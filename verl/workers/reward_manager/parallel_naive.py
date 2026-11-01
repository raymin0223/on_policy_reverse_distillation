# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager
from verl.workers.reward_manager.naive import _hit_max_response_length, _truncate_for_console


_DEFAULT_PARALLEL_DATA_SOURCES = frozenset({"code", "livecodebench", "humanevalplus"})


def _score_failure_dict(message: str) -> dict[str, Any]:
    return {
        "score": 0.0,
        "acc": 0.0,
        "pred": "",
        "incorrect_format": 0,
        "error_in_test_cases": 1,
        "timed_out": 0,
        "truncated": 0,
        "truncated_and_missing_answer": 0,
        "format_reward": 1.0,
        "feedback": f"Reward scoring failed: {message}",
    }


def _parallel_score_one(
    compute_score,
    data_source: str,
    response_str: str,
    ground_truth: str,
    extra_info: dict[str, Any],
):
    try:
        return compute_score(
            data_source=data_source,
            solution_str=response_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )
    except BaseException as exc:
        return _score_failure_dict(f"{type(exc).__name__}: {exc}")


@register("parallel_naive")
class ParallelNaiveRewardManager(AbstractRewardManager):
    """Naive reward manager with completion-level parallelism for code tasks.

    This manager is intentionally opt-in. It keeps the same reward contract as
    NaiveRewardManager, but scores code-like completions in a bounded thread
    pool. The code reward function already runs tests in subprocesses, so this
    avoids pickling custom reward functions while still using many CPU cores.
    """

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        reward_num_workers: int = 8,
        val_reward_num_workers: int | None = None,
        parallel_data_sources: list[str] | tuple[str, ...] | None = None,
        overlong_buffer_cfg=None,
        max_resp_len=None,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self._init_overlong_buffer(overlong_buffer_cfg, max_resp_len)
        self.reward_num_workers = max(1, int(reward_num_workers))
        self.val_reward_num_workers = (
            max(1, int(val_reward_num_workers))
            if val_reward_num_workers is not None
            else self.reward_num_workers
        )
        self.parallel_data_sources = (
            frozenset(str(x) for x in parallel_data_sources)
            if parallel_data_sources is not None
            else _DEFAULT_PARALLEL_DATA_SOURCES
        )

    def _max_workers_for_jobs(self, items: list[dict[str, Any]], job_indices: list[int]) -> int:
        if any(str(items[i]["extra_info"].get("split", "")) == "test" for i in job_indices):
            return min(self.val_reward_num_workers, len(job_indices))
        return min(self.reward_num_workers, len(job_indices))

    def __call__(self, data: DataProto, return_dict: bool = True) -> torch.Tensor | dict[str, Any]:
        reward_from_rm_scores = self._extract_reward_from_rm_scores(data, return_dict)
        if reward_from_rm_scores is not None:
            return reward_from_rm_scores

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print_data_sources = {}

        items: list[dict[str, Any]] = []
        parallel_jobs: list[int] = []
        scores: list[Any] = [None] * len(data)

        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            data_source = str(data_item.non_tensor_batch[self.reward_fn_key])
            extra_info = dict(data_item.non_tensor_batch.get("extra_info", {}) or {})
            extra_info["num_turns"] = data_item.non_tensor_batch.get("__num_turns__", None)
            extra_info["rollout_reward_scores"] = data_item.non_tensor_batch.get("reward_scores", {})
            extra_info["truncated"] = _hit_max_response_length(response_ids, valid_response_length)

            item = {
                "prompt_str": prompt_str,
                "response_str": response_str,
                "ground_truth": data_item.non_tensor_batch["reward_model"]["ground_truth"],
                "data_source": data_source,
                "extra_info": extra_info,
                "valid_response_length": valid_response_length,
                "response_ids": response_ids,
            }
            items.append(item)

            if data_source in self.parallel_data_sources:
                parallel_jobs.append(i)
            else:
                scores[i] = _parallel_score_one(
                    self.compute_score,
                    data_source,
                    response_str,
                    item["ground_truth"],
                    extra_info,
                )

        if parallel_jobs:
            max_workers = self._max_workers_for_jobs(items, parallel_jobs)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        _parallel_score_one,
                        self.compute_score,
                        items[i]["data_source"],
                        items[i]["response_str"],
                        items[i]["ground_truth"],
                        items[i]["extra_info"],
                    ): i
                    for i in parallel_jobs
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        scores[idx] = future.result()
                    except BaseException as exc:
                        scores[idx] = _score_failure_dict(f"{type(exc).__name__}: {exc}")

        for i, score in enumerate(scores):
            if isinstance(score, dict):
                reward = score["score"]
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            valid_response_length = items[i]["valid_response_length"]
            reward = self._apply_overlong_penalty(
                reward, valid_response_length, reward_extra_info, items[i]["response_ids"]
            )
            reward_tensor[i, valid_response_length.item() - 1] = reward

            data_source = items[i]["data_source"]
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", _truncate_for_console(items[i]["prompt_str"]))
                print("[response]", _truncate_for_console(items[i]["response_str"]))
                print("[ground_truth]", _truncate_for_console(items[i]["ground_truth"]))
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", _truncate_for_console(value))
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
