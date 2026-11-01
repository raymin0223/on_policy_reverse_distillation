# Copyright 2023-2025 SGLang Team
# Copyright Amazon.com, Inc. or its affiliates.
# Copyright 2025 ModelBest Inc. and/or its affiliates
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

from abc import ABC, abstractmethod
from typing import Any, Callable

import torch

from verl.protocol import DataProto
from verl.workers.reward_manager.overlong_buffer import (
    OverlongBuffer,
    build_overlong_buffer,
    valid_response_lengths,
)

RawRewardFn = Callable[..., Any]


class AbstractRewardManager(ABC):
    @abstractmethod
    def __init__(
        self,
        tokenizer: Any,
        num_examine: int,
        compute_score: RawRewardFn | None,
        reward_fn_key: str = "data_source",
        **kwargs: Any,
    ):
        pass

    # --- DAPO-style soft overlong punishment (see verl/workers/reward_manager/overlong_buffer.py) ---

    def _init_overlong_buffer(self, overlong_buffer_cfg: Any = None, max_resp_len: Any = None) -> None:
        """Validate the overlong buffer config. Call from a manager's ``__init__``."""
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len
        self.overlong_buffer: OverlongBuffer | None = build_overlong_buffer(overlong_buffer_cfg, max_resp_len)

    @property
    def _overlong_buffer(self) -> OverlongBuffer | None:
        return getattr(self, "overlong_buffer", None)

    def _overlong_buffer_for_batch(self, response_width: int | None) -> OverlongBuffer | None:
        """The buffer, but only for batches generated under the cap M it was configured for.

        Validation typically generates with `rollout.val_kwargs.response_length`, which can differ
        from `data.max_response_length`. The penalty is defined relative to a specific generation
        cap, so applying a train-cap penalty to a batch generated under a different cap would be
        meaningless (and would silently corrupt validation scores). Skip those batches instead.
        """
        overlong_buffer = self._overlong_buffer
        if overlong_buffer is None or response_width is None:
            return overlong_buffer
        if int(response_width) != overlong_buffer.max_resp_len:
            if not getattr(self, "_warned_overlong_response_width", False):
                self._warned_overlong_response_width = True
                print(
                    f"[overlong_buffer] skipping penalty: batch responses are padded to "
                    f"{int(response_width)} tokens but the buffer is configured for "
                    f"max_response_length={overlong_buffer.max_resp_len} (this is expected for "
                    f"validation batches generated with a different response length)."
                )
            return None
        return overlong_buffer

    def _eos_completed(self, response_ids: torch.Tensor, valid_response_length: Any) -> bool:
        """Whether the response ended on an EOS token rather than being cut off at the hard cap.

        Only ``tokenizer.eos_token_id`` counts; a model with several stop ids (e.g. Qwen's
        <|im_end|>) reports EOS only for its configured eos_token_id.
        """
        eos_token_id = getattr(getattr(self, "tokenizer", None), "eos_token_id", None)
        if eos_token_id is None:
            return False
        length = int(
            valid_response_length.item() if hasattr(valid_response_length, "item") else valid_response_length
        )
        if length < 1 or length > int(response_ids.shape[-1]):
            return False
        return int(response_ids[length - 1].item()) == int(eos_token_id)

    def _apply_overlong_penalty(
        self,
        reward: float,
        valid_response_length: Any,
        reward_extra_info: dict[str, list] | None = None,
        response_ids: torch.Tensor | None = None,
    ) -> float:
        """Add the soft overlong penalty to one trajectory's scalar task reward.

        Returns ``reward`` unchanged when the feature is disabled. Must be called exactly once
        per trajectory, on the aggregated task reward, before group normalization.
        """
        response_width = None if response_ids is None else int(response_ids.shape[-1])
        overlong_buffer = self._overlong_buffer_for_batch(response_width)
        if overlong_buffer is None:
            return reward

        base_reward = float(reward)
        penalty = overlong_buffer.penalty(valid_response_length)
        final_reward = base_reward + penalty

        if overlong_buffer.log and reward_extra_info is not None:
            length = int(
                valid_response_length.item()
                if hasattr(valid_response_length, "item")
                else valid_response_length
            )
            reward_extra_info["overlong_reward"].append(penalty)  # DAPO-compatible key name
            reward_extra_info["overlong_penalty"].append(penalty)
            reward_extra_info["overlong"].append(overlong_buffer.is_overlong(length))
            reward_extra_info["base_reward"].append(base_reward)
            reward_extra_info["final_reward"].append(final_reward)
            reward_extra_info["response_length"].append(float(length))
            reward_extra_info["reached_max_response_length"].append(length >= overlong_buffer.max_resp_len)
            if response_ids is not None:
                reward_extra_info["eos_completed"].append(self._eos_completed(response_ids, length))

        return final_reward

    @abstractmethod
    def __call__(
        self,
        data: DataProto,
        return_dict: bool = False,
    ) -> torch.Tensor | dict[str, Any]:
        pass

    def _extract_reward_from_rm_scores(
        self, data: DataProto, return_dict: bool = False
    ) -> torch.Tensor | dict[str, Any] | None:
        """
        Extract reward from already-computed rm_scores if available.
        This is used when use_reward_loop=True and rewards are already computed during generate_sequences.

        Args:
            data: DataProto object containing the batch data
            return_dict: Whether to return a dictionary with reward_tensor and reward_extra_info

        Returns:
            If rm_scores exists:
                - If return_dict=True: dict with "reward_tensor" and "reward_extra_info"
                - If return_dict=False: torch.Tensor of rm_scores
            If rm_scores doesn't exist: None
        """
        if "rm_scores" not in data.batch.keys():
            return None

        reward_tensor = data.batch["rm_scores"]
        reward_extra_info: dict[str, Any] = {}
        if return_dict:
            reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
            reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}

        reward_tensor, overlong_extra_info = self._apply_overlong_penalty_to_token_level_scores(
            data, reward_tensor
        )
        reward_extra_info.update(overlong_extra_info)

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        else:
            return reward_tensor

    def _apply_overlong_penalty_to_token_level_scores(
        self, data: DataProto, reward_tensor: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, list]]:
        """Add the overlong penalty to a precomputed token-level score tensor (the rm_scores path).

        The penalty lands on each sequence's final valid response token — the same slot the
        reward managers scatter the scalar reward into — so the summed sequence reward that
        feeds group normalization matches the per-sample path exactly.
        """
        responses = data.batch.get("responses", None)
        overlong_buffer = self._overlong_buffer_for_batch(
            None if responses is None else int(responses.shape[-1])
        )
        if overlong_buffer is None:
            return reward_tensor, {}

        lengths = valid_response_lengths(data).to(reward_tensor.device)
        penalties = overlong_buffer.penalty_batch(lengths).to(dtype=reward_tensor.dtype)

        reward_tensor = reward_tensor.clone()
        rows = torch.arange(reward_tensor.shape[0], device=reward_tensor.device)
        last_idx = lengths.long() - 1
        base_rewards = reward_tensor.sum(dim=-1)
        reward_tensor[rows, last_idx] += penalties

        if not overlong_buffer.log:
            return reward_tensor, {}

        lengths_list = [float(x) for x in lengths.tolist()]
        penalties_list = [float(x) for x in penalties.tolist()]
        base_list = [float(x) for x in base_rewards.tolist()]
        extra_info: dict[str, list] = {
            "overlong_reward": penalties_list,  # DAPO-compatible key name
            "overlong_penalty": list(penalties_list),
            "overlong": [length > overlong_buffer.expected_len for length in lengths_list],
            "base_reward": base_list,
            "final_reward": [b + p for b, p in zip(base_list, penalties_list, strict=True)],
            "response_length": lengths_list,
            "reached_max_response_length": [
                length >= overlong_buffer.max_resp_len for length in lengths_list
            ],
        }
        if responses is not None:
            extra_info["eos_completed"] = [
                self._eos_completed(responses[i], int(length)) for i, length in enumerate(lengths_list)
            ]
        return reward_tensor, extra_info
