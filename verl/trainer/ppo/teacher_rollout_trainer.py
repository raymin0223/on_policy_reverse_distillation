import asyncio
import json
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional
import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from verl import DataProto
from verl.experimental.agent_loop import AsyncLLMServerManager
from verl.trainer.ppo.opsd_trainer import OPSDRayPPOTrainer
from verl.trainer.ppo.utils import Role
from verl.utils import hf_tokenizer
from verl.utils.dataset.rl_dataset import build_plain_completion_prompt, use_plain_completion_prompt_format
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask
from verl.workers.rollout.server_pool import RolloutServerPool


class TeacherVLLMRolloutManager:
    """Standalone teacher rollout path backed by dedicated vLLM replicas."""

    def __init__(self, config, resource_pool, tokenizer):
        self.config = config
        self.rollout_config = self._build_teacher_rollout_config()
        self.model_config = self._build_teacher_model_config()
        trust_remote_code = bool(self.model_config.get("trust_remote_code", False))
        self.tokenizer = hf_tokenizer(self.model_config.tokenizer_path, trust_remote_code=trust_remote_code)
        self.tokenizer.padding_side = "left"
        distill_cfg = self.config.actor_rollout_ref.actor.distillation
        self.keep_awake = False
        self.pad_token_id = self.tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0
        self.server_pool = RolloutServerPool(
            rollout_config=self.rollout_config,
            model_config=self.model_config,
            n_gpus_per_node=self.config.trainer.n_gpus_per_node,
            resource_pool=resource_pool,
        )
        self.server_manager = AsyncLLMServerManager(config, self.server_pool.server_handles)
        if self.rollout_config.free_cache_engine and (not self.keep_awake):
            self.sleep()

    def _build_teacher_rollout_config(self):
        rollout_cfg = OmegaConf.create(
            OmegaConf.to_container(self.config.actor_rollout_ref.rollout, resolve=True)
        )
        distill_cfg = self.config.actor_rollout_ref.actor.distillation
        teacher_tp_size = None
        if teacher_tp_size is not None:
            rollout_cfg.tensor_model_parallel_size = int(teacher_tp_size)
        teacher_dp_size = None
        if teacher_dp_size is not None:
            rollout_cfg.data_parallel_size = int(teacher_dp_size)
        return rollout_cfg

    def _build_teacher_model_config(self):
        model_cfg = OmegaConf.to_container(self.config.actor_rollout_ref.model, resolve=True)
        distill_cfg = self.config.actor_rollout_ref.actor.distillation
        teacher_model_path = distill_cfg.get("teacher_model_path")
        teacher_lora_adapter_path = None
        if teacher_model_path is None or not str(teacher_model_path).strip():
            raise ValueError("teacher_model_path is required to launch teacher vLLM rollout.")
        model_cfg["path"] = str(teacher_model_path)
        model_cfg["hf_config_path"] = str(teacher_model_path)
        model_cfg["tokenizer_path"] = str(teacher_model_path)
        model_cfg["lora_adapter_path"] = (
            str(teacher_lora_adapter_path) if teacher_lora_adapter_path is not None else None
        )
        if teacher_lora_adapter_path is not None:
            inferred_lora_rank = self._infer_lora_rank(str(teacher_lora_adapter_path))
            if inferred_lora_rank > 0:
                model_cfg["lora_rank"] = inferred_lora_rank
        return OmegaConf.create(model_cfg)

    def _infer_lora_rank(self, adapter_path: str) -> int:
        adapter_config_path = Path(
            copy_to_local(adapter_path, use_shm=self.config.actor_rollout_ref.model.use_shm)
        )
        adapter_config_path = adapter_config_path / "adapter_config.json"
        if not adapter_config_path.exists():
            return int(self.config.actor_rollout_ref.model.get("lora_rank", 0))
        with adapter_config_path.open("r", encoding="utf-8") as fp:
            adapter_config = json.load(fp)
        return int(adapter_config.get("r", self.config.actor_rollout_ref.model.get("lora_rank", 0)))

    def wake_up(self):
        self.server_pool.wake_up()

    def sleep(self):
        self.server_pool.sleep()

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        prompts = self._rebuild_prompts_with_teacher_tokenizer(prompts)
        required_prompt_keys = {"input_ids", "attention_mask", "position_ids"}
        missing_prompt_keys = required_prompt_keys - set(prompts.batch.keys())
        if missing_prompt_keys:
            raise ValueError(
                f"teacher vLLM rollout missing prompt tensor keys: {sorted(missing_prompt_keys)}"
            )
        total_start_time = time.perf_counter()
        wake_elapsed = 0.0
        if self.rollout_config.free_cache_engine and (not self.keep_awake):
            wake_start_time = time.perf_counter()
            self.wake_up()
            wake_elapsed = time.perf_counter() - wake_start_time
        generate_start_time = time.perf_counter()
        sleep_elapsed = 0.0
        try:
            outputs = self._generate(prompts)
        finally:
            if self.rollout_config.free_cache_engine and (not self.keep_awake):
                sleep_start_time = time.perf_counter()
                self.sleep()
                sleep_elapsed = time.perf_counter() - sleep_start_time
        generate_elapsed = time.perf_counter() - generate_start_time - sleep_elapsed
        total_elapsed = time.perf_counter() - total_start_time
        timing = {
            "teacher_generation_timing/wake": wake_elapsed,
            "teacher_generation_timing/generate_only": generate_elapsed,
            "teacher_generation_timing/sleep": sleep_elapsed,
            "teacher_generation_timing/keep_awake": 1.0 if self.keep_awake else 0.0,
        }
        return self._build_output(prompts=prompts, outputs=outputs, elapsed=total_elapsed, timing=timing)

    @staticmethod
    def _to_message_list(raw_prompt: Any) -> list[dict]:
        if isinstance(raw_prompt, np.ndarray):
            raw_prompt = raw_prompt.tolist()
        if raw_prompt is None:
            return []
        if isinstance(raw_prompt, list):
            return [
                dict(item) if isinstance(item, dict) else {"role": "user", "content": str(item)}
                for item in raw_prompt
            ]
        if isinstance(raw_prompt, dict):
            return [raw_prompt]
        return [{"role": "user", "content": str(raw_prompt)}]

    def _rebuild_prompts_with_teacher_tokenizer(self, prompts: DataProto) -> DataProto:
        raw_prompts = (
            prompts.non_tensor_batch.get("raw_prompt") if prompts.non_tensor_batch is not None else None
        )
        if raw_prompts is None:
            if prompts.batch is None:
                raise ValueError("teacher vLLM rollout requires raw_prompt or prompt tensors.")
            return prompts
        messages = [self._to_message_list(raw_prompt) for raw_prompt in raw_prompts]
        max_prompt_length = prompts.meta_info.get("max_prompt_length", None)
        if max_prompt_length is None:
            max_prompt_length = OmegaConf.select(self.config, "data.max_prompt_length")
        if use_plain_completion_prompt_format(self.config.data):
            include_system = bool(self.config.data.get("plain_completion_include_system", True))
            prompt_texts = [
                build_plain_completion_prompt(message_list, include_system=include_system)
                for message_list in messages
            ]
            tokenize_kwargs = {
                "return_tensors": "pt",
                "padding": True,
                "truncation": True,
                "add_special_tokens": False,
            }
            if max_prompt_length is not None:
                tokenize_kwargs["max_length"] = int(max_prompt_length)
            prompt_dict = self.tokenizer(prompt_texts, **tokenize_kwargs)
        else:
            enable_thinking = (
                self.config.data.apply_chat_template_kwargs.get("enable_thinking", True)
                if self.config.data.apply_chat_template_kwargs
                else True
            )
            assistant_final_flags = [
                bool(message_list) and message_list[-1].get("role") == "assistant"
                for message_list in messages
            ]
            if any(assistant_final_flags) and (not all(assistant_final_flags)):
                raise ValueError(
                    "Mixed teacher prompt batches with and without assistant prefill are not supported."
                )
            continue_final_message = bool(assistant_final_flags) and all(assistant_final_flags)
            apply_kwargs = {
                "tokenize": True,
                "return_tensors": "pt",
                "return_dict": True,
                "continue_final_message": continue_final_message,
                "add_generation_prompt": not continue_final_message,
                "enable_thinking": enable_thinking,
                "padding": True,
                "truncation": True,
            }
            if max_prompt_length is not None:
                apply_kwargs["max_length"] = int(max_prompt_length)
            prompt_dict = self.tokenizer.apply_chat_template(messages, **apply_kwargs)
        attention_mask = prompt_dict["attention_mask"]
        return DataProto.from_dict(
            tensors={
                "input_ids": prompt_dict["input_ids"],
                "attention_mask": attention_mask,
                "position_ids": compute_position_id_with_mask(attention_mask),
            },
            non_tensors=prompts.non_tensor_batch,
            meta_info=prompts.meta_info,
        )

    def _generate(self, prompts: DataProto):
        prompt_ids_rows = self._extract_prompt_ids(prompts)
        response_length = int(prompts.meta_info.get("response_length", self.rollout_config.response_length))
        do_sample = bool(prompts.meta_info.get("do_sample", self.rollout_config.do_sample))
        sampling_params = {"max_tokens": response_length, "logprobs": False}
        if do_sample:
            sampling_params.update(
                {
                    "temperature": float(
                        prompts.meta_info.get("temperature", self.rollout_config.temperature)
                    ),
                    "top_p": float(prompts.meta_info.get("top_p", self.rollout_config.get("top_p", 1.0))),
                    "top_k": int(prompts.meta_info.get("top_k", self.rollout_config.get("top_k", 0))),
                }
            )
        else:
            sampling_params["temperature"] = 0.0

        async def run_all():
            return await asyncio.gather(
                *[
                    self.server_manager.generate(
                        request_id=uuid.uuid4().hex,
                        prompt_ids=prompt_ids,
                        sampling_params=dict(sampling_params),
                    )
                    for prompt_ids in prompt_ids_rows
                ]
            )

        return asyncio.run(run_all())

    @staticmethod
    def _extract_prompt_ids(prompts: DataProto) -> list[list[int]]:
        prompt_rows: list[list[int]] = []
        input_ids = prompts.batch["input_ids"]
        attention_mask = prompts.batch["attention_mask"]
        for row_ids, row_mask in zip(input_ids, attention_mask, strict=True):
            valid_positions = row_mask.to(dtype=torch.bool)
            prompt_rows.append(row_ids[valid_positions].tolist())
        return prompt_rows

    def _build_output(
        self, prompts: DataProto, outputs: list, elapsed: float, timing: Optional[dict] = None
    ) -> DataProto:
        prompt_ids = prompts.batch["input_ids"].cpu()
        prompt_attention_mask = prompts.batch["attention_mask"].cpu()
        response_length = int(prompts.meta_info.get("response_length", self.rollout_config.response_length))
        batch_size = prompt_ids.size(0)
        responses = torch.full((batch_size, response_length), self.pad_token_id, dtype=prompt_ids.dtype)
        response_mask = torch.zeros((batch_size, response_length), dtype=prompt_attention_mask.dtype)
        for idx, output in enumerate(outputs):
            token_ids = output.token_ids[:response_length]
            if len(token_ids) == 0:
                continue
            token_tensor = torch.tensor(token_ids, dtype=prompt_ids.dtype)
            token_count = token_tensor.numel()
            responses[idx, :token_count] = token_tensor
            response_mask[idx, :token_count] = 1
        attention_mask = torch.cat([prompt_attention_mask, response_mask], dim=1)
        input_ids = torch.cat([prompt_ids, responses], dim=1)
        position_ids = compute_position_id_with_mask(attention_mask)
        batch = TensorDict(
            {
                "prompts": prompt_ids,
                "responses": responses,
                "response_mask": response_mask,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        timing = dict(timing or {})
        timing.update(
            {
                "teacher_generation_timing/max": elapsed,
                "teacher_generation_timing/min": elapsed,
                "teacher_generation_timing/topk_ratio": 1.0,
                "teacher_generation_timing/mean": elapsed,
            }
        )
        return DataProto(
            batch=batch,
            non_tensor_batch={
                "teacher_response_text": np.array(
                    [
                        self.tokenizer.decode(output.token_ids[:response_length], skip_special_tokens=True)
                        for output in outputs
                    ],
                    dtype=object,
                ),
                "teacher_response_length": np.array(
                    [float(min(len(output.token_ids), response_length)) for output in outputs],
                    dtype=np.float32,
                ),
            },
            meta_info={"timing": timing},
        )


class TeacherRolloutRayPPOTrainer(OPSDRayPPOTrainer):
    """Reverse distillation trainer with teacher-rollout guided reprompting."""

    _CACHED_TEACHER_PREFIX_KEY = "reverse_teacher_prefix_text"
    _CACHED_TEACHER_SUFFIX_KEY = "reverse_teacher_suffix_text"
    _CACHED_TEACHER_SPLIT_IDX_KEY = "reverse_teacher_split_idx"
    _CACHED_TEACHER_SPLIT_RATIO_KEY = "reverse_teacher_split_ratio"
    _CACHED_TEACHER_IS_CORRECT_KEY = "reverse_teacher_is_correct"
    _CACHED_TEACHER_FALLBACK_KEY = "reverse_teacher_generation_fallback"
    _CACHED_TEACHER_ERROR_KEY = "reverse_teacher_generation_error"
    _ROLLOUT_REPEAT_COUNTS_META_KEY = "rollout_repeat_counts"

    def init_workers(self):
        super().init_workers()
        self.teacher_rollout_manager = None
        if self._should_use_teacher_vllm_rollout():
            actor_role = (
                Role.ActorRolloutRef
                if Role.ActorRolloutRef in self.role_worker_mapping
                else Role.ActorRollout
            )
            resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
            self.teacher_rollout_manager = TeacherVLLMRolloutManager(
                config=self.config, resource_pool=resource_pool, tokenizer=self.tokenizer
            )

    def _should_use_teacher_vllm_rollout(self) -> bool:
        distill_cfg = self.config.actor_rollout_ref.actor.get("distillation", None)
        if distill_cfg is None:
            return False
        feedback_teacher_stats_path = None
        if feedback_teacher_stats_path is not None and str(feedback_teacher_stats_path).strip():
            return False
        teacher_model_path = distill_cfg.get("teacher_model_path", None)
        if teacher_model_path is None or not str(teacher_model_path).strip():
            return False
        return str(self.config.actor_rollout_ref.rollout.name).lower() == "vllm"

    def _generate_teacher_rollout_sequences(self, teacher_prompt_batch: DataProto) -> DataProto:
        if self.teacher_rollout_manager is not None:
            return self.teacher_rollout_manager.generate_sequences(teacher_prompt_batch)
        if not hasattr(self.actor_rollout_wg, "generate_teacher_sequences"):
            raise AttributeError("generate_teacher_sequences is unavailable on actor_rollout_wg")
        return self.actor_rollout_wg.generate_teacher_sequences(teacher_prompt_batch)

    def _build_uid_index(self, uids: np.ndarray) -> tuple[dict[Any, list[int]], list[int]]:
        uid_to_indices: dict[Any, list[int]] = defaultdict(list)
        first_indices: list[int] = []
        for idx, uid in enumerate(uids):
            if uid not in uid_to_indices:
                first_indices.append(idx)
            uid_to_indices[uid].append(idx)
        return (uid_to_indices, first_indices)

    @staticmethod
    def _slice_prompt_position_ids(position_ids: torch.Tensor, response_length: int) -> torch.Tensor:
        if position_ids.dim() == 2:
            return position_ids[:, :-response_length]
        if position_ids.dim() == 3:
            return position_ids[:, :, :-response_length]
        raise ValueError(f"Unsupported position_ids rank: {position_ids.dim()}")

    @staticmethod
    def _to_message_list(raw_prompt: Any) -> list[dict]:
        if isinstance(raw_prompt, np.ndarray):
            raw_prompt = raw_prompt.tolist()
        if raw_prompt is None:
            return []
        if isinstance(raw_prompt, str):
            return [{"role": "user", "content": raw_prompt}]
        if isinstance(raw_prompt, dict):
            return [dict(raw_prompt)]
        if isinstance(raw_prompt, list | tuple):
            if len(raw_prompt) == 0:
                return []
            first_item = raw_prompt[0]
            if isinstance(first_item, dict):
                return [dict(msg) for msg in raw_prompt]
            if isinstance(first_item, str):
                return [{"role": "user", "content": "".join((str(part) for part in raw_prompt))}]
        return [{"role": "user", "content": str(raw_prompt)}]

    def _extract_prompt_text(self, raw_prompt: Any) -> str:
        messages = self._to_message_list(raw_prompt)
        if len(messages) == 0:
            return ""
        content = messages[-1].get("content", "")
        return content if isinstance(content, str) else str(content)

    def _tokenize_plain_prompt_texts(
        self, prompt_texts: list[str], max_length: Optional[int] = None
    ) -> dict[str, torch.Tensor]:
        tokenize_kwargs = {
            "add_special_tokens": False,
            "return_tensors": "pt",
            "padding": True,
            "truncation": True,
        }
        if max_length is not None:
            tokenize_kwargs["max_length"] = int(max_length)
        prompt_dict = self.tokenizer(prompt_texts, **tokenize_kwargs)
        attention_mask = prompt_dict["attention_mask"]
        position_ids = compute_position_id_with_mask(attention_mask)
        return {
            "input_ids": prompt_dict["input_ids"],
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }

    def _tokenize_prompts(
        self, messages: list[list[dict]], max_length: Optional[int] = None
    ) -> dict[str, torch.Tensor]:
        if use_plain_completion_prompt_format(self.config.data):
            include_system = bool(self.config.data.get("plain_completion_include_system", True))
            prompt_texts = [
                build_plain_completion_prompt(message_list, include_system=include_system)
                for message_list in messages
            ]
            return self._tokenize_plain_prompt_texts(prompt_texts, max_length=max_length)
        enable_thinking = (
            self.config.data.apply_chat_template_kwargs.get("enable_thinking", True)
            if self.config.data.apply_chat_template_kwargs
            else True
        )
        assistant_final_flags = [
            bool(message_list) and message_list[-1].get("role") == "assistant" for message_list in messages
        ]
        if any(assistant_final_flags) and (not all(assistant_final_flags)):
            raise ValueError(
                "Mixed prompt batches with and without final assistant-prefill messages are not supported."
            )
        continue_final_message = bool(assistant_final_flags) and all(assistant_final_flags)
        apply_kwargs = {
            "tokenize": True,
            "return_tensors": "pt",
            "return_dict": True,
            "continue_final_message": continue_final_message,
            "add_generation_prompt": not continue_final_message,
            "enable_thinking": enable_thinking,
            "padding": True,
            "truncation": True,
        }
        if max_length is not None:
            apply_kwargs["max_length"] = int(max_length)
        prompt_dict = self.tokenizer.apply_chat_template(messages, **apply_kwargs)
        attention_mask = prompt_dict["attention_mask"]
        position_ids = compute_position_id_with_mask(attention_mask)
        return {
            "input_ids": prompt_dict["input_ids"],
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        gen_batch = super()._get_gen_batch(batch)
        if getattr(self, "_in_validation", False):
            return gen_batch
        distillation_cfg = self.config.actor_rollout_ref.actor.get("distillation", None)
        loss_mode = self.config.actor_rollout_ref.actor.policy_loss.get("loss_mode", "vanilla")
        if distillation_cfg is None or True:
            return gen_batch
        if "raw_prompt" not in batch.non_tensor_batch:
            raise KeyError("reverse_distill requires non_tensor_batch['raw_prompt'], but it is missing.")
        teacher_first_indices = list(range(batch.batch.batch_size[0]))
        teacher_prompt_batch = self._build_teacher_prompt_batch(
            batch=batch, first_indices=teacher_first_indices
        )
        if teacher_prompt_batch.batch is None:
            raise RuntimeError(
                "teacher_prompt_batch.batch is None before teacher rollout generation. Expected tokenized prompt tensors (input_ids/attention_mask/position_ids)."
            )
        required_prompt_keys = {"input_ids", "attention_mask", "position_ids"}
        missing_prompt_keys = required_prompt_keys - set(teacher_prompt_batch.batch.keys())
        if missing_prompt_keys:
            raise RuntimeError(
                f"teacher_prompt_batch is missing required tensor keys: {sorted(missing_prompt_keys)}"
            )
        try:
            teacher_rollout = self._generate_teacher_rollout_sequences(teacher_prompt_batch)
        except Exception as exc:
            raise RuntimeError(
                "Teacher rollout generation failed in reverse_distill; fallback is disabled."
            ) from exc
        reward_threshold = float(distillation_cfg.success_reward_threshold)
        self._cache_teacher_rollout_info(
            batch=batch,
            teacher_rollout=teacher_rollout,
            distillation_cfg=distillation_cfg,
            reward_threshold=reward_threshold,
            fallback_flag=0.0,
            error_flag=0.0,
        )
        prefix_texts = batch.non_tensor_batch.get(self._CACHED_TEACHER_PREFIX_KEY)
        if prefix_texts is None:
            raise RuntimeError(
                "Teacher prefix cache is missing after teacher rollout generation in reverse_distill."
            )
        try:
            max_reprompt_len = int(distillation_cfg.get("max_reprompt_len", 10240))
            if use_plain_completion_prompt_format(self.config.data):
                include_system = bool(self.config.data.get("plain_completion_include_system", True))
                prompt_texts = []
                for i in range(batch.batch.batch_size[0]):
                    base_prompt = build_plain_completion_prompt(
                        batch.non_tensor_batch["raw_prompt"][i], include_system=include_system
                    )
                    prompt_texts.append(f"{base_prompt}{str(prefix_texts[i])}")
                prompt_tensors = self._tokenize_plain_prompt_texts(prompt_texts, max_length=max_reprompt_len)
            else:
                messages = [
                    self._build_prompt_with_prefix(
                        batch.non_tensor_batch["raw_prompt"][i], str(prefix_texts[i])
                    )
                    for i in range(batch.batch.batch_size[0])
                ]
                prompt_tensors = self._tokenize_prompts(messages, max_length=max_reprompt_len)
        except Exception as exc:
            raise RuntimeError(
                "Failed to tokenize reprompted teacher-prefix prompts in reverse_distill; fallback is disabled."
            ) from exc
        preserved_tensors = {}
        if gen_batch.batch is not None:
            for key in gen_batch.batch.keys():
                if key in {"input_ids", "attention_mask", "position_ids", "prompts", "responses"}:
                    continue
                preserved_tensors[key] = gen_batch.batch[key]
        preserved_tensors.update(prompt_tensors)
        gen_meta_info = dict(gen_batch.meta_info)
        return DataProto.from_dict(
            tensors=preserved_tensors, non_tensors=gen_batch.non_tensor_batch, meta_info=gen_meta_info
        )

    def _validate(self, merged: bool = False, defer_reward: bool = False):
        self._in_validation = True
        try:
            return super()._validate(merged=merged, defer_reward=defer_reward)
        finally:
            self._in_validation = False

    def _build_teacher_prompt_batch(self, batch: DataProto, first_indices: list[int]) -> DataProto:
        rollout_cfg = self.config.actor_rollout_ref.rollout
        distillation_cfg = self.config.actor_rollout_ref.actor.get("distillation", None)
        max_prompt_len = self.config.data.get("max_prompt_length", None)
        if "raw_prompt" not in batch.non_tensor_batch:
            raise KeyError("reverse_distill requires non_tensor_batch['raw_prompt'], but it is missing.")
        try:
            messages = [
                self._to_message_list(batch.non_tensor_batch["raw_prompt"][idx]) for idx in first_indices
            ]
            prompt_tensors = self._tokenize_prompts(messages, max_length=max_prompt_len)
            prompt_input_ids = prompt_tensors["input_ids"]
            prompt_attention_mask = prompt_tensors["attention_mask"]
            prompt_position_ids = prompt_tensors["position_ids"]
            response_length = int(rollout_cfg.response_length)
        except Exception as exc:
            raise RuntimeError(
                "Failed to tokenize teacher prompts for rollout generation in reverse_distill; fallback is disabled."
            ) from exc
        teacher_temperature = batch.meta_info.get("temperature", rollout_cfg.temperature)
        teacher_top_p = float(rollout_cfg.get("top_p", 1.0))
        if distillation_cfg is not None:
            teacher_temperature = 1.0
            teacher_top_p = 1.0
        prompt_meta = {
            "temperature": teacher_temperature,
            "response_length": response_length,
            "do_sample": rollout_cfg.do_sample,
            "top_p": teacher_top_p,
            "top_k": rollout_cfg.get("top_k", 0),
            "max_prompt_length": max_prompt_len,
        }
        return DataProto.from_dict(
            tensors={
                "input_ids": prompt_input_ids,
                "attention_mask": prompt_attention_mask,
                "position_ids": prompt_position_ids,
            },
            non_tensors={
                "raw_prompt": np.array(
                    [batch.non_tensor_batch["raw_prompt"][idx] for idx in first_indices], dtype=object
                )
            },
            meta_info=prompt_meta,
        )

    @staticmethod
    def _response_mask(attention_mask: torch.Tensor, responses: torch.Tensor) -> torch.Tensor:
        response_length = responses.size(1)
        return attention_mask[:, -response_length:]

    def _maybe_build_distillation_batch(
        self,
        batch: DataProto,
        reward_tensor: torch.Tensor,
        reward_extra_infos_dict: Optional[dict[str, list]] = None,
    ) -> Optional[tuple[DataProto, dict[str, float]]]:
        distillation_cfg = self.config.actor_rollout_ref.actor.get("distillation", None)
        loss_mode = self.config.actor_rollout_ref.actor.policy_loss.get("loss_mode", "vanilla")
        if distillation_cfg is None or True:
            return None
        metric_prefix = loss_mode
        enable_on_policy_distill = False
        reverse_loss_alpha = 0.5
        reverse_loss_schedule = "constant".lower()
        reverse_batch, reverse_metrics = self._build_reverse_distillation_batch(
            batch=batch,
            reward_tensor=reward_tensor,
            reward_extra_infos_dict=reward_extra_infos_dict,
            metric_prefix=metric_prefix,
        )
        metrics = {
            f"{metric_prefix}/enable_on_policy_distill": 1.0 if enable_on_policy_distill else 0.0,
            f"{metric_prefix}/reverse_loss_alpha_config": reverse_loss_alpha,
            f"{metric_prefix}/reverse_loss_alpha_schedule": 1.0 if reverse_loss_schedule == "linear" else 0.0,
            **reverse_metrics,
        }
        if not enable_on_policy_distill:
            metrics[f"{metric_prefix}/opsd_weight_config"] = 1.0
            metrics[f"{metric_prefix}/on_policy_weight_config"] = 0.0
            return (reverse_batch, metrics)
        on_policy_batch, on_policy_metrics = self._build_on_policy_distillation_batch(
            teacher_input_ids=reverse_batch.batch["teacher_input_ids"],
            teacher_attention_mask=reverse_batch.batch["teacher_attention_mask"],
            teacher_position_ids=reverse_batch.batch["teacher_position_ids"],
            on_policy_distillation_mask=reverse_batch.batch["on_policy_suffix_mask"],
            metric_prefix=metric_prefix,
        )
        metrics[f"{metric_prefix}/opsd_weight_config"] = reverse_loss_alpha
        metrics[f"{metric_prefix}/on_policy_weight_config"] = 1.0 - reverse_loss_alpha
        metrics[f"{metric_prefix}/on_policy_prompt_uses_reprompt"] = 1.0
        metrics.update(on_policy_metrics)
        return (reverse_batch.union(on_policy_batch), metrics)
