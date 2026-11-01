"""Source-to-learner policy optimization (S2L-PO).

S2L-PO keeps the ordinary PPO/GRPO objective and changes only rollout
collection and the policy used as the denominator of the token-level PPO
ratio.  Weak-policy samples retain their generation log probabilities all the
way to the actor batch; strong-policy samples use the usual frozen old actor.
"""

import asyncio
import math
import uuid
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.teacher_rollout_trainer import TeacherVLLMRolloutManager
from verl.trainer.ppo.utils import Role
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask


S2L_IS_WEAK_KEY = "s2l_is_weak"
S2L_SOURCE_LOG_PROBS_KEY = "source_policy_log_probs"
S2L_SOURCE_KEY = "rollout_source"


def s2l_phase_allocation(
    step: int,
    *,
    total_steps: int,
    t_mix: int,
    num_phases: int,
    group_size: int,
) -> tuple[int, int, int]:
    """Return ``(phase, weak_count, strong_count)`` for a 1-based step.

    The first half of the phases spans ``min(t_mix, total_steps)`` updates and
    anneals weak samples to zero.  The second half spans the remaining updates
    and is strong-only.  ``ceil`` gives the requested default allocation
    ``8,7,6,5,4,3,2,0`` while the last mixing phase is pinned to zero.
    """

    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}.")
    if step <= 0:
        raise ValueError(f"step must be 1-based and positive, got {step}.")
    if t_mix < 0:
        raise ValueError(f"t_mix must be non-negative, got {t_mix}.")
    if num_phases < 2 or num_phases % 2 != 0:
        raise ValueError(f"num_phases must be an even integer >= 2, got {num_phases}.")
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}.")

    step = min(step, total_steps)
    mix_phases = num_phases // 2
    strong_only_phases = num_phases - mix_phases
    mix_steps = min(t_mix, total_steps)

    if mix_steps > 0 and step <= mix_steps:
        phase = min(((step - 1) * mix_phases) // mix_steps, mix_phases - 1)
        if mix_phases == 1 or phase == mix_phases - 1:
            weak_count = 0
        else:
            weak_count = math.ceil(group_size * (mix_phases - 1 - phase) / (mix_phases - 1))
    else:
        tail_steps = total_steps - mix_steps
        if tail_steps <= 0:
            phase = mix_phases - 1
        else:
            tail_step = step - mix_steps
            tail_phase = min(((tail_step - 1) * strong_only_phases) // tail_steps, strong_only_phases - 1)
            phase = mix_phases + tail_phase
        weak_count = 0

    weak_count = min(max(int(weak_count), 0), group_size)
    return int(phase), weak_count, group_size - weak_count


class S2LWeakVLLMRolloutManager(TeacherVLLMRolloutManager):
    """Dedicated frozen weak-policy vLLM pool that always returns token log-probs."""

    def __init__(self, config, resource_pool, tokenizer):
        strong_vocab = tokenizer.get_vocab()
        strong_special_ids = (tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.bos_token_id)
        super().__init__(config=config, resource_pool=resource_pool, tokenizer=tokenizer)
        weak_special_ids = (
            self.tokenizer.pad_token_id,
            self.tokenizer.eos_token_id,
            self.tokenizer.bos_token_id,
        )
        if strong_vocab != self.tokenizer.get_vocab() or strong_special_ids != weak_special_ids:
            raise ValueError(
                "S2L-PO requires identical weak/strong token-id mappings and pad/eos/bos ids so weak "
                "actions and token log-probabilities can be consumed by the strong model without conversion."
            )

    def _s2l_config(self):
        return self.config.algorithm.s2l_po

    def _build_teacher_rollout_config(self):
        rollout_cfg = OmegaConf.create(
            OmegaConf.to_container(self.config.actor_rollout_ref.rollout, resolve=True)
        )
        s2l_cfg = self._s2l_config()
        weak_tp_size = s2l_cfg.get("weak_rollout_tensor_model_parallel_size", None)
        if weak_tp_size is not None:
            rollout_cfg.tensor_model_parallel_size = int(weak_tp_size)
        weak_dp_size = s2l_cfg.get("weak_rollout_data_parallel_size", None)
        if weak_dp_size is not None:
            rollout_cfg.data_parallel_size = int(weak_dp_size)
        return rollout_cfg

    def _build_teacher_model_config(self):
        model_cfg = OmegaConf.to_container(self.config.actor_rollout_ref.model, resolve=True)
        s2l_cfg = self._s2l_config()
        weak_model_path = s2l_cfg.get("weak_model_path", None)
        weak_lora_adapter_path = s2l_cfg.get("weak_lora_adapter_path", None)
        if weak_model_path is None or not str(weak_model_path).strip():
            raise ValueError("algorithm.s2l_po.weak_model_path is required when S2L-PO is enabled.")

        model_cfg["path"] = str(weak_model_path)
        model_cfg["hf_config_path"] = str(weak_model_path)
        model_cfg["tokenizer_path"] = str(weak_model_path)
        model_cfg["lora_adapter_path"] = (
            str(weak_lora_adapter_path) if weak_lora_adapter_path is not None else None
        )
        if weak_lora_adapter_path is not None:
            adapter_config_path = Path(
                copy_to_local(str(weak_lora_adapter_path), use_shm=model_cfg.get("use_shm", False))
            )
            adapter_config_path = adapter_config_path / "adapter_config.json"
            if adapter_config_path.exists():
                import json

                with adapter_config_path.open("r", encoding="utf-8") as fp:
                    model_cfg["lora_rank"] = int(json.load(fp).get("r", model_cfg.get("lora_rank", 0)))
        return OmegaConf.create(model_cfg)

    def _rebuild_prompts_with_teacher_tokenizer(self, prompts: DataProto) -> DataProto:
        weak_prompts = super()._rebuild_prompts_with_teacher_tokenizer(prompts)
        # Async agent-loop inputs intentionally carry only non-tensor
        # ``raw_prompt`` data.  In that path there are no strong prompt tensors
        # to compare or return, so retain the weak-tokenizer reconstruction and
        # pad it to the fixed prompt width used by AgentLoopWorker outputs.
        if prompts.batch is None:
            return self._pad_prompts_to_rollout_width(weak_prompts)

        for row, (strong_ids, strong_mask, weak_ids, weak_mask) in enumerate(
            zip(
                prompts.batch["input_ids"],
                prompts.batch["attention_mask"],
                weak_prompts.batch["input_ids"],
                weak_prompts.batch["attention_mask"],
                strict=True,
            )
        ):
            strong_valid = strong_ids[strong_mask.bool()].detach().cpu()
            weak_valid = weak_ids[weak_mask.bool()].detach().cpu()
            if not torch.equal(strong_valid, weak_valid):
                raise ValueError(
                    "S2L-PO requires token-compatible weak/strong tokenizers; "
                    f"prompt row {row} produced different token ids."
                )
        # Preserve the strong tokenizer's fixed prompt width so weak and strong
        # rollout tensors can be merged without retokenizing sampled actions.
        return prompts

    def _pad_prompts_to_rollout_width(self, prompts: DataProto) -> DataProto:
        target_width = int(self.rollout_config.prompt_length)
        input_ids = prompts.batch["input_ids"]
        attention_mask = prompts.batch["attention_mask"]
        current_width = int(input_ids.shape[-1])

        if current_width > target_width:
            input_ids = input_ids[:, -target_width:]
            attention_mask = attention_mask[:, -target_width:]
        elif current_width < target_width:
            pad_width = target_width - current_width
            input_padding = torch.full(
                (input_ids.shape[0], pad_width),
                int(self.pad_token_id),
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            mask_padding = torch.zeros(
                (attention_mask.shape[0], pad_width),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            input_ids = torch.cat((input_padding, input_ids), dim=-1)
            attention_mask = torch.cat((mask_padding, attention_mask), dim=-1)

        return DataProto.from_dict(
            tensors={
                "input_ids": input_ids,
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
        sampling_params: dict[str, Any] = {"max_tokens": response_length, "logprobs": True}
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

    def _build_output(self, prompts: DataProto, outputs: list, elapsed: float, timing: Optional[dict] = None):
        result = super()._build_output(prompts=prompts, outputs=outputs, elapsed=elapsed, timing=timing)
        response_length = result.batch["responses"].shape[-1]
        weak_log_probs = torch.zeros((len(outputs), response_length), dtype=torch.float32)
        for row, output in enumerate(outputs):
            token_count = min(len(output.token_ids), response_length)
            if output.log_probs is None or len(output.log_probs) < token_count:
                raise RuntimeError("Weak vLLM rollout did not return one log-probability per sampled token.")
            if token_count:
                weak_log_probs[row, :token_count] = torch.as_tensor(
                    output.log_probs[:token_count], dtype=torch.float32
                )
        result.batch[S2L_SOURCE_LOG_PROBS_KEY] = weak_log_probs
        return result


class S2LPORayPPOTrainer(RayPPOTrainer):
    """DAPO/GRPO trainer with scheduled weak/strong mixed rollout groups."""

    _ROLLOUT_TENSOR_KEYS = (
        "prompts",
        "responses",
        "response_mask",
        "input_ids",
        "attention_mask",
        "position_ids",
    )

    def init_workers(self):
        super().init_workers()
        self._validate_s2l_config()
        actor_role = (
            Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        )
        resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
        self.weak_rollout_manager = S2LWeakVLLMRolloutManager(
            config=self.config,
            resource_pool=resource_pool,
            tokenizer=self.tokenizer,
        )
        self._strong_generate_sequences = self.async_rollout_manager.generate_sequences
        self.async_rollout_manager.generate_sequences = self._generate_s2l_sequences
        self._s2l_step_metrics: dict[str, float] = {}

    def _validate_s2l_config(self):
        cfg = self.config.algorithm.s2l_po
        group_size = int(cfg.group_size)
        rollout_n = int(self.config.actor_rollout_ref.rollout.n)
        if group_size != rollout_n:
            raise ValueError(
                "S2L-PO requires algorithm.s2l_po.group_size == actor_rollout_ref.rollout.n; "
                f"got {group_size} and {rollout_n}."
            )
        if str(self.config.actor_rollout_ref.rollout.name).lower() != "vllm":
            raise ValueError("S2L-PO currently requires actor_rollout_ref.rollout.name=vllm.")
        if bool(self.config.reward_model.get("use_reward_loop", False)):
            raise ValueError("S2L-PO currently supports the ordinary (non reward-loop) DAPO reward path.")
        if bool(self.config.actor_rollout_ref.rollout.multi_turn.enable):
            raise ValueError("S2L-PO currently supports complete single-turn rollouts only.")
        if bool(self.config.actor_rollout_ref.actor.get("calculate_entropy", False)):
            raise ValueError(
                "S2L-PO requires actor.calculate_entropy=false to avoid evaluating weak rows with old strong."
            )
        if float(self.config.actor_rollout_ref.actor.get("entropy_coeff", 0.0)) != 0.0:
            raise ValueError(
                "S2L-PO requires actor.entropy_coeff=0 to avoid evaluating weak rows with old strong."
            )
        rollout_correction = self.config.algorithm.get("rollout_correction", None)
        if rollout_correction is not None:
            rollout_is = rollout_correction.get("rollout_is", None)
            if rollout_is is not None and str(rollout_is).lower() not in {"", "none", "null"}:
                raise ValueError(
                    "S2L-PO forbids additional rollout importance-weight correction; set rollout_is=null."
                )
            if bool(rollout_correction.get("bypass_mode", False)):
                raise ValueError("S2L-PO requires rollout_correction.bypass_mode=false.")
        # Exercise validation for all scalar constraints before any rollout starts.
        s2l_phase_allocation(
            1,
            total_steps=max(int(self.total_training_steps), 1),
            t_mix=int(cfg.t_mix),
            num_phases=int(cfg.num_phases),
            group_size=group_size,
        )

    def _allocation(self, step: int) -> tuple[int, int, int]:
        cfg = self.config.algorithm.s2l_po
        return s2l_phase_allocation(
            step,
            total_steps=int(self.total_training_steps),
            t_mix=int(cfg.t_mix),
            num_phases=int(cfg.num_phases),
            group_size=int(cfg.group_size),
        )

    @staticmethod
    def _rollout_indices(batch_size: int, group_size: int, weak_count: int) -> tuple[np.ndarray, np.ndarray]:
        if batch_size % group_size != 0:
            raise ValueError(
                f"S2L rollout batch size {batch_size} is not divisible by group_size={group_size}."
            )
        slots = np.arange(batch_size, dtype=np.int64) % group_size
        return np.flatnonzero(slots < weak_count), np.flatnonzero(slots >= weak_count)

    @staticmethod
    def _complete_rollout_non_tensors(non_tensors: dict, batch_size: int) -> dict:
        """Add fields emitted by the ordinary single-turn AgentLoop path."""

        completed = {key: value.copy() for key, value in non_tensors.items()}
        if "multi_modal_inputs" not in completed:
            empty_multi_modal = np.empty(batch_size, dtype=object)
            empty_multi_modal[:] = [{} for _ in range(batch_size)]
            completed["multi_modal_inputs"] = empty_multi_modal
        if "__num_turns__" not in completed:
            completed["__num_turns__"] = np.ones(batch_size, dtype=np.int32)
        return completed

    def _generate_s2l_sequences(self, prompts: DataProto) -> DataProto:
        # Validation remains strong-only and follows the unmodified DAPO path.
        if bool(prompts.meta_info.get("validate", False)) or "global_steps" not in prompts.meta_info:
            return self._strong_generate_sequences(prompts)

        step = int(prompts.meta_info["global_steps"])
        cfg = self.config.algorithm.s2l_po
        group_size = int(cfg.group_size)
        phase, weak_count, strong_count = self._allocation(step)
        weak_indices, strong_indices = self._rollout_indices(len(prompts), group_size, weak_count)

        outputs: list[tuple[np.ndarray, DataProto]] = []
        if weak_indices.size:
            weak_output = self.weak_rollout_manager.generate_sequences(prompts.select_idxs(weak_indices))
            outputs.append((weak_indices, weak_output))
        if strong_indices.size:
            strong_output = self._strong_generate_sequences(prompts.select_idxs(strong_indices))
            outputs.append((strong_indices, strong_output))

        concat_indices = np.concatenate([indices for indices, _ in outputs])
        inverse_order = torch.from_numpy(np.argsort(concat_indices))
        tensors = {}
        for key in self._ROLLOUT_TENSOR_KEYS:
            if any(key not in output.batch for _, output in outputs):
                raise KeyError(f"S2L rollout output is missing required tensor {key!r}.")
            tensors[key] = torch.cat([output.batch[key] for _, output in outputs], dim=0)[inverse_order]

        response_shape = tensors["responses"].shape
        source_log_probs = torch.zeros(response_shape, dtype=torch.float32)
        is_weak = torch.zeros(response_shape[0], dtype=torch.bool)
        if weak_indices.size:
            weak_output = outputs[0][1]
            weak_concat_rows = torch.arange(weak_indices.size)
            weak_rows = inverse_order.new_tensor(weak_indices, dtype=torch.long)
            # ``weak_indices`` are positions in the final order, while weak output
            # rows are contiguous and preserve their input order.
            source_log_probs[weak_rows] = weak_output.batch[S2L_SOURCE_LOG_PROBS_KEY][weak_concat_rows]
            is_weak[weak_rows] = True

        tensors[S2L_SOURCE_LOG_PROBS_KEY] = source_log_probs
        tensors[S2L_IS_WEAK_KEY] = is_weak
        non_tensors = self._complete_rollout_non_tensors(prompts.non_tensor_batch, len(prompts))
        non_tensors[S2L_SOURCE_KEY] = np.where(is_weak.numpy(), "weak", "strong").astype(object)

        timing: dict[str, float] = {}
        for _, output in outputs:
            for key, value in output.meta_info.get("timing", {}).items():
                timing[key] = timing.get(key, 0.0) + float(value)
        self._s2l_step_metrics = {
            "s2l/phase": float(phase),
            "s2l/weak_rollouts_per_group": float(weak_count),
            "s2l/strong_rollouts_per_group": float(strong_count),
            "s2l/weak_fraction": float(weak_count / group_size),
        }
        return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info={"timing": timing})

    @staticmethod
    def _merge_source_denominators(
        source_log_probs: torch.Tensor,
        is_weak: torch.Tensor,
        strong_indices: torch.Tensor,
        strong_old_log_probs: Optional[torch.Tensor],
    ) -> torch.Tensor:
        denominators = source_log_probs.detach().clone().float()
        if strong_indices.numel():
            if strong_old_log_probs is None:
                raise ValueError("strong_old_log_probs is required when strong rollout rows are present.")
            denominators[strong_indices] = strong_old_log_probs.detach().float()
        if not torch.isfinite(denominators[is_weak]).all():
            raise ValueError("Weak source-policy log-probabilities contain non-finite values.")
        return denominators

    def _compute_old_log_prob(self, batch: DataProto, calculate_entropy: bool = False):
        if S2L_IS_WEAK_KEY not in batch.batch:
            return super()._compute_old_log_prob(batch, calculate_entropy=calculate_entropy)
        if calculate_entropy:
            raise ValueError("S2L-PO does not evaluate old-strong entropy on weak rollout rows.")

        is_weak = batch.batch[S2L_IS_WEAK_KEY].bool()
        strong_indices = torch.nonzero(~is_weak, as_tuple=False).flatten()
        strong_result = None
        old_log_prob_mfu = 0
        if strong_indices.numel():
            strong_batch = batch.select_idxs(strong_indices)
            strong_result, old_log_prob_mfu = super()._compute_old_log_prob(
                strong_batch,
                calculate_entropy=False,
            )

        denominators = self._merge_source_denominators(
            source_log_probs=batch.batch[S2L_SOURCE_LOG_PROBS_KEY],
            is_weak=is_weak,
            strong_indices=strong_indices,
            strong_old_log_probs=(
                strong_result.batch["old_log_probs"] if strong_result is not None else None
            ),
        )
        # Preserve the exact source-policy denominator in the batch for audits
        # and use that same tensor as PPO's token-level old_log_probs.
        batch.batch[S2L_SOURCE_LOG_PROBS_KEY] = denominators
        return DataProto.from_dict(tensors={"old_log_probs": denominators}), old_log_prob_mfu

    def _update_actor(self, batch: DataProto) -> DataProto:
        actor_output = super()._update_actor(batch)
        metric_store = actor_output.meta_info.setdefault("metrics", {})
        for key, value in self._s2l_step_metrics.items():
            metric_store[key] = [value]
        return actor_output
