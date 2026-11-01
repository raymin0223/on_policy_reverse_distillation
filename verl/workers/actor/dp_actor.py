from oprd.geometry import linear_schedule, support_indices, guidance_direction, precondition_loss_gradient

"\nSingle Process Actor\n"
import logging
import math
import os
from contextlib import nullcontext, suppress
from typing import Optional
import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor
import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import compute_kdrl_k2_loss
from verl.trainer.ppo.core_algos import (
    agg_loss,
    global_token_mean_batch_info,
    compute_direct_opd_topk_dense_reward,
    compute_direct_opd_topk_loss,
    compute_w2s_reverse_kl_loss,
    compute_distillation_loss,
    get_policy_loss_fn,
    kl_penalty,
    direct_opd_topk_reward_stats,
    select_w2s_proxy_topk,
    update_direct_opd_anchor_alpha,
)
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import (
    gather_outputs_and_unpad,
    slice_input_tensor,
    ulysses_pad,
    ulysses_pad_and_slice_inputs,
)
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor"]
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))
_EARLY_MARKER_FORMAT_BOXED = 3


def _topk_logsumexp_chunk_rows() -> int:
    raw_value = os.getenv("VERL_TOPK_LOGSUMEXP_CHUNK_ROWS", "1024")
    try:
        return max(int(raw_value), 0)
    except ValueError:
        logger.warning("Invalid VERL_TOPK_LOGSUMEXP_CHUNK_ROWS=%r; using 1024", raw_value)
        return 1024


def _oprd_entropy_chunk_rows() -> int:
    """Bound the temporary full-vocabulary FP32 tensor used by OPRD metrics."""
    raw_value = os.getenv("VERL_OPRD_ENTROPY_CHUNK_ROWS", "256")
    try:
        return max(int(raw_value), 1)
    except ValueError:
        logger.warning("Invalid VERL_OPRD_ENTROPY_CHUNK_ROWS=%r; using 256", raw_value)
        return 256


def _chunked_logsumexp_last_dim(tensor: torch.Tensor) -> torch.Tensor:
    chunk_rows = _topk_logsumexp_chunk_rows()
    if chunk_rows <= 0:
        return torch.logsumexp(tensor, dim=-1, keepdim=True)
    if tensor.dim() == 2:
        if tensor.size(0) <= chunk_rows:
            return torch.logsumexp(tensor, dim=-1, keepdim=True)
        chunks = [torch.logsumexp(chunk, dim=-1, keepdim=True) for chunk in tensor.split(chunk_rows, dim=0)]
        return torch.cat(chunks, dim=0)
    if tensor.dim() == 3:
        if tensor.size(0) * tensor.size(1) <= chunk_rows:
            return torch.logsumexp(tensor, dim=-1, keepdim=True)
        rows = []
        for batch_logits in tensor:
            chunks = [
                torch.logsumexp(chunk, dim=-1, keepdim=True)
                for chunk in batch_logits.split(chunk_rows, dim=0)
            ]
            rows.append(torch.cat(chunks, dim=0))
        return torch.stack(rows, dim=0)
    return torch.logsumexp(tensor, dim=-1, keepdim=True)


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(
        self,
        config: ActorConfig,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
        data_parallel_group=None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.data_parallel_group = data_parallel_group
        self.teacher_module: Optional[nn.Module] = None
        self.external_teacher_module: Optional[nn.Module] = None
        self.on_policy_teacher_module: Optional[nn.Module] = None
        self.oprd_base_module: Optional[nn.Module] = None
        self.oprd_student_base_module: Optional[nn.Module] = None
        self.w2s_student_base_module: Optional[nn.Module] = None
        self._direct_opd_anchor_alpha: Optional[float] = None
        self._direct_opd_shift_stats: Optional[torch.Tensor] = None
        initial_distillation_cfg = getattr(self.config, "distillation", None)
        if initial_distillation_cfg is not None and initial_distillation_cfg.get(
            "delta_anchor_adaptive_alpha_enabled", False
        ):
            self._direct_opd_anchor_alpha = self._direct_opd_alpha_config(initial_distillation_cfg)[0]
        role = "Ref" if actor_optimizer is None else "Actor"
        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1
        self.use_dynamic_bsz = self.config.get("use_dynamic_bsz", False)
        global_batch_info = self.config.get("global_batch_info", None)
        if global_batch_info is None:
            global_batch_info = {}
            with suppress(Exception):
                self.config.global_batch_info = global_batch_info
        if (
            actor_optimizer is not None
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and (torch.distributed.get_world_size() > 1)
        ):
            global_batch_info.setdefault("auto_dp_sync", True)
        self.use_prefix_grouper = self.config.get("use_prefix_grouper", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_prefix_grouper={self.use_prefix_grouper}")
        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits
        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)
            else entropy_from_logits
        )
        self.device_name = get_device_name()
        self.param_dtype = PrecisionType.to_dtype(self.config.fsdp_config.get("dtype", "bfloat16"))
        if self.param_dtype == torch.float16:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            self.scaler = ShardedGradScaler(growth_interval=400)
        else:
            self.scaler = None
        if self.config.get("calculate_sum_pi_squared", False):
            self.calculate_sum_pi_squared_from_logits = (
                torch.compile(verl_F.calculate_sum_pi_squared_from_logits, dynamic=True)
                if self.config.get("use_torch_compile", True)
                else verl_F.calculate_sum_pi_squared_from_logits
            )
            assert not (self.use_fused_kernels or self.use_prefix_grouper), (
                f"calculate_sum_pi_squared is not supported with self.use_fused_kernels={self.use_fused_kernels!r} or self.use_prefix_grouper={self.use_prefix_grouper!r} for now."
            )

    @staticmethod
    def _direct_opd_alpha_config(distillation_cfg) -> tuple[float, float, float, float]:
        mode = str(distillation_cfg.get("direct_opd_mode", "topk")).strip().lower().replace("-", "_")
        default_alpha_init = 2.5 if mode in {"topk", "dense", "dense_topk"} else 1.0
        direct_kl_coef = distillation_cfg.get("direct_kl_coef", None)
        alpha_init = float(
            direct_kl_coef
            if direct_kl_coef is not None
            else distillation_cfg.get("delta_anchor_alpha_init", default_alpha_init)
        )
        epsilon = float(distillation_cfg.get("delta_anchor_alpha_epsilon", 0.01))
        alpha_min = float(distillation_cfg.get("delta_anchor_alpha_min", 0.5))
        alpha_max = float(distillation_cfg.get("delta_anchor_alpha_max", 2.5))
        update_direct_opd_anchor_alpha(alpha_init, 0.0, epsilon, alpha_min, alpha_max)
        return (alpha_init, epsilon, alpha_min, alpha_max)

    @staticmethod
    def _direct_opd_mode(distillation_cfg) -> str:
        mode = str(distillation_cfg.get("direct_opd_mode", "topk")).strip().lower().replace("-", "_")
        mode = {"dense": "topk", "dense_topk": "topk", "sampled_token": "sampled"}.get(mode, mode)
        if mode != "topk":
            raise ValueError(f"distillation.direct_opd_mode must be 'topk', got {mode!r}.")
        return mode

    @staticmethod
    def _direct_opd_topk_config(distillation_cfg) -> tuple[int, str]:
        top_k = int(distillation_cfg.get("direct_opd_top_k", 16))
        source = str(distillation_cfg.get("direct_opd_top_k_source", "student")).strip().lower()
        if top_k <= 0:
            raise ValueError(f"distillation.direct_opd_top_k must be positive, got {top_k}.")
        if source != "student":
            raise ValueError(
                f"Paper-faithful Direct-OPD Top-K requires direct_opd_top_k_source='student', got {source!r}."
            )
        return (top_k, source)

    @staticmethod
    def _w2s_opd_config(distillation_cfg) -> tuple[int, str, str, float]:
        top_k = int(distillation_cfg.get("w2s_opd_teacher_top_k", 32))
        source = (
            str(distillation_cfg.get("w2s_opd_top_k_source", "proxy_teacher"))
            .strip()
            .lower()
            .replace("-", "_")
        )
        divergence = (
            str(distillation_cfg.get("w2s_opd_divergence", "reverse_kl")).strip().lower().replace("-", "_")
        )
        amplification = float(distillation_cfg.get("w2s_amplification", 1.0))
        if top_k != 32:
            raise ValueError(
                f"Paper-faithful W2S-OPD requires Top-K support 32 (w2s_opd_teacher_top_k=32), got {top_k}."
            )
        if source != "proxy_teacher":
            raise ValueError(
                f"Paper-faithful W2S-OPD requires w2s_opd_top_k_source='proxy_teacher', got {source!r}."
            )
        if divergence not in {"reverse_kl", "rkl"}:
            raise ValueError(
                f"Paper-faithful W2S-OPD requires w2s_opd_divergence='reverse_kl', got {divergence!r}."
            )
        if not math.isfinite(amplification) or amplification < 0.0:
            raise ValueError(
                f"W2S-OPD w2s_amplification must be finite and non-negative, got {amplification}."
            )
        return (top_k, source, "reverse_kl", amplification)

    def _prepare_direct_opd_anchor_alpha(self, distillation_cfg) -> float:
        alpha_init, _, _, _ = self._direct_opd_alpha_config(distillation_cfg)
        if self._direct_opd_anchor_alpha is None:
            self._direct_opd_anchor_alpha = alpha_init
        self._direct_opd_shift_stats = None
        return float(self._direct_opd_anchor_alpha)

    def _accumulate_direct_opd_dense_reward(
        self, dense_reward: torch.Tensor, response_mask: torch.Tensor
    ) -> None:
        reward_sum, reward_count = direct_opd_topk_reward_stats(dense_reward, response_mask)
        stats = torch.stack((reward_sum, reward_count))
        self._direct_opd_shift_stats = (
            stats if self._direct_opd_shift_stats is None else self._direct_opd_shift_stats + stats
        )

    def _finalize_direct_opd_anchor_alpha(self, distillation_cfg) -> dict[str, float]:
        if self._direct_opd_anchor_alpha is None:
            raise RuntimeError("Direct-OPD adaptive alpha was finalized before initialization.")
        stats = self._direct_opd_shift_stats
        if stats is None:
            stats = torch.zeros(2, dtype=torch.float32, device=get_device_id())
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and (torch.distributed.get_world_size() > 1)
        ):
            torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
        shift_sum, token_count = stats.tolist()
        if token_count <= 0.0:
            raise RuntimeError("Direct-OPD adaptive alpha received no sampled distillation tokens.")
        mean_shift = float(shift_sum / token_count)
        alpha_used = float(self._direct_opd_anchor_alpha)
        _, epsilon, alpha_min, alpha_max = self._direct_opd_alpha_config(distillation_cfg)
        alpha_next = update_direct_opd_anchor_alpha(
            alpha_used, mean_shift, epsilon=epsilon, alpha_min=alpha_min, alpha_max=alpha_max
        )
        self._direct_opd_anchor_alpha = alpha_next
        self._direct_opd_shift_stats = None
        return {
            "distillation/direct_opd_anchor_alpha": alpha_used,
            "distillation/direct_opd_anchor_alpha_next": alpha_next,
            "distillation/direct_opd_mean_teacher_shift": mean_shift,
            "distillation/direct_opd_shift_token_count": float(token_count),
        }

    def export_direct_opd_anchor_alpha_state(self) -> dict[str, float | int | str]:
        if self._direct_opd_anchor_alpha is None:
            raise RuntimeError("Direct-OPD adaptive alpha state has not been initialized.")
        distillation_cfg = getattr(self.config, "distillation", None)
        if distillation_cfg is None:
            raise RuntimeError("Cannot export Direct-OPD alpha without distillation config.")
        return {
            "format_version": 2,
            "mode": self._direct_opd_mode(distillation_cfg),
            "alpha": float(self._direct_opd_anchor_alpha),
        }

    def load_direct_opd_anchor_alpha_state(self, state: dict[str, object]) -> None:
        format_version = int(state.get("format_version", -1))
        if format_version not in {1, 2}:
            raise ValueError(f"Unsupported Direct-OPD alpha state: {state!r}.")
        distillation_cfg = getattr(self.config, "distillation", None)
        if distillation_cfg is None:
            raise ValueError("Cannot restore Direct-OPD alpha without distillation config.")
        configured_mode = self._direct_opd_mode(distillation_cfg)
        saved_mode = "sampled" if format_version == 1 else str(state.get("mode", "")).strip().lower()
        if saved_mode != configured_mode:
            raise ValueError(
                f"Refusing to restore Direct-OPD alpha across algorithm modes: checkpoint={saved_mode!r}, configured={configured_mode!r}."
            )
        _, epsilon, alpha_min, alpha_max = self._direct_opd_alpha_config(distillation_cfg)
        alpha = float(state["alpha"])
        update_direct_opd_anchor_alpha(alpha, 0.0, epsilon, alpha_min, alpha_max)
        self._direct_opd_anchor_alpha = alpha

    def _update_teacher(self) -> None:
        """Full-parameter EMA for OPSD's self teacher; frozen teachers stay fixed."""
        cfg = self.config.distillation
        if self.config.policy_loss.loss_mode not in {"opsd", "guided_opsd"}:
            return
        if cfg.teacher_regularization != "ema" or cfg.teacher_update_rate == 0:
            return
        if self.teacher_module is None or self.teacher_module is self.actor_module:
            raise ValueError("OPSD EMA requires a separate full teacher model")
        rate = cfg.teacher_update_rate
        with torch.no_grad():
            for teacher, student in zip(
                self.teacher_module.parameters(), self.actor_module.parameters(), strict=True
            ):
                teacher.mul_(1.0 - rate).add_(
                    student.to(device=teacher.device, dtype=teacher.dtype), alpha=rate
                )

    @staticmethod
    def _has_non_empty_multi_modal_inputs(multi_modal_inputs) -> bool:
        if multi_modal_inputs is None:
            return False
        for inputs in multi_modal_inputs:
            if inputs is None:
                continue
            inputs = getattr(inputs, "data", inputs)
            if isinstance(inputs, dict):
                if not inputs:
                    continue
                for value in inputs.values():
                    if value is None:
                        continue
                    if isinstance(value, torch.Tensor) and value.numel() == 0:
                        continue
                    return True
            else:
                return True
        return False

    def _compute_kdrl_beta(
        self, meta_info: Optional[dict] = None, *, global_step: Optional[int] = None
    ) -> float:
        """Compute KDRL beta from the trainer's one-based global step."""
        distillation_cfg = getattr(self.config, "distillation", None)
        if distillation_cfg is None:
            return 0.0
        beta_init = float(distillation_cfg.get("kdrl_beta_init", 0.005))
        delta = float(distillation_cfg.get("kdrl_delta", 0.0001))
        beta_min = float(distillation_cfg.get("kdrl_beta_min", 0.0))
        meta_info = meta_info or {}
        if global_step is None:
            global_step = int(meta_info.get("global_steps", 0) or 0)
        schedule_step = max(int(global_step) - 1, 0)
        return max(beta_init - delta * schedule_step, beta_min)

    @staticmethod
    def _normalize_oprd_invalid_format_mask(
        invalid_format_mask, *, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        if invalid_format_mask is None:
            raise ValueError(
                "OPRD oprd_skip_invalid_format=true requires batch['oprd_invalid_format_mask']; check that the configured reward extra key is produced before actor update."
            )
        if not torch.is_tensor(invalid_format_mask):
            invalid_format_mask = torch.as_tensor(invalid_format_mask)
        invalid_format_mask = invalid_format_mask.to(device=device, dtype=torch.bool).reshape(-1)
        if invalid_format_mask.numel() != batch_size:
            raise ValueError(
                f"OPRD oprd_skip_invalid_format expected oprd_invalid_format_mask with {batch_size} values, got {invalid_format_mask.numel()}."
            )
        return invalid_format_mask

    def _forward_micro_batch(
        self,
        micro_batch: dict[str, torch.Tensor],
        temperature: float,
        calculate_entropy: bool = False,
        return_all_logps: bool = False,
        distill_topk: Optional[int] = None,
        topk_indices: Optional[torch.Tensor] = None,
        topk_indices_only: bool = False,
        return_model_topk_indices: bool = False,
        detach_topk_logps: bool = False,
        module: Optional[nn.Module] = None,
        oprd_hook_config: Optional[dict] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Returns:
            dict[str, torch.Tensor]:
                log_probs: (bs, response_len)
                if calculate_entropy is True:
                    entropys: (bs, response_len)
                if calculate_sum_pi_squared is False:
                    sum_pi_squared: (bs, response_len)
                if distill_topk or topk_indices is set:
                    topk_logps: (bs, response_len, k)
                    topk_indices: (bs, response_len, k)
                    model_topk_indices: (bs, response_len, distill_topk) when requested
        """
        calculate_sum_pi_squared = self.config.get("calculate_sum_pi_squared", False)
        sum_pi_squared_checkpointing = self.config.get("sum_pi_squared_checkpointing", False)
        use_topk = distill_topk is not None or topk_indices is not None
        compute_all_logps = return_all_logps and (not use_topk)
        return_topk_indices = use_topk and topk_indices is None
        return_model_topk_indices = return_model_topk_indices and use_topk
        if topk_indices_only and (not use_topk):
            raise ValueError("topk_indices_only=True requires distill_topk or topk_indices.")
        if topk_indices_only and (not return_topk_indices):
            raise ValueError("topk_indices_only=True requires topk_indices to be computed in this call.")
        if (return_all_logps or use_topk) and self.use_fused_kernels:
            raise ValueError("Logit distillation requires disabling fused kernels.")
        if oprd_hook_config is not None and self.use_fused_kernels:
            raise NotImplementedError("OPRD hook requires disabling fused kernels.")
        model = module or self.actor_module
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])
        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:
                position_ids = position_ids.transpose(0, 1)
            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)
                is_mask_all_zero = attention_mask.sum() == 0
                if is_mask_all_zero:
                    input_ids_rmpad = torch.zeros(
                        (1, self.ulysses_sequence_parallel_size),
                        device=input_ids.device,
                        dtype=input_ids.dtype,
                    )
                    if position_ids.dim() == 3:
                        position_ids_rmpad = torch.zeros(
                            (position_ids.shape[0], 1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )
                    else:
                        position_ids_rmpad = torch.zeros(
                            (1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )
                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)
                response_pred_mask = torch.zeros(
                    (batch_size, seqlen), dtype=torch.bool, device=attention_mask.device
                )
                response_pred_mask[:, -response_length - 1 : -1] = attention_mask[
                    :, -response_length - 1 : -1
                ].bool()
                response_pred_mask_rmpad = index_first_axis(
                    rearrange(response_pred_mask.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).squeeze(-1)
                response_position_rmpad = None
                sample_id_rmpad = None
                if oprd_hook_config is not None:
                    response_position = torch.full(
                        (batch_size, seqlen), -1, dtype=torch.long, device=attention_mask.device
                    )
                    response_position[:, -response_length - 1 : -1] = torch.arange(
                        response_length, dtype=torch.long, device=attention_mask.device
                    ).unsqueeze(0)
                    response_position_rmpad = index_first_axis(
                        rearrange(response_position.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).squeeze(-1)
                    sample_id = torch.arange(batch_size, dtype=torch.long, device=attention_mask.device)
                    sample_id = sample_id.unsqueeze(1).expand(batch_size, seqlen)
                    sample_id_rmpad = index_first_axis(
                        rearrange(sample_id.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).squeeze(-1)
                response_advantages_rmpad = None
                if oprd_hook_config is not None:
                    oprd_advantages = oprd_hook_config.get("advantages")
                    if oprd_advantages is None:
                        pass
                    else:
                        if oprd_advantages.shape != (batch_size, response_length):
                            raise ValueError(
                                f"OPRD expected advantages with shape {(batch_size, response_length)}, got {tuple(oprd_advantages.shape)}."
                            )
                        full_response_advantages = torch.zeros(
                            (batch_size, seqlen), dtype=oprd_advantages.dtype, device=oprd_advantages.device
                        )
                        full_response_advantages[:, -response_length - 1 : -1] = oprd_advantages
                        response_advantages_rmpad = index_first_axis(
                            rearrange(full_response_advantages.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                        ).squeeze(-1)
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(getattr(model, "module", model).config, "vision_config")
                    if is_vlm_model:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )
                    response_pred_mask_rmpad, _, _ = ulysses_pad_and_slice_inputs(
                        response_pred_mask_rmpad.unsqueeze(0).to(dtype=input_ids_rmpad.dtype),
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )
                    response_pred_mask_rmpad = response_pred_mask_rmpad.squeeze(0).bool()
                    if response_position_rmpad is not None:
                        response_position_rmpad, _, _ = ulysses_pad_and_slice_inputs(
                            response_position_rmpad.unsqueeze(0),
                            position_ids_rmpad=None,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                        response_position_rmpad = response_position_rmpad.squeeze(0)
                    if sample_id_rmpad is not None:
                        sample_id_rmpad, _, _ = ulysses_pad_and_slice_inputs(
                            sample_id_rmpad.unsqueeze(0),
                            position_ids_rmpad=None,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                        sample_id_rmpad = sample_id_rmpad.squeeze(0)
                    if response_advantages_rmpad is not None:
                        response_advantages_rmpad, _, _ = ulysses_pad_and_slice_inputs(
                            response_advantages_rmpad.unsqueeze(0),
                            position_ids_rmpad=None,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                        response_advantages_rmpad = response_advantages_rmpad.squeeze(0)
                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True
                output = model(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )
                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)
                    entropy_rmpad = output.entropy.squeeze(0)
                else:
                    logits_rmpad = output.logits.squeeze(0)
                    logits_rmpad.div_(temperature)
                    if oprd_hook_config is not None and module is None:
                        self._register_oprd_hook_rmpad(
                            logits_rmpad=logits_rmpad,
                            input_ids_rmpad=input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            input_ids_rmpad_rolled=input_ids_rmpad_rolled,
                            response_pred_mask_rmpad=response_pred_mask_rmpad,
                            response_position_rmpad=response_position_rmpad,
                            sample_id_rmpad=sample_id_rmpad,
                            response_advantages_rmpad=response_advantages_rmpad,
                            temperature=temperature,
                            oprd_hook_config=oprd_hook_config,
                        )
                    all_logps_rmpad = torch.log_softmax(logits_rmpad, dim=-1) if compute_all_logps else None
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad, labels=input_ids_rmpad_rolled, inplace_backward=inplace_backward
                    )
                    if calculate_entropy:
                        entropy_rmpad = (
                            self.compute_entropy_from_logits(logits_rmpad)
                            if not self.config.entropy_checkpointing
                            else torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )
                        )
                    if use_topk:
                        model_topk_indices_rmpad = None
                        logits_for_topk_rmpad = logits_rmpad.detach() if detach_topk_logps else logits_rmpad
                        if topk_indices is None:
                            topk = min(distill_topk, logits_for_topk_rmpad.shape[-1])
                            if topk_indices_only:
                                topk_indices_rmpad = torch.topk(logits_for_topk_rmpad, topk, dim=-1).indices
                                topk_logits_rmpad = None
                            else:
                                topk_logits_rmpad, topk_indices_rmpad = torch.topk(
                                    logits_for_topk_rmpad, topk, dim=-1
                                )
                            if return_model_topk_indices:
                                model_topk_indices_rmpad = topk_indices_rmpad
                        else:
                            topk = topk_indices.size(-1)
                            full_topk_indices = torch.zeros(
                                batch_size, seqlen, topk, device=topk_indices.device, dtype=topk_indices.dtype
                            )
                            full_topk_indices[:, -response_length - 1 : -1, :] = topk_indices
                            topk_indices_rmpad = index_first_axis(
                                rearrange(full_topk_indices, "b s k -> (b s) k"), indices
                            )
                            if self.use_ulysses_sp:
                                topk_indices_rmpad = slice_input_tensor(
                                    topk_indices_rmpad.unsqueeze(0), dim=1, padding=True
                                ).squeeze(0)
                            topk_logits_rmpad = (
                                None
                                if topk_indices_only
                                else torch.gather(logits_for_topk_rmpad, dim=-1, index=topk_indices_rmpad)
                            )
                            if return_model_topk_indices:
                                model_topk = min(
                                    distill_topk if distill_topk is not None else topk,
                                    logits_for_topk_rmpad.shape[-1],
                                )
                                model_topk_indices_rmpad = torch.topk(
                                    logits_for_topk_rmpad, model_topk, dim=-1
                                ).indices
                        if not topk_indices_only:
                            topk_logps_rmpad = torch.zeros_like(topk_logits_rmpad)
                            if response_pred_mask_rmpad.any():
                                response_rows = response_pred_mask_rmpad.nonzero(as_tuple=False).flatten()
                                chunk_rows = _topk_logsumexp_chunk_rows()
                                if chunk_rows <= 0:
                                    row_chunks = (response_rows,)
                                else:
                                    row_chunks = response_rows.split(chunk_rows, dim=0)
                                for row_chunk in row_chunks:
                                    resp_logits_chunk = logits_for_topk_rmpad.index_select(0, row_chunk)
                                    resp_topk_logits_chunk = topk_logits_rmpad.index_select(0, row_chunk)
                                    resp_logsumexp_chunk = torch.logsumexp(
                                        resp_logits_chunk, dim=-1, keepdim=True
                                    )
                                    topk_logps_rmpad[row_chunk] = (
                                        resp_topk_logits_chunk - resp_logsumexp_chunk
                                    ).to(dtype=topk_logps_rmpad.dtype)
                    if calculate_sum_pi_squared:
                        sum_pi_squared_rmpad = (
                            self.calculate_sum_pi_squared_from_logits(logits_rmpad)
                            if not sum_pi_squared_checkpointing
                            else torch.utils.checkpoint.checkpoint(
                                self.calculate_sum_pi_squared_from_logits, logits_rmpad
                            )
                        )
                if self.use_ulysses_sp:
                    log_probs = gather_outputs_and_unpad(
                        log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                        )
                    if use_topk:
                        if not topk_indices_only:
                            topk_logps_rmpad = gather_outputs_and_unpad(
                                topk_logps_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                            )
                        if return_topk_indices:
                            topk_indices_rmpad = gather_outputs_and_unpad(
                                topk_indices_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                            )
                        if return_model_topk_indices:
                            model_topk_indices_rmpad = gather_outputs_and_unpad(
                                model_topk_indices_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                            )
                    if calculate_sum_pi_squared:
                        sum_pi_squared_rmpad = gather_outputs_and_unpad(
                            sum_pi_squared_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                        )
                if is_mask_all_zero:
                    log_probs = log_probs[:0]
                    if calculate_entropy:
                        entropy_rmpad = entropy_rmpad[:0]
                    if compute_all_logps:
                        all_logps_rmpad = all_logps_rmpad[:0]
                    if use_topk:
                        if not topk_indices_only:
                            topk_logps_rmpad = topk_logps_rmpad[:0]
                        if return_topk_indices:
                            topk_indices_rmpad = topk_indices_rmpad[:0]
                        if return_model_topk_indices:
                            model_topk_indices_rmpad = model_topk_indices_rmpad[:0]
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                if calculate_sum_pi_squared:
                    full_sum_pi_squared = pad_input(
                        hidden_states=sum_pi_squared_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                if compute_all_logps:
                    full_all_logps = pad_input(
                        hidden_states=all_logps_rmpad, indices=indices, batch=batch_size, seqlen=seqlen
                    )
                if use_topk:
                    if not topk_indices_only:
                        full_topk_logps = pad_input(
                            hidden_states=topk_logps_rmpad, indices=indices, batch=batch_size, seqlen=seqlen
                        )
                    if return_model_topk_indices:
                        full_model_topk_indices = pad_input(
                            hidden_states=model_topk_indices_rmpad,
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen,
                        )
                    if return_topk_indices:
                        full_topk_indices = pad_input(
                            hidden_states=topk_indices_rmpad, indices=indices, batch=batch_size, seqlen=seqlen
                        )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
                )
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]
                if calculate_sum_pi_squared:
                    sum_pi_squared = full_sum_pi_squared.squeeze(-1)[:, -response_length - 1 : -1]
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]
                if compute_all_logps:
                    all_logps = full_all_logps[:, -response_length - 1 : -1, :]
                if use_topk:
                    if not topk_indices_only:
                        topk_logps = full_topk_logps[:, -response_length - 1 : -1, :]
                    if return_topk_indices:
                        topk_indices = full_topk_indices[:, -response_length - 1 : -1, :]
                    if return_model_topk_indices:
                        model_topk_indices = full_model_topk_indices[:, -response_length - 1 : -1, :]
            else:
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )
                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]
                else:
                    logits = output.logits
                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]
                    if oprd_hook_config is not None and module is None:
                        self._register_oprd_hook_non_rmpad(
                            logits=logits,
                            micro_batch=micro_batch,
                            temperature=temperature,
                            response_mask=micro_batch.get(
                                "response_mask", torch.ones_like(micro_batch["responses"])
                            ),
                            oprd_hook_config=oprd_hook_config,
                        )
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if compute_all_logps:
                        all_logps = torch.log_softmax(logits, dim=-1)
                    if use_topk:
                        model_topk_indices = None
                        logits_for_topk = logits.detach() if detach_topk_logps else logits
                        if topk_indices is None:
                            topk = min(distill_topk, logits_for_topk.size(-1))
                            if topk_indices_only:
                                topk_indices = torch.topk(logits_for_topk, topk, dim=-1).indices
                                topk_logits = None
                            else:
                                topk_logits, topk_indices = torch.topk(logits_for_topk, topk, dim=-1)
                            if return_model_topk_indices:
                                model_topk_indices = topk_indices
                        else:
                            topk = topk_indices.size(-1)
                            topk_logits = (
                                None
                                if topk_indices_only
                                else torch.gather(logits_for_topk, dim=-1, index=topk_indices)
                            )
                            if return_model_topk_indices:
                                model_topk = min(
                                    distill_topk if distill_topk is not None else topk,
                                    logits_for_topk.size(-1),
                                )
                                model_topk_indices = torch.topk(logits_for_topk, model_topk, dim=-1).indices
                        if not topk_indices_only:
                            logsumexp = _chunked_logsumexp_last_dim(logits_for_topk)
                            topk_logps = topk_logits - logsumexp
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)
                    if calculate_sum_pi_squared:
                        sum_pi_squared = (
                            self.calculate_sum_pi_squared_from_logits(logits)
                            if not sum_pi_squared_checkpointing
                            else torch.utils.checkpoint.checkpoint(
                                self.calculate_sum_pi_squared_from_logits, logits
                            )
                        )
            outputs = {"log_probs": log_probs}
            if calculate_entropy:
                outputs["entropys"] = entropy
            if calculate_sum_pi_squared:
                outputs["sum_pi_squared"] = sum_pi_squared
            if compute_all_logps:
                outputs["all_logps"] = all_logps
            if use_topk:
                if not topk_indices_only:
                    outputs["topk_logps"] = topk_logps
                if return_topk_indices:
                    outputs["topk_indices"] = topk_indices
                if return_model_topk_indices:
                    outputs["model_topk_indices"] = model_topk_indices
            return outputs

    def _clip_grad_norm_for(self, module: nn.Module, optimizer: torch.optim.Optimizer):
        assert self.config.grad_clip is not None
        if optimizer is None:
            raise ValueError("optimizer must be initialized before stepping.")
        if self.scaler is not None:
            self.scaler.unscale_(optimizer)
        if isinstance(module, FSDP):
            grad_norm = module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(module.parameters(), max_norm=self.config.grad_clip)
        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()
        return grad_norm

    def _optimizer_step_for(self, module: nn.Module, optimizer: torch.optim.Optimizer):
        grad_norm = self._clip_grad_norm_for(module, optimizer)
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            optimizer.zero_grad()
            return grad_norm
        if self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
        return grad_norm

    def _optimizer_step(self):
        return self._optimizer_step_for(self.actor_module, self.actor_optimizer)

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy: bool = False) -> dict[str, torch.Tensor]:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            dict[str, torch.Tensor]: a dict containing keys
                - ``log_probs``: tensor of shape [batch_size, response_length]. torch.float32.
                - ``entropys``: tensor of shape [batch_size, response_length]. torch.float32.
                - ``sum_pi_squared``: tensor of shape [batch_size, response_length]. torch.float32.
        """
        calculate_sum_pi_squared = self.config.get("calculate_sum_pi_squared", False)
        self.actor_module.eval()
        log_prob_target = "actor".lower()
        distillation_cfg = getattr(self.config, "distillation", None)
        target_module = self.actor_module
        target_scope = lambda _module: nullcontext()
        target_was_training = target_module.training
        target_module.eval()
        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        pad_token_id = data.meta_info.get("pad_token_id", 0)
        has_multi_modal_inputs = self._has_non_empty_multi_modal_inputs(
            data.non_tensor_batch.get("multi_modal_inputs")
        )
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)
        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)
        log_probs_lst = []
        entropy_lst = []
        sum_pi_squared_lst = []
        try:
            for micro_batch in micro_batches:
                micro_batch = micro_batch.to(get_device_id())
                model_inputs = {
                    **micro_batch.batch,
                    **micro_batch.non_tensor_batch,
                    "pad_token_id": pad_token_id,
                }
                with torch.no_grad():
                    with target_scope(target_module):
                        outputs = self._forward_micro_batch(
                            model_inputs,
                            temperature=temperature,
                            calculate_entropy=calculate_entropy,
                            module=target_module,
                        )
                log_probs_lst.append(outputs["log_probs"])
                if calculate_entropy:
                    entropy_lst.append(outputs["entropys"])
                if calculate_sum_pi_squared:
                    sum_pi_squared_lst.append(outputs["sum_pi_squared"])
        finally:
            if target_module is not self.actor_module and target_was_training:
                target_module.train()
        log_probs = torch.concat(log_probs_lst, dim=0)
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if calculate_sum_pi_squared:
            sum_pi_squared = torch.concat(sum_pi_squared_lst, dim=0)
        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)
            if calculate_sum_pi_squared:
                sum_pi_squared = restore_dynamic_batch(sum_pi_squared, batch_idx_list)
        outputs = {"log_probs": log_probs}
        if calculate_entropy:
            outputs["entropys"] = entropys
        if calculate_sum_pi_squared:
            outputs["sum_pi_squared"] = sum_pi_squared
        return outputs

    def _prepare_direct_opd_topk_batch(self, data: DataProto, distillation_cfg) -> dict[str, float]:
        """Freeze student Top-K rewards and update alpha before this batch's actor step."""
        top_k, _ = self._direct_opd_topk_config(distillation_cfg)
        if self.use_dynamic_bsz:
            raise ValueError("Direct-OPD Top-K requires actor.use_dynamic_bsz=false.")
        if self.oprd_base_module is None:
            raise ValueError(
                "Direct-OPD Top-K requires the frozen pre-RL teacher in distillation.delta_teacher_base_model_path."
            )
        teacher_rl_module = self.external_teacher_module or self.on_policy_teacher_module
        if teacher_rl_module is None:
            raise ValueError("Direct-OPD Top-K requires a frozen post-RL external teacher module.")
        required_keys = {"responses", "response_mask", "input_ids", "attention_mask", "position_ids"}
        missing = required_keys - set(data.batch.keys())
        if missing:
            raise KeyError(f"Direct-OPD Top-K prepass is missing batch keys: {sorted(missing)}")
        micro_batch_size = int(self.config.ppo_micro_batch_size_per_gpu)
        if micro_batch_size <= 0:
            raise ValueError(
                f"Direct-OPD Top-K requires a positive actor.ppo_micro_batch_size_per_gpu, got {micro_batch_size}."
            )
        temperature = float(data.meta_info["temperature"])
        pad_token_id = data.meta_info.get("pad_token_id", 0)
        old_topk_ids: list[torch.Tensor] = []
        dense_rewards: list[torch.Tensor] = []
        max_weight_sum_error = 0.0
        teacher_delta_abs_sum = 0.0
        teacher_delta_count = 0
        actor_was_training = self.actor_module.training
        teacher_rl_was_training = teacher_rl_module.training
        teacher_ref_was_training = self.oprd_base_module.training
        self.actor_module.eval()
        teacher_rl_module.eval()
        self.oprd_base_module.eval()
        try:
            for micro_batch in data.split(micro_batch_size):
                micro_batch = micro_batch.to(get_device_id())
                model_inputs = {
                    **micro_batch.batch,
                    **micro_batch.non_tensor_batch,
                    "pad_token_id": pad_token_id,
                }
                teacher_input_ids = model_inputs.get("teacher_input_ids", model_inputs["input_ids"])
                teacher_attention_mask = model_inputs.get(
                    "teacher_attention_mask", model_inputs["attention_mask"]
                )
                teacher_position_ids = model_inputs.get("teacher_position_ids", model_inputs["position_ids"])
                teacher_responses = model_inputs.get("teacher_responses", model_inputs["responses"])
                teacher_inputs = {
                    "responses": teacher_responses,
                    "input_ids": teacher_input_ids,
                    "attention_mask": teacher_attention_mask,
                    "position_ids": teacher_position_ids,
                    "pad_token_id": pad_token_id,
                }
                with torch.no_grad():
                    student_old_outputs = self._forward_micro_batch(
                        model_inputs,
                        temperature=temperature,
                        calculate_entropy=False,
                        distill_topk=top_k,
                        detach_topk_logps=True,
                    )
                    student_topk_ids = student_old_outputs["topk_indices"]
                    student_old_topk_logps = student_old_outputs["topk_logps"]
                    teacher_rl_outputs = self._forward_micro_batch(
                        teacher_inputs,
                        temperature=temperature,
                        calculate_entropy=False,
                        topk_indices=student_topk_ids,
                        detach_topk_logps=True,
                        module=teacher_rl_module,
                    )
                    teacher_ref_outputs = self._forward_micro_batch(
                        teacher_inputs,
                        temperature=temperature,
                        calculate_entropy=False,
                        topk_indices=student_topk_ids,
                        detach_topk_logps=True,
                        module=self.oprd_base_module,
                    )
                    dense_reward, student_topk_weight, teacher_delta = compute_direct_opd_topk_dense_reward(
                        student_old_topk_log_probs=student_old_topk_logps,
                        teacher_rl_topk_log_probs=teacher_rl_outputs["topk_logps"],
                        teacher_ref_topk_log_probs=teacher_ref_outputs["topk_logps"],
                    )
                    response_mask = model_inputs["response_mask"]
                    self._accumulate_direct_opd_dense_reward(dense_reward, response_mask)
                    max_weight_sum_error = max(
                        max_weight_sum_error, (student_topk_weight.sum(dim=-1) - 1.0).abs().max().item()
                    )
                    active_delta = teacher_delta[
                        response_mask.to(dtype=torch.bool).unsqueeze(-1).expand_as(teacher_delta)
                    ]
                    if active_delta.numel() > 0:
                        teacher_delta_abs_sum += active_delta.abs().sum().item()
                        teacher_delta_count += active_delta.numel()
                    old_topk_ids.append(student_topk_ids.detach().cpu())
                    dense_rewards.append(dense_reward.detach().cpu())
                del micro_batch
        finally:
            if actor_was_training:
                self.actor_module.train()
            if teacher_rl_was_training:
                teacher_rl_module.train()
            if teacher_ref_was_training:
                self.oprd_base_module.train()
        if not old_topk_ids:
            raise RuntimeError("Direct-OPD Top-K prepass received an empty rollout batch.")
        if getattr(data.batch, "is_locked", False):
            data.batch = data.batch.copy()
        data.batch["direct_opd_topk_ids"] = torch.cat(old_topk_ids, dim=0)
        data.batch["direct_opd_topk_dense_reward"] = torch.cat(dense_rewards, dim=0)
        return {
            "distillation/direct_opd_topk_prepass": 1.0,
            "distillation/direct_opd_topk_k": float(top_k),
            "distillation/direct_opd_topk_source_student": 1.0,
            "distillation/direct_opd_topk_weight_sum_max_error": max_weight_sum_error,
            "distillation/direct_opd_topk_teacher_delta_abs_mean": teacher_delta_abs_sum / teacher_delta_count
            if teacher_delta_count
            else 0.0,
        }

    def _prepare_w2s_opd_topk_batch(self, data: DataProto, distillation_cfg) -> dict[str, float]:
        """Build the frozen full-vocabulary proxy, then retain only proxy Top-32."""
        top_k, _, _, amplification = self._w2s_opd_config(distillation_cfg)
        if self.use_dynamic_bsz:
            raise ValueError("W2S-OPD Top-K requires actor.use_dynamic_bsz=false.")
        positive_module = self.external_teacher_module or self.on_policy_teacher_module
        negative_module = self.oprd_base_module
        student_base_module = self.w2s_student_base_module
        if positive_module is None:
            raise ValueError("W2S-OPD requires a frozen positive weak model.")
        if negative_module is None:
            raise ValueError("W2S-OPD requires a frozen negative weak model.")
        if student_base_module is None:
            raise ValueError("W2S-OPD requires a frozen initial-student base model.")
        required_keys = {"responses", "response_mask", "input_ids", "attention_mask", "position_ids"}
        missing = required_keys - set(data.batch.keys())
        if missing:
            raise KeyError(f"W2S-OPD Top-K prepass is missing batch keys: {sorted(missing)}")
        micro_batch_size = int(self.config.ppo_micro_batch_size_per_gpu)
        if micro_batch_size <= 0:
            raise ValueError(
                f"W2S-OPD requires a positive actor.ppo_micro_batch_size_per_gpu, got {micro_batch_size}."
            )
        temperature = float(data.meta_info["temperature"])
        pad_token_id = data.meta_info.get("pad_token_id", 0)
        proxy_topk_ids_cpu: list[torch.Tensor] = []
        proxy_topk_logp_cpu: list[torch.Tensor] = []
        max_proxy_prob_sum_error = 0.0
        modules = (positive_module, negative_module, student_base_module)
        training_states = tuple((module.training for module in modules))
        for module in modules:
            module.eval()
        try:
            for micro_batch in data.split(micro_batch_size):
                micro_batch = micro_batch.to(get_device_id())
                model_inputs = {
                    **micro_batch.batch,
                    **micro_batch.non_tensor_batch,
                    "pad_token_id": pad_token_id,
                }
                frozen_inputs = {
                    "responses": model_inputs.get("teacher_responses", model_inputs["responses"]),
                    "input_ids": model_inputs.get("teacher_input_ids", model_inputs["input_ids"]),
                    "attention_mask": model_inputs.get(
                        "teacher_attention_mask", model_inputs["attention_mask"]
                    ),
                    "position_ids": model_inputs.get("teacher_position_ids", model_inputs["position_ids"]),
                    "pad_token_id": pad_token_id,
                }
                with torch.no_grad():
                    positive_outputs = self._forward_micro_batch(
                        frozen_inputs,
                        temperature=temperature,
                        calculate_entropy=False,
                        return_all_logps=True,
                        module=positive_module,
                    )
                    proxy_score = positive_outputs["all_logps"].detach().float()
                    proxy_score.mul_(amplification)
                    del positive_outputs
                    negative_outputs = self._forward_micro_batch(
                        frozen_inputs,
                        temperature=temperature,
                        calculate_entropy=False,
                        return_all_logps=True,
                        module=negative_module,
                    )
                    negative_logp = negative_outputs["all_logps"]
                    if tuple(negative_logp.shape) != tuple(proxy_score.shape):
                        raise ValueError(
                            f"W2S-OPD positive/negative full-vocabulary shapes differ: positive={tuple(proxy_score.shape)}, negative={tuple(negative_logp.shape)}."
                        )
                    proxy_score.add_(negative_logp.detach(), alpha=-amplification)
                    del negative_logp, negative_outputs
                    student_base_outputs = self._forward_micro_batch(
                        frozen_inputs,
                        temperature=temperature,
                        calculate_entropy=False,
                        return_all_logps=True,
                        module=student_base_module,
                    )
                    student_base_logp = student_base_outputs["all_logps"]
                    if tuple(student_base_logp.shape) != tuple(proxy_score.shape):
                        raise ValueError(
                            f"W2S-OPD student-base/proxy full-vocabulary shapes differ: student_base={tuple(student_base_logp.shape)}, proxy={tuple(proxy_score.shape)}."
                        )
                    proxy_score.add_(student_base_logp.detach())
                    del student_base_logp, student_base_outputs
                    proxy_topk_ids, proxy_topk_logp = select_w2s_proxy_topk(proxy_score, top_k=top_k)
                    del proxy_score
                    max_proxy_prob_sum_error = max(
                        max_proxy_prob_sum_error, (proxy_topk_logp.exp().sum(dim=-1) - 1.0).abs().max().item()
                    )
                    proxy_topk_ids_cpu.append(proxy_topk_ids.cpu())
                    proxy_topk_logp_cpu.append(proxy_topk_logp.cpu())
                del micro_batch
        finally:
            for module, was_training in zip(modules, training_states, strict=True):
                if was_training:
                    module.train()
        if not proxy_topk_ids_cpu:
            raise RuntimeError("W2S-OPD Top-K prepass received an empty rollout batch.")
        if getattr(data.batch, "is_locked", False):
            data.batch = data.batch.copy()
        data.batch["w2s_opd_proxy_topk_ids"] = torch.cat(proxy_topk_ids_cpu, dim=0)
        data.batch["w2s_opd_proxy_topk_logp"] = torch.cat(proxy_topk_logp_cpu, dim=0)
        return {
            "distillation/w2s_opd_topk_prepass": 1.0,
            "distillation/w2s_opd_topk_k": float(top_k),
            "distillation/w2s_opd_topk_source_proxy_teacher": 1.0,
            "distillation/w2s_opd_amplification": amplification,
            "distillation/w2s_opd_adaptive_alpha_used": 0.0,
            "distillation/w2s_opd_proxy_prob_sum_max_error": max_proxy_prob_sum_error,
        }

    def update_policy(self, data: DataProto):
        """Update a paper objective, including frozen-support baseline prepasses."""
        distillation_cfg = getattr(self.config, "distillation", None)
        delta_target_enabled = bool(
            distillation_cfg is not None and distillation_cfg.get("delta_target_enabled", False)
        )
        w2s_opd_enabled = bool(
            distillation_cfg is not None and distillation_cfg.get("w2s_opd_enabled", False)
        )
        if w2s_opd_enabled:
            if self.actor_optimizer is None:
                raise RuntimeError("W2S-OPD training requires an actor optimizer.")
            if self.config.ppo_epochs != 1:
                raise RuntimeError("W2S-OPD requires one actor epoch per rollout batch.")
            prepass_metrics = self._prepare_w2s_opd_topk_batch(data, distillation_cfg)
            metrics = self._update_policy_single_pass(data)
            metrics.update(prepass_metrics)
            return metrics
        direct_opd_topk_enabled = bool(
            delta_target_enabled and self._direct_opd_mode(distillation_cfg) == "topk"
        )
        direct_opd_adaptive_alpha_enabled = bool(
            distillation_cfg is not None
            and distillation_cfg.get("delta_anchor_adaptive_alpha_enabled", False)
        )
        if direct_opd_adaptive_alpha_enabled and (not delta_target_enabled):
            raise ValueError("Direct-OPD adaptive alpha requires distillation.delta_target_enabled=true.")
        if direct_opd_adaptive_alpha_enabled or direct_opd_topk_enabled:
            if self.actor_optimizer is None:
                raise RuntimeError("Direct-OPD training requires an actor optimizer.")
            if self.config.ppo_epochs != 1:
                raise RuntimeError(
                    "Direct-OPD requires one PPO epoch so each sampled support contributes once."
                )
            direct_opd_mode = self._direct_opd_mode(distillation_cfg)
            self._prepare_direct_opd_anchor_alpha(distillation_cfg)
            completed = False
            try:
                if direct_opd_mode == "topk":
                    prepass_metrics = self._prepare_direct_opd_topk_batch(data, distillation_cfg)
                    if direct_opd_adaptive_alpha_enabled:
                        alpha_metrics = self._finalize_direct_opd_anchor_alpha(distillation_cfg)
                        alpha_metrics["distillation/direct_opd_alpha_update_before_actor"] = 1.0
                        alpha_metrics["distillation/direct_opd_dense_reward_mean"] = alpha_metrics[
                            "distillation/direct_opd_mean_teacher_shift"
                        ]
                        alpha_metrics["distillation/direct_opd_dense_reward_count"] = alpha_metrics[
                            "distillation/direct_opd_shift_token_count"
                        ]
                    else:
                        self._direct_opd_shift_stats = None
                        alpha_metrics = {
                            "distillation/direct_opd_anchor_alpha": float(self._direct_opd_anchor_alpha),
                            "distillation/direct_opd_anchor_alpha_next": float(self._direct_opd_anchor_alpha),
                            "distillation/direct_opd_alpha_update_before_actor": 0.0,
                        }
                    alpha_metrics["distillation/direct_opd_actor_alpha"] = float(
                        self._direct_opd_anchor_alpha
                    )
                    metrics = self._update_policy_single_pass(data)
                    metrics.update(prepass_metrics)
                    metrics.update(alpha_metrics)
                else:
                    metrics = self._update_policy_single_pass(data)
                    metrics.update(self._finalize_direct_opd_anchor_alpha(distillation_cfg))
                    metrics["distillation/direct_opd_alpha_update_before_actor"] = 0.0
                completed = True
                return metrics
            finally:
                if not completed:
                    self._direct_opd_shift_stats = None
        return self._update_policy_single_pass(data)

    def _update_policy_single_pass(self, data):
        from oprd.runtime.metrics import ProjectionMetrics

        previous = dict(self.config.global_batch_info)
        projection_metrics = None
        if self.config.distillation.get("oprd_enabled", False):
            projection_metrics = ProjectionMetrics(
                get_device_id(), getattr(self, "data_parallel_group", None)
            )
        self._oprd_projection_metrics = projection_metrics
        try:
            metrics = self._update_policy_normalized(data)
            if projection_metrics is not None:
                metrics.update(projection_metrics.finish())
            return metrics
        finally:
            self._oprd_projection_metrics = None
            self.config.global_batch_info.clear()
            self.config.global_batch_info.update(previous)

    def _update_policy_normalized(self, data):
        """One PPO epoch with the objectives and guidance directions in the paper."""
        from oprd.runtime.metrics import TokenMetrics

        token_metrics = TokenMetrics(get_device_id(), getattr(self, "data_parallel_group", None))
        self.actor_module.train()
        cfg = self.config.distillation
        mode = self.config.policy_loss.loss_mode
        temperature = data.meta_info["temperature"]
        metrics = {"actor/pg_loss": 0.0, "actor/kl_loss": 0.0}
        updated = False
        optimizer_steps = 0
        kdrl = bool(cfg.get("kdrl_enabled", False))
        if kdrl:
            metrics.update({"distillation/kdrl_grpo_loss": 0.0, "distillation/kdrl_k2_loss": 0.0})
        beta = self._compute_kdrl_beta(data.meta_info) if kdrl else 0.0
        direct = bool(cfg.get("delta_target_enabled", False))
        w2s = bool(cfg.get("w2s_opd_enabled", False))
        oprd = bool(cfg.get("oprd_enabled", False))
        for _ in range(self.config.ppo_epochs):
            for mini in data.split(self.config.ppo_mini_batch_size):
                # Same DP-wide optimizer-minibatch normalization as the current
                # source actor / Verl engine, extended to every paper objective.
                self.config.global_batch_info.clear()
                self.config.global_batch_info.update(
                    global_token_mean_batch_info(
                        mini.batch["response_mask"],
                        get_device_id(),
                        getattr(self, "data_parallel_group", None),
                    )
                )
                if self.use_dynamic_bsz:
                    micros, _ = prepare_dynamic_batch(
                        mini,
                        max_token_len=self.config.ppo_max_token_len_per_gpu
                        * self.ulysses_sequence_parallel_size,
                    )
                else:
                    micros = mini.split(self.config.ppo_micro_batch_size_per_gpu)
                    self.gradient_accumulation = len(micros)
                self.actor_optimizer.zero_grad()
                for micro in micros:
                    micro = micro.to(get_device_id())
                    inputs = {
                        **micro.batch,
                        **micro.non_tensor_batch,
                        "pad_token_id": data.meta_info.get("pad_token_id", 0),
                    }
                    mask = inputs["response_mask"]
                    # Each microbatch contributes a partial token sum. Do not
                    # average it again by rows or accumulation steps.
                    scale = 1.0
                    teacher = (
                        self.external_teacher_module or self.on_policy_teacher_module or self.teacher_module
                    )
                    hook = None
                    hook_metrics = {}
                    if oprd:
                        hook = {
                            key.removeprefix("oprd_"): value
                            for key, value in cfg.items()
                            if key.startswith("oprd_")
                        }
                        hook.update(
                            current_step=int(data.meta_info["global_steps"]),
                            teacher_module=self.teacher_module
                            if cfg.oprd_h1_source == "opsd_kl"
                            else teacher,
                            base_module=self.oprd_base_module,
                            student_base_module=None,
                            batch_size=mask.shape[0],
                            response_length=mask.shape[1],
                            advantages=inputs["advantages"],
                            metric_buffer=hook_metrics,
                            projection_metrics=self._oprd_projection_metrics,
                            grad_scale=float(self.scaler.get_scale()) if self.scaler is not None else 1.0,
                            opsd_teacher_input_ids=inputs.get("teacher_input_ids"),
                            opsd_teacher_attention_mask=inputs.get("teacher_attention_mask"),
                            opsd_teacher_position_ids=inputs.get("teacher_position_ids"),
                            opsd_distillation_mask=inputs.get("distillation_mask"),
                        )
                    if hook is not None and cfg.oprd_skip_invalid_format:
                        hook["invalid_format_mask"] = self._normalize_oprd_invalid_format_mask(
                            inputs.get("oprd_invalid_format_mask"),
                            batch_size=mask.shape[0],
                            device=mask.device,
                        )
                    topk = cfg.distillation_topk if mode == "opsd" else None
                    support = (
                        inputs.get("direct_opd_topk_ids")
                        if direct
                        else inputs.get("w2s_opd_proxy_topk_ids")
                        if w2s
                        else None
                    )
                    outputs = self._forward_micro_batch(
                        inputs,
                        temperature=temperature,
                        calculate_entropy=False,
                        distill_topk=topk,
                        topk_indices=support,
                        oprd_hook_config=hook,
                    )
                    logp = outputs["log_probs"]
                    old = inputs["old_log_probs"]
                    auxiliary = None
                    if mode == "kd" and (not direct) and (not w2s) or mode == "opsd" or (kdrl and beta > 0):
                        if mode == "opsd":
                            target_inputs = {
                                **inputs,
                                "input_ids": inputs["teacher_input_ids"],
                                "attention_mask": inputs["teacher_attention_mask"],
                                "position_ids": inputs["teacher_position_ids"],
                            }
                            target_model = self.teacher_module
                        else:
                            target_inputs, target_model = (inputs, teacher)
                        if target_model is None:
                            raise ValueError("This method requires a loaded teacher model")
                        with torch.no_grad():
                            with nullcontext():
                                auxiliary = self._forward_micro_batch(
                                    target_inputs,
                                    temperature=temperature,
                                    topk_indices=outputs.get("topk_indices"),
                                    module=target_model,
                                )
                    if direct:
                        loss, stats = compute_direct_opd_topk_loss(
                            outputs["topk_logps"],
                            inputs["direct_opd_topk_dense_reward"],
                            mask,
                            global_batch_info=self.config.global_batch_info,
                        )
                    elif w2s:
                        loss, stats = compute_w2s_reverse_kl_loss(
                            outputs["topk_logps"],
                            inputs["w2s_opd_proxy_topk_logp"],
                            mask,
                            global_batch_info=self.config.global_batch_info,
                        )
                    elif mode in {"kd", "opsd"}:
                        loss, stats = compute_distillation_loss(
                            logp,
                            auxiliary["log_probs"],
                            mask,
                            cfg,
                            old_log_probs=old,
                            student_topk_log_probs=outputs.get("topk_logps"),
                            teacher_topk_log_probs=auxiliary.get("topk_logps"),
                            distillation_topk_indices=outputs.get("topk_indices"),
                            student_topk_indices=outputs.get("topk_indices"),
                            teacher_topk_indices=outputs.get("topk_indices"),
                            distillation_mask=inputs.get("distillation_mask") if mode == "opsd" else None,
                            rollout_is_weights=inputs.get("rollout_is_weights"),
                            loss_agg_mode=self.config.loss_agg_mode,
                            global_batch_info=self.config.global_batch_info,
                        )
                    else:
                        policy = get_policy_loss_fn("vanilla" if mode == "guided_opsd" else mode)
                        result = policy(
                            old_log_prob=old,
                            log_prob=logp,
                            advantages=inputs["advantages"],
                            response_mask=mask,
                            loss_agg_mode=self.config.loss_agg_mode,
                            config=self.config,
                            rollout_is_weights=inputs.get("rollout_is_weights"),
                        )
                        if len(result) == 2:
                            loss, stats = result
                        else:
                            loss, clipfrac, ppo_kl, clipfrac_lower = result
                            stats = {
                                "actor/pg_clipfrac": float(clipfrac),
                                "actor/ppo_kl": float(ppo_kl),
                                "actor/pg_clipfrac_lower": float(clipfrac_lower),
                            }
                        if kdrl:
                            metrics["distillation/kdrl_grpo_loss"] += float(loss.detach())
                        if kdrl and beta > 0:
                            kd_loss, kd_stats = compute_kdrl_k2_loss(
                                logp,
                                auxiliary["log_probs"],
                                mask,
                                loss_agg_mode=self.config.loss_agg_mode,
                                global_batch_info=self.config.global_batch_info,
                            )
                            loss = loss + beta * kd_loss
                            metrics["distillation/kdrl_k2_loss"] += float(kd_loss.detach())
                            stats.update(kd_stats)
                        if kdrl:
                            stats["distillation/kdrl_beta"] = beta
                    if self.config.use_kl_loss:
                        penalty = kl_penalty(
                            logprob=logp,
                            ref_logprob=inputs["ref_log_prob"],
                            kl_penalty=self.config.kl_loss_type,
                        )
                        kl = agg_loss(
                            loss_mat=penalty,
                            loss_mask=mask,
                            loss_agg_mode=self.config.loss_agg_mode,
                            **self.config.global_batch_info,
                        )
                        coef = self._direct_opd_anchor_alpha if direct else self.config.kl_loss_coef
                        loss = loss + coef * kl
                        metrics["actor/kl_loss"] += float(kl.detach()) * scale
                    metrics["actor/pg_loss"] += float(loss.detach()) * scale
                    scaled = loss * scale
                    if self.scaler is not None:
                        self.scaler.scale(scaled).backward()
                    else:
                        scaled.backward()
                    token_metrics.record(stats, mask.sum())
                    append_to_dict(
                        metrics, {key: value for key, value in stats.items() if key not in TokenMetrics.KEYS}
                    )
                    append_to_dict(metrics, hook_metrics)
                norm = self._optimizer_step()
                optimizer_steps += 1
                updated = updated or bool(torch.isfinite(norm))
                append_to_dict(metrics, {"actor/grad_norm": float(norm.detach())})
        self.actor_optimizer.zero_grad()
        if updated:
            self._update_teacher()
        for key in ("actor/pg_loss", "actor/kl_loss"):
            metrics[key] /= max(optimizer_steps, 1)
        metrics.update(token_metrics.finish())
        if kdrl:
            metrics["distillation/kdrl_grpo_loss"] /= max(optimizer_steps, 1)
            metrics["distillation/kdrl_k2_loss"] /= max(optimizer_steps, 1)
            metrics["distillation/kdrl_weighted_k2_loss"] = beta * metrics["distillation/kdrl_k2_loss"]
            metrics["distillation/kdrl_total_loss"] = (
                metrics["distillation/kdrl_grpo_loss"] + metrics["distillation/kdrl_weighted_k2_loss"]
            )
            metrics["distillation/kdrl_k2_enabled"] = float(beta > 0)
        if (mode == "kd" and not direct and not w2s) or mode == "agent_global_token_opd":
            metrics["distillation/opd_loss"] = metrics["actor/pg_loss"]
        metrics["actor/global_token_mean_enabled"] = 1.0
        return metrics

    def _register_oprd_hook_non_rmpad(
        self, *, logits, micro_batch, temperature, response_mask, oprd_hook_config
    ):
        cfg = oprd_hook_config
        source = cfg.get("h1_source", "teacher_base")
        teacher = cfg["teacher_module"]
        length = logits.size(1)
        hinted = source == "opsd_kl"

        def forward(model, hint=False):
            prefix = "opsd_teacher_" if hint else ""
            values = cfg if hint else micro_batch
            out = model(
                input_ids=values[prefix + "input_ids"],
                attention_mask=values[prefix + "attention_mask"],
                position_ids=values[prefix + "position_ids"],
                use_cache=False,
                logits_to_keep=length + 1,
            )
            return out.logits[:, -length - 1 : -1].float() / temperature

        with torch.no_grad():
            with nullcontext():
                teacher_logits = forward(teacher, hinted)
            reference = forward(cfg["base_module"]) if source == "teacher_base" else None
            active = response_mask.bool()
            if cfg.get("skip_invalid_format", False):
                invalid = self._normalize_oprd_invalid_format_mask(
                    cfg.get("invalid_format_mask"), batch_size=active.size(0), device=active.device
                )
                active = active & ~invalid[:, None]
            if hinted:
                hint_mask = cfg["opsd_distillation_mask"].bool()
                active = active & (hint_mask[:, None] if hint_mask.ndim == 1 else hint_mask)
            indices, mask = support_indices(logits.detach(), micro_batch["responses"], cfg["topk"])
            direction = guidance_direction(logits.detach(), teacher_logits, reference, indices, mask, source)
            direction *= active.unsqueeze(-1)
        self._install_oprd_hook(
            logits, indices, direction, cfg, support_mask=mask, response_mask=response_mask
        )

    def _register_oprd_hook_rmpad(
        self,
        *,
        logits_rmpad,
        input_ids_rmpad,
        position_ids_rmpad,
        input_ids_rmpad_rolled,
        response_pred_mask_rmpad,
        response_position_rmpad,
        sample_id_rmpad,
        response_advantages_rmpad,
        temperature,
        oprd_hook_config,
    ):
        cfg = oprd_hook_config
        if self.use_ulysses_sp:
            raise ValueError("The paper OPRD configuration uses sequence parallel size 1")
        rows = response_pred_mask_rmpad.nonzero(as_tuple=False).flatten()
        if rows.numel() == 0:
            return
        source = cfg.get("h1_source", "teacher_base")
        teacher = cfg["teacher_module"]
        hinted = source == "opsd_kl"
        with torch.no_grad():
            if hinted:
                with nullcontext():
                    out = teacher(
                        input_ids=cfg["opsd_teacher_input_ids"],
                        attention_mask=cfg["opsd_teacher_attention_mask"],
                        position_ids=cfg["opsd_teacher_position_ids"],
                        use_cache=False,
                        logits_to_keep=cfg["response_length"] + 1,
                    )
                samples = sample_id_rmpad[rows].long()
                positions = response_position_rmpad[rows].long()
                teacher_logits = (
                    out.logits[:, -cfg["response_length"] - 1 : -1][samples, positions].float() / temperature
                )
            else:
                out = teacher(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    use_cache=False,
                    logits_to_keep=rows,
                )
                teacher_logits = out.logits.squeeze(0).float() / temperature
            reference = None
            if source == "teacher_base":
                out = cfg["base_module"](
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    use_cache=False,
                    logits_to_keep=rows,
                )
                reference = out.logits.squeeze(0).float() / temperature
            student = logits_rmpad.detach()[rows]
            indices, mask = support_indices(student, input_ids_rmpad_rolled[rows], cfg["topk"])
            direction = guidance_direction(student, teacher_logits, reference, indices, mask, source)
            active = torch.ones(rows.shape, device=rows.device, dtype=torch.bool)
            if cfg.get("skip_invalid_format", False):
                invalid = self._normalize_oprd_invalid_format_mask(
                    cfg.get("invalid_format_mask"), batch_size=cfg["batch_size"], device=rows.device
                )
                active &= ~invalid[sample_id_rmpad[rows].long()]
            if hinted:
                hint_mask = cfg["opsd_distillation_mask"].bool()
                active &= hint_mask[samples] if hint_mask.ndim == 1 else hint_mask[samples, positions]
            direction *= active.unsqueeze(-1)
        self._install_oprd_hook(logits_rmpad, indices, direction, cfg, rows, support_mask=mask)

    def _install_oprd_hook(
        self, logits, indices, direction, cfg, rows=None, *, support_mask=None, response_mask=None
    ):
        lam = float(cfg.get("lambda", 0.5))
        steps = int(cfg.get("signed_lambda_schedule_steps", 1))
        step = max(int(cfg.get("current_step", 1)), 1)
        plus = linear_schedule(
            float(cfg.get("lambda_plus_start", lam)), float(cfg.get("lambda_plus_end", lam)), steps, step
        )
        minus = linear_schedule(
            float(cfg.get("lambda_minus_start", lam)), float(cfg.get("lambda_minus_end", lam)), steps, step
        )
        metrics = cfg.get("metric_buffer")
        if metrics is not None:
            metrics.update({"distillation/oprd_lambda_plus": plus, "distillation/oprd_lambda_minus": minus})

        def hook(grad):
            diagnostics = cfg.get("projection_metrics")
            if diagnostics is not None:
                selected = (grad if rows is None else grad[rows]).gather(-1, indices)
                diagnostics.record(selected, direction, support_mask, plus, minus, response_mask)
            if rows is None:
                return precondition_loss_gradient(grad, indices, direction, plus, minus)
            result = grad.clone()
            result[rows] = precondition_loss_gradient(grad[rows], indices, direction, plus, minus)
            return result

        logits.register_hook(hook)
