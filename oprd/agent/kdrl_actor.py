"""ViLaSR KDRL adapter using root Verl's unchanged PPO and k2 objectives.

Only sampled-action log probabilities survive the frozen teacher forward.
The teacher uses the student's packed pixels/mRoPE and a chunked LM head.
RL advantages are never replaced; both losses share ONE global token mean.
"""

from contextvars import ContextVar
from dataclasses import dataclass
from types import SimpleNamespace

import torch
import torch.distributed as dist

from verl.trainer.ppo.core_algos import (
    compute_kdrl_k2_loss,
    compute_policy_loss_vanilla,
    register_policy_loss,
)
from verl.utils.device import get_device_id
from verl.workers.actor.dp_actor import DataParallelPPOActor
from .opd import sampled_token_logprobs
from .opd_vilasr_inputs import shared_teacher_inputs
from .oprd import teacher_inputs_on_device

_ACTIVE_KDRL = ContextVar("vilasr_kdrl_update", default=None)


def beta_at_step(config, step):
    if not isinstance(step, int) or isinstance(step, bool) or step < 1:
        raise ValueError("KDRL requires a one-based optimizer update step")
    # Reuse the exact implemented legacy KDRL annealing, including step-1.
    proxy = SimpleNamespace(config=SimpleNamespace(distillation=config))
    return DataParallelPPOActor._compute_kdrl_beta(proxy, global_step=step)


@dataclass
class UpdateState:
    beta: float
    teacher_log_probs: object = None
    sums: object = None

    def accumulate(self, rl_loss, kd_loss, diff, teacher, mask):
        with torch.no_grad():
            count = mask.sum(dtype=torch.float32)
            values = torch.stack(
                (
                    rl_loss.detach().float() * count,
                    kd_loss.detach().float() * count,
                    (diff.detach().float() * mask).sum(),
                    (teacher.detach().float() * mask).sum(),
                    count,
                )
            )
            self.sums = values if self.sums is None else self.sums + values

    def metrics(self, group=None):
        if self.teacher_log_probs is not None or self.sums is None:
            raise RuntimeError("KDRL update did not consume its teacher targets and losses")
        sums = self.sums.clone()
        if dist.is_initialized():
            dist.all_reduce(sums, group=group)
        rl, kd, diff, teacher, count = sums.cpu().tolist()
        if count <= 0:
            raise RuntimeError("KDRL update has no valid response tokens")
        return {
            "distillation/kdrl_k2_enabled": float(self.beta > 0),
            "distillation/kdrl_beta": self.beta,
            "distillation/kdrl_teacher_forward_skipped": float(self.beta == 0),
            "distillation/grpo_loss_coef": 1.0,
            "distillation/kdrl_grpo_loss": rl / count,
            "distillation/kdrl_k2_loss": kd / count,
            "distillation/kdrl_weighted_k2_loss": self.beta * kd / count,
            "distillation/kdrl_total_loss": (rl + self.beta * kd) / count,
            "distillation/kdrl_k2_log_prob_diff_mean": diff / count,
            "distillation/kdrl_teacher_logp_mean": teacher / count,
            "distillation/kdrl_k2_token_count": count if self.beta > 0 else 0.0,
        }


@register_policy_loss("agent_global_token_kdrl")
def kdrl_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    loss_agg_mode="token-mean",
    config=None,
    rollout_is_weights=None,
):
    if loss_agg_mode != "token-mean" or rollout_is_weights is not None:
        raise ValueError("ViLaSR KDRL requires token-mean without extra rollout IS")
    state = _ACTIVE_KDRL.get()
    if state is None:
        raise RuntimeError("KDRL loss requires its update context")
    valid = response_mask.bool()
    # Tool observations/padding are excluded BEFORE nonlinear arithmetic, so
    # e.g. an unused +/-inf padded log probability cannot produce 0*NaN.
    logp = torch.where(valid, log_prob, 0.0)
    old = torch.where(valid, old_log_prob, 0.0)
    advantage = torch.where(valid, advantages, 0.0)
    rl_loss, metrics = compute_policy_loss_vanilla(
        old, logp, advantage, response_mask, loss_agg_mode, config, rollout_is_weights
    )
    if state.beta > 0:
        teacher = state.teacher_log_probs
        if teacher is None or teacher.shape != log_prob.shape or teacher.device != log_prob.device:
            raise RuntimeError("KDRL teacher/student targets must match the current microbatch")
        teacher = torch.where(valid, teacher.detach(), 0.0)
        kd_loss, _ = compute_kdrl_k2_loss(
            logp,
            teacher,
            response_mask,
            loss_agg_mode=loss_agg_mode,
            global_batch_info=config.global_batch_info,
        )
        diff = logp.detach().float() - teacher.float()
        state.teacher_log_probs = None
    else:
        if state.teacher_log_probs is not None:
            raise RuntimeError("KDRL beta=0 must not evaluate the teacher")
        teacher = torch.zeros_like(logp)
        diff = teacher
        kd_loss = logp.sum() * 0.0
    # Diagnostics expect local means, while backward uses global partial sums.
    weight = (
        response_mask.sum()
        * config.global_batch_info["dp_size"]
        / config.global_batch_info["batch_num_tokens"]
    )
    state.accumulate(
        rl_loss / weight.clamp_min(1e-12), kd_loss / weight.clamp_min(1e-12), diff, teacher, response_mask
    )

    return rl_loss + state.beta * kd_loss, metrics


class AgentKDRLActor(DataParallelPPOActor):
    def __init__(self, *args, kdrl_config, **kwargs):
        super().__init__(*args, **kwargs)
        self.kdrl = kdrl_config
        self.teacher = None

    def _load_teacher(self):
        from transformers import Qwen2_5_VLForConditionalGeneration

        self.teacher = (
            Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.kdrl.teacher_model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                local_files_only=True,
            )
            .eval()
            .requires_grad_(False)
        )

    def update_policy(self, data):
        if _ACTIVE_KDRL.get() is not None:
            raise RuntimeError("Nested KDRL actor update")
        step = data.meta_info.get("global_steps")
        state = UpdateState(beta_at_step(self.kdrl, step))
        token = _ACTIVE_KDRL.set(state)
        try:
            if state.beta > 0:
                if self.teacher is None:
                    self._load_teacher()
                self.teacher.to(get_device_id())
            metrics = super().update_policy(data)
            metrics.update(state.metrics(getattr(self, "data_parallel_group", None)))
            return metrics
        finally:
            _ACTIVE_KDRL.reset(token)
            # Retain CPU weights, but skip even CPU->GPU transfer at beta=0.
            if state.beta > 0 and self.teacher is not None:
                self.teacher.cpu()
                torch.cuda.empty_cache()

    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False, **kwargs):
        state = _ACTIVE_KDRL.get()
        if state is not None and state.beta > 0 and torch.is_grad_enabled():
            if state.teacher_log_probs is not None:
                raise RuntimeError("Unconsumed KDRL microbatch teacher targets")
            from verl.models.transformers.qwen2_vl import qwen2_vl_forward

            with torch.no_grad(), torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
                target = torch.zeros_like(micro_batch["advantages"], dtype=torch.float32)
                mask = micro_batch["response_mask"].bool()
                if mask.any():
                    inputs, rows, labels, _ = shared_teacher_inputs(
                        micro_batch, self.kdrl.teacher_max_model_len
                    )
                    device = micro_batch["input_ids"].device
                    inputs = teacher_inputs_on_device(inputs, device)
                    hidden = qwen2_vl_forward(self.teacher, **inputs)[0].squeeze(0)
                    logp = sampled_token_logprobs(
                        hidden,
                        self.teacher.lm_head,
                        rows.to(device),
                        labels.to(device),
                        temperature,
                        chunk_size=self.kdrl.logprob_chunk_size,
                    )
                    target[mask] = logp.float()
                    del hidden, inputs, logp
                state.teacher_log_probs = target
        return super()._forward_micro_batch(
            micro_batch, temperature, calculate_entropy=calculate_entropy, **kwargs
        )
