"""Use the pinned actor's FSDP/optimizer loop with a pure OPD policy loss."""

import torch
import torch.distributed as dist
from verl.utils.device import get_device_id
from verl.workers.actor.dp_actor import DataParallelPPOActor
from . import token_mean  # noqa: F401 -- register losses in each Ray worker process
from .opd import sampled_token_logprobs
from .oprd import response_rows, teacher_inputs_on_device
from .opd_vilasr_inputs import shared_teacher_inputs


class AgentOPDActor(DataParallelPPOActor):
    def __init__(self, *args, opd_config, **kwargs):
        super().__init__(*args, **kwargs)
        self.opd = opd_config
        self.teacher = None
        self.opd_active = False

    def _load_teacher(self):
        from transformers import Qwen2_5_VLForConditionalGeneration

        self.teacher = (
            Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.opd.teacher_model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                local_files_only=True,
            )
            .eval()
            .requires_grad_(False)
        )

    def update_policy(self, data):
        if self.teacher is None:
            self._load_teacher()
        try:
            self.teacher.to(get_device_id())
            self.opd_active = True
            self.opd_stats = [0.0, 0.0, 0]

            metrics = super().update_policy(data)
            totals = torch.tensor(self.opd_stats, dtype=torch.float64, device=get_device_id())
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(totals, group=getattr(self, "data_parallel_group", None))
            total, gap, count = totals.cpu().tolist()
            metrics.update(
                {
                    "distillation/sampled_rkl_enabled": 1.0,
                    "distillation/sampled_rkl_teacher_logp_mean": total / max(count, 1),
                    "distillation/sampled_rkl_advantage_mean": gap / max(count, 1),
                    "distillation/sampled_rkl_is_clip_low": 0.5,
                    "distillation/sampled_rkl_is_clip_high": 2.0,
                    "distillation/grpo_loss_coef": 0.0,
                }
            )
            return metrics
        finally:
            self.opd_active = False
            self.teacher.cpu()
            torch.cuda.empty_cache()

    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False, **kwargs):
        if self.opd_active and torch.is_grad_enabled():
            from verl.models.transformers.qwen2_vl import qwen2_vl_forward

            _, labels = response_rows(micro_batch)
            # Evaluate the frozen teacher before retaining student activations.
            with torch.no_grad(), torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
                target = torch.zeros_like(micro_batch["advantages"])
                if labels.numel():
                    inputs, rows, aligned, _ = shared_teacher_inputs(
                        micro_batch, self.opd.teacher_max_model_len
                    )
                    if not torch.equal(aligned, labels.to(device=aligned.device)):
                        raise ValueError("OPD teacher/student generated text token identities differ")
                    device = micro_batch["input_ids"].device
                    inputs = teacher_inputs_on_device(inputs, device)
                    hidden = qwen2_vl_forward(self.teacher, **inputs)[0].squeeze(0)
                    logp = sampled_token_logprobs(
                        hidden,
                        self.teacher.lm_head,
                        rows.to(device),
                        aligned.to(device),
                        temperature,
                        chunk_size=getattr(self.opd, "logprob_chunk_size", 16),
                    )
                    mask = micro_batch["response_mask"].bool()
                    target[mask] = logp.to(target.dtype)
                    self.opd_stats[0] += logp.sum().item()
                    self.opd_stats[1] += (logp - micro_batch["old_log_probs"][mask]).sum().item()
                    self.opd_stats[2] += logp.numel()
                    del hidden, inputs, logp
                # Upstream has already captured a reference to this tensor.
                micro_batch["advantages"].copy_(target)
        return super()._forward_micro_batch(
            micro_batch, temperature, calculate_entropy=calculate_entropy, **kwargs
        )
