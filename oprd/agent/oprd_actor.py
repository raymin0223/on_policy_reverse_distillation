"""OPRD with shared visual inputs and frozen hidden prepasses before student forward."""

import torch
from verl.workers.actor.dp_actor import DataParallelPPOActor
from . import token_mean  # noqa: F401 -- register losses in each Ray worker process
from verl.utils.device import get_device_id
from .oprd import (
    project_gradient,
    residual_direction,
    response_rows,
    selected_head_logits,
    signed_lambdas,
    student_support,
    teacher_inputs_on_device,
)
from .opd_vilasr_inputs import shared_teacher_inputs


class AgentOPRDActor(DataParallelPPOActor):
    def __init__(self, *args, oprd_config, **kwargs):
        super().__init__(*args, **kwargs)
        self.oprd = oprd_config
        if not self.use_remove_padding or self.use_ulysses_sp or self.use_fused_kernels:
            raise ValueError("Agent OPRD requires packed, unfused logits with sequence parallel size 1")
        self.teacher_models = None
        self.oprd_step = None

    def update_policy(self, data):
        from oprd.runtime.metrics import ProjectionMetrics
        from .oprd_metrics import OPRDMetrics

        self.oprd_metrics = OPRDMetrics()
        self.projection_metrics = ProjectionMetrics(
            get_device_id(), getattr(self, "data_parallel_group", None)
        )
        # Never use a process-local counter: resume must retain the schedule.
        self.oprd_step = int(data.meta_info["global_steps"])
        lp, lm = signed_lambdas(self.oprd, self.oprd_step)
        if self.teacher_models is None:
            self._load_teachers()
        try:
            for model in self.teacher_models:
                model.to(get_device_id())

            metrics = super().update_policy(data)
            metrics.update(self.oprd_metrics.finish())
            metrics.update(self.projection_metrics.finish())
            metrics.update(
                {
                    "distillation/oprd_lambda_plus": lp,
                    "distillation/oprd_lambda_minus": lm,
                    "distillation/oprd_signed_lambda_schedule_steps": self.oprd.schedule_steps,
                    "distillation/oprd_enabled": 1.0,
                    "distillation/oprd_active": 1.0,
                    "distillation/oprd_current_step": self.oprd_step,
                    "distillation/oprd_signed_lambda_schedule_enabled": 1.0,
                }
            )
            return metrics
        finally:
            self.oprd_step = None
            self.oprd_metrics = None
            self.projection_metrics = None
            # Release both frozen BF16 copies before vLLM wakes for rollout.
            for model in self.teacher_models:
                model.cpu()
            torch.cuda.empty_cache()

    def _load_teachers(self):
        from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

        self.shared_vocab_size = len(
            AutoTokenizer.from_pretrained(self.oprd.base_model_path, local_files_only=True)
        )
        # Reuse pixels/grids/mRoPE; no processor loading, PIL payload or image re-tokenization.
        self.teacher_models = [
            Qwen2_5_VLForConditionalGeneration.from_pretrained(
                path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                local_files_only=True,
            )
            .eval()
            .requires_grad_(False)
            for path in (self.oprd.teacher_model_path, self.oprd.base_model_path)
        ]

    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False, **kwargs):
        # No teacher/reference work on old-logp or validation forwards.
        if self.oprd_step is None or not torch.is_grad_enabled():
            return super()._forward_micro_batch(
                micro_batch, temperature, calculate_entropy=calculate_entropy, **kwargs
            )
        rows, labels = response_rows(micro_batch)
        if not rows.numel():
            return super()._forward_micro_batch(
                micro_batch, temperature, calculate_entropy=calculate_entropy, **kwargs
            )
        active = None
        if self.oprd.get("skip_invalid_format", False):
            invalid = self._normalize_oprd_invalid_format_mask(
                micro_batch.get("oprd_invalid_format_mask"),
                batch_size=micro_batch["responses"].shape[0],
                device=rows.device,
            )
            # Keep every trajectory in the GRPO loss and token denominator.
            # Only the projection direction is disabled for invalid trajectories.
            valid_grid = (~invalid).unsqueeze(-1).expand_as(micro_batch["attention_mask"])
            active = valid_grid[micro_batch["attention_mask"].bool()][rows]
            if not active.any():
                diagnostics = getattr(self, "projection_metrics", None)
                if diagnostics is not None:
                    diagnostics.skipped(rows.numel())
                return super()._forward_micro_batch(
                    micro_batch, temperature, calculate_entropy=calculate_entropy, **kwargs
                )
        lp, lm = signed_lambdas(self.oprd, self.oprd_step)
        from verl.models.transformers.qwen2_vl import qwen2_vl_forward

        # Run frozen models sequentially before retaining any student activations.
        # Keep only predictor hidden states: O(valid response tokens * teacher hidden).
        with torch.no_grad(), torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            inputs, teacher_rows, teacher_labels, _ = shared_teacher_inputs(
                micro_batch, self.oprd.teacher_max_model_len
            )
            if not torch.equal(teacher_labels, labels.to(device=teacher_labels.device)):
                raise ValueError("Teacher/student sampled token identities differ")
            device = micro_batch["input_ids"].device
            inputs = teacher_inputs_on_device(inputs, device)
            predictor_hidden = []
            for model in self.teacher_models:
                hidden = qwen2_vl_forward(model, **inputs)[0].squeeze(0)
                predictor_hidden.append(hidden.index_select(0, teacher_rows.to(device)))
                del hidden
            del inputs

        def attach_projection(module, args, inputs, output):
            logits = output.logits
            if logits.ndim != 3 or logits.shape[0] != 1:
                raise ValueError("Expected one packed multimodal student sequence")
            with torch.no_grad(), torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
                ids, support_mask = student_support(
                    logits.detach().squeeze(0),
                    rows,
                    labels,
                    self.oprd.topk,
                    self.shared_vocab_size,
                    chunk_size=self.oprd.support_chunk_size,
                )
                relative_rows = torch.arange(rows.numel(), device=rows.device)
                support_logits = [
                    selected_head_logits(
                        hidden, model.lm_head, relative_rows, ids, chunk_size=self.oprd.head_chunk_size
                    )
                    for hidden, model in zip(predictor_hidden, self.teacher_models)
                ]
                h = residual_direction(*support_logits, support_mask)
                if active is not None:
                    h = h * active.unsqueeze(-1)
                diagnostics = getattr(self, "oprd_metrics", None)
                if diagnostics is not None:
                    diagnostics.forward(*support_logits, h, support_mask)
            # Only action+top10 coordinates survive into backward; free frozen hiddens.
            predictor_hidden.clear()

            def correct_gradient(gradient):
                projection_metrics = getattr(self, "projection_metrics", None)
                if projection_metrics is not None:
                    projection_metrics.record(
                        gradient.squeeze(0)[rows[:, None], ids], h, support_mask, lp, lm
                    )
                if diagnostics is not None:
                    advantage_grid = torch.zeros_like(micro_batch["attention_mask"], dtype=torch.float32)
                    length = micro_batch["responses"].shape[-1]
                    advantage_grid[:, -length - 1 : -1] = micro_batch["advantages"]
                    advantages = advantage_grid[micro_batch["attention_mask"].bool()][rows]
                    info = self.config.global_batch_info
                    # Undo the actual DP-wide token-mean factor, independently
                    # of how many action tokens happen to be in this microbatch.
                    loss_scale = info["dp_size"] / max(info["batch_num_tokens"], 1)
                    if getattr(self, "scaler", None) is not None:
                        loss_scale *= self.scaler.get_scale()
                    diagnostics.backward(
                        gradient.squeeze(0),
                        rows,
                        ids,
                        h,
                        support_mask,
                        lp,
                        lm,
                        loss_scale,
                        advantages,
                    )
                # project_gradient reads the argument and returns a new gradient;
                # do not mutate an autograd hook's incoming tensor in place.
                return project_gradient(gradient.squeeze(0), rows, ids, h, lp, lm).unsqueeze(0)

            logits.register_hook(correct_gradient)

        handle = self.actor_module.register_forward_hook(attach_projection, with_kwargs=True)
        try:
            return super()._forward_micro_batch(
                micro_batch, temperature, calculate_entropy=calculate_entropy, **kwargs
            )
        finally:
            handle.remove()
            predictor_hidden.clear()
