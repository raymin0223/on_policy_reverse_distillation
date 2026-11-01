"""Observational OPRD diagnostics, token-weighted across microbatches and ranks."""

import math
import torch
import torch.distributed as dist


class OPRDMetrics:
    def __init__(self):
        self.stats = {}

    def record(self, name, values, mode="mean"):
        values = values.detach().float().reshape(-1)
        if mode == "min":
            values = -values
        count = values.numel()
        # Keep reductions on the GPU; synchronize only once per optimizer update.
        summary = torch.stack(
            (
                values.double().sum(),
                values.new_tensor(count, dtype=torch.float64),
                values.double().max() if count else values.new_tensor(-math.inf, dtype=torch.float64),
            )
        )
        key = ("distillation/oprd_" + name, mode)
        if key in self.stats:
            old = self.stats[key]
            summary = torch.stack(
                (old[0] + summary[0], old[1] + summary[1], torch.maximum(old[2], summary[2]))
            )
        self.stats[key] = summary

    def forward(self, teacher, base, h, mask):
        raw = (teacher.float() - base.float()) * mask
        entries = h[mask]
        self.record("h_raw_norm_mean", raw.norm(dim=-1))
        self.record("h_norm_mean", h.norm(dim=-1))
        self.record("h_zero_fraction", h.norm(dim=-1) == 0)
        self.record("support_size_mean", mask.sum(-1))
        self.record("h_entry_positive_frac", entries > 0)
        self.record("h_entry_negative_frac", entries < 0)
        self.record("h_entry_abs_mean", entries.abs())
        for prefix, values in (("h_action", h[:, 0]), ("action_residual", raw[:, 0])):
            self.record(prefix + "_mean", values)
            self.record(prefix + "_positive_frac", values > 0)
            self.record(prefix + "_negative_frac", values < 0)

    def backward(self, gradient, rows, ids, h, mask, lp, lm, grad_scale, advantages):
        g = gradient[rows[:, None], ids].detach().float() * mask / grad_scale
        coef = (g * h).sum(-1)
        norm = g.norm(dim=-1).clamp_min(1e-12)
        ratio = coef.abs() / norm
        for prefix in ("h", "h1"):
            self.record(prefix + "_dot_grad_mean", coef)
            self.record(prefix + "_dot_grad_abs_mean", coef.abs())
            self.record(prefix + "_dot_grad_abs_over_grad_norm_mean", ratio)
            self.record(prefix + "_dot_grad_abs_over_grad_norm_max", ratio, "max")
            for name, select in (("positive", coef > 0), ("negative", coef < 0), ("zero", coef == 0)):
                self.record(prefix + "_dot_grad_" + name + "_frac", select)
            self.record(prefix + "_dot_update_positive_frac", coef < 0)
            self.record(prefix + "_dot_update_negative_frac", coef > 0)
            for bucket, low, high in (
                ("000_001", 0, 0.01),
                ("001_005", 0.01, 0.05),
                ("005_010", 0.05, 0.1),
                ("010_020", 0.1, 0.2),
                ("020_050", 0.2, 0.5),
                ("050_100", 0.5, math.inf),
            ):
                selected = (ratio >= low) & (ratio < high)
                key = prefix + "_dot_grad_abs_over_grad_norm_bin_" + bucket
                self.record(key + "_frac", selected)
                self.record(key + "_count", selected, "sum")
        update = -coef
        self.record("h_dot_policy_update_token_count", torch.ones_like(coef), "sum")
        self.record("h_dot_policy_update_mean", update)
        self.record("h_dot_policy_update_abs_mean", update.abs())
        self.record("h_dot_policy_update_dot_positive_frac", update > 0)
        self.record("h_dot_policy_update_dot_negative_frac", update < 0)
        for name, population in (("positive", advantages > 0), ("negative", advantages < 0)):
            selected = update[population]
            key = "h_dot_policy_update_when_advantage_" + name
            self.record(key + "_token_count", torch.ones_like(selected), "sum")
            for sign, values in (("positive", selected > 0), ("negative", selected < 0)):
                self.record(key + "_dot_" + sign + "_count", values, "sum")
                self.record(key + "_dot_" + sign + "_frac", values)
        cosine = (update / norm).clamp(-1, 1)
        theta = cosine.acos()
        for suffix, population in (
            ("", torch.ones_like(coef, dtype=torch.bool)),
            ("_adv_nonzero", advantages != 0),
        ):
            prefix = "h_neg_g_angle" + suffix
            self.record(prefix + "_token_count", torch.ones_like(coef[population]), "sum")
            self.record(prefix + "_theta_rad_mean", theta[population])
            self.record(prefix + "_theta_deg_mean", theta[population] * (180 / math.pi))
            self.record(prefix + "_cos_theta_mean", cosine[population])
            for name, half in (("aligned", cosine > 0), ("anti_aligned", cosine < 0)):
                self.record(prefix + "_" + name + "_frac", half[population])
                selected = theta[population & half]
                self.record(prefix + "_" + name + "_token_count", torch.ones_like(selected), "sum")
                self.record(prefix + "_" + name + "_theta_rad_mean", selected)
                self.record(prefix + "_" + name + "_theta_deg_mean", selected * (180 / math.pi))
                self.record(prefix + "_" + name + "_theta_deg_min", selected * (180 / math.pi), "min")
                self.record(prefix + "_" + name + "_theta_deg_max", selected * (180 / math.pi), "max")
        delta = torch.where(coef < 0, lp, lm)[:, None] * coef[:, None] * h
        delta_ratio = delta.norm(dim=-1) / norm
        self.record("effective_delta_over_grad_norm_mean", delta_ratio)
        self.record("effective_delta_over_grad_norm_max", delta_ratio, "max")
        self.record("correction_nonzero_token_fraction", delta.norm(dim=-1) > 0)

    def finish(self):
        distributed = dist.is_available() and dist.is_initialized()
        keys = sorted(self.stats)
        if distributed:
            all_keys = [None] * dist.get_world_size()
            dist.all_gather_object(all_keys, keys)
            keys = sorted(set().union(*(set(k) for k in all_keys)))
        if not keys:
            return {}
        device = (
            next(iter(self.stats.values())).device
            if self.stats
            else torch.device("cuda", torch.cuda.current_device())
        )
        empty = torch.tensor([0.0, 0.0, -math.inf], device=device, dtype=torch.float64)
        values = torch.stack([self.stats.get(key, empty) for key in keys])
        sums, maxima = values[:, :2].contiguous(), values[:, 2].contiguous()
        if distributed:
            dist.all_reduce(sums, op=dist.ReduceOp.SUM)
            dist.all_reduce(maxima, op=dist.ReduceOp.MAX)
        result = {}
        for (name, mode), (total, count), maximum in zip(keys, sums.cpu().tolist(), maxima.cpu().tolist()):
            if mode == "sum":
                result[name] = total
            elif count:
                result[name] = -maximum if mode == "min" else maximum if mode == "max" else total / count
        result.update(
            {
                "distillation/oprd_metrics_token_weighted": 1.0,
                "distillation/oprd_h_normalization_uniform_center_l2": 1.0,
                "distillation/oprd_projection_coefficient_mode_cosine_similarity": 0.0,
                "distillation/oprd_h_dot_policy_update_advantage_split_available": 1.0,
                "distillation/oprd_h_neg_g_angle_adv_nonzero_available": 1.0,
            }
        )
        return result
