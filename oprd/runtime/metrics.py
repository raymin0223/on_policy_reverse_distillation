"""Observational actor diagnostics, weighted by response tokens across DP ranks."""

import torch
import torch.distributed as dist


class TokenMetrics:
    """Combine microbatch token means without giving short sequences extra weight."""

    SUM_KEYS = ("distillation/kdrl_k2_token_count",)
    KEYS = (
        "actor/pg_clipfrac",
        "actor/ppo_kl",
        "actor/pg_clipfrac_lower",
        "distillation/sampled_rkl_advantage_mean",
        "distillation/sampled_rkl_ratio_mean",
        "distillation/sampled_rkl_clipfrac",
        "distillation/sampled_rkl_clipfrac_upper",
        "distillation/sampled_rkl_clipfrac_lower",
        "distillation/sampled_rkl_advantage_ratio_clipfrac",
        "distillation/kdrl_k2_log_prob_diff_mean",
    ) + SUM_KEYS

    def __init__(self, device, group=None):
        self.group = group
        self.totals = torch.zeros(len(self.KEYS), 2, dtype=torch.float64, device=device)

    @torch.no_grad()
    def record(self, metrics, token_count):
        for i, key in enumerate(self.KEYS):
            if key in metrics:
                value = torch.as_tensor(metrics[key], device=self.totals.device, dtype=torch.float64)
                if key in self.SUM_KEYS:
                    self.totals[i, 0] += value
                    self.totals[i, 1] += 1
                else:
                    # Empty masked means can be NaN; they carry no weight.
                    self.totals[i, 0] += torch.where(token_count > 0, value * token_count, 0.0)
                    self.totals[i, 1] += token_count

    def finish(self):
        values = self.totals.clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(values, group=self.group)
        return {
            key: total if key in self.SUM_KEYS else total / count
            for key, (total, count) in zip(self.KEYS, values.cpu().tolist())
            if count
        }


class ProjectionMetrics:
    def __init__(self, device, group=None):
        self.group = group
        # count, active direction, measurable alignment, cosine, relative change, changed
        self.totals = torch.zeros(6, dtype=torch.float64, device=device)

    @torch.no_grad()
    def skipped(self, token_count):
        """Include format-invalid trajectories even when no teacher forward is needed."""
        self.totals[0] += token_count

    @torch.no_grad()
    def record(self, selected_gradient, direction, support_mask, lp, lm, response_mask=None):
        g = selected_gradient.detach().float() * support_mask
        h = direction.detach().float()
        if response_mask is not None:
            g, h = g[response_mask.bool()], h[response_mask.bool()]
        g, h = g.reshape(-1, g.shape[-1]), h.reshape(-1, h.shape[-1])
        gn, hn = g.norm(dim=-1), h.norm(dim=-1)
        dot = (g * h).sum(-1)
        active = hn > 0
        measurable = active & (gn > 0)
        # The policy update is the negative loss gradient. Zero directions and
        # zero gradients have no angle; do not count them as orthogonal updates.
        cosine = (-dot / (gn * hn).clamp_min(1e-30)).clamp(-1, 1)
        change = torch.where(dot < 0, lp, lm) * dot.abs() * hn
        ratio = change / gn.clamp_min(1e-30)
        self.totals += torch.stack(
            (
                gn.new_tensor(gn.numel(), dtype=torch.float64),
                active.double().sum(),
                measurable.double().sum(),
                cosine[measurable].double().sum(),
                ratio.double().sum(),
                (change > 0).double().sum(),
            )
        )

    def finish(self):
        values = self.totals.clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(values, group=self.group)
        count, active, measurable, cosine, ratio, changed = values.cpu().tolist()
        prefix = "distillation/oprd_projection/"
        result = {
            prefix + "token_count": count,
            prefix + "active_fraction": active / max(count, 1),
            prefix + "correction_fraction": changed / max(count, 1),
            prefix + "relative_change_mean": ratio / max(count, 1),
            prefix + "alignment_token_count": measurable,
        }
        if measurable:
            result[prefix + "update_alignment_cosine"] = cosine / measurable
        return result
