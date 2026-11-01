"""Sampled reverse-KL, matching the existing code/math OPD objective."""

import torch
from verl.trainer.ppo.core_algos import agg_loss


def sampled_rkl_loss(student, old, teacher, mask, clip_low=0.5, clip_high=2.0, global_batch_info=None):
    # Teacher-forced scores on student samples; no reward-derived advantage.
    active = mask.bool()
    student = torch.where(active, student.float(), 0.0)
    old = torch.where(active, old.detach().float(), 0.0)
    teacher = torch.where(active, teacher.detach().float(), 0.0)
    advantage = teacher - old
    raw_ratio = (student - old).clamp(-20, 20).exp()
    ratio = raw_ratio.clamp(clip_low, clip_high)
    denominator = active.sum().clamp_min(1)
    loss = agg_loss(-advantage * ratio, active, "token-mean", **(global_batch_info or {}))
    upper = ((raw_ratio > clip_high) & active).sum() / denominator
    lower = ((raw_ratio < clip_low) & active).sum() / denominator
    kl = ((old - student) * active).sum() / denominator
    return loss, upper, kl.detach(), lower


def sampled_token_logprobs(hidden, head, rows, labels, temperature=1.0, chunk_size=16):
    """Normalize over the full teacher LM head, only for generated predictor rows.

    Bounded chunks avoid a sequence-length x vocabulary allocation. Do not use
    a top-k normalizer: OPD needs actual sampled-token log probabilities.
    """
    result = []
    for row, target in zip(rows.split(chunk_size), labels.split(chunk_size)):
        logits = head(hidden.index_select(0, row)).float() / temperature
        result.append(logits.gather(1, target[:, None]).squeeze(1) - logits.logsumexp(-1))
    return torch.cat(result)
