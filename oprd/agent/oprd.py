"""Agent OPRD math, shared by the worker and CPU checks.

Matches the repository's teacher_base / student_topk / uniform_center_l2 /
inner_product / signed_projection profile in verl/workers/actor/dp_actor.py.
"""

import torch
from torch.utils._pytree import tree_map


def teacher_inputs_on_device(inputs, device):
    """Move hook kwargs to the unwrapped teachers without changing dtype or values."""
    return tree_map(
        lambda value: (
            value.to(device=device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        ),
        inputs,
    )


def signed_lambdas(config, step):
    # Same one-based schedule as the code/math benchmarks: first update starts
    # at zero, update 45 reaches .5, and subsequent updates stay there.
    if step < 1:
        raise ValueError("OPRD requires the trainer's one-based global_steps")
    progress = min((step - 1) / max(config.schedule_steps - 1, 1), 1.0)
    return (
        float(config.lambda_plus),
        config.lambda_minus_start + progress * (config.lambda_minus_end - config.lambda_minus_start),
    )


def response_rows(micro_batch):
    """Map generated-token labels to their preceding rows in the packed logits."""
    mask = micro_batch["attention_mask"].bool()
    predictors = torch.zeros_like(mask)
    length = micro_batch["responses"].shape[-1]
    predictors[:, -length - 1 : -1] = micro_batch["response_mask"].bool()
    indices = mask.flatten().nonzero().squeeze(-1)
    rows = predictors.flatten()[indices].nonzero().squeeze(-1)
    labels = micro_batch["input_ids"].flatten()[indices].roll(-1)[rows]
    return rows, labels


def student_support(logits, rows, labels, topk, vocab_size, chunk_size=32):
    # Restrict to real, shared tokenizer IDs (Qwen models can have differently
    # padded LM heads). Keep the sampled action once even when it is in top-k.
    chunks = [
        logits.index_select(0, part)[:, :vocab_size].topk(topk, dim=-1).indices
        for part in rows.split(chunk_size)
    ]
    top_ids = torch.cat(chunks)
    ids = torch.cat((labels[:, None], top_ids), dim=-1)
    mask = torch.cat((torch.ones_like(labels[:, None], dtype=torch.bool), top_ids != labels[:, None]), dim=-1)
    return ids, mask


def residual_direction(teacher, base, support_mask):
    mask = support_mask.float()
    h = (teacher.float() - base.float()) * mask
    h = (h - h.sum(-1, keepdim=True) / mask.sum(-1, keepdim=True).clamp_min(1)) * mask
    return h / h.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def project_gradient(gradient, rows, ids, h, lambda_plus, lambda_minus):
    """Amplify the signed projection of the loss gradient, as in existing OPRD."""
    g = gradient[rows[:, None], ids].float()
    coefficient = (g * h).sum(-1, keepdim=True)
    scale = torch.where(coefficient < 0, lambda_plus, lambda_minus)
    delta = scale * coefficient * h
    result = gradient.clone()
    # Duplicate action entries have h=0, so accumulate without double counting.
    result.index_put_((rows[:, None], ids), delta.to(result.dtype), accumulate=True)
    return result


def selected_head_logits(hidden, head, rows, ids, chunk_size=128):
    """Score only action + top-k; never allocate teacher sequence x vocabulary."""
    result = []
    for row_chunk, id_chunk in zip(rows.split(chunk_size), ids.split(chunk_size)):
        states = hidden.index_select(0, row_chunk)
        weights = head.weight[id_chunk]
        scores = torch.einsum("rh,rkh->rk", states, weights).float()
        if head.bias is not None:
            scores += head.bias[id_chunk].float()
        result.append(scores)
    return torch.cat(result)
