"""The paper's support-local logit-gradient transform.

Backpropagation supplies a loss gradient, whose sign is opposite to the
ascent direction g in the paper. Negative alignment therefore uses lambda+.
"""

import torch


def linear_schedule(start, end, steps, step):
    if steps < 1 or step < 1:
        raise ValueError("Schedule steps and update index must be positive")
    progress = 1.0 if steps == 1 else min((step - 1) / (steps - 1), 1.0)
    return start + (end - start) * progress


def support_indices(student_logits, actions, topk=10):
    if topk < 1:
        raise ValueError("topk must be positive")
    ids = student_logits.topk(min(topk, student_logits.shape[-1]), dim=-1).indices
    actions = actions.unsqueeze(-1)
    indices = torch.cat((actions, ids), dim=-1)
    # Keep the sampled token once even when it is already in the top-k set.
    mask = torch.cat((torch.ones_like(actions, dtype=torch.bool), ids.ne(actions)), dim=-1)
    return indices, mask


def guidance_direction(student, teacher, reference, indices, mask, source="teacher_base"):
    student, teacher = student.float(), teacher.float()
    if source == "teacher_base":
        if reference is None:
            raise ValueError("OPRD requires the teacher pre-RL reference")
        h = (teacher - reference.float()).gather(-1, indices) * mask
        h = (h - h.sum(-1, keepdim=True) / mask.sum(-1, keepdim=True).clamp_min(1)) * mask
    elif source == "teacher_student":
        logits = student.gather(-1, indices).masked_fill(~mask, -torch.inf)
        weights = logits.softmax(-1).nan_to_num()
        shift = (teacher - student).gather(-1, indices)
        h = weights * (shift - (shift * weights).sum(-1, keepdim=True)) * mask
    elif source == "opsd_kl":
        h = (student.softmax(-1) - teacher.softmax(-1)).gather(-1, indices) * mask
    else:
        raise ValueError(f"Unsupported guidance direction: {source}")
    return h / h.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def precondition_loss_gradient(gradient, indices, direction, lambda_plus, lambda_minus):
    if lambda_plus < 0 or lambda_minus < 0:
        raise ValueError("OPRD branch strengths must be nonnegative")
    selected = gradient.float().gather(-1, indices)
    coefficient = (selected * direction).sum(-1, keepdim=True)
    strength = torch.where(coefficient < 0, lambda_plus, lambda_minus)
    change = strength * coefficient * direction
    result = gradient.clone()
    result.scatter_add_(-1, indices, change.to(result.dtype))
    return result
