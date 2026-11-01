"""Agent diagnostics using the repository's math/code metric conventions."""

from collections import defaultdict

import numpy as np


def numeric_stats(prefix, values):
    """Same finite-only mean/max/min reduction as math/code reward extras."""
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    values = np.asarray(values)
    if values.size == 0 or values.dtype.kind in ("S", "U"):
        return {}
    try:
        values = values.astype(np.float64).reshape(-1)
    except (ValueError, TypeError):
        return {}
    values = values[np.isfinite(values)]
    if not values.size:
        return {}
    return {
        f"{prefix}/{name}": float(fn(values))
        for name, fn in (("mean", np.mean), ("max", np.max), ("min", np.min))
    }


def group_score_metrics(batch):
    """Match math/code GRPO group metrics, including aborted-row exclusion."""
    tensors = getattr(batch, "batch", {})
    if "uid" not in batch.non_tensor_batch or "token_level_scores" not in tensors:
        return {}
    scores = tensors["token_level_scores"].sum(-1).detach().float().cpu().numpy()
    valid = tensors["response_mask"].sum(-1).detach().cpu().numpy() > 0
    uids = np.asarray(batch.non_tensor_batch["uid"]).reshape(-1)
    if len(uids) != len(scores):
        return {}
    groups = defaultdict(list)
    for uid, score, keep in zip(uids, scores, valid, strict=True):
        if keep:
            groups[uid].append(float(score))
    if not groups:
        return {}
    counts = dict(all_zero=0, all_one=0, all_same=0, mixed=0, singleton=0)
    stds = []
    for values in groups.values():
        values = np.asarray(values, dtype=np.float64)
        stds.append(float(np.std(values)))
        counts["singleton"] += len(values) <= 1
        same = np.max(values) - np.min(values) <= 1e-6
        counts["all_same"] += same
        counts["mixed"] += not same
        if same:
            counts["all_zero"] += abs(values[0]) <= 1e-6
            counts["all_one"] += abs(values[0] - 1) <= 1e-6
    prefix = "grpo/group_score"
    return {
        f"{prefix}/group_count": float(len(groups)),
        **{f"{prefix}/{key}_frac": float(value / len(groups)) for key, value in counts.items()},
        f"{prefix}/std_mean": float(np.mean(stds)),
        f"{prefix}/std_max": float(np.max(stds)),
    }


def training_metrics(batch):
    values = batch.non_tensor_batch
    result = {}
    # Keep correctness separate from the shaped reward; use the common 'acc' name.
    for source, target in (
        ("accuracy", "acc"),
        ("score", "score"),
        ("incorrect_format", "incorrect_format"),
        ("curiosity_bonus", "curiosity_bonus"),
        ("redundancy_penalty", "redundancy_penalty"),
        ("valid_tool_count", "valid_tool_count"),
        ("group_tool_rate", "group_tool_rate"),
    ):
        if source in values:
            result.update(numeric_stats(f"reward_extra/{target}", values[source]))
    if "valid_tool_count" in values:
        result.update(numeric_stats("tool_call_counts", values["valid_tool_count"]))
    stops = list(values.get("traj_stop_reason", []))
    reasons = {
        "answer",
        "response_budget",
        "action_length_limit",
        "max_turns_reached",
        "image_limit",
        "tool_error",
        "observation_response_budget",
    } | set(stops)
    if stops:
        for reason in sorted(reasons):
            result[f"verl_tool/stop_reason/{reason}/mean"] = stops.count(reason) / len(stops)
    tensors = batch.batch
    if "attention_mask" in tensors and "response_mask" in tensors:
        cap = tensors["responses"].shape[-1]
        trajectory = tensors["attention_mask"][:, -cap:].sum(-1)
        actions = tensors["response_mask"].sum(-1)
        result.update(numeric_stats("trajectory_length", trajectory))
        result.update(numeric_stats("observation_length", trajectory - actions))
        result["trajectory_length/clip_ratio"] = float((trajectory >= cap).float().mean())
    result.update(group_score_metrics(batch))
    return result
