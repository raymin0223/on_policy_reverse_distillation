"""Reward dispatch for the datasets evaluated in the OPRD paper."""


def default_compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    from .feedback import compute_score

    return compute_score(data_source, solution_str, ground_truth, extra_info)


compute_score = default_compute_score
