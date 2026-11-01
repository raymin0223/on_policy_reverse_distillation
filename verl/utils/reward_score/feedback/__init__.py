"""Programmatic rewards used by the OPRD paper."""

from verl.utils.reward_score import reasoning_gym
from verl.utils.reward_score.feedback import math


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    if data_source in reasoning_gym.SCORERS:
        result = reasoning_gym.compute_score(solution_str, ground_truth, extra_info, task=data_source)
    elif data_source in {
        "math",
        "competition_math",
        "math_dapo",
        "dapo_math",
        "aime24",
        "aime25",
        "hmmt25",
        "olympiadbench",
    }:
        result = math.compute_score(solution_str, ground_truth, extra_info)
    elif data_source in {
        "lcbv5",
        "primeintellect",
        "taco",
        "deepcoder_rlvr",
        "codecontests",
        "codecontests_official_test",
        "apps",
        "apps_competition",
    }:
        from verl.utils.reward_score import deepcoder_rlvr

        result = deepcoder_rlvr.compute_score(solution_str, ground_truth, extra_info)
    else:
        raise ValueError(f"Dataset outside paper scope: {data_source}")
    if "incorrect_format" in result:
        result.setdefault("format_reward", 1.0 - float(result["incorrect_format"]))
    return result
