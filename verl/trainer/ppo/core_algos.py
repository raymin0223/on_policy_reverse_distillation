# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO-like algorithms.
"""

__all__ = ["register_adv_est", "get_adv_estimator_fn", "AdvantageEstimator"]
from collections import defaultdict
from enum import Enum
from typing import Any, Callable, Optional
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
import verl.utils.torch_functional as verl_F
from verl.trainer.config import AlgoConfig
from verl.utils import as_torch_index, group_mean_std
from verl.utils.import_utils import deprecated
from verl.workers.config import ActorConfig

PolicyLossFn = Callable[
    [
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        str,
        Optional[DictConfig | ActorConfig],
        torch.Tensor | None,
    ],
    tuple[torch.Tensor, dict[str, Any]],
]
POLICY_LOSS_REGISTRY: dict[str, PolicyLossFn] = {}


def register_policy_loss(name: str) -> Callable[[PolicyLossFn], PolicyLossFn]:
    """Register a policy loss function with the given name.

    Args:
        name (str): The name to register the policy loss function under.

    Returns:
        function: Decorator function that registers the policy loss function.
    """

    def decorator(func: PolicyLossFn) -> PolicyLossFn:
        POLICY_LOSS_REGISTRY[name] = func
        return func

    return decorator


def get_policy_loss_fn(name):
    """Get the policy loss with a given name.

    Args:
        name: `(str)`
            The name of the policy loss.

    Returns:
        `(callable)`: The policy loss function.
    """
    loss_name = name
    if loss_name not in POLICY_LOSS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(POLICY_LOSS_REGISTRY.keys())}"
        )
    return POLICY_LOSS_REGISTRY[loss_name]


class AdvantageEstimator(str, Enum):
    """Using an enumeration class to avoid spelling errors in adv_estimator.

    Note(haibin.lin): this enum class is immutable after creation. Extending this
    enum for new estimators may not be necessary since users can always just call
    `verl.trainer.ppo.core_algos.register` with string name for a custom advantage
    estimator instead.
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    OPO = "opo"
    GRPO_PASSK = "grpo_passk"
    GPG = "gpg"
    RLOO_VECTORIZED = "rloo_vectorized"
    GRPO_VECTORIZED = "grpo_vectorized"
    OPTIMAL_TOKEN_BASELINE = "optimal_token_baseline"
    TIR_OPTIMAL_TOKEN_BASELINE = "tir_optimal_token_baseline"


ADV_ESTIMATOR_REGISTRY: dict[str, Any] = {}


def register_adv_est(name_or_enum: str | AdvantageEstimator) -> Any:
    """Decorator to register a advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    """

    def decorator(fn):
        name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
        if name in ADV_ESTIMATOR_REGISTRY and ADV_ESTIMATOR_REGISTRY[name] != fn:
            raise ValueError(
                f"Adv estimator {name} has already been registered: {ADV_ESTIMATOR_REGISTRY[name]} vs {fn}"
            )
        ADV_ESTIMATOR_REGISTRY[name] = fn
        return fn

    return decorator


def get_adv_estimator_fn(name_or_enum):
    """Get the advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    Returns:
        `(callable)`: The advantage estimator function.
    """
    name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
    if name not in ADV_ESTIMATOR_REGISTRY:
        raise ValueError(f"Unknown advantage estimator simply: {name}")
    return ADV_ESTIMATOR_REGISTRY[name]


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        """Update the KL coefficient based on current KL divergence.

        Args:
            current_kl (float): Current KL divergence value.
            n_steps (int): Number of steps taken.
        """
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        """Update method for fixed KL controller (no-op).

        Args:
            current_kl (float): Current KL divergence value (unused).
            n_steps (int): Number of steps taken (unused).
        """
        pass


def get_kl_controller(kl_ctrl):
    """Factory function to create appropriate KL controller based on configuration.

    Args:
        kl_ctrl: Configuration object containing KL controller settings.

    Returns:
        KL controller instance (FixedKLController or AdaptiveKLController).

    Raises:
        NotImplementedError: If controller type is not supported.
        AssertionError: If adaptive controller horizon is not positive.
    """
    if kl_ctrl.type == "fixed":
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    elif kl_ctrl.type == "adaptive":
        assert kl_ctrl.horizon > 0, f"horizon must be larger than 0. Got {kl_ctrl.horizon}"
        return AdaptiveKLController(
            init_kl_coef=kl_ctrl.kl_coef, target_kl=kl_ctrl.target_kl, horizon=kl_ctrl.horizon
        )
    else:
        raise NotImplementedError


@register_adv_est(AdvantageEstimator.GAE)
def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        values: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma is `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        nextvalues = 0
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]
        for t in reversed(range(gen_len)):
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam_ = delta + gamma * lam * lastgaelam
            nextvalues = values[:, t] * response_mask[:, t] + (1 - response_mask[:, t]) * nextvalues
            lastgaelam = lastgaelam_ * response_mask[:, t] + (1 - response_mask[:, t]) * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, response_mask)
    return (advantages, returns)


@register_adv_est(AdvantageEstimator.GRPO)
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-06,
    norm_adv_by_std_in_grpo: bool = False,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length)
        index: `(np.ndarray)`
            index array for grouping
        epsilon: `(float)`
            small value to avoid division by zero
        norm_adv_by_std_in_grpo: `(bool)`
            whether to scale the GRPO advantage
        config: `(Optional[AlgoConfig])`
            algorithm configuration object

    Note:
        If norm_adv_by_std_in_grpo is True, the advantage is scaled by the std, as in the original GRPO.
        If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        Returns: `(torch.Tensor)`
            shape is (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)
    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask
    return (scores, scores)


@register_adv_est(AdvantageEstimator.GRPO_VECTORIZED)
def compute_grpo_vectorized_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-06,
    norm_adv_by_std_in_grpo: bool = False,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized GRPO（outcome-only）:
      For each group g:
      a_i = \\frac{r_i - \\mu_g}{\\sigma_g} (or without dividing by \\sigma_g),
      then broadcast the scalar across the token dimension (multiplied by response_mask).。
    """
    with torch.no_grad():
        scores = token_level_rewards.sum(dim=-1)
        g = as_torch_index(index, device=scores.device)
        mean_g, std_g, _ = group_mean_std(scores, g, eps=epsilon)
        if norm_adv_by_std_in_grpo:
            scalars = (scores - mean_g[g]) / (std_g[g] + epsilon)
        else:
            scalars = scores - mean_g[g]
        advantages = scalars.unsqueeze(-1) * response_mask
        return (advantages, advantages)


@register_adv_est(AdvantageEstimator.GRPO_PASSK)
def compute_grpo_passk_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-06,
    norm_adv_by_std_in_grpo: bool = False,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for Pass@k using a GRPO-style outcome reward formulation.
    Only the best response per group gets a non-zero advantage: r_max - r_second_max.

    Implemented as described in https://arxiv.org/abs/2503.19595.

    Args:
        token_level_rewards: (bs, response_length)
        response_mask: (bs, response_length)
        index: (bs,) → group ID per sample
        epsilon: float for numerical stability
        config: (AlgoConfig) algorithm settings, which contains "norm_adv_by_std_in_grpo"

    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length)
    """
    assert config is not None
    norm_adv_by_std_in_grpo = config.get("norm_adv_by_std_in_grpo", False)
    scores = token_level_rewards.sum(dim=-1)
    advantages = torch.zeros_like(scores)
    id2scores = defaultdict(list)
    id2indices = defaultdict(list)
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            idx = index[i]
            id2scores[idx].append(scores[i])
            id2indices[idx].append(i)
        for idx in id2scores:
            rewards = torch.stack(id2scores[idx])
            if rewards.numel() < 2:
                raise ValueError(
                    f"Pass@k requires at least 2 samples per group. Got {rewards.numel()} for group {idx}."
                )
            topk, topk_idx = torch.topk(rewards, 2)
            r_max, r_second_max = (topk[0], topk[1])
            i_max = id2indices[idx][topk_idx[0].item()]
            advantage = r_max - r_second_max
            if norm_adv_by_std_in_grpo:
                std = torch.std(rewards)
                advantage = advantage / (std + epsilon)
            advantages[i_max] = advantage
    advantages = advantages.unsqueeze(-1) * response_mask
    return (advantages, advantages)


@register_adv_est(AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE)
def compute_reinforce_plus_plus_baseline_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: torch.Tensor,
    epsilon: float = 1e-06,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RF++-baseline (https://arxiv.org/abs/2501.03262), operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)
    id2score = defaultdict(list)
    id2mean = {}
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        scores = verl_F.masked_whiten(scores, response_mask) * response_mask
    return (scores, scores)


@register_adv_est(AdvantageEstimator.RLOO)
def compute_rloo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-06,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)
    id2score = defaultdict(list)
    id2mean = {}
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[
                    index[i]
                ] * response_num / (response_num - 1)
        scores = scores.unsqueeze(-1) * response_mask
    return (scores, scores)


@register_adv_est(AdvantageEstimator.OPO)
def compute_opo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-06,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for OPO based on https://arxiv.org/pdf/2505.23585

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = response_mask.sum(dim=-1)
    scores = token_level_rewards.sum(dim=-1)
    id2score = defaultdict(list)
    id2len = defaultdict(list)
    id2bsl = {}
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
            id2len[index[i]].append(response_length[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2bsl[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                score_tensor = torch.stack(id2score[idx])
                len_tensor = torch.stack(id2len[idx])
                id2bsl[idx] = (len_tensor * score_tensor).sum() / len_tensor.sum()
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2bsl[index[i]]
        scores = scores.unsqueeze(-1) * response_mask
    return (scores, scores)


@register_adv_est(AdvantageEstimator.REINFORCE_PLUS_PLUS)
def compute_reinforce_plus_plus_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    assert config is not None
    gamma = config.gamma
    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0
        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            running_return = running_return * response_mask[:, t]
        advantages = verl_F.masked_whiten(returns, response_mask)
        advantages = advantages * response_mask
    return (advantages, returns)


@register_adv_est(AdvantageEstimator.REMAX)
def compute_remax_outcome_advantage(
    token_level_rewards: torch.Tensor,
    reward_baselines: torch.Tensor,
    response_mask: torch.Tensor,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    with torch.no_grad():
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1) * response_mask
    return (advantages, returns)


@register_adv_est(AdvantageEstimator.GPG)
def compute_gpg_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-06,
    f_norm: float = 1.0,
    alpha: float = 1.0,
    config=None,
    **kwargs,
):
    """
    Compute advantage for GPG, operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        index: `(np.ndarray)`
            shape: (bs,)
        epsilon: (float)
        f_norm: (float)
        alpha: (float)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)
    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    with torch.no_grad():
        bsz = scores.shape[0]
        m = torch.count_nonzero(scores)
        alpha = bsz / m.clamp(min=1)
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = alpha * (scores[i] - id2mean[index[i]]) / f_norm
        scores = scores.unsqueeze(-1) * response_mask
    return (scores, scores)


@register_adv_est(AdvantageEstimator.RLOO_VECTORIZED)
def compute_rloo_vectorized_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-06,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)
    with torch.no_grad():
        inv = torch.from_numpy(np.unique(index, return_inverse=True)[1]).to(scores.device)
        c = torch.bincount(inv)[inv].to(scores.dtype)
        adv = (c * scores - torch.bincount(inv, weights=scores)[inv]) / (c - 1).clamp_min(1) * (c > 1)
        adv = adv.unsqueeze(-1) * response_mask
    return (adv, adv)


@register_adv_est(AdvantageEstimator.OPTIMAL_TOKEN_BASELINE)
def compute_optimal_token_baseline_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    old_log_probs: torch.Tensor,
    sum_pi_squared: torch.Tensor,
    rollout_is_weights: torch.Tensor = None,
    handle_zero_tail: bool = False,
    epsilon: float = 1e-08,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantages using Optimal Token Baseline (OTB).

    Unlike the group mean based baseline which uses a single baseline per trajectory,
    this computes a unique baseline for each timestep using cumulative path variance.

    Theory:
        For each timestep t in each prompt group:
            B_t* = E[G_t × W_t] / E[W_t]
        where W_t = Σ_{j=1}^t ||s_j||² (cumulative path-variance proxy)
        and ||s_j||² = 1 - 2π_j + Σπ²

    The cumulative sum W_t captures the "realized energy" of trajectory has been up to timestep t,
    giving higher weight to predicting rewards on high-variance paths.

    Args:
        token_level_rewards: Rewards at each token position [shape: (bs, response_length)]
        response_mask: Binary mask for valid tokens (1) vs padding (0) [shape: (bs, response_length)]
        index: Prompt indices for grouping trajectories from same prompt [shape: (bs,)]
        old_log_probs: Log probabilities from training policy during generation [shape: (bs, response_length)]
        sum_pi_squared: Sum of squared probabilities over vocabulary Σπ² [shape: (bs, response_length)]
        rollout_is_weights: Pre-computed IS weights for W correction [shape: (bs, response_length)],
            None if not using IS
        handle_zero_tail: If True, zero baselines will be set in the portion of the longest trajectory
            that extends beyond the second-longest trajectory in the prompt group.
            Default: False
        epsilon: Small constant for numerical stability (default: 1e-8)

    Returns:
        advantages: OTB advantage estimates [shape: (bs, response_length)]
        returns: Cumulative rewards (returns) from each position [shape: (bs, response_length)]

    Note on Rollout Importance Sampling:
        When rollout_is_weights is provided, W_t is scaled by ρ̄²(t) to minimize MSE under truncated IS:
            B_t* = Σ[G_t × ρ̄²(t) × W_t] / Σ[ρ̄²(t) × W_t]
    """
    with torch.no_grad():
        batch_size, seq_len = token_level_rewards.shape
        device = token_level_rewards.device
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        pi_t = torch.exp(old_log_probs)
        w_per_timestep = 1 - 2 * pi_t + sum_pi_squared
        if rollout_is_weights is not None:
            w_per_timestep = w_per_timestep * rollout_is_weights**2
        w_cumulative = (w_per_timestep * response_mask).cumsum(dim=-1)
        prompt_groups = defaultdict(list)
        for i in range(batch_size):
            prompt_groups[index[i]].append(i)
        baselines = torch.zeros_like(returns)
        for _, trajectory_indices in prompt_groups.items():
            N = len(trajectory_indices)
            if N == 1:
                continue
            traj_idx = torch.tensor(trajectory_indices, device=device)
            returns_group = returns[traj_idx]
            w_cumulative_group = w_cumulative[traj_idx]
            mask_group = response_mask[traj_idx]
            numerator = (returns_group * w_cumulative_group * mask_group).sum(dim=0)
            denominator = (w_cumulative_group * mask_group).sum(dim=0) + epsilon
            baseline_per_step = numerator / denominator
            baselines[traj_idx] = baseline_per_step.unsqueeze(0).expand(N, -1)
            if handle_zero_tail:
                response_lengths = mask_group.sum(dim=-1)
                sorted_lengths, _ = torch.sort(response_lengths)
                max_length = int(sorted_lengths[-1].item())
                second_max_length = int(sorted_lengths[-2].item())
                max_length_idx = (response_lengths == max_length).nonzero(as_tuple=True)[0]
                if max_length_idx.numel() == 1 and max_length > second_max_length:
                    max_length_traj_idx = trajectory_indices[int(max_length_idx[0])]
                    baselines[max_length_traj_idx, second_max_length:] = 0.0
        advantages = (returns - baselines) * response_mask
    return (advantages, returns)


@register_adv_est(AdvantageEstimator.TIR_OPTIMAL_TOKEN_BASELINE)
def compute_multi_turn_optimal_token_baseline_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    old_log_probs: torch.Tensor,
    sum_pi_squared: torch.Tensor,
    rollout_is_weights: torch.Tensor = None,
    handle_zero_tail: bool = True,
    epsilon: float = 1e-08,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantages using Optimal Token Baseline (OTB).

    Unlike the group mean based baseline which uses a single baseline per trajectory,
    this computes a unique baseline for each timestep using cumulative path variance.

    Theory:
        For each timestep t in each prompt group:
            B_t* = E[G_t × W_t] / E[W_t]
        where W_t = Σ_{j=1}^t ||s_j||² (cumulative path-variance proxy)
        and ||s_j||² = 1 - 2π_j + Σπ²

    The cumulative sum W_t captures the "realized energy" of trajectory has been up to timestep t,
    giving higher weight to predicting rewards on high-variance paths.

    Args:
        token_level_rewards: Rewards at each token position [shape: (bs, response_length)]
        response_mask: Binary mask for valid tokens (1) vs padding (0) [shape: (bs, response_length)]
        index: Prompt indices for grouping trajectories from same prompt [shape: (bs,)]
        old_log_probs: Log probabilities from training policy during generation [shape: (bs, response_length)]
        sum_pi_squared: Sum of squared probabilities over vocabulary Σπ² [shape: (bs, response_length)]
        rollout_is_weights: Pre-computed IS weights for W correction [shape: (bs, response_length)],
            None if not using IS
        handle_zero_tail: If True, zero baselines will be set in the portion of the longest trajectory
            that extends beyond the second-longest trajectory in the prompt group.
            Default: False
        epsilon: Small constant for numerical stability (default: 1e-8)

    Returns:
        advantages: OTB advantage estimates [shape: (bs, response_length)]
        returns: Cumulative rewards (returns) from each position [shape: (bs, response_length)]

    Note on Rollout Importance Sampling:
        When rollout_is_weights is provided, W_t is scaled by ρ̄²(t) to minimize MSE under truncated IS:
            B_t* = Σ[G_t × ρ̄²(t) × W_t] / Σ[ρ̄²(t) × W_t]
    """
    with torch.no_grad():
        token_returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        pi_t = torch.exp(old_log_probs)
        w_per_timestep = 1 - 2 * pi_t + sum_pi_squared
        if rollout_is_weights is not None:
            w_per_timestep = w_per_timestep * rollout_is_weights**2
        w_cumulative = (w_per_timestep * response_mask).cumsum(dim=-1)
        response_lengths = response_mask.sum(dim=-1).to(dtype=torch.long)
        max_response_length = int(response_lengths.max().item()) if response_lengths.numel() > 0 else 0
        all_w_values = w_cumulative.new_zeros((len(response_lengths), max_response_length))
        all_returns = torch.zeros_like(all_w_values)
        for i in range(len(response_lengths)):
            length = int(response_lengths[i].item())
            if length == 0:
                continue
            mask = response_mask[i].bool()
            all_w_values[i, :length] = w_cumulative[i, mask]
            all_returns[i, :length] = token_returns[i, mask]
        prompt_groups = defaultdict(list)
        for i in range(len(response_lengths)):
            if response_lengths[i] == 0:
                continue
            prompt_groups[index[i]].append(i)
        baselines = torch.zeros_like(all_returns)
        for _, trajectory_indices in prompt_groups.items():
            N = len(trajectory_indices)
            traj_idx = torch.tensor(trajectory_indices, device=all_returns.device)
            if N == 1:
                baselines[traj_idx[0]] = 0.0
                continue
            w_group = all_w_values[traj_idx]
            R_group = all_returns[traj_idx]
            b_star = (R_group * w_group).sum(dim=0) / (w_group.sum(dim=0) + epsilon)
            baselines[traj_idx] = b_star.to(baselines.dtype)
            if handle_zero_tail:
                response_lengths_group = response_lengths[traj_idx]
                sorted_lengths, _ = torch.sort(response_lengths_group)
                max_length = int(sorted_lengths[-1].item())
                second_max_length = int(sorted_lengths[-2].item())
                max_length_idx = (response_lengths_group == max_length).nonzero(as_tuple=True)[0]
                if max_length_idx.numel() == 1 and max_length > second_max_length:
                    max_length_traj_idx = trajectory_indices[int(max_length_idx[0])]
                    baselines[max_length_traj_idx, second_max_length:] = 0.0
        all_advantages = all_returns - baselines
        advantages = torch.zeros_like(token_returns)
        for i in range(len(response_lengths)):
            if response_lengths[i] == 0:
                continue
            advantages[i, response_mask[i].bool()] = all_advantages[i, : response_lengths[i]]
        advantages = advantages * response_mask
    return (advantages, token_returns)


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    """Compute token-level rewards with KL penalty.

    Args:
        token_level_scores (torch.Tensor): Token-level reward scores.
        old_log_prob (torch.Tensor): Log probabilities from current policy.
        ref_log_prob (torch.Tensor): Log probabilities from reference policy.
        kl_ratio (float): KL penalty coefficient.

    Returns:
        torch.Tensor: Token-level rewards with KL penalty applied.
    """
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def global_token_mean_batch_info(
    loss_mask: torch.Tensor, collective_device=None, data_parallel_group=None
) -> dict[str, int]:
    """Normalize one optimizer mini-batch, before fixed or dynamic micro-batching.

    FSDP averages gradients across ranks, so ``agg_loss`` must multiply each
    micro-batch's partial token sum by the DP size after dividing by the
    DP-wide valid-token count. The group excludes sequence-parallel peers,
    matching the current engine worker path.
    """
    token_count = (
        loss_mask.sum()
        .detach()
        .to(
            device=collective_device if collective_device is not None else loss_mask.device, dtype=torch.int64
        )
    )
    dp_size = 1
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        dp_size = torch.distributed.get_world_size(group=data_parallel_group)
        torch.distributed.all_reduce(
            token_count, op=torch.distributed.ReduceOp.SUM, group=data_parallel_group
        )
    if token_count.item() <= 0:
        raise ValueError("token-mean requires valid response tokens in every optimizer mini-batch")
    return {"dp_size": dp_size, "batch_num_tokens": token_count.item()}


def agg_loss(
    loss_mat: torch.Tensor,
    loss_mask: torch.Tensor,
    loss_agg_mode: str,
    dp_size: int = 1,
    batch_num_tokens: Optional[int] = None,
    global_batch_size: Optional[int] = None,
    loss_scale_factor: Optional[int] = None,
    auto_dp_sync: bool = False,
):
    """
    Aggregate the loss across global batch to ensure the loss is invariant to fsdp/megatron parallelism.

    NOTE: The returned loss has different behaviors for different backend:
    - FSDP: the loss is directly used for backward.
    - Megatron: the loss should be scaled by `num_microbatches` and `cp_size` for pp schedule.

    Args:
        loss_mat: micro batch loss matrix, (bs, response_length)
        loss_mask: micro batch loss mask, (bs, response_length)
        loss_agg_mode: method to aggregate the loss matrix into a scalar
        dp_size: data parallel size
        batch_num_tokens: number of valid tokens in global batch
        global_batch_size: global batch size
        loss_scale_factor: scale factor for "seq-mean-token-sum-norm" mode. If None, uses loss_mask.shape[-1].
            Set this to a constant value to ensure consistent normalization throughout training.
        auto_dp_sync: if True, infer global token/sequence counts from torch.distributed when explicit
            global batch info is not provided. This is intended for legacy data-parallel workers.

    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if (
        auto_dp_sync
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and (dp_size == 1)
    ):
        world_size = torch.distributed.get_world_size()
        if world_size > 1 and loss_agg_mode == "token-mean":
            dp_size = world_size
            if batch_num_tokens is None:
                batch_num_tokens = loss_mask.sum().detach().to(device=loss_mask.device, dtype=torch.float32)
                torch.distributed.all_reduce(batch_num_tokens, op=torch.distributed.ReduceOp.SUM)
                batch_num_tokens = batch_num_tokens.clamp(min=1.0)
        elif world_size > 1 and loss_agg_mode in {"seq-mean-token-sum", "seq-mean-token-mean"}:
            dp_size = world_size
            if global_batch_size is None:
                seq_mask = (torch.sum(loss_mask, dim=-1) > 0).to(dtype=torch.float32)
                global_batch_size = seq_mask.sum().detach().to(device=loss_mask.device, dtype=torch.float32)
                torch.distributed.all_reduce(global_batch_size, op=torch.distributed.ReduceOp.SUM)
                global_batch_size = global_batch_size.clamp(min=1.0)

    def _clamp_min_one(value):
        return value.clamp(min=1.0) if torch.is_tensor(value) else max(value, 1.0)

    if loss_agg_mode == "token-mean":
        if batch_num_tokens is None:
            batch_num_tokens = loss_mask.sum().clamp(min=1.0)
        loss = verl_F.masked_sum(loss_mat, loss_mask) / batch_num_tokens * dp_size
    elif loss_agg_mode == "seq-mean-token-sum":
        masked_loss_mat = torch.where(loss_mask.bool(), loss_mat, torch.zeros_like(loss_mat)) * loss_mask
        seq_losses = torch.sum(masked_loss_mat, dim=-1)
        seq_mask = (torch.sum(loss_mask, dim=-1) > 0).float()
        if global_batch_size is None:
            global_batch_size = seq_mask.sum().clamp(min=1.0)
        else:
            global_batch_size = _clamp_min_one(global_batch_size)
        loss = verl_F.masked_sum(seq_losses, seq_mask) / global_batch_size * dp_size
    elif loss_agg_mode == "seq-mean-token-mean":
        masked_loss_mat = torch.where(loss_mask.bool(), loss_mat, torch.zeros_like(loss_mat)) * loss_mask
        seq_token_count = torch.sum(loss_mask, dim=-1)
        seq_losses = torch.sum(masked_loss_mat, dim=-1) / (seq_token_count + 1e-08)
        seq_mask = (seq_token_count > 0).float()
        if global_batch_size is None:
            global_batch_size = seq_mask.sum().clamp(min=1.0)
        else:
            global_batch_size = _clamp_min_one(global_batch_size)
        loss = verl_F.masked_sum(seq_losses, seq_mask) / global_batch_size * dp_size
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        masked_loss_mat = torch.where(loss_mask.bool(), loss_mat, torch.zeros_like(loss_mat)) * loss_mask
        seq_losses = torch.sum(masked_loss_mat, dim=-1)
        if loss_scale_factor is None:
            loss_scale_factor = loss_mask.shape[-1]
        loss = torch.sum(seq_losses) / loss_scale_factor
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")
    return loss


def _compute_distillation_is_ratio(
    *,
    student_log_probs: torch.Tensor,
    old_log_probs: Optional[torch.Tensor],
    is_clip: float,
    is_clip_low: Optional[float] = None,
    is_clip_high: Optional[float] = None,
    loss_mask: torch.Tensor,
    metrics: dict[str, Any],
    metric_prefix: str,
    include_ratio_max: bool = False,
    detach_ratio: bool = True,
) -> torch.Tensor:
    if old_log_probs is None:
        raise ValueError("old_log_probs is required for distillation IS ratio.")
    if is_clip <= 0.0:
        raise ValueError(f"distillation.is_clip must be positive for IS clipping, got {is_clip}.")
    clip_min = float(is_clip_low) if is_clip_low is not None else None
    clip_max = float(is_clip_high) if is_clip_high is not None else float(is_clip)
    if clip_min is not None and clip_min <= 0.0:
        raise ValueError(f"distillation.is_clip_low must be positive when provided, got {clip_min}.")
    if clip_max <= 0.0:
        raise ValueError(
            f"distillation.is_clip_high/is_clip must be positive for IS clipping, got {clip_max}."
        )
    if clip_min is not None and clip_min > clip_max:
        raise ValueError(
            f"distillation IS clipping requires is_clip_low <= is_clip_high/is_clip, got is_clip_low={clip_min}, upper={clip_max}."
        )
    log_ratio = torch.clamp(student_log_probs - old_log_probs, min=-20.0, max=20.0)
    if detach_ratio:
        log_ratio = log_ratio.detach()
    unclipped_ratio = torch.exp(log_ratio)
    ratio = unclipped_ratio.clamp(min=clip_min, max=clip_max)
    metrics[f"{metric_prefix}_ratio_mean"] = verl_F.masked_mean(ratio.detach(), loss_mask).detach().item()
    if include_ratio_max:
        metrics[f"{metric_prefix}_ratio_max"] = ratio.detach().masked_fill(loss_mask <= 0, 0.0).max().item()
    metrics[f"{metric_prefix}_clipfrac_upper"] = (
        verl_F.masked_mean((unclipped_ratio > clip_max).float(), loss_mask).detach().item()
    )
    metrics[f"{metric_prefix}_clipfrac_lower"] = (
        verl_F.masked_mean(
            (unclipped_ratio < clip_min).float()
            if clip_min is not None
            else torch.zeros_like(unclipped_ratio),
            loss_mask,
        )
        .detach()
        .item()
    )
    metrics[f"{metric_prefix}_clipfrac"] = (
        metrics[f"{metric_prefix}_clipfrac_upper"] + metrics[f"{metric_prefix}_clipfrac_lower"]
    )
    return ratio


def compute_kdrl_k2_loss(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    distillation_mask: Optional[torch.Tensor] = None,
    loss_agg_mode: str = "token-mean",
    auto_dp_sync: bool = False,
    global_batch_info: Optional[dict] = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute KDRL's sampled-token k2 auxiliary loss.

    Rollout tokens are sampled from the student and teacher-forced through the
    frozen teacher. The estimator is evaluated in float32:

        0.5 * (log pi_student - log pi_teacher) ** 2.
    """
    if student_log_probs.shape != teacher_log_probs.shape:
        raise ValueError(
            f"KDRL k2 requires student and teacher sampled-token log-probs to have the same shape, got student={tuple(student_log_probs.shape)} and teacher={tuple(teacher_log_probs.shape)}."
        )
    if response_mask.shape != student_log_probs.shape:
        raise ValueError(
            f"KDRL k2 requires response_mask to match sampled-token log-probs, got mask={tuple(response_mask.shape)} and log_probs={tuple(student_log_probs.shape)}."
        )
    loss_mask = response_mask
    if distillation_mask is not None:
        if distillation_mask.dim() == 1:
            loss_mask = loss_mask * distillation_mask.unsqueeze(1)
        elif distillation_mask.dim() == 2:
            if distillation_mask.shape != response_mask.shape:
                raise ValueError(
                    f"KDRL k2 rank-2 distillation_mask must match response_mask, got distillation_mask={tuple(distillation_mask.shape)} and response_mask={tuple(response_mask.shape)}."
                )
            loss_mask = loss_mask * distillation_mask
        else:
            raise ValueError(
                f"KDRL k2 distillation_mask must be rank 1 or 2, got shape={tuple(distillation_mask.shape)}."
            )
    log_prob_diff = student_log_probs.float() - teacher_log_probs.detach().float()
    per_token_loss = 0.5 * log_prob_diff.square()
    auto_dp_sync_active = (
        auto_dp_sync
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and (torch.distributed.get_world_size() > 1)
    )
    agg_loss_kwargs = {"auto_dp_sync": auto_dp_sync and not global_batch_info}
    if loss_agg_mode == "token-mean" and not global_batch_info and (not auto_dp_sync_active):
        agg_loss_kwargs["batch_num_tokens"] = loss_mask.sum().clamp(min=1.0)
    loss = agg_loss(
        loss_mat=per_token_loss,
        loss_mask=loss_mask,
        loss_agg_mode=loss_agg_mode,
        **(global_batch_info or agg_loss_kwargs),
    )
    metrics = {
        "distillation/kdrl_k2_enabled": 1.0,
        "distillation/kdrl_k2_log_prob_diff_mean": verl_F.masked_mean(
            log_prob_diff.detach(), loss_mask
        ).item(),
        "distillation/kdrl_k2_token_count": loss_mask.sum().detach().item(),
    }
    return (loss, metrics)


def compose_delta_target_scores(
    student_base_log_probs: torch.Tensor,
    teacher_rl_log_probs: torch.Tensor,
    teacher_base_log_probs: torch.Tensor,
    scale: float = 1.0,
    anchor_alpha: float = 1.0,
    student_old_log_probs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compose sampled-token scores for a student-anchored teacher delta."""
    shapes = {
        tuple(student_base_log_probs.shape),
        tuple(teacher_rl_log_probs.shape),
        tuple(teacher_base_log_probs.shape),
    }
    if len(shapes) != 1:
        raise ValueError(
            f"Delta target scores require matching sampled-token shapes, got student_base={tuple(student_base_log_probs.shape)}, teacher_rl={tuple(teacher_rl_log_probs.shape)}, teacher_base={tuple(teacher_base_log_probs.shape)}."
        )
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"Delta target scale must be finite and positive, got {scale}.")
    anchor_alpha = float(anchor_alpha)
    if not np.isfinite(anchor_alpha) or anchor_alpha <= 0.0:
        raise ValueError(f"Delta target anchor alpha must be finite and positive, got {anchor_alpha}.")
    if anchor_alpha != 1.0:
        if student_old_log_probs is None:
            raise ValueError("A non-unit delta anchor alpha requires sampled student old log-probabilities.")
        if tuple(student_old_log_probs.shape) != tuple(student_base_log_probs.shape):
            raise ValueError(
                f"Delta target scores require matching sampled-token old-policy shape, got student_old={tuple(student_old_log_probs.shape)}, student_base={tuple(student_base_log_probs.shape)}."
            )
    scores = student_base_log_probs.detach().float() + scale * (
        teacher_rl_log_probs.detach().float() - teacher_base_log_probs.detach().float()
    )
    if anchor_alpha != 1.0:
        scores = scores + (1.0 - anchor_alpha) * (
            student_old_log_probs.detach().float() - student_base_log_probs.detach().float()
        )
    return scores


def update_direct_opd_anchor_alpha(
    alpha: float,
    mean_teacher_shift: float,
    epsilon: float = 0.01,
    alpha_min: float = 0.5,
    alpha_max: float = 2.5,
) -> float:
    """Apply Direct-OPD's multiplicative adaptive-KL controller."""
    alpha = float(alpha)
    mean_teacher_shift = float(mean_teacher_shift)
    epsilon = float(epsilon)
    alpha_min = float(alpha_min)
    alpha_max = float(alpha_max)
    values = (alpha, mean_teacher_shift, epsilon, alpha_min, alpha_max)
    if not all((np.isfinite(value) for value in values)):
        raise ValueError(f"Direct-OPD alpha controller requires finite values, got {values}.")
    if alpha <= 0.0 or epsilon < 0.0 or epsilon >= 1.0:
        raise ValueError(
            f"Direct-OPD alpha controller requires alpha > 0 and epsilon in [0, 1), got alpha={alpha}, epsilon={epsilon}."
        )
    if alpha_min <= 0.0 or alpha_min > alpha_max or (not alpha_min <= alpha <= alpha_max):
        raise ValueError(
            f"Direct-OPD alpha controller requires 0 < alpha_min <= alpha <= alpha_max, got alpha_min={alpha_min}, alpha={alpha}, alpha_max={alpha_max}."
        )
    sign = 1.0 if mean_teacher_shift > 0.0 else -1.0 if mean_teacher_shift < 0.0 else 0.0
    return float(np.clip(alpha * (1.0 + epsilon * sign), alpha_min, alpha_max))


def compute_direct_opd_topk_dense_reward(
    student_old_topk_log_probs: torch.Tensor,
    teacher_rl_topk_log_probs: torch.Tensor,
    teacher_ref_topk_log_probs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the detached dense Direct-OPD reward on a student Top-K support.

    Only the student weights are renormalized on the truncated support.  Teacher
    log-probabilities remain full-vocabulary log-probabilities gathered at the
    exact same token ids.
    """
    shapes = {
        tuple(student_old_topk_log_probs.shape),
        tuple(teacher_rl_topk_log_probs.shape),
        tuple(teacher_ref_topk_log_probs.shape),
    }
    if len(shapes) != 1:
        raise ValueError(
            f"Direct-OPD Top-K tensors must have identical [B, T, K] shapes, got student_old={tuple(student_old_topk_log_probs.shape)}, teacher_rl={tuple(teacher_rl_topk_log_probs.shape)}, teacher_ref={tuple(teacher_ref_topk_log_probs.shape)}."
        )
    if student_old_topk_log_probs.ndim != 3 or student_old_topk_log_probs.size(-1) <= 0:
        raise ValueError(
            f"Direct-OPD Top-K tensors must be rank-3 with a non-empty support, got {tuple(student_old_topk_log_probs.shape)}."
        )
    student_topk_weight = torch.softmax(student_old_topk_log_probs.detach().float(), dim=-1)
    teacher_delta = teacher_rl_topk_log_probs.detach().float() - teacher_ref_topk_log_probs.detach().float()
    dense_reward = (student_topk_weight * teacher_delta).detach()
    return (dense_reward, student_topk_weight.detach(), teacher_delta.detach())


def direct_opd_topk_reward_stats(
    dense_reward: torch.Tensor, response_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return paper-faithful dense reward sum/count for adaptive alpha."""
    if dense_reward.ndim != 3:
        raise ValueError(f"dense_reward must have shape [B, T, K], got {tuple(dense_reward.shape)}.")
    if tuple(dense_reward.shape[:2]) != tuple(response_mask.shape):
        raise ValueError(
            f"Direct-OPD dense reward and response mask shapes do not align: reward={tuple(dense_reward.shape)}, mask={tuple(response_mask.shape)}."
        )
    mask = response_mask.to(device=dense_reward.device, dtype=dense_reward.dtype)
    reward_sum = (dense_reward.detach().float() * mask.unsqueeze(-1)).sum()
    reward_count = mask.float().sum() * dense_reward.size(-1)
    return (reward_sum, reward_count)


def compute_direct_opd_topk_loss(
    current_student_topk_log_probs: torch.Tensor,
    dense_reward: torch.Tensor,
    response_mask: torch.Tensor,
    distillation_mask: Optional[torch.Tensor] = None,
    global_batch_info: Optional[dict] = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute dense Top-K Direct-OPD without a PPO ratio or Top-K log-softmax."""
    if tuple(current_student_topk_log_probs.shape) != tuple(dense_reward.shape):
        raise ValueError(
            f"Direct-OPD current log-probs and dense reward must have identical shapes, got student={tuple(current_student_topk_log_probs.shape)}, reward={tuple(dense_reward.shape)}."
        )
    if current_student_topk_log_probs.ndim != 3:
        raise ValueError(
            f"Direct-OPD current Top-K log-probs must have shape [B, T, K], got {tuple(current_student_topk_log_probs.shape)}."
        )
    if tuple(current_student_topk_log_probs.shape[:2]) != tuple(response_mask.shape):
        raise ValueError(
            f"Direct-OPD current Top-K log-probs and response mask do not align: student={tuple(current_student_topk_log_probs.shape)}, mask={tuple(response_mask.shape)}."
        )
    active_mask = response_mask
    if distillation_mask is not None:
        if tuple(distillation_mask.shape) != tuple(response_mask.shape):
            raise ValueError(
                f"Direct-OPD distillation mask must match response mask, got distillation={tuple(distillation_mask.shape)}, response={tuple(response_mask.shape)}."
            )
        active_mask = active_mask * distillation_mask
    active_mask = active_mask.to(device=current_student_topk_log_probs.device, dtype=torch.float32)
    reward = dense_reward.detach().to(device=current_student_topk_log_probs.device, dtype=torch.float32)
    dense_pg_loss_per_position = -(reward * current_student_topk_log_probs.float()).sum(dim=-1)
    dense_pg_loss = agg_loss(
        dense_pg_loss_per_position, active_mask, "token-mean", **(global_batch_info or {})
    )
    active_count = active_mask.sum().detach()
    reward_mass = reward.sum(dim=-1)
    metrics = {
        "distillation/direct_opd_topk_dense_pg_loss": dense_pg_loss.detach().item(),
        "distillation/direct_opd_topk_reward_position_mean": verl_F.masked_mean(reward_mass, active_mask)
        .detach()
        .item(),
        "distillation/direct_opd_topk_reward_abs_mean": verl_F.masked_mean(
            reward.abs().sum(dim=-1), active_mask
        )
        .detach()
        .item(),
        "distillation/direct_opd_topk_active_positions": active_count.item(),
        "distillation/direct_opd_topk_k": float(current_student_topk_log_probs.size(-1)),
        "distillation/direct_opd_topk_ppo_ratio_used": 0.0,
    }
    return (dense_pg_loss, metrics)


def select_w2s_proxy_topk(proxy_score: torch.Tensor, top_k: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    """Compress a detached full-vocabulary W2S proxy to its exact Top-K support."""
    if proxy_score.ndim != 3:
        raise ValueError(f"W2S-OPD proxy scores must have shape [B, T, V], got {tuple(proxy_score.shape)}.")
    top_k = int(top_k)
    vocab_size = proxy_score.size(-1)
    if top_k <= 0 or top_k > vocab_size:
        raise ValueError(f"W2S-OPD top_k must be in [1, {vocab_size}], got {top_k}.")
    proxy_topk_score, proxy_topk_ids = torch.topk(proxy_score.detach().float(), k=top_k, dim=-1)
    proxy_topk_logp = torch.log_softmax(proxy_topk_score, dim=-1).detach()
    return (proxy_topk_ids.detach(), proxy_topk_logp)


def compute_w2s_proxy_topk(
    student_base_logp: torch.Tensor,
    positive_logp: torch.Tensor,
    negative_logp: torch.Tensor,
    amplification: float,
    top_k: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose the full-vocabulary W2S proxy before selecting its Top-K.

    All frozen-model inputs are detached.  Selection happens only after
    ``student_base + amplification * (positive - negative)`` has been formed;
    no constituent model's Top-K is used as a candidate set.
    """
    shapes = {tuple(student_base_logp.shape), tuple(positive_logp.shape), tuple(negative_logp.shape)}
    if len(shapes) != 1 or student_base_logp.ndim != 3:
        raise ValueError(
            f"W2S-OPD frozen full-vocabulary scores must have identical [B, T, V] shapes, got student_base={tuple(student_base_logp.shape)}, positive={tuple(positive_logp.shape)}, negative={tuple(negative_logp.shape)}."
        )
    amplification = float(amplification)
    if not np.isfinite(amplification) or amplification < 0.0:
        raise ValueError(f"W2S-OPD amplification must be finite and non-negative, got {amplification}.")
    proxy_score = student_base_logp.detach().float().clone()
    proxy_score.add_(positive_logp.detach().float(), alpha=amplification)
    proxy_score.add_(negative_logp.detach().float(), alpha=-amplification)
    return select_w2s_proxy_topk(proxy_score, top_k=top_k)


def compute_w2s_reverse_kl_loss(
    current_student_score_on_proxy_ids: torch.Tensor,
    proxy_topk_logp: torch.Tensor,
    response_mask: torch.Tensor,
    distillation_mask: Optional[torch.Tensor] = None,
    global_batch_info: Optional[dict] = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute restricted-support KL(current student || frozen W2S proxy)."""
    if tuple(current_student_score_on_proxy_ids.shape) != tuple(proxy_topk_logp.shape):
        raise ValueError(
            f"W2S-OPD student/proxy Top-K tensors must have identical shapes, got student={tuple(current_student_score_on_proxy_ids.shape)}, proxy={tuple(proxy_topk_logp.shape)}."
        )
    if current_student_score_on_proxy_ids.ndim != 3:
        raise ValueError(
            f"W2S-OPD Top-K tensors must have shape [B, T, K], got {tuple(current_student_score_on_proxy_ids.shape)}."
        )
    if tuple(current_student_score_on_proxy_ids.shape[:2]) != tuple(response_mask.shape):
        raise ValueError(
            f"W2S-OPD Top-K tensors and response mask do not align: student={tuple(current_student_score_on_proxy_ids.shape)}, mask={tuple(response_mask.shape)}."
        )
    active_mask = response_mask
    if distillation_mask is not None:
        if tuple(distillation_mask.shape) != tuple(response_mask.shape):
            raise ValueError(
                f"W2S-OPD distillation mask must match response mask, got distillation={tuple(distillation_mask.shape)}, response={tuple(response_mask.shape)}."
            )
        active_mask = active_mask * distillation_mask
    active_mask = active_mask.to(device=current_student_score_on_proxy_ids.device, dtype=torch.float32)
    student_topk_logp = torch.log_softmax(current_student_score_on_proxy_ids.float(), dim=-1)
    student_topk_prob = student_topk_logp.exp()
    frozen_proxy_logp = proxy_topk_logp.detach().to(device=student_topk_logp.device, dtype=torch.float32)
    kl_per_position = (student_topk_prob * (student_topk_logp - frozen_proxy_logp)).sum(dim=-1)
    w2s_loss = agg_loss(kl_per_position, active_mask, "token-mean", **(global_batch_info or {}))
    proxy_prob_sum_error = (frozen_proxy_logp.exp().sum(dim=-1) - 1.0).abs().max()
    student_prob_sum_error = (student_topk_prob.sum(dim=-1) - 1.0).abs().max()
    metrics = {
        "distillation/w2s_opd_reverse_kl_loss": w2s_loss.detach().item(),
        "distillation/w2s_opd_kl_per_position_mean": verl_F.masked_mean(
            kl_per_position.detach(), active_mask
        ).item(),
        "distillation/w2s_opd_active_positions": active_mask.sum().detach().item(),
        "distillation/w2s_opd_topk_k": float(current_student_score_on_proxy_ids.size(-1)),
        "distillation/w2s_opd_proxy_prob_sum_max_error": proxy_prob_sum_error.detach().item(),
        "distillation/w2s_opd_student_prob_sum_max_error": student_prob_sum_error.detach().item(),
        "distillation/w2s_opd_direction_student_to_proxy": 1.0,
        "distillation/w2s_opd_ppo_ratio_used": 0.0,
        "distillation/w2s_opd_ppo_clipping_used": 0.0,
        "distillation/w2s_opd_grpo_reward_used": 0.0,
        "distillation/w2s_opd_explicit_student_base_kl_used": 0.0,
    }
    return (w2s_loss, metrics)


def compute_distillation_loss(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    distillation_config: Any,
    old_log_probs: Optional[torch.Tensor] = None,
    student_all_log_probs: Optional[torch.Tensor] = None,
    teacher_all_log_probs: Optional[torch.Tensor] = None,
    student_topk_log_probs: Optional[torch.Tensor] = None,
    teacher_topk_log_probs: Optional[torch.Tensor] = None,
    distillation_topk_indices: Optional[torch.Tensor] = None,
    student_topk_indices: Optional[torch.Tensor] = None,
    teacher_topk_indices: Optional[torch.Tensor] = None,
    distillation_mask: Optional[torch.Tensor] = None,
    loss_agg_mode: str = "token-mean",
    rollout_is_weights: Optional[torch.Tensor] = None,
    auto_dp_sync: bool = False,
    eos_token_id: Optional[int] = None,
    eos_guard_threshold: Optional[float] = None,
    exclude_eos_from_distill: bool = False,
    global_batch_info: Optional[dict] = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if bool(getattr(distillation_config, "kdrl_enabled", False)):
        return compute_kdrl_k2_loss(
            student_log_probs=student_log_probs,
            teacher_log_probs=teacher_log_probs,
            response_mask=response_mask,
            distillation_mask=distillation_mask,
            loss_agg_mode=loss_agg_mode,
            auto_dp_sync=auto_dp_sync,
            global_batch_info=global_batch_info,
        )
    metrics = {}
    sampled_rkl_use_importance_sampling = bool(
        getattr(distillation_config, "sampled_rkl_use_importance_sampling", False)
    )
    loss_mask = response_mask
    if distillation_mask is not None:
        if distillation_mask.dim() == 1:
            loss_mask = loss_mask * distillation_mask.unsqueeze(1)
        elif distillation_mask.dim() == 2:
            loss_mask = loss_mask * distillation_mask
        else:
            raise ValueError(
                f"distillation_mask must be rank-1 or rank-2 tensor, got shape={tuple(distillation_mask.shape)}"
            )

    def _exclusion_eps(dtype: torch.dtype) -> float:
        return 0.0001 if dtype in {torch.float16, torch.bfloat16} else 1e-07

    def _remaining_log_norm_after_token_exclusion(
        log_probs: torch.Tensor, token_mask: torch.Tensor, eps: float
    ) -> torch.Tensor:
        neg_large = torch.finfo(log_probs.dtype).min
        token_logp = log_probs.masked_fill(~token_mask, neg_large).float().logsumexp(dim=-1, keepdim=True)
        visible_remaining_logp = (
            log_probs.masked_fill(token_mask, neg_large).float().logsumexp(dim=-1, keepdim=True)
        )
        token_remaining_prob = (1.0 - token_logp.exp()).clamp(min=eps)
        visible_remaining_prob = visible_remaining_logp.exp().clamp(min=eps)
        return torch.log(torch.maximum(token_remaining_prob, visible_remaining_prob))

    def _exclude_vocab_token_log_probs(log_probs: torch.Tensor, token_id: int) -> torch.Tensor:
        if token_id < 0 or token_id >= log_probs.size(-1):
            return log_probs
        eps = _exclusion_eps(log_probs.dtype)
        token_mask = torch.zeros_like(log_probs, dtype=torch.bool)
        token_mask[..., token_id] = True
        log_norm = _remaining_log_norm_after_token_exclusion(log_probs, token_mask, eps)
        adjusted = log_probs.float() - log_norm
        neg_large = torch.finfo(log_probs.dtype).min
        adjusted = adjusted.clone()
        adjusted[..., token_id] = neg_large
        return adjusted.to(log_probs.dtype)

    def _exclude_topk_token_log_probs(
        log_probs: torch.Tensor, topk_indices: Optional[torch.Tensor], token_id: int
    ) -> torch.Tensor:
        if topk_indices is None:
            return log_probs
        token_mask = topk_indices == token_id
        if not bool(token_mask.any().item()):
            return log_probs
        eps = _exclusion_eps(log_probs.dtype)
        log_norm = _remaining_log_norm_after_token_exclusion(log_probs, token_mask, eps)
        adjusted = log_probs.float() - log_norm
        neg_large = torch.finfo(log_probs.dtype).min
        return adjusted.masked_fill(token_mask, neg_large).to(log_probs.dtype)

    def _normalize_token_id_list(value) -> list[int]:
        if value is None:
            return []
        if isinstance(value, str):
            text = value.strip()
            if text.lower() in {"", "none", "null", "[]"}:
                return []
            if text.startswith("[") and text.endswith("]"):
                text = text[1:-1]
            items = [part.strip() for part in text.replace(";", ",").split(",")]
        else:
            try:
                items = list(value)
            except TypeError:
                items = [value]
        token_ids: list[int] = []
        for item in items:
            if item is None:
                continue
            if isinstance(item, str) and item.strip().lower() in {"", "none", "null"}:
                continue
            token_id = int(item)
            if token_id >= 0:
                token_ids.append(token_id)
        return sorted(set(token_ids))

    def _record_topk_overlap_metrics(
        support_indices: Optional[torch.Tensor],
        student_indices: Optional[torch.Tensor],
        teacher_indices: Optional[torch.Tensor],
        student_log_probs_on_support: torch.Tensor,
        teacher_log_probs_on_support: torch.Tensor,
    ) -> None:
        if support_indices is None or student_indices is None or teacher_indices is None:
            return
        if support_indices.dim() != 3 or student_indices.dim() != 3 or teacher_indices.dim() != 3:
            return
        if (
            support_indices.shape[:2] != loss_mask.shape
            or student_indices.shape[:2] != loss_mask.shape
            or teacher_indices.shape[:2] != loss_mask.shape
        ):
            return
        active_mask = loss_mask.float()
        overlap_count = (
            (student_indices.unsqueeze(-1) == teacher_indices.unsqueeze(-2)).any(dim=-1).float().sum(dim=-1)
        )
        overlap_denominator = float(max(1, min(student_indices.size(-1), teacher_indices.size(-1))))
        overlap_ratio = overlap_count / overlap_denominator
        metrics["distillation/topk_overlap_ratio"] = (
            verl_F.masked_mean(overlap_ratio, active_mask).detach().item()
        )
        metrics["distillation/topk_overlap_count"] = (
            verl_F.masked_mean(overlap_count, active_mask).detach().item()
        )
        support_width = min(
            support_indices.size(-1),
            student_log_probs_on_support.size(-1),
            teacher_log_probs_on_support.size(-1),
        )
        if support_width <= 0:
            return
        support_indices = support_indices[..., :support_width]
        student_support_log_probs = student_log_probs_on_support[..., :support_width].float()
        teacher_support_log_probs = teacher_log_probs_on_support[..., :support_width].float()
        support_in_student = (support_indices.unsqueeze(-1) == student_indices.unsqueeze(-2)).any(dim=-1)
        support_in_teacher = (support_indices.unsqueeze(-1) == teacher_indices.unsqueeze(-2)).any(dim=-1)
        overlap_support_mask = support_in_student & support_in_teacher
        student_min = torch.finfo(student_log_probs_on_support.dtype).min / 2
        teacher_min = torch.finfo(teacher_log_probs_on_support.dtype).min / 2
        finite_support_mask = (
            torch.isfinite(student_support_log_probs)
            & torch.isfinite(teacher_support_log_probs)
            & (student_support_log_probs > student_min)
            & (teacher_support_log_probs > teacher_min)
        )
        overlap_support_mask = overlap_support_mask & finite_support_mask
        overlap_token_mask = active_mask.unsqueeze(-1) * overlap_support_mask.to(dtype=active_mask.dtype)
        token_advantage = teacher_support_log_probs - student_support_log_probs
        metrics["distillation/topk_overlap_token_advantage"] = (
            verl_F.masked_mean(token_advantage, overlap_token_mask).detach().item()
        )
        metrics["distillation/topk_overlap_token_count"] = overlap_token_mask.sum().detach().item()

    eos_token_id_int = None
    if eos_token_id is not None:
        try:
            eos_token_id_int = int(eos_token_id)
        except (TypeError, ValueError):
            eos_token_id_int = None
    incompatible_token_ids = _normalize_token_id_list(
        getattr(distillation_config, "teacher_tokenizer_incompatible_token_ids", None)
    )
    if incompatible_token_ids:
        metrics["distillation/tokenizer_incompatible_vocab_ids_excluded"] = float(len(incompatible_token_ids))
    if distillation_config.full_logit_distillation:
        use_topk = distillation_config.distillation_topk is not None
        if use_topk:
            if student_topk_log_probs is None or teacher_topk_log_probs is None:
                raise ValueError(
                    "top-k distillation requires student_topk_log_probs and teacher_topk_log_probs."
                )

            def add_tail(log_probs: torch.Tensor) -> torch.Tensor:
                log_s = torch.logsumexp(log_probs.float(), dim=-1, keepdim=True)
                log_s = torch.clamp(log_s, max=-1e-07)
                tail_log = torch.log(-torch.expm1(log_s))
                return torch.cat([log_probs, tail_log.to(log_probs.dtype)], dim=-1)

            def renorm_topk_log_probs(logp: torch.Tensor) -> torch.Tensor:
                logZ = torch.logsumexp(logp, dim=-1, keepdim=True)
                return logp - logZ

            student_distill_log_probs = student_topk_log_probs
            teacher_distill_log_probs = teacher_topk_log_probs
            if (
                eos_token_id_int is not None
                and eos_guard_threshold is not None
                and (distillation_topk_indices is not None)
            ):
                eos_match = distillation_topk_indices == eos_token_id_int
                if bool(eos_match.any().item()):
                    neg_large = torch.finfo(teacher_topk_log_probs.dtype).min
                    eos_logps = teacher_topk_log_probs.masked_fill(~eos_match, neg_large).logsumexp(dim=-1)
                    eos_probs = eos_logps.exp()
                else:
                    eos_probs = torch.zeros_like(loss_mask)
                eos_drop_mask = (loss_mask > 0) & (eos_probs > float(eos_guard_threshold))
                dropped = eos_drop_mask.float().sum().detach()
                active = (loss_mask > 0).float().sum().detach().clamp(min=1.0)
                loss_mask = loss_mask.masked_fill(eos_drop_mask, 0.0)
                any_eos_drop = bool(eos_drop_mask.any().item())
                metrics.update(
                    {
                        "distillation/eos_guard_available": 1.0,
                        "distillation/eos_guard_threshold": float(eos_guard_threshold),
                        "distillation/eos_guard_dropped_token_count": dropped.item(),
                        "distillation/eos_guard_dropped_token_fraction": (dropped / active).item(),
                        "distillation/eos_guard_dropped_row_fraction": eos_drop_mask.any(dim=-1)
                        .float()
                        .mean()
                        .item(),
                        "distillation/eos_guard_eos_prob_mean": eos_probs.masked_select(eos_drop_mask)
                        .mean()
                        .detach()
                        .item()
                        if any_eos_drop
                        else 0.0,
                        "distillation/eos_guard_eos_prob_max": eos_probs.masked_select(eos_drop_mask)
                        .max()
                        .detach()
                        .item()
                        if any_eos_drop
                        else 0.0,
                    }
                )
            elif eos_guard_threshold is not None:
                metrics.update(
                    {
                        "distillation/eos_guard_available": 0.0,
                        "distillation/eos_guard_threshold": float(eos_guard_threshold),
                    }
                )
            if exclude_eos_from_distill and eos_token_id_int is not None:
                student_distill_log_probs = _exclude_topk_token_log_probs(
                    student_distill_log_probs, distillation_topk_indices, eos_token_id_int
                )
                teacher_distill_log_probs = _exclude_topk_token_log_probs(
                    teacher_distill_log_probs, distillation_topk_indices, eos_token_id_int
                )
                metrics["distillation/eos_excluded_from_distill"] = 1.0
            for token_id in incompatible_token_ids:
                student_distill_log_probs = _exclude_topk_token_log_probs(
                    student_distill_log_probs, distillation_topk_indices, token_id
                )
                teacher_distill_log_probs = _exclude_topk_token_log_probs(
                    teacher_distill_log_probs, distillation_topk_indices, token_id
                )
            if distillation_config.distillation_add_tail and (not incompatible_token_ids):
                student_distill_log_probs = add_tail(student_distill_log_probs)
                teacher_distill_log_probs = add_tail(teacher_distill_log_probs)
            else:
                student_distill_log_probs = renorm_topk_log_probs(student_distill_log_probs)
                teacher_distill_log_probs = renorm_topk_log_probs(teacher_distill_log_probs)
                if incompatible_token_ids and distillation_config.distillation_add_tail:
                    metrics["distillation/tokenizer_incompatible_topk_tail_disabled"] = 1.0
            _record_topk_overlap_metrics(
                support_indices=distillation_topk_indices,
                student_indices=student_topk_indices,
                teacher_indices=teacher_topk_indices,
                student_log_probs_on_support=student_distill_log_probs,
                teacher_log_probs_on_support=teacher_distill_log_probs,
            )
        else:
            if student_all_log_probs is None or teacher_all_log_probs is None:
                raise ValueError(
                    "full_logit_distillation requires student_all_log_probs and teacher_all_log_probs."
                )
            student_distill_log_probs = student_all_log_probs
            teacher_distill_log_probs = teacher_all_log_probs
            if (
                eos_token_id_int is not None
                and eos_guard_threshold is not None
                and (0 <= eos_token_id_int < teacher_all_log_probs.size(-1))
            ):
                eos_probs = teacher_all_log_probs[..., eos_token_id_int].exp()
                eos_drop_mask = (loss_mask > 0) & (eos_probs > float(eos_guard_threshold))
                dropped = eos_drop_mask.float().sum().detach()
                active = (loss_mask > 0).float().sum().detach().clamp(min=1.0)
                loss_mask = loss_mask.masked_fill(eos_drop_mask, 0.0)
                any_eos_drop = bool(eos_drop_mask.any().item())
                metrics.update(
                    {
                        "distillation/eos_guard_available": 1.0,
                        "distillation/eos_guard_threshold": float(eos_guard_threshold),
                        "distillation/eos_guard_dropped_token_count": dropped.item(),
                        "distillation/eos_guard_dropped_token_fraction": (dropped / active).item(),
                        "distillation/eos_guard_dropped_row_fraction": eos_drop_mask.any(dim=-1)
                        .float()
                        .mean()
                        .item(),
                        "distillation/eos_guard_eos_prob_mean": eos_probs.masked_select(eos_drop_mask)
                        .mean()
                        .detach()
                        .item()
                        if any_eos_drop
                        else 0.0,
                        "distillation/eos_guard_eos_prob_max": eos_probs.masked_select(eos_drop_mask)
                        .max()
                        .detach()
                        .item()
                        if any_eos_drop
                        else 0.0,
                    }
                )
            elif eos_guard_threshold is not None:
                metrics.update(
                    {
                        "distillation/eos_guard_available": 0.0,
                        "distillation/eos_guard_threshold": float(eos_guard_threshold),
                    }
                )
            if exclude_eos_from_distill and eos_token_id_int is not None:
                student_distill_log_probs = _exclude_vocab_token_log_probs(
                    student_distill_log_probs, eos_token_id_int
                )
                teacher_distill_log_probs = _exclude_vocab_token_log_probs(
                    teacher_distill_log_probs, eos_token_id_int
                )
                metrics["distillation/eos_excluded_from_distill"] = 1.0
            for token_id in incompatible_token_ids:
                student_distill_log_probs = _exclude_vocab_token_log_probs(
                    student_distill_log_probs, token_id
                )
                teacher_distill_log_probs = _exclude_vocab_token_log_probs(
                    teacher_distill_log_probs, token_id
                )
        if distillation_config.alpha == 0.0:
            kl_loss = F.kl_div(
                student_distill_log_probs, teacher_distill_log_probs, reduction="none", log_target=True
            )
        elif distillation_config.alpha == 1.0:
            kl_loss = F.kl_div(
                teacher_distill_log_probs, student_distill_log_probs, reduction="none", log_target=True
            )
        else:
            alpha = torch.tensor(
                distillation_config.alpha,
                dtype=student_distill_log_probs.dtype,
                device=student_distill_log_probs.device,
            )
            mixture_log_probs = torch.logsumexp(
                torch.stack(
                    [
                        student_distill_log_probs + torch.log(1 - alpha),
                        teacher_distill_log_probs + torch.log(alpha),
                    ]
                ),
                dim=0,
            )
            kl_teacher = F.kl_div(
                mixture_log_probs, teacher_distill_log_probs, reduction="none", log_target=True
            )
            kl_student = F.kl_div(
                mixture_log_probs, student_distill_log_probs, reduction="none", log_target=True
            )
            kl_loss = torch.lerp(kl_student, kl_teacher, alpha)
        per_token_loss = kl_loss.sum(-1)
    else:
        assert distillation_config.alpha == 1.0, (
            "Only reverse KL is supported for non-full-logit distillation"
        )
        if sampled_rkl_use_importance_sampling:
            if old_log_probs is None:
                raise ValueError(
                    "sampled_rkl_use_importance_sampling requires old_log_probs from the rollout student policy."
                )
            advantages = teacher_log_probs - old_log_probs
            gap_gate_beta = getattr(distillation_config, "sampled_rkl_gap_gate_beta", None)
            if isinstance(gap_gate_beta, str) and gap_gate_beta.lower() in {"", "none", "null"}:
                gap_gate_beta = None
            gap_gate_enabled = gap_gate_beta is not None and float(gap_gate_beta) > 0.0
            if gap_gate_enabled:
                gap_gate_beta = float(gap_gate_beta)
                gate_delta = teacher_log_probs.detach() - student_log_probs.detach()
                gap_gate = torch.sigmoid(gap_gate_beta * gate_delta)
                advantages = advantages * gap_gate
                metrics["distillation/sampled_rkl_gap_gate_enabled"] = 1.0
                metrics["distillation/sampled_rkl_gap_gate_beta"] = gap_gate_beta
                metrics["distillation/sampled_rkl_gap_gate_mean"] = (
                    verl_F.masked_mean(gap_gate, loss_mask).detach().item()
                )
                metrics["distillation/sampled_rkl_gap_delta_mean"] = (
                    verl_F.masked_mean(gate_delta, loss_mask).detach().item()
                )
                metrics["distillation/sampled_rkl_gap_delta_positive_fraction"] = (
                    verl_F.masked_mean((gate_delta > 0).to(dtype=gate_delta.dtype), loss_mask).detach().item()
                )
            else:
                metrics["distillation/sampled_rkl_gap_gate_enabled"] = 0.0
            advantage_ratio_clip_low = getattr(
                distillation_config, "sampled_rkl_advantage_ratio_clip_low", None
            )
            advantage_ratio_clip_high = getattr(
                distillation_config, "sampled_rkl_advantage_ratio_clip_high", None
            )
            if isinstance(advantage_ratio_clip_low, str) and advantage_ratio_clip_low.lower() in {
                "",
                "none",
                "null",
            }:
                advantage_ratio_clip_low = None
            if isinstance(advantage_ratio_clip_high, str) and advantage_ratio_clip_high.lower() in {
                "",
                "none",
                "null",
            }:
                advantage_ratio_clip_high = None
            if advantage_ratio_clip_low is not None or advantage_ratio_clip_high is not None:
                clip_low = float(advantage_ratio_clip_low) if advantage_ratio_clip_low is not None else None
                clip_high = (
                    float(advantage_ratio_clip_high) if advantage_ratio_clip_high is not None else None
                )
                if clip_low is not None and clip_low <= 0.0:
                    raise ValueError(
                        f"distillation.sampled_rkl_advantage_ratio_clip_low must be positive when provided, got {clip_low}."
                    )
                if clip_high is not None and clip_high <= 0.0:
                    raise ValueError(
                        f"distillation.sampled_rkl_advantage_ratio_clip_high must be positive when provided, got {clip_high}."
                    )
                if clip_low is not None and clip_high is not None and (clip_low > clip_high):
                    raise ValueError(
                        f"sampled-RKL advantage ratio clipping requires low <= high, got low={clip_low}, high={clip_high}."
                    )
                unclipped_advantages = advantages
                min_log = float(np.log(clip_low)) if clip_low is not None else None
                max_log = float(np.log(clip_high)) if clip_high is not None else None
                if min_log is not None and max_log is not None:
                    advantages = torch.clamp(advantages, min=min_log, max=max_log)
                elif min_log is not None:
                    advantages = torch.clamp(advantages, min=min_log)
                elif max_log is not None:
                    advantages = torch.clamp(advantages, max=max_log)
                clipped_tokens = torch.ne(advantages, unclipped_advantages).float()
                metrics["distillation/sampled_rkl_advantage_ratio_clip_enabled"] = 1.0
                metrics["distillation/sampled_rkl_advantage_ratio_clip_low"] = (
                    float(clip_low) if clip_low is not None else 0.0
                )
                metrics["distillation/sampled_rkl_advantage_ratio_clip_high"] = (
                    float(clip_high) if clip_high is not None else 0.0
                )
                metrics["distillation/sampled_rkl_advantage_ratio_clipfrac"] = (
                    verl_F.masked_mean(clipped_tokens, loss_mask).detach().item()
                )
            else:
                metrics["distillation/sampled_rkl_advantage_ratio_clip_enabled"] = 0.0
            advantages = advantages.detach()
            is_clip = distillation_config.is_clip
            if is_clip is not None:
                ratio = _compute_distillation_is_ratio(
                    student_log_probs=student_log_probs,
                    old_log_probs=old_log_probs,
                    is_clip=is_clip,
                    is_clip_low=getattr(distillation_config, "is_clip_low", None),
                    is_clip_high=getattr(distillation_config, "is_clip_high", None),
                    loss_mask=loss_mask,
                    metrics=metrics,
                    metric_prefix="distillation/sampled_rkl",
                    detach_ratio=False,
                )
            else:
                ratio = torch.exp(torch.clamp(student_log_probs - old_log_probs, min=-20.0, max=20.0))
            per_token_loss = -advantages * ratio
            metrics["distillation/sampled_rkl_importance_sampling"] = 1.0
            metrics["distillation/sampled_rkl_advantage_mean"] = (
                verl_F.masked_mean(advantages, loss_mask).detach().item()
            )
            if is_clip is None:
                metrics["distillation/sampled_rkl_ratio_mean"] = (
                    verl_F.masked_mean(ratio.detach(), loss_mask).detach().item()
                )
        else:
            log_ratio = student_log_probs - teacher_log_probs
            per_token_loss = log_ratio.detach() * student_log_probs
    is_clip = distillation_config.is_clip
    if is_clip is not None and (not sampled_rkl_use_importance_sampling):
        ratio = _compute_distillation_is_ratio(
            student_log_probs=student_log_probs,
            old_log_probs=old_log_probs,
            is_clip=is_clip,
            is_clip_low=getattr(distillation_config, "is_clip_low", None),
            is_clip_high=getattr(distillation_config, "is_clip_high", None),
            loss_mask=loss_mask,
            metrics=metrics,
            metric_prefix="distillation/is",
            include_ratio_max=True,
        )
        per_token_loss = per_token_loss * ratio
    if rollout_is_weights is not None:
        per_token_loss = per_token_loss * rollout_is_weights
    auto_dp_sync_active = (
        auto_dp_sync
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and (torch.distributed.get_world_size() > 1)
    )
    agg_loss_kwargs = {"auto_dp_sync": auto_dp_sync and not global_batch_info}
    if loss_agg_mode == "token-mean" and not global_batch_info and (not auto_dp_sync_active):
        agg_loss_kwargs["batch_num_tokens"] = loss_mask.sum().clamp(min=1.0)
    loss = agg_loss(
        loss_mat=per_token_loss,
        loss_mask=loss_mask,
        loss_agg_mode=loss_agg_mode,
        **(global_batch_info or agg_loss_kwargs),
    )
    return (loss, metrics)


@deprecated("verl.trainer.ppo.core_algos.compute_policy_loss_vanilla")
def compute_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        clip_ratio_c (float, optional):
            Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
            Defaults to 3.0.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
    """
    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )
    negative_approx_kl = log_prob - old_log_prob
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )
    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    return (pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower)


@register_policy_loss("guided_opsd")
@register_policy_loss("vanilla")
def compute_policy_loss_vanilla(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
    collect_metrics: bool = True,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        config: `(verl.trainer.config.ActorConfig)`:
            config for the actor.
        rollout_is_weights: `(torch.Tensor)`:
            Optional per-token rollout importance-sampling weights, shape
            (batch_size, response_length).
        collect_metrics: Whether to materialize detached PPO diagnostics. Disabling
            this does not alter the policy loss or its gradient.
    """
    assert config is not None
    assert not isinstance(config, AlgoConfig)
    clip_ratio = config.clip_ratio
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
    clip_ratio_c = config.get("clip_ratio_c", 3.0)
    cliprange = clip_ratio
    cliprange_low = clip_ratio_low
    cliprange_high = clip_ratio_high
    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )
    negative_approx_kl = log_prob - old_log_prob
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask) if collect_metrics else None
    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    pg_clipfrac = (
        verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
        if collect_metrics
        else None
    )
    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = (
        verl_F.masked_mean(torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask)
        if collect_metrics
        else None
    )
    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights
    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )
    pg_metrics = (
        {
            "actor/pg_clipfrac": pg_clipfrac.detach().item(),
            "actor/ppo_kl": ppo_kl.detach().item(),
            "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
        }
        if collect_metrics
        else {}
    )
    return (pg_loss, pg_metrics)


def compute_entropy_loss(logits, response_mask, loss_agg_mode: str = "token-mean"):
    """Compute categorical entropy loss (For backward compatibility)

    Args:
        logits (torch.Tensor): shape is (bs, response_length, vocab_size)
        response_mask (torch.Tensor): shape is (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    token_entropy = verl_F.entropy_from_logits(logits)
    entropy_loss = agg_loss(loss_mat=token_entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    return entropy_loss


def compute_value_loss(
    vpreds: torch.Tensor,
    returns: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    cliprange_value: float,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute the clipped value-function loss for PPO.

    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (torch.FloatTensor):
            Predicted values from the value head, shape (batch_size, response_length).
        values (torch.FloatTensor):
            Old (baseline) values from the value head, shape (batch_size, response_length).
        returns (torch.FloatTensor):
            Ground-truth returns, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the value loss calculation.
        cliprange_value (float):
            Clip range for value prediction updates.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".

    Returns:
        vf_loss (torch.FloatTensor):
            A scalar tensor containing the aggregated value-function loss.
        vf_clipfrac (float):
            Fraction of elements where the clipped loss was used.
    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    clipped_vf_losses = torch.max(vf_losses1, vf_losses2)
    vf_loss = 0.5 * agg_loss(loss_mat=clipped_vf_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), response_mask)
    return (vf_loss, vf_clipfrac)


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob. Optionally using straight through to bind k2 on other
    kl penalty compute method for unbiased KL gradient estimation.
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:
        kl_estimate
    """
    forward_score = kl_penalty_forward(logprob, ref_logprob, kl_penalty)
    if not kl_penalty.endswith("+") or kl_penalty in ("mse", "k2"):
        return forward_score
    "\n    The expectation of k1 and k3 estimator is the expectaed value of KL, but the expected gradient of k1 and k3\n    estimator is not the expectaed gradient of KL. On the other hand k2 estimator gives right gradient estimator,\n    so we use a straight through trick here if the kl_penalty method ends with '+', .e.g., k3+.\n    "
    backward_score = 0.5 * (logprob - ref_logprob).square()
    return backward_score - backward_score.detach() + forward_score.detach()


def kl_penalty_forward(
    logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty
) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:
        kl_estimate
    """
    if kl_penalty in ("kl", "k1"):
        return logprob - ref_logprob
    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()
    if kl_penalty in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()
    if kl_penalty in ("low_var_kl", "k3"):
        kl = ref_logprob - logprob
        kl = torch.clamp(kl, min=-20, max=20)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)
    if kl_penalty == "full":
        raise NotImplementedError
    raise NotImplementedError


def compute_pf_ppo_reweight_data(data, reweight_method: str = "pow", weight_pow: float = 2.0):
    """Reweight the data based on the token_level_scores.

    Args:
        data: DataProto object, containing batch, non_tensor_batch and meta_info
        reweight_method: str, choices: "pow", "max_min", "max_random"
        weight_pow: float, the power of the weight

    Returns:

    """

    @torch.no_grad()
    def compute_weights(scores: torch.Tensor, reweight_method: str, weight_pow: float) -> torch.Tensor:
        """Compute importance weights for resampling based on scores.

        Args:
            scores (torch.Tensor): Tensor of scores to compute weights from.
            reweight_method (str): Method for computing weights ('pow', 'max_min', 'max_random').
            weight_pow (float): Power exponent for 'pow' method.

        Returns:
            torch.Tensor: Computed importance weights.

        Raises:
            ValueError: If reweight_method is not supported.
        """
        if reweight_method == "pow":
            weights = torch.pow(torch.abs(scores), weight_pow)
        elif reweight_method == "max_min":
            max_score = torch.max(scores)
            min_score = torch.min(scores)
            weights = torch.where((scores == max_score) | (scores == min_score), 1.0, 0.0)
        elif reweight_method == "max_random":
            max_score = torch.max(scores)
            weights = torch.where(scores == max_score, 0.4, 0.1)
        else:
            raise ValueError(f"Unsupported reweight_method: {reweight_method}")
        return weights

    scores = data.batch["token_level_scores"].sum(dim=-1)
    weights = compute_weights(scores, reweight_method, weight_pow)
    weights = torch.clamp(weights + 1e-08, min=1e-08)
    batch_size = scores.shape[0]
    sample_indices = torch.multinomial(weights, batch_size, replacement=True)
    resampled_batch = {key: tensor[sample_indices] for key, tensor in data.batch.items()}
    sample_indices_np = sample_indices.numpy()
    resampled_non_tensor_batch = {}
    for key, array in data.non_tensor_batch.items():
        if isinstance(array, np.ndarray):
            resampled_non_tensor_batch[key] = array[sample_indices_np]
        else:
            resampled_non_tensor_batch[key] = [array[i] for i in sample_indices_np]
    resampled_meta_info = {}
    for key, value in data.meta_info.items():
        if isinstance(value, list) and len(value) == batch_size:
            resampled_meta_info[key] = [value[i] for i in sample_indices_np]
        else:
            resampled_meta_info[key] = value
    resampled_batch_td = type(data.batch)(resampled_batch)
    resampled_batch_td.batch_size = data.batch.batch_size
    return type(data)(
        batch=resampled_batch_td, non_tensor_batch=resampled_non_tensor_batch, meta_info=resampled_meta_info
    )
