"""Agent losses share the root actor's DP-wide optimizer token normalization."""

from verl.trainer.ppo.core_algos import register_policy_loss, compute_policy_loss_vanilla
from .opd import sampled_rkl_loss


@register_policy_loss("agent_global_token_grpo")
@register_policy_loss("agent_global_token_oprd")
def global_token_oprd_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    loss_agg_mode="token-mean",
    config=None,
    rollout_is_weights=None,
):
    if loss_agg_mode != "token-mean" or rollout_is_weights is not None:
        raise ValueError("Agent requires token-mean without rollout IS")
    return compute_policy_loss_vanilla(
        old_log_prob, log_prob, advantages, response_mask, loss_agg_mode, config, rollout_is_weights
    )


@register_policy_loss("agent_global_token_opd")
def global_token_opd_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    loss_agg_mode="token-mean",
    config=None,
    rollout_is_weights=None,
):
    if loss_agg_mode != "token-mean" or rollout_is_weights is not None:
        raise ValueError("Agent requires token-mean without rollout IS")
    return sampled_rkl_loss(
        log_prob, old_log_prob, advantages, response_mask, global_batch_info=config.global_batch_info
    )
