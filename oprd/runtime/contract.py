"""Reject non-paper runtime choices before allocating models or starting Ray."""

from dataclasses import fields
from verl.workers.config.actor import SelfDistillationConfig

PAPER_LOSSES = {
    "vanilla",
    "kd",
    "opsd",
    "guided_opsd",
    "agent_global_token_grpo",
    "agent_global_token_opd",
    "agent_global_token_kdrl",
    "agent_global_token_oprd",
}


def validate(config):
    if config.trainer.nnodes != 1 or config.trainer.n_gpus_per_node != 4:
        raise ValueError("Paper launchers require one node with four GPUs (DP=4)")
    if config.actor_rollout_ref.rollout.tensor_model_parallel_size != 1:
        raise ValueError("Paper rollouts require tensor parallelism 1 (TP=1)")
    if config.data.prompt_format != "chat" or config.data.prepend_dropped_system_prompt_to_user:
        raise ValueError(
            "The final paper inputs use chat messages without reinserting dropped system prompts"
        )
    if config.data.apply_chat_template_kwargs.get("enable_thinking", False):
        raise ValueError("The final paper prompts do not enable the optional thinking template")
    actor = config.actor_rollout_ref.actor
    if actor.policy_loss.loss_mode not in PAPER_LOSSES:
        raise ValueError("This release only implements the paper's policy objectives")
    if config.algorithm.adv_estimator != "grpo":
        raise ValueError("Paper experiments use GRPO advantages")
    if actor.strategy != "fsdp" or actor.ulysses_sequence_parallel_size != 1:
        raise ValueError("Paper runtime uses FSDP with sequence parallelism 1")
    if actor.loss_agg_mode != "token-mean" or actor.ppo_epochs != 1 or actor.entropy_coeff != 0:
        raise ValueError("Paper comparisons use one epoch and a global token mean without entropy bonus")
    if config.algorithm.rollout_correction.get("bypass_mode", False):
        raise ValueError("Rollout bypass objectives are outside the paper")
    cfg = actor.distillation
    allowed = {f.name for f in fields(SelfDistillationConfig)} | {"_target_"}
    unknown = set(cfg) - allowed
    if unknown:
        raise ValueError(f"Unknown paper distillation fields: {sorted(unknown)}")
    if cfg.teacher_regularization not in {"fixed", "teacher", "ema"}:
        raise ValueError("Paper teachers are fixed, or full-parameter EMA for OPSD")
    if cfg.direct_opd_mode != "topk" or cfg.w2s_opd_divergence != "reverse_kl":
        raise ValueError("Only the paper's Direct-OPD and W2S-OPD definitions are retained")
