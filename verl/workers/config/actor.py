# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import math
from dataclasses import dataclass, field
from typing import Any, Optional
from omegaconf import MISSING
from verl.base_config import BaseConfig
from verl.trainer.config import CheckpointConfig
from verl.utils.profiler.config import ProfilerConfig
from .engine import FSDPEngineConfig, McoreEngineConfig
from .model import HFModelConfig
from .optimizer import OptimizerConfig

__all__ = [
    "SelfDistillationConfig",
    "PolicyLossConfig",
    "RouterReplayConfig",
    "ActorConfig",
    "FSDPActorConfig",
    "McoreActorConfig",
]


@dataclass
class SelfDistillationConfig(BaseConfig):
    """Frozen teacher, OPRD, and paper-baseline configuration."""

    full_logit_distillation: bool = True
    alpha: float = 0.5
    kdrl_enabled: bool = False
    kdrl_beta_init: float = 0.005
    kdrl_delta: float = 0.0001
    kdrl_beta_min: float = 0.0
    delta_target_enabled: bool = False
    delta_target_scale: float = 1.0
    delta_teacher_base_model_path: Optional[str] = None
    direct_opd_mode: str = "topk"
    direct_opd_top_k: int = 16
    direct_opd_top_k_source: str = "student"
    delta_anchor_adaptive_alpha_enabled: bool = False
    delta_anchor_alpha_init: float = 1.0
    direct_kl_coef: Optional[float] = None
    delta_anchor_alpha_epsilon: float = 0.01
    delta_anchor_alpha_min: float = 0.5
    delta_anchor_alpha_max: float = 2.5
    w2s_opd_enabled: bool = False
    w2s_opd_teacher_top_k: int = 32
    w2s_opd_top_k_source: str = "proxy_teacher"
    w2s_opd_divergence: str = "reverse_kl"
    w2s_amplification: float = 1.0
    w2s_positive_model_path: Optional[str] = None
    w2s_positive_model_revision: Optional[str] = None
    w2s_negative_model_path: Optional[str] = None
    w2s_negative_model_revision: Optional[str] = None
    w2s_student_base_model_path: Optional[str] = None
    w2s_student_base_model_revision: Optional[str] = None
    success_reward_threshold: float = 1.0
    teacher_regularization: str = "fixed"
    teacher_update_rate: float = 0.01
    teacher_model_path: Optional[str] = None
    multi_teacher_enabled: bool = False
    multi_teacher_domains: list[str] = field(default_factory=list)
    multi_teacher_model_paths: list[str] = field(default_factory=list)
    multi_teacher_strict_domain_partition: bool = True
    keep_external_teacher_on_gpu: bool = False
    teacher_tokenizer_mismatch_policy: str = "auto_mask_special"
    teacher_tokenizer_incompatible_token_ids: list[int] = field(default_factory=list)
    teacher_tokenizer_base_vocab_match: int = 1
    teacher_num_rollouts: int = 8
    distillation_topk: Optional[int] = None
    distillation_topk_source: str = "student"
    oprd_enabled: bool = False
    oprd_lambda: float = 0.5
    oprd_signed_lambda_schedule_enabled: bool = True
    oprd_lambda_plus_start: float = 1.0
    oprd_lambda_plus_end: float = 1.0
    oprd_lambda_minus_start: float = 0.0
    oprd_lambda_minus_end: float = 1.0
    oprd_signed_lambda_schedule_steps: int = 50
    oprd_topk: int = 64
    oprd_base_model_path: Optional[str] = None
    oprd_h1_source: str = "teacher_base"
    oprd_h_normalization: str = "uniform_center_l2"
    oprd_opsd_hint_source: str = "student"
    oprd_opsd_weak_teacher_num_rollouts: int = 8
    oprd_skip_invalid_format: bool = True
    oprd_invalid_format_key: str = "incorrect_format"
    oprd_invalid_format_threshold: float = 0.5
    distillation_add_tail: bool = True
    max_reprompt_len: int = 10240
    reprompt_truncation: str = "right"
    dont_reprompt_on_self_success: bool = False
    solution_selection: str = "first"
    successful_student_hint_source: str = "peer_student"
    remove_thinking_from_demonstration: bool = False
    strip_thinking_tags_from_demonstration: bool = True
    is_clip: Optional[float] = None
    is_clip_low: Optional[float] = None
    is_clip_high: Optional[float] = None
    is_old_log_prob_source: Optional[str] = None
    sampled_rkl_use_importance_sampling: bool = False
    sampled_rkl_advantage_ratio_clip_low: Optional[float] = None
    sampled_rkl_advantage_ratio_clip_high: Optional[float] = None
    reprompt_template: str = "{prompt}{solution}{feedback}\n\nCorrectly solve the original question.\n"
    solution_template: str = "\nCorrect solution:\n\n{successful_previous_attempt}\n\n"

    def __post_init__(self):
        if self.oprd_topk < 1:
            raise ValueError("OPRD top-k must be positive")
        if self.oprd_h1_source not in {"teacher_base", "teacher_student", "opsd_kl"}:
            raise ValueError("Unsupported guidance direction")


@dataclass
class RouterReplayConfig(BaseConfig):
    """Configuration for router replay in MoE models.

    This configuration controls the routing behavior for Mixture of Experts (MoE) models,
    allowing for deterministic training through route recording and replay.

    Args:
        mode (str): Router replay mode. Options: 'disabled', 'R2', 'R3'.
            - 'disabled': No router replay functionality
            - 'R2': Use Router Replay routing strategy
            - 'R3': Use Rollout Router Replay routing strategy
        record_file (Optional[str]): File path to save recorded routing decisions.
            Required when mode is 'record', 'R2', or 'R3'.
        replay_file (Optional[str]): File path to load recorded routing decisions for replay.
            Required when mode is 'replay'.
    """

    mode: str = "disabled"
    record_file: Optional[str] = None
    replay_file: Optional[str] = None

    def __post_init__(self):
        """Validate router replay configuration."""
        valid_modes = ["disabled", "R2", "R3"]
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid router_replay mode: {self.mode}. Must be one of {valid_modes}")


@dataclass
class PolicyLossConfig(BaseConfig):
    """Policy loss selection and upstream PPO clipping settings.

    Paper entrypoints select vanilla (GRPO/OPRD/KDRL), kd (OPD),
    opsd, or guided_opsd (the OPSD-direction ablation).
    """

    loss_mode: str = "vanilla"


@dataclass
class ActorConfig(BaseConfig):
    """Configuration for actor model training.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        strategy (str): Training strategy. Must be specified.
        ppo_mini_batch_size (int): Mini-batch size for PPO training.
        ppo_micro_batch_size (Optional[int]): Micro-batch size for PPO training.
            If None, uses ppo_micro_batch_size_per_gpu.
        ppo_micro_batch_size_per_gpu (Optional[int]): Micro-batch size per GPU for PPO training.
        use_dynamic_bsz (bool): Whether to use dynamic batch sizing.
        ppo_max_token_len_per_gpu (int): Maximum token length per GPU for PPO training.
        clip_ratio (float): PPO clipping ratio for policy loss.
        clip_ratio_low (float): Lower bound for PPO clipping ratio.
        clip_ratio_high (float): Upper bound for PPO clipping ratio.
        policy_loss (PolicyLossConfig): Configuration for policy loss computation.
        clip_ratio_c (float): Clipping ratio for critic loss.
        loss_agg_mode (str): Loss aggregation mode. Options: 'token-mean', 'sample-mean'.
        loss_scale_factor (Optional[int]): Scale factor for 'seq-mean-token-sum-norm' loss aggregation mode.
            If None, uses response_length. Set to a constant to ensure consistent normalization.
        entropy_coeff (float): Entropy coefficient for regularization.
        tau_pos (float): Positive tau for SAPO smoothing (>= 1.0 keeps rewards stable).
        tau_neg (float): Negative tau for SAPO smoothing (> tau_pos for asymmetry).
        use_kl_loss (bool): Whether to use KL divergence loss.
        use_torch_compile (bool): Whether to use torch.compile for optimization.
        kl_loss_coef (float): KL divergence loss coefficient.
        kl_loss_type (str): Type of KL loss to use.
        ppo_epochs (int): Number of PPO epochs per training step.
        shuffle (bool): Whether to shuffle data during training.
        checkpoint (CheckpointConfig): Configuration for checkpointing.
        optim (OptimizerConfig): Configuration for optimizer.
        use_fused_kernels (bool): Whether to use custom fused kernels (e.g., FlashAttention, fused MLP).
        data_loader_seed (int): Seed for data loader. If None, uses global seed.
        router_replay (RouterReplayConfig): Configuration for router replay in MoE models.
    """

    _mutable_fields = BaseConfig._mutable_fields | {
        "ppo_mini_batch_size",
        "ppo_micro_batch_size",
        "ppo_micro_batch_size_per_gpu",
        "ppo_infer_micro_batch_size_per_gpu",
        "engine",
        "model_config",
    }
    strategy: str = MISSING
    ppo_mini_batch_size: int = 256
    ppo_micro_batch_size: Optional[int] = None
    ppo_micro_batch_size_per_gpu: Optional[int] = None
    ppo_infer_micro_batch_size_per_gpu: Optional[int] = None
    use_dynamic_bsz: bool = False
    ppo_max_token_len_per_gpu: int = 16384
    ppo_infer_max_token_len_per_gpu: int = 16384
    clip_ratio: float = 0.2
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.2
    freeze_vision_tower: bool = False
    policy_loss: PolicyLossConfig = field(default_factory=PolicyLossConfig)
    clip_ratio_c: float = 3.0
    loss_agg_mode: str = "token-mean"
    loss_scale_factor: Optional[int] = None
    entropy_coeff: float = 0
    tau_pos: float = 1.0
    tau_neg: float = 1.05
    calculate_entropy: bool = False
    use_kl_loss: bool = False
    use_prefix_grouper: bool = False
    use_torch_compile: bool = True
    kl_loss_coef: float = 0.001
    kl_loss_type: str = "low_var_kl"
    ppo_epochs: int = 1
    shuffle: bool = False
    data_loader_seed: int = 1
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    use_fused_kernels: bool = False
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    engine: BaseConfig = field(default_factory=BaseConfig)
    rollout_n: int = MISSING
    model_config: HFModelConfig = field(default_factory=BaseConfig)
    router_replay: RouterReplayConfig = field(default_factory=RouterReplayConfig)
    distillation: SelfDistillationConfig = field(default_factory=SelfDistillationConfig)
    global_batch_info: dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate actor configuration parameters."""
        assert self.strategy != MISSING
        assert self.rollout_n != MISSING
        if not self.use_dynamic_bsz:
            if self.ppo_micro_batch_size is not None and self.ppo_micro_batch_size_per_gpu is not None:
                raise ValueError(
                    "[actor] You have set both 'actor.ppo_micro_batch_size' AND 'actor.ppo_micro_batch_size_per_gpu'. Please remove 'actor.ppo_micro_batch_size' because only '*_ppo_micro_batch_size_per_gpu' is supported (the former is deprecated)."
                )
            else:
                assert not (
                    self.ppo_micro_batch_size is None and self.ppo_micro_batch_size_per_gpu is None
                ), (
                    "[actor] Please set at least one of 'actor.ppo_micro_batch_size' or 'actor.ppo_micro_batch_size_per_gpu' if use_dynamic_bsz is not enabled."
                )
        valid_loss_agg_modes = [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ]
        if self.loss_agg_mode not in valid_loss_agg_modes:
            raise ValueError(f"Invalid loss_agg_mode: {self.loss_agg_mode}")

    def validate(self, n_gpus: int, train_batch_size: int, model_config: dict = None):
        """Validate actor configuration with runtime parameters."""
        if not self.use_dynamic_bsz:
            if train_batch_size < self.ppo_mini_batch_size:
                raise ValueError(
                    f"train_batch_size ({train_batch_size}) must be >= actor.ppo_mini_batch_size ({self.ppo_mini_batch_size})"
                )
            sp_size = getattr(self, "ulysses_sequence_parallel_size", 1)
            if self.ppo_micro_batch_size is not None:
                if self.ppo_mini_batch_size % self.ppo_micro_batch_size != 0:
                    raise ValueError(
                        f"ppo_mini_batch_size ({self.ppo_mini_batch_size}) must be divisible by ppo_micro_batch_size ({self.ppo_micro_batch_size})"
                    )
                if self.ppo_micro_batch_size * sp_size < n_gpus:
                    raise ValueError(
                        f"ppo_micro_batch_size ({self.ppo_micro_batch_size}) * ulysses_sequence_parallel_size ({sp_size}) must be >= n_gpus ({n_gpus})"
                    )

    @staticmethod
    def _check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
        """Validate mutually exclusive micro batch size configuration options."""
        param = "ppo_micro_batch_size"
        param_per_gpu = f"{param}_per_gpu"
        if mbs is None and mbs_per_gpu is None:
            raise ValueError(
                f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'."
            )
        if mbs is not None and mbs_per_gpu is not None:
            raise ValueError(
                f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove '{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
            )


@dataclass
class McoreActorConfig(ActorConfig):
    """Configuration for Megatron actor models.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        strategy (str): Training strategy set to 'megatron' for Megatron parallelism.
        load_weight (bool): Whether to load model weights from checkpoint.
        megatron (dict[str, Any]): Configuration for Megatron parallelism settings.
        profile (dict[str, Any]): Configuration for profiling settings.
    """

    strategy: str = "megatron"
    load_weight: bool = True
    megatron: McoreEngineConfig = field(default_factory=McoreEngineConfig)
    profile: dict[str, Any] = field(default_factory=dict)
    use_rollout_log_probs: bool = False

    def __post_init__(self):
        """Validate FSDP actor configuration parameters."""
        super().__post_init__()
        self.engine = self.megatron


@dataclass
class FSDPActorConfig(ActorConfig):
    """Configuration for FSDP actor models.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        strategy (str): Training strategy set to 'fsdp' for Fully Sharded Data Parallel.
        grad_clip (float): Gradient clipping threshold.
        ulysses_sequence_parallel_size (int): [DEPRECATED] Ulysses sequence parallel size for long sequences.
        entropy_from_logits_with_chunking (bool): Whether to compute entropy from logits
            with chunking for memory efficiency.
        entropy_checkpointing (bool): Whether to use gradient checkpointing for entropy computation.
        fsdp_config (dict[str, Any]): Configuration for FSDP settings.
        use_remove_padding (bool): Whether to remove padding tokens in inputs during training
    """

    strategy: str = "fsdp"
    grad_clip: float = 1.0
    ulysses_sequence_parallel_size: int = 1
    entropy_from_logits_with_chunking: bool = False
    entropy_checkpointing: bool = False
    fsdp_config: FSDPEngineConfig = field(default_factory=FSDPEngineConfig)
    use_remove_padding: bool = False
    use_rollout_log_probs: bool = False
    calculate_sum_pi_squared: bool = False
    sum_pi_squared_checkpointing: bool = False

    def __post_init__(self):
        """Validate FSDP actor configuration parameters."""
        super().__post_init__()
        self.engine = self.fsdp_config
        if self.ulysses_sequence_parallel_size > 1:
            self.fsdp_config.ulysses_sequence_parallel_size = self.ulysses_sequence_parallel_size

    def validate(self, n_gpus: int, train_batch_size: int, model_config: dict = None):
        """Validate FSDP actor configuration with runtime parameters."""
        super().validate(n_gpus, train_batch_size, model_config)
        if self.strategy in {"fsdp", "fsdp2"} and self.ulysses_sequence_parallel_size > 1:
            if model_config and (not model_config.get("use_remove_padding", False)):
                raise ValueError(
                    "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."
                )
