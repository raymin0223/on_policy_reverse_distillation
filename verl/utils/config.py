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

import random
from dataclasses import is_dataclass
from typing import Any, Optional
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

__all__ = ["apply_global_seed", "omega_conf_to_dataclass", "update_dict_with_config", "validate_config"]


def apply_global_seed(config: DictConfig) -> int | None:
    """Propagate a top-level seed to the trainer's seed knobs.

    A single ``seed=<int>`` override is easier to reason about than setting
    separate seeds for data order, rollout engines, validation, and model
    workers. If ``config.seed`` is null or absent, this function is a no-op.
    """
    seed = OmegaConf.select(config, "seed")
    if seed is None:
        return None
    seed = int(seed)
    seed_paths = (
        "data.seed",
        "actor_rollout_ref.rollout.seed",
        "actor_rollout_ref.rollout.val_kwargs.seed",
        "actor_rollout_ref.actor.data_loader_seed",
        "actor_rollout_ref.actor.fsdp_config.seed",
        "actor_rollout_ref.actor.megatron.seed",
        "actor_rollout_ref.ref.fsdp_config.seed",
        "actor_rollout_ref.ref.megatron.seed",
        "critic.data_loader_seed",
        "critic.model.fsdp_config.seed",
        "critic.megatron.seed",
        "reward_model.model.fsdp_config.seed",
        "reward_model.megatron.seed",
    )
    for path in seed_paths:
        parent = path.rsplit(".", 1)[0]
        if OmegaConf.select(config, parent) is not None:
            OmegaConf.update(config, path, seed, merge=False, force_add=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    return seed


def omega_conf_to_dataclass(config: DictConfig | dict, dataclass_type: Optional[type[Any]] = None) -> Any:
    """
    Convert an OmegaConf DictConfig to a dataclass.

    Args:
        config: The OmegaConf DictConfig or dict to convert.
        dataclass_type: The dataclass type to convert to. When dataclass_type is None,
            the DictConfig must contain _target_ to be instantiated via hydra.instantiate API.

    Returns:
        The dataclass instance.
    """
    if not config:
        return dataclass_type if dataclass_type is None else dataclass_type()
    if not isinstance(config, DictConfig | ListConfig | dict | list):
        return config
    if dataclass_type is None:
        assert "_target_" in config, (
            f"When dataclass_type is not provided, config must contain _target_. See trainer/config/ppo_trainer.yaml algorithm section for an example. Got config: {config}"
        )
        from hydra.utils import instantiate

        return instantiate(config, _convert_="partial")
    if not is_dataclass(dataclass_type):
        raise ValueError(f"{dataclass_type} must be a dataclass")
    cfg = OmegaConf.create(config)
    cfg_from_dataclass = OmegaConf.structured(dataclass_type)
    cfg_merged = OmegaConf.merge(cfg_from_dataclass, cfg)
    config_object = OmegaConf.to_object(cfg_merged)
    return config_object


def update_dict_with_config(dictionary: dict, config: DictConfig):
    for key in dictionary:
        if hasattr(config, key):
            dictionary[key] = getattr(config, key)


def validate_config(config: DictConfig, use_reference_policy: bool, use_critic: bool) -> None:
    """Validate an OmegaConf DictConfig.

    Args:
        config (DictConfig): The OmegaConf DictConfig to validate.
        use_reference_policy (bool): is ref policy needed
        use_critic (bool): is critic needed
    """
    if config.actor_rollout_ref.actor.checkpoint.get("eval_model_only", False):
        if not config.trainer.get("val_only", False):
            raise ValueError("checkpoint.eval_model_only is allowed only with trainer.val_only=true")
        if config.trainer.get("save_best_validation_checkpoint", False):
            raise ValueError("checkpoint.eval_model_only requires save_best_validation_checkpoint=false")
    if config.actor_rollout_ref.rollout.get("name", None) == "hf":
        raise ValueError("actor_rollout_ref.rollout.name=hf is forbidden; use vllm for generation.")
    n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
    if not config.actor_rollout_ref.actor.use_dynamic_bsz:
        if config.actor_rollout_ref.actor.strategy == "megatron":
            model_parallel_size = (
                config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size
                * config.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size
            )
            assert (
                n_gpus % (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size)
                == 0
            ), (
                f"n_gpus ({n_gpus}) must be divisible by model_parallel_size ({model_parallel_size}) times context_parallel_size ({config.actor_rollout_ref.actor.megatron.context_parallel_size})"
            )
            megatron_dp = n_gpus // (
                model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size
            )
            minimal_bsz = megatron_dp * config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
        else:
            minimal_bsz = n_gpus
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % minimal_bsz == 0, (
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by minimal possible batch size ({minimal_bsz})"
        )

    def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
        """Validate mutually exclusive micro batch size configuration options.

        Ensures that users don't set both deprecated micro_batch_size and
        the new micro_batch_size_per_gpu parameters simultaneously.

        Args:
            mbs: Deprecated micro batch size parameter value.
            mbs_per_gpu: New micro batch size per GPU parameter value.
            name (str): Configuration section name for error messages.

        Raises:
            ValueError: If both parameters are set or neither is set.
        """
        settings = {
            "reward_model": "micro_batch_size",
            "actor_rollout_ref.ref": "log_prob_micro_batch_size",
            "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
        }
        if name in settings:
            param = settings[name]
            param_per_gpu = f"{param}_per_gpu"
            if mbs is None and mbs_per_gpu is None:
                raise ValueError(
                    f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'."
                )
            if mbs is not None and mbs_per_gpu is not None:
                raise ValueError(
                    f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove '{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
                )

    actor_config = omega_conf_to_dataclass(config.actor_rollout_ref.actor)
    actor_config.validate(n_gpus, config.data.train_batch_size, config.actor_rollout_ref.model)
    if not config.actor_rollout_ref.actor.use_dynamic_bsz:
        if use_reference_policy:
            check_mutually_exclusive(
                config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.ref",
            )
        check_mutually_exclusive(
            config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
            config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
            "actor_rollout_ref.rollout",
        )
    if (
        config.reward_model.enable
        and (not config.reward_model.use_dynamic_bsz)
        and (not config.reward_model.use_reward_loop)
    ):
        check_mutually_exclusive(
            config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model"
        )
    if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
        print("NOTICE: You have both enabled in-reward kl and kl loss.")
    if use_critic:
        critic_config = omega_conf_to_dataclass(config.critic)
        critic_config.validate(n_gpus, config.data.train_batch_size)
    if config.data.get("val_batch_size", None) is not None:
        print(
            "WARNING: val_batch_size is deprecated."
            + " Validation datasets are sent to inference engines as a whole batch,"
            + " which will schedule the memory themselves."
        )
    if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
        assert config.actor_rollout_ref.rollout.temperature > 0, (
            "validation gen temperature should be greater than 0 when enabling do_sample"
        )
    if (
        config.actor_rollout_ref.model.get("lora_rank", 0) > 0
        and config.actor_rollout_ref.rollout.name == "vllm"
    ):
        assert config.actor_rollout_ref.model.lora_rank <= 512, (
            "LoRA rank in vLLM must be less than or equal to 512"
        )
    print("[validate_config] All configuration checks passed successfully!")
