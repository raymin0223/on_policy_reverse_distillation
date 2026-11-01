# Copyright 2025 Individual Contributor: Thibaut Barroyer
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
from __future__ import annotations

import inspect
import multiprocessing
import warnings
from collections import defaultdict
from functools import partial
from typing import TYPE_CHECKING, Any, Optional, cast

import numpy as np
import ray
import torch

from verl.utils.reward_score import default_compute_score
from verl.utils.transferqueue_utils import tqbridge

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from verl import DataProto
    from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
    from verl.trainer.config.config import ModuleConfig, RewardManagerConfig
    from verl.workers.reward_manager.abstract import AbstractRewardManager, RawRewardFn
else:
    try:
        from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
    except ImportError:
        RewardManagerBase = None  # type: ignore[assignment,misc]


def _call_with_kwargs(raw_fn, extra_kwargs, *args, **kwargs):
    """Calls `raw_fn` by merging `extra_kwargs` into call-time `kwargs`, with `extra_kwargs` taking precedence.

    This function is used to merge additional keyword arguments with the original function's arguments.
    """
    merged_kwargs = {**kwargs, **extra_kwargs}
    return raw_fn(*args, **merged_kwargs)


async def _call_with_kwargs_async(raw_fn, extra_kwargs, *args, **kwargs):
    """Calls `raw_fn` by merging `extra_kwargs` into call-time `kwargs`, with `extra_kwargs` taking precedence.

    This function is used to merge additional keyword arguments with the original function's arguments.
    """
    merged_kwargs = {**kwargs, **extra_kwargs}
    return await raw_fn(*args, **merged_kwargs)


def get_custom_reward_fn(config: DictConfig) -> Optional[RawRewardFn]:
    """Load and return a custom reward function from external file.

    Dynamically imports a reward function from a specified file path and wraps
    it with additional keyword arguments from the configuration.

    Args:
        config (dict): Configuration dictionary containing custom_reward_function
                      settings with 'path', 'name', and 'reward_kwargs' fields.

    Returns:
        callable or None: Wrapped reward function with merged kwargs, or None
                         if no custom reward function is configured.

    Raises:
        FileNotFoundError: If the specified reward function file doesn't exist.
        RuntimeError: If there's an error loading the module from file.
        AttributeError: If the specified function name isn't found in the module.
    """

    reward_fn_config = config.get("custom_reward_function") or {}
    module_path = reward_fn_config.get("path")
    if not module_path:
        return None

    fn_name = reward_fn_config.get("name")
    assert fn_name is not None

    from verl.utils.import_utils import load_extern_object

    raw_fn = load_extern_object(module_path=module_path, object_name=fn_name)

    reward_kwargs = dict(reward_fn_config.get("reward_kwargs", {}))
    if not inspect.iscoroutinefunction(raw_fn):
        return partial(_call_with_kwargs, raw_fn, reward_kwargs)
    else:
        return partial(_call_with_kwargs_async, raw_fn, reward_kwargs)


def _resolve_overlong_buffer_kwargs(
    config: DictConfig, reward_manager_cls: type, reward_kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Wire `reward_model.overlong_buffer` + `data.max_response_length` into the reward manager.

    Explicit `reward_kwargs` (e.g. `+reward_model.reward_kwargs.overlong_buffer_cfg.*` on the CLI)
    win over the config defaults, matching upstream DAPO scripts.
    """
    from verl.workers.reward_manager.overlong_buffer import build_overlong_buffer

    overlong_buffer_cfg = reward_kwargs.get("overlong_buffer_cfg", config.reward_model.get("overlong_buffer"))
    max_resp_len = reward_kwargs.get("max_resp_len", config.data.get("max_response_length"))

    # Validates enable/len/penalty_factor/max_resp_len; returns None when disabled.
    overlong_buffer = build_overlong_buffer(overlong_buffer_cfg, max_resp_len)
    if overlong_buffer is None:
        return {}

    accepted = inspect.signature(reward_manager_cls.__init__).parameters
    missing = [name for name in ("overlong_buffer_cfg", "max_resp_len") if name not in accepted]
    if missing:
        raise ValueError(
            f"reward_model.overlong_buffer.enable=True, but reward manager "
            f"{reward_manager_cls.__name__} does not accept {missing}. Use a reward manager that "
            f"supports the overlong buffer (naive, parallel_naive, dapo), or disable the feature."
        )

    max_prompt_len = config.data.get("max_prompt_length")
    rollout_cfg = config.get("actor_rollout_ref", {}).get("rollout", {})
    max_model_len = rollout_cfg.get("max_model_len") if rollout_cfg else None
    if max_prompt_len is not None and max_model_len:
        if int(max_prompt_len) + overlong_buffer.max_resp_len > int(max_model_len):
            raise ValueError(
                f"data.max_prompt_length ({max_prompt_len}) + data.max_response_length "
                f"({overlong_buffer.max_resp_len}) exceeds rollout.max_model_len ({max_model_len}); "
                "responses could be truncated below max_response_length, making the overlong penalty "
                "mis-scaled."
            )

    return {"overlong_buffer_cfg": overlong_buffer_cfg, "max_resp_len": overlong_buffer.max_resp_len}


def load_reward_manager(
    config: DictConfig, tokenizer: Any, num_examine: int, is_validation: bool = False, **reward_kwargs: Any
) -> AbstractRewardManager:
    """
    Load and initialize a reward manager based on the configuration.

    Args:
        config: PPO trainer configuration object containing reward_model fields.
        tokenizer: Tokenizer object used for processing text.
        num_examine: Number of samples to examine.
        is_validation: True for the validation reward manager. The overlong buffer is a
            TRAINING-ONLY reward shaping, so it is never wired into a validation manager --
            validation scores stay the raw task reward.
        **reward_kwargs: Additional keyword arguments for the reward manager.

    Returns:
        An instance of the specified reward manager class.
    """

    # Try to get a custom reward function based on the configuration
    # user defined reward manager can be registered in custom_reward_fn
    compute_score = get_custom_reward_fn(config)
    final_compute_score = compute_score

    reward_manager_cfg: RewardManagerConfig = config.reward_manager
    reward_manager_cls: type[AbstractRewardManager]
    if reward_manager_cfg.source == "register":
        from verl.workers.reward_manager import get_reward_manager_cls

        reward_manager_cls = get_reward_manager_cls(reward_manager_cfg.name)
    elif reward_manager_cfg.source == "importlib":
        from verl.utils.import_utils import load_extern_object

        module_cfg: ModuleConfig | None = reward_manager_cfg.module
        assert module_cfg is not None and module_cfg.path is not None, (
            f"Module path is required when {reward_manager_cfg.source=}, but got {module_cfg=}"
        )
        reward_manager_cls_name = reward_manager_cfg.name
        reward_manager_cls = cast(
            "type[AbstractRewardManager]",
            load_extern_object(module_path=module_cfg.path, object_name=reward_manager_cls_name),
        )

    if compute_score is None:
        sandbox_config = config.reward_model.get("sandbox_fusion")
        sandbox_url = sandbox_config.get("url") if sandbox_config else None
        memory_limit_mb = sandbox_config.get("memory_limit_mb", 1024) if sandbox_config else 1024
        if sandbox_url:
            sandbox_manager = multiprocessing.Manager()
            # Create a semaphore to control concurrent access to the sandbox
            _concurrent_semaphore = sandbox_manager.Semaphore(sandbox_config.get("max_concurrent", 64))
            final_compute_score = partial(
                default_compute_score,
                sandbox_fusion_url=sandbox_url,
                concurrent_semaphore=_concurrent_semaphore,
                memory_limit_mb=memory_limit_mb,
            )
        else:
            final_compute_score = default_compute_score

    # Instantiate and return the reward manager with the specified parameters
    # RewardManagerBase subclasses (like RateLimitedRewardLoopManager) don't accept num_examine
    # while AbstractRewardManager subclasses (like NaiveRewardManager) do
    if RewardManagerBase is not None and issubclass(reward_manager_cls, RewardManagerBase):
        # RewardManagerBase-based managers use a different signature
        return reward_manager_cls(
            config=config,
            tokenizer=tokenizer,
            compute_score=final_compute_score,
            **reward_kwargs,
        )
    else:
        # Traditional AbstractRewardManager-based managers.
        # Validation is never length-penalized: it evaluates the policy, it does not shape it.
        if is_validation:
            reward_kwargs = {
                k: v for k, v in reward_kwargs.items() if k not in ("overlong_buffer_cfg", "max_resp_len")
            }
        else:
            reward_kwargs = {
                **reward_kwargs,
                **_resolve_overlong_buffer_kwargs(config, reward_manager_cls, reward_kwargs),
            }
        return reward_manager_cls(
            tokenizer=tokenizer,
            num_examine=num_examine,
            compute_score=final_compute_score,
            reward_fn_key=config.data.reward_fn_key,
            **reward_kwargs,
        )


@tqbridge(put_data=False)
def compute_reward(data: DataProto, reward_fn: AbstractRewardManager) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute reward for a batch of data.
    Args:
        data: DataProto object containing the input data.
        reward_fn: Reward function to compute the reward.
    Returns:
        Tuple of reward tensor and extra info dictionary.
    """
    try:
        reward_result = reward_fn(data, return_dict=True)
        reward_tensor = reward_result["reward_tensor"]
        reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
    except Exception as e:
        print(f"Error in reward_fn: {e}")
        reward_tensor = reward_fn(data)
        reward_extra_infos_dict = {}

    return reward_tensor, reward_extra_infos_dict


@ray.remote(num_cpus=1)
def compute_reward_async(data: DataProto, config=None, tokenizer=None, reward_fn=None):
    """
    Load the reward manager and compute the reward for a batch of data.
    This is meant to be run in a separate Ray worker.
    """
    if reward_fn is None:
        assert config is not None and tokenizer is not None, (
            "config and tokenizer must not be None when reward_fn is None"
        )

        warnings.warn("using config and tokenizer with compute_reward_async is deprecated", stacklevel=2)
        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )

    return compute_reward(data, reward_fn)


@ray.remote(num_cpus=1)
def compute_validation_reward_payload_async(
    payload: dict[str, Any], config=None, tokenizer=None, reward_fn=None
):
    """Compute deferred validation rewards in a separate Ray worker process."""
    if reward_fn is None:
        assert config is not None and tokenizer is not None, (
            "config and tokenizer must not be None when reward_fn is None"
        )
        reward_fn = load_reward_manager(
            config,
            tokenizer,
            num_examine=1,
            is_validation=True,
            **config.reward_model.get("reward_kwargs", {}),
        )

    reward_extra_infos_dict: dict[str, list] = defaultdict(list)
    sample_scores = []

    for test_batch in payload["reward_batches"]:
        result = reward_fn(test_batch, return_dict=True)
        reward_tensor = result["reward_tensor"]
        scores = reward_tensor.sum(-1).cpu().tolist()
        sample_scores.extend(scores)

        reward_extra_infos_dict["reward"].extend(scores)
        reward_extra_info = result.get("reward_extra_info", {})
        for key, values in reward_extra_info.items():
            if key not in reward_extra_infos_dict:
                reward_extra_infos_dict[key] = []
            if isinstance(values, np.ndarray):
                reward_extra_infos_dict[key].extend(values.tolist())
            else:
                reward_extra_infos_dict[key].extend(values if isinstance(values, list) else [values])

    scored_payload = dict(payload)
    scored_payload["sample_scores"] = sample_scores
    scored_payload["reward_extra_infos_dict"] = reward_extra_infos_dict
    scored_payload.pop("reward_batches", None)
    return scored_payload
