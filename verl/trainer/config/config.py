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

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Optional

from verl.base_config import BaseConfig

__all__ = [
    "CHECKPOINT_REQUIRED_CONTENTS",
    "CheckpointConfig",
    "ProfileConfig",
    "BaseModelConfig",
    "validate_training_checkpoint_contents",
]


CHECKPOINT_REQUIRED_CONTENTS = ("model", "optimizer", "extra")
_CHECKPOINT_ALLOWED_CONTENTS = frozenset((*CHECKPOINT_REQUIRED_CONTENTS, "hf_model"))


def validate_training_checkpoint_contents(
    *,
    save_contents: Sequence[str],
    load_contents: Sequence[str],
    config_name: str = "checkpoint",
) -> None:
    """Reject partial training checkpoints and partial resume configuration."""

    required = set(CHECKPOINT_REQUIRED_CONTENTS)
    for operation, raw_contents in (("save", save_contents), ("load", load_contents)):
        if isinstance(raw_contents, (str, bytes)) or not isinstance(raw_contents, Sequence):
            raise ValueError(
                f"{config_name}.{operation}_contents must be a list containing "
                f"{list(CHECKPOINT_REQUIRED_CONTENTS)!r}; got {raw_contents!r}."
            )

        contents = list(raw_contents)
        missing = required.difference(contents)
        unexpected = set(contents).difference(_CHECKPOINT_ALLOWED_CONTENTS)
        duplicates = sorted({item for item in contents if contents.count(item) > 1})
        if missing or unexpected or duplicates:
            details = []
            if missing:
                details.append(f"missing={sorted(missing)!r}")
            if unexpected:
                details.append(f"unsupported={sorted(unexpected)!r}")
            if duplicates:
                details.append(f"duplicates={duplicates!r}")
            raise ValueError(
                f"Partial or malformed training checkpoint configuration is forbidden: "
                f"{config_name}.{operation}_contents={contents!r} ({', '.join(details)}). "
                f"Every training checkpoint save and resume must include "
                f"{list(CHECKPOINT_REQUIRED_CONTENTS)!r}."
            )


@dataclass
class CheckpointConfig(BaseConfig):
    """Configuration for model checkpointing.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        save_contents (list[str]): What to include in saved checkpoints.
            Options: 'model', 'optimizer', 'extra', 'hf_model'.
        load_contents (list[str]): Contents to load from checkpoint. Defaults to same as save_contents.
        async_save (bool): Whether to save checkpoints asynchronously. Supported by Megatron and FSDP actors.
    """

    save_contents: list[str] = field(default_factory=lambda: ["model", "optimizer", "extra"])
    load_contents: list[str] = field(default_factory=lambda: ["model", "optimizer", "extra"])
    async_save: bool = False
    eval_model_only: bool = False

    def __post_init__(self) -> None:
        validate_training_checkpoint_contents(
            save_contents=self.save_contents,
            load_contents=["model", "optimizer", "extra"] if self.eval_model_only else self.load_contents,
        )
        if self.eval_model_only and list(self.load_contents) != ["model"]:
            raise ValueError("eval_model_only requires checkpoint.load_contents=['model']")


@dataclass
class ProfileConfig(BaseConfig):
    """Configuration for profiling.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        profile_ranks (Optional[list[int]]): List of ranks to profile. None means all ranks.
        step_start (int): Starting step for profiling.
        step_end (int): Ending step for profiling.
        save_path (Optional[str]): Path to save profiling results.
    """

    profile_ranks: Optional[list[int]] = None
    step_start: int = -1
    step_end: int = -1
    save_path: Optional[str] = None


@dataclass
class BaseModelConfig(BaseConfig):
    """Base configuration for a model.
    Contains core settings for loading and initializing a pretrained model checkpoint.

    Args:
        path (str): Path to pretrained model weights.
        tokenizer_path (Optional[str]): Tokenizer path (defaults to actor's model path if not set).
        override_config (dict): Hugging Face config override.
        external_lib (Optional[str]): External model implementation (optional).
        trust_remote_code (bool): Whether to trust remote code from Hugging Face models.
    """

    path: str = "~/models/deepseek-llm-7b-chat"
    tokenizer_path: Optional[str] = None
    override_config: dict[str, Any] = field(default_factory=dict)
    external_lib: Optional[str] = None
    trust_remote_code: bool = False


@dataclass
class ModuleConfig(BaseConfig):
    """Configuration for external Python module, which can be loaded, executed (and optionally, ``import``ed).

    Args:
        path (str, optional): Path to the module file to load and execute.
        name (str, optional): Name of the module to ``import``. Format: ``"import.path.to.module"``.
            If ``None``, the module will be loaded with a hased name and
                will not be added to ``sys.modules``, thus can not be ``import``ed as ``name``.
    """

    path: Optional[str] = None
    name: Optional[str] = None


@dataclass
class RewardManagerConfig(BaseConfig):
    """Configuration for reward manager.

        A reward manager defines the mechanism of computing rule-based reward and handling different reward sources.

    Args:
        source (str): Source of the reward manager. Options: ``"register"``, ``"importlib"``. Default: ``"register"``.
        name (str, optional):
            - When ``source`` is ``"register"``, the name is used in `get_reward_manager_cls(name)``.
                See ``verl/experimental/reward/reward_manager.py`` for options. Default: ``"naive"``.
            - When ``source`` is ``"importlib"``, the name is used in ``getattr(module, name)``,
                e.g., ``"DAPORewardManager"``.
        module (ModuleConfig, optional): Optional configuration for the external module defining the reward manager,
    """

    source: str = "register"
    name: str = "naive"
    module: Optional[ModuleConfig] = field(default_factory=ModuleConfig)

    def __post_init__(self):
        super().__post_init__()
        if self.source == "register":
            from verl.workers.reward_manager.registry import REWARD_MANAGER_REGISTRY

            assert self.name in REWARD_MANAGER_REGISTRY, (
                f"Reward manager is not registered: {self.name=} ,{REWARD_MANAGER_REGISTRY.keys()=}"
            )
        elif self.source == "importlib":
            # NOTE: The existence is not checked since it depends on which machine the config is initialized on.
            assert self.module is not None and self.module.path is not None, (
                "When source is importlib, module.path should be set."
            )
