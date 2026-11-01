"""Full finetuning is an invariant of the public release, not a selectable mode."""

from collections.abc import Mapping
from pathlib import Path


def validate_model(module):
    """Reject adapters auto-detected by a model loader, including remote model IDs."""
    if getattr(module, "peft_config", None) or getattr(module, "_hf_peft_config_loaded", False):
        raise ValueError("An adapter model was loaded; this release only supports full finetuning")


def validate(config):
    from omegaconf import OmegaConf

    if OmegaConf.is_config(config):
        config = OmegaConf.to_container(config, resolve=False)

    def visit(value, prefix=""):
        if isinstance(value, Mapping):
            for key, item in value.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                name = str(key).lower()
                if (
                    "lora" in name
                    or "peft" in name
                    or name.endswith("adapter_path")
                    or name in {"target_modules", "exclude_modules", "full_finetuning_only"}
                ):
                    raise ValueError(
                        f"Unsupported configuration option {path}: this release only supports full finetuning"
                    )
                if isinstance(item, str) and (name == "path" or name.endswith("model_path")):
                    model = Path(item).expanduser()
                    if (model / "adapter_config.json").is_file():
                        raise ValueError(f"{path} points to an adapter; supply a complete model checkpoint")
                visit(item, path)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item, prefix)

    visit(config)
