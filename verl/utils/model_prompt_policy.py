"""Prompt policy registry for model families used in local experiments.

The important contract is explicit:
- models in BASE_CHAT_MODEL_NAMES use chat templates with system prompts dropped;
- models in NON_BASE_CHAT_MODEL_NAMES use chat templates with system prompts kept.

Aliases cover experiment/checkpoint names where ``owner/model`` is rendered as
``owner-model``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable


BASE_CHAT_MODEL_NAMES = frozenset(
    {
        "Qwen/Qwen3-0.6B-Base",
        "Qwen/Qwen3-1.7B-Base",
        "Qwen/Qwen3-4B-Base",
        "Qwen/Qwen3-8B-Base",
        "Qwen/Qwen3-14B-Base",
        "Qwen/Qwen3-32B-Base",
    }
)

NON_BASE_CHAT_MODEL_NAMES = frozenset(
    {
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3-1.7B",
        "Qwen/Qwen3-4B",
        "Qwen/Qwen3-8B",
        "Qwen/Qwen3-14B",
        "Qwen/Qwen3-32B",
        "Qwen/Qwen3.5-9B",
        "microsoft/Phi-3-mini-128k-instruct",
        "microsoft/Phi-3-small-128k-instruct",
        "mistralai/Ministral-3-3B-Instruct-2512",
        "mistralai/Ministral-3-3B-Instruct-2512-BF16",
        "mistralai/Ministral-3-3B-Reasoning-2512",
        "mistralai/Ministral-3-8B-Instruct-2512",
        "mistralai/Ministral-3-8B-Instruct-2512-BF16",
        "mistralai/Ministral-3-8B-Reasoning-2512",
        "mistralai/Ministral-3-14B-Instruct-2512",
        "mistralai/Ministral-3-14B-Instruct-2512-BF16",
        "mistralai/Ministral-3-14B-Reasoning-2512",
        "google/gemma-4-E2B-it",
        "google/gemma-4-E4B-it",
    }
)


def normalize_model_name(value: str | None) -> str:
    return str(value or "").strip().lower().rstrip("/")


def _aliases(model_name: str) -> tuple[str, ...]:
    normalized = normalize_model_name(model_name)
    return (
        normalized,
        normalized.replace("/", "-"),
        normalized.replace("/", "_"),
    )


BASE_CHAT_MODEL_ALIASES = frozenset(alias for name in BASE_CHAT_MODEL_NAMES for alias in _aliases(name))
NON_BASE_CHAT_MODEL_ALIASES = frozenset(
    alias for name in NON_BASE_CHAT_MODEL_NAMES for alias in _aliases(name)
)
STRICT_CHAT_MODEL_ALIASES = BASE_CHAT_MODEL_ALIASES | NON_BASE_CHAT_MODEL_ALIASES


def _contains_alias(haystack: str, aliases: Iterable[str]) -> bool:
    for alias in sorted(aliases, key=len, reverse=True):
        if not alias:
            continue
        pattern = rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])"
        if re.search(pattern, haystack):
            return True
    return False


def prompt_model_kind(*values: str | None) -> str | None:
    """Return ``base``/``non_base``/None from explicit model registry entries."""

    normalized_values = [normalize_model_name(value) for value in values if value]
    for value in normalized_values:
        if value in BASE_CHAT_MODEL_ALIASES:
            return "base"
        if value in NON_BASE_CHAT_MODEL_ALIASES:
            return "non_base"

    haystack = " ".join(normalized_values)
    if _contains_alias(haystack, BASE_CHAT_MODEL_ALIASES):
        return "base"
    if _contains_alias(haystack, NON_BASE_CHAT_MODEL_ALIASES):
        return "non_base"
    return None


def is_base_chat_model(*values: str | None) -> bool:
    return prompt_model_kind(*values) == "base"


def is_non_base_chat_model(*values: str | None) -> bool:
    return prompt_model_kind(*values) == "non_base"


def is_strict_chat_model(*values: str | None) -> bool:
    return prompt_model_kind(*values) is not None


def should_drop_system_prompt(*values: str | None) -> bool | None:
    kind = prompt_model_kind(*values)
    if kind is None:
        return None
    return kind == "base"


def cli() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Classify a model according to the prompt policy registry.")
    parser.add_argument("values", nargs="+")
    parser.add_argument(
        "--field",
        choices=["kind", "drop_system_prompt", "prompt_format"],
        default="kind",
    )
    args = parser.parse_args()

    kind = prompt_model_kind(*args.values)
    if args.field == "kind":
        print(kind or "")
    elif args.field == "drop_system_prompt":
        drop = should_drop_system_prompt(*args.values)
        print("" if drop is None else str(drop).lower())
    elif args.field == "prompt_format":
        print("chat" if kind is not None else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
