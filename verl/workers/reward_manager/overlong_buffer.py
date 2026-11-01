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

"""DAPO-style Soft Overlong Punishment.

Given the generator's hard cap ``M = max_response_length`` and a buffer of the last
``B = overlong_buffer.len`` response tokens, a response of valid length ``L`` gets

    ratio   = clamp((L - (M - B)) / B, 0, 1)
    penalty = -penalty_factor * ratio

so the reward is untouched while ``L <= M - B``, decays linearly across the buffer, and
reaches exactly ``-penalty_factor`` at ``L == M``. The penalty is added to the scalar task
reward before any group normalization; it never changes the generated tokens.

``penalty_factor=1.0`` assumes the usual 0/1 correctness reward scale — with a differently
scaled task reward, scale the factor to match.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

__all__ = ["OverlongBuffer", "build_overlong_buffer", "valid_response_lengths"]


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Read a key from a DictConfig, dict, dataclass, or namespace config object."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    getter = getattr(cfg, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            pass
    return getattr(cfg, key, default)


def _as_int(value: Any) -> int:
    return int(value.item() if hasattr(value, "item") else value)


@dataclass(frozen=True)
class OverlongBuffer:
    """Validated soft overlong punishment. Construct via :func:`build_overlong_buffer`."""

    max_resp_len: int
    buffer_len: int
    penalty_factor: float
    log: bool = False

    @property
    def expected_len(self) -> int:
        """``T = M - B``: the longest response that escapes the penalty entirely."""
        return self.max_resp_len - self.buffer_len

    def is_overlong(self, valid_response_length: Any) -> bool:
        return _as_int(valid_response_length) > self.expected_len

    def penalty(self, valid_response_length: Any) -> float:
        """Scalar penalty in ``[-penalty_factor, 0]`` for one trajectory."""
        length = _as_int(valid_response_length)
        if length < 1:
            raise ValueError(
                f"overlong buffer needs a valid response length >= 1 to index the final response token, "
                f"but got {length}"
            )
        ratio = (length - self.expected_len) / self.buffer_len
        ratio = min(max(ratio, 0.0), 1.0)
        return -self.penalty_factor * ratio

    def penalty_batch(self, valid_response_lengths: torch.Tensor) -> torch.Tensor:
        """Vectorized :meth:`penalty` over a 1-D tensor of lengths. Matches the scalar path."""
        lengths = valid_response_lengths.to(dtype=torch.float32)
        if int(lengths.min().item()) < 1:
            raise ValueError(
                "overlong buffer needs a valid response length >= 1 to index the final response token, "
                f"but got a minimum of {int(lengths.min().item())}"
            )
        ratio = ((lengths - self.expected_len) / self.buffer_len).clamp(min=0.0, max=1.0)
        return -self.penalty_factor * ratio


def build_overlong_buffer(overlong_buffer_cfg: Any, max_resp_len: Any) -> OverlongBuffer | None:
    """Return an :class:`OverlongBuffer`, or ``None`` when the feature is off.

    Raises ValueError on an enabled-but-invalid configuration rather than silently
    degrading to a no-op penalty.
    """
    if overlong_buffer_cfg is None:
        return None
    if not bool(_cfg_get(overlong_buffer_cfg, "enable", False)):
        return None

    if max_resp_len is None:
        raise ValueError(
            "overlong_buffer.enable=True requires a max response length "
            "(set data.max_response_length, or pass max_resp_len to the reward manager)"
        )
    max_resp_len = _as_int(max_resp_len)

    buffer_len = _cfg_get(overlong_buffer_cfg, "len", None)
    if buffer_len is None:
        raise ValueError("overlong_buffer.enable=True requires overlong_buffer.len")
    buffer_len = _as_int(buffer_len)
    if buffer_len <= 0:
        raise ValueError(f"overlong_buffer.len must be > 0, but got {buffer_len}")
    if max_resp_len < buffer_len:
        raise ValueError(
            f"max_response_length ({max_resp_len}) must be >= overlong_buffer.len ({buffer_len})"
        )

    penalty_factor = float(_cfg_get(overlong_buffer_cfg, "penalty_factor", 0.0) or 0.0)
    if penalty_factor < 0:
        raise ValueError(
            f"overlong_buffer.penalty_factor must be >= 0 (the penalty sign is applied internally), "
            f"but got {penalty_factor}"
        )

    return OverlongBuffer(
        max_resp_len=max_resp_len,
        buffer_len=buffer_len,
        penalty_factor=penalty_factor,
        log=bool(_cfg_get(overlong_buffer_cfg, "log", False)),
    )


def valid_response_lengths(data) -> torch.Tensor:
    """Per-sequence count of generated response tokens, from the attention mask.

    Prompt tokens and right-padding are excluded, so this is L — not the decoded string
    length and not the padded tensor width.
    """
    prompt_length = data.batch["prompts"].shape[-1]
    return data.batch["attention_mask"][:, prompt_length:].sum(dim=-1)
