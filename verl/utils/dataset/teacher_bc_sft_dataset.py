"""Token-exact SFT dataset for offline teacher behavior-cloning trajectories."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import torch

from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.dataset.multiturn_sft_dataset import (
    MultiTurnSFTDataset,
    convert_nested_value_to_list_recursive,
)


def _tokenizer_vocab_fingerprint(tokenizer) -> str:
    encoded = json.dumps(
        sorted(tokenizer.get_vocab().items()),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class TeacherBCSFTDataset(MultiTurnSFTDataset):
    """Train on the exact prompt and response token IDs emitted by vLLM.

    The stored response includes vLLM's stopping token.  This matters because
    Qwen3 Base stops on ``<|endoftext|>`` while Qwen3 Instruct normally stops on
    ``<|im_end|>``.  Reconstructing an assistant turn from text would discard
    that distinction.  We also render the prompt again with the SFT tokenizer
    and require exact equality with the IDs vLLM actually consumed.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.pad_mode != DatasetPadMode.NO_PADDING:
            raise ValueError("TeacherBCSFTDataset requires data.pad_mode=no_padding")
        if self.processor is not None:
            raise ValueError("TeacherBCSFTDataset currently supports text-only behavior cloning")
        if self.tools is not None:
            raise ValueError("TeacherBCSFTDataset does not support tool-call trajectories")

        config = kwargs.get("config")
        if config is None and len(args) >= 3:
            config = args[2]
        config = config or {}
        prompt_key = config.get("prompt_token_ids_key", "prompt_token_ids")
        response_key = config.get("response_token_ids_key", "response_token_ids")
        vocab_hash_key = config.get("tokenizer_vocab_sha256_key", "tokenizer_vocab_sha256")
        required = {prompt_key, response_key, vocab_hash_key}
        missing = sorted(required - set(self.dataframe.columns))
        if missing:
            raise ValueError(f"TeacherBCSFTDataset parquet is missing token-exact columns: {missing}")

        self.prompt_token_ids = (
            self.dataframe[prompt_key].apply(convert_nested_value_to_list_recursive).tolist()
        )
        self.response_token_ids = (
            self.dataframe[response_key].apply(convert_nested_value_to_list_recursive).tolist()
        )
        expected_vocab_hashes = set(self.dataframe[vocab_hash_key].astype(str).tolist())
        if len(expected_vocab_hashes) != 1:
            raise ValueError(
                f"Expected one teacher tokenizer vocabulary hash, got {sorted(expected_vocab_hashes)}"
            )
        expected_vocab_hash = next(iter(expected_vocab_hashes))
        actual_vocab_hash = _tokenizer_vocab_fingerprint(self.tokenizer)
        if actual_vocab_hash != expected_vocab_hash:
            raise ValueError(
                "Teacher and SFT tokenizers do not have the same token-to-ID vocabulary: "
                f"teacher={expected_vocab_hash}, sft={actual_vocab_hash}"
            )

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        messages = self.messages[item]
        if not messages or str(messages[-1].get("role", "")).lower() != "assistant":
            raise ValueError(f"SFT row {item} must end in exactly one assistant rollout")

        enable_thinking = self.enable_thinking[item] if self.enable_thinking is not None else None
        apply_kwargs = dict(self.apply_chat_template_kwargs)
        if enable_thinking is not None:
            apply_kwargs["enable_thinking"] = bool(enable_thinking)

        prompt_inputs = self.tokenizer.apply_chat_template(
            messages[:-1],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            **apply_kwargs,
        )
        rendered_prompt_ids = prompt_inputs["input_ids"][0]
        prompt_ids = torch.tensor(
            [int(token_id) for token_id in self.prompt_token_ids[item]], dtype=torch.long
        )
        response_ids = torch.tensor(
            [int(token_id) for token_id in self.response_token_ids[item]], dtype=torch.long
        )
        if not torch.equal(rendered_prompt_ids, prompt_ids):
            raise AssertionError(
                f"SFT row {item} prompt IDs differ from the IDs consumed during teacher generation "
                f"(rendered_tokens={len(rendered_prompt_ids)}, stored_tokens={len(prompt_ids)})"
            )
        if len(response_ids) == 0:
            raise ValueError(f"SFT row {item} contains no teacher response token IDs")
        if int(response_ids.min()) < 0 or int(response_ids.max()) >= len(self.tokenizer):
            raise ValueError(
                f"SFT row {item} contains response token IDs outside the SFT tokenizer vocabulary"
            )

        input_ids = torch.cat((prompt_ids, response_ids))
        if len(input_ids) > self.max_length:
            raise ValueError(f"sequence_length={len(input_ids)} is larger than max_length={self.max_length}")

        loss_mask = torch.zeros_like(input_ids)
        loss_mask[len(prompt_ids) :] = 1
        position_ids = torch.arange(len(input_ids), dtype=torch.long)
        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }
