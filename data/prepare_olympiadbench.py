#!/usr/bin/env python3
"""Prepare the canonical OlympiadBench English text-only math subset for eval."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

from data.format.math import SYSTEM_PROMPT


HF_REPO = "Hothan/OlympiadBench"
HF_CONFIG = "OE_TO_maths_en_COMP"
HF_REVISION = "91184b52131e7fc9455fef848035173aea8cc01a"
EXPECTED_ROWS = 674


def _clean_gold(value: Any) -> str:
    """Remove display-math delimiters while preserving the answer expression."""

    answer = str(value).strip().replace("$", "")
    if answer.startswith(r"\(") and answer.endswith(r"\)"):
        answer = answer[2:-2].strip()
    if answer.startswith(r"\[") and answer.endswith(r"\]"):
        answer = answer[2:-2].strip()
    return answer


def _instruction(example: dict[str, Any]) -> str:
    answer_type = str(example.get("answer_type") or "answer").strip().lower()
    multiple = bool(example.get("is_multiple_answer"))
    answer_clause = (
        "Put every requested final value together inside exactly one LaTeX box, "
        "separating multiple answers with commas."
        if multiple
        else "Put the final answer inside exactly one LaTeX box."
    )
    unit_clause = (
        " State the unit outside the box; do not include the unit inside the box."
        if example.get("unit") not in (None, "")
        else ""
    )
    return (
        f"The expected answer type is {answer_type}. Please reason step by step. {answer_clause}{unit_clause}"
    )


def _convert(example: dict[str, Any], index: int) -> dict[str, Any]:
    final_answers = example.get("final_answer")
    if not isinstance(final_answers, list) or len(final_answers) != 1:
        raise ValueError(
            f"Expected one canonical final_answer string for id={example.get('id')}, got {final_answers!r}"
        )

    question = str(example["question"]).strip()
    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{question}\n\n{_instruction(example)}"},
    ]
    extra_info = {
        "index": f"olympiadbench:{example['id']}",
        "id": example["id"],
        "question": question,
        "raw_question": question,
        "split": "train",
        "hf_repo": HF_REPO,
        "hf_config": HF_CONFIG,
        "hf_revision": HF_REVISION,
        "answer_type": example.get("answer_type"),
        "is_multiple_answer": example.get("is_multiple_answer"),
        "unit": example.get("unit"),
        "error": example.get("error"),
        "subfield": example.get("subfield"),
        "source_index": index,
    }
    return {
        # Reuse the repository's established math verifier and mean@16 policy.
        "data_source": "competition_math",
        "prompt": prompt,
        "ability": "MATH",
        "reward_model": {
            "ground_truth": _clean_gold(final_answers[0]),
            "style": "rule-lighteval/MATH_v2",
        },
        "extra_info": extra_info,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/math/evaluation/olympiadbench_oe_to_maths_en_comp.parquet"),
    )
    parser.add_argument("--limit", type=int, default=-1)
    args = parser.parse_args()

    dataset = load_dataset(
        HF_REPO,
        HF_CONFIG,
        split="train",
        revision=HF_REVISION,
    )
    if len(dataset) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} rows from {HF_REPO}/{HF_CONFIG}, found {len(dataset)}")
    if args.limit >= 0:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    rows = [_convert(example, index) for index, example in enumerate(dataset)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), args.output, compression="zstd")
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
