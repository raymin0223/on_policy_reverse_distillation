#!/usr/bin/env python3
"""Reformat official DAPO-Math parquet files into the self-distillation prompt style."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.format.math import SYSTEM_PROMPT


DAPO_LEADING_INSTRUCTION_RE = re.compile(
    r"^Solve the following math problem step by step\.\s*"
    r"The last line of your response should be of the form Answer: \$Answer "
    r"\(without quotes\) where \$Answer is the answer to the problem\.\s*\n*",
)
DAPO_TRAILING_INSTRUCTION_RE = re.compile(
    r'\s*\n*Remember to put your answer on its own line after "Answer:"\.?\s*$',
)
BOXED_SUFFIX = r"Please reason step by step, and put your final answer within \boxed{}."


def _as_messages(prompt: Any) -> list[dict[str, Any]]:
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    if isinstance(prompt, str):
        try:
            prompt = json.loads(prompt)
        except json.JSONDecodeError:
            prompt = {"role": "user", "content": prompt}
    if isinstance(prompt, dict):
        prompt = [prompt]
    if not isinstance(prompt, list):
        raise TypeError(f"Unsupported prompt type: {type(prompt)!r}")

    messages = []
    for message in prompt:
        if hasattr(message, "as_py"):
            message = message.as_py()
        if not isinstance(message, dict):
            raise TypeError(f"Unsupported message type: {type(message)!r}")
        messages.append(dict(message))
    return messages


def _reformat_problem_text(text: str) -> str:
    text = DAPO_LEADING_INSTRUCTION_RE.sub("", text)
    text = DAPO_TRAILING_INSTRUCTION_RE.sub("", text).strip()
    if text.endswith(BOXED_SUFFIX):
        return text
    return f"{text}\n{BOXED_SUFFIX}"


def _reformat_prompt(prompt: Any) -> list[dict[str, Any]]:
    messages = _as_messages(prompt)
    user = next((m for m in messages if m.get("role") == "user"), messages[0] if messages else {})
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _reformat_problem_text(str(user.get("content", "")))},
    ]


def _process_row(row: dict[str, Any], data_source: str | None) -> dict[str, Any]:
    row = dict(row)
    row["prompt"] = _reformat_prompt(row["prompt"])
    row["system"] = SYSTEM_PROMPT
    if data_source:
        row["data_source"] = data_source
    return row


def convert_parquet(
    input_path: Path,
    output_path: Path,
    *,
    data_source: str | None = None,
    batch_size: int = 4096,
    limit: int | None = None,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_file = pq.ParquetFile(input_path)

    writer: pq.ParquetWriter | None = None
    schema: pa.Schema | None = None
    written = 0
    try:
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            rows = []
            for row in batch.to_pylist():
                if limit is not None and written + len(rows) >= limit:
                    break
                rows.append(_process_row(row, data_source))

            if not rows:
                break

            table = pa.Table.from_pylist(rows)
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(output_path, schema, compression="zstd")
            elif schema is not None and table.schema != schema:
                table = table.cast(schema)

            writer.write_table(table)
            written += len(rows)

            if limit is not None and written >= limit:
                break
    finally:
        if writer is not None:
            writer.close()

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Official DAPO-style parquet file.")
    parser.add_argument("--output", required=True, type=Path, help="Output parquet path.")
    parser.add_argument(
        "--data-source",
        default=None,
        help="Optional override for the data_source column. By default the source value is preserved.",
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=-1, help="Optional row limit for smoke tests.")
    args = parser.parse_args()

    limit = None if args.limit is None or args.limit < 0 else args.limit
    count = convert_parquet(
        args.input,
        args.output,
        data_source=args.data_source,
        batch_size=args.batch_size,
        limit=limit,
    )
    print(f"Wrote {count} rows to {args.output}")


if __name__ == "__main__":
    main()
