from __future__ import annotations
import json
import os
from typing import Any
from data.code_rlvr_prompts import build_messages
from data.prepare_deepcoder_rlvr import normalize_tests
from datasets import Dataset, load_dataset

DEEPCODER_DATASET = "agentica-org/DeepCoder-Preview-Dataset"
DEEPCODER_REVISION = os.getenv("DEEPCODER_DATASET_REVISION", "177913a7bd43791646ef6a43645caa3c871ab3db")
CODECONTESTS_REVISION = os.getenv("CODECONTESTS_DATASET_REVISION", "802411c3010cb00d1b05bad57ca77365a3c699d6")
EVAL_MEMORY_LIMIT_BYTES = 4 * 1024**3
LCB_EVAL_TIMEOUT_SECONDS = 6.0
OTHER_EVAL_TIMEOUT_SECONDS = 12.0
EVAL_MEMORY_LIMIT_MODE = "address_space_data_rss"


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def prepare_eval_tests(tests: Any, data_source: str) -> list[dict[str, Any]]:
    if not isinstance(tests, list) or not tests:
        raise ValueError(f"{data_source} eval tests must be a non-empty list")
    timeout_seconds = LCB_EVAL_TIMEOUT_SECONDS if data_source == "lcbv5" else OTHER_EVAL_TIMEOUT_SECONDS
    prepared = []
    for test_case in tests:
        if not isinstance(test_case, dict):
            raise ValueError(f"{data_source} eval test case must be a dictionary")
        prepared_case = dict(test_case)
        metadata = dict(prepared_case.get("metadata") or {})
        metadata.update(
            {
                "timeout_seconds": timeout_seconds,
                "memory_limit_bytes": EVAL_MEMORY_LIMIT_BYTES,
                "memory_limit_mode": EVAL_MEMORY_LIMIT_MODE,
            }
        )
        prepared_case["metadata"] = metadata
        prepared.append(prepared_case)
    return prepared


def _make_row(
    *,
    data_source: str,
    problem: str,
    tests: Any,
    starter_code: str = "",
    index: str,
    source_split: str,
    logical_split: str = "test",
    eval_metric: str = "mean@4",
) -> dict[str, Any]:
    prepared_tests = prepare_eval_tests(tests, data_source)
    prompt = build_messages(problem, starter_code=starter_code)
    return {
        "data_source": data_source,
        "question": problem,
        "prompt": prompt,
        "raw_prompt": prompt,
        "ability": "code",
        "reward_model": {"style": "rule", "ground_truth": _json_dumps(prepared_tests)},
        "extra_info": {
            "split": logical_split,
            "index": index,
            "source_split": source_split,
            "eval_metric": eval_metric,
            "test_selection": "all_hidden_tests",
            "test_count": len(prepared_tests),
        },
        "uid": f"{data_source}:{index}",
    }


def build_deepcoder_lcbv5() -> Dataset:
    ds = load_dataset(DEEPCODER_DATASET, "lcbv5", split="test", revision=DEEPCODER_REVISION)
    rows = []
    for idx, row in enumerate(ds):
        row = dict(row)
        problem = str(row["problem"])
        tests = normalize_tests(row, "lcbv5")
        rows.append(
            _make_row(
                data_source="lcbv5",
                problem=problem,
                tests=tests,
                starter_code=str(row.get("starter_code") or ""),
                index=str(idx),
                source_split="lcbv5/test",
            )
        )
    return Dataset.from_list(rows)


def _pair_tests(inputs: Any, outputs: Any) -> list[dict[str, Any]]:
    inputs = list(inputs or [])
    outputs = list(outputs or [])
    return [{"input": inp, "output": out, "metadata": {}} for inp, out in zip(inputs, outputs, strict=True)]


def _codecontests_tests(row: dict[str, Any]) -> list[dict[str, Any]]:
    tests = []
    for key in ("public_tests", "private_tests", "generated_tests"):
        block = row.get(key) or {}
        if isinstance(block, dict):
            tests.extend(_pair_tests(block.get("input"), block.get("output")))
    if not tests:
        raise ValueError("CodeContests row has no tests")
    return tests
