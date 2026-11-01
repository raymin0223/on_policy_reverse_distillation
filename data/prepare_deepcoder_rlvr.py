from __future__ import annotations
import json
import os
from typing import Any
from datasets import Dataset, concatenate_datasets, load_dataset
from data.code_rlvr_prompts import build_user_prompt

DATASET_NAME = "agentica-org/DeepCoder-Preview-Dataset"
DATASET_REVISION = os.getenv("DEEPCODER_DATASET_REVISION", "177913a7bd43791646ef6a43645caa3c871ab3db")
TRAIN_SOURCES = ("lcbv5", "primeintellect", "taco")
DEFAULT_MIN_TRAIN_TESTS = 5
DEFAULT_MAX_TRAIN_TESTS = 15
LCBV5_TIMEOUT_SECONDS = 6.0
OTHER_TRAIN_TIMEOUT_SECONDS = 12.0
TACO_CALL_MEMORY_LIMIT_BYTES = 4 * 1024**3
TACO_STDIN_MEMORY_LIMIT_BYTES = 10 * 1024**3
LOCAL_CODE_MEMORY_LIMIT_BYTES = 4 * 1024**3
MEMORY_LIMIT_MODE_FULL = "address_space_data_rss"
MEMORY_LIMIT_MODE_VIRTUAL = "virtual_memory"


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _first_present(row: dict[str, Any], keys: tuple[str, ...], default: Any = "") -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return default


def extract_problem(row: dict[str, Any]) -> str:
    return _as_text(
        _first_present(
            row, ("problem", "question", "prompt", "question_content", "problem_statement", "description")
        )
    )


def extract_starter_code(row: dict[str, Any]) -> str:
    return _as_text(_first_present(row, ("starter_code", "starter", "code_stub"), "")).strip()


def _parse_json_field(value: Any, field_name: str) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    if value is None:
        raise ValueError(f"missing {field_name}")
    return json.loads(json.dumps(value))


def _parse_metadata(value: Any) -> dict[str, Any]:
    if value in (None, ""):
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        return json.loads(value)
    return dict(value)


def _extract_tests_obj(row: dict[str, Any]) -> Any:
    tests = _first_present(row, ("tests", "input_output", "verification_info", "test_cases"), None)
    return _parse_json_field(tests, "tests")


def _metadata_fn_name(metadata: dict[str, Any]) -> str:
    return str(metadata.get("func_name") or metadata.get("fn_name") or "").strip()


def _attach_fn_name(test_obj: Any, fn_name: str) -> Any:
    if not fn_name:
        return test_obj
    if isinstance(test_obj, dict):
        metadata = dict(test_obj.get("metadata") or {})
        metadata["func_name"] = fn_name
        test_obj["metadata"] = metadata
        test_obj.setdefault("fn_name", fn_name)
        return test_obj
    if isinstance(test_obj, list):
        for test in test_obj:
            if isinstance(test, dict):
                metadata = dict(test.get("metadata") or {})
                metadata["func_name"] = fn_name
                test["metadata"] = metadata
        return test_obj
    raise ValueError(f"cannot attach fn_name to tests of type {type(test_obj).__name__}")


def _standard_io_to_lcb_format(test_obj: Any) -> list[dict[str, Any]]:
    if isinstance(test_obj, dict) and "inputs" in test_obj and ("outputs" in test_obj):
        inputs = test_obj["inputs"]
        outputs = test_obj["outputs"]
        if len(inputs) != len(outputs):
            raise ValueError("inputs and outputs have different lengths")
        return [
            {"input": inp, "output": out, "metadata": {}} for inp, out in zip(inputs, outputs, strict=True)
        ]
    if isinstance(test_obj, list):
        out = []
        for test in test_obj:
            if not isinstance(test, dict):
                raise ValueError("test case must be a dictionary")
            out.append(
                {
                    "input": test.get("input", test.get("inputs", "")),
                    "output": test.get("output", test.get("outputs", "")),
                    "metadata": dict(test.get("metadata") or {}),
                }
            )
        return out
    raise ValueError(f"unsupported standard IO test format: {type(test_obj).__name__}")


def taco_to_lcb_format(test_obj: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(test_obj, dict) or "inputs" not in test_obj or "outputs" not in test_obj:
        raise ValueError("TACO tests must be a dictionary with inputs and outputs")
    inputs = test_obj["inputs"]
    outputs = test_obj["outputs"]
    if len(inputs) != len(outputs):
        raise ValueError("TACO inputs and outputs have different lengths")
    fn_name = str(test_obj.get("fn_name") or "").strip()
    metadata = {"func_name": fn_name} if fn_name else {}
    return [
        {"input": inp, "output": out, "metadata": dict(metadata)}
        for inp, out in zip(inputs, outputs, strict=True)
    ]


def normalize_tests(row: dict[str, Any], source_name: str) -> Any:
    tests = _extract_tests_obj(row)
    if source_name == "lcbv5":
        metadata = _parse_metadata(row.get("metadata"))
        return _attach_fn_name(tests, _metadata_fn_name(metadata))
    if source_name == "primeintellect":
        return _standard_io_to_lcb_format(tests)
    if source_name == "taco":
        return taco_to_lcb_format(tests)
    raise ValueError(f"unsupported source: {source_name}")


def _test_input_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _test_fn_name(test_case: dict[str, Any]) -> str:
    metadata = test_case.get("metadata") or {}
    return str(
        test_case.get("fn_name")
        or test_case.get("func_name")
        or metadata.get("fn_name")
        or metadata.get("func_name")
        or ""
    ).strip()


def select_training_tests(
    tests: list[dict[str, Any]],
    *,
    min_tests: int = DEFAULT_MIN_TRAIN_TESTS,
    max_tests: int = DEFAULT_MAX_TRAIN_TESTS,
) -> list[dict[str, Any]]:
    if min_tests <= 0:
        raise ValueError(f"min_tests must be positive, got {min_tests}")
    if max_tests < min_tests:
        raise ValueError(f"max_tests ({max_tests}) must be >= min_tests ({min_tests})")
    if len(tests) < min_tests:
        raise ValueError(f"training row has only {len(tests)} tests; at least {min_tests} are required")
    if len(tests) <= max_tests:
        return list(tests)
    ranked = sorted(
        enumerate(tests), key=lambda item: (-len(_test_input_text(item[1].get("input"))), item[0])
    )
    return [test for _, test in ranked[:max_tests]]


def _training_execution_policy(source_name: str, test_case: dict[str, Any]) -> dict[str, Any]:
    timeout_seconds = LCBV5_TIMEOUT_SECONDS if source_name == "lcbv5" else OTHER_TRAIN_TIMEOUT_SECONDS
    memory_limit_bytes: int | None = LOCAL_CODE_MEMORY_LIMIT_BYTES
    memory_limit_mode = MEMORY_LIMIT_MODE_FULL
    if source_name == "taco":
        if _test_fn_name(test_case):
            memory_limit_bytes = TACO_CALL_MEMORY_LIMIT_BYTES
            memory_limit_mode = MEMORY_LIMIT_MODE_FULL
        else:
            memory_limit_bytes = TACO_STDIN_MEMORY_LIMIT_BYTES
            memory_limit_mode = MEMORY_LIMIT_MODE_VIRTUAL
    return {
        "timeout_seconds": timeout_seconds,
        "memory_limit_bytes": memory_limit_bytes,
        "memory_limit_mode": memory_limit_mode,
    }


def prepare_training_tests(
    tests: list[dict[str, Any]],
    source_name: str,
    *,
    min_tests: int = DEFAULT_MIN_TRAIN_TESTS,
    max_tests: int = DEFAULT_MAX_TRAIN_TESTS,
) -> list[dict[str, Any]]:
    selected = select_training_tests(tests, min_tests=min_tests, max_tests=max_tests)
    prepared = []
    for test_case in selected:
        prepared_case = dict(test_case)
        metadata = dict(prepared_case.get("metadata") or {})
        metadata.update(_training_execution_policy(source_name, prepared_case))
        prepared_case["metadata"] = metadata
        prepared.append(prepared_case)
    return prepared


def convert_row(
    row: dict[str, Any],
    source_name: str,
) -> dict[str, Any]:
    problem = extract_problem(row)
    starter_code = extract_starter_code(row)
    prompt_text = build_user_prompt(problem, starter_code=starter_code)
    normalized_tests = normalize_tests(row, source_name)
    ground_truth = json.dumps(normalized_tests, ensure_ascii=False)
    prompt = [{"role": "user", "content": prompt_text}]
    return {
        "data_source": source_name,
        "question": problem,
        "prompt": prompt,
        "raw_prompt": prompt,
        "ability": "code",
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
    }


def load_train_source(dataset_name: str, source_name: str) -> Dataset:
    kwargs = {}
    if dataset_name == DATASET_NAME:
        kwargs["revision"] = DATASET_REVISION
    return load_dataset(dataset_name, source_name, split="train", **kwargs)


def process_source(
    dataset_name: str,
    source_name: str,
    *,
    max_examples: int | None = None,
    min_tests_per_example: int = DEFAULT_MIN_TRAIN_TESTS,
    max_tests_per_example: int = DEFAULT_MAX_TRAIN_TESTS,
) -> Dataset:
    ds = load_train_source(dataset_name, source_name)
    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))
    rows = []
    for row in ds:
        converted = convert_row(
            dict(row),
            source_name,
        )
        normalized_tests = json.loads(converted["reward_model"]["ground_truth"])
        prepared_tests = prepare_training_tests(
            normalized_tests, source_name, min_tests=min_tests_per_example, max_tests=max_tests_per_example
        )
        converted["reward_model"]["ground_truth"] = json.dumps(prepared_tests, ensure_ascii=False)
        converted["extra_info"] = {
            "split": "train",
            "source": source_name,
            "original_test_count": len(normalized_tests),
            "selected_test_count": len(prepared_tests),
            "test_selection": "longest_input_desc"
            if len(normalized_tests) > max_tests_per_example
            else "all",
        }
        rows.append(converted)
    return Dataset.from_list(rows)


def build_train_dataset(
    dataset_name: str = DATASET_NAME,
    *,
    max_examples_per_source: int | None = None,
    min_tests_per_example: int = DEFAULT_MIN_TRAIN_TESTS,
    max_tests_per_example: int = DEFAULT_MAX_TRAIN_TESTS,
    seed: int = 42,
) -> Dataset:
    datasets = [
        process_source(
            dataset_name,
            source_name,
            max_examples=max_examples_per_source,
            min_tests_per_example=min_tests_per_example,
            max_tests_per_example=max_tests_per_example,
        )
        for source_name in TRAIN_SOURCES
    ]
    ds = concatenate_datasets(datasets)
    return ds.shuffle(seed=seed)
