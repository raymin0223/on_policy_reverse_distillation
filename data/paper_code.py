"""Pinned evaluation subsets, local checker metadata, and the EvalPlus snapshot."""

import gzip
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


def prepare_benchmarks(root):
    from datasets import load_dataset
    from oprd.prepare import write_rows
    from data.prepare_code_rlvr_eval import _make_row, _codecontests_tests
    from data.prepare_deepcoder_rlvr import normalize_tests
    from data.code_rlvr_prompts import build_messages

    manifest = json.loads(Path(__file__).with_name("evaluation_manifest.json").read_text())
    dataset = load_dataset("deepmind/code_contests", split="test", revision=manifest["codecontests_revision"])
    rows = []
    for item in manifest["codecontests"]:
        raw = dataset[item["source_index"]]
        row = _make_row(
            data_source="codecontests_official_test",
            problem=raw["description"],
            tests=_codecontests_tests(raw),
            index=item["uid"].split(":", 1)[1],
            source_split="test",
        )
        tests = json.loads(row["reward_model"]["ground_truth"])
        if len(tests) != item["test_count"]:
            raise ValueError("CodeContests test count changed")
        for test in tests:
            test["metadata"].update(
                timeout_seconds=item["timeout_seconds"],
                memory_limit_bytes=item["memory_limit_bytes"],
                local_code_eval=True,
                output_checker=item["checker"],
            )
        row["reward_model"]["ground_truth"] = json.dumps(tests, ensure_ascii=False)
        rows.append(row)
    write_rows(root / "codecontests/test.parquet", rows)
    dataset = load_dataset("codeparrot/apps", split="test", revision=manifest["apps_revision"])
    selected = {int(x["task_id"]) for x in manifest["apps_task_ids"]}
    rows = []
    for raw in dataset:
        if int(raw["problem_id"]) not in selected:
            continue
        input_output = json.loads(raw["input_output"])
        tests = normalize_tests({"tests": input_output}, "taco")
        prepared = _make_row(
            data_source="apps_competition",
            problem=raw["question"],
            tests=tests,
            starter_code=raw.get("starter_code") or "",
            index=str(raw["problem_id"]),
            source_split="competition/test",
        )
        prepared["reward_model"]["ground_truth"] = json.dumps(input_output)
        rows.append(prepared)
    if len(rows) != len(selected):
        raise ValueError("APPS subset changed")
    write_rows(root / "apps_competition/test.parquet", rows)
    from oprd.code.common import prepare_official

    prepare_official()
    human = manifest["humanevalplus"]
    with urlopen(human["source_url"], timeout=120) as response:
        all_rows = [json.loads(line) for line in gzip.decompress(response.read()).decode().splitlines()]
    selected = set(human["task_ids"])
    snapshot = [row for row in all_rows if row["task_id"] in selected]
    if len(snapshot) != human["included_count"]:
        raise ValueError("HumanEval+ subset changed")
    directory = root / "humanevalplus"
    directory.mkdir(parents=True, exist_ok=True)
    # Explicit snapshot override prevents a later installed EvalPlus release changing the tests.
    (directory / "snapshot.jsonl").write_text("".join(json.dumps(row) + "\n" for row in snapshot))
    rows = []
    for raw in snapshot:
        rows.append(
            {
                "data_source": "humanevalplus",
                "prompt": [{"role": "user", "content": human_prompt(raw["prompt"])}],
                "reward_model": {"style": "rule", "ground_truth": raw["task_id"]},
                "extra_info": {"task_id": raw["task_id"], "index": raw["task_id"]},
                "ability": "code",
            }
        )
    write_rows(directory / "test.parquet", rows)


def human_prompt(specification):
    return (
        "You will be given a Python function specification. Implement the function correctly.\n"
        "Preserve the original function name, arguments, and return-value contract.\n"
        "Please reason step by step in plain prose, including the algorithm and time/space complexity.\n"
        "Then output the complete implementation, including the function declaration, necessary imports,\n"
        "and helper functions, inside exactly one fenced Python code block as the last thing in your response.\n"
        "Do not convert the function into a stdin/stdout program.\n\n### Function specification:\n"
        + specification
        + "\n### Answer:\n"
    )
