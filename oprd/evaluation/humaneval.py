"""Score every generated sample with both official HumanEval+ test suites."""

import json
from pathlib import Path
import subprocess
from verl.utils.reward_score.deepcoder_rlvr import extract_single_python_code_block


def score_file(responses, rows, snapshot, python):
    records = [json.loads(line) for line in responses.read_text().splitlines()]
    expected = {(i, j) for i in range(len(rows)) for j in range(4)}
    if (
        len(records) != len(expected)
        or {(r["problem_index"], r["sample_index"]) for r in records} != expected
    ):
        raise ValueError("HumanEval+ requires four responses for every selected question")
    samples = responses.with_suffix(".evalplus.jsonl")
    formats = []
    with samples.open("x") as stream:
        for record in records:
            parsed = extract_single_python_code_block(record["response"])
            formats.append(parsed.code is not None)
            stream.write(
                json.dumps(
                    {
                        "task_id": rows[record["problem_index"]]["extra_info"]["task_id"],
                        "solution": parsed.code or "raise RuntimeError('FORMAT_ERROR')",
                    }
                )
                + "\n"
            )
    subprocess.run(
        [
            # Keep the venv symlink: resolving it selects the system interpreter.
            str(Path(python).expanduser().absolute()),
            str(Path(__file__).with_name("evalplus_official.py")),
            "--snapshot",
            str(snapshot),
            "--samples",
            str(samples),
        ],
        check=True,
    )
    results = json.loads(samples.with_name(samples.stem + "_eval_results.json").read_text())["eval"]
    scores = []
    for record, format_ok in zip(records, formats, strict=True):
        task = rows[record["problem_index"]]["extra_info"]["task_id"]
        result = results[task][record["sample_index"]]
        scores.append(
            float(format_ok and result["base_status"] == "pass" and result["plus_status"] == "pass")
        )
    responses.with_suffix(".scores.json").write_text(json.dumps(scores) + "\n")
    return scores
