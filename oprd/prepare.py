"""Prepare the datasets used in the paper; generated data stays outside version control."""

import argparse
import json
from pathlib import Path
from .paper import REASONING_TASKS, ADDITIONAL_TASKS


def write_rows(path, rows):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")


def reasoning(task, root, count=20000, test_size=200, seed=0):
    from data import reasoning_gym_tasks as gym
    from data.quantum_lock import generate_unique_quantum_lock_problems
    from data.string_manipulation import generate_unique_string_manipulation_problems

    generators = {
        name: getattr(gym, "generate_" + name + "_problems")
        for name in REASONING_TASKS + ADDITIONAL_TASKS
        if name not in {"quantum_lock", "string_manipulation"}
    }
    generators.update(
        quantum_lock=generate_unique_quantum_lock_problems,
        string_manipulation=generate_unique_string_manipulation_problems,
    )
    generated = generators[task](num_problems=count, seed=seed)
    problems = generated[0] if isinstance(generated, tuple) else generated
    if not 0 < test_size < len(problems):
        raise ValueError("Require disjoint, nonempty training and test sets")
    train, test = gym.split_train_test(problems, test_size)
    if task == "binary_matrix":
        train, test, _ = gym.dedup_binary_matrix_train_against_test(problems, test_size, seed)
    for split, subset in [("train", train), ("test", test)]:
        rows = []
        for i, problem in enumerate(subset):
            raw = gym.make_reasoning_row(task, i, problem, split)
            rows.append(
                {
                    "data_source": task,
                    "prompt": [{"role": "user", "content": raw["prompt"]}],
                    "ability": "reasoning_gym",
                    "reward_model": {"style": "rule", "ground_truth": raw["answer"]},
                    "extra_info": {"index": str(i), "split": split, "problem": raw["prompt"]},
                }
            )
        write_rows(root / task / (split + ".parquet"), rows)
    (root / task / "sample_config.json").write_text(
        json.dumps({"seed": seed, "train_size": count - test_size, "test_size": test_size}, indent=2) + "\n"
    )


def mathematics(root):
    from datasets import load_dataset
    from data.format.math import SYSTEM_PROMPT
    from data.prepare_dapo_math import _process_row
    from data.prepare_olympiadbench import HF_REPO, HF_CONFIG, HF_REVISION, EXPECTED_ROWS, _convert
    from data.paper_math import LOCK, load_source, source_spec, verify_prepared

    train = [_process_row(row, "dapo_math") for row in load_source("dapo")]
    verify_prepared("dapo", train)
    write_rows(root / "math/train.parquet", train)
    for name, answer_key in [
        ("aime24", "solution"),
        ("aime25", "answer"),
        ("hmmt25", "answer"),
    ]:
        dataset = load_source(name)
        split = source_spec(name)["split"]
        rows = []
        for i, row in enumerate(dataset):
            answer = str(row[answer_key])
            if name == "aime24":
                from data.utils.math import extract_boxed_answer

                answer = extract_boxed_answer(answer) or answer
            question = str(row["problem"])
            rows.append(
                {
                    "data_source": name,
                    "prompt": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": question
                            + "\nPlease reason step by step, and put your final answer within \\boxed{}.",
                        },
                    ],
                    "ability": "math",
                    "reward_model": {"style": "rule", "ground_truth": answer},
                    "extra_info": {"index": str(i), "problem": question, "split": split},
                }
            )
        verify_prepared(name, rows)
        write_rows(root / name / "test.parquet", rows)
    dataset = load_dataset(HF_REPO, HF_CONFIG, split="train", revision=HF_REVISION)
    if len(dataset) != EXPECTED_ROWS:
        raise ValueError("OlympiadBench revision/count mismatch")
    write_rows(root / "olympiadbench/test.parquet", [_convert(row, i) for i, row in enumerate(dataset)])
    (root / "math/sources.lock.json").write_text(LOCK.read_text())


def code(root):
    from data.prepare_deepcoder_rlvr import build_train_dataset
    from data.prepare_code_rlvr_eval import (
        build_deepcoder_lcbv5,
    )

    train = build_train_dataset()
    if len(train) != 24287:
        raise ValueError(f"Expected 24287 DeepCoder training rows, got {len(train)}")
    (root / "code").mkdir(parents=True, exist_ok=True)
    train.to_parquet(str(root / "code/train.parquet"))
    lcb = build_deepcoder_lcbv5()
    write_rows(root / "lcbv5/test.parquet", list(lcb))
    write_rows(root / "code/test.parquet", list(lcb))
    from data.paper_code import prepare_benchmarks

    prepare_benchmarks(root)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--task", choices=("math", "code", "reasoning") + REASONING_TASKS + ADDITIONAL_TASKS, required=True
    )
    p.add_argument("--output-root", type=Path, default=Path("datasets"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--count", type=int, default=20000)
    p.add_argument("--test-size", type=int, default=200)
    a = p.parse_args()
    if a.task == "math":
        mathematics(a.output_root)
    elif a.task == "code":
        code(a.output_root)
    else:
        for task in REASONING_TASKS + ADDITIONAL_TASKS if a.task == "reasoning" else (a.task,):
            reasoning(task, a.output_root, a.count, a.test_size, a.seed)


if __name__ == "__main__":
    main()
