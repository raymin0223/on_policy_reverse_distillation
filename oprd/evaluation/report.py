"""Aggregate complete checkpoint evaluations with equal benchmark weights."""

import argparse
import json
from pathlib import Path


def aggregate(records, checkpoints, benchmarks):
    checkpoints, benchmarks = list(checkpoints), list(benchmarks)
    if not checkpoints or not benchmarks:
        raise ValueError("Select at least one checkpoint and one benchmark")
    if len(set(checkpoints)) != len(checkpoints) or len(set(benchmarks)) != len(benchmarks):
        raise ValueError("Checkpoint and benchmark selections must be unique")
    cells = {}
    for record in records:
        key = (int(record["step"]), str(record["benchmark"]))
        if key in cells:
            raise ValueError(f"Duplicate result {key}")
        value = float(record["mean_score"])
        if not 0 <= value <= 100:
            raise ValueError("Scores must be percentages in [0,100]")
        cells[key] = value
    expected = {(step, benchmark) for step in checkpoints for benchmark in benchmarks}
    if set(cells) != expected:
        raise ValueError(f"Incomplete or unexpected cells: {sorted(expected ^ set(cells))}")
    per_benchmark = {b: sum(cells[s, b] for s in checkpoints) / len(checkpoints) for b in benchmarks}
    return {
        "benchmarks": per_benchmark,
        "average": sum(per_benchmark.values()) / len(benchmarks),
        "checkpoints": list(checkpoints),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results", type=Path, required=True, help="JSON list of step, benchmark, mean_score records"
    )
    p.add_argument("--steps", type=int, nargs="+", default=[30, 60, 90, 120, 150])
    p.add_argument("--benchmarks", nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    out = aggregate(json.loads(a.results.read_text()), a.steps, a.benchmarks)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
