"""Four independent single-GPU evaluation workers and ordered result merging."""

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


def add_arguments(parser):
    parser.add_argument("--tensor-parallel-size", type=int, choices=(1,), default=1)
    parser.add_argument("--data-parallel-size", type=int, choices=(4,), default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--worker-rank", type=int, choices=range(4), help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)


def bounds(count, rank):
    """Contiguous, balanced shards retain the original global question indices."""
    return count * rank // 4, count * (rank + 1) // 4


def devices():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    result = [part.strip() for part in visible.split(",")]
    if len(result) != 4 or len(set(result)) != 4 or any(not d or d == "-1" for d in result):
        raise ValueError("Evaluation requires four distinct CUDA_VISIBLE_DEVICES (DP=4, TP=1)")
    return result


def validate_arguments(args, count):
    if count <= 0:
        raise ValueError("Empty evaluation")
    if args.batch_size <= 0 or (args.limit is not None and args.limit <= 0):
        raise ValueError("--batch-size and --limit must be positive")
    if not 0 < args.gpu_memory_utilization < 1:
        raise ValueError("--gpu-memory-utilization must be between zero and one")
    if (args.worker_rank is None) != (args.worker_output is None):
        raise ValueError("Worker rank and output must be supplied together")


def records(path):
    with Path(path).open() as stream:
        for line in stream:
            yield json.loads(line)


def merge_shards(shards, output, count, samples):
    """Validate every question/sample exactly once before exposing the result."""
    if len(shards) != 4:
        raise ValueError("Expected exactly four evaluation shards")
    merged = shards[0].parent / "merged.jsonl"
    with merged.open("x") as target:
        for rank, shard in enumerate(shards):
            rows = iter(records(shard))
            start, stop = bounds(count, rank)
            for problem in range(start, stop):
                for sample in range(samples):
                    row = next(rows, None)
                    if row is None or (row.get("problem_index"), row.get("sample_index")) != (
                        problem, sample
                    ):
                        raise ValueError(f"Missing, duplicate or misordered evaluation record in {shard}")
                    target.write(json.dumps(row, ensure_ascii=False) + "\n")
            if next(rows, None) is not None:
                raise ValueError(f"Unexpected extra evaluation record in {shard}")
    # Same filesystem: publish atomically and fail if another run created output.
    os.link(merged, output)


def run_workers(module, argv, output, count, samples):
    selected = devices()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.with_suffix(".summary.json").exists():
        raise FileExistsError(f"Evaluation already exists: {output}")
    work = output.with_name(output.name + ".workers")
    if work.exists():
        raise FileExistsError(f"Inspect the previous worker logs, then move/remove {work} before retrying")
    work.mkdir()
    print(f"Evaluating {count} questions with DP=4, TP=1; worker logs: {work}", flush=True)
    shards = [work / f"rank-{rank}.jsonl" for rank in range(4)]
    processes, logs = [], []
    try:
        for rank, device in enumerate(selected):
            log = (work / f"rank-{rank}.log").open("w")
            logs.append(log)
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": device}
            # A parent distributed launcher must not turn a worker into a multi-GPU job.
            for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
                env.pop(key, None)
            processes.append(subprocess.Popen(
                [sys.executable, "-m", module, *argv, "--worker-rank", str(rank),
                 "--worker-output", str(shards[rank])],
                env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            ))
        while True:
            states = [process.poll() for process in processes]
            for rank, state in enumerate(states):
                if state is not None and state != 0:
                    raise RuntimeError(f"Evaluation worker {rank} exited {state}; see {work}/rank-{rank}.log")
            if all(state == 0 for state in states):
                break
            time.sleep(0.2)
        merge_shards(shards, output, count, samples)
    except BaseException:
        # Terminate entire worker groups, including vLLM engine subprocesses.
        for process in processes:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 5
        for process in processes:
            try:
                process.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass
        for process in processes:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        print(f"Evaluation did not complete; worker logs and partial shards remain in {work}", file=sys.stderr)
        raise
    finally:
        for log in logs:
            log.close()
    shutil.rmtree(work)
