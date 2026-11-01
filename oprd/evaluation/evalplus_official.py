"""CPU entry point for the official EvalPlus commit containing the upstream #241 fix."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

EVALPLUS_VERSION = "0.4.0.dev13"
EVALPLUS_COMMIT = "6eb1e199c2e370518e7bf3e8eee6322c2b23c89c"


def runtime_info():
    dist = importlib.metadata.distribution("evalplus")
    origin = json.loads(dist.read_text("direct_url.json") or "{}")
    if dist.version != EVALPLUS_VERSION or origin.get("vcs_info", {}).get("commit_id") != EVALPLUS_COMMIT:
        raise ValueError(
            f"Requires official EvalPlus commit {EVALPLUS_COMMIT}; see requirements/evalplus.txt"
        )
    root = Path(dist.locate_file("evalplus"))
    hashes = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*.py"))
    }
    return dict(
        evalplus_version=dist.version,
        evalplus_commit=EVALPLUS_COMMIT,
        numpy_version=importlib.metadata.version("numpy"),
        psutil_version=importlib.metadata.version("psutil"),
        source_tree_sha256=hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--samples", type=Path)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--describe", action="store_true")
    args = parser.parse_args()
    runtime = runtime_info()
    if args.describe:
        print(json.dumps(runtime))
        return
    if not args.snapshot or not args.samples:
        parser.error("--snapshot and --samples are required")
    if args.parallel < 1:
        parser.error("--parallel must be positive")
    if args.samples.with_name(args.samples.stem + "_eval_results.json").exists():
        raise ValueError("Refusing cached EvalPlus results")
    os.environ["HUMANEVAL_OVERRIDE_PATH"] = str(args.snapshot.resolve())
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["EVALPLUS_MAX_MEMORY_BYTES"] = str(4 * 1024**3)
    # Use EvalPlus's default group timeout (60 seconds), never an inherited override.
    os.environ.pop("EVALPLUS_TIMEOUT_PER_TASK", None)
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ["XDG_CACHE_HOME"] = str(Path(__file__).resolve().parents[2] / ".cache/evalplus-6eb1e19")
    # The override is read at import time, so these imports must remain here.
    from evalplus.evaluate import evaluate

    args.samples.with_name("evalplus_runtime.json").write_text(json.dumps(runtime, indent=2))
    evaluate(
        dataset="humaneval",
        samples=str(args.samples.resolve()),
        parallel=args.parallel,
        min_time_limit=4,
        gt_time_limit_factor=4,
        mini=False,
        noextreme=False,
        base_only=False,
    )


if __name__ == "__main__":
    main()
