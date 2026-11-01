"""Download and verify the math source snapshots used by the paper recipes."""

import hashlib
import json
from pathlib import Path

LOCK = Path(__file__).with_name("math_sources.lock.json")


def source_spec(name):
    return json.loads(LOCK.read_text())["datasets"][name]


def read_verified_source(path, spec):
    path = Path(path)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != spec["sha256"]:
        raise ValueError(f"Math source checksum mismatch: {spec['repo']} / {spec['filename']}")
    if spec["filename"].endswith(".parquet"):
        import pyarrow.parquet as pq

        rows = pq.read_table(path).to_pylist()
    elif spec["filename"].endswith(".jsonl"):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    else:
        raise ValueError(f"Unsupported math source format: {spec['filename']}")
    if len(rows) != spec["rows"]:
        raise ValueError(f"Math source row count mismatch: {spec['repo']}")
    return rows


def load_source(name):
    from huggingface_hub import hf_hub_download

    spec = source_spec(name)
    path = hf_hub_download(
        repo_id=spec["repo"], repo_type="dataset",
        filename=spec["filename"], revision=spec["revision"],
    )
    return read_verified_source(path, spec)


def input_digest(rows):
    """Hash actual prompts/answers in order, independent of parquet metadata."""
    inputs = [
        {
            "prompt": [{"role": m["role"], "content": m["content"]} for m in row["prompt"]],
            "answer": str(row["reward_model"]["ground_truth"]),
        }
        for row in rows
    ]
    encoded = json.dumps(inputs, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def verify_prepared(name, rows):
    spec = source_spec(name)
    if len(rows) != spec["rows"] or input_digest(rows) != spec["prepared_inputs_sha256"]:
        raise ValueError(f"Prepared {name} prompts/answers differ from the verified paper inputs")
