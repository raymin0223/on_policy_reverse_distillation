"""Rebuild the paper's 10,464-question holdout from public source row indices."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from .prepare import find_one, digest, family, scene, resolve_image, image_identity, record


def prepare(root):
    import pyarrow.parquet as pq
    from oprd.prepare import write_rows

    spec = json.loads(Path(__file__).with_name("holdout_indices.json").read_text())
    raw_path = find_one(root / "raw/rl", "vilasr_rl_data.jsonl")
    if digest(raw_path) != spec["raw_rl_sha256"]:
        raise ValueError("Pinned ViLaSR source changed")
    raw = [json.loads(line) for line in raw_path.read_text().splitlines()]
    train = pq.read_table(root / "prepared/train.parquet").to_pylist()
    val = pq.read_table(root / "prepared/holdout_val.parquet").to_pylist()
    train_ids = {r["extra_info"]["source_index"] for r in train}
    val_ids = {r["extra_info"]["source_index"] for r in val}
    selected = set(spec["additional_source_indices"]) | val_ids
    if len(train) != 19200 or len(val_ids) != 128 or selected & train_ids or len(selected) != 10464:
        raise ValueError("Training/validation membership differs from the paper split")
    candidates = [
        {
            "raw": raw[i],
            "index": i,
            "family": family(raw[i]),
            "paths": [resolve_image(root / "raw/rl", p, "rl") for p in raw[i]["image_path"]],
        }
        for i in sorted(selected)
    ]
    paths = sorted(
        {p for r in train for p in r["extra_info"]["images"]} | {p for r in candidates for p in r["paths"]}
    )
    with ThreadPoolExecutor(max_workers=16) as pool:
        hashes = dict(zip(paths, pool.map(image_identity, paths)))
    train_pixels = {hashes[p] for r in train for p in r["extra_info"]["images"]}
    train_scenes = {
        scene({"family": family(raw[r["extra_info"]["source_index"]]), "paths": r["extra_info"]["images"]})
        for r in train
    }
    seen = set()
    counts = Counter()
    rows = []
    for item in candidates:
        if scene(item) in train_scenes or train_pixels.intersection(hashes[p] for p in item["paths"]):
            raise ValueError("Evaluation overlaps a training image or scene")
        original = item["raw"]
        key = (
            tuple(hashes[p] for p in item["paths"]),
            original["question"],
            tuple(original.get("options") or []),
        )
        if key in seen:
            raise ValueError("Duplicate evaluation question/image")
        seen.add(key)
        counts[item["family"]] += 1
        row = record(original, item["paths"], "vilasr_holdout", item["index"])
        row["extra_info"]["family"] = item["family"]
        rows.append(row)
    if dict(counts) != spec["expected_counts"]:
        raise ValueError(f"Unexpected family counts: {counts}")
    write_rows(root / "prepared/holdout_full.parquet", rows)
    print(json.dumps({"questions": len(rows), "families": dict(counts)}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("datasets/agent"))
    prepare(p.parse_args().root.resolve())


if __name__ == "__main__":
    main()
