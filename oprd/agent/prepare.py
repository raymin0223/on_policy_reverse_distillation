"""CPU-only compilation of the pinned ViLaSR release. Never starts training.

Balanced RL batches are an OPRD experimental design, not the original paper's
sampler. External benchmarks are never used to select difficulty or reward.
"""

import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import random

from PIL import Image
from oprd.agent.vilasr_contract import MAX_PIXELS, MIN_PIXELS, VERSION, answer_score, prompt_constants

ROOT = Path(__file__).resolve().parents[2]
FAMILIES = ("maze", "GPT4Scene", "SR-91k", "VQA")
EXTERNAL = {
    "spatialeval": "SpatialEval_spatialreal.json",
    "vsibench": "vsi_bench.json",
    "sparbench": "spar_bench.json",
    "mmsibench": "mmsi_bench.json",
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def find_one(root, name):
    found = list(root.rglob(name))
    if len(found) != 1:
        raise ValueError(
            f"Expected exactly one {name} under {root}, got {len(found)}. Finish extraction first."
        )
    return found[0]


def resolve_image(root, name, prefix=None):
    name = str(name)
    if ".." in Path(name).parts:
        raise ValueError(f"Unsafe image path {name}")
    relative = name.removeprefix("./")
    if Path(relative).is_absolute():
        if not prefix or f"/{prefix}/" not in relative:
            raise ValueError(f"Unrecognized original-machine path {name}")
        relative = prefix + "/" + relative.split(f"/{prefix}/", 1)[1]
    candidates = [root / relative]
    if prefix:
        candidates += [root / prefix / relative, root / relative.removeprefix(prefix + "/")]
    valid = {p.resolve() for p in candidates if p.is_file() and p.resolve().is_relative_to(root.resolve())}
    if len(valid) != 1:
        raise ValueError(f"Image missing or ambiguous: {name} under {root}")
    return str(valid.pop())


@lru_cache(maxsize=100000)
def image_identity(path):
    # Decoded pixels, not filenames/encoded JPEG bytes, define exact image reuse.
    with Image.open(path) as image:
        image = image.convert("RGB")
        h = hashlib.sha256(f"{image.size}".encode())
        h.update(image.tobytes())
        return h.hexdigest()


@lru_cache(maxsize=100000)
def displayed_size(path):
    from qwen_vl_utils.vision_process import smart_resize

    with Image.open(path) as image:
        h, w = smart_resize(
            image.height, image.width, factor=28, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
        )
    return w, h


def family(row):
    source = row.get("source", row.get("data_type"))
    return source if source in FAMILIES[:3] else "VQA"


def scene(row):
    # One scene's video frames must not occur on both sides of the RL holdout.
    if row["family"] in ("GPT4Scene", "SR-91k") or len(row["paths"]) > 1:
        return row["family"], str(Path(row["paths"][0]).parent)
    return row["family"], image_identity(row["paths"][0])


def prompt(row, paths):
    constants = prompt_constants()
    kind = row.get("question_type", row.get("problem_type"))
    question = row["question"]
    if row.get("original_question_type") in ("position_matching", "camera_motion_infer"):
        question += "\nThe values represent the bounding box coordinates normalized to a 0-1000 scale, with the top-left corner as the origin of the image."
    if kind == "multiple choice" and row.get("options"):
        question += "\n" + "\n".join(str(x).strip() for x in row["options"])
    suffix = constants["TYPE_TEMPLATE"].get(kind, "" if kind == "vci" else None)
    if suffix is None:
        raise ValueError(f"Unverifiable/unsupported question type: {kind}")
    if len(paths) > 1 and row.get("data_type") != "multi_view":
        question = (
            f"These are frames from a video, numbered from 1 to {len(paths)} in sequence.\n\n"
            "Answer the question with appropriate tools:\n" + question + suffix
        )
    elif len(paths) == 1:
        question += "\nThe index of the given image is 1." + suffix
    else:
        question += suffix
    # No answer, thought, solution, or other metadata is interpolated here.
    content = ""
    for i, path in enumerate(paths, 1):
        width, height = displayed_size(path)
        content += f"<image>The index of the given image is {i} (width: {width}, height: {height}).\n"
    content += constants["PROMPT_TEMPLATE"].format(question=question)
    width, height = displayed_size(paths[0])
    return [
        {"role": "system", "content": constants["SYSTEM_PROMPT"].format(width=width, height=height)},
        {"role": "user", "content": content},
    ]


def record(row, paths, source, index, external=False):
    kind = row.get("question_type", row.get("problem_type"))
    gold = {
        "type": kind,
        "answer": str(row["answer"]),
        "numeric_metric": "exact2" if external and kind == "numerical" else "mra",
    }
    if answer_score(gold["answer"], gold) != 1.0:
        raise ValueError(f"Gold fails its own verifier: {source}/{index}")
    return {
        "prompt": prompt(row, paths),
        "images": [{"image": p, "max_pixels": MAX_PIXELS, "min_pixels": MIN_PIXELS} for p in paths],
        "data_source": source,
        "agent_name": "vilasr_agent",
        "ability": "spatial_reasoning",
        "reward_model": {"style": "rule", "ground_truth": gold},
        "extra_info": {
            "index": index,
            "qid": f"{source}:{row.get('question_id', index)}:{index}",
            "source_index": index,
            "images": paths,
            "original_question_type": str(row.get("original_question_type", kind)),
            "source": str(row.get("source", row.get("data_type", source))),
        },
    }


def balanced(rows, steps=300, per_family=16, seed=42):
    rng = random.Random(seed)
    pools = {name: [r for r in rows if r["family"] == name] for name in FAMILIES}
    for name, pool in pools.items():
        if len(pool) < steps * per_family:
            raise ValueError(
                f"Only {len(pool)} eligible {name} rows; need {steps * per_family}, no silent repeat"
            )
        rng.shuffle(pool)
    output = []
    for step in range(steps):
        batch = [row for name in FAMILIES for row in pools[name][step * per_family : (step + 1) * per_family]]
        rng.shuffle(batch)
        output.extend(batch)
    return output


def prepare_rl(root):
    import pyarrow as pa
    import pyarrow.parquet as pq

    output = root / "prepared"
    if (output / "manifest.json").exists():
        raise ValueError("Prepared manifest already exists; use a different root to regenerate")
    external_hashes, external_records, stats = set(), {}, {}
    for name, filename in EXTERNAL.items():
        rows = json.loads((root / "downloads/eval" / filename).read_text())
        converted = []
        for i, row in enumerate(rows):
            paths = [resolve_image(root / "raw/eval", p) for p in row["image_path"]]
            external_hashes.update(image_identity(p) for p in paths)
            converted.append(record(row, paths, name, i, external=True))
        external_records[name] = converted
        stats[name] = {
            "count": len(rows),
            "types": dict(Counter(x["reward_model"]["ground_truth"]["type"] for x in converted)),
        }
    raw = find_one(root / "raw/rl", "vilasr_rl_data.jsonl")
    candidates, excluded, identities = [], Counter(), set()
    for i, line in enumerate(raw.read_text().splitlines()):
        row = json.loads(line)
        paths = [resolve_image(root / "raw/rl", p, "rl") for p in row["image_path"]]
        hashes = tuple(image_identity(p) for p in paths)
        key = (hashes, row["question"], tuple(row.get("options") or []))
        if key in identities:
            excluded["duplicate_question_image"] += 1
            continue
        identities.add(key)
        if external_hashes.intersection(hashes):
            excluded["external_exact_image_overlap"] += 1
            continue
        candidates.append({"family": family(row), "paths": paths, "raw": row, "index": i})
    # Hold out entire scenes/images, then choose 32 questions per family.
    rng, held_scenes, holdout = random.Random(314159), set(), []
    for name in FAMILIES:
        pool = [r for r in candidates if r["family"] == name]
        rng.shuffle(pool)
        for row in pool:
            key = scene(row)
            if key in held_scenes:
                continue
            held_scenes.add(key)
            holdout.append(row)
            if sum(r["family"] == name for r in holdout) == 32:
                break
        if sum(r["family"] == name for r in holdout) != 32:
            raise ValueError(f"Insufficient distinct holdout scenes for {name}")
    validation_images = {image_identity(p) for row in holdout for p in row["paths"]}
    eligible = [
        r
        for r in candidates
        if scene(r) not in held_scenes
        and not validation_images.intersection(image_identity(p) for p in r["paths"])
    ]
    selected = balanced(eligible)
    tables = {
        "train": [record(r["raw"], r["paths"], "vilasr_" + r["family"], r["index"]) for r in selected],
        "holdout_val": [record(r["raw"], r["paths"], "vilasr_holdout", r["index"]) for r in holdout],
    }
    output.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, rows in tables.items():
        path = output / f"{name}.parquet"
        temp = path.with_suffix(".parquet.tmp")
        pq.write_table(pa.Table.from_pylist(rows), temp, compression="zstd")
        temp.replace(path)
        files[name] = {"count": len(rows), "sha256": digest(path)}
    dump(
        output / "manifest.json",
        {
            "ready": True,
            "version": VERSION,
            "lock": json.loads((root / "assets.lock.json").read_text()),
            "source_sha256": digest(raw),
            "raw_counts": dict(Counter(family(json.loads(x)) for x in raw.read_text().splitlines())),
            "excluded": dict(excluded),
            "eligible_counts": dict(Counter(r["family"] for r in eligible)),
            "steps": 300,
            "batch": 64,
            "rollouts_per_prompt": 8,
            "per_family_per_batch": 16,
            "seed": 42,
            "files": files,
            "external": stats,
            "rl_external_exact_pixel_overlap_removed": True,
            "sft_overlap_audited": False,
            "caveat": "RL holdout is disjoint from RL training images/scenes, not certified unseen during public 7B SFT. Near-duplicate/semantic overlap is not excluded.",
        },
    )


def prepare_sft(root, exclude_corrupt_sft_sample=False):
    output = root / "prepared_sft"
    if (output / "manifest.json").exists():
        raise ValueError("SFT preparation already exists")
    dataset_info, files, excluded = {}, {}, []
    if exclude_corrupt_sft_sample:
        from oprd.agent.sft_exclusions import REVISION

        if json.loads((root / "assets.lock.json").read_text())["data"]["revision"] != REVISION:
            raise ValueError("Approved exclusion does not apply to this data revision")
    for name in ("vqa", "maze", "GPT4Scene", "SR_91k"):
        path = find_one(root / "raw/cold_start", f"{name}.json")
        rows = json.loads(path.read_text())
        source_count = len(rows)
        if exclude_corrupt_sft_sample:
            from oprd.agent.sft_exclusions import filter_approved_row

            rows, removed = filter_approved_row(name, rows)
            excluded.extend(removed)
        for row in rows:
            row["images"] = [resolve_image(root / "raw/cold_start", p, "cold_start") for p in row["images"]]
            if sum(x["content"].count("<image>") for x in row["conversations"]) != len(row["images"]):
                raise ValueError(f"SFT image placeholders disagree in {name}/{row.get('question_id')}")
            if any(x["role"] not in ("system", "user", "assistant") for x in row["conversations"]):
                raise ValueError("Unexpected SFT role")
            turns = (
                row["conversations"][1:]
                if row["conversations"][0]["role"] == "system"
                else row["conversations"]
            )
            if len(turns) % 2 or any(
                m["role"] != ("user" if i % 2 == 0 else "assistant") for i, m in enumerate(turns)
            ):
                raise ValueError("SFT conversations must alternate user/assistant and end with an assistant")
        target = output / f"{name}.json"
        dump(target, rows)
        files[name] = {
            "count": len(rows),
            "source_count": source_count,
            "sha256": digest(target),
            "original_sha256": digest(path),
        }
        dataset_info[name + "_cold_start"] = {
            "file_name": target.name,
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "images": "images"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        }
    dump(output / "dataset_info.json", dataset_info)
    if len(excluded) != int(exclude_corrupt_sft_sample):
        raise ValueError("SFT exclusion count differs from explicit authorization")
    dump(output / "excluded_trajectories.json", excluded)
    dump(
        output / "manifest.json",
        {
            "ready": True,
            "files": files,
            "excluded_trajectories": excluded,
            "transformation": "Absolute local image paths; retained conversations unchanged. Only explicitly listed corrupt trajectory excluded.",
            "length_audit_complete": False,
            "external_overlap_audited": False,
        },
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT / "datasets/agent")
    p.add_argument("--part", choices=("rl", "sft", "all"), default="all")
    p.add_argument("--exclude-corrupt-sft-sample", action="store_true")
    a = p.parse_args()
    if a.part in ("rl", "all"):
        prepare_rl(a.root)
    if a.part in ("sft", "all"):
        prepare_sft(a.root, exclude_corrupt_sft_sample=a.exclude_corrupt_sft_sample)


if __name__ == "__main__":
    main()
