"""Pinned CPU-only downloads and safe, restartable extraction. Never starts training."""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import stat
import zipfile

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[2]
LOCK = Path(__file__).with_name("vilasr_assets.lock.json")


def safe_extract(archive, target, excluded_members=()):
    target = target.resolve()
    target.mkdir(parents=True, exist_ok=True)
    marker = target / f".{archive.name}.complete.json"
    if marker.exists():
        if sorted(json.loads(marker.read_text()).get("excluded_members", [])) != sorted(excluded_members):
            raise ValueError(f"Extraction exclusions changed for {archive.name}")
        return
    with zipfile.ZipFile(archive) as z:
        if not set(excluded_members).issubset(z.namelist()):
            raise ValueError("Approved excluded member is absent from the archive")
        for member in z.infolist():
            dest = (target / member.filename).resolve()
            if not dest.is_relative_to(target) or stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError(f"Unsafe ZIP member {member.filename}")
            if member.filename in excluded_members:
                print(f"Known corrupt sample excluded: {archive.name}/{member.filename}", flush=True)
                continue
            if member.is_dir():
                dest.mkdir(parents=True, exist_ok=True)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            # Parts may contain a shared metadata file. Never silently overwrite it.
            if dest.exists():
                import zlib

                crc = 0
                with dest.open("rb") as f:
                    for block in iter(lambda: f.read(4 << 20), b""):
                        crc = zlib.crc32(block, crc)
                if dest.stat().st_size != member.file_size or crc != member.CRC:
                    raise ValueError(f"Conflicting archive member {dest}")
                continue
            tmp = dest.with_name(dest.name + ".vilasr-extract-partial")
            # zipfile verifies the member CRC while reading to EOF.
            with z.open(member) as src, tmp.open("wb") as out:
                shutil.copyfileobj(src, out, length=4 << 20)
            tmp.replace(dest)
    marker.write_text(
        json.dumps(
            {
                "archive": str(archive),
                "bytes": archive.stat().st_size,
                "excluded_members": list(excluded_members),
            }
        )
        + "\n"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT / "datasets/agent")
    p.add_argument("--part", choices=["all", "models", "rl", "sft", "eval"], default="all")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--exclude-corrupt-sft-sample", action="store_true")
    a = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    lock = json.loads(LOCK.read_text())
    a.root.mkdir(parents=True, exist_ok=True)
    marker = a.root / "assets.lock.json"
    if marker.exists() and json.loads(marker.read_text()) != lock:
        raise ValueError("Destination contains different revisions")
    marker.write_text(json.dumps(lock, indent=2) + "\n")
    selected = ["models", "rl", "eval", "sft"] if a.part == "all" else [a.part]
    for part in selected:
        if part == "models":
            for name in ("student7b", "teacher3b_base"):
                item = lock[name]
                print(f"Downloading {name} at {item['revision']}", flush=True)
                snapshot_download(
                    item["repo"],
                    revision=item["revision"],
                    local_dir=a.root / "models" / name,
                    max_workers=a.workers,
                    allow_patterns=[
                        "*.json",
                        "*.jinja",
                        "*.txt",
                        "*.safetensors",
                        "README.md",
                        "trainer_log.jsonl",
                    ],
                    ignore_patterns=["training_args.bin"],
                )
            continue
        key = "eval" if part == "eval" else "data"
        item = lock[key]
        patterns = {
            "rl": ["rl_part*.zip", "dataset_info.json"],
            "sft": ["cold_start_part*.zip", "dataset_info.json"],
            "eval": ["*.json", "eval_data.zip"],
        }[part]
        dest = a.root / "downloads" / key
        print(f"Downloading {part} at {item['revision']}", flush=True)
        snapshot_download(
            item["repo"],
            repo_type="dataset",
            revision=item["revision"],
            local_dir=dest,
            allow_patterns=patterns,
            max_workers=a.workers,
        )
        pattern = {"rl": "rl_part*.zip", "sft": "cold_start_part*.zip", "eval": "eval_data.zip"}[part]
        target = a.root / {"rl": "raw/rl", "sft": "raw/cold_start", "eval": "raw/eval"}[part]
        for archive in sorted(
            dest.glob(pattern),
            key=lambda x: int(re.search(r"part(\d+)", x.name)[1]) if "part" in x.name else 0,
        ):
            print(f"Extracting {archive.name}", flush=True)
            exclusions = ()
            if part == "sft" and a.exclude_corrupt_sft_sample:
                from oprd.agent.sft_exclusions import archive_exclusions

                exclusions = archive_exclusions(archive, item["revision"])
            safe_extract(archive, target, exclusions)
    print("Requested assets downloaded/extracted. No GPU, Ray, W&B or training started.", flush=True)


if __name__ == "__main__":
    main()
