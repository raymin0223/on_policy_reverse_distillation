"""Pinned official APPS scorer and bounded CPU subprocess configuration."""

import hashlib
import json
import os
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
CPU_PYTHON = ROOT / ".venv-apps/bin/python"
OFFICIAL_APPS = ROOT / ".cache/apps-official/testing_util.py"
MEMORY = 4 * 1024**3


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def prepare_official():
    lock = json.loads(Path(__file__).with_name("apps.lock.json").read_text())
    if not OFFICIAL_APPS.exists():
        url = f"https://raw.githubusercontent.com/hendrycks/apps/{lock['commit']}/eval/testing_util.py"
        with urlopen(url, timeout=120) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != lock["sha256"]:
            raise ValueError("Official APPS evaluator checksum mismatch")
        OFFICIAL_APPS.parent.mkdir(parents=True, exist_ok=True)
        OFFICIAL_APPS.write_bytes(content)
    if sha256(OFFICIAL_APPS) != lock["sha256"]:
        raise ValueError("Official APPS evaluator changed")


def cpu_env():
    env = {k: os.environ[k] for k in ("PATH", "LANG", "LC_ALL", "TMPDIR") if k in os.environ}
    env.update(
        PYTHONPATH=str(ROOT),
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        MPLBACKEND="Agg",
        PYTHONHASHSEED="42",
    )
    return env
