"""Build/audit the 3B cold-start SFT config without loading model weights.

Original ViLaSR recipe: full language-model SFT, frozen vision/projector,
3 epochs, lr=1e-5, cosine, warmup=.1, cutoff=18432, global batch=32.
Our four-GPU adaptation uses accumulation=8 and ZeRO-2. SDPA is an explicitly
conservative initial choice, not a measured throughput optimum.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess

import yaml
from oprd.agent.prepare import ROOT, dump, digest
from oprd.runtime import tracking


def configuration(root):
    return {
        "model_name_or_path": str(root / "models/teacher3b_base"),
        "trust_remote_code": False,
        "image_max_pixels": 1003520,
        "video_max_pixels": 1003520,
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "full",
        "freeze_vision_tower": True,
        "freeze_multi_modal_projector": True,
        "freeze_language_model": False,
        "flash_attn": "sdpa",
        "deepspeed": str(root / "prepared_sft/ds_zero2.json"),
        "dataset": "vqa_cold_start,maze_cold_start,GPT4Scene_cold_start,SR_91k_cold_start",
        "dataset_dir": str(root / "prepared_sft"),
        "template": "qwen2_vl",
        "cutoff_len": 18432,
        "packing": False,
        "train_on_prompt": False,
        "mask_history": False,
        "overwrite_cache": False,
        "preprocessing_num_workers": 8,
        "dataloader_num_workers": 2,
        "output_dir": str(root / "models/teacher3b_sft"),
        "overwrite_output_dir": False,
        "logging_steps": 10,
        "logging_first_step": True,
        "save_steps": 200,
        "save_total_limit": 2,
        "save_only_model": False,
        "report_to": "wandb",
        "run_name": "agent-teacher-sft",
        "plot_loss": False,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 1e-5,
        "num_train_epochs": 3.0,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "bf16": True,
        "gradient_checkpointing": True,
        "ddp_timeout": 18000,
        "seed": 42,
        "data_seed": 42,
        "push_to_hub": False,
    }


def audit(root):
    import math
    from transformers import AutoProcessor
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
    from llamafactory.data.template import get_template_and_fix_tokenizer
    from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
    from llamafactory.hparams import DataArguments
    from PIL import Image
    from functools import lru_cache

    manifest_path = root / "prepared_sft/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    processor = AutoProcessor.from_pretrained(
        root / "models/teacher3b_base", local_files_only=True, trust_remote_code=False
    )
    processor.image_max_pixels, processor.image_min_pixels = 1003520, 1024
    args = DataArguments(template="qwen2_vl", cutoff_len=10**9)
    template = get_template_and_fix_tokenizer(processor.tokenizer, args)
    encoder = SupervisedDatasetProcessor(template, processor.tokenizer, processor, args)
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")

    @lru_cache(maxsize=300000)
    def tokens(path):
        with Image.open(path) as im:
            w, h = im.size
        # LLaMA-Factory regularizes area before Qwen's patch-grid resize.
        for bound, predicate in ((1003520, w * h > 1003520), (1024, w * h < 1024)):
            if predicate:
                factor = math.sqrt(bound / (w * h))
                w, h = int(w * factor), int(h * factor)
        w, h = max(w, 28), max(h, 28)
        if w / h > 200:
            w = h * 180
        if h / w > 200:
            h = w * 180
        ip = processor.image_processor
        h, w = smart_resize(
            h, w, factor=ip.patch_size * ip.merge_size, min_pixels=ip.min_pixels, max_pixels=ip.max_pixels
        )
        return h // 28 * (w // 28)

    result = {
        "manifest_sha256": digest(manifest_path),
        "model_config_sha256": digest(root / "models/teacher3b_base/config.json"),
        "datasets": {},
        "ready": True,
        "cpu_only": True,
    }
    for name, item in manifest["files"].items():
        path = root / "prepared_sft" / f"{name}.json"
        if digest(path) != item["sha256"]:
            raise ValueError("Changed SFT data")
        rows = json.loads(path.read_text())
        maximum, overlong = 0, 0
        for row_index, row in enumerate(rows):
            messages = [
                {
                    **m,
                    "content": m["content"].replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>"),
                }
                for m in row["conversations"]
            ]
            system = messages.pop(0)["content"] if messages[0]["role"] == "system" else ""
            pairs = template.encode_multiturn(processor.tokenizer, messages, system=system, tools="")
            ids = [token for pair in pairs for part in pair for token in part]
            if ids.count(image_token_id) != len(row["images"]):
                raise ValueError("SFT image tokens and images disagree")
            length = len(ids) + sum(tokens(p) - 1 for p in row["images"])
            if row_index < 2:
                turns = (
                    row["conversations"][1:]
                    if row["conversations"][0]["role"] == "system"
                    else row["conversations"]
                )
                actual, labels = encoder._encode_data_example(
                    turns[:-1], turns[-1:], system, "", row["images"], [], []
                )
                if len(actual) != length or not any(x != -100 for x in labels):
                    raise ValueError(
                        "Fast SFT length/mask check differs from the actual LLaMA-Factory encoder"
                    )
            maximum = max(maximum, length)
            overlong += length > 18432
            if (row_index + 1) % 2000 == 0 or row_index + 1 == len(rows):
                print(
                    f"SFT length audit {name}: {row_index + 1}/{len(rows)}, "
                    f"max_tokens={maximum}, overlong={overlong}",
                    flush=True,
                )
        result["datasets"][name] = {
            "count": len(rows),
            "max_tokens": maximum,
            "overlong": overlong,
            "actual_encoder_parity_samples": min(2, len(rows)),
        }
        result["ready"] &= overlong == 0
    dump(root / "prepared_sft/length_audit.json", result)
    if not result["ready"]:
        raise ValueError(
            "Overlong SFT trajectories detected. Inspect length_audit.json before choosing truncation/filtering; no training started."
        )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    tracking.add_arguments(p)
    p.add_argument("--root", type=Path, default=ROOT / "datasets/agent")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mode", choices=("config", "prepare", "preflight", "train"), default="preflight")
    a = p.parse_args()
    tracking.configure_environment(a)
    config = configuration(a.root)
    config["seed"] = config["data_seed"] = a.seed
    os.environ.setdefault("WANDB_DIR", config["output_dir"])
    if a.wandb_mode == "disabled":
        config["report_to"] = "none"
    if a.mode == "config":
        print(yaml.safe_dump(config, sort_keys=False))
        return
    if a.mode == "prepare":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        audit(a.root)
    ready = json.loads((a.root / "prepared_sft/length_audit.json").read_text())
    if not ready["ready"] or ready["manifest_sha256"] != digest(a.root / "prepared_sft/manifest.json"):
        raise ValueError("SFT input audit missing/outdated")
    manifest = json.loads((a.root / "prepared_sft/manifest.json").read_text())
    for name, value in manifest["files"].items():
        if digest(a.root / "prepared_sft" / f"{name}.json") != value["sha256"]:
            raise ValueError("SFT dataset changed after the input audit")
    if ready["model_config_sha256"] != digest(a.root / "models/teacher3b_base/config.json"):
        raise ValueError("SFT model changed after the input audit")
    if a.mode != "train":
        print("CPU SFT input preflight passed; no training launched")
        return
    import torch

    if torch.cuda.device_count() != 4:
        raise ValueError("Exactly four GPUs are required")
    output = Path(config["output_dir"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            "Existing SFT output must not be overwritten; reuse a completed checkpoint with --teacher-initialization in oprd.experiments, or choose a new data root"
        )
    env_dir = ROOT / ".venv-vilasr-sft"
    cli = env_dir / "bin/llamafactory-cli"
    if not cli.is_file():
        raise FileNotFoundError("Install requirements/agent-sft.txt in .venv-vilasr-sft first")
    # W&B follows the selected mode (offline by default); no HF upload.
    dump(
        a.root / "prepared_sft/ds_zero2.json",
        {
            "bf16": {"enabled": "auto"},
            "zero_optimization": {"stage": 2, "overlap_comm": True, "contiguous_gradients": True},
            "gradient_accumulation_steps": "auto",
            "train_micro_batch_size_per_gpu": "auto",
            "train_batch_size": "auto",
            "gradient_clipping": "auto",
        },
    )
    config_path = a.root / "prepared_sft/train_3b.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    subprocess.run(
        [str(cli), "train", str(config_path)],
        check=True,
        env={
            **os.environ,
            "PATH": str(env_dir / "bin") + os.pathsep + os.environ["PATH"],
            "FORCE_TORCHRUN": "1",
            "NNODES": "1",
            "NPROC_PER_NODE": "4",
            "WANDB_MODE": a.wandb_mode,
            "HF_HUB_OFFLINE": "1",
        },
    )
    state = json.loads((output / "trainer_state.json").read_text())
    if state.get("epoch", 0) < 2.999:
        raise ValueError("SFT did not finish three epochs")
    dump(
        output / "vilasr_sft_complete.json",
        {
            "complete": True,
            "epochs": 3,
            "config": config,
            "audit": ready,
            "trainer_state_sha256": digest(output / "trainer_state.json"),
        },
    )


if __name__ == "__main__":
    main()
