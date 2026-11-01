"""Full-parameter W2SR-P behavior cloning on filtered teacher demonstrations."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
from .runtime import tracking


def main():
    p = argparse.ArgumentParser(description=__doc__)
    tracking.add_arguments(p)
    p.add_argument("--model", required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-length", type=int, default=22528)
    p.add_argument("--gpus", type=int, choices=(4,), default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    tracking.configure_environment(a)
    config = {
        "data.train_files": str(a.data.resolve()),
        "data.val_files": None,
        "data.train_batch_size": 64,
        "data.micro_batch_size_per_gpu": 1,
        "data.max_token_len_per_gpu": a.max_length,
        "data.use_dynamic_bsz": True,
        "data.max_length": a.max_length,
        "data.truncation": "error",
        "data.custom_cls.path": "pkg://verl.utils.dataset.teacher_bc_sft_dataset",
        "data.custom_cls.name": "TeacherBCSFTDataset",
        "data.pad_mode": "no_padding",
        "model.path": a.model,
        "model.use_remove_padding": True,
        "model.enable_gradient_checkpointing": True,
        "engine.strategy": "fsdp2",
        "engine.fsdp_size": a.gpus,
        "engine.ulysses_sequence_parallel_size": 1,
        "engine.model_dtype": "bfloat16",
        "engine.dtype": "bfloat16",
        "optim.optimizer": "AdamW",
        "optim.lr": 2e-5,
        "optim.lr_warmup_steps": 10,
        "optim.weight_decay": 0.01,
        "optim.betas": [0.9, 0.999],
        "optim.clip_grad": 1.0,
        "optim.lr_scheduler_type": "constant",
        "trainer.seed": a.seed,
        "trainer.total_epochs": 3,
        "trainer.save_freq": 1000000000,
        "trainer.test_freq": -1,
        "trainer.resume_mode": "disable",
        "trainer.max_ckpt_to_keep": 1,
        "trainer.default_local_dir": str(a.output.resolve()),
        **tracking.trainer_overrides(a),
        "trainer.experiment_name": "w2sr-p",
        "checkpoint.save_contents": ["model", "optimizer", "extra", "hf_model"],
    }
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={a.gpus}",
        "-m",
        "verl.trainer.sft_trainer",
    ] + ["++" + k + "=" + json.dumps(v) for k, v in config.items()]
    if a.dry_run:
        print(shlex.join(cmd))
    else:
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
