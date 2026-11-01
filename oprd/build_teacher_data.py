#!/usr/bin/env python3
"""Build offline behavior-cloning SFT data from verifiable teacher rollouts.

The parent process reconstructs the first ``steps * batch_size`` rows consumed by
VERL's seeded RL sampler.  It then launches one independent vLLM worker per GPU.
Each worker generates ``n`` candidates, scores them with the repository reward
function, and chooses one valid correct candidate uniformly with a fixed seed.

This script is intentionally local-only: it does not initialize W&B, Ray, or any
remote checkpoint/data upload integration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=("knights_knaves", "math_dapo", "string_manipulation"),
        required=True,
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--generation-seed", type=int, default=42)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--data-parallel-size", type=int, choices=(4,), default=4)
    parser.add_argument("--data-parallel-devices", default=None)
    parser.add_argument("--filter-overlong-prompts-workers", type=int, default=8)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--manifest-only", action="store_true", help="Build/validate the RL-matched manifest and exit"
    )
    parser.add_argument("--worker-id", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--num-workers", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--manifest", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.drop_system_prompt = args.task != "math_dapo"
    args.enable_thinking = False
    return args


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if hasattr(value, "item"):
        return _jsonable(value.item())
    return str(value)


def _messages(value: Any) -> list[dict[str, Any]]:
    converted = _jsonable(value)
    if not isinstance(converted, list):
        raise TypeError(f"Expected a list of chat messages, got {type(converted).__name__}")
    return [dict(message) for message in converted]


def _config_payload(args: argparse.Namespace) -> dict[str, Any]:
    with args.data.open("rb") as stream:
        source_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "schema_version": 3,
        "task": args.task,
        "data": str(args.data.resolve()),
        "data_sha256": source_hash,
        "model": args.model,
        "tokenizer": args.tokenizer or args.model,
        "steps": args.steps,
        "train_batch_size": args.train_batch_size,
        "num_prompts": args.steps * args.train_batch_size,
        "n": args.n,
        "data_seed": args.data_seed,
        "generation_seed": args.generation_seed,
        "selection_seed": args.selection_seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_prompt_length": args.max_prompt_length,
        "max_new_tokens": args.max_new_tokens,
        "max_model_len": args.max_model_len,
        "drop_system_prompt": args.drop_system_prompt,
        "enable_thinking": args.enable_thinking,
        "preserve_rollout_token_ids": True,
    }


def _fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tokenizer_vocab_fingerprint(tokenizer) -> str:
    """Hash the complete token-to-ID map used by vLLM generation."""

    encoded = json.dumps(
        sorted(tokenizer.get_vocab().items()),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_or_validate_config(args: argparse.Namespace) -> str:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = _config_payload(args)
    payload["fingerprint"] = _fingerprint(payload)
    path = args.output_dir / "generation_config.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(
                f"Existing generation config differs from this run: {path}. "
                "Use a new output directory to avoid mixing rollout populations."
            )
    else:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return str(payload["fingerprint"])


def _build_manifest(args: argparse.Namespace) -> Path:
    """Reproduce the RL dataset filtering and seeded Stateful RandomSampler."""

    from omegaconf import OmegaConf
    from transformers import AutoTokenizer

    from verl.trainer.main_ppo import create_rl_sampler
    from verl.utils.dataset.rl_dataset import RLHFDataset

    manifest_path = args.output_dir / "manifest.parquet"
    if manifest_path.exists():
        manifest = pd.read_parquet(manifest_path)
        expected = args.steps * args.train_batch_size
        if len(manifest) != expected:
            raise RuntimeError(f"Expected {expected} manifest rows, found {len(manifest)} in {manifest_path}")
        return manifest_path

    source = args.data.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    indexed_source = args.output_dir / ".indexed_source.parquet"
    source_df = pd.read_parquet(source)
    source_df["bc_source_row_idx"] = range(len(source_df))
    source_df.to_parquet(indexed_source, index=False)

    tokenizer_source = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    data_config = OmegaConf.create(
        {
            "prompt_key": "prompt",
            "max_prompt_length": args.max_prompt_length,
            "filter_overlong_prompts": True,
            "filter_overlong_prompts_workers": args.filter_overlong_prompts_workers,
            "shuffle": True,
            "seed": args.data_seed,
            "sampler": None,
            "drop_system_prompt": args.drop_system_prompt,
            "prepend_dropped_system_prompt_to_user": False,
            "prompt_format": "chat",
            "plain_completion_include_system": not args.drop_system_prompt,
            "apply_chat_template_kwargs": {"enable_thinking": args.enable_thinking},
            "return_raw_chat": True,
            "truncation": "error",
            "use_shm": False,
        }
    )

    try:
        dataset = RLHFDataset(
            data_files=str(indexed_source),
            tokenizer=tokenizer,
            processor=None,
            config=data_config,
        )
        expected = args.steps * args.train_batch_size
        if len(dataset) < expected:
            raise RuntimeError(f"RL dataset has only {len(dataset)} usable rows; need {expected}")

        sampler = create_rl_sampler(data_config, dataset)
        sampled_indices = []
        for dataset_index in sampler:
            sampled_indices.append(int(dataset_index))
            if len(sampled_indices) == expected:
                break

        rows: list[dict[str, Any]] = []
        for sample_order, dataset_index in enumerate(sampled_indices):
            item = dataset[dataset_index]
            rows.append(
                {
                    "rl_sample_order": sample_order,
                    "rl_step": sample_order // args.train_batch_size + 1,
                    "rl_batch_slot": sample_order % args.train_batch_size,
                    "rl_dataset_index": dataset_index,
                    "source_row_idx": int(item["bc_source_row_idx"]),
                    "prompt": _jsonable(item["raw_prompt"]),
                    "data_source": str(item.get("data_source") or args.task),
                    "reward_model": _jsonable(item.get("reward_model") or {}),
                    "extra_info": _jsonable(item.get("extra_info") or {}),
                }
            )
        pd.DataFrame(rows).to_parquet(manifest_path, index=False)
    finally:
        indexed_source.unlink(missing_ok=True)

    print(f"[manifest] wrote {len(rows)} RL-matched rows to {manifest_path}", flush=True)
    return manifest_path


def _device_groups(args: argparse.Namespace) -> list[str]:
    raw = args.data_parallel_devices or os.environ.get("CUDA_VISIBLE_DEVICES", "")
    devices = [item.strip() for item in raw.split(",") if item.strip()]
    if not devices:
        devices = [str(index) for index in range(args.data_parallel_size)]
    if len(devices) != args.data_parallel_size or len(set(devices)) != args.data_parallel_size:
        raise ValueError(
            f"Expected exactly {args.data_parallel_size} generation GPUs, got {devices}. "
            "Set CUDA_VISIBLE_DEVICES to four comma-separated device IDs."
        )
    return devices


def _worker_command(
    args: argparse.Namespace,
    *,
    worker_id: int,
    num_workers: int,
    manifest: Path,
    worker_output: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--task",
        args.task,
        "--data",
        str(args.data),
        "--model",
        args.model,
        "--output-dir",
        str(args.output_dir),
        "--steps",
        str(args.steps),
        "--train-batch-size",
        str(args.train_batch_size),
        "--n",
        str(args.n),
        "--data-seed",
        str(args.data_seed),
        "--generation-seed",
        str(args.generation_seed),
        "--selection-seed",
        str(args.selection_seed),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--top-k",
        str(args.top_k),
        "--max-prompt-length",
        str(args.max_prompt_length),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--max-model-len",
        str(args.max_model_len),
        "--chunk-size",
        str(args.chunk_size),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--dtype",
        args.dtype,
        "--worker-id",
        str(worker_id),
        "--num-workers",
        str(num_workers),
        "--manifest",
        str(manifest),
        "--worker-output",
        str(worker_output),
    ]
    if args.tokenizer:
        command.extend(("--tokenizer", args.tokenizer))
    if args.trust_remote_code:
        command.append("--trust-remote-code")
    if args.local_files_only:
        command.append("--local-files-only")
    return command


def _read_jsonl_by_order(path: Path, *, repair_tail: bool = False) -> dict[int, dict[str, Any]]:
    """Read complete records, optionally repairing an interrupted final append.

    Only an unterminated, invalid JSON tail can be discarded. Corrupt complete
    lines, missing IDs and duplicates must fail instead of silently losing data.
    """
    records: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return records
    truncate_at = None
    needs_newline = False
    with path.open("rb") as stream:
        line_number = 0
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            line_number += 1
            needs_newline = not line.endswith(b"\n")
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                if repair_tail and needs_newline:
                    truncate_at = offset
                    break
                raise ValueError(f"Malformed generation record at {path}:{line_number}") from exc
            if not isinstance(record, dict) or type(record.get("rl_sample_order")) is not int:
                raise ValueError(f"Invalid rl_sample_order at {path}:{line_number}")
            order = record["rl_sample_order"]
            if order in records:
                raise ValueError(f"Duplicate rl_sample_order={order} at {path}:{line_number}")
            records[order] = record
    if repair_tail:
        if truncate_at is not None:
            with path.open("r+b") as stream:
                stream.truncate(truncate_at)
            print(f"[resume] removed unfinished final record from {path}", flush=True)
        elif needs_newline:
            with path.open("ab") as stream:
                stream.write(b"\n")
    return records


def _validate_records(records: dict[int, dict[str, Any]], manifest: pd.DataFrame, *, complete: bool) -> None:
    expected = {int(row["rl_sample_order"]): row for row in manifest.to_dict(orient="records")}
    if len(expected) != len(manifest):
        raise ValueError("Duplicate rl_sample_order in generation manifest")
    unexpected = sorted(set(records) - set(expected))
    missing = sorted(set(expected) - set(records))
    if unexpected or (complete and missing):
        raise RuntimeError(
            f"Generation rows do not match manifest: missing={missing[:20]}, unexpected={unexpected[:20]}"
        )
    for order, record in records.items():
        for key in ("rl_step", "rl_batch_slot", "source_row_idx"):
            if record.get(key) != expected[order][key]:
                raise ValueError(f"Generation row {order} has mismatched {key}")


def _score_candidate(row: dict[str, Any], text: str, *, truncated: bool) -> dict[str, Any]:
    from verl.utils.reward_score.feedback import compute_score

    reward_model = row.get("reward_model") or {}
    ground_truth = reward_model.get("ground_truth", "")
    extra_info = dict(row.get("extra_info") or {})
    extra_info["truncated"] = truncated
    result = compute_score(
        str(row.get("data_source") or ""),
        text,
        str(ground_truth),
        extra_info,
    )
    if not isinstance(result, dict):
        result = {"score": float(result), "acc": float(result)}
    return _jsonable(result)


def _run_worker(args: argparse.Namespace) -> None:
    if (
        args.worker_id is None
        or args.num_workers is None
        or args.manifest is None
        or args.worker_output is None
    ):
        raise ValueError("Incomplete internal worker arguments")

    manifest = pd.read_parquet(args.manifest).sort_values("rl_sample_order")
    manifest = manifest[manifest["rl_sample_order"] % args.num_workers == args.worker_id]
    existing = _read_jsonl_by_order(args.worker_output, repair_tail=True)
    _validate_records(existing, manifest, complete=False)
    pending = [
        row for row in manifest.to_dict(orient="records") if int(row["rl_sample_order"]) not in existing
    ]
    if not pending:
        print(f"[worker {args.worker_id}] nothing to do; shard is complete", flush=True)
        return

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer_source = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    tokenizer_vocab_sha256 = _tokenizer_vocab_fingerprint(tokenizer)
    llm = LLM(
        model=args.model,
        tokenizer=tokenizer_source,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=True,
        disable_log_stats=True,
        seed=args.generation_seed + args.worker_id,
    )

    args.worker_output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.worker_output.open("a", encoding="utf-8") as stream:
            for start in range(0, len(pending), args.chunk_size):
                chunk = pending[start : start + args.chunk_size]
                prompts = [
                    tokenizer.apply_chat_template(
                        _messages(row["prompt"]),
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=args.enable_thinking,
                    )
                    for row in chunk
                ]
                sampling_params = [
                    SamplingParams(
                        n=args.n,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        max_tokens=args.max_new_tokens,
                        seed=args.generation_seed + int(row["rl_sample_order"]),
                    )
                    for row in chunk
                ]
                outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
                for row, request_output in zip(chunk, outputs, strict=True):
                    prompt_token_ids = [int(token_id) for token_id in (request_output.prompt_token_ids or [])]
                    if not prompt_token_ids:
                        raise RuntimeError(
                            f"vLLM returned no prompt_token_ids for rl_sample_order={row['rl_sample_order']}"
                        )
                    candidates: list[dict[str, Any]] = []
                    correct_indices: list[int] = []
                    for candidate_index, output in enumerate(request_output.outputs):
                        text = str(output.text)
                        finish_reason = str(output.finish_reason or "")
                        token_ids = list(output.token_ids or [])
                        truncated = finish_reason == "length" or (
                            len(token_ids) >= args.max_new_tokens and finish_reason not in {"stop", "eos"}
                        )
                        score = _score_candidate(row, text, truncated=truncated)
                        valid_correct = (
                            float(score.get("acc", 0.0)) >= 1.0
                            and int(score.get("incorrect_format", 0)) == 0
                            and int(score.get("truncated", int(truncated))) == 0
                        )
                        candidates.append(
                            {
                                "candidate_index": candidate_index,
                                "num_tokens": len(token_ids),
                                "finish_reason": finish_reason,
                                "score": score,
                                "valid_correct": valid_correct,
                                "text": text if valid_correct else None,
                                "token_ids": token_ids if valid_correct else None,
                            }
                        )
                        if valid_correct:
                            correct_indices.append(candidate_index)

                    sample_order = int(row["rl_sample_order"])
                    selected_index = None
                    selected_response = None
                    selected_num_tokens = None
                    selected_response_token_ids = None
                    selected_finish_reason = None
                    selected_stop_reason = None
                    if correct_indices:
                        chooser = random.Random(args.selection_seed + sample_order)
                        selected_index = chooser.choice(correct_indices)
                        selected = candidates[selected_index]
                        selected_response = selected["text"]
                        selected_num_tokens = selected["num_tokens"]
                        selected_response_token_ids = selected["token_ids"]
                        selected_finish_reason = selected["finish_reason"]
                        raw_stop_reason = request_output.outputs[selected_index].stop_reason
                        selected_stop_reason = json.dumps(_jsonable(raw_stop_reason), ensure_ascii=False)
                        if not selected_response_token_ids:
                            raise RuntimeError(
                                f"Selected rollout has no response token IDs for rl_sample_order={sample_order}"
                            )

                    candidate_summary = []
                    for candidate in candidates:
                        summary = dict(candidate)
                        summary.pop("text", None)
                        summary.pop("token_ids", None)
                        candidate_summary.append(summary)
                    record = {
                        "rl_sample_order": sample_order,
                        "rl_step": int(row["rl_step"]),
                        "rl_batch_slot": int(row["rl_batch_slot"]),
                        "source_row_idx": int(row["source_row_idx"]),
                        "has_correct": selected_index is not None,
                        "num_correct": len(correct_indices),
                        "selected_rollout_index": selected_index,
                        "selected_response": selected_response,
                        "selected_num_tokens": selected_num_tokens,
                        "prompt_token_ids": prompt_token_ids,
                        "selected_response_token_ids": selected_response_token_ids,
                        "selected_finish_reason": selected_finish_reason,
                        "selected_stop_reason": selected_stop_reason,
                        "selected_last_token_id": (
                            int(selected_response_token_ids[-1]) if selected_response_token_ids else None
                        ),
                        "tokenizer_vocab_sha256": tokenizer_vocab_sha256,
                        "candidate_summary": candidate_summary,
                    }
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stream.flush()
                    print(
                        f"[worker {args.worker_id}] order={sample_order} "
                        f"correct={len(correct_indices)}/{args.n} selected={selected_index}",
                        flush=True,
                    )
    finally:
        try:
            llm.llm_engine.shutdown()
        except Exception:
            pass


def _merge_outputs(args: argparse.Namespace, manifest_path: Path, shard_paths: list[Path]) -> None:
    manifest = pd.read_parquet(manifest_path).sort_values("rl_sample_order")
    records: dict[int, dict[str, Any]] = {}
    for path in shard_paths:
        shard = _read_jsonl_by_order(path)
        duplicates = sorted(set(records) & set(shard))
        if duplicates:
            raise ValueError(f"Duplicate generation rows across shards: {duplicates[:20]} in {path}")
        records.update(shard)
    _validate_records(records, manifest, complete=True)

    generated = pd.DataFrame([records[index] for index in sorted(records)])
    merged = manifest.merge(
        generated, on=["rl_sample_order", "rl_step", "rl_batch_slot", "source_row_idx"], validate="one_to_one"
    )

    coverage = merged[
        [
            "rl_sample_order",
            "rl_step",
            "rl_batch_slot",
            "source_row_idx",
            "data_source",
            "has_correct",
            "num_correct",
            "selected_rollout_index",
            "selected_num_tokens",
            "selected_finish_reason",
            "selected_stop_reason",
            "selected_last_token_id",
            "tokenizer_vocab_sha256",
        ]
    ].copy()
    coverage["candidate_summary_json"] = [
        json.dumps(value, ensure_ascii=False) for value in merged["candidate_summary"]
    ]
    coverage.to_parquet(args.output_dir / "coverage.parquet", index=False)

    selected = merged[merged["has_correct"]].copy().sort_values("rl_sample_order")
    messages = []
    for prompt, response in zip(selected["prompt"], selected["selected_response"], strict=True):
        conversation = _messages(prompt)
        conversation.append({"role": "assistant", "content": str(response)})
        messages.append(conversation)

    sft = pd.DataFrame(
        {
            "messages": messages,
            "enable_thinking": [args.enable_thinking] * len(selected),
            "data_source": selected["data_source"].tolist(),
            "source_row_idx": selected["source_row_idx"].astype(int).tolist(),
            "rl_sample_order": selected["rl_sample_order"].astype(int).tolist(),
            "rl_step": selected["rl_step"].astype(int).tolist(),
            "rl_batch_slot": selected["rl_batch_slot"].astype(int).tolist(),
            "teacher_model": [args.model] * len(selected),
            "rollout_n": [args.n] * len(selected),
            "selected_rollout_index": selected["selected_rollout_index"].astype(int).tolist(),
            "selected_num_tokens": selected["selected_num_tokens"].astype(int).tolist(),
            "prompt_token_ids": selected["prompt_token_ids"].tolist(),
            "response_token_ids": selected["selected_response_token_ids"].tolist(),
            "finish_reason": selected["selected_finish_reason"].tolist(),
            "stop_reason": selected["selected_stop_reason"].tolist(),
            "last_token_id": selected["selected_last_token_id"].astype(int).tolist(),
            "tokenizer_vocab_sha256": selected["tokenizer_vocab_sha256"].tolist(),
        }
    )
    sft.to_parquet(args.output_dir / "train.parquet", index=False)

    summary = {
        **_config_payload(args),
        "num_manifest_prompts": len(manifest),
        "num_sft_examples": len(sft),
        "coverage": len(sft) / len(manifest),
        "mean_correct_rollouts": float(merged["num_correct"].mean()),
        "finish_reason_counts": {
            str(key): int(value) for key, value in selected["selected_finish_reason"].value_counts().items()
        },
        "last_token_id_counts": {
            str(int(key)): int(value)
            for key, value in selected["selected_last_token_id"].value_counts().items()
        },
        "tokenizer_vocab_sha256": (
            str(selected["tokenizer_vocab_sha256"].iloc[0]) if len(selected) > 0 else None
        ),
        "output_train_parquet": str((args.output_dir / "train.parquet").resolve()),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def _run_parent(args: argparse.Namespace) -> None:
    _write_or_validate_config(args)
    manifest = _build_manifest(args)
    if args.manifest_only:
        print(f"[manifest] manifest-only run complete: {manifest}", flush=True)
        return
    devices = _device_groups(args)
    shards_dir = args.output_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    shard_paths = [shards_dir / f"worker_{index:02d}.jsonl" for index in range(len(devices))]

    processes: list[tuple[int, subprocess.Popen[Any]]] = []
    try:
        for worker_id, device in enumerate(devices):
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = device
            environment["WANDB_MODE"] = "disabled"
            environment["WANDB_DISABLED"] = "true"
            for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
                environment.pop(key, None)
            command = _worker_command(
                args,
                worker_id=worker_id,
                num_workers=len(devices),
                manifest=manifest,
                worker_output=shard_paths[worker_id],
            )
            print(f"[parent] launching worker {worker_id} on physical GPU {device}", flush=True)
            processes.append((worker_id, subprocess.Popen(
                command, cwd=REPO_ROOT, env=environment, start_new_session=True,
            )))
        while True:
            states = [(worker_id, process.poll()) for worker_id, process in processes]
            failures = [(worker_id, state) for worker_id, state in states if state not in (None, 0)]
            if failures:
                raise RuntimeError(f"Generation workers failed: {failures}")
            if all(state == 0 for _, state in states):
                break
            time.sleep(0.2)
    except BaseException:
        # Also stop each worker's vLLM subprocesses on failure or interruption.
        for _, process in processes:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 5
        for _, process in processes:
            try:
                process.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass
        for _, process in processes:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        print(f"Generation did not complete; resumable shards remain in {shards_dir}", file=sys.stderr)
        raise
    _merge_outputs(args, manifest, shard_paths)


def main() -> None:
    args = _parse_args()
    args.data = args.data.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.steps <= 0 or args.train_batch_size <= 0 or args.n <= 0:
        raise ValueError("steps, train batch size, and n must all be positive")
    if args.worker_id is not None:
        _run_worker(args)
    else:
        _run_parent(args)


if __name__ == "__main__":
    main()
