"""Single-trajectory Agent evaluation with the training tools and verifier."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from oprd.evaluation import parallel as eval_parallel
from .vilasr_contract import MAX_PIXELS, MIN_PIXELS


def messages_for(row):
    messages = [dict(turn) for turn in row["prompt"]]
    parts = messages[1]["content"].split("<image>")
    if len(parts) != len(row["images"]) + 1:
        raise ValueError("Image placeholder mismatch")
    content = [{"type": "text", "text": parts[0]}] if parts[0] else []
    for image, trailing in zip(row["images"], parts[1:], strict=True):
        content.append({"type": "image", **image})
        if trailing:
            content.append({"type": "text", "text": trailing})
    messages[1]["content"] = content
    return messages


def summarize(path, seed):
    families = defaultdict(lambda: [0.0, 0])
    for row in eval_parallel.records(path):
        value = families[row["family"]]
        value[0] += row["validation_rule_score"]
        value[1] += 1
    scores = {k: 100 * total / count for k, (total, count) in families.items()}
    count = sum(value[1] for value in families.values())
    return {
        "questions": count, "seed": seed, "family_mean": scores,
        "family_macro": sum(scores.values()) / len(scores),
        "question_mean": 100 * sum(value[0] for value in families.values()) / count,
        "data_parallel_size": 4, "tensor_parallel_size": 1,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--data", type=Path, default=Path("datasets/agent/prepared/holdout_full.parquet"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=8)
    eval_parallel.add_arguments(p)
    p.add_argument("--limit", type=int)
    argv = sys.argv[1:] if argv is None else argv
    a = p.parse_args(argv)
    import pyarrow.parquet as pq

    rows = pq.read_table(a.data).to_pylist()
    if a.limit is not None:
        rows = rows[: a.limit]
    elif len(rows) != 10464:
        raise ValueError("Full paper evaluation requires 10464 questions; use --limit for a smoke test")
    eval_parallel.validate_arguments(a, len(rows))
    if a.worker_rank is None:
        eval_parallel.run_workers("oprd.agent.evaluate", argv, a.output, len(rows), 1)
        summary = summarize(a.output, a.seed)
        a.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        return
    first, stop = eval_parallel.bounds(len(rows), a.worker_rank)
    rows = rows[first:stop]
    if not rows:
        a.worker_output.touch(exist_ok=False)
        return
    from transformers import AutoProcessor
    from qwen_vl_utils import process_vision_info
    from vllm import LLM, SamplingParams
    from .generation import EvalLimits, generate_aligned

    processor = AutoProcessor.from_pretrained(a.model, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    engine = LLM(
        model=a.model,
        tensor_parallel_size=1,
        dtype="bfloat16",
        gpu_memory_utilization=a.gpu_memory_utilization,
        max_num_seqs=32,
        max_num_batched_tokens=32768,
        mm_processor_cache_gb=0,
        max_model_len=24576,
        limit_mm_per_prompt={"image": 42},
        seed=a.seed,
    )
    sampling = SamplingParams(n=1, temperature=0.6, top_p=0.95, top_k=20, max_tokens=4096)
    with a.worker_output.open("x") as stream:
        for start in range(0, len(rows), a.batch_size):
            batch = rows[start : start + a.batch_size]
            inputs = []
            for row in batch:
                messages = messages_for(row)
                images, _ = process_vision_info(messages)
                inputs.append(
                    {
                        "prompt": processor.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        ),
                        "multi_modal_data": {"image": images},
                        "qid": row["extra_info"]["qid"],
                        "data_source": row["data_source"],
                        "gold": row["reward_model"]["ground_truth"],
                    }
                )
            results = generate_aligned(
                engine, processor, processor.tokenizer, inputs, sampling, EvalLimits(seed=a.seed)
            )
            for index, (row, result) in enumerate(zip(batch, results, strict=True), first + start):
                family = row["extra_info"]["family"]
                record = {**result, "family": family, "problem_index": index, "sample_index": 0}
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()


if __name__ == "__main__":
    main()
