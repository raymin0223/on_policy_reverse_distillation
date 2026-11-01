"""Evaluate a text checkpoint using the paper's task-specific sampling protocol."""

import argparse
import json
from pathlib import Path
import sys
from .evaluation import parallel as eval_parallel
from .paper import TASKS, mean_score


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=[x for x in TASKS if x != "agent"], required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    eval_parallel.add_arguments(p)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--limit", type=int)
    p.add_argument("--apps-python", default=".venv-apps/bin/python")
    p.add_argument("--evalplus-python", default=".venv-evalplus/bin/python")
    argv = sys.argv[1:] if argv is None else argv
    a = p.parse_args(argv)
    import pyarrow.parquet as pq

    spec = TASKS[a.task]
    rows = pq.read_table(a.data).to_pylist()
    if a.limit is not None:
        rows = rows[: a.limit]
    eval_parallel.validate_arguments(a, len(rows))
    humaneval = all(row["data_source"] == "humanevalplus" for row in rows)
    if a.worker_rank is None:
        eval_parallel.run_workers("oprd.evaluate", argv, a.output, len(rows), spec.evaluation_n)
        if humaneval:
            from .evaluation.humaneval import score_file

            scores = score_file(a.output, rows, a.data.parent / "snapshot.jsonl", a.evalplus_python)
        else:
            scores = [row["score"] for row in eval_parallel.records(a.output)]
        summary = {
            "model": a.model, "task": a.task, "data": str(a.data), "seed": a.seed,
            "questions": len(rows), "samples_per_question": spec.evaluation_n,
            "mean_score": mean_score(scores), "data_parallel_size": 4, "tensor_parallel_size": 1,
        }
        a.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        return
    first, stop = eval_parallel.bounds(len(rows), a.worker_rank)
    rows = rows[first:stop]
    if not rows:
        a.worker_output.touch(exist_ok=False)
        return
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from .rewards import compute_score

    tokenizer = AutoTokenizer.from_pretrained(a.model)
    llm = LLM(
        model=a.model,
        tensor_parallel_size=1,
        dtype="bfloat16",
        gpu_memory_utilization=a.gpu_memory_utilization,
        max_num_seqs=16 if a.task == "code" else 128,
        max_num_batched_tokens=8192,
        max_model_len=2048 + spec.evaluation_length,
        seed=a.seed,
    )
    sampling = SamplingParams(
        n=spec.evaluation_n,
        temperature=spec.evaluation_temperature,
        top_p=spec.evaluation_top_p,
        top_k=20,
        max_tokens=spec.evaluation_length,
        seed=a.seed,
    )
    from oprd.code import common as apps_runtime

    # A venv's bin/python must retain its path even when it is a symlink.
    apps_runtime.CPU_PYTHON = Path(a.apps_python).expanduser().absolute()
    with a.worker_output.open("x") as f:
        for start in range(0, len(rows), a.batch_size):
            batch = rows[start : start + a.batch_size]
            prompts = []
            for row in batch:
                messages = row["prompt"]
                if not spec.system_prompt:
                    messages = [m for m in messages if m["role"] != "system"]
                prompts.append(
                    tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                    )
                )
            generations = llm.generate(prompts, sampling)
            for i, (row, result) in enumerate(zip(batch, generations, strict=True), first + start):
                for j, out in enumerate(result.outputs):
                    if row["data_source"] == "apps_competition":
                        from oprd.code.apps_evaluator import evaluate_submission
                        from verl.utils.reward_score.deepcoder_rlvr import extract_single_python_code_block

                        extracted = extract_single_python_code_block(out.text)
                        score = (
                            0.0
                            if extracted.code is None
                            else float(
                                evaluate_submission(
                                    {"input_output": json.loads(row["reward_model"]["ground_truth"])},
                                    extracted.code,
                                )["reward"]
                            )
                        )
                    else:
                        score = (
                            None
                            if humaneval
                            else float(
                                compute_score(
                                    row["data_source"],
                                    out.text,
                                    row["reward_model"]["ground_truth"],
                                    row.get("extra_info"),
                                )["score"]
                            )
                        )
                    record = {
                        "problem_index": i,
                        "sample_index": j,
                        "response": out.text,
                        "score": score,
                        "finish_reason": out.finish_reason,
                        "tokens": len(out.token_ids),
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
