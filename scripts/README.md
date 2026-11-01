# Paper experiment scripts

The `01`–`06` prefixes are navigation order, not paper section numbers. The table below maps them to arXiv v2. The directories are numbered consecutively for the retained paper experiments and additional results. Section 4.3 student-behavior and qualitative analysis is outside the public reproduction scope. Each script expands a named recipe from `oprd/recipes.py`; shared training defaults remain in `oprd/paper.py` and `oprd/train.py`.

All recipes target **one node with four NVIDIA B200 GPUs: DP=4, TP=1**. RL uses FSDP with BF16 mixed precision, padding removal, and gradient checkpointing. W2SR-P SFT uses FSDP2; Agent cold-start SFT uses ZeRO-2, also on four GPUs. Scripts select GPUs `0,1,2,3` unless `CUDA_VISIBLE_DEVICES` is already set. Prompt and minibatch sizes are global across the four GPUs. Offline evaluation launches four independent single-GPU vLLM engines (DP=4, TP=1) and partitions questions across them; sampling counts, lengths, temperatures, and scoring rules retain the task-specific protocol.

| Directory | Paper section | Scripts |
|---|---|---|
| `01_weak_to_strong/` | §3.2, successor model transfer | `math.sh`, `reasoning.sh`, `code.sh`, `agent.sh` |
| `02_multi_teacher/` | §3.3, specialist consolidation | `multi_teacher.sh` |
| `03_strong_to_weak/` | §3.4, reversed capacity ordering | `math.sh`, `knights.sh` |
| `04_method_comparison/` | §4.1 and baseline appendix | `method_comparison.sh`, `w2sr_p.sh` |
| `05_guidance_design/` | §4.2 and alignment appendix | `teacher_strength.sh`, `guidance_directions.sh`, `guidance_directions_main_teacher.sh`, `lambda_strength.sh`, `alignment_branches.sh`, `branch_schedules.sh` |
| `06_additional_results/` | §4.4 and additional-results appendix | `math_base.sh`, `additional_reasoning.sh`, `reference_length_bias.sh`, `vanishing_reward.sh` |

Section 4.3 analysis code and launchers are excluded from the public tree. Cost/memory profiling also remains outside these launchers. See [validation](../docs/validation.md) and [protocol notes](../docs/protocol_notes.md).

## Run a complete experiment

Activate the training environment first, or set `PYTHON=/path/to/python`. Scripts also work when invoked from another working directory.

```bash
# Review the exact commands; no model loading, downloads, or files are created.
bash scripts/01_weak_to_strong/math.sh --dry-run

# Data, teachers, student methods, and checkpoint evaluation are explicit stages.
bash scripts/01_weak_to_strong/math.sh --stage prepare
bash scripts/01_weak_to_strong/math.sh --stage teacher
bash scripts/01_weak_to_strong/math.sh --stage train
bash scripts/01_weak_to_strong/math.sh --stage evaluate
```

The default stage is `train`, and the main scripts run GRPO, OPD, KDRL, and OPRD sequentially. `--stage all` runs preparation, teacher training, student training, and evaluation sequentially. It is a full experiment, not a smoke test.

All recipes use full finetuning; there is no LoRA/adapter switch. The default student seed is 42 except Binary Matrix (1234); `--seed` overrides it. `--teacher-seed` independently defaults to 42 so student-seed repeats reuse the same frozen teachers.

Select a smaller matrix when needed:

```bash
bash scripts/01_weak_to_strong/reasoning.sh \
  --tasks knights_knaves --methods oprd --dry-run
bash scripts/05_guidance_design/teacher_strength.sh \
  --tasks quantum_lock --variants teacher_60 --dry-run
bash scripts/01_weak_to_strong/code.sh \
  --stage evaluate --methods oprd --steps 150 --dry-run
```

All commands accept `--help`, `--dry-run`, `--json`, `--seed`, `--data-root`, `--output-root`, and `--teacher-root`. `--steps` selects evaluation checkpoints; it does not shorten training. `--teacher-initialization` reuses an Agent 3B SFT checkpoint for teacher GRPO and the OPRD reference; `--stage all` then skips SFT reconstruction. `--teacher` and `--teacher-reference` accept local checkpoints for a single task. They cannot silently replace a teacher-quality or reference ablation. Missing inputs stop the run with their expected location.

## Checkpoint and result layout

```text
datasets/                         Prepared public data, ignored by Git
outputs/
  teachers/<task>/seed-42/
    global_step_75/actor/huggingface/
  01_weak_to_strong/
    weak_to_strong_math/math/oprd/default/seed-42/
      global_step_30/actor/huggingface/
      evaluation/step-30-aime24.jsonl
      wandb/offline-run-.../
  05_guidance_design/
    teacher_strength/quantum_lock/oprd/teacher_60/seed-42/
```

All RL runs validate before training and every 15 updates. Student models are retained at five evenly spaced points: 30/60/90/120/150 for 150-update runs, 45/90/135/180/225 for Agent, and 60/120/180/240/300 for multi-teacher. The current best validation model is retained too, with no duplicate when it overlaps a regular snapshot. Only the latest committed evaluation checkpoint retains full resume state. Agent training diagnostics use 128 questions × 4 samples; the separate paper evaluation uses the full holdout × 1 sample. `--stage evaluate` selects the paper checkpoints by default; `--steps` can select any saved update. It never averages model weights or runs `oprd.evaluation.report` automatically.

Teachers are shared between scripts. Teacher runs retain five evenly spaced snapshots, the current best, the selected main checkpoint and every later-reference/teacher-strength checkpoint needed by related recipes. Off-cadence snapshots are also evaluated. Teacher budgets match the corresponding student: 150 updates for single-domain Math/reasoning and strong-to-weak, or 300 per specialist when launched from the multi-teacher section. Code extends to 435 and Agent to 450 to reach their required paper checkpoints (students: 150 and 225). Running multi-teacher preparation after a completed 150-update specialist run resumes that shared teacher to 300; its required frozen paper checkpoints remain retained. The student always reads the selected frozen checkpoint, not the final or best teacher automatically. Strong-to-weak teachers have separate `strong_math` and `strong_knights_knaves` directories.

The export must contain `config.json` and model weights. If using an existing sharded training checkpoint, first export it with the bundled FSDP model merger and pass the exported directory. Existing evaluation files are not overwritten. To start an independent repetition, use a different seed or output root.

RL checkpoints include model shards, optimizer state, resume metadata, and a full Hugging Face model export. Only scheduled, current-best and latest-resume models survive cleanup; only the latest retains optimizer and other resume state. A complete method matrix can require several TB; read the [storage and execution guide](../docs/reproduction.md#storage-and-execution-notes) before `--stage all`.

## W&B: offline by default

Default settings are `WANDB_MODE=offline` and `WANDB_PROJECT=on-policy-reverse-distillation`. Runs also log to the console. The runtime stores offline RL/SFT logs inside each run directory, and no login or API-key argument is needed. The repository contains no API keys, account-specific entity, or historical run IDs. Nothing uploads or syncs automatically.

```bash
# Local logs, custom project:
bash scripts/01_weak_to_strong/math.sh --methods oprd \
  --wandb-mode offline --wandb-project my-oprd-experiments

# Disable W&B entirely:
bash scripts/01_weak_to_strong/math.sh --methods oprd --wandb-mode disabled

# Explicit online mode, using authentication already configured outside the repo:
bash scripts/01_weak_to_strong/math.sh --methods oprd \
  --wandb-mode online --wandb-project my-oprd-experiments --wandb-entity my-team
```

The same options work with `python -m oprd.train`, `oprd.sft`, and `oprd.agent.sft`. Environment variables can supply these public settings too; command-line options take precedence. Do not place credentials in scripts, JSON manifests, Hydra overrides, or tracked environment files.

For RL runs, start with the `important/` dashboard views; the original metric names remain available.

| Metrics | Meaning |
| --- | --- |
| `important/train/task_score`, `important/train/reward` | Task score before the training length penalty, and the reward used for learning |
| `important/train/invalid_format`, `response_length/*`, `grpo/group_score/*` | Format failure rate, generated action-token lengths, and within-prompt reward diversity |
| `actor/pg_loss`, `actor/grad_norm`, `actor/lr`, `actor/pg_clipfrac` | Optimizer objective, gradient norm, learning rate, and PPO clipping diagnostics |
| `distillation/opd_loss`, `distillation/kdrl_*loss`, `distillation/kdrl_beta` | Method objective and KDRL's RL, auxiliary and weighted auxiliary components |
| `distillation/oprd_lambda_*`, `distillation/oprd_projection/*` | Scheduled strengths, fraction of tokens with guidance/correction, relative correction norm, and update–guidance alignment |
| `val-core/<dataset>/score/mean@N`, `best_validation/*` | Unpenalized validation score, current best score and its update number |
| `tool_call_counts/*`, `verl_tool/stop_reason/*`, `trajectory_length/*`, `observation_length/*` | Agent tool use, termination reasons, total trajectory length including observations, and observation-token length |

Scores and rates are logged on a 0–1 scale; paper reports multiply scores by 100. `mean@N` averages individual response scores; it is not any-of-N success. Agent's online validation is a training diagnostic, separate from the full paper evaluation. OPRD projection metrics use the unique sampled-action + top-10 support and token weighting across microbatches and GPUs. Relative changes include zero corrections on invalid-format responses; alignment excludes zero directions/gradients. These diagnostics do not change the gradient. Loss components use the optimizer's global token normalization. Async evaluation is logged at its source update; later W&B rows may wait for its reward job to finish. Offline files live under `<run>/wandb/`; checkpoints and evaluation JSON files are saved separately from W&B.

## Special workflows

**Multi-teacher:** prepare/train the four main reasoning teachers first. The multi-teacher script writes a local mapping under `outputs/02_multi_teacher/` and trains with 16 prompts per task, 300 updates, and the original 75-update coefficient schedule.

**Agent:** `prepare` downloads models/RL/evaluation assets. To reconstruct the 3B teacher initialization, install the separate `.venv-vilasr-sft` environment and run `--stage teacher-sft`; this stage downloads/prepares the cold-start data before auditing and training. Then run `teacher`, `train`, and `evaluate`. To reuse an existing 3B SFT checkpoint, pass `--teacher-initialization /path/to/3b-sft`; `all` then skips SFT reconstruction. The default `train` stage always consumes existing checkpoints. The SFT stage uses four GPUs. Student training reads the public 7B SFT initialization, teacher update 450, and the teacher's own 3B SFT reference. Evaluation uses all 10,464 held-out questions, not the 128-question validation subset.

**W2SR-P:** `w2sr_p.sh --stage train --tasks knights_knaves` first builds the verified teacher demonstrations, then performs three-epoch SFT. Evaluate the exported SFT model explicitly with `--stage evaluate --tasks knights_knaves --model /path/to/exported-model`; RL update numbers do not identify SFT checkpoints.


For detailed prompts, metrics, dataset provenance, evaluator environments, and baseline definitions, see [the reproduction guide](../docs/reproduction.md).
