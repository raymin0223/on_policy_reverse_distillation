# Reproducing the paper

The reference is [arXiv:2609.08798v2](https://arxiv.org/abs/2609.08798v2). `oprd.paper` defines the public experiment defaults. All commands below run from the repository root. Paths such as `/path/to/teacher` are inputs you supply; no private checkpoint service is required.

For section-level experiment matrices, start with [scripts/README.md](../scripts/README.md). All scripts target four B200 GPUs and default to W&B offline under `on-policy-reverse-distillation`. [Protocol notes](protocol_notes.md) document task-specific defaults and runtime behavior.

## Configuration and four-GPU execution

The executable configuration is split by responsibility: `oprd/paper.py` holds task defaults, `oprd/recipes.py` selects experiment matrices/variants, and `oprd/train.py` builds runtime options on top of the bundled `verl/trainer/config/` Hydra schemas. There are no duplicate top-level JSON configuration snapshots. The section shell scripts select recipes; editing a shell wrapper is usually unnecessary. Top-level `oprd/` files contain execution entrypoints, experiment definitions, and the core method. `oprd/runtime/` contains the training backend and shared setup/validation; `oprd/evaluation/` contains sharding, official scoring, and optional benchmark-table aggregation. `agent/` and `code/` contain domain-specific components. The main training, preparation, and evaluation commands keep their existing module names.

Training and evaluation use **one node, DP=4, TP=1**. RL uses FSDP; W2SR-P SFT uses FSDP2 and Agent cold-start SFT uses ZeRO-2. Offline evaluation splits questions across four independent vLLM processes, each restricted to one device from `CUDA_VISIBLE_DEVICES`. Set exactly four distinct device IDs, such as `CUDA_VISIBLE_DEVICES=0,1,2,3`; GPU UUIDs are also accepted. All samples for a question stay together, and workers merge in original question/sample order. Even small `--limit` checks use this launch layout, with empty shards skipping model loading. Public entrypoints reject TP values other than 1 and DP sizes other than 4.

The table below lists this release's B200 memory and sequence limits. Settings may differ from those used in the original experiments.

| Workload | vLLM GPU memory fraction | Maximum sequences / batched tokens |
|---|---:|---|
| 8B text RL | 0.60 | 128 / 8,192 |
| 14B Code RL, including frozen 8B policies | 0.35 | 16 / 8,192 |
| 7B Agent RL | 0.60 | 64 / 32,768 |
| Text offline evaluation | 0.85 | 128 / 8,192; Code uses 16 sequences |
| Agent offline evaluation | 0.85 | 32 / 32,768; multimodal processor cache disabled |

These are per-engine memory fractions, not caps on total training memory. RL also holds actor/optimizer and frozen-policy state. The 14B profile leaves more room for those allocations. Offline evaluation can reserve more memory because it only loads the evaluated model. Both evaluation CLIs accept `--gpu-memory-utilization` for local tuning. Four-B200 small-model smoke checks do not establish full-size/long-context fit; the 8B/14B settings still need verification during an actual run. Keep paper sampling counts and length budgets intact when tuning execution memory.

## Shared training protocol

All four main methods use student on-policy rollouts, GRPO advantages without standard-deviation normalization, eight responses per prompt, and one PPO epoch. AdamW uses learning rate 1e-6, betas (0.9, 0.999), weight decay 0.01, gradient clipping 1, ten warmup steps, and a constant schedule. PPO clipping is 0.2 below and 0.28 above. The main methods have no initial-policy KL or entropy penalty.

The main text comparisons and teacher training use token-level rollout importance correction (threshold 2.0, no batch normalization). OPD additionally clips its current/old policy ratio to [0.5, 2.0]. Agent and the separate appendix baselines have distinct correction settings; see [protocol notes](protocol_notes.md#rollout-correction-and-distillation-inputs).

| Task | Weak teacher checkpoint | Train response limit | Prompt batch / minibatch setting | OPRD ramp / KDRL decay horizon |
|---|---:|---:|---|---:|
| Math | 75 | 20,480 | 64 / 64 | 30 |
| Knights & Knaves | 105 | 8,192 | 64 / 32 | 75 |
| Quantum Lock | 105 | 8,192 | 64 / 32 | 75 |
| String Manipulation | 75 | 8,192 | 64 / 32 | 75 |
| Countdown | 105 | 8,192 | 64 / 32 | 75 |
| Code | 435 | 16,384 | 64 / 32 | 45 |
| Agent | 450, after 3-epoch SFT | 16,384 including observations | 64 / 64 | 45 |

Text prompts have a 2,048-token budget. Math and Code apply a soft overlength penalty in the last 4,096 response tokens. Math uses the non-thinking system prompt; reasoning and code use non-thinking chat formatting without a system message.

OPRD holds λ+ at 0.5 (1.0 for Agent and the original Binary Matrix comparison) while ramping λ− from zero to the same value. The inherited one-based update convention starts the ramp at zero on update 1 and reaches its endpoint at update K. KDRL uses `max(0.005 - 0.005/K * (step-1), 0)`. Resume reads the saved global step. Backpropagated loss gradients have the opposite sign to the ascent gradient in the paper; the implementation accounts for that when selecting λ+ and λ−.

## Data and evaluation provenance

```bash
python -m oprd.prepare --task math
python -m oprd.prepare --task reasoning
python -m oprd.prepare --task code
```

Math downloads use the exact source files in [`data/math_sources.lock.json`](../data/math_sources.lock.json). This is an executable download lock: `data/paper_math.py` reads its repository, commit revision, filename, row count, and SHA-256 before preparing the data. The four pinned sources were compared with the original experiment files, including question order, full prompts, and answers:

| Input | Repository | Pinned revision | Rows |
|---|---|---|---:|
| DAPO English training | `open-r1/DAPO-Math-17k-Processed` | `31dd309567e3da778038cc87d868b6097a3ccf68` | 14,116 |
| AIME24 | `math-ai/aime24` | `83a7f387baaa524a8bda0022eac0541582297103` | 30 |
| AIME25 | `math-ai/aime25` | `563bb8404243c5f09de6ec262f2db674fe5bce9b` | 30 |
| HMMT25 | `MathArena/hmmt_feb_2025` | `6fdc4277120810ff75aa22d2d5489b91f7a262a1` | 30 |

The downloader passes `revision` directly to Hugging Face, then checks the downloaded bytes. Preparation also hashes the ordered, fully formatted prompts/answers so preprocessing or row-order changes cannot silently change the inputs. The AIME/HMMT boxed-answer instruction is separated from the question by one newline. Completed preparation copies the lock to `datasets/math/sources.lock.json` for run provenance. These locks are consumed by the downloader. OlympiadBench keeps its existing pinned revision/count checks; Code and Agent retain their existing revision/checksum controls.

Reasoning data uses seed 0 and 20,000 generated questions per task, with the final 200 reserved for evaluation. Quantum Lock and String Manipulation use the original indexed, deduplicated generators. Binary Matrix preserves the held-out set and refills training rows that overlap it. `--count 30 --test-size 5` provides a small preparation smoke test; these are not paper-sized splits.

`data/evaluation_manifest.json` records CodeContests' validated source indices and checker/resource settings, APPS Competition's retained task IDs, and HumanEval+'s original v0.1.10 snapshot. The retained CodeContests cohort has 106 questions; the HumanEval+ cohort has 163 questions and excludes HumanEval/32, matching the original experiment's canonical-solution validation. Preparation downloads public originals and rebuilds these subsets; it does not package private test files. LCBv5 and DeepCoder revisions are pinned in the data modules. Code execution tests must run in an isolated environment suitable for executing model-generated Python.

APPS uses the pinned official `hendrycks/apps` checker with the original per-test comparison semantics, a 10-second submission wall limit, and a 4 GiB memory limit. Its `pyext` dependency uses a separate Python 3.10 environment (Linux with libseccomp):

```bash
python3.10 -m venv .venv-apps
.venv-apps/bin/pip install -r requirements/apps.txt
```

`oprd.prepare --task code` downloads the checksum-verified official scorer. Supply `--apps-python .venv-apps/bin/python` to `oprd.evaluate` if using a different environment location.

HumanEval+ uses a separate official evaluator environment:

```bash
python3.12 -m venv .venv-evalplus
.venv-evalplus/bin/pip install -r requirements/evalplus.txt
python -m oprd.evaluate --task code --model /path/to/code-checkpoint \
  --data datasets/humanevalplus/test.parquet \
  --output outputs/evaluation/humanevalplus.jsonl \
  --evalplus-python .venv-evalplus/bin/python
```

The evaluator requires the pinned EvalPlus commit and both original and additional tests to pass. Invalid code-block formatting contributes zero. It reports the arithmetic mean of all four sample scores, not pass@4.

Text evaluation limits are 38,912 tokens for Math, 8,192 for reasoning, and 30,720 for Code. `--limit` is a smoke-test option and must not be used for reported results. Evaluation outputs use exclusive creation, preventing accidental replacement of existing responses.

## Main and additional single-teacher experiments

Train the teacher with `--teacher-training`, then pass the chosen frozen checkpoint to each student method:

Teacher runs use the student's full training horizon, extended when a required paper checkpoint is later: Math/reasoning and strong-to-weak 150, Code 435 (student 150), Agent 450 (student 225). The multi-teacher section uses 300 updates per specialist, matching its student. Retention is five evenly spaced models plus the current best and all paper teacher/reference checkpoints required by the shared teacher directory. For example, the Math teacher keeps 30/60/75/90/120/150 plus best; the student still uses teacher update 75. Direct `oprd.train --steps N` explicitly overrides the training horizon for bounded checks.

Without `--output`, direct teacher launches save under `outputs/teachers/<task>/seed-<seed>/`, separately from student runs. Use the same seed/output root when intentionally reusing a teacher across entrypoints; the section runner's teacher seed defaults to 42.

```bash
python -m oprd.train --task math --teacher-training --output outputs/teachers/math
python -m oprd.train --task math --method oprd \
  --teacher /path/to/math-teacher-step75 \
  --teacher-reference Qwen/Qwen3-4B --output outputs/math-oprd

python -m oprd.train --task code --teacher-training --output outputs/teachers/code
python -m oprd.train --task code --method oprd \
  --teacher /path/to/code-teacher-step435 \
  --teacher-reference Qwen/Qwen3-8B-Base --output outputs/code-oprd
```

Repeat with `--method grpo`, `opd`, and `kdrl`. The teacher argument is unnecessary for GRPO. Use `--task math_base` for the Qwen3-Base math comparison: teacher update 90, minibatch 32, coefficient horizon 75, and evaluation response limit 30,720, matching the original appendix scripts. Use `zebra_puzzles`, `color_cube_rotation`, and `binary_matrix` for the additional reasoning tasks.

Color Cube and Binary Matrix study reference-induced length bias. For OPRD, explicitly supply the teacher's step-30 reference for Color Cube or step-45 reference for Binary Matrix. Supply `Qwen/Qwen3-4B-Base` explicitly to reproduce their step-0 reference ablation. These later references belong to the same weak-teacher training run as the selected post-RL checkpoint.

Append `--dry-run` to inspect any launch. Additional Hydra overrides are supported as trailing `key=value` arguments; changing them can change the paper protocol.

## Multi-teacher transfer

Use four GPUs and one specialist per main reasoning task. Create a local JSON file:

```json
{
  "knights_knaves": "/path/to/knights-teacher-step105",
  "quantum_lock": "/path/to/quantum-teacher-step105",
  "string_manipulation": "/path/to/string-teacher-step75",
  "countdown": "/path/to/countdown-teacher-step105"
}
```

```bash
python -m oprd.train --task multi_teacher --method oprd \
  --teachers /path/to/teachers.json --output outputs/multi-oprd
```

The sampler draws 16 prompts per domain per batch and routes each domain to its teacher. Train for 300 updates and evaluate checkpoints 60, 120, 180, 240, and 300 on all four test sets. The corresponding `grpo` and `opd` recipes are the paper's Mix-RL and MOPD baselines. KDRL is also supported.

## Agent: cold-start SFT, RL, and visual tools

Download the pinned ViLaSR assets and preserve the documented corrupt SFT-sample exclusion:

```bash
python -m oprd.agent.download --exclude-corrupt-sft-sample
python -m oprd.agent.prepare --exclude-corrupt-sft-sample
python -m oprd.agent.holdout
```

The RL split contains 19,200 questions, 4,800 per family, arranged in batches of 16 per family. The full holdout is reconstructed from public row indices and verifies question, decoded-image, and scene disjointness from RL training. Expected counts are Maze 4,111; GPT4Scene 673; SR-91k 818; Spatial VQA 4,862. The 128-question `holdout_val` file is an inexpensive training diagnostic; use `holdout_full` for the paper table. Public cold-start SFT and pretraining exposure are separate from RL split disjointness.

The download includes the public **7B ViLaSR SFT student**, plus raw Qwen2.5-VL-3B-Instruct for reconstructing the teacher. The pinned ViLaSR release has no matching 3B cold-start checkpoint, so our 3B teacher received three-epoch SFT before GRPO. An existing completed 3B SFT checkpoint can be reused directly; reconstruction is only needed when that checkpoint is unavailable. See [initialization provenance](protocol_notes.md#agent-initialization-provenance).

To reconstruct it, create the separate SFT environment and run the explicit stage:

```bash
python3.12 -m venv .venv-vilasr-sft
.venv-vilasr-sft/bin/pip install -r requirements/agent-sft.txt
bash scripts/01_weak_to_strong/agent.sh --stage teacher-sft
```

The SFT recipe uses three epochs, full language-model tuning with frozen vision/projector, learning rate 1e-5, cosine decay, 0.1 warmup, cutoff 18,432, global batch 32, and four-GPU ZeRO-2. The section stage downloads/prepares the SFT corpus, audits its lengths and launches the separate SFT environment. W&B defaults to offline. The ordinary `train` stage never reruns SFT. For `--stage all`, supply `--teacher-initialization /path/to/3b-sft` to reuse a completed checkpoint; omit it to include SFT reconstruction. Return to the RL environment for teacher/student GRPO:

```bash
python -m oprd.train --task agent --teacher-training \
  --student datasets/agent/models/teacher3b_sft --output outputs/teachers/agent

python -m oprd.train --task agent --method oprd \
  --student datasets/agent/models/student7b \
  --teacher /path/to/agent-teacher-step450 \
  --teacher-reference datasets/agent/models/teacher3b_sft \
  --output outputs/agent-oprd

python -m oprd.agent.evaluate --model /path/to/agent-student-checkpoint \
  --output outputs/evaluation/agent-step225.jsonl
```

The OPRD reference is the teacher's **SFT checkpoint**, not the raw 3B Instruct model. The shared Object Mapper / Path Tracer implementation enforces 12 tool calls, 42 total images, 4,096 action tokens, 8,192 observation tokens, and a 16,384-token trajectory budget including observations. Tool observations receive no policy-gradient loss. Numeric answers use MRA; malformed or unfinished trajectories score zero. Report the equal-weight family macro average.

## Appendix baselines

| Paper name | Entry point | Key behavior |
|---|---|---|
| W2SR-P | `oprd.build_teacher_data` + `oprd.sft` | Verified, nontruncated teacher demonstrations; full SFT for 3 epochs |
| S2L-PO | `--method s2l_po` | 16 rollout-mixture phases; weak-policy log probabilities; initial-student KL 0.001 |
| OPSD | `--method opsd` | Correct weak-teacher draft, student fallback, EMA student 0.05; JSD 0.5 on top-100 plus tail |
| Direct-OPD | `--method direct_opd` | Frozen student top-16 dense teacher-shift reward; adaptive initial-policy anchor |
| W2S-OPD | `--method w2s_opd` | Frozen full-vocabulary proxy from student initial policy + teacher log-policy shift; proxy top-32 reverse KL |

For W2SR-P, the builder follows the same 150×64 prompt-occurrence stream and samples eight weak-teacher responses per occurrence. Only correct, valid, nontruncated responses are eligible:

```bash
python -m oprd.build_teacher_data --task knights_knaves \
  --data datasets/knights_knaves/train.parquet --model /path/to/teacher-step105 \
  --output-dir outputs/w2sr-p-data --max-new-tokens 8192 --max-model-len 10240
# Inspect the builder's printed output paths, then use its SFT parquet:
python -m oprd.sft --model Qwen/Qwen3-8B-Base \
  --data /path/to/teacher-demonstrations.parquet --max-length 10240 \
  --output outputs/w2sr-p
```

Math uses builder task `math_dapo`, maximum new tokens 20,480 and model length 22,528. The SFT learning rate is 2e-5, global batch 64, three epochs, with token-exact demonstration records. `oprd.sft` is the W2SR-P recipe; the Agent teacher's SFT recipe is separate.

The demonstration builder saves completed teacher responses and resumes generation when rerun with the same input data and generation settings. Use a new output directory if either changes. This applies to demonstration generation; `oprd.sft` starts a new SFT run (`resume_mode=disable`).

## Component studies and table reporting

| Paper experiment | Reproduction control |
|---|---|
| Scaling strength | `--lambda VALUE` |
| Positive branch only | `--branch positive_only` |
| Negative branch only | `--branch negative_only` |
| Fixed symmetric scaling | `--branch symmetric --lambda 1.0` |
| Decaying positive branch | `--branch decay_positive` |
| Default negative-branch ramp | `--branch ramp_negative --schedule-steps K` |
| Guidance construction | `--direction teacher_shift`, `opd`, or `opsd`; use weak-teacher step 60 for this comparison |
| Privileged-draft source | `--hint-source student`, `weak_teacher`, or `weak_teacher_then_student` |
| Teacher-strength sensitivity | Supply each selected checkpoint from the same teacher run with `--teacher` |
| Zero correction / reference equality | `--lambda 0`, or use the same teacher and reference checkpoint |
| Additional reference comparison | Set `--teacher-reference` to the base or later teacher checkpoint |

Numerical validation covers the rank-one formula, branch signs, duplicate support coordinates, schedules, zero teacher shift, and orthogonal gradients. The stationary-point argument concerns the paper's logit-space transform; it is not a general parameter-space convergence claim.

Section 4.3 student/teacher response-style comparisons and qualitative analyses are outside the public reproduction scope. Their scripts, feature extraction, and analysis code are not included.

Table aggregation is optional postprocessing; neither training nor evaluation invokes it automatically. It averages scores, never checkpoint weights. To request it, create a JSON list containing `step`, `benchmark`, and `mean_score` (0–100) for every cell. Missing or duplicate evaluations are rejected:

```bash
python -m oprd.evaluation.report --results outputs/math-results.json \
  --benchmarks aime24 aime25 hmmt25 olympiadbench --output outputs/math-table.json
```

Agent uses `--steps 45 90 135 180 225`; multi-teacher uses `--steps 60 120 180 240 300`. GPU cost/memory numbers require the original hardware and fixed-length profiling protocol; ordinary training timings are not interchangeable with that benchmark. Timing and memory profiling is not included in the release validation.

## Validation status

The maintainer suite covers CPU numerical, scorer, objective-update, and configuration checks; see the [validation record](validation.md). These do not replace full distributed GPU validation. Model downloads, all-dataset regeneration, teacher SFT/RL, end-to-end multimodal training, and the paper's full benchmark runs must be completed before claiming that this release reproduces the published scores. No trained checkpoints are silently replaced with base models.

## Storage and execution notes

Read this before launching `--stage all`, including when delegating runs to a coding agent. The default stage is `train`; main scripts run four methods sequentially. `all` also prepares data, trains teachers, and evaluates checkpoints. It is a full experiment. `--dry-run` prints the plan without loading models or creating outputs. `--steps` selects evaluation checkpoints and does not reduce training updates.

| Location | Saved contents |
|---|---|
| Hugging Face cache (`HF_HOME`, or the client's default cache) | Downloaded base models and dataset snapshots |
| `datasets/` (or `--data-root`) | Prepared parquet files, manifests, Code scorer assets; Agent images, downloads, and cold-start SFT models |
| `outputs/teachers/` (or `--teacher-root`) | Frozen-teacher training runs and their intermediate checkpoints |
| `outputs/<section>/.../seed-<seed>/` (or `--output-root`) | Student checkpoints, resume state, run metadata, and offline W&B logs |
| `global_step_<N>/actor/` | Model shards and metadata; optimizer, RNG/scheduler and method-specific state such as OPSD EMA remain only at the latest committed checkpoint |
| `global_step_<N>/actor/huggingface/` | A complete loadable model export, config, and tokenizer/processor; this duplicates model storage alongside the resume shards |
| `evaluation/*.jsonl`, `*.summary.json` | Generated responses, per-sample scores or trajectories, and aggregate metrics; HumanEval+ additionally writes EvalPlus inputs/results and `*.scores.json` |
| `<evaluation-output>.workers/` | Temporary worker response shards and logs; removed after successful merge, retained on failure for diagnosis |
| W2SR-P builder output | Prompt-occurrence manifest, selected responses/token IDs, candidate score summaries, and filtered token-exact SFT demonstrations |

RL model retention is **five evenly spaced student snapshots + the current best validation checkpoint**. For a 150-update run the five steps are 30/60/90/120/150; for Agent's 225 updates, 45/90/135/180/225; for multi-teacher's 300 updates, 60/120/180/240/300. An overlap is stored once. Best can be step 0. A previous best is removed when superseded unless it is also a scheduled snapshot. Teachers additionally preserve teacher/reference steps consumed by related recipes, including selected main teachers and teacher-strength variants. Teacher horizons are 150 for single-domain Math/reasoning and strong-to-weak, 300 per specialist in the multi-teacher section, 435 for Code, and 450 for Agent.

All RL runs evaluate before training and every 15 updates, plus the final update. Explicit model snapshots outside that cadence are also evaluated. Every evaluation saves a complete resume checkpoint; **only the latest committed evaluation checkpoint retains optimizer, RNG/scheduler, dataloader and distillation sidecars**. While training, this rolling checkpoint can add one temporary model snapshot beyond the five permanent models and best; unfinished saves and validation candidates can add temporary space. A failed new save does not authorize deleting the previous resume state. Normal cleanup runs only after checkpoint commitment. `max_actor_ckpt_to_keep=null` prevents worker-side rotation; trainer-side retention implements these rules.

W2SR-P SFT still keeps its final checkpoint; Agent cold-start SFT still saves every 200 steps with a two-checkpoint retention limit plus its final export. The RL validation/retention policy does not add a new held-out SFT validation protocol.

**Plan for hundreds of GB per RL run and TB-scale complete matrices.** Saved dtype matters: with FP32 master weights, retained model shards cost about 4 bytes/parameter, plus 2–4 bytes/parameter for a separate full export. Older 8B snapshots therefore cost roughly 48–64 GB each, and older 14B snapshots 84–112 GB each. One latest Adam optimizer adds about 8 bytes/parameter (64 GB for 8B; 112 GB for 14B). Five 14B snapshots plus one optimizer are roughly 532–672 GB, before a distinct best model, method-specific state, metadata or temporary saves (decimal GB, estimates rather than measured sizes). OPSD also needs one latest EMA teacher shard set. A new full save temporarily coexists with the previous resume state; reserve headroom for both. Measure the first checkpoint and extrapolate. Model export also needs CPU RAM, and evaluation merging briefly stores both worker shards and merged results.

Choose a large volume before starting, and use the same roots for every stage:

```bash
export HF_HOME=/large-volume/huggingface
bash scripts/01_weak_to_strong/code.sh --stage all --methods oprd \
  --data-root /large-volume/oprd/datasets \
  --teacher-root /large-volume/oprd/teachers \
  --output-root /large-volume/oprd/outputs --dry-run
# Inspect the plan and available space, then remove --dry-run to execute.
df -h /large-volume
du -sh /large-volume/oprd/* /large-volume/huggingface
```

Keep the five reported student exports, selected teacher exports, and any intermediate teacher/reference exports used by later ablations. Do not set retention to one before evaluating the required checkpoint set. To resume training, retain the complete checkpoint, including optimizer, RNG/scheduler, loader state, and method-specific sidecars; a Hugging Face export alone is only sufficient for inference or a new initialization. After a completed run, resume shards can be archived separately if resumption is no longer needed. The RL trainer prunes superseded non-best/non-scheduled models and old resume state automatically; it does not upload offline logs.

For resuming OPSD and OPSD-guided OPRD from the latest checkpoint, keep all four `distill_teacher_world_size_4_rank_*.pt` shards and matching `distill_resume_state_world_size_4_rank_*.pt` metadata inside the actor checkpoint. Missing/corrupt EMA state is an error; it cannot be reconstructed from the final student export. Restore a complete checkpoint, or use a fresh output directory for a deliberately new run.

RL training uses `resume_mode=auto`: reusing the same run path can resume it. Choose a new output root/seed for an independent run. Evaluation refuses to overwrite existing results. On a worker failure, inspect its retained logs before moving/removing the incomplete `.workers` directory and retrying. HumanEval+ generation and official scoring are separate steps within the evaluator; a scoring failure can leave generated responses without a summary. Never treat a partial output as a completed benchmark.
