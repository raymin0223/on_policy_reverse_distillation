# Training protocol

The executable task defaults are in `oprd/paper.py`, experiment variants in `oprd/recipes.py`, and runtime options in `oprd/train.py`. The [release note](../README.md#release-note) describes the scope of this codebase. All RL runs use one node with four B200 GPUs, FSDP DP4, rollout TP1, and full finetuning.

## Task and appendix settings

- Main Math uses Qwen3-4B teacher update 75, minibatch 64, coefficient horizon 30, and evaluation response limit 38,912.
- Math-Base uses Qwen3-4B-Base teacher update 90, minibatch 32, coefficient horizon 75, and evaluation response limit 30,720.
- Strong-to-weak Math uses Qwen3-8B teacher update 105 with a Qwen3-1.7B student. Knights & Knaves uses Qwen3-8B-Base teacher update 105 with the instruction-tuned Qwen3-0.6B student. Both students use minibatch 64 and OPRD horizon 45. Math teacher minibatch is 64; Knights & Knaves teacher minibatch is 32. All use prompt batch 64 and eight responses per prompt.
- The §4.4 vanishing-reward study uses Qwen3-1.7B-Base, minibatch 32 and coefficient horizon 75, with the 8B-Base teacher at update 105.
- Teacher-strength experiments use Knights & Knaves and Quantum Lock teachers at updates 15/60/105/150.
- Color Cube uses teacher reference update 30; Binary Matrix uses update 45. Reference ablations compare these with the teacher's pre-RL policy.
- Binary Matrix uses λ=1.0 and student/data-order seed 1234. Other text tasks use λ=0.5 and seed 42. Dataset generation uses seed 0; section-level teacher training independently defaults to seed 42.

## Training rewards and format handling

Reward scoring runs concurrently in the asynchronous rollout loop. Math/Math-Base and Code teacher and student training use a soft length penalty over the last 4,096 response tokens. For Code, the penalty is zero through 12,288 tokens and reaches −1 at 16,384. Reasoning and Agent use no length penalty. Online and offline evaluation use raw task scores without training penalties.

OPRD leaves the verifier-driven GRPO gradient unchanged on responses marked `incorrect_format`; these responses remain in the loss and token-mean denominator. OPRD projection is applied to format-valid responses, including incorrect answers. Math/reasoning check the boxed-answer format; Code checks the required Python code-block format. Syntax errors or failing tests alone are not Code format failures. Agent checks the complete terminal/tool protocol and skips projection on every action token of an invalid trajectory; observation tokens remain excluded from the loss.

In an internal comparison, we did not observe a clear performance difference between applying OPRD guidance to invalid-format responses and leaving their GRPO gradients unchanged. We therefore enable invalid-format skipping by default across tasks. This comparison concerns how invalid-format responses are handled within OPRD, not the overall performance of GRPO versus OPRD.

The switches are `actor_rollout_ref.actor.distillation.oprd_skip_invalid_format=true` for text and `actor_rollout_ref.agent_oprd.skip_invalid_format=true` for Agent. Set the corresponding switch to `false` to apply OPRD regardless of format. This controls only OPRD projection, not the task scorer. A zero underlying GRPO gradient remains zero with either setting.

## Strong-to-weak KDRL schedule

Both strong-to-weak KDRL runs start at β=0.005, keep β≥0.001, and train for 150 updates:

| Task | Decrement per update | First update at the floor |
|---|---:|---:|
| Math, 8B → 1.7B | 0.0001 | 41 |
| Knights & Knaves, 8B-Base → 0.6B | 0.000088889 | 46 |

The actor uses `β(step) = max(0.005 − decrement × (step − 1), 0.001)`. These overrides apply only to KDRL in the two strong-to-weak recipes. Other coefficient schedules are defined by the task and method recipe.

## Rollout correction and distillation inputs

GRPO, OPD, KDRL and OPRD text comparisons, including teacher training, use token-level rollout importance correction with threshold 2.0 and no batch normalization. The detached weight is `min(exp(old_log_prob − rollout_log_prob), 2.0)`. OPD also clips its current/old policy ratio to [0.5, 2.0]. KDRL applies rollout weights to its RL term; its K2 term is unweighted.

S2L-PO, pure OPSD, Direct-OPD, W2S-OPD and Agent use no additional rollout correction. OPRD with an OPSD guidance direction keeps text rollout correction. SFT and evaluation do not use rollout-loss correction.

OPSD and OPSD-guided OPRD allow 23,040 reprompt tokens for Math and 10,752 for reasoning, including prompt, demonstration and template allowance. Resumption restores the EMA teacher together with the student.

## Loss normalization (public release)

All RL/distillation methods use the valid response-token count over the whole optimizer minibatch across data-parallel ranks. Each rank backpropagates its partial token-loss sum multiplied by DP size and divided by that global count; FSDP gradient averaging produces the global token mean. There is no additional averaging by microbatch count. Padding and Agent observations have response mask zero. Objective-specific exclusions contribute zero numerator within the common response-token denominator.

## Agent initialization provenance

The student starts from the released ViLaSR 7B cold-start SFT checkpoint. The teacher starts from Qwen2.5-VL-3B-Instruct followed by three-epoch cold-start SFT, then GRPO; OPRD uses that 3B SFT checkpoint as its reference. The pinned upstream assets and download checks are recorded in `oprd/agent/vilasr_assets.lock.json`.

Teacher SFT uses 34,745 trajectories after the documented corrupt-image exclusion, full language-model training, frozen vision/projector, batch 32, three epochs, LR 1e-5 and cosine scheduling. Both reconstruction and reuse of an existing matching SFT checkpoint are supported.

## Validation, checkpoints and reporting

RL runs validate before training, every 15 updates and at the final update. Five evenly spaced snapshots and the current best validation model are retained, with no duplicate when they overlap. Teacher runs additionally retain the selected teacher/reference checkpoints consumed by related recipes. Off-cadence snapshot points are also evaluated.

Teacher horizons match their students unless a required paper checkpoint is later: 150 for single-domain Math/reasoning and strong-to-weak, 435 for Code (student 150), and 450 for Agent (student 225). The multi-teacher section runs each specialist for 300 updates. Students read the configured frozen teacher checkpoints, including 75/105 for the specialists; they do not automatically select the final or best teacher.

Only the latest committed evaluation checkpoint keeps optimizer, RNG/scheduler, dataloader and method-specific resume state. Older scheduled/best snapshots retain model weights and exports. Pending saves and validation candidates require temporary space. Completed runs exit when resumed at or beyond their requested horizon; a larger horizon continues from the latest resume checkpoint.

Best selection uses raw `score/mean@n`: Math n=16, Code n=4, reasoning n=1 and Agent's online diagnostic n=4. Multi-domain validation averages domains equally. Agent online validation uses 128 questions × 4 samples; paper evaluation uses the full holdout × 1 sample. Offline evaluation uses DP4/TP1 and merges question shards before scoring/aggregation.

`oprd.evaluation.report` optionally averages already-computed checkpoint scores for table reporting. It never averages weights or changes training/model selection. See the [reproduction guide](reproduction.md#storage-and-execution-notes) for output files and disk budgeting.

## Data and method scope

Math source revisions and hashes are pinned in `data/math_sources.lock.json`; preparation verifies source bytes and ordered, formatted inputs. Code and Agent also use pinned assets and checksums. See [data provenance](reproduction.md#data-and-evaluation-provenance).

Supported methods are GRPO, OPD, KDRL, OPRD, S2L-PO, OPSD, Direct-OPD, W2S-OPD and W2SR-P SFT, plus the listed guidance/reference/branch experiments. Agent and multi-teacher compare the four main methods. The §4.3 qualitative and response-style analysis is outside the release scope. Full finetuning is required; adapter checkpoints and LoRA/PEFT options are unsupported.
