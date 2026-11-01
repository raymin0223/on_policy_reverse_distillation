"""Launch paper experiments, or print their exact Hydra command with --dry-run."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
from .paper import TASKS, METHODS, REASONING_TASKS, checkpoint_steps
from .recipes import required_teacher_checkpoints
from .runtime import tracking

ROOT = Path(__file__).resolve().parents[1]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    tracking.add_arguments(p)
    p.add_argument("--task", choices=TASKS, required=True)
    p.add_argument("--method", choices=METHODS, default="oprd")
    p.add_argument("--student")
    p.add_argument("--teacher", help="Frozen post-RL Hugging Face checkpoint path or model ID")
    p.add_argument(
        "--teacher-reference",
        help="Teacher's own pre-RL checkpoint; later reference for length-bias ablations",
    )
    p.add_argument(
        "--teachers", type=Path, help="JSON mapping of the four domains to their frozen checkpoints"
    )
    p.add_argument("--data-root", type=Path, default=ROOT / "datasets")
    p.add_argument("--output", type=Path)
    p.add_argument(
        "--seed", type=int, help="Defaults to the task's original seed (42, or 1234 for Binary Matrix)"
    )
    p.add_argument("--gpus", type=int, choices=(4,), default=4)
    p.add_argument("--steps", type=int)
    p.add_argument("--lambda", dest="scale", type=float, default=None)
    p.add_argument("--schedule-steps", type=int)
    p.add_argument(
        "--branch",
        choices=("ramp_negative", "positive_only", "negative_only", "symmetric", "decay_positive"),
        default="ramp_negative",
    )
    p.add_argument("--direction", choices=("teacher_shift", "opd", "opsd"), default="teacher_shift")
    p.add_argument(
        "--hint-source",
        choices=("student", "weak_teacher", "weak_teacher_then_student"),
        default="weak_teacher_then_student",
    )
    p.add_argument(
        "--teacher-training",
        action="store_true",
        help="Train a teacher with GRPO; match the student horizon or reach the required paper checkpoints",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("overrides", nargs="*", help="Additional Hydra key=value overrides")
    return p


def build_overrides(a):
    task = TASKS[a.task]
    if a.seed is None:
        a.seed = task.seed
    if a.gpus < 1 or a.seed < 0:
        raise ValueError("gpus must be positive and seed nonnegative")
    if a.task == "agent":
        return build_agent_overrides(a)
    if a.task == "multi_teacher" and a.method not in {"grpo", "opd", "kdrl", "oprd"}:
        raise ValueError("The multi-teacher comparison uses Mix-RL, MOPD, KDRL, and OPRD")
    student = a.student or (task.teacher_base if a.teacher_training else task.student)
    required_steps = required_teacher_checkpoints(a.task) if a.teacher_training else []
    steps = a.steps if a.steps is not None else max([task.steps, *required_steps])
    saved_steps = sorted(set(checkpoint_steps(steps)) | {step for step in required_steps if step <= steps})
    method = "grpo" if a.teacher_training else a.method
    data_task = "math" if a.task == "math_base" else a.task
    root = a.data_root.resolve()
    train_files = [str(root / data_task / "train.parquet")]
    val_files = [str(root / data_task / "test.parquet")]
    if a.task.startswith("math"):
        val_files = [str(root / "aime24" / "test.parquet")]
    if a.task == "multi_teacher":
        train_files = [str(root / name / "train.parquet") for name in REASONING_TASKS]
        val_files = [str(root / name / "test.parquet") for name in REASONING_TASKS]
    default_output = (
        ROOT / "outputs" / "teachers" / a.task / f"seed-{a.seed}"
        if a.teacher_training else ROOT / "outputs" / f"{a.task}-{method}-seed{a.seed}"
    )
    output = (a.output or default_output).resolve()
    # Main text methods use token-level rollout importance correction.
    # See docs/protocol_notes.md for the current training settings.
    # S2L-PO, OPSD, Direct-OPD and W2S-OPD retain separate objective protocols.
    correct_rollout = method in {"grpo", "opd", "kdrl", "oprd"}
    d = {
        "seed": a.seed,
        "data.seed": a.seed,
        "data.train_files": train_files,
        "data.val_files": val_files,
        "data.train_batch_size": 64,
        "data.max_prompt_length": 2048,
        "data.max_response_length": task.response_length,
        "data.prompt_key": "prompt",
        "data.return_raw_chat": True,
        "data.prompt_format": "chat",
        "data.prepend_dropped_system_prompt_to_user": False,
        "data.drop_system_prompt": not task.system_prompt,
        "data.apply_chat_template_kwargs.enable_thinking": False,
        "actor_rollout_ref.model.path": student,
        "actor_rollout_ref.model.use_remove_padding": True,
        "actor_rollout_ref.model.enable_gradient_checkpointing": True,
        "actor_rollout_ref.model.override_config.attn_implementation": "flash_attention_2",
        "actor_rollout_ref.actor.strategy": "fsdp",
        "actor_rollout_ref.actor.ppo_mini_batch_size": task.mini_batch,
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.actor.ppo_epochs": 1,
        "actor_rollout_ref.actor.use_dynamic_bsz": False,
        "actor_rollout_ref.actor.optim.lr": 1e-6,
        "actor_rollout_ref.actor.optim.lr_warmup_steps": 10,
        "actor_rollout_ref.actor.optim.lr_scheduler_type": "constant",
        "actor_rollout_ref.actor.optim.weight_decay": 0.01,
        "actor_rollout_ref.actor.clip_ratio_low": 0.2,
        "actor_rollout_ref.actor.clip_ratio_high": 0.28,
        "actor_rollout_ref.actor.entropy_coeff": 0.0,
        "actor_rollout_ref.actor.use_kl_loss": False,
        "actor_rollout_ref.actor.grad_clip": 1.0,
        "actor_rollout_ref.actor.loss_agg_mode": "token-mean",
        "actor_rollout_ref.actor.policy_loss.loss_mode": "vanilla",
        "actor_rollout_ref.rollout.name": "vllm",
        "actor_rollout_ref.rollout.mode": "async",
        "actor_rollout_ref.rollout.tensor_model_parallel_size": 1,
        "actor_rollout_ref.rollout.n": 8,
        "actor_rollout_ref.rollout.temperature": 1.0,
        "actor_rollout_ref.rollout.top_p": 1.0,
        "actor_rollout_ref.rollout.top_k": -1,
        "actor_rollout_ref.rollout.response_length": task.response_length,
        "actor_rollout_ref.rollout.prompt_length": 2048,
        "actor_rollout_ref.rollout.gpu_memory_utilization": 0.6,
        # 64 prompts x 8 responses / 4 GPUs: cap each text engine at 128 sequences.
        "actor_rollout_ref.rollout.max_num_seqs": 128,
        "actor_rollout_ref.rollout.max_num_batched_tokens": 8192,
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.rollout.calculate_log_probs": correct_rollout,
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.rollout.val_kwargs.n": task.evaluation_n,
        "trainer.validation_response_length": task.evaluation_length,
        "actor_rollout_ref.rollout.max_model_len": 2048 + task.evaluation_length,
        "actor_rollout_ref.rollout.val_kwargs.temperature": task.evaluation_temperature,
        "actor_rollout_ref.rollout.val_kwargs.top_p": task.evaluation_top_p,
        "actor_rollout_ref.rollout.val_kwargs.top_k": 20,
        "actor_rollout_ref.rollout.val_kwargs.do_sample": True,
        "algorithm.adv_estimator": "grpo",
        "algorithm.norm_adv_by_std_in_grpo": False,
        "algorithm.use_kl_in_reward": False,
        "algorithm.rollout_correction.rollout_is": "token" if correct_rollout else None,
        "algorithm.rollout_correction.rollout_is_threshold": 2.0,
        "algorithm.rollout_correction.rollout_is_batch_normalize": False,
        "trainer.n_gpus_per_node": a.gpus,
        "trainer.nnodes": 1,
        "trainer.total_training_steps": steps,
        "trainer.total_epochs": 100,
        "trainer.save_freq": 0,
        "trainer.save_steps": saved_steps,
        "trainer.save_on_validation": True,
        "trainer.keep_only_latest_resume_state": True,
        "trainer.max_actor_ckpt_to_keep": None,
        "trainer.save_best_validation_checkpoint": True,
        "trainer.best_validation_metric": f"score/mean@{task.evaluation_n}",
        "trainer.best_validation_metric_aggregate": "mean",
        "trainer.best_validation_checkpoint_min_step": 0,
        "trainer.test_freq": task.validation_interval,
        **tracking.trainer_overrides(a),
        "trainer.experiment_name": output.name,
        "trainer.default_local_dir": str(output),
        "trainer.resume_mode": "auto",
        "trainer.val_before_train": True,
        "trainer.use_legacy_worker_impl": "enable",
        "actor_rollout_ref.actor.checkpoint.save_contents": ["model", "optimizer", "extra", "hf_model"],
        "custom_reward_function.path": str(ROOT / "oprd" / "rewards.py"),
        "custom_reward_function.name": "compute_score",
    }
    if a.task == "code":
        # Original DP4/B200 launcher: leave room for the 14B actor and two 8B frozen models.
        d.update(
            {
                "actor_rollout_ref.rollout.gpu_memory_utilization": 0.35,
                "actor_rollout_ref.rollout.max_num_seqs": 16,
                "actor_rollout_ref.rollout.max_num_batched_tokens": 8192,
                "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": 18432,
                "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu": 18432,
            }
        )
    if a.task == "multi_teacher":
        if a.gpus != 4:
            raise ValueError("Multi-teacher domain partitioning requires four GPUs")
        d.update(
            {
                "data.sampler.class_path": "pkg://verl.experimental.dataset.domain_balanced_sampler",
                "data.sampler.class_name": "DomainBalancedSampler",
                "data.domain_balanced_sampler.domains": list(REASONING_TASKS),
                "data.domain_balanced_sampler.samples_per_domain": 16,
                "data.domain_balanced_sampler.shuffle_within_batch": True,
            }
        )
    c = "actor_rollout_ref.actor.distillation."
    if method != "grpo":
        if a.task == "multi_teacher":
            if not a.teachers:
                raise ValueError("--teachers is required for the multi-teacher experiment")
            teachers = json.loads(a.teachers.read_text())
            if set(teachers) != set(REASONING_TASKS):
                raise ValueError("--teachers must contain exactly the four main reasoning domains")
            d.update(
                {
                    c + "multi_teacher_enabled": True,
                    c + "multi_teacher_domains": list(REASONING_TASKS),
                    c + "multi_teacher_model_paths": [teachers[name] for name in REASONING_TASKS],
                }
            )
        elif not a.teacher:
            raise ValueError(
                "--teacher must point to the frozen post-RL teacher; a raw base model is not a substitute"
            )
        else:
            d[c + "teacher_model_path"] = a.teacher
        d.update(
            {
                c + "teacher_regularization": "teacher",
                c + "teacher_update_rate": 0.0,
                c + "alpha": 1.0,
                c + "full_logit_distillation": False,
            }
        )
    warmup = a.schedule_steps or task.schedule_steps
    if method in {"opd", "direct_opd", "w2s_opd"}:
        d["actor_rollout_ref.actor.policy_loss.loss_mode"] = "kd"
        d[c + "sampled_rkl_use_importance_sampling"] = True
    if method == "opd":
        d.update({c + "is_clip": 2.0, c + "is_clip_low": 0.5, c + "is_clip_high": 2.0})
    if method == "kdrl":
        d.update(
            {
                c + "kdrl_enabled": True,
                c + "kdrl_beta_init": 0.005,
                c + "kdrl_beta_min": 0.0,
                c + "kdrl_delta": 0.005 / warmup,
            }
        )
    if method in {"oprd", "direct_opd", "w2s_opd"}:
        reference = a.teacher_reference or task.teacher_base
        if (
            a.task in {"color_cube_rotation", "binary_matrix"}
            and method == "oprd"
            and not a.teacher_reference
        ):
            raise ValueError(
                "Specify --teacher-reference: paper defaults use teacher step 30 (Color Cube) or 45 (Binary Matrix); pass the base model explicitly for the step-0 ablation"
            )
        d[c + "oprd_base_model_path"] = reference
        if method == "direct_opd":
            d.update(
                {
                    c + "delta_target_enabled": True,
                    c + "delta_teacher_base_model_path": reference,
                    c + "direct_opd_mode": "topk",
                    c + "direct_opd_top_k": 16,
                    c + "delta_anchor_adaptive_alpha_enabled": True,
                    c + "delta_anchor_alpha_init": 2.5,
                    c + "delta_anchor_alpha_epsilon": 0.01,
                    c + "delta_anchor_alpha_min": 0.5,
                    c + "delta_anchor_alpha_max": 2.5,
                    "actor_rollout_ref.actor.use_kl_loss": True,
                    "actor_rollout_ref.actor.kl_loss_coef": 1.0,
                    "actor_rollout_ref.actor.kl_loss_type": "low_var_kl",
                }
            )
        elif method == "w2s_opd":
            d.update(
                {
                    c + "w2s_opd_enabled": True,
                    c + "w2s_positive_model_path": a.teacher,
                    c + "w2s_negative_model_path": reference,
                    c + "w2s_student_base_model_path": student,
                    c + "w2s_opd_teacher_top_k": 32,
                    c + "w2s_amplification": 1.0,
                    c + "sampled_rkl_use_importance_sampling": False,
                }
            )
        else:
            scale = task.oprd_lambda if a.scale is None else a.scale
            if scale < 0 or warmup < 1:
                raise ValueError("OPRD requires nonnegative lambda and a positive schedule horizon")
            plus_start = plus_end = scale
            minus_start, minus_end = 0.0, scale
            if a.branch == "positive_only":
                minus_end = 0.0
            elif a.branch == "negative_only":
                plus_start = plus_end = 0.0
                minus_start = scale
            elif a.branch == "symmetric":
                minus_start = scale
            elif a.branch == "decay_positive":
                plus_end = 0.0
                minus_end = 0.0
            d.update(
                {
                    c + "oprd_enabled": True,
                    c + "oprd_lambda": scale,
                    c + "oprd_topk": 10,
                    c + "oprd_h1_source": {
                        "teacher_shift": "teacher_base",
                        "opd": "teacher_student",
                        "opsd": "opsd_kl",
                    }[a.direction],
                    c + "oprd_h_normalization": "uniform_center_l2"
                    if a.direction == "teacher_shift"
                    else "policy_gradient_l2"
                    if a.direction == "opd"
                    else "l2",
                    c + "oprd_signed_lambda_schedule_enabled": True,
                    c + "oprd_signed_lambda_schedule_steps": warmup,
                    c + "oprd_lambda_plus_start": plus_start,
                    c + "oprd_lambda_plus_end": plus_end,
                    c + "oprd_lambda_minus_start": minus_start,
                    c + "oprd_lambda_minus_end": minus_end,
                    c + "oprd_skip_invalid_format": True,
                }
            )
    if method == "opsd" or method == "oprd" and a.direction == "opsd":
        d.update(
            {
                "actor_rollout_ref.actor.policy_loss.loss_mode": "opsd"
                if method == "opsd"
                else "guided_opsd",
                c + "teacher_regularization": "ema",
                c + "teacher_update_rate": 0.05,
                c + "alpha": 0.5,
                c + "full_logit_distillation": True,
                c + "distillation_topk": 100,
                c + "distillation_add_tail": True,
                c + "dont_reprompt_on_self_success": False,
                c + "solution_selection": "random",
                c + "success_reward_threshold": 0.5,
                c + "max_reprompt_len": 2048 + task.response_length + 512,
                c + "oprd_opsd_hint_source": a.hint_source,
                c + "oprd_opsd_weak_teacher_num_rollouts": 8,
            }
        )
    if method == "s2l_po":
        d.update(
            {
                "algorithm.s2l_po.enabled": True,
                "algorithm.s2l_po.weak_model_path": a.teacher,
                "algorithm.s2l_po.num_phases": 16,
                "algorithm.s2l_po.t_mix": 75,
                "algorithm.s2l_po.group_size": 8,
                "actor_rollout_ref.actor.use_kl_loss": True,
                "actor_rollout_ref.actor.kl_loss_coef": 0.001,
            }
        )
    if task.length_penalty_buffer:
        d.update(
            {
                "reward_model.reward_manager": "dapo",
                # Public reruns use the same task penalty for teachers and students.
                "reward_model.reward_kwargs.overlong_buffer_cfg.enable": True,
                "reward_model.reward_kwargs.overlong_buffer_cfg.len": task.length_penalty_buffer,
                "reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor": 1.0,
                "reward_model.reward_kwargs.max_resp_len": task.response_length,
            }
        )
    return d


def build_agent_overrides(a):
    from .agent.launch import build_overrides

    return build_overrides(a)


def command(a):
    d = build_overrides(a)
    encode = lambda value: json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return (
        [sys.executable, "-m", "oprd.runtime"]
        + ["++" + key + "=" + encode(value) for key, value in d.items()]
        + a.overrides
    )


def main():
    p = parser()
    a = p.parse_args()
    tracking.configure_environment(a)
    try:
        cmd = command(a)
    except (ValueError, FileNotFoundError) as e:
        p.error(str(e))
    if a.dry_run:
        print(shlex.join(cmd))
        return
    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
