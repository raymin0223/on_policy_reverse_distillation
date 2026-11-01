"""Paper Agent settings: ViLaSR SFT initialization, GRPO teacher450, 225 updates."""

from copy import copy


def build_overrides(a):
    from oprd.train import build_overrides as text_overrides, ROOT
    from oprd.paper import TASKS, checkpoint_steps
    from oprd.recipes import required_teacher_checkpoints

    if a.method not in {"grpo", "opd", "kdrl", "oprd"}:
        raise ValueError("Agent compares GRPO, OPD, KDRL, and OPRD")
    if not a.student:
        raise ValueError(
            "--student must be the downloaded ViLaSR 7B SFT checkpoint (or the 3B SFT checkpoint for teacher training)"
        )
    if a.method != "grpo" and not a.teacher_training and not a.teacher:
        raise ValueError("Agent teacher methods require --teacher at GRPO update 450")
    if a.method == "oprd" and not a.teacher_training and not a.teacher_reference:
        raise ValueError("Agent OPRD requires --teacher-reference at the teacher's 3-epoch SFT checkpoint")
    base = copy(a)
    base.task = "code"
    base.method = "grpo"
    base.teacher_training = False
    d = text_overrides(base)
    task = TASKS["agent"]
    required_steps = required_teacher_checkpoints("agent") if a.teacher_training else []
    steps = a.steps if a.steps is not None else max([task.steps, *required_steps])
    saved_steps = sorted(set(checkpoint_steps(steps)) | {step for step in required_steps if step <= steps})
    for key in list(d):
        if key.startswith("reward_model.reward_kwargs"):
            del d[key]
    default_output = (
        ROOT / "outputs" / "teachers" / "agent" / f"seed-{a.seed}"
        if a.teacher_training else ROOT / "outputs" / f"agent-{a.method}-seed{a.seed}"
    )
    out = (a.output or default_output).resolve()
    d.update(
        {
            "data.train_files": [str(a.data_root.resolve() / "agent/prepared/train.parquet")],
            "data.val_files": [str(a.data_root.resolve() / "agent/prepared/holdout_val.parquet")],
            "data.shuffle": False,
            "data.validation_shuffle": False,
            "data.image_patch_size": 14,
            "data.filter_overlong_prompts": False,
            "data.truncation": "error",
            "data.return_raw_input_ids": True,
            "trainer.validation_response_length": 16384,
            "data.max_prompt_length": 8192,
            "data.max_response_length": 16384,
            "data.drop_system_prompt": False,
            "data.return_multi_modal_inputs": False,
            "actor_rollout_ref.model.path": a.student,
            "actor_rollout_ref.actor.ppo_mini_batch_size": 64,
            "actor_rollout_ref.actor.use_dynamic_bsz": True,
            "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": 32768,
            "actor_rollout_ref.actor.fsdp_config.model_dtype": "fp32",
            "actor_rollout_ref.actor.fsdp_config.mixed_precision.param_dtype": "bf16",
            "actor_rollout_ref.actor.fsdp_config.mixed_precision.reduce_dtype": "fp32",
            "actor_rollout_ref.actor.fsdp_config.mixed_precision.buffer_dtype": "fp32",
            "actor_rollout_ref.rollout.prompt_length": 8192,
            "actor_rollout_ref.rollout.response_length": 16384,
            "actor_rollout_ref.rollout.max_model_len": 24576,
            "actor_rollout_ref.rollout.limit_images": 42,
            "actor_rollout_ref.rollout.gpu_memory_utilization": 0.6,
            "actor_rollout_ref.rollout.max_num_seqs": 64,
            "actor_rollout_ref.rollout.max_num_batched_tokens": 32768,
            "actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb": 2,
            "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz": True,
            "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu": 32768,
            "actor_rollout_ref.rollout.agent.num_workers": 16,
            "actor_rollout_ref.rollout.calculate_log_probs": False,
            "algorithm.rollout_correction.rollout_is": None,
            "actor_rollout_ref.rollout.agent.default_agent_loop": "vilasr_agent",
            "actor_rollout_ref.rollout.agent.agent_loop_config_path": str(
                ROOT / "oprd/agent/agent_loop.yaml"
            ),
            "actor_rollout_ref.rollout.multi_turn.enable": True,
            "actor_rollout_ref.rollout.val_kwargs.n": 4,
            "actor_rollout_ref.rollout.val_kwargs.temperature": 0.6,
            "actor_rollout_ref.rollout.val_kwargs.top_p": 0.95,
            "actor_rollout_ref.agent_protocol.train_max_action_length": 4096,
            "actor_rollout_ref.agent_protocol.val_max_action_length": 4096,
            "actor_rollout_ref.agent_protocol.max_observation_length": 8192,
            "actor_rollout_ref.agent_protocol.train_max_tool_turns": 12,
            "actor_rollout_ref.agent_protocol.val_max_tool_turns": 12,
            "actor_rollout_ref.agent_protocol.eval_seed": a.seed,
            "custom_reward_function.path": None,
            "reward_model.reward_manager": "vilasr_rule",
            "trainer.total_training_steps": steps,
            "trainer.save_steps": saved_steps,
            "trainer.best_validation_metric": "score/mean@4",
            "trainer.val_before_train": True,
            "trainer.test_freq": 15,
            "trainer.default_local_dir": str(out),
            "trainer.experiment_name": out.name,
        }
    )
    method = "grpo" if a.teacher_training else a.method
    if method == "grpo":
        d["actor_rollout_ref.actor.policy_loss.loss_mode"] = "agent_global_token_grpo"
        d["actor_rollout_ref.agent_grpo.enabled"] = True
    if method != "grpo":
        prefix = "actor_rollout_ref.agent_" + method + "."
        profile = {
            "enabled": True,
            "teacher_model_path": a.teacher,
            "teacher_max_model_len": 32768,
            "logprob_chunk_size": 128,
        }
        if method == "oprd":
            scale = 1.0 if a.scale is None else a.scale
            profile.update(
                base_model_path=a.teacher_reference,
                topk=10,
                lambda_plus=scale,
                lambda_minus_start=0.0,
                lambda_minus_end=scale,
                schedule_steps=a.schedule_steps or 45,
                support_chunk_size=32,
                head_chunk_size=128,
                skip_invalid_format=True,
            )
        elif method == "kdrl":
            profile.update(kdrl_beta_init=0.005, kdrl_beta_min=0.0, kdrl_delta=0.005 / 45, schedule_steps=45)
        d.update({prefix + k: v for k, v in profile.items()})
        d["actor_rollout_ref.actor.policy_loss.loss_mode"] = "agent_global_token_" + method
    return d
