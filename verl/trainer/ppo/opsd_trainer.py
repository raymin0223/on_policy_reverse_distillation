import re
from collections import defaultdict
from typing import Any, Optional
import torch
from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.dataset.rl_dataset import build_plain_completion_prompt, use_plain_completion_prompt_format
from verl.utils.model import compute_position_id_with_mask


class OPSDRayPPOTrainer(RayPPOTrainer):
    """OPSD-specific trainer with distillation reprompt construction."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        distillation_cfg = self.config.actor_rollout_ref.actor.get("distillation", {})
        self.tokenizer.truncation_side = distillation_cfg.get("reprompt_truncation", "error")

    @staticmethod
    def _collect_feedback(
        include_environment_feedback: bool, reward_extra_infos_dict: Optional[dict[str, Any]], batch_size: int
    ) -> list[Any]:
        feedback_list: list[Any] = [None] * batch_size
        if include_environment_feedback and reward_extra_infos_dict is not None:
            raw_feedback = reward_extra_infos_dict.get("feedback", [])
            for i in range(min(len(raw_feedback), batch_size)):
                if raw_feedback[i] and isinstance(raw_feedback[i], str) and raw_feedback[i].strip():
                    feedback_list[i] = raw_feedback[i]
        return feedback_list

    def _collect_solutions_by_uid(
        self, batch: DataProto, reward_tensor: torch.Tensor, success_reward_threshold: float
    ) -> dict[Any, list[int]]:
        seq_scores = reward_tensor.sum(dim=-1).detach().cpu().numpy()
        uids = batch.non_tensor_batch["uid"]
        success_by_uid: dict[Any, list[int]] = defaultdict(list)
        for idx, uid in enumerate(uids):
            if seq_scores[idx] >= success_reward_threshold:
                success_by_uid[uid].append(idx)
        return success_by_uid

    @staticmethod
    def _remove_thinking_trace(text: str) -> str:
        return re.sub("<\\s*think\\b[^>]*>.*?</\\s*think\\s*>\\s*", "", text, flags=re.DOTALL | re.IGNORECASE)

    @staticmethod
    def _strip_thinking_tags(text: str) -> str:
        return re.sub("</?\\s*think\\b[^>]*>", "", text, flags=re.IGNORECASE)

    @classmethod
    def _prepare_demonstration_text(
        cls,
        text: str,
        *,
        remove_thinking_from_demonstration: bool = False,
        strip_thinking_tags_from_demonstration: bool = False,
    ) -> str:
        if remove_thinking_from_demonstration:
            return cls._remove_thinking_trace(text)
        if strip_thinking_tags_from_demonstration:
            return cls._strip_thinking_tags(text)
        return text

    @staticmethod
    def _strip_plain_completion_solution_marker(prompt_text: str) -> str:
        return re.sub("\\n*Solution:\\s*$", "", prompt_text.rstrip())

    @staticmethod
    def _solution_selection_mode(distillation_cfg) -> str:
        selection = str(distillation_cfg.get("solution_selection", "first")).strip().lower()
        selection = {"sample": "random", "rand": "random"}.get(selection, selection)
        if selection not in {"first", "random", "shortest"}:
            raise ValueError(
                f"distillation.solution_selection must be one of ['first', 'random', 'shortest'], got {selection!r}"
            )
        return selection

    def _get_solution(
        self,
        idx: int,
        success_by_uid: dict[Any, list[int]],
        uids: list[Any],
        response_texts: list[str],
        dont_reprompt_on_self_success: bool = False,
        remove_thinking_from_demonstration: bool = False,
        strip_thinking_tags_from_demonstration: bool = False,
        solution_selection: str = "first",
    ) -> Optional[str]:
        uid = uids[idx]
        solution_idxs = success_by_uid[uid]
        if dont_reprompt_on_self_success:
            solution_idxs = [j for j in solution_idxs if j != idx]
        if len(solution_idxs) == 0:
            return None
        if solution_selection == "random":
            solution_idx = solution_idxs[int(torch.randint(low=0, high=len(solution_idxs), size=(1,)).item())]
        elif solution_selection == "shortest":
            solution_idx = min(solution_idxs, key=lambda j: len(response_texts[j]))
        else:
            solution_idx = solution_idxs[0]
        solution_str = response_texts[solution_idx]
        return self._prepare_demonstration_text(
            solution_str,
            remove_thinking_from_demonstration=remove_thinking_from_demonstration,
            strip_thinking_tags_from_demonstration=strip_thinking_tags_from_demonstration,
        )

    @classmethod
    def _get_solution_from_text_pool(
        cls,
        uid: Any,
        solution_texts_by_uid: dict[Any, list[str]],
        *,
        remove_thinking_from_demonstration: bool = False,
        strip_thinking_tags_from_demonstration: bool = False,
        solution_selection: str = "first",
    ) -> Optional[str]:
        """Select a demonstration from an externally generated correct-rollout pool."""
        candidates = [str(text) for text in solution_texts_by_uid.get(uid, []) if str(text).strip()]
        if not candidates:
            return None
        if solution_selection == "random":
            solution = candidates[int(torch.randint(low=0, high=len(candidates), size=(1,)).item())]
        elif solution_selection == "shortest":
            solution = min(candidates, key=len)
        else:
            solution = candidates[0]
        return cls._prepare_demonstration_text(
            solution,
            remove_thinking_from_demonstration=remove_thinking_from_demonstration,
            strip_thinking_tags_from_demonstration=strip_thinking_tags_from_demonstration,
        )

    @staticmethod
    def _build_on_policy_distillation_batch(batch: DataProto) -> tuple[DataProto, dict[str, float]]:
        device = batch.batch["input_ids"].device
        batch_size = batch.batch.batch_size[0]
        on_policy_distillation_mask = torch.ones((batch_size,), dtype=torch.float32, device=device)
        return (
            DataProto.from_dict(
                tensors={
                    "on_policy_teacher_input_ids": batch.batch["input_ids"],
                    "on_policy_teacher_attention_mask": batch.batch["attention_mask"],
                    "on_policy_teacher_position_ids": batch.batch["position_ids"],
                    "on_policy_distillation_mask": on_policy_distillation_mask,
                }
            ),
            {"distillation/on_policy_target_sample_fraction": 1.0},
        )

    def _maybe_build_distillation_batch(
        self,
        batch: DataProto,
        reward_tensor: torch.Tensor,
        reward_extra_infos_dict: Optional[dict[str, list]] = None,
    ) -> Optional[tuple[DataProto, dict[str, float]]]:
        distillation_cfg = self.config.actor_rollout_ref.actor.get("distillation", None)
        loss_mode = self.config.actor_rollout_ref.actor.policy_loss.get("loss_mode", "vanilla")
        if distillation_cfg is None or loss_mode != "opsd":
            return None
        enable_on_policy_distill = False
        mix_alpha = 1.0
        mix_alpha_schedule = "fixed".lower()
        mix_schedule_is_linear = 1.0 if mix_alpha_schedule == "linear" else 0.0
        mix_schedule_is_step = 1.0 if mix_alpha_schedule == "step" else 0.0
        opsd_batch, opsd_metrics = self._build_opsd_distillation_batch(
            batch=batch,
            reward_tensor=reward_tensor,
            reward_extra_infos_dict=reward_extra_infos_dict,
            metric_prefix="distillation",
        )
        metrics = {
            **opsd_metrics,
            "distillation/on_policy_enabled": 1.0 if enable_on_policy_distill else 0.0,
            "distillation/mix_alpha_config": mix_alpha,
            "distillation/mix_alpha_schedule": mix_schedule_is_linear,
            "distillation/mix_alpha_schedule_step": mix_schedule_is_step,
        }
        if not enable_on_policy_distill:
            return (opsd_batch, metrics)
        on_policy_batch, on_policy_metrics = self._build_on_policy_distillation_batch(batch=batch)
        metrics.update(on_policy_metrics)
        return (opsd_batch.union(on_policy_batch), metrics)

    def _build_opsd_distillation_batch(
        self,
        batch: DataProto,
        reward_tensor: torch.Tensor,
        reward_extra_infos_dict: Optional[dict[str, list]] = None,
        metric_prefix: str = "distillation",
        solution_texts_by_uid: Optional[dict[Any, list[str]]] = None,
        exclude_self_solution: Optional[bool] = None,
    ) -> tuple[DataProto, dict[str, float]]:
        distillation_cfg = self.config.actor_rollout_ref.actor.distillation
        device = batch.batch["input_ids"].device
        response_mask = batch.batch["response_mask"]
        responses = batch.batch["responses"]
        response_ids_cpu = responses.detach().cpu().tolist()
        response_lens = response_mask.sum(dim=-1).detach().cpu().tolist()
        trimmed_response_ids = [
            response_ids[: int(valid_len)]
            for response_ids, valid_len in zip(response_ids_cpu, response_lens, strict=True)
        ]
        response_texts = self.tokenizer.batch_decode(trimmed_response_ids, skip_special_tokens=True)
        prompt_texts = [msgs[-1]["content"] for msgs in batch.non_tensor_batch["raw_prompt"]]
        batch_size = batch.batch.batch_size[0]
        feedback_list = self._collect_feedback(
            include_environment_feedback=False,
            reward_extra_infos_dict=reward_extra_infos_dict,
            batch_size=batch_size,
        )
        solution_selection = self._solution_selection_mode(distillation_cfg)
        success_by_uid = self._collect_solutions_by_uid(
            batch, reward_tensor, success_reward_threshold=distillation_cfg.success_reward_threshold
        )
        if solution_texts_by_uid is None:
            dont_reprompt_on_self_success = (
                distillation_cfg.dont_reprompt_on_self_success
                if exclude_self_solution is None
                else bool(exclude_self_solution)
            )
            solution_strs = [
                self._get_solution(
                    i,
                    success_by_uid,
                    batch.non_tensor_batch["uid"],
                    response_texts,
                    dont_reprompt_on_self_success,
                    distillation_cfg.get("remove_thinking_from_demonstration", False),
                    distillation_cfg.get("strip_thinking_tags_from_demonstration", True),
                    solution_selection,
                )
                for i in range(batch_size)
            ]
        else:
            solution_strs = [
                self._get_solution_from_text_pool(
                    batch.non_tensor_batch["uid"][i],
                    solution_texts_by_uid,
                    remove_thinking_from_demonstration=distillation_cfg.get(
                        "remove_thinking_from_demonstration", False
                    ),
                    strip_thinking_tags_from_demonstration=distillation_cfg.get(
                        "strip_thinking_tags_from_demonstration", True
                    ),
                    solution_selection=solution_selection,
                )
                for i in range(batch_size)
            ]
        feedback_only_without_solution = False
        feedback_used = [
            feedback_list[i] is not None and (not feedback_only_without_solution or solution_strs[i] is None)
            for i in range(batch_size)
        ]
        distillation_active = [solution_strs[i] is not None or feedback_used[i] for i in range(batch_size)]

        def _build_teacher_message(i: int) -> list[dict]:
            system_messages = batch.non_tensor_batch["raw_prompt"][i][:-1]
            if not distillation_active[i]:
                return system_messages + [{"role": "user", "content": ""}]
            has_solution = solution_strs[i] is not None
            has_feedback = feedback_list[i] is not None
            use_feedback = has_feedback and (not feedback_only_without_solution or not has_solution)
            solution_section = ""
            if has_solution:
                solution_section = distillation_cfg.solution_template.format(
                    successful_previous_attempt=solution_strs[i]
                )
            feedback_section = ""
            if use_feedback:
                feedback_section = "\nThe following is feedback from your unsuccessful earlier attempt:\n\n{feedback_raw}\n\n".format(
                    feedback_raw=feedback_list[i]
                )
            if use_feedback or has_solution:
                reprompt_text = distillation_cfg.reprompt_template.format(
                    prompt=prompt_texts[i], solution=solution_section, feedback=feedback_section
                )
            else:
                reprompt_text = prompt_texts[i]
            return system_messages + [{"role": "user", "content": reprompt_text}]

        if use_plain_completion_prompt_format(self.config.data):
            include_system = bool(self.config.data.get("plain_completion_include_system", True))
            teacher_prompt_texts = []
            for i in range(batch_size):
                if not distillation_active[i]:
                    teacher_prompt_texts.append(
                        build_plain_completion_prompt(
                            batch.non_tensor_batch["raw_prompt"][i], include_system=include_system
                        )
                    )
                    continue
                base_prompt = self._strip_plain_completion_solution_marker(
                    build_plain_completion_prompt(
                        batch.non_tensor_batch["raw_prompt"][i], include_system=include_system
                    )
                )
                solution_section = ""
                if solution_strs[i] is not None:
                    solution_section = distillation_cfg.solution_template.format(
                        successful_previous_attempt=solution_strs[i]
                    )
                feedback_section = ""
                if feedback_used[i]:
                    feedback_section = "\nThe following is feedback from your unsuccessful earlier attempt:\n\n{feedback_raw}\n\n".format(
                        feedback_raw=feedback_list[i]
                    )
                reprompt_text = distillation_cfg.reprompt_template.format(
                    prompt=base_prompt, solution=solution_section, feedback=feedback_section
                ).rstrip()
                teacher_prompt_texts.append(f"{reprompt_text}\n\nSolution:")
            teacher_prompt = self.tokenizer(
                teacher_prompt_texts,
                add_special_tokens=False,
                return_tensors="pt",
                max_length=distillation_cfg.max_reprompt_len,
                padding=True,
                truncation=True,
            )
        else:
            messages = [_build_teacher_message(i) for i in range(batch_size)]
            enable_thinking = (
                self.config.data.apply_chat_template_kwargs.get("enable_thinking", True)
                if self.config.data.apply_chat_template_kwargs
                else True
            )
            teacher_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
                continue_final_message=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
                max_length=distillation_cfg.max_reprompt_len,
                padding=True,
                truncation=True,
            )
        teacher_input_ids = torch.cat([teacher_prompt["input_ids"].to(device), responses], dim=1)
        teacher_attention_mask = torch.cat(
            [teacher_prompt["attention_mask"].to(device), response_mask], dim=1
        )
        teacher_position_ids = compute_position_id_with_mask(teacher_attention_mask)
        distillation_mask = torch.tensor(distillation_active, dtype=torch.float32, device=device)
        uids = set(batch.non_tensor_batch["uid"])
        if solution_texts_by_uid is None:
            successful_uid_count = len([uid for uid in uids if len(success_by_uid[uid]) > 0])
        else:
            successful_uid_count = len([uid for uid in uids if solution_texts_by_uid.get(uid)])
        num_with_feedback_available = sum((1 for f in feedback_list if f is not None))
        num_with_feedback_used = sum((1 for f in feedback_used if f))
        num_with_solution = sum((1 for s in solution_strs if s is not None))
        metrics = {
            f"{metric_prefix}/success_group_fraction": successful_uid_count / max(len(uids), 1),
            f"{metric_prefix}/success_sample_fraction": num_with_solution / batch_size,
            f"{metric_prefix}/feedback_available_fraction": num_with_feedback_available / batch_size,
            f"{metric_prefix}/feedback_used_fraction": num_with_feedback_used / batch_size,
            f"{metric_prefix}/reprompt_sample_fraction": distillation_mask.float().mean().item(),
        }
        return (
            DataProto.from_dict(
                tensors={
                    "teacher_input_ids": teacher_input_ids,
                    "teacher_attention_mask": teacher_attention_mask,
                    "teacher_position_ids": teacher_position_ids,
                    "distillation_mask": distillation_mask,
                }
            ),
            metrics,
        )
