from collections import defaultdict
from typing import Any, Optional
import numpy as np
import torch
from verl import DataProto
from verl.trainer.ppo.teacher_rollout_trainer import TeacherRolloutRayPPOTrainer


class GuidedOPSDRayPPOTrainer(TeacherRolloutRayPPOTrainer):
    """GRPO trainer that supplies correct-rollout hints to the OPSD-OPRD hook.

    Student rollouts are always generated from the original prompt.  A separate
    reprompt is built only for the frozen/EMA scoring model used to construct the
    KL-gradient direction inside the actor update.
    """

    @staticmethod
    def _normalize_hint_source(value: Any) -> str:
        source = str(value or "student").strip().lower().replace("-", "_")
        source = {
            "self": "student",
            "student_success": "student",
            "self_success": "student",
            "teacher": "weak_teacher",
            "external_teacher": "weak_teacher",
            "weak": "weak_teacher",
            "teacher_then_student": "weak_teacher_then_student",
            "weak_then_student": "weak_teacher_then_student",
            "weak_teacher_fallback_student": "weak_teacher_then_student",
        }.get(source, source)
        if source not in {"student", "weak_teacher", "weak_teacher_then_student"}:
            raise ValueError(
                f"distillation.oprd_opsd_hint_source must be one of ['student', 'weak_teacher', 'weak_teacher_then_student'], got {source!r}."
            )
        return source

    def _should_use_teacher_vllm_rollout(self) -> bool:
        distillation_cfg = self.config.actor_rollout_ref.actor.get("distillation", None)
        loss_mode = self.config.actor_rollout_ref.actor.policy_loss.get("loss_mode", "vanilla")
        if distillation_cfg is not None and loss_mode == "guided_opsd":
            if self._normalize_hint_source(distillation_cfg.get("oprd_opsd_hint_source", "student")) not in {
                "weak_teacher",
                "weak_teacher_then_student",
            }:
                return False
        return super()._should_use_teacher_vllm_rollout()

    def _build_student_solution_pool(
        self, batch: DataProto, reward_tensor: torch.Tensor, *, reward_threshold: float
    ) -> dict[Any, list[str]]:
        """Collect verified current-policy rollouts by prompt uid."""
        response_mask = batch.batch["response_mask"]
        response_ids = batch.batch["responses"].detach().cpu().tolist()
        response_lens = response_mask.sum(dim=-1).detach().cpu().tolist()
        response_texts = self.tokenizer.batch_decode(
            [ids[: int(length)] for ids, length in zip(response_ids, response_lens, strict=True)],
            skip_special_tokens=True,
        )
        success_by_uid = self._collect_solutions_by_uid(batch, reward_tensor, reward_threshold)
        pools: dict[Any, list[str]] = {}
        for uid, indices in success_by_uid.items():
            candidates = [str(response_texts[idx]) for idx in indices if str(response_texts[idx]).strip()]
            if candidates:
                pools[uid] = candidates
        return pools

    @staticmethod
    def _select_preferred_solution_pool(
        uids: list[Any],
        weak_teacher_pool: dict[Any, list[str]],
        student_pool: dict[Any, list[str]],
        *,
        allow_student_fallback: bool,
        metric_prefix: str,
    ) -> tuple[dict[Any, list[str]], dict[str, float]]:
        """Select exactly one group-level hint, preferring the weak teacher."""
        selected: dict[Any, list[str]] = {}
        weak_selected = 0
        student_selected = 0
        no_hint = 0
        unique_uids = list(dict.fromkeys(uids))
        for uid in unique_uids:
            candidates = [str(text) for text in weak_teacher_pool.get(uid, []) if str(text).strip()]
            source = "weak_teacher"
            if not candidates and allow_student_fallback:
                candidates = [str(text) for text in student_pool.get(uid, []) if str(text).strip()]
                source = "student"
            if not candidates:
                no_hint += 1
                continue
            selected_idx = int(torch.randint(low=0, high=len(candidates), size=(1,)).item())
            selected[uid] = [candidates[selected_idx]]
            if source == "weak_teacher":
                weak_selected += 1
            else:
                student_selected += 1
        denominator = max(len(unique_uids), 1)
        metrics = {
            f"{metric_prefix}/selected_weak_teacher_uid_fraction": weak_selected / denominator,
            f"{metric_prefix}/selected_student_fallback_uid_fraction": student_selected / denominator,
            f"{metric_prefix}/no_correct_hint_uid_fraction": no_hint / denominator,
        }
        return (selected, metrics)

    def _build_weak_teacher_solution_pool(
        self, batch: DataProto, *, reward_threshold: float, num_rollouts: int, metric_prefix: str
    ) -> tuple[dict[Any, list[str]], dict[str, float]]:
        if num_rollouts <= 0:
            raise ValueError(
                f"distillation.oprd_opsd_weak_teacher_num_rollouts must be positive, got {num_rollouts}."
            )
        if self.teacher_rollout_manager is None:
            raise ValueError(
                "weak_teacher OPSD-OPRD currently requires rollout.name=vllm and distillation.teacher_model_path so a dedicated weak-teacher rollout pool can be launched."
            )
        uids = batch.non_tensor_batch["uid"]
        _, first_indices = self._build_uid_index(uids)
        unique_batch = batch.select_idxs(first_indices)
        teacher_prompts = self._build_teacher_prompt_batch(batch=batch, first_indices=first_indices)
        teacher_prompts = teacher_prompts.repeat(repeat_times=num_rollouts, interleave=True)
        teacher_rollout = self._generate_teacher_rollout_sequences(teacher_prompts)
        repeated_unique = unique_batch.repeat(repeat_times=num_rollouts, interleave=True)
        teacher_tensors = {key: value for key, value in teacher_rollout.batch.items()}
        teacher_non_tensors = {key: value.copy() for key, value in repeated_unique.non_tensor_batch.items()}
        teacher_non_tensors.update(
            {key: value.copy() for key, value in teacher_rollout.non_tensor_batch.items()}
        )
        teacher_eval_batch = DataProto.from_dict(
            tensors=teacher_tensors, non_tensors=teacher_non_tensors, meta_info=dict(batch.meta_info)
        )
        if "response_mask" not in teacher_eval_batch.batch:
            teacher_eval_batch.batch["response_mask"] = self._response_mask(
                teacher_eval_batch.batch["attention_mask"], teacher_eval_batch.batch["responses"]
            )
        teacher_reward_tensor, _ = self._compute_or_extract_reward(
            teacher_eval_batch, reward_fn=self.reward_fn, return_dict=False
        )
        correct = teacher_reward_tensor.sum(dim=-1) >= reward_threshold
        response_mask = teacher_eval_batch.batch["response_mask"]
        responses = teacher_eval_batch.batch["responses"]
        response_texts = teacher_eval_batch.non_tensor_batch.get("teacher_response_text")
        if response_texts is None:
            response_texts = np.asarray(
                [
                    self.tokenizer.decode(response[: int(mask.sum().item())], skip_special_tokens=True)
                    for response, mask in zip(responses, response_mask, strict=True)
                ],
                dtype=object,
            )
        repeated_uids = repeated_unique.non_tensor_batch["uid"]
        pools: dict[Any, list[str]] = defaultdict(list)
        for uid, text, is_correct in zip(repeated_uids, response_texts, correct.tolist(), strict=True):
            if is_correct and str(text).strip():
                pools[uid].append(str(text))
        unique_uids = list(dict.fromkeys(uids.tolist()))
        pool_sizes = [len(pools[uid]) for uid in unique_uids]
        metrics = {
            f"{metric_prefix}/weak_teacher_rollout_count": float(correct.numel()),
            f"{metric_prefix}/weak_teacher_correct_fraction": correct.float().mean().item()
            if correct.numel()
            else 0.0,
            f"{metric_prefix}/weak_teacher_hint_uid_fraction": float(sum((size > 0 for size in pool_sizes)))
            / max(len(pool_sizes), 1),
            f"{metric_prefix}/weak_teacher_correct_pool_size_mean": float(np.mean(pool_sizes))
            if pool_sizes
            else 0.0,
        }
        return (dict(pools), metrics)

    def _maybe_build_distillation_batch(
        self,
        batch: DataProto,
        reward_tensor: torch.Tensor,
        reward_extra_infos_dict: Optional[dict[str, list]] = None,
    ) -> Optional[tuple[DataProto, dict[str, float]]]:
        distillation_cfg = self.config.actor_rollout_ref.actor.get("distillation", None)
        loss_mode = self.config.actor_rollout_ref.actor.policy_loss.get("loss_mode", "vanilla")
        if distillation_cfg is None or loss_mode != "guided_opsd":
            if distillation_cfg is not None and loss_mode == "opsd":
                hint_source = self._normalize_hint_source(
                    distillation_cfg.get("oprd_opsd_hint_source", "student")
                )
                if hint_source in {"weak_teacher", "weak_teacher_then_student"}:
                    metric_prefix = "distillation"
                    weak_teacher_pool, source_metrics = self._build_weak_teacher_solution_pool(
                        batch,
                        reward_threshold=float(distillation_cfg.success_reward_threshold),
                        num_rollouts=int(distillation_cfg.get("oprd_opsd_weak_teacher_num_rollouts", 8)),
                        metric_prefix=metric_prefix,
                    )
                    student_pool = {}
                    allow_student_fallback = hint_source == "weak_teacher_then_student"
                    if allow_student_fallback:
                        student_pool = self._build_student_solution_pool(
                            batch,
                            reward_tensor,
                            reward_threshold=float(distillation_cfg.success_reward_threshold),
                        )
                    uids = batch.non_tensor_batch["uid"]
                    uid_values = uids.tolist() if hasattr(uids, "tolist") else list(uids)
                    solution_pool, selection_metrics = self._select_preferred_solution_pool(
                        uid_values,
                        weak_teacher_pool,
                        student_pool,
                        allow_student_fallback=allow_student_fallback,
                        metric_prefix=metric_prefix,
                    )
                    distillation_batch, metrics = self._build_opsd_distillation_batch(
                        batch=batch,
                        reward_tensor=reward_tensor,
                        reward_extra_infos_dict=None,
                        metric_prefix=metric_prefix,
                        solution_texts_by_uid=solution_pool,
                    )
                    enable_on_policy_distill = False
                    mix_alpha = 1.0
                    mix_alpha_schedule = "fixed".lower()
                    metrics.update(source_metrics)
                    metrics.update(selection_metrics)
                    metrics.update(
                        {
                            f"{metric_prefix}/hint_source_weak_teacher_then_student": float(
                                allow_student_fallback
                            ),
                            f"{metric_prefix}/on_policy_enabled": float(enable_on_policy_distill),
                            f"{metric_prefix}/mix_alpha_config": mix_alpha,
                            f"{metric_prefix}/mix_alpha_schedule": float(mix_alpha_schedule == "linear"),
                            f"{metric_prefix}/mix_alpha_schedule_step": float(mix_alpha_schedule == "step"),
                        }
                    )
                    if not enable_on_policy_distill:
                        return (distillation_batch, metrics)
                    on_policy_batch, on_policy_metrics = self._build_on_policy_distillation_batch(batch=batch)
                    metrics.update(on_policy_metrics)
                    return (distillation_batch.union(on_policy_batch), metrics)
            return super()._maybe_build_distillation_batch(batch, reward_tensor, reward_extra_infos_dict)
        metric_prefix = "guided_opsd"
        hint_source = self._normalize_hint_source(distillation_cfg.get("oprd_opsd_hint_source", "student"))
        solution_pool = None
        source_metrics: dict[str, float] = {}
        if hint_source in {"weak_teacher", "weak_teacher_then_student"}:
            solution_pool, source_metrics = self._build_weak_teacher_solution_pool(
                batch,
                reward_threshold=float(distillation_cfg.success_reward_threshold),
                num_rollouts=int(distillation_cfg.get("oprd_opsd_weak_teacher_num_rollouts", 8)),
                metric_prefix=metric_prefix,
            )
            if hint_source == "weak_teacher_then_student":
                student_pool = self._build_student_solution_pool(
                    batch, reward_tensor, reward_threshold=float(distillation_cfg.success_reward_threshold)
                )
                uids = batch.non_tensor_batch["uid"]
                uid_values = uids.tolist() if hasattr(uids, "tolist") else list(uids)
                solution_pool, selection_metrics = self._select_preferred_solution_pool(
                    uid_values,
                    solution_pool,
                    student_pool,
                    allow_student_fallback=True,
                    metric_prefix=metric_prefix,
                )
                source_metrics.update(selection_metrics)
        distillation_batch, metrics = self._build_opsd_distillation_batch(
            batch=batch,
            reward_tensor=reward_tensor,
            reward_extra_infos_dict=None,
            metric_prefix=metric_prefix,
            solution_texts_by_uid=solution_pool,
            exclude_self_solution=True,
        )
        metrics.update(source_metrics)
        metrics[f"{metric_prefix}/hint_source_student"] = float(hint_source == "student")
        metrics[f"{metric_prefix}/hint_source_weak_teacher"] = float(hint_source == "weak_teacher")
        metrics[f"{metric_prefix}/hint_source_weak_teacher_then_student"] = float(
            hint_source == "weak_teacher_then_student"
        )
        return (distillation_batch, metrics)
