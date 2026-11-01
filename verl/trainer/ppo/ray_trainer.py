"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from string import Template
from typing import Any, Optional
import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_variance_proxy_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import (
    compute_reward,
    compute_reward_async,
    compute_validation_reward_payload_async,
)
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.fsdp_checkpoint_manager import fsync_parent_directory
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.metric import reduce_metrics
from verl.utils.model import compute_position_id_with_mask
from verl.utils.py_functional import rename_dict
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import (
    calculate_workload,
    get_seqlen_balanced_partitions,
    log_seqlen_unbalance,
)
from verl.utils.torch_functional import masked_mean
from verl.utils.torch_functional import postprocess_data
from verl.utils.tracking import ValidationGenerationsLogger, select_random_uid_validation_samples
from verl.workers.config import FSDPEngineConfig
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

ROLLOUT_REPEAT_COUNTS_META_KEY = "rollout_repeat_counts"
OPTIMIZER_RETENTION_MARKER = ".preserve_optimizer_state"
ASYNC_CHECKPOINT_PROTOCOL_MARKER = ".async_checkpoint_protocol.json"


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes,
                use_gpu=True,
                max_colocate_count=3,
                name_prefix=resource_pool_name,
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool
        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )

    def release_resource_pools(
        self, *, timeout_seconds: float = 30.0, poll_interval_seconds: float = 0.1
    ) -> dict[str, int]:
        """Release owned Ray placement groups without creating new ones.

        Training workers live inside these placement groups. Removing the groups
        terminates the colocated actors and returns both physical GPU memory and
        Ray's logical GPU reservations while the CPU-only trainer can finish
        checkpoint uploads and logging.
        """
        placement_groups = []
        seen_placement_groups = set()
        owning_pools = []

        def placement_group_key(placement_group):
            placement_group_id = getattr(placement_group, "id", None)
            try:
                return placement_group_id.hex() if placement_group_id is not None else id(placement_group)
            except Exception:
                return id(placement_group)

        for resource_pool in self.resource_pool_dict.values():
            if getattr(resource_pool, "detached", False):
                continue
            existing_groups = getattr(resource_pool, "pgs", None)
            if not existing_groups:
                continue
            owning_pools.append((resource_pool, list(existing_groups)))
            for placement_group in existing_groups:
                group_key = placement_group_key(placement_group)
                if group_key in seen_placement_groups:
                    continue
                seen_placement_groups.add(group_key)
                placement_groups.append(placement_group)
        if not placement_groups:
            return {"requested": 0, "confirmed": 0, "failed": 0, "pending": 0}
        pending = {}
        failed_group_keys = set()
        for placement_group in placement_groups:
            group_key = placement_group_key(placement_group)
            try:
                ray.util.remove_placement_group(placement_group)
                pending[group_key] = placement_group
            except Exception as exc:
                failed_group_keys.add(group_key)
                print(f"Warning: failed to request Ray placement-group removal: {exc}")
        deadline = time.monotonic() + max(0.0, float(timeout_seconds))
        poll_interval_seconds = max(0.01, float(poll_interval_seconds))
        while pending and time.monotonic() < deadline:
            for group_key, placement_group in list(pending.items()):
                try:
                    state = ray.util.placement_group_table(placement_group)
                except Exception:
                    continue
                if not state or str(state.get("state", "")).upper() == "REMOVED":
                    pending.pop(group_key, None)
            if pending:
                time.sleep(min(poll_interval_seconds, max(0.0, deadline - time.monotonic())))
        for resource_pool, existing_groups in owning_pools:
            failed_groups = [
                placement_group
                for placement_group in existing_groups
                if placement_group_key(placement_group) in failed_group_keys
            ]
            resource_pool.pgs = failed_groups or None
        failed = len(failed_group_keys)
        confirmed = len(placement_groups) - failed - len(pending)
        if pending:
            print(
                f"Warning: timed out waiting for {len(pending)} Ray placement group(s) to report REMOVED; removal remains requested."
            )
        print(
            f"Released training Ray placement groups before checkpoint upload wait: requested={len(placement_groups) - failed}, confirmed={confirmed}, failed={failed}, pending={len(pending)}"
        )
        return {
            "requested": len(placement_groups) - failed,
            "confirmed": confirmed,
            "failed": failed,
            "pending": len(pending),
        }

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray._private.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )
    kld = kld * response_mask
    beta = kl_ctrl.value
    token_level_rewards = token_level_scores - beta * kld
    current_kl = masked_mean(kld, mask=response_mask, axis=-1)
    current_kl = torch.mean(current_kl, dim=0).item()
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards
    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}
    return (data, metrics)


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = False,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    if adv_estimator == AdvantageEstimator.GAE:
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data, config.pf_ppo.get("reweight_method"), config.pf_ppo.get("weight_pow")
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        grpo_calculation_mask = data.batch["response_mask"]
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]
        if adv_estimator in (
            AdvantageEstimator.OPTIMAL_TOKEN_BASELINE,
            AdvantageEstimator.TIR_OPTIMAL_TOKEN_BASELINE,
        ):
            assert "sum_pi_squared" in data.batch, (
                "Step-dependent optimal baseline requires sum_pi_squared from actor. Please set actor.calculate_sum_pi_squared=True in config."
            )
            adv_kwargs["sum_pi_squared"] = data.batch["sum_pi_squared"]
            rollout_is_weights = data.batch.get("rollout_is_weights", None)
            adv_kwargs["rollout_is_weights"] = rollout_is_weights
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "error"
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"
        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping or Role.ActorRolloutRef in role_worker_mapping, (
                f"role_worker_mapping.keys()={role_worker_mapping.keys()!r}"
            )
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.config)
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_reward_loop = self.config.reward_model.use_reward_loop
        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name, experiment_name=self.config.trainer.experiment_name
        )
        self._hf_upload_processes: list[tuple[int, subprocess.Popen, str]] = []
        self._hf_upload_scheduled_steps: set[int] = set()
        self._hf_upload_pending_checkpoint_steps: set[int] = set()
        self._pending_best_checkpoint_commits: dict[int, dict[str, Any]] = {}
        self._pending_checkpoint_cleanup_steps: set[int] = set()
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = (
            lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        )
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)
        self.use_prefix_grouper = self.config.actor_rollout_ref.actor.get("use_prefix_grouper", False)
        self.use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                is_train=True,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                is_train=False,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn
        num_workers = self.config.data["dataloader_num_workers"]
        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )
        val_batch_size = self.config.data.val_batch_size
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )
        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"
        print(
            f"Size of train dataloader: {len(self.train_dataloader)} (dataset={len(self.train_dataset)}), Size of val dataloader: {len(self.val_dataloader)} (dataset={len(self.val_dataset)}, batch_size={val_batch_size})"
        )
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps
        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")
        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(
        self,
        inputs,
        outputs,
        gts,
        scores,
        reward_extra_infos_dict,
        dump_path,
        *,
        uid_limit: int | None = None,
        selection_seed: int = 42,
        step: int | None = None,
    ):
        """Dump rollout/validation samples as JSONL, optionally sampling whole uid groups."""
        os.makedirs(dump_path, exist_ok=True)
        dump_step = self.global_steps if step is None else int(step)
        filename = os.path.join(dump_path, f"{dump_step}.jsonl")
        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "ground_truth": gts,
            "score": scores,
            "step": [dump_step] * n,
        }
        dump_label = getattr(self, "_validation_generation_dump_label", None)
        if dump_label:
            base_data["validation_path"] = [dump_label] * n
        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v
        entries = [{k: v[i] for k, v in base_data.items()} for i in range(n)]
        if self._get_validation_generation_wrong_only():
            entries = self._filter_wrong_validation_samples(
                entries, success_threshold=self._get_validation_generation_success_threshold()
            )
        if uid_limit is not None and uid_limit > 0:
            entries = select_random_uid_validation_samples(entries, int(uid_limit), seed=selection_seed)
        lines = [json.dumps(entry, ensure_ascii=False) for entry in entries]
        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"Dumped generations to {filename}")

    def _resolve_validation_data_dir(self) -> str | None:
        if not self._validation_generations_enabled():
            return None
        explicit_dump_path = self.config.trainer.get("validation_data_dir", None)
        if explicit_dump_path:
            return explicit_dump_path
        return os.path.join(
            "validation_generations",
            str(self.config.trainer.project_name),
            str(self.config.trainer.experiment_name),
        )

    def _validation_generations_enabled(self) -> bool:
        return int(self.config.trainer.get("log_val_generations", 0)) > 0

    def _get_validation_generation_uid_limit(self) -> int:
        return max(1, int(self.config.trainer.get("log_val_generation_uid_count", 16)))

    def _get_validation_generation_wrong_only(self) -> bool:
        return bool(self.config.trainer.get("log_val_generation_wrong_only", False))

    def _get_validation_generation_success_threshold(self) -> float:
        return float(self.config.trainer.get("log_val_generation_success_threshold", 1.0))

    @staticmethod
    def _middle_truncate_text(value: Any, max_chars: int) -> str:
        text = "" if value is None else str(value)
        max_chars = max(0, int(max_chars))
        if max_chars <= 0 or len(text) <= max_chars:
            return text
        head_chars = max_chars // 2
        tail_chars = max_chars - head_chars
        omitted = len(text) - head_chars - tail_chars
        return f"{text[:head_chars]}\n...[truncated {omitted} chars]...\n{text[-tail_chars:]}"

    def _print_validation_generation_previews(
        self, samples: list[dict[str, Any]], *, step: int, max_samples: int, jsonl_path: str | None = None
    ) -> None:
        if max_samples <= 0:
            return
        if not samples:
            path_text = f" full_jsonl={jsonl_path}" if jsonl_path else ""
            print(
                f"[VAL_GENERATION_PREVIEW] step={step} showing=0/0 no_samples_after_filter{path_text}",
                flush=True,
            )
            return
        preview_samples = samples[: min(max_samples, len(samples))]
        path_text = f" full_jsonl={jsonl_path}" if jsonl_path else ""
        print(
            f"[VAL_GENERATION_PREVIEW] step={step} showing={len(preview_samples)}/{len(samples)} input_chars<=1200 output_chars<=4000{path_text}",
            flush=True,
        )
        for index, sample in enumerate(preview_samples, start=1):
            extras = []
            for key in (
                "reward",
                "correct",
                "pred",
                "acc",
                "all_passed",
                "passed_tests",
                "total_tests",
                "passed_tests_ratio",
                "truncated",
                "incorrect_format",
                "format_reward",
                "timed_out",
                "error",
                "boxed_answer_count",
                "validation_path",
                "data_source",
            ):
                if key in sample:
                    extras.append(f"{key}={sample.get(key)!r}")
            extra_text = " ".join(extras)
            print(
                f"[VAL_GENERATION_PREVIEW:{index}] uid={sample.get('uid')!r} score={sample.get('score')!r} ground_truth={sample.get('ground_truth')!r} {extra_text}".rstrip()
            )
            print("[VAL_GENERATION_INPUT_BEGIN]")
            print(self._middle_truncate_text(sample.get("input"), 1200))
            print("[VAL_GENERATION_INPUT_END]")
            print("[VAL_GENERATION_OUTPUT_BEGIN]")
            print(self._middle_truncate_text(sample.get("output"), 4000))
            print("[VAL_GENERATION_OUTPUT_END]")

    @staticmethod
    def _filter_wrong_validation_samples(
        samples: list[dict[str, Any]], success_threshold: float
    ) -> list[dict[str, Any]]:
        wrong_samples: list[dict[str, Any]] = []
        for sample in samples:
            try:
                score = float(sample.get("score", 0.0))
            except (TypeError, ValueError):
                continue
            if score < success_threshold:
                wrong_samples.append(sample)
        return wrong_samples

    @staticmethod
    def _flatten_validation_data_sources(data_source_batches: list) -> list[Any]:
        flattened: list[Any] = []
        for values in data_source_batches:
            if isinstance(values, np.ndarray):
                flattened.extend(values.tolist())
            elif isinstance(values, (list, tuple)):
                flattened.extend(values)
            else:
                flattened.append(values)
        return flattened

    @staticmethod
    def _normalize_validation_metadata_list(values: Any, expected_length: int) -> list[Any]:
        if values is None:
            return [None] * expected_length
        if isinstance(values, np.ndarray):
            values = values.tolist()
        elif isinstance(values, tuple):
            values = list(values)
        elif isinstance(values, (str, bytes)):
            values = [values]
        elif not isinstance(values, list):
            values = list(values)
        if len(values) != expected_length:
            raise ValueError(f"Expected {expected_length} metadata values, got {len(values)}")
        return values

    def _build_validation_sample_records(
        self,
        *,
        inputs,
        outputs,
        scores,
        gts=None,
        uids=None,
        data_sources=None,
        reward_extra_infos_dict: dict[str, Any] | None = None,
        step: int | None = None,
    ) -> list[dict[str, Any]]:
        record_count = len(inputs)
        record_step = self.global_steps if step is None else int(step)
        outputs = self._normalize_validation_metadata_list(outputs, record_count)
        scores = self._normalize_validation_metadata_list(scores, record_count)
        ground_truths = self._normalize_validation_metadata_list(gts, record_count)
        uids = self._normalize_validation_metadata_list(uids, record_count)
        data_sources = self._normalize_validation_metadata_list(data_sources, record_count)
        extra_info_lists: dict[str, list[Any]] = {}
        for key, values in (reward_extra_infos_dict or {}).items():
            if key == "reward":
                continue
            try:
                normalized_values = self._normalize_validation_metadata_list(values, record_count)
            except ValueError:
                continue
            extra_info_lists[key] = normalized_values
        records: list[dict[str, Any]] = []
        dump_label = getattr(self, "_validation_generation_dump_label", None)
        for index, input_text in enumerate(inputs):
            try:
                numeric_score = float(scores[index])
            except (TypeError, ValueError):
                numeric_score = 0.0
            record = {
                "step": record_step,
                "sample_index": index,
                "uid": uids[index],
                "data_source": data_sources[index],
                "input": input_text,
                "output": outputs[index],
                "ground_truth": ground_truths[index],
                "score": scores[index],
                "reward": scores[index],
                "correct": numeric_score >= self._get_validation_generation_success_threshold(),
            }
            if dump_label:
                record["validation_path"] = dump_label
            for key, values in extra_info_lists.items():
                record[key] = values[index]
            records.append(record)
        return records

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch
            ]
            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id", batch.non_tensor_batch["request_id"].tolist()
                )
            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    def _maybe_log_val_generations(
        self,
        inputs,
        outputs,
        scores,
        *,
        gts=None,
        uids=None,
        data_sources=None,
        reward_extra_infos_dict: dict[str, Any] | None = None,
        step: int | None = None,
    ):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""
        generations_to_log = self.config.trainer.log_val_generations
        if int(generations_to_log) <= 0:
            return
        record_step = self.global_steps if step is None else int(step)
        collection_step = max(int(self.global_steps), record_step)
        samples = self._build_validation_sample_records(
            inputs=inputs,
            outputs=outputs,
            scores=scores,
            gts=gts,
            uids=uids,
            data_sources=data_sources,
            reward_extra_infos_dict=reward_extra_infos_dict,
            step=record_step,
        )
        if self._get_validation_generation_wrong_only():
            samples = self._filter_wrong_validation_samples(
                samples, success_threshold=self._get_validation_generation_success_threshold()
            )
        samples = select_random_uid_validation_samples(
            samples, self._get_validation_generation_uid_limit(), seed=42
        )
        val_data_dir = self._resolve_validation_data_dir()
        validation_jsonl_path = os.path.join(val_data_dir, f"{record_step}.jsonl") if val_data_dir else None
        self._print_validation_generation_previews(
            samples, step=record_step, max_samples=int(generations_to_log), jsonl_path=validation_jsonl_path
        )
        if bool(self.config.trainer.get("log_val_generations_upload_table", False)):
            configured_loggers = self.config.trainer.logger
            if isinstance(configured_loggers, str):
                configured_loggers = [configured_loggers]
            source_step_loggers = [backend for backend in configured_loggers if backend == "wandb"]
            collection_step_loggers = [backend for backend in configured_loggers if backend != "wandb"]
            if source_step_loggers:
                self.validation_generations_logger.log(source_step_loggers, samples, record_step)
            if collection_step_loggers:
                self.validation_generations_logger.log(collection_step_loggers, samples, collection_step)

    def _compute_or_extract_reward(
        self, batch: DataProto, reward_fn=None, return_dict: bool = False, sum_reward: bool = False
    ) -> tuple[torch.Tensor, dict[str, Any]] | torch.Tensor | dict[str, Any]:
        """
        Compute or extract reward from batch.

        When use_reward_loop=True, rewards are already computed during generate_sequences
        and stored in rm_scores. This method directly extracts them instead of calling
        reward functions which would only perform format conversion.

        Args:
            batch: DataProto containing the batch data
            reward_fn: Reward function to use if rm_scores doesn't exist (for training/validation)
            return_dict: Whether to return dict format with reward_extra_info (for validation)
            sum_reward: Whether to sum reward tensor along last dimension (for REMAX baseline)

        Returns:
            If return_dict=True: dict with "reward_tensor" and "reward_extra_info"
            If return_dict=False and sum_reward=True: summed reward_tensor (1D tensor)
            If return_dict=False and sum_reward=False: reward_tensor (2D tensor)
        """
        if "rm_scores" in batch.batch.keys():
            reward_tensor = batch.batch["rm_scores"]
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)
            if return_dict:
                reward_extra_keys = batch.meta_info.get("reward_extra_keys", [])
                reward_extra_info = (
                    {key: batch.non_tensor_batch[key] for key in reward_extra_keys}
                    if reward_extra_keys
                    else {}
                )
                return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
            else:
                if sum_reward:
                    return reward_tensor
                reward_extra_keys = batch.meta_info.get("reward_extra_keys", [])
                reward_extra_infos_dict = (
                    {key: batch.non_tensor_batch[key] for key in reward_extra_keys}
                    if reward_extra_keys
                    else {}
                )
                return (reward_tensor, reward_extra_infos_dict)
        if reward_fn is None:
            raise ValueError("reward_fn must be provided when rm_scores is not available.")
        if return_dict:
            result = reward_fn(batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)
            reward_extra_info = result.get("reward_extra_info", {})
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        else:
            reward_tensor, reward_extra_infos_dict = compute_reward(batch, reward_fn)
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)
            return (reward_tensor, reward_extra_infos_dict)

    def _maybe_attach_oprd_invalid_format_mask(
        self, batch: DataProto, reward_tensor: torch.Tensor, reward_extra_infos_dict: Optional[dict[str, Any]]
    ) -> dict[str, float]:
        distillation_cfg = self.config.actor_rollout_ref.actor.get("distillation", None)
        agent_cfg = self.config.actor_rollout_ref.get("agent_oprd", {})
        if agent_cfg.get("enabled", False):
            format_cfg, prefix = agent_cfg, ""
        elif distillation_cfg is not None and distillation_cfg.get("oprd_enabled", False):
            format_cfg, prefix = distillation_cfg, "oprd_"
        else:
            return {}
        if not format_cfg.get(prefix + "skip_invalid_format", False):
            return {}
        reward_extra_infos_dict = reward_extra_infos_dict or {}
        invalid_format_key = str(format_cfg.get(prefix + "invalid_format_key", "incorrect_format"))
        if not invalid_format_key:
            raise ValueError(
                "OPRD invalid_format_key must be non-empty when skipping invalid-format guidance."
            )
        if invalid_format_key not in reward_extra_infos_dict:
            available_keys = sorted((str(key) for key in reward_extra_infos_dict.keys()))
            raise ValueError(
                f"OPRD invalid-format gating requires reward extra {invalid_format_key!r}; available reward extra keys: {available_keys}."
            )
        batch_size = int(batch.batch["responses"].size(0))
        try:
            values = np.asarray(reward_extra_infos_dict[invalid_format_key], dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"OPRD invalid-format gating expected numeric reward extra {invalid_format_key!r}."
            ) from exc
        values_tensor = torch.as_tensor(values, dtype=torch.float32, device=reward_tensor.device).reshape(-1)
        if values_tensor.numel() != batch_size:
            raise ValueError(
                f"OPRD invalid-format gating expected reward extra {invalid_format_key!r} with {batch_size} values, got {values_tensor.numel()}."
            )
        threshold = float(format_cfg.get(prefix + "invalid_format_threshold", 0.5))
        invalid_format_mask = values_tensor >= threshold
        if getattr(batch.batch, "is_locked", False):
            batch.batch = batch.batch.copy()
        batch.batch["oprd_invalid_format_mask"] = invalid_format_mask.to(dtype=torch.bool)
        return {
            "distillation/oprd_skip_invalid_format_enabled": 1.0,
            "distillation/oprd_invalid_format_threshold": threshold,
            "distillation/oprd_invalid_format_rollout_fraction": invalid_format_mask.to(dtype=torch.float32)
            .mean()
            .detach()
            .item(),
        }

    def _maybe_build_distillation_batch(
        self,
        batch: DataProto,
        reward_tensor: torch.Tensor,
        reward_extra_infos_dict: Optional[dict[str, list]] = None,
    ) -> Optional[tuple[DataProto, dict[str, float]]]:
        return None

    @staticmethod
    def _safe_format_template(template: str, **values: Any) -> str:
        try:
            return str(template).format(**values)
        except Exception as exc:
            raise ValueError(f"Failed to format advantage feedback template: {exc}") from exc

    def _decode_response_texts(self, batch: DataProto) -> tuple[list[str], list[int]]:
        responses = batch.batch["responses"]
        response_mask = batch.batch["response_mask"]
        response_ids_cpu = responses.detach().cpu().tolist()
        response_lens = [int(x) for x in response_mask.sum(dim=-1).detach().cpu().tolist()]
        trimmed = [ids[:valid_len] for ids, valid_len in zip(response_ids_cpu, response_lens, strict=True)]
        return (self.tokenizer.batch_decode(trimmed, skip_special_tokens=True), response_lens)

    @staticmethod
    def _response_prefix_ids(
        *, response_ids: torch.Tensor, response_mask: torch.Tensor, ratio: float
    ) -> list[int]:
        valid_len = int(response_mask.sum().detach().cpu().item())
        if valid_len <= 0:
            return []
        split_len = min(valid_len, max(1, int(round(valid_len * float(ratio)))))
        return [int(token_id) for token_id in response_ids[:split_len].detach().cpu().tolist()]

    @staticmethod
    def _patch_reward_extra_infos(
        base_infos: Optional[dict[str, Any]],
        replacement_infos: Optional[dict[str, Any]],
        *,
        target_indices: list[int],
        total_count: int,
    ) -> dict[str, list[Any]]:
        patched: dict[str, list[Any]] = {}
        for key, values in (base_infos or {}).items():
            if isinstance(values, np.ndarray):
                values = values.tolist()
            elif not isinstance(values, list):
                values = list(values)
            patched[key] = list(values)
        for key, values in (replacement_infos or {}).items():
            if isinstance(values, np.ndarray):
                values = values.tolist()
            elif not isinstance(values, list):
                values = list(values)
            if len(values) != len(target_indices):
                continue
            if key not in patched or len(patched[key]) != total_count:
                patched[key] = [None] * total_count
            for dst_idx, value in zip(target_indices, values, strict=True):
                patched[key][dst_idx] = value
        return patched

    @staticmethod
    def _set_replacement_non_tensor_values(
        batch: DataProto,
        *,
        target_indices: list[int],
        key: str,
        values: list[Any],
        default_value: Any,
        dtype: Any = object,
    ) -> None:
        if key not in batch.non_tensor_batch or len(batch.non_tensor_batch[key]) != len(batch):
            batch.non_tensor_batch[key] = np.array([default_value] * len(batch), dtype=dtype)
        for dst_idx, value in zip(target_indices, values, strict=True):
            batch.non_tensor_batch[key][dst_idx] = value

    @classmethod
    def _replace_batch_rows(
        cls, batch: DataProto, *, target_indices: list[int], replacement: DataProto
    ) -> None:
        if len(target_indices) != len(replacement):
            raise ValueError(
                f"replacement row count must match target indices, got {len(replacement)} and {len(target_indices)}."
            )
        idx_tensor = torch.tensor(target_indices, dtype=torch.long)
        if batch.batch is not None:
            for key, value in replacement.batch.items():
                if key not in batch.batch:
                    continue
                target = batch.batch[key]
                if tuple(target.shape[1:]) != tuple(value.shape[1:]):
                    raise ValueError(
                        f"Cannot replace tensor key {key!r}: shape tail differs {tuple(target.shape[1:])} vs {tuple(value.shape[1:])}."
                    )
                idx_on_device = idx_tensor.to(device=target.device)
                batch.batch[key][idx_on_device] = value.to(device=target.device, dtype=target.dtype)
        for key, values in replacement.non_tensor_batch.items():
            if key not in batch.non_tensor_batch or len(batch.non_tensor_batch[key]) != len(batch):
                batch.non_tensor_batch[key] = np.array([None] * len(batch), dtype=object)
            for dst_idx, value in zip(target_indices, values.tolist(), strict=True):
                batch.non_tensor_batch[key][dst_idx] = value

    @staticmethod
    def _to_message_list(raw_prompt: Any) -> list[dict]:
        if isinstance(raw_prompt, np.ndarray):
            raw_prompt = raw_prompt.tolist()
        if raw_prompt is None:
            return []
        if isinstance(raw_prompt, str):
            return [{"role": "user", "content": raw_prompt}]
        if isinstance(raw_prompt, dict):
            return [dict(raw_prompt)]
        if isinstance(raw_prompt, list | tuple):
            if len(raw_prompt) == 0:
                return []
            first_item = raw_prompt[0]
            if isinstance(first_item, dict):
                return [dict(msg) for msg in raw_prompt]
            if isinstance(first_item, str):
                return [{"role": "user", "content": "".join((str(part) for part in raw_prompt))}]
        return [{"role": "user", "content": str(raw_prompt)}]

    def _tokenize_teacher_prompt_messages(
        self, messages: list[list[dict]], max_length: Optional[int] = None
    ) -> dict[str, torch.Tensor]:
        apply_chat_template_kwargs = OmegaConf.select(self.config, "data.apply_chat_template_kwargs") or {}
        enable_thinking = apply_chat_template_kwargs.get("enable_thinking", True)
        assistant_final_flags = [
            bool(message_list) and message_list[-1].get("role") == "assistant" for message_list in messages
        ]
        if any(assistant_final_flags) and (not all(assistant_final_flags)):
            raise ValueError(
                "Mixed prompt batches with and without final assistant-prefill messages are not supported."
            )
        continue_final_message = bool(assistant_final_flags) and all(assistant_final_flags)
        apply_kwargs = {
            "tokenize": True,
            "return_tensors": "pt",
            "return_dict": True,
            "continue_final_message": continue_final_message,
            "add_generation_prompt": not continue_final_message,
            "enable_thinking": enable_thinking,
            "padding": True,
            "truncation": True,
        }
        if max_length is not None:
            apply_kwargs["max_length"] = int(max_length)
        original_truncation_side = getattr(self.tokenizer, "truncation_side", None)
        if original_truncation_side == "error":
            self.tokenizer.truncation_side = "left"
        try:
            prompt_dict = self.tokenizer.apply_chat_template(messages, **apply_kwargs)
        finally:
            if original_truncation_side is not None:
                self.tokenizer.truncation_side = original_truncation_side
        attention_mask = prompt_dict["attention_mask"]
        return {
            "input_ids": prompt_dict["input_ids"],
            "attention_mask": attention_mask,
            "position_ids": compute_position_id_with_mask(attention_mask),
        }

    @staticmethod
    def _repeat_batch_for_rollout(batch: DataProto, repeat_times: int) -> DataProto:
        rollout_repeat_counts = batch.meta_info.get(ROLLOUT_REPEAT_COUNTS_META_KEY)
        if rollout_repeat_counts is None:
            return batch.repeat(repeat_times=repeat_times, interleave=True)
        repeat_counts = [int(count) for count in rollout_repeat_counts]
        if len(repeat_counts) != len(batch):
            raise ValueError(
                f"rollout repeat count length must match batch size, got {len(repeat_counts)} and {len(batch)}."
            )
        if any((count < 0 for count in repeat_counts)):
            raise ValueError(f"rollout repeat counts must be non-negative, got {repeat_counts}")
        if all((count == repeat_times for count in repeat_counts)):
            return batch.repeat(repeat_times=repeat_times, interleave=True)
        expanded_indices = np.repeat(np.arange(len(repeat_counts), dtype=np.int64), repeat_counts)
        if expanded_indices.size == 0:
            raise ValueError("rollout repeat counts produced an empty batch.")
        return batch.select_idxs(expanded_indices)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = (
            set({"data_source", "reward_model", "extra_info", "uid", "raw_prompt"})
            & batch.non_tensor_batch.keys()
        )
        batch_keys_to_pop = []
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop, non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop)
        )
        max_prompt_length = OmegaConf.select(self.config, "data.max_prompt_length")
        if max_prompt_length is not None:
            gen_batch.meta_info["max_prompt_length"] = max_prompt_length
        apply_chat_template_kwargs = OmegaConf.select(self.config, "data.apply_chat_template_kwargs")
        if apply_chat_template_kwargs:
            gen_batch.meta_info["apply_chat_template_kwargs"] = OmegaConf.to_container(
                apply_chat_template_kwargs, resolve=True
            )
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)
        return gen_batch

    def _validation_seed(self):
        return self.config.actor_rollout_ref.rollout.val_kwargs.get("seed", 42)

    def _add_validation_seed(self, meta_info: dict) -> dict:
        seed = self._validation_seed()
        if seed is not None:
            meta_info["seed"] = int(seed)
        return meta_info

    def _validate(self, merged: bool = False, defer_reward: bool = False):
        data_source_lst = []
        reward_batches = []
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_turns = []
        sample_uids = []
        val_data_dir = self._resolve_validation_data_dir()
        need_generation_text = self._validation_generations_enabled() or bool(val_data_dir)
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )
            if (
                self.config.reward_model.enable
                and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model"
            ):
                return {}
            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)
            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            val_response_length = self.config.actor_rollout_ref.rollout.val_kwargs.get(
                "response_length", None
            )
            if val_response_length is not None:
                test_gen_batch.meta_info["response_length"] = int(val_response_length)
            self._add_validation_seed(test_gen_batch.meta_info)
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(
                    test_gen_batch_padded
                )
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            if need_generation_text:
                output_ids = test_output_gen_batch.batch["responses"]
                output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
                sample_outputs.extend(output_texts)
            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True
            if need_generation_text:
                input_ids = test_batch.batch["prompts"]
                input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
                sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])
            reward_batches.append(test_batch)
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])
            data_source_lst.append(
                test_batch.non_tensor_batch.get("data_source", ["unknown"] * len(test_batch))
            )
        payload = {
            "validation_step": int(self.global_steps),
            "reward_batches": reward_batches,
            "data_source_lst": data_source_lst,
            "sample_inputs": sample_inputs,
            "sample_outputs": sample_outputs,
            "sample_gts": sample_gts,
            "sample_turns": sample_turns,
            "sample_uids": sample_uids,
            "val_data_dir": val_data_dir,
        }
        if defer_reward:
            return payload
        scored_payload = self._score_validation_reward_payload(payload)
        return self._finalize_scored_validation_reward_payload(scored_payload, merged=merged)

    def _score_validation_reward_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)
        sample_scores = []
        for test_batch in payload["reward_batches"]:
            result = self._compute_or_extract_reward(
                test_batch, reward_fn=self.val_reward_fn, return_dict=True
            )
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)
            reward_extra_infos_dict["reward"].extend(scores)
            reward_extra_info = result.get("reward_extra_info", {})
            for key, values in reward_extra_info.items():
                if key not in reward_extra_infos_dict:
                    reward_extra_infos_dict[key] = []
                if isinstance(values, np.ndarray):
                    reward_extra_infos_dict[key].extend(values.tolist())
                else:
                    reward_extra_infos_dict[key].extend(values if isinstance(values, list) else [values])
        scored_payload = dict(payload)
        scored_payload["sample_scores"] = sample_scores
        scored_payload["reward_extra_infos_dict"] = reward_extra_infos_dict
        scored_payload.pop("reward_batches", None)
        return scored_payload

    def _finalize_scored_validation_reward_payload(self, payload: dict[str, Any], merged: bool = False):
        data_source_lst = payload["data_source_lst"]
        sample_inputs = payload["sample_inputs"]
        sample_outputs = payload["sample_outputs"]
        sample_gts = payload["sample_gts"]
        sample_scores = payload["sample_scores"]
        sample_turns = payload["sample_turns"]
        sample_uids = payload["sample_uids"]
        val_data_dir = payload["val_data_dir"]
        reward_extra_infos_dict = payload["reward_extra_infos_dict"]
        validation_step = int(payload.get("validation_step", self.global_steps))
        flat_data_sources = self._flatten_validation_data_sources(data_source_lst)
        self._maybe_log_val_generations(
            inputs=sample_inputs,
            outputs=sample_outputs,
            scores=sample_scores,
            gts=sample_gts,
            uids=sample_uids,
            data_sources=flat_data_sources,
            reward_extra_infos_dict=reward_extra_infos_dict,
            step=validation_step,
        )
        if val_data_dir:
            reward_extra_infos_to_dump = dict(reward_extra_infos_dict)
            reward_extra_infos_to_dump["uid"] = [None if uid is None else str(uid) for uid in sample_uids]
            reward_extra_infos_to_dump["data_source"] = flat_data_sources
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=val_data_dir,
                uid_limit=self._get_validation_generation_uid_limit(),
                step=validation_step,
            )
        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), (
                f"{key_info}: len(lst)={len(lst)!r}, len(sample_scores)={len(sample_scores)!r}"
            )
        if merged:
            print("_merge_validation_results validate result will be merged")
            return {
                "data_sources": data_source_lst,
                "sample_uids": sample_uids,
                "sample_turns": sample_turns,
                "reward_extra_infos_dict": reward_extra_infos_dict,
            }
        data_sources = np.asarray(flat_data_sources, dtype=object)
        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def _val_metrics_update(self, data_sources, sample_uids, reward_extra_infos_dict, sample_turns):
        data_src2var2metric2val = process_validation_metrics(
            data_sources, sample_uids, reward_extra_infos_dict
        )
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_vars = {"acc" if "acc" in var2metric2val else "reward", "score"}
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        var_name in core_vars
                        and any((metric_name.startswith(pfx) for pfx in ["mean", "maj", "best", "pass"]))
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val
        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()
        return metric_dict

    def _merge_validation_results(self, result_a, result_b):
        if result_a is None and result_b is None:
            return {}
        if result_a is None:
            result_a = {
                "data_sources": [],
                "sample_uids": [],
                "sample_turns": [],
                "reward_extra_infos_dict": {},
            }
        if result_b is None:
            result_b = {
                "data_sources": [],
                "sample_uids": [],
                "sample_turns": [],
                "reward_extra_infos_dict": {},
            }
        if not result_a.get("data_sources") and (not result_b.get("data_sources")):
            return {}
        data_sources = np.concatenate(result_a["data_sources"] + result_b["data_sources"], axis=0)
        sample_uids = result_a["sample_uids"] + result_b["sample_uids"]
        sample_turns = result_a["sample_turns"] + result_b["sample_turns"]
        reward_extra_infos_dict = {}
        all_keys = set(result_a["reward_extra_infos_dict"].keys()) | set(
            result_b["reward_extra_infos_dict"].keys()
        )
        for key in all_keys:
            list_a = result_a["reward_extra_infos_dict"].get(key, [])
            list_b = result_b["reward_extra_infos_dict"].get(key, [])
            reward_extra_infos_dict[key] = list_a + list_b
        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {
            pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()
        }
        actor_role = (
            Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        )
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[actor_role],
                config=self.config.actor_rollout_ref,
                role=str(actor_role),
            )
            self.resource_pool_to_cls[resource_pool][str(actor_role)] = actor_rollout_cls
        else:
            raise NotImplementedError
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            from verl.workers.config import CriticConfig

            critic_cfg: CriticConfig = omega_conf_to_dataclass(self.config.critic)
            if self.use_legacy_worker_impl == "disable":
                from verl.workers.engine_workers import TrainingWorkerConfig

                orig_critic_cfg = critic_cfg
                if orig_critic_cfg.strategy == "fsdp":
                    engine_config: FSDPEngineConfig = orig_critic_cfg.model.fsdp_config
                    engine_config.infer_max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
                    engine_config.max_token_len_per_gpu = critic_cfg.ppo_max_token_len_per_gpu
                else:
                    raise NotImplementedError(
                        f"Unknown strategy orig_critic_cfg.strategy={orig_critic_cfg.strategy!r}"
                    )
                critic_cfg = TrainingWorkerConfig(
                    model_type="value_model",
                    model_config=orig_critic_cfg.model_config,
                    engine_config=engine_config,
                    optimizer_config=orig_critic_cfg.optim,
                    checkpoint_config=orig_critic_cfg.checkpoint,
                )
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls
        if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls
        if not self.use_reward_loop:
            if self.use_rm:
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
                rm_cls = RayClassWithInitArgs(
                    self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model
                )
                self.resource_pool_to_cls[resource_pool][str(Role.RewardModel)] = rm_cls
        else:
            can_reward_loop_parallelize = not self.use_rm or self.config.reward_model.enable_resource_pool
            if not can_reward_loop_parallelize:
                from verl.experimental.reward_loop import RewardLoopManager

                self.config.reward_model.n_gpus_per_node = self.config.trainer.n_gpus_per_node
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
                self.reward_loop_manager = RewardLoopManager(
                    config=self.config, rm_resource_pool=resource_pool
                )
        all_wg = {}
        wg_kwargs = {}
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = (
                self.config.trainer.ray_wait_register_center_timeout
            )
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(
                        self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options"
                    )
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(
                        self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options"
                    )
                )
        wg_kwargs["device_name"] = self.device_name
        worker_group_master_port = OmegaConf.select(self.config.ray_kwargs, "worker_group_master_port")
        worker_group_master_addr = OmegaConf.select(self.config.ray_kwargs, "worker_group_master_addr")
        if worker_group_master_port is not None and worker_group_master_addr is None:
            worker_group_master_addr = ray.util.get_node_ip_address().strip("[]")
        for wg_idx, (resource_pool, class_dict) in enumerate(self.resource_pool_to_cls.items()):
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            local_wg_kwargs = dict(wg_kwargs)
            if worker_group_master_port is not None:
                local_wg_kwargs["master_addr"] = str(worker_group_master_addr)
                local_wg_kwargs["master_port"] = str(int(worker_group_master_port) + wg_idx)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, **local_wg_kwargs
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            if self.use_legacy_worker_impl == "disable":
                self.critic_wg.reset()
                from functools import partial
                from verl.workers.utils.losses import value_loss

                value_loss_ = partial(value_loss, config=orig_critic_cfg)
                self.critic_wg.set_loss_fn(value_loss_)
            else:
                self.critic_wg.init_model()
        if self.use_reference_policy and (not self.ref_in_actor):
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()
            else:
                assert str(Role.ActorRolloutRef) in all_wg, f"all_wg.keys()={all_wg.keys()!r}"
                self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]
        self.rm_wg = None
        if self.use_rm and (not self.use_reward_loop):
            self.rm_wg = all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()
        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()
        if self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg
        self.async_rollout_mode = True
        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get(
            "agent_loop_manager_class"
        )
        if manager_class_fqn:
            AgentLoopManager = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            from verl.experimental.agent_loop import AgentLoopManager
        if self.config.reward_model.enable and self.config.reward_model.enable_resource_pool:
            rm_resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
        else:
            rm_resource_pool = None
        self.async_rollout_manager = AgentLoopManager(
            config=self.config, worker_group=self.actor_rollout_wg, rm_resource_pool=rm_resource_pool
        )

    def _get_save_steps(self) -> set[int]:
        if hasattr(self, "_cached_save_steps"):
            return self._cached_save_steps
        raw_save_steps = self.config.trainer.get("save_steps", [])
        save_steps: set[int] = set()
        if raw_save_steps is None:
            self._cached_save_steps = save_steps
            return save_steps
        if isinstance(raw_save_steps, str):
            raw_save_steps = raw_save_steps.strip()
            raw_save_steps = raw_save_steps.strip("[]")
            candidates = [x.strip() for x in raw_save_steps.split(",") if x.strip()]
        elif isinstance(raw_save_steps, int):
            candidates = [raw_save_steps]
        else:
            candidates = raw_save_steps
        for candidate in candidates:
            try:
                step = int(candidate)
            except (TypeError, ValueError):
                print(f"Warning: invalid trainer.save_steps entry {candidate!r}; ignoring.")
                continue
            if step <= 0:
                print(f"Warning: trainer.save_steps entry {step} must be > 0; ignoring.")
                continue
            save_steps.add(step)
        self._cached_save_steps = save_steps
        return save_steps

    def _get_validation_steps(self) -> set[int]:
        if hasattr(self, "_cached_validation_steps"):
            return self._cached_validation_steps
        raw_validation_steps = self.config.trainer.get("validation_steps", [])
        validation_steps: set[int] = set()
        if raw_validation_steps is None:
            raw_validation_steps = []
        if isinstance(raw_validation_steps, str):
            raw_validation_steps = raw_validation_steps.strip().strip("[]")
            candidates = [x.strip() for x in raw_validation_steps.split(",") if x.strip()]
        elif isinstance(raw_validation_steps, int):
            candidates = [raw_validation_steps]
        else:
            candidates = raw_validation_steps
        for candidate in candidates:
            try:
                step = int(candidate)
            except (TypeError, ValueError):
                print(f"Warning: invalid trainer.validation_steps entry {candidate!r}; ignoring.")
                continue
            if step <= 0:
                print(f"Warning: trainer.validation_steps entry {step} must be > 0; ignoring.")
                continue
            validation_steps.add(step)
        # Custom horizons can put a model snapshot between regular evaluations.
        # Evaluate that snapshot too, so every new resume target has been evaluated.
        if self.config.trainer.get("save_on_validation", False):
            validation_steps.update(self._get_save_steps())
        self._cached_validation_steps = validation_steps
        return validation_steps

    @staticmethod
    def _env_bool(name: str, default: bool = False) -> bool:
        value = os.environ.get(name)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "on"}

    def _fsdp_frequency_checkpoint_policy_enabled(self) -> bool:
        """Whether model-only frequency retention is supported by this actor engine."""
        try:
            actor_config = self.config.actor_rollout_ref.actor
        except (AttributeError, KeyError, TypeError):
            return True
        strategy = actor_config.get("strategy", "fsdp")
        return str(strategy).strip().lower() in {"fsdp", "fsdp2"}

    def _async_checkpoint_protocol_marker_path(self) -> str:
        return os.path.join(str(self.config.trainer.default_local_dir), ASYNC_CHECKPOINT_PROTOCOL_MARKER)

    def _async_checkpoint_protocol_cutover_step(self) -> int | None:
        marker_path = self._async_checkpoint_protocol_marker_path()
        if not os.path.isfile(marker_path):
            return None
        try:
            with open(marker_path, encoding="utf-8") as stream:
                payload = json.load(stream)
            step = int(payload["first_async_step"])
            if step < 0:
                raise ValueError("first_async_step must be non-negative")
            return step
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            print(f"Warning: invalid async checkpoint protocol marker {marker_path}: {exc}")
            return None

    def _existing_async_checkpoint_evidence_steps(self) -> list[int]:
        checkpoint_root = str(self.config.trainer.default_local_dir)
        evidence_steps: list[int] = []
        try:
            names = os.listdir(checkpoint_root)
        except OSError:
            return evidence_steps
        for name in names:
            if not name.startswith("global_step_"):
                continue
            try:
                candidate_step = int(name.removeprefix("global_step_"))
            except ValueError:
                continue
            if self._checkpoint_transaction_has_async_evidence(candidate_step):
                evidence_steps.append(candidate_step)
        return evidence_steps

    def _effective_async_checkpoint_cutover_step(self) -> int | None:
        persisted_cutover = self._async_checkpoint_protocol_cutover_step()
        if persisted_cutover is not None:
            return persisted_cutover
        evidence_steps = self._existing_async_checkpoint_evidence_steps()
        return min(evidence_steps) if evidence_steps else None

    def _ensure_async_checkpoint_protocol_marker(self, step: int) -> None:
        """Record the sync-to-async cutover before the first async save starts."""
        marker_path = self._async_checkpoint_protocol_marker_path()
        if os.path.exists(marker_path):
            cutover = self._async_checkpoint_protocol_cutover_step()
            if cutover is None:
                raise RuntimeError(f"Refusing async save with invalid protocol marker: {marker_path}")
            return
        evidence_steps = self._existing_async_checkpoint_evidence_steps()
        first_async_step = min([int(step), *evidence_steps])
        os.makedirs(os.path.dirname(marker_path), exist_ok=True)
        temporary_path = f"{marker_path}.tmp.{uuid.uuid4().hex}"
        try:
            with open(temporary_path, "w", encoding="utf-8") as stream:
                json.dump({"version": 1, "first_async_step": first_async_step}, stream, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, marker_path)
            fsync_parent_directory(marker_path)
        finally:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)

    def _optimizer_retention_marker_path(self, step: int) -> str:
        return os.path.join(
            str(self.config.trainer.default_local_dir), f"global_step_{int(step)}", OPTIMIZER_RETENTION_MARKER
        )

    def _checkpoint_is_explicit_save_asset(self, step: int) -> bool:
        """Return whether a checkpoint was requested deliberately rather than by cadence.

        Note this says nothing about optimizer state: save_steps keep model weights,
        but only the latest checkpoint keeps its optimizer shards. See
        ``_prune_stale_optimizer_states``.
        """
        step = int(step)
        return step in self._get_save_steps() or os.path.isfile(self._optimizer_retention_marker_path(step))

    def _delete_uploaded_save_steps_enabled(self) -> bool:
        return self._env_bool("RD_HF_DELETE_UPLOADED_SAVE_STEPS", False)

    def _build_hf_checkpoint_upload_cmd(self, step: int) -> list[str] | None:
        cmd = None
        if cmd is None:
            return None
        cmd.extend(["--checkpoint-step", str(int(step)), "--skip-latest-alias", "--skip-metadata"])
        if self._final_only_optimizer_policy_enabled():
            cmd.extend(
                [
                    "--final-optimizer-step",
                    str(self._final_training_step()),
                    "--checkpoint-save-freq",
                    str(int(self.config.trainer.get("save_freq", 0) or 0)),
                    "--reconcile-remote-policy",
                    "--squash-history-after-delete",
                ]
            )
        return cmd

    def _build_hf_final_upload_cmd(self) -> list[str] | None:
        cmd = None
        if cmd is None:
            return None
        cmd.extend(["--upload-preserved-checkpoints", "--skip-existing", "--skip-latest-alias"])
        if self._final_only_optimizer_policy_enabled():
            cmd.extend(
                [
                    "--final-optimizer-step",
                    str(self._final_training_step()),
                    "--checkpoint-save-freq",
                    str(int(self.config.trainer.get("save_freq", 0) or 0)),
                    "--reconcile-remote-policy",
                    "--squash-history-after-delete",
                ]
            )
        return cmd

    def _final_only_optimizer_policy_enabled(self) -> bool:
        return self._env_bool("RD_FINAL_ONLY_OPTIMIZER", True)

    def _final_training_step(self) -> int:
        configured_step = self.config.trainer.get("total_training_steps", None)
        configured_step = int(configured_step) if configured_step is not None else 0
        step = configured_step if configured_step > 0 else int(getattr(self, "total_training_steps", 0))
        if step <= 0:
            raise ValueError("RD_FINAL_ONLY_OPTIMIZER requires a positive total training step count")
        return step

    def _delete_uploaded_save_step_checkpoint(self, step: int) -> None:
        if not self._delete_uploaded_save_steps_enabled():
            return
        if self._checkpoint_is_explicit_save_asset(step):
            print(f"Keeping explicit save-step checkpoint after HF upload: step={step}")
            return
        checkpoint_root = os.path.abspath(str(self.config.trainer.default_local_dir))
        checkpoint_dir = os.path.abspath(os.path.join(checkpoint_root, f"global_step_{step}"))
        if os.path.dirname(checkpoint_dir) != checkpoint_root:
            print(f"Warning: refusing to delete checkpoint outside run root: {checkpoint_dir}")
            return
        if not os.path.isdir(checkpoint_dir):
            return
        print(f"Deleting uploaded HF save-step checkpoint: {checkpoint_dir}")
        shutil.rmtree(checkpoint_dir, ignore_errors=True)

    def _async_upload_delete_preserved_steps(self) -> set[int]:
        preserved_steps: set[int] = set()
        global_steps = int(getattr(self, "global_steps", -1))
        if global_steps >= 0:
            preserved_steps.add(global_steps)
        if global_steps > 0:
            preserved_steps.add(global_steps - 1)
        tracker_path = os.path.join(
            str(self.config.trainer.default_local_dir), "latest_checkpointed_iteration.txt"
        )
        try:
            if os.path.isfile(tracker_path):
                with open(tracker_path, encoding="utf-8") as f:
                    preserved_steps.add(int(f.read().strip()))
        except (OSError, ValueError):
            pass
        return preserved_steps

    def _best_validation_checkpoint_enabled(self) -> bool:
        return bool(self.config.trainer.get("save_best_validation_checkpoint", False))

    def _best_validation_checkpoint_metadata_path(self) -> str:
        filename = str(
            self.config.trainer.get(
                "best_validation_checkpoint_metadata_filename", "best_validation_checkpoint.json"
            )
        )
        return os.path.join(self.config.trainer.default_local_dir, filename)

    def _checkpoint_transaction_complete_marker_path(self, step: int) -> str:
        return os.path.join(
            str(self.config.trainer.default_local_dir),
            f"global_step_{int(step)}",
            "actor",
            ".checkpoint_complete.json",
        )

    def _checkpoint_transaction_is_complete(self, step: int) -> bool:
        if not self._actor_checkpoint_async_save_enabled():
            return True
        if not self._fsdp_frequency_checkpoint_policy_enabled():
            return not bool(getattr(self, "_async_checkpoint_save_pending", False))
        if not os.path.isfile(self._checkpoint_transaction_complete_marker_path(step)):
            actor_path = os.path.join(
                str(self.config.trainer.default_local_dir), f"global_step_{int(step)}", "actor"
            )
            cutover_step = self._effective_async_checkpoint_cutover_step()
            if (
                cutover_step is not None
                and int(step) < cutover_step
                and (not self._checkpoint_transaction_has_async_evidence(step))
            ):
                tracker_path = os.path.join(
                    str(self.config.trainer.default_local_dir), "latest_checkpointed_iteration.txt"
                )
                try:
                    with open(tracker_path, encoding="utf-8") as tracker_file:
                        tracker_step = int(tracker_file.read().strip())
                except (OSError, ValueError):
                    return False
                return bool(
                    tracker_step >= int(step)
                    and os.path.isdir(actor_path)
                    and (not self._checkpoint_transaction_has_failure_marker(step))
                )
            if (
                cutover_step is None
                and getattr(self, "_loaded_checkpoint_step", None) is not None
                and (not self._run_has_async_checkpoint_evidence())
            ):
                tracker_path = os.path.join(
                    str(self.config.trainer.default_local_dir), "latest_checkpointed_iteration.txt"
                )
                try:
                    with open(tracker_path, encoding="utf-8") as tracker_file:
                        tracker_step = int(tracker_file.read().strip())
                except (OSError, ValueError):
                    return False
                return bool(
                    tracker_step >= int(step)
                    and os.path.isdir(actor_path)
                    and (not self._checkpoint_transaction_has_failure_marker(step))
                )
            return bool(
                getattr(self, "_loaded_checkpoint_step", None) == int(step)
                and os.path.isdir(actor_path)
                and (not self._checkpoint_transaction_has_failure_marker(step))
            )
        tracker_path = os.path.join(
            str(self.config.trainer.default_local_dir), "latest_checkpointed_iteration.txt"
        )
        try:
            with open(tracker_path, encoding="utf-8") as tracker_file:
                return int(tracker_file.read().strip()) >= int(step)
        except (OSError, ValueError):
            return False

    def _checkpoint_transaction_has_failure_marker(self, step: int) -> bool:
        actor_path = os.path.join(
            str(self.config.trainer.default_local_dir), f"global_step_{int(step)}", "actor"
        )
        try:
            marker_names = os.listdir(actor_path)
        except OSError:
            return False
        return any(
            (
                name == ".trainer_sidecar_failed.json"
                or name.startswith(".async_checkpoint_failed_")
                or name.startswith(".worker_sidecars_failed_")
                for name in marker_names
            )
        )

    def _checkpoint_transaction_has_async_evidence(self, step: int) -> bool:
        actor_path = os.path.join(
            str(self.config.trainer.default_local_dir), f"global_step_{int(step)}", "actor"
        )
        try:
            marker_names = os.listdir(actor_path)
        except OSError:
            return False
        return any(
            (
                name == ".checkpoint_complete.json"
                or name == ".trainer_sidecar_failed.json"
                or name.startswith(".async_checkpoint_")
                or name.startswith(".worker_sidecars_")
                for name in marker_names
            )
        )

    def _run_has_async_checkpoint_evidence(self) -> bool:
        return bool(self._existing_async_checkpoint_evidence_steps())

    def _load_best_validation_checkpoint_metadata(self) -> dict[str, Any] | None:
        if hasattr(self, "_best_validation_checkpoint_metadata_loaded"):
            return getattr(self, "_best_validation_checkpoint_metadata", None)
        self._best_validation_checkpoint_metadata_loaded = True
        self._best_validation_checkpoint_metadata = None
        path = self._best_validation_checkpoint_metadata_path()
        if not os.path.isfile(path):
            return None
        try:
            with open(path, encoding="utf-8") as f:
                metadata = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Warning: failed to read best validation checkpoint metadata {path}: {exc}")
            return None
        self._best_validation_checkpoint_metadata = metadata
        return metadata

    def _get_best_validation_checkpoint_step(self) -> int | None:
        metadata = self._load_best_validation_checkpoint_metadata()
        if not metadata:
            return None
        try:
            step = int(metadata.get("step"))
        except (TypeError, ValueError):
            return None
        return step if step >= 0 else None

    def _write_best_validation_checkpoint_metadata(
        self, *, step: int, score: float, metric_selector: str, metric_keys: list[str]
    ) -> None:
        from verl.utils.fs import local_mkdir_safe

        checkpoint_root = self.config.trainer.default_local_dir
        local_mkdir_safe(checkpoint_root)
        metadata = {
            "step": int(step),
            "checkpoint_dir": f"global_step_{int(step)}",
            "score": float(score),
            "metric": metric_selector,
            "metric_keys": metric_keys,
            "mode": str(self.config.trainer.get("best_validation_metric_mode", "max")).lower(),
            "updated_at_unix": time.time(),
        }
        path = self._best_validation_checkpoint_metadata_path()
        temporary_path = f"{path}.tmp.{uuid.uuid4().hex}"
        try:
            with open(temporary_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, sort_keys=True)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary_path, path)
            fsync_parent_directory(path)
        finally:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)
        self._best_validation_checkpoint_metadata_loaded = True
        self._best_validation_checkpoint_metadata = metadata

    def _commit_or_defer_best_validation_checkpoint(
        self, *, step: int, score: float, metric_selector: str, metric_keys: list[str]
    ) -> None:
        payload = {
            "step": int(step),
            "score": float(score),
            "metric_selector": str(metric_selector),
            "metric_keys": list(metric_keys),
        }
        if self._checkpoint_transaction_is_complete(step):
            self._write_best_validation_checkpoint_metadata(**payload)
            None
            self._cleanup_non_preserved_checkpoint_dirs()
            return
        self._pending_best_checkpoint_commits[int(step)] = payload
        self._pending_checkpoint_cleanup_steps.add(int(step))

    def _defer_checkpoint_cleanup_until_complete(self, step: int) -> None:
        if self._checkpoint_transaction_is_complete(step):
            self._cleanup_non_preserved_checkpoint_dirs()
            return
        self._pending_checkpoint_cleanup_steps.add(int(step))

    def _commit_completed_checkpoint_actions(self) -> None:
        """Publish metadata and run cleanup only after a successful save wait."""
        pending_commits = getattr(self, "_pending_best_checkpoint_commits", {})
        pending_cleanup = getattr(self, "_pending_checkpoint_cleanup_steps", set())
        committed_or_obsolete_steps: set[int] = set()
        for step in sorted(pending_commits):
            if not self._checkpoint_transaction_is_complete(step):
                continue
            payload = pending_commits[step]
            if self._best_validation_score_improved(float(payload["score"]), include_pending=False):
                self._write_best_validation_checkpoint_metadata(**payload)
                None
            committed_or_obsolete_steps.add(step)
        for step in committed_or_obsolete_steps:
            pending_commits.pop(step, None)
        completed_cleanup_steps = {
            step for step in pending_cleanup if self._checkpoint_transaction_is_complete(step)
        }
        pending_cleanup.difference_update(completed_cleanup_steps)
        if committed_or_obsolete_steps or completed_cleanup_steps:
            self._cleanup_non_preserved_checkpoint_dirs()

    def _discard_pending_checkpoint_actions_after_failure(self) -> None:
        """Keep durable metadata/retention unchanged after a failed transaction."""
        getattr(self, "_pending_best_checkpoint_commits", {}).clear()
        getattr(self, "_pending_checkpoint_cleanup_steps", set()).clear()
        getattr(self, "_hf_upload_pending_checkpoint_steps", set()).clear()

    def _extract_best_validation_score(self, val_metrics: dict[str, Any]) -> tuple[float, list[str]] | None:
        metric_selector = str(self.config.trainer.get("best_validation_metric", "acc/mean@16"))
        candidates: list[tuple[str, float]] = []
        for key, value in val_metrics.items():
            key_str = str(key)
            if key_str != metric_selector and (not key_str.endswith(f"/{metric_selector}")):
                continue
            try:
                value_float = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(value_float):
                candidates.append((key_str, value_float))
        if not candidates:
            print(f"Warning: best validation metric '{metric_selector}' not found in validation metrics.")
            return None
        if len(candidates) == 1:
            key, score = candidates[0]
            return (score, [key])
        aggregate = str(self.config.trainer.get("best_validation_metric_aggregate", "mean")).lower()
        values = [score for _, score in candidates]
        if aggregate == "max":
            score = max(values)
        elif aggregate == "min":
            score = min(values)
        else:
            score = float(np.mean(values))
        return (score, [key for key, _ in candidates])

    def _effective_best_validation_checkpoint_metadata(
        self, *, include_pending: bool = True
    ) -> dict[str, Any] | None:
        """Return the best durable or in-flight candidate for save decisions."""
        candidates: list[dict[str, Any]] = []
        durable = self._load_best_validation_checkpoint_metadata()
        if durable is not None:
            candidates.append(durable)
        if include_pending:
            candidates.extend(getattr(self, "_pending_best_checkpoint_commits", {}).values())
        mode = str(self.config.trainer.get("best_validation_metric_mode", "max")).lower()
        valid_candidates: list[tuple[float, dict[str, Any]]] = []
        for candidate in candidates:
            try:
                candidate_score = float(candidate["score"])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(candidate_score):
                valid_candidates.append((candidate_score, candidate))
        if not valid_candidates:
            return None
        selector = min if mode == "min" else max
        return selector(valid_candidates, key=lambda item: item[0])[1]

    def _best_validation_score_improved(self, score: float, *, include_pending: bool = True) -> bool:
        mode = str(self.config.trainer.get("best_validation_metric_mode", "max")).lower()
        min_delta = float(self.config.trainer.get("best_validation_metric_min_delta", 0.0) or 0.0)
        metadata = self._effective_best_validation_checkpoint_metadata(include_pending=include_pending)
        if not metadata or metadata.get("score") is None:
            return True
        try:
            best_score = float(metadata["score"])
        except (TypeError, ValueError):
            return True
        if mode == "min":
            return score < best_score - min_delta
        return score > best_score + min_delta

    def _maybe_update_best_validation_checkpoint(
        self,
        val_metrics: dict[str, Any],
        *,
        step: int,
        save_checkpoint: bool,
        checkpoint_already_saved: bool = False,
    ) -> dict[str, float]:
        metrics: dict[str, float] = {}
        if not self._best_validation_checkpoint_enabled():
            return metrics
        min_step = int(self.config.trainer.get("best_validation_checkpoint_min_step", 0) or 0)
        if int(step) < min_step:
            return metrics
        score_payload = self._extract_best_validation_score(val_metrics)
        if score_payload is None:
            return metrics
        score, metric_keys = score_payload
        metrics["best_validation/current_score"] = float(score)
        current_best = self._effective_best_validation_checkpoint_metadata()
        if current_best and current_best.get("score") is not None:
            try:
                metrics["best_validation/best_score"] = float(current_best["score"])
            except (TypeError, ValueError):
                pass
        if current_best and current_best.get("step") is not None:
            try:
                metrics["best_validation/best_step"] = float(current_best["step"])
            except (TypeError, ValueError):
                pass
        if not self._best_validation_score_improved(score):
            metrics["best_validation/improved"] = 0.0
            return metrics
        metric_selector = str(self.config.trainer.get("best_validation_metric", "acc/mean@16"))
        print(
            f"Best validation checkpoint improved: step={step}, score={score}, metric={metric_selector}, keys={metric_keys}"
        )
        if save_checkpoint and (not checkpoint_already_saved):
            self._save_checkpoint(exclude_from_max_ckpt_to_keep=True, cleanup=False, defer_hf_upload=False)
        self._commit_or_defer_best_validation_checkpoint(
            step=int(step), score=score, metric_selector=metric_selector, metric_keys=metric_keys
        )
        self._best_validation_checkpoint_saved_step = int(step)
        metrics["best_validation/improved"] = 1.0
        metrics["best_validation/best_score"] = float(score)
        metrics["best_validation/best_step"] = float(step)
        return metrics

    def _maybe_save_best_validation_checkpoint(
        self, val_metrics: dict[str, Any], *, checkpoint_already_saved: bool = False
    ) -> dict[str, float]:
        save_on_validation = bool(self.config.trainer.get("save_on_validation", False))
        if save_on_validation and not checkpoint_already_saved:
            self._save_checkpoint(exclude_from_max_ckpt_to_keep=True, cleanup=False, defer_hf_upload=False)
            checkpoint_already_saved = True
        if checkpoint_already_saved:
            self._validation_checkpoint_saved_step = int(self.global_steps)
        metrics = self._maybe_update_best_validation_checkpoint(
            val_metrics,
            step=int(self.global_steps),
            save_checkpoint=True,
            checkpoint_already_saved=checkpoint_already_saved,
        )
        if save_on_validation and not metrics.get("best_validation/improved", False):
            self._defer_checkpoint_cleanup_until_complete(int(self.global_steps))
        return metrics

    def _loaded_current_step_checkpoint_is_reusable(self) -> bool:
        """Return whether initial validation can reuse the checkpoint just loaded.

        Merely finding a same-step directory is insufficient: a fresh run may
        encounter stale output.  `_loaded_checkpoint_step` is published only
        after actor, optimizer, lag state, and dataloader restoration all
        succeed.  Requiring the actor directory to remain present also handles
        `del_local_ckpt_after_load=true`, where a new save is necessary.
        """
        if getattr(self, "_loaded_checkpoint_step", None) != int(self.global_steps):
            return False
        actor_path = os.path.join(
            str(self.config.trainer.default_local_dir), f"global_step_{int(self.global_steps)}", "actor"
        )
        return os.path.isdir(actor_path)

    @staticmethod
    def _validation_metrics_are_best(val_metrics: dict[str, Any]) -> bool:
        try:
            return float(val_metrics.get("best_validation/improved", 0.0) or 0.0) == 1.0
        except (TypeError, ValueError):
            return False

    def _print_validation_metrics(self, val_metrics: dict[str, Any], *, label: str = "Validation") -> None:
        print_label = "Best validation" if self._validation_metrics_are_best(val_metrics) else label
        pprint(f"{print_label} metrics: {val_metrics}")

    @staticmethod
    def _metric_scalar_float(value: Any) -> float | None:
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                return None
            value = value.detach().float().cpu().item()
        elif isinstance(value, np.ndarray):
            if value.size != 1:
                return None
            value = value.reshape(-1)[0]
        elif isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, (bool, int, float)):
            value = float(value)
            if np.isfinite(value):
                return value
        return None

    @staticmethod
    def _is_important_validation_acc_metric(key: str) -> bool:
        if not key.startswith(("val-core/", "val-x0-core/", "val-x1-core/", "val-frozen-x1-core/")):
            return False
        return re.search(r"/(score|acc|reward)/mean@\d+$", key) is not None

    @staticmethod
    def _is_important_loss_metric(key: str) -> bool:
        if key == "actor/pg_loss":
            return True
        if not key.startswith("distillation/"):
            return False
        metric_name = key.rsplit("/", 1)[-1]
        return metric_name.endswith(
            ("_loss", "_loss_raw", "_loss_weighted", "_pg_loss", "_ce_loss", "_awr_loss", "_kl_loss")
        )

    @staticmethod
    def _is_important_distillation_alignment_metric(key: str) -> bool:
        if not key.startswith("distillation/"):
            return False
        metric_name = key.rsplit("/", 1)[-1]
        return metric_name in {"topk_overlap_ratio", "topk_overlap_token_advantage"} or key.startswith(
            "distillation/oprd_projection/"
        )

    @classmethod
    def _add_important_metric_views(cls, metrics: dict[str, Any]) -> None:
        """Small dashboard views; preserve every original metric and its units."""
        views = {}
        aliases = {
            "train/task_score": (
                "reward_extra/score/mean",
                "reward_extra/acc/mean",
                "reward_extra/accuracy/mean",
            ),
            "train/reward": ("critic/rewards/mean",),
            "train/invalid_format": ("reward_extra/incorrect_format/mean",),
            "train/response_tokens": ("response_length/mean",),
            "train/constant_reward_groups": ("grpo/group_score/all_same_frac",),
            "train/grad_norm": ("actor/grad_norm",),
            "train/learning_rate": ("actor/lr",),
        }
        for alias, candidates in aliases.items():
            for key in candidates:
                value = cls._metric_scalar_float(metrics.get(key))
                if value is not None:
                    views[f"important/{alias}"] = value
                    break
        for key, value in metrics.items():
            if (
                cls._is_important_validation_acc_metric(key)
                or cls._is_important_loss_metric(key)
                or cls._is_important_distillation_alignment_metric(key)
                or key.startswith("best_validation/")
                or key
                in {
                    "distillation/oprd_lambda_plus",
                    "distillation/oprd_lambda_minus",
                    "distillation/kdrl_beta",
                }
            ):
                scalar = cls._metric_scalar_float(value)
                if scalar is not None:
                    views[f"important/{key}"] = scalar
        metrics.update(views)

    def _cleanup_non_preserved_checkpoint_dirs(self):
        """Keep scheduled models, the current best, latest resume, and pending saves."""
        checkpoint_root = self.config.trainer.default_local_dir
        if not checkpoint_root or not os.path.isdir(checkpoint_root):
            return
        preserved_steps = set(self._get_save_steps())
        try:
            save_freq = int(self.config.trainer.get("save_freq", 0) or 0)
        except (TypeError, ValueError):
            save_freq = 0
        best_validation_step = self._get_best_validation_checkpoint_step()
        if best_validation_step is not None:
            preserved_steps.add(best_validation_step)
        tracker_path = os.path.join(checkpoint_root, "latest_checkpointed_iteration.txt")
        tracker_step = None
        try:
            if os.path.isfile(tracker_path):
                with open(tracker_path, encoding="utf-8") as tracker_file:
                    tracker_step = int(tracker_file.read().strip())
                    preserved_steps.add(tracker_step)
        except (OSError, ValueError):
            pass
        preserved_steps.add(int(self.global_steps))
        preserved_steps.update(
            int(record["validation_step"])
            for record in getattr(self, "_async_validation_futures", [])
            if record.get("candidate_checkpoint_saved", False)
        )
        preserved_steps.update(getattr(self, "_pending_best_checkpoint_commits", {}).keys())
        preserved_steps.update(getattr(self, "_pending_checkpoint_cleanup_steps", set()))
        for name in os.listdir(checkpoint_root):
            if not name.startswith("global_step_"):
                continue
            step_text = name.removeprefix("global_step_")
            try:
                step = int(step_text)
            except ValueError:
                continue
            if self._fsdp_frequency_checkpoint_policy_enabled() and save_freq > 0 and (step % save_freq == 0):
                preserved_steps.add(step)
            if os.path.isfile(os.path.join(checkpoint_root, name, OPTIMIZER_RETENTION_MARKER)):
                preserved_steps.add(step)
            transaction_incomplete = not self._checkpoint_transaction_is_complete(step)
            transaction_failed = self._checkpoint_transaction_has_failure_marker(step)
            markerless_older_than_resume_target = (
                transaction_incomplete and tracker_step is not None and (tracker_step > step)
            )
            failed_non_target_transaction = (
                transaction_incomplete and transaction_failed and (step != tracker_step)
            )
            if failed_non_target_transaction:
                path = os.path.join(checkpoint_root, name)
                if os.path.isdir(path):
                    print(f"Removing failed checkpoint transaction: {path}")
                    shutil.rmtree(path, ignore_errors=True)
                continue
            if markerless_older_than_resume_target:
                path = os.path.join(checkpoint_root, name)
                if os.path.isdir(path):
                    print(f"Removing incomplete checkpoint transaction: {path}")
                    shutil.rmtree(path, ignore_errors=True)
                continue
            if (
                self._actor_checkpoint_async_save_enabled()
                and transaction_incomplete
                and (not transaction_failed)
                and (not markerless_older_than_resume_target)
            ):
                preserved_steps.add(step)
            if step in preserved_steps:
                continue
            path = os.path.join(checkpoint_root, name)
            if os.path.isdir(path):
                print(f"Removing non-preserved checkpoint directory: {path}")
                shutil.rmtree(path, ignore_errors=True)
        self._prune_stale_optimizer_states()

    def _prune_stale_optimizer_states(self):
        """Prune old resume payloads only after a newer checkpoint is committed.

        Public recipes keep optimizer, RNG/scheduler, dataloader, EMA teacher and
        adaptive-distillation state only at the latest committed resume target.
        Scheduled/best model shards and Hugging Face exports remain available.
        """
        if not self._fsdp_frequency_checkpoint_policy_enabled():
            return
        latest_resume_only = bool(self.config.trainer.get("keep_only_latest_resume_state", False))
        if not latest_resume_only and os.environ.get("RD_PRUNE_STALE_OPTIMIZER", "true").strip().lower() in {
            "0",
            "false",
            "no",
            "off",
        }:
            return
        checkpoint_root = self.config.trainer.default_local_dir
        if not checkpoint_root or not os.path.isdir(checkpoint_root):
            return
        step_dirs: dict[int, str] = {}
        for name in os.listdir(checkpoint_root):
            if not name.startswith("global_step_"):
                continue
            try:
                step_dirs[int(name.removeprefix("global_step_"))] = name
            except ValueError:
                continue
        if not step_dirs:
            return
        # A newer directory can contain a failed or still-writing save. It is
        # never sufficient evidence to discard the last working resume state.
        try:
            with open(
                os.path.join(checkpoint_root, "latest_checkpointed_iteration.txt"), encoding="utf-8"
            ) as stream:
                latest_step = int(stream.read().strip())
        except (OSError, ValueError):
            return
        if latest_step not in step_dirs or not self._checkpoint_transaction_is_complete(latest_step):
            return
        for step, name in sorted(step_dirs.items()):
            if step >= latest_step or not self._checkpoint_transaction_is_complete(step):
                continue
            if not latest_resume_only and os.path.isfile(
                os.path.join(checkpoint_root, name, OPTIMIZER_RETENTION_MARKER)
            ):
                continue
            step_dir = os.path.join(checkpoint_root, name)
            for dirpath, _dirs, files in os.walk(step_dir):
                for fname in files:
                    stale_resume = fname.startswith("optim_world_size_") and fname.endswith(".pt")
                    if latest_resume_only:
                        stale_resume |= fname == "data.pt" and dirpath == step_dir
                        stale_resume |= fname.startswith(
                            (
                                "extra_state_world_size_",
                                "distill_teacher_world_size_",
                                "distill_resume_state_world_size_",
                            )
                        ) and fname.endswith(".pt")
                    if stale_resume:
                        optim_path = os.path.join(dirpath, fname)
                        try:
                            os.remove(optim_path)
                            print(f"Pruned stale resume state: {optim_path}", flush=True)
                        except OSError as exc:
                            print(f"Failed to prune resume state {optim_path}: {exc}", flush=True)

    def _async_validation_reward_enabled(self) -> bool:
        return bool(self.config.trainer.get("async_validation_reward", False))

    def _async_validation_max_pending(self) -> int:
        return max(1, int(self.config.trainer.get("async_validation_max_pending", 1) or 1))

    def _async_validation_reward_num_cpus(self) -> int:
        reward_kwargs = self.config.reward_model.get("reward_kwargs", {}) or {}
        val_workers = reward_kwargs.get("val_reward_num_workers", reward_kwargs.get("reward_num_workers", 1))
        return max(1, int(val_workers or 1))

    def _shutdown_async_validation_executor(self) -> None:
        return None

    def _should_save_async_validation_candidate_checkpoint(self, validation_step: int) -> bool:
        if self.config.trainer.get("save_on_validation", False):
            return True
        if not self._best_validation_checkpoint_enabled():
            return False
        min_step = int(self.config.trainer.get("best_validation_checkpoint_min_step", 0) or 0)
        return int(validation_step) >= min_step

    @staticmethod
    def _stamp_validation_step(metrics: dict[str, Any], step: int) -> None:
        """Expose every validation record on both validation and training step axes."""
        source_step = float(step)
        metrics["validation/source_step"] = source_step
        metrics["training/global_step"] = source_step

    @staticmethod
    def _define_async_validation_metrics(logger, metrics: dict[str, Any]) -> None:
        logger.define_metrics("validation/source_step")
        metric_keys = [
            key
            for key in metrics.keys()
            if key != "validation/source_step" and (not str(key).startswith("val-hourly/"))
        ]
        if metric_keys:
            logger.define_metrics(metric_keys, step_metric="validation/source_step")

    def _schedule_async_validation_reward(
        self,
        logger,
        *,
        step_validation_due: bool,
        hourly_validation_due: bool,
        current_step_training_seconds: float = 0.0,
    ) -> None:
        self._log_completed_async_validations(logger, blocking=False)
        while len(getattr(self, "_async_validation_futures", [])) >= self._async_validation_max_pending():
            self._log_completed_async_validations(logger, blocking=True, max_to_log=1)
        validation_step = int(self.global_steps)
        payload = self._validate(defer_reward=True)
        if not payload:
            return
        candidate_checkpoint_saved = False
        if self._should_save_async_validation_candidate_checkpoint(validation_step):
            print(f"Saving async validation candidate checkpoint for step {validation_step}.")
            self._save_checkpoint(exclude_from_max_ckpt_to_keep=True, cleanup=False, defer_hf_upload=False)
            candidate_checkpoint_saved = True
            self._async_validation_candidate_checkpoint_steps.add(validation_step)
            self._validation_checkpoint_saved_step = validation_step
        validation_reward_num_cpus = self._async_validation_reward_num_cpus()
        future = compute_validation_reward_payload_async.options(num_cpus=validation_reward_num_cpus).remote(
            payload, config=self.config, tokenizer=self.tokenizer
        )
        self._async_validation_futures.append(
            {
                "future": future,
                "validation_step": validation_step,
                "step_validation_due": bool(step_validation_due),
                "hourly_validation_due": bool(hourly_validation_due),
                "current_step_training_seconds": float(current_step_training_seconds or 0.0),
                "candidate_checkpoint_saved": candidate_checkpoint_saved,
                "submitted_at": time.monotonic(),
            }
        )
        logger.add_wandb_step_barrier(validation_step)
        print(
            f"Scheduled async validation reward: step={validation_step}, pending={len(self._async_validation_futures)}, num_cpus={validation_reward_num_cpus}"
        )

    def _log_completed_async_validations(
        self, logger, *, blocking: bool, max_to_log: int | None = None
    ) -> list[dict[str, Any]]:
        pending = list(getattr(self, "_async_validation_futures", []))
        if not pending:
            return []
        remaining = []
        completed = []
        logged_count = 0
        for record_index, record in enumerate(pending):
            if max_to_log is not None and logged_count >= max_to_log:
                remaining.extend(pending[record_index:])
                break
            future = record["future"]
            should_collect = blocking
            if not blocking:
                ready, _ = ray.wait([future], timeout=0)
                should_collect = bool(ready)
            if not should_collect:
                remaining.extend(pending[record_index:])
                break
            scored_payload = ray.get(future)
            val_metrics = self._finalize_scored_validation_reward_payload(scored_payload)
            best_metrics = self._maybe_update_best_validation_checkpoint(
                val_metrics,
                step=int(record["validation_step"]),
                save_checkpoint=False,
                checkpoint_already_saved=bool(record["candidate_checkpoint_saved"]),
            )
            val_metrics.update(best_metrics)
            if (
                record["candidate_checkpoint_saved"]
                and float(best_metrics.get("best_validation/improved", 0.0) or 0.0) != 1.0
            ):
                self._defer_checkpoint_cleanup_until_complete(int(record["validation_step"]))
            log_metrics = dict(val_metrics)
            self._stamp_validation_step(log_metrics, record["validation_step"])
            log_metrics["validation/async_reward_elapsed_seconds"] = max(
                time.monotonic() - float(record["submitted_at"]), 0.0
            )
            if record["hourly_validation_due"]:
                hourly_val_metrics = self._build_hourly_validation_metrics(
                    val_metrics, additional_training_seconds=float(record["current_step_training_seconds"])
                )
                self._define_hourly_validation_metrics(logger, hourly_val_metrics)
                log_metrics.update(hourly_val_metrics)
            self._add_important_metric_views(log_metrics)
            self._define_async_validation_metrics(logger, log_metrics)
            validation_step = int(record["validation_step"])
            logger.log_async_validation(
                data=log_metrics,
                source_step=validation_step,
                collection_step=max(int(self.global_steps), validation_step),
            )
            logger.release_wandb_step_barrier(validation_step)
            self._print_validation_metrics(log_metrics, label="Async validation")
            completed.append(
                {
                    "validation_step": int(record["validation_step"]),
                    "metrics": val_metrics,
                    "step_validation_due": bool(record["step_validation_due"]),
                    "hourly_validation_due": bool(record["hourly_validation_due"]),
                }
            )
            logged_count += 1
        self._async_validation_futures = remaining
        if completed:
            self._cleanup_non_preserved_checkpoint_dirs()
        return completed

    def _save_checkpoint(
        self, exclude_from_max_ckpt_to_keep: bool = False, cleanup: bool = True, defer_hf_upload: bool = True
    ):
        from verl.utils.fs import local_mkdir_safe

        async_save = self._actor_checkpoint_async_save_enabled()
        critic_checkpoint = self.config.critic.get("checkpoint", {}) if self.use_critic else {}
        critic_async_save = bool(critic_checkpoint.get("async_save", False)) if critic_checkpoint else False
        if critic_async_save:
            raise RuntimeError(
                "Asynchronous critic checkpoints cannot safely publish through the trainer's shared root tracker without a coordinated multi-role transaction."
            )
        if async_save and self.use_critic:
            raise RuntimeError(
                "Asynchronous actor checkpoints are currently supported only for critic-free training; a coordinated actor+critic transaction is required before enabling this combination."
            )
        if async_save:
            self._wait_async_checkpoint_save(cleanup=False)
            if self._fsdp_frequency_checkpoint_policy_enabled():
                self._ensure_async_checkpoint_protocol_marker(self.global_steps)
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )
        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")
        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(
                self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor"
            )
        )
        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated, set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None)
            if not remove_previous_ckpt_in_save
            else 1
        )
        self.actor_rollout_wg.save_checkpoint(
            actor_local_path,
            actor_remote_path,
            self.global_steps,
            max_ckpt_to_keep=max_actor_ckpt_to_keep,
            exclude_from_max_ckpt_to_keep=exclude_from_max_ckpt_to_keep,
        )
        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path,
                critic_remote_path,
                self.global_steps,
                max_ckpt_to_keep=max_critic_ckpt_to_keep,
                exclude_from_max_ckpt_to_keep=exclude_from_max_ckpt_to_keep,
            )
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_temporary_path = None
        try:
            local_mkdir_safe(local_global_step_folder)
            dataloader_temporary_path = f"{dataloader_local_path}.tmp.{uuid.uuid4().hex}"
            dataloader_state_dict = self.train_dataloader.state_dict()
            torch.save(dataloader_state_dict, dataloader_temporary_path)
            with open(dataloader_temporary_path, "rb") as data_file:
                os.fsync(data_file.fileno())
            os.replace(dataloader_temporary_path, dataloader_local_path)
            fsync_parent_directory(dataloader_local_path)
        except BaseException as exc:
            if async_save:
                try:
                    self._write_trainer_sidecar_failure_marker(actor_local_path, exc)
                except Exception:
                    print("Warning: failed to publish trainer checkpoint-sidecar failure marker")
            raise
        finally:
            if dataloader_temporary_path is not None and os.path.exists(dataloader_temporary_path):
                os.remove(dataloader_temporary_path)
        if async_save:
            print("skip write latest_checkpointed_iteration.txt when async_save is True")
            self._async_checkpoint_save_pending = True
            if cleanup:
                if not hasattr(self, "_pending_checkpoint_cleanup_steps"):
                    self._pending_checkpoint_cleanup_steps = set()
                self._pending_checkpoint_cleanup_steps.add(int(self.global_steps))
            if defer_hf_upload:
                None
            return
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        tracker_temporary_path = f"{local_latest_checkpointed_iteration}.tmp.{uuid.uuid4().hex}"
        try:
            with open(tracker_temporary_path, "w", encoding="utf-8") as f:
                f.write(str(self.global_steps))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tracker_temporary_path, local_latest_checkpointed_iteration)
            fsync_parent_directory(local_latest_checkpointed_iteration)
        finally:
            if os.path.exists(tracker_temporary_path):
                os.remove(tracker_temporary_path)
        if defer_hf_upload:
            None
        if cleanup:
            self._cleanup_non_preserved_checkpoint_dirs()

    def _maybe_save_scheduled_checkpoint(self, *, is_last_step=False, esi_close_to_expiration=False) -> bool:
        explicit = int(self.global_steps) in self._get_save_steps()
        frequency = int(self.config.trainer.get("save_freq", 0))
        scheduled = frequency > 0 and (
            is_last_step or int(self.global_steps) % frequency == 0 or esi_close_to_expiration
        )
        if not explicit and not scheduled:
            return False
        already_saved = (
            getattr(self, "_best_validation_checkpoint_saved_step", None) == self.global_steps
            or getattr(self, "_validation_checkpoint_saved_step", None) == self.global_steps
            or self.global_steps in getattr(self, "_async_validation_candidate_checkpoint_steps", set())
        )
        if already_saved:
            return False
        self._save_checkpoint(
            exclude_from_max_ckpt_to_keep=explicit or self._fsdp_frequency_checkpoint_policy_enabled()
        )
        return True

    def _actor_checkpoint_async_save_enabled(self) -> bool:
        checkpoint_config = self.config.actor_rollout_ref.actor.checkpoint
        if hasattr(checkpoint_config, "async_save"):
            return bool(checkpoint_config.async_save)
        return bool(checkpoint_config.get("async_save", False))

    @staticmethod
    def _write_trainer_sidecar_failure_marker(actor_local_path: str, exc: BaseException) -> None:
        """Wake the rank-0 writer immediately when durable data.pt staging fails."""
        os.makedirs(actor_local_path, exist_ok=True)
        path = os.path.join(actor_local_path, ".trainer_sidecar_failed.json")
        temporary_path = f"{path}.tmp.{uuid.uuid4().hex}"
        try:
            with open(temporary_path, "w", encoding="utf-8") as stream:
                json.dump({"error": f"{type(exc).__name__}: {exc}"[:4096]}, stream, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, path)
            fsync_parent_directory(path)
        finally:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)

    def _poll_async_checkpoint_save(self) -> None:
        """Abort at the next rollout boundary after any background writer error."""
        if not self._actor_checkpoint_async_save_enabled() or not bool(
            getattr(self, "_async_checkpoint_save_pending", False)
        ):
            return
        poll = getattr(self.actor_rollout_wg, "raise_if_async_checkpoint_save_failed", None)
        if not callable(poll):
            self._async_checkpoint_save_pending = False
            self._discard_pending_checkpoint_actions_after_failure()
            raise RuntimeError(
                "Actor checkpoint async_save=true requires a worker-side nonblocking checkpoint-health RPC."
            )
        try:
            results = poll()
        except BaseException:
            self._async_checkpoint_save_pending = False
            self._discard_pending_checkpoint_actions_after_failure()
            raise
        if isinstance(results, (list, tuple)) and results and all((bool(result) for result in results)):
            self._async_checkpoint_save_pending = False
            self._commit_completed_checkpoint_actions()
            None

    def _wait_async_checkpoint_save(self, cleanup: bool = False) -> None:
        if not self._actor_checkpoint_async_save_enabled():
            return
        wait = getattr(self.actor_rollout_wg, "wait_async_checkpoint_save", None)
        if not callable(wait):
            self._async_checkpoint_save_pending = False
            self._discard_pending_checkpoint_actions_after_failure()
            raise RuntimeError(
                "Actor checkpoint async_save=true requires a worker-side blocking checkpoint-wait RPC."
            )
        try:
            wait()
        except BaseException:
            self._async_checkpoint_save_pending = False
            self._discard_pending_checkpoint_actions_after_failure()
            raise
        self._async_checkpoint_save_pending = False
        self._commit_completed_checkpoint_actions()
        None
        if cleanup:
            self._cleanup_non_preserved_checkpoint_dirs()

    def _preflight_resume_dataloader_state(self, global_step_folder: str) -> dict[str, Any] | None:
        """Reject incomplete scientific resumes before loading actor shards."""
        step_match = re.search("global_step_(\\d+)$", global_step_folder.rstrip(os.sep))
        if step_match is None:
            raise ValueError(f"Malformed checkpoint directory: {global_step_folder}")
        global_step = int(step_match.group(1))
        actor_path = os.path.join(global_step_folder, "actor")
        if not os.path.isdir(actor_path):
            raise FileNotFoundError(f"Checkpoint actor directory is missing: {actor_path}")
        marker_names = [name for name in os.listdir(actor_path) if name.startswith(".")]
        complete_marker = os.path.join(actor_path, ".checkpoint_complete.json")
        transaction_marked = os.path.isfile(complete_marker)
        async_marker_present = any(
            (
                name.startswith((".async_checkpoint_", ".worker_sidecars_", ".trainer_sidecar_"))
                for name in marker_names
            )
        )
        if async_marker_present and (not transaction_marked):
            raise RuntimeError(
                f"Refusing to resume an incomplete asynchronous checkpoint transaction before actor load: {global_step_folder}"
            )
        if transaction_marked:
            try:
                with open(complete_marker, encoding="utf-8") as stream:
                    marker_payload = json.load(stream)
                marker_step = int(marker_payload.get("global_step", -1))
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"Invalid checkpoint complete marker: {complete_marker}") from exc
            if marker_step != global_step:
                raise RuntimeError(
                    f"Checkpoint complete marker step mismatch: path={global_step}, marker={marker_step}"
                )
            tracker_path = os.path.join(
                os.path.dirname(global_step_folder), "latest_checkpointed_iteration.txt"
            )
            try:
                with open(tracker_path, encoding="utf-8") as stream:
                    tracker_step = int(stream.read().strip())
            except (OSError, ValueError) as exc:
                raise RuntimeError(
                    f"Committed async checkpoint has no valid root tracker: {tracker_path}"
                ) from exc
            if tracker_step < global_step:
                raise RuntimeError(
                    f"Checkpoint payload marker exists, but its final root-tracker commit is missing: checkpoint_step={global_step}, tracker_step={tracker_step}"
                )
        data_path = os.path.join(global_step_folder, "data.pt")
        dataloader_required = transaction_marked or None
        if not os.path.isfile(data_path):
            if dataloader_required:
                raise FileNotFoundError(
                    f"A complete dataloader state is mandatory for this scientific resume, but data.pt is missing: {data_path}"
                )
            return None
        try:
            return torch.load(data_path, map_location="cpu", weights_only=False)
        except BaseException as exc:
            raise RuntimeError(f"Failed to preload checkpoint dataloader state: {data_path}") from exc

    def _load_checkpoint(self):
        self._loaded_checkpoint_step = None
        if self.config.trainer.resume_mode == "disable":
            return 0
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        elif self.config.trainer.resume_mode == "resume_path":
            resume_from_path = self.config.trainer.resume_from_path
            if OmegaConf.is_config(resume_from_path):
                resume_from_path = OmegaConf.to_container(resume_from_path, resolve=True)
            assert resume_from_path is not None, "resume ckpt must be set"
            assert not isinstance(resume_from_path, (dict, list, tuple)), "resume ckpt must be a scalar path"
            resume_from_path = os.fspath(resume_from_path)
            assert "global_step_" in resume_from_path, "resume ckpt must specify the global_steps"
            global_step_folder = resume_from_path
            if not os.path.isabs(global_step_folder):
                working_dir = os.getcwd()
                global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        self.global_steps = int(global_step_folder.split("global_step_")[-1])
        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")
        dataloader_state_dict = self._preflight_resume_dataloader_state(global_step_folder)
        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if dataloader_state_dict is not None:
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")
        self._loaded_checkpoint_step = int(self.global_steps)

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm and (not self.use_reward_loop):
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm and (not self.use_reward_loop):
                self.rm_wg.stop_profile()

    def _get_dp_size(self, worker_group, role: str) -> int:
        """Get data parallel size from worker group dispatch info.

        This method retrieves the data parallel size by querying the dispatch info
        for the specified role. The dispatch info is cached for subsequent calls.

        Args:
            worker_group: The worker group to query dispatch info from.
            role: The role name (e.g., "actor", "critic") to get DP size for.

        Returns:
            The data parallel size (number of DP ranks).
        """
        if role not in worker_group._dispatch_info:
            dp_rank_mapping = worker_group._query_dispatch_info(role)
            worker_group._dispatch_info[role] = dp_rank_mapping
        else:
            dp_rank_mapping = worker_group._dispatch_info[role]
        return max(dp_rank_mapping) + 1

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens.

        When use_prefix_grouper is enabled, uses group-level balancing to keep samples with
        the same uid together on the same rank for prefix sharing optimization.
        """
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)
        workload_lst = calculate_workload(global_seqlen_lst)
        dp_size = self._get_dp_size(self.actor_rollout_wg, "actor")
        distillation_cfg = self.config.actor_rollout_ref.actor.get("distillation", None)
        multi_teacher_enabled = bool(
            distillation_cfg is not None and distillation_cfg.get("multi_teacher_enabled", False)
        )
        if multi_teacher_enabled:
            rank_domains = [str(value) for value in distillation_cfg.get("multi_teacher_domains", [])]
            if len(rank_domains) != dp_size:
                raise ValueError(
                    f"multi-teacher domain count must equal actor DP size; domains={len(rank_domains)}, dp_size={dp_size}"
                )
            if "data_source" not in batch.non_tensor_batch:
                raise ValueError("multi-teacher balancing requires non_tensor_batch['data_source']")
            if batch_size % dp_size != 0:
                raise ValueError(
                    f"multi-teacher batch size must be divisible by DP size; batch={batch_size}, dp={dp_size}"
                )
            expected_rank_size = batch_size // dp_size
            data_sources = np.asarray(batch.non_tensor_batch["data_source"], dtype=object).reshape(-1)
            global_partition_lst = []
            for rank, domain in enumerate(rank_domains):
                domain_indices = np.flatnonzero(data_sources == domain).tolist()
                if len(domain_indices) != expected_rank_size:
                    raise ValueError(
                        f"multi-teacher batches must contain equal domain counts; rank={rank}, domain={domain!r}, expected={expected_rank_size}, actual={len(domain_indices)}"
                    )
                global_partition_lst.append(domain_indices)
                metrics[f"multi_teacher/rank_{rank}_{domain}_rows"] = float(len(domain_indices))
        elif getattr(self, "use_prefix_grouper", False) and "uid" in batch.non_tensor_batch:
            from verl.utils.seqlen_balancing import get_group_balanced_partitions

            uid_list = list(batch.non_tensor_batch["uid"])
            seqlen_list = global_seqlen_lst.tolist()
            num_groups = len(set(uid_list))
            if num_groups % dp_size != 0:
                raise ValueError(
                    f"PrefixGrouper with balance_batch requires num_uid_groups ({num_groups}) % dp_size ({dp_size}) == 0. This ensures each rank gets equal number of groups. Current batch_size={batch_size}, adjust batch_size to be a multiple of dp_size * rollout.n."
                )
            global_partition_lst = get_group_balanced_partitions(
                seqlen_list=seqlen_list, uid_list=uid_list, k_partitions=dp_size
            )
        elif keep_minibatch:
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(workload_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(dp_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    workload_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=dp_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(
                workload_lst, k_partitions=dp_size, equal_size=True
            )
        if not getattr(self, "use_prefix_grouper", False):
            for idx, partition in enumerate(global_partition_lst):
                partition.sort(key=lambda x: (workload_lst[x], x))
                ordered_partition = partition[::2] + partition[1::2][::-1]
                global_partition_lst[idx] = ordered_partition
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst.tolist(), partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _compute_values(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            batch_td = left_right_2_no_padding(batch_td)
            tu.assign_non_tensor(batch_td, compute_loss=False)
            output = self.critic_wg.infer_batch(batch_td)
            output = output.get()
            values = tu.get(output, "values")
            values = no_padding_2_padding(values, batch_td)
            values = tu.get_tensordict({"values": values.float()})
            values = DataProto.from_tensordict(values)
        else:
            values = self.critic_wg.compute_values(batch)
        return values

    def _compute_ref_log_prob(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            batch_td = left_right_2_no_padding(batch_td)
            metadata = {"calculate_entropy": False, "compute_loss": False}
            if self.ref_in_actor:
                metadata["no_lora_adapter"] = True
            tu.assign_non_tensor(batch_td, **metadata)
            if self.ref_in_actor:
                output = self.actor_rollout_wg.compute_log_prob(batch_td)
            else:
                output = self.ref_policy_wg.compute_ref_log_prob(batch_td)
            log_probs = tu.get(output, "log_probs")
            log_probs = no_padding_2_padding(log_probs, batch_td)
            ref_log_prob = tu.get_tensordict({"ref_log_prob": log_probs.float()})
            ref_log_prob = DataProto.from_tensordict(ref_log_prob)
        else:
            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
        return ref_log_prob

    def _compute_old_log_prob(self, batch: DataProto, calculate_entropy: bool = False):
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            batch_td = left_right_2_no_padding(batch_td)
            tu.assign_non_tensor(batch_td, calculate_entropy=calculate_entropy, compute_loss=False)
            output = self.actor_rollout_wg.compute_log_prob(batch_td)
            log_probs = tu.get(output, "log_probs")
            old_log_prob_mfu = tu.get(output, "metrics")["mfu"]
            log_probs = no_padding_2_padding(log_probs, batch_td)
            old_log_prob_tensors = {"old_log_probs": log_probs.float()}
            if calculate_entropy:
                entropy = tu.get(output, "entropy")
                entropy = no_padding_2_padding(entropy, batch_td)
                old_log_prob_tensors["entropys"] = entropy.float()
            old_log_prob = tu.get_tensordict(old_log_prob_tensors)
            old_log_prob = DataProto.from_tensordict(old_log_prob)
        else:
            previous_calculate_entropy = batch.meta_info.get("calculate_entropy", None)
            batch.meta_info["calculate_entropy"] = calculate_entropy
            try:
                old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            finally:
                if previous_calculate_entropy is None:
                    batch.meta_info.pop("calculate_entropy", None)
                else:
                    batch.meta_info["calculate_entropy"] = previous_calculate_entropy
            old_log_prob_mfu = 0
        return (old_log_prob, old_log_prob_mfu)

    def _update_actor(self, batch: DataProto) -> DataProto:
        rollout_config = self.config.actor_rollout_ref.rollout
        batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
        batch.meta_info["temperature"] = rollout_config.temperature
        if batch.meta_info.get("eos_token_id", None) is None:
            batch.meta_info["eos_token_id"] = self.tokenizer.eos_token_id
        if batch.meta_info.get("pad_token_id", None) is None:
            batch.meta_info["pad_token_id"] = self.tokenizer.pad_token_id
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            batch_td = left_right_2_no_padding(batch_td)
            calculate_entropy = self.config.actor_rollout_ref.actor.entropy_coeff != 0.0
            ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            ppo_epochs = self.config.actor_rollout_ref.actor.ppo_epochs
            seed = self.config.actor_rollout_ref.actor.data_loader_seed
            shuffle = self.config.actor_rollout_ref.actor.shuffle
            tu.assign_non_tensor(
                batch_td,
                calculate_entropy=calculate_entropy,
                global_batch_size=ppo_mini_batch_size,
                mini_batch_size=ppo_mini_batch_size,
                epochs=ppo_epochs,
                seed=seed,
                dataloader_kwargs={"shuffle": shuffle},
            )
            actor_output = self.actor_rollout_wg.update_actor(batch_td)
            actor_output = tu.get(actor_output, "metrics")
            actor_output = rename_dict(actor_output, "actor/")
            actor_output["perf/mfu/actor"] = actor_output.pop("actor/mfu")
            actor_output = DataProto.from_single_dict(data={}, meta_info={"metrics": actor_output})
        else:
            actor_output = self.actor_rollout_wg.update_actor(batch)
        return actor_output

    def _update_critic(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            batch_td = left_right_2_no_padding(batch_td)
            ppo_mini_batch_size = self.config.critic.ppo_mini_batch_size
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            ppo_epochs = self.config.critic.ppo_epochs
            seed = self.config.critic.data_loader_seed
            shuffle = self.config.critic.shuffle
            tu.assign_non_tensor(
                batch_td,
                global_batch_size=ppo_mini_batch_size,
                mini_batch_size=ppo_mini_batch_size,
                epochs=ppo_epochs,
                seed=seed,
                dataloader_kwargs={"shuffle": shuffle},
            )
            output = self.critic_wg.train_mini_batch(batch_td)
            output = output.get()
            output = tu.get(output, "metrics")
            output = rename_dict(output, "critic/")
            output["perf/mfu/critic"] = output.pop("critic/mfu")
            critic_output = DataProto.from_single_dict(data={}, meta_info={"metrics": output})
        else:
            critic_output = self.critic_wg.update_critic(batch)
        return critic_output

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
            group_name=self.config.trainer.get("group_name", None),
        )
        self.global_steps = 0
        self._load_checkpoint()
        if self.global_steps >= self.total_training_steps and not self.config.trainer.get("val_only", False):
            # Shared teachers may already have reached a longer recipe's horizon.
            # Do not generate another batch or overwrite retained checkpoints.
            print(
                f"Training already complete: checkpoint step {self.global_steps} "
                f">= requested horizon {self.total_training_steps}"
            )
            logger.finish()
            return
        self._init_validation_schedule()
        current_epoch = self.global_steps // len(self.train_dataloader)
        distillation_cfg = self.config.actor_rollout_ref.actor.get("distillation", None)
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            print(
                f"Starting initial validation: prompts={len(self.val_dataset)}, repeat_n={self.config.actor_rollout_ref.rollout.val_kwargs.n}, effective_generations={len(self.val_dataset) * self.config.actor_rollout_ref.rollout.val_kwargs.n}, val_batch_size={self.val_dataloader.batch_size}"
            )
            if self._async_validation_reward_enabled() and (not self.config.trainer.get("val_only", False)):
                initial_hourly_due = self._next_hourly_validation_train_time is not None
                self._schedule_async_validation_reward(
                    logger, step_validation_due=True, hourly_validation_due=initial_hourly_due
                )
                if initial_hourly_due:
                    self._mark_hourly_validation_complete()
            else:
                val_metrics = self._validate()
                assert val_metrics, f"val_metrics={val_metrics!r}"
                self._stamp_validation_step(val_metrics, self.global_steps)
                val_metrics.update(
                    self._maybe_save_best_validation_checkpoint(
                        val_metrics,
                        checkpoint_already_saved=self._loaded_current_step_checkpoint_is_reusable(),
                    )
                )
                self._print_validation_metrics(val_metrics, label="Initial validation")
                init_metrics = dict(val_metrics)
                if self._next_hourly_validation_train_time is not None:
                    hourly_val_metrics = self._build_hourly_validation_metrics(val_metrics)
                    self._define_hourly_validation_metrics(logger, hourly_val_metrics)
                    init_metrics.update(hourly_val_metrics)
                    self._mark_hourly_validation_complete()
                self._add_important_metric_views(init_metrics)
                logger.log(data=init_metrics, step=self.global_steps, wandb_commit=True)
                if self.config.trainer.get("val_only", False):
                    None
                    logger.finish()
                    return
        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()
        progress_bar = tqdm(
            total=self.total_training_steps, initial=self.global_steps, desc="Training Progress"
        )
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0
        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False
        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                self._poll_async_checkpoint_save()
                None
                self._log_completed_async_validations(logger, blocking=False)
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                timing_raw = {}
                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                if "uid" not in batch.non_tensor_batch:
                    batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                    )
                gen_batch = self._get_gen_batch(batch)
                gen_batch.meta_info["global_steps"] = self.global_steps
                rollout_repeat_counts = gen_batch.meta_info.get(ROLLOUT_REPEAT_COUNTS_META_KEY)
                if rollout_repeat_counts is not None:
                    batch.meta_info[ROLLOUT_REPEAT_COUNTS_META_KEY] = list(rollout_repeat_counts)
                gen_batch_output = self._repeat_batch_for_rollout(
                    gen_batch, repeat_times=self.config.actor_rollout_ref.rollout.n
                )
                is_last_step = self.global_steps >= self.total_training_steps
                step_train_start_time = time.monotonic()
                with marked_timer("step", timing_raw):
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")
                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(
                                    gen_baseline_batch
                                )
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(
                                    gen_baseline_batch
                                )
                            batch = batch.union(gen_baseline_output)
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                if not self.use_reward_loop:
                                    rm_scores = self.rm_wg.compute_rm_score(batch)
                                else:
                                    assert self.reward_loop_manager is not None, "RewardLoopManager is None"
                                    rm_scores = self.reward_loop_manager.compute_rm_score(batch)
                                batch = batch.union(rm_scores)
                            reward_baseline_tensor = self._compute_or_extract_reward(
                                batch, reward_fn=self.reward_fn, sum_reward=True
                            )
                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))
                            batch.batch["reward_baselines"] = reward_baseline_tensor
                            del rm_scores, gen_baseline_batch, gen_baseline_output
                    batch = self._repeat_batch_for_rollout(
                        batch, repeat_times=self.config.actor_rollout_ref.rollout.n
                    )
                    batch = batch.union(gen_batch_output)
                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)
                    batch.meta_info["global_token_num"] = torch.sum(
                        batch.batch["attention_mask"], dim=-1
                    ).tolist()
                    images_seqlens_all = []
                    for multi_modal_input in batch.non_tensor_batch["multi_modal_inputs"]:
                        if "image_grid_thw" not in multi_modal_input.keys():
                            continue
                        images_seqlens_all.extend(multi_modal_input["images_seqlens"].tolist())
                    batch.meta_info["images_seqlens"] = images_seqlens_all
                    with marked_timer("reward", timing_raw, color="yellow"):
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            if not self.use_reward_loop:
                                reward_tensor = self.rm_wg.compute_rm_score(batch)
                            else:
                                assert self.reward_loop_manager is not None, "RewardLoopManager is None"
                                reward_tensor = self.reward_loop_manager.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)
                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(
                                data=batch, config=self.config, tokenizer=self.tokenizer
                            )
                        else:
                            reward_tensor, reward_extra_infos_dict = self._compute_or_extract_reward(
                                batch, reward_fn=self.reward_fn, return_dict=False
                            )
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get(
                        "bypass_mode", False
                    )
                    if bypass_recomputing_logprobs:
                        from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                        apply_bypass_mode(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            actor_config = self.config.actor_rollout_ref.actor
                            compute_old_entropy = bool(
                                actor_config.get("calculate_entropy", False)
                                or float(actor_config.get("entropy_coeff", 0.0)) != 0.0
                                or actor_config.get("log_old_entropy", False)
                            )
                            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(
                                batch, calculate_entropy=compute_old_entropy
                            )
                            old_log_prob_metrics = {"perf/mfu/actor_infer": old_log_prob_mfu}
                            if compute_old_entropy and "entropys" in old_log_prob.batch:
                                entropys = old_log_prob.batch["entropys"]
                                response_masks = batch.batch["response_mask"]
                                entropy_agg = agg_loss(
                                    loss_mat=entropys,
                                    loss_mask=response_masks,
                                    loss_agg_mode=actor_config.loss_agg_mode,
                                    loss_scale_factor=actor_config.loss_scale_factor,
                                )
                                old_log_prob_metrics["actor/entropy"] = entropy_agg.detach().item()
                            metrics.update(old_log_prob_metrics)
                            if "entropys" in old_log_prob.batch:
                                distillation_cfg = actor_config.get("distillation", None)
                                keep_old_entropys = False
                                if keep_old_entropys:
                                    old_log_prob.batch["old_entropys"] = old_log_prob.batch["entropys"]
                                old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))
                    assert "old_log_probs" in batch.batch, (
                        f'"old_log_prob" not in batch.batch.keys()={batch.batch.keys()!r}'
                    )
                    if self.use_reference_policy:
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            ref_log_prob = self._compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self._compute_values(batch)
                            batch = batch.union(values)
                    with marked_timer("adv", timing_raw, color="brown"):
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async and future_reward is not None:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor
                        metrics.update(
                            self._maybe_attach_oprd_invalid_format_mask(
                                batch, reward_tensor, reward_extra_infos_dict
                            )
                        )
                        distillation_data = self._maybe_build_distillation_batch(
                            batch, reward_tensor, reward_extra_infos_dict
                        )
                        if distillation_data is not None:
                            distillation_batch, distillation_metrics = distillation_data
                            batch = batch.union(distillation_batch)
                            metrics.update(distillation_metrics)
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update(
                                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            )
                            batch.meta_info["reward_extra_keys"] = list(reward_extra_infos_dict.keys())
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch,
                                kl_ctrl=self.kl_ctrl_in_reward,
                                kl_penalty=self.config.algorithm.kl_penalty,
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and (not bypass_recomputing_logprobs)
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import (
                                compute_rollout_correction_and_add_to_batch,
                            )

                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(
                                batch, rollout_corr_config
                            )
                            metrics.update(is_metrics)
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", False)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self._update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        batch.meta_info["global_steps"] = self.global_steps
                        batch.meta_info["total_training_steps"] = self.total_training_steps
                        with marked_timer("update_actor", timing_raw, color="red"):
                            actor_output = self._update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)
                explicit_validation_due = self.global_steps in self._get_validation_steps()
                frequency_validation_due = self.config.trainer.test_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                )
                step_validation_due = self.val_reward_fn is not None and (
                    explicit_validation_due or frequency_validation_due
                )
                current_step_training_seconds = max(time.monotonic() - step_train_start_time, 0.0)
                hourly_validation_due = self._should_run_hourly_validation(
                    is_last_step, current_step_training_seconds=current_step_training_seconds
                )
                if step_validation_due or hourly_validation_due:
                    with marked_timer("testing", timing_raw, color="green"):
                        if self._async_validation_reward_enabled() and (not is_last_step):
                            self._schedule_async_validation_reward(
                                logger,
                                step_validation_due=step_validation_due,
                                hourly_validation_due=hourly_validation_due,
                                current_step_training_seconds=current_step_training_seconds,
                            )
                        else:
                            if self._async_validation_reward_enabled():
                                self._log_completed_async_validations(logger, blocking=True)
                            val_metrics: dict = self._validate()
                            self._stamp_validation_step(val_metrics, self.global_steps)
                            val_metrics.update(self._maybe_save_best_validation_checkpoint(val_metrics))
                            if is_last_step:
                                last_val_metrics = val_metrics
                            else:
                                self._print_validation_metrics(val_metrics)
                    if self._async_validation_reward_enabled() and (not is_last_step):
                        if hourly_validation_due:
                            self._mark_hourly_validation_complete(
                                elapsed_training_seconds=self._elapsed_training_seconds(
                                    additional_seconds=current_step_training_seconds
                                )
                            )
                    else:
                        if step_validation_due:
                            metrics.update(val_metrics)
                        if hourly_validation_due:
                            hourly_val_metrics = self._build_hourly_validation_metrics(
                                val_metrics, additional_training_seconds=current_step_training_seconds
                            )
                            self._define_hourly_validation_metrics(logger, hourly_val_metrics)
                            metrics.update(hourly_val_metrics)
                            self._mark_hourly_validation_complete(
                                elapsed_training_seconds=self._elapsed_training_seconds(
                                    additional_seconds=current_step_training_seconds
                                )
                            )
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                with marked_timer("save_checkpoint", timing_raw, color="green"):
                    self._maybe_save_scheduled_checkpoint(
                        is_last_step=is_last_step, esi_close_to_expiration=esi_close_to_expiration
                    )
                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and (not next_step_profile)
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile
                completed_step_training_seconds = self._get_completed_step_training_seconds(timing_raw)
                self._pure_training_seconds += completed_step_training_seconds
                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                        "training/elapsed_hours": self._elapsed_training_hours(),
                    }
                )
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                gradient_norm = metrics.get("actor/grad_norm", None)
                metrics.update(compute_variance_proxy_metrics(batch=batch, gradient_norm=gradient_norm))
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)
                self._add_important_metric_views(metrics)
                logger.log(data=metrics, step=self.global_steps, wandb_commit=True)
                progress_bar.update(1)
                self.global_steps += 1
                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )
                if is_last_step:
                    self._log_completed_async_validations(logger, blocking=True)
                    None
                    if last_val_metrics is not None:
                        self._print_validation_metrics(last_val_metrics, label="Final validation")
                    progress_bar.close()
                    logger.finish()
                    return
                if hasattr(self.train_dataset, "on_batch_end"):
                    self.train_dataset.on_batch_end(batch=batch)
        self._log_completed_async_validations(logger, blocking=True)
        None
        logger.finish()

    def _get_hourly_validation_freq_seconds(self) -> float:
        return 0.0

    def _init_validation_schedule(self) -> None:
        self._fit_start_time_monotonic = time.monotonic()
        self._pure_training_seconds = 0.0
        self._async_validation_futures = []
        self._async_validation_candidate_checkpoint_steps = set()
        self._hourly_validation_freq_seconds = self._get_hourly_validation_freq_seconds()
        if self._hourly_validation_freq_seconds > 0:
            self._next_hourly_validation_train_time = self._hourly_validation_freq_seconds
        else:
            self._next_hourly_validation_train_time = None

    def _elapsed_training_seconds(self, additional_seconds: float = 0.0) -> float:
        base_seconds = float(getattr(self, "_pure_training_seconds", 0.0) or 0.0)
        return max(base_seconds + max(additional_seconds, 0.0), 0.0)

    def _elapsed_training_hours(self, additional_seconds: float = 0.0) -> float:
        return self._elapsed_training_seconds(additional_seconds=additional_seconds) / 3600.0

    @staticmethod
    def _get_completed_step_training_seconds(timing_raw: dict[str, float]) -> float:
        step_seconds = float(timing_raw.get("step", 0.0) or 0.0)
        non_training_seconds = sum(
            (
                float(timing_raw.get(metric_name, 0.0) or 0.0)
                for metric_name in ("testing", "save_checkpoint", "stop_profile")
            )
        )
        return max(step_seconds - non_training_seconds, 0.0)

    def _should_run_hourly_validation(
        self, is_last_step: bool, current_step_training_seconds: float = 0.0
    ) -> bool:
        if self.val_reward_fn is None or self._next_hourly_validation_train_time is None:
            return False
        if is_last_step:
            return True
        return (
            self._elapsed_training_seconds(additional_seconds=current_step_training_seconds)
            >= self._next_hourly_validation_train_time
        )

    def _mark_hourly_validation_complete(self, elapsed_training_seconds: Optional[float] = None) -> None:
        if self._next_hourly_validation_train_time is None:
            return
        if elapsed_training_seconds is None:
            elapsed_training_seconds = self._elapsed_training_seconds()
        while self._next_hourly_validation_train_time <= elapsed_training_seconds:
            self._next_hourly_validation_train_time += self._hourly_validation_freq_seconds

    @staticmethod
    def _rewrite_hourly_validation_metric_key(metric_name: str) -> str:
        prefix, sep, suffix = metric_name.partition("/")
        if sep and prefix.startswith("val-"):
            return f"{prefix.replace('val-', 'val-hourly-', 1)}/{suffix}"
        return f"val-hourly/{metric_name}"

    def _build_hourly_validation_metrics(
        self, val_metrics: dict[str, float], additional_training_seconds: float = 0.0
    ) -> dict[str, float]:
        hourly_metrics: dict[str, float] = {
            "val-hourly/elapsed_hours": self._elapsed_training_hours(
                additional_seconds=additional_training_seconds
            ),
            "val-hourly/global_step": float(self.global_steps),
        }
        for metric_name, metric_value in val_metrics.items():
            hourly_metrics[self._rewrite_hourly_validation_metric_key(metric_name)] = metric_value
        return hourly_metrics

    @staticmethod
    def _define_hourly_validation_metrics(logger, hourly_metrics: dict[str, float]) -> None:
        step_metric = "val-hourly/elapsed_hours"
        logger.define_metrics(["val-hourly/elapsed_hours", "val-hourly/global_step"])
        metric_keys = [
            key
            for key in hourly_metrics.keys()
            if key not in {"val-hourly/elapsed_hours", "val-hourly/global_step"}
        ]
        if metric_keys:
            logger.define_metrics(metric_keys, step_metric=step_metric)
