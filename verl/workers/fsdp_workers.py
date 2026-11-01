"""
The main entry point to run the PPO algorithm
"""

import copy
import datetime
import hashlib
import json
import logging
import os
import re
import warnings
from dataclasses import asdict
from typing import Any, Optional
import numpy as np
import psutil
import torch
import torch.distributed
import torch.distributed as dist
from codetiming import Timer
from omegaconf import DictConfig, OmegaConf, open_dict
from safetensors.torch import save_file
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import FullStateDictConfig, ShardedStateDictConfig, StateDictType

try:
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor
import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.activation_offload import enable_activation_offloading
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager, atomic_torch_save
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.dataset.rl_dataset import build_plain_completion_prompt, use_plain_completion_prompt_format
from verl.utils.device import (
    get_device_id,
    get_device_name,
    get_nccl_backend,
    get_torch_device,
    set_expandable_segments,
)
from verl.utils.flops_counter import FlopsCounter
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    _reshape_scalar_parameters_for_fsdp,
    apply_fsdp2,
    collect_lora_params,
    fsdp2_load_full_state_dict,
    fsdp_version,
    get_fsdp_state_ctx,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    get_shard_placement_fn,
    init_fn,
    layered_summon_lora_params,
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
    replace_lora_wrapper,
)
from verl.utils.import_utils import import_external_libs
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils.model import (
    compute_position_id_with_mask,
    convert_weight_keys,
    expand_packed_qwen_weights_for_vllm,
)
from verl.utils.profiler import (
    DistProfiler,
    DistProfilerExtension,
    ProfilerConfig,
    log_gpu_memory_usage,
    simple_timer,
)
from verl.utils.profiler.performance import reduce_timing, topk_reduce_ratio_min_max
from verl.utils.py_functional import convert_to_regular_types
from verl.utils.ray_utils import get_event_loop
from verl.workers.config import FSDPCriticConfig, FSDPEngineConfig, HFModelConfig, RolloutConfig
from verl.workers.config.optimizer import build_optimizer


def _check_distillation_resume_errors(local_error: Exception | None, context: str) -> None:
    """Fail on every rank before a peer enters FSDP state-loading collectives."""
    if dist.is_available() and dist.is_initialized():
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, str(local_error) if local_error is not None else None)
        failures = [f"rank {rank}: {error}" for rank, error in enumerate(errors) if error is not None]
        if failures:
            raise RuntimeError(f"{context}: " + "; ".join(failures)) from local_error
    elif local_error is not None:
        raise local_error


def _w2s_model_artifact_manifest(path: str) -> dict[str, object]:
    """Build a cheap, relocation-stable identity manifest for a frozen model artifact."""
    normalized_path = os.path.realpath(path) if os.path.exists(path) else str(path)
    digest = hashlib.sha256()
    files_seen = 0
    total_weight_bytes = 0
    metadata_names = {
        "adapter_config.json",
        "config.json",
        "generation_config.json",
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    }
    weight_suffixes = (".bin", ".pt", ".pth", ".safetensors")
    if os.path.isdir(normalized_path):
        for root, dirnames, filenames in os.walk(normalized_path):
            dirnames.sort()
            for filename in sorted(filenames):
                full_path = os.path.join(root, filename)
                relative_path = os.path.relpath(full_path, normalized_path).replace(os.sep, "/")
                try:
                    size = os.path.getsize(full_path)
                except OSError:
                    continue
                is_metadata = filename in metadata_names or filename.endswith(".index.json")
                is_weight = filename.endswith(weight_suffixes)
                if not (is_metadata or is_weight):
                    continue
                files_seen += 1
                digest.update(relative_path.encode("utf-8"))
                digest.update(str(size).encode("ascii"))
                if is_weight:
                    total_weight_bytes += size
                else:
                    try:
                        with open(full_path, "rb") as stream:
                            for chunk in iter(lambda: stream.read(1048576), b""):
                                digest.update(chunk)
                    except OSError:
                        digest.update(b"<unreadable>")
    else:
        digest.update(str(path).encode("utf-8"))
    return {
        "manifest_sha256": digest.hexdigest(),
        "manifest_file_count": files_seen,
        "weight_bytes": total_weight_bytes,
    }


def _w2s_frozen_source_metadata(
    *,
    role: str,
    source_path: str,
    local_path: str,
    revision: Optional[str] = None,
    resolved_commit_hash: Optional[str] = None,
    adapter_source_path: Optional[str] = None,
    adapter_local_path: Optional[str] = None,
) -> dict[str, object]:
    """Describe and fingerprint one immutable W2S frozen policy source."""
    metadata: dict[str, object] = {
        "role": role,
        "source_path": str(source_path),
        "revision": str(revision) if revision is not None and str(revision).strip() else None,
        "resolved_commit_hash": str(resolved_commit_hash)
        if resolved_commit_hash is not None and str(resolved_commit_hash).strip()
        else None,
        "model_artifact": _w2s_model_artifact_manifest(local_path),
        "adapter_source_path": str(adapter_source_path)
        if adapter_source_path is not None and str(adapter_source_path).strip()
        else None,
        "adapter_artifact": _w2s_model_artifact_manifest(adapter_local_path)
        if adapter_local_path is not None and str(adapter_local_path).strip()
        else None,
    }
    identity_payload = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    metadata["identity_sha256"] = hashlib.sha256(identity_payload).hexdigest()
    return metadata


def _w2s_module_commit_hash(module: Optional[torch.nn.Module]) -> Optional[str]:
    """Read the Hugging Face resolved commit hash through common FSDP wrappers."""
    candidates = [module]
    if module is not None:
        candidates.extend([getattr(module, "module", None), getattr(module, "_fsdp_wrapped_module", None)])
    for candidate in candidates:
        config = getattr(candidate, "config", None) if candidate is not None else None
        commit_hash = getattr(config, "_commit_hash", None) if config is not None else None
        if commit_hash is not None and str(commit_hash).strip():
            return str(commit_hash)
    return None


from verl.workers.rollout import get_rollout_class
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))
device_name = get_device_name()


def _distributed_init_method_from_env():
    if os.environ.get("WG_BACKEND") == "ray":
        return None
    return os.environ.get("DIST_INIT_METHOD") or None


def create_device_mesh(world_size, fsdp_size):
    if fsdp_size < 0 or fsdp_size >= world_size:
        device_mesh = init_device_mesh(device_name, mesh_shape=(world_size,), mesh_dim_names=["fsdp"])
    else:
        device_mesh = init_device_mesh(
            device_name, mesh_shape=(world_size // fsdp_size, fsdp_size), mesh_dim_names=["ddp", "fsdp"]
        )
    return device_mesh


def get_sharding_strategy(device_mesh, zero3_enable=True):
    from torch.distributed.fsdp import ShardingStrategy

    if zero3_enable:
        fsdp_strategy = ShardingStrategy.FULL_SHARD
        hsdp_strategy = ShardingStrategy.HYBRID_SHARD
    else:
        fsdp_strategy = ShardingStrategy.SHARD_GRAD_OP
        hsdp_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2
    if device_mesh.ndim == 1:
        sharding_strategy = fsdp_strategy
    elif device_mesh.ndim == 2:
        sharding_strategy = hsdp_strategy
    else:
        raise NotImplementedError(f"Get device mesh ndim={device_mesh.ndim}, but only support 1 or 2")
    return sharding_strategy


def get_vl_model_vision_tower(vl_model_instance):
    """
    Util to extract Vision Tower from a VL model instance
    """
    if hasattr(vl_model_instance, "model") and hasattr(vl_model_instance.model, "visual"):
        return vl_model_instance.model.visual
    elif hasattr(vl_model_instance, "visual"):
        return vl_model_instance.visual
    return None


class ActorRolloutRefWorker(Worker, DistProfilerExtension):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    def __init__(self, config: DictConfig, role: str, **kwargs):
        Worker.__init__(self)
        self.config = config
        import torch.distributed

        device_name = get_device_name()
        if not torch.distributed.is_initialized():
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            backend = "gloo" if device_name == "cpu" else f"cpu:gloo,{device_name}:{get_nccl_backend()}"
            torch.distributed.init_process_group(
                backend=backend,
                rank=rank,
                world_size=world_size,
                timeout=datetime.timedelta(seconds=self.config.get("nccl_timeout", 600)),
                init_method=_distributed_init_method_from_env(),
            )
        world_size = torch.distributed.get_world_size()
        self.device_mesh = create_device_mesh(
            world_size=world_size, fsdp_size=self.config.actor.fsdp_config.fsdp_size
        )
        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.actor.get("ulysses_sequence_parallel_size", 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                device_name, mesh_shape=(dp, self.ulysses_sequence_parallel_size), mesh_dim_names=["dp", "sp"]
            )
        if self.ulysses_device_mesh is not None:
            is_collect = self.ulysses_device_mesh["sp"].get_local_rank() == 0
            self._register_dispatch_collect_info(
                "actor", dp_rank=self.ulysses_device_mesh["dp"].get_local_rank(), is_collect=is_collect
            )
        else:
            self._register_dispatch_collect_info("actor", dp_rank=self.rank, is_collect=True)
        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)
        self._lora_rank = 0
        self._is_lora = False
        self.role = role
        assert self.role in ["actor", "rollout", "ref", "actor_rollout", "actor_rollout_ref"]
        self._is_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._is_rollout = self.role in ["rollout", "actor_rollout", "actor_rollout_ref"]
        self._is_ref = self.role in ["ref", "actor_rollout_ref"]
        self.use_orig_params = self.config.actor.fsdp_config.get("use_orig_params", False)
        if self._is_actor:
            omega_profiler_config = config.actor.get("profiler", {})
        elif self._is_rollout:
            omega_profiler_config = config.rollout.get("profiler", {})
        elif self._is_ref:
            omega_profiler_config = config.ref.get("profiler", {})
        else:
            raise ValueError(
                f"Invalid role {self.role}, should be one of ['actor', 'rollout', 'ref', 'actor_rollout', 'actor_rollout_ref']"
            )
        profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)
        if omega_profiler_config.get("tool", None) in ["npu", "nsys", "torch", "torch_memory"]:
            tool_config = omega_conf_to_dataclass(
                omega_profiler_config.get("tool_config", {}).get(omega_profiler_config.get("tool"))
            )
        else:
            tool_config = None
        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config)
        )
        self._is_offload_param = False
        self._is_offload_optimizer = False
        if self._is_actor:
            self._is_offload_param = self.config.actor.fsdp_config.get("param_offload", False)
            self._is_offload_optimizer = self.config.actor.fsdp_config.get("optimizer_offload", False)
        elif self._is_ref:
            self._is_offload_param = self.config.ref.fsdp_config.get("param_offload", False)
        if self._is_actor:
            self.config.actor.ppo_mini_batch_size *= self.config.rollout.n
            self.config.actor.ppo_mini_batch_size //= (
                self.device_mesh.size() // self.ulysses_sequence_parallel_size
            )
            assert self.config.actor.ppo_mini_batch_size > 0, (
                f"ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be larger than 0 after normalization"
            )
            if self.config.actor.ppo_micro_batch_size is not None:
                self.config.actor.ppo_micro_batch_size //= (
                    self.device_mesh.size() // self.ulysses_sequence_parallel_size
                )
                self.config.actor.ppo_micro_batch_size_per_gpu = self.config.actor.ppo_micro_batch_size
            if self.config.actor.ppo_micro_batch_size_per_gpu is not None:
                assert (
                    self.config.actor.ppo_mini_batch_size % self.config.actor.ppo_micro_batch_size_per_gpu
                    == 0
                ), (
                    f"normalized ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be divisible by ppo_micro_batch_size_per_gpu {self.config.actor.ppo_micro_batch_size_per_gpu}"
                )
                assert (
                    self.config.actor.ppo_mini_batch_size // self.config.actor.ppo_micro_batch_size_per_gpu
                    > 0
                ), (
                    f"normalized ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be larger than ppo_micro_batch_size_per_gpu {self.config.actor.ppo_micro_batch_size_per_gpu}"
                )
        if self._is_rollout and self.config.rollout.log_prob_micro_batch_size is not None:
            self.config.rollout.log_prob_micro_batch_size //= (
                self.device_mesh.size() // self.ulysses_sequence_parallel_size
            )
            self.config.rollout.log_prob_micro_batch_size_per_gpu = (
                self.config.rollout.log_prob_micro_batch_size
            )
        if self._is_ref and self.config.ref.log_prob_micro_batch_size is not None:
            self.config.ref.log_prob_micro_batch_size //= (
                self.device_mesh.size() // self.ulysses_sequence_parallel_size
            )
            self.config.ref.log_prob_micro_batch_size_per_gpu = self.config.ref.log_prob_micro_batch_size

    def _build_model_optimizer(
        self,
        model_path,
        fsdp_config: FSDPEngineConfig,
        optim_config,
        override_model_config,
        use_remove_padding=False,
        use_fused_kernels=False,
        enable_gradient_checkpointing=False,
        trust_remote_code=False,
        use_liger=False,
        role="actor",
        enable_activation_offload=False,
        use_prefix_grouper=False,
        use_tiled_mlp=False,
        tiled_mlp_shards=4,
        lora_adapter_path_override=None,
        lora_adapter_path_override_is_set: bool = False,
        lora_rank_override=None,
        frozen_validation_adapter_path=None,
        force_cpu_offload: Optional[bool] = None,
        lora_is_trainable_override: Optional[bool] = None,
        wrap_fsdp: bool = True,
    ):
        from torch.distributed.fsdp import CPUOffload, MixedPrecision
        from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForImageTextToText

        try:
            from transformers import AutoModelForVision2Seq
        except ImportError:
            AutoModelForVision2Seq = AutoModelForImageTextToText
        from verl.utils.model import (
            get_generation_config,
            print_model_size,
            sanitize_hf_config_for_model_load,
            update_model_config,
        )
        from verl.utils.torch_dtypes import PrecisionType

        assert role in ["actor", "ref"]
        if use_tiled_mlp and self.config.actor.strategy == "fsdp":
            raise ValueError("TiledMLP requires FSDP2. Set `actor_rollout_ref.actor.strategy=fsdp2`.")
        log_gpu_memory_usage(f"Before init {role} from HF AutoModel", logger=logger)
        local_path = model_path
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        self.processor = hf_processor(local_path, trust_remote_code=trust_remote_code)
        if self.config.model.get("custom_chat_template", None) is not None:
            if self.processor is not None:
                self.processor.chat_template = self.config.model.custom_chat_template
            else:
                self.tokenizer.chat_template = self.config.model.custom_chat_template
        torch_dtype = fsdp_config.get("model_dtype", None)
        if torch_dtype is None:
            torch_dtype = torch.float32 if self._is_actor else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)
        attn_implementation = override_model_config.get("attn_implementation", "flash_attention_2")
        if attn_implementation == "eager":
            raise ValueError(
                "attn_implementation=eager is forbidden; use flash_attention_2/flash_attention_3 instead."
            )
        actor_model_config = AutoConfig.from_pretrained(
            local_path, trust_remote_code=trust_remote_code, attn_implementation=attn_implementation
        )
        sanitize_hf_config_for_model_load(actor_model_config)
        if self.ulysses_sequence_parallel_size > 1 and hasattr(actor_model_config, "vision_config"):
            raise ValueError(
                "Ulysses vision eager fallback is forbidden; disable Ulysses SP or use a text-only model."
            )
        if (
            getattr(actor_model_config, "model_type", None) == "qwen2_5_vl"
            and attn_implementation == "flash_attention_3"
            and hasattr(actor_model_config, "vision_config")
        ):
            actor_model_config.vision_config._attn_implementation = "flash_attention_2"
        if getattr(actor_model_config, "model_type", None) == "kimi_vl":
            actor_model_config.text_config.topk_method = "greedy"
        self.generation_config = get_generation_config(local_path, trust_remote_code=trust_remote_code)
        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config)
        update_model_config(actor_model_config, override_config_kwargs=override_config_kwargs)
        if self.rank == 0:
            print(f"Model config after override: {actor_model_config}")
        full_init_on_all_ranks = not wrap_fsdp or (
            torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
            and (
                os.environ.get("VERL_FSDP_FULL_INIT_ON_ALL_RANKS", "0").strip().lower()
                in {"1", "true", "yes", "on"}
            )
        )
        if (
            torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
            and (not full_init_on_all_ranks)
        ):
            mesh_coord = self.device_mesh.get_coordinate()
            is_fsdp_source_rank = mesh_coord is not None and mesh_coord[-1] == 0
        else:
            is_fsdp_source_rank = True
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not getattr(actor_model_config, "tie_word_embeddings", False)
            and (not full_init_on_all_ranks),
            mesh=None if full_init_on_all_ranks else self.device_mesh,
        )
        if self.rank == 0 and full_init_on_all_ranks:
            print(
                "FSDP init: loading full model weights on every rank and skipping sync_module_states broadcast."
            )
        elif (
            self.rank == 0 and torch.distributed.is_initialized() and (torch.distributed.get_world_size() > 1)
        ):
            print("FSDP init: loading HF checkpoint only on source ranks; other ranks build from config.")
        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            has_remote_code = hasattr(actor_model_config, "auto_map") and any(
                (actor_model_config.architectures[0] in val for val in actor_model_config.auto_map.values())
            )
            if has_remote_code:
                auto_class = next(
                    (
                        k
                        for k, v in actor_model_config.auto_map.items()
                        if actor_model_config.architectures[0] in v
                    )
                )
                match auto_class:
                    case "AutoModelForVision2Seq":
                        actor_module_class = AutoModelForVision2Seq
                    case "AutoModelForCausalLM":
                        actor_module_class = AutoModelForCausalLM
                    case "AutoModelForImageTextToText":
                        actor_module_class = AutoModelForImageTextToText
                    case _:
                        actor_module_class = AutoModel
            elif type(actor_model_config) in AutoModelForVision2Seq._model_mapping.keys():
                actor_module_class = AutoModelForVision2Seq
            elif type(actor_model_config) in AutoModelForCausalLM._model_mapping.keys():
                actor_module_class = AutoModelForCausalLM
            elif type(actor_model_config) in AutoModelForImageTextToText._model_mapping.keys():
                actor_module_class = AutoModelForImageTextToText
            else:
                actor_module_class = AutoModel
            if is_fsdp_source_rank:
                actor_module = actor_module_class.from_pretrained(
                    pretrained_model_name_or_path=local_path,
                    torch_dtype=torch_dtype,
                    config=actor_model_config,
                    trust_remote_code=trust_remote_code,
                    attn_implementation=attn_implementation,
                )
            else:
                actor_module = actor_module_class.from_config(
                    actor_model_config, trust_remote_code=trust_remote_code
                )
            if use_liger:
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance

                _apply_liger_kernel_to_instance(model=actor_module)
            fused_kernel_options = self.config.model.get("fused_kernel_options", None)
            fused_kernels_backend = (
                fused_kernel_options.get("impl_backend", None) if fused_kernel_options is not None else None
            )
            apply_monkey_patch(
                model=actor_module,
                use_remove_padding=use_remove_padding,
                ulysses_sp_size=self.ulysses_sequence_parallel_size,
                use_fused_kernels=use_fused_kernels,
                fused_kernels_backend=fused_kernels_backend,
                use_prefix_grouper=use_prefix_grouper,
                use_tiled_mlp=use_tiled_mlp,
                tiled_mlp_shards=tiled_mlp_shards,
            )
            from oprd.runtime.full_finetuning import validate_model

            validate_model(actor_module)
            actor_module.to(torch_dtype)
            if enable_gradient_checkpointing:
                actor_module.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
        if not wrap_fsdp:
            if optim_config is not None:
                raise ValueError("unsharded model replicas are inference-only")
            actor_module.to(torch.device(device_name, get_device_id()))
            actor_module.eval()
            for parameter in actor_module.parameters():
                parameter.requires_grad_(False)
            if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
                torch.distributed.barrier()
            log_gpu_memory_usage(f"After init unsharded {role} replica", logger=logger)
            return (actor_module, None, None, actor_model_config)
        self.use_orig_params = fsdp_config.get("use_orig_params", False)
        if self.config.actor.get("freeze_vision_tower", False):
            vision_tower = get_vl_model_vision_tower(actor_module)
            if vision_tower is not None:
                vision_tower.requires_grad_(False)
                self.use_orig_params = True
                if self.rank == 0:
                    print("[actor model] Vision tower is set to not trainable.")
            elif self.rank == 0:
                print("[actor model] No vision tower found.")
        torch.distributed.barrier()
        if self.rank == 0:
            print_model_size(actor_module)
        log_gpu_memory_usage(f"After init {role} from HF AutoModel", logger=logger)
        mixed_precision_config = fsdp_config.get("mixed_precision", None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get("reduce_dtype", "fp32"))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get("buffer_dtype", "fp32"))
        else:
            param_dtype = PrecisionType.to_dtype(fsdp_config.dtype)
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32
        mixed_precision = MixedPrecision(
            param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype
        )
        auto_wrap_policy = get_fsdp_wrap_policy(
            module=actor_module, config=fsdp_config.get("wrap_policy", None), is_lora=self._is_lora
        )
        if self.rank == 0:
            print(f"wrap_policy: {auto_wrap_policy}")
        fsdp_mesh = self.device_mesh
        fsdp_enable_zero3 = fsdp_config.reshard_after_forward
        sharding_strategy = get_sharding_strategy(fsdp_mesh, fsdp_enable_zero3)
        if force_cpu_offload is None:
            cpu_offload = None if role == "actor" else CPUOffload(offload_params=True)
        else:
            cpu_offload = CPUOffload(offload_params=True) if force_cpu_offload else None
        fsdp_strategy = self.config.actor.strategy
        if fsdp_strategy == "fsdp":
            reshaped_scalar_params = _reshape_scalar_parameters_for_fsdp(actor_module)
            if reshaped_scalar_params > 0 and self.rank == 0:
                print(f"Reshaped {reshaped_scalar_params} scalar parameters to 1D tensors for FSDP.")
            actor_module_fsdp = FSDP(
                actor_module,
                cpu_offload=cpu_offload,
                param_init_fn=init_fn,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=sharding_strategy,
                mixed_precision=mixed_precision,
                sync_module_states=not full_init_on_all_ranks,
                device_mesh=self.device_mesh,
                use_orig_params=self.use_orig_params,
                forward_prefetch=fsdp_config.get("forward_prefetch", False),
            )
        elif fsdp_strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, (
                "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            )
            mp_policy = MixedPrecisionPolicy(
                param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=True
            )
            if role == "actor" and fsdp_config.offload_policy:
                cpu_offload = CPUOffloadPolicy(pin_memory=True)
                self._is_offload_param = False
                self._is_offload_optimizer = False
            elif force_cpu_offload is None:
                cpu_offload = None if role == "actor" else CPUOffloadPolicy(pin_memory=True)
            else:
                cpu_offload = CPUOffloadPolicy(pin_memory=True) if force_cpu_offload else None
            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "mp_policy": mp_policy,
                "offload_policy": cpu_offload,
                "reshard_after_forward": fsdp_config.reshard_after_forward,
                "shard_placement_fn": get_shard_placement_fn(fsdp_size=self.device_mesh.shape[-1]),
            }
            full_state = actor_module.state_dict()
            apply_fsdp2(actor_module, fsdp_kwargs, fsdp_config)
            fsdp2_load_full_state_dict(actor_module, full_state, fsdp_mesh, cpu_offload)
            actor_module_fsdp = actor_module
        else:
            raise NotImplementedError(f"not implement {fsdp_strategy}")
        if enable_activation_offload:
            enable_activation_offloading(actor_module_fsdp, fsdp_strategy, enable_gradient_checkpointing)
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            torch.distributed.barrier()
        log_gpu_memory_usage(f"After {role} FSDP init", logger=logger)
        if role == "actor" and optim_config is not None:
            from verl.utils.torch_functional import (
                get_constant_schedule_with_warmup,
                get_cosine_schedule_with_warmup,
            )

            actor_optimizer = build_optimizer(actor_module_fsdp.parameters(), optim_config)
            total_steps = optim_config.get("total_training_steps", 0)
            num_warmup_steps = int(optim_config.get("lr_warmup_steps", -1))
            lr_scheduler_type = optim_config.get("lr_scheduler_type", "constant")
            min_lr_ratio = optim_config.get("min_lr_ratio", 0.0)
            num_cycles = optim_config.get("num_cycles", 0.5)
            if num_warmup_steps < 0:
                num_warmup_steps_ratio = optim_config.get("lr_warmup_steps_ratio", 0.0)
                num_warmup_steps = int(num_warmup_steps_ratio * total_steps)
            if self.rank == 0:
                print(f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}")
            if lr_scheduler_type == "constant":
                actor_lr_scheduler = get_constant_schedule_with_warmup(
                    optimizer=actor_optimizer, num_warmup_steps=num_warmup_steps
                )
            elif lr_scheduler_type == "cosine":
                actor_lr_scheduler = get_cosine_schedule_with_warmup(
                    optimizer=actor_optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=total_steps,
                    min_lr_ratio=min_lr_ratio,
                    num_cycles=num_cycles,
                )
            else:
                raise NotImplementedError(f"LR scheduler type {lr_scheduler_type} is not supported")
            log_gpu_memory_usage(f"After {role} optimizer init", logger=logger)
        else:
            actor_optimizer = None
            actor_lr_scheduler = None
        return (actor_module_fsdp, actor_optimizer, actor_lr_scheduler, actor_model_config)

    def _build_rollout(self, trust_remote_code=False):
        from torch.distributed.device_mesh import init_device_mesh

        rollout_config: RolloutConfig = omega_conf_to_dataclass(self.config.rollout)
        model_config: HFModelConfig = omega_conf_to_dataclass(self.config.model, dataclass_type=HFModelConfig)
        self.model_config = model_config
        infer_tp = self.config.rollout.tensor_model_parallel_size * self.config.rollout.data_parallel_size
        infer_pp = self.config.rollout.pipeline_model_parallel_size
        infer_world_size = infer_tp * infer_pp
        dp = self.world_size // infer_world_size
        assert self.world_size % infer_world_size == 0, (
            f"rollout world_size: {self.world_size} is not divisible by infer_world_size: {infer_world_size}"
        )
        rollout_device_mesh = init_device_mesh(
            device_name, mesh_shape=(dp, infer_tp, infer_pp), mesh_dim_names=["dp", "infer_tp", "infer_pp"]
        )
        rollout_name = self.config.rollout.name
        self.rollout_device_mesh = rollout_device_mesh
        if rollout_name == "hf":
            self._register_dispatch_collect_info("rollout", dp_rank=self.rank, is_collect=True)
        else:
            is_collect = (
                rollout_device_mesh["infer_tp"].get_local_rank() == 0
                and rollout_device_mesh["infer_pp"].get_local_rank() == 0
            )
            self._register_dispatch_collect_info(
                "rollout", dp_rank=rollout_device_mesh["dp"].get_local_rank(), is_collect=is_collect
            )
        self.torch_random_states = get_torch_device().get_rng_state()
        gen_dp_rank = rollout_device_mesh["dp"].get_local_rank()
        get_torch_device().manual_seed(gen_dp_rank + 1000)
        self.gen_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.torch_random_states)
        log_gpu_memory_usage(f"Before building {self.config.rollout.name} rollout", logger=logger)
        self.rollout = get_rollout_class(rollout_config.name, rollout_config.mode)(
            config=rollout_config, model_config=model_config, device_mesh=rollout_device_mesh
        )
        log_gpu_memory_usage(f"After building {self.config.rollout.name} rollout", logger=logger)
        force_full_state_dict = os.environ.get("VERL_FSDP_FULL_INIT_ON_ALL_RANKS", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if (torch.distributed.get_world_size() == 1 or force_full_state_dict) and fsdp_version(
            self.actor_module_fsdp
        ) == 1:
            FSDP.set_state_dict_type(
                self.actor_module_fsdp,
                state_dict_type=StateDictType.FULL_STATE_DICT,
                state_dict_config=FullStateDictConfig(rank0_only=False),
            )
        elif fsdp_version(self.actor_module_fsdp) == 1:
            FSDP.set_state_dict_type(
                self.actor_module_fsdp,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )
        self.base_sync_done: bool = "dummy" not in self.config.rollout.load_format
        self.layered_summon = self.config.rollout.get("layered_summon", False)

    async def rollout_mode(self, sync_weights: bool = True):
        """Context switch hybridengine to rollout mode."""
        aggressive_empty_cache(force_sync=True)
        if not sync_weights and self.base_sync_done:
            set_expandable_segments(False)
            if self.config.rollout.free_cache_engine:
                await self.rollout.resume(tags=["weights"])
                await self.rollout.resume(tags=["kv_cache"])
            self.torch_random_states = get_torch_device().get_rng_state()
            get_torch_device().set_rng_state(self.gen_random_states)
            return
        if not sync_weights and (not self.base_sync_done):
            logger.info(
                "skip rollout weight sync requested before initial sync; syncing initial rollout weights once"
            )
        log_gpu_memory_usage("Before load_fsdp_model_to_gpu", logger=logger)
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        log_gpu_memory_usage("After load_fsdp_model_to_gpu", logger=logger)
        peft_config = None
        lora_params = None
        peft_model = getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
        if hasattr(peft_model, "peft_config"):
            peft_config = peft_model.peft_config.get("default", None)
            params = collect_lora_params(
                module=self.actor_module_fsdp,
                layered_summon=self.config.rollout.get("layered_summon", False),
                base_sync_done=self.base_sync_done,
            )
            if not self.base_sync_done:
                lora_params = collect_lora_params(
                    module=self.actor_module_fsdp,
                    layered_summon=self.config.rollout.get("layered_summon", False),
                    base_sync_done=True,
                )
        else:
            params = self.actor_module_fsdp.state_dict()
        params = expand_packed_qwen_weights_for_vllm(
            params, getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
        )
        if peft_config is not None and (not self.base_sync_done):
            params = {replace_lora_wrapper(k, peft_config): v for k, v in params.items()}
        if isinstance(params, dict):
            sample_keys = list(params.keys())[:8]
            sync_msg = f"[VERL_ROLLOUT_SYNC] rank={self.rank} base_sync_done={self.base_sync_done} peft={peft_config is not None} param_count={len(params)} sample_keys={sample_keys}"
            logger.warning(sync_msg)
            print(sync_msg, flush=True)
        if (
            peft_config is not None
            and self.base_sync_done
            and (getattr(self.rollout, "sleep_level", None) == 2)
        ):
            base_model_params = collect_lora_params(
                module=self.actor_module_fsdp, layered_summon=self.layered_summon, base_sync_done=False
            )
            base_model_params = expand_packed_qwen_weights_for_vllm(
                base_model_params,
                getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp),
            )
            base_model_params = {
                replace_lora_wrapper(k, peft_config): v for k, v in base_model_params.items()
            }
        log_gpu_memory_usage("Before offload_fsdp_model_to_cpu", logger=logger)
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        log_gpu_memory_usage("After offload_fsdp_model_to_cpu", logger=logger)
        set_expandable_segments(False)
        if peft_config is not None and self.base_sync_done:
            per_tensor_param = params.items() if isinstance(params, dict) else params
        else:
            device = get_device_id()
            per_tensor_param = (
                (
                    name,
                    param.to(device, non_blocking=True).full_tensor()
                    if isinstance(param, DTensor)
                    else param,
                )
                for name, param in params.items()
            )
        if self.config.rollout.free_cache_engine:
            await self.rollout.resume(tags=["weights"])
        log_gpu_memory_usage("After resume weights", logger=logger)
        if (
            peft_config is not None
            and self.base_sync_done
            and (getattr(self.rollout, "sleep_level", None) == 2)
        ):
            per_tensor_base_params = (
                (
                    name,
                    param.to(device, non_blocking=True).full_tensor()
                    if isinstance(param, DTensor)
                    else param,
                )
                for name, param in base_model_params.items()
            )
            await self.rollout.update_weights(per_tensor_base_params, base_sync_done=False)
            del base_model_params, per_tensor_base_params
        await self.rollout.update_weights(
            per_tensor_param, peft_config=peft_config, base_sync_done=self.base_sync_done
        )
        if lora_params is not None:
            lora_msg = f"[VERL_ROLLOUT_SYNC] rank={self.rank} loading initial LoRA tensors lora_count={len(lora_params)} sample_keys={list(lora_params.keys())[:8]}"
            logger.warning(lora_msg)
            print(lora_msg, flush=True)
            await self.rollout.update_weights(
                lora_params.items(), peft_config=peft_config, base_sync_done=True
            )
        log_gpu_memory_usage("After update_weights", logger=logger)
        del params, per_tensor_param, lora_params
        aggressive_empty_cache(force_sync=True)
        if self.config.rollout.free_cache_engine:
            await self.rollout.resume(tags=["kv_cache"])
        log_gpu_memory_usage("After resume kv_cache", logger=logger)
        self.base_sync_done = True
        self.torch_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.gen_random_states)

    async def trainer_mode(self):
        """Context switch hybridengine to trainer mode."""
        if self.config.rollout.free_cache_engine:
            log_gpu_memory_usage("Before rollout offload", logger=logger)
            await self.rollout.release()
            log_gpu_memory_usage("After rollout offload", logger=logger)
        self.actor_module_fsdp.train()
        aggressive_empty_cache(force_sync=True)
        set_expandable_segments(True)
        self.gen_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.torch_random_states)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from verl.workers.actor import DataParallelPPOActor

        import_external_libs(self.config.model.get("external_lib", None))
        override_model_config = OmegaConf.to_container(
            OmegaConf.create(self.config.model.get("override_config", {}))
        )
        use_remove_padding = self.config.model.get("use_remove_padding", False)
        use_shm = self.config.model.get("use_shm", False)
        use_fused_kernels = self.config.model.get("use_fused_kernels", False)
        use_prefix_grouper = hasattr(self.config, "actor") and self.config.actor.get(
            "use_prefix_grouper", False
        )
        actor_distillation_cfg_for_ref = (
            self.config.actor.get("distillation", None) if self._is_actor else None
        )
        ref_used_as_oprd_student_base = False
        actor_local_path = None
        local_frozen_validation_adapter_path = None
        if self._is_actor or self._is_rollout:
            if self._is_actor:
                optim_config = self.config.actor.optim
                fsdp_config = omega_conf_to_dataclass(self.config.actor.fsdp_config)
                frozen_validation_adapter_path = None
                if frozen_validation_adapter_path is not None and str(frozen_validation_adapter_path).strip():
                    local_frozen_validation_adapter_path = copy_to_local(
                        str(frozen_validation_adapter_path), use_shm=use_shm
                    )
            else:
                optim_config = None
                fsdp_config = FSDPEngineConfig()
            local_path = copy_to_local(self.config.model.path, use_shm=use_shm)
            actor_local_path = local_path
            tiled_mlp_config = self.config.model.get("tiled_mlp", {})
            use_tiled_mlp = tiled_mlp_config.get("enabled", False)
            tiled_mlp_shards = tiled_mlp_config.get("num_shards", 4)
            self.actor_module_fsdp, self.actor_optimizer, self.actor_lr_scheduler, self.actor_model_config = (
                self._build_model_optimizer(
                    model_path=local_path,
                    fsdp_config=fsdp_config,
                    optim_config=optim_config,
                    override_model_config=override_model_config,
                    use_remove_padding=use_remove_padding,
                    use_fused_kernels=use_fused_kernels,
                    enable_gradient_checkpointing=self.config.model.get(
                        "enable_gradient_checkpointing", False
                    ),
                    trust_remote_code=self.config.model.get("trust_remote_code", False),
                    use_liger=self.config.model.get("use_liger", False),
                    role="actor",
                    enable_activation_offload=self.config.model.get("enable_activation_offload", False),
                    use_prefix_grouper=use_prefix_grouper,
                    use_tiled_mlp=use_tiled_mlp,
                    tiled_mlp_shards=tiled_mlp_shards,
                    frozen_validation_adapter_path=local_frozen_validation_adapter_path,
                )
            )
            if fsdp_version(self.actor_module_fsdp) == 1:
                self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module
            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)
                log_gpu_memory_usage("After offload actor model during init", logger=logger)
            if self._is_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)
                log_gpu_memory_usage("After offload actor optimizer during init", logger=logger)
        if self._is_actor:
            actor_cfg = omega_conf_to_dataclass(self.config.actor)
            agent_opd = self.config.get("agent_opd", {})
            agent_oprd = self.config.get("agent_oprd", {})
            agent_kdrl = self.config.get("agent_kdrl", {})
            if (
                sum((bool(profile.get("enabled", False)) for profile in (agent_opd, agent_oprd, agent_kdrl)))
                > 1
            ):
                raise ValueError("Agent OPD, OPRD and KDRL cannot be enabled together")
            actor_class = DataParallelPPOActor
            agent_kwargs = {}
            if agent_opd.get("enabled", False):
                from oprd.agent.opd_actor import AgentOPDActor

                actor_class = AgentOPDActor
                agent_kwargs["opd_config"] = agent_opd
            elif agent_oprd.get("enabled", False):
                from oprd.agent.oprd_actor import AgentOPRDActor

                actor_class = AgentOPRDActor
                agent_kwargs["oprd_config"] = agent_oprd
            elif agent_kdrl.get("enabled", False):
                from oprd.agent.kdrl_actor import AgentKDRLActor

                actor_class = AgentKDRLActor
                agent_kwargs["kdrl_config"] = agent_kdrl
            if self.config.get("agent_grpo", {}).get("enabled", False):
                from oprd.agent.grpo_actor import AgentGRPOActor

                actor_class = AgentGRPOActor
            self.actor = actor_class(
                config=actor_cfg,
                actor_module=self.actor_module_fsdp,
                actor_optimizer=self.actor_optimizer,
                data_parallel_group=self.ulysses_device_mesh["dp"].get_group()
                if self.ulysses_device_mesh is not None
                else None,
                **agent_kwargs,
            )
            if getattr(self, "tokenizer", None) is not None:
                self.actor.tokenizer = self.tokenizer
            if local_frozen_validation_adapter_path is not None:
                self.actor.set_frozen_validation_adapter("frozen_validation")
        defer_rollout_build_for_oprd_student_base = self._is_rollout and ref_used_as_oprd_student_base
        if self._is_rollout and (not defer_rollout_build_for_oprd_student_base):
            self._build_rollout(trust_remote_code=self.config.model.get("trust_remote_code", False))
        if self._is_ref:
            ref_model_path = self.config.model.path
            ref_model = self.config.ref.get("model", None)
            if ref_model is not None:
                ref_model_path = ref_model.get("path", self.config.model.path)
            if self.rank == 0:
                print("reference model:", ref_model_path)
            ref_local_path = copy_to_local(ref_model_path, use_shm=use_shm)
            ref_tiled_mlp_config = self.config.ref.get("tiled_mlp", None)
            if ref_tiled_mlp_config is None:
                ref_tiled_mlp_config = self.config.model.get("tiled_mlp", {})
            ref_use_tiled_mlp = ref_tiled_mlp_config.get("enabled", False)
            ref_tiled_mlp_shards = ref_tiled_mlp_config.get("num_shards", 4)
            self.ref_module_fsdp = self._build_model_optimizer(
                model_path=ref_local_path,
                fsdp_config=omega_conf_to_dataclass(self.config.ref.fsdp_config),
                optim_config=None,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="ref",
                use_prefix_grouper=use_prefix_grouper,
                use_tiled_mlp=ref_use_tiled_mlp,
                tiled_mlp_shards=ref_tiled_mlp_shards,
                force_cpu_offload=False if ref_used_as_oprd_student_base else None,
            )[0]
            OmegaConf.set_struct(self.config.ref, True)
            with open_dict(self.config.ref):
                self.config.ref.use_remove_padding = use_remove_padding
                self.config.ref.use_fused_kernels = use_fused_kernels
                if use_prefix_grouper:
                    self.config.ref.use_prefix_grouper = use_prefix_grouper
            self.ref_policy = DataParallelPPOActor(config=self.config.ref, actor_module=self.ref_module_fsdp)
            if getattr(self, "tokenizer", None) is not None:
                self.ref_policy.tokenizer = self.tokenizer
        if self._is_actor:
            distillation_cfg = self.config.actor.get("distillation", None)
            loss_mode = self.config.actor.policy_loss.get("loss_mode", "vanilla")
            kdrl_enabled = distillation_cfg is not None and bool(distillation_cfg.get("kdrl_enabled", False))
            vanilla_on_policy_mix_enabled = (
                distillation_cfg is not None and loss_mode == "vanilla" and kdrl_enabled
            )
            oprd_enabled = distillation_cfg is not None and bool(distillation_cfg.get("oprd_enabled", False))
            delta_target_enabled = distillation_cfg is not None and bool(
                distillation_cfg.get("delta_target_enabled", False)
            )
            w2s_opd_enabled = distillation_cfg is not None and bool(
                distillation_cfg.get("w2s_opd_enabled", False)
            )
            if delta_target_enabled and oprd_enabled:
                raise ValueError("Delta-target KDRL/OPD baselines require distillation.oprd_enabled=false.")
            if w2s_opd_enabled and (delta_target_enabled or oprd_enabled):
                raise ValueError("W2S-OPD cannot be combined with Direct-OPD delta targets or OPRD.")
            if w2s_opd_enabled:
                missing_w2s_sources = [
                    key
                    for key in (
                        "w2s_positive_model_path",
                        "w2s_negative_model_path",
                        "w2s_student_base_model_path",
                    )
                    if distillation_cfg.get(key, None) is None or not str(distillation_cfg.get(key)).strip()
                ]
                if missing_w2s_sources:
                    raise ValueError(
                        f"W2S-OPD requires explicit immutable frozen-model paths; it will not derive student_base from the current/resumed actor. Missing: {missing_w2s_sources}."
                    )
            oprd_h1_source = (
                str(distillation_cfg.get("oprd_h1_source", "teacher_base")).strip().lower().replace("-", "_")
                if distillation_cfg is not None
                else "teacher_base"
            )
            if oprd_enabled and oprd_h1_source not in {"teacher_base", "teacher_student", "opsd_kl"}:
                raise ValueError(f"Unsupported paper guidance direction: {oprd_h1_source}")
            oprd_requires_base = (
                oprd_enabled and oprd_h1_source == "teacher_base" or delta_target_enabled or w2s_opd_enabled
            )
            oprd_requires_student_base = False
            if distillation_cfg is not None and (
                loss_mode in {"opsd", "guided_opsd", "kd"} or vanilla_on_policy_mix_enabled or oprd_enabled
            ):

                def _build_frozen_self_teacher(model_path_override=None):
                    """Build a separate self-teacher module without deepcopy(FSDP)."""
                    if actor_local_path is None:
                        raise RuntimeError(
                            "actor_local_path is not initialized for self-teacher construction."
                        )
                    frozen_model_path = actor_local_path
                    if model_path_override is not None and str(model_path_override).strip():
                        frozen_model_path = copy_to_local(str(model_path_override), use_shm=use_shm)
                    saved_tokenizer = getattr(self, "tokenizer", None)
                    saved_processor = getattr(self, "processor", None)
                    saved_generation_config = getattr(self, "generation_config", None)
                    prev_offload_param = self._is_offload_param
                    prev_offload_optimizer = self._is_offload_optimizer
                    teacher_module = self._build_model_optimizer(
                        model_path=frozen_model_path,
                        fsdp_config=omega_conf_to_dataclass(self.config.actor.fsdp_config),
                        optim_config=None,
                        override_model_config=override_model_config,
                        use_remove_padding=use_remove_padding,
                        use_fused_kernels=use_fused_kernels,
                        enable_gradient_checkpointing=self.config.model.get(
                            "enable_gradient_checkpointing", False
                        ),
                        trust_remote_code=self.config.model.get("trust_remote_code", False),
                        use_liger=self.config.model.get("use_liger", False),
                        role="actor",
                        enable_activation_offload=self.config.model.get("enable_activation_offload", False),
                        use_prefix_grouper=self.config.actor.get("use_prefix_grouper", False),
                        use_tiled_mlp=self.config.model.get("tiled_mlp", {}).get("enabled", False),
                        tiled_mlp_shards=self.config.model.get("tiled_mlp", {}).get("num_shards", 4),
                    )[0]
                    self.tokenizer = saved_tokenizer
                    self.processor = saved_processor
                    self.generation_config = saved_generation_config
                    self._is_offload_param = prev_offload_param
                    self._is_offload_optimizer = prev_offload_optimizer
                    teacher_module.eval()
                    for param in teacher_module.parameters():
                        param.requires_grad_(False)
                    return teacher_module

                def _build_external_teacher():
                    teacher_model_path = (
                        distillation_cfg.get("w2s_positive_model_path", None)
                        if w2s_opd_enabled
                        else distillation_cfg.get("teacher_model_path", None)
                    )
                    if w2s_opd_enabled and (
                        teacher_model_path is None or not str(teacher_model_path).strip()
                    ):
                        teacher_model_path = distillation_cfg.get("teacher_model_path", None)
                    multi_teacher_enabled = bool(distillation_cfg.get("multi_teacher_enabled", False))
                    self._multi_teacher_expected_domain = None
                    if multi_teacher_enabled:
                        multi_teacher_domains = [
                            str(value) for value in distillation_cfg.get("multi_teacher_domains", [])
                        ]
                        multi_teacher_model_paths = [
                            str(value) for value in distillation_cfg.get("multi_teacher_model_paths", [])
                        ]
                        world_size = torch.distributed.get_world_size()
                        if (
                            len(multi_teacher_domains) != world_size
                            or len(multi_teacher_model_paths) != world_size
                        ):
                            raise ValueError(
                                f"multi-teacher mode requires exactly one domain and model path per actor rank; world_size={world_size}, domains={len(multi_teacher_domains)}, paths={len(multi_teacher_model_paths)}"
                            )
                        if len(set(multi_teacher_domains)) != len(multi_teacher_domains):
                            raise ValueError(
                                f"multi-teacher domains contain duplicates: {multi_teacher_domains}"
                            )
                        teacher_model_path = multi_teacher_model_paths[self.rank]
                        self._multi_teacher_expected_domain = multi_teacher_domains[self.rank]
                        print(
                            f"[multi-teacher] rank={self.rank} domain={self._multi_teacher_expected_domain} model={teacher_model_path}"
                        )
                    if teacher_model_path is None or not str(teacher_model_path).strip():
                        return None
                    if not _requires_external_teacher_module():
                        if self.rank == 0:
                            print(
                                "Skipping actor-side external teacher module; teacher rollout/cache covers it."
                            )
                        return None
                    teacher_lora_adapter_path = None if multi_teacher_enabled else None
                    if teacher_lora_adapter_path is not None and (not str(teacher_lora_adapter_path).strip()):
                        teacher_lora_adapter_path = None
                    teacher_tiled_mlp_config = self.config.ref.get("tiled_mlp", None)
                    if teacher_tiled_mlp_config is None:
                        teacher_tiled_mlp_config = self.config.model.get("tiled_mlp", {})
                    teacher_use_tiled_mlp = teacher_tiled_mlp_config.get("enabled", False)
                    teacher_tiled_mlp_shards = teacher_tiled_mlp_config.get("num_shards", 4)
                    keep_teacher_on_gpu = (
                        bool(distillation_cfg.get("keep_external_teacher_on_gpu", False)) or oprd_enabled
                    )
                    if self.rank == 0 and keep_teacher_on_gpu:
                        print("Keeping the frozen teacher on GPU for per-token scoring.")
                    saved_tokenizer = getattr(self, "tokenizer", None)
                    saved_processor = getattr(self, "processor", None)
                    saved_generation_config = getattr(self, "generation_config", None)
                    prev_offload_param = self._is_offload_param
                    prev_offload_optimizer = self._is_offload_optimizer
                    teacher_local_path = copy_to_local(str(teacher_model_path), use_shm=use_shm)
                    teacher_module, _, _, _ = self._build_model_optimizer(
                        model_path=teacher_local_path,
                        fsdp_config=omega_conf_to_dataclass(self.config.ref.fsdp_config),
                        optim_config=None,
                        override_model_config=override_model_config,
                        use_remove_padding=use_remove_padding,
                        use_fused_kernels=use_fused_kernels,
                        enable_gradient_checkpointing=self.config.model.get(
                            "enable_gradient_checkpointing", False
                        ),
                        trust_remote_code=self.config.model.get("trust_remote_code", False),
                        use_liger=self.config.model.get("use_liger", False),
                        role="ref",
                        use_prefix_grouper=use_prefix_grouper,
                        use_tiled_mlp=teacher_use_tiled_mlp,
                        tiled_mlp_shards=teacher_tiled_mlp_shards,
                        lora_adapter_path_override=teacher_lora_adapter_path,
                        lora_adapter_path_override_is_set=True,
                        lora_rank_override=0 if teacher_lora_adapter_path is None else None,
                        force_cpu_offload=False if keep_teacher_on_gpu else None,
                        lora_is_trainable_override=False,
                        wrap_fsdp=not multi_teacher_enabled,
                    )
                    self.tokenizer = saved_tokenizer
                    self.processor = saved_processor
                    self.generation_config = saved_generation_config
                    self._is_offload_param = prev_offload_param
                    self._is_offload_optimizer = prev_offload_optimizer
                    teacher_module.eval()
                    for param in teacher_module.parameters():
                        param.requires_grad_(False)
                    return teacher_module

                def _build_oprd_base():
                    base_model_path = (
                        distillation_cfg.get("w2s_negative_model_path", None)
                        if w2s_opd_enabled
                        else distillation_cfg.get("delta_teacher_base_model_path", None)
                        if delta_target_enabled
                        else distillation_cfg.get("oprd_base_model_path", None)
                    )
                    if base_model_path is None or not str(base_model_path).strip():
                        return None
                    saved_tokenizer = getattr(self, "tokenizer", None)
                    saved_processor = getattr(self, "processor", None)
                    saved_generation_config = getattr(self, "generation_config", None)
                    prev_offload_param = self._is_offload_param
                    prev_offload_optimizer = self._is_offload_optimizer
                    base_local_path = copy_to_local(str(base_model_path), use_shm=use_shm)
                    base_module, _, _, _ = self._build_model_optimizer(
                        model_path=base_local_path,
                        fsdp_config=omega_conf_to_dataclass(self.config.ref.fsdp_config),
                        optim_config=None,
                        override_model_config=override_model_config,
                        use_remove_padding=use_remove_padding,
                        use_fused_kernels=use_fused_kernels,
                        enable_gradient_checkpointing=self.config.model.get(
                            "enable_gradient_checkpointing", False
                        ),
                        trust_remote_code=self.config.model.get("trust_remote_code", False),
                        use_liger=self.config.model.get("use_liger", False),
                        role="ref",
                        use_prefix_grouper=use_prefix_grouper,
                        use_tiled_mlp=self.config.model.get("tiled_mlp", {}).get("enabled", False),
                        tiled_mlp_shards=self.config.model.get("tiled_mlp", {}).get("num_shards", 4),
                        force_cpu_offload=False,
                        lora_rank_override=0,
                        lora_is_trainable_override=False,
                    )
                    self.tokenizer = saved_tokenizer
                    self.processor = saved_processor
                    self.generation_config = saved_generation_config
                    self._is_offload_param = prev_offload_param
                    self._is_offload_optimizer = prev_offload_optimizer
                    base_module.eval()
                    for param in base_module.parameters():
                        param.requires_grad_(False)
                    return base_module

                def _requires_external_teacher_module():
                    return (
                        loss_mode == "kd"
                        or vanilla_on_policy_mix_enabled
                        or (oprd_enabled and oprd_h1_source != "opsd_kl")
                        or (distillation_cfg.get("teacher_regularization") == "teacher")
                    )

                teacher_regularization = distillation_cfg.get("teacher_regularization", "fixed")
                trust_region_ref_module = getattr(self, "ref_module_fsdp", None)
                external_teacher_module = _build_external_teacher()
                self.actor.external_teacher_module = external_teacher_module
                self.actor.on_policy_teacher_module = external_teacher_module
                self.actor.oprd_base_module = _build_oprd_base() if oprd_requires_base else None
                if oprd_requires_base and self.actor.oprd_base_module is None:
                    required_path_name = (
                        "distillation.w2s_negative_model_path"
                        if w2s_opd_enabled
                        else "distillation.delta_teacher_base_model_path"
                        if delta_target_enabled
                        else "distillation.oprd_base_model_path"
                    )
                    raise ValueError(f"Teacher-base scoring requires {required_path_name} to be set.")
                self.actor.w2s_student_base_module = (
                    _build_frozen_self_teacher(distillation_cfg.get("w2s_student_base_model_path", None))
                    if w2s_opd_enabled
                    else None
                )
                if w2s_opd_enabled:
                    positive_source = str(distillation_cfg.get("w2s_positive_model_path"))
                    negative_source = str(distillation_cfg.get("w2s_negative_model_path"))
                    student_base_source = str(distillation_cfg.get("w2s_student_base_model_path"))
                    positive_adapter_source = None
                    positive_local = copy_to_local(positive_source, use_shm=use_shm)
                    negative_local = copy_to_local(negative_source, use_shm=use_shm)
                    student_base_local = copy_to_local(student_base_source, use_shm=use_shm)
                    positive_adapter_local = (
                        copy_to_local(str(positive_adapter_source), use_shm=use_shm)
                        if positive_adapter_source is not None and str(positive_adapter_source).strip()
                        else None
                    )
                    self.actor.w2s_frozen_model_metadata = {
                        "format_version": 1,
                        "student_base": _w2s_frozen_source_metadata(
                            role="student_base",
                            source_path=student_base_source,
                            local_path=student_base_local,
                            revision=distillation_cfg.get("w2s_student_base_model_revision", None),
                            resolved_commit_hash=_w2s_module_commit_hash(self.actor.w2s_student_base_module),
                        ),
                        "positive": _w2s_frozen_source_metadata(
                            role="positive",
                            source_path=positive_source,
                            local_path=positive_local,
                            revision=distillation_cfg.get("w2s_positive_model_revision", None),
                            resolved_commit_hash=_w2s_module_commit_hash(external_teacher_module),
                            adapter_source_path=positive_adapter_source,
                            adapter_local_path=positive_adapter_local,
                        ),
                        "negative": _w2s_frozen_source_metadata(
                            role="negative",
                            source_path=negative_source,
                            local_path=negative_local,
                            revision=distillation_cfg.get("w2s_negative_model_revision", None),
                            resolved_commit_hash=_w2s_module_commit_hash(self.actor.oprd_base_module),
                        ),
                    }
                else:
                    self.actor.w2s_frozen_model_metadata = None
                self.actor.oprd_student_base_module = (
                    trust_region_ref_module if oprd_requires_student_base else None
                )
                if loss_mode in {"kd"}:
                    if external_teacher_module is None:
                        raise ValueError(
                            f"{loss_mode} loss_mode requires distillation.teacher_model_path (external teacher) to be set."
                        )
                    self.actor.teacher_module = external_teacher_module
                elif vanilla_on_policy_mix_enabled:
                    if external_teacher_module is None:
                        raise ValueError(
                            f"{loss_mode} with on-policy distillation requires distillation.teacher_model_path (external teacher) to be set."
                        )
                    self.actor.teacher_module = external_teacher_module
                elif teacher_regularization == "teacher":
                    if external_teacher_module is None:
                        raise ValueError(
                            "teacher regularization requires distillation.teacher_model_path (external teacher)."
                        )
                    self.actor.teacher_module = external_teacher_module
                elif teacher_regularization == "ema":
                    self.actor.teacher_module = _build_frozen_self_teacher()
                else:
                    self.actor.teacher_module = _build_frozen_self_teacher()
        if self._is_actor:
            self.flops_counter = FlopsCounter(self.actor_model_config)
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=self.actor.actor_optimizer,
                lr_scheduler=self.actor_lr_scheduler,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                checkpoint_config=self.config.actor.checkpoint,
            )
        if not self._is_actor and self._is_rollout:
            checkpoint_contents = OmegaConf.create({"load_contents": ["model"], "save_contents": []})
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=None,
                lr_scheduler=None,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                checkpoint_config=checkpoint_contents,
            )

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="red", role="actor_update")
    def update_actor(self, data: DataProto):
        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=get_device_id())
        with self.ulysses_sharding_manager:
            data = data.to("cpu")
            data.meta_info.setdefault("pad_token_id", self.tokenizer.pad_token_id)
            expected_domain = getattr(self, "_multi_teacher_expected_domain", None)
            distillation_cfg = self.config.actor.get("distillation", None)
            strict_multi_teacher_partition = bool(
                distillation_cfg is not None
                and distillation_cfg.get("multi_teacher_strict_domain_partition", True)
            )
            if expected_domain is not None and strict_multi_teacher_partition:
                if "data_source" not in data.non_tensor_batch:
                    raise ValueError("multi-teacher actor update requires non_tensor_batch['data_source']")
                local_domains = {str(value) for value in data.non_tensor_batch["data_source"].tolist()}
                if local_domains != {expected_domain}:
                    raise ValueError(
                        f"multi-teacher rank {self.rank} owns {expected_domain!r}, but received domains={sorted(local_domains)}"
                    )
            with Timer(name="update_policy", logger=None) as timer:
                metrics = self.actor.update_policy(data=data)
            delta_time = timer.last
            global_num_tokens = data.meta_info["global_token_num"]
            images_seqlens = data.meta_info.get("images_seqlens", None)
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(
                global_num_tokens, delta_time, images_seqlens=images_seqlens
            )
            metrics["perf/mfu/actor"] = (
                estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size
            )
            metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / 1073741824
            metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / 1073741824
            metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / 1073741824
            if self.actor_optimizer is not None and len(self.actor_optimizer.param_groups) > 0:
                lr = self.actor_optimizer.param_groups[0]["lr"]
            else:
                lr = self.actor_lr_scheduler.get_last_lr()[0]
            metrics["actor/lr"] = lr.item() if torch.is_tensor(lr) else lr
            self.actor_lr_scheduler.step()
            output = DataProto(meta_info={"metrics": metrics})
            output = output.to("cpu")
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during update_actor", logger=logger)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
            log_gpu_memory_usage("After offload actor optimizer during update_actor", logger=logger)
        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"))
    @DistProfiler.annotate(color="red", role="rollout_generate")
    def generate_sequences(self, prompts: DataProto):
        assert self._is_rollout
        prompts = prompts.to(get_device_id())
        teacher_tokenizer = self._get_teacher_tokenizer_for_generation()
        teacher_pad_token_id = teacher_tokenizer.pad_token_id
        if teacher_pad_token_id is None:
            teacher_pad_token_id = (
                teacher_tokenizer.eos_token_id if teacher_tokenizer.eos_token_id is not None else 0
            )
        meta_info = {"eos_token_id": teacher_tokenizer.eos_token_id, "pad_token_id": teacher_pad_token_id}
        prompts.meta_info.update(meta_info)
        timing_generate = {}
        if self._is_actor:
            loop = get_event_loop()
            loop.run_until_complete(self.rollout_mode())
            log_gpu_memory_usage("After switch to rollout mode", logger=logger)
        with simple_timer("generate_sequences", timing_generate):
            output = self.rollout.generate_sequences(prompts=prompts)
        if self._is_actor:
            loop.run_until_complete(self.trainer_mode())
            log_gpu_memory_usage("After switch to trainer mode", logger=logger)
        timing_generate_topk_ratio, timing_generate_min, timing_generate_max = topk_reduce_ratio_min_max(
            timing_generate["generate_sequences"]
        )
        timing_generate = reduce_timing(timing_generate)
        timing_generate.update(
            {
                "generation_timing/max": timing_generate_max,
                "generation_timing/min": timing_generate_min,
                "generation_timing/topk_ratio": timing_generate_topk_ratio,
            }
        )
        output.meta_info["timing"] = timing_generate
        output = output.to("cpu")
        get_torch_device().empty_cache()
        return output

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

    def _get_data_apply_chat_template_kwargs(self, prompts: Optional[DataProto] = None) -> dict:
        apply_kwargs = None
        if prompts is not None:
            apply_kwargs = prompts.meta_info.get("apply_chat_template_kwargs", None)
        if apply_kwargs is None:
            apply_kwargs = OmegaConf.select(self.config, "data.apply_chat_template_kwargs")
        if apply_kwargs is None:
            return {}
        if isinstance(apply_kwargs, DictConfig):
            apply_kwargs = OmegaConf.to_container(apply_kwargs, resolve=True)
        return dict(apply_kwargs)

    def _get_data_max_prompt_length(self, prompts: DataProto) -> Optional[int]:
        max_prompt_length = prompts.meta_info.get("max_prompt_length", None)
        if max_prompt_length is None:
            max_prompt_length = OmegaConf.select(self.config, "data.max_prompt_length")
        return max_prompt_length

    def _get_teacher_tokenizer_for_generation(self):
        cached = getattr(self, "_teacher_generation_tokenizer", None)
        if cached is not None:
            return cached
        distillation_cfg = OmegaConf.select(self.config, "actor_rollout_ref.actor.distillation")
        teacher_model_path = None
        if distillation_cfg is not None:
            teacher_model_path = distillation_cfg.get("teacher_model_path", None)
        if teacher_model_path is None or not str(teacher_model_path).strip():
            raise ValueError(
                "generate_teacher_sequences requires distillation.teacher_model_path for teacher tokenization."
            )
        trust_remote_code = bool(
            OmegaConf.select(self.config, "actor_rollout_ref.model.trust_remote_code") or False
        )
        tokenizer = hf_tokenizer(str(teacher_model_path), trust_remote_code=trust_remote_code)
        self._teacher_generation_tokenizer = tokenizer
        return tokenizer

    def _rebuild_teacher_prompt_batch_from_raw(self, prompts: DataProto) -> DataProto:
        raw_prompts = prompts.non_tensor_batch.get("raw_prompt")
        if raw_prompts is None:
            raise ValueError(
                "generate_teacher_sequences received DataProto without prompt tensors and without non_tensor_batch['raw_prompt'] for reconstruction."
            )
        max_prompt_length = self._get_data_max_prompt_length(prompts)
        messages = [self._to_message_list(raw_prompt) for raw_prompt in raw_prompts]
        teacher_tokenizer = self._get_teacher_tokenizer_for_generation()
        if use_plain_completion_prompt_format(self.config.data):
            include_system = bool(self.config.data.get("plain_completion_include_system", True))
            prompt_texts = [
                build_plain_completion_prompt(message_list, include_system=include_system)
                for message_list in messages
            ]
            tokenize_kwargs = {
                "return_tensors": "pt",
                "padding": True,
                "truncation": True,
                "add_special_tokens": False,
            }
            if max_prompt_length is not None:
                tokenize_kwargs["max_length"] = int(max_prompt_length)
            prompt_dict = teacher_tokenizer(prompt_texts, **tokenize_kwargs)
        else:
            data_apply_kwargs = self._get_data_apply_chat_template_kwargs(prompts)
            enable_thinking = data_apply_kwargs.get("enable_thinking", True)
            apply_kwargs = {
                "tokenize": True,
                "return_tensors": "pt",
                "return_dict": True,
                "continue_final_message": False,
                "add_generation_prompt": True,
                "enable_thinking": enable_thinking,
                "padding": True,
                "truncation": True,
            }
            if max_prompt_length is not None:
                apply_kwargs["max_length"] = int(max_prompt_length)
            prompt_dict = teacher_tokenizer.apply_chat_template(messages, **apply_kwargs)
        attention_mask = prompt_dict["attention_mask"]
        rebuilt_tensors = {
            "input_ids": prompt_dict["input_ids"],
            "attention_mask": attention_mask,
            "position_ids": compute_position_id_with_mask(attention_mask),
        }
        return DataProto.from_dict(
            tensors=rebuilt_tensors, non_tensors=prompts.non_tensor_batch, meta_info=prompts.meta_info
        )

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="orange", role="teacher_rollout_generate")
    def generate_teacher_sequences(self, prompts: DataProto):
        """Generate one teacher rollout per prompt using actor-side teacher module."""
        assert self._is_actor

        if prompts.non_tensor_batch.get("raw_prompt") is not None:
            prompts = self._rebuild_teacher_prompt_batch_from_raw(prompts)
        elif prompts.batch is None:
            prompts = self._rebuild_teacher_prompt_batch_from_raw(prompts)
        required_prompt_keys = {"input_ids", "attention_mask", "position_ids"}
        missing_prompt_keys = required_prompt_keys - set(prompts.batch.keys())
        if missing_prompt_keys:
            prompts = self._rebuild_teacher_prompt_batch_from_raw(prompts)
            missing_prompt_keys = required_prompt_keys - set(prompts.batch.keys())
            if missing_prompt_keys:
                raise ValueError(
                    f"generate_teacher_sequences missing prompt tensor keys: {sorted(missing_prompt_keys)}"
                )
        prompts = prompts.to(get_device_id())
        teacher_module = self.actor.external_teacher_module or self.actor.teacher_module
        if teacher_module is None:
            raise ValueError(
                "Teacher rollout requires distillation.teacher_model_path to identify the frozen teacher checkpoint."
            )
        if not hasattr(teacher_module, "generate"):
            raise ValueError(
                "Teacher module does not expose .generate(); use an external teacher model or a generate-capable reference model for teacher rollout generation."
            )
        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)
        timing_generate = {}
        with simple_timer("generate_teacher_sequences", timing_generate):
            from verl.workers.rollout.hf_rollout import HFRollout

            teacher_rollout = HFRollout(module=teacher_module, config=self.config.rollout)
            output = teacher_rollout.generate_sequences(prompts=prompts)
        response_mask = output.batch.get("response_mask", None)
        if response_mask is None:
            response_mask = output.batch["attention_mask"][:, -output.batch["responses"].size(1) :]
        response_lengths = response_mask.sum(dim=-1).detach().cpu().numpy().astype(np.float32)
        response_ids = output.batch["responses"].detach().cpu().tolist()
        trimmed_response_ids = [
            ids[: int(length)] for ids, length in zip(response_ids, response_lengths.tolist(), strict=True)
        ]
        output.non_tensor_batch["teacher_response_text"] = np.array(
            self._get_teacher_tokenizer_for_generation().batch_decode(
                trimmed_response_ids, skip_special_tokens=True
            ),
            dtype=object,
        )
        output.non_tensor_batch["teacher_response_length"] = response_lengths
        teacher_module.eval()
        timing_generate_topk_ratio, timing_generate_min, timing_generate_max = topk_reduce_ratio_min_max(
            timing_generate["generate_teacher_sequences"]
        )
        timing_generate = reduce_timing(timing_generate)
        timing_generate.update(
            {
                "teacher_generation_timing/max": timing_generate_max,
                "teacher_generation_timing/min": timing_generate_min,
                "teacher_generation_timing/topk_ratio": timing_generate_topk_ratio,
            }
        )
        output.meta_info["timing"] = timing_generate
        output = output.to("cpu")
        get_torch_device().empty_cache()
        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="orange", role="frozen_validation_generate")
    def generate_frozen_validation_sequences(self, prompts: DataProto):
        """Generate validation sequences with rollout/vLLM after temporarily switching to the frozen LoRA adapter."""
        assert self._is_rollout
        prompts = prompts.to(get_device_id())
        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)
        timing_generate = {}
        switched_mode = False
        loop = get_event_loop()
        with simple_timer("generate_frozen_validation_sequences", timing_generate):
            with self.actor.use_frozen_validation_adapter():
                if self._is_actor and (not loop.is_running()):
                    loop.run_until_complete(self.rollout_mode())
                    switched_mode = True
                log_gpu_memory_usage("After switch to frozen rollout mode", logger=logger)
                output = self.rollout.generate_sequences(prompts=prompts)
                if switched_mode:
                    loop.run_until_complete(self.trainer_mode())
                log_gpu_memory_usage("After switch back from frozen rollout mode", logger=logger)
        timing_generate_topk_ratio, timing_generate_min, timing_generate_max = topk_reduce_ratio_min_max(
            timing_generate["generate_frozen_validation_sequences"]
        )
        timing_generate = reduce_timing(timing_generate)
        timing_generate.update(
            {
                "frozen_validation_generation_timing/max": timing_generate_max,
                "frozen_validation_generation_timing/min": timing_generate_min,
                "frozen_validation_generation_timing/topk_ratio": timing_generate_topk_ratio,
            }
        )
        output.meta_info["timing"] = timing_generate
        output = output.to("cpu")
        get_torch_device().empty_cache()
        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="blue", role="actor_compute_log_prob")
    def compute_log_prob(self, data: DataProto):
        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        from contextlib import nullcontext

        is_lora = data.meta_info.pop("is_lora", False)
        disable_lora_adapter = data.meta_info.pop("disable_lora_adapter", False)
        adapter_ctx = (
            self.actor.actor_module.disable_adapter() if is_lora or disable_lora_adapter else nullcontext()
        )
        config_source = self.config.ref if is_lora else self.config.rollout
        data.meta_info["micro_batch_size"] = config_source.log_prob_micro_batch_size_per_gpu
        data.meta_info["max_token_len"] = config_source.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = config_source.log_prob_use_dynamic_bsz
        data.meta_info["temperature"] = self.config.rollout.temperature
        data.meta_info.setdefault("pad_token_id", self.tokenizer.pad_token_id)
        calculate_entropy = bool(data.meta_info.pop("calculate_entropy", not is_lora))
        with self.ulysses_sharding_manager:
            with adapter_ctx:
                outputs = self.actor.compute_log_prob(data=data, calculate_entropy=calculate_entropy)
            if not is_lora:
                tensors = {"old_log_probs": outputs["log_probs"]}
            else:
                tensors = {"ref_log_prob": outputs["log_probs"]}
            if calculate_entropy:
                tensors["entropys"] = outputs["entropys"]
            if "sum_pi_squared" in outputs:
                tensors["sum_pi_squared"] = outputs["sum_pi_squared"]
            output = DataProto.from_dict(
                tensors=tensors, meta_info={"temperature": self.config.rollout.temperature}
            )
        output = output.to("cpu")
        if self.world_size > 1 and fsdp_version(self.actor.actor_module) == 1:
            self.actor.actor_module._handle.reshard(True)
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during compute_log_prob", logger=logger)
        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="olive", role="ref_compute_log_prob")
    def compute_ref_log_prob(self, data: DataProto):
        if self._is_lora:
            data.meta_info["is_lora"] = True
            return self.compute_log_prob(data)
        assert self._is_ref
        micro_batch_size = self.config.ref.log_prob_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["temperature"] = self.config.rollout.temperature
        data.meta_info["max_token_len"] = self.config.ref.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.ref.log_prob_use_dynamic_bsz
        data.meta_info.setdefault("pad_token_id", self.tokenizer.pad_token_id)
        with self.ulysses_sharding_manager:
            data = data.to("cpu")
            outputs = self.ref_policy.compute_log_prob(data=data, calculate_entropy=False)
            output = DataProto.from_dict(tensors={"ref_log_prob": outputs["log_probs"]})
        output = output.to("cpu")
        if self.world_size > 1:
            if fsdp_version(self.ref_policy.actor_module) == 1:
                self.ref_policy.actor_module._handle.reshard(True)
            elif fsdp_version(self.ref_policy.actor_module) == 2:
                self.ref_policy.actor_module.reshard()
        return output

    def _save_aux_fsdp_module_checkpoint(
        self, module: Optional[FSDP], local_path: str, filename_prefix: str
    ) -> Optional[str]:
        if module is None:
            return None
        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with get_fsdp_state_ctx(module, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, None):
                module_state_dict = module.state_dict()
                target_path = os.path.join(
                    local_path, f"{filename_prefix}_world_size_{self.world_size}_rank_{self.rank}.pt"
                )
                atomic_torch_save(module_state_dict, target_path)
        return target_path

    def _load_aux_fsdp_module_checkpoint(
        self, module: Optional[FSDP], local_path: str, filename_prefix: str
    ) -> bool:
        local_error = None
        module_state_dict = None
        try:
            if module is None or local_path is None:
                raise RuntimeError(
                    "Cannot restore the distillation teacher without a module and checkpoint path"
                )
            remote_path = os.path.join(
                local_path, f"{filename_prefix}_world_size_{self.world_size}_rank_{self.rank}.pt"
            )
            local_aux_path = copy_to_local(remote_path)
            # Stage the shard before any rank enters the FSDP load context.
            # A missing EMA shard must never reset the teacher to its initial model.
            module_state_dict = torch.load(local_aux_path, weights_only=False)
        except Exception as exc:
            local_error = exc
        _check_distillation_resume_errors(local_error, "Distillation teacher checkpoint restore failed")
        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with get_fsdp_state_ctx(module, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, None):
                module.load_state_dict(module_state_dict)
        return True

    def _save_distillation_resume_sidecar(self, local_path: str) -> list[str]:
        if not self._is_actor or local_path is None:
            return []
        distillation_cfg = self.config.actor.get("distillation", None)
        loss_mode = self.config.actor.policy_loss.get("loss_mode", "vanilla")
        direct_opd_adaptive_alpha_enabled = bool(
            distillation_cfg is not None
            and distillation_cfg.get("delta_anchor_adaptive_alpha_enabled", False)
        )
        w2s_opd_enabled = bool(
            distillation_cfg is not None and distillation_cfg.get("w2s_opd_enabled", False)
        )
        if distillation_cfg is None or loss_mode not in {"opsd", "guided_opsd", "kd"}:
            return []
        teacher_regularization = str(distillation_cfg.get("teacher_regularization", "fixed")).lower()
        sidecar_state: dict[str, object] = {"teacher_regularization": teacher_regularization}
        saved_paths: list[str] = []
        if direct_opd_adaptive_alpha_enabled:
            export_state = getattr(self.actor, "export_direct_opd_anchor_alpha_state", None)
            if not callable(export_state):
                raise RuntimeError(
                    "Direct-OPD adaptive alpha is enabled, but the actor cannot export its state."
                )
            sidecar_state["direct_opd_anchor_alpha"] = export_state()
        if w2s_opd_enabled:
            frozen_metadata = getattr(self.actor, "w2s_frozen_model_metadata", None)
            if not isinstance(frozen_metadata, dict):
                raise RuntimeError("W2S-OPD cannot checkpoint without frozen-model source metadata.")
            sidecar_state["w2s_opd"] = {
                "format_version": 1,
                "top_k": int(distillation_cfg.get("w2s_opd_teacher_top_k", 32)),
                "amplification": float(distillation_cfg.get("w2s_amplification", 1.0)),
                "frozen_models": frozen_metadata,
            }
            sidecar_path = os.path.join(
                local_path, f"distill_resume_state_world_size_{self.world_size}_rank_{self.rank}.pt"
            )
            atomic_torch_save(sidecar_state, sidecar_path)
            return [sidecar_path]
        sidecar_path = os.path.join(
            local_path, f"distill_resume_state_world_size_{self.world_size}_rank_{self.rank}.pt"
        )
        atomic_torch_save(sidecar_state, sidecar_path)
        saved_paths.append(sidecar_path)
        if teacher_regularization in {"fixed", "ema"}:
            saved_path = self._save_aux_fsdp_module_checkpoint(
                self.actor.teacher_module, local_path, "distill_teacher"
            )
            if saved_path is not None:
                saved_paths.append(saved_path)
        return saved_paths

    def _load_distillation_resume_sidecar(self, local_path: str) -> None:
        if not self._is_actor or local_path is None:
            return
        distillation_cfg = self.config.actor.get("distillation", None)
        loss_mode = self.config.actor.policy_loss.get("loss_mode", "vanilla")
        direct_opd_adaptive_alpha_enabled = bool(
            distillation_cfg is not None
            and distillation_cfg.get("delta_anchor_adaptive_alpha_enabled", False)
        )
        w2s_opd_enabled = bool(
            distillation_cfg is not None and distillation_cfg.get("w2s_opd_enabled", False)
        )
        if distillation_cfg is None or loss_mode not in {"opsd", "guided_opsd", "kd"}:
            return
        teacher_regularization = str(distillation_cfg.get("teacher_regularization", "fixed")).lower()
        remote_sidecar_path = os.path.join(
            local_path, f"distill_resume_state_world_size_{self.world_size}_rank_{self.rank}.pt"
        )
        sidecar_state: dict[str, object] = {}
        sidecar_error = None
        try:
            local_sidecar_path = copy_to_local(remote_sidecar_path)
            if os.path.exists(local_sidecar_path):
                sidecar_state = torch.load(local_sidecar_path, weights_only=False)
            if not isinstance(sidecar_state, dict):
                raise ValueError(f"Invalid distillation resume state: {remote_sidecar_path}")
            saved_regularization = sidecar_state.get("teacher_regularization")
            if teacher_regularization == "ema" and saved_regularization is None:
                raise FileNotFoundError(
                    f"EMA checkpoint is missing mandatory teacher state metadata: {remote_sidecar_path}"
                )
            if saved_regularization is not None and saved_regularization != teacher_regularization:
                raise ValueError(
                    f"Teacher regularization changed across resume: saved={saved_regularization}, current={teacher_regularization}"
                )
        except Exception as exc:
            sidecar_error = exc
            sidecar_state = {}
        _check_distillation_resume_errors(sidecar_error, "Distillation resume metadata restore failed")
        if direct_opd_adaptive_alpha_enabled:
            alpha_state = sidecar_state.get("direct_opd_anchor_alpha")
            if alpha_state is None:
                raise FileNotFoundError(
                    f"Direct-OPD checkpoint is missing its mandatory adaptive-alpha state: {remote_sidecar_path}."
                )
            load_state = getattr(self.actor, "load_direct_opd_anchor_alpha_state", None)
            if not callable(load_state):
                raise RuntimeError(
                    "Direct-OPD adaptive alpha is enabled, but the actor cannot restore its state."
                )
            load_state(alpha_state)
        if w2s_opd_enabled:
            saved_w2s_state = sidecar_state.get("w2s_opd")
            if not isinstance(saved_w2s_state, dict):
                raise FileNotFoundError(
                    f"W2S-OPD checkpoint is missing mandatory frozen-model metadata: {remote_sidecar_path}. Refusing to derive student_base from the resumed actor."
                )
            current_frozen_metadata = getattr(self.actor, "w2s_frozen_model_metadata", None)
            saved_frozen_metadata = saved_w2s_state.get("frozen_models")
            if saved_frozen_metadata != current_frozen_metadata:
                raise ValueError(
                    f"W2S-OPD frozen model identity changed across resume. The student base, positive, and negative sources must remain exactly fixed. saved={saved_frozen_metadata!r}, current={current_frozen_metadata!r}."
                )
            saved_top_k = int(saved_w2s_state.get("top_k", -1))
            current_top_k = int(distillation_cfg.get("w2s_opd_teacher_top_k", 32))
            saved_amplification = float(saved_w2s_state.get("amplification", 1e309 - 1e309))
            current_amplification = float(distillation_cfg.get("w2s_amplification", 1.0))
            if saved_top_k != current_top_k or not np.isclose(
                saved_amplification, current_amplification, rtol=0.0, atol=0.0
            ):
                raise ValueError(
                    f"W2S-OPD algorithm settings changed across resume: saved_top_k={saved_top_k}, current_top_k={current_top_k}, saved_amplification={saved_amplification}, current_amplification={current_amplification}."
                )
            return
        if teacher_regularization in {"fixed", "ema"}:
            self._load_aux_fsdp_module_checkpoint(self.actor.teacher_module, local_path, "distill_teacher")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(
        self,
        local_path,
        hdfs_path=None,
        global_step=0,
        max_ckpt_to_keep=None,
        exclude_from_max_ckpt_to_keep=False,
    ):
        from verl.utils.logger import log_with_rank

        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        async_save_enabled = getattr(self.checkpoint_manager, "_async_save_enabled", None)
        async_save = bool(async_save_enabled()) if callable(async_save_enabled) else False
        additional_payloads: list[tuple[object, str, str]] = []
        worker_sidecar_paths: list[str] = []
        try:
            manager_save_error: BaseException | None = None
            try:
                self.checkpoint_manager.save_checkpoint(
                    local_path=local_path,
                    hdfs_path=hdfs_path,
                    global_step=global_step,
                    max_ckpt_to_keep=max_ckpt_to_keep,
                    exclude_from_max_ckpt_to_keep=exclude_from_max_ckpt_to_keep,
                    additional_payloads=additional_payloads,
                    wait_for_worker_sidecars=async_save,
                    wait_for_trainer_sidecar=async_save,
                )
            except BaseException as exc:
                manager_save_error = exc
            if async_save:
                self.checkpoint_manager._raise_if_distributed_staging_failed(
                    manager_save_error, context="Async checkpoint manager submission"
                )
            elif manager_save_error is not None:
                raise manager_save_error
            distillation_sidecar_error: BaseException | None = None
            try:
                worker_sidecar_paths.extend(self._save_distillation_resume_sidecar(local_path))
            except BaseException as exc:
                distillation_sidecar_error = exc
            if async_save:
                self.checkpoint_manager._raise_if_distributed_staging_failed(
                    distillation_sidecar_error, context="Async distillation-sidecar staging"
                )
            elif distillation_sidecar_error is not None:
                raise distillation_sidecar_error
            dist.barrier()
            if self._is_lora and hasattr(
                getattr(self, "actor_module", self.actor_module_fsdp), "peft_config"
            ):
                lora_save_path = os.path.join(local_path, "lora_adapter")
                peft_model = getattr(self, "actor_module", self.actor_module_fsdp)
                peft_config = {}
                if dist.get_rank() == 0:
                    os.makedirs(lora_save_path, exist_ok=True)
                    peft_config = asdict(peft_model.peft_config.get("default", {}))
                    peft_config["task_type"] = peft_config["task_type"].value
                    peft_config["peft_type"] = peft_config["peft_type"].value
                    peft_config["target_modules"] = list(peft_config["target_modules"])
                try:
                    if fsdp_version(self.actor_module_fsdp) > 0:
                        self.actor_module_fsdp = self.actor_module_fsdp.to(get_device_name())
                        lora_params = layered_summon_lora_params(self.actor_module_fsdp)
                        if dist.get_rank() == 0:
                            lora_model_path = os.path.join(lora_save_path, "adapter_model.safetensors")
                            lora_config_path = os.path.join(lora_save_path, "adapter_config.json")
                            save_file(lora_params, lora_model_path)
                            with open(lora_config_path, "w", encoding="utf-8") as f:
                                json.dump(peft_config, f, ensure_ascii=False, indent=4)
                            worker_sidecar_paths.extend((lora_model_path, lora_config_path))
                except Exception as e:
                    log_with_rank(
                        f"Save LoRA Adapter Error ({e})",
                        rank=dist.get_rank(),
                        logger=logger,
                        log_only_rank_0=True,
                    )
                dist.barrier()
                log_with_rank(
                    f"[rank-{self.rank}]: Saved LoRA adapter to: {lora_save_path}",
                    rank=dist.get_rank(),
                    logger=logger,
                    log_only_rank_0=True,
                )
            if async_save:
                self.checkpoint_manager.mark_worker_sidecars_complete(
                    local_path, global_step=global_step, required_paths=worker_sidecar_paths
                )
        except BaseException as exc:
            if async_save:
                try:
                    self.checkpoint_manager.mark_worker_sidecars_failed(local_path, exc)
                except Exception:
                    logger.exception("Failed to publish worker checkpoint-sidecar failure marker")
            raise
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def wait_async_checkpoint_save(self):
        if hasattr(self, "checkpoint_manager"):
            self.checkpoint_manager.wait_async_save_distributed()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def raise_if_async_checkpoint_save_failed(self):
        """Poll a pending writer without waiting for successful serialization."""
        if hasattr(self, "checkpoint_manager"):
            return self.checkpoint_manager.raise_if_async_save_failed_distributed()
        return True

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        assert self._is_actor or (not self._is_actor and self._is_rollout), (
            f"Checkpoint loading is only supported for Actor or standalone Rollout Workers, but got {self._is_actor} and {self._is_rollout}"
        )
        if local_path is None:
            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            if self._is_offload_optimizer:
                offload_fsdp_optimizer(self.actor_optimizer)
            return
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        self.checkpoint_manager.load_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load
        )
        None
        self._load_distillation_resume_sidecar(local_path)
        if self._is_actor and self.actor_optimizer is not None and (not self._is_offload_optimizer):
            load_fsdp_optimizer(self.actor_optimizer, device_id=get_device_id())
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(self.actor_optimizer)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def start_profile(self, **kwargs) -> None:
        """Start profiling for the current rank in the current training step."""
        self.profiler.start(**kwargs)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def stop_profile(self) -> None:
        """Stop profiling for the current rank in the current training step."""
        self.profiler.stop()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def dump_memory_snapshot(self, tag: str = "manual", sub_dir: str = None) -> None:
        """Manually trigger a CUDA memory snapshot dump on all ranks."""
        if hasattr(self, "profiler") and hasattr(self.profiler, "_impl"):
            try:
                if hasattr(self.profiler._impl, "sampler"):
                    out_dir = OmegaConf.select(self.config, "actor.profiler.save_path") or "."
                    self.profiler._impl.sampler.dump_memory_snapshot(
                        out_dir=out_dir, tag=tag, sub_dir=sub_dir
                    )
            except Exception:
                pass


class CriticWorker(Worker, DistProfilerExtension):
    def __init__(self, config: FSDPCriticConfig):
        Worker.__init__(self)
        omega_profiler_config = config.get("profiler", {})
        profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)
        if omega_profiler_config.get("tool", None) in ["npu", "nsys", "torch", "torch_memory"]:
            tool_config = omega_conf_to_dataclass(
                omega_profiler_config.get("tool_config", {}).get(omega_profiler_config.get("tool"))
            )
        else:
            tool_config = None
        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config)
        )
        import torch.distributed

        self.config = config
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend=get_nccl_backend(),
                timeout=datetime.timedelta(seconds=self.config.get("nccl_timeout", 600)),
                init_method=_distributed_init_method_from_env(),
            )
        self.config: FSDPCriticConfig = config
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh

        fsdp_size = self.config.model.fsdp_config.fsdp_size
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=fsdp_size)
        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                device_name, mesh_shape=(dp, self.ulysses_sequence_parallel_size), mesh_dim_names=["dp", "sp"]
            )
        if self.ulysses_device_mesh is not None:
            is_collect = self.ulysses_device_mesh["sp"].get_local_rank() == 0
            self._register_dispatch_collect_info(
                "critic", dp_rank=self.ulysses_device_mesh["dp"].get_local_rank(), is_collect=is_collect
            )
        else:
            self._register_dispatch_collect_info("critic", dp_rank=self.rank, is_collect=True)
        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)
        self._is_offload_param = self.config.model.fsdp_config.param_offload
        self._is_offload_optimizer = self.config.model.fsdp_config.optimizer_offload
        self.config.ppo_mini_batch_size *= self.config.rollout_n
        self.config.ppo_mini_batch_size //= (
            torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size
        )
        if self.config.ppo_micro_batch_size is not None:
            self.config.ppo_micro_batch_size //= (
                torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size
            )
            self.config.forward_micro_batch_size //= (
                torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size
            )
            self.config.ppo_micro_batch_size_per_gpu = self.config.ppo_micro_batch_size
            self.config.forward_micro_batch_size_per_gpu = self.config.forward_micro_batch_size
        if self.config.ppo_micro_batch_size_per_gpu is not None:
            assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size_per_gpu == 0, (
                f"normalized ppo_mini_batch_size {self.config.ppo_mini_batch_size} should be divisible by ppo_micro_batch_size_per_gpu {self.config.ppo_micro_batch_size_per_gpu}"
            )
            assert self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu > 0, (
                f"normalized ppo_mini_batch_size {self.config.ppo_mini_batch_size} should be larger than ppo_micro_batch_size_per_gpu {self.config.ppo_micro_batch_size_per_gpu}"
            )
        self._is_lora = (
            self.config.model.get("lora_adapter_path") is not None
            or self.config.model.get("lora_rank", 0) > 0
        )
        self.use_orig_params = self.config.model.fsdp_config.get("use_orig_params", False)

    def _build_critic_model_optimizer(self, config):
        from torch.distributed.fsdp import MixedPrecision
        from verl.utils.model import load_valuehead_model, print_model_size
        from verl.utils.torch_dtypes import PrecisionType

        use_shm = config.model.get("use_shm", False)
        local_path = copy_to_local(config.model.path, use_shm=use_shm)
        tokenizer_path = copy_to_local(config.model.tokenizer_path, use_shm=use_shm)
        self.tokenizer = hf_tokenizer(
            tokenizer_path, trust_remote_code=config.model.get("trust_remote_code", False)
        )
        self.processor = hf_processor(
            tokenizer_path, trust_remote_code=config.model.get("trust_remote_code", False)
        )
        if self.config.model.get("custom_chat_template", None) is not None:
            if self.processor is not None:
                self.processor.chat_template = self.config.model.custom_chat_template
            else:
                self.tokenizer.chat_template = self.config.model.custom_chat_template
        override_config = OmegaConf.to_container(
            OmegaConf.create(self.config.model.get("override_config", {}))
        )
        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_config)
        if self.rank == 0:
            print(f"Critic overriding config {override_config_kwargs}")
        torch_dtype = self.config.model.fsdp_config.get("model_dtype", "fp32")
        torch_dtype = PrecisionType.to_dtype(torch_dtype)
        from transformers import AutoConfig
        from verl.utils.model import sanitize_hf_config_for_model_load

        attn_implementation = override_config.get("attn_implementation", "flash_attention_2")
        if attn_implementation == "eager":
            raise ValueError(
                "attn_implementation=eager is forbidden; use flash_attention_2/flash_attention_3 instead."
            )
        critic_model_config = AutoConfig.from_pretrained(
            local_path,
            attn_implementation=attn_implementation,
            trust_remote_code=config.model.get("trust_remote_code", False),
        )
        sanitize_hf_config_for_model_load(critic_model_config)
        if self.ulysses_sequence_parallel_size > 1 and hasattr(critic_model_config, "vision_config"):
            raise ValueError(
                "Ulysses vision eager fallback is forbidden; disable Ulysses SP or use a text-only model."
            )
        critic_model_config.num_labels = 1
        if getattr(critic_model_config, "model_type", None) == "kimi_vl":
            critic_model_config.text_config.topk_method = "greedy"
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not getattr(critic_model_config, "tie_word_embeddings", False),
            mesh=self.device_mesh,
        )
        tiled_mlp_config = config.model.get("tiled_mlp", {})
        use_tiled_mlp = tiled_mlp_config.get("enabled", False)
        tiled_mlp_shards = tiled_mlp_config.get("num_shards", 4)
        if use_tiled_mlp and config.strategy == "fsdp":
            raise ValueError("TiledMLP requires FSDP2. Set `critic.strategy=fsdp2`.")
        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            critic_model_config.classifier_dropout = 0.0
            critic_model_config.hidden_dropout = "0"
            critic_model_config.summary_dropout_prob = 0.0
            critic_module = load_valuehead_model(
                local_path, torch_dtype, critic_model_config, config.model.get("trust_remote_code", False)
            )
            use_remove_padding = config.model.get("use_remove_padding", False)
            apply_monkey_patch(
                model=critic_module,
                use_remove_padding=use_remove_padding,
                ulysses_sp_size=self.ulysses_sequence_parallel_size,
                use_tiled_mlp=use_tiled_mlp,
                tiled_mlp_shards=tiled_mlp_shards,
            )
            critic_module.to(torch_dtype)
            if config.model.get("enable_gradient_checkpointing", False):
                critic_module.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
        if self.rank == 0:
            print_model_size(critic_module)
        self.critic_model_config = critic_model_config
        fsdp_config = self.config.model.fsdp_config
        mixed_precision_config = fsdp_config.get("mixed_precision", None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get("reduce_dtype", "fp32"))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get("buffer_dtype", "fp32"))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32
        mixed_precision = MixedPrecision(
            param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype
        )
        auto_wrap_policy = get_fsdp_wrap_policy(
            module=critic_module, config=self.config.model.fsdp_config.wrap_policy, is_lora=self._is_lora
        )
        log_gpu_memory_usage("Before critic FSDP", logger=None)
        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)
        self.use_orig_params = fsdp_config.get("use_orig_params", False)
        if self.config.model.get("freeze_vision_tower", False):
            vision_tower = get_vl_model_vision_tower(critic_module)
            if vision_tower is not None:
                vision_tower.requires_grad_(False)
                self.use_orig_params = True
                if self.rank == 0:
                    print("[critic model] Vision tower is set to not trainable.")
            elif self.rank == 0:
                print("[critic model] No vision tower found.")
        if config.strategy == "fsdp":
            reshaped_scalar_params = _reshape_scalar_parameters_for_fsdp(critic_module)
            if reshaped_scalar_params > 0 and self.rank == 0:
                print(f"Reshaped {reshaped_scalar_params} scalar parameters to 1D tensors for critic FSDP.")
            critic_module = FSDP(
                critic_module,
                param_init_fn=init_fn,
                use_orig_params=self.use_orig_params,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=sharding_strategy,
                mixed_precision=mixed_precision,
                sync_module_states=True,
                forward_prefetch=self.config.model.fsdp_config.forward_prefetch,
                device_mesh=self.device_mesh,
                cpu_offload=None,
            )
        elif config.strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, (
                "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            )
            mp_policy = MixedPrecisionPolicy(
                param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=True
            )
            offload_policy = None
            if fsdp_config.offload_policy:
                self._is_offload_param = False
                self._is_offload_optimizer = False
                offload_policy = CPUOffloadPolicy(pin_memory=True)
            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "mp_policy": mp_policy,
                "offload_policy": offload_policy,
                "reshard_after_forward": fsdp_config.reshard_after_forward,
                "shard_placement_fn": get_shard_placement_fn(fsdp_size=self.device_mesh.shape[-1]),
            }
            full_state = critic_module.state_dict()
            apply_fsdp2(critic_module, fsdp_kwargs, fsdp_config)
            fsdp2_load_full_state_dict(critic_module, full_state, fsdp_mesh, offload_policy)
        else:
            raise NotImplementedError(f"Unknown strategy {config.strategy}")
        if config.model.get("enable_activation_offload", False):
            enable_gradient_checkpointing = config.model.get("enable_gradient_checkpointing", False)
            enable_activation_offloading(critic_module, config.strategy, enable_gradient_checkpointing)
        log_gpu_memory_usage("After critic FSDP", logger=None)
        critic_optimizer = build_optimizer(critic_module.parameters(), config.optim)
        total_steps = config.optim.get("total_training_steps", 0)
        num_warmup_steps = int(config.optim.get("lr_warmup_steps", -1))
        lr_scheduler_type = config.optim.get("lr_scheduler_type", "constant")
        if num_warmup_steps < 0:
            num_warmup_steps_ratio = config.optim.get("lr_warmup_steps_ratio", 0.0)
            num_warmup_steps = int(num_warmup_steps_ratio * total_steps)
        if self.rank == 0:
            print(f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}")
        from verl.utils.torch_functional import (
            get_constant_schedule_with_warmup,
            get_cosine_schedule_with_warmup,
        )

        if lr_scheduler_type == "constant":
            critic_lr_scheduler = get_constant_schedule_with_warmup(
                optimizer=critic_optimizer, num_warmup_steps=num_warmup_steps
            )
        elif lr_scheduler_type == "cosine":
            min_lr_ratio = config.optim.get("min_lr_ratio", 0.0)
            num_cycles = config.optim.get("num_cycles", 0.5)
            critic_lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=critic_optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_steps,
                min_lr_ratio=min_lr_ratio,
                num_cycles=num_cycles,
            )
        else:
            raise NotImplementedError(f"LR scheduler type {lr_scheduler_type} is not supported")
        return (critic_module, critic_optimizer, critic_lr_scheduler)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        import_external_libs(self.config.model.get("external_lib", None))
        from verl.workers.critic import DataParallelPPOCritic

        self.critic_module, self.critic_optimizer, self.critic_lr_scheduler = (
            self._build_critic_model_optimizer(self.config)
        )
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)
            log_gpu_memory_usage("After offload critic model during init", logger=logger)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)
            log_gpu_memory_usage("After offload critic optimizer during init", logger=logger)
        self.critic = DataParallelPPOCritic(
            config=self.config, critic_module=self.critic_module, critic_optimizer=self.critic_optimizer
        )
        self.flops_counter = FlopsCounter(self.critic_model_config)
        self.checkpoint_manager = FSDPCheckpointManager(
            model=self.critic_module,
            optimizer=self.critic_optimizer,
            lr_scheduler=self.critic_lr_scheduler,
            processing_class=self.processor if self.processor is not None else self.tokenizer,
            checkpoint_config=self.config.checkpoint,
        )

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="critic"))
    @DistProfiler.annotate(color="cyan", role="compute_values")
    def compute_values(self, data: DataProto):
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)
        micro_batch_size = self.config.forward_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["max_token_len"] = self.config.forward_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.use_dynamic_bsz
        with self.ulysses_sharding_manager:
            data = data.to("cpu")
            values = self.critic.compute_values(data=data)
            output = DataProto.from_dict(tensors={"values": values})
        output = output.to("cpu")
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)
        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="critic"))
    @DistProfiler.annotate(color="pink", role="critic_update")
    def update_critic(self, data: DataProto):
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.critic_optimizer, device_id=get_device_id())
        with self.ulysses_sharding_manager:
            data = data.to("cpu")
            with Timer(name="update_critic", logger=None) as timer:
                metrics = self.critic.update_critic(data=data)
            delta_time = timer.last
            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics["perf/mfu/critic"] = (
                estimated_flops * self.config.ppo_epochs / promised_flops / self.world_size
            )
            if self.critic_optimizer is not None and len(self.critic_optimizer.param_groups) > 0:
                lr = self.critic_optimizer.param_groups[0]["lr"]
            else:
                lr = self.critic_lr_scheduler.get_last_lr()[0]
            metrics["critic/lr"] = lr.item() if torch.is_tensor(lr) else lr
            self.critic_lr_scheduler.step()
            output = DataProto(batch=None, meta_info={"metrics": metrics})
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)
        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(
        self,
        local_path,
        hdfs_path=None,
        global_step=0,
        max_ckpt_to_keep=None,
        exclude_from_max_ckpt_to_keep=False,
    ):
        import torch

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)
        self.checkpoint_manager.save_checkpoint(
            local_path=local_path,
            hdfs_path=hdfs_path,
            global_step=global_step,
            max_ckpt_to_keep=max_ckpt_to_keep,
            exclude_from_max_ckpt_to_keep=exclude_from_max_ckpt_to_keep,
        )
        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=True):
        import torch

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)
        self.checkpoint_manager.load_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load
        )
        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(self.critic_optimizer)


class RewardModelWorker(Worker, DistProfilerExtension):
    """
    Note that we only implement the reward model that is subclass of AutoModelForTokenClassification.
    """

    def __init__(self, config):
        Worker.__init__(self)
        omega_profiler_config = config.get("profiler", {})
        profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)
        if omega_profiler_config.get("tool", None) in ["npu", "nsys", "torch", "torch_memory"]:
            tool_config = omega_conf_to_dataclass(
                omega_profiler_config.get("tool_config", {}).get(omega_profiler_config.get("tool"))
            )
        else:
            tool_config = None
        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config)
        )
        import torch.distributed

        self.config = config
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend=get_nccl_backend(),
                timeout=datetime.timedelta(seconds=self.config.get("nccl_timeout", 600)),
                init_method=_distributed_init_method_from_env(),
            )
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh

        fsdp_size = self.config.model.fsdp_config.fsdp_size
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=fsdp_size)
        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                device_name, mesh_shape=(dp, self.ulysses_sequence_parallel_size), mesh_dim_names=["dp", "sp"]
            )
        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)
        if self.ulysses_device_mesh is not None:
            is_collect = self.ulysses_device_mesh["sp"].get_local_rank() == 0
            self._register_dispatch_collect_info(
                "reward", dp_rank=self.ulysses_device_mesh["dp"].get_local_rank(), is_collect=is_collect
            )
        else:
            self._register_dispatch_collect_info("reward", dp_rank=self.rank, is_collect=True)
        self.use_remove_padding = self.config.model.get("use_remove_padding", False)
        if self.config.micro_batch_size is not None:
            self.config.micro_batch_size //= torch.distributed.get_world_size()
            self.config.micro_batch_size_per_gpu = self.config.micro_batch_size

    def _build_model(self, config):
        from torch.distributed.fsdp import CPUOffload
        from transformers import AutoConfig, AutoModelForTokenClassification
        from verl.utils.model import sanitize_hf_config_for_model_load

        use_shm = config.model.get("use_shm", False)
        local_path = copy_to_local(config.model.path, use_shm=use_shm)
        if self.config.model.input_tokenizer is None:
            self._do_switch_chat_template = False
        else:
            self._do_switch_chat_template = True
            input_tokenizer_local_path = copy_to_local(config.model.input_tokenizer, use_shm=use_shm)
            self.input_tokenizer = hf_tokenizer(
                input_tokenizer_local_path, trust_remote_code=config.model.get("trust_remote_code", False)
            )
            self.tokenizer = hf_tokenizer(
                local_path, trust_remote_code=config.model.get("trust_remote_code", False)
            )
        trust_remote_code = config.model.get("trust_remote_code", False)
        override_config = OmegaConf.to_container(OmegaConf.create(config.model.get("override_config", {})))
        if override_config.get("attn_implementation", "flash_attention_2") == "eager":
            raise ValueError(
                "attn_implementation=eager is forbidden; use flash_attention_2/flash_attention_3 instead."
            )
        model_config = AutoConfig.from_pretrained(
            local_path,
            trust_remote_code=trust_remote_code,
            attn_implementation=override_config.get("attn_implementation", "flash_attention_2"),
        )
        sanitize_hf_config_for_model_load(model_config)
        model_config.num_labels = 1
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not getattr(model_config, "tie_word_embeddings", False), mesh=self.device_mesh
        )
        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_config.classifier_dropout = 0.0
            reward_module = AutoModelForTokenClassification.from_pretrained(
                pretrained_model_name_or_path=local_path,
                config=model_config,
                torch_dtype=torch.bfloat16,
                trust_remote_code=trust_remote_code,
            )
            apply_monkey_patch(
                model=reward_module,
                use_remove_padding=config.model.get("use_remove_padding", False),
                ulysses_sp_size=self.ulysses_sequence_parallel_size,
            )
            reward_module.to(torch.bfloat16)
        auto_wrap_policy = get_fsdp_wrap_policy(module=reward_module, config=self.config.model.fsdp_config)
        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)
        if config.strategy == "fsdp":
            reshaped_scalar_params = _reshape_scalar_parameters_for_fsdp(reward_module)
            if reshaped_scalar_params > 0 and self.rank == 0:
                print(f"Reshaped {reshaped_scalar_params} scalar parameters to 1D tensors for reward FSDP.")
            reward_module = FSDP(
                reward_module,
                param_init_fn=init_fn,
                use_orig_params=False,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=sharding_strategy,
                sync_module_states=True,
                cpu_offload=CPUOffload(offload_params=True),
                forward_prefetch=self.config.model.fsdp_config.forward_prefetch,
                device_mesh=self.device_mesh,
            )
        elif config.strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, (
                "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            )
            cpu_offload = CPUOffloadPolicy(pin_memory=True)
            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "offload_policy": cpu_offload,
                "reshard_after_forward": config.model.fsdp_config.reshard_after_forward,
                "shard_placement_fn": get_shard_placement_fn(fsdp_size=self.device_mesh.shape[-1]),
            }
            full_state = reward_module.state_dict()
            apply_fsdp2(reward_module, fsdp_kwargs, config.model.fsdp_config)
            fsdp2_load_full_state_dict(reward_module, full_state, fsdp_mesh, cpu_offload)
        else:
            raise NotImplementedError(f"Unknown strategy: {config.strategy}")
        return reward_module

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        import_external_libs(self.config.model.get("external_lib", None))
        self.reward_module = self._build_model(config=self.config)

    def _forward_micro_batch(self, micro_batch):
        from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
        from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs

        with torch.no_grad(), torch.autocast(device_type=device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            if position_ids.dim() == 3:
                position_ids = position_ids.transpose(0, 1)
            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)
                if self.ulysses_sequence_parallel_size > 1:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad, position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size
                    )
                output = self.reward_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    use_cache=False,
                )
                reward_rmpad = output.logits
                reward_rmpad = reward_rmpad.squeeze(0)
                if self.ulysses_sequence_parallel_size > 1:
                    reward_rmpad = gather_outputs_and_unpad(
                        reward_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                    )
                rm_score = pad_input(reward_rmpad, indices=indices, batch=batch_size, seqlen=seqlen).squeeze(
                    -1
                )
            else:
                output = self.reward_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                )
                rm_score = output.logits
                rm_score = rm_score.squeeze(-1)
            eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)
            rm_score = rm_score[torch.arange(batch_size), eos_mask_idx]
            return rm_score

    def _expand_to_token_level(self, data: DataProto, scores: torch.Tensor):
        batch_size = data.batch.batch_size[0]
        attention_mask = data.batch["attention_mask"]
        position_ids = data.batch["position_ids"]
        response_length = data.batch["responses"].shape[-1]
        if position_ids.dim() == 3:
            position_ids = position_ids[:, 0, :]
        eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)
        token_level_scores = torch.zeros_like(attention_mask, dtype=scores.dtype)
        token_level_scores[torch.arange(batch_size), eos_mask_idx] = scores
        token_level_scores = token_level_scores[:, -response_length:]
        return token_level_scores

    def _switch_chat_template(self, data: DataProto):
        src_max_length = data.batch["attention_mask"].shape[-1]
        src_tokenizer = self.input_tokenizer
        target_tokenizer = self.tokenizer
        rm_input_ids = []
        rm_attention_mask = []
        for i in range(data.batch.batch_size[0]):
            if not isinstance(data.non_tensor_batch["raw_prompt"][i], list | np.ndarray):
                raise TypeError(
                    f"raw_prompt must be a list or numpy array, got {type(data.non_tensor_batch['raw_prompt'][i])}"
                )
            chat: list = list(data.non_tensor_batch["raw_prompt"][i])
            response_ids = data.batch["responses"][i]
            response_length = response_ids.shape[-1]
            valid_response_length = data.batch["attention_mask"][i][-response_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            response = src_tokenizer.decode(valid_response_ids)
            response = response.replace(src_tokenizer.eos_token, "")
            chat.append({"role": "assistant", "content": response})
            prompt_with_chat_template = target_tokenizer.apply_chat_template(
                chat, add_generation_prompt=False, tokenize=False
            )
            if self.rank == 0 and i == 0:
                print(f"Switch template. chat: {prompt_with_chat_template}")
            max_length = self.config.get("max_length", src_max_length)
            if max_length is None:
                max_length = src_max_length
            model_inputs = target_tokenizer(
                prompt_with_chat_template, return_tensors="pt", add_special_tokens=False
            )
            input_ids, attention_mask = verl_F.postprocess_data(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                max_length=max_length,
                pad_token_id=target_tokenizer.pad_token_id,
                left_pad=False,
                truncation=self.config.get("truncation", "right"),
            )
            rm_input_ids.append(input_ids)
            rm_attention_mask.append(attention_mask)
        rm_input_ids = torch.cat(rm_input_ids, dim=0)
        rm_attention_mask = torch.cat(rm_attention_mask, dim=0)
        rm_position_ids = compute_position_id_with_mask(rm_attention_mask)
        rm_inputs = {
            "input_ids": rm_input_ids,
            "attention_mask": rm_attention_mask,
            "position_ids": rm_position_ids,
        }
        return DataProto.from_dict(rm_inputs)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="reward"))
    @DistProfiler.annotate(color="brown", role="compute_rm_score")
    def compute_rm_score(self, data: DataProto):
        import itertools
        from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches

        data = data.to(get_device_id())
        if self._do_switch_chat_template:
            rm_data = self._switch_chat_template(data)
        else:
            rm_input_ids = data.batch["input_ids"]
            rm_attention_mask = data.batch["attention_mask"]
            rm_position_ids = data.batch["position_ids"]
            rm_inputs = {
                "input_ids": rm_input_ids,
                "attention_mask": rm_attention_mask,
                "position_ids": rm_position_ids,
            }
            rm_data = DataProto.from_dict(rm_inputs)
        rm_data = rm_data.to(get_device_id())
        with self.ulysses_sharding_manager:
            use_dynamic_bsz = self.config.use_dynamic_bsz
            if use_dynamic_bsz:
                max_token_len = (
                    self.config.forward_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                )
                micro_batches, indices = rearrange_micro_batches(
                    batch=rm_data.batch, max_token_len=max_token_len
                )
            else:
                micro_batches = rm_data.batch.split(self.config.micro_batch_size_per_gpu)
            output = []
            for micro_batch in micro_batches:
                rm_score = self._forward_micro_batch(micro_batch)
                output.append(rm_score)
            scores = torch.cat(output, dim=0)
            if use_dynamic_bsz:
                indices = list(itertools.chain.from_iterable(indices))
                assert len(indices) == scores.size(0), f"{len(indices)} vs. {scores.size()}"
                revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
                scores = scores[revert_indices]
            token_level_scores = self._expand_to_token_level(data, scores)
            output = DataProto.from_dict(tensors={"rm_scores": token_level_scores})
        if self.world_size > 1 and fsdp_version(self.reward_module) == 1:
            self.reward_module._handle.reshard(True)
        output = output.to("cpu")
        return output


class AsyncActorRolloutRefWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    async def wake_up(self, sync_weights: bool = True):
        await self.rollout_mode(sync_weights=sync_weights)
        return True

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    async def sleep(self):
        await self.trainer_mode()
        return True

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    def get_zeromq_address(self):
        return self.rollout.get_zeromq_address()

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD, blocking=False)
    async def chat_completion(self, json_request):
        ret = await self.rollout.chat_completion(json_request)
        return ret

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD, blocking=False)
    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
    ) -> list[int]:
        ret = await self.rollout.generate(prompt_ids, sampling_params, request_id, image_data=image_data)
        return ret
