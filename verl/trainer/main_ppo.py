"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other mpain.
"""

import os
import socket
import hydra
import ray
from omegaconf import OmegaConf
from verl.experimental.dataset.sampler import AbstractSampler
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import apply_global_seed, validate_config
from verl.utils.device import auto_set_device, is_cuda_available
from verl.utils.import_utils import load_extern_object


def _patch_ray_ports_from_env() -> None:
    """Allow launch scripts to avoid Ray local-cluster port collisions."""
    env_to_kwarg = {
        "RAY_DASHBOARD_AGENT_LISTEN_PORT": "dashboard_agent_listen_port",
        "RAY_METRICS_EXPORT_PORT": "metrics_export_port",
        "RAY_GCS_SERVER_PORT": "gcs_server_port",
        "RAY_RUNTIME_ENV_AGENT_PORT": "runtime_env_agent_port",
        "RAY_NODE_MANAGER_PORT": "node_manager_port",
        "RAY_OBJECT_MANAGER_PORT": "object_manager_port",
    }
    port_values = {}
    for env_name, kwarg_name in env_to_kwarg.items():
        port = os.environ.get(env_name)
        if not port or port.strip().lower() in {"auto", "none", "null"}:
            continue
        try:
            port_values[kwarg_name] = int(port)
        except ValueError as exc:
            raise ValueError(f"{env_name} must be an integer, got {port!r}") from exc
    if not port_values:
        return
    import ray._private.parameter as ray_parameter

    original_init = ray_parameter.RayParams.__init__
    if getattr(original_init, "_rd_ray_ports_patched", False):
        return

    def patched_init(self, *args, **kwargs):
        for kwarg_name, port_int in port_values.items():
            kwargs.setdefault(kwarg_name, port_int)
        return original_init(self, *args, **kwargs)

    patched_init._rd_ray_ports_patched = True
    ray_parameter.RayParams.__init__ = patched_init


def _should_use_on_policy_distillation(config) -> bool:
    distillation_cfg = config.actor_rollout_ref.actor.get("distillation", None)
    if distillation_cfg is None:
        return False
    loss_mode = config.actor_rollout_ref.actor.policy_loss.get("loss_mode", "vanilla")
    if loss_mode == "kd":
        return True
    return loss_mode == "vanilla" and bool(distillation_cfg.get("kdrl_enabled", False))


def _should_configure_teacher_tokenizer_compatibility(config) -> bool:
    distillation_cfg = config.actor_rollout_ref.actor.get("distillation", None)
    if distillation_cfg is None:
        return False
    teacher_model_path = distillation_cfg.get("teacher_model_path", None)
    if teacher_model_path is None or not str(teacher_model_path).strip():
        return False
    policy = (
        str(distillation_cfg.get("teacher_tokenizer_mismatch_policy", "auto_mask_special")).strip().lower()
    )
    if policy in {"", "none", "null", "off", "disabled"}:
        return False
    loss_mode = config.actor_rollout_ref.actor.policy_loss.get("loss_mode", "vanilla")
    distill_loss_modes = {"opsd", "kd"}
    return (
        loss_mode in distill_loss_modes
        or (loss_mode == "vanilla" and (bool(distillation_cfg.get("kdrl_enabled", False)) or False))
        or False
    )


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    """Main entry point for PPO training with Hydra configuration management.

    Args:
        config_dict: Hydra configuration dictionary containing training parameters.
    """
    auto_set_device(config)
    run_ppo(config)


def run_ppo(config, task_runner_class=None) -> None:
    from oprd.runtime.full_finetuning import validate
    from oprd.runtime.contract import validate as validate_runtime

    validate(config)
    validate_runtime(config)
    """Initialize Ray cluster and run distributed PPO training process.

    Args:
        config: Training configuration object containing all necessary parameters
                for distributed PPO training including Ray initialization settings,
                model paths, and training hyperparameters.
        task_runner_class: For recipe to change TaskRunner.
    """
    from oprd.runtime.tokenizers import _validate_w2s_tokenizer_identity

    _validate_w2s_tokenizer_identity(config)
    if _should_configure_teacher_tokenizer_compatibility(config):
        from oprd.runtime.tokenizers import _configure_teacher_tokenizer_compatibility

        _configure_teacher_tokenizer_compatibility(config)
    apply_global_seed(config)
    ray_initialized_here = False
    if not ray.is_initialized():
        _patch_ray_ports_from_env()
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        if config.transfer_queue.enable:
            runtime_env_vars = runtime_env_kwargs.get("env_vars", {})
            runtime_env_vars["TRANSFER_QUEUE_ENABLE"] = "1"
            runtime_env_kwargs["env_vars"] = runtime_env_vars
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        ray_init_kwargs = OmegaConf.to_container(ray_init_kwargs, resolve=True)
        runtime_env_vars = ray_init_kwargs.get("runtime_env", {}).get("env_vars", None)
        if runtime_env_vars is not None:
            ray_init_kwargs["runtime_env"]["env_vars"] = {
                str(key): str(value) for key, value in runtime_env_vars.items()
            }
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**ray_init_kwargs)
        ray_initialized_here = True
    if task_runner_class is None:
        task_runner_class = ray.remote(num_cpus=1)(TaskRunner)
    try:
        if (
            is_cuda_available
            and config.global_profiler.tool == "nsys"
            and (config.global_profiler.get("steps") is not None)
            and (len(config.global_profiler.get("steps", [])) > 0)
        ):
            from verl.utils.import_utils import is_nvtx_available

            assert is_nvtx_available(), "nvtx is not available in CUDA platform. Please 'pip3 install nvtx'"
            nsight_options = OmegaConf.to_container(
                config.global_profiler.global_tool_config.nsys.controller_nsight_options
            )
            runner = task_runner_class.options(runtime_env={"nsight": nsight_options}).remote()
        else:
            runner = task_runner_class.remote()
        ray.get(runner.run.remote(config))
        timeline_json_file = config.ray_kwargs.get("timeline_json_file", None)
        if timeline_json_file:
            ray.timeline(filename=timeline_json_file)
    finally:
        if ray_initialized_here and ray.is_initialized():
            ray.shutdown()


class TaskRunner:
    """Ray remote class for executing distributed PPO training tasks.

    This class encapsulates the main training logic and runs as a Ray remote actor
    to enable distributed execution across multiple nodes and GPUs.

    Attributes:
        role_worker_mapping: Dictionary mapping Role enums to Ray remote worker classes
        mapping: Dictionary mapping Role enums to resource pool IDs for GPU allocation
    """

    def __init__(self):
        self.role_worker_mapping = {}
        self.mapping = {}

    def get_trainer_class(self, config):
        s2l_cfg = config.algorithm.get("s2l_po", None)
        if s2l_cfg is not None and bool(s2l_cfg.get("enabled", False)):
            from verl.trainer.ppo.s2l_po_trainer import S2LPORayPPOTrainer

            return S2LPORayPPOTrainer
        return RayPPOTrainer

    def add_actor_rollout_worker(self, config):
        """Add actor rollout worker based on the actor strategy."""
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import Role

        use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
        distillation_cfg = config.actor_rollout_ref.actor.get("distillation", None)
        loss_mode = config.actor_rollout_ref.actor.policy_loss.get("loss_mode", "vanilla")
        distill_loss_modes = {"opsd", "guided_opsd", "kd"}
        teacher_forward_distillation_enabled = distillation_cfg is not None and (
            loss_mode in distill_loss_modes
            or (loss_mode == "vanilla" and (bool(distillation_cfg.get("kdrl_enabled", False)) or False))
            or False
        )
        teacher_regularization = (
            distillation_cfg.get("teacher_regularization", "fixed") if distillation_cfg is not None else None
        )
        oprd_h1_source = (
            str(distillation_cfg.get("oprd_h1_source", "teacher_base")).strip().lower().replace("-", "_")
            if distillation_cfg is not None
            else "teacher_base"
        )
        oprd_needs_student_base_ref = (
            distillation_cfg is not None
            and bool(distillation_cfg.get("oprd_enabled", False))
            and (oprd_h1_source == "teacher_base_residual")
        )
        distillation_needs_ref = (
            distillation_cfg is not None
            and loss_mode in distill_loss_modes
            and (teacher_regularization in {"trust-region"})
        )
        if distillation_needs_ref and need_reference_policy(config):
            raise ValueError("Distillation loss cannot share the reference policy with KL regularization.")
        if teacher_forward_distillation_enabled and use_legacy_worker_impl == "disable":
            raise ValueError(
                "Teacher-based distillation losses require the legacy worker implementation. Set trainer.use_legacy_worker_impl=auto/enable."
            )
        if distillation_needs_ref and use_legacy_worker_impl == "disable":
            raise ValueError(
                "Distillation loss requires the legacy worker implementation to colocate the teacher."
            )
        if use_legacy_worker_impl == "disable":
            from verl.workers.engine_workers import ActorRolloutRefWorker

            actor_rollout_cls = ActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup
            lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
            if lora_rank <= 0:
                lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
            ref_in_actor = (
                lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
            )
            if (need_reference_policy(config) or oprd_needs_student_base_ref) and (not ref_in_actor):
                role = Role.ActorRolloutRef
            else:
                role = Role.ActorRollout
            self.role_worker_mapping[role] = ray.remote(actor_rollout_cls)
            self.mapping[role] = "global_pool"
            return (actor_rollout_cls, ray_worker_group_cls)
        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker

            actor_rollout_cls = AsyncActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup
        elif config.actor_rollout_ref.actor.strategy == "megatron":
            from verl.workers.megatron_workers import AsyncActorRolloutRefWorker

            actor_rollout_cls = AsyncActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup
        else:
            raise NotImplementedError
        actor_role = (
            Role.ActorRolloutRef
            if distillation_needs_ref or oprd_needs_student_base_ref
            else Role.ActorRollout
        )
        self.role_worker_mapping[actor_role] = ray.remote(actor_rollout_cls)
        self.mapping[actor_role] = "global_pool"
        return (actor_rollout_cls, ray_worker_group_cls)

    def add_critic_worker(self, config):
        """Add critic worker to role mapping."""
        use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
        if config.critic.strategy in {"fsdp", "fsdp2"}:
            if use_legacy_worker_impl in ["auto", "enable"]:
                from verl.workers.fsdp_workers import CriticWorker
            elif use_legacy_worker_impl == "disable":
                from verl.workers.engine_workers import TrainingWorker

                CriticWorker = TrainingWorker
                print("Using new worker implementation")
            else:
                raise ValueError(f"Invalid use_legacy_worker_impl: {use_legacy_worker_impl}")
        elif config.critic.strategy == "megatron":
            from verl.workers.megatron_workers import CriticWorker
        else:
            raise NotImplementedError
        from verl.trainer.ppo.ray_trainer import Role

        self.role_worker_mapping[Role.Critic] = ray.remote(CriticWorker)
        self.mapping[Role.Critic] = "global_pool"

    def init_resource_pool_mgr(self, config):
        """Initialize resource pool manager."""
        global_pool_id = "global_pool"
        resource_pool_spec = {global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes}
        if config.reward_model.enable_resource_pool:
            if config.reward_model.n_gpus_per_node <= 0:
                raise ValueError("config.reward_model.n_gpus_per_node must be greater than 0")
            if config.reward_model.nnodes <= 0:
                raise ValueError("config.reward_model.nnodes must be greater than 0")
            reward_pool = [config.reward_model.n_gpus_per_node] * config.reward_model.nnodes
            resource_pool_spec["reward_pool"] = reward_pool
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager

        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec, mapping=self.mapping
        )
        return resource_pool_manager

    def add_reward_model_worker(self, config):
        """Add reward model worker if enabled."""
        from verl.trainer.ppo.ray_trainer import Role

        if config.reward_model.enable:
            use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
            if use_legacy_worker_impl in ["auto", "enable", "disable"]:
                if config.reward_model.strategy in {"fsdp", "fsdp2"}:
                    from verl.workers.fsdp_workers import RewardModelWorker
                elif config.reward_model.strategy == "megatron":
                    from verl.workers.megatron_workers import RewardModelWorker
                else:
                    raise NotImplementedError
            else:
                raise ValueError(f"Invalid use_legacy_worker_impl: {use_legacy_worker_impl}")
            self.role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            if config.reward_model.enable_resource_pool:
                self.mapping[Role.RewardModel] = "reward_pool"
            else:
                self.mapping[Role.RewardModel] = "global_pool"

    def add_ref_policy_worker(self, config, ref_policy_cls):
        """Add reference policy worker if KL loss or KL reward is used."""
        from verl.trainer.ppo.ray_trainer import Role

        use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
        if use_legacy_worker_impl == "disable":
            return
        if need_reference_policy(config):
            if Role.ActorRolloutRef in self.role_worker_mapping:
                return
            self.role_worker_mapping[Role.RefPolicy] = ray.remote(ref_policy_cls)
            self.mapping[Role.RefPolicy] = "global_pool"

    def run(self, config):
        """Execute the main PPO training workflow.

        This method sets up the distributed training environment, initializes
        workers, datasets, and reward functions, then starts the training process.

        Args:
            config: Training configuration object containing all parameters needed
                   for setting up and running the PPO training process.
        """
        from pprint import pprint
        from omegaconf import OmegaConf
        from verl.utils.fs import copy_to_local

        apply_global_seed(config)
        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)
        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)
        self.add_reward_model_worker(config)
        self.add_ref_policy_worker(config, actor_rollout_cls)
        validate_config(
            config=config, use_reference_policy=need_reference_policy(config), use_critic=need_critic(config)
        )
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)
        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        val_reward_num_examine = int(config.trainer.get("val_reward_num_examine", 1))
        val_reward_fn = load_reward_manager(
            config,
            tokenizer,
            num_examine=val_reward_num_examine,
            is_validation=True,
            **config.reward_model.get("reward_kwargs", {}),
        )
        resource_pool_manager = self.init_resource_pool_mgr(config)
        from verl.utils.dataset.rl_dataset import collate_fn

        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)
        trainer = self.get_trainer_class(config)(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        trainer.init_workers()
        trainer.fit()


def create_rl_dataset(data_paths, data_config, tokenizer, processor, is_train=True, max_samples: int = -1):
    """Create a dataset.

    Arguments:
        data_paths: List of paths to data files.
        data_config: The data config.
        tokenizer (Tokenizer): The tokenizer.
        processor (Processor): The processor.

    Returns:
        dataset (Dataset): The dataset.
    """
    from verl.utils.dataset.rl_dataset import get_dataset_class

    dataset_cls = get_dataset_class(data_config)
    dataset_config = data_config
    val_random_sample = bool(data_config.get("val_random_sample", False))
    if not is_train and max_samples > 0 and data_config.get("shuffle", False) and (not val_random_sample):
        dataset_config = data_config.copy()
        dataset_config["shuffle"] = False
    dataset = dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=dataset_config,
        max_samples=max_samples,
    )
    return dataset


def create_rl_sampler(data_config, dataset):
    """Create a sampler for the dataset.

    Arguments:
        data_config: The data config.
        dataset (Dataset): The dataset.

    Returns:
        sampler (Sampler): The sampler.
    """
    import torch
    from torch.utils.data import SequentialSampler
    from torchdata.stateful_dataloader.sampler import RandomSampler

    if data_config.sampler is not None and data_config.sampler.get("class_path", None) is not None:
        curriculum_class = load_extern_object(data_config.sampler.class_path, data_config.sampler.class_name)
        sampler = curriculum_class(data_source=dataset, data_config=data_config)
        assert isinstance(sampler, AbstractSampler)
        assert data_config.get("dataloader_num_workers", 8) == 0, (
            "If using curriculum, num_workers must be 0 to prevent data caching. If the dataloader caches data before the batch is done the curriculum sampler won't have the opportunity to reorder it. "
        )
    elif data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        seed = data_config.get("seed")
        if seed is not None:
            train_dataloader_generator.manual_seed(seed)
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        sampler = SequentialSampler(data_source=dataset)
    return sampler


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C). Shutting down Ray...")
        if ray.is_initialized():
            ray.shutdown()
