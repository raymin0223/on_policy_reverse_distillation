import asyncio

from verl.single_controller.ray.base import split_resource_pool
from verl.workers.rollout.replica import get_rollout_replica_class


class RolloutServerPool:
    """Shared rollout replica lifecycle for vLLM/SGLang-style async servers."""

    def __init__(
        self,
        *,
        rollout_config,
        model_config,
        n_gpus_per_node: int,
        resource_pool=None,
        nnodes: int | None = None,
        is_reward_model: bool = False,
    ):
        self.rollout_config = rollout_config
        self.model_config = model_config
        self.n_gpus_per_node = n_gpus_per_node
        self.resource_pool = resource_pool
        self.nnodes = nnodes
        self.is_reward_model = is_reward_model
        self.rollout_replica_class = get_rollout_replica_class(self.rollout_config.name)
        self._initialize_llm_servers()

    def _initialize_llm_servers(self):
        rollout_world_size = (
            self.rollout_config.tensor_model_parallel_size
            * self.rollout_config.data_parallel_size
            * self.rollout_config.pipeline_model_parallel_size
        )
        world_size = (
            self.resource_pool.world_size
            if self.resource_pool is not None
            else self.n_gpus_per_node * self.nnodes
        )
        num_replicas = world_size // rollout_world_size
        self.rollout_replicas = [
            self.rollout_replica_class(
                replica_rank=replica_rank,
                config=self.rollout_config,
                model_config=self.model_config,
                gpus_per_node=self.n_gpus_per_node,
                is_reward_model=self.is_reward_model,
            )
            for replica_rank in range(num_replicas)
        ]
        if self.resource_pool is not None:
            split_resource_pools = split_resource_pool(self.resource_pool, split_size=rollout_world_size)
            assert len(split_resource_pools) == len(self.rollout_replicas)
            self._run_all(
                [
                    replica.init_colocated(resource_pool)
                    for replica, resource_pool in zip(
                        self.rollout_replicas, split_resource_pools, strict=True
                    )
                ]
            )
        else:
            self._run_all([replica.init_standalone() for replica in self.rollout_replicas])

        self.server_handles = [replica.server_handle for replica in self.rollout_replicas]
        self.server_addresses = [replica.server_address for replica in self.rollout_replicas]

    def wake_up(self):
        self._run_all([replica.wake_up() for replica in self.rollout_replicas])

    def sleep(self):
        self._run_all([replica.sleep() for replica in self.rollout_replicas])

    def clear_kv_cache(self):
        self._run_all([replica.clear_kv_cache() for replica in self.rollout_replicas])

    @staticmethod
    def _run_all(tasks: list):
        async def run_all():
            await asyncio.gather(*tasks)

        asyncio.run(run_all())
