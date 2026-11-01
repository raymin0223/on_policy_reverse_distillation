# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os

from verl.single_controller.ray.base import RayResourcePool
from verl.workers.config import HFModelConfig, RewardModelConfig
from verl.workers.rollout.server_pool import RolloutServerPool

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class RewardModelManager:
    """Reward model manager."""

    def __init__(
        self,
        config: RewardModelConfig,
        resource_pool: RayResourcePool = None,
    ):
        """
        Initialize the reward model manager.

        Args:
            config (RewardModelConfig): Reward model configuration.
            resource_pool (RayResourcePool, optional): Resource pool. Defaults to None.
        """
        self.config = config
        self.resource_pool = resource_pool
        self._initialize_llm_servers()
        self._initialize_router()
        assert self.config.rollout.skip_tokenizer_init is False, (
            "Reward model should not skip tokenizer init."
        )
        if self.config.rollout.free_cache_engine:
            self.sleep()

    def _initialize_llm_servers(self):
        rollout_config = self.config.rollout
        model_config = HFModelConfig(
            path=self.config.model.path,
            external_lib=self.config.model.external_lib,
            trust_remote_code=self.config.model.trust_remote_code,
        )
        self.tokenizer = model_config.get_processor()
        self.server_pool = RolloutServerPool(
            rollout_config=rollout_config,
            model_config=model_config,
            n_gpus_per_node=self.config.n_gpus_per_node,
            resource_pool=self.resource_pool,
            nnodes=self.config.nnodes,
            is_reward_model=True,
        )
        self.rollout_replicas = self.server_pool.rollout_replicas
        self.server_handles = self.server_pool.server_handles
        self.server_addresses = self.server_pool.server_addresses

    def _initialize_router(self):
        worker_urls = [f"http://{server_address}" for server_address in self.server_addresses]

        # TODO (dyy): sglang router is not ready yet.
        # if self.config.rollout.name == "sglang":
        #     from .router.inner_sglang_router import launch_router_process
        # else:
        #     from .router.naive_router import launch_router_process

        from .router.naive_router import launch_router_process

        self.router_address, _ = launch_router_process(worker_urls=worker_urls)

    def get_router_address(self):
        return self.router_address

    def wake_up(self):
        """Wake up all rollout replica instances."""
        self.server_pool.wake_up()

    def sleep(self):
        """Sleep all rollout replica instances."""
        self.server_pool.sleep()
