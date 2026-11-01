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

from verl.experimental.reward_loop.reward_manager.naive import NaiveRewardManager
from verl.experimental.reward_loop.reward_manager.registry import register


@register("dapo")
class DAPORewardManager(NaiveRewardManager):
    """Use the shared, validated soft-buffer implementation for async rewards.

    Both manager names accept ``reward_kwargs.overlong_buffer_cfg`` and apply
    soft length penalties only during training. Rewards are scalar, and
    optional penalty-statistics logging defaults to off.
    """
