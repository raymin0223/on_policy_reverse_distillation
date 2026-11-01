"""GRPO uses the same root actor and normalization as all paper methods."""

from verl.workers.actor.dp_actor import DataParallelPPOActor
from . import token_mean  # noqa: F401 -- register losses in each Ray worker process


class AgentGRPOActor(DataParallelPPOActor):
    pass
