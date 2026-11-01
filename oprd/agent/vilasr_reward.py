"""Shared terminal verifier for training and held-out validation."""

from collections import defaultdict
import torch
from verl.workers.reward_manager import register
from .contracts import validate_training_groups
from .vilasr_contract import score_response


@register("vilasr_rule")
class ViLaSRRewardManager:
    def __init__(self, tokenizer, num_examine=0, **kwargs):
        self.tokenizer = tokenizer

    def __call__(self, data, return_dict=False):
        if "rm_scores" in data.batch:
            raise ValueError("Precomputed/learned reward is not allowed")
        if not data.meta_info.get("validate", False):
            validate_training_groups([x.non_tensor_batch.get("uid") for x in data])
        tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        extra = defaultdict(list)
        for i, item in enumerate(data):
            meta = item.non_tensor_batch
            result = score_response(
                meta.get("model_segments"),
                meta.get("traj_stop_reason"),
                meta["reward_model"]["ground_truth"],
                meta.get("tool_interact_info"),
            )
            indices = item.batch["response_mask"].bool().nonzero(as_tuple=True)[0]
            if len(indices):
                tensor[i, indices[-1]] = result["score"]
            elif result["score"]:
                raise ValueError("Positive reward without model tokens")
            for key, value in result.items():
                extra[key].append(value)
        result = {"reward_tensor": tensor, "reward_extra_info": dict(extra)}
        return result if return_dict else tensor
