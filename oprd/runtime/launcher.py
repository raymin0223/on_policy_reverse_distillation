"""Hydra entry point for the paper's retained training methods."""

import hydra
import ray
from verl.trainer.main_ppo import TaskRunner, run_ppo
from verl.utils.device import auto_set_device


class PaperTaskRunner(TaskRunner):
    def run(self, config):
        if config.reward_model.reward_manager == "vilasr_rule":
            from oprd.agent import vilasr_reward, token_mean, kdrl_actor  # noqa: F401 -- registries
        return super().run(config)

    def get_trainer_class(self, config):
        if config.algorithm.get("s2l_po", {}).get("enabled", False):
            from verl.trainer.ppo.s2l_po_trainer import S2LPORayPPOTrainer

            return S2LPORayPPOTrainer
        mode = config.actor_rollout_ref.actor.policy_loss.loss_mode
        if mode in {"opsd", "guided_opsd"}:
            from verl.trainer.ppo.guided_opsd_trainer import GuidedOPSDRayPPOTrainer

            return GuidedOPSDRayPPOTrainer
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer

        return RayPPOTrainer


@hydra.main(config_path="../../verl/trainer/config", config_name="ppo_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(PaperTaskRunner))


if __name__ == "__main__":
    main()
