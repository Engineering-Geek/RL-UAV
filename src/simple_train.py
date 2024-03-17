import os

import ray

from src.register import register
from ray.rllib.algorithms.ppo import PPO, PPOTorchPolicy
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.examples.env.multi_agent import MultiAgentCartPole
import random


config = PPOConfig()
config = config.training(gamma=0.9, lr=0.01, kl_coeff=0.3, train_batch_size=128)
config = config.resources(num_gpus=0)
config = config.rollouts(num_rollout_workers=1)
config = config.environment(MultiAgentCartPole,
                            env_config={"n_agents": 2, "n_images": 1, "frame_skip": 1, "depth_render": True})
config = config.multi_agent(
    policies={
        "policy_0": (PPOTorchPolicy, MultiAgentCartPole.observation_space, MultiAgentCartPole.action_space, {}),
        "policy_1": (PPOTorchPolicy, MultiAgentCartPole.observation_space, MultiAgentCartPole.action_space, {})
    },
    policy_mapping_fn=lambda agent_id: f"policy_{random.randint(0, 1)}"
)
config = config.resources(num_gpus=1)


stop = {
    "timesteps_total": 10000,
}
results = tune.Tuner(
    "PPO",
    param_space=config.to_dict(),
    run_config=air.RunConfig(stop=stop, verbose=1),
).fit()

ray.shutdown()

