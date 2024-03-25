from typing import Optional, Tuple
import gymnasium as gym
import numpy as np
import ray
from ray import tune
from ray.rllib.env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig

# Assuming the SimpleHoverDroneEnv is correctly imported
from environments.SimpleHoverDroneEnv import SimpleHoverDroneEnv


# Ensure that the environment is registered
def env_creator(env_config):
    return SimpleHoverDroneEnv(env_config)


register_env("simple_multi_agent_env", env_creator)

# Initialize Ray
if not ray.is_initialized():
    ray.init()

# Define the PPO configuration
config = PPOConfig() \
    .environment(env="simple_multi_agent_env", env_config={"n_agents": 1}) \
    .framework("torch") \
    .rollouts(num_rollout_workers=15) \
    .training(num_sgd_iter=10, lr=0.0003) \
    .to_dict()

# Start the training
results = tune.run(
    "PPO",
    config=config,
    stop={"training_iteration": 1000},
    checkpoint_at_end=True,
    verbose=1
)

# Output the results
for trial in results.trials:
    print(trial.last_result)

# Shutdown Ray at the end of the training
ray.shutdown()
