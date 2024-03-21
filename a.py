import os
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env

# Assuming your custom environment is defined in custom_env.py
from environments.SimpleDroneEnv import SimpleDroneEnv


def env_creator(env_config):
    return SimpleDroneEnv(env_config.get("num_agents", 3))


# Register your environment
register_env("simple_multi_agent_env", env_creator)

# Initialize Ray
if not ray.is_initialized():
    ray.init()

# Define the configuration for PPO
config = {
    "env": "simple_multi_agent_env",
    "num_workers": 1,
    "env_config": {"num_agents": 3},
    "framework": "torch",  # or "tf" for TensorFlow
    # Add other PPO-specific and RLlib-specific configurations as needed
}

# Run the training
tune.run(
    PPO,
    config=config,
    stop={"training_iteration": 1000},  # Stopping condition
    checkpoint_at_end=True,
    verbose=1
)

# Shutdown Ray
ray.shutdown()
