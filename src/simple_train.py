import os
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from gym_drone.environments.battle_royal import LidarBattleRoyal
import numpy as np

# Initialize Ray
ray.init(ignore_reinit_error=True)

# Define the environment
env_config = {
    "render_mode": None,
    "world_bounds": np.array([[-10, -10, -0.1], [10, 10, 5]]),
    "respawn_box": np.array([[-5, -5, 0.5], [5, 5, 5]]),
    "num_agents": 5,
    "n_phi": 16,
    "n_theta": 16,
    "ray_max_distance": 10,
    "kwargs": {
        "noise": 0.005,
    }
}

# Configure and initialize the PPO algorithm
config = PPOConfig() \
    .environment(env=LidarBattleRoyal, env_config=env_config) \
    .framework("torch") \
    .rollouts(num_rollout_workers=1, num_envs_per_worker=1) \
    .training(gamma=0.99, lr=5e-5, train_batch_size=5000,
              model={"fcnet_hiddens": [256, 256], "fcnet_activation": "relu"}) \
    .resources(num_gpus=1 if ray.cluster_resources().get("GPU", 0) > 0 else 0) \
    .evaluation(evaluation_interval=10, evaluation_duration=10)

ppo_algo = config.build()

# Training loop
for i in range(100):
    print(f"Training iteration: {i}")
    result = ppo_algo.train()
    print(f"Episode reward mean: {result['episode_reward_mean']}")
    
    if i % 10 == 0:
        checkpoint = ppo_algo.save()
        print(f"Checkpoint saved at {checkpoint}")

ray.shutdown()
