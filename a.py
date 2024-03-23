from typing import Dict, Union, Optional

import numpy as np
from ray.rllib import BaseEnv, Policy, RolloutWorker
from ray.rllib.algorithms import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.typing import PolicyID, AgentID

from environments.SimpleDroneEnvs import SimpleHoverDroneEnv
from ray.tune.registry import register_env
import ray

import logging

logging.basicConfig(level=logging.INFO)


class BatchIntegrityChecker(DefaultCallbacks):
    def on_postprocess_traj(
            self,
            *,
            worker,
            base_env,
            policies: Dict[str, Policy],
            episode: Episode,
            agent_id: Optional[str] = None,
            policy_id: Optional[str] = None,
            postprocessed_batch,
            original_batches,
            **kwargs,
    ) -> None:
        # Extract episode IDs from the batch
        episode_ids = postprocessed_batch['eps_id']
        unique_episode_ids = np.unique(episode_ids)

        if len(unique_episode_ids) > 1:
            # Log detailed information about the problematic batch
            logging.error(f"Batch contains steps from multiple trajectories! Unique episode IDs: {unique_episode_ids}")
            logging.error(f"Batch size: {postprocessed_batch.count}")
            logging.error(f"Actions in the batch: {postprocessed_batch['actions']}")
            logging.error(f"Rewards in the batch: {postprocessed_batch['rewards']}")
            logging.error(f"Dones in the batch: {postprocessed_batch['dones']}")
            logging.error(f"Episode IDs in the batch: {episode_ids}")

            # Raise an error to halt training and investigate
            raise ValueError("Batch contains steps from multiple trajectories!")
        else:
            logging.info(f"Batch verification passed: All steps belong to episode ID {unique_episode_ids[0]}.")


def env_creator(env_config):
    return SimpleHoverDroneEnv(
        n_agents=env_config.get("n_agents", 5),
        spacing=env_config.get("spacing", 3.0),
        map_bounds=env_config.get("map_bounds", np.array([[-10, -10, -0.01], [10, 10, 10]])),
        spawn_box=env_config.get("spawn_box", np.array([[-1, -1, 0.1], [5, 1, 1]])),
        dt=env_config.get("dt", 0.01),
        render_mode=env_config.get("render_mode", None),
        fps=env_config.get("fps", 60),
        sim_rate=env_config.get("sim_rate", 1),
        update_distances=env_config.get("update_distances", True)
    )


register_env("SimpleHoverDroneEnv", env_creator)

# Initialize Ray
ray.init(ignore_reinit_error=True)

# Configure the PPO algorithm
config = PPOConfig() \
    .environment(env="SimpleHoverDroneEnv") \
    .framework("torch") \
    .rollouts(num_rollout_workers=0) \
    .training(gamma=0.99, lr=5e-4, train_batch_size=4000) \
    .resources(num_gpus=0) \
    .evaluation(evaluation_interval=10, evaluation_num_episodes=10) \
    .reporting(min_time_s_per_iteration=1)\

# Build the PPO algorithm
ppo_agent = config.build()

# Train the PPO agent
for i in range(100):  # Number of training iterations
    result = ppo_agent.train()
    print(f"Iteration: {i}, episode_reward_mean: {result['episode_reward_mean']}")

    if i % 10 == 0:  # Save the model every 10 iterations
        checkpoint = ppo_agent.save()
        print(f"Checkpoint saved at: {checkpoint}")

