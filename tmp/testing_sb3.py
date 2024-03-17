from ray.rllib.algorithms.ppo import PPO, PPOConfig, PPOTorchPolicy
from src.Environments.multi_agent import MultiAgentDroneEnvironment, SimpleShooter

env = MultiAgentDroneEnvironment(
    n_agents=2,
    n_images=1,
    frame_skip=1,
    depth_render=True,
    Drone=SimpleShooter,
)


