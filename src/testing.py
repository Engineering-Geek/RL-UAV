import numpy as np
from ray.rllib.utils import check_env

from gym_drone.environments.battle_royal import LidarBattleRoyal

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

env = LidarBattleRoyal(env_config)
# for _ in range(5):
#     env.step(env.action_space.sample())
# obs, _, _, _, _ = env.step(env.action_space.sample())
# for k, ob in obs.items():
#     ob_summary = f"{k}: "
#     for o in ob:
#         ob_summary += f"(shape: {o.shape}, "
#         ob_summary += f"type: {o.dtype}, "
#         ob_summary += f"min: {o.min()}, "
#         ob_summary += f"max: {o.max()}), "
#     print(ob_summary)
# print(env.observation_space)

check_env(env)
env.close()
