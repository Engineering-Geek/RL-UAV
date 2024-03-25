from environments.core import HoverDrone, MultiAgentDroneEnvironment
import numpy as np
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO
from stable_baselines3.common.env_checker import check_env
from ray.rllib.env.env_context import EnvContext
from scipy.spatial.transform import Rotation

from environments.core.sensors import Lidar, Accelerometer, Gyroscope


def parametric_target_position(t: float) -> np.ndarray:
    # period = 20
    # x = 1 * np.sin(2 * np.pi * t / period)
    # y = 1 * np.cos(2 * np.pi * t / period)
    # z = 1
    x = 0
    y = 0
    z = 1
    return np.array([x, y, z])


class SimpleDrone(HoverDrone):
    def __init__(self, **kwargs):
        super(SimpleDrone, self).__init__(**kwargs)
        self.max_distance_to_target_reward = 1000.
        self.orientation_reward_weight = 50.
        self.hit_penalty = 50.
        
    @property
    def target_position(self) -> np.ndarray:
        return parametric_target_position(self.data.time)
    
    @property
    def reward(self) -> float:
        # Compute the distance to the target only if target_position is specified.
        if self.target_position is not None:
            distance = np.linalg.norm(self.data.xpos[self.body_id] - self.target_position)
            # Clip the distance to be within [0, self.max_distance_to_target_reward]
            clipped_distance = np.clip(distance, 0, self.max_distance_to_target_reward)
            distance_to_target_reward = clipped_distance * self.model.opt.timestep
        else:
            distance_to_target_reward = self.max_distance_to_target_reward * self.model.opt.timestep
        
        # Calculate the orientation reward
        # Convert the quaternion to a rotation matrix and extract the z column vector
        z_axis = -Rotation.from_quat(self.data.xquat[self.body_id]).as_matrix()[:, 2]
        orientation_reward = np.dot(z_axis, [0, 0, 1]) * self.orientation_reward_weight * self.model.opt.timestep
        
        # Apply hit penalty
        hit_penalty = self.hit_penalty if self.hit_floor or self.out_of_bounds else 0
        
        # Sum up the components to get the total reward
        reward = distance_to_target_reward + orientation_reward - hit_penalty
        return reward
    
    @property
    def done(self) -> bool:
        return False
    
    @property
    def truncated(self) -> bool:
        return False
    
    @property
    def log_info(self) -> dict:
        return {}
    
    def custom_update(self):
        if self.hit_floor or self.out_of_bounds:
            self.reset()
    
    def reset_custom_flags(self):
        pass


class SimpleHoverDroneEnv(MultiAgentDroneEnvironment):
    def __init__(self, config: EnvContext):
        lidar_sensor = Lidar(
            n_thetas=config.get("n_thetas", 8),
            n_phis=config.get("n_phis", 8),
            ray_distance=config.get("ray_distance", 5)
        )
        accelerometer = Accelerometer()
        gyroscope = Gyroscope()
        
        map_bounds = np.array([[-5, -5, -0.01], [5, 5, 5]])
        spawn_box = np.array([[-1, -1, 0], [1, 1, 3]])
        
        super(SimpleHoverDroneEnv, self).__init__(
            n_agents=config.get("n_agents", 1),
            spacing=1.0,
            DroneClass=SimpleDrone,
            sensors={lidar_sensor, accelerometer, gyroscope},
            map_bounds=config.get("map_bounds", map_bounds),
            spawn_box=config.get("spawn_box", spawn_box),
            dt=config.get("dt", 0.01),
            render_mode=config.get("render_mode", None),
        )
        self.reset(seed=config.worker_index * config.num_workers)
    
