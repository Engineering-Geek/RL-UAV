from environments.core import ClosestDronesSensor, HoverDrone, MultiAgentDroneEnvironment
import numpy as np
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO
from stable_baselines3.common.env_checker import check_env

from environments.core import Lidar, Accelerometer, Gyroscope


class SimpleDrone(HoverDrone):
    def __init__(self, **kwargs):
        super(SimpleDrone, self).__init__(**kwargs)
    
    @property
    def reward(self) -> float:
        reward = 0
        reward += self.model.opt.timestep   # staying alive reward
        reward -= 50 * self.out_of_bounds   # out of bounds penalty
        reward -= 100 * self.hit_floor      # hit floor penalty
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
        pass


class SimpleDroneEnv(MultiAgentDroneEnvironment):
    def __init__(self, num_agents: int = 3):
        lidar_sensor = Lidar(
            n_thetas=8,
            n_phis=8,
            ray_distance=10
        )
        accelerometer = Accelerometer()
        gyroscope = Gyroscope()
        
        map_bounds = np.array([[-5, -5, -0.01], [5, 5, 5]])
        spawn_box = np.array([[-5, -5, 0], [5, 5, 5]])
        
        super(SimpleDroneEnv, self).__init__(
            n_agents=num_agents,
            spacing=1.0,
            DroneClass=SimpleDrone,
            sensors={lidar_sensor, accelerometer, gyroscope},
            map_bounds=map_bounds,
            spawn_box=spawn_box,
            dt=0.01,
            render_mode=None
        )
    
