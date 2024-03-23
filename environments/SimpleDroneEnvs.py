import numpy as np
from scipy.spatial.transform import Rotation as R

from environments.core import HoverDrone, MultiAgentDroneEnvironment, ShootingDrone
from environments.core import Lidar, Accelerometer, Gyroscope


class SimpleHoverDrone(HoverDrone):
    
    def __init__(self, **kwargs):
        super(SimpleHoverDrone, self).__init__(**kwargs)
        self.initial_position = self.data.xpos[self.body_id].copy()
        self.global_z = np.array([0, 0, 1])
        self.orientation_weight = 2  # How close to the global z-axis the drone should be
        self.position_weight = 0.1  # How close to the initial position the drone should be
    
    @property
    def orientation_reward(self) -> float:
        return np.dot(
            R.from_quat(self.data.xquat[self.body_id]).apply([0, 0, 1]),
            self.global_z
        ) * self.orientation_weight
    
    @property
    def position_reward(self) -> float:
        return np.linalg.norm(self.data.xpos[self.body_id] - self.initial_position) * self.position_weight
    
    @property
    def reward(self) -> float:
        reward = 0
        reward += self.model.opt.timestep  # staying alive reward
        reward -= 50 * self.out_of_bounds  # out of bounds penalty
        reward -= 100 * self.hit_floor  # hit floor penalty
        reward += self.orientation_reward * self.model.opt.timestep  # orientation reward
        reward += self.position_reward * self.model.opt.timestep
        return reward
    
    @property
    def done(self) -> bool:
        if (self.hit_floor or self.got_shot or self.crash_into_drone or
                self.makes_simulation_unstable or self.out_of_bounds):
            return True
        return False
    
    @property
    def truncated(self) -> bool:
        if self.out_of_bounds:
            return True
        return False
    
    @property
    def log_info(self) -> dict:
        return {}
    
    def custom_update(self):
        if self.hit_floor or self.got_shot or self.out_of_bounds:
            self.reset()
    

class SimpleShootingDrone(ShootingDrone):
    def __init__(self, **kwargs):
        super(SimpleShootingDrone, self).__init__(**kwargs)
        self.initial_position = self.data.xpos[self.body_id].copy()
        self.global_z = np.array([0, 0, 1])
        self.orientation_weight = 2     # How close to the global z-axis the drone should be
        self.position_weight = 0.1      # How close to the initial position the drone should be
    
    @property
    def orientation_reward(self) -> float:
        return np.dot(
            R.from_quat(self.data.xquat[self.body_id]).apply([0, 0, 1]),
            self.global_z
        ) * self.orientation_weight
    
    @property
    def position_reward(self) -> float:
        return np.linalg.norm(self.data.xpos[self.body_id] - self.initial_position) * self.position_weight
    
    @property
    def reward(self) -> float:
        reward = 0
        reward += self.model.opt.timestep  # staying alive reward
        reward -= 50 * self.out_of_bounds  # out of bounds penalty
        reward -= 100 * self.hit_floor  # hit floor penalty
        reward -= 5 * self.got_shot  # got shot penalty, low value to encourage shooting and avoid confusion
        reward += 100 * self.shot_drone  # shot drone reward, high value to encourage shooting and avoid confusion
        reward -= 0.5 * self.just_shot  # just shot penalty, low value to encourage shooting
        reward += self.orientation_reward * self.model.opt.timestep  # orientation reward
        reward += self.position_reward * self.model.opt.timestep
        return reward
    
    @property
    def done(self) -> bool:
        if (self.hit_floor or self.got_shot or self.crash_into_drone or
                self.makes_simulation_unstable or self.out_of_bounds):
            return True
        return False
    
    @property
    def truncated(self) -> bool:
        if self.out_of_bounds:
            return True
        return False
    
    @property
    def log_info(self) -> dict:
        return {}
    
    def custom_update(self):
        if self.hit_floor or self.got_shot or self.out_of_bounds:
            self.reset()


class SimpleHoverDroneEnv(MultiAgentDroneEnvironment):
    def __init__(self, **kwargs):
        lidar_sensor = Lidar(
            n_thetas=kwargs.get("n_thetas", 8),  # number of rays in horizontal direction (theta)
            n_phis=kwargs.get("n_phis", 8),  # number of rays in vertical direction (phi)
            ray_distance=kwargs.get("ray_distance", 10.0)  # distance of the rays
        )
        accelerometer = Accelerometer()
        gyroscope = Gyroscope()
        
        map_bounds = kwargs.get("map_bounds", np.array([[-10, -10, -0.01], [10, 10, 10]]))
        spawn_box = kwargs.get("spawn_box", np.array([[-1, -1, 0.1], [5, 1, 1]]))
        
        super(SimpleHoverDroneEnv, self).__init__(
            n_agents=kwargs.get("n_agents", 5),
            spacing=kwargs.get("spacing", 3.0),
            DroneClass=SimpleHoverDrone,
            sensor=lidar_sensor,
            map_bounds=map_bounds,
            spawn_box=spawn_box,
            dt=kwargs.get("dt", 0.01),
            render_mode=kwargs.get("render_mode", None),
            fps=kwargs.get("fps", 60),
            sim_rate=kwargs.get("sim_rate", 1),
            update_distances=kwargs.get("update_distances", True)
        )
    

class SimpleShootingDroneEnv(MultiAgentDroneEnvironment):
    def __init__(self, **kwargs):
        lidar_sensor = Lidar(
            n_thetas=kwargs.get("n_thetas", 8),  # number of rays in horizontal direction (theta)
            n_phis=kwargs.get("n_phis", 8),  # number of rays in vertical direction (phi)
            ray_distance=kwargs.get("ray_distance", 10.0)  # distance of the rays
        )
        accelerometer = Accelerometer()
        gyroscope = Gyroscope()
        
        map_bounds = kwargs.get("map_bounds", np.array([[-10, -10, -0.01], [10, 10, 10]]))
        spawn_box = kwargs.get("spawn_box", np.array([[-1, -1, 0.1], [5, 1, 1]]))
        
        super(SimpleShootingDroneEnv, self).__init__(
            n_agents=kwargs.get("n_agents", 5),
            spacing=kwargs.get("spacing", 3.0),
            DroneClass=SimpleShootingDrone,
            sensor=lidar_sensor,
            map_bounds=map_bounds,
            spawn_box=spawn_box,
            dt=kwargs.get("dt", 0.01),
            render_mode=kwargs.get("render_mode", None),
            fps=kwargs.get("fps", 60),
            sim_rate=kwargs.get("sim_rate", 1),
            update_distances=kwargs.get("update_distances", True)
        )

