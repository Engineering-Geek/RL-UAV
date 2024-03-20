from typing import List, Tuple

import numpy as np
from mujoco._structs import _MjModelActuatorViews
from numpy import cos
from ray.rllib.utils.typing import EnvObsType, EnvActionType
from gymnasium.spaces import Box, MultiBinary, Tuple as TupleSpace

from tmp.Environments import MultiAgentDroneEnvironment, BaseDrone
from src.utils.simulation_metrics import time_ratio


class SimpleShooter(BaseDrone):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # ---------------------------------------------------------
        
        self.dtype = np.float32
        
        # region Reward Flags
        
        # Flags for sparse rewards and penalties and actions
        self._hit_floor = False
        self._got_hit = False
        self._hit_target = False
        self._out_of_bounds = False
        self._bullet_out_of_bounds = False
        self.just_shot = False
        
        # Rewards and Penalties for Continuous actions
        self._aimed_at_angles: List[float] = []
        self._aiming_at_angles: List[float] = []
        # endregion
        
        # ---------------------------------------------------------
        
        # region Reward Variables and Functions
        
        # Reward and Penalty parameters
        self.staying_alive_reward = kwargs.get("staying_alive_reward", 1.0)  # Reward for staying alive per second
        self.hit_reward = kwargs.get("hit_reward", 10.0)  # Reward for hitting the target
        self.hit_penalty = kwargs.get("hit_penalty", 10.0)  # Penalty for getting hit
        self.floor_penalty = kwargs.get("floor_penalty", 10.0)  # Penalty for hitting the floor
        self.out_of_bounds_penalty = kwargs.get("out_of_bounds_penalty", 10.0)  # Penalty for going out of bounds
        self.bullet_out_of_bounds_penalty = (
            kwargs.get("bullet_out_of_bounds_penalty", 10.0))  # Penalty for bullet going out of bounds
        self.aim_reward = kwargs.get("aim_reward", 10.0)  # Reward for aiming at the target
        self.aim_penalty = kwargs.get("aim_penalty", 10.0)  # Penalty for being aimed at
        self._shooting_penalty = kwargs.get("shooting_penalty", 1.0)  # Penalty for shooting
        self.instability_penalty = kwargs.get("instability_penalty", .1)  # Penalty for instability
        # endregion
        
        self.actuator_model: _MjModelActuatorViews = self.model.actuator(self._actuators[0].name)
        self.max_actuation = self.actuator_model.ctrlrange[1]
    
    def got_hit(self):
        self._got_hit = True
        self.reset()  # Respawning the drone after getting hit
    
    def hit_floor(self):
        self._hit_floor = True
        self.reset()  # Respawning the drone after hitting the floor
    
    def on_bullet_out_of_bounds(self):
        self._bullet_out_of_bounds = True
    
    def on_out_of_bounds(self):
        self._out_of_bounds = True
        self.reset()  # Respawning the drone after going out of bounds
    
    def scored_hit(self):
        self._hit_target = True
    
    def aiming_at(self, drones: List[Tuple[BaseDrone, float]]):
        self._aiming_at_angles = [angle for drone, angle in drones]
    
    def aimed_at(self, drones: List[Tuple[BaseDrone, float]]):
        self._aimed_at_angles = [angle for drone, angle in drones]
    
    def env_update(self):
        pass
    
    def reset_params(self):
        self._hit_floor = False
        self._got_hit = False
        self._hit_target = False
        self._out_of_bounds = False
        self._bullet_out_of_bounds = False
        self.just_shot = False
        
        # No need to reset the aiming and aimed at angles because they are updated every step
        # self._aimed_at_angles = []
        # self._aiming_at_angles = []
    
    @property
    def reward(self):
        reward = 0
        if self._hit_floor:
            reward -= self.floor_penalty
        if self._got_hit:
            reward -= self.hit_penalty
        if self._hit_target:
            reward += self.hit_reward
        if self._out_of_bounds:
            reward -= self.out_of_bounds_penalty
        if self._bullet_out_of_bounds:
            reward -= self.bullet_out_of_bounds_penalty
        if self.alive:
            reward += self.staying_alive_reward * self.model.opt.timestep
        if self.just_shot:
            reward -= self._shooting_penalty
        
        # Formula for Aiming Reward
        for angle in self._aiming_at_angles:
            reward += self.aim_reward * cos(angle) * self.model.opt.timestep
        
        # Formula for Aimed At Penalty
        for angle in self._aimed_at_angles:
            reward -= self.aim_penalty * cos(angle) * self.model.opt.timestep
            
        # Formula for Instability Penalty
        reward -= self.instability_penalty * np.linalg.norm(self.acceleration) * self.model.opt.timestep
        
        return reward
    
    @property
    def done(self):
        return self._hit_floor or self._got_hit
    
    @property
    def truncated(self) -> bool:
        return self._out_of_bounds
    
    @property
    def info(self):
        return {
            "hit_floor": self._hit_floor,
            "got_hit": self._got_hit,
            "hit_target": self._hit_target,
            "out_of_bounds": self._out_of_bounds,
            "bullet_out_of_bounds": self._bullet_out_of_bounds,
        }
    
    @property
    def observation_space(self) -> EnvObsType:
        """
        Observation:
            - 0 images
            - vector and distance to nearest other drone (add noise, normalize distance between 0 and 1 using boundaries)
            - current gyro and accelerometer readings (add noise), normalize between 0 and 1
            - current position (x, y, z) (between 0 and 1, use boundaries)
        All observations are continuous and in a 1D array
        """
        # Note, map bounds is a 2x3 array, row 1 is the lower bounds and row 2 is the upper bounds
        min_bounds = self.map_bounds[0]
        max_bounds = self.map_bounds[1]
        max_bounds = max_bounds - min_bounds
        agent_ids = self.drone_positions.keys()
        sample_unit_vector = np.array([self.unit_vector_to_other_drones[agent_id]
                                       for agent_id in agent_ids if agent_id != self.agent_id]).flatten()
        sample_distance = np.array([self.distance_to_other_drones[agent_id]
                                    for agent_id in agent_ids if agent_id != self.agent_id]
                                   ).flatten() / np.linalg.norm(max_bounds)
        sample_gyro = self.frame_quaternion / 2 + 0.5
        sample_position = self.body.xpos / max_bounds
        onehot_size = sample_unit_vector.size + 1 + sample_gyro.size + sample_position.size + sample_distance.size
        return Box(low=-1, high=1, shape=(onehot_size,), dtype=self.dtype)
    
    @property
    def action_space(self) -> EnvActionType:
        """
        Action:
            - 0, 1, 2, 3: motor power (continuous)
            - 4: shoot (binary)
        """
        return TupleSpace([
            Box(low=0, high=1, shape=(4,), dtype=self.dtype),
            MultiBinary(1)
        ])
    
    def observation(self, render: bool = False) -> EnvObsType:
        # Note, map bounds is a 2x3 array, row 1 is the lower bounds and row 2 is the upper bounds
        min_bounds = self.map_bounds[0]
        max_bounds = self.map_bounds[1]
        max_bounds = max_bounds - min_bounds
        agent_ids = self.drone_positions.keys()
        sample_unit_vector = np.array([self.unit_vector_to_other_drones[agent_id]
                                       for agent_id in agent_ids if agent_id != self.agent_id]).flatten()
        sample_distance = np.array([self.distance_to_other_drones[agent_id]
                                    for agent_id in agent_ids if agent_id != self.agent_id]
                                   ).flatten() / np.linalg.norm(max_bounds)
        sample_gyro = self.frame_quaternion / 2 + 0.5
        sample_position = self.body.xpos / max_bounds
        return np.concatenate((sample_unit_vector, sample_distance, sample_gyro, sample_position))
    
    def act(self, action: EnvActionType):
        motor_powers, shoot = action
        actuators = self._actuators
        for actuator, motor_power in zip(actuators, motor_powers):
            actuator.ctrl = motor_power
        
        if shoot:
            self.just_shot = True
            self.bullet.shoot()


class SimpleShootingEnvironment(MultiAgentDroneEnvironment):
    def __init__(self, n_agents: int, **kwargs):
        bottom_left = np.array([-50, -50, 0])
        top_right = np.array([50, 50, 50])
        super().__init__(
            n_images=0,
            n_agents=n_agents,
            depth_render=False,
            Drone=SimpleShooter,
            map_bounds=np.array([bottom_left, top_right]),
            kwargs=kwargs
        )


if __name__ == "__main__":
    env = SimpleShootingEnvironment(n_agents=5, dt=0.01)
    ratio = time_ratio(env, 100)
    print(f"Time Ratio: {ratio}")
    
    
