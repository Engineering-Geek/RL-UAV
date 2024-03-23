from abc import abstractmethod, ABC
from copy import deepcopy
from typing import List, Dict, Tuple

import numpy as np
import quaternion
from gymnasium.spaces import Box, Tuple as TupleSpace, Dict as DictSpace, Discrete
from mujoco import MjModel, MjData
from ray.rllib.utils.typing import AgentID, EnvActionType

from environments.core.sensors import Sensor


class Drone(ABC):
    """
    An abstract base class representing a drone in a simulation environment. It includes basic properties,
    methods for sensor updates, shooting mechanisms, and resetting states, as well as abstract methods
    that must be implemented by subclasses.

    Attributes:
        model (MjModel): The MuJoCo model associated with the drone.
        data (MjData): The MuJoCo data structure associated with the drone.
        agent_id (AgentID): The unique identifier for the drone.
        spawn_box (np.ndarray): The spawning area for the drone.
        gammas (np.ndarray): The angles between the local +x-axis and the vectors pointing to other drones.
        distances (np.ndarray): The distances to other drones.
        sensors (List[Sensor]): The list of sensors attached to the drone.
        bullet_speed (float): The speed of the bullets fired by the drone.

    Methods:
        update_sensors: Updates all sensors attached to the drone.
        reset_bullet: Resets the bullet's state if the drone has a shooting mechanism.
        reset: Resets the drone's state to initial conditions.
        shoot: Activates the drone's shooting mechanism if available.
        reset_default_flags: Resets the default status flags of the drone.
        observation: Returns the current observation of the drone.
        observation_space: Returns the observation space of the drone.
        act: Abstract method to apply an action to the drone.
        action_space: Abstract property that defines the action space of the drone.
        reward: Abstract property that defines the reward for the drone's current state.
        done: Abstract property that checks if the drone's episode is done.
        truncated: Abstract property that checks if the drone's episode is truncated.
        log_info: Abstract property that returns logging information for the drone.
        custom_update: Abstract method for custom updates within the simulation step.
    """
    def __init__(self, model: MjModel, data: MjData, agent_id: AgentID, spawn_box: np.ndarray, gammas: np.ndarray,
                 distances: np.ndarray, sensor: Sensor, bullet_speed: float = 10.0, max_qacc=1000.0):
        self.model = model
        self.data = data
        self.agent_id = agent_id
        self.spawn_box = spawn_box
        self.gammas = gammas  # angle between the local +x-axis and the vector pointing to other drones
        self.distances = distances  # distance to other drones
        self.sensor = sensor
        self.max_qacc = max_qacc
        
        self._initial_xpos = deepcopy(self.data.xpos[self.agent_id])
        x_0, y_0, z_0 = self._initial_xpos
        x_min, y_min, z_min = self.spawn_box[0] + np.array([x_0, y_0, 0])
        x_max, y_max, z_max = self.spawn_box[1] + np.array([x_0, y_0, 0])
        self.spawn_box = np.array([[x_min, y_min, z_min], [x_max, y_max, z_max]])
        self.spawn_angles = np.array([[0, 0, 0], [2 * np.pi, 2 * np.pi, 2 * np.pi]])
        
        self.body_id: int = self.model.body(f"drone_{self.agent_id}").id
        self.free_joint_id: int = self.model.joint(f"drone_{self.agent_id}_free_joint").id
        self.free_qp_adr: int = self.model.jnt_qposadr[self.free_joint_id]
        self.free_qv_adr: int = self.model.jnt_dofadr[self.free_joint_id]
        self.collision_geom_1_id: int = self.model.geom(f"drone_{self.agent_id}_collision_1").id
        self.collision_geom_2_id: int = self.model.geom(f"drone_{self.agent_id}_collision_2").id
        self.bullet_body_id: int = self.model.body(f"bullet_{self.agent_id}").id
        self.bullet_free_joint_id: int = self.model.joint(f"bullet_{self.agent_id}").id
        self.bullet_geom_id: int = self.model.geom(f"bullet_{self.agent_id}").id
        self.bullet_spawn_point: int = self.model.site(f"bullet_spawn_position_{self.agent_id}").id
        motor_names = [f"front_left_{self.agent_id}", f"back_right_{self.agent_id}", f"front_right_{self.agent_id}",
                       f"back_left_{self.agent_id}"]
        self.actuator_ids: List[int] = [self.model.actuator(name).id for name in motor_names]
        
        self.bullet_rgba = np.array([1, 0, 0, 1], dtype=np.float64)
        self.bullet_speed = bullet_speed
        
        self.just_shot = False
        self.shot_while_no_bullet = False
        self.shot_drone = False
        self.hit_floor = False
        self.got_shot = False
        self.out_of_bounds = False
        self.bullet_out_of_bounds = False
        self.bullet_aloft = False
        self.crash_into_drone = False
        
    @property
    def makes_simulation_unstable(self):
        qacc = np.linalg.norm(self.data.qacc[self.free_qv_adr: self.free_qv_adr + 6])
        return qacc > self.max_qacc
    
    def update_sensor(self):
        self.sensor.update()
    
    def reset_bullet(self):
        self.model.body_contype[self.bullet_body_id] = 0
        self.model.body_conaffinity[self.bullet_body_id] = 0
        self.data.qpos[self.bullet_free_joint_id: self.bullet_free_joint_id + 7].fill(0)
        self.data.qvel[self.bullet_free_joint_id: self.bullet_free_joint_id + 6].fill(0)
        self.bullet_aloft = False
    
    def reset(self):
        pos = np.random.uniform(*self.spawn_box) + self._initial_xpos[:3]
        quat = quaternion.from_euler_angles(np.random.uniform(*self.spawn_angles))
        self.data.qpos[self.free_qp_adr: self.free_qp_adr + 7] = np.concatenate([pos, quaternion.as_float_array(quat)])
        self.data.qvel[self.free_qv_adr: self.free_qv_adr + 6].fill(0)
        self.reset_bullet()
    
    def shoot(self):
        if not self.bullet_aloft:
            bullet_pos = self.data.xpos[self.agent_id] + quaternion.rotate_vectors(
                self.data.xquat[self.agent_id], np.array([0.5, 0, 0])
            )
            bullet_quat = self.data.xquat[self.agent_id]
            bullet_vel = quaternion.rotate_vectors(bullet_quat, np.array([self.bullet_speed, 0, 0]))
            self.data.qpos[self.bullet_free_joint_id: self.bullet_free_joint_id + 7] = np.concatenate(
                [bullet_pos, quaternion.as_float_array(bullet_quat)]
            )
            self.data.qvel[self.bullet_free_joint_id: self.bullet_free_joint_id + 6] = bullet_vel
            self.model.body_contype[self.bullet_body_id] = 1
            self.model.body_conaffinity[self.bullet_body_id] = 1
            self.just_shot = True
        else:
            self.shot_while_no_bullet = True
    
    def reset_default_flags(self):
        self.just_shot = False
        self.shot_while_no_bullet = False
        self.shot_drone = False
        self.hit_floor = False
        self.got_shot = False
        self.out_of_bounds = False
        self.bullet_out_of_bounds = False
    
    @property
    def observation(self) -> np.ndarray:
        return self.sensor.observation
    
    @property
    def observation_space(self) -> Box:
        return self.sensor.observation_space
    
    @abstractmethod
    def act(self, action: EnvActionType) -> None:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def action_space(self) -> EnvActionType:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def reward(self) -> float:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def done(self) -> bool:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def truncated(self) -> bool:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def log_info(self) -> dict:
        raise NotImplementedError
    
    @abstractmethod
    def custom_update(self):
        raise NotImplementedError


class ShootingDrone(Drone, ABC):
    """
    Represents a drone with shooting capabilities, extending the Drone class. It provides specific implementations
    for action space and the act method to incorporate shooting actions.

    Methods:
        action_space: Overrides the Drone class's action_space to include shooting actions.
        act: Overrides the Drone class's act method to handle movement and shooting actions.
    """
    def __init__(self, **kwargs):
        super(ShootingDrone, self).__init__(**kwargs)
    
    @property
    def action_space(self) -> EnvActionType:
        return TupleSpace(
            (Box(low=0, high=self.model.actuator_ctrlrange[self.actuator_ids[0], 1], shape=(4,), dtype=np.float64),
             Discrete(1)))
    
    def act(self, action: EnvActionType) -> None:
        self.data.ctrl[self.actuator_ids] = action[0]
        if action[1]:
            self.shoot()


class HoverDrone(Drone, ABC):
    """
    Represents a drone designed for hovering, extending the Drone class. It provides specific implementations
    for the action space and the act method tailored for hovering actions.

    Methods:
        action_space: Overrides the Drone class's action_space to suit hover-specific actions.
        act: Overrides the Drone class's act method to handle hover-specific actions.
    """
    def __init__(self, **kwargs):
        super(HoverDrone, self).__init__(**kwargs)
    
    @property
    def action_space(self) -> EnvActionType:
        return Box(low=0, high=self.model.actuator_ctrlrange[self.actuator_ids[0], 1], shape=(4,), dtype=np.float64)
    
    def act(self, action: EnvActionType) -> None:
        self.data.ctrl[self.actuator_ids] = action
