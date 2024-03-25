from abc import abstractmethod, ABC
from copy import deepcopy
from typing import List, Dict

import numpy as np
import quaternion
from gymnasium.spaces import Box, Tuple as TupleSpace, Dict as DictSpace, Discrete
from mujoco import MjModel, MjData
from mujoco._structs import _MjModelBodyViews, _MjDataBodyViews, _MjModelGeomViews, \
    _MjDataGeomViews, _MjModelJointViews, _MjDataJointViews, _MjDataActuatorViews, _MjModelActuatorViews
from ray.rllib.utils.typing import AgentID, EnvActionType

from environments.core.sensors import Sensor


class Drone(ABC):
    def __init__(self, model: MjModel, data: MjData, agent_id: AgentID, spawn_box: np.ndarray, gammas: np.ndarray,
                 distances: np.ndarray, sensors: List[Sensor], bullet_speed: float = 10.0):
        self.model = model
        self.data = data
        self.agent_id = agent_id
        self.spawn_box = spawn_box
        self.gammas = gammas  # angle between the local +x-axis and the vector pointing to other drones
        self.distances = distances  # distance to other drones
        self.sensors = sensors
        
        self._initial_xpos = deepcopy(self.data.xpos[self.agent_id])
        x_0, y_0, z_0 = self._initial_xpos
        x_min, y_min, z_min = self.spawn_box[0] + np.array([x_0, y_0, 0])
        x_max, y_max, z_max = self.spawn_box[1] + np.array([x_0, y_0, 0])
        self.spawn_box = np.array([[x_min, y_min, z_min], [x_max, y_max, z_max]])
        self.spawn_angles = np.array([[0, 0, 0], [2 * np.pi, 2 * np.pi, 2 * np.pi]])

        motor_names = [f"front_left_{self.agent_id}", f"back_right_{self.agent_id}",
                       f"front_right_{self.agent_id}", f"back_left_{self.agent_id}"]
        
        self.body_model: _MjModelBodyViews = self.model.body(f"drone_{self.agent_id}")
        self.free_joint_model: _MjModelJointViews = self.model.joint(f"drone_{self.agent_id}_free_joint")
        self.geom1_model: _MjModelGeomViews = self.model.geom(f"drone_{self.agent_id}_collision_1")
        self.geom2_model: _MjModelGeomViews = self.model.geom(f"drone_{self.agent_id}_collision_2")
        self.bullet_body_model: _MjModelBodyViews = self.model.body(f"bullet_{self.agent_id}")
        self.bullet_free_joint_model: _MjModelJointViews = self.model.joint(f"bullet_{self.agent_id}")
        self.bullet_geom_model: _MjModelGeomViews = self.model.geom(f"bullet_{self.agent_id}")
        self.bullet_spawn_point_model: _MjModelGeomViews = self.model.site(f"bullet_spawn_position_{self.agent_id}")
        self.actuators_model: List[_MjModelActuatorViews] = [self.model.actuator(name) for name in motor_names]
        
        self.body_data: _MjDataBodyViews = self.data.body(f"drone_{self.agent_id}")
        self.free_joint_data: _MjDataJointViews = self.data.joint(f"drone_{self.agent_id}_free_joint")
        self.geom1_data: _MjDataGeomViews = self.data.geom(f"drone_{self.agent_id}_collision_1")
        self.geom2_data: _MjDataGeomViews = self.data.geom(f"drone_{self.agent_id}_collision_2")
        self.bullet_body_data: _MjDataBodyViews = self.data.body(f"bullet_{self.agent_id}")
        self.bullet_free_joint_data: _MjDataJointViews = self.data.joint(f"bullet_{self.agent_id}")
        self.bullet_geom_data: _MjDataGeomViews = self.data.geom(f"bullet_{self.agent_id}")
        self.bullet_spawn_point_data: _MjDataGeomViews = self.data.site(f"bullet_spawn_position_{self.agent_id}")
        self.actuators_data: List[_MjDataActuatorViews] = [self.data.actuator(name) for name in motor_names]
        
        self.body_id: int = self.body_model.id
        self.free_joint_id: int = self.free_joint_model.id
        self.free_joint_qpos_address: int = self.free_joint_model.qposadr[0]
        self.free_joint_qvel_address: int = self.free_joint_model.dofadr[0]
        self.collision_geom_1_id: int = self.geom1_model.id
        self.collision_geom_2_id: int = self.geom2_model.id
        self.bullet_body_id: int = self.bullet_body_model.id
        self.bullet_free_joint_id: int = self.bullet_free_joint_model.id
        self.bullet_geom_id: int = self.bullet_geom_model.id
        self.bullet_spawn_point: int = self.bullet_spawn_point_model.id
        self.actuator_ids: List[int] = [actuator.id for actuator in self.actuators_model]
        
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
    
    def update_sensors(self):
        for sensor in self.sensors:
            sensor.update()
    
    def reset_bullet(self):
        self.model.body_contype[self.bullet_body_id] = 0
        self.model.body_conaffinity[self.bullet_body_id] = 0
        self.data.qpos[self.bullet_free_joint_id: self.bullet_free_joint_id + 7].fill(0)
        self.data.qvel[self.bullet_free_joint_id: self.bullet_free_joint_id + 6].fill(0)
        self.bullet_aloft = False
    
    def reset(self):
        pos = np.random.uniform(*self.spawn_box) + self._initial_xpos[:3]
        quat = quaternion.from_euler_angles(np.random.uniform(*self.spawn_angles))
        self.data.qpos[self.free_joint_qpos_address: self.free_joint_qpos_address + 7] = np.concatenate(
            [pos, quaternion.as_float_array(quat)])
        self.data.qvel[self.free_joint_qvel_address: self.free_joint_qvel_address + 6].fill(0)
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
        self.crash_into_drone = False
        
    @abstractmethod
    def reset_custom_flags(self):
        raise NotImplementedError
    
    @property
    def observation(self) -> Dict:
        observations = dict()
        for sensor in self.sensors:
            observations.update(sensor.observation)
        return observations
    
    @property
    def observation_space(self) -> DictSpace:
        observation_space = dict()
        for sensor in self.sensors:
            observation_space.update(sensor.observation_space)
        return DictSpace(observation_space)
    
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
    def __init__(self, **kwargs):
        super(ShootingDrone, self).__init__(**kwargs)
    
    @property
    def action_space(self) -> EnvActionType:
        return TupleSpace((Box(low=0, high=self.model.actuator_ctrlrange[self.actuator_ids[0], 1], shape=(4,), dtype=np.float64),
                           Discrete(1)))
    
    def act(self, action: EnvActionType) -> None:
        self.data.ctrl[self.actuator_ids] = action[0]
        if action[1]:
            self.shoot()
            

class HoverDrone(Drone, ABC):
    def __init__(self, **kwargs):
        super(HoverDrone, self).__init__(**kwargs)
    
    @property
    def action_space(self) -> EnvActionType:
        return Box(low=0, high=self.model.actuator_ctrlrange[self.actuator_ids[0], 1], shape=(4,), dtype=np.float64)
    
    def act(self, action: EnvActionType) -> None:
        self.data.ctrl[self.actuator_ids] = action
    