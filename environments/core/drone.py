from abc import abstractmethod, ABC
from copy import copy, deepcopy
from typing import Union, List, Dict

import numpy as np
import quaternion
from gymnasium.spaces import Box, Tuple as TupleSpace, Dict as DictSpace, Discrete
from mujoco import MjModel, MjData, mj_multiRay, Renderer, mjtObj, mjtJoint, mj_name2id
from mujoco import mjtGeom
from ray.rllib.utils.typing import AgentID, EnvActionType, EnvObsType
from scipy.spatial.transform import Rotation


class Sensor(ABC):
    model: MjModel
    data: MjData
    agent_id: AgentID
    
    def __init__(self):
        pass
    
    @abstractmethod
    def environment_init(self, model: MjModel, data: MjData, agent_id: AgentID, **kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def update(self):
        raise NotImplementedError
    
    @property
    @abstractmethod
    def observation(self) -> dict:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def observation_space(self) -> DictSpace:
        raise NotImplementedError
    
    def __call__(self) -> dict:
        return self.observation


class Lidar(Sensor):
    body_id: int
    free_joint_qpos_address: int
    
    def __init__(self, n_thetas: int, n_phis: int, ray_distance: float):
        super().__init__()
        self.n_thetas = n_thetas
        self.n_phis = n_phis
        self.n_rays = n_thetas * n_phis
        
        self._directional_unit_vectors = np.array(
            [[np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
             for theta in np.linspace(0, np.pi, n_thetas)
             for phi in np.linspace(0, 2 * np.pi, n_phis)]
        )
        self.initial_ray_unit_vectors = np.array(
            [
                [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
                for theta in np.linspace(0, np.pi, n_thetas) for phi in np.linspace(0, 2 * np.pi, n_phis)
            ]
        )
        self.flattened_ray_unit_vectors = self.initial_ray_unit_vectors.copy().flatten()
        self.distances = np.zeros(n_phis * n_thetas, dtype=np.float64)
        self.intersecting_geoms = np.zeros(n_phis * n_thetas, dtype=np.int32)
        self.ray_max_distance = float(ray_distance)
        self.num_rays = n_phis * n_thetas
    
    def environment_init(self, model: MjModel, data: MjData, agent_id: AgentID, **kwargs):
        self.model = model
        self.data = data
        self.agent_id = agent_id
        self.body_id = mj_name2id(self.model, mjtObj.mjOBJ_BODY, f"drone_{self.agent_id}")
        self.free_joint_qpos_address = self.model.jnt_qposadr[mj_name2id(self.model, mjtJoint.mjJNT_FREE, f"drone_{self.agent_id}_free_joint")]
    
    def update(self):
        if not all(self.data.xquat[self.body_id] == 0):
            self.flattened_ray_unit_vectors[:] = Rotation.from_quat(
                self.data.qpos[self.free_joint_qpos_address + 3: self.free_joint_qpos_address + 7]  # [x, y, z, w]
            ).apply(self.initial_ray_unit_vectors).flatten()  # [x, y, z] -> [x1, y1, z1, x2, y2, z2, ...]
            mj_multiRay(
                m=self.model,
                d=self.data,
                pnt=self.data.xpos[self.body_id],
                vec=self.flattened_ray_unit_vectors,
                geomgroup=None,
                flg_static=1,
                bodyexclude=self.body_id,
                geomid=self.intersecting_geoms,
                dist=self.distances,
                nray=self.num_rays,
                cutoff=self.ray_max_distance
            )
            self.distances[self.distances < 0] = self.ray_max_distance
            np.clip(self.distances, a_min=0, a_max=self.ray_max_distance, out=self.distances)
    
    @property
    def observation(self) -> dict:
        return {'Lidar': copy(self.distances)}
    
    @property
    def observation_space(self) -> DictSpace:
        return DictSpace({
            'Lidar': Box(low=0, high=self.ray_max_distance, shape=(self.n_rays,), dtype=np.float64)
        })


class Camera(Sensor):
    camera_id: Union[int, str]
    
    def __init__(self, depth: bool, renderer: Renderer, max_distance: float = 10.0):
        super().__init__()
        self.depth = depth
        self.renderer = renderer
        self.max_distance = max_distance
        height, width = self.renderer.height, self.renderer.width
        if depth:
            dimensions = (height, width)
        else:
            dimensions = (height, width, 3)
        self.image = np.zeros(dimensions)
    
    def environment_init(self, model: MjModel, data: MjData, agent_id: AgentID, **kwargs):
        self.model = model
        self.data = data
        self.agent_id = agent_id
        
        self.camera_id = mj_name2id(self.model, mjtObj.mjOBJ_CAMERA, f"camera{agent_id}")
        self.renderer.update_scene(data=self.data, camera=self.camera_id)
    
    def update(self):
        self.renderer.render(out=self.image)
        if self.depth:
            np.divide(self.image, self.max_distance, out=self.image)
            np.clip(self.image, 0, 1, out=self.image)
        else:
            np.divide(self.image, 255, out=self.image)
            np.clip(self.image, 0, 1, out=self.image)
    
    @property
    def observation(self) -> dict:
        return {'Camera': copy(self.image)}
    
    @property
    def observation_space(self) -> DictSpace:
        return DictSpace({
            'Camera': Box(low=0, high=1, shape=self.image.shape, dtype=np.float64)
        })


class ClosestDronesSensor(Sensor):
    distances: np.ndarray
    gammas: np.ndarray
    max_distance: float
    world_bounds: np.ndarray
    gamma_sin: np.ndarray
    gamma_cos: np.ndarray
    num_drones: int
    
    def __init__(self):
        super().__init__()
    
    def environment_init(self, model: MjModel, data: MjData, agent_id: AgentID, **kwargs):
        self.agent_id = agent_id
        
        if "world_bounds" not in kwargs:
            raise ValueError("Missing required parameter: world_bounds")
        self.world_bounds = kwargs["world_bounds"]
        self.max_distance = np.linalg.norm(self.world_bounds[1] - self.world_bounds[0])
        
        if "gammas" not in kwargs:
            raise ValueError("Missing required parameter: gammas")
        self.gammas = kwargs["gammas"]
        
        if "distances" not in kwargs:
            raise ValueError("Missing required parameter: distances")
        self.distances = kwargs["distances"]
        
        self.num_drones = len(self.distances)
        
        self.gamma_sin = np.zeros((self.num_drones,))
        self.gamma_cos = np.zeros((self.num_drones,))
    
    def update(self):
        np.sin(self.gammas[self.agent_id], out=self.gamma_sin)
        np.cos(self.gammas[self.agent_id], out=self.gamma_cos)
    
    @property
    def observation(self) -> dict:
        return {
            "Distances": self.distances[self.agent_id],
            "SinCos": np.array([self.gamma_sin, self.gamma_cos])
        }
    
    @property
    def observation_space(self) -> DictSpace:
        return DictSpace({
            "Distances": Box(low=0, high=self.max_distance, shape=(self.num_drones,), dtype=np.float64),
            "SinCos": Box(low=-1, high=1, shape=(2, self.num_drones), dtype=np.float64)
        })


class Gyroscope(Sensor):
    sensor_id: int
    
    def __init__(self):
        super().__init__()
        self.return_data = np.zeros(3)
    
    def environment_init(self, model: MjModel, data: MjData, agent_id: AgentID, **kwargs):
        self.model = model
        self.data = data
        self.agent_id = agent_id
        self.sensor_id = mj_name2id(self.model, mjtObj.mjOBJ_SENSOR, f"gyro{self.agent_id}")
    
    def update(self):
        if self.model.sensor_cutoff[self.sensor_id]:
            np.clip(
                self.data.sensordata[self.sensor_id: self.sensor_id + 3] / self.model.sensor_cutoff[self.sensor_id],
                a_min=-1, a_max=1, out=self.return_data
            )
    
    @property
    def observation(self) -> dict:
        return {'Gyro': copy(self.return_data)}
    
    @property
    def observation_space(self) -> DictSpace:
        return DictSpace({
            'Gyro': Box(low=-1, high=1, shape=(3,), dtype=np.float64)
        })


class Accelerometer(Sensor):
    sensor_id: int
    
    def __init__(self):
        super().__init__()
        self.return_data = np.zeros(3)
        
    def environment_init(self, model: MjModel, data: MjData, agent_id: AgentID, **kwargs):
        self.model = model
        self.data = data
        self.agent_id = agent_id
        self.sensor_id = mj_name2id(self.model, mjtObj.mjOBJ_SENSOR, f"accelerometer{self.agent_id}")
        
    def update(self):
        if self.model.sensor_cutoff[self.sensor_id]:
            np.clip(
                self.data.sensordata[self.sensor_id: self.sensor_id + 3] / self.model.sensor_cutoff[self.sensor_id],
                a_min=-1, a_max=1, out=self.return_data
            )
            
    @property
    def observation(self) -> dict:
        return {'Accelerometer': copy(self.return_data)}
    
    @property
    def observation_space(self) -> DictSpace:
        return DictSpace({
            'Accelerometer': Box(low=-1, high=1, shape=(3,), dtype=np.float64)
        })
    

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
        
        self.body_id: int = mj_name2id(self.model, mjtObj.mjOBJ_BODY, f"drone_{self.agent_id}")
        self.free_joint_id: int = mj_name2id(self.model, mjtJoint.mjJNT_FREE, f"drone_{self.agent_id}_free_joint")
        self.free_joint_qpos_address: int = self.model.jnt_qposadr[self.free_joint_id]
        self.free_joint_qvel_address: int = self.model.jnt_dofadr[self.free_joint_id]
        self.collision_geom_1_id: int = mj_name2id(self.model, mjtGeom.mjGEOM_BOX, f"drone_{self.agent_id}_collision_1")
        self.collision_geom_2_id: int = mj_name2id(self.model, mjtGeom.mjGEOM_BOX, f"drone_{self.agent_id}_collision_2")
        self.bullet_body_id: int = mj_name2id(self.model, mjtObj.mjOBJ_BODY, f"bullet_{self.agent_id}")
        self.bullet_free_joint_id: int = mj_name2id(self.model, mjtJoint.mjJNT_FREE, f"bullet_{self.agent_id}")
        self.bullet_geom_id: int = mj_name2id(self.model, mjtGeom.mjGEOM_CAPSULE, f"bullet_{self.agent_id}")
        self.bullet_spawn_point: int = mj_name2id(model, mjtObj.mjOBJ_SITE, f"bullet_spawn_position_{self.agent_id}")
        motor_names = [f"front_left_{self.agent_id}", f"back_right_{self.agent_id}", f"front_right_{self.agent_id}", f"back_left_{self.agent_id}"]
        self.actuator_ids: List[int] = [mj_name2id(self.model, mjtObj.mjOBJ_ACTUATOR, name) for name in motor_names]
        
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
    