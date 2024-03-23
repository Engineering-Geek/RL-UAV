from abc import abstractmethod, ABC
from copy import copy
from typing import Union

import numpy as np
from gymnasium.spaces import Box, Dict as DictSpace
from mujoco import MjModel, MjData, mj_multiRay, Renderer, mjtObj, mjtJoint, mj_name2id
from ray.rllib.utils.typing import AgentID
from scipy.spatial.transform import Rotation


class Sensor(ABC):
    """
    An abstract base class representing a sensor in a simulation environment. It provides a structure for sensor
    initialization, updating, and defining observations in simulation environments.

    Methods:
        environment_init: Abstract method to initialize the sensor in the environment.
        update: Abstract method to update the sensor's readings based on the environment's current state.
        observation: Abstract property that returns the current observation from the sensor.
        observation_space: Abstract property that returns the observation space of the sensor.
    """
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
    def observation(self) -> np.ndarray:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def observation_space(self) -> Box:
        raise NotImplementedError
    
    def __call__(self) -> np.ndarray:
        return self.observation


class Lidar(Sensor):
    """
    A Lidar sensor class that extends Sensor, simulating a lidar sensor's functionality within the environment.
    It calculates distances in various directions using ray casting.

    Attributes:
        n_thetas (int): The number of vertical divisions in the lidar scan.
        n_phis (int): The number of horizontal divisions in the lidar scan.
        ray_max_distance (float): The maximum distance each ray can detect.
    
    Methods:
        environment_init: Initializes the Lidar sensor within the MuJoCo environment.
        update: Updates the distances detected by the Lidar sensor in all specified directions.
        observation: Returns the current Lidar distance readings.
        observation_space: Returns the observation space for the Lidar sensor.
    """
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
        self.free_joint_qpos_address = self.model.jnt_qposadr[
            mj_name2id(self.model, mjtJoint.mjJNT_FREE, f"drone_{self.agent_id}_free_joint")]
    
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
            np.divide(self.distances, self.ray_max_distance, out=self.distances)
            np.clip(self.distances, a_min=0, a_max=1, out=self.distances)
    
    @property
    def observation(self) -> np.ndarray:
        return copy(self.distances)
    
    @property
    def observation_space(self) -> Box:
        return Box(low=0, high=1, shape=(self.num_rays,), dtype=np.float64)


class Camera(Sensor):
    """
    A Camera sensor class that extends Sensor, simulating a camera within the environment. It can capture both
    RGB and depth images.

    Attributes:
        depth (bool): Flag indicating whether the camera captures depth images.
        renderer (Renderer): The renderer used for capturing images.
        max_distance (float): The maximum detectable distance for depth sensing.
    
    Methods:
        environment_init: Initializes the Camera sensor within the MuJoCo environment.
        update: Updates the camera image based on the current state of the environment.
        observation: Returns the current camera image.
        observation_space: Returns the observation space for the camera sensor.
    """
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
    def observation(self) -> np.ndarray:
        return copy(self.image)
    
    @property
    def observation_space(self) -> Box:
        return Box(low=0, high=1, shape=self.image.shape, dtype=np.float64)


class Gyroscope(Sensor):
    """
    A Gyroscope sensor class that extends Sensor, simulating a gyroscope within the environment.

    Methods:
        environment_init: Initializes the Gyroscope sensor within the MuJoCo environment.
        update: Updates the gyroscope readings based on the current state of the environment.
        observation: Returns the current gyroscope readings.
        observation_space: Returns the observation space for the Gyroscope sensor.
    """
    sensor_id: int
    
    def __init__(self):
        super().__init__()
        self.return_data = np.zeros(3)
    
    def environment_init(self, model: MjModel, data: MjData, agent_id: AgentID, **kwargs):
        self.model = model
        self.data = data
        self.agent_id = agent_id
        self.sensor_id = self.model.sensor(f"gyro{self.agent_id}").id
    
    def update(self):
        if self.model.sensor_cutoff[self.sensor_id]:
            np.clip(
                self.data.sensordata[self.sensor_id: self.sensor_id + 3] / self.model.sensor_cutoff[self.sensor_id],
                a_min=-1, a_max=1, out=self.return_data
            )
    
    @property
    def observation(self) -> np.ndarray:
        return copy(self.return_data)
    
    @property
    def observation_space(self) -> Box:
        return Box(low=-1, high=1, shape=(3,), dtype=np.float64)


class Accelerometer(Sensor):
    """
    An Accelerometer sensor class that extends Sensor, simulating an accelerometer within the environment.

    Methods:
        environment_init: Initializes the Accelerometer sensor within the MuJoCo environment.
        update: Updates the accelerometer readings based on the current state of the environment.
        observation: Returns the current accelerometer readings.
        observation_space: Returns the observation space for the Accelerometer sensor.
    """
    sensor_id: int
    
    def __init__(self):
        super().__init__()
        self.return_data = np.zeros(3)
    
    def environment_init(self, model: MjModel, data: MjData, agent_id: AgentID, **kwargs):
        self.model = model
        self.data = data
        self.agent_id = agent_id
        self.sensor_id = self.model.sensor(f"accelerometer{self.agent_id}").id
    
    def update(self):
        if self.model.sensor_cutoff[self.sensor_id]:
            np.clip(
                self.data.sensordata[self.sensor_id: self.sensor_id + 3] / self.model.sensor_cutoff[self.sensor_id],
                a_min=-1, a_max=1, out=self.return_data
            )
    
    @property
    def observation(self) -> np.ndarray:
        return copy(self.return_data)
    
    @property
    def observation_space(self) -> Box:
        return Box(low=-1, high=1, shape=(3,), dtype=np.float64)
