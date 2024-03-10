# region Imports and global variables
import os
from abc import ABC, abstractmethod
from typing import SupportsFloat, Any, Tuple

import numpy as np
from gymnasium.core import ActType, ObsType
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Space, Box, Dict
from gymnasium.utils import EzPickle
from mujoco import MjModel, MjData, mj_step
from mujoco._structs import _MjDataBodyViews
import quaternion
from numpy.random import default_rng

XML_PATH = os.path.join(os.path.dirname(__file__), "assets/UAV/scene.xml")


# endregion

# region Drone and Target Classes
class Drone:
    """
    The Drone class represents a drone in the simulation.

    :param data: The MjData instance from the simulation
    :type data: MjData
    :param spawn_box: The box in which the drone can spawn
    :type spawn_box: np.ndarray
    :param spawn_max_velocity: The maximum velocity at which the drone can spawn
    :type spawn_max_velocity: float
    :param rng: The random number generator used for spawning the drone
    :type rng: np.random.Generator, optional
    """
    def __init__(self,
                 data: MjData,
                 spawn_box: np.ndarray,
                 spawn_max_velocity: float,
                 rng: np.random.Generator = default_rng()
                 ):
        """
        Initialize the Drone object
        :param data:
        :param spawn_box:
        :param spawn_max_velocity:
        """
        self.data = data
        self.body: _MjDataBodyViews = data.body('drone')
        self.spawn_box = spawn_box
        self.spawn_max_velocity = spawn_max_velocity
        self.rng = rng
        self.id = self.body.id
    
    @property
    def position(self) -> np.ndarray:
        return self.body.xpos
    
    @position.setter
    def position(self, value: np.ndarray) -> None:
        self.body.xpos = value
    
    @property
    def velocity(self) -> np.ndarray:
        return self.body.cvel[:3]
    
    @velocity.setter
    def velocity(self, value: np.ndarray) -> None:
        self.body.cvel[:3] = value
    
    @property
    def angular_velocity(self) -> np.ndarray:
        return self.body.qvel[3:6]
    
    @angular_velocity.setter
    def angular_velocity(self, value: np.ndarray) -> None:
        self.body.qvel[3:6] = value
    
    @property
    def imu_accel(self) -> np.ndarray:
        return self.data.sensor('imu_accel').data
    
    @property
    def imu_gyro(self) -> np.ndarray:
        return self.data.sensor('imu_gyro').data
    
    @property
    def imu_orientation(self) -> np.ndarray:
        return self.data.sensor('imu_orientation').data
    
    def reset(self):
        """
        Reset the drone's position, orientation, velocity, and angular velocity.
        """
        self.position = self.spawn_box[0] + (self.spawn_box[1] - self.spawn_box[0]) * self.rng.random(3)
        self.velocity = self.spawn_max_velocity * self.rng.random(3)


class Target:
    """
    The Target class represents a target in the simulation.

    :param data: The MjData instance from the simulation
    :type data: MjData
    :param spawn_box: The box in which the target can spawn
    :type spawn_box: np.ndarray
    :param spawn_max_velocity: The maximum velocity at which the target can spawn
    :type spawn_max_velocity: float
    :param spawn_max_angular_velocity: The maximum angular velocity at which the target can spawn
    :type spawn_max_angular_velocity: float
    :param rng: The random number generator used for spawning the target
    :type rng: np.random.Generator, optional
    """
    def __init__(self,
                 data: MjData,
                 spawn_box: np.ndarray,
                 spawn_max_velocity: float,
                 spawn_max_angular_velocity: float,
                 rng: np.random.Generator = default_rng()
                 ):
        """
        Initialize the Target object
        :param data:
        :param spawn_box:
        :param spawn_max_velocity:
        :param spawn_max_angular_velocity:
        """
        self.data = data
        self.body: _MjDataBodyViews = data.body('target')
        self.spawn_box = spawn_box
        self.spawn_max_velocity = spawn_max_velocity
        self.spawn_max_angular_velocity = spawn_max_angular_velocity
        self.rng = rng
        self.id = self.body.id
    
    @property
    def position(self) -> np.ndarray:
        return self.body.xpos
    
    @position.setter
    def position(self, value: np.ndarray) -> None:
        self.body.xpos = value
    
    @property
    def velocity(self) -> np.ndarray:
        return self.body.cvel[:3]
    
    @velocity.setter
    def velocity(self, value: np.ndarray) -> None:
        self.body.cvel[:3] = value
    
    @property
    def orientation(self) -> np.ndarray:
        return self.body.qpos[3:7]
    
    @orientation.setter
    def orientation(self, value: np.ndarray) -> None:
        self.body.qpos[3:7] = value
    
    def reset(self):
        """
        Reset the target's position, orientation, velocity, and angular velocity.
        """
        self.position = self.spawn_box[0] + (self.spawn_box[1] - self.spawn_box[0]) * self.rng.random(3)
        self.velocity = self.spawn_max_velocity * self.rng.random(3)
# endregion


# region BaseMujocoEnv
class _BaseRegime(MujocoEnv, EzPickle, ABC):
    """
    _BaseRegime is a custom environment that extends the
    MujocoEnv. It is designed to simulate a drone in a 3D space
    with a target. The drone's task is to reach the target.

    :param kwargs: Keyword arguments for the _BaseRegime environment
    :type kwargs: dict
    """
    
    # region Initialization
    def __init__(self, **kwargs):
        """
        Initialize the _BaseRegime environment.
        :key xml_path: Path to the XML file that describes the
        :key sim_rate: Simulation rate
        :key dt: Simulation timestep
        :key height: Height of the camera
        :key width: Width of the camera
        :key drone_spawn_box: Box in which the drone can spawn. First
            numpy array is the lower bound and the second numpy array is
            the upper bound.
        :key drone_spawn_angle_range: Range of angles in which the drone
            can spawn. First numpy array is the lower bound and the second
            numpy array is the upper bound.
        :key drone_spawn_max_velocity: Maximum velocity at which the drone
            can spawn.
        :key drone_spawn_max_angular_velocity: Maximum angular velocity at
            which the drone can spawn.
        :key target_spawn_box: Box in which the target can spawn. First
            numpy array is the lower bound and the second numpy array is
            the upper bound.
        """
        # region Initialize MujocoEnv
        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": 500,
            # render_fps will be set to dt^-1 after the model loads
        }
        height = kwargs.get('height', 480)
        width = kwargs.get('width', 640)
        observation_space: Space[ObsType] = Dict({
            "imu_accel":
                Box(low=-np.inf, high=np.inf, shape=(3,)),
            "imu_gyro":
                Box(low=-np.inf, high=np.inf, shape=(4,)),
            "imu_orientation":
                Box(low=-np.inf, high=np.inf, shape=(3,)),
            "image_0":
                Box(low=0, high=255, shape=(height, width, 3), dtype=np.uint8),
            "image_1":
                Box(low=0, high=255, shape=(height, width, 3), dtype=np.uint8),
        })
        
        MujocoEnv.__init__(
            self,
            model_path=XML_PATH if 'model_path' not in kwargs else kwargs['model_path'],
            frame_skip=kwargs.get('frame_skip', 1),
            observation_space=observation_space,
            height=height,
            width=width,
        )
        EzPickle.__init__(
            self,
            XML_PATH,
            kwargs.get('frame_skip', 1),
            kwargs.get('dt', self.dt),
            **kwargs,
        )
        # Set action space
        self.action_space: Space = (
            Box(low=-1.0, high=1.0, shape=(self.model.nu,)))
        
        # Set simulation timestep
        self.model.opt.timestep = kwargs.get('dt', self.dt)
        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }
        # endregion
        
        # region Initialize Random Number Generator and Model/Data
        # Initialize random number generator
        self.rng = default_rng()
        
        # Initialize model and data attributes
        self.model: MjModel = self.model
        self.data: MjData = self.data
        # endregion
        
        # region Drone Parameters
        self.drone = Drone(
            data=self.data,
            spawn_box=kwargs.get('drone_spawn_box',
                                 np.array([np.array([-10, -10, 0.5]), np.array([10, 10, 0.5])])),
            spawn_max_velocity=kwargs.get('drone_spawn_max_velocity', 0.5),
            rng=self.rng
        )
        # endregion
        
        # region Target Parameters
        self.target = Target(
            data=self.data,
            spawn_box=kwargs.get('target_spawn_box',
                                 np.array([np.array([-10, -10, 0.5]), np.array([10, 10, 0.5])])),
            spawn_max_velocity=kwargs.get('target_spawn_max_velocity', 0.5),
            spawn_max_angular_velocity=kwargs.get('target_spawn_max_angular_velocity', 0.5),
            rng=self.rng
        )
        # endregion
    
    # endregion
    
    # region Properties
    @property
    def drone_accel(self) -> np.ndarray:
        return self.data.sensor('imu_accel').data
    
    @property
    def drone_gyro(self) -> np.ndarray:
        return self.data.sensor('imu_gyro').data
    
    @property
    def drone_orientation(self) -> np.ndarray:
        return self.data.sensor('imu_orientation').data
    
    @property
    def camera_0(self) -> np.ndarray:
        return self.mujoco_renderer.render('rgb_array', 0)
    
    @property
    def camera_1(self) -> np.ndarray:
        return self.mujoco_renderer.render('rgb_array', 1)
    
    @property
    def target_position(self) -> np.ndarray:
        return self.data.body('target').xpos
    
    @property
    def target_velocity(self) -> np.ndarray:
        return self.data.body('target').cvel[:3]
    
    @property
    def drone_target_vector(self) -> np.ndarray:
        return self.target.position - self.drone.position
    
    @property
    def drone_hit_ground(self) -> bool:
        drone_id = self.model.geom('drone').id
        floor_id = self.model.geom('floor').id
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if (contact.geom1 == drone_id and contact.geom2 == floor_id) or (
                    contact.geom1 == floor_id and contact.geom2 == drone_id):
                return True
        return False
    
    # endregion
    
    # region Methods
    def place_target(self,
                     target_pos: np.ndarray,
                     target_orientation: np.ndarray) -> None:
        """
        Place the target at a specific position and orientation.

        :param target_pos: The position to place the target
        :type target_pos: np.ndarray
        :param target_orientation: The orientation to set for the target
        :type target_orientation: np.ndarray
        """
        self.data.body('target').qpos[:3] = target_pos
        self.data.body('target').qpos[3:] = target_orientation
    
    def move_target(self, target_pos: np.ndarray) -> None:
        """
        Move the target to a specific position.

        :param target_pos: The position to move the target to
        :type target_pos: np.ndarray
        """
        self.data.body('target').xpos = target_pos
    
    def pre_simulation(self) -> None:
        """
        Perform any necessary operations before the simulation.
        """
        pass
    
    def reset_model(self) -> ObsType:
        """
        Reset the model of the environment.

        :return: The initial observation of the environment
        :rtype: ObsType
        """
        self.drone.reset()
        self.target.reset()
        return self.observation
    
    # endregion
    
    # region Step Logic
    def step(
            self, action: ActType
    ) -> Tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Perform a step in the environment using the given action.

        :param action: The action to perform
        :type action: ActType
        :return: A tuple containing the new observation, reward, whether the episode was truncated, whether the episode is done, and additional metrics
        :rtype: Tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]
        """
        self.pre_simulation()
        self.do_simulation(action, self.frame_skip)
        return (self.observation, self.reward,
                self.truncated, self.done, self.metrics)
    
    @property
    def observation(self) -> ObsType:
        return {
            "drone_position": self.data.qpos[0:3],
            "drone_velocity": self.data.qvel[0:3],
            "drone_orientation": self.data.qpos[3:7],
            "drone_angular_velocity": self.data.qvel[3:6],
            "image_0": self.mujoco_renderer.render('rgb_array', 0),
            "image_1": self.mujoco_renderer.render('rgb_array', 1),
        }
    
    @property
    @abstractmethod
    def reward(self) -> SupportsFloat:
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
    def metrics(self) -> dict[str, Any]:
        raise NotImplementedError
    
    # endregion

# endregion
