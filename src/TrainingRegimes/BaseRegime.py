# region Imports and global variables
import os
from abc import ABC, abstractmethod
from typing import SupportsFloat, Any, Tuple

import numpy as np
from gymnasium.core import ActType, ObsType
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Space, Box, Dict
from gymnasium.utils import EzPickle
from mujoco import MjModel, MjData
from mujoco._structs import _MjDataBodyViews
from numpy.random import default_rng

XML_PATH_MAIN = os.path.join(os.path.dirname(__file__), "assets/UAV/scene.xml")
XML_PATH_REDUCED = os.path.join(os.path.dirname(__file__), "assets/ReducedUAV/scene.xml")


# endregion

# region Drone and Target Classes
class Drone:
    """
    Represents a drone in the simulation environment. It manages the drone's state, including its position,
    velocity, and IMU sensor readings.

    :param data: The MjData instance containing the simulation state.
    :param spawn_box: A numpy array defining the boundaries for the drone's initial position.
    :param spawn_max_velocity: The maximum initial velocity of the drone.
    :param rng: An instance of a random number generator (optional).
    """
    def __init__(self,
                 data: MjData,
                 spawn_box: np.ndarray,
                 spawn_max_velocity: float,
                 rng: np.random.Generator = default_rng()
                 ):
        """
        Initializes the Drone instance with the provided parameters.

        :param data: The MjData instance from the simulation environment.
        :param spawn_box: A 2x3 numpy array defining the min and max coordinates for spawning the drone.
        :param spawn_max_velocity: The maximum magnitude of the drone's initial velocity vector.
        :param rng: A numpy random number generator instance for stochasticity in initial conditions.
        """
        self.data = data
        self.body: _MjDataBodyViews = data.body('drone')
        self.spawn_box = spawn_box
        self.spawn_max_velocity = spawn_max_velocity
        self.rng = rng
        self.id = self.body.id
    
    @property
    def position(self) -> np.ndarray:
        """
        Gets the drone's position.

        :return: A numpy array representing the drone's current position.
        """
        return self.body.xpos
    
    @position.setter
    def position(self, value: np.ndarray) -> None:
        """
        Sets the drone's position.

        :param value: A numpy array representing the new position for the drone.
        """
        self.body.xpos = value
    
    @property
    def velocity(self) -> np.ndarray:
        """
        Gets the drone's velocity.

        :return: A numpy array representing the drone's current velocity.
        """
        return self.body.cvel[:3]
    
    @velocity.setter
    def velocity(self, value: np.ndarray) -> None:
        """
        Sets the drone's velocity.

        :param value: A numpy array representing the new velocity for the drone.
        """
        self.body.cvel[:3] = value
    
    @property
    def imu_accel(self) -> np.ndarray:
        """
        Gets the drone's accelerometer readings from the IMU.

        :return: A numpy array representing the drone's current accelerometer readings.
        """
        return self.data.sensor('imu_accel').data
    
    @property
    def imu_gyro(self) -> np.ndarray:
        """
        Gets the drone's gyroscope readings from the IMU.

        :return: A numpy array representing the drone's current gyroscope readings.
        """
        return self.data.sensor('imu_gyro').data
    
    @property
    def imu_orientation(self) -> np.ndarray:
        """
        Gets the drone's orientation readings from the IMU.

        :return: A numpy array representing the drone's current orientation.
        """
        return self.data.sensor('imu_orientation').data
    
    def reset(self):
        """
        Resets the drone's position and velocity to initial values within the defined spawn box and velocity limits.
        """
        self.position = self.spawn_box[0] + (self.spawn_box[1] - self.spawn_box[0]) * self.rng.random(3)
        self.velocity = self.spawn_max_velocity * self.rng.random(3)


class Target:
    """
    Represents a target in the simulation environment. It manages the target's state, including its position,
    velocity, and orientation.

    :param data: The MjData instance containing the simulation state.
    :param spawn_box: A numpy array defining the boundaries for the target's initial position.
    :param spawn_max_velocity: The maximum initial velocity of the target.
    :param spawn_max_angular_velocity: The maximum initial angular velocity of the target.
    :param rng: An instance of a random number generator (optional).
    """
    def __init__(self,
                 data: MjData,
                 spawn_box: np.ndarray,
                 spawn_max_velocity: float,
                 spawn_max_angular_velocity: float,
                 rng: np.random.Generator = default_rng()
                 ):
        """
        Initializes the Target instance with the provided parameters.

        :param data: The MjData instance from the simulation environment.
        :param spawn_box: A 2x3 numpy array defining the min and max coordinates for spawning the target.
        :param spawn_max_velocity: The maximum magnitude of the target's initial velocity vector.
        :param spawn_max_angular_velocity: The maximum magnitude of the target's initial angular velocity vector.
        :param rng: A numpy random number generator instance for stochasticity in initial conditions.
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
        """
        Gets the target's position.

        :return: A numpy array representing the target's current position.
        """
        return self.body.xpos
    
    @position.setter
    def position(self, value: np.ndarray) -> None:
        """
        Sets the target's position.

        :param value: A numpy array representing the new position for the target.
        """
        self.body.xpos = value
    
    @property
    def velocity(self) -> np.ndarray:
        """
        Gets the target's velocity.

        :return: A numpy array representing the target's current velocity.
        """
        return self.body.cvel[:3]
    
    @velocity.setter
    def velocity(self, value: np.ndarray) -> None:
        """
        Sets the target's velocity.

        :param value: A numpy array representing the new velocity for the target.
        """
        self.body.cvel[:3] = value
    
    @property
    def orientation(self) -> np.ndarray:
        """
        Gets the target's orientation.

        :return: A numpy array representing the target's current orientation.
        """
        return self.body.qpos[3:7]
    
    @orientation.setter
    def orientation(self, value: np.ndarray) -> None:
        """
        Sets the target's orientation.

        :param value: A numpy array representing the new orientation for the target.
        """
        self.body.qpos[3:7] = value
    
    def reset(self):
        """
        Resets the target's position, velocity, and orientation to initial values within the defined spawn box and velocity limits.
        """
        self.position = self.spawn_box[0] + (self.spawn_box[1] - self.spawn_box[0]) * self.rng.random(3)
        self.velocity = self.spawn_max_velocity * self.rng.random(3)


# endregion


# region BaseMujocoEnv
class BaseRegime(MujocoEnv, EzPickle, ABC):
    """
    Base class for creating training regimes in a Mujoco simulation environment. It sets up the environment,
    including the drone and target, and defines the necessary properties and methods that should be implemented
    by subclasses.

    This class should not be instantiated directly but extended by subclasses to define specific training regimes.

    :param kwargs: Keyword arguments for environment configuration.
    """
    # region Initialization
    def __init__(self, **kwargs):
        """
        Initialize the BaseRegime environment with configuration parameters.

        :keyword xml_path: Path to the XML model file. It can be a string indicating a predefined model ("Main" or "Reduced")
            or a path to a custom model file. (default: "Reduced")
            Type: str
        :keyword height: The height of the rendered images from the simulation, used in defining the observation space.
            Type: int, default: 480
        :keyword width: The width of the rendered images from the simulation, used in defining the observation space.
            Type: int, default: 640
        :keyword n_camera: Number of cameras/views to be included in the observation space. Each camera adds an image to
            the observation space. Type: int, default: 2
        :keyword frame_skip: Number of frames to skip for each simulation step. A higher frame_skip results in faster
            simulation at the cost of less control precision. Type: int, default: 1
        :keyword dt: The time step for the simulation. It determines the update frequency of the simulation.
            Type: float, default: value from the loaded Mujoco model or 0.01 if not specified
        :keyword drone_spawn_box: A 2x3 array specifying the lower and upper bounds ([min, max]) for the drone's initial
            position. Each row is a 3D coordinate. Type: numpy.ndarray, shape: (2, 3)
        :keyword drone_spawn_max_velocity: Maximum magnitude of the drone's initial velocity. Type: float
        :keyword target_spawn_box: A 2x3 array specifying the lower and upper bounds ([min, max]) for the target's initial
            position. Each row is a 3D coordinate. Type: numpy.ndarray, shape: (2, 3)
        :keyword target_spawn_max_velocity: Maximum magnitude of the target's initial velocity. Type: float
        :keyword target_spawn_max_angular_velocity: Maximum magnitude of the target's initial angular velocity.
            Type: float
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
        self.n_camera = kwargs.get('n_camera', 2)
        observation_space_dict = {
            "imu_accel": Box(low=-np.inf, high=np.inf, shape=(3,)),
            "imu_gyro": Box(low=-np.inf, high=np.inf, shape=(4,)),
            "imu_orientation": Box(low=-np.inf, high=np.inf, shape=(3,)),
        }
        
        for i in range(self.n_camera):
            observation_space_dict[f"image_{i}"] = Box(low=0, high=255, shape=(height, width, 3), dtype=np.uint8)
        
        observation_space: Space[ObsType] = Dict(observation_space_dict)
        
        # if model_path is a valid file, use it. If it is "Reduced", use the reduced model. Else, use the main model.
        model_path = kwargs.get('xml_path', 'Reduced')
        if model_path == "Reduced":
            model_path = XML_PATH_REDUCED
        elif model_path == "Main":
            model_path = XML_PATH_MAIN
        elif not os.path.isfile(model_path):
            raise ValueError(f"Model path {model_path} is not a valid file. "
                             f"Expected a valid file or 'Reduced' or 'Main'.")
        MujocoEnv.__init__(
            self,
            model_path=model_path,
            frame_skip=kwargs.get('frame_skip', 1),
            observation_space=observation_space,
            height=height,
            width=width,
        )
        EzPickle.__init__(
            self,
            XML_PATH_MAIN,
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
    def drone_target_vector(self) -> np.ndarray:
        """
        Compute and return the vector from the drone to the target.

        :return: A numpy array representing the vector from the drone to the target.
        """
        return self.target.position - self.drone.position
    
    @property
    def drone_hit_ground(self) -> bool:
        """
        Check if the drone has hit the ground.

        :return: True if the drone has made contact with the ground, False otherwise.
        """
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
    def pre_simulation(self) -> None:
        """
        Perform any necessary actions before each simulation step. This method should be overridden by subclasses.
        """
        pass
    
    def reset_model(self) -> ObsType:
        """
        Reset the environment to an initial state and return the initial observation.

        :return: The initial observation after resetting the environment.
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
        Execute one time step within the environment.

        :param action: The action to be executed.
        :return: A tuple containing the new observation, reward, done flag, truncated flag, and info dictionary.
        """
        self.pre_simulation()
        self.do_simulation(action, self.frame_skip)
        return (self.observation, self.reward,
                self.truncated, self.done, self.metrics)
    
    @property
    def observation(self) -> ObsType:
        """
        Get the current observation of the environment.

        :return: The current environment observation.
        """
        observation = {
            "acceleration": self.drone.imu_accel,
            "gyro": self.drone.imu_gyro,
            "orientation": self.drone.imu_orientation
        }
        for i in range(self.n_camera):
            observation[f"image_{i}"] = self.mujoco_renderer.render('rgb_array', i)
        return observation
    
    @property
    @abstractmethod
    def reward(self) -> SupportsFloat:
        """
        Calculate and return the current reward. This method should be implemented by subclasses.

        :return: The current reward.
        """
        raise NotImplementedError
    
    @property
    @abstractmethod
    def done(self) -> bool:
        """
        Determine whether the episode is done. This method should be implemented by subclasses.

        :return: True if the episode is finished, False otherwise.
        """
        raise NotImplementedError
    
    @property
    @abstractmethod
    def truncated(self) -> bool:
        """
        Determine whether the episode is truncated. This method should be implemented by subclasses.

        :return: True if the episode is truncated, False otherwise.
        """
        raise NotImplementedError
    
    @property
    @abstractmethod
    def metrics(self) -> dict[str, Any]:
        """
        Return additional data or metrics for logging purposes. This method should be implemented by subclasses.

        :return: A dictionary containing metrics or additional information.
        """
        raise NotImplementedError
    
    # endregion


# endregion

class _TestRegime(BaseRegime):
    """
    A test regime class for demonstration purposes, extending the BaseRegime class.

    This class implements the abstract methods of BaseRegime with basic or dummy functionality.
    """
    def __init__(self, **kwargs):
        """
        Initialize the _TestRegime environment.

        :param kwargs: Configuration parameters for the environment.
        """
        super().__init__(**kwargs)
    
    def pre_simulation(self) -> None:
        """
        Perform any necessary actions before each simulation step. Overridden to set control to zero.
        """
        self.data.ctrl[:] = 0
    
    @property
    def reward(self) -> SupportsFloat:
        """
        Calculate and return the current reward. Overridden to return a fixed reward of 0.

        :return: The current reward, which is 0 in this test regime.
        """
        return 0
    
    @property
    def done(self) -> bool:
        """
        Determine whether the episode is done. Overridden to always return False.

        :return: False, indicating the episode is never considered done in this test regime.
        """
        return False
    
    @property
    def truncated(self) -> bool:
        """
        Determine whether the episode is truncated. Overridden to always return False.

        :return: False, indicating the episode is never considered truncated in this test regime.
        """
        return False
    
    @property
    def metrics(self) -> dict[str, Any]:
        """
        Return additional data or metrics for logging purposes. Overridden to return an empty dictionary.

        :return: An empty dictionary, as no additional metrics are provided in this test regime.
        """
        return {}




