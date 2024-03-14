from typing import List, Union, Dict

import numpy as np
import quaternion
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from gymnasium.spaces import Box, MultiBinary, Dict as SpaceDict
from mujoco import MjModel, MjData
from mujoco._structs import _MjDataBodyViews, _MjDataGeomViews, _MjDataSiteViews, _MjDataActuatorViews
from numpy.random import Generator, default_rng
from ray.rllib.env.multi_agent_env import AgentID
from ray.rllib.utils.typing import EnvObsType, EnvActionType


class Bullet:
    """
    Represents a bullet within a MuJoCo simulation environment.

    Attributes:
        model (MjModel): The MuJoCo model of the environment.
        data (MjData): The MuJoCo data structure with the simulation's current state.
        index (int): The index of the drone associated with this bullet.
        shoot_velocity (float): The initial velocity when the bullet is shot.
        x_bounds (np.ndarray): The environment's boundaries along the x-axis.
        y_bounds (np.ndarray): The environment's boundaries along the y-axis.
        z_bounds (np.ndarray): The environment's boundaries along the z-axis.
    """
    
    def __init__(self, model: MjModel, data: MjData, index: int, shoot_velocity: float,
                 x_bounds: np.ndarray, y_bounds: np.ndarray, z_bounds: np.ndarray) -> None:
        """
        Initialize the Bullet object with the necessary parameters.

        Parameters:
            model (MjModel): The Mujoco model of the environment.
            data (MjData): The Mujoco data of the environment.
            index (int): Index of the drone associated with this bullet.
            shoot_velocity (float): Initial velocity of the bullet when fired.
            x_bounds (np.ndarray): Boundary limits of the environment along the x-axis.
            y_bounds (np.ndarray): Boundary limits of the environment along the y-axis.
            z_bounds (np.ndarray): Boundary limits of the environment along the z-axis.
        """
        # Basic attributes
        self.model = model
        self.data = data
        self.index = index
        self.shoot_velocity = shoot_velocity
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.z_bounds = z_bounds
        
        # Getting the specific bullet body views from the data based on the drone index
        self.bullet_body: _MjDataBodyViews = self.data.body(f"drone{self.index}_bullet")
        
        # Identifying the end of the gun barrel to calculate the shooting direction
        self.barrel_end: _MjDataSiteViews = self.data.site(f"drone{self.index}_gun_barrel_end")
        
        # Accessing the drone's body to get its current position and velocity
        self.parent_drone_body: _MjDataBodyViews = self.data.body(f"drone{self.index}")
        
        # Getting bullet geometry for collision detection and other operations
        self.geom: _MjDataGeomViews = self.data.geom(f"drone{self.index}_bullet_geom")
        self._geom_id: int = self.geom.id  # Store the geometry ID for future reference
        
        # Initialize the bullet's starting position
        self.starting_position = self.bullet_body.xpos.copy()
        
        # Set the initial flying status of the bullet to False
        self._is_flying = False
    
    def reset(self) -> None:
        """
        Resets the bullet's position to its starting position, sets its velocity to zero, and marks it as not flying.

        This method is typically called when a bullet goes out of bounds or when initializing the simulation.
        """
        self.bullet_body.xpos = self.starting_position.copy()
        self.bullet_body.cvel[:] = 0
        self._is_flying = False
    
    def shoot(self) -> None:
        """
        Fires the bullet by setting its velocity in the direction of the drone's gun barrel end.

        The bullet is only shot if it is not already flying. This method calculates the bullet's trajectory based
        on the drone's orientation and adds the shoot velocity to the bullet's current velocity.
        """
        if not self._is_flying:
            aim_direction = self.data.site_xmat[self.barrel_end.id].reshape(3, 3)[:, 0]
            bullet_velocity = aim_direction * self.shoot_velocity + self.parent_drone_body.cvel[:3]
            
            self.bullet_body.xpos = self.barrel_end.xpos
            self.bullet_body.cvel[:3] = bullet_velocity
            self._is_flying = True
    
    def update(self) -> None:
        """
        Updates the bullet's flying status and position.

        Checks if the bullet is within the environment bounds. If it's out of bounds, the bullet is reset.
        This method should be called at each time step of the simulation to update the bullet's state.
        """
        if self._is_flying and not (self.x_bounds[0] <= self.bullet_body.xpos[0] <= self.x_bounds[1] and
                                    self.y_bounds[0] <= self.bullet_body.xpos[1] <= self.y_bounds[1] and
                                    self.z_bounds[0] <= self.bullet_body.xpos[2] <= self.z_bounds[1]):
            self.reset()
    
    @property
    def geom_id(self) -> int:
        """
        Gets the geometry ID of the bullet.

        Returns:
            int: The geometry ID, used for collision detection and other simulation interactions.
        """
        return self._geom_id
    
    @property
    def is_flying(self) -> bool:
        """
        Checks if the bullet is currently flying (in motion).

        Returns:
            bool: True if the bullet is flying, False otherwise.
        """
        return self._is_flying


class Drone:
    """
    Represents a drone in a MuJoCo-based simulation environment, capable of shooting bullets and capturing images.

    :param model: The MuJoCo model of the environment.
    :type model: MjModel
    :param data: The MuJoCo data of the environment.
    :type data: MjData
    :param renderer: Renderer for the MuJoCo environment.
    :type renderer: MujocoRenderer
    :param n_images: Number of images the drone should capture.
    :type n_images: int
    :param depth_render: Specifies if depth rendering is used instead of RGB.
    :type depth_render: bool
    :param index: Index of the drone in the environment.
    :type index: int
    :param agent_id: Unique identifier for the agent.
    :type agent_id: AgentID
    :param spawn_box: The 3D area in which the drone can be spawned.
    :type spawn_box: np.ndarray
    :param max_spawn_velocity: Maximum velocity at spawn.
    :type max_spawn_velocity: float
    :param spawn_angle_range: Range of possible spawn angles.
    :type spawn_angle_range: np.ndarray
    :param rng: Random number generator instance.
    :type rng: Generator, optional
    :param map_bounds: Boundaries of the map where the drone operates.
    :type map_bounds: np.ndarray, optional
    :param bullet_max_velocity: Maximum velocity of the bullet.
    :type bullet_max_velocity: float
    :param height: The height of the captured image.
    :type height: int
    :param width: The width of the captured image.
    :type width: int

    The Drone class encapsulates functionalities for drone movement, shooting bullets, and capturing images within a simulation.
    It interacts with the MuJoCo simulation environment and maintains the state and behavior of the drone.
    """
    
    def __init__(self, model: MjModel, data: MjData, renderer: MujocoRenderer, n_images: int,
                 depth_render: bool, index: int, agent_id: AgentID, spawn_box: np.ndarray, max_spawn_velocity: float,
                 spawn_angle_range: np.ndarray, map_bounds: np.ndarray, rng: Generator = default_rng(),
                 bullet_max_velocity: float = 50, height: int = 32, width: int = 32) -> None:
        """
        Initializes the Drone object, setting up its attributes and preparing it for simulation.

        :param model: The MuJoCo model associated with the simulation environment. This model defines the physical properties
                      and configurations of all entities in the simulation, including the drone itself.
        :type model: MjModel

        :param data: The data structure containing the current state of the simulation. It includes information about all
                     entities defined in the model, such as their positions, velocities, and other dynamic properties.
        :type data: MjData

        :param renderer: An instance of the renderer used for generating visualizations of the simulation. This renderer
                         can produce images from the drone's perspective, which can be used for observation or analysis.
        :type renderer: MujocoRenderer

        :param n_images: Specifies the number of images the drone should capture in each simulation step. This can be used
                         to simulate multiple cameras or different types of sensory inputs.
        :type n_images: int

        :param depth_render: A boolean indicating whether the images captured by the drone should be depth maps instead of
                             RGB images. Depth maps can provide valuable information for certain types of analysis or control.
        :type depth_render: bool

        :param index: A unique identifier for the drone within the simulation. This index is used to differentiate between
                      multiple drones or other entities.
        :type index: int

        :param agent_id: A unique identifier used to track the drone within a larger multi-agent simulation or environment.
                         This ID is crucial for coordinating interactions between multiple agents.
        :type agent_id: AgentID

        :param spawn_box: Defines the 3D area within which the drone can be initialized at the start of the simulation. The
                          spawn box is defined by minimum and maximum coordinates in each dimension.
        :type spawn_box: np.ndarray

        :param max_spawn_velocity: The maximum velocity at which the drone can be spawned. This parameter can be used to
                                   introduce variability in initial conditions for the simulation.
        :type max_spawn_velocity: float

        :param spawn_angle_range: Specifies the range of angles at which the drone can be initialized. This helps in setting
                                  up diverse initial orientations for the drone.
        :type spawn_angle_range: np.ndarray

        :param rng: An optional random number generator that can be used for initializing the drone with random positions,
                    velocities, or orientations. Using a consistent RNG can be useful for reproducibility.
        :type rng: Generator, optional

        :param map_bounds: An optional parameter that defines the boundaries of the environment in which the drone operates.
                           These bounds can be used to enforce constraints on the drone's movement.
        :type map_bounds: np.ndarray, optional

        :param bullet_max_velocity: The maximum velocity at which bullets fired by the drone can travel. This setting affects
                                    the physics of bullets within the simulation.
        :type bullet_max_velocity: float

        :param height: The height of the images captured by the drone. This parameter defines the resolution of the visual
                       input received from the drone's perspective.
        :type height: int

        :param width: The width of the images captured by the drone, defining the horizontal resolution of the drone's visual
                      input.
        :type width: int

        This method sets up the drone with the specified parameters, initializing its position, orientation, and other
        properties based on the provided values. It prepares the drone for interacting with its environment and executing
        simulation steps.
        """
        
        # Basic drone attributes
        self.model = model
        self.data = data
        self.index = index
        self.agent_id = agent_id
        self.spawn_box = spawn_box.reshape(2, 3) if spawn_box.shape == (3, 2) else spawn_box
        self.spawn_angle_range = spawn_angle_range.reshape(2, 3) if spawn_angle_range.shape == (
            3, 2) else spawn_angle_range
        self.max_spawn_velocity = max_spawn_velocity
        self.map_bounds = map_bounds
        self.n_images = n_images
        self.depth_render = depth_render
        self.height = height
        self.width = width
        self.rng = rng
        self.starting_position = self.data.body(f"drone{index}").xpos.copy()
        
        # Bullet initialization
        self.bullet = Bullet(self.model, self.data, self.index, bullet_max_velocity, *self.map_bounds)
        
        # Drone status flags
        self.got_hit = False
        self.scored_hit = False
        self.alive = True
        
        # Renderer for image capturing
        self.renderer = renderer
        
        # Camera IDs for image rendering
        self._camera_1_id = self.data.camera(f"drone{index}_camera_1").id
        self._camera_2_id = self.data.camera(f"drone{index}_camera_2").id
        
        # Lazy initialization of image arrays
        self._image_1 = None
        self._image_2 = None
        
        # Drone's physical components in the simulation
        self._body = self.data.body(f"drone{index}")
        self._geom = self.data.geom(f"drone{index}_geom")
        self._gyro = self.data.sensor(f"drone{index}_imu_gyro")
        self._accelerometer = self.data.sensor(f"drone{index}_imu_accel")
        self._frame_quat = self.data.sensor(f"drone{index}_imu_orientation")
        self._actuators: List[_MjDataActuatorViews] = [self.data.actuator(f"drone{index}_motor{i}") for i in
                                                       range(1, 5)]
        
        # Default position when the drone is considered 'dead' or inactive
        self._dead_position = self._body.xpos.copy()
        self._dead_position[2] = -10  # Set a specific Z value to denote 'dead' status
        
        # Store the initial orientation for potential reset purposes
        self.initial_quat = self.frame_quaternion
    
    # region Properties
    @property
    def images(self) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """
        Retrieves the latest captured images from the drone's cameras. The method supports both single and dual camera
        configurations. When two cameras are present, it returns a tuple of images.

        :return: An array representing a single image when one camera is used, or a tuple of arrays representing images from two cameras.
        :rtype: Union[np.ndarray, tuple[np.ndarray, np.ndarray]]
        """
        # Create the image arrays lazily if they don't exist yet
        if self._image_1 is None:
            if self.depth_render:
                self._image_1 = np.ndarray((self.height, self.width), dtype=np.uint8)
            else:
                self._image_1 = np.ndarray((self.height, self.width, 3), dtype=np.uint8)
        
        if self.n_images == 2 and self._image_2 is None:
            if self.depth_render:
                self._image_2 = np.ndarray((self.height, self.width), dtype=np.uint8)
            else:
                self._image_2 = np.ndarray((self.height, self.width, 3), dtype=np.uint8)
        
        # Update and render the scene if needed (logic unchanged)
        self._image_1 = self.renderer.render(
            render_mode="rgb_array" if not self.depth_render else "depth_array",
            camera_id=self._camera_1_id,
        )
        if self.n_images == 1:
            return self._image_1
        
        self._image_2 = self.renderer.render(
            render_mode="rgb_array" if not self.depth_render else "depth_array",
            camera_id=self._camera_2_id,
        )
        return self._image_1, self._image_2
    
    @property
    def in_map_bounds(self) -> bool:
        """
        Checks if the drone is within the predefined map boundaries.

        :return: True if the drone's current position is within the map boundaries, False otherwise.
        :rtype: bool
        """
        return all(self.map_bounds[:, 0] <= self.position) and all(self.position <= self.map_bounds[:, 1])
    
    @property
    def position(self) -> np.ndarray:
        """
        Provides the current position of the drone in the simulation environment.

        :return: The position of the drone as a NumPy array.
        :rtype: np.ndarray
        """
        return self._body.xpos
    
    @property
    def velocity(self) -> np.ndarray:
        """
        Provides the current velocity of the drone.

        :return: The velocity of the drone as a NumPy array.
        :rtype: np.ndarray
        """
        return self._body.cvel
    
    @property
    def acceleration(self) -> np.ndarray:
        """
        Provides the current acceleration of the drone, derived from the accelerometer data.

        :return: The acceleration of the drone as a NumPy array.
        :rtype: np.ndarray
        """
        return self._accelerometer.data
    
    @property
    def orientation(self) -> np.ndarray:
        """
        Provides the current orientation of the drone in quaternion format.

        :return: The orientation of the drone as a NumPy array (quaternion).
        :rtype: np.ndarray
        """
        return self._body.xquat
    
    @property
    def angular_velocity(self) -> np.ndarray:
        """
        Provides the current angular velocity of the drone.

        :return: The angular velocity of the drone as a NumPy array.
        :rtype: np.ndarray
        """
        return self._gyro.data
    
    @property
    def frame_quaternion(self) -> np.ndarray:
        """
        Provides the frame quaternion of the drone, representing its orientation in the simulation.

        :return: The frame quaternion of the drone.
        :rtype: np.ndarray
        """
        return self._frame_quat.data
    
    @property
    def motor_velocity(self):
        """
        Provides the current velocities of the drone's motors.

        :return: An array of motor velocities.
        :rtype: np.ndarray
        """
        return np.array([actuator.velocity for actuator in self._actuators])
    
    @property
    def motor_torque(self):
        """
        Provides the current torques of the drone's motors.

        :return: An array of motor torques.
        :rtype: np.ndarray
        """
        return np.array([actuator.moment for actuator in self._actuators])
    
    @property
    def motor_controls(self) -> np.ndarray:
        """
        Provides the current control inputs for the drone's motors.

        :return: An array of control inputs for each motor.
        :rtype: np.ndarray
        """
        return np.array([actuator.ctrl for actuator in self._actuators])
    
    @property
    def body(self) -> _MjDataBodyViews:
        """
        Provides access to the drone's body data within the simulation.

        :return: The body data of the drone.
        :rtype: _MjDataBodyViews
        """
        return self._body
    
    @property
    def geom(self) -> _MjDataGeomViews:
        """
        Provides access to the drone's geometry data within the simulation.

        :return: The geometry data of the drone.
        :rtype: _MjDataGeomViews
        """
        return self._geom
    
    # endregion
    
    # region Reset, Act, Reward, Update
    def reset(self):
        """
        Resets the drone's state to its initial conditions, including position, orientation, and velocity.
        """
        pos = self.rng.uniform(self.spawn_box[0], self.spawn_box[1]) + self.starting_position
        vel = self.rng.uniform(-self.max_spawn_velocity, self.max_spawn_velocity, size=3)
        yaw_pitch_roll = self.rng.uniform(self.spawn_angle_range[0], self.spawn_angle_range[1])
        quat = quaternion.from_euler_angles(yaw_pitch_roll)
        self._body.xpos = pos
        self._body.cvel[:3] = vel
        self._body.xquat = np.array([quat.w, quat.x, quat.y, quat.z])
        self.initial_quat = self.frame_quaternion
        self.bullet.reset()
        return self.observation(render=self.n_images > 0)
    
    def observation(self, render: bool = False) -> EnvObsType:
        """
        Generates an observation of the drone's current state, including its position, velocity, orientation, and images
            captured by its cameras.

        :param render: Whether to render the images from the drone's cameras.
        :type render: bool
        :return: A dictionary containing various observations of the drone's state.
        :rtype: Dict
        """
        observations = {
            "position": self.position,
            "velocity": self.velocity,
            "acceleration": self.acceleration,
            "orientation": self.orientation,
            "angular_velocity": self.angular_velocity,
            "frame_quaternion": self.frame_quaternion
        }
        if self.n_images == 1:
            observations["image"] = self.images if render else self._image_1
        elif self.n_images == 2:
            observations["images"] = self.images if render else (self._image_1, self._image_2)
        return observations
    
    @property
    def observation_space(self) -> EnvObsType:
        """
        Defines the observation space for the drone, including the position, velocity, orientation, and images captured
        by its cameras.

        :return: A dictionary specifying the observation space for the drone.
        :rtype: Dict
        """
        obs_space = {
            "position": Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "velocity": Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "acceleration": Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "orientation": Box(low=-1, high=1, shape=(4,), dtype=np.float32),
            "angular_velocity": Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "frame_quaternion": Box(low=-1, high=1, shape=(4,), dtype=np.float32),
        }
        if self.n_images >= 1:
            obs_space["image_1"] = Box(low=0, high=255,
                                       shape=(self.height, self.width, 3 if not self.depth_render else 1),
                                       dtype=np.uint8)
        if self.n_images == 2:
            obs_space["image_2"] = Box(low=0, high=255,
                                       shape=(self.height, self.width, 3 if not self.depth_render else 1),
                                       dtype=np.uint8)
        return SpaceDict(obs_space)
    
    def act(self, action: EnvActionType):
        """
        Applies the specified action to the drone, affecting its motor controls and potentially firing a bullet.

        :param action: The control inputs for the drone's motors and a flag indicating whether to shoot a bullet.
        :type action: np.ndarray
        """
        shoot = bool(action["shoot"][0])
        motor_controls = action["motor"]
        for actuator, value in zip(self._actuators, motor_controls):
            actuator.ctrl = value
        if shoot:
            self.bullet.shoot()
    
    @property
    def action_space(self) -> EnvActionType:
        """
        Defines the action space for the drone, including motor controls and a flag for shooting a bullet.

        :return: A dictionary specifying the action space for the drone.
        :rtype: Dict
        """
        low, high = self.model.actuator_ctrlrange.T[:, 0]
        return SpaceDict({
            "motor": Box(low=low, high=high, shape=(4,), dtype=np.float32),
            "shoot": MultiBinary(n=1)
        })
    
    @property
    def reward(self) -> float:
        return 0.0
    
    def dead_update(self):
        """
        Defines the behavior of the drone when it is considered 'dead' or inactive in the simulation.
        """
        pass
    
    def update(self):
        """
        Updates the drone's state based on the current simulation state, including checking bullet status and drone aliveness.
        """
        self.bullet.update()
        if not self.alive:
            self.dead_update()
    # endregion
