from typing import List, Union, Optional, Tuple

import numpy as np
import quaternion
from gymnasium.core import ActType
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from gymnasium.spaces import Box, MultiBinary
from gymnasium.utils import EzPickle
from mujoco import MjModel, MjData
from mujoco._structs import _MjDataBodyViews, _MjContactList, _MjDataGeomViews
from numpy.random import Generator, default_rng
from ray.rllib.env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict, EnvObsType, EnvActionType, AgentID

from src.utils.multiagent_model_generator import save_multiagent_model


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
        self._bullet_body = self.data.body(f"drone{self.index}_bullet")
        
        # Identifying the end of the gun barrel to calculate the shooting direction
        self._barrel_end = self.data.site(f"drone{self.index}_gun_barrel_end")
        
        # Accessing the drone's body to get its current position and velocity
        self._parent_drone_body = self.data.body(f"drone{self.index}")
        
        # Getting bullet geometry for collision detection and other operations
        self._geom = self.data.geom(f"drone{self.index}_bullet_geom")
        self._geom_id = self._geom.id  # Store the geometry ID for future reference
        
        # Initialize the bullet's starting position
        self.starting_position = self._bullet_body.xpos.copy()
        
        # Set the initial flying status of the bullet to False
        self._is_flying = False
    
    def reset(self) -> None:
        """
        Resets the bullet's position to its starting position, sets its velocity to zero, and marks it as not flying.

        This method is typically called when a bullet goes out of bounds or when initializing the simulation.
        """
        self._bullet_body.xpos = self.starting_position.copy()
        self._bullet_body.cvel[:] = 0
        self._is_flying = False
    
    def shoot(self) -> None:
        """
        Fires the bullet by setting its velocity in the direction of the drone's gun barrel end.

        The bullet is only shot if it is not already flying. This method calculates the bullet's trajectory based
        on the drone's orientation and adds the shoot velocity to the bullet's current velocity.
        """
        if not self._is_flying:
            aim_direction = self.data.site_xmat[self._barrel_end.id].reshape(3, 3)[:, 0]
            bullet_velocity = aim_direction * self.shoot_velocity + self._parent_drone_body.cvel
            
            self._bullet_body.xpos = self._barrel_end.xpos
            self._bullet_body.cvel = bullet_velocity
            self._is_flying = True
    
    def update(self) -> None:
        """
        Updates the bullet's flying status and position.

        Checks if the bullet is within the environment bounds. If it's out of bounds, the bullet is reset.
        This method should be called at each time step of the simulation to update the bullet's state.
        """
        if self._is_flying and not (self.x_bounds[0] <= self._bullet_body.xpos[0] <= self.x_bounds[1] and
                                    self.y_bounds[0] <= self._bullet_body.xpos[1] <= self.y_bounds[1] and
                                    self.z_bounds[0] <= self._bullet_body.xpos[2] <= self.z_bounds[1]):
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
                 spawn_angle_range: np.ndarray, rng: Generator = default_rng(), map_bounds: np.ndarray = None,
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
        self.spawn_box = spawn_box.reshape(2, 3)
        self.spawn_angle_range = spawn_angle_range.reshape(2, 3)
        self.max_spawn_velocity = max_spawn_velocity
        self.map_bounds = map_bounds or np.array([[-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf]])
        self.n_images = n_images
        self.depth_render = depth_render
        self.height = height
        self.width = width
        self.rng = rng
        
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
        self._actuators = [self.data.actuator(f"drone{index}_motor{i}") for i in range(1, 5)]
        
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
    
    # region Spaces
    @property
    def action_space(self) -> EnvActionType:
        shoot_space = MultiBinary(1)
        motor_space = Box(0, 1, shape=(4,))
        return {"shoot": shoot_space, "motor": motor_space}
    
    @property
    def observation_space(self) -> EnvObsType:
        observation_space = {
            "position": Box(-np.inf, np.inf, shape=(3,)),
            "velocity": Box(-np.inf, np.inf, shape=(3,)),
            "acceleration": Box(-np.inf, np.inf, shape=(3,)),
            "orientation": Box(-1, 1, shape=(4,)),
            "angular_velocity": Box(-np.inf, np.inf, shape=(3,)),
            "frame_quaternion": Box(-1, 1, shape=(4,))
        }
        if self.n_images == 1:
            if self.depth_render:
                observation_space["image"] = Box(0, 1, shape=(self.height, self.width))
            else:
                observation_space["image"] = Box(0, 255, shape=(self.height, self.width, 3))
        elif self.n_images == 2:
            # Update this part
            if self.depth_render:
                observation_space["images"] = Box(0, 1, shape=(self.height * 2, self.width))
            else:
                observation_space["images"] = Box(0, 255, shape=(self.height, self.width * 2, 3))
        return observation_space
    
    # endregion
    
    # region Reset, Act, Reward, Update
    def reset(self):
        """
        Resets the drone's state to its initial conditions, including position, orientation, and velocity.
        """
        pos = self.rng.uniform(self.spawn_box[0], self.spawn_box[1])
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
    
    def act(self, action: dict[str, np.ndarray]):
        """
        Applies the specified action to the drone, affecting its motor controls and potentially firing a bullet.

        :param action: A dictionary specifying the motor control values and whether to shoot a bullet.
        :type action: dict[str, np.ndarray]
        """
        shoot = bool(action["shoot"][0])
        motor_controls = action["motor"]
        for actuator, value in zip(self._actuators, motor_controls):
            actuator.ctrl = value
        if shoot:
            self.bullet.shoot()
    
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


class BaseMultiAgentEnvironment(MujocoEnv, EzPickle, MultiAgentEnv):
    """
    A multi-agent environment for drone simulation based on the MuJoCo physics engine.
    It supports rendering, interaction, and control of multiple drones capable of shooting and perception.

    The environment initializes with a specified number of drones, each with its own attributes and capabilities,
    like shooting bullets and capturing images. The environment facilitates the interaction between these drones
    and tracks their states throughout the simulation.
    """
    
    def __init__(self, n_agents: int, n_images: int, depth_render: bool, spawn_boxes: List[np.ndarray],
                 spawn_angles: List[np.ndarray], **kwargs) -> None:
        """
        Initializes a multi-agent environment tailored for drone simulations, using the MuJoCo physics engine.
        This environment supports multiple drones, each with capabilities for shooting and capturing images.
    
        :param int n_agents: The number of drone agents to be initialized in the environment.
        :param int n_images: The number of images each drone should capture per simulation step. It affects the
                             observation space of each drone.
        :param bool depth_render: Determines if the drones should capture depth images (True) or RGB images (False).
        :param list[np.ndarray] spawn_boxes: A list of 2x3 NumPy arrays where each array specifies the min and max
                                             (x, y, z) coordinates of the spawn box for each drone.
        :param list[np.ndarray] spawn_angles: A list of 2x3 NumPy arrays where each array specifies the min and max
                                              (roll, pitch, yaw) angles for the initial orientation of each drone.
        :keyword float spacing: (Optional) Defines the spacing between agents in the environment. Default is 2.0.
        :keyword str save_dir: (Optional) Directory path to save the generated MuJoCo model file. If not specified,
                               the model is not saved.
        :keyword int max_spawn_velocity: (Optional) The maximum magnitude of the initial velocity that can be assigned
                                         to a drone. Default is 1.
        :keyword np.ndarray map_bounds: (Optional) A 2x3 array specifying the min and max (x, y, z) coordinates of the
                                         environment boundaries. Default is a large area around the origin.
        :keyword int fps: (Optional) The number of frames per second at which the simulation should run. This affects
                          the frequency at which the environment's state is rendered and updated. Default is 30.
        :keyword float dt: (Optional) The timestep duration for the simulation in seconds. Default is 0.01.
        :keyword float max_time: (Optional) The maximum duration for the simulation in simulation time units. Default
                                 is 100.
    
        This method sets up the environment with the specified number of agents and configures each agent based on the
        provided parameters and keyword arguments.
        """
        self._n_images = None
        self.i = 0
        self.n_agents = n_agents
        
        # Load or generate a simulation model based on the provided agent configuration.
        self.model_path = save_multiagent_model(n_agents, kwargs.get('spacing', 2.0), kwargs.get('save_dir', None))
        
        # Initialize the MuJoCo environment with the generated model and specified rendering options.
        height, width = kwargs.get('render_height', 32), kwargs.get('render_width', 32)
        MujocoEnv.__init__(self, model_path=self.model_path, frame_skip=kwargs.get('frame_skip', 1),
                           observation_space=None, render_mode=kwargs.get('render_mode', 'rgb_array'),
                           height=height, width=width)
        
        # Update the environment's metadata with rendering details.
        self.metadata['render_fps'] = int(np.round(1.0 / self.dt))
        
        # Instantiate and configure each drone agent within the environment.
        self.drones = [self._create_drone(i, agent_id, n_images, depth_render, spawn_boxes, spawn_angles, **kwargs)
                       for i, agent_id in enumerate(range(n_agents))]
        
        # Setup simulation properties, such as timestep and rendering frequency.
        self._setup_simulation_properties(**kwargs)
    
    def _create_drone(self, index: int, agent_id: int, n_images: int, depth_render: bool, spawn_boxes: List[np.ndarray],
                      spawn_angles: List[np.ndarray], **kwargs) -> Drone:
        """
        Creates a drone object with specified parameters.

        :param index: Index of the drone in the environment.
        :param agent_id: Unique identifier for the agent.
        :param n_images: Number of images the drone captures in each step.
        :param depth_render: Specifies if the drone captures depth images.
        :param spawn_boxes: Bounding boxes for drone spawning locations.
        :param spawn_angles: Initial orientation angle ranges for drones.
        :param kwargs: Additional arguments for drone creation.
        :return: An instance of the Drone class.
        """
        return Drone(model=self.model, data=self.data, renderer=self.mujoco_renderer, n_images=n_images,
                     depth_render=depth_render, index=index, agent_id=agent_id, spawn_box=spawn_boxes[index],
                     max_spawn_velocity=kwargs.get('max_spawn_velocity', 1),
                     spawn_angle_range=spawn_angles[index], rng=kwargs.get('rng', default_rng()),
                     map_bounds=kwargs.get('map_bounds', np.array([[-100, 100], [-100, 100], [0, 100]])),
                     bullet_max_velocity=kwargs.get('bullet_max_velocity', 50))
    
    def _setup_simulation_properties(self, **kwargs):
        """
        Sets up additional properties for the simulation, such as rendering frequency and timestep adjustments.

        :param kwargs: Additional arguments including timestep and frames per second settings.
        """
        _fps = kwargs.get('fps', 30)
        dt = kwargs.get('dt', 0.01)
        time_steps_per_second = int(1.0 / dt)
        self.render_every = time_steps_per_second // _fps
        self.model.opt.timestep = dt
        self.data = MjData(self.model)  # Reinitialize data with the updated model.
        self.i = 0  # Frame counter.
        self.max_time = kwargs.get('max_time', 100)  # Maximum simulation time.
        self._action_space_in_preferred_format = True
        self._observation_space_in_preferred_format = True
        
        # Initialize collision tracking dictionaries.
        self._bullet_geom_ids_to_drones = {}
        self._drone_geom_ids_to_drones = {}
        for drone in self.drones:
            self._bullet_geom_ids_to_drones[drone.bullet.geom_id] = drone
            self._drone_geom_ids_to_drones[drone.geom.id] = drone
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[MultiAgentDict, MultiAgentDict]:
        """
        Resets the environment to an initial state, reinitializing the positions, orientations, and other relevant
        state variables of all drones. This method is typically called at the beginning of an episode.

        :param seed: An optional seed to ensure deterministic reset behavior. Not currently used.
        :type seed: Optional[int]
        :param options: An optional dictionary of reset options. Not currently used but can be extended for future functionalities.
        :type options: Optional[dict]
        :return: A tuple containing the initial observations and metrics after the reset.
        :rtype: Tuple[MultiAgentDict, MultiAgentDict]
        """
        self.i = 0
        self.reset_model()
        return self.observation(False), self.metrics
    
    def reset_model(self):
        """
        Resets the internal state of each drone model. This method is called by the main reset function to reinitialize
        each drone's state within the environment.
        """
        for drone in self.drones:
            drone.reset()
    
    def step(self, action: ActType) \
            -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        """
        Advances the environment by one timestep. Applies the provided actions to the drones, updates their states,
        checks for collisions, and computes the rewards.

        :param action: A dictionary mapping each drone's agent ID to its corresponding action.
        :type action: ActType
        :return: A tuple containing the new observations, rewards, truncation flags, done flags, and additional info for
            each agent.
        :rtype: Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]
        """
        self.i += 1
        for drone in self.drones:
            drone.act(action[drone.agent_id])
            drone.update()
        shooting_drones, hit_drones = self.collisions()
        for drone in shooting_drones:
            drone.scored_hit = True
        for drone in hit_drones:
            drone.got_hit = True
            drone.alive = False
        observation = self.observation(render=self._n_images > 0 and self.i % self.render_every == 0)
        reward = self.reward
        truncated = self.truncated
        done = self.done
        info = self.metrics
        return observation, reward, truncated, done, info
    
    def collisions(self) -> tuple[list[Drone], list[Drone]]:
        """
        Checks for and processes collisions between drones and bullets within the environment.

        :return: A tuple of two lists containing the drones that have shot and been hit, respectively.
        :rtype: tuple[list[Drone], list[Drone]]
        """
        contact_list: _MjContactList = self.data.contact
        if not contact_list.dim:
            return [], []
        contact_geom_pairs: np.ndarray = np.array(contact_list.geom)
        
        # Initialize sets to keep track of shooting and hit drones.
        shooting_drones = set()
        hit_drones = set()
        
        # Loop through each contact pair to detect collisions.
        for geom_1_id, geom_2_id in contact_geom_pairs:
            # Check each pair to see if it involves a bullet that is flying and a drone.
            for bullet_geom_id, drone_geom_id in [(geom_1_id, geom_2_id), (geom_2_id, geom_1_id)]:
                # Check if the bullet is flying and has hit a drone.
                if (bullet_geom_id in self._bullet_geom_ids_to_drones and
                        drone_geom_id in self._drone_geom_ids_to_drones):
                    shooting_drone = self._bullet_geom_ids_to_drones[bullet_geom_id]
                    hit_drone = self._drone_geom_ids_to_drones[drone_geom_id]
                    
                    # Only consider the collision if the bullet is flying.
                    if shooting_drone.bullet.is_flying:
                        shooting_drones.add(shooting_drone)
                        hit_drones.add(hit_drone)
                        break  # Stop checking if we've found a valid collision.
            
            # Check for Drone - Drone Collisions, ensuring we're not considering the bullet twice.
            if geom_1_id in self._drone_geom_ids_to_drones and geom_2_id in self._drone_geom_ids_to_drones:
                hit_drones.add(self._drone_geom_ids_to_drones[geom_1_id])
                hit_drones.add(self._drone_geom_ids_to_drones[geom_2_id])
        
        return list(shooting_drones), list(hit_drones)
    
    def observation(self, render: bool = False) -> MultiAgentDict:
        """
        Constructs the observation for each drone in the environment.

        :param render: Determines whether to render the observation visually.
        :type render: bool
        :return: A dictionary mapping each agent's ID to its observation.
        :rtype: MultiAgentDict
        """
        return dict({
            drone.agent_id: drone.observation(render) for drone in self.drones
        })
    
    @property
    def reward(self) -> MultiAgentDict:
        """
        Computes the reward for each drone based on the current state of the environment.

        :return: A dictionary mapping each agent's ID to its computed reward.
        :rtype: MultiAgentDict
        """
        return dict({
            drone.agent_id: drone.reward for drone in self.drones
        })
    
    @property
    def truncated(self) -> MultiAgentDict:
        """
        Determines whether the current episode should be truncated for each drone.

        :return: A dictionary mapping each agent's ID to a boolean indicating whether its episode should be truncated.
        :rtype: MultiAgentDict
        """
        # Cases for general truncation: time limit reached, alive drones < 2
        if self.data.time >= self.max_time or sum(drone.alive for drone in self.drones) < 2:
            return MultiAgentDict({drone.agent_id: True for drone in self.drones})
        
        # Cases for individual truncation: drone is dead
        truncations = {}
        for drone in self.drones:
            truncations[drone.agent_id] = not drone.alive
    
    @property
    def done(self) -> MultiAgentDict:
        """
        Determines whether the current episode is done for each drone.

        :return: A dictionary mapping each agent's ID to a boolean indicating whether its episode is done.
        :rtype: MultiAgentDict
        """
        only_one_alive = sum(drone.alive for drone in self.drones) == 1
        return {drone.agent_id: only_one_alive for drone in self.drones}
    
    @property
    def metrics(self):
        """
        Collects and returns additional metrics about the environment's current state.

        :return: A dictionary of metrics.
        :rtype: dict
        """
        return {}


def test_multi_agent_env():
    spawn_lower_bound = np.array([-10, -10, 0])
    spawn_upper_bound = np.array([10, 10, 10])
    spawn_lower_angle = np.array([-1, -1, -1])
    spawn_upper_angle = np.array([1, 1, 1])
    env = BaseMultiAgentEnvironment(
        n_agents=2,
        n_images=2,
        depth_render=False,
        spawn_boxes=[np.array([spawn_lower_bound, spawn_upper_bound]) for _ in range(2)],
        spawn_angles=[np.array([spawn_lower_angle, spawn_upper_angle]) for _ in range(2)]
    )
    env.reset()
    drones = env.drones
    # print(obs)
    for i in range(1000):
        action = {
            drone.agent_id: {
                "shoot": np.array([0]),
                "motor": np.array([0.5, 0.5, 0.5, 0.5])
            } for drone in drones
        }
        obs, reward, truncated, done, info = env.step(action)
        print(obs, reward, done, info)
