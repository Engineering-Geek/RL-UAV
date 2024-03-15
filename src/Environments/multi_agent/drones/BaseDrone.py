"""
This module defines the Bullet and BaseDrone classes for a MuJoCo-based simulation environment. The Bullet class
represents a projectile fired by a drone, while BaseDrone is an abstract base class that outlines the structure and
capabilities of a drone within the simulation, including movement, image capture, and interaction with other entities.

Classes:
    Bullet: Defines the properties and behaviors of a bullet within the simulation.
    BaseDrone: An abstract class that specifies the required attributes and methods for a drone in the environment.
"""

from __future__ import annotations

from abc import abstractmethod, ABC
from typing import List, Union, Dict

import numpy as np
import quaternion
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from gymnasium.spaces import Box, MultiBinary, Dict as SpaceDict
from mujoco import MjModel, MjData
from mujoco._structs import _MjDataBodyViews, _MjDataGeomViews, _MjDataSiteViews, _MjDataActuatorViews, \
    _MjDataSensorViews, _MjDataJointViews, _MjModelBodyViews, _MjModelGeomViews
from numpy.random import Generator, default_rng
from ray.rllib.env.multi_agent_env import AgentID
from ray.rllib.utils.typing import EnvObsType, EnvActionType


class Bullet:
    """
    Represents a bullet within a MuJoCo simulation environment, capable of being fired and interacting with other entities.

    Attributes:
        parent (BaseDrone): The drone that fired the bullet.
        shoot_velocity (float): The initial velocity with which the bullet is fired.
        bullet_body_data (_MjDataBodyViews): Reference to the bullet's body in the simulation.
        barrel_end (_MjDataSiteViews): Reference to the drone's barrel end site to determine the firing direction.
        body_free_joint (_MjDataJointViews): Joint associated with the bullet for positional updates.
        bullet_geom_data (_MjDataGeomViews): Geometry of the bullet for collision detection.
        _geom_id (int): Internal ID of the bullet's geometry.
        starting_position (np.ndarray): The position from which the bullet is fired.
        is_flying (bool): Flag indicating whether the bullet is currently in motion.
    """
    
    def __init__(self, parent: BaseDrone, shoot_velocity: float) -> None:
        """
        Initializes a Bullet instance with the given parent drone and shoot velocity.

        Parameters:
            parent (BaseDrone): The drone associated with this bullet.
            shoot_velocity (float): The velocity at which the bullet is fired.
        """
        # Basic attributes
        self.shoot_velocity = shoot_velocity
        self.parent = parent
        
        # Getting the specific bullet body views from the data based on the drone index
        self.bullet_body_data: _MjDataBodyViews = self.parent.data.body(f"drone{self.parent.index}_bullet")
        self.bullet_body_model: _MjModelBodyViews = self.parent.model.body(f"drone{self.parent.index}_bullet")
        
        # Identifying the end of the gun barrel to calculate the shooting direction
        self.barrel_end: _MjDataSiteViews = self.parent.data.site(f"drone{self.parent.index}_gun_barrel_end")
        
        # Accessing the drone's joint to set the bullet's position
        self.body_free_joint: _MjDataJointViews = self.parent.data.joint(f"drone{self.parent.index}_bullet_joint")
        
        # Getting bullet geometry for collision detection and other operations
        self.bullet_geom_data: _MjDataGeomViews = self.parent.data.geom(f"drone{self.parent.index}_bullet_geom")
        self.bullet_geom_model: _MjModelGeomViews = self.parent.model.geom(f"drone{self.parent.index}_bullet_geom")
        self._geom_id: int = self.bullet_geom_data.id  # Store the geometry ID for future reference
        
        # Initialize the bullet's starting position
        self.starting_position = self.body_free_joint.qpos.copy()
        
    @property
    def is_flying(self) -> bool:
        """
        Checks whether the bullet is currently in motion.

        Returns:
            bool: True if the bullet is flying, otherwise False.
        """
        return bool(self.bullet_geom_model.conaffinity)
    
    def reset(self) -> None:
        """
        Resets the bullet to its initial state, typically called when the bullet goes out of bounds or the simulation
            resets.
        """
        self.starting_position = self.starting_position.copy()
        self.bullet_body_data.cvel[:] = 0
        self.bullet_geom_model.conaffinity = 0
        self.bullet_geom_model.contype = 0
        self.bullet_geom_model.rgba[-1] = 0
    
    def shoot(self) -> None:
        """
        Fires the bullet, calculating its trajectory based on the drone's orientation and updating its velocity.
        """
        
        if not self.is_flying:
            aim_direction = self.parent.body.xmat.reshape(3, 3) @ np.array([1, 0, 0])
            bullet_velocity = aim_direction * self.shoot_velocity + self.parent.body.cvel[3:]
            self.body_free_joint.qpos[:3] = self.barrel_end.xpos
            self.body_free_joint.qvel[:3] = bullet_velocity
            self.bullet_geom_model.conaffinity = 1
            self.bullet_geom_model.contype = 1
            self.bullet_geom_model.rgba[-1] = 1
    
    @property
    def out_of_bounds(self) -> bool:
        """
        Checks whether the bullet has left the designated simulation area.

        Returns:
            bool: True if the bullet is out of bounds, otherwise False.
        """
        x_bounds, y_bounds, z_bounds = self.parent.map_bounds
        if (x_bounds[0] <= self.bullet_body_data.xpos[0] <= x_bounds[1] and
                y_bounds[0] <= self.bullet_body_data.xpos[1] <= y_bounds[1] and
                z_bounds[0] <= self.bullet_body_data.xpos[2] <= z_bounds[1]):
            return True
        else:
            return False
    
    @property
    def geom_id(self) -> int:
        """
        Retrieves the geometry ID of the bullet.

        Returns:
            int: The geometry ID of the bullet.
        """
        return self._geom_id


class BaseDrone(ABC):
    """
    Abstract base class for a drone in a MuJoCo-based simulation, specifying required methods and attributes.

    Subclasses must implement various methods to define the drone's behavior, including movement, image capture,
    and interaction with the simulation environment.
    
    # Execution order in step in MultiAgentDroneEnvironment:
        1. update the environment (env_update)
        2. step the environment
        3. check for any collisions (bullet, drone, floor) and call the appropriate methods (got_hit, scored_hit, hit_floor, bullet_hit_floor) for each Drone
        4. check if the drone is out of bounds (out_of_bounds, bullet_out_of_bounds)
        5. reset parameters if necessary; very useful for sparse rewards
    
    
    # Implementing Sparse Rewards and Penalties:
        1. Declare a boolean flag for when the drone should incur a penalty or reward with a default value of False
        2. Declare the functions or reward values for the penalty or reward
        3. Switch the flag in the appropriate methods (got_hit, scored_hit, hit_floor, bullet_hit_floor, out_of_bounds, bullet_out_of_bounds)
        4. In the reward method, check the flag and return the appropriate reward or penalty function (or value)
        5. Reset the flag in the reset_params method, which is called at the beginning of each step
        
    # Implementing Continuous Rewards and Penalties:
        1. Declare a variable to store the reward or penalty function (or value)
        2. Calculate the reward or penalty in the reward method
    
    View the SimpleDrone class for an example implementation of the above steps (src/Environments/multi_agent/drones/SimpleDrone.py).
    
    Attributes:
        just_shot (bool): A flag indicating whether the drone has just fired a bullet.
        model (MjModel): The MuJoCo model associated with the simulation environment.
        data (MjData): The data structure containing the current state of the simulation.
        index (int): A unique identifier for the drone within the simulation.
        agent_id (AgentID): A unique identifier used to track the drone within a larger multi-agent simulation or environment.
        spawn_box (np.ndarray): The 3D area within which the drone can be initialized at the start of the simulation.
        spawn_angle_range (np.ndarray): The range of angles at which the drone can be initialized.
        max_spawn_velocity (float): The maximum velocity at which the drone can be spawned.
        map_bounds (np.ndarray): The boundaries of the environment in which the drone operates.
        n_images (int): The number of images the drone should capture in each simulation step.
        depth_render (bool): A boolean indicating whether the images captured by the drone should be depth maps instead of RGB images.
        height (int): The height of the images captured by the drone.
        width (int): The width of the images captured by the drone.
        fire_threshold (float): The threshold for firing a bullet.
        rng (Generator): An optional random number generator that can be used for initializing the drone with random positions, velocities, or orientations.
        starting_position (np.ndarray): The position from which the drone was spawned.
        max_time (float): The maximum time for an episode (in seconds).
        bullet (Bullet): The bullet fired by the drone.
        alive (bool): A flag indicating whether the drone is currently active or alive.
        renderer (MujocoRenderer): An instance of the renderer used for generating visualizations of the simulation.
        _camera_1_id (int): The ID of the first camera used by the drone to capture images.
        _camera_2_id (int): The ID of the second camera used by the drone to capture images.
        _image_1 (np.ndarray): An array representing the image captured by the first camera.
        _image_2 (np.ndarray): An array representing the image captured by the second camera.
        _body (_MjDataBodyViews): The drone's body within the simulation.
        _free_joint (_MjDataJointViews): The free joint associated with the drone.
        _geom (_MjDataGeomViews): The geometry of the drone within the simulation.
        _gyro (_MjDataSensorViews): The gyroscope sensor associated with the drone.
        _accelerometer (_MjDataSensorViews): The accelerometer sensor associated with the drone.
        _frame_quat (_MjDataSensorViews): The sensor providing the drone's orientation in quaternion format.
        _floor_geom_id (int): The ID of the floor geometry within the simulation.
        _actuators (List[_MjDataActuatorViews]): A list of actuators associated with the drone.
        _dead_position (np.ndarray): The default position when the drone is considered 'dead' or inactive.
        initial_quat (np.ndarray): The initial orientation of the drone.
    

    Abstract Methods to Implement:
        - reset: Resets the drone's state to its initial conditions.
        - observation: Generates an observation of the drone's current state.
        - act: Applies an action to the drone, influencing its motors and potentially other actuators.
        - env_update: Updates the environment based on the drone's current state.
        - reward: Calculates the reward or penalty for the drone based on its current state.
        - reset_params: Resets the drone's reward flags and other parameters.
        - got_hit: A method to call when the drone gets hit by a bullet.
        - hit_floor: A method to call when the drone hits the floor.
        - on_bullet_out_of_bounds: A method to call when the bullet goes out of bounds.
        - on_out_of_bounds: A method to call when the drone goes out of bounds.
        - scored_hit: A method to call when the drone scores a hit on a target.
    """
    
    def __init__(self, model: MjModel, data: MjData, renderer: MujocoRenderer, n_images: int,
                 depth_render: bool, index: int, agent_id: AgentID, spawn_box: np.ndarray, max_spawn_velocity: float,
                 spawn_angle_range: np.ndarray, map_bounds: np.ndarray, rng: Generator = default_rng(),
                 bullet_max_velocity: float = 50, height: int = 32, width: int = 32,
                 fire_threshold: float = 0.9, max_time: float = 30) -> None:
        """
        Initializes a BaseDrone instance with the specified parameters,
            setting up its simulation environment and attributes.

        :param model: The MuJoCo model associated with the simulation environment. This model defines the physical
                      properties and configurations of all entities in the simulation, including the drone itself.
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
        
        :param fire_threshold: The threshold for firing a bullet.
        :type fire_threshold: float
        
        :param max_time: The maximum time for an episode (in seconds).
        :type max_time: float

        This method sets up the drone with the specified parameters, initializing its position, orientation, and other
        properties based on the provided values. It prepares the drone for interacting with its environment and executing
        simulation steps.
        """
        
        # Basic drone attributes
        self.just_shot = False
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
        self.fire_threshold = fire_threshold
        self.rng = rng
        self.starting_position = self.data.body(f"drone{index}").xpos.copy()
        self.max_time = max_time
        
        # Bullet initialization
        self.bullet = Bullet(self, bullet_max_velocity)
        
        # Drone status flags
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
        self._body: _MjDataBodyViews = self.data.body(f"drone{index}")
        self._free_joint: _MjDataJointViews = self.data.joint(f"drone{index}_free_joint")
        self._geom: _MjDataGeomViews = self.data.geom(f"drone{index}_geom")
        self._gyro: _MjDataSensorViews = self.data.sensor(f"drone{index}_imu_gyro")
        self._accelerometer: _MjDataSensorViews = self.data.sensor(f"drone{index}_imu_accel")
        self._frame_quat: _MjDataSensorViews = self.data.sensor(f"drone{index}_imu_orientation")
        self._floor_geom_id: int = self.data.geom(f"floor").id
        self._actuators: List[_MjDataActuatorViews] = [self.data.actuator(f"drone{index}_motor{i}") for i in
                                                       range(1, 5)]
        self._prop_geoms: List[_MjDataGeomViews] = [self.data.geom(f"drone{index}_prop{i}_geom") for i in range(1, 5)]
        
        self.contact_geom_ids = [self.geom.id] + [prop_geom.id for prop_geom in self._prop_geoms]
        
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

        :return: An array representing a single image when one camera is used, or a tuple of arrays representing images
            from two cameras.
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
    def in_bounds(self) -> bool:
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
        Resets the drone's state to its initial conditions. This method is typically called at the beginning of an
        episode or after the drone has finished a task or mission. The reset process involves setting the drone's
        position, velocity, orientation, and other relevant states to their initial values or randomly generating
        them based on specified ranges.
        """
        pos = self.rng.uniform(self.spawn_box[0], self.spawn_box[1]) + self.starting_position
        vel = self.rng.uniform(-self.max_spawn_velocity, self.max_spawn_velocity, size=3)
        yaw_pitch_roll = self.rng.uniform(self.spawn_angle_range[0], self.spawn_angle_range[1])
        quat = quaternion.from_euler_angles(yaw_pitch_roll)
        self._free_joint.qpos[:3] = pos
        self._free_joint.qvel[:3] = vel
        self._body.xquat = np.array([quat.w, quat.x, quat.y, quat.z])
        self.initial_quat = self.frame_quaternion
        self.bullet.reset()
        self.reset_params()
        return self.observation(render=self.n_images > 0)
    
    def observation(self, render: bool = False) -> EnvObsType:
        """
        Generates an observation of the drone's current state, which typically includes its position, velocity,
        orientation, and images captured from its cameras. This method forms the basis for how the drone perceives
        its environment, providing essential data for decision-making or control algorithms.

        Parameters:
            render (bool): If True, the drone's cameras will capture new images to be included in the observation.

        Returns:
            EnvObsType: A structured observation of the drone's current state, which may include numerical data
                        and images.
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
        Applies an action to the drone, influencing its motors and potentially other actuators like firing mechanisms.
        This method is how the drone interacts with the environment based on decisions made by a controller or agent.

        Parameters:
            action (EnvActionType): A structured action that includes control signals for the drone's actuators.
        """
        shoot = action["shoot"][0]
        shoot = shoot > self.fire_threshold
        motor_controls = action["motor"]
        for actuator, control in zip(self._actuators, motor_controls):
            actuator.ctrl = control
        if shoot and not self.bullet.is_flying:
            self.bullet.shoot()
            self.just_shot = True
    
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
    # endregion
    
    # region Abstract Methods
    @property
    @abstractmethod
    def reward(self) -> float:
        """
        Calculates and returns the reward for the drone's current state and recent actions within the environment.

        Returns:
            float: The calculated reward value.
        """
        raise NotImplementedError
    
    @abstractmethod
    def env_update(self):
        """
        Updates the environment based on the drone's actions and interactions. This method should encompass changes
        in the simulation state that result from the drone's behavior, such as movements, collisions, or other effects.
        """
        raise NotImplementedError
    
    @abstractmethod
    def got_hit(self):
        """
        Handles the drone's response to being hit, such as by a projectile or another drone. This method defines the
        logic for updating the drone's state or the simulation environment in response to the hit.
        """
        raise NotImplementedError
    
    @abstractmethod
    def scored_hit(self):
        """
        Handles the event where the drone successfully hits a target. This method should define the logic for updating
        the drone's state or the simulation environment based on the successful hit.
        """
        raise NotImplementedError
    
    @abstractmethod
    def hit_floor(self):
        """
        Handles the drone's collision with the floor. This method should contain logic to respond to such an event,
        potentially affecting the drone's state or the simulation outcome.
        """
        raise NotImplementedError
    
    @abstractmethod
    def on_out_of_bounds(self):
        """
        Handles the drone going out of bounds, which may involve resetting its state or setting rewards and penalties.
        """
        raise NotImplementedError
    
    @abstractmethod
    def on_bullet_out_of_bounds(self):
        """
        Handles the event where a bullet fired by the drone hits the floor. This method should define the logic for
        how the drone or the simulation responds to this event.
        """
        raise NotImplementedError
    
    @abstractmethod
    def reset_params(self):
        """
        Resets the drone's internal parameters to their initial values. This method is typically called when the drone
        is reset at the beginning of an episode or after finishing a task. Meant for subclass-specific parameters.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def truncated(self) -> bool:
        """
        Determines whether the current drone's episode should be truncated. This could be based on various factors, such
        as the drone's state, elapsed time, or other conditions within the simulation.

        Returns:
            bool: True if the episode should be truncated, False otherwise.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def done(self) -> bool:
        """
        Determines whether the current episode is completed. This status can influence the simulation's control flow,
        determining whether to reset the environment or continue running.

        Returns:
            bool: True if the episode is done, False otherwise.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def info(self) -> Dict[str, float]:
        """
        Provides a set of metrics for monitoring the drone's performance and state within the simulation. These metrics
        can be used for analysis, debugging, or guiding the training process of learning algorithms.

        Returns:
            Dict[str, float]: A dictionary of metric names and their corresponding values.
        """
        raise NotImplementedError
    
    # endregion
    @property
    def prop_geoms(self):
        return self._prop_geoms
