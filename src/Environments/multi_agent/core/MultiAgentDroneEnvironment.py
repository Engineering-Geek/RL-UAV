"""
Base Multi-Agent Environment for Drone Simulation with MuJoCo
=============================================================

This module provides a base environment class for simulating multi-agent drone interactions
using the MuJoCo physics engine. It facilitates the rendering, interaction, and control of
multiple drones capable of performing actions like shooting and perception within a shared
simulation space.

Classes
-------
BaseMultiAgentEnvironment
    A multi-agent environment for drone simulation based on the MuJoCo physics engine.

Detailed Description
--------------------
The `BaseMultiAgentEnvironment` class inherits from `MujocoEnv`, `EzPickle`, and `MultiAgentEnv`.
It initializes with a specified number of drone agents, each defined by its capabilities and
attributes. The environment tracks the state of each drone, including its position, orientation,
velocity, and other relevant properties.

The environment supports various customization options, such as the number of drones, the type of
images they capture (RGB or depth), their initial spawning conditions, and the boundaries of the
simulation space. Users can interact with the environment through the provided API, controlling
the drones and querying their states.

The environment also features collision detection, enabling it to process interactions between
drones and between drones and bullets. This functionality is crucial for simulating more complex
scenarios where drones might engage in combat or avoid obstacles.

Usage
-----
To create an instance of the environment, specify the desired configuration parameters, such as
the number of agents, whether to use depth rendering, and the initial conditions for each drone.
Once instantiated, the environment can be reset, stepped through with actions, and queried for
observations, rewards, and other information.

Example
-------
.. code-block:: python

    if __name__ == "__main__":
        env = BaseMultiAgentEnvironment(
            n_agents=2,
            n_images=2,
            depth_render=False,
            drone_class=BaseDrone   # Replace with a custom drone class
        )
        env.reset()
        for i in range(1000):
            action = env.action_space.sample()
            obs, reward, truncated, done, info = env.step(action)

Notes
-----
- The environment is designed for use with the MuJoCo physics engine and requires a valid
  MuJoCo license.
- While the environment supports rendering, it may require additional configuration and
  resources, especially when running on headless servers or in environments without a display.

See Also
--------
`gymnasium.envs.mujoco.MujocoEnv`, `gymnasium.utils.EzPickle`, `ray.rllib.env.MultiAgentEnv`

"""

from typing import Optional, Tuple, Sequence, Type, Dict, List

import numpy as np
from gymnasium import Space
from gymnasium.core import ActType, ObsType
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Dict as SpaceDict
from gymnasium.utils import EzPickle
from mujoco import MjData, mj_step
from mujoco._structs import _MjContactList, _MjDataGeomViews
from numpy.random import default_rng
from ray.rllib.env import MultiAgentEnv
from ray.rllib.env.multi_agent_env import AgentID
from ray.rllib.utils.typing import MultiAgentDict

from Environments.multi_agent.core.BaseDrone import BaseDrone, Bullet
from src.utils.multiagent_model_generator import save_multiagent_model


class MultiAgentDroneEnvironment(MujocoEnv, EzPickle, MultiAgentEnv):
    """
    Simulates a multi-agent environment for drone interactions using MuJoCo.

    This class provides a detailed simulation environment for multiple drones, supporting functionalities like
    movement, perception, and interaction within a shared 3D space. It extends the capabilities of MuJoCoEnv,
    allowing for advanced physics-based simulations with high precision and realism.

    Attributes
    ----------
    metadata : dict
        Metadata for the environment, including available render modes and rendering FPS.
    n_agents : int
        The number of drone agents in the environment.
    n_images : int
        The number of images each drone captures per simulation step.
    drones : List[BaseDrone]
        A list of drones present in the environment, each represented by an instance of BaseDrone or its subclass.

    Parameters
    ----------
    n_agents : int
        The number of drone agents to initialize within the environment.
    n_images : int
        The number of images to capture per drone per simulation step.
    depth_render : bool
        If set to True, drones capture depth images; otherwise, they capture RGB images.
    Drone : Type[BaseDrone]
        The drone class to be used for creating drone instances in the environment.
    **kwargs
        Additional keyword arguments for environment configuration, including render settings and drone initialization parameters.

    Methods
    -------
    reset(seed: Optional[int] = None, options: Optional[dict] = None)
        Resets the environment to an initial state, reinitializing drones' positions and states.
    step(action: ActType)
        Advances the environment state by one timestep, processing drones' actions and interactions.
    drone_aim()
        Calculates the angles between drones' forward directions and their lines of sight to other drones.
    collisions()
        Detects and processes collisions within the environment, including drone-drone and drone-bullet collisions.
    check_bullet_bounds()
        Checks and processes out-of-bound bullets in the simulation.
    check_drone_bounds()
        Verifies and handles drones going out of the designated bounds.
    observation(render: bool = False)
        Constructs and returns the observation for each drone in the environment.
    reward()
        Computes and returns the reward for each drone based on the current simulation state.
    truncated()
        Indicates whether the episode should be truncated for each drone.
    done()
        Determines whether the episode has ended for each drone.
    info()
        Provides additional information and metrics about the current state of the environment.
    """
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 500
    }
    
    def __init__(self, n_agents: int, n_images: int, depth_render: bool, Drone: Type[BaseDrone], **kwargs) -> None:
        """
        Initializes a multi-agent environment tailored for drone simulations, using the MuJoCo physics engine.
        This environment supports multiple drones, each with capabilities for shooting and capturing images.
    
        :param int n_agents: The number of drone agents to be initialized in the environment.
        :param int n_images: The number of images each drone should capture per simulation step. It affects the
                             observation space of each drone.
        :param bool depth_render: Determines if the drones should capture depth images (True) or RGB images (False).
        :param Type[BaseDrone] Drone: The class type for the drones to be instantiated in the environment.
        :keyword Sequence[np.ndarray] spawn_boxes: (Optional) A list of 2x3 NumPy arrays where each array specifies the
                                                    min and max (x, y, z) coordinates of the spawn box for each drone.
        :keyword Sequence[np.ndarray] spawn_angles: (Optional) A list of 2x3 NumPy arrays where each array specifies the
                                                    min and max (roll, pitch, yaw) angles for the initial orientation of
                                                    each drone.
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
        self.n_images = kwargs.get('n_images', 1)
        self.i = 0
        self.n_agents = n_agents
        
        # Load or generate a simulation model based on the provided agent configuration.
        self.model_path = save_multiagent_model(n_agents, kwargs.get('spacing', 2.0), kwargs.get('save_dir', None))
        
        # Initialize the MuJoCo environment with the generated model and specified rendering options.
        height, width = kwargs.get('render_height', 32), kwargs.get('render_width', 32)
        self.frame_skip = kwargs.get('frame_skip', 1)
        self.metadata['render_fps'] = int(np.round(1.0 / (0.002 * self.frame_skip)))
        MujocoEnv.__init__(self, model_path=self.model_path, frame_skip=self.frame_skip,
                           observation_space=Space(), render_mode=kwargs.get('render_mode', 'rgb_array'),
                           height=height, width=width)
        
        # Update the environment's metadata with rendering details.
        self.metadata['render_fps'] = int(np.round(1.0 / self.dt))
        
        spawn_angles = kwargs.get('spawn_angles', [
            np.array([[-np.pi / 6, -np.pi / 6, -np.pi / 6], [np.pi / 6, np.pi / 6, np.pi / 6]])
            for _ in range(n_agents)
        ])
        
        spawn_boxes = kwargs.get('spawn_boxes', [
            np.array([[-10, -10, 0], [10, 10, 10]])
            for _ in range(n_agents)
        ])
        # Setup simulation properties, such as timestep and rendering frequency.
        self._setup_simulation_properties(**kwargs)
        # Instantiate and configure each drone agent within the environment.
        self.drones = [self._create_drone(Drone, i, agent_id, n_images, depth_render, spawn_boxes, spawn_angles,
                                          **kwargs) for i, agent_id in enumerate(range(n_agents))]
        self.observation_space: SpaceDict[AgentID, Space[ObsType]] = SpaceDict({
            drone.agent_id: drone.observation_space for drone in self.drones})
        self.action_space: SpaceDict[AgentID, Space[ObsType]] = SpaceDict({
            drone.agent_id: drone.action_space for drone in self.drones})
        self._action_space_in_preferred_format = True
        self._observation_space_in_preferred_format = True
        self._agent_ids = [drone.agent_id for drone in self.drones]
        
        # Initialize collision tracking dictionaries.
        self._bullet_geom_ids_to_drones = {}
        self._drone_geom_ids_to_drones = {}
        for drone in self.drones:
            # Mapping bullets to their respective drones.
            self._bullet_geom_ids_to_drones[drone.bullet.geom_id] = drone
            
            # Mapping drone parts to their respective drones.
            self._drone_geom_ids_to_drones[drone.geom.id] = drone
            self._drone_geom_ids_to_drones[drone.prop_geoms[0].id] = drone
            self._drone_geom_ids_to_drones[drone.prop_geoms[1].id] = drone
            self._drone_geom_ids_to_drones[drone.prop_geoms[2].id] = drone
            self._drone_geom_ids_to_drones[drone.prop_geoms[3].id] = drone
        
        self.actuators = np.array([0] * self.model.nu)
        self.floor_geom: _MjDataGeomViews = self.data.geom('floor')
        self.time: float = self.data.time
        
        MultiAgentEnv.__init__(self)
    
    def _create_drone(self, drone_class: Type[BaseDrone], index: int, agent_id: int,
                      n_images: int, depth_render: bool, spawn_boxes: Sequence[np.ndarray],
                      spawn_angles: Sequence[np.ndarray], **kwargs) -> BaseDrone:
        """
        Creates a drone object with specified parameters.

        :param drone_class: The class type for the drone to be instantiated.
        :param index: Index of the drone in the environment.
        :param agent_id: Unique identifier for the agent.
        :param n_images: Number of images the drone captures in each step.
        :param depth_render: Specifies if the drone captures depth images.
        :param spawn_boxes: Bounding boxes for drone spawning locations.
        :param spawn_angles: Initial orientation angle ranges for drones.
        :param kwargs: Additional arguments for drone creation.
        :return: An instance of the Drone class.
        """
        return drone_class(model=self.model, data=self.data, renderer=self.mujoco_renderer,
                           n_images=n_images, depth_render=depth_render, index=index + 1, agent_id=agent_id,
                           spawn_box=spawn_boxes[index], max_spawn_velocity=kwargs.get('max_spawn_velocity', 1),
                           spawn_angle_range=spawn_angles[index], rng=kwargs.get('rng', default_rng()),
                           map_bounds=kwargs.get('map_bounds', np.array([[-100, 100], [-100, 100], [0, 100]])),
                           bullet_max_velocity=kwargs.get('bullet_max_velocity', 50), max_time=self.max_time)
    
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
    
    def reset(self, *, seed: Optional[int] = None,
              options: Optional[dict] = None) -> Tuple[MultiAgentDict, MultiAgentDict]:
        """
        Resets the environment to an initial state, reinitializing the positions, orientations, and other relevant
        state variables of all drones. This method is typically called at the beginning of an episode.

        :param seed: An optional seed to ensure deterministic reset behavior. Not currently used.
        :type seed: Optional[int]
        :param options: An optional dictionary of reset options. Not currently used but can be extended for future
            functionalities.
        :type options: Optional[dict]
        :return: A tuple containing the initial observations and metrics after the reset.
        :rtype: Tuple[MultiAgentDict, MultiAgentDict]
        """
        self.i = 0
        self.reset_model()
        observation = self.observation(render=self.n_images > 0 and self.i % self.render_every == 0)
        info = self.info
        return observation, info
    
    def reset_model(self):
        """
        Resets the internal state of each drone model. This method is called by the main reset function to reinitialize
        each drone's state within the environment.
        """
        for drone in self.drones:
            drone.reset()
    
    @property
    def ctrl(self) -> np.ndarray:
        """
        Returns the control signals for each drone in the environment.

        :return: A 1D NumPy array containing the control signals for each drone.
        :rtype: np.ndarray
        """
        return np.concatenate([drone.motor_controls for drone in self.drones]).flatten()
    
    def step(self, action: ActType) \
            -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        """
        Executes one timestep within the environment, processing the actions taken by each drone, updating their states,
        evaluating collisions, and determining rewards based on their actions and interactions.

        The method follows these steps:
        1. Increment the internal timestep counter.
        2. Apply the specified actions to each drone and update their states accordingly.
        3. Advance the physical simulation by a specified number of timesteps.
        4. Detect and process any collisions that occur during the timestep.
        5. Evaluate and process the landing of drones or bullets on the ground.
        6. Check and handle any drones or bullets that go out of bounds.
        7. Calculate and assign rewards based on the positioning of drones within each other's camera views.
        8. Compile and return the new observations, rewards, termination flags, and additional information for each agent.

        :param action: A dictionary mapping each drone's agent ID to its corresponding action.
        :type action: ActType
        :return: A tuple containing updated observations, rewards, termination flags, done flags, and additional info for
                 each agent, in respective order.
        :rtype: Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]
        """
        self.i += 1
        # Action application and state update
        for drone in self.drones:
            drone.env_update()
            drone.act(action[drone.agent_id])
        
        # Simulation step
        mj_step(self.model, self.data, self.frame_skip)
        
        # Collision detection and processing
        shooting_drones, hit_drones, bullet_floor_contacts, drone_floor_contacts = self.collisions()
        for drone in shooting_drones:
            drone.scored_hit()
        for drone in hit_drones:
            drone.got_hit()
        for bullet_id in bullet_floor_contacts:
            self._bullet_geom_ids_to_drones[bullet_id].hit_floor()
        for drone_id in drone_floor_contacts:
            self._drone_geom_ids_to_drones[drone_id].hit_floor()
        
        # Boundary checks
        self.check_drone_bounds()
        self.check_bullet_bounds()
        
        # Reward calculation based on drone positioning
        drones_aiming_at, drones_aimed_at = self.drone_aim()
        for drone in self.drones:
            drone.aiming_at(drones_aiming_at[drone])
            drone.aimed_at(drones_aimed_at[drone])
            
        observation = self.observation(render=self.n_images > 0 and self.i % self.render_every == 0)
        reward = self.reward
        truncated = self.truncated
        done = self.done
        info = self.info
        
        return observation, reward, truncated, done, info
    
    def drone_aim(self) -> Tuple[Dict[BaseDrone, List[Tuple[BaseDrone, float]]],
                                 Dict[BaseDrone, List[Tuple[BaseDrone, float]]]]:
        """
        Determines the visibility and angular relationship between drones in the simulation. For each pair of drones,
        it calculates the angle between the direction one drone is facing and the line of sight to the other drone.
    
        The method calculates the angle :math:`\\theta_{ij}` between the forward direction of drone_i (:math:`f_i`)
        and the unit vector (:math:`u_{ij}`) pointing from drone_i to drone_j. The angle is computed using the dot
        product formula:
    
        .. math::
    
            \\theta_{ij} = \\arccos\\left(\\frac{f_i \\cdot u_{ij}}{||f_i|| \\times ||u_{ij}||}\\right)
    
        where:
    
        - :math:`f_i` is the forward unit vector of drone_i.
        - :math:`u_{ij}` is the unit vector from drone_i to drone_j.
        - :math:`\\cdot` denotes the dot product.
        - :math:`||x||` denotes the norm of vector :math:`x`.
    
        Results are stored in two dictionaries mapping each drone to a list of tuples, where each tuple contains
        a visible drone and the corresponding angle.
    
        :return: A tuple of two dictionaries. The first maps each drone to a list of drones it can see along with the angle.
                 The second maps each drone to a list of drones that can see it, along with the angle.
        :rtype: Tuple[Dict[BaseDrone, List[Tuple[BaseDrone, float]]], Dict[BaseDrone, List[Tuple[BaseDrone, float]]]]
        """
        
        drones_aiming_at = {drone: [] for drone in self.drones}
        drones_aimed_at = {drone: [] for drone in self.drones}
        
        positions = {drone: drone.position for drone in self.drones}
        directions = {drone: drone.forward_unit_vector for drone in self.drones}
        
        for aiming_drone in self.drones:
            for aimed_at_drone in self.drones:
                if aimed_at_drone is not aiming_drone:
                    aiming_drone_pos = positions[aiming_drone]
                    aiming_drone_dir = directions[aiming_drone]
                    aimed_at_drone_pos = positions[aimed_at_drone]

                    drone_to_drone_vector = aimed_at_drone_pos - aiming_drone_pos
                    drone_to_drone_vector /= np.linalg.norm(drone_to_drone_vector)

                    theta = np.arccos(np.clip(np.dot(aiming_drone_dir, drone_to_drone_vector), -1.0, 1.0))

                    # Update both drones' records in a single pass
                    drones_aiming_at[aiming_drone].append((aimed_at_drone, theta))
                    drones_aimed_at[aimed_at_drone].append((aiming_drone, theta))
        
        return drones_aiming_at, drones_aimed_at
    
    def collisions(self) -> tuple[List[BaseDrone], List[BaseDrone], List[Bullet], List[BaseDrone]]:
        """
        Checks for and processes collisions between drones and bullets within the environment. Drone includes both
            body and 4 propellers.

        :return: A tuple of two lists containing the drones that have shot and been hit, respectively.
        :rtype: tuple[List[BaseDrone], List[Drone]]
        """
        contact_list: _MjContactList = self.data.contact
        if not hasattr(contact_list, 'geom1') or not hasattr(contact_list, 'geom2'):
            return [], [], [], []
        contact_geom_pairs: np.ndarray = np.array([contact_list.geom1, contact_list.geom2]).T
        
        # Initialize sets to keep track of shooting and hit drones.
        shooting_drones = set()
        hit_drones = set()
        bullet_floor_contacts = set()
        drone_floor_contacts = set()
        
        # Loop through each contact pair to detect collisions.
        for geom_1_id, geom_2_id in contact_geom_pairs:
            # Check each pair to see if it involves a bullet that is flying and a drone.
            for geom_1, geom_2 in [(geom_1_id, geom_2_id), (geom_2_id, geom_1_id)]:
                # Check if the bullet has hit a drone (except the parent drone).
                # region Bullet - Drone Collisions
                if (geom_1 in self._bullet_geom_ids_to_drones.keys() and
                        geom_2 in self._drone_geom_ids_to_drones.keys() and
                        self._bullet_geom_ids_to_drones[geom_1].geom.id != geom_2):
                    shooting_drone = self._bullet_geom_ids_to_drones[geom_1]
                    hit_drone = self._drone_geom_ids_to_drones[geom_2]
                    shooting_drones.add(shooting_drone)
                    hit_drones.add(hit_drone)
                # endregion
                # region Bullet - Floor Collisions
                elif geom_1 in self._bullet_geom_ids_to_drones.keys() and \
                        geom_2 == self.floor_geom.id:
                    bullet_floor_contacts.add(geom_1)
                # endregion
                # region Drone - Floor Collisions
                elif geom_1 in self._drone_geom_ids_to_drones.keys() and \
                        geom_2 == self.floor_geom.id:
                    drone_floor_contacts.add(geom_1)
                # endregion
            
            # Check for Drone - Drone Collisions, ensuring we're not considering the bullet twice.
            if geom_1_id in self._drone_geom_ids_to_drones and geom_2_id in self._drone_geom_ids_to_drones:
                hit_drones.add(self._drone_geom_ids_to_drones[geom_1_id])
                hit_drones.add(self._drone_geom_ids_to_drones[geom_2_id])
        
        return list(shooting_drones), list(hit_drones), list(bullet_floor_contacts), list(drone_floor_contacts)
    
    def check_bullet_bounds(self):
        """
        Checks if any bullet has gone out of bounds and resets it if necessary.
        """
        for drone in self.drones:
            if drone.bullet.out_of_bounds:
                drone.on_bullet_out_of_bounds()
    
    def check_drone_bounds(self):
        """
        Checks if any drone has gone out of bounds and resets it if necessary.
        """
        for drone in self.drones:
            if not drone.in_bounds:
                drone.on_out_of_bounds()
    
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
        return {drone.agent_id: drone.reward for drone in self.drones}
    
    @property
    def truncated(self) -> MultiAgentDict:
        """
        Determines whether the current episode should be truncated for each drone.

        :return: A dictionary mapping each agent's ID to a boolean indicating whether its episode should be truncated.
        :rtype: MultiAgentDict
        """
        return {drone.agent_id: drone.truncated for drone in self.drones}
    
    @property
    def done(self) -> MultiAgentDict:
        """
        Determines whether the current episode is done for each drone.

        :return: A dictionary mapping each agent's ID to a boolean indicating whether its episode is done.
        :rtype: MultiAgentDict
        """
        return {drone.agent_id: drone.done for drone in self.drones}
    
    @property
    def info(self) -> MultiAgentDict:
        """
        Collects and returns additional metrics about the environment's current state.

        :return: A dictionary of metrics.
        :rtype: MultiAgentDict
        """
        return {drone.agent_id: drone.info for drone in self.drones}
