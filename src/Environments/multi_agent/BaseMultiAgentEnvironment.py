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
            depth_render=False
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

from typing import Optional, Tuple, Sequence

import numpy as np
from gymnasium import Space
from gymnasium.core import ActType, ObsType
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Dict as SpaceDict
from gymnasium.utils import EzPickle
from mujoco import MjData
from mujoco._structs import _MjContactList
from numpy.random import default_rng
from ray.rllib.env import MultiAgentEnv
from ray.rllib.env.multi_agent_env import AgentID
from ray.rllib.utils.typing import MultiAgentDict

from src.Environments.multi_agent.Drones import Drone
from src.utils.multiagent_model_generator import save_multiagent_model


class BaseMultiAgentEnvironment(MujocoEnv, EzPickle, MultiAgentEnv):
    """
    A multi-agent environment for drone simulation based on the MuJoCo physics engine.
    It supports rendering, interaction, and control of multiple drones capable of shooting and perception.

    The environment initializes with a specified number of drones, each with its own attributes and capabilities,
    like shooting bullets and capturing images. The environment facilitates the interaction between these drones
    and tracks their states throughout the simulation.
    """
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 500
    }
    
    def __init__(self, n_agents: int, n_images: int, depth_render: bool, **kwargs) -> None:
        """
        Initializes a multi-agent environment tailored for drone simulations, using the MuJoCo physics engine.
        This environment supports multiple drones, each with capabilities for shooting and capturing images.
    
        :param int n_agents: The number of drone agents to be initialized in the environment.
        :param int n_images: The number of images each drone should capture per simulation step. It affects the
                             observation space of each drone.
        :param bool depth_render: Determines if the drones should capture depth images (True) or RGB images (False).
        :keyword Sequence[np.ndarray] spawn_boxes: (Optional) A list of 2x3 NumPy arrays where each array specifies the min and max
                                                   (x, y, z) coordinates of the spawn box for each drone.
        :keyword Sequence[np.ndarray] spawn_angles: (Optional) A list of 2x3 NumPy arrays where each array specifies the min and max
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
        self.n_images = kwargs.get('n_images', 1)
        self.i = 0
        self.n_agents = n_agents
        
        # Load or generate a simulation model based on the provided agent configuration.
        self.model_path = save_multiagent_model(n_agents, kwargs.get('spacing', 2.0), kwargs.get('save_dir', None))
        
        # Initialize the MuJoCo environment with the generated model and specified rendering options.
        height, width = kwargs.get('render_height', 32), kwargs.get('render_width', 32)
        MujocoEnv.__init__(self, model_path=self.model_path, frame_skip=kwargs.get('frame_skip', 1),
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
        # Instantiate and configure each drone agent within the environment.
        self.drones = [self._create_drone(i, agent_id, n_images, depth_render, spawn_boxes, spawn_angles, **kwargs)
                       for i, agent_id in enumerate(range(n_agents))]
        self.observation_space: SpaceDict[AgentID, Space[ObsType]] = SpaceDict({
            drone.agent_id: drone.observation_space for drone in self.drones})
        self.action_space: SpaceDict[AgentID, Space[ObsType]] = SpaceDict({
            drone.agent_id: drone.action_space for drone in self.drones})
        self._action_space_in_preferred_format = True
        self._observation_space_in_preferred_format = True
        self._agent_ids = [drone.agent_id for drone in self.drones]
        MultiAgentEnv.__init__(self)
        # Setup simulation properties, such as timestep and rendering frequency.
        self._setup_simulation_properties(**kwargs)
    
    def _create_drone(self, index: int, agent_id: int, n_images: int, depth_render: bool,
                      spawn_boxes: Sequence[np.ndarray],
                      spawn_angles: Sequence[np.ndarray], **kwargs) -> Drone:
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
                     depth_render=depth_render, index=index + 1, agent_id=agent_id, spawn_box=spawn_boxes[index],
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
        
        # Initialize collision tracking dictionaries.
        self._bullet_geom_ids_to_drones = {}
        self._drone_geom_ids_to_drones = {}
        for drone in self.drones:
            self._bullet_geom_ids_to_drones[drone.bullet.geom_id] = drone
            self._drone_geom_ids_to_drones[drone.geom.id] = drone
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[
        MultiAgentDict, MultiAgentDict]:
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
        observation = self.observation(render=self.n_images > 0 and self.i % self.render_every == 0)
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
            return {drone.agent_id: True for drone in self.drones}
        
        # Cases for individual truncation: drone is dead
        return {drone.agent_id: not drone.alive for drone in self.drones}
    
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
    env = BaseMultiAgentEnvironment(
        n_agents=2,
        n_images=2,
        depth_render=False
    )
    env.reset()
    drones = env.drones
    # print(obs)
    for i in range(1000):
        action = env.action_space.sample()
        obs, reward, truncated, done, info = env.step(action)


if __name__ == "__main__":
    test_multi_agent_env()
