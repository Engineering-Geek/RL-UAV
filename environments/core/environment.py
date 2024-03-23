import logging
from typing import Type, Set, Optional, Tuple, Dict, List

import numpy as np
from gymnasium.spaces import Dict as DictSpace
from mujoco import MjData, viewer, mj_resetData, mj_step
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict, AgentID

from environments.core.drone import Drone
from environments.core.sensors import Sensor
from environments.models.model_generator import get_model

logging.basicConfig(level=logging.INFO, filename="logs.log")
logger = logging.getLogger(__name__)


class MultiAgentDroneEnvironment(MultiAgentEnv):
    """
    A multi-agent environment for drone simulation using MuJoCo as the physics engine and integrating with the Ray RLlib framework.

    This environment supports multiple drones navigating and interacting in a 3D space. It integrates sensors, collision handling,
    and other drone-specific behaviors, providing a comprehensive setup for simulating drone operations in a multi-agent context.

    Attributes:
        model (MjModel): The MuJoCo model of the environment.
        data (MjData): The data structure for the MuJoCo model.
        map_bounds (np.ndarray): The boundaries of the map.
        spawn_box (np.ndarray): The spawning box from which drones start.
        n_agents (int): The number of agents (drones) in the environment.
        fps (int): The frames per second for the simulation.
        sim_rate (int): The simulation rate.
        render_every (int): How often to render the scene.
        dt (float): The time step for the simulation.
        DroneClass (Type[Drone]): The class of the drone agents.
        time_limit (float): The time limit for the simulation.
        drones (Dict[AgentID, Drone]): A dictionary mapping agent IDs to drone instances.
        drone_geom_ids_to_drones (Dict[int, Drone]): A dictionary mapping drone geometry IDs to drone instances.
        bullet_geom_ids_to_drones (Dict[int, Drone]): A dictionary mapping bullet geometry IDs to drone instances.
        boundary_geom_ids (Set[int]): A set of boundary geometry IDs.
        renderer (viewer.Handle): The MuJoCo viewer handle.
        drone_positions (np.ndarray): The positions of the drones.
        _obs_space_in_preferred_format (bool): A flag indicating if the observation space is in the preferred format.
        _action_space_in_preferred_format (bool): A flag indicating if the action space is in the preferred format.

    Parameters:
        n_agents (int): The number of drone agents.
        spacing (float): The spacing between drones.
        DroneClass (Type[Drone]): The drone class to be instantiated for each agent.
        sensors (Set[Sensor]): A set of sensors to be attached to each drone.
        map_bounds (np.ndarray): The boundaries of the map.
        spawn_box (np.ndarray): The spawning box for drones.
        dt (float): The simulation time step.
        fps (int, optional): Frames per second for the simulation. Defaults to 30.
        sim_rate (int, optional): The rate at which the simulation steps are run. Defaults to 1.
        bullet_speed (float, optional): The speed of bullets fired by drones. Defaults to 10.0.
        update_distances (bool, optional): Whether to update distances and angles between drones. Defaults to False.
        render_mode (str, optional): The mode for rendering ('human' for on-screen). Defaults to None.
        time_limit (float, optional): The time limit for each episode. Defaults to 60.0.
    """
    
    def __init__(self, n_agents: int, spacing: float, DroneClass: Type[Drone], sensor: Sensor,
                 map_bounds: np.ndarray, spawn_box: np.ndarray, dt: float, fps: int = 30, sim_rate: int = 1,
                 bullet_speed: float = 10.0, update_distances: bool = False, render_mode: str = None,
                 time_limit: float = 60.0):
        """
        Initialize the multi-agent drone environment.
        
        Args:
            n_agents (int): The number of drone agents.
            spacing (float): The spacing between drones.
            DroneClass (Type[Drone]): The drone class to be instantiated for each agent.
            sensor (Sensor): The sensor to be attached to each drone.
            map_bounds (np.ndarray): The boundaries of the map.
            spawn_box (np.ndarray): The spawning box for drones.
            dt (float): The simulation time step.
            fps (int, optional): Frames per second for the simulation. Defaults to 30.
            sim_rate (int, optional): The rate at which the simulation steps are run. Defaults to 1.
            bullet_speed (float, optional): The speed of bullets fired by drones. Defaults to 10.0.
            update_distances (bool, optional): Whether to update distances and angles between drones. Defaults to False.
            render_mode (str, optional): The mode for rendering ('human' for on-screen). Defaults to None.
            time_limit (float, optional): The time limit for each episode. Defaults to 60.0.
        """
        model = get_model(
            num_agents=n_agents,
            spacing=spacing,
            drone_spawn_height=0.25,
            map_bounds=map_bounds
        )
        model.opt.timestep = dt
        self.model = model
        self.data = MjData(self.model)
        self.map_bounds = map_bounds
        self.spawn_box = spawn_box
        self._agent_ids: Set[AgentID] = set(range(n_agents))
        self.n_agents = n_agents
        self.fps = fps
        self.sim_rate = sim_rate
        self.render_every = int(fps / sim_rate)
        self.i = 0
        self.dt = dt
        self.DroneClass = DroneClass
        self.time_limit = time_limit
        
        self.gammas = np.zeros((n_agents, n_agents))
        self.drone_to_drone_vectors = np.zeros((n_agents, n_agents, 3))
        self.distances = np.zeros((n_agents, n_agents))
        self.drone_to_drone_vectors = np.zeros((n_agents, n_agents, 3))
        self.update_distances = update_distances
        self.max_distance = np.linalg.norm(self.map_bounds[1] - self.map_bounds[0])
        
        self.drones: Dict[AgentID, Drone] = self._init_drones(sensor=sensor, bullet_speed=bullet_speed)
        
        self.drone_geom_ids_to_drones: Dict[int, Drone] = {
            drone.collision_geom_1_id: drone for drone in self.drones.values()}
        self.drone_geom_ids_to_drones.update({drone.collision_geom_2_id: drone for drone in self.drones.values()})
        self.drone_geom_ids = set(self.drone_geom_ids_to_drones.keys())
        
        self.bullet_geom_ids_to_drones: Dict[int, Drone] = {
            drone.bullet_geom_id: drone for drone in self.drones.values() if drone.bullet_geom_id is not None}
        self.bullet_geom_ids = set(self.bullet_geom_ids_to_drones.keys())
        
        self.boundary_geom_ids = set()
        names = ["top_wall", "bottom_wall", "left_wall", "right_wall"]
        for name in names:
            _id = self.model.geom(name).id
            self.boundary_geom_ids.add(_id)
        floor_id = self.model.geom("floor").id
        self.boundary_geom_ids.add(floor_id)
        
        self.renderer: viewer.Handle = viewer.launch_passive(self.model, self.data) if render_mode == "human" else None
        
        self.drone_positions = np.zeros((n_agents, 3))
        self._obs_space_in_preferred_format = True
        self._action_space_in_preferred_format = True
        super().__init__()
    
    def _init_drones(self, sensor: Sensor, bullet_speed: float) -> dict:
        """
        Initializes the drones within the environment.

        Parameters:
            sensor (Sensor): A set of sensors to be attached to each drone.
            bullet_speed (float): The speed of bullets fired by drones.

        Returns:
            Dict[AgentID, Drone]: A dictionary mapping agent IDs to initialized drone instances.
        """
        drones = {}
        for agent_id in self._agent_ids:
            drones[agent_id] = self.DroneClass(
                model=self.model,
                data=self.data,
                agent_id=agent_id,
                spawn_box=self.spawn_box,
                gammas=self.gammas[agent_id],
                distances=self.distances[agent_id],
                sensor=sensor,
                bullet_speed=bullet_speed
            )
            drones[agent_id].sensor.environment_init(
                model=self.model,
                data=self.data,
                agent_id=agent_id,
            )
        return drones
    
    def update_distances_and_gammas(self):
        """
        Updates the distances and relative angles (gammas) between all pairs of drones.

        This method calculates the Euclidean distance and the relative angle for each pair of drones.
        The angle is computed using the arctangent function, considering the x and y position differences.

        .. math::

            \text{For each pair of drones } (i, j):

            \text{distance}_{ij} = \sqrt{pow(x_j - x_i, 2) + pow(y_j - y_i, 2) + pow(z_j - z_i, 2)}

            \gamma_{ij} = \arctan2(\frac{y_i - y_j}{x_i - x_j})

        Where:
        - \( x_i, y_i, z_i \) are the x, y, and z coordinates of drone \( i \).
        - \( \text{distance}_{ij} \) is the Euclidean distance between drone \( i \) and drone \( j \).
        - \( \gamma_{ij} \) is the relative angle between drone \( i \) and drone \( j \), considering the x and y axes.
        """
        # Update drone positions
        for i, agent_id in enumerate(self._agent_ids):
            self.drone_positions[i, :] = self.data.xpos[self.drones[agent_id].body_id]
        
        # Calculate drone-to-drone relationships if enabled
        if self.update_distances:
            # Calculate relative positions
            np.subtract(self.drone_positions, self.drone_positions[:, np.newaxis, :], out=self.drone_to_drone_vectors)
            
            # Compute distances
            self.distances = np.linalg.norm(self.drone_to_drone_vectors, axis=2)
            np.fill_diagonal(self.distances, np.nan)  # Avoid self-distance
            
            # Calculate angles using arctan2, taking care of the axes order (y, x)
            np.arctan2(self.drone_to_drone_vectors[..., 1], self.drone_to_drone_vectors[..., 0], out=self.gammas)
            np.fill_diagonal(self.gammas, np.nan)  # Avoid self-angle
    
    def collisions(self) -> None:
        """
        Checks and processes collisions between drones, bullets, and the environment.
        """
        for contact in self.data.contact:
            geom_1_id, geom_2_id = contact.geom1, contact.geom2
            # region Collision Object Retrieval
            drone_bullet_1 = self.bullet_geom_ids_to_drones.get(geom_1_id)
            drone_1 = self.drone_geom_ids_to_drones.get(geom_1_id)
            
            drone_bullet_2 = self.bullet_geom_ids_to_drones.get(geom_2_id)
            drone_2 = self.drone_geom_ids_to_drones.get(geom_2_id)
            # endregion
            
            # region Bullet to Drone Collision
            if drone_bullet_1 and drone_2 and drone_bullet_1 != drone_2:
                drone_bullet_1.shot_drone = True
                drone_2.got_shot = True
                continue
            if drone_bullet_2 and drone_1 and drone_bullet_2 != drone_1:
                drone_bullet_2.shot_drone = True
                drone_1.got_shot = True
                continue
            # endregion
            
            # region Drone to Boundary Collision
            if drone_1 and geom_2_id in self.boundary_geom_ids:
                drone_1.hit_floor = True
                continue
            if drone_2 and geom_1_id in self.boundary_geom_ids:
                drone_2.hit_floor = True
                continue
            # endregion
            
            # region Drone to Drone Collision
            if drone_1 and drone_2 and drone_1 != drone_2:
                drone_1.crash_into_drone = True
                drone_2.crash_into_drone = True
                continue
            # endregion
    
    @property
    def observation(self) -> MultiAgentDict:
        """
        Returns the current observation for each agent.

        Returns:
            MultiAgentDict: A dictionary mapping each agent ID to its observation.
        """
        return {agent_id: self.drones[agent_id].observation for agent_id in self._agent_ids}
    
    @property
    def observation_space(self) -> DictSpace:
        """
        Returns the observation space for each agent.
        
        Returns:
            DictSpace: A dictionary space mapping each agent ID to its observation space.
        """
        return DictSpace({agent_id: self.drones[agent_id].observation_space for agent_id in self._agent_ids})
    
    def action(self, action_dict: MultiAgentDict) -> None:
        """
        Applies the given actions to the corresponding drones in the environment.

        Parameters:
            action_dict (MultiAgentDict): A dictionary mapping each agent ID to its action.
        """
        for agent_id, action in action_dict.items():
            self.drones[agent_id].act(action)
    
    @property
    def action_space(self) -> DictSpace:
        """
        The action space for the environment.

        Returns:
            DictSpace: A dictionary space mapping each agent ID to its action space.
        """
        return DictSpace({agent_id: self.drones[agent_id].action_space for agent_id in self._agent_ids})
    
    @property
    def reward(self) -> MultiAgentDict:
        """
        Computes and returns the current reward for each agent.

        Returns:
            MultiAgentDict: A dictionary mapping each agent ID to its reward.
        """
        return {agent_id: self.drones[agent_id].reward for agent_id in self._agent_ids}
    
    @property
    def simulation_unstable(self) -> bool:
        """
        Checks and returns the simulation stability status for each agent.

        Returns:
            MultiAgentDict: A dictionary mapping each agent ID to a boolean indicating whether the simulation is unstable.
        """
        return any(self.drones[agent_id].makes_simulation_unstable for agent_id in self._agent_ids)
    
    @property
    def done(self) -> MultiAgentDict:
        """
        Checks and returns the done status for each agent.

        Returns:
            MultiAgentDict: A dictionary mapping each agent ID to a boolean indicating whether it's done.
        """
        dones = {agent_id: self.drones[agent_id].done for agent_id in self._agent_ids}
        dones["__all__"] = (self.data.time > self.time_limit or self.simulation_unstable)
        return dones
    
    @property
    def truncated(self) -> MultiAgentDict:
        """
        Checks and returns the truncated status for each agent.

        Returns:
            MultiAgentDict: A dictionary mapping each agent ID to a boolean indicating whether it's truncated.
        """
        trucated = {agent_id: self.drones[agent_id].truncated for agent_id in self._agent_ids}
        trucated["__all__"] = self.data.time > self.time_limit or self.simulation_unstable
        return trucated
    
    @property
    def info_log(self) -> dict:
        """
        Provides additional information about each agent for logging purposes.

        Returns:
            dict: A dictionary mapping each agent ID to its logging information.
        """
        return {agent_id: self.drones[agent_id].log_info for agent_id in self._agent_ids}
    
    def update_sensors(self):
        """
        Updates the sensors for each drone, typically called once every simulation step.
        """
        if self.i % self.render_every == 0:
            for drone in self.drones.values():
                drone.update_sensor()
        self.i += 1
    
    def custom_update(self):
        """
        Custom update function for the environment, typically called once every simulation step.
        """
        for drone in self.drones.values():
            drone.custom_update()
    
    def reset_default_flags(self):
        """
        Resets the default flags for each drone, typically called once every simulation step.
        """
        for drone in self.drones.values():
            drone.reset_default_flags()
    
    def _step(self):
        """
        Executes one step in the environment, typically called once every simulation step.
        """
        mj_step(self.model, self.data, self.sim_rate)
    
    def step(self, action_dict: MultiAgentDict) -> Tuple[
        MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict
    ]:
        """
        Executes one step in the environment, processing actions, updating states, and returning the results.

        Parameters:
            action_dict (MultiAgentDict): A dictionary mapping each agent ID to its action.

        Returns:
            Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
                A tuple containing the observation, reward, done, truncated, and info log dictionaries.
        """
        self.reset_default_flags()
        self.action(action_dict)
        
        self._step()
        
        self.update_distances_and_gammas()
        self.update_sensors()
        
        self.collisions()
        
        self.custom_update()
        
        return self.observation, self.reward, self.done, self.done, self.info_log
    
    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ) -> Tuple[MultiAgentDict, MultiAgentDict]:
        """
        Resets the environment to an initial state.

        Parameters:
            seed (Optional[int]): An optional seed to ensure determinism.
            options (Optional[dict]): Additional options for resetting the environment.

        Returns:
            Tuple[MultiAgentDict, MultiAgentDict]: The initial observation and logging information.
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.i = 0
        mj_resetData(self.model, self.data)
        
        self.reset_default_flags()
        
        for drone in self.drones.values():
            drone.reset()
        
        self.update_distances_and_gammas()
        self.update_sensors()
        
        initial_observation, info_log = self.observation, self.info_log
        return initial_observation, info_log
    
    def close(self) -> None:
        """
        Closes the environment, releasing any resources.
        """
        if self.renderer:
            self.renderer.close()
    
    def render(self) -> None:
        """
        Renders the environment. If a viewer is active, it updates the viewer with the current state of the environment.
        """
        if self.renderer:
            self.renderer.sync()
