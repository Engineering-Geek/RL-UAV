from typing import Type, Set, Optional, Tuple, Dict

import numpy as np
from gymnasium.spaces import Dict as DictSpace
from mujoco import MjData, viewer, mj_resetData, mj_step
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict, AgentID

from environments.core.drone import Drone
from environments.core.sensors import Sensor, ClosestDronesSensor
from mujoco_xml.model_generator import get_model


class MultiAgentDroneEnvironment(MultiAgentEnv):
    def __init__(self, n_agents: int, spacing: float, DroneClass: Type[Drone], sensors: Set[Sensor],
                 map_bounds: np.ndarray, spawn_box: np.ndarray, dt: float, fps: int = 30, sim_rate: int = 1,
                 n_thetas: int = None, n_phi: int = None,
                 bullet_speed: float = 10.0, update_distances: bool = False, render_mode: str = None,
                 time_limit: float = 60.0):
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
        self.n_thetas = n_thetas
        self.n_phis = n_phi
        self.dt = dt
        self.DroneClass = DroneClass
        self.time_limit = time_limit
        
        self.gammas = np.zeros((n_agents, n_agents))
        self.drone_to_drone_vectors = np.zeros((n_agents, n_agents, 3))
        self.distances = np.zeros((n_agents, n_agents))
        self.drone_to_drone_vectors = np.zeros((n_agents, n_agents, 3))
        self.update_distances = True if any([isinstance(sensor, ClosestDronesSensor)
                                             for sensor in sensors]) else update_distances
        self.max_distance = np.linalg.norm(self.map_bounds[1] - self.map_bounds[0])
        
        self.drones: Dict[AgentID, Drone] = self._init_drones(sensors=sensors, bullet_speed=bullet_speed)
        self.drone_body_ids: Set[int] = {drone.body_id for drone in self.drones.values()}
        
        self.drone_geom_ids_to_drones: Dict[int, Drone] = {drone.collision_geom_1_id: drone for drone in
                                                           self.drones.values()}
        self.drone_geom_ids_to_drones.update({drone.collision_geom_2_id: drone for drone in self.drones.values()})
        self.drone_geom_ids = set(self.drone_geom_ids_to_drones.keys())
        
        self.bullet_geom_ids_to_drones: Dict[int, Drone] = {drone.bullet_geom_id: drone for drone in
                                                            self.drones.values() if drone.bullet_geom_id is not None}
        self.bullet_geom_ids = set(self.bullet_geom_ids_to_drones.keys())
        
        self.boundary_geom_ids = set()
        names = ["top_wall", "bottom_wall", "left_wall", "right_wall"]
        for name in names:
            self.boundary_geom_ids.add(self.model.geom(name).id)
        self.boundary_geom_ids.add(self.model.geom("floor").id)
        
        self.renderer: viewer.Handle = viewer.launch_passive(self.model, self.data) if render_mode == "human" else None
        
        self.drone_positions = np.zeros((n_agents, 3))
        self._obs_space_in_preferred_format = True
        self._action_space_in_preferred_format = True
        super().__init__()
    
    def _init_drones(self, sensors: Set[Sensor], bullet_speed: float) -> dict:
        drones = dict()
        sensors = list(sensors)
        for agent_id in self._agent_ids:
            for sensor in sensors:
                sensor.environment_init(
                    model=self.model, data=self.data, agent_id=agent_id, world_bounds=self.map_bounds,
                    gammas=self.gammas, distances=self.distances)
            
            # Create the Drone instance with all sensors
            drones[agent_id] = self.DroneClass(
                model=self.model, data=self.data, agent_id=agent_id, spawn_box=self.spawn_box, gammas=self.gammas,
                distances=self.distances, sensors=sensors, bullet_speed=bullet_speed
            )
        
        return drones
    
    def update_distances_and_gammas(self):
        # Update drone positions
        for i, agent_id in enumerate(self._agent_ids):
            self.drone_positions[i, :] = self.data.xpos[self.drones[agent_id].body_id]
        
        # Calculate drone-to-drone relationships if enabled
        if self.update_distances:
            # Calculate relative positions
            np.subtract(self.drone_positions[:, np.newaxis, :], self.drone_positions, out=self.drone_to_drone_vectors)
            
            # Compute distances
            self.distances = np.linalg.norm(self.drone_to_drone_vectors, axis=2)
            np.fill_diagonal(self.distances, np.nan)  # Avoid self-distance
            
            # Calculate angles using arctan2, taking care of the axes order (y, x)
            np.arctan2(self.drone_to_drone_vectors[..., 1], self.drone_to_drone_vectors[..., 0], out=self.gammas)
            np.fill_diagonal(self.gammas, np.nan)  # Avoid self-angle
    
    def collisions(self) -> None:
        """
        Checks and processes collisions between drones, bullets, targets, and the environment.
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
            
            # region Bullet to Boundary Collision
            if drone_bullet_1 and geom_2_id in self.boundary_geom_ids:
                drone_bullet_1.bullet_out_of_bounds = True
                continue
            if drone_bullet_2 and geom_1_id in self.boundary_geom_ids:
                drone_bullet_2.bullet_out_of_bounds = True
                continue
            # endregion
    
    def out_of_bounds(self) -> None:
        # TODO: Optimize if possible
        for drone in self.drones.values():
            drone.out_of_bounds = np.any(drone.data.xpos[drone.body_id] < self.map_bounds[0]) or \
                                  np.any(drone.data.xpos[drone.body_id] > self.map_bounds[1])
            drone.bullet_out_of_bounds = np.any(drone.data.xpos[drone.bullet_body_id] < self.map_bounds[0]) or \
                                         np.any(drone.data.xpos[drone.bullet_body_id] > self.map_bounds[1])
    
    @property
    def observation(self) -> MultiAgentDict:
        return {agent_id: self.drones[agent_id].observation for agent_id in self._agent_ids}
    
    @property
    def observation_space(self) -> DictSpace:
        return DictSpace({agent_id: self.drones[agent_id].observation_space for agent_id in self._agent_ids})
    
    def action(self, action_dict: MultiAgentDict) -> None:
        for agent_id, action in action_dict.items():
            self.drones[agent_id].act(action)
    
    @property
    def action_space(self) -> DictSpace:
        return DictSpace({agent_id: self.drones[agent_id].action_space for agent_id in self._agent_ids})
    
    @property
    def reward(self) -> MultiAgentDict:
        return {agent_id: self.drones[agent_id].reward for agent_id in self._agent_ids}
    
    @property
    def done(self) -> MultiAgentDict:
        dones = {agent_id: self.drones[agent_id].done for agent_id in self._agent_ids}
        dones["__all__"] = any(dones.values())
        return dones
    
    @property
    def truncated(self) -> MultiAgentDict:
        trucated = {agent_id: self.drones[agent_id].truncated for agent_id in self._agent_ids}
        trucated["__all__"] = self.data.time > self.time_limit
        return trucated
    
    @property
    def info_log(self) -> dict:
        return {agent_id: self.drones[agent_id].log_info for agent_id in self._agent_ids}
    
    def update_sensors(self):
        if self.i % self.render_every == 0:
            for drone in self.drones.values():
                drone.update_sensors()
        self.i += 1
    
    def custom_drone_updates(self):
        for drone in self.drones.values():
            drone.custom_update()
    
    def reset_default_flags(self):
        for drone in self.drones.values():
            drone.reset_default_flags()
    
    def reset_custom_flags(self):
        for drone in self.drones.values():
            drone.reset_custom_flags()
    
    def step(self, action_dict: MultiAgentDict) -> Tuple[
        MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict
    ]:
        self.reset_default_flags()
        self.reset_custom_flags()
        self.action(action_dict)
        mj_step(self.model, self.data, self.sim_rate)
        self.update_distances_and_gammas()
        self.update_sensors()
        self.collisions()
        self.out_of_bounds()
        self.custom_drone_updates()
        return self.observation, self.reward, self.done, self.truncated, self.info_log
    
    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ) -> Tuple[MultiAgentDict, MultiAgentDict]:
        if seed is not None:
            np.random.seed(seed)
        mj_resetData(self.model, self.data)
        self.reset_default_flags()
        for drone in self.drones.values():
            drone.reset()
        self.update_distances_and_gammas()
        self.update_sensors()
        self.custom_drone_updates()
        return self.observation, self.info_log
    
    def close(self) -> None:
        if self.renderer:
            self.renderer.close()
    
    def render(self) -> None:
        if self.renderer:
            self.renderer.sync()
