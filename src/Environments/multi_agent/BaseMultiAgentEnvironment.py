from typing import List, Union, Optional, Tuple, Dict

import numpy as np
import quaternion
from quaternion import numpy_quaternion as np_quaternion
from gymnasium.core import ActType
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from gymnasium.spaces import Box, Dict, MultiBinary, Space
from gymnasium.utils import EzPickle
from mujoco import MjModel, MjData
from mujoco._structs import _MjDataBodyViews, _MjDataSensorViews, _MjDataActuatorViews, _MjDataSiteViews, _MjContactList
from mujoco._structs import _MjDataGeomViews
from numpy.random import Generator, default_rng
from ray.rllib.env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict, EnvObsType, EnvActionType, AgentID

from utils.multiagent_model_generator import save_multiagent_model


class Bullet:
    """
    The Bullet class represents a bullet in the simulation.

    Attributes
    ----------
    model : MjModel
        The Mujoco model of the environment.
    data : MjData
        The Mujoco data of the environment.
    index : int
        The index of the drone that the bullet belongs to.
    shoot_velocity : float
        The velocity at which the bullet is shot.
    x_bounds : np.ndarray
        The x-axis boundaries of the environment.
    y_bounds : np.ndarray
        The y-axis boundaries of the environment.
    z_bounds : np.ndarray
        The z-axis boundaries of the environment.

    Methods
    -------
    reset():
        Resets the bullet's position, velocity, and flying status.
    shoot():
        Shoots the bullet if it is not already flying.
    update():
        Updates the bullet's status. If the bullet is out of bounds, it is reset.
    geom_id():
        Returns the geometry ID of the bullet.
    is_flying():
        Returns whether the bullet is currently flying.
    """
    
    def __init__(self, model: MjModel, data: MjData, index: int, shoot_velocity: float, x_bounds: np.ndarray,
                 y_bounds: np.ndarray, z_bounds: np.ndarray) -> None:
        """
        Constructs all the necessary attributes for the Bullet object.

        Parameters
        ----------
            model : MjModel
                The Mujoco model of the environment.
            data : MjData
                The Mujoco data of the environment.
            index : int
                The index of the drone that the bullet belongs to.
            shoot_velocity : float
                The velocity at which the bullet is shot.
            x_bounds : np.ndarray
                The x-axis boundaries of the environment.
            y_bounds : np.ndarray
                The y-axis boundaries of the environment.
            z_bounds : np.ndarray
                The z-axis boundaries of the environment.
        """
        
        self.model = model
        self.data = data
        self.index = index
        self.shoot_velocity = shoot_velocity
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.z_bounds = z_bounds
        self._bullet_body: _MjDataBodyViews = self.data.body(f"drone{self.index}_bullet")
        self._barrel_end: _MjDataSiteViews = self.data.site(f"drone{self.index}_gun_barrel_end")
        self._parent_drone_body: _MjDataBodyViews = self.data.body(f"drone{self.index}")
        self._geom: _MjDataGeomViews = self.data.geom(f"drone{self.index}_bullet_geom")
        self._geom_id: int = self._geom.id
        self.starting_position = self._bullet_body.xpos.copy()
        self._is_flying = False
    
    def reset(self) -> None:
        """
        Resets the bullet's position, velocity, and flying status.
        """
        self._bullet_body.xpos = self.starting_position.copy()
        self._bullet_body.cvel[:] = 0
        self._is_flying = False
    
    def shoot(self) -> None:
        """
        Shoots the bullet if it is not already flying.
        """
        if not self._is_flying:
            aim_direction = self.data.site_xmat[self._barrel_end.id].reshape(3, 3)[:, 0]
            bullet_velocity = aim_direction * self.shoot_velocity + self._parent_drone_body.cvel
            
            self._bullet_body.xpos = self._barrel_end.xpos
            self._bullet_body.cvel = bullet_velocity
            self._is_flying = True
    
    def update(self) -> None:
        """
        Updates the bullet's status. If the bullet is out of bounds, it is reset.
        """
        if self._is_flying and not (self.x_bounds[0] <= self._bullet_body.xpos <= self.x_bounds[1] and
                                    self.y_bounds[0] <= self._bullet_body.ypos <= self.y_bounds[1] and
                                    self.z_bounds[0] <= self._bullet_body.zpos <= self.z_bounds[1]):
            self.reset()
    
    @property
    def geom_id(self):
        """
        Returns the geometry ID of the bullet.

        Returns
        -------
        int
            The geometry ID of the bullet.
        """
        return self._geom_id
    
    @property
    def is_flying(self):
        """
        Returns whether the bullet is currently flying.

        Returns
        -------
        bool
            True if the bullet is flying, False otherwise.
        """
        return self._is_flying


class Drone:
    def __init__(self, model: MjModel, data: MjData, renderer: MujocoRenderer, n_images: int,
                 depth_render: bool, index: int, agent_id: AgentID, spawn_box: np.ndarray, max_spawn_velocity: float,
                 spawn_angle_range: np.ndarray, rng: Generator = default_rng(), map_bounds: np.ndarray = None,
                 bullet_max_velocity: float = 50, height: int = 32, width: int = 32) -> None:
        self.model = model
        self.data = data
        self.index = index
        self.agent_id = agent_id
        self.spawn_box = spawn_box.reshape(2, 3)
        self.max_spawn_velocity = max_spawn_velocity
        self.spawn_angle_range = spawn_angle_range.reshape(2, 3)
        self._camera_1_id: int = self.data.camera(f"drone{int(self.index)}_camera_1").id
        self._camera_2_id: int = self.data.camera(f"drone{int(self.index)}_camera_2").id
        self.n_images = n_images
        self.map_bounds = map_bounds
        self.bullet = Bullet(self.model, self.data, self.index, bullet_max_velocity, *self.map_bounds)
        self.got_hit = self.scored_hit = False
        self.alive = True
        self.depth_render = depth_render
        self.height = height
        self.width = width
        
        self.renderer = renderer
        
        self._image_1: np.ndarray = None
        self._image_2: np.ndarray = None
        
        self._body: _MjDataBodyViews = self.data.body(f"drone{int(self.index)}")
        self._geom: _MjDataGeomViews = self.data.geom(f"drone{int(self.index)}_geom")
        self._gyro: _MjDataSensorViews = self.data.sensor(f"drone{int(self.index)}_imu_gyro")
        self._accelerometer: _MjDataSensorViews = self.data.sensor(f"drone{int(self.index)}_imu_accel")
        self._frame_quat: _MjDataSensorViews = self.data.sensor(f"drone{int(self.index)}_imu_orientation")
        self._actuators: List[_MjDataActuatorViews] = [self.data.actuator(f"drone{int(self.index)}_motor{i}") for i in
                                                       range(1, 5)]
        self._dead_position = self._body.xpos
        self._dead_position[2] = -10
        
        self.initial_quat = self.frame_quaternion
        self.rng = rng
    
    # region Properties
    @property
    def images(self) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
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
        return all(self.map_bounds[:, 0] <= self.position) and all(self.position <= self.map_bounds[:, 1])
    
    @property
    def position(self) -> np.ndarray:
        return self._body.xpos
    
    @property
    def velocity(self) -> np.ndarray:
        return self._body.cvel
    
    @property
    def acceleration(self) -> np.ndarray:
        return self._accelerometer.data
    
    @property
    def orientation(self) -> np.ndarray:
        return self._body.xquat
    
    @property
    def angular_velocity(self) -> np.ndarray:
        return self._gyro.data
    
    @property
    def frame_quaternion(self) -> np.ndarray:
        return self._frame_quat.data
    
    @property
    def motor_velocity(self):
        return np.array([actuator.velocity for actuator in self._actuators])
    
    @property
    def motor_torque(self):
        return np.array([actuator.moment for actuator in self._actuators])
    
    @property
    def motor_controls(self) -> np.ndarray:
        return np.array([actuator.ctrl for actuator in self._actuators])
    
    @property
    def body(self) -> _MjDataBodyViews:
        return self._body
    
    @property
    def geom(self) -> _MjDataGeomViews:
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
        pass
    
    def update(self):
        self.bullet.update()
        if not self.alive:
            self.dead_update()
    # endregion


class BaseMultiAgentEnvironment(MujocoEnv, EzPickle, MultiAgentEnv):
    metadata = {
        'render_modes': ['human', 'rgb_array', 'depth_array'],
        'render_fps': 500   # This is changed later on in the code; declared here to avoid errors
    }
    
    def __init__(self, n_agents: int, n_images: int, depth_render: bool, spawn_boxes: List[np.ndarray],
                 spawn_angles: List[np.ndarray], **kwargs) -> None:
        self.n_agents = n_agents
        
        self.model_path = save_multiagent_model(n_agents, kwargs.get('spacing', 2.0), kwargs.get('save_dir', None))

        height, width = kwargs.get('render_height', 32), kwargs.get('render_width', 32)
        MujocoEnv.__init__(
            self=self,
            model_path=self.model_path,
            frame_skip=kwargs.get('frame_skip', 1),
            observation_space=None,
            render_mode=kwargs.get('render_mode', 'rgb_array'),
            height=height,
            width=width
        )
        self.model: MjModel = self.model
        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }
        self._agent_ids = []
        self.drones = []
        for agent_id, i in enumerate(range(1, n_agents + 1)):
            self._agent_ids.append(agent_id)
            self.drones.append(Drone(
                model=self.model,
                data=self.data,
                renderer=self.mujoco_renderer,
                n_images=n_images,
                depth_render=depth_render,
                index=i,
                agent_id=agent_id,
                spawn_box=spawn_boxes[i - 1],
                max_spawn_velocity=kwargs.get('max_spawn_velocity', 1),
                spawn_angle_range=spawn_angles[i - 1],
                rng=kwargs.get('rng', default_rng()),
                map_bounds=kwargs.get('map_bounds', np.array([[-100, 100], [-100, 100], [0, 100]])),
                bullet_max_velocity=kwargs.get('bullet_max_velocity', 50)
            ))
        
        self._agent_ids = set(self._agent_ids)
        EzPickle.__init__(self)
        
        self._n_agents = n_agents
        self._n_images = n_images
        self._depth_render = depth_render
        self._spawn_boxes = spawn_boxes
        self._spawn_angles = spawn_angles
        _fps = kwargs.get('fps', 30)
        dt = kwargs.get('dt', 0.01)
        time_steps_per_second = int(1.0 / dt)
        self.render_every = time_steps_per_second // _fps
        self.model.opt.timestep = dt
        self.data = MjData(self.model)  # reinitialize data with new model
        self.i = 0
        self._bullet_geom_ids_to_drones = {}
        self._drone_geom_ids_to_drones = {}
        for drone in self.drones:
            self._bullet_geom_ids_to_drones[drone.bullet.geom_id] = drone
            self._drone_geom_ids_to_drones[drone.geom.id] = drone
        self.max_time = kwargs.get('max_time', 100)
        self._action_space_in_preferred_format = True
        self._observation_space_in_preferred_format = True
        
        MultiAgentEnv.__init__(self)
    
    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ) -> Tuple[MultiAgentDict, MultiAgentDict]:
        self.i = 0
        self.reset_model()
        return self.observation(False), self.metrics
    
    def reset_model(self):
        for drone in self.drones:
            drone.reset()
    
    def step(
            self, action: ActType
    ) -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
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
        return dict({
            drone.agent_id: drone.observation(render) for drone in self.drones
        })
    
    @property
    def reward(self) -> MultiAgentDict:
        return dict({
            drone.agent_id: drone.reward for drone in self.drones
        })
    
    @property
    def truncated(self) -> MultiAgentDict:
        # Cases for general truncation: time limit reached, alive drones < 2
        if self.data.time >= self.max_time or sum(drone.alive for drone in self.drones) < 2:
            return MultiAgentDict({drone.agent_id: True for drone in self.drones})
        
        # Cases for individual truncation: drone is dead
        truncations = {}
        for drone in self.drones:
            truncations[drone.agent_id] = not drone.alive
    
    @property
    def done(self) -> MultiAgentDict:
        only_one_alive = sum(drone.alive for drone in self.drones) == 1
        return {drone.agent_id: only_one_alive for drone in self.drones}
    
    @property
    def metrics(self):
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

