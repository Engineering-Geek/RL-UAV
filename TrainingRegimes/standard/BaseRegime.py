from abc import ABC, abstractmethod
from typing import Dict, Sequence, Tuple, List, Union
from numpy import ndarray
from numpy.random import default_rng
from mujoco import MjModel, MjData, Renderer, mj_step
from mujoco._structs import _MjDataBodyViews
import numpy as np
import quaternion


class State:
    """
    A class to represent the state of the environment at a given time step.
    """
    
    def __init__(self, data: MjModel, observation: Sequence[ndarray],
                 reward: float, done: bool, metrics: Dict[str, float]):
        """
        Constructor for the State class.
        :param data: The MjModel data representing the current state of the environment.
        :param observation: The current observation of the environment.
        :param reward: The reward obtained in the current state.
        :param done: A boolean indicating whether the episode is done.
        :param metrics: A dictionary containing additional information about the current state.
        """
        self.data = data
        self.observation = observation
        self.reward = reward
        self.done = done
        self.metrics = metrics
        self.historical_data = []
        self.observations = []
    
    def append_historical_data(self):
        """
        Appends the current state of the environment to the historical data.
        """
        self.historical_data.append(self.data.__copy__())
    
    def append_observation(self):
        """
        Appends the current observation to the list of observations.
        """
        self.observations.append(self.observation)


class BaseRegime(ABC):
    """
    An abstract base class for different training regimes.
    """
    
    def __init__(self, drone_xml_path, **kwargs):
        """
        Constructor for the BaseRegime class.
        :param drone_xml_path: The path to the XML file defining the drone model.
        :param kwargs: Optional parameters for the Renderer.
        """
        self.model: MjModel = MjModel.from_xml_path(drone_xml_path)
        self.renderer = Renderer(self.model, kwargs.get('width', 64), kwargs.get('height', 64))
        self._rng = default_rng()
        drone_alpha_range = kwargs.get('alpha_range', (-np.pi, np.pi))
        drone_beta_range = kwargs.get('beta_range', (-np.pi, np.pi))
        drone_gamma_range = kwargs.get('gamma_range', (-np.pi, np.pi))
        drone_x_range = np.array(kwargs.get('x_range', (-10., 10.)))
        drone_y_range = np.array(kwargs.get('y_range', (-10., 10.)))
        drone_z_range = np.array(kwargs.get('z_range', (.2, 10.)))
        self.drone_xpos_range = np.array([drone_x_range, drone_y_range, drone_z_range])
        self.drone_euler_range = np.array([drone_alpha_range, drone_beta_range, drone_gamma_range])
        target_x_range = np.array(kwargs.get('target_x_range', (-10., 10.)))
        target_y_range = np.array(kwargs.get('target_y_range', (-10., 10.)))
        target_z_range = np.array(kwargs.get('target_z_range', (.2, 10.)))
        self.target_xpos_range = np.array([target_x_range, target_y_range, target_z_range])
        self._frame_1 = np.zeros((self.renderer.width, self.renderer.height, 3), dtype=np.uint8)
        self._frame_2 = np.zeros((self.renderer.width, self.renderer.height, 3), dtype=np.uint8)
        self._append_historical_data = kwargs.get('append_historical_data', False)
        self._append_observations = kwargs.get('append_observations', False)
        self.drone_name = kwargs.get('drone_name', 'drone')
        self.target_name = kwargs.get('target_name', 'target')
        self.camera_1_name = kwargs.get('camera_1_name', 'camera_1')
        self.camera_2_name = kwargs.get('camera_2_name', 'camera_2')
        self.max_sim_time = kwargs.get('max_sim_time', 10.0)
        _sample_data = MjData(self.model)
        self.action_space = self.model.nu
        observation = self.observe(_sample_data)
        if isinstance(observation, ndarray):
            self.observation_space = observation.shape
        else:
            self.observation_space = [item.shape for item in observation]
    
    def _generate_random_drone_quat(self):
        """
        Generates a random quaternion for the drone's orientation.
        """
        alpha, beta, gamma = [self._rng.uniform(*angle_range) for angle_range in self.drone_euler_range]
        return quaternion.as_float_array(quaternion.from_euler_angles(alpha, beta, gamma))
    
    def _reposition_drone(self, data: MjData) -> None:
        """
        Repositions the drone to a random position and orientation.
        :param data: The MjModel data representing the current state of the environment.
        """
        drone_body_view: _MjDataBodyViews = data.body(self.drone_name)
        drone_body_view.xpos = np.array([self._rng.uniform(*range_) for range_ in self.drone_xpos_range])
        drone_body_view.xquat = self._generate_random_drone_quat()
    
    def _reposition_target(self, data: MjData) -> None:
        """
        Repositions the target to a random position.
        :param data: The MjModel data representing the current state of the environment.
        """
        target_body_view: _MjDataBodyViews = data.body(self.target_name)
        target_body_view.xpos = np.array([self._rng.uniform(*range_) for range_ in self.target_xpos_range])
    
    def reset(self, seed: int = -1) -> State:
        """
        Resets the regime with an optional seed.
        :param seed: The seed for the random number generator. If -1, a new random number generator is created.
        :return: The MjModel data representing the initial state of the environment.
        """
        if seed != -1:
            self._rng = np.random.Generator(np.random.PCG64(seed))
        else:
            self._rng = np.random.default_rng()
        data = MjData(self.model)
        self._reposition_drone(data)
        self._reposition_target(data)
        return State(data, self.observe(data), self.reward(data), self.done(data), self.metrics(data))
    
    @abstractmethod
    def done(self, data: MjData) -> bool:
        """
        Checks if the current episode is done.
        :param data: The MjModel data representing the current state of the environment.
        :return: A boolean indicating whether the episode is done.
        """
        raise NotImplementedError
    
    @abstractmethod
    def reward(self, data: MjData) -> float:
        """
        Calculates the reward for the current state.
        :param data: The MjModel data representing the current state of the environment.
        :return: The reward for the current state.
        """
        raise NotImplementedError
    
    @abstractmethod
    def observe(self, data: MjData) -> Union[List[ndarray], ndarray]:
        """
        Observes the current state.
        :param data: The MjModel data representing the current state of the environment.
        :return: The observation of the current state.
        """
        raise NotImplementedError
    
    @abstractmethod
    def metrics(self, data: MjData) -> Dict[str, float]:
        """
        Calculates the metrics for the current state.
        :param data: The MjModel data representing the current state of the environment.
        :return: A dictionary containing the metrics for the current state.
        """
        raise NotImplementedError
    
    def render(self, data: MjData) -> Tuple[ndarray, ndarray]:
        """
        Renders the current state.
        :param data: The MjModel data representing the current state of the environment.
        :return: Two frames representing the rendered state.
        """
        self.renderer.update_scene(data, self.camera_1_name)
        self.renderer.render(out=self._frame_1)
        self.renderer.update_scene(data, self.camera_2_name)
        self.renderer.render(out=self._frame_2)
        return self._frame_1, self._frame_2
    
    def step(self, state: State, action: np.ndarray) -> State:
        """
        Performs a step in the regime with the given action.
        :param state: The current state.
        :param action: The action to perform.
        :return: The new state after performing the action.
        """
        state.data.ctrl = action
        mj_step(self.model, state.data)
        state.observation = self.observe(state.data)
        state.done = self.done(state.data)
        state.reward = self.reward(state.data)
        state.metrics = self.metrics(state.data)
        if self._append_historical_data:
            state.append_historical_data()
        if self._append_observations:
            state.append_observation()
        return state


class _TestRegime(BaseRegime):
    """
    A simple test regime for debugging purposes.
    """
    
    def done(self, data: MjData) -> bool:
        """
        Checks if the current episode is done.
        :param data: The MjModel data representing the current state of the environment.
        :return: A boolean indicating whether the episode is done.
        """
        return False
    
    def reward(self, data: MjData) -> float:
        """
        Calculates the reward for the current state.
        :param data: The MjModel data representing the current state of the environment.
        :return: The reward for the current state.
        """
        return 0
    
    def observe(self, data: MjData) -> ndarray:
        """
        Observes the current state.
        :param data: The MjModel data representing the current state of the environment.
        :return: The observation of the current state.
        """
        self.render(data)
        return np.zeros(3)
    
    def metrics(self, data: MjData) -> Dict[str, float]:
        """
        Calculates the metrics for the current state.
        :param data: The MjModel data representing the current state of the environment.
        :return: A dictionary containing the metrics for the current state.
        """
        return {'test_metric': 0}
    

def test_regime():
    regime = _TestRegime(
        drone_xml_path='../../models/UAV/scene.xml',
    )
    state = regime.reset()
    for _ in range(10):
        state = regime.step(state, np.zeros(4))
        print(state.reward)
