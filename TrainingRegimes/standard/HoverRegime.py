from typing import Tuple, List, Dict, Union

from mujoco import MjData, mj_step
import numpy as np
from numpy import ndarray

from TrainingRegimes.standard.BaseRegime import BaseRegime, State


class HoverRegime(BaseRegime):
    def __init__(self, drone_xml_path, **kwargs):
        super().__init__(drone_xml_path, **kwargs)
        self.time_penalty = kwargs.get('time_penalty', 0.01)
        self.crash_penalty = kwargs.get('crash_penalty', 100.0)
        self.motor_power_penalty = kwargs.get('motor_power_penalty', 0.01)
        self.distance_to_target_reward = kwargs.get('distance_to_target_reward', 100.0)
        self.min_distance_to_target = kwargs.get('min_distance_to_target', 0.001)
        self.max_distance_to_target = kwargs.get('max_distance_to_target', 100.0)
        
    def vector_to_target(self, data: MjData) -> ndarray:
        drone_xpos = data.body(self.drone_name).xpos
        target_xpos = data.body(self.target_name).xpos
        return target_xpos - drone_xpos
        
    def distance_to_target(self, data: MjData) -> float:
        return np.clip(np.linalg.norm(self.vector_to_target(data)),
                       self.min_distance_to_target, self.max_distance_to_target)
        
    def reward(self, data: MjData) -> float:
        reward = 0.0
        reward -= self.time_penalty
        reward -= self.motor_power_penalty * data.ctrl @ data.ctrl
        reward += self.distance_to_target_reward / self.distance_to_target(data)
        if data.ncon > 0:
            reward -= self.crash_penalty
        return reward
    
    def done(self, data: MjData) -> bool:
        if data.ncon > 0 and any(data.contact[i].geom1 == 0 or data.contact[i].geom2 == 0 for i in range(data.ncon)):
            return True
        return data.time >= self.max_sim_time
    
    def observe(self, data: MjData) -> Union[List[ndarray], ndarray]:
        return np.concatenate([data.xpos, data.qpos, data.xvel, data.qvel, data.ctrl])
    
    def metrics(self, data: MjData) -> Dict[str, float]:
        return {
            'distance_to_target': self.distance_to_target(data),
            'time': data.time,
            'current position': data.xpos,
        }


def test_hover_regime():
    regime = HoverRegime('../../models/UAV/scene.xml')
    state = regime.reset()
    state.append_historical_data()
    state.append_observation()
    while not state.done:
        state = regime.step(state, np.array([0.0, 0.0, 0.0, 0.0]))
        state.append_historical_data()
        state.append_observation()
    print(state.reward)
    print(state.done)
    print(state.metrics)
    print(len(state.historical_data))
    print(len(state.observations))

