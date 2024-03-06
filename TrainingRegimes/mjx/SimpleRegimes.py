from typing import Dict

import jax.numpy
from brax.mjx.base import State as MjxState
from brax import envs
from jax import numpy as jnp, lax
from jax import Array

from TrainingRegimes.mjx.BaseRegime import BaseRegime


class HoverRegime(BaseRegime):
    def __init__(self, scene_file_path: str, target_point: Array,
                 time_penalty: float, accuracy_reward_coefficient: float, **kwargs):
        super().__init__(scene_file_path, **kwargs)
        self.target_point = target_point
        self.time_penalty = time_penalty
        self.accuracy_reward_coefficient = accuracy_reward_coefficient
    
    def calculate_reward(self, mjx_state: MjxState) -> jnp.ndarray:
        def true_fun(_):
            return jax.numpy.clip(
                1.0 / (self.accuracy_reward_coefficient * jnp.linalg.norm(mjx_state.qpos[:3] - self.target_point)),
                0,
                100000
            ) - self.time_penalty * mjx_state.time
        
        def false_fun(_):
            return -self.time_penalty * 1000
        
        condition = mjx_state.time > 10
        reward = lax.cond(condition, true_fun, false_fun, None)
        
        return reward
    
    def is_simulation_done(self, mjx_state: MjxState) -> jnp.ndarray:
        pos = mjx_state.qpos[:3]
        condition_1 = jnp.linalg.norm(pos - self.target_point) < 0.1
        condition_2 = mjx_state.time > 10
        return jnp.logical_or(condition_1, condition_2)
    
    def get_simulation_metrics(self, mjx_state: MjxState) -> Dict[str, jnp.ndarray]:
        pos = mjx_state.qpos[:3]
        vel = mjx_state.qvel[:3]
        return {
            'distance to target': jnp.linalg.norm(pos - self.target_point),
            'velocity': jnp.linalg.norm(vel),
        }


envs.register_environment('HoverRegime', HoverRegime)
