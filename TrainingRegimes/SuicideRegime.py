from typing import SupportsFloat, Any

import pytest
from numpy.linalg import norm

from TrainingRegimes._BaseRegime import _BaseRegime


class SuicideDroneRegime(_BaseRegime):
    """
    This class represents the environment for a suicide drone. It inherits from the _BaseDroneEnv class.
    The drone's goal is to reach a target. The simulation ends when the drone hits the ground,
    the time exceeds the maximum time, or the drone reaches the goal.
    The drone receives rewards based on its distance to the target, its velocity, and whether it has reached the goal.
    It receives penalties for time and crashing.
    """
    
    def __init__(self, tolerance_distance: float, max_time: float,
                 reward_distance_coefficient: float, reward_distance_exp: float, reward_distance_max: float,
                 reward_goal: float, reward_velocity_coefficient: float, reward_velocity_exp: float,
                 penalty_time: float, penalty_crash: float, **kwargs):
        """
        Initialize the SuicideDroneRegime.

        :param tolerance_distance: The maximum distance from the target to end the simulation
        :param max_time: The longest the simulation can go on for (seconds)
        :param reward_distance_coefficient: Coefficient for the reward based on the distance to the target.
        :param reward_distance_exp: Exponent for the reward based on the distance to the target.
        :param reward_distance_max: Maximum reward based on the distance to the target.
        :param reward_goal: Reward for reaching the goal.
        :param reward_velocity_coefficient: Coefficient for the reward based on the drone's velocity.
        :param reward_velocity_exp: Exponent for the reward based on the drone's velocity.
        :param penalty_time: Penalty for time.
        :param penalty_crash: Penalty for crashing.
        :param kwargs: Additional arguments for the base class.
        """
        super().__init__(**kwargs)
        self._tolerance_distance = tolerance_distance
        self._max_time = max_time
        self._reward_distance_coefficient = reward_distance_coefficient
        self._reward_distance_exp = reward_distance_exp
        self._reward_distance_max = reward_distance_max
        self._reward_goal = reward_goal
        self._reward_velocity_coefficient = reward_velocity_coefficient
        self._reward_velocity_exp = reward_velocity_exp
        self._penalty_time = penalty_time
        self._penalty_crash = penalty_crash
    
    @property
    def reward_distance(self) -> float:
        """
        Calculate the reward based on the distance to the target.
        If the norm of the drone target vector is 0, return the maximum reward.
        """
        try:
            return min((self._reward_distance_coefficient /
                        norm(self.drone_target_vector)) ** self._reward_distance_exp,
                       self._reward_distance_max)
        except ZeroDivisionError:
            return self._reward_distance_max
    
    @property
    def reward_time(self) -> float:
        return -self._penalty_time
    
    @property
    def reward_crash(self) -> float:
        return -self._penalty_crash if self.drone_hit_ground else 0
    
    @property
    def reward_goal(self) -> float:
        return self._reward_goal if self.goal_reached else 0
    
    @property
    def reward_velocity(self) -> float:
        return (self._reward_velocity_coefficient * (norm(self.drone.velocity))
                ** self._reward_velocity_exp)
    
    @property
    def reward(self) -> float:
        return (self.reward_distance + self.reward_goal + self.reward_velocity
                - self._penalty_time - self._penalty_crash)
    
    @property
    def metrics(self) -> dict[str, Any]:
        return {
            "distance": norm(self.drone_target_vector),
            "goal_reached": self.goal_reached,
            "goal_velocity": norm(self.drone.velocity),
            "hit_ground": self.drone_hit_ground,
            "time": self.data.time
        }
    
    @property
    def done(self) -> bool:
        return self.drone_hit_ground or self.truncated or self.goal_reached
    
    @property
    def truncated(self) -> bool:
        return self.data.time > self._max_time
    
    @property
    def goal_reached(self) -> bool:
        return norm(self.drone_target_vector) < self._tolerance_distance
