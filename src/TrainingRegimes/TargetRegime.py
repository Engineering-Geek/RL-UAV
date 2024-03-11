from typing import Any

from numpy.linalg import norm

from src.TrainingRegimes.BaseRegime import BaseRegime


class TargetRegime(BaseRegime):
    """
    A regime that defines the behavior and objectives for a drone in a target-reaching scenario within a simulation environment.
    It extends BaseRegime with specific reward and penalty mechanisms related to reaching a target, maintaining velocity,
    and avoiding crashes.

    :param tolerance_distance: The distance within which the drone is considered to have reached the target. Type: float
    :param max_time: The maximum duration for which the simulation or episode is allowed to run. If exceeded, the episode
                     is considered truncated. Type: float
    :param reward_distance_coefficient: Coefficient for the reward based on the inverse of the distance to the target.
                                        The reward increases as the drone gets closer to the target. Type: float
    :param reward_distance_exp: The exponent applied to the inverse distance reward calculation. Type: float
    :param reward_distance_max: The maximum reward granted for distance to the target. Prevents the reward from becoming
                                excessively large as the distance approaches zero. Type: float
    :param reward_goal: The reward given when the drone reaches the target. Type: float
    :param reward_velocity_coefficient: Coefficient for the reward based on the drone's velocity. Type: float
    :param reward_velocity_exp: The exponent applied to the velocity in the velocity-based reward calculation. Type: float
    :param reward_velocity_max: The maximum reward granted for the drone's velocity. Type: float
    :param penalty_time: The penalty applied at each timestep, encouraging the drone to reach the target faster. Type: float
    :param penalty_crash: The penalty applied if the drone crashes, i.e., hits the ground. Type: float
    :param kwargs: Additional keyword arguments passed to the base class (BaseRegime).
    """

    def __init__(self, tolerance_distance: float, max_time: float,
                 reward_distance_coefficient: float, reward_distance_exp: float, reward_distance_max: float,
                 reward_goal: float, reward_velocity_coefficient: float, reward_velocity_exp: float,
                 reward_velocity_max: float, penalty_time: float, penalty_crash: float, **kwargs):
        """
        Initializes the TargetRegime with specific parameters related to the target-reaching task.

        :param tolerance_distance: The threshold distance within which the target is considered as reached.
        :param max_time: The time limit for the episode after which it is truncated.
        :param reward_distance_coefficient: Multiplier for the distance-based reward.
        :param reward_distance_exp: Exponent for scaling the distance-based reward.
        :param reward_distance_max: Cap for the distance-based reward to prevent infinite values.
        :param reward_goal: Reward for reaching the target.
        :param reward_velocity_coefficient: Multiplier for the velocity-based reward.
        :param reward_velocity_exp: Exponent for scaling the velocity-based reward.
        :param reward_velocity_max: Maximum possible velocity-based reward.
        :param penalty_time: Time penalty incurred each timestep to encourage faster completion.
        :param penalty_crash: Penalty for crashing, which discourages collision with the ground.
        :param kwargs: Additional arguments passed to the BaseRegime initializer.
        """
        super().__init__(**kwargs)
        self._tolerance_distance = float(tolerance_distance)
        self._max_time = float(max_time)
        self._reward_distance_coefficient = float(reward_distance_coefficient)
        self._reward_distance_exp = float(reward_distance_exp)
        self._reward_distance_max = float(reward_distance_max)
        self._reward_goal = float(reward_goal)
        self._reward_velocity_coefficient = float(reward_velocity_coefficient)
        self._reward_velocity_exp = float(reward_velocity_exp)
        self._reward_velocity_max = float(reward_velocity_max)
        self._penalty_time = float(penalty_time)
        self._penalty_crash = float(penalty_crash)
    
    @property
    def reward_distance(self) -> float:
        """
        Calculate the distance-based reward, which is inversely proportional to the distance between the drone and the target.
        The reward is capped by `reward_distance_max` to prevent it from becoming infinitely large as the distance approaches zero.

        :return: The distance-based reward, scaled and exponentiated based on the distance to the target, with a maximum limit.
        """
        try:
            return min((self._reward_distance_coefficient /
                        norm(self.drone_target_vector)) ** self._reward_distance_exp,
                       self._reward_distance_max)
        except ZeroDivisionError:
            return self._reward_distance_max
    
    @property
    def reward_time(self) -> float:
        """
        Calculate the time-based penalty at each timestep to encourage the drone to reach its target faster.

        :return: The time-based penalty, which is a constant negative value defined by `penalty_time`.
        """
        return -self._penalty_time
    
    @property
    def reward_crash(self) -> float:
        """
        Calculate the crash penalty, applied if the drone hits the ground.

        :return: The crash penalty if the drone hits the ground; otherwise, zero.
        """
        return -self._penalty_crash if self.drone_hit_ground else 0.
    
    @property
    def reward_goal(self) -> float:
        """
        Calculate the reward for reaching the goal. This reward is granted when the drone's distance to the target is
        less than or equal to `tolerance_distance`.

        :return: The reward for reaching the goal if the goal is reached; otherwise, zero.
        """
        return self._reward_goal if self.goal_reached else 0.
    
    @property
    def reward_velocity(self) -> float:
        """
        Calculate the velocity-based reward, which is proportional to the drone's current velocity.

        :return: The velocity-based reward, scaled and exponentiated based on the drone's current velocity, with a maximum limit.
        """
        return (self._reward_velocity_coefficient * (norm(self.drone.velocity))
                ** self._reward_velocity_exp)
    
    @property
    def reward(self) -> float:
        """
        Aggregate the total reward for the current timestep, combining distance, time, crash, goal-reaching, and velocity-based rewards.

        :return: The total reward for the current timestep.
        """
        return (self.reward_distance + self.reward_goal + self.reward_velocity
                - self._penalty_time - self._penalty_crash)
    
    @property
    def metrics(self) -> dict[str, Any]:
        """
        Collect and return various metrics related to the drone's performance and the environment state.

        :return: A dictionary containing metrics such as distance to the target, goal-reaching status, drone's velocity,
                 whether the drone hit the ground, and the simulation time.
        """
        return {
            "distance": norm(self.drone_target_vector),
            "goal_reached": self.goal_reached,
            "goal_velocity": norm(self.drone.velocity),
            "hit_ground": self.drone_hit_ground,
            "time": self.data.time
        }
    
    @property
    def done(self) -> bool:
        """
        Determine whether the episode has concluded. An episode is considered done if the drone crashes, reaches the target,
        or the simulation time exceeds `max_time`.

        :return: True if the episode is done; otherwise, False.
        """
        return bool(self.drone_hit_ground or self.truncated or self.goal_reached)
    
    @property
    def truncated(self) -> bool:
        """
        Check if the episode is truncated due to exceeding the maximum allowed simulation time (`max_time`).

        :return: True if the simulation time exceeds `max_time`; otherwise, False.
        """
        return bool(self.data.time > self._max_time)
    
    @property
    def goal_reached(self) -> bool:
        """
        Determine whether the drone has reached the target by checking if its distance to the target is less than or equal
        to `tolerance_distance`.

        :return: True if the drone has reached the target; otherwise, False.
        """
        return bool(norm(self.drone_target_vector) < self._tolerance_distance)
