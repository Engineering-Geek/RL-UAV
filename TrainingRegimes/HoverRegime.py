from numpy.linalg import norm

from SuicideRegime import SuicideRegime


class HoverRegime(SuicideRegime):
    """
    This class represents the environment for a hover drone. It inherits from the _BaseDroneEnv class.
    The drone's goal is to hover in place. The simulation ends when the drone hits the ground,
    the time exceeds the maximum time, or the drone reaches the goal.
    The drone receives rewards based on its distance to the target, its velocity, and whether it has reached the goal.
    It receives penalties for time and crashing into the floor.
    """
    
    def __init__(self, tolerance_distance: float, max_time: float,
                 reward_distance_coefficient: float, reward_distance_exp: float, reward_distance_max: float,
                 reward_goal: float, reward_velocity_coefficient: float, reward_velocity_exp: float,
                 penalty_time: float, penalty_crash: float, **kwargs):
        """
        Initialize the HoverRegime.

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
        super().__init__(tolerance_distance, max_time, reward_distance_coefficient, reward_distance_exp,
                         reward_distance_max, reward_goal, reward_velocity_coefficient, reward_velocity_exp,
                         penalty_time, penalty_crash, **kwargs)
    
    def reward_velocity(self) -> float:
        """
        Calculate the reward based on the drone's velocity.
        The reward increases as the drone's velocity decreases, unlike the SuicideRegime.
        """
        return (self._reward_velocity_coefficient * (norm(self.drone.velocity))
                ** -self._reward_velocity_exp)
