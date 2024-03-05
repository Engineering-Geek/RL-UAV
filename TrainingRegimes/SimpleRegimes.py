import jax

from TrainingRegimes.BaseRegime import BaseRegime
from brax.envs.base import State as BraxState


class HoverRegime(BaseRegime):
    """
    Simple hover regime. This regime is used to train the agent to hover in place.
    """

    def calculate_reward(self, brax_state):
        """
        Calculates the reward for the given state.
        In this case, the reward is higher the closer the drone is to the target altitude.

        Args:
            brax_state: The current state of the Brax environment.

        Returns:
            The calculated reward.
        """
        target_altitude = 0.3
        current_altitude = brax_state.qp.pos[2]
        reward = -abs(target_altitude - current_altitude)
        return reward

    def is_simulation_done(self, brax_state):
        """
        Checks if the simulation is done for the given state.
        In this case, the simulation is done if the drone has fallen to the ground.

        Args:
            brax_state: The current state of the Brax environment.

        Returns:
            A boolean indicating whether the simulation is done.
        """
        return brax_state.qp.pos[2] <= 0

    def get_simulation_metrics(self, brax_state):
        """
        Gets the simulation metrics for the given state.
        In this case, the metrics are the current altitude and the target altitude.

        Args:
            brax_state: The current state of the Brax environment.

        Returns:
            A dictionary of simulation metrics.
        """
        metrics = {
            'current_altitude': brax_state.qp.pos[2],
            'target_altitude': 0.3
        }
        return metrics


def test_hover_regime():
    # Initialize a random number generator
    rng = jax.random.PRNGKey(0)

    # Create an instance of HoverRegime
    hover_regime = HoverRegime('../models/skydio_x2/scene.xml')

    # Reset the simulation
    initial_state = hover_regime.reset(rng)

    # Perform a step
    action = jax.random.uniform(rng, (hover_regime.sys.nu,))
    new_state = hover_regime.step(initial_state, action)

    # Check if the returned state is an instance of BraxState
    assert isinstance(new_state, BraxState), "The returned state is not an instance of BraxState"

    print("HoverRegime test passed successfully.")


# Run the test
test_hover_regime()
