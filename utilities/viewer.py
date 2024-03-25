from time import sleep

from mujoco import viewer, MjModel, MjData
from environments.core import MultiAgentDroneEnvironment
from environments.SimpleHoverDroneEnv import SimpleHoverDroneEnv
from ray.rllib.env.env_context import EnvContext


def view_environment(environment: MultiAgentDroneEnvironment, n: int = -1):
    """
    View the environment in a Mujoco viewer.

    Args:
        environment (MultiAgentDroneEnvironment): Environment to view.
        n (int): Number of steps to run the environment for. If -1, the environment will run indefinitely.
    """
    with viewer.launch_passive(environment.model, environment.data) as handler:
        while n != 0 and handler.is_running():
            o, r, d, t, i = environment.step(environment.action_space.sample())
            handler.sync()
            n -= 1


if __name__ == "__main__":
    context = EnvContext(
        env_config={
            "n_agents": 1,
        },
        worker_index=0,
    )
    env = SimpleHoverDroneEnv(config=context)
    view_environment(env)
