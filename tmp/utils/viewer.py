from time import time, sleep

import numpy as np
from mujoco import viewer

from tmp.Environments import MultiAgentDroneEnvironment
from src.utils.multiagent_model_generator import multiagent_model
from ray.rllib.env.multi_agent_env import MultiAgentDict


def launch_multi_agent_environment(env: MultiAgentDroneEnvironment, duration: int = None, limp: bool = False,
                                   set_action: MultiAgentDict = None):
    """
    Launches and visualizes a multi-agent environment simulation using MuJoCo's passive viewer.

    This function steps through the environment, applying actions to each agent and updating the viewer at each
    timestep. The simulation can be run in different modes based on the parameters: normal (random actions),
    limp (no movement), or with a predetermined set of actions.

    Parameters
    ----------
    env : MultiAgentDroneEnvironment
        The multi-agent MuJoCo environment to be launched. The environment should be configured prior to calling this
        function.
    duration : int, optional
        The duration in simulation seconds for which the environment should run. If None, the simulation will run
        indefinitely. Default is None.
    limp : bool, optional
        If True, all agents will receive a zero action (no movement). If False, agents will act based on `set_action` or
        sample from their action space if `set_action` is None. Default is False.
    set_action : MultiAgentDict, optional
        A dictionary specifying actions for each agent. If provided, these actions are applied to the agents at each
        timestep. If None, actions are sampled from the environment's action space. Default is None.

    Notes
    -----
    - The viewer is launched in passive mode, meaning it only visualizes the environment without allowing user
        interaction.
    - The function supports three modes of operation: using `set_action` to apply a specific action to each agent,
        `limp` mode where no actions are applied, and normal mode where actions are sampled from the environment's
        action space.

    """
    env.reset()
    continue_condition = True if not duration else env.time < duration
    with viewer.launch_passive(env.model, env.data) as v:
        while continue_condition:
            start = time()
            if set_action:
                action = set_action
            elif limp:
                action = {
                    agent_id: {
                        "motor": np.array([0.0, 0.0, 0.0, 0.0]),
                        "shoot": np.array([1]),
                    } for agent_id in env.get_agent_ids()
                }
            else:
                action = env.action_space_sample()
            env.step(action)
            end = time()
            sleep(max(0, env.dt - (end - start)))
            v.sync()


def launch_multi_agent():
    model = multiagent_model(n_drones=2)
    viewer.launch(model)


if __name__ == "__main__":
    # env = MultiAgentDroneEnvironment(n_agents=1, n_images=0, depth_render=False, Drone=SimpleDrone)
    # launch_multi_agent_environment(env, duration=1000, limp=True)
    launch_multi_agent()
