import numpy as np
from mujoco import viewer
from src.Environments.multi_agent.BaseMultiAgentEnvironment import BaseMultiAgentEnvironment
from ray.rllib.env.multi_agent_env import MultiAgentDict


def launch_multi_agent_environment(env: BaseMultiAgentEnvironment, duration: int = None, limp: bool = False,
                                   set_action: MultiAgentDict = None):
    """
    Launches and visualizes a multi-agent environment simulation using MuJoCo's passive viewer.

    This function steps through the environment, applying actions to each agent and updating the viewer at each timestep. The simulation can be run in different modes based on the parameters: normal (random actions), limp (no movement), or with a predetermined set of actions.

    Parameters
    ----------
    env : BaseMultiAgentEnvironment
        The multi-agent MuJoCo environment to be launched. The environment should be configured prior to calling this function.
    duration : int, optional
        The duration in simulation seconds for which the environment should run. If None, the simulation will run indefinitely. Default is None.
    limp : bool, optional
        If True, all agents will receive a zero action (no movement). If False, agents will act based on `set_action` or sample from their action space if `set_action` is None. Default is False.
    set_action : MultiAgentDict, optional
        A dictionary specifying actions for each agent. If provided, these actions are applied to the agents at each timestep. If None, actions are sampled from the environment's action space. Default is None.

    Notes
    -----
    - The viewer is launched in passive mode, meaning it only visualizes the environment without allowing user interaction.
    - The function supports three modes of operation: using `set_action` to apply a specific action to each agent, `limp` mode where no actions are applied, and normal mode where actions are sampled from the environment's action space.

    Examples
    --------
    >>> env = BaseMultiAgentEnvironment(n_agents=2, n_images=2, depth_render=False)
    >>> launch_multi_agent_environment(env, duration=1000, limp=True)
    This will launch the environment with all agents not moving for 1000 simulation seconds.

    """
    env.reset()
    continue_condition = True if not duration else env.data.time < duration
    with viewer.launch_passive(env.model, env.data) as v:
        while continue_condition:
            if set_action:
                action = set_action
            elif limp:
                action = {
                    agent_id: {
                        "motor": np.array([0.0, 0.0, 0.0]),
                        "shoot": np.array([0]),
                    } for agent_id in env.get_agent_ids()
                }
            else:
                action = env.action_space_sample()
            env.step(action)
            v.sync()

