import numpy as np
from mujoco import viewer
from src.Environments.multi_agent.BaseMultiAgentEnvironment import BaseMultiAgentEnvironment
from ray.rllib.env.multi_agent_env import MultiAgentDict


def launch_multi_agent_environment(env: BaseMultiAgentEnvironment, duration: int = None, limp: bool = False,
                                   set_action: MultiAgentDict = None):
    # duration is in seconds
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
            # print(id(env.data))
            v.sync()
