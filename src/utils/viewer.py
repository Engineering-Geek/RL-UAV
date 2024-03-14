import numpy as np
from mujoco import viewer
from src.Environments.multi_agent.BaseMultiAgentEnvironment import BaseMultiAgentEnvironment


def launch_multi_agent_environment(env: BaseMultiAgentEnvironment, duration: int = None, limp: bool = False):
    # duration is in seconds
    env.reset()
    continue_condition = True if not duration else env.data.time < duration
    with viewer.launch_passive(env.model, env.data) as v:
        while continue_condition:
            if limp:
                action = {{
                    "motor": np.array([0.0, 0.0, 0.0]),
                    "shoot": np.array([0]),
                } for _ in range(env.n_agents)}
            else:
                action = env.action_space_sample()
            env.step(action)
            v.sync()
