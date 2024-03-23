from environments.core import MultiAgentDroneEnvironment
from environments.SimpleDroneEnvs import SimpleHoverDroneEnv, SimpleShootingDroneEnv
import pytest


@pytest.fixture
def simple_shooting_drone_outputs():
    return step_output(SimpleShootingDroneEnv())


@pytest.fixture
def simple_hover_drone_outputs():
    return step_output(SimpleHoverDroneEnv())


def step_output(env: MultiAgentDroneEnvironment, n: int = 1000):
    actions = [env.action_space_sample() for _ in range(n)]
    observations = []
    rewards = []
    dones = []
    truncateds = []
    infos = []
    
    for action in actions:
        obs, reward, truncated, done, info = env.step(action)
        observations.append(obs)
        rewards.append(reward)
        dones.append(done)
        truncateds.append(truncated)
        infos.append(info)
        
    return observations, rewards, dones, truncateds, infos


# def observation_space_match(observation_space, observations):
#     """
#     Check if the observations match the observation space.
#     Things to check for:
#         - Shape
#         - Type
#         - Range
#     """
#     sample_observation
#     for observation in observations:
#         for agent_id in observation.keys():
#             # Each observation for each agent should be a dictionary
#             assert isinstance(observation[agent_id], dict)
#             for sensor in observation[agent_id].keys():
#                 pass
