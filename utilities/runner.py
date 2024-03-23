import time
from copy import deepcopy
from typing import Hashable

from tqdm import tqdm

from environments.SimpleDroneEnvs import SimpleHoverDroneEnv
from environments.core import MultiAgentDroneEnvironment
import logging
logging.basicConfig(level=logging.INFO)


def run(environment: MultiAgentDroneEnvironment, num_steps: int = 1000):
    environment.reset()
    for _ in range(num_steps):
        action = environment.action_space_sample()
        obs, reward, done, truc, info = environment.step(action)
        environment.render()
        time.sleep(environment.model.opt.timestep)
    environment.close()


def get_readings(environment: MultiAgentDroneEnvironment, num_steps: int = 1000):
    observations, rewards, dones, truncs, infos = [], [], [], [], []
    observation, info = environment.reset()
    observations.append(observation)
    
    for _ in tqdm(range(num_steps), desc="Running environment..."):
        action = environment.action_space_sample()
        observation, reward, done, truc, info = environment.step(action)
        observations.append(observation)
        rewards.append(reward)
        dones.append(done)
        truncs.append(truc)
        infos.append(info)
    environment.close()
    return observations, rewards, dones, truncs, infos


# @pytest.fixture
# def environment() -> MultiAgentDroneEnvironment:
#     return SimpleHoverDroneEnv(render_mode=None)


def _test_environment(environment: MultiAgentDroneEnvironment):
    observations, rewards, dones, truncs, infos = get_readings(environment, 1000)
    
    # Check the initial reward structure
    assert isinstance(rewards[0], dict), "Initial rewards must be a dict"
    for reward in tqdm(rewards, desc="Checking rewards"):
        assert isinstance(reward, dict), "Each reward must be a dict"
        for agent_id, value in reward.items():
            assert isinstance(agent_id, Hashable), f"Agent ID {agent_id} is not hashable"
            assert isinstance(value, float), f"Value for agent_id {agent_id} is not of type float"
    
    # Check the initial done structure
    assert isinstance(dones[0], dict), "Initial dones must be a dict"
    for done in tqdm(dones, desc="Checking dones"):
        assert isinstance(done, dict), "Each done must be a dict"
        for agent_id, value in done.items():
            assert isinstance(value, bool), f"Value for agent_id {agent_id} is not of type bool"
    # All dones must have a __all__ key
    assert all("__all__" in done.keys() for done in dones), "All dones must have a __all__ key"
    
    # Check the initial trunc structure
    assert isinstance(truncs[0], dict), "Initial truncs must be a dict"
    for trunc in tqdm(truncs, desc="Checking truncs"):
        assert isinstance(trunc, dict), "Each trunc must be a dict"
        for agent_id, value in trunc.items():
            assert isinstance(value, bool), f"Value for agent_id {agent_id} is not of type bool"
    # All truncs must have a __all__ key
    assert all("__all__" in trunc.keys() for trunc in truncs), "All truncs must have a __all__ key"
    
    # Check the initial info structure
    assert isinstance(infos[0], dict), "Initial infos must be a dict"
    for info in tqdm(infos, desc="Checking infos"):
        assert isinstance(info, dict), "Each info must be a dict"
        for agent_id, sub_info in info.items():
            assert isinstance(sub_info, dict), f"Value for agent_id {agent_id} is not of type dict"
            for key, value in sub_info.items():
                assert isinstance(key, str), "Key in info must be a string"
                assert isinstance(value, float), "Value in info must be a float"
    
    # Check that all observations are in the observation space
    for observation in tqdm(observations, desc="Checking observations"):
        assert environment.observation_space.contains(observation), "Observation not in observation space"


if __name__ == '__main__':
    env = SimpleHoverDroneEnv(render_mode=None)
    _test_environment(env)
