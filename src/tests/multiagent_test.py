import numpy as np
import pytest
from src.Environments.multi_agent.BaseMultiAgentEnvironment import BaseMultiAgentEnvironment
from time import time
from typing import List
from logging import getLogger

logger = getLogger(__name__)


@pytest.fixture
def base_multi_agent_environment() -> List[BaseMultiAgentEnvironment]:
    n_agents = [1, 2, 3]
    n_images = [0, 1, 2]
    envs = []
    for n_agent in n_agents:
        for n_image in n_images:
            start = time()
            envs.append(BaseMultiAgentEnvironment(
                n_agents=n_agent,
                n_images=n_image,
                frame_skip=1,
                depth_render=False
            ))
            end = time()
            logger.info(f"Time to create environment with {n_agent} agents and {n_image} images: {end - start}")
    return envs


def test_speed(base_multi_agent_environment):
    n = 1000
    logger.info(f"Stress Testing Each BaseMultiAgentEnvironment through {n} steps")
    for env in base_multi_agent_environment:
        n_agents = env.n_agents
        n_images = env.n_images
        start = time()
        env.reset()
        for _ in range(n):
            env.step(env.action_space_sample())
            env.render()
        env.close()
        end = time()
        logger.info(f"\tAverage Speed for {n_agents} agents and {n_images} images: {n / (end - start)} steps/second")
    logger.info("Stress Test Passed")


def test_bullet_logic(base_multi_agent_environment):
    env = base_multi_agent_environment[0]
    env.reset()
    tolerance = 0.1
    acceptable_error = 0.1
    no_shot_correct = 0
    for _ in range(100):
        env.step({
            0: {
                "motor": np.array([0.0, 0.0, 0.0]),
                "shoot": np.array([0]),
            },
        })
        if env.drones[0].bullet.bullet_body.cvel[:3] == pytest.approx(np.array([0.0, 0.0, 0.0]), abs=tolerance):
            no_shot_correct += 1
    shot_correct = 0
    for _ in range(100):
        env.step({
            0: {
                "motor": np.array([0.0, 0.0, 0.0]),
                "shoot": np.array([1]),
            },
        })
        if env.drones[0].bullet.bullet_body.cvel[:3] != pytest.approx(np.array([0.0, 0.0, 0.0]), abs=tolerance):
            shot_correct += 1
    env.close()
    logger.info(f"Bullet Logic Test Passed: {no_shot_correct} no shots correct, {shot_correct} shots correct")
    assert no_shot_correct > acceptable_error * 100, f"no_shot_correct: {no_shot_correct} out of 100, too low"
    assert shot_correct > acceptable_error * 100, f"shot_correct: {shot_correct} out of 100, too low"


def test_reset_model(base_multi_agent_environment):
    for env in base_multi_agent_environment:
        env.reset_model()
        for drone in env.drones:
            assert drone.alive == True, f"Drone {drone.agent_id} is not alive after reset_model"


def test_observation(base_multi_agent_environment):
    for env in base_multi_agent_environment:
        obs = env.observation()
        assert len(obs) == env.n_agents, "Number of observations does not match number of agents"


def test_reward(base_multi_agent_environment):
    for env in base_multi_agent_environment:
        rewards = env.reward
        assert len(rewards) == env.n_agents, "Number of rewards does not match number of agents"


def test_truncated(base_multi_agent_environment):
    for env in base_multi_agent_environment:
        truncations = env.truncated
        assert len(truncations) == env.n_agents, "Number of truncations does not match number of agents"


def test_done(base_multi_agent_environment):
    for env in base_multi_agent_environment:
        dones = env.done
        assert len(dones) == env.n_agents, "Number of done flags does not match number of agents"


def test_metrics(base_multi_agent_environment):
    for env in base_multi_agent_environment:
        metrics = env.metrics
        assert isinstance(metrics, dict), "Metrics is not a dictionary"
