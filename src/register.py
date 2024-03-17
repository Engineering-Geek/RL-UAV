from typing import Type
from ray.tune.registry import register_env
from src.Environments.multi_agent import SimpleShooter, MultiAgentDroneEnvironment, BaseDrone
from yaml import safe_load


def create_environment(Drone: Type[BaseDrone], n_agents: int, n_images: int,
                       frame_skip: int, depth_render: bool, **kwargs) -> MultiAgentDroneEnvironment:
    env = MultiAgentDroneEnvironment(
        n_agents=n_agents,
        n_images=n_images,
        frame_skip=frame_skip,
        depth_render=depth_render,
        Drone=Drone,
        **kwargs
    )
    return env


def register(name: str = "MultiAgentDroneEnvironment", environment_creator: callable = None) -> None:
    register_env(name, environment_creator if environment_creator else create_environment)


register()
