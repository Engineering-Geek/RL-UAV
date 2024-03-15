from src.Environments.multi_agent import MultiAgentDroneEnvironment, SimpleDrone, BaseDrone
from src.utils.viewer import launch_multi_agent_environment


def main():
    env = MultiAgentDroneEnvironment(n_agents=1, n_images=0, depth_render=False, Drone=SimpleDrone)
    launch_multi_agent_environment(env, limp=True)

