import cProfile
from environments.core import MultiAgentDroneEnvironment
from environments.SimpleHoverDroneEnv import SimpleHoverDroneEnv
from tqdm import tqdm
import argparse


def profile(environment: MultiAgentDroneEnvironment, n: int = 1000, save_path: str = "profile_results.prof"):
    environment.reset()
    actions = [environment.action_space.sample() for _ in tqdm(range(n), desc="Sampling actions")]
    profiler = cProfile.Profile()
    profiler.enable()
    for action in tqdm(actions, desc="Profiling"):
        environment.step(action)
    profiler.disable()
    profiler.dump_stats(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile the environment")
    parser.add_argument("--n", type=int, default=10000, help="Number of steps to profile")
    parser.add_argument("--save_path", type=str, default="profile_results.prof",
                        help="Path to save the profile results")
    args = parser.parse_args()

    env = SimpleHoverDroneEnv()
    profile(env, args.n, args.save_path)
    


