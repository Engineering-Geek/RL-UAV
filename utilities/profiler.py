import cProfile
from environments.core import MultiAgentDroneEnvironment
from tqdm import tqdm


def profile(environment: MultiAgentDroneEnvironment, num_steps: int = 10000, filename: str = "profile.prof"):
    profiler = cProfile.Profile()
    random_actions = [environment.action_space_sample() for _ in tqdm(range(num_steps), desc="Generating random actions")]
    environment.reset()
    profiler.enable()
    for i in tqdm(range(num_steps), desc="Profiling"):
        environment.step(random_actions[i])
    profiler.disable()
    environment.close()
    profiler.dump_stats(filename)
    
    
if __name__ == '__main__':
    from environments.SimpleDroneEnvs import SimpleHoverDroneEnv
    env = SimpleHoverDroneEnv()
    profile(env)

