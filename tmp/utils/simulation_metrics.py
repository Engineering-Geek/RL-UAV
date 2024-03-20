from gymnasium.envs.mujoco import MujocoEnv
from tqdm import tqdm


def time_ratio(env: MujocoEnv, simulation_seconds: int = 1000):
    """
    Calculate the ratio of simulation time to real time for a given MuJoCo environment.

    This function steps through a specified number of seconds in the simulation, sampling random actions at each timestep. It measures the real time taken to complete the simulation and computes the ratio of simulation time to real time, which can be useful for performance analysis or tuning the simulation.

    Parameters
    ----------
    env : MujocoEnv
        The MuJoCo environment to be tested. The environment should be already configured and ready to use.
    simulation_seconds : int, optional
        The number of seconds to simulate in the environment. Default is 1000.

    Returns
    -------
    float
        The ratio of simulation seconds to real seconds. A higher ratio indicates a faster simulation, where more simulation time is covered in less real time.

    Examples
    --------
    >>> from gymnasium.envs.mujoco import AntEnv
    >>> env = AntEnv()
    >>> ratio = time_ratio(env, simulation_seconds=1000)
    >>> print(f"Simulation to real-time ratio: {ratio}")

    Notes
    -----
    - The function uses `env.action_space.sample()` to generate random actions at each timestep.
    - The environment is reset at the beginning of the function and closed at the end.
    - The tqdm library is used to show a progress bar during the simulation.

    """
    from time import time
    start = time()
    env.reset()
    total_timesteps = round(simulation_seconds // env.dt)
    for _ in tqdm(range(total_timesteps)):
        env.step(env.action_space.sample())
    env.close()
    end = time()
    return simulation_seconds / (end - start)

