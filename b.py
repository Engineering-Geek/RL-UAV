import gymnasium as gym
from environments.SimpleDroneEnvs import SimpleHoverDroneEnv

# Replace 'YourEnv' with your environment's ID or class
env = SimpleHoverDroneEnv(
    n_agents=5,
    spacing=3.0,
    dt=0.01,
    render_mode=None,
    fps=60,
    sim_rate=1,
    update_distances=True
)

num_episodes = 100  # Number of episodes to run the test

for episode in range(num_episodes):
    observations, log = env.reset()
    dones = {agent_id: False for agent_id in observations.keys()}
    dones["__all__"] = False
    step_count = 0

    while not dones["__all__"]:
        actions = {agent_id: env.action_space[agent_id].sample() for agent_id in observations.keys()}  # Sample random actions

        next_observations, rewards, dones, truncs, infos = env.step(actions)
        
        if dones["__all__"]:
            print(f"Episode {episode + 1} finished after {step_count + 1} steps for all agents.")
        step_count += 1

