import numpy as np

from gym_drone import LidarBattleRoyal
from mujoco import viewer

from gym_drone import LidarBattleRoyal
from time import sleep, time
from tqdm import tqdm

spawn_box = np.array([[-5, -5, 0.5], [5, 5, 5]])
world_bounds = np.array([[-10, -10, -0.1], [10, 10, 5]])
env = LidarBattleRoyal(render_mode="human", world_bounds=world_bounds,
                       respawn_box=spawn_box, num_agents=5, n_phi=16, n_theta=16, ray_max_distance=10,)


observation, log = env.reset()

for i in tqdm(range(100000)):
    action = env.action_space.sample()
    observation, reward, trunc, done, info = env.step(action)
    shot_drone = any([info[agent_id]["shot_drone"] for agent_id in range(env.num_agents)])
    hit_floor = any([info[agent_id]["hit_floor"] for agent_id in range(env.num_agents)])
    env.render()
    if shot_drone:
        print("Drone shot!")
        env.render()
    if hit_floor:
        print("Drone hit the floor!")
        env.render()
env.close()


# model = env.model.__copy__()
# data = env.data.__copy__()
# env.close()
# viewer.launch(model, data)



