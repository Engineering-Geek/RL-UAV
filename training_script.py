from gym_drone.gym_drone import SuicideDroneEnv
import numpy as np

environment = SuicideDroneEnv(
    tolerance_distance=0.1,
    max_time=20,
    reward_goal=100,
    reward_distance_exp=1,
    reward_distance_max=100,
    penalty_crash=100,
    penalty_time=1,
    reward_velocity_exp=1,
    reward_distance_coefficient=1,
    reward_velocity_coefficient=1,
)

observation = environment.reset()

for _ in range(1000):
    action = environment.action_space.sample()
    observation, reward, done, info = environment.step(action)
    if done:
        observation = environment.reset()

