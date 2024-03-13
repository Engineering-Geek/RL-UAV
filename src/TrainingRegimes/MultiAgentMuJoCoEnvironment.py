from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.utils import EzPickle
from utils.multiagent_model_generator import multiagent_model
from mujoco import MjModel, MjData, viewer


model: MjModel = multiagent_model(6, 2.0)
data: MjData = MjData(model)

print(model.nbody)


