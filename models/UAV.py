from datetime import datetime
import functools
import jax
from jax import numpy as jp
import numpy as np
from typing import Any, Dict, Sequence, Tuple, Union

from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.envs.base import Env, PipelineEnv, State
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import html, mjcf, model

from etils import epath
from flax import struct
from matplotlib import pyplot as plt
from ml_collections import config_dict
from mujoco import MjModel, MjData, mjx
import mujoco as mj


class UAV(PipelineEnv):
    def __init__(self, steps_per_frame: int = 1, debug_pipeline: bool = False):
        uav: MjModel = MjModel.from_xml_path('models/scene.xml')
        uav.opt.iterations = 10
        uav.opt.timestep = 0.01
        uav.opt.gravity = [0, 0, -9.81]
        uav.opt.solver = mj.mjtSolver.mjSOL_CG
        uav.opt.ls_iterations = 10
        sys = mjcf.load_model(uav)
        super().__init__(
            sys=sys,
            backend='mjx',
            n_frames=steps_per_frame,
            debug=debug_pipeline
        )

    def reset(self, rng: jax.Array) -> State:
        rng, q_pos_rng, q_vel_rng = jax.random.split(rng, 3)
        q_pos = jax.random.uniform(q_pos_rng, (self.sys.nq,), minval=-0.1, maxval=0.1)
        q_vel = jax.random.uniform(q_vel_rng, (self.sys.nv,), minval=-0.1, maxval=0.1)
        reward, done, zero = jp.zeros(3)
        metrics = {
            'distance to target': zero,
            'velocity': zero,
            'angular velocity': zero,
            'max force': zero,
            'max torque': zero
        }


