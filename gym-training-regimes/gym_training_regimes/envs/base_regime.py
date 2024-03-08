from typing import SupportsFloat, Any

import numpy as np
from gymnasium.core import ActType, ObsType
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Space, Box, Sequence, Dict
from numpy.random import default_rng, Generator
from mujoco import MjModel, MjData


class CustomMujocoEnv(MujocoEnv):
    def __init__(self, xml_path, **kwargs):
        super().__init__(model_path=xml_path, frame_skip=kwargs.get('frame_skip', 1))
        self.rng = default_rng()
        self.model: MjModel     # Declare the model attribute
        self.data: MjData           # Declare the data attribute
        
        self.observation_space: Space = Dict({
            "drone_position": Box(low=-np.inf, high=np.inf, shape=(3,)),
            "drone_velocity": Box(low=-np.inf, high=np.inf, shape=(3,)),
            "drone_orientation": Box(low=-np.inf, high=np.inf, shape=(4,)),
            "drone_angular_velocity": Box(low=-np.inf, high=np.inf, shape=(3,)),
            "target_vector": Box(low=-np.inf, high=np.inf, shape=(3,)),
            "image_0": Box(low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8),
        })
        self.action_space: Space = Box(low=-1.0, high=1.0, shape=(self.model.nu,))
    
    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.do_simulation(action, self.frame_skip)
        observation = self._get_obs()
        reward = self.compute_reward()
        done = self.check_done()
        info = self.compute_metrics()
        
        return observation, reward, done, info
    
    def _get_obs(self):
        # Implement your observation logic here
        return np.concatenate([
            self.data.qpos.flat,  # Joint positions
            self.data.qvel.flat,  # Joint velocities
        ])
    
    def compute_reward(self):
        # Implement your reward logic here
        return 0.0
    
    def check_done(self):
        # Implement your logic to check if the episode is done
        return False
    
    def compute_metrics(self):
        # Implement any additional metrics you want to track
        return {}
    
    def reset_model(self):
        # Required by MujocoEnv, called by reset()
        qpos = self.init_qpos + self.rng.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.rng.random(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()
    
    def viewer_setup(self):
        # Adjust the camera settings if needed
        self.viewer.cam.distance = self.model.stat.extent * 0.5
    
    def get_camera_images(self):
        """
        Returns the RGB arrays from cameras 0 and 1.
        """
        width = self.viewer.cam.width
        height = self.viewer.cam.height
        
        # Get the image from camera 0
        self.viewer.cam.type = 2  # Set camera type to fixed
        self.viewer.cam.fixedcamid = 0  # Set the camera to fixed camera 0
        img_0 = self.sim.render(width, height, camera_name="camera0")
        
        # Get the image from camera 1
        self.viewer.cam.fixedcamid = 1  # Switch to fixed camera 1
        img_1 = self.sim.render(width, height, camera_name="camera1")
        
        return img_0, img_1
