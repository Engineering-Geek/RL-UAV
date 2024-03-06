from abc import abstractmethod, ABC
from pathlib import Path
from typing import Union, Tuple, Dict, List, Any

import jax
from jax import numpy as jnp, Array, lax
from jax.scipy.spatial.transform import Rotation
from jax.random import normal, uniform

import numpy as np

from brax.envs.base import PipelineEnv
from brax.envs.base import State as BraxState
from brax.io import mjcf
from brax.mjx.base import State as MjxState

from mujoco import MjModel, Renderer, MjvOption, MjData
from mujoco.mjx import get_data, get_data_into


class BaseRegime(PipelineEnv, ABC):
    """
    BaseRegime is an abstract base class that defines the basic structure for a simulation regime.
    It inherits from PipelineEnv and ABC (Abstract Base Class).

    Attributes:
        simulation_rate (int): The rate at which the simulation is run.
        _mj_model (MjModel): The Mujoco model of the environment.
        _data (MjData): The Mujoco data of the environment.
        _mj_renderer (Renderer): The Mujoco renderer for visualizing the environment.
        _mj_visualization_options (MjvOption): The Mujoco visualization options.
        _camera_frame_rate (int): The frame rate of the camera.
        _number_of_cameras (int): The number of cameras in the environment.
        _gyro_noise (float): The noise in the gyroscope readings.
        _vel_noise (float): The noise in the velocity readings.
        _camera_noise (float): The noise in the camera images.
        camera_images (Array): The images captured by the cameras.
        _initial_position_noise (float): The noise in the initial position of the drone.
        _initial_velocity_noise (float): The noise in the initial velocity of the drone.
        _initial_position_limit (Tuple[float, float]): The bounds for the initial position of the drone.
        _initial_angle_limit (Tuple[float, float]): The bounds for the initial angle of the drone.
    """
    
    def __init__(self, scene_file_path: Union[str, Path], **kwargs):
        """
        The constructor for the BaseRegime class.

        Args:
            scene_file_path (Union[str, Path]): The path to the scene file.
            **kwargs: Arbitrary keyword arguments.
                simulation_rate (int, optional): The rate at which the simulation is run. Defaults to 1.
                debug_pipeline (bool, optional): If True, enables debugging for the pipeline. Defaults to False.
                image_height (int, optional): The height of the image to be rendered. Defaults to 256.
                image_width (int, optional): The width of the image to be rendered. Defaults to 256.
                visualization_options (MjvOption, optional): The Mujoco visualization options. Defaults to MjvOption().
                camera_frame_rate (int, optional): The frame rate of the camera. Defaults to 60.
                number_of_cameras (int, optional): The number of cameras in the environment. Defaults to 2.
                gyro_noise (float, optional): The noise in the gyroscope readings. Defaults to 0.01.
                vel_noise (float, optional): The noise in the velocity readings. Defaults to 0.01.
                camera_noise (float, optional): The noise in the camera images. Defaults to 0.01.
                initial_position_noise (float, optional): The noise in the initial position of the drone. Defaults to 0.01.
                initial_velocity_noise (float, optional): The noise in the initial velocity of the drone. Defaults to 0.01.
                initial_position_bounds (Tuple[float, float], optional): The bounds for the initial position of the drone. Defaults to (-1.0, 1.0).
                initial_angle_bounds (Tuple[float, float], optional): The bounds for the initial angle of the drone. Defaults to (-np.pi, np.pi).
        """
        self.simulation_rate = kwargs.get('simulation_rate', 1)
        super().__init__(
            sys=mjcf.load_model(MjModel.from_xml_path(scene_file_path)),
            n_frames=self.simulation_rate,
            backend='mjx',
            debug=kwargs.get('debug_pipeline', False)
        )
        self._mj_model: MjModel = self.sys.mj_model
        self._data: MjData = MjData(self._mj_model)
        self._mj_renderer: Renderer = Renderer(self._mj_model, kwargs.get('image_height', 256),
                                               kwargs.get('image_width', 256))
        self._mj_visualization_options: MjvOption = kwargs.get('visualization_options', MjvOption())
        self._camera_frame_rate: int = kwargs.get('camera_frame_rate', 60)
        self._number_of_cameras: int = kwargs.get('number_of_cameras', 2)
        self._gyro_noise: float = kwargs.get('gyro_noise', 0.01)
        self._vel_noise: float = kwargs.get('vel_noise', 0.01)
        self._camera_noise: float = kwargs.get('camera_noise', 0.01)
        self.camera_images = jnp.array(
            [jnp.zeros((kwargs.get('image_height', 256), kwargs.get('image_width', 256), 3))
             for _ in range(self._number_of_cameras)])
        self._initial_position_noise = kwargs.get('initial_position_noise', 0.01)
        self._initial_velocity_noise = kwargs.get('initial_velocity_noise', 0.01)
        self._initial_position_limit = kwargs.get('initial_position_bounds', (-1.0, 1.0))
        self._initial_angle_limit = kwargs.get('initial_angle_bounds', (-np.pi, np.pi))

    def get_camera_images(self, mjx_state: MjxState) -> Tuple[jax.Array, ...]:
        """
        Get the images from the cameras.

        Args:
            mjx_state (MjxState): The state of the Mujoco environment.

        Returns:
            Tuple[jax.Array, ...]: A tuple of images from the cameras.
        """
        if mjx_state is None:
            return tuple(self.camera_images)
        get_data_into(self._data, mjx_state)
        camera_images = [self._render_and_convert_to_image(i) for i in range(self._number_of_cameras)]
        return tuple(camera_images)

    def _render_and_convert_to_image(self, camera_id):
        """
        Render the environment and convert it to an image.

        Args:
            camera_id (int): The id of the camera.

        Returns:
            jnp.array: The rendered image.
        """
        self._mj_renderer.update_scene(self._data, camera_id, self._mj_visualization_options)
        return jnp.array(self._mj_renderer.render())

    @abstractmethod
    def calculate_reward(self, mjx_state: MjxState) -> jnp.ndarray:
        """
        Calculate the reward for the current state.

        Args:
            mjx_state (MjxState): The state of the Mujoco environment.

        Returns:
            jnp.ndarray: The calculated reward.

        Raises:
            NotImplementedError: This method must be implemented by a subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def is_simulation_done(self, mjx_state: MjxState) -> jnp.ndarray:
        """
        Check if the simulation is done.

        Args:
            mjx_state (MjxState): The state of the Mujoco environment.

        Returns:
            jnp.ndarray: True if the simulation is done, False otherwise.

        Raises:
            NotImplementedError: This method must be implemented by a subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def get_simulation_metrics(self, mjx_state: MjxState) -> Dict[str, jnp.ndarray]:
        """
        Get the metrics of the simulation.

        Args:
            mjx_state (MjxState): The state of the Mujoco environment.

        Returns:
            Dict[str, jnp.ndarray]: The metrics of the simulation.

        Raises:
            NotImplementedError: This method must be implemented by a subclass.
        """
        raise NotImplementedError

    def observe_environment(self, state_i: MjxState, state_j: MjxState) -> list[Array]:
        """
        Observe the environment.

        Args:
            state_i (MjxState): The initial state of the Mujoco environment.
            state_j (MjxState): The final state of the Mujoco environment.

        Returns:
            list[Array]: A list of observations from the environment.
        """
        
        def true_fun(_):
            self.get_camera_images(state_j)
            return self.camera_images
        
        def false_fun(_):
            return self.camera_images
        
        condition = (self._number_of_cameras > 0) and (
                    state_j.time % self._camera_frame_rate - self.simulation_rate <= 0)
        self.camera_images = lax.cond(condition, true_fun, false_fun, None)
        
        gyro_state_i = state_i.qpos[3:7]
        gyro_state_j = state_j.qpos[3:7]
        gyro = gyro_state_j - gyro_state_i

        vel_i = state_i.qvel[:3]
        vel_j = state_j.qvel[:3]
        acc = (vel_j - vel_i) / (self.dt * self.simulation_rate)
        vel = (vel_i + vel_j) / 2

        rng = jax.random.PRNGKey(0)
        gyro += normal(rng, gyro.shape) * self._gyro_noise
        vel += normal(rng, vel.shape) * self._vel_noise
        acc += normal(rng, acc.shape) * self._vel_noise

        if self._number_of_cameras > 0:
            # self.camera_images = [img + normal(rng, img.shape) * self._camera_noise for img in self.camera_images]
            return [vel, acc, gyro] + list(self.camera_images)
        else:
            return [vel, acc, gyro]

    def step(self, brax_state: BraxState, action: jax.Array) -> BraxState:
        """
        Perform a step in the simulation.

        Args:
            brax_state (BraxState): The initial state of the Brax environment.
            action (jax.Array): The action to be performed.

        Returns:
            BraxState: The final state of the Brax environment.
        """
        mjx_state_i: BraxState = brax_state.pipeline_state
        mjx_state_j: BraxState = self.pipeline_step(mjx_state_i, action)
        return brax_state.replace(
            pipeline_state=mjx_state_j,
            obs=self.observe_environment(mjx_state_i, mjx_state_j),
            reward=self.calculate_reward(mjx_state_j),
            done=self.is_simulation_done(mjx_state_j),
            metrics=self.get_simulation_metrics(mjx_state_j)
        )

    def reset(self, rng: jax.Array) -> BraxState:
        """
        Reset the simulation.

        Args:
            rng (jax.Array): The random number generator.

        Returns:
            BraxState: The initial state of the Brax environment.
        """
        rng, rng1, rng2, rng3, rng4, rng5 = jax.random.split(rng, 6)
        initial_positions = self.sys.qpos0 + normal(rng1, self.sys.qpos0.shape) * self._initial_position_noise
        initial_velocities = normal(rng2, (self.sys.nv,)) * self._initial_velocity_noise
        min_val, max_val = self._initial_position_limit
        x = uniform(rng3, (), minval=min_val, maxval=max_val)
        y = uniform(rng4, (), minval=min_val, maxval=max_val)
        z = uniform(rng5, (), minval=min_val, maxval=max_val)
        initial_positions = initial_positions.at[0:3].set(jnp.array([x, y, z]))

        # Randomize the starting angle of the drone
        rng6, rng7, rng8 = jax.random.split(rng, 3)
        min_val, max_val = self._initial_angle_limit
        alpha = uniform(rng6, (), minval=min_val, maxval=max_val)
        beta = uniform(rng7, (), minval=min_val, maxval=max_val)
        gamma = uniform(rng8, (), minval=min_val, maxval=max_val)

        # Convert Euler angles to quaternion for the rotation
        initial_quaternion = Rotation.from_euler('xyz', jnp.array([alpha, beta, gamma])).as_quat()
        initial_positions = initial_positions.at[3:7].set(initial_quaternion)

        mjx_state: MjxState = self.pipeline_init(initial_positions, initial_velocities)

        brax_state = BraxState(
            pipeline_state=mjx_state,
            obs=self.observe_environment(mjx_state, mjx_state),
            reward=self.calculate_reward(mjx_state),
            done=self.is_simulation_done(mjx_state),
            metrics=self.get_simulation_metrics(mjx_state)
        )
        return brax_state

    def __enter__(self):
        """
        Enter the context of the BaseRegime object.

        Returns:
            BaseRegime: The current object.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the context of the BaseRegime object.

        Args:
            exc_type (type): The type of the exception.
            exc_val (Exception): The instance of the exception.
            exc_tb (traceback): The traceback of the exception.
        """
        self.close()

    def close(self):
        """
        Close the renderer.
        """
        self._mj_renderer.close()