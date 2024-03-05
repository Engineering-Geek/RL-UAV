from abc import abstractmethod, ABC
from pathlib import Path
from typing import Union, Tuple, Dict

import jax
from jax import numpy as jnp
from jax.scipy.spatial.transform import Rotation

import numpy as np

from brax.envs.base import PipelineEnv
from brax.envs.base import State as BraxState
from brax.io import mjcf
from brax.mjx.base import State as MjxState

from mujoco import MjModel, Renderer, MjvOption
from mujoco.mjx import get_data


class BaseRegime(PipelineEnv, ABC):
    """
    BaseRegime is an abstract base class that provides a common interface for different simulation regimes.
    It inherits from PipelineEnv and ABC (Abstract Base Class).
    """

    def __init__(self, scene_file_path: Union[str, Path], **kwargs):
        """
        Initializes the BaseRegime instance with the given parameters.

        Args:
            scene_file_path (Union[str, Path]): The path to the scene file.
            **kwargs: Variable length argument list providing the following optional parameters:
                simulation_rate (int, optional): The rate of the simulation. Defaults to 1.
                debug_pipeline (bool, optional): If True, the pipeline will be debugged. Defaults to False.
                image_height (int, optional): The height of the image. Defaults to 256.
                image_width (int, optional): The width of the image. Defaults to 256.
                visualization_options (MjvOption, optional): The visualization options for the Mujoco renderer. Defaults to MjvOption().
                camera_frame_rate (int, optional): The frame rate of the camera. Defaults to 60.
                number_of_cameras (int, optional): The number of cameras. Defaults to 2.
                initial_position_noise (float, optional): The noise to add to the initial position. Defaults to 0.01.
                initial_velocity_noise (float, optional): The noise to add to the initial velocity. Defaults to 0.01.
                initial_position_bounds (Tuple[float, float], optional): The bounds for the initial position. Defaults to (-1.0, 1.0).
                initial_angle_bounds (Tuple[float, float], optional): The bounds for the initial angle. Defaults to (-np.pi, np.pi).
        """
        self._mj_model = MjModel.from_xml_path(scene_file_path)
        self.simulation_rate = kwargs.get('simulation_rate', 1)
        super().__init__(
            sys=mjcf.load_model(self._mj_model),
            n_frames=self.simulation_rate,
            backend='mjx',
            debug=kwargs.get('debug_pipeline', False)
        )
        self._mj_renderer: Renderer = Renderer(self._mj_model, kwargs.get('image_height', 256), kwargs.get('image_width', 256))
        self._mj_visualization_options: MjvOption = kwargs.get('visualization_options', MjvOption())
        self._camera_frame_rate: int = kwargs.get('camera_frame_rate', 60)
        self._number_of_cameras: int = kwargs.get('number_of_cameras', 2)
        self.simulation_time = 0
        self.camera_images = jnp.array(
            [jnp.zeros((
                kwargs.get('image_height', 256),
                kwargs.get('image_width', 256), 3))
                for _ in range(self._number_of_cameras)
            ])

        self.initial_position_noise = kwargs.get('initial_position_noise', 0.01)
        self.initial_velocity_noise = kwargs.get('initial_velocity_noise', 0.01)
        self.initial_position_bounds = kwargs.get('initial_position_bounds', (-1.0, 1.0))
        self.initial_angle_bounds = kwargs.get('initial_angle_bounds', (-np.pi, np.pi))

    def get_camera_images(self, brax_state: BraxState) -> Tuple[jax.Array, ...]:
        """
        Returns the images from the environment.

        Args:
            brax_state (BraxState): The current state of the Brax environment.

        Returns:
            Tuple[jax.Array, ...]: A tuple of JAX arrays representing the images from the environment.
        """
        if isinstance(brax_state, MjxState):
            mjx_state: MjxState = brax_state
        else:
            mjx_state: MjxState = brax_state.pipeline_state
        mjx_state = get_data(self._mj_model, mjx_state)
        camera_images = [self._render_and_convert_to_image(mjx_state, i) for i in range(self._number_of_cameras)]
        return tuple(camera_images)

    def _render_and_convert_to_image(self, mjx_state, camera_id):
        """
        Renders the given state and converts it to an image.

        Args:
            mjx_state: The current state of the Mujoco environment.
            camera_id: The ID of the camera to use for rendering.

        Returns:
            jnp.array: A JAX array representing the rendered image.
        """
        self._mj_renderer.update_scene(mjx_state, camera_id, self._mj_visualization_options)
        return jnp.array(self._mj_renderer.render())

    @abstractmethod
    def calculate_reward(self, brax_state: BraxState) -> jnp.ndarray:
        """
        Calculates the reward for the given state.

        This method must be implemented by subclasses.

        Args:
            brax_state (BraxState): The current state of the Brax environment.

        Returns:
            jnp.ndarray: The calculated reward.
        """
        raise NotImplementedError

    @abstractmethod
    def is_simulation_done(self, brax_state: BraxState) -> jnp.ndarray:
        """
        Checks if the simulation is done for the given state.

        This method must be implemented by subclasses.

        Args:
            brax_state (BraxState): The current state of the Brax environment.

        Returns:
            jnp.ndarray: A boolean array indicating whether the simulation is done.
        """
        raise NotImplementedError

    @abstractmethod
    def get_simulation_metrics(self, brax_state: BraxState) -> Dict[str, jnp.ndarray]:
        """
        Gets the simulation metrics for the given state.

        This method must be implemented by subclasses.

        Args:
            brax_state (BraxState): The current state of the Brax environment.

        Returns:
            Dict[str, jnp.ndarray]: A dictionary of simulation metrics.
        """
        raise NotImplementedError

    def observe_environment(self, brax_state: BraxState) -> jnp.ndarray:
        """
        Observes the environment and returns the observation.

        Args:
            brax_state (BraxState): The current state of the Brax environment.

        Returns:
            jnp.ndarray: The observation of the environment.
        """
        rng, rng_gyro, rng_linacc, rng_quat = jax.random.split(jax.random.PRNGKey(0), 4)

        # Manually calculate gyroscope data (angular velocity in this context)
        gyro_data = brax_state.qvel[3:6]  # Assuming angular velocity is stored in these indices
        gyro_data = gyro_data + jax.random.normal(rng_gyro, gyro_data.shape) * self.initial_velocity_noise  # Adding Gaussian noise

        # Manually calculate linear acceleration
        linacc_data = brax_state.qacc[0:3]  # Assuming linear acceleration is stored in these indices
        linacc_data = linacc_data + jax.random.normal(rng_linacc, linacc_data.shape) * self.initial_velocity_noise  # Adding Gaussian noise

        # Extract quaternion from the state (assuming it's part of the state)
        quaternion_data = brax_state.qpos[3:7]  # Assuming quaternion is stored in these indices
        quaternion_data = quaternion_data + jax.random.normal(rng_quat, quaternion_data.shape) * self.initial_position_noise  # Adding Gaussian noise

        if (self._number_of_cameras > 0) and (
                self.simulation_time % self._camera_frame_rate - self.simulation_rate <= 0):
            self.get_camera_images(brax_state)
        return jnp.concatenate(
            [gyro_data, linacc_data, quaternion_data, *self.camera_images])

    def step(self, brax_state: BraxState, action: jax.Array) -> BraxState:
        """
        Steps the environment with the given action and returns the new state.

        Args:
            brax_state (BraxState): The current state of the Brax environment.
            action (jax.Array): The action to apply.

        Returns:
            BraxState: The new state of the Brax environment after stepping.
        """
        mjx_state_i: MjxState = brax_state.pipeline_state
        mjx_state_j: MjxState = self.pipeline_step(mjx_state_i, action)
        self.simulation_time += self.dt * self.simulation_rate
        reward = self.calculate_reward(mjx_state_j)
        done = self.is_simulation_done(mjx_state_j)
        metrics = self.get_simulation_metrics(mjx_state_j)
        observation = self.observe_environment(mjx_state_j)
        return brax_state.replace(
            pipeline_state=mjx_state_j,
            observation=observation,
            reward=reward,
            done=done,
            metrics=metrics
        )

    def reset(self, rng: jax.Array) -> BraxState:
        self.simulation_time = 0

        # Introduce Gaussian variations to qpos and qvel
        rng, rng1, rng2, rng3, rng4, rng5 = jax.random.split(rng, 6)
        initial_positions = self.sys.qpos0 + jax.random.normal(rng1, self.sys.qpos0.shape) * self.initial_position_noise
        initial_velocities = jax.random.normal(rng2, (self.sys.nv,)) * self.initial_velocity_noise

        # Randomize the starting location of the drone
        x = jax.random.uniform(rng3, (), minval=self.initial_position_bounds[0], maxval=self.initial_position_bounds[1])
        y = jax.random.uniform(rng4, (), minval=self.initial_position_bounds[0], maxval=self.initial_position_bounds[1])
        z = jax.random.uniform(rng5, (), minval=self.initial_position_bounds[0], maxval=self.initial_position_bounds[1])
        initial_positions = initial_positions.at[0:3].set(jnp.array([x, y, z]))

        # Randomize the starting angle of the drone
        rng6, rng7, rng8 = jax.random.split(rng, 3)
        alpha = jax.random.uniform(rng6, (), minval=self.initial_angle_bounds[0], maxval=self.initial_angle_bounds[1])
        beta = jax.random.uniform(rng7, (), minval=self.initial_angle_bounds[0], maxval=self.initial_angle_bounds[1])
        gamma = jax.random.uniform(rng8, (), minval=self.initial_angle_bounds[0], maxval=self.initial_angle_bounds[1])

        # Convert Euler angles to quaternion for the rotation
        initial_quaternion = Rotation.from_euler('xyz', jnp.array([alpha, beta, gamma])).as_quat()
        initial_positions = initial_positions.at[3:7].set(initial_quaternion)

        mjx_state: MjxState = self.pipeline_init(initial_positions, initial_velocities)

        brax_state = BraxState(
            pipeline_state=mjx_state,
            observation=self.observe_environment(mjx_state),
            reward=jnp.zeros((), dtype=jnp.float32),
            done=jnp.zeros((), dtype=jnp.bool_),
            metrics={}
        )
        return brax_state

    def __enter__(self):
        """
        Defines the behavior of the class when it is used in a 'with' statement.

        Returns:
            self: The instance of the class.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Defines the behavior of the class when it is exited from a 'with' statement.

        Args:
            exc_type: The type of the exception that caused the 'with' statement to be exited, if any.
            exc_val: The instance of the exception that caused the 'with' statement to be exited, if any.
            exc_tb: The traceback of the exception that caused the 'with' statement to be exited, if any.
        """
        self.close()

    def close(self):
        """
        Closes the Renderer object associated with the instance of the BaseRegime class.

        This method is responsible for releasing the resources held by the Renderer object.
        It should be called when the BaseRegime object is no longer needed, to ensure
        proper cleanup of the resources. If the Renderer object has already been closed,
        this method will have no effect.
        """
        self._mj_renderer.close()
