## Files

- `BaseRegime.py`: This is an abstract base class that provides a common interface for different simulation regimes. It includes methods for observing the environment, stepping through the simulation, calculating rewards, and checking if the simulation is done.

    The `BaseRegime` class takes the following arguments:

    - `scene_file_path (Union[str, Path])`: The path to the scene file.
    - `simulation_rate (int, optional)`: The rate of the simulation. Defaults to 1.
    - `debug_pipeline (bool, optional)`: If True, the pipeline will be debugged. Defaults to False.
    - `image_height (int, optional)`: The height of the image. Defaults to 256.
    - `image_width (int, optional)`: The width of the image. Defaults to 256.
    - `visualization_options (MjvOption, optional)`: The visualization options for the Mujoco renderer. Defaults to MjvOption().
    - `camera_frame_rate (int, optional)`: The frame rate of the camera. Defaults to 60.
    - `number_of_cameras (int, optional)`: The number of cameras. Defaults to 2.
    - `initial_position_noise (float, optional)`: The noise to add to the initial position. Defaults to 0.01.
    - `initial_velocity_noise (float, optional)`: The noise to add to the initial velocity. Defaults to 0.01.
    - `initial_position_bounds (Tuple[float, float], optional)`: The bounds for the initial position. Defaults to (-1.0, 1.0).
    - `initial_angle_bounds (Tuple[float, float], optional)`: The bounds for the initial angle. Defaults to (-np.pi, np.pi).

- `SimpleRegimes.py`: This file contains simple training regimes, such as the `HoverRegime`, which trains the drone to hover at a specific altitude.

## Usage

To use a training regime, import the desired class from its file and instantiate it with the necessary parameters. For example:

```python
from TrainingRegimes.SimpleRegimes import HoverRegime

hover_regime = HoverRegime(name="Hover", duration=1000, target_altitude=0.3)