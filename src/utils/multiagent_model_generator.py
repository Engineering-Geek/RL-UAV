import os
import math
from typing import Tuple, List

from mujoco import MjModel, viewer

# Define directories for saving and reading XML files
SAVE_DIR = os.path.join(os.path.dirname(__file__), '../Environments/assets/MultiUAV/tmp')
SCENE_DIR = os.path.join(os.path.dirname(__file__), '../Environments/assets/MultiUAV/scene.xml')
UAV_DIR = os.path.join(os.path.dirname(__file__), '../Environments/assets/MultiUAV/UAV.xml')

# Create the save directory if it does not exist
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)


def _clear_dir():
    """
    Remove all .xml files in the save directory except 'scene.xml' and 'UAV.xml'.

    This function ensures that the save directory is clean before generating new UAV configurations.
    """
    for f in os.listdir(SAVE_DIR):
        if f.endswith('.xml') and f != 'scene.xml' and f != 'UAV.xml':
            os.remove(os.path.join(SAVE_DIR, f))


def _get_drone_positions(n_drones: int, spacing: float) -> List[Tuple[float, float]]:
    """
    Calculate the positions for drones to be placed in a 2D grid centered around the origin.

    :param n_drones: The number of drones to position.
    :param spacing: The spacing between drones in the grid.
    :return: A list of tuples where each tuple represents the (x, y) position of a drone.
    """
    side_length = math.ceil(math.sqrt(n_drones))  # Side length of the grid
    positions = []
    offset = -((side_length - 1) * spacing / 2)  # Calculate offset to center the grid
    for i in range(n_drones):
        x_pos = (i % side_length) * spacing + offset
        y_pos = (i // side_length) * spacing + offset
        positions.append((x_pos, y_pos))
    return positions


def save_multiagent_model(n_drones: int, spacing: float = 2.0, save_dir: str = None) -> str:
    """
    Generate individual UAV XML files and a scene file including all UAVs, then save them to a directory.

    :param n_drones: The number of drones to include in the scene.
    :param spacing: The spacing between drones in the generated grid.
    :param save_dir: The directory to save the generated XML files. Defaults to SAVE_DIR.
    :return: The path to the generated scene file.
    """
    _clear_dir()  # Clear the directory to remove old files
    drone_positions = _get_drone_positions(n_drones, spacing)
    include_files_str = ""
    
    # Generate UAV XML files
    for i, (x_pos, y_pos) in enumerate(drone_positions):
        with open(UAV_DIR, 'r') as file:
            content = file.read()
        content = content.replace('{{index}}', str(i + 1))
        content = content.replace('{{x_pos}}', str(x_pos))
        content = content.replace('{{y_pos}}', str(y_pos))
        
        drone_file_name = f"drone_{i + 1}.xml"
        drone_file_path = os.path.join(SAVE_DIR, drone_file_name)
        with open(drone_file_path, 'w') as file:
            file.write(content)
        
        include_files_str += f'<include file="{drone_file_name}"/>\n'
    
    # Generate the scene XML file
    with open(SCENE_DIR, 'r') as file:
        scene_content = file.read()
    
    scene_content = scene_content.replace('{{include_file}}', include_files_str.strip())
    save_dir = save_dir or SAVE_DIR
    scene_file_path = os.path.join(save_dir, 'scene_with_drones.xml')
    with open(scene_file_path, 'w') as file:
        file.write(scene_content)
    
    return scene_file_path


def multiagent_model(n_drones: int, spacing: float = 2.0) -> MjModel:
    """
    Load a multi-agent MuJoCo model with a specified number of drones positioned in a 2D grid.

    :param n_drones: The number of drones to include in the model.
    :param spacing: The spacing between drones in the grid.
    :return: An instance of MjModel representing the multi-agent scene.
    """
    scene_file_path = save_multiagent_model(n_drones, spacing)
    model = MjModel.from_xml_path(scene_file_path)
    return model


def test_multiagent_model():
    model = multiagent_model(4)
    viewer.launch(model)
