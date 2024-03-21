"""
Simulation Model Generation for MuJoCo Drones and Targets
=========================================================

This module provides functions to generate and organize the XML configuration files required for simulating drones and targets in a MuJoCo environment. It includes functionalities to calculate positions for drones and targets, generate XML files for the scene, drones, and targets, and save these configurations for use in MuJoCo simulations.

Functions
---------
- read_template_files : Read the XML template files for the scene, drone, and target.
- generate_drone_positions : Generate x, y, z positions for the agents evenly spaced in a circle.
- generate_target_positions : Generate x, y, z positions for the targets randomly placed within the map bounds.
- generate_target_dimensions : Generate dimensions for the targets based on the specified range.
- create_scene_xml : Create the XML for the scene and write it to a file.
- create_agent_and_target_xml : Create and save the XML files for drones and targets with generated positions and dimensions.
- model : Main function to generate and save the scene, drones, and targets XML configuration.

This module is designed to be used in conjunction with MuJoCo simulations, particularly those involving multiple drones
and target objects in a defined environment.
"""

import numpy as np
import os

from mujoco import MjModel


def read_template_files() -> tuple:
    """
    Read the XML template files for the scene, drone, and target.

    Returns:
        tuple: Containing the content of scene.xml, drone.xml, and target.xml respectively.
    """
    base_path = f"{os.path.dirname(__file__)}"
    with open(f"{base_path}/scene.xml") as f:
        scene_xml = f.read()
    with open(f"{base_path}/drone.xml") as f:
        drone_xml = f.read()
    return scene_xml, drone_xml


def generate_drone_positions(num_agents: int, spacing: float, drone_height: float) -> np.ndarray:
    """
    Generate x, y, z positions for the agents evenly spaced in a circle.

    Args:
        num_agents (int): Number of agents.
        spacing (float): Distance between each agent's center.
        drone_height (float): Height at which drones are placed.

    Returns:
        np.ndarray: BaseDrone positions array.
    """
    indices = np.arange(num_agents)
    radians = 2 * np.pi / num_agents
    length = spacing / np.sqrt(3)
    
    drone_positions = np.zeros((num_agents, 3))
    drone_positions[:, 0] = length * np.cos(radians * indices)
    drone_positions[:, 1] = length * np.sin(radians * indices)
    drone_positions[:, 2] = drone_height
    return drone_positions


def create_scene_xml(scene_xml: str, num_agents: int, map_bounds: np.ndarray, save_dir: str):
    """
    Create the XML for the scene and write it to a file.

    Args:
        scene_xml (str): Template XML for the scene.
        num_agents (int): Number of agents (drones).
        map_bounds (np.ndarray): World bounds for placing the fence.
        save_dir (str): Directory to save the generated scene XML.
    """
    
    files = "\n"
    
    for i in range(num_agents):
        files += f'\t<include file="drone{i}.xml"/>\n'
        
    scene_xml = scene_xml.replace("{{files}}", files)
    
    # Define the fence XML based on the world bounds
    xmin, ymin, _ = map_bounds[0]
    xmax, ymax, _ = map_bounds[1]
    
    # Define four walls to form a fence around the perimeter
    fence_xml = ""
    wall_thickness = 0.1  # thickness of the wall
    wall_height = 1.0  # height of the wall
    fence_color = "1 0 0 1"  # RGBA color for the fence (red)
    
    # Left wall
    fence_xml += f'<geom name="left_wall" type="box" size="{wall_thickness / 2} {(ymax - ymin) / 2} {wall_height / 2}" pos="{xmin - wall_thickness / 2} {(ymin + ymax) / 2} {wall_height / 2}" rgba="{fence_color}"/>\n'
    # Right wall
    fence_xml += f'<geom name="right_wall" type="box" size="{wall_thickness / 2} {(ymax - ymin) / 2} {wall_height / 2}" pos="{xmax + wall_thickness / 2} {(ymin + ymax) / 2} {wall_height / 2}" rgba="{fence_color}"/>\n'
    # Bottom wall
    fence_xml += f'<geom name="bottom_wall" type="box" size="{(xmax - xmin) / 2} {wall_thickness / 2} {wall_height / 2}" pos="{(xmin + xmax) / 2} {ymin - wall_thickness / 2} {wall_height / 2}" rgba="{fence_color}"/>\n'
    # Top wall
    fence_xml += f'<geom name="top_wall" type="box" size="{(xmax - xmin) / 2} {wall_thickness / 2} {wall_height / 2}" pos="{(xmin + xmax) / 2} {ymax + wall_thickness / 2} {wall_height / 2}" rgba="{fence_color}"/>\n'
    
    # Replace the fence placeholder
    scene_xml = scene_xml.replace("{{fence}}", fence_xml)
    
    with open(f"{save_dir}/scene.xml", "w") as f:
        f.write(scene_xml)


def create_agent_and_target_xml(drone_xml: str, drone_positions: np.ndarray, save_dir: str):
    """
    Create the XML files for drones and targets, replacing placeholders with generated positions and dimensions.

    Args:
        drone_xml (str): Template XML for a single drone.
        drone_positions (np.ndarray): Array containing the positions of drones.
        save_dir (str): Directory to save the generated drone and target XML files.
    """
    # Create the XML for the drones
    for index, pos in enumerate(drone_positions):
        updated_drone_xml = drone_xml.replace("{{index}}", str(index))
        updated_drone_xml = updated_drone_xml.replace("{{x}}", str(pos[0]))
        updated_drone_xml = updated_drone_xml.replace("{{y}}", str(pos[1]))
        updated_drone_xml = updated_drone_xml.replace("{{z}}", str(pos[2]))
        with open(f"{save_dir}/drone{index}.xml", "w") as f:
            f.write(updated_drone_xml)


def get_model(num_agents: int, spacing: float, drone_spawn_height: float, map_bounds: np.ndarray) -> MjModel:
    """
    Generate and save the scene, drones, and targets XML configuration for the simulation. Returns the MuJoCo model.

    Args:
        num_agents (int): Number of agents (drones).
        spacing (float): Distance between each agent's center.
        drone_spawn_height (float): Height at which drones are placed.
        map_bounds (np.ndarray): 2x3 array with lower and upper bounds for target positions.
    """
    scene_xml, drone_xml = read_template_files()
    
    save_dir = f"{os.path.dirname(__file__)}/generated"
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        for file in os.listdir(save_dir):
            os.remove(os.path.join(save_dir, file))
    
    drone_positions = generate_drone_positions(num_agents, spacing, drone_spawn_height)
    
    create_scene_xml(scene_xml, num_agents, map_bounds, save_dir)
    create_agent_and_target_xml(drone_xml, drone_positions, save_dir)
    
    return MjModel.from_xml_path(f"{save_dir}/scene.xml")
