from xml.etree import ElementTree as ET
from pprint import pprint

import re

# Let's start by reading the uploaded URDF file to understand its structure.
stl_dir = '/home/engineering-geek/PycharmProjects/RL-UAV/UAV'
urdf_path = '/UAV/assets/robot.urdf'

# Read the content of the URDF file

with open(urdf_path, 'r') as file:
    urdf_content = file.read()

# Parse the URDF XML content

urdf_xml = ET.fromstring(urdf_content)

# Initialize the MJCF XML structure

mjcf_root = ET.Element("mujoco")

worldbody = ET.SubElement(mjcf_root, "worldbody")


# Function to convert URDF origin to MJCF position (pos) and orientation (quat) attributes

def urdf_origin_to_mjcf_attributes(origin):
    # Extract xyz and rpy (roll, pitch, yaw) from URDF origin, if present

    xyz = origin.get('xyz', '0 0 0').split(' ')

    rpy = origin.get('rpy', '0 0 0').split(' ')

    # Convert rpy to quaternion for MJCF (placeholder, actual conversion requires more complex math)

    quat = "1 0 0 0"  # This is a placeholder. A proper rpy to quaternion conversion is needed here.

    return {'pos': " ".join(xyz), 'quat': quat}


# Iterate through URDF links and convert them to MJCF bodies

for link in urdf_xml.findall('link'):

    body = ET.SubElement(worldbody, "body", name=link.get('name'))

    # Handle visual elements - assuming single mesh per link for simplicity

    visual = link.find('visual')

    if visual is not None:

        geom_type = "mesh"  # Default to mesh for this example

        geom_mesh = visual.find('geometry').find('mesh').get('filename').replace('package://', stl_dir)  # Assuming mesh type for simplicity

        material = visual.find('material')

        color_rgba = "1 1 1 1"  # Default color

        if material is not None:

            color = material.find('color')

            if color is not None:
                color_rgba = color.get('rgba')

        geom = ET.SubElement(body, "geom", type=geom_type, mesh=geom_mesh, rgba=color_rgba)

        # Convert origin if present

        origin = visual.find('origin')

        if origin is not None:
            geom.attrib.update(urdf_origin_to_mjcf_attributes(origin))

# Convert MJCF structure to string for display

mjcf_str = ET.tostring(mjcf_root, encoding='unicode', method='xml')

with open('/home/engineering-geek/PycharmProjects/RL-UAV/UAV/robot.xml', 'w') as file:
    file.write(mjcf_str)
