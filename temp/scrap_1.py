from mujoco import MjModel
from mujoco.mjx import make_data, step, get_data
from time import time

from brax.io import mjcf

# Load the model
model: MjModel = MjModel.from_xml_path('../models/skydio_x2/scene.xml')

# Load the model in mjcf to use with mjx
mjx_model = mjcf.load_model(model)

# Create mjx data from the mjx model
mjx_data = make_data(mjx_model)

# Step the simulation
start = time()
mjx_data = step(mjx_model, mjx_data)
print(f"Time taken: {time() - start}")
# Print the current position of the drone

print(mjx_data.qpos)

# cast mjx_data to MjData
# data = get_data(model, mjx_data)
