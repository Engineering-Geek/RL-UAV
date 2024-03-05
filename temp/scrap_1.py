from mujoco import MjModel
from mujoco.mjx import make_data, step

from brax.io import mjcf

# Load the model
model: MjModel = MjModel.from_xml_path('../models/skydio_x2/scene.xml')

# Load the model in mjcf to use with mjx
mjx_model = mjcf.load_model(model)

# Create mjx data from the mjx model
mjx_data = make_data(mjx_model)

# Step the simulation
for _ in range(5):
    mjx_data = step(mjx_model, mjx_data)

# Access sensor data from mjx_data
# mjx_data exposes the sensor data similarly to MjData, but the access might differ slightly
# depending on the Brax version and its integration with mjx
sensor_data = mjx_data.sensordata  # This will give you an array of all sensor data
print(sensor_data)

# If you know the specific index or name of the sensor, you can access it directly
# For example, if you know the index of a sensor is 0, you can do:
specific_sensor_data = sensor_data[0]

print(specific_sensor_data)
