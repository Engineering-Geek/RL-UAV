import mujoco as mj
import mujoco.viewer
import numpy as np

# Generate random heights
heights = np.random.rand(10000)  # For a 100x100 heightfield

# Load the XML model
XML = """
<mujoco>
    <worldbody>
        <body name="terrain" pos="0 0 0">
            <geom name="terrain_geom" type="hfield" hfield="hfield_asset" size="1 1 1" pos="0 0 0"/>
        </body>
    </worldbody>
    <asset>
        <hfield name="hfield_asset" file="heightmap.png" size="10 10 1 1"/>
    </asset>
</mujoco>

"""
model: mj.MjModel = mj.MjModel.from_xml_string(XML)

# Update the model's heightfield data
# model.hfield_data[:] = heights.flatten()
mj.viewer.launch(model)
