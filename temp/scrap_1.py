import numpy as np
from mujoco import MjModel, MjData, Renderer, MjvOption, mj_step, viewer, mj_saveModel
from mujoco._structs import _MjContactList
from mujoco.mjx import make_data, step, get_data
import time
import cv2

from brax.io import mjcf

# Load the model
model: MjModel = MjModel.from_xml_path('../models/skydio_x2/scene.xml')
data: MjData = MjData(model)

# data.ctrl = np.array([3.] * 4)
# viewer.launch(model, data)

# Create a renderer
# renderer: Renderer = Renderer(model, 128, 128)
# print(dir(data.camera('front_camera_left')))

# frames = []
# with viewer.launch_passive(model, data) as v:
for _ in range(5000):
    # Step the simulation
    mj_step(model, data)
    # v.sync()
    # time.sleep(0.01)
    if data.ncon > 0:
        contact_list: _MjContactList = data.contact
        for i in range(data.ncon):
            print(model.geom(data.contact[i].geom1).name, model.geom(data.contact[i].geom2).name)
        break
        # print(data.body('target_body').xpos)
#     renderer.update_scene(data, 0)
#     frames.append(renderer.render())
#     print(data.xpos[1])
#
# cv2.namedWindow('image', cv2.WINDOW_NORMAL)
# for frame in frames:
#     cv2.imshow('image', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
#     cv2.waitKey(1)
#     time.sleep(0.01)
# cv2.destroyAllWindows()
# renderer.close()
