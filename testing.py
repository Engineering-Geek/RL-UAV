from mujoco import MjModel, MjData, viewer


m = MjModel.from_xml_path('UAV/scene.xml')
# m = MjModel.from_xml_path('mujoco_menagerie/skydio_x2/scene.xml')
viewer.launch(m)

