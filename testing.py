from mujoco import MjModel, MjData, viewer, mj_step


m = MjModel.from_xml_path('models/simple_scene.xml')
# m = MjModel.from_xml_path('mujoco_menagerie/skydio_x2/simple_scene.xml')
# m = MjModel.from_xml_path('simple_scene.xml')
d = MjData(m)

viewer.launch(m, d)
