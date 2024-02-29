from mujoco import MjModel, MjData, viewer, mj_step


m = MjModel.from_xml_path('models/scene.xml')
d = MjData(m)

viewer.launch(m, d)
