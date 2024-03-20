from mujoco import MjModel, MjData, viewer


def debug():
    model = MjModel.from_xml_path("/home/engineering-geek/PycharmProjects/RL-UAV/src/Environments/assets/MultiUAV/tmp/scene_with_drones.xml")
    data = MjData(model)
    viewer.launch(model, data)


if __name__ == "__main__":
    debug()

