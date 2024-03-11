from mujoco import MjModel, MjData, viewer


def debug():
    model = MjModel.from_xml_path("ReducedUAV/scene.xml")
    data = MjData(model)
    viewer.launch(model, data)


if __name__ == "__main__":
    debug()

