from mujoco import MjModel, mj_saveLastXML, mj_saveModel


def urdf_to_mj_xml(urdf_path, xml_path):
    m = MjModel.from_xml_path(urdf_path)
    mj_saveLastXML(xml_path, m)


def urdf_to_mjcf(urdf_path, mjcf_path):
    m = MjModel.from_xml_path(urdf_path)
    mj_saveModel(mjcf_path, m)
