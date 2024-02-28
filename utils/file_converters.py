from mujoco import MjModel, mj_saveLastXML, mj_saveModel
from logging import getLogger

logger = getLogger(__name__)


def convert_urdf(urdf_path, output_path, save_func, overwrite=False):
    """
    Convert a URDF file to another format.

    Parameters:
    urdf_path (str): Path to the URDF file.
    output_path (str): Path to the output file.
    save_func (function): Function to save the output file.
    overwrite (bool): Whether to overwrite the output file if it exists.
    """
    assert urdf_path.endswith('.urdf'), 'urdf_path should end with .urdf'
    logger.debug(f'Converting {urdf_path} to {output_path}')
    if not overwrite:
        try:
            with open(output_path, 'r'):
                logger.warning(f'{output_path} already exists. Set overwrite=True to overwrite')
                return
        except FileNotFoundError:
            logger.error(f'File {output_path} not found')
            pass
    m = MjModel.from_xml_path(urdf_path)
    save_func(output_path, m)
    logger.info('Remember to manually edit the output file')


def urdf_to_mj_xml(urdf_path, xml_path, overwrite=False):
    assert xml_path.endswith('.xml'), 'xml_path should end with .xml'
    convert_urdf(urdf_path, xml_path, mj_saveLastXML, overwrite)


def urdf_to_mjcf(urdf_path, mjcf_path, overwrite=False):
    assert mjcf_path.endswith('.mjcf'), 'mjcf_path should end with .mjcf'
    # must be in the same directory; mjcf is a binary format
    assert urdf_path[:-5] == mjcf_path[:-5], 'urdf_path and mjcf_path should have the same name'
    convert_urdf(urdf_path, mjcf_path, mj_saveModel, overwrite)
