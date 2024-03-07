from mujoco import MjModel, mj_saveLastXML, mj_saveModel
from logging import getLogger

logger = getLogger(__name__)


def _convert(input_path, output_path, save_func, overwrite=False):
    """
    Convert a URDF file to another format.

    Parameters:
    urdf_path (str): Path to the URDF file.
    output_path (str): Path to the output file.
    save_func (function): Function to save the output file.
    overwrite (bool): Whether to overwrite the output file if it exists.
    """
    assert input_path.endswith('.urdf') or input_path.endswith('.xml'), 'input_path should end with .urdf or .xml'
    logger.debug(f'Converting {input_path} to {output_path}')
    if not overwrite:
        try:
            with open(output_path, 'r'):
                logger.warning(f'{output_path} already exists. Set overwrite=True to overwrite')
                return
        except FileNotFoundError:
            logger.error(f'File {output_path} not found')
            pass
    m = MjModel.from_xml_path(input_path)
    save_func(output_path, m)
    logger.info('Remember to manually edit the output file')


def urdf_to_mj_xml(urdf_path, xml_path, overwrite=False):
    assert xml_path.endswith('.xml'), 'xml_path should end with .xml'
    _convert(urdf_path, xml_path, mj_saveLastXML, overwrite)


def urdf_to_mjcf(urdf_path, mjcf_path, overwrite=False):
    assert mjcf_path.endswith('.mjcf'), 'mjcf_path should end with .mjcf'
    # must be in the same directory; mjcf is a binary format
    assert urdf_path[:-5] == mjcf_path[:-5], 'urdf_path and mjcf_path should have the same name'
    _convert(urdf_path, mjcf_path, mj_saveModel, overwrite)


def mj_xml_to_mjcf(xml_path, mjcf_path, overwrite=False):
    assert mjcf_path.endswith('.mjcf'), 'mjcf_path should end with .mjcf'
    # must be in the same directory; mjcf is a binary format
    assert xml_path[:-4] == mjcf_path[:-5], 'xml_path and mjcf_path should have the same name'
    model: MjModel = MjModel.from_xml_path(xml_path)
    with open(mjcf_path, 'wb') as f:
        mj_saveModel(model, f)
