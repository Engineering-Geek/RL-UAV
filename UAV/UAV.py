import mujoco as mj
from mujoco import MjModel, MjData
from mujoco import mjx
from pathlib import Path
from typing import Tuple, Union, List


class UAV:
    def __init__(self, xml_path: Union[str, Path], device: str = None):
        self._model: MjModel = MjModel.from_xml_path(xml_path)
        self._data: MjData = MjData(self._model)
        self._mjx_model: mjx.Model = mjx.put_model(self._model, device=device)
        self._mjx_data: mjx.Data = mjx.put_data(self._model, self._data, device=device)

