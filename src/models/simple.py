from typing import Dict, Tuple, List

import numpy as np
import torch
from ray.rllib.utils.typing import AgentID
from torch import nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, Dropout

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.annotations import override

from src.Environments.multi_agent import MultiAgentDroneEnvironment, SimpleShooter


class SimpleNN(nn.Module, TorchModelV2):
    def __init__(self, input_length: int, image_shape: Tuple[int, int, int], input_names: List[str] = None):
        super(SimpleNN, self).__init__()
        # input_length does not include the image
        # output is 4 (0 - 1, float) for motor and 1 (0 or 1, bool) for shoot
        # Note that sample_input has an 'image' key which is either a 1D or 3D tensor
        
        self.input_length = input_length
        self.image_shape = image_shape
        self.input_names = input_names
        
        self.fc1 = Linear(in_features=input_length, out_features=64)
        self.fc2 = Linear(in_features=64, out_features=64)
        
        self.image_cnn = Sequential(
            nn.Conv2d(in_channels=image_shape[0], out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        self.fc4 = Linear(in_features=64 * 7 * 7, out_features=512)
        self.fc5 = Linear(in_features=512, out_features=512)
        self.out = Linear(in_features=512, out_features=5)
        
    @override(TorchModelV2)
    def forward(self,
                input_dict: Dict[AgentID, torch.Tensor],
                state: List[torch.Tensor],
                seq_lens: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        print(input_dict.keys())
        x = torch.cat([input_dict[name] for name in self.input_names], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        image = input_dict["image"]
        image = self.image_cnn(image)
        
        x = torch.cat([x, image], dim=1)
        
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.out(x)
        
        return x, state


if __name__ == "__main__":
    env = MultiAgentDroneEnvironment(
        n_agents=2,
        n_images=1,
        frame_skip=1,
        depth_render=False,
        Drone=SimpleShooter,
    )
