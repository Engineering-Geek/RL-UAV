import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, Dropout

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.annotations import override


class SimpleNN(nn.Module, TorchModelV2):
    def __init__(self, input_dim: int):
        super(SimpleNN, self).__init__()
        # output is 4 (0 - 1, float) for motor and 1 (0 or 1, bool) for shoot
        self.fc1 = Linear(input_dim, 64)
        self.fc2 = Linear(64, 64)
        self.fc3 = Linear(64, 5)
        self.cnn = nn.Conv2d(1, 32, 8, stride=4)
        self.lstm = nn.LSTM(10, 20, 2)
        self.lstm2 = nn.LSTM(20, 30, 3)
        self.lstm3 = nn.LSTM(30, 40, 4)
        self.lstm4 = nn.LSTM(40, 50, 5)
        self.lstm5 = nn.LSTM(50, 60, 6)
        self.lstm6 = nn.LSTM(60, 70, 7)
        self.lstm7 = nn.LSTM(70, 80, 8)

    def forward(self, input_dict, state, seq_lens):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        motor_output = x[:, :4]
        shoot_output = x[:, 4:]
        return {
            "motor": motor_output,
            "shoot": shoot_output
        }


if __name__ == "__main__":
    model = SimpleNN(10)
    print(model)
    print(model(torch.rand(1, 10)))


