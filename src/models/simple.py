import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, Dropout


class SimpleNN(nn.Module):
    def __init__(self, input_dim: int):
        super(SimpleNN, self).__init__()
        # output is 4 (0 - 1, float) for motor and 1 (0 or 1, bool) for shoot
        self.fc1 = Linear(input_dim, 64)
        self.fc2 = Linear(64, 64)
        self.fc3 = Linear(64, 5)

    def forward(self, x):
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


