import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleDNN(nn.Module):
    def __init__(self):
        super(SimpleDNN, self).__init__()
        self.fc0 = nn.Linear(40 * 40, 128)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.fc0(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x, x

def test():
    data = torch.rand(128, 1, 40, 40)
    net = SimpleDNN()
    result, _ = net(data)
    print(result.shape)


test()
