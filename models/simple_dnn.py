import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleDNN(nn.Module):
    def __init__(self, classes=3):
        super(SimpleDNN, self).__init__()
        self.fc1 = nn.Linear(40 * 40, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, 512)
        self.fc6 = nn.Linear(512, classes)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x, x

def test():
    data = torch.rand(128, 1, 40, 40)
    net = SimpleDNN()
    result, _ = net(data)
    print(result.shape)

if __name__ == "__main__":
    test()
