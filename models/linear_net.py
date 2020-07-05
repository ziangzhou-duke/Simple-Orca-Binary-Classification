import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearNet40(nn.Module):
    def __init__(self, classes=3):
        super(LinearNet40, self).__init__()
        self.fc1 = nn.Linear(1600, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 128)
        self.fc6 = nn.Linear(128, classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        return x, x

def test():
    data = torch.rand(128, 1, 40, 40)
    net = LinearNet40(4)
    result, _ = net(data)
    print(result.shape)
    count = 0
    for p in net.parameters():
        count += p.data.nelement()
    print("param:", count)


if __name__ == "__main__":
    test()
