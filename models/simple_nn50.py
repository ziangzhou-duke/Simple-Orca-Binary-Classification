import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNet(nn.Module):
    def __init__(self, classes=3):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 4 * 3, 128)
        self.fc2 = nn.Linear(128, classes)
        self.dropout = nn.Dropout(0.5)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 3)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        emb = x
        x = self.fc2(x)
        return x, emb

def test():
    data = torch.rand(128, 1, 50, 40)
    net = SimpleNet()
    result, _ = net(data)
    print(result.shape)

if __name__ == "__main__":
    test()
