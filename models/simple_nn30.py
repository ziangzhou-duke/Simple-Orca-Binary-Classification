import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNet30(nn.Module):
    def __init__(self, classes=3):
        super(SimpleNet30, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(64 * 2 * 3, 128)
        self.fc2 = nn.Linear(128, classes)
        self.dropout = nn.Dropout(0.5)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # print(x.shape)
        x = x.view(-1, 64 * 2 * 3)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        emb = x
        x = self.fc2(x)
        return x, emb

def test():
    data = torch.rand(128, 1, 32, 40)
    net = SimpleNet30()
    result, _ = net(data)
    print(result.shape)

if __name__ == "__main__":
    test()
