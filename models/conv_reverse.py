import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.reverse_gradient import ReverseLayerF

class ConvNet40Reverse(nn.Module):
    def __init__(self, classes=3):
        super(ConvNet40Reverse, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, classes)
        self.fc3 = nn.Linear(128, 2)
        self.dropout = nn.Dropout(0.5)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x, LAMBDA):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        emb = x
        x1 = self.fc2(emb)
        reverse_x = ReverseLayerF(emb, LAMBDA)
        x2 = self.fc3(emb)
        return x1, x2, emb

class ConvNet40DDC(nn.Module):
    def __init__(self, classes=3):
        super(ConvNet40DDC, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, classes)
        self.dropout = nn.Dropout(0.5)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x1, x2):
        x1 = self.pool(F.relu(self.conv1(x1)))
        x1 = self.pool(F.relu(self.conv2(x1)))
        x1 = self.pool(F.relu(self.conv3(x1)))
        x1 = x1.view(-1, 64 * 3 * 3)
        x1 = F.relu(self.fc1(x1))
        x1 = self.dropout(x1)
        emb1 = x1

        x2 = self.pool(F.relu(self.conv1(x2)))
        x2 = self.pool(F.relu(self.conv2(x2)))
        x2 = self.pool(F.relu(self.conv3(x2)))
        x2 = x2.view(-1, 64 * 3 * 3)
        x2 = F.relu(self.fc1(x2))
        x2 = self.dropout(x2)
        emb2 = x2

        out = self.fc2(x1)
        return out, emb1, emb2

def test():
    data = torch.rand(128, 1, 40, 40)
    net = ConvNet40(4)
    result, _ = net(data)
    print(result.shape)
    count = 0
    for p in net.parameters():
        count += p.data.nelement()
    print("param:", count)


if __name__ == "__main__":
    test()
