import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUAvg(nn.Module):
    def __init__(self, classes):
        super(GRUAvg, self).__init__()
        self.feature_dim = 40
        self.hidden_dim = 128
        self.output_dim = classes
        self.num_layers = 2
        self.gru = nn.GRU(self.feature_dim, self.hidden_dim, self.num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(self.hidden_dim, classes)

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[2], x.shape[3])
        self.gru.flatten_parameters()
        # self.hidden = self.init_hidden(x.shape[1])
        x, ht = self.gru(x)
        x = x.mean(1)
        x = self.fc(x)
        return x, x

def test():
    data = torch.rand(256, 1, 100, 40)
    net = GRUAvg(4)
    result, _ = net(data)
    print(result.shape)

if __name__ == "__main__":
    test()
