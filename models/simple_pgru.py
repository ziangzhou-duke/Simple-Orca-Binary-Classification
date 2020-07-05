import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleGRU(nn.Module):
    def __init__(self, output_dim):
        super(SimpleGRU, self).__init__()
        self.feature_dim = 40
        self.hidden_dim = 128
        self.output_dim = output_dim
        self.num_layers = 2
        self.gru = nn.GRU(self.feature_dim, self.hidden_dim, self.num_layers, batch_first=True, dropout=0.2)
        self.hidden2out = nn.Linear(self.hidden_dim, self.output_dim)


    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[2], x.shape[3])
        self.gru.flatten_parameters()
        # self.hidden = self.init_hidden(x.shape[1])
        x, ht = self.gru(x)
        # print(x.shape, ht.shape)
        # print(x[-1].shape)
        # print(x[:,-1,:].shape)
        x = self.hidden2out(x[:,-1,:])
        return x, x

def test():
    data = torch.rand(256, 1, 99, 40)
    print(data.shape)
    net = SimpleGRU(4)
    result, _ = net(data)
    print(result.shape)

if __name__ == "__main__":
    test()
