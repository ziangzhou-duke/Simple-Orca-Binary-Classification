import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleLSTM(nn.Module):
    def __init__(self, output_dim):
        super(SimpleLSTM, self).__init__()
        self.feature_dim = 40
        self.hidden_dim = 128
        self.output_dim = output_dim
        self.num_layers = 4
        self.blstm = nn.LSTM(self.feature_dim, self.hidden_dim, self.num_layers, batch_first=True, dropout=0.2)
        self.hidden2out = nn.Linear(self.hidden_dim, self.output_dim)


    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[2], x.shape[3])
        self.blstm.flatten_parameters()
        # self.hidden = self.init_hidden(x.shape[1])
        x, (ht, ct) = self.blstm(x)
        x = self.hidden2out(ht[-1])
        return x, x

def test():
    data = torch.rand(256, 1, 99, 40)
    print(data.shape)
    net = SimpleLSTM(4)
    result, _ = net(data)
    print(result.shape)

if __name__ == "__main__":
    test()
