import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyTrans(nn.Module):

    def __init__(self, classes=4):
        super(TinyTrans, self).__init__()
        self.fc_front = nn.Linear(40, 256)
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.fc = nn.Linear(256, 30)

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[2], x.shape[3])
        x = x.transpose(0, 1)
        x = self.fc_front(x)
        x = self.transformer_encoder(x)
        x = x.mean(axis=0)
        x = self.fc(x)
        return x, x

def test():
    a = torch.rand(128, 1, 40, 40)
    net = TinyTrans(4)
    c, d = net(a)
    print(c.shape)

test()
