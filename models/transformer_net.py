import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.front_resnet import ResNet34
from modules.pool_avg import AvgPool
from modules.back_fc_embd import Classifier

class TinyTrans(nn.Module):

    def __init__(self, classes=0, in_planes=16, embedding_size=64):

        super(TinyTrans, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=40, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.fc = nn.Linear(64, 4)

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[2], x.shape[3])
        x = x.transpose(0, 1)
        out = self.transformer_encoder(x)
        print(out.shape)
        out = torch.mean(out, dim=0, keepdim=False)
        out = self.fc(out)
        return out, out

def test():
    net = TinyTrans()
    # a = torch.rand(99, 128, 64)
    a = torch.rand(128, 1, 40, 40)
    b, _ = net(a)
    print(b.shape)

test()

        
