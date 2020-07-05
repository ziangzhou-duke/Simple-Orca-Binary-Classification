import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.front_cnn import ConvNet
from modules.pool_avg import AvgPool
from modules.back_fc import Classifier


class ConvAvgNet(nn.Module):

    def __init__(self, classes, input_channels=16, embedding_size=64):

        super(ConvAvgNet, self).__init__()
        self.front = ConvNet(input_channels)
        self.pool = AvgPool()
        self.back = Classifier(classes, input_channels*16, embedding_size)

    def forward(self, x):
        out = self.front(x)
        out = self.pool(out)
        out, embd = self.back(out)
        return out

        
