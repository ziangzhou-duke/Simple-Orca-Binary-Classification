import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.front_preact_resnet import PreActResNet34
from modules.pool_avg import AvgPool
from modules.back_fc_embd import Classifier


class PreActResNet34AvgNet(nn.Module):

    def __init__(self, classes, in_planes=16, embedding_size=64):

        super(PreActResNet34AvgNet, self).__init__()
        self.front = PreActResNet34(in_planes)
        self.pool = AvgPool()
        self.back = Classifier(classes, in_planes*8, embedding_size)

    def forward(self, x):
        out = self.front(x)
        out = self.pool(out)
        out, embd = self.back(out)
        return out, embd

        
