import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.front_resnet import ResNet34
from modules.pool_avg import AvgPool
from modules.back_fc_embd import Classifier
from encoding.nn import Encoding, Normalize


class ResNet34AvgNet(nn.Module):

    def __init__(self, classes, in_planes=16, embedding_size=64, D=256, K=64):

        super(ResNet34AvgNet, self).__init__()
        self.front = ResNet34(in_planes)
        #self.reduce = nn.Conv2d(in_planes*8, D, kernel_size=1, stride=1, padding=0, bias=False)# 1x1 convolution to reduce channels
        self.pool = Encoding(D=D, K=K)
        self.back = Classifier(classes, D*K, embedding_size)

    def forward(self, x):
        out = self.front(x)
        
        #out = self.reduce(out)
        #out = out.view(out.shape[0], out.shape[1], out.shape[3])
        #out = self.pool(out) / out.shape[2]
        out = self.pool(out) / out.shape[3]
        out = out.view(out.size()[0], -1)
        
        out, embd = self.back(out)
        return out, embd

        
