import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.front_resnet import ResNet34
from modules.pool_avg import AvgPool
from modules.back_fc_embd import Classifier
from modules.reverse_gradient import ReverseLayerF

class ResNet34AvgDANet(nn.Module):

    def __init__(self, num_class, num_domain, in_planes=16, embedding_size=[512, 64]):

        super(ResNet34AvgDANet, self).__init__()
        self.front = ResNet34(in_planes)
        self.pool = AvgPool()
        self.speaker_classifier = Classifier(num_class, in_planes*8, embedding_size[0])
        self.domain_classifier = Classifier(num_domain, in_planes*8, embedding_size[1])

    def forward(self, x, alpha, is_target=False, is_diagnoise=False):
        feature = self.front(x)
        feature = self.pool(feature)
        
        if is_diagnoise==True:
            return feature
        
        if is_target == False:
            class_out, embd = self.speaker_classifier(feature)
            reverse_feature = ReverseLayerF.apply(feature, alpha)
            domain_out, _ = self.domain_classifier(reverse_feature)
            return class_out, domain_out, embd
        else:
            reverse_feature = ReverseLayerF.apply(feature, alpha)
            domain_out, _ = self.domain_classifier(reverse_feature)
            return domain_out

        
