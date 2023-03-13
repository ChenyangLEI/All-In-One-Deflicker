from collections import namedtuple

import torch
from torchvision import models


class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X, layer):

        output = []

        h = self.slice1(X)
        h_relu1_2 = h
        output.append(h_relu1_2)
        if layer == 0:
            return output

        h = self.slice2(h)
        h_relu2_2 = h
        output.append(h_relu2_2)
        if layer == 1:
            return output

        h = self.slice3(h)
        h_relu3_3 = h
        output.append(h_relu3_3)
        if layer == 2:
            return output

        h = self.slice4(h)
        h_relu4_3 = h
        output.append(h_relu4_3)
        if layer == 3:
            return output

        #vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        #out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)

        #return out
