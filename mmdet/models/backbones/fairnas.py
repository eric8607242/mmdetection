  
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..registry import BACKBONES

def stem(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def separable_conv(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, 1, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU6(inplace=True),
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
    )


def conv_before_pooling(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, kernel_size, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]
        self.stride = stride
        padding = kernel_size // 2
        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        self.conv1 = nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, padding, groups=hidden_dim, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.conv3 = nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(oup)

    def forward(self, x):
        inputs = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu6(x, inplace=True)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu6(x, inplace=True)
        x = self.conv3(x)
        x = self.bn3(x)
        if self.use_res_connect:
            return inputs + x
        else:
            return x

@BACKBONES.register_module
class FairNasA(nn.Module):
    def __init__(self, n_class=1000, input_size=224):
        super(FairNasA, self).__init__()
        assert input_size % 32 == 0
        mb_config = [
            # expansion, out_channel, kernel_size, stride,
            [3, 32, 5, 2],
            [3, 32, 3, 1],
            [3, 40, 7, 2],
            [3, 40, 3, 1],
            [3, 40, 3, 1],
            [3, 40, 3, 1],
            [3, 80, 3, 2],
            [3, 80, 3, 1],
            [3, 80, 3, 1],
            [6, 80, 3, 1],
            [3, 96, 3, 1],
            [3, 96, 3, 1],
            [3, 96, 3, 1],
            [3, 96, 3, 1],
            [6, 192, 7, 2],
            [6, 192, 7, 1],
            [6, 192, 3, 1],
            [6, 192, 3, 1],
            [6, 320, 5, 1],
        ]
        input_channel = 16
        last_channel = 1280

        self.last_channel = last_channel
        self.stem = stem(3, 32, 2)
        self.separable_conv = separable_conv(32, 16)
        self.mb_module = list()
        for t, c, k, s in mb_config:
            output_channel = c
            self.mb_module.append(InvertedResidual(input_channel, output_channel, k, s, expand_ratio=t))
            input_channel = output_channel
        self.mb_module = nn.ModuleList(self.mb_module)
        self.conv_before_pooling = conv_before_pooling(input_channel, self.last_channel)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, n_class),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.separable_conv(x)
        out = []
        for i, l in enumerate(self.mb_module):
            x = l(x)
            if i in [1, 5, 13, 17]:
                out.append(x)
#         x = self.mb_module(x)
        x = self.conv_before_pooling(x)
#         out.append(x)
#         x = x.mean(3).mean(2)
#         x = self.classifier(x)
        return tuple(out)
    
    def init_weights(self, pretrained=False):
        model_dict = torch.load("/home/jovyan/mmdetection/mmdet/models/backbones/FairNAS/pretrained/FairNasC.pth.tar")
        self.load_state_dict(model_dict["model_state"])
        
