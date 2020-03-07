
import math
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from .layers.misc import (
    BatchNorm2d,
    Conv2d,
    interpolate,
    _NewEmptyTensorOp
)

NORMAL_PRIMITIVES = {
    "block_type" : ["MB"],
    "kernel_size" : [3, 5, 7],
    "stride" : [1],
    "channel_size" : [-1],
    "expansion" : [3, 6],
    "group" : [1],
    "se": [True, False],
    "activation" : ["relu"]
}

NORMAL_CS_PRIMITIVES = {
    "block_type" : ["MB"],
    "kernel_size" : [3, 5, 7],
    "stride" : [1],
    "channel_size" : [i for i in range(1, 40)],
    "expansion" : [3, 6],
    "group" : [1],
    "se": [True, False],
    "activation" : ["relu"]
}

STRIDE_PRIMITIVES = {
    "block_type" : ["MB"],
    "kernel_size" : [3, 5, 7],
    "stride" : [2],
    "channel_size" : [i for i in range(1, 40)],
    "expansion" : [3, 6],
    "group" : [1],
    "se": [True, False],
    "activation" : ["relu"]
}

def make_divisible(x, divisible_by=8):
    return int(np.ceil(x*1. / divisible_by)* divisible_by)

def get_layer_structure(kernel, 
                              stride, 
                              channel_size, 
                              expansion,
                              group,
                              se,
                              activation,
                              block_type="MB"
                            ):
        block_name = "{}_k{}_s{}_cs{}_e{}_g{}_se{}_at{}".format(
                                                          block_type,
                                                          kernel, 
                                                          stride, 
                                                          channel_size, 
                                                          expansion,
                                                          group, 
                                                          se, 
                                                          activation)
        block_name = block_name.replace(".", "-")
        return {
            block_name : lambda input_depth, **kwargs : PDBlock(
                    input_depth,
                    channel_size, 
                    kernel,
                    stride,
                    activation,
                    block_type=block_type,
                    expansion= expansion,
                    group=group,
                    se=se,
                )
            }

def conv_1x1_bn(input_depth, output_depth):
    return nn.Sequential(
        nn.Conv2d(input_depth, output_depth, 1, 1, 0, bias=False),
        nn.BatchNorm2d(output_depth),
        nn.ReLU6(inplace=True)
    )


class HSwish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x+3, inplace=True) / 6
        return out

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


class ConvBNRelu(nn.Sequential):
    def __init__(self, 
                 input_depth,
                 output_depth,
                 kernel,
                 stride,
                 pad,
                 activation="relu",
                 group=1,
                 *args,
                 **kwargs):

        super(ConvBNRelu, self).__init__()

        assert activation in ["hswish", "relu", None]
        assert stride in [1, 2, 4]

        self.add_module("conv", nn.Conv2d(input_depth, output_depth, kernel, stride, pad, groups=group, bias=False))
        self.add_module("bn", nn.BatchNorm2d(output_depth))

        if activation == "relu":
            self.add_module("relu", nn.ReLU6(inplace=True))
        elif activation == "hswish":
            self.add_module("hswish", HSwish())

class SEModule(nn.Module):
    reduction = 4
    def __init__(self, C):
        super(SEModule, self).__init__()
        mid = max(C // self.reduction, 8)
        conv1 = Conv2d(C, mid, 1, 1, 0)
        conv2 = Conv2d(mid, C, 1, 1, 0)

        self.operation = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), conv1, nn.ReLU(inplace=True), conv2, nn.Sigmoid()    
            )

    def forward(self, x):
        return x * self.operation(x)


class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        N, C, H, W = x.size()
        g = self.groups
        return x.view(N,g,int(C//g),H,W).permute(0,2,1,3,4).contiguous().view(N,C,H,W)

class BasicConv(nn.Module):
    def __init__(self, 
                input_depth, 
                output_depth, 
                kernel, 
                stride, 
                activation,
                group=1,
                se=False,
                *args,
                **kwargs):

        super(BasicConv, self).__init__()
        self.use_res_connect = True if (stride==1 and input_depth == output_depth) else False

        self.conv1 = ConvBNRelu(input_depth, 
                                input_depth,
                                kernel=kernel,
                                stride=1,
                                pad=(kernel//2),
                                activation=activation)

        self.conv2 = ConvBNRelu(input_depth,
                                output_depth,
                                kernel=kernel,
                                stride=stride,
                                pad=(kernel//2),
                                activation=activation)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)

        y = y + x if self.use_res_connect else y
        return y

class MBConv(nn.Module):
    def __init__(self,
                 input_depth,
                 output_depth,
                 expansion,
                 kernel,
                 stride,
                 activation,
                 group=1,
                 se=False,
                 *args,
                 **kwargs):
        super(MBConv, self).__init__()
        self.use_res_connect = True if (stride==1 and input_depth == output_depth) else False
        mid_depth = int(input_depth * expansion)

        self.group = group
        #if self.group > 1:
        #    self.shuffle = ShuffleBlock(group)

        if input_depth == mid_depth:
            self.point_wise = nn.Sequential()
        else:
            self.point_wise = ConvBNRelu(input_depth,
                                         mid_depth,
                                         kernel=1,
                                         stride=1,
                                         pad=0,
                                         activation=activation,
                                         group=group,
                                    )

        self.depthwise = ConvBNRelu(mid_depth,
                                    mid_depth,
                                    kernel=kernel,
                                    stride=stride,
                                    pad=(kernel//2),
                                    activation=activation,
                                    group=mid_depth,
                                )

        self.point_wise_1 = ConvBNRelu(mid_depth,
                                       output_depth,
                                       kernel=1,
                                       stride=1,
                                       pad=0,
                                       activation=None,
                                       group=group,
                                    )
        self.se = SEModule(mid_depth) if se else None

    def forward(self, x):
        y = self.point_wise(x)
        y = self.depthwise(y)

        #y = self.shuffle(y) if self.group > 1 else y

        y = self.se(y) if self.se is not None else y
        y = self.point_wise_1(y)

        y = y + x if self.use_res_connect else y
        return y



class PDBlock(nn.Module):
    def __init__(self, 
                 input_depth,
                 channel_size,
                 kernel,
                 stride,
                 activation="relu",
                 block_type="MB",
                 expansion=1,
                 group=1,
                 se=False):

        super(PDBlock, self).__init__()
        
        #output_depth = int(input_depth+channel_size*8)
        output_depth = int(16+channel_size*8) if channel_size != -1 else int(input_depth)

        if block_type == "MB":
            self.block = MBConv(input_depth,
                                output_depth,
                                expansion,
                                kernel,
                                stride,
                                activation,
                                group,
                                se)

        elif block_type == "BC":
            self.block = BasicConv(input_depth,
                                   output_depth,
                                   kernel,
                                   stride,
                                   activation,
                                   group,
                                   se)
            


    def forward(self, x):
        y = self.block(x)
        return y


class ApproximateBlock(nn.Module):
    def __init__(self, n_class=1000, input_size=16, width_mult=1., input_channel=16, interverted_residual_setting=None):
        super(ApproximateBlock, self).__init__()
        block = InvertedResidual
        last_channel = 1280
        self.start_index = 0
        if interverted_residual_setting is None:
                interverted_residual_setting = IMAGENET_APPROXIMATE

        self.last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = nn.ModuleList()
        # building inverted residual blocks
        for index, (t, c, n, s, k) in enumerate(interverted_residual_setting):
            output_channel = make_divisible(c * width_mult) if t > 1 else c
            for i in range(n):
                if i == 0:
                    self.features.add_module(str(index), block(input_channel, output_channel, s, expand_ratio=t, kernel=k))
                else:
                    self.features.add_module(str(index), block(input_channel, output_channel, 1, expand_ratio=t, kernel=k))
                input_channel = output_channel
        # building last several layers
        self.features.add_module("last", conv_1x1_bn(input_channel, self.last_channel))
        self.drop = nn.Dropout(0.2)

        # building classifier
        self.classifier = nn.Linear(self.last_channel, n_class)

        self._initialize_weights()

    def forward(self, x):
        for l in self.features[self.start_index:]:
            x = l(x)
#         x = x.mean(3).mean(2)
#         x = self.drop(x)
#         x = self.classifier(x)
        return x

    def add_index(self):
        self.start_index += 1

    def set_index(self, index):
        self.start_index = index

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()    
def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, kernel):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        
        if expand_ratio == 1:
            self.point_wise = nn.Sequential()
        else:
            self.point_wise = ConvBNRelu(inp,
                                         hidden_dim,
                                         kernel=1,
                                         stride=1,
                                         pad=0,
                                         activation="relu",
                                         group=1,
                                    )

        self.depthwise = ConvBNRelu(hidden_dim,
                                    hidden_dim,
                                    kernel=kernel,
                                    stride=stride,
                                    pad=(kernel//2),
                                    activation="relu",
                                    group=hidden_dim,
                                )

        self.point_wise_1 = ConvBNRelu(hidden_dim,
                                       oup,
                                       kernel=1,
                                       stride=1,
                                       pad=0,
                                       activation=None,
                                       group=1,
                                    )

    def forward(self, x):
        if self.use_res_connect:
            y = self.point_wise(x)
            y = self.depthwise(y)
            y = self.point_wise_1(y)
            return x + y
        else:
            y = self.point_wise(x)
            y = self.depthwise(y)
            y = self.point_wise_1(y)
            return y


