import math
import random
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models import load_checkpoint

from ..registry import BACKBONES

class ConvBNRelu(nn.Sequential):
    def __init__(self, 
                 input_channel,
                 output_channel,
                 kernel,
                 stride,
                 pad,
                 activation="hswish",
                 bn=True,
                 group=1,
                 *args,
                 **kwargs):

        super(ConvBNRelu, self).__init__()

        assert activation in ["hswish", "relu", None]
        assert stride in [1, 2, 4]

        self.add_module("conv", nn.Conv2d(input_channel, output_channel, kernel, stride, pad, groups=group, bias=False))
        if bn:
            self.add_module("bn", nn.BatchNorm2d(output_channel, momentum=0.001))

        if activation == "relu":
            self.add_module("relu", nn.ReLU6(inplace=True))
        elif activation == "hswish":
            self.add_module("hswish", HSwish())

class HSigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(HSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        out = F.relu6(x + 3, inplace=self.inplace) / 6
        return out
            
class SEModule(nn.Module):
    def __init__(self,  in_channel,
                 reduction=4,
                 squeeze_act=nn.ReLU(inplace=True),
                 excite_act=HSigmoid(inplace=True)):
        super(SEModule, self).__init__()
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.squeeze_conv = nn.Conv2d(in_channels=in_channel,
                                      out_channels=in_channel // reduction,
                                      kernel_size=1,
                                      bias=True)
        self.squeeze_act = squeeze_act
        self.excite_conv = nn.Conv2d(in_channels=in_channel // reduction,
                                     out_channels=in_channel,
                                     kernel_size=1,
                                     bias=True)
        self.excite_act = excite_act

    def forward(self, inputs):
        feature_pooling = self.global_pooling(inputs)
        feature_squeeze_conv = self.squeeze_conv(feature_pooling)
        feature_squeeze_act = self.squeeze_act(feature_squeeze_conv)
        feature_excite_conv = self.excite_conv(feature_squeeze_act)
        feature_excite_act = self.excite_act(feature_excite_conv)
        return inputs * feature_excite_act

def conv_1x1_bn(input_channel, output_channel):
    return nn.Sequential(
        nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=False),
        nn.BatchNorm2d(output_channel, momentum=0.001),
#         nn.ReLU6(inplace=True)
        HSwish()
    )

class Flatten(nn.Module):
    def forward(self, x):
        return x.mean(3).mean(2)


class HSwish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x+3, inplace=True) / 6
        return out

class MixedBlock(nn.Module):
    def __init__(self,
                 input_channel,
                 output_channel,
                 stride,
                 split_block=4,
                 kernels=[3, 5, 7, 9],
                 axis=1,
                 group=1,
                 activation="hswish",
                 block_type="MB"):
        super(MixedBlock, self).__init__()
        self.blocks = nn.ModuleList()

        self.skip_index = None
        for i, k in enumerate(kernels):
            if k == "skip":
                operation = nn.Conv2d(input_channel, output_channel, 1, stride, 0, bias=False)
                self.skip_index = i
            else:
                operation = ConvBNRelu(
                            input_channel,
                            output_channel,
                            kernel=k,
                            stride=stride,
                            pad=(k//2),
                            activation=activation,
                            group=group,
                        )
            self.blocks.append(operation)
        self.split_block = split_block

        # Order without skip operation
        self.split_block = split_block-1
        self.order = np.random.permutation(self.split_block)
        self.index = 0
        random.shuffle(self.order)
        # =============================
        self.skip_operation = False

        self.arch_param = None

    def forward(self, x, arch_flag=False):
        if not arch_flag:
            index = self.order[self.index] if not self.skip_operation else self.skip_index
            x = self.blocks[index](x) if index != self.skip_index else self.blocks[index](x)*0
            return x, self.skip_operation
        else:
            block_probability = self.arch_param.cuda()
            # If skip connection, then output set 0
            x = sum(b(x)*p if i != self.skip_index else b(x)*0 for i, (b, p) in enumerate(zip(self.blocks, block_probability)) if p > 1e-2)
            return x, False
    
    def set_arch_param(self, arch_param):
        self.arch_param = arch_param

    def set_training_order(self, reset=False, skip=False):
        """
        Choose the convolution operation. If skip is true, choose skip operation
        """
        self.skip_operation = False
        if reset:
            self.order = np.random.permutation(self.split_block)
            random.shuffle(self.order)
            self.index = 0

        if skip:
            self.skip_operation = True
        else:
            self.index += 1
            if self.index == self.split_block:
                self.order = np.random.permutation(self.split_block)
                random.shuffle(self.order)
                self.index = 0

class MPDBlock(nn.Module):
    """Mixed path depthwise block"""
    def __init__(self,
                 input_channel,
                 output_channel,
                 stride,
                 split_block=4,
                 kernels=[3, 5, 7, 9],
                 axis=1,
                 activation="hswish",
                 block_type="MB",
                 search=False):
        super(MPDBlock, self).__init__()
        self.block_input_channel = input_channel//split_block
        self.block_output_channel = output_channel//split_block

        self.split_block = len(kernels)
        self.blocks = nn.ModuleList()

        for b in range(split_block):
            if search:
                operation = MixedBlock(
                            self.block_input_channel,
                            self.block_output_channel,
                            stride=stride,
                            split_block=len(kernels),
                            kernels=kernels,
                            group=self.block_output_channel,
                            activation=activation,
                        )
            else:
                operation = nn.Conv2d(
                            self.block_input_channel, 
                            self.block_output_channel, 
                            kernels[b], 
                            stride, 
                            (kernels[b]//2), 
                            groups=self.block_output_channel, 
                            bias=False)
            self.blocks.append(operation)
            
        if not search:
            self.bn = nn.BatchNorm2d(output_channel, momentum=0.001)
#             self.relu = nn.ReLU6(inplace=True)
            self.act_fn = HSwish()

        self.axis = axis 
        self.search = search

    def forward(self, x, arch_flag=False):
        split_x = torch.split(x, self.block_input_channel, dim=self.axis)
        # =================== SBN
        skip_connection_num = 0
        output_list = []
        skip_flag = False
        for x_i, conv_i in zip(split_x, self.blocks):
            if self.search:
                output, skip_flag = conv_i(x_i, arch_flag)
                skip_connection_num += 1 if skip_flag else 0
            else:
                output = conv_i(x_i)

            output_list.append(output)

        x = torch.cat(output_list, dim=self.axis)

        if not self.search:
            x = self.bn(x)
#             x = self.relu(x)
            x = self.act_fn(x)
        return x, skip_connection_num
        # ===================

    def set_arch_param(self, arch_param):
        for i, l in enumerate(self.blocks):
            l.set_arch_param(arch_param[i*self.split_block:(i+1)*self.split_block])


    def set_training_order(self, active_block, reset=False, static=False):
        # ================ Choose active_block
        if static and active_block == 0:
            # At least choose one block in static layer
            active_block = random.randint(1, len(self.blocks))

        #active_order = np.random.permutation(len(self.blocks))
        active_order = np.array([0, 1, 2, 3, 4, 5])
        active_blocks = active_order[:active_block]

        for b_num, b in enumerate(self.blocks):
            if b_num in active_blocks:
                b.set_training_order(reset, skip=False)
            else:
                b.set_training_order(reset, skip=True)

class MBConv(nn.Module):
    def __init__(self,
                 input_channel,
                 output_channel,
                 expansion,
                 kernels,
                 stride,
                 activation,
                 split_block=1,
                 group=1,
                 se=True,
                 search=False,
                 *args,
                 **kwargs):
        super(MBConv, self).__init__()
        self.use_res_connect = True if (stride==1 and input_channel == output_channel) else False
        mid_depth = int(input_channel * expansion)

        self.group = group

        if input_channel == mid_depth:
            self.point_wise = nn.Sequential()
        else:
            self.point_wise = ConvBNRelu(input_channel,
                                         mid_depth,
                                         kernel=1,
                                         stride=1,
                                         pad=0,
                                         activation=activation,
                                         group=group,
                                    )

        self.depthwise = MPDBlock(mid_depth,
                                  mid_depth,
                                  stride,
                                  split_block=expansion,
                                  kernels=kernels,
                                  activation=activation,
                                  search=search)

        self.point_wise_1 = ConvBNRelu(mid_depth,
                                       output_channel,
                                       kernel=1,
                                       stride=1,
                                       pad=0,
                                       activation=None,
                                       bn=False if search else True,
                                       group=group,
                                    )
        self.se = SEModule(mid_depth) if se else None

        self.expansion = expansion
        self.search = search
        if self.search:
            # reference : https://arxiv.org/pdf/1908.06022.pdf
            self.learnable_stabilizer = nn.Conv2d(input_channel, output_channel, 1, stride, 0, bias=False)
            # ====================
            # reference : https://arxiv.org/pdf/2001.05887.pdf
            self.sbn = nn.ModuleList()
            for i in range(expansion):
                self.sbn.append(nn.BatchNorm2d(output_channel, momentum=0.001))
            self.skip_connection_num = 0
            # ===================
            self.input_temp = None

            # =================== Training order
            self.index = 0

            self.order = [2, 3, 4, 5, 6]
            self.order = np.array(self.order)
            random.shuffle(self.order)

            self.order_distill = [2, 3, 4, 5, 6]
            self.order_distill = np.array(self.order_distill)
            random.shuffle(self.order_distill)
            # ===================

    def forward(self, x, arch_flag=False):
        y = self.point_wise(x)
        y, skip_connection_num = self.depthwise(y) if not arch_flag else self.depthwise(y, arch_flag)

        y = self.se(y) if self.se is not None else y
        if self.search:
            self.input_temp = y
        y = self.point_wise_1(y)

        if self.search:
            # ============== SBN
            if arch_flag:
                skip_connection_num = round(self.skip_connection_num)
            if skip_connection_num != self.expansion:
                y = self.sbn[skip_connection_num](y)
            # ==============

        if skip_connection_num == self.expansion and self.search:
            y = x
            # Skip connection
            #y = self.learnable_stabilizer(y)
        y = y + x if self.use_res_connect else y

        return y

    def set_arch_param(self, arch_param):
        self.depthwise.set_arch_param(arch_param)

        # Count continue skip num
        self.skip_connection_num = 0
        kernel_nums = len(arch_param) // self.expansion
        for i in range(0, len(arch_param), kernel_nums):
            split_arch_param = arch_param[i:i+kernel_nums]
            self.skip_connection_num += split_arch_param[-1].item()

    def set_training_order(self,reset=False, state=None, static=False):
        expansion = None
        if reset:
            self.index = 0
            random.shuffle(self.order)
            random.shuffle(self.order_distill)
            if state is None:
                expansion = self.order[self.index]
            else:
                expansion = self.order_distill[self.index]
            self.index += 1
        else:
            if state == "Max":
                expansion = 6
            elif state == "Min":
                expansion = 1
            elif state == "Random":
                expansion = self.order_distill[self.index]

                self.index += 1
                if self.index == len(self.order_distill):
                    self.index = 0
                    random.shuffle(self.order_distill)
            else:
                expansion = self.order[self.index]

                self.index += 1
                if self.index == len(self.order):
                    self.index = 0
                    random.shuffle(self.order)

        self.depthwise.set_training_order(expansion, reset, static)
                

    def bn_statics_tracking(self):
        mean_list = []
        var_list = []
        if self.input_temp is not None:
            print(torch.mean(self.input_temp))
        for bn in self.sbn:
            state_dict = bn.state_dict()
            running_mean = state_dict["running_mean"]
            running_var = state_dict["running_var"]

            mean_list.append(running_mean)
            var_list.append(running_var)

        return mean_list, var_list


SGNAS_A = []
SGNAS_B = []
SGNAS_C = []


@BACKBONES.register_module
class SGNAS(nn.Module):
    def __init__(self, l_cfgs=BASIC_CFGS, dataset="imagenet", classes=1000):
        super(Model, self).__init__()
        if dataset[:5] == "cifar":
            self.first = ConvBNRelu(input_channel=3, output_channel=32, kernel=3, stride=1,
                                    pad=3//2, activation="hswish")
        elif dataset == "imagenet" or dataset == "imagenet_lmdb":
            self.first = ConvBNRelu(input_channel=3, output_channel=32, kernel=3, stride=2,
                                    pad=3//2, activation="hswish")

        input_channel = 32
        output_channel = 16
        self.first_mb = MBConv(input_channel=input_channel,
                               output_channel=output_channel,
                               expansion=1,
                               kernels=[3],
                               stride=1,
                               activation="hswish",
                               split_block=1,
                               se=True)
               
        input_channel = output_channel
        self.stages = nn.ModuleList()
        for l_cfg in l_cfgs:
            expansion, output_channel, kernel, stride, split_block, se = l_cfg
            self.stages.append(MBConv(input_channel=input_channel,
                                      output_channel=output_channel,
                                      expansion=expansion,
                                      kernels=kernel,
                                      stride=stride,
                                      activation="hswish",
                                      split_block=split_block,
                                      se=True))
            input_channel = output_channel

        self.last_stage = conv_1x1_bn(input_channel, 1280)
        self.classifier = nn.Sequential(
                            nn.Dropout(0.2),
                            nn.Linear(1280, classes))

        #self._initialize_weights()

    def forward(self, x):
        x = self.first(x)
        x = self.first_mb(x)
        out = []
        for , l in enumerate(self.stages):
            x = l(x)
            if i in [2, 6, 14, 18]:
                out.append(y)
        x = self.last_stage(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)

        return x

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
                n = m.weight.size(0)  # fan-out
                init_range = 1.0 / math.sqrt(n)
                m.weight.data.uniform_(-init_range, init_range)
                m.bias.data.zero_()

    def init_weights(self, pretrained=False):
        load_checkpoint(self, "~/SGNAS/SGNAS_A_best.pth.tar", False)


