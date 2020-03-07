import math

import torch
import torch.nn as nn

from utils.net_builder import NORMAL_PRIMITIVES, NORMAL_CS_PRIMITIVES, STRIDE_PRIMITIVES, ApproximateBlock, PDBlock, ConvBNRelu
from utils.lookup_table_builder import get_structure_list
from config import CONFIG

class Network(nn.Module):
    def __init__(self, classes, layers_info):
        super(Network, self).__init__()
        self.dataset = CONFIG["dataloading"]["dataset"]

        if self.dataset == "cifar10":
            self.first = ConvBNRelu(input_depth=3, output_depth=32, kernel=3, stride=1, pad=3//2, activation="relu")
        elif self.dataset == "imagenet":
            self.first = ConvBNRelu(input_depth=3, output_depth=32, kernel=3, stride=2, pad=3//2, activation="relu")

        self.stages = nn.Sequential()
        self.stages.add_module("0", PDBlock(32, 0, 3, 1))
        output_depth = 16
        for i, layer in enumerate(layers_info):
            input_depth = output_depth
            layer_name = layer["name"]
            cs = int(layer_name.split("_")[3][2:])
            if cs != -1:
                output_depth = 16 + 8*cs

            layer_num = i+1
            if layer_num in CONFIG["train_settings"][self.dataset]["stride_layer"]:
                self.stride_layer_structure_list = get_structure_list(STRIDE_PRIMITIVES, i)
                self.stages.add_module(str(layer_num), self.stride_layer_structure_list[layer_name](input_depth))
            elif layer_num in CONFIG["train_settings"][self.dataset]["cs_layer"]:
                self.normal_cs_layer_structure_list = get_structure_list(NORMAL_CS_PRIMITIVES, i)
                self.stages.add_module(str(layer_num), self.normal_cs_layer_structure_list[layer_name](input_depth))
            else:
                self.normal_layer_structure_list = get_structure_list(NORMAL_PRIMITIVES, i)
                self.stages.add_module(str(layer_num), self.normal_layer_structure_list[layer_name](input_depth))


        self.appro_block = ApproximateBlock(n_class=classes, input_channel=output_depth, interverted_residual_setting = [])
        
        self._initialize_weights()

    def forward(self, x):
        y = self.first(x)
        y = self.stages(y)
        y = self.appro_block(y)
        return y

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
