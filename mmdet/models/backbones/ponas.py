import math

import torch
import torch.nn as nn

from .utils.util import get_logger, create_directories_from_list, check_tensor_in_list, load_layers_info, load_dataparallel_weight, load_search_weight 
from .utils.net_builder import NORMAL_PRIMITIVES, NORMAL_CS_PRIMITIVES, STRIDE_PRIMITIVES, ApproximateBlock, PDBlock, ConvBNRelu
from .utils.lookup_table_builder import get_structure_list

from ..registry import BACKBONES

@BACKBONES.register_module
class Network(nn.Module):
    def __init__(self, classes=1000):
        super(Network, self).__init__()
        self.dataset = "imagenet"
        layers_info = load_layers_info("/home/jovyan/mmdetection/mmdet/models/backbones/330.json")

        if self.dataset == "cifar10":
            self.first = ConvBNRelu(input_depth=3, output_depth=32, kernel=3, stride=1, pad=3//2, activation="relu")
        elif self.dataset == "imagenet":
            self.first = ConvBNRelu(input_depth=3, output_depth=32, kernel=3, stride=2, pad=3//2, activation="relu")

        self.stages = nn.ModuleList()
        self.stages.append(PDBlock(32, 0, 3, 1))
        output_depth = 16
        for i, layer in enumerate(layers_info):
            input_depth = output_depth
            layer_name = layer["name"]
            cs = int(layer_name.split("_")[3][2:])
            if cs != -1:
                output_depth = 16 + 8*cs

            layer_num = i+1
            if layer_num in [1, 3, 7, 15]:
                self.stride_layer_structure_list = get_structure_list(STRIDE_PRIMITIVES, i)
                self.stages.append(self.stride_layer_structure_list[layer_name](input_depth))
            elif layer_num in [1, 11, 19]:
                self.normal_cs_layer_structure_list = get_structure_list(NORMAL_CS_PRIMITIVES, i)
                self.stages.append(self.normal_cs_layer_structure_list[layer_name](input_depth))
            else:
                self.normal_layer_structure_list = get_structure_list(NORMAL_PRIMITIVES, i)
                self.stages.append(self.normal_layer_structure_list[layer_name](input_depth))


        self.appro_block = ApproximateBlock(n_class=classes, input_channel=output_depth, interverted_residual_setting = [])
        
    def forward(self, x):
        y = self.first(x)
        out = []
        for i, l in enumerate(self.stages):
            y = l(y)
#             print("---------------------------------")
#             print(i)
#             print(y.shape)
            if i in [18, 14, 10]:
                out.append(y)
        y = self.appro_block(y)
        out.append(y)
        return tuple(out)

    def init_weights(self, pretrained=False):
        new_state_dict = load_dataparallel_weight("/home/jovyan/mmdetection/mmdet/models/backbones/330.pth")
        self.load_state_dict(new_state_dict)
