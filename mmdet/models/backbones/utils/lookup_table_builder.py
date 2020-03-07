import gc
import time
import torch
import torch.nn as nn
import itertools
from collections import OrderedDict

from .net_builder import get_layer_structure, NORMAL_PRIMITIVES, STRIDE_PRIMITIVES
from .countmacs import MAC_Counter

IMAGE_ARCHITECTURE = [
        # [4, 7, 14]
        # [2, 11, 17]
        # expansion, cs, min_cs
        #(1, -2, -2),
        (6, 2, 2),(6, -1, -1), #24
        (6, 3, 3),(6, -1, -1),(6, -1, -1),(6, -1, -1), #40
        (6, 8, 8),(6, -1, -1),(6, -1, -1),(6, -1, -1), #80
        (6, 10, 10),(6, -1, -1),(6, -1, -1),(6, -1, -1), #96
        (6, 22, 22),(6, -1, -1),(6, -1, -1),(6, -1, -1), #192
        (6, 38, 38) #320
]

SEARCH_SPACE = OrderedDict([
    
])
def get_structure_list(PRIMITIVES, layer_num):
    """Get the all structure in layer search space
    """
    dataset = "imagenet"
    max_structure = IMAGE_ARCHITECTURE[layer_num] if dataset == "imagenet" else CIFAR_ARCHITECTURE[layer_num]
    max_expansion, max_cs, min_cs = max_structure

    layer_structure_list = {}
    if PRIMITIVES["stride"][0] == 1 and max_cs == 0:
        layer_structure_list = {"identity" : lambda input_depth, **kwargs: nn.Conv2d(input_depth, input_depth, 1)}
        #layer_structure_list = {"identity" : lambda input_depth, **kwargs: nn.Sequential()}
    for combination in itertools.product(PRIMITIVES["kernel_size"],
                                         PRIMITIVES["stride"],
                                         PRIMITIVES["channel_size"],
                                         PRIMITIVES["expansion"],
                                         PRIMITIVES["group"],
                                         PRIMITIVES["se"],
                                         PRIMITIVES["activation"]):
        cs = combination[2]
        e = combination[3]
        se = combination[5]
        activation = combination[6]

        if cs > max_cs or cs < min_cs or e > max_expansion:
            continue

        #if max_expansion == 6 and e <= 3:
        #    continue

        layer_structure = get_layer_structure(*combination)
        layer_structure_list = {**layer_structure_list, **layer_structure}

    if "BC" in PRIMITIVES["block_type"]:
        for combination in itertools.product(PRIMITIVES["kernel_size"],
                                             PRIMITIVES["stride"],
                                             PRIMITIVES["channel_size"],
                                             [1],
                                             [1],
                                             [False],
                                             PRIMITIVES["activation"]):
            cs = combination[2]
            activation = combination[6]

            if cs > max_cs or cs < min_cs:
                continue

            layer_structure = get_layer_structure(*combination, block_type="BC")
            layer_structure_list = {**layer_structure_list, **layer_structure}

        
    return layer_structure_list


class LookUpTable:
    def __init__(self, search_space=SEARCH_SPACE):
        pass

    def calculate_latency(self, layer_structure_list, input_depth, input_size):
        layer_info_table = {"latency" : {}, "macs" : {}}
        output_table = {}

        ops_names = [op_name for op_name in layer_structure_list]
        for op_name in ops_names:
            op = layer_structure_list[op_name](input_depth)
            input_sample = torch.randn((1, input_depth, input_size, input_size))
            output_sample = op(input_sample)

            
            counter = MAC_Counter(op, [1, input_depth, input_size, input_size])
            macs = counter.print_summary(False)
            macs = macs["total_gmacs"]
            
            start = time.time()
            op(input_sample)
            total_time = time.time() - start
            layer_info_table["latency"][op_name] = total_time
            layer_info_table["macs"][op_name] = macs*1000
            output_table[op_name] = output_sample.shape
            del op

 
        return layer_info_table, output_table

