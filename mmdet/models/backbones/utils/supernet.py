import json
import math
import numpy as np
from collections import OrderedDict 

import torch
import torch.nn as nn

from utils.countmacs import MAC_Counter
from utils.net_builder import conv_1x1_bn, ConvBNRelu, Flatten, NORMAL_PRIMITIVES, NORMAL_CS_PRIMITIVES, STRIDE_PRIMITIVES, PDBlock, ApproximateBlock
from utils.lookup_table_builder import get_structure_list

from config import CONFIG, IMAGENET_APPROXIMATE, CIFAR_APPROXIMATE, CIFAR_ARCHITECTURE, IMAGE_ARCHITECTURE

class MixedOperation(nn.Module):
    def __init__(self, layer_structure_list, input_depth, layer_info, output_table):
        super(MixedOperation, self).__init__()

        self.ops_names = [op_name for op_name in layer_structure_list]
        print(self.ops_names)
        self.ops = nn.ModuleList([layer_structure_list[op_name](input_depth)
                                  for op_name in self.ops_names])

        self.weight_p = [1.0 / len(self.ops_names) for i in range(len(self.ops_names))]

        self.output_table = [output_table[op_name] for op_name in self.ops_names]

        self.latency = layer_info["latency"]
        self.macs = layer_info["macs"]
        self.macs_list = np.array([self.macs[name] for name in self.ops_names])

        self.input_shape = None
        self.structure_num = None

        self.meta_learning = True
        self.best_index = None

        self.macs_budget = CONFIG["budget"]["macs"]
        self.path_to_candidate_table = CONFIG["train_settings"]["path_to_candidate_table"]

    def load_weight(self, appro_state_dict=None):
        state_dict = self.ops[-1].state_dict() if appro_state_dict is None else appro_state_dict
        for op in self.ops:
            op_param = op.state_dict()
            for k, v in state_dict.items():
                block_name = "block." + k

                if block_name in op_param:
                    op_param_shape = op_param[block_name].shape

                    if len(v.shape) == 1:
                        op_param[block_name] = v[:op_param_shape[0]]
                    elif len(v.shape) == 4:
                            t_size = op_param_shape[2]
                            a_size = v.size(2)
                            initial = a_size - t_size - 1 if a_size > t_size else 0

                            op_param[block_name] = v[:op_param_shape[0], :op_param_shape[1], initial:initial+op_param_shape[2], initial:initial+op_param_shape[3]]
                    else:
                        op_param[block_name] = v

            op.load_state_dict(op_param)

        self.meta_learning = False
        
    def forward(self, x, structure_index):
        self.input_shape = x.shape
        soft_mask_variables = torch.Tensor([0.0 for i in range(len(self.weight_p))])

        if self.meta_learning:
            c = len(self.weight_p)-1
        else:
            c = structure_index

        soft_mask_variables[int(c)] += 1

        output = [m * op(x) if m != 0.0 else torch.zeros([self.input_shape[0], *opshape[1:]]).cuda() for m, op, opshape in zip(soft_mask_variables, self.ops, self.output_table)] 
        output = self._merge_output(output)
        return output

    def get_block_num(self):
        return len(self.ops_names)

    def _merge_output(self, output):
        """Merge the output of candidate, sum the same depth output, and cat the
           different depth output
        """
        output_size = output[0].size(2)
        output_dict = {}
        output_merge = []

        for layer in output:
            layer_depth = layer.shape[1]
            layer_size = layer.shape[2]

            key = str(layer_depth) + str(layer_size)

            if key in output_dict:
                output_dict[key].append(layer)
            else:
                output_dict[key] = [layer]

        for key, value in output_dict.items():
            list_sum = sum(value)
            output_merge.append(list_sum)

        max_shape = 0
        batch_shape = 0
        for i in output_merge:
            batch_shape = i.shape[0]
            shape = i.shape[1]
            if shape > max_shape:
                max_shape = shape

        output = torch.zeros(batch_shape, max_shape, output_size, output_size).cuda()
        for i in output_merge:
            output[:, :i.shape[1]] += i

        return output


    def get_structure_nums(self):
        return len(self.ops_names)

    def get_latency(self, index):
        return self.latency[self.ops_names[index]]

    def get_block_ratio(self, accuracy_list):
        min_accuracy = accuracy_list.min().item()
        min_macs = self.macs_list.min()

        ratio_list = np.array([0 for i in range(len(self.macs_list))])
        for i in range(len(ratio_list)):
            accuracy = accuracy_list[i]
            macs = self.macs_list[i]

            ratio_list = (accuracy / min_accuracy) / (macs / min_macs)
        return ratio_list
        

    def get_candidate_info(self):
        return self.ops_names, self.macs_list

    def get_best_structure(self, index, accuracy_list=None, model_macs=None):
        self.best_index = accuracy_list.argmax()

        output_shape = self._get_output_shape(self.best_index)
        output_depth = output_shape[1]
        output_size = output_shape[2]

        return self.ops_names[self.best_index], output_depth, output_size
    
    def get_best_weight(self):
        return self.ops[self.best_index].state_dict()

    def _get_output_shape(self, index):
        return self.ops[index](torch.randn(self.input_shape).cuda()).shape


class DPNet_SuperNet(nn.Module):
    def __init__(self, lookup_table, device, cnt_classes=1000):
        super(DPNet_SuperNet, self).__init__()
        self.layer_num = 1
        self.input_depth = 16
        self.classes = cnt_classes
        self.device = device

        self.normal_layer_structure_list = get_structure_list(NORMAL_PRIMITIVES, self.layer_num-1)
        self.normal_cs_layer_structure_list = get_structure_list(NORMAL_CS_PRIMITIVES, self.layer_num-1)
        self.stride_layer_structure_list = get_structure_list(STRIDE_PRIMITIVES, self.layer_num-1)

        self.lookup_table = lookup_table
        self.stages = nn.Sequential()

        self.dataset = CONFIG["dataloading"]["dataset"]

        if self.dataset == "cifar10":
            self.interverted_residual_setting = CIFAR_APPROXIMATE
            self.first = ConvBNRelu(input_depth=3, output_depth=32, kernel=3, stride=1,
                                    pad=3//2, activation="relu")
            self.feature_size = 32
        elif self.dataset == "imagenet":
            self.interverted_residual_setting = IMAGENET_APPROXIMATE
            self.first = ConvBNRelu(input_depth=3, output_depth=32, kernel=3, stride=2,
                                    pad=3//2, activation="relu")
            self.feature_size = 112

        self.stages.add_module(str(self.layer_num-1), PDBlock(32, 0, 3, 1))
        self.stages_to_search = None

        self.appro_block = ApproximateBlock(n_class=self.classes)
        self._initialize_weights()

    def stage_init(self):
        self.appro_block.set_index(1)

        if self.dataset == "cifar10":
            layer_info, output_table = self.lookup_table.calculate_latency(self.normal_cs_layer_structure_list, 16, self.feature_size)
            self.stages_to_search = MixedOperation(self.normal_cs_layer_structure_list, 16, layer_info, output_table)
        elif self.dataset == "imagenet":
            layer_info, output_table = self.lookup_table.calculate_latency(self.stride_layer_structure_list, 16, self.feature_size)
            self.stages_to_search = MixedOperation(self.stride_layer_structure_list, 16, layer_info, output_table)

    def transfer_weight(self):
        appro_state_dict = self.appro_block.features[self.layer_num-1].state_dict()
        self.stages_to_search.load_weight(appro_state_dict)

    def get_model_info(self, device):
        if self.dataset == "cifar10":
            input_sample = torch.randn((1, 3, 32, 32))
            input_size = 32
        elif self.dataset == "imagenet":
            input_sample = torch.randn((1, 3, 224, 224))
            input_size = 224

        input_sample = input_sample.to(device)
        first_macs = self._count_layer_macs(self.first, 3, input_size)
        first_output = self.first(input_sample)

        input_depth, input_size = first_output.size(1), first_output.size(2)
        stages_macs = self._count_layer_macs(self.stages, input_depth, input_size)
        stages_output = self.stages(first_output)

        search_output = self.stages_to_search(stages_output, 0)

        input_depth, input_size = search_output.size(1), search_output.size(2)
        appro_macs = self._count_layer_macs(self.appro_block, input_depth, input_size)

        total_macs = first_macs + stages_macs + appro_macs

        return total_macs


    def _count_layer_macs(self, layer, input_depth, input_size):
        counter = MAC_Counter(layer, [1, input_depth, input_size, input_size])

        macs = counter.print_summary(False)
        macs = macs["total_gmacs"] * 1000

        return macs
    
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


    def forward(self, x, structure_index=None):
        latency = None
        y = self.first(x)
        y = self.stages(y)
        
        if self.stages_to_search is not None:
            y = self.stages_to_search(y, structure_index)

        y = self.appro_block(y)

        return y

    def get_structure_name(self, layer):
        if layer in CONFIG["train_settings"][self.dataset]["stride_layer"]:
            return [op_name for op_name in self.stride_layer_structure_list]
        elif layer in CONFIG["train_settings"][self.dataset]["cs_layer"]:
            return [op_name for op_name in self.normal_cs_layer_structure_list]
       	return [op_name for op_name in self.normal_layer_structure_list]


    def add_stage_module(self, index, accuracy_list=None):
        input_depth = self.input_depth
        model_macs = self.get_model_info(self.device)

        layer_name, output_depth, output_size = self.stages_to_search.get_best_structure(index, accuracy_list, model_macs)
        best_weight = self.stages_to_search.get_best_weight()

        if self.layer_num in CONFIG["train_settings"][self.dataset]["stride_layer"]:
            operation = self.stride_layer_structure_list[layer_name](self.input_depth)
            operation.load_state_dict(best_weight)
            self.stages.add_module(str(self.layer_num), operation)

        elif self.layer_num in CONFIG["train_settings"][self.dataset]["cs_layer"]:
            operation = self.normal_cs_layer_structure_list[layer_name](self.input_depth)
            operation.load_state_dict(best_weight)
            self.stages.add_module(str(self.layer_num), operation)
        else:
            operation = self.normal_layer_structure_list[layer_name](self.input_depth)
            operation.load_state_dict(best_weight)
            self.stages.add_module(str(self.layer_num), operation)

        self.input_depth = output_depth
        self.layer_num += 1

        self.change_channel_size()

        del self.stages_to_search

        return (layer_name, input_depth, output_depth, output_size)

    def change_channel_size(self):
        origin_depth = self.interverted_residual_setting[self.layer_num-2][1]
        depth_diff = origin_depth - self.input_depth
        print(origin_depth)

        if self.input_depth != origin_depth:
            for i in range(len(self.interverted_residual_setting)):
                if self.interverted_residual_setting[i][1] == origin_depth:
                    self.interverted_residual_setting[i][1] = self.input_depth

        print(self.interverted_residual_setting)

        appro_state_dict = self.appro_block.state_dict()
        start_index = self.appro_block.start_index

        # from self.layer_num-1 because need to calculate search stages
        self.appro_block = ApproximateBlock(n_class=self.classes, interverted_residual_setting=self.interverted_residual_setting)
        self.appro_block.set_index(start_index)

        self.load_appro_block(appro_state_dict)

    def load_appro_block(self, appro_state_dict):
        new_appro_state_dict = self.appro_block.state_dict()
        
        for k, v in appro_state_dict.items():
            if k in new_appro_state_dict:
                new_appro_shape = new_appro_state_dict[k].shape

                if len(v.shape) == 4:
                    new_appro_state_dict[k] = v[:new_appro_shape[0], :new_appro_shape[1]] 
                elif len(v.shape) == 1:
                    new_appro_state_dict[k] = v[:new_appro_shape[0]]
                else:
                    new_appro_state_dict[k] = v

        self.appro_block.load_state_dict(new_appro_state_dict)

    def add_search_module(self, layers_info):
        _, _, output_depth, output_size = layers_info
        output_table = {}

        if self.layer_num in CONFIG["train_settings"][self.dataset]["stride_layer"]:
            self.stride_layer_structure_list = get_structure_list(STRIDE_PRIMITIVES, self.layer_num-1)
            layer_info, output_table = self.lookup_table.calculate_latency(self.stride_layer_structure_list, output_depth, output_size)
            self.stages_to_search = MixedOperation(self.stride_layer_structure_list, output_depth, layer_info, output_table)
        elif self.layer_num in CONFIG["train_settings"][self.dataset]["cs_layer"]:
            self.normal_cs_layer_structure_list = get_structure_list(NORMAL_CS_PRIMITIVES, self.layer_num-1)
            layer_info, output_table = self.lookup_table.calculate_latency(self.normal_cs_layer_structure_list, output_depth, output_size)
            self.stages_to_search = MixedOperation(self.normal_cs_layer_structure_list, output_depth, layer_info, output_table)
        else:
            self.normal_layer_structure_list = get_structure_list(NORMAL_PRIMITIVES, self.layer_num-1)
            layer_info, output_table = self.lookup_table.calculate_latency(self.normal_layer_structure_list, output_depth, output_size)
            self.stages_to_search = MixedOperation(self.normal_layer_structure_list, output_depth, layer_info, output_table)

        self.appro_block.add_index()

    def get_structure_nums(self):
        return self.stages_to_search.get_structure_nums()

    def set_mode(self, mode):
        self.stages_to_search.set_mode(mode)

    def get_block_num(self):
        return self.stages_to_search.get_block_num()

    def get_latency(self, index):
        return self.stages_to_search.get_latency(index)

    def get_candidate_info(self):
        ops_names, macs_list = self.stages_to_search.get_candidate_info()
        return ops_names, macs_list

