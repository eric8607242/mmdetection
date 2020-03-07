import os
import json
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class AverageMeter(object):
    def __init__(self, name=''):
        self._name = name
        self.avg = 0.0
        self.sum = 0.0
        self.cnt = 0.0

    def reset(self):
        self.avg = 0.0
        self.sum = 0.0
        self.cnt = 0.0

    def update(self, val, n=1):
        self.sum += val*n
        self.cnt += n
        self.avg = self.sum/self.cnt
    
    def __str__(self):
        return "%s: %.5f" % (self._name, self.avg)

    def get_avg(self):
        return self.avg

    def __repr__(self):
        return self.__str__()

def get_logger(file_path):
    logger = logging.getLogger("dpnet")
    log_format = "%(asctime)s | %(message)s"
    formatter = logging.Formatter(log_format, datefmt="%m/%d %I:%M:%S %p")

    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger

def create_directories_from_list(list_of_directories):
    for directory in list_of_directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

def accuracy(output, target, topk=(1,)):
    """Compute the precision for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()

    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0/batch_size))

    return res

def check_tensor_in_list(atensor, alist):
    if any([(atensor == t_).all() for t_ in alist if atensor.shape == t_.shape]):
        return True
    return False

def save(model, model_path):
    torch.save(model.state_dict(), model_path)

def save_best_layer(layer_structure, layers_path):
    with open(layers_path, "w") as f:
        json.dump(layer_structure, f, sort_keys=True)

def save_candidate_table(candidate_table, candidate_table_path):
    with open(candidate_table_path, "w") as f:
        json.dump(candidate_table, f)

def load_candidate_table(candidate_table_path):
    candidate_table = None
    with open(candidate_table_path) as f:
        candidate_table = json.load(f)

    return candidate_table

def load_layers_info(path_to_info):
    with open(path_to_info) as f:
        layers_info = json.load(f)

        return layers_info

def load_dataparallel_weight(load_path):
    checkpoint = torch.load(load_path)

    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        name = k[7:]
        new_state_dict[name] = v

    return new_state_dict

def load_search_weight(model, load_path):
    search_dict = torch.load(load_path)

    model_dict = model.state_dict()

    search_dict = {k:v for k, v in search_dict.items() if k in model_dict }
    model_dict.update(search_dict)

    return model_dict

def get_best_structure(model, criterion, loader, device):
    structure_num = model.module.get_structure_nums()
    structure_loss = np.array([0.0 for i in range(structure_num)])
    model.eval()

    structure_accuracy = [0 for i in range(structure_num)]

    for structure_index in range(structure_num):
        top1 = AverageMeter()
        with torch.no_grad():
            for step, (X, y) in enumerate(loader):
                X, y = X.to(device), y.to(device)
                N = X.shape[0]

                outs = model(X, structure_index)
                loss = criterion(outs, y)
                
                prec1, _ = accuracy(outs, y, topk=(1, 5))
                top1.update(prec1.item(), N)

                structure_loss[structure_index] += loss.item()

            if CONFIG["train_settings"]["efficiency"]:
                latency = model.module.get_latency(structure_index)
                structure_accuracy[structure_index] = top1.get_avg() / latency
            else:
                structure_accuracy[structure_index] = top1.get_avg()

            print("Structure {} Prec{:.1%}".format(structure_index, top1.get_avg()))

    return structure_loss.argsort(), structure_accuracy
    
def save_structure_sort(layer_num, structure_list, structure_path):
    structure_sort = {}
    with open(structure_path) as f:
        structure_sort = json.load(f)

    structure_sort[layer_num] = structure_list 

    with open(structure_path, "w") as f:
        json.dump(structure_sort, f)

def normalize(l):
    np_l = np.array(l)
    max_value = np_l.max(0)
    min_value = np_l.min(0)

    normalize = [0 for i in range(len(np_l))]
    for i in range(len(normalize)):
        normalize[i] = (np_l[i] - min_value) / (max_value - min_value)

    return normalize

def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))

def label_smoothing(pred, target, eta=0.1):
    '''
    Refer from https://arxiv.org/pdf/1512.00567.pdf
    :param target: N,
    :param n_classes: int
    :param eta: float
    :return:
        N x C onehot smoothed vector
    '''
    n_classes = pred.size(1)
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros_like(pred)
    onehot_target.scatter_(1, target, 1)
    return onehot_target * (1 - eta) + eta / n_classes * 1


def cross_encropy_with_label_smoothing(pred, target, eta=0.1):
    onehot_target = label_smoothing(pred, target, eta=eta)
    return cross_entropy_for_onehot(pred, onehot_target)
