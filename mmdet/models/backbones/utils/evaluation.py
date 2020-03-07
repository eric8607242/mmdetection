import gc
import time
import timeit
import torch

from config import CONFIG

def calculate_latency(model, input_depth, input_size):
    input_sample = torch.randn((1, input_depth, input_size, input_size))
    #globals()["model"], globals()["input_sample"] = model, input_sample
    end = time.time()
    model(input_sample)
    inference_time = time.time() - end
    
    #total_time = timeit.timeit("output=model(input_sample, 0)", setup="gc.enable()", \
    #                           globals=globals(), number=CONFIG["lookup_table"]["number_of_runs"])

    #average_time = total_time / CONFIG["lookup_table"]["number_of_runs"] /1e-6
    #return average_time
    return inference_time

def calculate_param_nums(model):
    #total_params = sum(p.numel() for p in model.parameters())
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params
