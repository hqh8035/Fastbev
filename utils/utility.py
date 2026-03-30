import torch
import numpy as np

def sigmoid(x):
    x = np.array(x)
    x = np.clip(x, -1e1, 1e1)
    y = 1 / (1 + np.exp(-x))
    return y


def compute_rms_norm(model_params):
    '''Compute root-mean-square norm.'''
    sum, num = 0, 0
    for param in model_params:
        if param.grad is not None:
            sum += torch.sum(torch.pow(param.grad.detach(), 2))
            num += 1
    norm = torch.sqrt(sum) if num != 0 else 0
    return norm
