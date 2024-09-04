
import torch 
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# checkpoint related
def get_adam1_from_storage(path):
    optimizer = torch.load(path)
    return optimizer[0][0][0]['exp_avg']

def get_adam2_from_storage(path):
    optimizer = torch.load(path)
    return optimizer[0][0][0]['exp_avg_sq']

def create_dynamic_map(signed=True):
    if signed:
        # For signed integers (range -128 to 127)
        base = torch.linspace(-1, 1, 256)
        return torch.sign(base) * torch.abs(base) ** 2
    else:
        # For unsigned integers (range 0 to 255)
        base = torch.linspace(0, 1, 256)
        return base ** 2