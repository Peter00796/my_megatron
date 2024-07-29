import torch
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import os 

def load_checkpoint(checkpoint_path):
    """Load a checkpoint and return the model state dict."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    return checkpoint['model']

def extract_parameters(state_dict):
    """Extract parameters from the state dict as numpy arrays"""
    return {name: param.cpu().numpy() for name, param in state_dict.items()}

def compare_parameters(param_sets):
    """Compare parameters across multiple checkpoints"""
    diff_dict = {}
    for param_name in param_sets[0].keys():
        diffs = []
        for i in range(1, len(param_sets)):
            diff = np.abs(param_sets[i][param_name] - param_sets[[i-1][param_name]])
            diffs.append(diff)
        diff_dict[param_name] = np.mean(diffs)









checkpoint_path = "/mnt/pengyanxin/Megatron-LLaMA/fine-tuned-ckpts/iter_0000050/mp_rank_00_000/model_optim_rng.pt"

state_dict = load_checkpoint(checkpoint_path)


print(state_dict)
