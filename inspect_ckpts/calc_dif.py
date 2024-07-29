import torch
import numpy as np
import matplotlib.pyplot as plt 


ckpt_path1 = "/mnt/pengyanxin/Megatron-LLaMA/fine-tuned-ckpts/iter_0000010/mp_rank_00_000/model_optim_rng.pt"
ckpt1 = torch.load(ckpt_path1, map_location='cpu')

ckpt_path2 = "/mnt/pengyanxin/Megatron-LLaMA/fine-tuned-ckpts/iter_0000050/mp_rank_00_000/model_optim_rng.pt"
ckpt2 = torch.load(ckpt_path2, map_location='cpu')

state_dict1 = ckpt1['model']
state_dict2 = ckpt2['model']

diff_state_dict = {}
def iterate_and_subtract(dict1, dict2, diff_dict):
    for key in dict1.keys(): 
        if key in dict2.keys():
            if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                diff_dict[key] = {}
                iterate_and_subtract(dict1[key], dict2[key], diff_dict[key])
            elif isinstance(dict1[key], torch.Tensor) and isinstance(dict2[key], torch.Tensor):
                diff_dict[key] = dict1[key] - dict2[key]
            else:
                raise ValueError(f"Mismatched types for key '{key}': {type(dict1[key])} vs {type(dict2[key])}")
        else:
            raise ValueError(f"Key '{key}' not found in second dict")

iterate_and_subtract(state_dict1, state_dict2, diff_state_dict)


def tensors_to_numpy(data):
    if isinstance(data, dict): 
        return {key: tensors_to_numpy(value) for key, value in data.items()}
    elif isinstance(data, torch.Tensor): 
        return data.cpu().numpy()
    else: 
        return data




print(diff_state_dict)
