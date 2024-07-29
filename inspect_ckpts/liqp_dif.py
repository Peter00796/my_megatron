import torch
import numpy as np

ckpt_path1 = "/mnt/public/algm/megrez/jobs/megrez_3b_dist_627B_from_scratch_relu_gqa/checkpoints/iter_0005000/mp_rank_00/distrib_optim.pt"
ckpt1 = torch.load(ckpt_path1, map_location='cpu')

ckpt_path2 = "/mnt/public/algm/megrez/jobs/megrez_3b_dist_627B_from_scratch_relu_gqa/checkpoints/iter_0010000/mp_rank_00/distrib_optim.pt"
ckpt2 = torch.load(ckpt_path2, map_location='cpu')
print(ckpt2.keys())
# print(ckpt2[0].values())
# print(ckpt2[0][1])
state_dict1 = list(ckpt1[0].values())[0]['param']
state_dict2 = list(ckpt2[0].values())[0]['param']


print(state_dict1)
print(state_dict2)

diff_distrib_optim = state_dict2 - state_dict1
print(state_dict1 - state_dict2)

def tensors_to_numpy(data):
    if isinstance(data, dict): 
        return {key: tensors_to_numpy(value) for key, value in data.items()}
    elif isinstance(data, torch.Tensor): 
        return data.cpu().numpy()
    else: 
        return data
    
diff_state_dict_numpy = tensors_to_numpy(diff_distrib_optim)
print(diff_state_dict_numpy)

def calculate_percentage_of_changes(tensor_diff):
    # Flatten the tensor to a 1D array
    
    # Count the number of non-zero elements
    num_non_zero = (tensor_diff != 0).sum()
    
    # Calculate the total number of elements
    total_elements = tensor_diff.size
    
    # Compute the percentage of changed elements
    percentage_changed = (num_non_zero / total_elements) * 100
    
    return percentage_changed

# Example usag
tensor_diff_to_plot = diff_state_dict_numpy
percentage_changed = calculate_percentage_of_changes(tensor_diff_to_plot)
print(f"Percentage of changed :{percentage_changed:.2f}%")