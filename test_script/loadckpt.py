import torch 

checkpoint_file = '/mnt/pengyanxin/mdl/my_megatron/checkpoints/iter_0000020/mp_rank_00/model_optim_rng.pt'

checkpoint = torch.load(checkpoint_file)

key_to_check = 'model'  # Change this to the key you want to check
# print(checkpoint[key_to_check].keys())
# Determine the maximum length of the tensor keys
max_key_length = max(len(tensor_key) for tensor_key in checkpoint[key_to_check].keys())

for tensor_key, tensor in checkpoint[key_to_check].items():
    print("|", end="")
    print(tensor_key.ljust(max_key_length), end="|")
    if tensor is not None:
        try:
            print(tensor.shape, end="")
        except:
            print("Not a tensor", end="")
    print("")  # To add a newline after each tensor key and shape


