import torch

def print_info(input):
    """
    This function takes in checkpoint parameters such as model, optimizer, and scheduler and prints out the information of each.
    The information contains the datatype, the shape, the datastructure
    """
    if isinstance(input, dict):
        for key, value in input.items():
            print(f'Key: {key}')
            print_info(value)
    elif isinstance(input, list):
        for i, value in enumerate(input):
            print(f'Index: {i}')
            print_info(value)
    elif isinstance(input, torch.Tensor):
        print(f'data type: {input.dtype}')
        print(f'Shape: {input.shape}')