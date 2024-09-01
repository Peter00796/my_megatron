import torch
import numpy as np

def generate_bitmask(delta, bitmasks):
    """
    This function is used for generating the bitmasks for the delta
    """
    for key, value in delta.items():
        if isinstance(value, dict):
            bitmasks[key] = {}
            generate_bitmask(value, bitmasks[key])
        elif isinstance(value, torch.Tensor):
            bitmask = (value != 0).to(torch.uint8)  # Use uint8 instead of bool
            bitmasks[key] = _compress_bitmask(bitmask)
        elif value is None:
            bitmasks[key] = None
        else:
            raise ValueError(f"Unsupported type for key '{key}': {type(value)}")


def _compress_bitmask(bitmask):
    bitmask = bitmask.view(-1).cpu().numpy()
    compressed_bitmask = np.packbits(bitmask)
    return torch.tensor(compressed_bitmask, dtype=torch.uint8)


def _decompress_bitmask(compressed_bitmask, original_shape):
    compressed_bitmask = compressed_bitmask.cpu().numpy()
    decompressed_bitmask = np.unpackbits(compressed_bitmask)
    num_bits = np.prod(original_shape)
    decompressed_bitmask = decompressed_bitmask[:num_bits]
    return torch.tensor(decompressed_bitmask, dtype=torch.uint8).view(original_shape)

def generate_sparse_delta_with_bitmask(delta, bitmask):
    """
    This function is usede for generating the sparse delta with bitmask
    This function will return a dictionary with the same structure as the delta
    """
    
    def process_delta_and_bitmask(delta, bitmask):
        sparse_delta_with_bitmask = {}
        for key in delta.keys():
            if isinstance(delta[key], dict):
                sparse_delta_with_bitmask[key] = process_delta_and_bitmask(delta[key], bitmask[key])
            elif isinstance(delta[key], torch.Tensor):
                flattened_delta = delta[key].view(-1)
                flattened_bitmask = _decompress_bitmask(bitmask[key], flattened_delta.shape)
                sparse_values = flattened_delta[flattened_bitmask == 1]
                sparse_delta_with_bitmask[key] = (sparse_values, bitmask[key])
            elif delta[key] is None:
                sparse_delta_with_bitmask[key] = None
            else:
                raise ValueError(f"Unsupported type for key '{key}': {type(delta[key])}")
        return sparse_delta_with_bitmask

    sparse_delta_with_bitmask = process_delta_and_bitmask(delta, bitmask)
    return sparse_delta_with_bitmask

def reconstruct_checkpoint_with_bitmask(previous_checkpoint, sparse_delta_with_bitmask):
    """
    This function is used for reconstructing the checkpoint with the sparse delta and bitmask
    @Params: 
    previous_checkpoint: the checkpoint before applying the sparse delta
    sparse_delta_with_bitmask: the sparse delta with bitmask
    """
    
    def process_reconstruction(previous_checkpoint, sparse_delta_with_bitmask):
        new_checkpoint = {}
        for key in previous_checkpoint.keys():
            if key in sparse_delta_with_bitmask:
                if isinstance(sparse_delta_with_bitmask[key], tuple):
                    sparse_values, compressed_bitmask = sparse_delta_with_bitmask[key]
                    flattened_bitmask = _decompress_bitmask(compressed_bitmask, previous_checkpoint[key].view(-1).shape)
                    delta = torch.zeros_like(previous_checkpoint[key]).view(-1)
                    delta[flattened_bitmask == 1] = sparse_values
                    delta = delta.view(previous_checkpoint[key].shape)
                    new_checkpoint[key] = previous_checkpoint[key] + delta
                elif isinstance(sparse_delta_with_bitmask[key], dict):
                    new_checkpoint[key] = process_reconstruction(previous_checkpoint[key], sparse_delta_with_bitmask[key])
                elif sparse_delta_with_bitmask[key] is None:
                    new_checkpoint[key] = previous_checkpoint[key]
                else:
                    raise ValueError(f"Unsupported type for key '{key}' in sparse_delta_with_bitmask: {type(sparse_delta_with_bitmask[key])}")
            else:
                new_checkpoint[key] = previous_checkpoint[key]
        return new_checkpoint

    return process_reconstruction(previous_checkpoint, sparse_delta_with_bitmask)

def model_diff(base_model, new_model, diff_state_dict):
    """
    This function is used to compute the difference between the base model and the new model
    """
    for key in base_model.keys():
        if key in new_model.keys():
            if isinstance(base_model[key], dict) and isinstance(new_model[key], dict):
                diff_state_dict[key] = {}
                model_diff(base_model[key], new_model[key], diff_state_dict[key])
            elif isinstance(base_model[key], torch.Tensor) and isinstance(new_model[key], torch.Tensor):
                diff_state_dict[key] = new_model[key] - base_model[key]
            elif base_model[key] is None and new_model[key] is None:
                diff_state_dict[key] = None
            else:
                raise ValueError(f"Mismatched types for key '{key}' in base and new model, base has the type {type(base_model[key])} and new has the type {type(new_model[key])}")
        else:
            raise ValueError(f"Key '{key}' not found in new model")
