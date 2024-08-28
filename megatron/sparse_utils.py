import torch
import numpy as np

def generate_bitmask(delta):
    """
    This function is used for generating the bitmasks for the delta
    """
    bitmasks = {} 
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
    return bitmasks


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

def model_diff(base_model, new_model):
    """
    This function is used to compute the difference between the base model and the new model
    """
    diff_state_dict = {}
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
    return diff_state_dict












def compress_indices(indices, dtype=torch.int32):
    # Compress indices to a specified lower-bit integer type
    return indices.to(dtype)

def decompress_indices(indices, dtype=torch.int64):
    # Decompress indices back to int64
    return indices.to(dtype)


# sparsification using traditional COO sparse matrix
def sparsification(diff_model, sparse_model, device='cuda'):
    """
    This function is used to perform sparsification on the diff_model
    We need to iterate through the diff_model and perform sparsification on the tensors
    """
    for key in diff_model:
        if isinstance(diff_model[key], dict):
            sparse_model[key] = {}
            sparsification(diff_model[key], sparse_model[key], device)
        elif isinstance(diff_model[key], torch.Tensor):
            # Perform sparsification using COO format
            tensor = diff_model[key].to(device)
            tensor_sparse = tensor.to_sparse()
            indices = tensor_sparse.indices()
            values = tensor_sparse.values()
            # Compress indices for storage or transmission

            # Here note that the latest pyTorch(torch 2.3.0) could apply torch.uint16 to directly convert the datatype of the indices to uint16
            # This is ideal since the indices tensor does not have to go through CPU to be converted to nupmy array
            # However now with torch 2.2.0 we have no choice but to convert to numpy array and then use uint 16
            
            
            # compressed_indices = indices.to(torch.uint16)
            # compressed_indices = compress_indices(indices, dtype=torch.uint16)
            
            indices_numpy = indices.cpu().numpy()
            compressed_indices = np.array(indices_numpy, dtype=np.uint16)

            # Store the compressed indices and values
            sparse_model[key] = {
                "size": tensor_sparse.size(),
                "compressed_indices": compressed_indices,
                "values": values
            }

        elif diff_model[key] is None:
            sparse_model[key] = None

def de_sparsification(sparse_model, diff_model):
    for key in sparse_model:
        if isinstance(sparse_model[key], dict) and "compressed_indices" in sparse_model[key]:
            compressed_indices_uint16 = sparse_model[key]["compressed_indices"]
            values = sparse_model[key]["values"]
            size = sparse_model[key]["size"]
            compress_indices_int64 = np.array(compressed_indices_uint16, dtype=np.int64)
            indices = torch.tensor(compress_indices_int64, dtype=torch.int64)
            indices = indices.to(values.device)

            tensor = torch.sparse_coo_tensor(indices, values, size, device=values.device)
            diff_model[key] = tensor.to_dense()
        elif isinstance(sparse_model[key], dict):
            diff_model[key] = {}
            de_sparsification(sparse_model[key], diff_model[key])
        elif sparse_model[key] is None:
            diff_model[key] = None



def incrementation(base_model, diff_model):
    recovered_ckpt = {}
    for key in base_model:
        if key in diff_model:
            if isinstance(base_model[key], dict) and isinstance(diff_model[key], dict):
                recovered_ckpt[key] = incrementation(base_model[key], diff_model[key])
            elif isinstance(base_model[key], torch.Tensor) and isinstance(diff_model[key], torch.Tensor):
                base_tensor = base_model[key].to(diff_model[key].device)
                recovered_ckpt[key] = base_tensor + diff_model[key]
            elif base_model[key] is None and diff_model[key] is None:
                recovered_ckpt[key] = None
            else:
                raise ValueError(f"Mismatched types for key '{key}' in base and diff model, base has the type {type(base_model[key])} and diff has the type {type(diff_model[key])}")
        else:
            raise ValueError(f"Key '{key}' not found in diff model")
    return recovered_ckpt


    