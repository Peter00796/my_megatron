from .util import *
from abc import ABC, abstractmethod
import torch
from abc import ABC, abstractmethod


class DynamicBlockwiseQuantizer:
    def __init__(self, block_size=4096, device=None):
        self.block_size = block_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.qmap_signed = self.create_dynamic_map(signed=True).to(self.device)
        self.qmap_unsigned = self.create_dynamic_map(signed=False).to(self.device)

    @staticmethod
    def create_dynamic_map(signed=True):
        if signed:
            base = torch.linspace(-1, 1, 256)
            return torch.sign(base) * torch.abs(base) ** 2
        else:
            base = torch.linspace(0, 1, 256)
            return base ** 2

    def quantize_tensor(self, data, signed=True):
        num_blocks = (data.numel() + self.block_size - 1) // self.block_size
        padded_data = torch.nn.functional.pad(data.flatten(), (0, num_blocks * self.block_size - data.numel()))
        blocks = padded_data.view(num_blocks, self.block_size)
        
        quantized_blocks = []
        block_info = []
        for block in blocks:
            block_min, block_max = block.min(), block.max()
            scale_factor = block_max - block_min
            scaled_data = 2 * (block - block_min) / scale_factor - 1 if signed else (block - block_min) / scale_factor
            qmap = self.qmap_signed if signed else self.qmap_unsigned
            quantized_indices = torch.argmin(torch.abs(scaled_data.unsqueeze(-1) - qmap), dim=-1)
            quantized_data = quantized_indices.to(torch.int8) - 128 if signed else quantized_indices.to(torch.int8)
            quantized_blocks.append(quantized_data)
            block_info.append({'min_val': block_min.item(), 'scale_factor': scale_factor.item()})
        
        quantized_data = torch.cat(quantized_blocks)[:data.numel()].reshape(data.shape)
        return quantized_data, block_info

    def dequantize_tensor(self, quantized_data, block_info, original_shape):
        num_blocks = len(block_info)
        flat_quantized = quantized_data.flatten()
        padded_length = num_blocks * self.block_size
        
        if flat_quantized.numel() < padded_length:
            flat_quantized = torch.nn.functional.pad(flat_quantized, (0, padded_length - flat_quantized.numel()))
        
        blocks = flat_quantized.view(num_blocks, self.block_size)
        
        dequantized_blocks = []
        for block, info in zip(blocks, block_info):
            qmap = self.qmap_signed if info['scale_factor'] != 0 else self.qmap_unsigned
            dequantized_block = qmap[block + 128] if info['scale_factor'] != 0 else qmap[block]
            dequantized_block = 0.5 * (dequantized_block + 1) * info['scale_factor'] + info['min_val'] if info['scale_factor'] != 0 else dequantized_block * info['scale_factor'] + info['min_val']
            dequantized_blocks.append(dequantized_block)
        
        dequantized_data = torch.cat(dequantized_blocks)[:np.prod(original_shape)].reshape(original_shape)
        return dequantized_data

    def quantize_optimizer_state(self, optimizer_state):
        quantized_state = {}
        block_info = {}
        for param_id, param_state in optimizer_state.items():
            quantized_param_state = {}
            block_info_param_state = {}
            for state_name, state_value in param_state.items():
                if torch.is_tensor(state_value):
                    quantized_data, info = self.quantize_tensor(state_value, signed=(state_name == 'exp_avg'))
                    quantized_param_state[state_name] = quantized_data
                    block_info_param_state[state_name] = info
                else:
                    quantized_param_state[state_name] = state_value
            quantized_state[param_id] = quantized_param_state
            block_info[param_id] = block_info_param_state
        return quantized_state, block_info

    def dequantize_optimizer_state(self, quantized_state, block_info):
        dequantized_state = {}
        for param_id, quantized_param_state in quantized_state.items():
            dequantized_param_state = {}
            for state_name, quantized_value in quantized_param_state.items():
                if torch.is_tensor(quantized_value):
                    dequantized_value = self.dequantize_tensor(
                        quantized_value,
                        block_info[param_id][state_name],
                        quantized_value.shape
                    )
                    dequantized_param_state[state_name] = dequantized_value
                else:
                    dequantized_param_state[state_name] = quantized_value
            dequantized_state[param_id] = dequantized_param_state
        return dequantized_state

def quantize_optimizer_state(optimizer_state, block_size=4096):
    quantizer = DynamicBlockwiseQuantizer(block_size=block_size)
    return quantizer.quantize_optimizer_state(optimizer_state)

def dequantize_optimizer_state(quantized_state, block_info):
    quantizer = DynamicBlockwiseQuantizer()
    return quantizer.dequantize_optimizer_state(quantized_state, block_info)