from .util import *
from abc import ABC, abstractmethod
import torch
from abc import ABC, abstractmethod

class Quantizer(ABC):
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def quantize_cluster(self, data, min_val, max_val, signed=True):
        pass

    @abstractmethod
    def dequantize_cluster(self, quantized_data, min_val, scale_factor, signed=True):
        pass

    def quantize(self, clusters, signed=True):
        quantized_clusters = []
        for i, cluster in enumerate(clusters):
            if cluster.numel() == 0:
                continue
            cluster = cluster.to(self.device)
            min_val, max_val = torch.min(cluster), torch.max(cluster)
            quantized_data, scale_factor = self.quantize_cluster(cluster, min_val, max_val, signed)
            quantized_clusters.append({
                'min_val': min_val.cpu(),
                'scale_factor': scale_factor.cpu(),
                'quantized_data': quantized_data.cpu()
            })
            torch.cuda.empty_cache()
        return quantized_clusters

    def quantize_all(self, clustered_optimizer, batch_size=10):
        quantized_optimizer = {}
        for key, state_dict in clustered_optimizer.items():
            quantized_state = {}
            for state_key, clusters in state_dict.items():
                signed = state_key == 'exp_avg'
                quantized_clusters = []
                for i in range(0, len(clusters), batch_size):
                    batch_clusters = clusters[i:i + batch_size]
                    quantized_batch = self.quantize(batch_clusters, signed)
                    quantized_clusters.extend(quantized_batch)
                quantized_state[state_key] = quantized_clusters
            quantized_optimizer[key] = quantized_state
        return quantized_optimizer

    def dequantize(self, quantized_clusters):
        dequantized_clusters = []
        for cluster_info in quantized_clusters:
            quantized_data = cluster_info['quantized_data'].to(self.device)
            min_val = cluster_info['min_val'].to(self.device)
            scale_factor = cluster_info['scale_factor'].to(self.device)
            signed = cluster_info.get('signed', True)
            dequantized_data = self.dequantize_cluster(quantized_data, min_val, scale_factor, signed)
            dequantized_clusters.append(dequantized_data.cpu())
            torch.cuda.empty_cache()
        return dequantized_clusters

    def dequantize_all(self, quantized_optimizer):
        dequantized_optimizer = {}
        for key, state_dict in quantized_optimizer.items():
            dequantized_state = {}
            for state_key, quantized_clusters in state_dict.items():
                dequantized_data = []
                for cluster_info in quantized_clusters:
                    dequantized_cluster = self.dequantize_cluster(
                        cluster_info['quantized_data'],
                        cluster_info['min_val'],
                        cluster_info['scale_factor'],
                        cluster_info.get('signed', True)
                    )
                    dequantized_data.append(dequantized_cluster)
                # Reconstruct full tensor, skip if there are no dequantized clusters
                if dequantized_data:
                    dequantized_data = [d.unsqueeze(0) if d.dim() == 0 else d for d in dequantized_data]
                    dequantized_state[state_key] = torch.cat(dequantized_data)
            
            dequantized_optimizer[key] = dequantized_state
        
        return dequantized_optimizer

    @staticmethod
    def calculate_percentage_loss(original, dequantized):
        return torch.abs((original - dequantized) / (original + 1e-10)) * 100

class Int8NaiveQuantizer(Quantizer):
    def quantize_cluster(self, data, min_val, max_val, signed=True):
        scale_factor = (max_val - min_val) / 255
        quantized_data = torch.round((data - min_val) / scale_factor).to(torch.int8)
        return quantized_data, scale_factor
    
    def dequantize_cluster(self, quantized_data, min_val, scale_factor, signed=True):
        dequantized_data = (quantized_data.to(torch.float32) * scale_factor) + min_val
        return dequantized_data

class Int8DynamicQuantizer(Quantizer):
    def __init__(self, device=None):
        super().__init__(device)
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
        
    def quantize_cluster(self, data, min_val, max_val, signed=True):
        scale_factor = max_val - min_val
        scaled_data = 2 * (data - min_val) / scale_factor - 1 if signed else (data - min_val) / scale_factor
        qmap = self.qmap_signed if signed else self.qmap_unsigned
        quantized_indices = torch.argmin(torch.abs(scaled_data.unsqueeze(-1) - qmap), dim=-1)
        quantized_data = quantized_indices.to(torch.int8) - 128 if signed else quantized_indices.to(torch.int8)
        return quantized_data, scale_factor
    
    def dequantize_cluster(self, quantized_data, min_val, scale_factor, signed=True):
        qmap = self.qmap_signed if signed else self.qmap_unsigned
        quantized_data = quantized_data.to(torch.int64)
        dequantized_data = qmap[quantized_data + 128] if signed else qmap[quantized_data]
        dequantized_data = 0.5 * (dequantized_data + 1) * scale_factor + min_val if signed else dequantized_data * scale_factor + min_val
        return dequantized_data

class Int8BlockwiseQuantizer(Int8DynamicQuantizer):
    def __init__(self, block_size=4096, device=None):
        super().__init__(device)
        self.block_size = block_size

    def quantize_all(self, clustered_optimizer):
        quantized_optimizer = {}
        for key, state_dict in clustered_optimizer.items():
            quantized_state = {}
            for state_key, clusters in state_dict.items():
                signed = state_key == 'exp_avg'
                quantized_clusters = self.quantize(clusters, signed)
                quantized_state[state_key] = quantized_clusters
            quantized_optimizer[key] = quantized_state
        return quantized_optimizer

    def dynamic_blockwise_quantize(self, data):
        num_blocks = (data.numel() + self.block_size - 1) // self.block_size
        padded_data = torch.nn.functional.pad(data.flatten(), (0, num_blocks * self.block_size - data.numel()))
        blocks = padded_data.view(num_blocks, self.block_size)
        
        quantized_blocks = []
        block_info = []
        for block in blocks:
            block_min, block_max = block.min(), block.max()
            quantized_data, scale_factor = self.quantize_cluster(block, block_min, block_max, signed=True)
            quantized_blocks.append(quantized_data)
            block_info.append({'min_val': block_min.item(), 'scale_factor': scale_factor.item()})
        
        quantized_data = torch.cat(quantized_blocks)[:data.numel()].reshape(data.shape)
        return quantized_data, block_info

    def dynamic_blockwise_dequantize(self, quantized_data, block_info, original_shape):
        num_blocks = len(block_info)
        flat_quantized = quantized_data.flatten()
        padded_length = num_blocks * self.block_size
        
        # Pad the quantized data if necessary
        if flat_quantized.numel() < padded_length:
            flat_quantized = torch.nn.functional.pad(flat_quantized, (0, padded_length - flat_quantized.numel()))
        
        blocks = flat_quantized.view(num_blocks, self.block_size)
        
        dequantized_blocks = []
        for block, info in zip(blocks, block_info):
            dequantized_block = self.dequantize_cluster(block, info['min_val'], info['scale_factor'], signed=True)
            dequantized_blocks.append(dequantized_block)
        
        dequantized_data = torch.cat(dequantized_blocks)[:np.prod(original_shape)].reshape(original_shape)
        return dequantized_data

    def quantize_cluster(self, data, min_val, max_val, signed=True):
        scale_factor = max_val - min_val
        num_blocks = (data.numel() + self.block_size - 1) // self.block_size
        padded_data = torch.nn.functional.pad(data.flatten(), (0, num_blocks * self.block_size - data.numel()))
        blocks = padded_data.view(num_blocks, self.block_size)
        
        quantized_blocks = []
        for block in blocks:
            block_min, block_max = block.min(), block.max()
            quantized, _ = super().quantize_cluster(block, block_min, block_max, signed)
            quantized_blocks.append(quantized)
        
        quantized_data = torch.cat(quantized_blocks)[:data.numel()].reshape(data.shape)
        return quantized_data, scale_factor
    
    def dequantize_cluster(self, quantized_data, min_val, scale_factor, signed=True):
        return super().dequantize_cluster(quantized_data, min_val, scale_factor, signed)

class FP16Quantizer(Quantizer):
    def quantize_cluster(self, data, min_val, max_val, signed=True):
        scale_factor = (max_val - min_val) / 255
        quantized_data = torch.round((data - min_val) / scale_factor).to(torch.float16)
        return quantized_data, scale_factor
    
    def dequantize_cluster(self, quantized_data, min_val, scale_factor, signed=True):
        dequantized_data = (quantized_data.to(torch.float32) * scale_factor) + min_val
        return dequantized_data
