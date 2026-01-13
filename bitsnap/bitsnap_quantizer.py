from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import torch


@dataclass
class BitSnapCompressedTensor:
    """Torch-save-friendly container for 4-bit quantized tensors."""

    # Marker to detect compressed objects inside nested optimizer state dicts.
    __bitsnap__: bool
    version: int

    # Stored on CPU for portability.
    codebook: torch.Tensor  # float32 [16]
    packed_uint8: torch.Tensor  # uint8 [ceil(numel/2)]

    numel: int
    original_shape: Tuple[int, ...]
    original_dtype: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "__bitsnap__": True,
            "version": self.version,
            "codebook": self.codebook,
            "packed_uint8": self.packed_uint8,
            "numel": int(self.numel),
            "original_shape": tuple(self.original_shape),
            "original_dtype": str(self.original_dtype),
        }


class BitSnapOptimizerQuantizer:
    """Online Adaptive Quantization for optimizer state tensors.

    - Codebook size: 16 (4-bit indices)
    - Learning: 0.1% strided sampling + per-code means + EMA update
    """

    def __init__(self, alpha: float = 0.9, codebook_size: int = 16, seed: Optional[int] = None):
        assert 0.0 < alpha < 1.0
        assert codebook_size == 16, "BitSnap demo currently targets 4-bit (16-entry) codebooks."
        self.alpha = float(alpha)
        g = None
        if seed is not None:
            g = torch.Generator()
            g.manual_seed(int(seed))
        # Keep codebook on CPU; we move it to devices as needed for quant/dequant.
        self.codebook = torch.randn((codebook_size,), dtype=torch.float32, generator=g, device="cpu")

    @staticmethod
    def _rank() -> int:
        try:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                return int(torch.distributed.get_rank())
        except Exception:
            pass
        return 0

    @staticmethod
    def _pick_sample_stride(numel: int, sample_rate: float = 0.001) -> int:
        # For strided sampling, stride ~= 1/sample_rate.
        # For small tensors, stride=1 samples all elements which is acceptable.
        if numel <= 0:
            return 1
        stride = int(round(1.0 / sample_rate))
        return max(1, stride)

    def learn(self, t: torch.Tensor) -> None:
        """Update codebook with EMA based on strided samples from tensor t."""
        if not isinstance(t, torch.Tensor):
            return
        if t.numel() == 0:
            return
        # Operate on CPU float32 samples for stability and portability.
        flat = t.detach().reshape(-1)
        # Keep sampling cheap: if tensor is on GPU, only bring samples to CPU.
        stride = self._pick_sample_stride(flat.numel(), sample_rate=0.001)
        sample = flat[::stride]
        if sample.numel() == 0:
            return
        sample = sample.float()
        if sample.device.type != "cpu":
            sample = sample.cpu()

        codebook = self.codebook  # cpu float32 [16]

        # Assign each sample to closest codebook value.
        # Chunk samples to avoid large (N,16) temporary allocations.
        counts = torch.zeros((16,), dtype=torch.int64, device="cpu")
        sums = torch.zeros((16,), dtype=torch.float32, device="cpu")

        chunk = 1_000_000  # 1e6 samples => 1e6*16 float ops, reasonable.
        for start in range(0, sample.numel(), chunk):
            x = sample[start : start + chunk]  # cpu float32 [m]
            # distances: [m,16]
            d = (x[:, None] - codebook[None, :]).abs()
            idx = torch.argmin(d, dim=1).to(torch.int64)  # [m]
            counts.scatter_add_(0, idx, torch.ones_like(idx, dtype=torch.int64))
            sums.scatter_add_(0, idx, x)

        # Observed per-code means; if empty, keep old.
        observed = codebook.clone()
        nonzero = counts > 0
        observed[nonzero] = sums[nonzero] / counts[nonzero].to(torch.float32)

        # EMA update.
        self.codebook = self.alpha * codebook + (1.0 - self.alpha) * observed

    @staticmethod
    def _pack_int4(indices: torch.Tensor, numel: int) -> torch.Tensor:
        """Pack 0..15 int4 indices into uint8: low nibble first element, high nibble second."""
        assert indices.dtype == torch.uint8
        assert indices.device.type == "cpu"
        if numel == 0:
            return torch.empty((0,), dtype=torch.uint8, device="cpu")
        if indices.numel() != numel:
            indices = indices[:numel]
        # Pad to even length.
        if (numel % 2) == 1:
            indices = torch.cat([indices, torch.zeros((1,), dtype=torch.uint8, device="cpu")], dim=0)
        low = indices[0::2] & 0x0F
        high = (indices[1::2] & 0x0F) << 4
        return (low | high).contiguous()

    @staticmethod
    def _unpack_int4(packed: torch.Tensor, numel: int) -> torch.Tensor:
        """Unpack uint8 into uint8 indices 0..15, returning shape [numel]."""
        assert packed.dtype == torch.uint8
        if numel == 0:
            return torch.empty((0,), dtype=torch.uint8, device=packed.device)
        low = packed & 0x0F
        high = (packed >> 4) & 0x0F
        # Interleave low/high.
        out = torch.empty((packed.numel() * 2,), dtype=torch.uint8, device=packed.device)
        out[0::2] = low
        out[1::2] = high
        return out[:numel]

    def compress(self, t: torch.Tensor) -> Dict[str, Any]:
        """Compress FP32 tensor to BitSnapCompressedTensor dict (codebook + packed uint8 indices)."""
        assert isinstance(t, torch.Tensor)
        original_shape = tuple(t.shape)
        numel = int(t.numel())
        original_dtype = str(t.dtype)

        # Learn/update codebook first (online).
        self.learn(t)

        if numel == 0:
            ct = BitSnapCompressedTensor(
                __bitsnap__=True,
                version=1,
                codebook=self.codebook.clone().cpu(),
                packed_uint8=torch.empty((0,), dtype=torch.uint8, device="cpu"),
                numel=0,
                original_shape=original_shape,
                original_dtype=original_dtype,
            )
            return ct.to_dict()

        # Quantize in chunks to avoid allocating huge (N,16) matrices.
        codebook = self.codebook.to(device=t.device, dtype=torch.float32)
        flat = t.detach().reshape(-1).to(dtype=torch.float32)

        indices_cpu = torch.empty((numel,), dtype=torch.uint8, device="cpu")
        # Keep this moderate: (chunk,16) temporary can be large on GPU.
        chunk = 262_144  # elements (not bytes)
        for start in range(0, numel, chunk):
            x = flat[start : start + chunk]  # on t.device
            # distances: [m,16]
            d = (x[:, None] - codebook[None, :]).abs()
            idx = torch.argmin(d, dim=1).to(torch.uint8)  # 0..15
            indices_cpu[start : start + idx.numel()] = idx.detach().to(device="cpu")

        packed = self._pack_int4(indices_cpu, numel=numel)
        ct = BitSnapCompressedTensor(
            __bitsnap__=True,
            version=1,
            codebook=self.codebook.clone().cpu(),
            packed_uint8=packed,
            numel=numel,
            original_shape=original_shape,
            original_dtype=original_dtype,
        )
        return ct.to_dict()

    def decompress(self, obj: Dict[str, Any], device: Optional[torch.device] = None) -> torch.Tensor:
        assert obj.get("__bitsnap__", False) is True
        version = int(obj.get("version", 1))
        if version != 1:
            raise ValueError(f"Unsupported BitSnapCompressedTensor version: {version}")

        codebook: torch.Tensor = obj["codebook"]
        packed: torch.Tensor = obj["packed_uint8"]
        numel: int = int(obj["numel"])
        original_shape: Sequence[int] = obj["original_shape"]

        if device is None:
            device = torch.device("cpu")

        if numel == 0:
            return torch.empty(tuple(original_shape), dtype=torch.float32, device=device)

        if packed.device != torch.device("cpu"):
            packed = packed.cpu()
        if codebook.device != torch.device("cpu"):
            codebook = codebook.cpu()

        idx = self._unpack_int4(packed, numel=numel).to(torch.int64)  # cpu
        cb = codebook.to(device=device, dtype=torch.float32)
        out = cb[idx.to(device=device)]
        return out.reshape(tuple(original_shape))

    def dry_run(self, t: torch.Tensor) -> None:
        """Compress + decompress and print MSE stats for verification."""
        rank = self._rank()
        ct = self.compress(t)
        device = t.device if t.is_cuda else torch.device("cpu")
        recon = self.decompress(ct, device=device)
        orig = t.detach().to(dtype=torch.float32)
        rec = recon.detach().to(dtype=torch.float32)
        mse = torch.mean((orig - rec) ** 2).item() if orig.numel() else 0.0

        before_bytes = int(t.numel()) * 4  # fp32
        after_bytes = int(ct["packed_uint8"].numel()) * 1 + 16 * 4  # packed + codebook
        shape_str = str(list(t.shape))
        print(
            f"[BitSnap] Rank {rank}: Compressed tensor shape {shape_str} | "
            f"MSE: {mse:.6g} | bytes: {before_bytes}->{after_bytes}"
        )

