import torch

from bitsnap.bitsnap_quantizer import BitSnapOptimizerQuantizer


def main() -> None:
    q = BitSnapOptimizerQuantizer(alpha=0.9, seed=1234)
    t = torch.randn((1024, 4096), dtype=torch.float32)
    q.dry_run(t)

    # Simple invariants.
    ct = q.compress(t)
    recon = q.decompress(ct, device=torch.device("cpu"))
    assert tuple(recon.shape) == tuple(t.shape)
    assert recon.dtype == torch.float32


if __name__ == "__main__":
    main()

