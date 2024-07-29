
import torch

def load_checkpoint(checkpoint_path):
    """Load a checkpoint and return the model state dict."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(checkpoint.keys())
    return checkpoint['0,1']

checkpoint_path = "/mnt/pengyanxin/Megatron-LLaMA/fine-tuned-ckpts/iter_0000050/mp_rank_00_000/distrib_optim.pt"

checkpoint = torch.load(checkpoint_path, map_location='cpu')

print(checkpoint[0]['param'])
