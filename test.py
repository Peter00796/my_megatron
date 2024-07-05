import torch
import torch.distributed as dist
import os

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def print_default_group_info():
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"Default Group - Rank: {rank}, World Size: {world_size}")

def main():
    world_size = 2
    processes = []

    for rank in range(world_size):
        p = torch.multiprocessing.Process(target=run, args=(rank, world_size))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

def run(rank, world_size):
    setup(rank, world_size)
    dist.barrier()
    print_default_group_info()

if __name__ == "__main__":
    main()
