"""Single-node CUDA process-group setup for inference."""
from contextlib import contextmanager
from datetime import timedelta
import os


@contextmanager
def cuda_ranks(supported):
    import torch
    import torch.distributed as dist

    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("LOCAL_RANK", 0))
    if world not in supported or int(os.environ.get("LOCAL_WORLD_SIZE", world)) != world:
        raise ValueError(f"This backend requires one node with GPU count in {supported}.")
    if not torch.cuda.is_available() or torch.cuda.device_count() < world:
        raise RuntimeError(f"{world} CUDA device(s) required for this run.")
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    try:
        yield device, rank, world
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
