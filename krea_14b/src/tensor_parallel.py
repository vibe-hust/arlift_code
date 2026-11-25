"""Column/row FFN partitioning; TP1 keeps the checkpoint's original module."""
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class TensorParallelFFN(nn.Module):
    def __init__(self, original, rank, world):
        super().__init__()
        first, activation, last = original
        if not dist.is_initialized() or first.out_features % world:
            raise ValueError('FFN width must divide the initialized TP group')
        width = first.out_features // world
        start, stop = rank * width, (rank + 1) * width
        self.first = nn.Linear(first.in_features, width, bias=first.bias is not None,
                               device=first.weight.device, dtype=first.weight.dtype)
        self.last = nn.Linear(width, last.out_features, bias=last.bias is not None,
                              device=last.weight.device, dtype=last.weight.dtype)
        with torch.no_grad():
            self.first.weight.copy_(first.weight[start:stop])
            if first.bias is not None:
                self.first.bias.copy_(first.bias[start:stop])
            self.last.weight.copy_(last.weight[:, start:stop])
            if last.bias is not None:
                self.last.bias.copy_(last.bias)
        self.activation = activation

    def forward(self, value):
        value = self.activation(self.first(value))
        value = F.linear(value, self.last.weight, None)
        dist.all_reduce(value)
        return value if self.last.bias is None else value + self.last.bias
