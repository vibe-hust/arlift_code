"""Optional two-GPU head/FFN tensor parallelism; native TP1 modules stay intact.

Hidden states and the sampler are replicated. Q/K/V and KV heads, plus FFN
intermediate channels, are partitioned. Row outputs and Q/K RMS statistics are
reduced over the node-local process group. No upstream source is modified.
"""
from types import MethodType
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F


def partition_bounds(width, rank, world):
    if type(width) is not int or width % world or not 0 <= rank < world:
        raise ValueError('Projection width must divide the tensor-parallel group')
    size = width//world
    return rank*size, (rank+1)*size


class ColumnLinear(nn.Module):
    def __init__(self, source, rank, world):
        super().__init__()
        start, stop = partition_bounds(source.out_features, rank, world)
        self.in_features, self.out_features = source.in_features, stop-start
        self.weight = nn.Parameter(source.weight[start:stop].detach().clone(), requires_grad=False)
        self.bias = (nn.Parameter(source.bias[start:stop].detach().clone(), requires_grad=False)
                     if source.bias is not None else None)

    def forward(self, value):
        return F.linear(value, self.weight, self.bias)


class RowLinear(nn.Module):
    def __init__(self, source, rank, world):
        super().__init__()
        start, stop = partition_bounds(source.in_features, rank, world)
        self.in_features, self.out_features = stop-start, source.out_features
        self.weight = nn.Parameter(source.weight[:, start:stop].detach().clone(), requires_grad=False)
        self.bias = (nn.Parameter(source.bias.detach().clone(), requires_grad=False)
                     if source.bias is not None else None)

    def forward(self, value):
        result = F.linear(value, self.weight, None)
        dist.all_reduce(result)
        return result if self.bias is None else result+self.bias


class GlobalRMSNorm(nn.Module):
    def __init__(self, source, rank, world):
        super().__init__()
        self.full_width = source.weight.numel()
        self.eps = source.eps
        start, stop = partition_bounds(self.full_width, rank, world)
        self.weight = nn.Parameter(source.weight[start:stop].detach().clone(), requires_grad=False)

    def forward(self, value):
        squared = value.float().square().sum(dim=-1, keepdim=True)
        dist.all_reduce(squared)
        normed = value.float()*torch.rsqrt(squared/self.full_width+self.eps)
        # Match WanRMSNorm: cast normalized values before multiplying the weight.
        return normed.type_as(value)*self.weight


def shard_generator(generator):
    if not dist.is_initialized() or dist.get_world_size() != 2:
        raise ValueError('Use an initialized node-local two-rank group')
    rank, world = dist.get_rank(), dist.get_world_size()
    description = []
    for index, block in enumerate(generator.model.blocks):
        for name in ('self_attn', 'cross_attn'):
            layer = getattr(block, name)
            if getattr(layer, 'tp_world_size', 1) != 1:
                raise ValueError('An attention module has already been partitioned')
            full_heads = layer.num_heads
            partition_bounds(full_heads, rank, world)
            for projection in ('q', 'k', 'v'):
                setattr(layer, projection, ColumnLinear(getattr(layer, projection), rank, world))
            for normalization in ('norm_q', 'norm_k'):
                original = getattr(layer, normalization)
                if not isinstance(original, nn.Identity):
                    setattr(layer, normalization, GlobalRMSNorm(original, rank, world))
            layer.o = RowLinear(layer.o, rank, world)
            layer.num_heads = full_heads//world
            layer.tp_world_size, layer.tp_rank = world, rank
            description.append({'layer': index, 'attention': name, 'global_heads': full_heads,
                                'local_heads': layer.num_heads, 'head_dim': layer.head_dim})
        first, activation, last = block.ffn
        block.ffn = nn.Sequential(ColumnLinear(first, rank, world), activation,
                                  RowLinear(last, rank, world))
        description[-1]['local_ffn_width'] = block.ffn[0].out_features
    return description


def prepare_pipeline(pipeline):
    """Allocate native cross-cache metadata using the same local head partition."""
    def initialize(self, batch_size, dtype, device):
        caches = []
        for block in self.generator.model.blocks:
            layer = block.cross_attn
            shape = (batch_size, 512, layer.num_heads, layer.head_dim)
            caches.append({'k': torch.zeros(shape, dtype=dtype, device=device),
                           'v': torch.zeros(shape, dtype=dtype, device=device), 'is_init': False})
        self.crossattn_cache_pos = caches
        self.crossattn_cache_neg = None
    pipeline._initialize_crossattn_cache = MethodType(initialize, pipeline)


def shared_noise(shape, rng, device, dtype):
    value = (torch.randn(shape, generator=rng, device=device, dtype=dtype)
             if dist.get_rank() == 0 else torch.empty(shape, device=device, dtype=dtype))
    dist.broadcast(value, src=0)
    return value


def shared_conditioning(encoder, prompt, device):
    if dist.get_rank() == 0:
        result = encoder(text_prompts=[prompt])['prompt_embeds'].contiguous()
        if result.dtype != torch.bfloat16 or result.ndim != 3:
            raise ValueError('Expected native BF16 prompt embeddings')
        shape = torch.tensor(result.shape, device=device, dtype=torch.long)
    else:
        shape = torch.empty(3, device=device, dtype=torch.long)
    dist.broadcast(shape, src=0)
    if dist.get_rank() != 0:
        result = torch.empty(tuple(shape.tolist()), dtype=torch.bfloat16, device=device)
    dist.broadcast(result, src=0)
    return {'prompt_embeds': result}
