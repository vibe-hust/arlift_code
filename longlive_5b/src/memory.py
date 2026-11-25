"""Branch-local LiteMem for native LongLive attention."""
from contextlib import contextmanager
from types import MethodType
import torch
from wan_5b.modules.attention import attention
from wan_5b.modules.causal_model import causal_rope_apply


class BranchAttention:
    def __init__(self, pipeline, token_hw, memrope):
        if pipeline.guidance_scale != 1.0 or pipeline.quantize_kv:
            raise ValueError("Expected BF16 conditional inference with unquantized KV.")
        self.pipeline = pipeline
        self.hw = tuple(token_hw)
        self.frame_tokens = self.hw[0] * self.hw[1]
        self.memrope = memrope
        self.sink_capacity = memrope["sink_frames"]
        self.recent_capacity = memrope["recent_frames"]
        if (memrope["long_memory_frames"], memrope["short_memory_frames"]) != (1, 1):
            raise ValueError("The branch memory requires one long and one short summary.")
        self.capacity = self.sink_capacity + 2 + self.recent_capacity + 8
        self.history = 0
        self.sink, self.recent, self.initialized = 0, 0, False
        self.block, self.phase = -1, "idle"
        self.clean_seen = set()
        self.events = []
        self.calls = {"denoise": 0, "terminal_prediction": 0, "clean_kv": 0}

    def allocate(self, device, dtype):
        caches = []
        for block in self.pipeline.generator.model.blocks:
            layer = block.self_attn
            shape = (1, self.capacity * self.frame_tokens, layer.num_heads, layer.head_dim)
            caches.append({'k': torch.zeros(shape, device=device, dtype=dtype),
                'v': torch.zeros(shape, device=device, dtype=dtype), 'quantized': False,
                'global_end_index': torch.zeros(1, device=device, dtype=torch.long),
                'local_end_index': torch.zeros(1, device=device, dtype=torch.long),
                'pinned_start': torch.full((1,), -1, device=device, dtype=torch.long),
                'pinned_len': torch.zeros(1, device=device, dtype=torch.long)})
        self.pipeline.kv_cache_pos = caches
        self.pipeline.kv_cache_neg = None
        self.pipeline._initialize_crossattn_cache(1, dtype, device)

    def begin_block(self, block):
        if block != self.block + 1 or self.phase not in ('idle', 'committed'):
            raise RuntimeError('Previous block did not commit clean KV')
        self.block = block
        self.phase = 'idle'

    def set_phase(self, phase, step=None):
        if phase not in self.calls:
            raise ValueError('Unknown attention phase')
        self.phase = phase
        self.clean_seen.clear()

    @contextmanager
    def installed(self):
        saved = []
        try:
            for index, block in enumerate(self.pipeline.generator.model.blocks):
                module = block.self_attn
                saved.append((module, 'forward' in module.__dict__, module.__dict__.get('forward')))

                def forward(layer, *args, _index=index, **kwargs):
                    return self.forward(layer, _index, *args, **kwargs)

                module.forward = MethodType(forward, module)
            yield self
        finally:
            for module, had_override, previous in reversed(saved):
                if had_override:
                    module.forward = previous
                else:
                    module.__dict__.pop('forward', None)

    def forward(self, layer, index, x, seq_lens, grid_sizes, freqs, block_mask,
                kv_cache=None, current_start=0, cache_start=None, t_scale=1.0,
                use_relative_rope=False, method='linear', original_seq_len=None,
                temporal_offset=0.0):
        b, s = x.shape[:2]
        grid = tuple(int(v) for v in grid_sizes[0].tolist())
        if (b != 1 or grid != (8, *self.hw) or s != 8 * self.frame_tokens
                or block_mask is not None or kv_cache is None or self.phase not in self.calls
                or current_start != self.block * 8 * self.frame_tokens
                or (cache_start is not None and cache_start != current_start)
                or t_scale != 1.0 or method != 'linear' or temporal_offset != 0.0):
            raise ValueError('Unexpected LongLive attention geometry/phase/position')
        n, d = layer.num_heads, layer.head_dim
        q = layer.norm_q(layer.q(x)).view(b, s, n, d)
        k = layer.norm_k(layer.k(x)).view(b, s, n, d)
        v = layer.v(x).view(b, s, n, d)
        start, end = self.history * self.frame_tokens, (self.history + 8) * self.frame_tokens
        if end > kv_cache['k'].shape[1]:
            raise RuntimeError('Branch cache exceeded capacity')
        # Noisy forwards touch scratch current slots only. History changes only
        # after a separate clean-KV forward on the final emitted block.
        kv_cache['k'][:, start:end].copy_(k)
        kv_cache['v'][:, start:end].copy_(v)
        query = causal_rope_apply(q, [grid], freqs, start_frame=self.history).type_as(v)
        # Rotate a transient view of the normalized raw cache, one frame at
        # a time to bound the native FP64/complex temporary.
        keys = torch.empty_like(kv_cache['k'][:, :end])
        for frame in range(self.history + 8):
            span = slice(frame * self.frame_tokens, (frame + 1) * self.frame_tokens)
            keys[:, span].copy_(causal_rope_apply(kv_cache['k'][:, span],
                [(1, *self.hw)], freqs, start_frame=frame).type_as(v))
        result = attention(query, keys, kv_cache['v'][:, :end], fa_version=2)
        self.calls[self.phase] += 1
        if self.phase == 'clean_kv':
            self.clean_seen.add(index)
        # Match this pinned backbone's (output, cache_update_info) interface.
        return layer.o(result.flatten(2)), (current_start + s, end, None)

    def commit(self):
        caches = self.pipeline.kv_cache_pos
        if self.phase != 'clean_kv' or self.clean_seen != set(range(len(caches))):
            raise RuntimeError('Cannot commit noisy, terminal-prediction or incomplete KV')
        old = {'sink': self.sink, 'recent': self.recent, 'memory': self.initialized,
               'history': self.history}
        next_layout = None
        evicted_count = 0
        for cache in caches:
            ft = self.frame_tokens
            current = {key: cache[key][:, self.history * ft:(self.history + 8) * ft].clone()
                       for key in ('k', 'v')}
            sink_take = min(self.sink_capacity - self.sink, 8)
            tail_start = (self.sink + (2 if self.initialized else 0)) * ft
            total_recent = self.recent + 8 - sink_take
            evicted = max(0, total_recent - self.recent_capacity)
            kept = total_recent - evicted
            for key in ('k', 'v'):
                tail = torch.cat((cache[key][:, tail_start:tail_start + self.recent * ft],
                                  current[key][:, sink_take * ft:]), dim=1)
                if sink_take:
                    cache[key][:, self.sink * ft:(self.sink + sink_take) * ft].copy_(
                        current[key][:, :sink_take * ft])
                sink = self.sink + sink_take
                initialized = self.initialized or evicted > 0
                if evicted:
                    # For each EMA summary, pool the evicted time and
                    # space tokens together, then broadcast to a memory frame.
                    pooled = tail[:, :evicted * ft].float().mean(dim=1, keepdim=True)
                    for offset, alpha in ((0, self.memrope['alpha_long']),
                                          (1, self.memrope['alpha_short'])):
                        span = slice((sink + offset) * ft, (sink + offset + 1) * ft)
                        value = ((1 - alpha) * cache[key][:, span].float() + alpha * pooled
                                 if self.initialized else pooled)
                        cache[key][:, span].copy_(value.to(cache[key].dtype))
                dest = (sink + (2 if initialized else 0)) * ft
                cache[key][:, dest:dest + kept * ft].copy_(tail[:, evicted * ft:])
            evicted_count = evicted
            next_layout = (sink, kept, initialized, sink + kept + (2 if initialized else 0))
        self.sink, self.recent, self.initialized, self.history = next_layout
        for cache in caches:
            cache['local_end_index'].fill_(self.history * self.frame_tokens)
        self.events.append({'block': self.block, 'before': old,
            'after': {'sink': self.sink, 'recent': self.recent,
                      'memory': self.initialized, 'history': self.history},
            'evicted_frames': evicted_count, 'current_frames': 8})
        self.phase = 'committed'
