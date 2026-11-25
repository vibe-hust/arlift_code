"""Compact memory and online RoPE for each Krea branch."""
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class MemRoPEConfig:
    sink_frames: int
    long_memory_frames: int
    short_memory_frames: int
    recent_frames: int
    current_frames: int
    alpha_long: float
    alpha_short: float
    online_rope: bool

    @property
    def history_capacity_frames(self):
        return self.sink_frames + self.long_memory_frames + self.short_memory_frames + self.recent_frames

    @property
    def capacity_frames(self):
        return self.history_capacity_frames + self.current_frames

    def validate(self):
        layout = (self.sink_frames, self.long_memory_frames, self.short_memory_frames,
                  self.recent_frames, self.current_frames)
        if any(type(x) is not int for x in layout) or layout != (1, 1, 1, 4, 3):
            raise ValueError("The Krea preset uses one anchor, two summaries, four recent frames and a three-frame block")
        if not (0 < self.alpha_long <= 1 and 0 < self.alpha_short <= 1 and self.online_rope):
            raise ValueError("MemRoPE requires alpha in (0,1] and Online RoPE")


class MemRoPEController:
    """Commit clean K/V once per completed three-latent-frame generation block.

    Before compression the packed layout is sink + recent.  After the first
    overflow it is sink + long + short + recent.  The two memory frames are
    absent, not zero-filled attention entries, until actual eviction occurs.
    """

    def __init__(self, caches, frame_tokens: int, config: MemRoPEConfig):
        config.validate()
        if frame_tokens < 1 or not caches:
            raise ValueError("MemRoPE needs a nonempty per-layer cache and positive frame token count")
        self.caches = caches
        self.frame_tokens = frame_tokens
        self.config = config
        self.committed_blocks = 0
        self.events = []
        for cache in caches:
            if cache["k"].shape != cache["v"].shape or cache["k"].shape[1] != config.capacity_frames * frame_tokens:
                raise ValueError("MemRoPE physical K/V cache differs from the configured capacity")
            cache["memrope_history_frames"] = 0
            cache["memrope_sink_frames"] = 0
            cache["memrope_recent_frames"] = 0
            cache["memrope_memory_initialized"] = False
            cache["memrope_phase"] = "idle"
            cache["memrope_clean_forward_done"] = False
            cache["local_end_index"] = 0
            cache["global_end_index"] = 0

    @property
    def first(self):
        return self.caches[0]

    @property
    def history_frames(self):
        return self.first["memrope_history_frames"]

    def begin(self, phase: str):
        if phase not in ("denoise", "clean_commit"):
            raise ValueError("Invalid MemRoPE forward phase")
        for cache in self.caches:
            if cache["memrope_history_frames"] != self.history_frames:
                raise RuntimeError("Layer caches disagree about temporal layout")
            cache["memrope_phase"] = phase
            if phase == "clean_commit":
                cache["memrope_clean_forward_done"] = False

    def commit(self):
        """Current slots contain the zero-timestep clean forward from every layer."""
        cfg, s = self.config, self.frame_tokens
        if any(c["memrope_phase"] != "clean_commit" or not c["memrope_clean_forward_done"]
               for c in self.caches):
            raise RuntimeError("Cannot commit noisy or incomplete MemRoPE K/V")
        if any(c["local_end_index"] != (c["memrope_history_frames"] + cfg.current_frames) * s
               for c in self.caches):
            raise RuntimeError("Every layer must finish the clean current-block forward")
        old_sink = self.first["memrope_sink_frames"]
        old_recent = self.first["memrope_recent_frames"]
        initialized = self.first["memrope_memory_initialized"]
        event = {"block": self.committed_blocks, "sink_before": old_sink,
                 "recent_before": old_recent, "memory_before": initialized,
                 "evicted_frames": 0, "memory_updated": False}
        for cache in self.caches:
            self._commit_layer(cache)
            cache["memrope_phase"] = "idle"
            cache["memrope_clean_forward_done"] = False
            cache["global_end_index"] = (self.committed_blocks + 1) * cfg.current_frames * s
            cache["local_end_index"] = cache["memrope_history_frames"] * s
        if any((c["memrope_history_frames"], c["memrope_sink_frames"],
                c["memrope_recent_frames"], c["memrope_memory_initialized"])
               != (self.first["memrope_history_frames"], self.first["memrope_sink_frames"],
                   self.first["memrope_recent_frames"], self.first["memrope_memory_initialized"])
               for c in self.caches):
            raise RuntimeError("MemRoPE layer layouts diverged")
        event.update({"sink_after": self.first["memrope_sink_frames"],
                      "recent_after": self.first["memrope_recent_frames"],
                      "memory_after": self.first["memrope_memory_initialized"],
                      "history_after": self.history_frames,
                      "evicted_frames": max(0, old_recent + cfg.current_frames
                                            - (cfg.sink_frames if old_sink == 0 else 0)
                                            - cfg.recent_frames)})
        event["memory_updated"] = event["evicted_frames"] > 0
        self.events.append(event)
        self.committed_blocks += 1
        return event

    def _commit_layer(self, cache):
        cfg, s = self.config, self.frame_tokens
        old_history = cache["memrope_history_frames"]
        sink = cache["memrope_sink_frames"]
        recent = cache["memrope_recent_frames"]
        initialized = cache["memrope_memory_initialized"]
        if old_history != sink + recent + (2 if initialized else 0):
            raise RuntimeError("Corrupt MemRoPE layout")
        # The current three frames are clean, normalized, unrotated K/V.
        start = old_history * s
        current_k = cache["k"][:, start:start + cfg.current_frames * s].clone()
        current_v = cache["v"][:, start:start + cfg.current_frames * s].clone()
        incoming_frames = cfg.current_frames
        if sink == 0:
            cache["k"][:, :cfg.sink_frames * s].copy_(current_k[:, :cfg.sink_frames * s])
            cache["v"][:, :cfg.sink_frames * s].copy_(current_v[:, :cfg.sink_frames * s])
            sink, recent = cfg.sink_frames, 0
            # A Krea block still has three latent frames. With a one-frame
            # sink, keep its other two frames as recent history immediately.
            current_k = current_k[:, cfg.sink_frames * s:]
            current_v = current_v[:, cfg.sink_frames * s:]
            incoming_frames -= cfg.sink_frames
        if incoming_frames:
            recent_start = (sink + (2 if initialized else 0)) * s
            combined_k = torch.cat((cache["k"][:, recent_start:recent_start + recent * s], current_k), dim=1)
            combined_v = torch.cat((cache["v"][:, recent_start:recent_start + recent * s], current_v), dim=1)
            total_recent = recent + incoming_frames
            evicted = max(0, total_recent - cfg.recent_frames)
            if evicted:
                # Official fixed EMA pools the evicted *time and space token*
                # dimensions together; every spatial site in a memory frame
                # receives the same per-head summary.  Accumulate in fp32.
                pooled_k = combined_k[:, :evicted * s].float().mean(dim=1, keepdim=True)
                pooled_v = combined_v[:, :evicted * s].float().mean(dim=1, keepdim=True)
                long_pos, short_pos = sink * s, (sink + 1) * s
                for tensor, pooled in ((cache["k"], pooled_k),
                                       (cache["v"], pooled_v)):
                    if initialized:
                        tensor[:, long_pos:long_pos + s].copy_(
                            ((1 - cfg.alpha_long) * tensor[:, long_pos:long_pos + s].float()
                             + cfg.alpha_long * pooled).to(tensor.dtype))
                        tensor[:, short_pos:short_pos + s].copy_(
                            ((1 - cfg.alpha_short) * tensor[:, short_pos:short_pos + s].float()
                             + cfg.alpha_short * pooled).to(tensor.dtype))
                    else:
                        tensor[:, long_pos:long_pos + s].copy_(pooled.to(tensor.dtype))
                        tensor[:, short_pos:short_pos + s].copy_(pooled.to(tensor.dtype))
                initialized = True
            kept = total_recent - evicted
            dest = (sink + (2 if initialized else 0)) * s
            cache["k"][:, dest:dest + kept * s].copy_(combined_k[:, evicted * s:])
            cache["v"][:, dest:dest + kept * s].copy_(combined_v[:, evicted * s:])
            recent = kept
        cache["memrope_sink_frames"] = sink
        cache["memrope_recent_frames"] = recent
        cache["memrope_memory_initialized"] = initialized
        cache["memrope_history_frames"] = sink + recent + (2 if initialized else 0)
        if cache["memrope_history_frames"] > cfg.history_capacity_frames:
            raise RuntimeError("MemRoPE history exceeded its fixed capacity")


def online_rotate_keys(raw_keys, grid_sizes, freqs, frame_tokens, rotate):
    """Apply the original 3D Krea RoPE once per retained latent frame.

    Per-frame rotation bounds the temporary float64 intermediate in Krea's
    causal_rope_apply.  The returned bf16 buffer is transient, not the cache.
    """
    if raw_keys.shape[1] % frame_tokens:
        raise ValueError("Online RoPE requires complete latent frames")
    frames = raw_keys.shape[1] // frame_tokens
    one_frame_grid = grid_sizes.clone()
    one_frame_grid[:, 0] = 1
    result = torch.empty_like(raw_keys)
    for frame in range(frames):
        start = frame * frame_tokens
        result[:, start:start + frame_tokens].copy_(
            rotate(raw_keys[:, start:start + frame_tokens], one_frame_grid, freqs,
                   start_frame=frame).to(raw_keys.dtype))
    return result
