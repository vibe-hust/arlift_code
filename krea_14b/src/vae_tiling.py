"""Bounded spatial tiling for the existing causal VAEDecoderWrapper.

Each spatial tile has its own persistent causal feature cache. The grid is fixed
for the lifetime of a video, so the same tile sees consecutive latent frames.
Spatial attention inside the decoder becomes tile-local; this is an explicit
VAE reconstruction strategy and is not pixel-identical to whole-frame decode.
"""

import math

import torch


class SpatialTiledVAEDecoder:
    def __init__(self, tile_size: int, overlap: int, cache_slots: int):
        if tile_size < 16 or overlap < 1 or overlap * 2 >= tile_size:
            raise ValueError("VAE tile size must be >=16 and overlap must be in [1, tile_size/2)")
        self.tile_size = tile_size
        self.overlap = overlap
        self.cache_slots = cache_slots
        self.shape = None
        self.tiles = []
        self.caches = {}
        self.normalizer = None
        self.blocks_decoded = 0
        self.active_tile = None

    @staticmethod
    def _starts(length: int, size: int, overlap: int):
        if length <= size:
            return [0]
        count = math.ceil((length - size) / (size - overlap)) + 1
        last = length - size
        return [round(index * last / (count - 1)) for index in range(count)]

    @staticmethod
    def _axis_weight(start: int, end: int, previous_end: int | None,
                     next_start: int | None, scale: int):
        weight = torch.ones((end - start) * scale, dtype=torch.float32)
        if previous_end is not None:
            pixels = (previous_end - start) * scale
            if pixels > 0:
                weight[:pixels] = (torch.arange(pixels, dtype=torch.float32) + 1) / (pixels + 1)
        if next_start is not None:
            pixels = (end - next_start) * scale
            if pixels > 0:
                weight[-pixels:] *= (pixels - torch.arange(pixels, dtype=torch.float32)) / (pixels + 1)
        return weight

    def _configure(self, height: int, width: int):
        if self.shape is not None:
            if self.shape != (height, width):
                raise ValueError("VAE tile grid changed within one video")
            return
        self.shape = (height, width)
        ys = self._starts(height, self.tile_size, self.overlap)
        xs = self._starts(width, self.tile_size, self.overlap)
        self.normalizer = torch.zeros((height * 8, width * 8), dtype=torch.float32)
        for yi, y0 in enumerate(ys):
            y1 = min(height, y0 + self.tile_size)
            wy = self._axis_weight(y0, y1,
                                   None if yi == 0 else ys[yi - 1] + min(height, self.tile_size),
                                   None if yi + 1 == len(ys) else ys[yi + 1], 8)
            for xi, x0 in enumerate(xs):
                x1 = min(width, x0 + self.tile_size)
                wx = self._axis_weight(x0, x1,
                                       None if xi == 0 else xs[xi - 1] + min(width, self.tile_size),
                                       None if xi + 1 == len(xs) else xs[xi + 1], 8)
                weight = wy[:, None] * wx[None, :]
                self.tiles.append((y0, y1, x0, x1, weight))
                self.caches[(y0, x0)] = [None] * self.cache_slots
                self.normalizer[y0 * 8:y1 * 8, x0 * 8:x1 * 8].add_(weight)
        if torch.any(self.normalizer <= 0):
            raise RuntimeError("VAE tile grid left uncovered output pixels")

    @property
    def tile_count(self):
        return len(self.tiles)

    @torch.inference_mode()
    def decode_block(self, decoder, latents: torch.Tensor):
        if latents.ndim != 5 or latents.shape[0] != 1 or latents.shape[1] != 3:
            raise ValueError("Tiled VAE expects one three-latent-frame generation block")
        height, width = latents.shape[-2:]
        self._configure(height, width)
        chunks = []
        for latent_index in range(latents.shape[1]):
            canvas = None
            frame_count = None
            for tile_index, (y0, y1, x0, x1, weight) in enumerate(self.tiles):
                self.active_tile = {"latent_in_block": latent_index, "tile_index": tile_index,
                                    "latent_bounds": [y0, y1, x0, x1]}
                key = (y0, x0)
                active_cache = [value.to(latents.device) if isinstance(value, torch.Tensor) else value
                                for value in self.caches[key]]
                pixels, next_cache = decoder(
                    latents[:, latent_index:latent_index + 1, :, y0:y1, x0:x1], *active_cache)
                if pixels.shape[0] != 1 or pixels.shape[2] != 3 or pixels.shape[-2:] != ((y1 - y0) * 8, (x1 - x0) * 8):
                    raise RuntimeError("VAE tile output has an unexpected spatial shape")
                if canvas is None:
                    frame_count = pixels.shape[1]
                    canvas = torch.zeros((1, frame_count, 3, height * 8, width * 8), dtype=torch.float32)
                elif pixels.shape[1] != frame_count:
                    raise RuntimeError("VAE tiles disagree on the temporal output length")
                tile_cpu = pixels.to("cpu", dtype=torch.float32)
                canvas[:, :, :, y0 * 8:y1 * 8, x0 * 8:x1 * 8].add_(
                    tile_cpu * weight[None, None, None, :, :])
                self.caches[key] = [value.detach().to("cpu") if isinstance(value, torch.Tensor) else value
                                    for value in next_cache]
                del active_cache, next_cache, pixels, tile_cpu
            canvas.div_(self.normalizer[None, None, None, :, :]).clamp_(-1, 1)
            chunks.append(canvas)
        self.blocks_decoded += 1
        self.active_tile = None
        return torch.cat(chunks, dim=1)
