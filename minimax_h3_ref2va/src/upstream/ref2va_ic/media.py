"""Draft frames in, refined video out.

read_draft takes 124 frames picked at 24 fps (frame i of the source is int(i * fps / 24)),
at least 113 real frames (shorter clips are refused), the tail repeats the last real frame; each frame is
cover-resized (bilinear; antialiased when shrinking) and center-cropped to the LQ canvas.
"""
from contextlib import nullcontext
from fractions import Fraction
from itertools import chain

import numpy as np

FRAMES = 124
MIN_REAL = 113


def _cover(x, height, width):
    from torch.nn import functional as F

    h, w = x.shape[-2:]
    if (h, w) == (height, width):
        return x
    scale = max(height / h, width / w)
    sh, sw = max(height, round(h * scale)), max(width, round(w * scale))
    x = F.interpolate(x, size=(sh, sw), mode='bilinear', align_corners=False, antialias=scale < 1)
    i, j = (sh - height) // 2, (sw - width) // 2
    return x[..., i:i + height, j:j + width]


def read_draft(path, height, width):
    """-> float [124, 3, height, width] in [0, 1] (CPU), number of real frames, source fps."""
    import av
    import torch
    out = torch.empty((FRAMES, 3, height, width), dtype=torch.float32)
    valid = 0
    with av.open(str(path)) as video:
        stream = video.streams.video[0]
        stream.thread_type = 'AUTO'
        fps = float(stream.average_rate or 24.)
        wanted = [int(i * fps / 24.) for i in range(FRAMES)]
        for index, frame in enumerate(video.decode(stream)):
            if index < wanted[valid]:
                continue
            x = torch.from_numpy(frame.to_ndarray(format='rgb24')).permute(2, 0, 1)[None].float() / 255.
            x = _cover(x, height, width)
            while valid < FRAMES and wanted[valid] == index:
                out[valid] = x[0]
                valid += 1
            if valid == FRAMES:
                break
    if valid < MIN_REAL:
        raise ValueError(f'{path}: {valid} frames at 24 fps; the model needs at least {MIN_REAL} (about 4.7 s)')
    out[valid:] = out[valid - 1]
    return out.clamp(0, 1), valid, fps


def _write_audio(source, source_stream, output, stream, max_samples):
    import av

    rate = stream.codec_context.sample_rate
    layout = stream.layout.name
    resampler = av.AudioResampler(format='fltp', layout=layout, rate=rate)
    written = 0
    for decoded in chain(source.decode(source_stream), (None,)):
        if decoded is not None:
            # Matroska PCM may carry a channel count without channel positions.
            if decoded.layout.name != layout and decoded.layout.name in ('1 channels', '2 channels'):
                sample_rate = decoded.sample_rate
                decoded = av.AudioFrame.from_ndarray(decoded.to_ndarray(), format=decoded.format.name, layout=layout)
                decoded.sample_rate = sample_rate
            decoded.pts = None
        for frame in resampler.resample(decoded):
            if written + frame.samples > max_samples:
                samples = np.ascontiguousarray(frame.to_ndarray()[:, :max_samples - written])
                frame = av.AudioFrame.from_ndarray(samples, format='fltp', layout=layout)
            frame.pts = written
            frame.time_base = Fraction(1, rate)
            frame.sample_rate = rate
            output.mux(stream.encode(frame))
            written += frame.samples
            if written == max_samples:
                output.mux(stream.encode(None))
                return
    output.mux(stream.encode(None))


def write_video(frames, path, audio_from=None, crf=12):
    """Write H.264 at 24 fps and mux the draft's first audio track through PyAV."""
    import av

    count, height, width, _ = frames.shape
    source_context = av.open(str(audio_from)) if audio_from is not None else nullcontext()
    with source_context as source, av.open(str(path), mode='w', container_options={'movflags': '+faststart'}) as output:
        video = output.add_stream('libx264', rate=24, options={'preset': 'slow', 'crf': str(crf)})
        video.width, video.height, video.pix_fmt = width, height, 'yuv420p'
        source_audio = next(iter(source.streams.audio), None) if source is not None else None
        if source_audio is not None:
            rate = source_audio.codec_context.sample_rate
            audio = output.add_stream('aac', rate=rate)
            layout = source_audio.codec_context.layout.name
            audio.layout = {'1 channels': 'mono', '2 channels': 'stereo'}.get(layout, layout)
            audio.bit_rate = 192000
        for index, pixels in enumerate(frames):
            frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(pixels), format='rgb24')
            frame.pts = index
            frame.time_base = Fraction(1, 24)
            output.mux(video.encode(frame))
        output.mux(video.encode(None))
        if source_audio is not None:
            _write_audio(source, source_audio, output, audio, round(count * rate / 24))
