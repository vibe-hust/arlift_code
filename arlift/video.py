"""Stream frames to FFmpeg and publish only a complete video."""
import os
from contextlib import contextmanager
from pathlib import Path
import subprocess
import tempfile


@contextmanager
def new_video(destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    with tempfile.TemporaryDirectory(dir=destination.parent) as scratch:
        path = Path(scratch) / "output.mp4"
        yield path
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError("The encoder produced no video.")
        os.link(path, destination)


@contextmanager
def rgb_writer(path, size, fps):
    width, height = size
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-n",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s:v", f"{width}x{height}",
           "-r", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "slow",
           "-crf", "16", "-pix_fmt", "yuv420p", str(path)]
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        yield process.stdin
        process.stdin.close()
        if process.wait():
            raise RuntimeError("FFmpeg failed to encode the video.")
    finally:
        if process.poll() is None:
            process.kill()
        process.stdin.close()
        process.wait()
