"""Validate inference settings before loading checkpoints or starting CUDA."""

import math

from .guidance import gaussian_parameters


def positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def finite_number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")
    return value


def image_size(value, name):
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must be [width, height].")
    return [positive_integer(dimension, name) for dimension in value]


def descending_schedule(sequence, name, first):
    if not isinstance(sequence, list) or len(sequence) < 2:
        raise ValueError(f"{name} must be a descending list ending at 0.")
    for value in sequence:
        finite_number(value, name)
    if sequence[0] != first or sequence[-1] != 0 or any(
        left <= right for left, right in zip(sequence, sequence[1:])
    ):
        raise ValueError(f"{name} must decrease strictly from {first} to 0.")


def validate_config(backbone, cfg):
    if not isinstance(cfg, dict):
        raise ValueError("The inference configuration must be a YAML mapping.")
    if type(cfg["seed"]) is not int or not 0 <= cfg["seed"] < 2**63:
        raise ValueError("The seed must be an integer in [0, 2**63).")
    native = image_size(cfg["native_size"], "native_size")
    output = image_size(cfg["output_size"], "output_size")
    if output != [2 * dimension for dimension in native]:
        raise ValueError("output_size must be exactly twice native_size on both axes.")
    frames = positive_integer(cfg["frames"], "frames")
    fps = positive_integer(cfg["fps"], "fps")
    gaussian = cfg["gaussian"]
    gaussian_parameters(gaussian["sigma"], gaussian["kernel_size"])

    if backbone == "minimax_h3_ref2va":
        if native != [1344, 768] or (frames, fps, cfg["audio_rate"]) != (124, 24, 32000):
            raise ValueError("MiniMax requires a 1344×768 draft with 124 frames at 24 fps and a 32 kHz draft audio rate.")
        descending_schedule(cfg["raw_sigmas"], "raw_sigmas", 1)
        if len(cfg["raw_sigmas"]) != 9:
            raise ValueError("MiniMax requires nine sigma values for eight HyperFlow steps.")
        shift = "video_shift"
        steps = 8
    else:
        blocks = positive_integer(cfg["blocks"], "blocks")
        block_latents = positive_integer(cfg["block_latents"], "block_latents")
        memory = cfg["memory"]
        for key in ("sink_frames", "long_memory_frames", "short_memory_frames", "recent_frames"):
            positive_integer(memory[key], f"memory.{key}")
        if (memory["long_memory_frames"], memory["short_memory_frames"]) != (1, 1):
            raise ValueError("LiteMem requires one long and one short summary.")
        for key in ("alpha_long", "alpha_short"):
            if not 0 < finite_number(memory[key], f"memory.{key}") <= 1:
                raise ValueError(f"memory.{key} must be in (0, 1].")
        shift = "timestep_shift"
        if backbone == "krea_14b":
            if native != [832, 480] or block_latents != 3:
                raise ValueError("Krea requires native_size [832, 480] and block_latents 3.")
            if (memory["sink_frames"], memory["recent_frames"]) != (1, 4):
                raise ValueError("Krea requires one anchor and four recent frames.")
            tile = cfg["vae_tile"]
            size = positive_integer(tile["size"], "vae_tile.size")
            overlap = positive_integer(tile["overlap"], "vae_tile.overlap")
            if size < 16 or 2 * overlap >= size:
                raise ValueError("VAE tile size must be at least 16 and overlap less than half the tile size.")
            descending_schedule(cfg["timesteps"], "timesteps", 1000)
            steps = len(cfg["timesteps"])
            available_frames = 4 * blocks * block_latents - 6
        else:
            if block_latents != 8 or any(dimension % 32 for dimension in native):
                raise ValueError("LongLive requires block_latents 8 and native dimensions divisible by 32.")
            if positive_integer(cfg["unipc_steps"], "unipc_steps") != 4:
                raise ValueError("LongLive requires four UniPC updates.")
            steps = 5  # Four UniPC updates and the separate terminal prediction.
            available_frames = 4 * blocks * block_latents - 3
        if frames > available_frames:
            raise ValueError(f"frames exceeds the {available_frames} frames available from the configured blocks.")

    if finite_number(cfg[shift], shift) <= 0:
        raise ValueError(f"{shift} must be positive.")
    weights = cfg["guidance_weights"]
    if not isinstance(weights, list) or len(weights) != steps:
        raise ValueError(f"guidance_weights must contain {steps} values.")
    if any(not 0 <= finite_number(weight, "guidance_weights") <= 1 for weight in weights):
        raise ValueError("Each guidance weight must be in [0, 1].")
