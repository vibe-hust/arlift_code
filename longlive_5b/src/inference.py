"""ARLIFT on the frozen LongLive 2.0 5B backbone."""
import gc
from contextlib import contextmanager
import os
from pathlib import Path
import sys

from arlift.devices import cuda_ranks
from arlift.guidance import fuse_clean, resize_reference
from arlift.video import new_video, rgb_writer

NATIVE = Path(__file__).resolve().parent / "upstream"


@contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def run_branch(cfg, generator, encoder, vae, conditioning, device, world, references=None):
    import torch
    from omegaconf import OmegaConf
    from pipeline import CausalDiffusionInferencePipeline
    from utils.config import normalize_config
    from .memory import BranchAttention
    from .sampling import configure_terminal, sample_block
    from .tensor_parallel import prepare_pipeline, shared_noise

    high = references is not None
    w, h = cfg["output_size" if high else "native_size"]
    block_frames = cfg["block_latents"]
    shape = (1, cfg["blocks"] * block_frames, 48, h // 16, w // 16)
    native = {"model_kwargs": {"model_name": "Wan2.2-TI2V-5B", "timestep_shift": cfg["timestep_shift"],
                               "num_frame_per_block": block_frames, "local_attn_size": 32},
              "data": {"image_or_video_shape": list(shape)},
              "inference": {"sampling_steps": cfg["unipc_steps"], "sink_size": 8, "guidance_scale": 1.0,
                            "multi_shot_sink": True, "multi_shot_rope_offset": 8, "streaming_vae": False,
                            "async_vae": False, "vae_type": "wan", "vae_device": None},
              "logging": {"seed": cfg["seed"]}}
    pipeline = CausalDiffusionInferencePipeline(normalize_config(OmegaConf.create(native)), device,
                                               generator=generator, text_encoder=encoder, vae=vae)
    if world == 2:
        prepare_pipeline(pipeline)
    state = BranchAttention(pipeline, (h // 32, w // 32), cfg["memory"])
    state.allocate(device, torch.bfloat16)
    outputs = []

    def forward(latents, timestep, condition, caches, offset):
        times = torch.full(latents.shape[:2], float(timestep), device=device, dtype=torch.float32)
        return generator(noisy_image_or_video=latents, conditional_dict=condition, timestep=times,
                         kv_cache=caches, crossattn_cache=pipeline.crossattn_cache_pos,
                         current_start=offset, cache_start=offset)

    try:
        with torch.random.fork_rng(devices=[device.index]), state.installed():
            torch.random.default_generator.manual_seed(cfg["seed"])
            torch.cuda.manual_seed(cfg["seed"])
            rng = torch.Generator(device=device).manual_seed(cfg["seed"])
            noise = (shared_noise(shape, rng, device, torch.bfloat16) if world == 2 else
                     torch.randn(shape, device=device, dtype=torch.bfloat16, generator=rng))
            for block in range(cfg["blocks"]):
                state.begin_block(block)
                for cache in pipeline.crossattn_cache_pos:
                    cache["is_init"] = False
                scheduler = pipeline._initialize_sample_scheduler(noise)
                configure_terminal(scheduler, generator.scheduler)
                start = block * block_frames
                offset = start * pipeline.frame_seq_length
                ref = resize_reference(references[block].to(device), shape[-2:]) if high else None

                def guide(clean, step):
                    if ref is None:
                        return clean
                    return fuse_clean(clean, ref, cfg["guidance_weights"][step], **cfg["gaussian"])

                current = sample_block(forward, scheduler, noise[:, start:start+block_frames], conditioning,
                                       pipeline.kv_cache_pos, offset, terminal=True,
                                       clean_hook=guide, phase_hook=state.set_phase)
                outputs.append(current.detach().cpu())
                state.set_phase("clean_kv")
                forward(current, 0, conditioning, pipeline.kv_cache_pos, offset)
                state.commit()
                if device.index == 0:
                    print(f"{'HR' if high else 'LR'} block {block + 1}/{cfg['blocks']}", flush=True)
        return outputs
    finally:
        pipeline.clear_cache()


def decode(blocks, vae, cfg, device, path):
    import torch
    from wan_5b.modules.vae2_2 import unpatchify

    vae.to(device=device, dtype=torch.bfloat16)
    model = vae.model
    if tuple(model.conv2.kernel_size) != (1, 1, 1):
        raise ValueError("Streaming decode requires the native 1×1×1 VAE convolution.")
    model.clear_cache()
    written = raw = 0
    try:
        with rgb_writer(path, cfg["output_size"], cfg["fps"]) as stream:
            for block in blocks:
                for latent in block.split(1, dim=1):
                    z = latent.to(device=device, dtype=torch.bfloat16).permute(0, 2, 1, 3, 4)
                    mean = vae.mean.to(z).view(1, 48, 1, 1, 1)
                    std = vae.std.to(z).view(1, 48, 1, 1, 1)
                    x = model.conv2(z / (1.0 / std) + mean)
                    model._conv_idx = [0]
                    pixels = model.decoder(x, feat_cache=model._feat_map, feat_idx=model._conv_idx, first_chunk=raw == 0)
                    pixels = unpatchify(pixels, patch_size=2).float().clamp(-1, 1)
                    frames = ((pixels * 0.5 + 0.5).clamp(0, 1) * 255).byte()[0].permute(1, 2, 3, 0).cpu()
                    for frame in frames:
                        if written < cfg["frames"]:
                            stream.write(frame.contiguous().numpy().tobytes())
                            written += 1
                        raw += 1
        if written != cfg["frames"] or raw != 4 * cfg["block_latents"] * cfg["blocks"] - 3:
            raise RuntimeError("Unexpected number of frames from the native Wan decoder.")
    finally:
        model.clear_cache()


def generate(cfg, weights, prompt, output, *, draft=None):
    import torch
    # The pinned upstream code imports its own top-level utils and wan_5b packages.
    sys.path.insert(0, str(NATIVE))
    from utils.wan_5b_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
    from utils.inference_utils import load_generator_checkpoint
    from .tensor_parallel import shard_generator, shared_conditioning

    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    with cuda_ranks((1, 2)) as (device, rank, world), working_directory(weights), torch.inference_mode():
        model = WanDiffusionWrapper(model_name="Wan2.2-TI2V-5B", timestep_shift=cfg["timestep_shift"],
                                    is_causal=True, local_attn_size=32, num_frame_per_block=cfg["block_latents"])
        load_generator_checkpoint(model, str(weights / "checkpoints/LongLive-2.0-5B/model_bf16.pt"), strict=True)
        model.to(dtype=torch.bfloat16).eval().requires_grad_(False)
        if world == 2:
            shard_generator(model)
        model.to(device)
        encoder = WanTextEncoder().to(dtype=torch.bfloat16).eval() if rank == 0 else torch.nn.Identity()
        conditioning = (shared_conditioning(encoder, prompt, device) if world == 2 else encoder(text_prompts=[prompt]))
        encoder.to("cpu")
        vae = WanVAEWrapper().to(dtype=torch.bfloat16).eval() if rank == 0 else torch.nn.Identity()
        low = run_branch(cfg, model, encoder, vae, conditioning, device, world)
        high = run_branch(cfg, model, encoder, vae, conditioning, device, world, low)
        del low, model
        gc.collect()
        torch.cuda.empty_cache()
        if rank == 0:
            with new_video(output) as path:
                decode(high, vae, cfg, device, path)
