"""Bidirectional Ref2VA refinement on a Pika draft with eight HyperFlow steps."""
import gc
import json
import os
from contextlib import ExitStack
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from arlift.devices import cuda_ranks
from arlift.video import new_video


def validate_draft(path, cfg):
    import av

    with av.open(str(path)) as video:
        if len(video.streams.video) != 1:
            raise ValueError("Expected one video stream in the draft.")
        stream = video.streams.video[0]
        if ([stream.width, stream.height] != cfg["native_size"]
                or stream.average_rate != cfg["fps"]
                or (stream.frames and stream.frames != cfg["frames"])):
            raise ValueError("Expected a 1344×768 draft with 124 frames at 24 fps.")
        if not stream.frames and sum(1 for _ in video.decode(stream)) != cfg["frames"]:
            raise ValueError("The draft must have exactly 124 frames.")
        return bool(video.streams.audio)


def encode_draft(draft, prompt_text, cfg, base, device):
    import torch
    from .text_encoder import load_dual_gpu_encoder
    from ref2va_ic import encode, media

    width, height = cfg["native_size"]
    pixels, count, fps = media.read_draft(str(draft), height, width)
    if (count, fps) != (cfg["frames"], cfg["fps"]):
        raise ValueError("Expected 124 frames at 24 fps.")
    vae = encode.load_vae(base, device)
    latent = encode.vae_encode(vae, pixels, cfg["seed"])
    del vae
    torch.cuda.empty_cache()
    text = load_dual_gpu_encoder(base, device)
    prompt = encode.encode_prompt(text, pixels, prompt_text)
    text.update_components(text_encoder=None)
    del text
    gc.collect()
    torch.cuda.empty_cache()
    return latent, prompt


def load_denoiser(base, adapter, device, cfg):
    import torch
    from diffusers import MiniMaxH3Transformer3DModel
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard
    from ref2va_ic import cp, hyperflow

    model = MiniMaxH3Transformer3DModel.from_pretrained(
        str(base), subfolder="transformer_ref", torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, local_files_only=True,
    ).requires_grad_(False)
    metadata = hyperflow.install(model, adapter, device)
    if json.loads(metadata["hyperflow_sigmas"]) != cfg["raw_sigmas"]:
        raise ValueError("The loaded HyperFlow weights do not contain the agreed hf8 noise grid.")
    sigmas = hyperflow.sigmas(metadata, cfg["video_shift"])
    model.set_attention_backend("native")
    mesh = init_device_mesh("cuda", (1, 2), mesh_dim_names=("replicate", "shard"))
    offload = CPUOffloadPolicy(pin_memory=False)
    block_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=False,
    )
    for block in model.transformer_blocks:
        fully_shard(block, mesh=mesh, mp_policy=block_policy, offload_policy=offload)
    root_policy = MixedPrecisionPolicy(reduce_dtype=torch.float32, cast_forward_inputs=False)
    fully_shard(model, mesh=mesh, mp_policy=root_policy, offload_policy=offload)
    model.eval()
    cp.enable_cp(model, 2, "native")
    return model, sigmas


def sample(model, sigmas, latent, prompt, silence, seed, target_shape, cfg, device, primary):
    import torch
    from .guidance import GaussianVideoGuide
    from ref2va_ic import cp, hyperflow, layout as lay

    with torch.inference_mode():
        ref, audio, text, layout, n = lay.pack(latent, target_shape, prompt, silence, device)
        reference = latent.permute(1, 0, 2, 3)[None].to(device=device, dtype=torch.float32)
        guide = GaussianVideoGuide(reference, target_shape[-2:], cfg["gaussian"])
        channels, frames, height, width = target_shape
        generator = torch.Generator(device=device).manual_seed(seed % (1 << 62))
        anchor = 0.999 * ref.float() + 0.001 * torch.randn(ref.shape, generator=generator, device=device)
        target = torch.randn(
            (1, frames * (height // 2) * (width // 2), channels * 4),
            generator=generator, device=device, dtype=torch.float32,
        )
        for step, (sigma, next_sigma, weight) in enumerate(zip(sigmas[:-1], sigmas[1:], cfg["guidance_weights"], strict=True), 1):
            times, _ = lay.row_timesteps(layout, sigma)
            model.time_embedder.endpoint_timesteps = hyperflow.endpoints(times, lay.canonical_sigma(sigma), next_sigma)
            with torch.autocast("cuda", enabled=False):
                velocity = cp.forward(model, torch.cat([anchor, target], 1), audio, text, layout, sigma, 2)[:, n:].float()
            if weight:
                # HyperFlow predicts data-ward velocity, so x0 = x_sigma + sigma * v.
                clean = lay.unpatch_rows(target + sigma * velocity, target_shape).permute(1, 0, 2, 3)[None]
                correction = guide.fuse(clean, weight) - clean
                velocity = velocity + lay.patch_rows(correction[0].permute(1, 0, 2, 3).contiguous()) / sigma
                del clean, correction
            target = target + (sigma - next_sigma) * velocity
            del velocity
            if primary:
                print(f"HF8 step {step}/8 complete, guidance={weight:.12g}", flush=True)
        if not torch.isfinite(target).all():
            raise RuntimeError("Ref2VA returned non-finite target latents.")
        return lay.unpatch_rows(target.cpu(), target_shape).bfloat16() if primary else None


def generate(cfg, weights, prompt, output, *, draft=None):
    import av
    import torch
    import torch.distributed as dist
    from safetensors.torch import load_file

    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    sys.path.insert(0, str(Path(__file__).resolve().parent / "upstream"))
    from pika_h3.runtime import validate_runtime
    from ref2va_ic import encode, media
    from .draft import generate_draft

    validate_runtime()
    base = weights / "MiniMax-H3"
    hyperflow = weights / "HyperFlow/lora/minimax_h3_hyperflow_8step_v1.0.safetensors"
    silence_file = weights / "HyperFlow/assets/silence_audio_124f.safetensors"
    with cuda_ranks((2,)) as (device, rank, _), ExitStack() as stack:
        if torch.cuda.device_count() != 2:
            raise RuntimeError("Expose exactly two GPUs with CUDA_VISIBLE_DEVICES for the shared text encoder.")
        if draft is None:
            location = [None]
            if rank == 0:
                location[0] = str(Path(stack.enter_context(TemporaryDirectory())) / "draft.mkv")
            dist.broadcast_object_list(location, src=0, device=device)
            draft = Path(location[0])
            generate_draft(cfg, weights, prompt, draft, device, rank)
            gc.collect()
            torch.cuda.empty_cache()
        payload = [None]
        if rank == 0:
            has_audio = validate_draft(draft, cfg)
            payload[0] = encode_draft(draft, prompt, cfg, base, device)
        dist.broadcast_object_list(payload, src=0, device=device)
        latent, conditioning = payload.pop()
        model, sigmas = load_denoiser(base, hyperflow, device, cfg)
        silence = load_file(str(silence_file))["audio_latents"]
        width, height = cfg["output_size"]
        shape = [latent.shape[0], latent.shape[1], height // 16, width // 16]
        refined = sample(model, sigmas, latent, conditioning, silence, cfg["seed"],
                         shape, cfg, device, rank == 0)
        del model, latent, conditioning
        gc.collect()
        torch.cuda.empty_cache()
        if rank == 0:
            pixels = encode.vae_decode(encode.load_vae(base, device), refined)
            if pixels.shape != (cfg["frames"], height, width, 3):
                raise RuntimeError(f"Unexpected decoded video shape: {pixels.shape}.")
            with new_video(output) as path:
                media.write_video(pixels, path, audio_from=draft if has_audio else None)
                with av.open(str(path)) as result:
                    stream = result.streams.video[0]
                    if ((stream.width, stream.height, stream.frames, stream.average_rate)
                            != (width, height, cfg["frames"], cfg["fps"])
                            or bool(result.streams.audio) != has_audio):
                        raise RuntimeError("The exported video does not match the expected format.")
        dist.barrier()
