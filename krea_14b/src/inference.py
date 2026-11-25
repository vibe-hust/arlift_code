"""Few-step causal inference with a fixed LR reference for each HR block."""
import gc
import json

from arlift.devices import cuda_ranks
from arlift.guidance import fuse_clean, resize_reference
from arlift.video import new_video, rgb_writer


def load_transformer(weights, device, rank, world):
    import torch
    from safetensors.torch import load_file
    from .wan.modules.causal_model import CausalWanModel
    from .wan.modules.model import rope_params
    from .tensor_parallel import TensorParallelFFN

    config = json.loads((weights / "krea/transformer/config.json").read_text())
    fields = ("model_type", "in_dim", "dim", "ffn_dim", "freq_dim", "out_dim",
              "num_heads", "num_layers", "text_len", "eps")
    with torch.device("meta"):
        model = CausalWanModel(**{k: config[k] for k in fields}, local_attn_size=-1, sink_size=0)
    d = model.dim // model.num_heads
    model.freqs = torch.cat([rope_params(1024, n) for n in (d - 4*(d//6), 2*(d//6), 2*(d//6))], dim=1)
    state = load_file(str(weights / "krea/krea-realtime-video-14b.safetensors"), device="cpu")
    # The official checkpoint stores the generator under the "model." prefix.
    model.load_state_dict({k.removeprefix("model."): v for k, v in state.items()}, strict=True, assign=True)
    del state
    for layer in model.blocks:
        if world == 2:
            layer.self_attn.enable_tensor_parallel(rank, world)
            layer.ffn = TensorParallelFFN(layer.ffn, rank, world)
    model = model.to(dtype=torch.bfloat16, device=device).eval().requires_grad_(False)
    if world == 1:
        for layer in model.blocks:
            layer.self_attn.fuse_projections()
    model.num_frame_per_block = 3
    return model


def encode_prompt(weights, prompt, device, rank, world):
    import torch
    import torch.distributed as dist
    from .text_encoder import WanTextEncoder

    if rank == 0:
        encoder = WanTextEncoder(weights).eval().to(dtype=torch.bfloat16)
        embedding = encoder(text_prompts=[prompt])["prompt_embeds"].to(device=device, dtype=torch.bfloat16).contiguous()
        del encoder
        torch.cuda.empty_cache()
    else:
        embedding = torch.empty((1, 512, 4096), device=device, dtype=torch.bfloat16)
    if world == 2:
        dist.broadcast(embedding, src=0)
    return embedding


def allocate_cache(model, token_count, memory, device):
    import torch
    from .memory import MemRoPEController

    kv, cross = [], []
    model.set_frame_seq_length(token_count)
    for block in model.blocks:
        layer = block.self_attn
        local_heads = layer.tp_num_heads if layer.tp_world_size > 1 else layer.num_heads
        dims = (1, memory.capacity_frames * token_count, local_heads, layer.head_dim)
        def zeros(shape):
            return torch.zeros(shape, dtype=torch.bfloat16, device=device)
        kv.append({"k": zeros(dims), "v": zeros(dims), "global_end_index": 0, "local_end_index": 0})
        cross_dims = (1, 512, model.num_heads, model.dim // model.num_heads)
        cross.append({"k": zeros(cross_dims), "v": zeros(cross_dims), "is_init": False})
    return kv, cross, MemRoPEController(kv, token_count, memory)


def run_branch(model, scheduler, embedding, cfg, device, rank, world, references=None):
    import torch
    import torch.distributed as dist
    from .memory import MemRoPEConfig

    high = references is not None
    width, height = cfg["output_size" if high else "native_size"]
    frames = cfg["block_latents"]
    latent_hw = (height // 8, width // 8)
    tokens = (height // 16) * (width // 16)
    memory = MemRoPEConfig(**cfg["memory"], current_frames=frames, online_rope=True)
    kv, cross, history = allocate_cache(model, tokens, memory, device)
    stages = torch.tensor(cfg["timesteps"], dtype=torch.float32, device=device)
    model_steps = scheduler.timesteps.to(device=device, dtype=torch.float64)
    model_sigmas = scheduler.sigmas.to(device=device, dtype=torch.float64)
    rng = torch.Generator(device=device)
    blocks = []

    def draw(shape):
        noise = (torch.randn(shape, dtype=torch.bfloat16, device=device, generator=rng)
                 if rank == 0 else torch.empty(shape, dtype=torch.bfloat16, device=device))
        if world > 1:
            dist.broadcast(noise, src=0)
        return noise

    def predict(noisy, t):
        times = torch.ones((1, frames), device=device, dtype=torch.int64) * t
        flow = model(noisy.permute(0, 2, 1, 3, 4), t=times, context=embedding,
                     seq_len=21 * tokens, kv_cache=kv, crossattn_cache=cross,
                     current_start=history.history_frames * tokens, cache_start=None).permute(0, 2, 1, 3, 4)
        # The native Wan wrapper evaluates the flow-to-clean conversion in FP64.
        idx = (model_steps.unsqueeze(0) - times.flatten().unsqueeze(1)).abs().argmin(dim=1)
        sigma = model_sigmas[idx].view(1, frames, 1, 1, 1)
        return (noisy.double() - sigma * flow.double()).to(flow.dtype)

    for block in range(cfg["blocks"]):
        rng.manual_seed(cfg["seed"] + block * 1000003)
        noisy = draw((1, frames, 16, *latent_hw))
        reference = resize_reference(references[block].to(device), latent_hw) if high else None
        history.begin("denoise")
        for i, stage in enumerate(stages):
            clean = predict(noisy, stage)
            if high:
                clean = fuse_clean(clean, reference, cfg["guidance_weights"][i], **cfg["gaussian"])
            if i + 1 < len(stages):
                flat = clean.flatten(0, 1)
                next_t = stages[i + 1] * torch.ones(frames, device=device, dtype=torch.long)
                noisy = scheduler.add_noise(flat, draw(flat.shape), next_t).unflatten(0, clean.shape[:2])
        history.begin("clean_commit")
        predict(clean.detach(), 0)
        history.commit()
        blocks.append(clean.detach().cpu())
        if rank == 0:
            print(f"{'HR' if high else 'LR'} block {block + 1}/{cfg['blocks']}", flush=True)
    return blocks


def decode(blocks, weights, cfg, device, path):
    import torch
    from .decoder import VAEDecoderWrapper
    from .vae_tiling import SpatialTiledVAEDecoder

    decoder = VAEDecoderWrapper()
    state = torch.load(weights / "Wan2.1-T2V-1.3B/Wan2.1_VAE.pth", map_location="cpu", weights_only=True)
    keys = {k: v for k, v in state.items() if "decoder." in k or "conv2" in k}
    missing, unexpected = decoder.load_state_dict(keys, strict=False)
    if set(missing) - {"mean", "std"} or unexpected:
        raise ValueError(f"Unexpected Wan decoder weights: {missing}, {unexpected}")
    del state, keys
    decoder.eval().to(device=device, dtype=torch.float16)
    tiles = SpatialTiledVAEDecoder(cfg["vae_tile"]["size"], cfg["vae_tile"]["overlap"], 55)
    written = 0
    with rgb_writer(path, cfg["output_size"], cfg["fps"]) as stream, torch.compiler.set_stance("force_eager"):
        for i, block in enumerate(blocks):
            pixels = tiles.decode_block(decoder, block.to(device=device, dtype=torch.float16))[0]
            if i == 0:
                pixels = pixels[3:]   # Native leading display trim.
            for frame in pixels:
                if written == cfg["frames"]:
                    break
                rgb = frame.cpu().add(1).mul(127.5).clamp(0, 255).byte().permute(1, 2, 0)
                stream.write(rgb.contiguous().numpy().tobytes())
                written += 1
        if written != cfg["frames"]:
            raise RuntimeError(f"Expected {cfg['frames']} frames, decoded {written}.")


def generate(cfg, weights, prompt, output, *, draft=None):
    import torch
    from .scheduler import FlowMatchScheduler

    with cuda_ranks((1, 2)) as (device, rank, world), torch.inference_mode():
        from .wan.modules.attention import FLASH_ATTN_2_AVAILABLE
        if not FLASH_ATTN_2_AVAILABLE:
            raise RuntimeError("Install FlashAttention 2 before running Krea.")
        generator = load_transformer(weights, device, rank, world)
        embedding = encode_prompt(weights, prompt, device, rank, world)
        scheduler = FlowMatchScheduler(shift=cfg["timestep_shift"], sigma_min=0.0, extra_one_step=True)
        scheduler.set_timesteps(1000, training=True)
        references = run_branch(generator, scheduler, embedding, cfg, device, rank, world)
        blocks = run_branch(generator, scheduler, embedding, cfg, device, rank, world, references)
        del generator, references
        gc.collect()
        torch.cuda.empty_cache()
        if rank == 0:
            with new_video(output) as path:
                decode(blocks, weights, cfg, device, path)
