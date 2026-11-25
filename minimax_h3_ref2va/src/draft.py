"""Generate the lossless Pika-H3-Flex draft used to condition Ref2VA."""
from fractions import Fraction
import gc


def write_draft(path, cfg, pixels, wave):
    import av
    import numpy as np

    width, height = cfg["native_size"]
    frames, fps = cfg["frames"], cfg["fps"]
    rate = cfg["audio_rate"]
    samples = round(frames * rate / fps)
    if pixels.shape != (frames, height, width, 3) or pixels.dtype != np.uint8:
        raise ValueError("Unexpected Pika LR RGB frame format.")
    if wave.ndim != 2 or wave.shape[0] != 2 or wave.shape[1] < samples:
        raise ValueError("Pika LR draft needs stereo audio covering all frames.")
    if not np.isfinite(wave[:, :samples]).all():
        raise ValueError("Pika LR draft audio is not finite.")

    with av.open(str(path), mode="w", format="matroska") as container:
        video = container.add_stream("ffv1", rate=fps)
        video.width, video.height, video.pix_fmt = width, height, "bgr0"
        audio = container.add_stream("pcm_f32le", rate=rate)
        audio.layout = "stereo"
        # Mux in time order. FFV1 encodes RGB losslessly and PCM keeps the float waveform.
        audio_start = 0
        for index, rgb in enumerate(pixels):
            container.mux(video.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")))
            audio_end = round((index + 1) * rate / fps)
            pcm = np.ascontiguousarray(wave[:, audio_start:audio_end].T, dtype=np.float32).reshape(1, -1)
            audio_frame = av.AudioFrame.from_ndarray(pcm, format="flt", layout="stereo")
            audio_frame.pts = audio_start
            audio_frame.sample_rate = rate
            audio_frame.time_base = Fraction(1, rate)
            container.mux(audio.encode(audio_frame))
            audio_start = audio_end
        container.mux(video.encode(None))
        container.mux(audio.encode(None))


def generate_draft(cfg, weights, prompt, path, device, rank):
    import torch
    import torch.distributed as dist
    from diffusers.models._modeling_parallel import ContextParallelConfig
    from pika_h3.assets import resolve_assets
    from pika_h3.core import geometry as G
    from pika_h3.core.conditioning import H3ConditionEncoder, Request
    from pika_h3.core.sampler import sample_draft_latents
    from pika_h3.core.loader import apply_lora, build_transformer, load_lora_weights
    from pika_h3.core.numerics import configure_numerics_backend, prepare_conditioning
    from pika_h3.core.schedules import H3SigmaSchedule
    from pika_h3.core.vae import H3Decoders
    from pika_h3.core.wrapper import H3DiffusionWrapper
    from .text_encoder import load_sharded_text_encoder

    assets = [None]
    if rank == 0:
        assets[0] = resolve_assets(weights / "MiniMax-H3", weights / "Pika-H3-Flex/fl2va")
    dist.broadcast_object_list(assets, src=0, device=device)
    base, adapter, manifest = assets[0]
    if manifest["sha256"] != "8ed6115d647ab11c1b38d04e63c4c6e1594128ae312f7b5635cd50563b7b9327":
        raise ValueError("The FL2VA adapter differs from the pinned Pika release.")
    policy = manifest["numerics_policy"]
    configure_numerics_backend(policy, runtime="inference")
    schedule = H3SigmaSchedule({"denoising_step_list": manifest["sampling"]["knots"],
                                "video_score_sigma_shift": manifest["sampling"]["video_shift"],
                                "audio_score_sigma_shift": manifest["sampling"]["audio_shift"]}, device=device)

    class ShardedEncoder(H3ConditionEncoder):
        def _load_text_encoder(self):
            if self._te is None:
                from transformers import AutoProcessor, AutoTokenizer
                root = self.text_encoder_checkpoint
                self._te = (AutoTokenizer.from_pretrained(root / "tokenizer"),
                            AutoProcessor.from_pretrained(root / "processor"),
                            load_sharded_text_encoder(root / "text_encoder"))
            return self._te

    payload = [None]
    if rank == 0:
        encoder = ShardedEncoder(str(base), device=device, keep_loaded=True)
        try:
            width, height = cfg["native_size"]
            payload[0] = encoder.encode(Request(task="t2va", prompt=prompt, width=width, height=height,
                                                num_frames=cfg["frames"], reference_rule="tiered", keyframe_fit="crop"))
        finally:
            encoder.free()
            del encoder
    dist.broadcast_object_list(payload, src=0, device=device)
    conditioning = payload.pop()
    gc.collect()
    torch.cuda.empty_cache()
    model = build_transformer(str(base), dtype=torch.bfloat16, device="cpu", subfolder=manifest["trunk"])
    apply_lora(model, rank=manifest["rank"], alpha=manifest["alpha"],
               target_modules=manifest["target_modules"], master_dtype=torch.bfloat16)
    load_lora_weights(model, str(adapter))
    model.requires_grad_(False).eval()
    model.set_attention_backend("native")
    model.enable_parallelism(config=ContextParallelConfig(ulysses_degree=2, ulysses_anything=True))
    for module in model.token_refiner.modules():
        processor = getattr(module, "processor", None)
        if hasattr(processor, "_parallel_config"):
            processor._parallel_config = None
    model.enable_group_offload(onload_device=device, offload_device=torch.device("cpu"), offload_type="block_level",
                               num_blocks_per_group=1, use_stream=True, record_stream=False, low_cpu_mem_usage=True)
    wrapper = H3DiffusionWrapper(model).eval()
    geom = conditioning["geometry"]
    with torch.no_grad():
        video, audio = sample_draft_latents(
            wrapper, conditional_dict=prepare_conditioning(conditioning, device, policy=policy),
            video_shape=(1, geom.num_latent_frames, G.VIDEO_LATENT_CHANNELS, geom.latent_height, geom.latent_width),
            audio_shape=(1, geom.num_audio_latents, G.AUDIO_EXTERNAL_DIM),
            sigmas_v=schedule.video_ladder, sigmas_a=schedule.audio_ladder, seed=cfg["seed"], device=device,
        )
    if rank == 0:
        draft_video, draft_audio = video[0].cpu(), audio[0].cpu()
    del video, audio, model, wrapper, conditioning
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()
    if rank == 0:
        with torch.inference_mode():
            decoders = H3Decoders(str(base), device=device)
            write_draft(path, cfg, decoders.decode_video(draft_video), decoders.decode_audio(draft_audio))
        del decoders
    dist.barrier()
