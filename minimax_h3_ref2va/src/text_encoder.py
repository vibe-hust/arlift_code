"""Load the shared H3 text encoder across the two GPUs in one node."""


def load_sharded_text_encoder(checkpoint):
    import torch
    from transformers import Qwen3VLForConditionalGeneration

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        checkpoint,
        dtype=torch.bfloat16,
        device_map="balanced",
        max_memory={0: "48GiB", 1: "48GiB"},
        local_files_only=True,
    ).eval()
    if set(model.hf_device_map.values()) != {0, 1}:
        raise RuntimeError(f"Expected Qwen3-VL on both local GPUs, got {model.hf_device_map}.")
    print("[cond] Qwen3-VL loaded across CUDA devices 0 and 1", flush=True)
    return model


def load_dual_gpu_encoder(model_dir, _device):
    from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3Ref2VATextEncoderStep

    encoder = MiniMaxH3Ref2VATextEncoderStep().init_pipeline(str(model_dir))
    encoder.load_components(names=["tokenizer", "processor"], pretrained_model_name_or_path=str(model_dir))
    encoder.update_components(text_encoder=load_sharded_text_encoder(model_dir / "text_encoder"))
    return encoder
