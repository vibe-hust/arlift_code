# Third-party notices

## Krea, Wan and LongLive

- Krea Realtime Video is distributed under CC BY-NC-SA 4.0. The Krea backend adapts the causal generator, sampler and tiled decoder for ARLIFT. The shared spatial-guidance implementation and the LongLive branch-memory adaptation also draw on the Krea integration. [Source revision](https://github.com/krea-ai/realtime-video/tree/acf4b7b4e1a3049a0e04782e751d12203334038d) · [License](licenses/Krea-Realtime.md)
- The Wan transformer, text-encoder, tokenizer and VAE components retain their upstream copyright notices and Apache 2.0 terms. [Wan 2.1](https://github.com/Wan-Video/Wan2.1) · [Wan 2.2](https://github.com/Wan-Video/Wan2.2) · [License](licenses/Wan-Apache-2.0.txt)
- The LongLive backend includes the native inference modules from NVlabs/LongLive. ARLIFT adds blockwise reference guidance, branch-local attention memory and a command-line entry point for the merged 5B checkpoint. LongLive publishes its code under Apache 2.0 and its weights under the NVIDIA Open Model License. [Source revision](https://github.com/NVlabs/LongLive/tree/242635807b4ebb89fa2609a84df7eacde6aed92f) · [Code license](licenses/LongLive.txt) · [Model](https://huggingface.co/Efficient-Large-Model/LongLive-2.0-5B)
- LiteMem uses fixed-rate memory summaries and online rotary position encoding adapted to each causal backbone. The underlying memory mechanism follows MemRoPE. [MemRoPE source](https://github.com/YoungRaeKimm/MemRoPE/tree/b89bf5b87ca84efa2e40e1c7cc43a44fe88ab865)

## MiniMax-H3, Pika-H3-Flex and HyperFlow

- MiniMax H3 is licensed under the MiniMax H3 Community License Agreement, Copyright © 2026 MiniMax. All Rights Reserved. The model and its derivatives have their own terms. [Base model](https://huggingface.co/MiniMaxAI/MiniMax-H3) · [License](licenses/Pika-H3-Flex/licenses/MiniMax-H3-LICENSE.txt)
- Pika-H3-Flex supplies the draft sampler, loader, FL2VA adapter and Diffusers patches. The adapted entry point accepts an explicit prompt and writes a lossless video draft. Its inference code and patches carry the upstream Apache 2.0 notices. [Pika source](https://github.com/Pika-Labs/Pika-H3-Flex/tree/6dacb6250e60b006d1df9efdaf04f0a54c1a6755) · [Code license](licenses/Pika-H3-Flex/LICENSE-CODE) · [Third-party notices](licenses/Pika-H3-Flex/THIRD_PARTY_NOTICES.md)
- The high-resolution pass uses Ref2VA encoding, context parallelism, layout, HyperFlow and media modules from Open-H3-Regenerate-2K-IC. ARLIFT adds early-step spatial guidance. The eight-step HyperFlow adapter is by the HyperFlow authors. The upstream notice credits the model and encoder authors and describes component-specific terms. [Source revision](https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K-IC/tree/54a7104b204fb46486cf85c2f600942118186d57) · [Notice](licenses/Open-H3-Regenerate/NOTICE) · [Model license](licenses/Open-H3-Regenerate/LICENSE) · [Qwen license](licenses/Open-H3-Regenerate/LICENSE-Qwen.txt)

The weight downloader retrieves each checkpoint from its original model repository at the revision in the [weight manifest](arlift/weights.json). Powered by MiniMax H3.
