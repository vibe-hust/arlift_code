# ARLIFT

[🌐 Project website](https://vibe-hust.github.io/arlift/) · [💻 Code repository](https://github.com/vibe-hust/arlift_code)

**High-resolution video generation with low-resolution draft guidance.** ARLIFT generates a low-resolution draft and uses its structure to guide a second pass at twice the width and height. This repository contains the inference code, model-specific settings, and scripts for downloading the pinned checkpoints.

| Backbone | Draft → output (width × height) | Output clip | Weight download |
| --- | --- | --- | --- |
| [Krea Realtime 14B](#krea-realtime-14b) | 832 × 480 → 1664 × 960 | 112 frames at 16 fps | ~41 GB |
| [LongLive 2.0 5B](#longlive-20-5b) | 1280 × 704 → 2560 × 1408 | 112 frames at 16 fps | ~44 GB |
| [MiniMax-H3 Ref2VA](#minimax-h3-ref2va) | 1344 × 768 → 2688 × 1536 | 124 frames at 24 fps, with audio from the draft | ~216 GB from text, ~146 GB with an existing draft |

<a id="getting-started"></a>

## 🚀 Getting started

```bash
git clone https://github.com/vibe-hust/arlift_code.git
cd arlift_code
```

Choose one backbone and run its commands from the repository root. The three default generation paths were tested end to end on Linux, each on a single node with two NVIDIA H100 GPUs (64 GiB class) and the listed checkpoints. The tests used prepared environments and verified the exported frame size, frame count, frame rate, and MiniMax audio track.

Krea and LongLive share a Python 3.11 environment and require FFmpeg with `libx264` on `PATH`. MiniMax uses a separate Python 3.12 environment. The examples use two visible GPUs on one machine.

<a id="krea-realtime-14b"></a>

## 🎬 Krea Realtime 14B

Set up the causal-model environment with a CUDA 12.8-compatible driver and the CUDA toolkit for building FlashAttention.

```bash
python3.11 -m venv .venv-ar
source .venv-ar/bin/activate
python -m pip install --upgrade pip setuptools wheel packaging ninja psutil
python -m pip install torch==2.8.0+cu128 torchvision==0.23.0+cu128 --index-url https://download.pytorch.org/whl/cu128
```

```bash
python -m pip install --no-build-isolation -r krea_14b/requirements.txt
python download_weights.py krea_14b
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 krea_14b/generate.py \
  --prompt-file examples/prompt.txt --output outputs/krea.mp4
```

Krea generates ten draft blocks of three latent frames each. The final clean estimate of each draft block guides the matching high-resolution block. Each branch maintains its own seven-slot LiteMem history. The defaults are in [the Krea configuration](krea_14b/inference.yaml).

<a id="longlive-20-5b"></a>

## 🎥 LongLive 2.0 5B

If starting with LongLive, run the first setup block under Krea to create `.venv-ar` and install PyTorch. Then install the LongLive dependencies and generate a clip.

```bash
source .venv-ar/bin/activate
python -m pip install --no-build-isolation -r longlive_5b/requirements.txt
python download_weights.py longlive_5b
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 longlive_5b/generate.py \
  --prompt-file examples/prompt.txt --output outputs/longlive.mp4
```

LongLive generates four blocks of eight latent frames. Each block uses four UniPC updates and a terminal clean prediction. Its guidance weights and seven-slot LiteMem settings are in [the LongLive configuration](longlive_5b/inference.yaml).

For a single-GPU run with either causal model, launch its `generate.py` with `CUDA_VISIBLE_DEVICES=0 python` in place of `CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2`.

<a id="minimax-h3-ref2va"></a>

## 🎞️ MiniMax-H3 Ref2VA

Use two visible GPUs on one machine, a CUDA 12.9-compatible driver, Git, and the system `patch` tool. The text encoder has a configured memory budget of 48 GiB per GPU. The transformer also uses host RAM for offloading. Apply and verify the two bundled Diffusers patches after installation.

```bash
python3.12 -m venv .venv-h3
source .venv-h3/bin/activate
python -m pip install --upgrade pip
python -m pip install -r minimax_h3_ref2va/requirements.txt
python minimax_h3_ref2va/tools/apply_diffusers_patches.py
python minimax_h3_ref2va/tools/apply_diffusers_patches.py --check
```

To generate a video from text, download the weights for both passes. The first pass generates a lossless Pika-H3-Flex RGB and audio draft. The second pass refines the video with the MiniMax-H3 Ref2VA transformer and HyperFlow adapter. The output carries the draft audio.

```bash
python download_weights.py minimax_h3_ref2va
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 minimax_h3_ref2va/generate.py \
  --prompt-file examples/prompt.txt --output outputs/minimax.mp4
```

To refine an existing draft, supply a **1344 × 768** video with **124 frames at 24 fps** and its matching text via `--prompt` or `--prompt-file`. If the draft was generated with a different seed, pass that seed with `--seed`. Audio is optional. The smaller download below contains just the components used for refinement.

```bash
python download_weights.py minimax_h3_ref2va --existing-draft
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 minimax_h3_ref2va/generate.py \
  --draft /path/to/draft.mp4 --prompt-file /path/to/matching_prompt.txt \
  --output outputs/minimax_from_draft.mp4
```

MiniMax applies spatial guidance in the first four of eight steps. Its noise schedule and guidance weights are in [the MiniMax configuration](minimax_h3_ref2va/inference.yaml).

<a id="writing-prompts"></a>

## ✍️ Writing prompts

Use a detailed prompt for a single continuous shot rather than a short caption. For English prompts, about 60 words in three sentences is a good starting point, as in the tested example below. This is a recommendation, not a minimum length. ARLIFT uses the low-resolution draft to guide the high-resolution pass, so describe one main action that fits the clip and make the subject, motion, and camera behavior explicit.

1. **Scene and subject.** Name the main subject, its setting, and the objects that matter in the frame.
2. **Action.** Describe what moves, how it moves, and which parts of the scene stay still.
3. **Camera and appearance.** Specify the framing, whether the camera is fixed or moving, and a few visible details such as lighting or materials.

The [bundled example](examples/prompt.txt) is the first prompt in the 24-prompt evaluation set used for Krea and LongLive:

> A woman sits beside a sunlit window in a modest living room with a sofa, a wooden table, and a leafy plant. She slowly turns her head toward the window while keeping her hands relaxed in her lap. The camera remains fixed in a medium shot, showing natural skin, cotton fabric, warm wood, and clear daylight in one continuous take.

Use `--prompt-file path/to/prompt.txt` for a UTF-8 text file containing one prompt, or pass text directly with `--prompt`. Text-to-video runs use the bundled example when neither option is given. The three configurations default to seed `610009`, which you can override with `--seed`. To compare two prompt versions, keep the backbone, generation settings, and seed fixed.

With this example, the default settings, and the two-H100 setup above, the published inference code reproduced the Krea and LongLive evaluation MP4s byte for byte. MiniMax also generated the example successfully from text.

<a id="weights-and-outputs"></a>

## 📦 Weights and outputs

- Checkpoints go under `weights/<backbone>/` by default. To use another disk, pass the same `--weights-dir /path/to/weights` to the download and generation commands. Model sources, revisions, and required files are in the [weight manifest](arlift/weights.json).
- Use `--list` with the download command to inspect sources before downloading. Use `--describe` on a generation command to validate and print its resolved configuration, or `--check-weights` to verify that the files and indexed shards exist. These checks run on CPU. For a refinement-only MiniMax installation, use `--check-weights` with `--draft /path/to/draft.mp4` and its matching prompt.
- `--output` selects the MP4 path. Existing files are protected. With `--config` you can pass a copy of a model's YAML. The dimensions, block lengths and schedules shipped here correspond to the paper settings.

<a id="attributions-and-licenses"></a>

## 📄 Attributions and licenses

See the [third-party notices](THIRD_PARTY_NOTICES.md) for source revisions, component-specific terms, and model licenses. Powered by MiniMax H3.
