"""Command-line interface shared by the three inference backends."""
import argparse
import importlib
import json
from pathlib import Path

from .config import validate_config
from .weights import check_weights

ROOT = Path(__file__).resolve().parents[1]


def main(backend):
    name = backend.name
    parser = argparse.ArgumentParser(description=f"Generate a 2× video using ARLIFT on {name}.")
    prompts = parser.add_mutually_exclusive_group()
    prompts.add_argument("--prompt", help="Text description of the video.")
    prompts.add_argument("--prompt-file", type=Path, help="UTF-8 text file containing one prompt.")
    parser.add_argument("--config", type=Path, default=backend / "inference.yaml")
    parser.add_argument("--weights-dir", type=Path, default=ROOT / "weights" / name)
    parser.add_argument("--output", type=Path, default=Path("outputs") / f"{name}.mp4")
    parser.add_argument("--seed", type=int, help="Override the seed in the configuration.")
    if name == "minimax_h3_ref2va":
        parser.add_argument("--draft", type=Path, help="Reuse a 1344×768, 124-frame, 24 fps LR draft.")
    else:
        parser.set_defaults(draft=None)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--describe", action="store_true", help="Print the resolved settings without loading weights.")
    action.add_argument("--check-weights", action="store_true", help="Verify the local weight layout without using a GPU.")
    args = parser.parse_args()
    if args.draft is not None and args.prompt is None and args.prompt_file is None:
        parser.error("--draft requires --prompt or --prompt-file with the matching text prompt.")
    try:
        import yaml
    except ImportError:
        parser.error("Install PyYAML to load inference.yaml.")
    try:
        cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        if args.seed is not None and isinstance(cfg, dict):
            cfg["seed"] = args.seed
        validate_config(name, cfg)
        prompt = args.prompt
        if args.prompt_file is not None:
            prompt = args.prompt_file.read_text(encoding="utf-8")
        if prompt is None:
            prompt = (ROOT / "examples/prompt.txt").read_text(encoding="utf-8")
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("The prompt must contain text.")
        weights = args.weights_dir.expanduser().resolve()
        output = args.output.expanduser().resolve()
        draft = args.draft.expanduser().resolve() if args.draft else None
        if output.suffix.lower() != ".mp4":
            raise ValueError("--output must name an MP4 file.")
        if args.describe:
            print(json.dumps({"backbone": name, "settings": cfg, "prompt": prompt,
                              "weights_dir": str(weights), "output": str(output),
                              **({"draft": str(draft) if draft else "generate from prompt"}
                                 if name == "minimax_h3_ref2va" else {})}, indent=2, ensure_ascii=False))
            return
        if draft is not None and not draft.is_file():
            raise ValueError(f"Draft file not found: {draft}")
        check_weights(name, weights, with_draft=draft is None)
        if args.check_weights:
            print(f"Weight files are present for {name}.")
            return
        if output.exists():
            raise FileExistsError(f"Choose a new output path: {output}")
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    module = importlib.import_module(f"{name}.src.inference")
    module.generate(cfg, weights, prompt, output, draft=draft)
