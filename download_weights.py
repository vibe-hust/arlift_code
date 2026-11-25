"""Download a backend's pinned public checkpoints to a local directory."""
import argparse
from pathlib import Path
from arlift.weights import SOURCES, check_weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backbone", choices=sorted(SOURCES))
    parser.add_argument("--weights-dir", type=Path)
    parser.add_argument("--existing-draft", action="store_true", help="Skip Pika draft-generation weights for MiniMax.")
    parser.add_argument("--list", action="store_true", help="List sources and destinations without downloading.")
    args = parser.parse_args()
    if args.existing_draft and args.backbone != "minimax_h3_ref2va":
        parser.error("--existing-draft applies to MiniMax only.")
    root = (args.weights_dir or Path(__file__).resolve().parent / "weights" / args.backbone).expanduser().resolve()
    for src in SOURCES[args.backbone]:
        if src.get("generate_draft") and args.existing_draft:
            continue
        folder = root / src["directory"]
        print(f"{src['repository']} @ {src['revision']} -> {folder}", flush=True)
        if not args.list:
            from huggingface_hub import snapshot_download
            snapshot_download(src["repository"], revision=src["revision"],
                              allow_patterns=src["files"] + ["LICENSE*", "NOTICE*", "licenses/*"], local_dir=folder)
    if not args.list:
        check_weights(args.backbone, root, with_draft=not args.existing_draft)


if __name__ == "__main__":
    main()
