"""Pinned model-file layouts. Downloading is a separate, explicit command."""
import json
from pathlib import Path

SOURCES = json.loads(Path(__file__).with_name("weights.json").read_text())


def check_weights(backend, root, with_draft=True):
    missing = []
    for src in SOURCES[backend]:
        if src.get("generate_draft") and not with_draft:
            continue
        folder = root / src["directory"]
        indices = set()
        for pattern in src["required"]:
            matches = [p for p in folder.glob(pattern) if p.is_file() and p.stat().st_size]
            if not matches:
                missing.append(str(folder / pattern))
            for file in matches:
                indices.update(file.parent.glob("*.safetensors.index.json"))
        for index in sorted(indices):
            for shard in sorted(set(json.loads(index.read_text())["weight_map"].values())):
                p = index.parent / shard
                if not p.is_file() or not p.stat().st_size:
                    missing.append(str(p))
    if missing:
        raise FileNotFoundError("Missing model files:\n  " + "\n  ".join(missing))
