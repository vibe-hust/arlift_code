# SPDX-License-Identifier: Apache-2.0
# Copyright © 2026 Mellis, Inc.
"""Validate the local FL2VA adapter used for draft generation."""
import hashlib
import json
from pathlib import Path

BASE_REPO = "MiniMaxAI/MiniMax-H3"
BASE_REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"
KNOTS = [1000, 750, 500, 250, 100, 40, 0]


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_assets(base, adapter_root):
    manifest = json.loads((adapter_root / "manifest.json").read_text())
    if manifest.get("format_version") != 1 or manifest.get("rank") != 128 or manifest.get("alpha") != 128:
        raise ValueError("unsupported adapter format/rank/alpha")
    if manifest.get("numerics_policy") != "bf16_conditioning_v1":
        raise ValueError("adapter numerical policy does not match this release")
    adapter = adapter_root / "adapter.safetensors"
    # Validate the exported release contract before loading any models.
    if manifest["trunk"] != "transformer":
        raise ValueError("adapter trunk does not match task")
    if manifest["sampling"]["knots"] != KNOTS or manifest["sampling"]["video_shift"] != 12 or manifest["sampling"]["audio_shift"] != 3 or manifest["sampling"]["method"] != "renoise":
        raise ValueError("unsupported adapter sampling contract")
    if file_sha256(adapter) != manifest["sha256"]:
        raise ValueError("adapter checksum mismatch")
    if manifest["base"] != {"repo_id": BASE_REPO, "revision": BASE_REVISION}:
        raise ValueError("adapter base identity mismatch")
    return Path(base), adapter, manifest
