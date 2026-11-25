# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0
"""Load the merged LongLive generator checkpoint."""
from __future__ import annotations

import torch
from utils.nvfp4_checkpoint import clean_fsdp_state_dict_keys, unwrap_generator_state_dict


def _torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_generator_checkpoint(generator, checkpoint_path: str, *, use_ema: bool = False, strict: bool | None = None):
    """Load a LongLive generator checkpoint into ``generator``."""
    checkpoint = _torch_load(checkpoint_path)
    state_dict = unwrap_generator_state_dict(checkpoint, use_ema=use_ema)
    if use_ema:
        state_dict = clean_fsdp_state_dict_keys(state_dict)
    if strict is None:
        strict = not use_ema
    return generator.load_state_dict(state_dict, strict=strict)
