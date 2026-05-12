#!/usr/bin/env python3
"""Generate shared inputs for the kernel-count diff bench.

NumPy seed 42; F32 normal; cast inside torch so BF16 bytes match what
PyTorch produces (matches the matched_config bench convention).

Outputs: ./inputs.safetensors with the following keys.

| key       | shape                | dtype |
|-----------|----------------------|-------|
| x         | [1, 4096, 2560]      | bf16  |
| y         | [1, 4096, 2560]      | bf16  |
| w_norm1   | [2560]               | bf16  |
| w_norm2   | [2560]               | bf16  |
| w_qkv     | [7680, 2560]         | bf16  |
| w_o       | [2560, 2560]         | bf16  |
| w_mlp1    | [10240, 2560]        | bf16  |
| w_mlp2    | [2560, 10240]        | bf16  |
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file


SEED = 42
B, SEQ, HIDDEN = 1, 4096, 2560
HEADS, HEAD_DIM = 20, 128
MLP = 4 * HIDDEN  # 10240

OUT_PATH = Path(__file__).parent / "inputs.safetensors"


def randn(shape, rng):
    arr = rng.standard_normal(size=shape, dtype=np.float32)
    # Small scale for weights to keep activations sane.
    return torch.from_numpy(arr).to(torch.bfloat16).contiguous()


def randn_scaled(shape, rng, scale):
    arr = rng.standard_normal(size=shape, dtype=np.float32) * scale
    return torch.from_numpy(arr).to(torch.bfloat16).contiguous()


def main():
    rng = np.random.default_rng(SEED)
    out: dict[str, torch.Tensor] = {}

    # Inputs (activations) — N(0,1) BF16.
    out["x"] = randn((B, SEQ, HIDDEN), rng)
    out["y"] = randn((B, SEQ, HIDDEN), rng)

    # Norm weights — start near 1 so rmsnorm doesn't blow up the activation
    # magnitude. We use 1 + small noise; matches typical init.
    n1 = rng.standard_normal(size=(HIDDEN,), dtype=np.float32) * 0.02 + 1.0
    n2 = rng.standard_normal(size=(HIDDEN,), dtype=np.float32) * 0.02 + 1.0
    out["w_norm1"] = torch.from_numpy(n1).to(torch.bfloat16).contiguous()
    out["w_norm2"] = torch.from_numpy(n2).to(torch.bfloat16).contiguous()

    # Linears — scale ~ 1/sqrt(fan_in), matches default He/Kaiming.
    out["w_qkv"]  = randn_scaled((3 * HIDDEN, HIDDEN), rng, 1.0 / (HIDDEN ** 0.5))
    out["w_o"]    = randn_scaled((HIDDEN, HIDDEN),     rng, 1.0 / (HIDDEN ** 0.5))
    out["w_mlp1"] = randn_scaled((MLP, HIDDEN),        rng, 1.0 / (HIDDEN ** 0.5))
    out["w_mlp2"] = randn_scaled((HIDDEN, MLP),        rng, 1.0 / (MLP ** 0.5))

    save_file(out, str(OUT_PATH), metadata={"seed": str(SEED)})
    size_mb = OUT_PATH.stat().st_size / 1e6
    print(f"Wrote {OUT_PATH} ({size_mb:.1f} MB) with {len(out)} tensors.")


if __name__ == "__main__":
    main()
