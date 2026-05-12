#!/usr/bin/env python3
"""Generate identical input tensors for the matched-config benchmark.

Both flame-core (Rust) and PyTorch (Python) sides load this exact file. No
side regenerates inputs — they read the same bytes.

We seed NumPy at 42, generate F32 normal, then cast to the target dtype
inside PyTorch (so we get correct BF16 rounding rather than a uint16
bit-twiddle which would differ from torch's internal cast). Saved via
safetensors.

Shapes covered:
- Elementwise / cast / reduction:
    small:  [1, 64,   2560]
    hot:    [1, 4096, 2560]   <- zimage hot training shape
    large:  [2, 4096, 2560]
- Norms:
    [1, 4096, 2560]  (zimage)
    [1, 4096, 4096]  (klein-block-ish)
- GEMM:
    square:  A=[4096,4096], B=[4096,4096]
    rect:    X=[1, 4096, 2560], W=[2560, 2560]  (+ optional bias [2560])
- Attention:
    Q=K=V=[1, 24, 4096, 128]
- Layout (transpose/permute):
    [1, 4096, 2560]            (rank-3)
    [1, 24, 4096, 128]         (rank-4 — attention layout)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file


SEED = 42
OUT_PATH = Path(__file__).parent / "inputs.safetensors"


def randn(shape, dtype, rng):
    # Generate in F32 from NumPy with fixed seed, then cast inside torch so
    # both sides see identical BF16 bytes.
    arr = rng.standard_normal(size=shape, dtype=np.float32)
    t = torch.from_numpy(arr)
    if dtype == torch.bfloat16:
        return t.to(torch.bfloat16).contiguous()
    if dtype == torch.float32:
        return t.contiguous()
    raise ValueError(f"unsupported dtype: {dtype}")


def main():
    rng = np.random.default_rng(SEED)
    tensors: dict[str, torch.Tensor] = {}

    # ── Tier 1/2/7 elementwise + reductions + cast ─────────────────────────
    for label, shape in [
        ("small", (1, 64, 2560)),
        ("hot", (1, 4096, 2560)),
        ("large", (2, 4096, 2560)),
    ]:
        tensors[f"x_bf16_{label}"] = randn(shape, torch.bfloat16, rng)
        tensors[f"y_bf16_{label}"] = randn(shape, torch.bfloat16, rng)
        # F32 source for BF16→F32→BF16 cast tier
        tensors[f"x_f32_{label}"] = randn(shape, torch.float32, rng)

    # ── Tier 3 norms ────────────────────────────────────────────────────────
    for label, shape, last_dim in [
        ("zimage", (1, 4096, 2560), 2560),
        ("klein", (1, 4096, 4096), 4096),
    ]:
        tensors[f"norm_x_{label}"] = randn(shape, torch.bfloat16, rng)
        # RMSNorm weight + LayerNorm weight/bias on last dim
        tensors[f"norm_w_{label}"] = randn((last_dim,), torch.bfloat16, rng)
        tensors[f"norm_b_{label}"] = randn((last_dim,), torch.bfloat16, rng)

    # ── Tier 4 GEMM ─────────────────────────────────────────────────────────
    # Square 4096×4096
    tensors["gemm_sq_a"] = randn((4096, 4096), torch.bfloat16, rng)
    tensors["gemm_sq_b"] = randn((4096, 4096), torch.bfloat16, rng)
    # Rect [1, 4096, 2560] @ [2560, 2560]
    tensors["gemm_rect_x"] = randn((1, 4096, 2560), torch.bfloat16, rng)
    # Note: PyTorch nn.Linear stores [out, in]; we provide both layouts
    # so each side uses what is natural.
    tensors["gemm_rect_w_oi"] = randn((2560, 2560), torch.bfloat16, rng)  # [out, in]
    tensors["gemm_rect_bias"] = randn((2560,), torch.bfloat16, rng)

    # ── Tier 5 attention ────────────────────────────────────────────────────
    tensors["attn_q"] = randn((1, 24, 4096, 128), torch.bfloat16, rng)
    tensors["attn_k"] = randn((1, 24, 4096, 128), torch.bfloat16, rng)
    tensors["attn_v"] = randn((1, 24, 4096, 128), torch.bfloat16, rng)

    # ── Tier 6 layout (re-use hot + attn shapes; new keys for clarity) ─────
    tensors["layout_r3"] = randn((1, 4096, 2560), torch.bfloat16, rng)
    tensors["layout_r4"] = randn((1, 24, 4096, 128), torch.bfloat16, rng)

    print(f"Generated {len(tensors)} tensors")
    for name, t in sorted(tensors.items()):
        print(f"  {name:24s}  shape={tuple(t.shape)}  dtype={t.dtype}")

    save_file(tensors, str(OUT_PATH))
    sz = OUT_PATH.stat().st_size
    print(f"\nWrote {OUT_PATH} ({sz / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
