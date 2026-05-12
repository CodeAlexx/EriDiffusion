#!/usr/bin/env python3
"""PyTorch side of the kernel-count diff bench.

Runs N forward+backward steps on a single transformer block. Designed to
be profiled by `nsys profile --trace=cuda`. Each step is one full graph;
we sync between steps so nsys can attribute kernels per step.

Usage (standalone, for sanity check, prints loss + step time):
    python3 pytorch_block.py --steps 6

For kernel counting, wrap in nsys (see run_diff.sh).

Pinned config:
- BF16 throughout (no autocast, no compile, no flash attn).
- SDPA backend: math-only (apples-to-apples vs flame which uses cuDNN/math).
- requires_grad=True on input + all weights.
- No optimizer step; gradients are populated but we drop them between steps.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file


B, SEQ, HIDDEN = 1, 4096, 2560
HEADS, HEAD_DIM = 20, 128
MLP = 4 * HIDDEN  # 10240
EPS = 1e-6


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    # match flame's `primitive_rms_norm` semantics: F32-internal compute, BF16 out
    dtype = x.dtype
    x_f = x.float()
    var = x_f.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x_f * torch.rsqrt(var + eps)
    return (x_normed * w.float()).to(dtype)


def block_forward(x, y, weights):
    """One transformer block, F32-internal rms_norm, BF16 elsewhere."""
    w_norm1 = weights["w_norm1"]
    w_norm2 = weights["w_norm2"]
    w_qkv = weights["w_qkv"]   # [3H, H]
    w_o = weights["w_o"]       # [H, H]
    w_mlp1 = weights["w_mlp1"] # [4H, H]
    w_mlp2 = weights["w_mlp2"] # [H, 4H]

    # --- attention ---
    h = rms_norm(x, w_norm1)
    qkv = F.linear(h, w_qkv)              # [B, seq, 3H]
    q, k, v = qkv.chunk(3, dim=-1)
    q = q.reshape(B, SEQ, HEADS, HEAD_DIM).permute(0, 2, 1, 3)
    k = k.reshape(B, SEQ, HEADS, HEAD_DIM).permute(0, 2, 1, 3)
    v = v.reshape(B, SEQ, HEADS, HEAD_DIM).permute(0, 2, 1, 3)

    # apples-to-apples: force math kernel (no flash, no mem-efficient).
    # flame's SDPA at this shape on sm86 also resolves to a non-flash path.
    with torch.nn.attention.sdpa_kernel(
        backends=[torch.nn.attention.SDPBackend.MATH]
    ):
        attn = F.scaled_dot_product_attention(q, k, v, scale=1.0 / (HEAD_DIM ** 0.5))

    attn = attn.permute(0, 2, 1, 3).contiguous().reshape(B, SEQ, HIDDEN)
    attn = F.linear(attn, w_o)
    x1 = x + attn

    # --- mlp ---
    h2 = rms_norm(x1, w_norm2)
    m = F.linear(h2, w_mlp1)
    m = F.silu(m)
    m = F.linear(m, w_mlp2)
    out = x1 + m

    loss = F.mse_loss(out, y)
    return loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default=str(Path(__file__).parent / "inputs.safetensors"))
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--profile-json", default=None,
                    help="Path to write per-kernel counts JSON via torch.profiler.")
    args = ap.parse_args()

    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    raw = load_file(args.inputs, device="cuda")
    weights = {k: v for k, v in raw.items() if k.startswith("w_")}
    for k in weights:
        weights[k] = weights[k].detach().requires_grad_(True)
    x = raw["x"].detach().requires_grad_(True)
    y = raw["y"].detach()  # target, no grad

    # Optional: NVTX ranges so nsys can split steps if user passes --trace=cuda,nvtx
    have_nvtx = hasattr(torch.cuda, "nvtx")

    # torch.profiler path — counts kernel launches per step via CUPTI. This
    # is the only reliable way to get per-kernel counts on this box, since
    # the system nsys 2023.4 / 2024.1 can't decode CUDA 12.8 events from a
    # PyTorch process ("Errors occurred while processing the raw events").
    prof = None
    if args.profile_json is not None:
        from torch.profiler import profile, ProfilerActivity
        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
            with_stack=False,
        )
        prof.__enter__()

    losses = []
    durations = []
    for i in range(args.steps):
        # zero grads (just drop them; no optimizer)
        for w in weights.values():
            if w.grad is not None:
                w.grad = None
        if x.grad is not None:
            x.grad = None

        torch.cuda.synchronize()
        if have_nvtx:
            torch.cuda.nvtx.range_push(f"step{i}")
        t0 = time.perf_counter()

        loss = block_forward(x, y, weights)
        loss.backward()

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        if have_nvtx:
            torch.cuda.nvtx.range_pop()

        durations.append((t1 - t0) * 1e3)
        losses.append(loss.item())
        print(f"step {i}: loss={loss.item():.4f}  wall={durations[-1]:.2f} ms")

    measured = durations[args.warmup:]
    if measured:
        med = sorted(measured)[len(measured) // 2]
        print(f"\nmedian measured step (steps {args.warmup}..{args.steps - 1}): {med:.2f} ms")
    print(f"total measured steps: {len(measured)}")
    print(f"warmup steps: {args.warmup}")

    if prof is not None:
        prof.__exit__(None, None, None)
        # Aggregate kernel events by name. Each event has a `device_type`
        # (we want CUDA == kernel) and a `count` of how many times it ran.
        from collections import defaultdict
        import json
        kern_counts: dict[str, int] = defaultdict(int)
        kern_time_us: dict[str, float] = defaultdict(float)
        total_kern_launches = 0
        total_kern_time_us = 0.0

        # key_averages aggregates by name across all events in the trace.
        avgs = prof.key_averages()
        for ev in avgs:
            # device_type 1 = CUDA on most builds; safer to check the
            # device_time attributes directly.
            dev_us = float(ev.device_time_total if hasattr(ev, "device_time_total") else ev.cuda_time_total)
            count = int(ev.count)
            # Anything with non-zero CUDA time is a kernel-launching op. But
            # we also have raw-kernel events (no CPU wrapper). Both show up
            # in key_averages on modern profiler; we keep both, deduplicated
            # by name.
            if dev_us > 0:
                kern_counts[ev.key] += count
                kern_time_us[ev.key] += dev_us
                total_kern_launches += count
                total_kern_time_us += dev_us

        # But the better count: walk raw events for ProfilerActivity.CUDA.
        # `events()` returns FunctionEvents with .device_type for kernel rows.
        try:
            cuda_kern_count = 0
            cuda_kern_time_us = 0.0
            kern_count_by_name: dict[str, int] = defaultdict(int)
            kern_time_by_name: dict[str, float] = defaultdict(float)
            for ev in prof.events():
                # device_type==DeviceType.CUDA → it's a kernel-launch event
                if str(getattr(ev, "device_type", "")).endswith("CUDA"):
                    cuda_kern_count += 1
                    cuda_kern_time_us += float(ev.device_time_total)
                    kern_count_by_name[ev.key] += 1
                    kern_time_by_name[ev.key] += float(ev.device_time_total)
            print(f"\n[profiler] CUDA kernel events: {cuda_kern_count} total, {cuda_kern_time_us / 1e3:.2f} ms")
            print(f"[profiler]   per step (total / {args.steps}): "
                  f"{cuda_kern_count / args.steps:.0f} launches, "
                  f"{cuda_kern_time_us / args.steps / 1e3:.2f} ms")

            top = sorted(kern_count_by_name.items(), key=lambda kv: -kv[1])[:10]
            json_payload = {
                "steps": args.steps,
                "warmup": args.warmup,
                "total_kernel_launches": cuda_kern_count,
                "total_kernel_time_us": cuda_kern_time_us,
                "top_kernels_by_count": [
                    {"name": n, "count": c, "time_us": kern_time_by_name[n]}
                    for n, c in top
                ],
                "all_kernels": [
                    {"name": n, "count": c, "time_us": kern_time_by_name[n]}
                    for n, c in sorted(kern_count_by_name.items(), key=lambda kv: -kv[1])
                ],
            }
            with open(args.profile_json, "w") as f:
                json.dump(json_payload, f, indent=2)
            print(f"[profiler] wrote {args.profile_json}")
        except Exception as e:
            print(f"[profiler] failed to extract per-event counts: {e}")


if __name__ == "__main__":
    main()
