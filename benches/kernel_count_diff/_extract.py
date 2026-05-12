#!/usr/bin/env python3
"""Parse profiler outputs from both sides and produce results.md.

- PT side: a JSON written by torch.profiler (see pytorch_block.py).
- flame side: nsys CSV stats (cuda_api_sum, cuda_gpu_kern_sum).

Both ultimately count GPU kernel launches via CUPTI; the numbers are
directly comparable.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Dict, Tuple


def parse_nsys_csv(path: Path) -> List[Dict[str, str]]:
    """nsys stats CSVs have a couple of preamble lines + a header row."""
    rows: List[Dict[str, str]] = []
    if not path.exists():
        return rows
    lines: list[str] = []
    with open(path) as f:
        for line in f:
            ls = line.strip()
            if not ls:
                continue
            if ls.startswith("#") or ls.startswith("Generating") or ls.startswith("Processing"):
                continue
            lines.append(line)
    if not lines:
        return rows
    reader = csv.DictReader(lines)
    for r in reader:
        rows.append({k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in r.items()})
    return rows


def find_col(rows, candidates):
    if not rows:
        return ""
    keys = list(rows[0].keys())
    for c in candidates:
        for k in keys:
            if c.lower() == k.lower():
                return k
    for c in candidates:
        for k in keys:
            if c.lower() in k.lower():
                return k
    return ""


def parse_int(s: str) -> int:
    s = s.replace(",", "").strip()
    if not s or s.lower() in ("n/a", "nan"):
        return 0
    try:
        return int(float(s))
    except ValueError:
        return 0


def parse_float(s: str) -> float:
    s = s.replace(",", "").strip()
    if not s or s.lower() in ("n/a", "nan"):
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def flame_kern_summary(rows):
    """From flame's cuda_gpu_kern_sum CSV, return total launches, total time
    (ns), and the top-10 kernels by count.
    """
    name_col = find_col(rows, ["Name", "Kernel Name", "Demangled Name"])
    inst_col = find_col(rows, ["Instances", "Num Calls", "Count"])
    time_col = find_col(rows, ["Total Time (ns)", "Total time (ns)", "Total Time"])
    if not name_col or not inst_col or not time_col:
        return 0, 0.0, []

    total_count = 0
    total_time = 0.0
    by_kernel = []
    for r in rows:
        nm = r.get(name_col, "")
        c = parse_int(r.get(inst_col, "0"))
        t = parse_float(r.get(time_col, "0"))
        total_count += c
        total_time += t
        by_kernel.append((nm, c, t))
    by_kernel.sort(key=lambda x: -x[1])
    return total_count, total_time, by_kernel[:10]


def flame_api_summary(rows):
    """From flame's cuda_api_sum CSV, return total cuda runtime API time."""
    time_col = find_col(rows, ["Total Time (ns)", "Total time (ns)", "Total Time"])
    if not time_col:
        return 0.0, 0
    total_time = 0.0
    total_calls = 0
    calls_col = find_col(rows, ["Num Calls", "Instances", "Count"])
    for r in rows:
        total_time += parse_float(r.get(time_col, "0"))
        if calls_col:
            total_calls += parse_int(r.get(calls_col, "0"))
    return total_time, total_calls


def shorten(name: str, n: int = 60) -> str:
    name = name.strip()
    if len(name) <= n:
        return name
    return name[: n - 3] + "..."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt-json", type=Path, required=True)
    ap.add_argument("--fl-api", type=Path, required=True)
    ap.add_argument("--fl-kern", type=Path, required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--warmup", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    # PT side
    pt = json.loads(args.pt_json.read_text())
    pt_total_launches = int(pt["total_kernel_launches"])
    pt_total_us = float(pt["total_kernel_time_us"])
    pt_top = [(k["name"], k["count"], k["time_us"]) for k in pt["top_kernels_by_count"]]
    pt_steps = int(pt["steps"])

    # flame side
    fl_kern_rows = parse_nsys_csv(args.fl_kern)
    fl_api_rows = parse_nsys_csv(args.fl_api)
    fl_total_launches, fl_total_ns, fl_top = flame_kern_summary(fl_kern_rows)
    fl_api_time_ns, fl_api_calls = flame_api_summary(fl_api_rows)
    fl_steps = args.steps

    # Per-step normalization: total over all steps (warmup included).
    pt_per = lambda n: n / pt_steps if pt_steps else 0
    fl_per = lambda n: n / fl_steps if fl_steps else 0

    pt_launches_step = pt_per(pt_total_launches)
    fl_launches_step = fl_per(fl_total_launches)
    pt_kerntime_step_ms = pt_per(pt_total_us) / 1e3
    fl_kerntime_step_ms = fl_per(fl_total_ns) / 1e6
    fl_apitime_step_ms = fl_per(fl_api_time_ns) / 1e6

    def ratio(a, b):
        return (a / b) if b else float("nan")

    out = []
    out.append("# Kernel-count diff — results")
    out.append("")
    out.append(f"- Harness steps: {args.steps} (warmup: {args.warmup})")
    out.append("- Per-step numbers are total / total_steps (consistent across both sides).")
    out.append("- PT side counted via `torch.profiler` (CUPTI). flame side counted via `nsys`.")
    out.append("  Both ultimately go through CUPTI on the GPU; the kernel counts are directly comparable.")
    out.append("- nsys 2023.4 / 2024.1 on this box cannot decode CUDA 12.8 events from a PyTorch process,")
    out.append("  hence the asymmetric tooling. Counts and per-kernel times are still apples-to-apples.")
    out.append("")
    out.append("## Summary table")
    out.append("")
    out.append("| metric | flame | pt | flame/pt |")
    out.append("|---|---|---|---|")
    out.append(f"| launches/step | {fl_launches_step:.0f} | {pt_launches_step:.0f} | {ratio(fl_launches_step, pt_launches_step):.2f}× |")
    out.append(f"| kernel_time/step (ms) | {fl_kerntime_step_ms:.2f} | {pt_kerntime_step_ms:.2f} | {ratio(fl_kerntime_step_ms, pt_kerntime_step_ms):.2f}× |")
    out.append(f"| avg kernel_time (µs) | {(fl_total_ns / fl_total_launches / 1e3) if fl_total_launches else 0:.1f} | {(pt_total_us / pt_total_launches) if pt_total_launches else 0:.1f} | — |")
    out.append("")
    out.append("Notes:")
    out.append(f"- flame API time/step: {fl_apitime_step_ms:.2f} ms (sum of all cuda runtime+driver API calls; includes cudaStreamSynchronize stalls).")
    out.append(f"- flame total cuda API calls/step: {fl_api_calls / fl_steps:.0f}.")
    out.append("")
    out.append("## Top-10 GPU kernels by launch count")
    out.append("")
    out.append("### flame")
    out.append("")
    out.append("| kernel | count | total_time_ms | per_step |")
    out.append("|---|---|---|---|")
    for name, c, t in fl_top:
        out.append(f"| `{shorten(name, 70)}` | {c} | {t / 1e6:.2f} | {c / fl_steps:.1f} |")
    out.append("")
    out.append("### pytorch")
    out.append("")
    out.append("| kernel | count | total_time_ms | per_step |")
    out.append("|---|---|---|---|")
    for name, c, t_us in pt_top:
        out.append(f"| `{shorten(name, 70)}` | {c} | {t_us / 1e3:.2f} | {c / pt_steps:.1f} |")

    out.append("")
    out.append("## Raw paths")
    out.append("")
    out.append(f"- PT profile JSON: `{args.pt_json}`")
    out.append(f"- flame api CSV: `{args.fl_api}`")
    out.append(f"- flame kern CSV: `{args.fl_kern}`")

    args.out.write_text("\n".join(out) + "\n")
    print("=" * 64)
    print(f"flame launches/step: {fl_launches_step:.0f}")
    print(f"pt    launches/step: {pt_launches_step:.0f}")
    print(f"ratio (flame/pt):    {ratio(fl_launches_step, pt_launches_step):.2f}x")
    print(f"flame kernel_time/step: {fl_kerntime_step_ms:.2f} ms")
    print(f"pt    kernel_time/step: {pt_kerntime_step_ms:.2f} ms")
    print(f"ratio (flame/pt):       {ratio(fl_kerntime_step_ms, pt_kerntime_step_ms):.2f}x")
    print("=" * 64)


if __name__ == "__main__":
    main()
