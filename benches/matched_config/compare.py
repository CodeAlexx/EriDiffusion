#!/usr/bin/env python3
"""Compare flame_results.json vs pt_results.json.

Produces a markdown table sorted by flame/pt ratio descending. Verdict:
- 🚨 if ratio > 2.0
- ⚠️ if 1.3 < ratio <= 2.0
- ✅ if ratio < 1.0
- (blank) if 1.0 <= ratio <= 1.3 (within ~30% noise)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
FLAME = HERE / "flame_results.json"
PT = HERE / "pt_results.json"
OUT_MD = HERE / "results.md"


def verdict(ratio: float) -> str:
    if ratio > 2.0:
        return "🚨"
    if ratio > 1.3:
        return "⚠️"
    if ratio < 1.0:
        return "✅"
    return ""


def main():
    if not FLAME.exists() or not PT.exists():
        print(f"ERROR: need both {FLAME.name} and {PT.name}", file=sys.stderr)
        sys.exit(1)

    flame = json.loads(FLAME.read_text())
    pt = json.loads(PT.read_text())

    flame_by_name = {o["name"]: o for o in flame["ops"]}
    pt_by_name = {o["name"]: o for o in pt["ops"]}

    common = sorted(set(flame_by_name) & set(pt_by_name))
    missing_in_flame = sorted(set(pt_by_name) - set(flame_by_name))
    missing_in_pt = sorted(set(flame_by_name) - set(pt_by_name))

    rows = []
    for name in common:
        f = flame_by_name[name]
        p = pt_by_name[name]
        ratio = f["median_us"] / p["median_us"] if p["median_us"] > 0 else float("inf")
        rows.append({
            "name": name,
            "shape": f["shape"],
            "flame_us": f["median_us"],
            "pt_us": p["median_us"],
            "ratio": ratio,
            "verdict": verdict(ratio),
        })

    # Sort by ratio descending — worst flame regressions at the top
    rows.sort(key=lambda r: -r["ratio"])

    # ── Markdown output ────────────────────────────────────────────────────
    lines = []
    lines.append("# Matched-config benchmark — flame-core vs PyTorch")
    lines.append("")
    lines.append(f"- PyTorch: `{pt.get('torch_version', '?')}` on `{pt.get('device', '?')}`")
    lines.append(f"- Iters per op: {flame.get('iters', '?')} (warmup {flame.get('warmup', '?')})")
    lines.append(f"- Ratio = `flame_median / pt_median` — `<1.0` means flame is faster.")
    lines.append("")
    lines.append("| Op | Shape | flame µs | PyTorch µs | flame/PT | Verdict |")
    lines.append("|---|---|---:|---:|---:|:---:|")
    for r in rows:
        lines.append(
            f"| `{r['name']}` | `{r['shape']}` "
            f"| {r['flame_us']:8.2f} | {r['pt_us']:8.2f} | {r['ratio']:5.2f}× | {r['verdict']} |"
        )

    # ── Summary stats per tier prefix ──────────────────────────────────────
    def tier_of(name: str) -> str:
        head = name.split("/", 1)[0]
        return head
    by_tier: dict[str, list[float]] = {}
    for r in rows:
        by_tier.setdefault(tier_of(r["name"]), []).append(r["ratio"])

    lines.append("")
    lines.append("## Per-op-family geomean ratio")
    lines.append("")
    lines.append("| Op family | n | geomean(flame/PT) |")
    lines.append("|---|---:|---:|")
    import math
    for k in sorted(by_tier):
        v = by_tier[k]
        gm = math.exp(sum(math.log(x) for x in v) / len(v))
        lines.append(f"| `{k}` | {len(v)} | {gm:5.2f}× |")

    if missing_in_flame or missing_in_pt:
        lines.append("")
        lines.append("## Missing ops")
        if missing_in_flame:
            lines.append(f"- Only in PT: {missing_in_flame}")
        if missing_in_pt:
            lines.append(f"- Only in flame: {missing_in_pt}")

    out_text = "\n".join(lines) + "\n"
    OUT_MD.write_text(out_text)
    print(out_text)
    print(f"\nWrote {OUT_MD}")


if __name__ == "__main__":
    main()
