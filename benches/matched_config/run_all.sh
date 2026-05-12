#!/usr/bin/env bash
# Matched-config bench orchestrator.
#
# Builds the flame Rust binary, regenerates inputs (idempotent — same seed),
# runs PyTorch then flame sequentially on the same GPU, then runs compare.
#
# We DO NOT run the two sides in parallel — they share the GPU and would
# pollute each other's timings. Sequential ensures a quiet GPU per side.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLAME_ROOT="$(cd "$HERE/../../../flame-core" && pwd)"
INPUTS="$HERE/inputs.safetensors"

echo "==> Step 1: regenerate inputs (seed=42)"
python3 "$HERE/gen_inputs.py"

echo
echo "==> Step 2: build flame matched_bench binary (release, --features cuda)"
( cd "$FLAME_ROOT" && cargo build --release --features cuda --bin matched_bench )

# Verify the GPU isn't already loaded from a stray process.
if command -v nvidia-smi >/dev/null 2>&1; then
    echo
    echo "==> GPU pre-flight"
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used,temperature.gpu \
               --format=csv,noheader
fi

echo
echo "==> Step 3: PyTorch side (quiet GPU)"
python3 "$HERE/pytorch_side.py"

echo
echo "==> Step 4: flame side (FLAME_ALLOC_POOL=0, production config)"
FLAME_ALLOC_POOL=0 "$FLAME_ROOT/target/release/matched_bench" \
    --inputs "$INPUTS" \
    --out    "$HERE/flame_results.json"

echo
echo "==> Step 5: compare → results.md"
python3 "$HERE/compare.py"

echo
echo "Done. Report: $HERE/results.md"
