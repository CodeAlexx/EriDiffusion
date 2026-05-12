#!/usr/bin/env bash
# Orchestrator: build flame bin, profile both PT and flame, extract
# kernel counts, print diff. Runs each side serially on a quiet GPU.
#
# Tool choice:
#   - PyTorch side: torch.profiler (CUPTI-based). The system nsys 2023.4
#     and 2024.1 cannot decode CUDA 12.8 events from a PyTorch process
#     ("Errors occurred while processing the raw events" then 0 kernels
#     in cuda_api_sum / cuda_gpu_kern_sum). torch.profiler hooks into the
#     same CUPTI underneath, so the counts are directly comparable to
#     nsys's GPU kernel counts.
#   - flame side: nsys (works fine for the static Rust binary).
#
# Usage:
#   ./run_diff.sh                # default: 6 steps, 2 warmup
#   STEPS=10 WARMUP=2 ./run_diff.sh
#
# Output: ./results.md plus raw artifacts in /tmp/.

set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
FLAME_CORE="$DIR/../../../flame-core"
NSYS="/usr/local/cuda-12.4/bin/nsys"

STEPS="${STEPS:-6}"
WARMUP="${WARMUP:-2}"

PT_JSON="/tmp/blockcount_pt.json"
FL_REP="/tmp/blockcount_fl.nsys-rep"
FL_API="/tmp/blockcount_fl_api.csv"
FL_KERN="/tmp/blockcount_fl_kern.csv"

echo "================================================================"
echo " kernel-count diff: PyTorch vs flame-core"
echo " steps=$STEPS  warmup=$WARMUP  (per-step divisor = $STEPS total)"
echo "================================================================"

# Step 1: generate inputs (idempotent — seed 42)
if [[ ! -f "$DIR/inputs.safetensors" ]]; then
  echo "[1/6] Generating inputs..."
  python3 "$DIR/gen_inputs.py"
else
  echo "[1/6] inputs.safetensors exists, skipping gen"
fi

# Step 2: build flame binary
echo "[2/6] Building flame block_count_bench..."
(cd "$FLAME_CORE" && cargo build --release --features cuda --bin block_count_bench 2>&1 | tail -5)

FLAME_BIN="$FLAME_CORE/target/release/block_count_bench"
if [[ ! -x "$FLAME_BIN" ]]; then
  echo "ERROR: $FLAME_BIN not built"
  exit 1
fi

# Step 3: PyTorch sanity check
echo "[3/6] PyTorch sanity check..."
python3 "$DIR/pytorch_block.py" --steps 3 --warmup 1 2>&1 | tail -6

echo ""
echo "[4/6] Profiling PyTorch via torch.profiler..."
python3 "$DIR/pytorch_block.py" --steps "$STEPS" --warmup "$WARMUP" --profile-json "$PT_JSON" 2>&1 | tail -12

echo ""
echo "[5/6] Profiling flame with nsys..."
rm -f "$FL_REP"
FLAME_ALLOC_POOL=0 "$NSYS" profile --trace=cuda --force-overwrite=true -o "${FL_REP%.nsys-rep}" \
  "$FLAME_BIN" --inputs "$DIR/inputs.safetensors" --steps "$STEPS" --warmup "$WARMUP" 2>&1 | tail -8

echo ""
echo "[6/6] Extracting kernel counts..."

# Extract flame CSVs from nsys
"$NSYS" stats --report cuda_api_sum --format csv --force-export=true \
  "$FL_REP" > "$FL_API" 2>/dev/null || true
"$NSYS" stats --report cuda_gpu_kern_sum --format csv --force-export=true \
  "$FL_REP" > "$FL_KERN" 2>/dev/null || true

# Parse + build results.md
python3 "$DIR/_extract.py" \
  --pt-json "$PT_JSON" \
  --fl-api "$FL_API" --fl-kern "$FL_KERN" \
  --steps "$STEPS" --warmup "$WARMUP" \
  --out "$DIR/results.md"

echo ""
echo "Done. Results written to: $DIR/results.md"
echo "Raw artifacts:"
echo "  PT:    $PT_JSON"
echo "  flame: $FL_REP  $FL_API  $FL_KERN"
