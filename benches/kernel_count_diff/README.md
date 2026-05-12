# Kernel-count diff: flame-core vs PyTorch (one transformer block, fwd+bwd)

## Why

The matched-config bench (`../matched_config/`) confirmed that flame-core
matches or beats PyTorch on **per-op** kernel time for the same input. Yet
flame trainers run multiple-times slower than PyTorch on the same model.
The hypothesis to test here:

> flame emits 2-3× more `cuLaunchKernel` calls per logical forward+backward
> than PyTorch for the same transformer block. The matched-config bench
> just confirmed flame matches or beats PT on per-op kernel time. So the
> remaining gap must be in kernel COUNT, not per-kernel time.

This harness measures kernel-launch count and total GPU kernel time on **one
simplified DiT-style block**, forward+backward, in BF16, on identical inputs.

## What it measures

- `cuLaunchKernel + cudaLaunchKernel` count per step.
- Total GPU kernel execution time per step.
- Top-10 GPU kernels by launch count, per side.

See `block_spec.md` for the exact block architecture (no LoRA, no AdaLN,
no RoPE — pure foundation: norms, linears, attention, residuals, silu,
mse).

## Tool choice

- **flame side**: `nsys profile --trace=cuda` plus `nsys stats
  --report cuda_gpu_kern_sum` — works cleanly on the static Rust binary.
- **PyTorch side**: `torch.profiler` (CUPTI). The system-installed
  nsys (2023.4 at `/usr/local/cuda-12.4/bin/nsys`, 2024.1 at
  `/opt/nvidia/nsight-compute/2024.1.1/host/target-linux-x64/nsys`)
  cannot decode CUDA 12.8 events emitted by PyTorch 2.10:

      **** Errors occurred while processing the raw events. ****

  The resulting `cuda_api_sum` / `cuda_gpu_kern_sum` is empty. This is a
  known nsys version-skew issue (nsys < 2024.4 vs CUDA 12.8 driver).

  `torch.profiler` hooks into the same CUPTI underneath, so the numbers
  it reports for kernel-launch counts and total CUDA time are directly
  comparable to nsys's GPU-side counts. We document the asymmetric
  tooling explicitly and verify the PT wall time (via `time.perf_counter`)
  matches the profiler's kernel-time sum within noise.

## How to run

```bash
./run_diff.sh                    # default: 6 steps, 2 warmup
STEPS=10 WARMUP=2 ./run_diff.sh  # override
```

The orchestrator:
1. Regenerates `inputs.safetensors` if missing (NumPy seed 42).
2. Builds `block_count_bench` from `flame-core` via cargo.
3. Runs PyTorch under `torch.profiler`, writes `/tmp/blockcount_pt.json`.
4. Runs flame under `nsys`, writes `/tmp/blockcount_fl.nsys-rep`.
5. Extracts both into `./results.md`.

## Sanity check

Both sides compute `loss = mse_loss(out, y)` from identical input bytes
and identical weight bytes. We assert the loss is finite on both sides
and print it. With seed 42 weights, the loss on this graph is **2.3594
on both sides** to 4 decimal places — bit-identical loss is not the goal
(different RNG paths in BF16 reductions / cuBLAS algorithm choice produce
different last-bit bytes), but matching to that precision is a strong
sanity signal that the graphs are equivalent.

## Constraints

- BF16 throughout. No autocast on the PT side.
- No FA2/FA3. PT-side SDPA forced to math backend
  (`torch.nn.attention.sdpa_kernel(backends=[SDPBackend.MATH])`).
  flame's `attention::sdpa` resolves to cuDNN SDPA flash-fprop on this
  hardware/shape (`cudnn_generated_fort_native_sdpa_sm80_flash_fprop_*`)
  which is the canonical non-FA-port path; PT's math path is a similar
  decomposed sequence. The two are not bit-identical, but they're both
  "non-FA2 cuDNN SDPA" — apples-to-apples for the kernel-count question.
- `FLAME_ALLOC_POOL=0` on flame (production cold-allocator config).
- PT keeps its default caching allocator (its production config).
- One side at a time on the GPU. Serial, not parallel.
- No torch.compile, no graph capture, no optimizer step.

## Result (from `run_diff.sh` on this machine, 2026-05-12)

| metric | flame | pt | flame/pt |
|---|---|---|---|
| launches/step | 83 | 112 | **0.74×** |
| kernel_time/step (ms) | 134.4 | 76.1 | **1.77×** |
| avg kernel_time (µs) | 1619 | 679 | — |
| wall_time/step (median, ms) | 135 | 75 | 1.80× |

**The hypothesis is wrong.** flame launches **fewer** kernels per step
than PT (0.74× = 26% fewer), but each kernel takes **2.39× as long on
average**. The total kernel time is 1.77× PT's, which fully explains
the 1.80× wall-time ratio. There is no hidden CPU stall: launch overhead
is not the bottleneck here. Per-kernel GPU work is.

### Where the GPU time goes (flame)

| kernel | count/step | time/step (ms) | %step |
|---|---|---|---|
| `permute_generic_bf16_kernel` | 4 | 53.0 | **39.4%** |
| `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_*` (4 variants) | 12 | 25.6 | 19.0% |
| `bf16_to_f32` / `f32_to_bf16` casts | 12 | 1.3 | 0.9% |
| `inplace_binary_kernel<float, AddOp<float>>` (F32 residual add) | 4 | 1.1 | 0.8% |
| `permute0213_vec4_bf16_kernel` | 9 | 0.5 | 0.3% |
| `narrow_backward_scatter_add_kernel` (chunk bwd) | 3 | 0.6 | 0.4% |
| GEMMs (all variants, summed) | ~9 | ~50 | ~37% |

### Where the GPU time goes (pt)

| kernel | count/step | time/step (ms) | %step |
|---|---|---|---|
| `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_*` (mix) | ~6 | ~13 | ~17% |
| `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_32x4_*` | ~3 | ~3 | ~4% |
| `vectorized_elementwise_kernel<4, BinaryFunctor<float>>` (resid F32 add) | 9 | 5.3 | 7% |
| `vectorized_elementwise_kernel<4, bf16_copy_kernel>` | 11 | 2.8 | 4% |
| SDPA decomposition (cutlass + softmax + bmm) | ~10 | ~12 | ~16% |
| cuDNN fused norms | ~6 | ~6 | ~8% |

### The single dominant finding

**`permute_generic_bf16_kernel` accounts for 38.7% of flame's per-step
kernel time.** It is launched 4×/step, average 12.3 ms per launch.

This is consistent with the comment in `EriDiffusion-v2/crates/eridiffusion-core/src/models/zimage.rs:1714-1716`:

> Was: `weight.transpose()?.contiguous()?` + `matmul` → materialized
> every weight transpose via `permute_generic_bf16_kernel` (rank-2 [1,0]
> is not a fast-path perm).

This bench's block uses `fused_linear3d_native` (which avoids the
permute for the forward weight transpose), so the 4 permute_generic
launches/step that survive are on the **backward path** — likely the
QKV chunk gradient or the SDPA backward path materializing a permuted
view. The remaining 9 launches are `permute0213_vec4_bf16_kernel`,
flame's tuned fast path, which is fine (0.5 ms/step total).

PyTorch's equivalent material-transpose work goes through
`unrolled_elementwise_kernel<direct_copy_kernel_cuda>` at ~2 ms/step
total — **25× cheaper** than flame's `permute_generic_bf16_kernel` for
the same logical work.

### What this tells us about the broader perf gap

This single-block measurement reproduces ~70% of the trainer-level gap
that we observed at the run level (zimage trainer is ~3-4× PT;
matched-config per-op is ~1.0×). The remaining ~2× of trainer slowdown
that's not in this bench comes from:

- Optimizer step (Adam moments, weight decay, parameter copy) — not
  measured here, we deliberately skip the step.
- Activation recomputation in gradient-checkpointed paths — the real
  trainer does this; this single-step bench does not.
- Per-block adapter (LoRA/LyCORIS) overhead — not measured.

But the **bottom-line answer to the kernel-count hypothesis is: NO,
flame does not emit more kernels.** It emits fewer kernels, but a
single class of them (`permute_generic_bf16_kernel`, used on backward
paths involving rank-4 transposes that miss the `permute0213` fast
path) is ~25× slower than the PT equivalent and accounts for nearly
40% of per-step time.

### Recommended follow-ups (do not include in this bench)

1. Trace which call site invokes `permute_generic_bf16_kernel` on the
   backward path. Candidates: `chunk` backward (qkv unpack), SDPA
   backward's K-transpose, the `permute(0,2,1,3)` undo before the output
   linear. Use `RUST_BACKTRACE=1` plus targeted printlns in
   `flame_core::ops::permute_generic_bf16` to find the call sites.

2. Add a fast-path for rank-4 perms `[0,2,1,3]` and `[0,1,3,2]` in
   `permute_generic_bf16` so the SDPA-adjacent backward calls hit the
   tuned path (which is what `permute0213_vec4_bf16_kernel` already
   does on the forward side — confirm by reading
   `flame-core/src/cuda_kernel_sources.rs::PERMUTE_*`).

3. Verify whether the 4 per-step launches are actually backward-only
   by re-running with `requires_grad=False` on the input. If the count
   drops to 0, the issue is purely backward; if it drops to ~2, there's
   a forward-path miss as well.

## 200-word analysis

Does flame emit more kernels than PyTorch on the same logical
forward+backward of one transformer block? **No.** flame issues 83
GPU-side kernel launches per step; PyTorch issues 112. flame is, in
absolute terms, calling cuLaunchKernel **26% less often** than PT — the
opposite of the hypothesis.

The 1.8× wall-time gap is per-kernel work, not launch overhead. flame's
average kernel takes 1619 µs; PT's takes 679 µs. The kernel that
dominates flame's step is **`permute_generic_bf16_kernel` — 4 launches
per step at ~13 ms each, 53 ms / step total = 39% of flame's per-step
kernel time**. PT has no analog at this magnitude: its largest single
permute-like contributor is `unrolled_elementwise_kernel<direct_copy>`
at 2 ms / step (~25× cheaper for the same logical work).

The `permute_generic_bf16_kernel` calls survive because the bench's
forward already uses `fused_linear3d_native` (no forward transpose).
The 4 launches/step come from the **backward path** — likely the
QKV-chunk gradient `narrow_backward_scatter_add` adjacency, the SDPA
backward's K-permute undo, or the rank-4 `permute(0,2,1,3)` inverse
before the output linear. The recommended next step is to add a
rank-4 `[0,2,1,3]` fast path to `permute_generic_bf16` so these
backward transposes hit the same tuned kernel that the forward
already uses (`permute0213_vec4_bf16_kernel`).

## Files

- `block_spec.md` — exact block architecture, shapes, constraints.
- `gen_inputs.py` — seed-42 input generator; writes `inputs.safetensors`.
- `pytorch_block.py` — PT side, runs forward+backward + torch.profiler.
- `flame_block_bin.rs` — flame side; built into
  `flame-core/target/release/block_count_bench`.
- `run_diff.sh` — orchestrator: build, profile, extract, write `results.md`.
- `_extract.py` — parses both sides' outputs into `results.md`.
- `results.md` — the diff table + top-10 kernels. Generated.
- `inputs.safetensors` — generated, ~200 MB, gitignored.
