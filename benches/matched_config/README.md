# Matched-config bench: flame-core vs PyTorch

A per-op, per-shape, **bit-identical-input** benchmark of flame-core (Rust) against PyTorch 2.10 on the same GPU. The goal is to find out, for each individual kernel, where flame is faster, slower, or equal.

Every prior comparison we ran had confounds: batch size mismatch, LyCORIS overhead on top of the flame side, FP8 quant on the PT side and not on flame, different optimizers, full step times that include data loading and optimizer state. Until you compare `flame.silu(x)` vs `torch.silu(x)` on the same input on the same GPU, you don't actually know per-op where flame stands. This harness does exactly that.

## What it does

1. `gen_inputs.py` generates 25 input tensors in F32 with NumPy seed 42, casts to BF16 via PyTorch (so we get correct round-to-even), and saves them to a single `inputs.safetensors` file (~530 MB).
2. `pytorch_side.py` loads that file on CUDA, runs each op 20 warmup + 100 timed iterations with `torch.cuda.Event`, records median + p90 in microseconds to `pt_results.json`. All ops run under `torch.inference_mode()`.
3. `flame_side.rs` is built as a Rust binary (`matched_bench`) in the `flame-core` crate. It loads the same safetensors via `flame_core::serialization::load_file`, runs the same ops with `cudaEventRecord` / `cudaEventSynchronize`, autograd off, writes `flame_results.json`.
4. `compare.py` diffs the two JSONs and produces `results.md` sorted by `flame_us / pt_us`.

## How to run

```bash
./run_all.sh
```

The orchestrator:
- regenerates inputs (idempotent; seed is fixed),
- builds the flame binary (`cargo build --release --features cuda --bin matched_bench`),
- runs PyTorch, then flame **sequentially** on the same GPU (parallel runs would pollute timings),
- writes `results.md`.

Build separately if needed:
```bash
cd ../../../flame-core
cargo build --release --features cuda --bin matched_bench
```

## Ops covered

| Tier | Ops | Shapes |
|---|---|---|
| 1 — elementwise | silu, gelu, add, mul, mul_scalar | `[1,64,2560]`, `[1,4096,2560]`, `[2,4096,2560]` |
| 2 — reductions  | sum (full), sum_dim(-1, keepdim), mean | same |
| 3 — norms       | rms_norm, layer_norm (with weight/bias) | `[1,4096,2560]`, `[1,4096,4096]` |
| 4 — GEMM        | matmul square 4096², matmul rect 4096×2560, linear-with-bias | as listed |
| 5 — attention   | sdpa(Q,K,V) no mask | `[1, 24, 4096, 128]` |
| 6 — layout      | material transpose, permute+reshape | rank-3 and rank-4 attn |
| 7 — cast        | BF16→F32, F32→BF16 | three elementwise shapes |

All ops run in BF16 except the cast tier, which is the BF16 ↔ F32 boundary.

## Methodology

- **Same input bytes.** NumPy seed 42, single safetensors file, both sides load it directly. No re-randomization on either side.
- **Same device, same default stream.** GPU 0. PyTorch uses its default stream; flame's `global_cuda_device` uses its global stream. We do not switch streams.
- **Warmup:** 20 iters before timing, discarded.
- **Timing:** 100 timed iters. Per iter: `cuda.Event.record()` → op → `cuda.Event.record()` → `event.synchronize()` → `start.elapsed_time(end)`. We sync the device once before each iteration so we measure the kernel's full runtime, not its launch overhead overlapping with the previous iter.
- **Forward only, no autograd.** PyTorch side runs in `torch.inference_mode()`; flame side calls `AutogradContext::set_enabled(false)` at startup.
- **`FLAME_ALLOC_POOL=0`.** flame's caching allocator is disabled — this matches the production cold-path config that EriDiffusion uses for `prepare_*` and inference binaries. PyTorch keeps its default caching allocator on (that is _its_ production config).
  - This is **not** maximally "fair" in some abstract sense; it is apples-to-apples vs how each library is actually used. The README documents this explicitly.
- **No interference.** PyTorch and flame run **sequentially**, not concurrently. Same quiet GPU on both sides.
- **No autograd, no autocast, no torch.compile.** We measure the raw kernel paths.

## What the verdict columns mean

- ✅ — flame faster than PT (ratio < 1.0)
- (blank) — within ~30% noise (1.0–1.3×)
- ⚠️ — flame 1.3×–2.0× slower
- 🚨 — flame >2× slower

## Caveats

- **Median + p90 only.** No variance reported. With 100 iters and a quiet GPU on a 3090 Ti, run-to-run jitter is ~3% on most ops. The small shapes (~5–15 µs) get noisier because kernel-launch overhead dominates.
- **The small shape favours flame.** flame's launch overhead is lower (fewer abstraction layers between op call and `cuLaunchKernel`) than PyTorch's, so on tiny tensors flame wins by a ratio that won't matter in real training. For a real perf signal look at the hot/large shapes.
- **`fused_linear3d_native` is flame's production linear path.** It does the weight transpose inside cuBLASLt. We compare it to PyTorch's `F.linear`, which also does an internal transpose via cuBLAS. This is what trainers actually call.
- **No flash attention.** flame's `sdpa` goes through cuDNN SDPA → memory-efficient path. PyTorch on this hardware will also pick cuDNN/mem-efficient. Neither side is using FA2/FA3.
- **`sum_dim_last` is comically slow on flame.** ~1.5 ms vs PT's 35 µs at `[1, 4096, 2560]`, a 43× regression. This is a real finding (the original task spec called it out as suspect — "BF16→F32→reduce→F32→BF16 round-trip"). Now we have a number on it.
- **`permute_reshape/r4` is 2.4× slower in flame.** Likely the `permute_generic` path doing a generic strided copy rather than a tuned transpose kernel.
- **`mul_scalar` and the cast small-shape numbers are launch-overhead dominated.** Don't read the 0.27× on `cast_f32_to_bf16/small` as a real 4× speedup; it's flame's lower per-launch CPU cost relative to PT.

## Result (from `run_all.sh` on this machine)

See `results.md` for the table. Headline:

- **Per-op family geomean (lower = flame faster):**
  - `silu` 0.73×, `gelu` 0.75×, `add` 0.74×, `mul` 0.75×, `mul_scalar` 0.66× — flame's elementwise BF16 kernels are ~25–35% faster than PyTorch.
  - `rms_norm` 0.86×, `layer_norm` 0.89× — flame fused norms beat PyTorch's by ~10–15%.
  - `sdpa` 0.94×, `linear_bias` 0.96×, `matmul` 1.09× — GEMM-bound ops are essentially a tie (both sides hit cuBLAS).
  - `transpose_contig` 0.90× — material transpose is slightly faster in flame.
  - `mean_full` 1.34×, `sum_full` 0.89× — full reductions are roughly tied.
  - `permute_reshape` 2.36×, `sum_dim_last` 20.73× — flame regressions. Real bugs to investigate.

- **Worst three flame ratios** (sorted by `flame/PT`):
  1. `sum_dim_last/hot` — 42.22× slower (1513 µs vs 36 µs).
  2. `sum_dim_last/large` — 26.84× slower.
  3. `sum_dim_last/small` — 7.86× slower.

  `sum_dim_last` is the dominant finding. The shape used (`[1, 4096, 2560]`, reduce last dim) is exactly the hidden-state shape used in training. Whatever flame is doing in `Tensor::sum_dim` here, it's not a single-pass row reduction — it appears to be paying for a transpose or a per-row sequential pass. PyTorch's `.sum(dim=-1)` finishes in ~35 µs (memory bandwidth bound, ~2.4 GB / 800 GB/s ≈ 30 µs). flame takes 1.5 ms, which is 40× off the memory-bandwidth floor.

  This is a real finding and is the highest-value target for flame-core kernel work.

## Files

- `gen_inputs.py` — input generator (seed 42).
- `pytorch_side.py` — PT bench (`pt_results.json`).
- `flame_side.rs` — flame bench, built into `flame-core/target/release/matched_bench`.
- `compare.py` — diffs the two JSONs (`results.md`).
- `run_all.sh` — orchestrator.
- `inputs.safetensors` — generated by `gen_inputs.py`, ~530 MB, gitignored.
- `pt_results.json`, `flame_results.json`, `results.md` — output.
