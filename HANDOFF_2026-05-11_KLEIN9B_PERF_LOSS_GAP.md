# Handoff 2026-05-11 — Klein 9B EDv2-vs-ai-toolkit perf + loss gap

A long session today comparing EDv2's klein9b LoKr training against
ai-toolkit's klein9b LoKr training on identical hyperparameters. Two
real signals surfaced; neither is root-caused.

## What got shipped today (committed + pushed)

| Commit | Repo | Change |
|---|---|---|
| `f9a7373` | flame-core (main) | `tests/rms_norm_vs_primitive_zimage.rs` + FLAME_INDEX note for `norm::rms_norm` as the training-safe entry — pinned bit-exact parity vs the F32 primitive chain |
| `658f57b` | EriDiffusion (master) | `zimage::primitive_rms_norm` body becomes thin delegate to `flame_core::norm::rms_norm` (kept name for diff minimization) |
| `cc6b0ac` | EriDiffusion | ScheduleFree `enter_eval_mode` formula corrected: `x = y/beta1 + z*(1-1/beta1)` (was inverted) on both `RAdamScheduleFree` and `ScheduleFreeWrapper` |
| `262ca2e` | EriDiffusion | Phase 2c `--init-lokr-norm` wired across the remaining 4 LoKr-capable models: acestep / ernie / klein / sd35 (zimage, qwenimage, wan22, anima, chroma, sdxl already had it) |
| `9e078e1` | EriDiffusion | `--conv-rank` / `--conv-alpha` CLI surface across 10 LoKr-capable trainers (acestep, anima, chroma, ernie, klein, qwenimage, sd35, sdxl, wan22, zimage). Plumbed into `LycorisBundleConfig`; **not yet consumed by any adapter constructor** — fields are inert until a trainer adds a conv-target build path |

## Two open signals to root-cause

### Signal A: EDv2 klein9b loss runs high vs ai-toolkit

Pinned by running the **same hyperparameters** on both stacks on the
same dataset:

| | EDv2 klein9b | ai-toolkit klein9b |
|---|---|---|
| Model | flux-2-klein-base-9B (BF16) | flux-2-klein-base-9B (BF16) |
| Algo | LoKr, factor=4, full-rank (rank=9999999999, alpha=9999999999) | LoKr, factor=4, `lokr_full_rank: true` (→ same rank/alpha internally) |
| Conv | conv=32, conv_alpha=24 | conv=32, conv_alpha=24 |
| Optimizer | AdamW | AdamW |
| lr | 3e-4 with 100-step linear warmup | 3e-4 with `constant_with_warmup` 100 steps |
| Dataset | `/home/alex/eri2` (118 imgs) | `/home/alex/eri2` (118 imgs) |
| Resolution | 512 cached, multi-bucket | multi-bucket 512/768/1024 |
| Init perturbation | `--init-lokr-norm 0.001` | ai-toolkit has no equivalent |
| Quantize | none (BF16) | `quantize: false` (BF16) |
| Per-step loss | bouncing 0.4-1.1, mean ~0.74 | bouncing 0.45-0.71, mean ~0.59 |

User's stated envelope from prior local runs: **0.6 ± 0.2 across small
datasets**. ai-toolkit matches; EDv2 sits ~25% high.

Possible causes, not yet bisected:
1. `--init-lokr-norm 0.001` perturbation biasing the model away from
   base initialization (ai-toolkit doesn't apply this)
2. EDv2's `init_perturbed_normal_lokr` math diverging from
   SimpleTuner's reference (`peft_init.py`) in some subtle way
3. EDv2's klein LoKr forward differing from ai-toolkit's at the
   adapter level (Kronecker product math, scale formula)
4. Different optimizer hyperparameters (β1, β2, eps, weight_decay)
   - need to compare both configs side-by-side

**Action for next session:** run EDv2 klein9b with
`--init-lokr-norm 0.0` first (no perturbation). If loss matches
ai-toolkit → init is the culprit. If still elevated → bisect by
swapping LoKr algo for vanilla LoRA on same hyperparams (rank=16) and
compare to ai-toolkit LoRA at the same settings.

### Signal B: EDv2 klein9b runs ~2× slower per step

| | s/step (batch=1, klein9b, full-rank LoKr) |
|---|---|
| ai-toolkit | ~6 s/step |
| EDv2 | ~11 s/step |

Prior `BASELINE_OT_KLEIN9B_ALINA.md` doc recorded EDv2 at 4.50 s/step
batch=1 vs OneTrainer 2.80 s/step batch=2 (alina baseline, rank=16).
That documented gap was 1.6× per step on rank=16 LoRA. Today's gap is
~1.9× on full-rank LoKr — slightly wider.

What I checked but DIDN'T profile:
- ai-toolkit venv: `accelerate 1.12.0`, `bitsandbytes 0.49.1`,
  `triton 3.3.0`. **No flash-attn, no sage-attn, no xformers** in the
  pip list. The flux2 extension module doesn't import either.
- So ai-toolkit's klein path is plain PyTorch SDPA + autograd + AdamW.
  Nothing exotic.

That means the 2× gap is **NOT** explained by ai-toolkit having a
custom attention backend EDv2 lacks. Real root cause unknown without
profiling. Candidates worth `nsys` profiling on EDv2:
- Excess autograd-recorded nodes per op vs PyTorch's C++ autograd
- flame-core matmul/gemm cublasLt config selection
- Block offload streaming (full-block-per-step vs ai-toolkit's
  `layer_offloading` mechanism — different implementation)
- BF16 dtype-conversion overhead in flame-core's strict-BF16 path
- AdamW kernel — flame-core has a multi-tensor fused AdamW
  (`flame-core/src/optimizers/multi_tensor_adam.rs`) but I didn't
  verify it's the one being dispatched on klein

**Action for next session:** `nsys profile -o klein_edv2.nsys-rep`
one EDv2 klein9b train step. Compare top-10 kernels by total time
against the same one-step trace from ai-toolkit. Wherever the EDv2
trace spends more time is where the gap lives.

## What did NOT go right today

- Eval-mode formula fix: shipped but unrelated to the gating bug user
  was seeing on Z-Image. The actual cause of "samples look broken at
  step N" was `--init-lokr-norm 1.0` being too aggressive (perturbed
  the model into noise). Dropped to 0.01 → clean samples. Both fixes
  were necessary; only the init-magnitude fix gated visible quality.
- RMSNorm swap: shipped, bit-exact parity proven, but did NOT
  fix or affect the klein 9B loss gap (klein doesn't use Z-Image's
  `primitive_rms_norm` wrapper). Pure speedup commit for Z-Image only.
- conv-rank / conv-alpha CLI: surfaces are in but no trainer
  currently builds conv adapters. Today's run set --conv-rank 32
  --conv-alpha 24 on klein9b but klein's bundle is linear-only so
  these were saved into LycorisBundleConfig metadata and not consumed.
- Thermal cap watchdog fired multiple times — 78°C is below klein 9B
  sampling at 1024² under sustained load. Stage 2 baseline samples
  redesigned to render one prompt at a time with cool-down between.
  Eventually dropped step-0 baseline sample entirely since klein
  trainer already saves before each periodic sample.
- No EDv2 klein9b training run completed today. ai-toolkit 100-step
  comparison launched but not finished at handoff time.

## How to pick this up

1. Read this doc + `BASELINE_OT_KLEIN9B_ALINA.md` (gives the
   prior-measured perf baseline for context).
2. Tackle Signal A (loss) before Signal B (perf). A wrong loss
   means we're training the wrong gradient direction; perf is
   moot if the result is bad.
3. Signal A first experiment: rerun EDv2 klein9b 100 steps with
   `--init-lokr-norm 0.0`. Compare per-step loss against today's
   ai-toolkit log at
   `/home/alex/EriDiffusion/EriDiffusion-v2/output/eri2_klein9b_lokr_512/aitoolkit_klein9b_100step.log`.
4. Signal B: only after Signal A is resolved or bisected. nsys
   profile one step on each stack, diff kernel timings.

## Open backlogs unchanged from yesterday's handoff

See `HANDOFF_2026-05-10_FEATURE_UNIFICATION_FOLLOWUP.md` — none of
those 11 backlogs were touched today.

## Files added today

- `/home/alex/EriDiffusion/flame-core/tests/rms_norm_vs_primitive_zimage.rs`
  (parity test, 5 cases, all pass)
- `/home/alex/EriDiffusion/EriDiffusion-v2/output/eri2_klein9b_lokr_512/run_pipeline.sh`
  (klein9b training pipeline with thermal watchdog at 78°C)
- `/home/alex/EriDiffusion/EriDiffusion-v2/output/eri2_klein9b_lokr_512/aitoolkit_klein9b_100step.log`
  (ai-toolkit comparison run log)
- `/home/alex/ai-toolkit/config/eri2_klein9b_lokr_full_100step.yaml`
  (ai-toolkit config matching EDv2 hyperparams)

## What I owe the next session
This handoff IS the owed thing. It records the gap honestly without
inventing root causes I didn't verify. The two signals are real,
neither is root-caused, and the path forward (Signal A → 0-init
control, Signal B → nsys profile) is concrete.
