# Session handoff — 2026-05-09

Long context-rich session. This doc is the canonical state-of-play for the
next session — read top-to-bottom before doing anything.

---

## TL;DR (60-second read)

**Wins**:
- HiDream-O1 inference port (1024² + 2048² verified, chunked SDPA shipped)
- Chroma trainer: 2 gradient bugs found + fixed → 418/418 LoRA-Bs converging.
  3000-step run completed step 2000 (killed at 2244), 4 checkpoints saved.
- Wan2.2 trainer: full forward port + 3-agent audit + 12 audit-driven fixes
  shipped across Phases 0-6
- Cross-model video utilities module shipped (decode/prep/shape)
- LyCORIS Phase 2b plumbing for 3 trainers (chroma + flux + klein)
- LyCORIS upstream gradient-isolation bug found + fixed in lycoris-rs +
  EDv2 bridge — verified by regression test
- Qwenimage same-class chroma-pattern bug found + fixed — explains the
  prior session's "qwen step-1000 visual imprint inconclusive"
- Misc: Wan22 LoRA PEFT-format saves, sample_chroma --lora wired,
  train_chroma --resume-lora wired, AdamW8bit wired in Wan trainer

**Open blockers**:
- 🔴 `CUDA_ERROR_MISALIGNED_ADDRESS` in flame-core autograd backward when
  LoCon F32 matmul runs. Plain `--algo lora` works fine. Blocks any real
  LyCORIS-algo training even with all the Phase 2b plumbing in place.
- 🟡 6 trainers still need LyCORIS Phase 2b plumbing (ernie, sdxl, sd35,
  zimage, qwenimage, anima)
- 🟡 Wan trainer is UNTESTED on GPU (chroma had it for the entire session)
- 🟡 Qwen + Chroma-pattern fix not yet GPU-validated
- 🟡 Chroma samples at step 2000 are visually poor — convergence verified
  but quality investigation deferred (timestep dist / latent norm / rank /
  sample cfg are likely culprits)

**Currently working on**: nothing active. Idle, awaiting direction.

---

## Section 1 — Done this session (chronological-ish)

### 1.1 HiDream-O1-Image (inference-flame repo)
- Phase 2a-2d port already shipped before session start; this session
  resolved the OOM-at-forward blocker
- Real fix: `flame_core::AutogradContext::set_enabled(false)` in
  `src/bin/hidream_o1_infer.rs::main()`. The original "OOM" was a
  28-step compute-graph leak, not a per-step memory issue.
- Added `--allow-any-resolution` smoke flag, `HIDREAM_MEM_LOG=1`
  instrumentation
- Implemented chunked SDPA: `src/models/hidream_o1/decoder.rs::chunked_sdpa`
  (default chunk 1024, env `HIDREAM_SDPA_CHUNK`). Layer 0 SDPA peak
  transient: 12.6 GB → ~30 MB (~400× lower). Wall: 41 min → 39 min at 2048².
- Verified outputs: `inference-flame/output/hidream_smoke_1024.png`,
  `hidream_2048_bokeh.png`, `hidream_2048_bokeh_chunked.png`
- Commits in `inference-flame` repo: `ade8378` (port), `9c2cfe4` (cleanup)

### 1.2 Chroma trainer fix (EDv2 master)
- 50-step smoke audit revealed 304 of 418 LoRA-B tensors zero — chroma
  pattern bug
- Bug 1: `rms_norm_head` (chroma.rs:1526) called inference-only
  `cuda_ops_bf16::rms_norm_bf16`. Fixed → `flame_core::norm::rms_norm`.
- Bug 2: `forward_lora` (chroma.rs:833) wrapped dual-block unpack +
  dual→single boundary cat in `AutogradContext::no_grad`. Fixed →
  removed guards.
- Defensive: inlined `attn_split_txt_img_bf16` as
  narrow+permute+contiguous+reshape (didn't actually fix anything; kept
  for clarity).
- 50-step v4 smoke verified: 418/418 LoRA-B non-zero.
- Commit `0a51a5e` on EDv2 master.

### 1.3 Wan 2.2 trainer port (Wan22 redo from handoff)
**Forward port** (~960 LOC):
- New module `eridiffusion-core/src/models/wan22_fwd/{mod,rope,head,block,forward}.rs`
- Port of `flame-diffusion-archive/wan-trainer/src/forward_impl/`
- `Wan22Model::forward` now dispatches to `wan22_fwd::forward_with_lora`

**Encoders** (~1870 LOC):
- `eridiffusion-core/src/encoders/wan22_vae.rs` (1259 LOC, Wan2.2 VAE
  encoder for TI2V-5B)
- `eridiffusion-core/src/encoders/umt5.rs` (611 LOC, UMT5-XXL with
  per-layer rel-pos bias)
- `Cargo.toml` added `half = "2"` dep

**Audit-driven fixes (Phases 0-6, 12 items)**:
- C1 — 5D→4D shape mismatch (squeeze in trainer + sampler)
- C3 — variant-aware VAE stride (5B uses /16, 14B uses /8)
- C4/M5 — `sample_wan22 --cfg` real CFG implementation +
  `--low-lora`/`--high-lora` real loading
- C2 — real Wan VAE + UMT5 in `prepare_wan22` (image input verified)
- M1 — BlockOffloader integration via `Wan22Model::load_swapped` +
  checkpoint_offload pattern + `--offload` flag
- H1 — cross-attn text padding mask threaded from cache through forward
- H3 — per-batch e0+e (saves ~16 GB at video seq lengths; broadcasts
  zero-impact)
- H5 — Wan LoRA save format → PEFT-compliant
  `blocks.{i}.{target}.lora_*.weight` with legacy-format back-compat
  on load
- H7 — `--max-grad-norm` CLI flag (default 1.0; SimpleTuner uses 0.1)
- H8 — `Tensor::randn_seeded` per-step derived seed (reproducible)
- H10 — cross-trainer grad-coverage diagnostic
  (`training/grad_coverage.rs`) wired into wan trainer at step 1 and
  every `--grad-coverage-every` steps
- M3 — dropped redundant outer record on `WanRope::apply`
  (`rope_fused_bf16` records `Op::RoPePrecomputed` itself)
- M4 — AdamW8bit fully wired (was warn-and-fallback)

**Cross-model video module** (Phase 6, user-requested for ltx2/anima
reuse):
- `eridiffusion-core/src/video/{mod,decode,prep,shape}.rs` (~500 LOC)
- `decode_to_tensor` — image (image crate) + video (ffmpeg subprocess) →
  `[1, 3, F, H, W]` BF16 in `[-1, 1]`
- `walk_dataset` + `snap_to_wan_frame_count` (Wan VAE temporal constraint)
- `latent_dims`/`pixel_dims`/`pixel_frames` shape converters
- `prepare_wan22` wired to use it — mp4/webm/mov decode now works

**3-agent audit** (builder + bug-fixer + skeptic) reports preserved:
`/tmp/wan_audit_{builder,bugfixer,skeptic,synthesis}.md`

**Commit**: `dd19ade` on EDv2 master (24 files, +4386/-272). The commit
message includes a `⚠️ WAN22 IS UNTESTED ON GPU` warning at the top.

### 1.4 Chroma 3000-step LoRA run + samples
- Trained on `boxjana_chroma_edv2_512` cache (22 samples, 512²).
- Killed at step 2244/3000 (out of 3000 target) to free GPU for sampling
- 4 saved checkpoints: step500/1000/1500/2000 in
  `output/chroma_boxjana_3k/`
- 3 sample images at 1024² generated using step2000 LoRA, 3 ladies
  prompts in caption-style. Pushed to
  https://github.com/CodeAlexx/Samples (commit `81df81b`):
  - `chroma_boxjana_step2000_01_mediterranean_balcony.png`
  - `chroma_boxjana_step2000_02_moody_library.png`
  - `chroma_boxjana_step2000_03_tokyo_neon_alley.png`
- **User assessment**: "convergence verified but samples terrible".
  Quality investigation deferred. Likely culprits:
  - Timestep distribution (chroma trainer uses `uniform`, but chroma
    upstream typically uses `logit_normal` w/ shift)
  - Latent norm (caches are RAW posterior; trainer applies SHIFT/SCALE
    per H3 audit fix — verify scale value)
  - Rank/alpha at 16/16 vs SimpleTuner's typical 32+
  - Sample-time `--cfg 3.6` on 26 steps may be wrong band
  - Training resolution 512 vs sample resolution 1024

### 1.5 LyCORIS Phase 2b plumbing — 3 trainers
**Trainers wired** (commit `e3c00bb` on EDv2 master, ~1400 LOC across 7 files):
- chroma: `ChromaLoraBundle` HashMap value type → `Arc<dyn AdapterModule>`
- flux: parallel-bundle design (`bundle: FluxLoraBundle` legacy +
  `lycoris_bundle: Option<FluxLycorisBundle>` mutually exclusive)
- klein: parallel-field design (`Vec<LoRALinear>` legacy +
  `Option<Vec<Arc<dyn AdapterModule>>>` for new). Klein's split-QKV
  preserved.

**New CLI flags** in all three: `--algo`, `--lokr-factor`,
`--oft-block-size`, `--oft-neumann-terms`, `--use-tucker`,
`--decompose-both`, `--dora`, `--dora-wd-on-out`.

**Legacy `--algo lora` byte-identical** in all three. Default routes to
the original LoRALinear constructor.

### 1.6 LyCORIS upstream gradient-isolation fix
**Bug**: `LycorisLinear::to_parameters()` (adapter.rs:335) wrapped cloned
Tensors in fresh Parameters. AdamW's `set_data` mutated the wrapper but
not the LycorisAdapter's internal `down/up/...` fields. Forward read
internal fields → optimizer mutations silently dropped. Same shape as
the chroma rms_norm bug.

**Fix**: lycoris-rs algorithms migrated leaf storage from `Tensor` →
`flame_core::parameter::Parameter`:
- `locon`: down, up, mid?
- `loha`: w1a, w1b, w2a, w2b, t1?, t2?
- `lokr`: w1?, w1a?, w1b?, w2?, w2a?, w2b?, t2?
- `full`: diff, diff_b?
- `oft`: blocks
- New `LycorisModule::parameters_handles() -> Vec<Parameter>` trait method
- Bridge: `LycorisLinear::to_parameters` / `LycorisBundle::to_parameters`
  in EDv2 now call `parameters_handles()` and clone the live Parameter
  handles.

**Verified**: regression test
`tests/autograd_smoke.rs::locon_set_data_through_handle_propagates_to_forward`
PASSES (pre-fix would show 0).

**Commits**:
- lycoris-rs: `aaebf5c` on `salvage-flame-2026-04` branch (Eri-Lycoris repo)
- EDv2: `43cd4d2` on master

**Side-finding**: end-to-end `train_chroma --algo locon` 5-step smoke
hits `CUDA_ERROR_MISALIGNED_ADDRESS` in flame-core autograd backward.
Plain `--algo lora` completes cleanly. Independent of the Parameter
migration — see Section 2 open items.

### 1.7 Qwenimage same-class bug fix
- Static audit found `cuda_ops_bf16::rms_norm_bf16` AND
  `cuda_ops_bf16::layer_norm_bf16` in qwen's `dual_stream_block_iflame`
  (the trainer's hot block).
- Bug: every QK head-norm + every img/txt pre-attn AND pre-FFN norm cuts
  gradient. Far worse scope than chroma's bug.
- Fixed identically: rms → `flame_core::norm::rms_norm`, layer_norm →
  hand-rolled F32 manual mean/var/rstd (mirror `wan22_fwd/block.rs`).
- Cross-check: `dual_stream_block_standalone` (offload-path block,
  qwenimage.rs:1329) was already clean — uses
  `flame_core::layer_norm::layer_norm` + autograd-aware `rms_norm_per_head`.
- **Most likely root cause** of the night-handoff "qwen step-1000 visual
  imprint inconclusive" finding.
- Commit `d5f2cb2` on EDv2 master.
- NOT GPU-validated yet.

### 1.8 Other small fixes
- `train_chroma --resume-lora` wired (was stub bail) — uses new
  `ChromaLoraBundle::load_weights` in-place loader
- `sample_chroma --lora` flag wired — uses new
  `ChromaLoraBundle::load_from_safetensors`
- Cross-trainer grad-coverage utility (`training/grad_coverage.rs`) —
  detects partial coverage at step 1 + every save_every. Currently wired
  in wan trainer; could be added to other trainers as one-liner.

### 1.9 Repo hygiene
- inference-flame: pushed (`9c2cfe4`)
- EDv2 master: 4 commits pushed (`0a51a5e` chroma fix, `dd19ade` wan/video,
  `e3c00bb` LyCORIS Phase 2b chroma+flux+klein, `43cd4d2` LyCORIS bridge fix,
  `d5f2cb2` qwen norm fix)
- Eri-Lycoris `salvage-flame-2026-04`: pushed (`aaebf5c`)
- Samples repo: pushed (`81df81b`)

---

## Section 2 — Open / TODO

### 2.1 GPU validation — needs running on a clean GPU
**Priority A — most critical**:
1. **Wan trainer GPU smoke** (TI2V-5B variant, 50-step + LoRA-B audit). Wan
   trainer has not run end-to-end on GPU at any point this session;
   chroma had the GPU the whole time. Code is forward-port-ready but
   *unverified*.
   ```
   cd /home/alex/EriDiffusion/EriDiffusion-v2
   ./target/release/prepare_wan22 --variant ti2v_5b --size 256 ...
   ./target/release/train_wan22 --steps 50 --grad-coverage-every 1 \
     --variant ti2v_5b --low-noise <ckpt> --offload ...
   ```

2. **Qwen GPU validation** (50-step + LoRA-B audit) — verifies today's
   `dual_stream_block_iflame` fix (commit `d5f2cb2`). Compare to a
   pre-fix run if any saved checkpoints exist on disk.

3. **`CUDA_ERROR_MISALIGNED_ADDRESS` triage** in flame-core autograd
   backward for LoCon F32 matmul. The agent's report:
   "aborts mid-step-0 backward inside `flame_core::autograd::compute_gradients`".
   Plain `--algo lora` works. Bisect: is it the F32 matmul kernel? The
   tucker/loha-style decomposition path? Reproducer:
   ```
   ./target/release/train_chroma --algo locon --steps 5 ...
   ```
   This is the actual blocker for any real LyCORIS-algo training.

**Priority B — quality/parity**:
4. **Chroma sample-quality investigation**. Filed for later. Suspects
   ranked: timestep distribution, latent norm value, rank/alpha sizing,
   sample-time cfg/steps, train-vs-sample resolution mismatch.

### 2.2 LyCORIS Phase 2b — 6 remaining trainers
Same plumbing pattern as chroma+flux+klein (legacy `--algo lora`
byte-identical, new algos via `Box<dyn AdapterModule>`):
- ernie
- sdxl
- sd35
- zimage
- qwenimage
- anima

Per-trainer ~100 LOC mostly mechanical. Recommended dispatch: 2 batches
of 3 agents (max-3-concurrent rule). **DO NOT dispatch yet** — first
fix `CUDA_ERROR_MISALIGNED_ADDRESS` so the agents have a working
end-to-end smoke target.

### 2.3 Same-class chroma-pattern audit pending for
The `cuda_ops_bf16::rms_norm_bf16` / `cuda_ops_bf16::layer_norm_bf16`
inference-only kernel in trainer hot paths surfaced in chroma AND qwen
this session. Worth a sweep:
- anima
- ernie
- sdxl
- sd35
- ltx2
- acestep
- zimage
- klein (already known-good — 240/240 in 1000-step LoRA, but verify the
  Phase 2b plumbing didn't regress)

Quick grep recipe per trainer:
```
grep -nE "cuda_ops_bf16::(rms_norm_bf16|layer_norm_bf16|silu)" \
  crates/eridiffusion-core/src/models/<model>.rs
```
If any hit lives inside a trainer-side forward (vs an inference-only
function), it's the bug.

### 2.4 In-loop sampling — 5 trainers without it
Discovered: handoff said "Qwen + Anima in-loop sampling" but qwenimage
ALREADY has it. Real gap:
- chroma — none. Has `sample_chroma --lora` (tested).
- anima — none. Has standalone sampler.
- flux — none.
- sdxl — none.
- acestep — none.

Pattern to copy: `train_qwenimage.rs::qwenimage_inline_sample`. ~80 LOC
each. Pre-encode 1-2 prompts, teardown activation pool + clear CUDA
mempool, call `<sampler>::sample_image(model, ...)` in-process.

### 2.5 Wan2.2 audit synthesis open items (Phases 4-6 leftovers)
- M2 — GPU patchify (port host loop to Conv3d-as-linear; perf, not
  correctness)
- M8 — gradient accumulation + gradient checkpointing
- M9 — UniPC scheduler (sampler currently Euler only)
- M10 — chunked FFN (memory)
- M11 — image conditioning (I2V first/last frame) — large port
- H4 — flame-core FP8 runtime DType (architectural; blocks 14B+14B on
  24 GB)
- H6 — Diffusers base-checkpoint key adapter (HF cross-load)

### 2.6 Phase B — modern features parity across all 12 trainers
DEFERRED at user request from prior session. Doc:
`EriDiffusion-v2/PARITY_SIMPLETUNER.md`. Common gaps across trainers:
- LR scheduler family (cosine_with_restarts, polynomial)
- timestep_bias strategies
- min-SNR / Huber loss family
- multi-resolution noise
- caption_dropout + null-text-cache
- validation harness (held-out cache + side-RNG)
- EMA + EMA-validation-swap
- Image augs (flip, brightness, contrast)
Some trainers have full Phase 0+ surface (chroma, qwen, klein), others
lag.

### 2.7 Original handoff items — still open
1. ✅ ~~Wan2.2 redo~~ — DONE (untested)
2. 🟡 LyCORIS Phase 2b — 3/9 done
3. ⏸ Phase B — DEFERRED at user request
4. 🟡 Qwen + Anima in-loop sampling — qwen has it, anima needs it
5. 🟡 Qwen re-confirm — likely now traced to today's qwen norm fix; rerun
   after GPU validation

---

## Section 3 — Currently working on

**Nothing active.** All commits pushed. Background agents all completed.
Idle, awaiting direction.

---

## Section 4 — File paths and commit reference

### Repos and branches
| Repo | Branch | Latest commit | Url |
|---|---|---|---|
| `EriDiffusion/inference-flame` | master | `9c2cfe4` | github.com/CodeAlexx/inference-flame |
| `EriDiffusion/EriDiffusion-v2` | master | `d5f2cb2` | github.com/CodeAlexx/EriDiffusion |
| `EriDiffusion/eri-lycoris` (= lycoris-rs) | salvage-flame-2026-04 | `aaebf5c` | github.com/CodeAlexx/Eri-Lycoris |
| `Samples` | main | `81df81b` | github.com/CodeAlexx/Samples |

### Critical files modified this session (EDv2)
- `crates/eridiffusion-core/src/adapter.rs` — LycorisLinear bridge to Parameter handles
- `crates/eridiffusion-core/src/lycoris.rs` — same fix at bundle layer
- `crates/eridiffusion-core/src/models/chroma.rs` — gradient bugs fixed,
  PEFT save/load, LyCORIS Phase 2b plumbing
- `crates/eridiffusion-core/src/models/flux.rs` — LyCORIS Phase 2b plumbing
- `crates/eridiffusion-core/src/models/klein.rs` — LyCORIS Phase 2b plumbing
- `crates/eridiffusion-core/src/models/wan22.rs` — Wan2.2 model wrapper, BlockOffloader, PEFT save
- `crates/eridiffusion-core/src/models/wan22_fwd/{mod,rope,head,block,forward}.rs` — Wan2.2 forward port
- `crates/eridiffusion-core/src/models/qwenimage.rs` — chroma-pattern norm fix
- `crates/eridiffusion-core/src/encoders/wan22_vae.rs` — Wan2.2 VAE encoder
- `crates/eridiffusion-core/src/encoders/umt5.rs` — UMT5-XXL
- `crates/eridiffusion-core/src/video/{mod,decode,prep,shape}.rs` — cross-model video module
- `crates/eridiffusion-core/src/training/grad_coverage.rs` — partial-coverage warn utility
- `crates/eridiffusion-cli/src/bin/train_wan22.rs` — Wan trainer with all audit fixes
- `crates/eridiffusion-cli/src/bin/sample_wan22.rs` — Wan sampler with CFG + LoRA
- `crates/eridiffusion-cli/src/bin/prepare_wan22.rs` — real prep with VAE+UMT5+video decode
- `crates/eridiffusion-cli/src/bin/train_chroma.rs` — `--resume-lora` + LyCORIS plumbing
- `crates/eridiffusion-cli/src/bin/train_flux.rs` — LyCORIS plumbing
- `crates/eridiffusion-cli/src/bin/train_klein.rs` — LyCORIS plumbing
- `crates/eridiffusion-cli/src/bin/sample_klein.rs` — bundle field add
- `crates/eridiffusion-cli/src/bin/sample_chroma.rs` — `--lora` flag

### lycoris-rs files modified
- `lycoris-rs/src/algorithms/{locon,loha,lokr,full,oft}.rs` — Parameter migration
- `lycoris-rs/src/lib.rs` — `LycorisModule::parameters_handles` trait method
- `lycoris-rs/src/loader.rs` — `Parameter::new(...)` wraps loaded Tensors
- `lycoris-rs/tests/{autograd_smoke,smoke,parity}.rs` — updated direct struct literals

### Outputs / data
- Chroma 3k LoRA checkpoints: `EriDiffusion-v2/output/chroma_boxjana_3k/chroma_lora_step{500,1000,1500,2000}.safetensors` (~262 MB each)
- Chroma 3k sample PNGs: `EriDiffusion-v2/output/chroma_boxjana_3k_samples/sample_{001,002,003}.png`
- HiDream PNGs: `inference-flame/output/hidream_{smoke_1024,2048_bokeh,2048_bokeh_chunked}.png`
- Boxjana cache: `/home/alex/datasets/boxjana_chroma_edv2_512` (22 samples)
- Audit reports: `/tmp/wan_audit_{builder,bugfixer,skeptic,synthesis}.md` (will be wiped by next reboot)

### Memory entries created/updated this session
- `feedback_edv2_output_dir.md` — EDv2 trainer outputs go in EDv2/output/
- `project_chroma_lora_broken_2026-05-09.md` — chroma fix details (was "broken", now "FIXED")
- `project_hidream_o1_2026-05-09.md` — HiDream port verified
- `project_wan22_audit_decisions_2026-05-09.md` — 5 user decisions on wan audit Q's

---

## Section 5 — Decisions already locked in

(Prior commits depend on these — don't reverse without changing the
relevant code.)

| Decision | Made when | Effect |
|---|---|---|
| Wan22 LoRA save format = PEFT-compliant `blocks.{i}.{target}.lora_*.weight` | this session | New saves use this; legacy `lora_wan_blocks_*` still loadable. |
| Wan22 target hardware = 24 GB → focus TI2V-5B, defer 14B until flame-core FP8 | this session | Audit-derived |
| Wan22 AdamW8bit wired (not silent fallback) | this session | Per `feedback_wan22_quant_exception` |
| Wan22 sample CFG implemented + `--negative-embed` flag | this session | Audit C4 |
| Wan22 per-batch e0 (not per-token) | this session | Saves ~16 GB at video seq lengths |
| EDv2 trainer outputs go in `EriDiffusion-v2/output/` (not inference-flame/output) | this session | Memory pinned |
| LyCORIS legacy `--algo lora` path BYTE-IDENTICAL to pre-Phase-2b | this session | Required for resume-from-checkpoint compatibility |

---

## Section 6 — Suggested order for next session

1. **Triage `CUDA_ERROR_MISALIGNED_ADDRESS`** in flame-core autograd
   backward (LoCon F32 matmul path). This is the actual end-to-end
   blocker for non-LoRA algo training. Without this, Phase 2b plumbing
   in chroma+flux+klein is theoretical.
2. **GPU-validate today's qwen fix** — 50-step run on the pre-existing
   qwen alina cache, audit LoRA-B zeros. Should now be 100% non-zero.
3. **GPU smoke for Wan22 trainer** — TI2V-5B 50-step run. First time
   the wan trainer touches GPU.
4. **Same-class audit sweep** for the remaining 7 trainers (Section 2.3).
5. **LyCORIS Phase 2b for 6 remaining trainers** — once #1 lands, dispatch
   2 batches of 3 agents.
6. (Lower priority) In-loop sampling for chroma, anima, flux, sdxl,
   acestep.

---

## Section 7 — Caveats / things to NOT do

- **Don't push wan trainer commits as "ready for users"** — the trainer
  has never run on GPU. Mark as untested/code-review-only until smoke
  passes.
- **Don't dispatch LyCORIS Phase 2b agents for the remaining 6 trainers
  yet** — wait until `CUDA_ERROR_MISALIGNED_ADDRESS` is fixed so the
  agents can verify their work end-to-end.
- **Don't change the legacy `--algo lora` code path** in chroma/flux/
  klein — byte-equivalence is what makes the Phase 2b commits safe.
- **Don't kill chroma 3k unless you have a step{N} checkpoint you're
  willing to live with** — the trainer doesn't auto-save on SIGTERM.
  (We killed it at step 2244 with step2000 saved; that worked because
  `--save-every 500` had landed step2000 60 steps prior.)
- **Don't trust the chroma sample quality as a parity benchmark** —
  user explicitly noted "convergence verified but samples terrible";
  that's a separate quality investigation, not a trainer bug.
