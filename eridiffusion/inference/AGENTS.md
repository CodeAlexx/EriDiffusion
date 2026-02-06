# EriDiffusion Inference – Agent Guardrails

## Must-Do Rules
- **BF16 storage only.** With `STRICT_BF16=1` every tensor created on device stays BF16. If you think you need FP32 staging, stop and escalate—do not land a workaround.
- **Stream every host copy.** Use the shared dumper (`PARITY_MAX_DUMP_MB`, default 16) to pull BF16 slices → pinned host → write. Never allocate a full-tensor FP32 clone for logging or parity.
- **Respect pressure valves.** Honor `TILE_VRAM_FRACTION` and `ARENA_TEMP_SOFTCAP_MB`. Kernels and graph glue must cooperate with tiling; large concats/attn passes that ignore the caps are regressions.
- **Mirror FlameCore semantics.** Any layout, dtype, or allocator tweak must land in FlameCore first (or simultaneously) and be mirrored into `FlameCore_clean_repo2` / C++ helpers before you close the task.

## Workflow Expectations
- Compose existing kernels (`bf16_repeat`, chunked concat, cuBLASLt BF16) instead of reintroducing ad-hoc compute. If a kernel gap exists, document it and get FlameCore to own the fix.
- Keep parity artifacts (`parity_output_dir`) focused: emit hashes + shape metadata, not raw FP32 dumps. Log the exact env (`STRICT_BF16`, `PARITY_MAX_DUMP_MB`, `TILE_VRAM_FRACTION`) alongside each capture.
- Whenever you touch memory-heavy paths (SDPA, VAE decode, latents concat), attach allocator stats (`device_fp32_alloc_bytes_during_infer`, arena high watermark) to the PR or doc update.

## Validation Checklist
- `cargo fmt` and `cargo clippy --all-targets -- -D warnings`.
- `cargo test --manifest-path inference/Cargo.toml --features cuda,bf16_u16 -- --nocapture strict_bf16_harness`.
- SD‑3.5 / SDXL smoke: `cargo test -p inference --features cuda,bf16_u16 -- --nocapture sd35_10steps` with `STRICT_BF16=1`, ensuring peak VRAM stays within the documented budget.
- If parity flagging is enabled, diff the produced artifacts against the C++ reference before merging.

## Escalation
Any attempted FP32 staging, ignored tiling cap, or unexplained OOM must be recorded in `docs/FLAME_SD35_BLOCKER_PLAN.md` and tagged for the FlameCore runtime team. Do **not** merge partial fixes—document, escalate, and wait for an approved plan.
