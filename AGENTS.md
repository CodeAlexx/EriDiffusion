# EriDiffusion Trainer – Agent Guardrails

## Hard Rules
- **GPU BF16 end-to-end.** With `STRICT_BF16=1` there is zero tolerance for FP32 device tensors or host staging. Do not revive `tensor.to_vec()` / `to_vec_f32()` or any CPU mirror of BF16 buffers.
- **Broadcast/repeat/index stay on device.** The helpers are already GPU-only; do not sneak in `to_vec()` or host iterators “for debugging”. Use the FlameCore kernels or open a blocker.
- **Keep runs tight.** Use focused tests (`cargo test -p training --features cuda,bf16_u16 -- --nocapture name_of_case`). If a loop exceeds 10 minutes, you’re running the wrong suite.
- **No surprise memory spikes.** Chunk parity dumps, enable tiling helpers, and watch `device_fp32_alloc_bytes_during_infer`—any non-zero value is a regression that must be fixed before merging.

## Safe Workflow
- Compose existing FlameCore ops; only request new kernels after proving the gap. Land changes in FlameCore → mirror → trainer.
- Keep SDXL/SD3 worklists (`docs/FLAME_SD35_*`) current. Every STRICT change or arena tweak should be recorded before you close the PR.
- Feature-gate diagnostics (`cfg!(feature = "strict_bf16")`). No ad-hoc env guards.

## Validation Checklist
- `cargo fmt`, `cargo clippy --all-targets -- -D warnings`, `cargo test --all-targets`.
- `cargo test --manifest-path crates/models/Cargo.toml --features cuda,bf16_u16 -- --nocapture sdxl_dtype_guard` with `STRICT_BF16=1`.
- Run `scripts/check_sdxl_dtype_guard.sh` (or equivalent) and log VRAM/time deltas; >0.5 % regression is a blocker.
- Record allocator metrics (`ARENA_TEMP_SOFTCAP_MB`, `TILE_VRAM_FRACTION`, `PARITY_MAX_DUMP_MB`) in the test log whenever you tweak them so the next agent can reproduce your settings.

## SDPA BF16 Tiling Plan (in progress)
1. **Remove FP32 fallback** – eliminate the SDPA `forward_f32` path under `STRICT_BF16` so oversize workspace requests error out instead of upcasting.
2. **Runtime tiler** – add a VRAM-aware planner (tile order Q → H → Dv) that allocates a single ring-buffer workspace and retries with smaller tiles on OOM.
3. **Integration & tests** – wire the planner into attention, expose counters/tripwires, and add CI contract tests (tight/loose budgets, strict BF16 parity, dtype checks).

## Escalation
If you must temporarily disable STRICT enforcement to unblock yourself, stop and escalate in `docs/FLAME_SD35_BLOCKER_PLAN.md`. We do not accept “temporary” BF16 relaxations—document the reproduction and wait for approval instead of merging hacks.
