//! Per-step progress log line + SerenityBoard metrics emission. Single source
//! of truth for all trainer binaries.
//!
//! Stdout format:
//! `[<tag>] step N/T | epoch e/E | loss X.XXXX | grad_norm X.XXXX | X.Xs/step | elapsed H:MM:SS | ETA H:MM:SS`
//!
//! `tag` is the trainer/run identifier, e.g. `u1-lora`, `zimage-lora`,
//! `klein-full-finetune`. An empty `tag` suppresses the bracket prefix.
//!
//! Board scalars (when writer is present): `loss/train`, `grad_norm`,
//! `lr/default`, `perf/steps_per_sec`. SerenityBoard's `training_reader.py`
//! reads these tags by default.

use std::time::Instant;

use crate::training::board::BoardWriter;

/// Log a single training step. Writes both human-readable stdout via `log::info!`
/// AND, when `board` is `Some`, writes scalars to SerenityBoard's SQLite DB.
///
/// `tag` is the run identifier shown in `[…]` brackets at the line start
/// (e.g. `SenseNova-U1-lora`, `zimage-lora`, `HiDream-O1-full-finetune`).
/// Empty string suppresses the bracket.
///
/// `step` is 0-indexed; the printed/stored step number is `step + 1`.
pub fn log_step(
    tag: &str,
    step: usize,
    total_steps: usize,
    dataset_len: usize,
    batch_size: usize,
    loss: f32,
    grad_norm: f32,
    lr: f32,
    t_start: Instant,
    board: Option<&BoardWriter>,
) {
    let bs = batch_size.max(1);
    let steps_per_epoch = (dataset_len.max(1) + bs - 1) / bs;
    let cur_epoch = (step / steps_per_epoch.max(1)) + 1;
    let total_epochs =
        (total_steps + steps_per_epoch.saturating_sub(1)) / steps_per_epoch.max(1);

    let elapsed = t_start.elapsed().as_secs_f32();
    let sec_per_step = elapsed / (step + 1) as f32;
    let elapsed_u = elapsed as u64;
    let eta_secs = (sec_per_step * (total_steps.saturating_sub(step + 1)) as f32) as u64;
    let (eh, em, es) = (elapsed_u / 3600, (elapsed_u % 3600) / 60, elapsed_u % 60);
    let (ah, am, as_) = (eta_secs / 3600, (eta_secs % 3600) / 60, eta_secs % 60);

    let prefix = if tag.is_empty() {
        String::new()
    } else {
        format!("[{tag}] ")
    };
    log::info!(
        "{prefix}step {}/{} | epoch {}/{} | loss {:.4} | grad_norm {:.4} | {:.1}s/step | elapsed {}:{:02}:{:02} | ETA {}:{:02}:{:02}",
        step + 1, total_steps, cur_epoch, total_epochs,
        loss, grad_norm, sec_per_step,
        eh, em, es, ah, am, as_,
    );

    if let Some(b) = board {
        let steps_per_sec = if sec_per_step > 0.0 { 1.0 / sec_per_step } else { 0.0 };
        b.log_scalars(
            (step + 1) as u64,
            &[
                ("loss/train", loss as f64),
                ("grad_norm", grad_norm as f64),
                ("lr/default", lr as f64),
                ("perf/steps_per_sec", steps_per_sec as f64),
            ],
        );
    }
}
