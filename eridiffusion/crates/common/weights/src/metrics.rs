use serde::Serialize;
use std::io::Write;

#[derive(Serialize)]
struct StepLog<'a> {
    step: usize,
    loss: f32,
    grad_norm: f32,
    seconds: f32,
    vram_gb: Option<f32>,
    note: Option<&'a str>,
}

pub fn log_step_metrics(step: usize, loss: f32, grad_norm: f32, seconds: f32) {
    let vram = super::registry::vram_used_gb().ok();
    let rec = StepLog { step, loss, grad_norm, seconds, vram_gb: vram, note: None };
    if let Ok(line) = serde_json::to_string(&rec) {
        let _ = writeln!(&mut std::io::stdout(), "{}", line);
    }
}
