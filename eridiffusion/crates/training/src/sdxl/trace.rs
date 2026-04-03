use std::sync::OnceLock;
use std::time::{Duration, Instant};

static LOAD_TRACE: OnceLock<bool> = OnceLock::new();

#[inline]
pub(crate) fn load_trace_enabled() -> bool {
    *LOAD_TRACE
        .get_or_init(|| matches!(std::env::var("SDXL_LOAD_TRACE").ok().as_deref(), Some("1")))
}

#[inline]
pub(crate) fn stage_start() -> Option<Instant> {
    load_trace_enabled().then(Instant::now)
}

#[inline]
pub(crate) fn log_stage(label: &str, duration: Duration) {
    if load_trace_enabled() {
        eprintln!("[sdxl-load] {:>6.3}s {}", duration.as_secs_f64(), label);
    }
}

#[inline]
pub(crate) fn log_transfer(label: &str, bytes: usize, duration: Duration) {
    if !load_trace_enabled() {
        return;
    }
    let gib = bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let secs = duration.as_secs_f64().max(f64::EPSILON);
    let gib_per_s = gib / secs;
    eprintln!(
        "[sdxl-load] {:>6.3}s {} — {:.2} GiB ({:.1} GiB/s)",
        duration.as_secs_f64(),
        label,
        gib,
        gib_per_s
    );
}
