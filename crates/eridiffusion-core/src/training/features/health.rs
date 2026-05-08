//! Phase 7: GPU health monitor + QK-clip diagnostic.
//!
//! `GpuHealthMonitor` polls NVML for temperature, ECC errors, and clock state
//! on a background thread and flips an abort flag when faults are detected.
//!
//! `QkClipLog` is infrastructure-only in Phase 7 — model integration is
//! Phase 7.5+. It records per-layer post-softmax max-abs attention magnitudes
//! and flushes to SerenityBoard at step boundaries.

use nvml_wrapper::{Nvml, error::NvmlError};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

/// GPU health monitor configuration. `spawn` consumes the struct and returns a
/// `HealthHandle` whose `abort_flag` flips to `true` when a fault is detected.
pub struct GpuHealthMonitor {
    pub nvml: Nvml,
    pub device_index: u32,
    pub check_interval_secs: u64,
    pub temp_threshold_c: u32,
    pub temp_sustained_secs: u64,
}

pub struct HealthHandle {
    pub abort_flag: Arc<AtomicBool>,
}

impl GpuHealthMonitor {
    /// Construct a monitor for `device_index`. Errors if NVML fails to init
    /// (driver missing, container without /dev/nvidia* mapped, etc.).
    pub fn new(device_index: u32) -> Result<Self, NvmlError> {
        Ok(Self {
            nvml: Nvml::init()?,
            device_index,
            check_interval_secs: 10,
            temp_threshold_c: 90,
            temp_sustained_secs: 30,
        })
    }

    /// Spawn a background thread that polls NVML at `check_interval_secs`. The
    /// returned `HealthHandle.abort_flag` flips to `true` and the thread exits
    /// when:
    ///   - GPU temperature ≥ `temp_threshold_c` for ≥ `temp_sustained_secs`
    ///   - NVML reports any uncorrected volatile ECC errors
    ///
    /// Transient NVML errors (e.g. driver hiccups) are logged and the loop
    /// continues — only confirmed faults trigger abort.
    pub fn spawn(self) -> HealthHandle {
        let abort = Arc::new(AtomicBool::new(false));
        let abort_clone = abort.clone();
        let cfg = (
            self.device_index,
            self.check_interval_secs,
            self.temp_threshold_c,
            self.temp_sustained_secs,
        );
        let nvml = self.nvml;
        thread::spawn(move || {
            let (device_index, check_interval_secs, temp_threshold_c, temp_sustained_secs) = cfg;
            let mut over_temp_since: Option<Instant> = None;
            loop {
                if abort_clone.load(Ordering::Relaxed) {
                    break;
                }
                match nvml.device_by_index(device_index) {
                    Ok(dev) => {
                        // Temperature
                        if let Ok(t) = dev.temperature(
                            nvml_wrapper::enum_wrappers::device::TemperatureSensor::Gpu,
                        ) {
                            if t >= temp_threshold_c {
                                let elapsed =
                                    over_temp_since.get_or_insert_with(Instant::now).elapsed();
                                if elapsed.as_secs() >= temp_sustained_secs {
                                    log::error!(
                                        "[health] GPU{} TEMP {}°C sustained {}s — aborting",
                                        device_index,
                                        t,
                                        temp_sustained_secs
                                    );
                                    abort_clone.store(true, Ordering::Relaxed);
                                    break;
                                }
                            } else {
                                over_temp_since = None;
                            }
                        }
                        // Uncorrected volatile ECC errors. NVML returns 0 on
                        // healthy hardware; any non-zero is a fault.
                        if let Ok(errs) = dev.total_ecc_errors(
                            nvml_wrapper::enum_wrappers::device::MemoryError::Uncorrected,
                            nvml_wrapper::enum_wrappers::device::EccCounter::Volatile,
                        ) {
                            if errs > 0 {
                                log::error!(
                                    "[health] GPU{} uncorrected ECC errors: {} — aborting",
                                    device_index,
                                    errs
                                );
                                abort_clone.store(true, Ordering::Relaxed);
                                break;
                            }
                        }
                    }
                    Err(e) => {
                        log::warn!("[health] NVML device error: {} — pausing checks", e);
                    }
                }
                thread::sleep(Duration::from_secs(check_interval_secs));
            }
        });
        HealthHandle { abort_flag: abort }
    }
}

/// Per-layer post-softmax max-abs attention magnitude tracker.
///
/// **Phase 7 status: infrastructure only.** No model wires this yet — the
/// `forward` paths in EDv2 models do not currently expose attention scores.
/// Future Phase 7.5+ work: hook into Klein/Z-Image/Flux attention blocks and
/// call `record(layer_idx, max_abs)` per step. The `log_to_board` helper
/// flushes via `BoardWriter::log_scalar`.
pub struct QkClipLog {
    pub max_per_layer: Vec<f32>,
}

impl QkClipLog {
    pub fn new(num_layers: usize) -> Self {
        Self {
            max_per_layer: vec![0.0; num_layers],
        }
    }

    /// Update the rolling max for `layer_idx`. Out-of-range indices are
    /// silently ignored so callers in tight loops don't pay a Result cost.
    pub fn record(&mut self, layer_idx: usize, max_abs: f32) {
        if layer_idx < self.max_per_layer.len() {
            self.max_per_layer[layer_idx] = self.max_per_layer[layer_idx].max(max_abs);
        }
    }

    /// Reset all per-layer max trackers to 0. Call between training steps if
    /// you want per-step rather than monotonic-rolling values.
    pub fn reset(&mut self) {
        for v in &mut self.max_per_layer {
            *v = 0.0;
        }
    }

    /// Flush per-layer max values to SerenityBoard as `qk_max/layer_<i>`.
    pub fn log_to_board(&self, board: &crate::training::board::BoardWriter, step: u64) {
        for (i, v) in self.max_per_layer.iter().enumerate() {
            board.log_scalar(&format!("qk_max/layer_{i}"), step, *v as f64);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qk_clip_records_max() {
        let mut log = QkClipLog::new(4);
        log.record(0, 1.0);
        log.record(0, 0.5); // smaller, must NOT replace max
        log.record(0, 2.0); // bigger, replaces
        log.record(2, 7.5);
        assert_eq!(log.max_per_layer[0], 2.0);
        assert_eq!(log.max_per_layer[1], 0.0);
        assert_eq!(log.max_per_layer[2], 7.5);
        assert_eq!(log.max_per_layer[3], 0.0);
    }

    #[test]
    fn qk_clip_record_out_of_range_is_silent() {
        let mut log = QkClipLog::new(2);
        log.record(99, 1.0); // must not panic
        assert_eq!(log.max_per_layer.len(), 2);
        assert_eq!(log.max_per_layer[0], 0.0);
    }

    #[test]
    fn qk_clip_reset_zeroes() {
        let mut log = QkClipLog::new(3);
        log.record(0, 5.0);
        log.record(2, 7.0);
        log.reset();
        assert!(log.max_per_layer.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn health_monitor_new_may_fail_without_driver() {
        // We cannot guarantee NVML is present in CI, so we just ensure the
        // function is callable and returns a proper Result. Either branch is
        // acceptable here — we don't want to gate tests on hardware.
        let _ = GpuHealthMonitor::new(0);
    }
}
