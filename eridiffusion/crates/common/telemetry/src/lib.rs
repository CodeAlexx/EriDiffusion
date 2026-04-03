//! Telemetry helpers (GPU-only). Provide simple diagnostics for memory usage and timing.

use anyhow::Result;
use eridiffusion_core::device::{device_manager, Device};
use eridiffusion_core::device::DeviceInfo;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

pub fn memory_mb(device: &Device) -> Result<(f64, f64)> {
    let mgr = device_manager();
    let info: DeviceInfo = mgr
        .get_device_info(device)
        .ok_or_else(|| anyhow::anyhow!("device info missing for {device:?}"))?;
    let avail_mb = info.available_memory as f64 / (1024.0 * 1024.0);
    let total_mb = info.total_memory as f64 / (1024.0 * 1024.0);
    Ok((avail_mb, total_mb))
}

pub fn format_memory(device: &Device) -> String {
    match memory_mb(device) {
        Ok((avail, total)) => format!("{avail:.2}/{total:.2} MB free"),
        Err(e) => format!("telemetry err: {e}"),
    }
}

pub fn now_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::from_secs(0))
        .as_millis()
}

pub fn elapsed_millis(start: SystemTime) -> u128 {
    SystemTime::now()
        .duration_since(start)
        .unwrap_or(Duration::from_secs(0))
        .as_millis()
}
