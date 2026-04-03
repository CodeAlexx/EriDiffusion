//! Minimal GPU memory telemetry helpers.
//! For now these provide lightweight fallbacks so higher-level code can compile.

#[derive(Debug, Clone, Copy)]
pub enum MemUnit {
    KB,
    MB,
    GB,
}

/// Return the estimated GPU allocation in megabytes for the given device index.
/// Stub implementation that currently returns 0.0; replace with NVML-backed query when available.
pub fn gpu_alloc_mb(_device_index: usize) -> f32 {
    0.0
}

/// Format a byte count into a human friendly value/unit pair.
pub fn format_bytes_auto(bytes: usize) -> (f32, MemUnit) {
    const KB: f32 = 1024.0;
    const MB: f32 = KB * 1024.0;
    const GB: f32 = MB * 1024.0;
    let b = bytes as f32;
    if b >= GB {
        (b / GB, MemUnit::GB)
    } else if b >= MB {
        (b / MB, MemUnit::MB)
    } else {
        (b / KB, MemUnit::KB)
    }
}
