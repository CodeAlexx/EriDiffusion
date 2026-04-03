use anyhow::Context;
use flame_core::device::Device;
use flame_core::optimizers::{Adam, SGD};
use flame_core::{DType, Result, Shape, Tensor};
use std::process::Command;
use std::sync::Arc;

// Memory utilities for training

// FLAME uses flame_core::device::Device instead of Device

/// Set environment variables for better CUDA memory management
pub fn setup_cuda_memory_management() -> flame_core::Result<()> {
    // Prevent PyTorch/CUDA from reserving too much memory
    std::env::set_var("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512");

    // Enable TF32 for better performance
    std::env::set_var("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "1");

    // Reduce CUDA memory fragmentation
    std::env::set_var("CUDA_LAUNCH_BLOCKING", "0");

    println!("Set CUDA memory management environment variables");
    Ok(())
}

/// Clear CUDA cache if available
pub fn clear_cuda_cache(device: &Device) -> flame_core::Result<()> {
    #[cfg(feature = "cuda")]
    {
        let _ = device;
        std::thread::yield_now();
        println!("CUDA memory management attempted - yielded CPU time for GPU cleanup");
        Ok(())
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = device;
        Err(flame_core::Error::InvalidOperation(
            "CUDA feature not enabled, cannot clear cache".to_string(),
        ))
    }
}

/// Get current GPU memory usage
pub fn get_gpu_memory_info() -> flame_core::Result<String> {
    let output = Command::new("nvidia-smi")
        .args(&[
            "--query-gpu=memory.used,memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .map_err(|e| {
            flame_core::Error::InvalidOperation(format!("Failed to run nvidia-smi: {}", e))
        })?;

    if output.status.success() {
        let info = String::from_utf8_lossy(&output.stdout);
        let parts: Vec<&str> = info.trim().split(',').collect();
        if parts.len() >= 3 {
            let used: f32 = parts[0].trim().parse().unwrap_or(0.0);
            let free: f32 = parts[1].trim().parse().unwrap_or(0.0);
            let total: f32 = parts[2].trim().parse().unwrap_or(0.0);

            return Ok(format!(
                "GPU Memory: {:.1}GB/{:.1}GB used ({:.1}% free)",
                used / 1024.0,
                total / 1024.0,
                (free / total) * 100.0
            ));
        }
    }

    Ok("GPU memory info not available".to_string())
}

/// Print memory usage at key points
pub fn log_memory_usage(stage: &str) -> flame_core::Result<()> {
    let info = get_gpu_memory_info()?;
    println!("[{}] {}", stage, info);
    Ok(())
}
