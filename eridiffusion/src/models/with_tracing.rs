//! Tracing wrappers for debugging and monitoring
//! This module provides wrapped versions of common layers with optional tracing

use crate::loaders::WeightLoader;
use flame_core::{DType, Shape, Tensor};
use flame_core::device::Device;
use flame_core::{Result};

/// Helper function to create a Conv2d layer with tracing support
pub fn conv2d_with_tracing(
    name: &str,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    config: flame_core::conv2d::Conv2dConfig,
    weights: &WeightLoader,
) -> Result<flame_core::conv2d::Conv2d> {
    if std::env::var("TRACE_LAYERS").is_ok() {
        println!("[TRACE] Creating Conv2d layer '{}': in={}, out={}, kernel={}", 
                 name, in_channels, out_channels, kernel_size);
    }
    
    flame_core::conv2d::conv2d_no_bias(
        in_channels,
        out_channels,
        kernel_size,
        config,
        weights.get(&format!("{}.weight", name))?,
    )
}