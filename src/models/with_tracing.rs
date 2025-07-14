//! Tracing wrappers for debugging and monitoring
//! This module provides wrapped versions of common layers with optional tracing

use candle_core::{Result, Tensor};
pub use candle_nn::{Conv2d, Conv2dConfig, VarBuilder};

/// Helper function to create a Conv2d layer with tracing support
pub fn conv2d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    config: Conv2dConfig,
    vs: VarBuilder,
) -> Result<Conv2d> {
    candle_nn::conv2d(in_channels, out_channels, kernel_size, config, vs)
}