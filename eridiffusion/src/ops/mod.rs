use flame_core::device::Device;
use flame_core::GradientMap;
use flame_core::{DType, Shape, Tensor};
// backward is a method on Tensor, not a standalone function
use flame_core::optimizers::{Adam, SGD};

// GPU-accelerated operations for Flux
//
// This module provides CUDA implementations of critical operations
// that would otherwise run on CPU and bottleneck training.

pub mod attention;
pub mod qk_norm;
pub mod rope;
pub mod streaming_layer_norm;
pub mod streaming_rms_norm;
pub mod streaming_rms_norm_fixed;

// Re-export FLAME's built-in layers
pub use flame_core::conv::Conv2d;
pub use flame_core::group_norm::GroupNorm;
pub use flame_core::layer_norm::LayerNorm;
pub use flame_core::linear::Linear;
pub use flame_core::norm::RMSNorm;

// Export our custom operations
pub use attention::attention;
pub use qk_norm::{apply_qk_norm, scaled_dot_product_attention, split_qkv};
pub use rope::{apply_rotary_emb, RotaryEmbedding};
pub use streaming_layer_norm::{
    apply_double_stream_norm, apply_single_stream_norm, extract_norm_weights,
    streaming_layer_norm_chunked, StreamingLayerNorm,
};
pub use streaming_rms_norm::{
    apply_double_stream_rms_norm, apply_single_stream_rms_norm, extract_rms_norm_weights,
    StreamingRMSNorm,
};
