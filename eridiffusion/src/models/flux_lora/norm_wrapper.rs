//! Normalization wrapper for Flux models
//! 
//! Provides LayerNorm for Flux following candle's implementation

use candle_core::{Tensor, Module, Result};
use candle_nn::{VarBuilder, LayerNorm};

/// LayerNorm wrapper for Flux (following candle's implementation)
pub struct FluxNorm {
    layer_norm: LayerNorm,
}

impl FluxNorm {
    /// Create a new Flux normalization layer
    /// Note: Flux doesn't store norm weights in the model - they're created as ones
    pub fn new(num_features: usize, vb: VarBuilder) -> Result<Self> {
        // Create weight as ones, no bias - matching candle's layer_norm function
        let device = vb.device();
        let dtype = vb.dtype();
        let weight = Tensor::ones(num_features, dtype, device)?;
        let layer_norm = LayerNorm::new_no_bias(weight, 1e-6);
        
        Ok(Self { layer_norm })
    }
}

impl Module for FluxNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.layer_norm.forward(x)
    }
}