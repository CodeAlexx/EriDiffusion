//! Lazy loading layers for Flux model
//! 
//! These layers defer weight loading until first use

use candle_core::{Device, DType, Module, Result, Tensor, D};
use candle_nn::{VarBuilder};
use std::sync::{Arc, Mutex, OnceLock};

use crate::trainers::flux_lazy_loader::LazyVarBuilder;

/// Lazy Linear layer that loads weights on first forward pass
pub struct LazyLinear {
    in_features: usize,
    out_features: usize,
    weight: OnceLock<Tensor>,
    bias: OnceLock<Option<Tensor>>,
    vb: LazyVarBuilder,
    use_bias: bool,
}

impl LazyLinear {
    pub fn new(
        in_features: usize, 
        out_features: usize,
        use_bias: bool,
        vb: LazyVarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            in_features,
            out_features,
            weight: OnceLock::new(),
            bias: OnceLock::new(),
            vb,
            use_bias,
        })
    }
    
    fn load_weights(&self) -> Result<()> {
        // Load weight if not already loaded
        self.weight.get_or_init(|| {
            self.vb.get("weight", &[self.out_features, self.in_features])
                .expect("Failed to load weight")
        });
        
        // Load bias if needed
        if self.use_bias {
            self.bias.get_or_init(|| {
                Some(self.vb.get("bias", &[self.out_features])
                    .expect("Failed to load bias"))
            });
        } else {
            self.bias.get_or_init(|| None);
        }
        
        Ok(())
    }
}

impl Module for LazyLinear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // Ensure weights are loaded
        self.load_weights()?;
        
        let weight = self.weight.get().unwrap();
        let bias = self.bias.get().unwrap();
        
        // Standard linear forward: y = xW^T + b
        let output = xs.matmul(&weight.t()?)?;
        
        if let Some(b) = bias {
            output.broadcast_add(b)
        } else {
            Ok(output)
        }
    }
}

/// Create a lazy linear layer
pub fn lazy_linear(
    in_features: usize,
    out_features: usize,
    vb: LazyVarBuilder,
) -> Result<LazyLinear> {
    LazyLinear::new(in_features, out_features, true, vb)
}

/// Create a lazy linear layer without bias
pub fn lazy_linear_no_bias(
    in_features: usize,
    out_features: usize,
    vb: LazyVarBuilder,
) -> Result<LazyLinear> {
    LazyLinear::new(in_features, out_features, false, vb)
}