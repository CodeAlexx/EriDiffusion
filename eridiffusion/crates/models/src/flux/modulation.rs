//! Modulation layers for Flux models
//! 
//! Implements the modulation mechanism used in Flux for conditioning
//! on time and guidance embeddings.

use candle_core::{Tensor, Module, Result, Device, DType};
use candle_nn::{VarBuilder, Linear, linear};

/// Modulation layer that produces shift, scale, and gate values
pub struct Modulation {
    linear: Linear,
    hidden_size: usize,
}

impl Modulation {
    pub fn new(hidden_size: usize, vb: VarBuilder) -> Result<Self> {
        // Output 3x hidden_size for shift, scale, and gate
        let linear = linear(hidden_size, hidden_size * 3, vb)?;
        
        Ok(Self {
            linear,
            hidden_size,
        })
    }
    
    /// Forward pass returns (shift, scale, gate) tensors
    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let out = self.linear.forward(x)?;
        
        // Split into shift, scale, gate
        let shift = out.narrow(1, 0, self.hidden_size)?;
        let scale = out.narrow(1, self.hidden_size, self.hidden_size)?;
        let gate = out.narrow(1, 2 * self.hidden_size, self.hidden_size)?;
        
        Ok((shift, scale, gate))
    }
}

/// Apply modulation: out = (x + shift) * scale
pub fn apply_modulation(x: &Tensor, shift: &Tensor, scale: &Tensor) -> Result<Tensor> {
    // Unsqueeze shift and scale for broadcasting
    let shift = shift.unsqueeze(1)?;
    let scale = scale.unsqueeze(1)?;
    
    // Apply modulation
    x.broadcast_add(&shift)?
        .broadcast_mul(&scale)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_modulation() -> Result<()> {
        let device = Device::Cpu;
        let batch_size = 2;
        let seq_len = 10;
        let hidden_size = 256;
        
        let vs = candle_nn::VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&vs, DType::F32, &device);
        
        let modulation = Modulation::new(hidden_size, vb)?;
        
        // Test input
        let vec = Tensor::randn(0.0, 1.0, &[batch_size, hidden_size], &device)?;
        let (shift, scale, gate) = modulation.forward(&vec)?;
        
        assert_eq!(shift.shape().dims(), &[batch_size, hidden_size]);
        assert_eq!(scale.shape().dims(), &[batch_size, hidden_size]);
        assert_eq!(gate.shape().dims(), &[batch_size, hidden_size]);
        
        // Test apply_modulation
        let x = Tensor::randn(0.0, 1.0, &[batch_size, seq_len, hidden_size], &device)?;
        let modulated = apply_modulation(&x, &shift, &scale)?;
        assert_eq!(modulated.shape().dims(), &[batch_size, seq_len, hidden_size]);
        
        Ok(())
    }
}