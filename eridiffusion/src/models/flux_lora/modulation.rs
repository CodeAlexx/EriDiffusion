//! Modulation layers for Flux
//! 
//! Implements shift and scale modulation used in Flux architecture

use candle_core::{Tensor, Module, Result, Device, DType, D};
use candle_nn::{VarBuilder, Linear, linear};

/// Activation function enum (temporary until networks crate is fixed)
#[derive(Debug, Clone, Copy)]
pub enum Activation {
    Silu,
    Gelu,
    Relu,
}

/// Modulation layer that produces shift, scale, and gate parameters
pub struct Modulation {
    linear: Linear,
    num_features: usize,
    double: bool,
}

impl Modulation {
    pub fn new(dim: usize, num_features: usize, double: bool, vb: VarBuilder) -> Result<Self> {
        let out_features = if double {
            num_features * 6  // shift1, scale1, gate1, shift2, scale2, gate2
        } else {
            num_features * 3  // shift, scale, gate
        };
        
        Ok(Self {
            linear: linear(dim, out_features, vb)?,
            num_features,
            double,
        })
    }
    
    pub fn forward(&self, x: &Tensor) -> Result<ModulationParams> {
        let out = self.linear.forward(x)?;
        
        if self.double {
            // Split into 6 parts
            let chunks = out.chunk(6, D::Minus1)?;
            Ok(ModulationParams::Double {
                shift1: chunks[0].clone(),
                scale1: chunks[1].clone(),
                gate1: chunks[2].clone(),
                shift2: chunks[3].clone(),
                scale2: chunks[4].clone(),
                gate2: chunks[5].clone(),
            })
        } else {
            // Split into 3 parts
            let chunks = out.chunk(3, D::Minus1)?;
            Ok(ModulationParams::Single {
                shift: chunks[0].clone(),
                scale: chunks[1].clone(),
                gate: chunks[2].clone(),
            })
        }
    }
}

/// Modulation parameters
pub enum ModulationParams {
    Single {
        shift: Tensor,
        scale: Tensor,
        gate: Tensor,
    },
    Double {
        shift1: Tensor,
        scale1: Tensor,
        gate1: Tensor,
        shift2: Tensor,
        scale2: Tensor,
        gate2: Tensor,
    },
}

/// Apply modulation to a normalized tensor
pub fn apply_modulation(x: &Tensor, scale: &Tensor, shift: &Tensor) -> Result<Tensor> {
    // Modulation: x = x * (1 + scale) + shift
    let one = Tensor::ones_like(scale)?;
    let scale_factor = (one + scale)?;
    x.broadcast_mul(&scale_factor)?.broadcast_add(shift)
}

/// Modulation output structure matching candle
#[derive(Clone)]
pub struct ModulationOut {
    pub shift: Tensor,
    pub scale: Tensor,
    pub gate: Tensor,
}

impl ModulationOut {
    pub fn scale_shift(&self, xs: &Tensor) -> Result<Tensor> {
        xs.broadcast_mul(&(&self.scale + 1.)?)?.broadcast_add(&self.shift)
    }

    pub fn gate(&self, xs: &Tensor) -> Result<Tensor> {
        self.gate.broadcast_mul(xs)
    }
}

/// Modulation2 for double blocks (matching candle's structure)
pub struct Modulation2 {
    lin: Linear,
}

impl Modulation2 {
    pub fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let lin = linear(dim, 6 * dim, vb.pp("lin"))?;
        Ok(Self { lin })
    }

    pub fn forward(&self, vec_: &Tensor) -> Result<(ModulationOut, ModulationOut)> {
        let ys = vec_
            .silu()?
            .apply(&self.lin)?
            .unsqueeze(1)?
            .chunk(6, D::Minus1)?;
        if ys.len() != 6 {
            candle_core::bail!("unexpected len from chunk {:?}", ys.len())
        }
        let mod1 = ModulationOut {
            shift: ys[0].clone(),
            scale: ys[1].clone(),
            gate: ys[2].clone(),
        };
        let mod2 = ModulationOut {
            shift: ys[3].clone(),
            scale: ys[4].clone(),
            gate: ys[5].clone(),
        };
        Ok((mod1, mod2))
    }
}

/// Modulation1 for single blocks
pub struct Modulation1 {
    lin: Linear,
}

impl Modulation1 {
    pub fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let lin = linear(dim, 3 * dim, vb.pp("lin"))?;
        Ok(Self { lin })
    }

    pub fn forward(&self, vec_: &Tensor) -> Result<ModulationOut> {
        let ys = vec_
            .silu()?
            .apply(&self.lin)?
            .unsqueeze(1)?
            .chunk(3, D::Minus1)?;
        if ys.len() != 3 {
            candle_core::bail!("unexpected len from chunk {:?}", ys.len())
        }
        Ok(ModulationOut {
            shift: ys[0].clone(),
            scale: ys[1].clone(),
            gate: ys[2].clone(),
        })
    }
}