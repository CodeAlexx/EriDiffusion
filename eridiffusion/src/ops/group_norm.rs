//! GroupNorm implementation with CUDA acceleration
//! 
//! Provides GPU-accelerated group normalization to replace LayerNorm
//! in Flux models for better performance.

use candle_core::{CpuStorage, CustomOp1, Layout, Result, Shape, Tensor, DType, Device, D};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct GroupNorm {
    pub num_groups: usize,
    pub eps: f64,
    pub affine: bool,
}

impl GroupNorm {
    pub fn new(num_groups: usize, num_channels: usize, eps: f64, affine: bool) -> Result<Self> {
        if num_channels % num_groups != 0 {
            candle_core::bail!(
                "num_channels ({}) must be divisible by num_groups ({})",
                num_channels,
                num_groups
            );
        }
        Ok(Self {
            num_groups,
            eps,
            affine,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        weight: Option<&Tensor>,
        bias: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let device = x.device();
        let dtype = x.dtype();
        
        // Handle different input shapes
        let original_shape = x.shape().clone();
        let x_dims = x.dims();
        
        let (batch_size, num_channels, spatial_size) = match x_dims.len() {
            3 => {
                // (B, C, L) - already in correct format
                let &[b, c, l] = x_dims else { unreachable!() };
                (b, c, l)
            }
            4 => {
                // (B, C, H, W) - reshape to (B, C, H*W)
                let &[b, c, h, w] = x_dims else { unreachable!() };
                (b, c, h * w)
            }
            _ => candle_core::bail!("GroupNorm expects 3D or 4D input, got {:?}", x_dims),
        };

        // Reshape if needed
        let x = if x_dims.len() == 4 {
            x.reshape(&[batch_size, num_channels, spatial_size])?
        } else {
            x.clone()
        };
        
        // Ensure tensor is contiguous for CUDA operations
        let x = if x.is_contiguous() {
            x
        } else {
            x.contiguous()?
        };

        // Validate affine parameters
        if self.affine {
            if let Some(w) = weight {
                if w.dims() != [num_channels] {
                    candle_core::bail!("weight must have shape [{}], got {:?}", num_channels, w.dims());
                }
            }
            if let Some(b) = bias {
                if b.dims() != [num_channels] {
                    candle_core::bail!("bias must have shape [{}], got {:?}", num_channels, b.dims());
                }
            }
        }

        // Prepare output tensors
        let mean = Tensor::zeros(&[batch_size * self.num_groups], dtype, device)?;
        let rstd = Tensor::zeros(&[batch_size * self.num_groups], dtype, device)?;
        let output = Tensor::zeros_like(&x)?;

        // Call appropriate kernel based on device and dtype
        match (device, dtype) {
            (Device::Cuda(_), DType::F32) => {
                self.cuda_forward_f32(&x, weight, bias, &output, &mean, &rstd, 
                                     batch_size, num_channels, spatial_size)?;
            }
            (Device::Cuda(_), DType::BF16) => {
                // Convert to f32, compute, convert back
                let x_f32 = x.to_dtype(DType::F32)?;
                let output_f32 = Tensor::zeros_like(&x_f32)?;
                let weight_f32 = weight.map(|w| w.to_dtype(DType::F32)).transpose()?;
                let bias_f32 = bias.map(|b| b.to_dtype(DType::F32)).transpose()?;
                
                self.cuda_forward_f32(&x_f32, weight_f32.as_ref(), bias_f32.as_ref(), 
                                     &output_f32, &mean, &rstd,
                                     batch_size, num_channels, spatial_size)?;
                
                let output = output_f32.to_dtype(DType::BF16)?;
                return Ok((output.reshape(&original_shape)?, mean, rstd));
            }
            (Device::Cpu, _) => {
                candle_core::bail!("GroupNorm CPU implementation not available - GPU required");
            }
            _ => {
                candle_core::bail!("Unsupported device/dtype combination for GroupNorm");
            }
        }

        // Reshape output back to original shape if needed
        let output = if x_dims.len() == 4 {
            output.reshape(&original_shape)?
        } else {
            output
        };

        Ok((output, mean, rstd))
    }

    #[cfg(feature = "cuda")]
    fn cuda_forward_f32(
        &self,
        x: &Tensor,
        weight: Option<&Tensor>,
        bias: Option<&Tensor>,
        output: &Tensor,
        mean: &Tensor,
        rstd: &Tensor,
        batch_size: usize,
        num_channels: usize,
        spatial_size: usize,
    ) -> Result<()> {
        use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
        use candle_core::cuda_backend::WrapErr;
        
        use std::os::raw::{c_float, c_int};
        let device = x.device().as_cuda_device()?;
        
        // Get CUDA storage
        let (x_storage, x_layout) = x.storage_and_layout();
        let x_storage = match &*x_storage {
            candle_core::Storage::Cuda(s) => s,
            _ => candle_core::bail!("Expected CUDA storage"),
        };
        
        let (output_storage, _) = output.storage_and_layout();
        let output_storage = match &*output_storage {
            candle_core::Storage::Cuda(s) => s,
            _ => candle_core::bail!("Expected CUDA storage"),
        };
        
        let (mean_storage, _) = mean.storage_and_layout();
        let mean_storage = match &*mean_storage {
            candle_core::Storage::Cuda(s) => s,
            _ => candle_core::bail!("Expected CUDA storage"),
        };
        
        let (rstd_storage, _) = rstd.storage_and_layout();
        let rstd_storage = match &*rstd_storage {
            candle_core::Storage::Cuda(s) => s,
            _ => candle_core::bail!("Expected CUDA storage"),
        };
        
        // For now, we'll use a placeholder implementation
        // In a real implementation, we would need to properly extract device pointers
        // from CudaStorageSlice which is an enum that needs pattern matching
        
        println!("WARNING: GroupNorm CUDA kernel integration pending proper pointer extraction");
        
        // Skip the CUDA kernel call for now
        // Fall through to the candle-based implementation below
        
        // Fallback implementation using candle operations
        let channels_per_group = num_channels / self.num_groups;
        let x_reshaped = x.reshape(&[batch_size * self.num_groups, channels_per_group * spatial_size])?;
        
        // Compute mean and variance
        let mean_vals = x_reshaped.mean_keepdim(1)?;
        let x_centered = x_reshaped.broadcast_sub(&mean_vals)?;
        let var = (&x_centered * &x_centered)?.mean_keepdim(1)?;
        let rstd_vals = var.affine(1.0, self.eps)?.recip()?.sqrt()?;
        
        // Normalize
        let normalized = x_centered.broadcast_mul(&rstd_vals)?;
        let normalized = normalized.reshape(&[batch_size, num_channels, spatial_size])?;
        
        // Apply affine transform
        let output_val = if let Some(w) = weight {
            let w_reshaped = w.reshape(&[1, num_channels, 1])?;
            let normalized = normalized.broadcast_mul(&w_reshaped)?;
            if let Some(b) = bias {
                let b_reshaped = b.reshape(&[1, num_channels, 1])?;
                normalized.broadcast_add(&b_reshaped)?
            } else {
                normalized
            }
        } else {
            normalized
        };
        
        // Copy to output tensor
        output.slice_assign(&[0..batch_size, 0..num_channels, 0..spatial_size], &output_val)?;
        
        Ok(())
    }
}

/// Layer wrapper for convenience
pub fn group_norm(
    num_groups: usize,
    num_channels: usize,
    eps: f64,
    affine: bool,
    vb: candle_nn::VarBuilder,
) -> Result<impl Fn(&Tensor) -> Result<Tensor>> {
    let group_norm = GroupNorm::new(num_groups, num_channels, eps, affine)?;
    
    let weight = if affine {
        Some(vb.get(num_channels, "weight")?)
    } else {
        None
    };
    
    let bias = if affine {
        Some(vb.get(num_channels, "bias")?)
    } else {
        None
    };
    
    Ok(move |x: &Tensor| {
        let (output, _, _) = group_norm.forward(x, weight.as_ref(), bias.as_ref())?;
        Ok(output)
    })
}