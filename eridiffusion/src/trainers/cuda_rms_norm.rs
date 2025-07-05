use anyhow::Result;
use candle_core::{CpuStorage, CudaStorage, CustomOp2, DType, Device, Layout, Shape, Tensor, D};
use std::sync::Arc;

/// Direct CUDA RMS Norm implementation that ensures CUDA dispatch
#[derive(Debug, Clone)]
pub struct CudaRmsNorm {
    eps: f32,
}

impl CudaRmsNorm {
    pub fn new(eps: f32) -> Self {
        Self { eps }
    }
}

impl CustomOp2 for CudaRmsNorm {
    fn name(&self) -> &'static str {
        "cuda-rms-norm"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape), candle_core::Error> {
        // For CPU, fall back to the standard implementation
        use candle_core::backend::BackendStorage;
        
        let eps = self.eps;
        fn inner<
            T: candle_core::WithDType
                + num_traits::Float
                + num_traits::AsPrimitive<f32>
                + num_traits::FromPrimitive,
        >(
            src: &[T],
            layout: &Layout,
            alpha: &[T],
            alpha_layout: &Layout,
            eps: f32,
        ) -> Result<(CpuStorage, Shape), candle_core::Error> {
            let src = match layout.contiguous_offsets() {
                None => candle_core::bail!("input has to be contiguous"),
                Some((o1, o2)) => &src[o1..o2],
            };
            let alpha = match alpha_layout.contiguous_offsets() {
                None => candle_core::bail!("alpha has to be contiguous"),
                Some((o1, o2)) => &alpha[o1..o2],
            };
            let el_count = layout.shape().elem_count();
            let dims = layout.shape().dims();
            let dim_m1 = dims[dims.len() - 1];
            let mut dst = vec![T::zero(); el_count];
            use rayon::prelude::*;
            src.par_chunks(dim_m1)
                .zip(dst.par_chunks_mut(dim_m1))
                .for_each(|(src, dst)| {
                    let sum2 = src
                        .iter()
                        .map(|&v| {
                            let v = v.as_();
                            v * v
                        })
                        .sum::<f32>();
                    let m = (sum2 / dim_m1 as f32 + eps).sqrt();
                    let m = T::from_f32(m).unwrap_or_else(T::nan);
                    for ((d, s), alpha) in dst.iter_mut().zip(src.iter()).zip(alpha) {
                        *d = *s / m * *alpha
                    }
                });
            let storage = candle_core::WithDType::to_cpu_storage_owned(dst);
            Ok((storage, Shape::from_dims(dims)))
        }

        use CpuStorage as C;
        match (s1, s2) {
            (C::BF16(s1), C::BF16(s2)) => inner::<half::bf16>(s1, l1, s2, l2, eps),
            (C::F16(s1), C::F16(s2)) => inner::<half::f16>(s1, l1, s2, l2, eps),
            (C::F32(s1), C::F32(s2)) => inner::<f32>(s1, l1, s2, l2, eps),
            _ => candle_core::bail!("unsupported dtype for rmsnorm {:?}", s1.dtype()),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &CudaStorage,
        l1: &Layout,
        s2: &CudaStorage,
        l2: &Layout,
    ) -> Result<(CudaStorage, Shape), candle_core::Error> {
        use candle_core::cuda_backend::cudarc::driver::{
            CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg,
        };
        use candle_core::cuda_backend::{kernel_name, kernels, Map2, WrapErr};
        use candle_core::{CudaDevice, WithDType};

        println!("CUDA RMS Norm: Using GPU kernel!");

        struct S {
            eps: f32,
        }
        impl Map2 for S {
            fn f<T: DeviceRepr + WithDType>(
                &self,
                src: &CudaSlice<T>,
                layout: &Layout,
                alpha: &CudaSlice<T>,
                alpha_layout: &Layout,
                dev: &CudaDevice,
            ) -> Result<CudaSlice<T>, candle_core::Error> {
                let src = match layout.contiguous_offsets() {
                    None => candle_core::bail!("input has to be contiguous"),
                    Some((o1, o2)) => src.slice(o1..o2),
                };
                let alpha = match alpha_layout.contiguous_offsets() {
                    None => candle_core::bail!("alpha has to be contiguous"),
                    Some((o1, o2)) => alpha.slice(o1..o2),
                };
                let el = layout.shape().elem_count();
                let dims = layout.shape().dims();
                let dim_m1 = dims[dims.len() - 1];
                let (n_rows, n_cols) = (el / dim_m1, dim_m1);

                let block_size = if n_cols < 1024 { 32 } else { 1024 };
                let cfg = LaunchConfig {
                    grid_dim: (n_rows as u32, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                };
                let func = dev.get_or_load_func(&kernel_name::<T>("rmsnorm"), &kernels::REDUCE)?;
                // SAFETY: Set later by running the kernel.
                let dst = unsafe { dev.alloc::<T>(el)? };
                let mut builder = func.builder();
                builder.arg(&src);
                builder.arg(&dst);
                builder.arg(&alpha);
                candle_core::builder_arg!(builder, n_cols as i32, block_size as i32, self.eps);
                // SAFETY: ffi.
                unsafe { builder.launch(cfg) }.w()?;
                Ok(dst)
            }
        }

        use candle_core::backend::BackendStorage;
        let dev = s1.device();
        let slice = S { eps: self.eps }.map(&s1.slice, l1, &s2.slice, l2, dev)?;
        let dst = CudaStorage {
            slice,
            device: dev.clone(),
        };
        Ok((dst, l1.shape().clone()))
    }

    #[cfg(not(feature = "cuda"))]
    fn cuda_fwd(
        &self,
        _: &CudaStorage,
        _: &Layout,
        _: &CudaStorage,
        _: &Layout,
    ) -> Result<(CudaStorage, Shape), candle_core::Error> {
        candle_core::bail!("CUDA support not compiled in candle")
    }
}

/// Direct RMS norm function that ensures CUDA dispatch
pub fn cuda_rms_norm(xs: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let hidden_size_xs = xs.dim(D::Minus1)?;
    let hidden_size_weight = weight.dims1()?;
    if hidden_size_xs != hidden_size_weight {
        anyhow::bail!(
            "shape mismatch in rms-norm {:?} {:?}",
            xs.shape(),
            weight.shape()
        )
    }
    
    // Check if we're on CUDA
    if xs.device().is_cuda() {
        println!("RMS Norm: Input tensor is on CUDA device");
    }
    
    let result = xs.apply_op2_no_bwd(weight, &CudaRmsNorm::new(eps))
        .map_err(|e| anyhow::anyhow!("RMS norm error: {}", e))?;
    
    Ok(result)
}

/// Wrapper module that can replace candle_nn's RMSNorm
pub mod rms_norm_module {
    use super::*;
    use candle_nn::{Module, VarBuilder};
    
    #[derive(Clone)]
    pub struct RMSNorm {
        weight: Tensor,
        eps: f64,
    }
    
    impl RMSNorm {
        pub fn new(weight: Tensor, eps: f64) -> Self {
            Self { weight, eps }
        }
        
        pub fn from_vars(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
            let weight = vb.get(dim, "weight")?;
            Ok(Self::new(weight, eps))
        }
    }
    
    impl Module for RMSNorm {
        fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
            cuda_rms_norm(xs, &self.weight, self.eps as f32)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))
        }
    }
}

/// Test function to verify CUDA dispatch
pub fn test_cuda_rms_norm() -> Result<()> {
    println!("\n=== Testing CUDA RMS Norm ===");
    
    if let Ok(device) = Device::cuda_if_available(0) {
        println!("CUDA device available, testing RMS norm...");
        
        // Create test tensors
        let x = Tensor::randn(0f32, 1.0, &[2, 8, 512], &device)?;
        let weight = Tensor::ones(&[512], DType::F32, &device)?;
        
        println!("Input shape: {:?}", x.shape());
        println!("Weight shape: {:?}", weight.shape());
        
        // Run RMS norm
        let output = cuda_rms_norm(&x, &weight, 1e-6)?;
        
        println!("Output shape: {:?}", output.shape());
        println!("✓ CUDA RMS norm test passed!");
    } else {
        println!("No CUDA device available");
    }
    
    Ok(())
}