use candle_core::{CudaDevice, CudaStorage, Device, DType, Result, Shape, Tensor};
use candle_nn::VarBuilder;
use cudarc::driver::{CudaFunction, CudaStream, DevicePtr, LaunchAsync, LaunchConfig};
use std::sync::Arc;

pub struct CudaRmsNorm {
    device: Arc<CudaDevice>,
    f32_func: CudaFunction,
    f16_func: CudaFunction,
}

impl CudaRmsNorm {
    pub fn new(device: &CudaDevice) -> Result<Self> {
        // PTX will be embedded at compile time
        let ptx = include_bytes!(concat!(env!("OUT_DIR"), "/rms_norm.ptx"));
        let ptx_str = std::str::from_utf8(ptx)
            .map_err(|e| candle_core::Error::Msg(format!("Invalid PTX: {}", e)))?;
        
        let cu_device = device.cu_device();
        let cu_module = cu_device.load_ptx(ptx_str, "rms_norm", &["rms_norm_f32", "rms_norm_f16"])
            .map_err(|e| candle_core::Error::Cuda(Box::new(e)))?;
            
        let f32_func = cu_module.get_func("rms_norm_f32")
            .map_err(|e| candle_core::Error::Cuda(Box::new(e)))?;
            
        let f16_func = cu_module.get_func("rms_norm_f16")
            .map_err(|e| candle_core::Error::Cuda(Box::new(e)))?;
        
        Ok(Self {
            device: Arc::new(device.clone()),
            f32_func,
            f16_func,
        })
    }
    
    pub fn forward(&self, x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        let shape = x.shape();
        let dims = shape.dims();
        
        if dims.len() < 2 {
            return Err(candle_core::Error::DimOutOfRange {
                shape: shape.clone(),
                dim: 2,
                op: "rms_norm",
            });
        }
        
        let hidden_size = dims[dims.len() - 1];
        let num_sequences = dims.iter().take(dims.len() - 1).product::<usize>();
        let batch_size = if dims.len() >= 3 { dims[0] } else { 1 };
        let seq_len = num_sequences / batch_size;
        
        // Determine block size based on hidden dimension
        let block_size = (hidden_size.min(1024).next_power_of_two() / 2).max(32);
        let shared_mem_size = block_size * std::mem::size_of::<f32>();
        
        let config = LaunchConfig {
            block_dim: (block_size as u32, 1, 1),
            grid_dim: (num_sequences as u32, 1, 1),
            shared_mem_size: shared_mem_size as u32,
        };
        
        match x.dtype() {
            DType::F32 => self.forward_f32(x, weight, eps, config, batch_size, seq_len, hidden_size),
            DType::F16 => self.forward_f16(x, weight, eps, config, batch_size, seq_len, hidden_size),
            dt => Err(candle_core::Error::Msg(format!(
                "Unsupported dtype {:?} for cuda_rms_norm", dt
            ))),
        }
    }
    
    fn forward_f32(
        &self,
        x: &Tensor,
        weight: &Tensor,
        eps: f32,
        config: LaunchConfig,
        batch_size: usize,
        seq_len: usize,
        hidden_size: usize,
    ) -> Result<Tensor> {
        let elem_count = x.elem_count();
        let cu_device = self.device.cu_device();
        let stream = self.device.cu_stream();
        
        let (x_storage, x_layout) = x.storage_and_layout();
        let x_cuda_storage = x_storage.as_cuda_slice::<f32>()?;
        let x_ptr = *x_cuda_storage.device_ptr();
        
        let (weight_storage, _) = weight.storage_and_layout();
        let weight_cuda_storage = weight_storage.as_cuda_slice::<f32>()?;
        let weight_ptr = *weight_cuda_storage.device_ptr();
        
        let output = unsafe {
            let output_ptr = cu_device.alloc::<f32>(elem_count)
                .map_err(|e| candle_core::Error::Cuda(Box::new(e)))?;
            
            let params = (
                x_ptr,
                weight_ptr,
                output_ptr.clone(),
                eps,
                batch_size as i32,
                seq_len as i32,
                hidden_size as i32,
            );
            
            self.f32_func.launch(config, params)
                .map_err(|e| candle_core::Error::Cuda(Box::new(e)))?;
            
            let storage = CudaStorage::wrap_cuda_slice(output_ptr, cu_device.clone());
            Tensor::from_storage(storage, x.shape().clone(), candle_core::backend::BackendStorage::Cuda, x.dtype())
        }?;
        
        Ok(output)
    }
    
    fn forward_f16(
        &self,
        x: &Tensor,
        weight: &Tensor,
        eps: f32,
        config: LaunchConfig,
        batch_size: usize,
        seq_len: usize,
        hidden_size: usize,
    ) -> Result<Tensor> {
        let elem_count = x.elem_count();
        let cu_device = self.device.cu_device();
        
        let (x_storage, _) = x.storage_and_layout();
        let x_cuda_storage = x_storage.as_cuda_slice::<half::f16>()?;
        let x_ptr = *x_cuda_storage.device_ptr();
        
        let (weight_storage, _) = weight.storage_and_layout();
        let weight_cuda_storage = weight_storage.as_cuda_slice::<half::f16>()?;
        let weight_ptr = *weight_cuda_storage.device_ptr();
        
        let output = unsafe {
            let output_ptr = cu_device.alloc::<half::f16>(elem_count)
                .map_err(|e| candle_core::Error::Cuda(Box::new(e)))?;
            
            let params = (
                x_ptr,
                weight_ptr,
                output_ptr.clone(),
                eps,
                batch_size as i32,
                seq_len as i32,
                hidden_size as i32,
            );
            
            self.f16_func.launch(config, params)
                .map_err(|e| candle_core::Error::Cuda(Box::new(e)))?;
            
            let storage = CudaStorage::wrap_cuda_slice(output_ptr, cu_device.clone());
            Tensor::from_storage(storage, x.shape().clone(), candle_core::backend::BackendStorage::Cuda, x.dtype())
        }?;
        
        Ok(output)
    }
}

// Module wrapper for integration with model
pub struct RmsNorm {
    weight: Tensor,
    eps: f32,
    cuda_impl: Option<Arc<CudaRmsNorm>>,
}

impl RmsNorm {
    pub fn new(dim: usize, eps: f32, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(dim, "weight")?;
        let device = weight.device();
        
        let cuda_impl = if device.is_cuda() {
            Some(Arc::new(CudaRmsNorm::new(device.as_cuda_device()?)?))
        } else {
            None
        };
        
        Ok(Self {
            weight,
            eps,
            cuda_impl,
        })
    }
    
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if let Some(cuda_impl) = &self.cuda_impl {
            cuda_impl.forward(x, &self.weight, self.eps)
        } else {
            // CPU fallback
            let x2 = x.sqr()?;
            let variance = x2.mean_keepdim(candle_core::D::Minus1)?;
            let x_normed = x.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
            x_normed.broadcast_mul(&self.weight)
        }
    }
}