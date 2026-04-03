#![deny(rust_2018_idioms)]

pub mod tokenizer;
pub mod masks;
pub mod clip_l;
pub mod openclip_g;
pub mod qwen;
pub mod qwen25_vl_7b;
pub mod qwen25_7b_instruct;
pub mod encode;
pub mod gpu_utils;

pub use tokenizer::{HfTokenizer, HfTokenizer as Tokenizer, TokenBatch};
pub use masks::{build_padding_mask, build_padding_mask_bool};

use eridiffusion_core::Device;
use flame_core::{CudaDevice, DType, Error as CoreError, Result as CoreResult, Shape, Tensor};

/// Build a Bool attention mask [B,1,1,T] from raw sequence lengths.
pub fn attn_mask_from_lengths(lengths: &[i32], max_len: i32, device: &Device) -> CoreResult<Tensor> {
    if max_len <= 0 { return Err(CoreError::InvalidInput("max_len must be > 0".into())); }
    let cuda = match device {
        Device::Cpu => return Err(CoreError::Unsupported("CPU device not supported".into())),
        Device::Cuda(idx) => CudaDevice::new(*idx)
            .map_err(|e| CoreError::InvalidInput(format!("cuda {idx}: {e}")))?,
    };
    let b = lengths.len();
    let t = max_len as usize;
    let mut data = Vec::with_capacity(b * t);
    for &len in lengths {
        let l = len.max(0) as usize;
        for pos in 0..t {
            data.push(if pos < l { 0.0 } else { 1.0 });
        }
    }
    let tensor = Tensor::from_vec(data, Shape::from_dims(&[b, 1, 1, t]), cuda)?;
    tensor.to_dtype(DType::Bool)
}
