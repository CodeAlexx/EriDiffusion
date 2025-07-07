//! Stable Diffusion family adapters (SD1.5, SD2, SDXL, SD3, SD3.5)

use anyhow::Result;
use candle_core::{Device, DType, Tensor};
use std::collections::HashMap;
use super::WeightAdapter;

/// Adapter for SD 1.5 and SD 2.x models
pub struct SD15Adapter;

impl WeightAdapter for SD15Adapter {
    fn can_adapt(&self, from_arch: &str, to_arch: &str) -> bool {
        (from_arch == "sd15" || from_arch == "sd2") && to_arch.starts_with("sd")
    }
    
    fn adapt_name(&self, name: &str) -> String {
        // Handle common SD naming variations
        name.replace("model.diffusion_model.", "")
            .replace("input_blocks", "down_blocks")
            .replace("output_blocks", "up_blocks")
            .replace("middle_block", "mid_block")
            .replace("time_embed", "time_embedding")
    }
    
    fn adapt_tensor(&self, name: &str, tensor: Tensor) -> Result<Vec<(String, Tensor)>> {
        // Most SD tensors can be used directly
        Ok(vec![(self.adapt_name(name), tensor)])
    }
}

/// Adapter for SDXL models
pub struct SDXLAdapter;

impl WeightAdapter for SDXLAdapter {
    fn can_adapt(&self, from_arch: &str, to_arch: &str) -> bool {
        from_arch == "sdxl" && (to_arch == "sdxl" || to_arch == "sd")
    }
    
    fn adapt_name(&self, name: &str) -> String {
        // SDXL specific mappings
        name.replace("conditioner.embedders.0.", "text_encoder.")
            .replace("conditioner.embedders.1.", "text_encoder_2.")
            .replace("add_embedding.linear_1", "add_embedding.linear1")
            .replace("add_embedding.linear_2", "add_embedding.linear2")
    }
    
    fn adapt_tensor(&self, name: &str, tensor: Tensor) -> Result<Vec<(String, Tensor)>> {
        Ok(vec![(self.adapt_name(name), tensor)])
    }
}

/// Adapter for SD3 and SD3.5 models
pub struct SD3Adapter {
    hidden_size: usize,
}

impl SD3Adapter {
    pub fn new(hidden_size: usize) -> Self {
        Self { hidden_size }
    }
}

impl WeightAdapter for SD3Adapter {
    fn can_adapt(&self, from_arch: &str, to_arch: &str) -> bool {
        (from_arch == "sd3" || from_arch == "sd35") && to_arch.starts_with("sd3")
    }
    
    fn adapt_name(&self, name: &str) -> String {
        // SD3 uses MMDiT architecture
        name.replace("model.diffusion_model.", "")
            .replace("joint_blocks", "mmdit_blocks")
            .replace("context_embedder", "text_embedder")
            .replace("x_embedder", "img_embedder")
            .replace("norm_out", "final_norm")
    }
    
    fn adapt_tensor(&self, name: &str, tensor: Tensor) -> Result<Vec<(String, Tensor)>> {
        // Handle SD3's combined QKV tensors
        if name.contains(".qkv.") {
            let is_weight = name.ends_with(".weight");
            let prefix = name.replace(".qkv.weight", "").replace(".qkv.bias", "");
            
            if is_weight {
                let (total_dim, in_dim) = tensor.dims2()?;
                let head_dim = total_dim / 3;
                
                Ok(vec![
                    (format!("{}.to_q.weight", prefix), tensor.narrow(0, 0, head_dim)?),
                    (format!("{}.to_k.weight", prefix), tensor.narrow(0, head_dim, head_dim)?),
                    (format!("{}.to_v.weight", prefix), tensor.narrow(0, head_dim * 2, head_dim)?),
                ])
            } else {
                let total_dim = tensor.dims1()?;
                let head_dim = total_dim / 3;
                
                Ok(vec![
                    (format!("{}.to_q.bias", prefix), tensor.narrow(0, 0, head_dim)?),
                    (format!("{}.to_k.bias", prefix), tensor.narrow(0, head_dim, head_dim)?),
                    (format!("{}.to_v.bias", prefix), tensor.narrow(0, head_dim * 2, head_dim)?),
                ])
            }
        } else {
            Ok(vec![(self.adapt_name(name), tensor)])
        }
    }
}