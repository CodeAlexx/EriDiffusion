//! Direct Flux model loader that bypasses VarBuilder's prefix issues

use candle_core::{Device, DType, Result, Tensor};
use candle_nn::{Linear, LayerNorm, VarBuilder, VarMap};
use std::collections::HashMap;
use std::path::Path;

use crate::models::flux_custom::{
    FluxConfig, FluxModelWithLoRA,
    blocks::{FluxDoubleBlockWithLoRA, FluxSingleBlockWithLoRA},
    utils::{MLP, EmbedND},
    lora::LinearWithLoRA,
};

/// Load Flux model directly without VarBuilder complications
pub fn load_flux_direct(
    checkpoint_path: &Path,
    config: &FluxConfig,
    device: &Device,
    dtype: DType,
) -> Result<FluxModelWithLoRA> {
    println!("Loading Flux model with direct tensor loading...");
    
    // Load all tensors from checkpoint
    let tensors = candle_core::safetensors::load(checkpoint_path, device)?;
    println!("Loaded {} tensors from checkpoint", tensors.len());
    
    // Create a wrapper for easier tensor access
    let loader = DirectLoader {
        tensors,
        device: device.clone(),
        dtype,
    };
    
    // Build model components directly
    let hidden_size = config.hidden_size;
    
    // Input projections
    let img_in = loader.load_linear("img_in", config.in_channels, hidden_size)?;
    let txt_in = loader.load_linear("txt_in", config.context_in_dim, hidden_size)?;
    
    // Time/vector embeddings
    let time_in = loader.load_mlp("time_in", 256, hidden_size)?;
    let vector_in = loader.load_mlp("vector_in", config.vec_in_dim, hidden_size)?;
    
    // Positional embeddings - create empty for now
    let pe_embedder = {
        let var_map = VarMap::new();
        let vb = VarBuilder::from_varmap(&var_map, dtype, device);
        EmbedND::new(
            hidden_size / config.axes_dim.len(),
            &config.axes_dim,
            vb,
        )?
    };
    
    // Double blocks
    let mut double_blocks = Vec::new();
    for i in 0..config.depth {
        double_blocks.push(loader.load_double_block(i, config)?);
    }
    
    // Single blocks
    let mut single_blocks = Vec::new();
    for i in 0..config.depth_single_blocks {
        single_blocks.push(loader.load_single_block(i, config)?);
    }
    
    // Final layer
    let final_layer = loader.load_linear_with_lora(
        "final_layer",
        hidden_size,
        config.in_channels,
    )?;
    
    // Guidance embedding (if needed)
    let guidance_in = if config.guidance_embed {
        Some(loader.load_mlp("guidance_in", 256, hidden_size)?)
    } else {
        None
    };
    
    // Manually construct the model
    Ok(FluxModelWithLoRA {
        config: config.clone(),
        img_in,
        txt_in,
        time_in,
        vector_in,
        pe_embedder,
        double_blocks,
        single_blocks,
        final_layer,
        guidance_in,
        hidden_size,
        num_heads: config.num_heads,
    })
}

struct DirectLoader {
    tensors: HashMap<String, Tensor>,
    device: Device,
    dtype: DType,
}

impl DirectLoader {
    fn get_tensor(&self, name: &str) -> Result<Tensor> {
        self.tensors
            .get(name)
            .ok_or_else(|| candle_core::Error::Msg(format!("Tensor {} not found", name)))?
            .to_device(&self.device)?
            .to_dtype(self.dtype)
    }
    
    fn load_linear(&self, prefix: &str, in_features: usize, out_features: usize) -> Result<Linear> {
        let weight = self.get_tensor(&format!("{}.weight", prefix))?;
        let bias = self.get_tensor(&format!("{}.bias", prefix)).ok();
        
        // Create a temporary VarMap for this linear layer
        let var_map = VarMap::new();
        var_map.set_one("weight", weight)?;
        if let Some(b) = bias {
            var_map.set_one("bias", b)?;
        }
        
        let vb = VarBuilder::from_varmap(&var_map, self.dtype, &self.device);
        candle_nn::linear(in_features, out_features, vb)
    }
    
    fn load_linear_with_lora(
        &self,
        prefix: &str,
        in_features: usize,
        out_features: usize,
    ) -> Result<LinearWithLoRA> {
        let linear = self.load_linear(prefix, in_features, out_features)?;
        
        // Create LinearWithLoRA manually
        let var_map = VarMap::new();
        let vb = VarBuilder::from_varmap(&var_map, self.dtype, &self.device);
        
        Ok(LinearWithLoRA {
            base: linear,
            lora: None,
            name: prefix.to_string(),
            trainable: false,
        })
    }
    
    fn load_mlp(&self, prefix: &str, in_features: usize, hidden_size: usize) -> Result<MLP> {
        // MLP layers use different naming in checkpoint
        let fc1 = if self.tensors.contains_key(&format!("{}.in_layer.weight", prefix)) {
            self.load_linear(&format!("{}.in_layer", prefix), in_features, hidden_size)?
        } else if self.tensors.contains_key(&format!("{}.mlp.0.weight", prefix)) {
            self.load_linear(&format!("{}.mlp.0", prefix), in_features, hidden_size)?
        } else {
            return Err(candle_core::Error::Msg(format!("MLP {} not found", prefix)));
        };
        
        let fc2 = if self.tensors.contains_key(&format!("{}.out_layer.weight", prefix)) {
            self.load_linear(&format!("{}.out_layer", prefix), hidden_size, hidden_size)?
        } else if self.tensors.contains_key(&format!("{}.mlp.2.weight", prefix)) {
            self.load_linear(&format!("{}.mlp.2", prefix), hidden_size, hidden_size)?
        } else {
            return Err(candle_core::Error::Msg(format!("MLP {} out layer not found", prefix)));
        };
        
        Ok(MLP { fc1, fc2 })
    }
    
    fn load_attention_qkv(
        &self,
        prefix: &str,
        hidden_size: usize,
    ) -> Result<(LinearWithLoRA, LinearWithLoRA, LinearWithLoRA)> {
        // Load combined QKV and split
        let qkv_weight = self.get_tensor(&format!("{}.qkv.weight", prefix))?;
        let qkv_bias = self.get_tensor(&format!("{}.qkv.bias", prefix)).ok();
        
        let (total_dim, _) = qkv_weight.dims2()?;
        let head_dim = total_dim / 3;
        
        // Split weights
        let q_weight = qkv_weight.narrow(0, 0, head_dim)?;
        let k_weight = qkv_weight.narrow(0, head_dim, head_dim)?;
        let v_weight = qkv_weight.narrow(0, head_dim * 2, head_dim)?;
        
        // Split bias if present
        let (q_bias, k_bias, v_bias) = if let Some(bias) = qkv_bias {
            let total_dim = bias.dims1()?;
            let head_dim = total_dim / 3;
            (
                Some(bias.narrow(0, 0, head_dim)?),
                Some(bias.narrow(0, head_dim, head_dim)?),
                Some(bias.narrow(0, head_dim * 2, head_dim)?),
            )
        } else {
            (None, None, None)
        };
        
        // Create linear layers
        let to_q = self.create_linear_from_tensors(q_weight, q_bias, hidden_size, hidden_size)?;
        let to_k = self.create_linear_from_tensors(k_weight, k_bias, hidden_size, hidden_size)?;
        let to_v = self.create_linear_from_tensors(v_weight, v_bias, hidden_size, hidden_size)?;
        
        Ok((
            LinearWithLoRA {
                base: to_q,
                lora: None,
                name: format!("{}.to_q", prefix),
                trainable: false,
            },
            LinearWithLoRA {
                base: to_k,
                lora: None,
                name: format!("{}.to_k", prefix),
                trainable: false,
            },
            LinearWithLoRA {
                base: to_v,
                lora: None,
                name: format!("{}.to_v", prefix),
                trainable: false,
            },
        ))
    }
    
    fn create_linear_from_tensors(
        &self,
        weight: Tensor,
        bias: Option<Tensor>,
        in_features: usize,
        out_features: usize,
    ) -> Result<Linear> {
        let var_map = VarMap::new();
        var_map.set_one("weight", weight)?;
        if let Some(b) = bias {
            var_map.set_one("bias", b)?;
        }
        
        let vb = VarBuilder::from_varmap(&var_map, self.dtype, &self.device);
        candle_nn::linear(in_features, out_features, vb)
    }
    
    fn load_double_block(&self, idx: usize, config: &FluxConfig) -> Result<FluxDoubleBlockWithLoRA> {
        let prefix = format!("double_blocks.{}", idx);
        let hidden_size = config.hidden_size;
        
        // Load attention QKV
        let (img_to_q, img_to_k, img_to_v) = self.load_attention_qkv(
            &format!("{}.img_attn", prefix),
            hidden_size,
        )?;
        let (txt_to_q, txt_to_k, txt_to_v) = self.load_attention_qkv(
            &format!("{}.txt_attn", prefix),
            hidden_size,
        )?;
        
        // Output projections
        let img_to_out = self.load_linear_with_lora(
            &format!("{}.img_attn.proj", prefix),
            hidden_size,
            hidden_size,
        )?;
        let txt_to_out = self.load_linear_with_lora(
            &format!("{}.txt_attn.proj", prefix),
            hidden_size,
            hidden_size,
        )?;
        
        // MLPs (0/2 naming in checkpoint)
        let img_mlp_fc1 = self.load_linear(&format!("{}.img_mlp.0", prefix), hidden_size, (hidden_size as f32 * config.mlp_ratio) as usize)?;
        let img_mlp_fc2 = self.load_linear(&format!("{}.img_mlp.2", prefix), (hidden_size as f32 * config.mlp_ratio) as usize, hidden_size)?;
        let txt_mlp_fc1 = self.load_linear(&format!("{}.txt_mlp.0", prefix), hidden_size, (hidden_size as f32 * config.mlp_ratio) as usize)?;
        let txt_mlp_fc2 = self.load_linear(&format!("{}.txt_mlp.2", prefix), (hidden_size as f32 * config.mlp_ratio) as usize, hidden_size)?;
        
        // Layer norms
        let img_norm1 = self.load_layer_norm(&format!("{}.img_norm1", prefix), hidden_size)?;
        let img_norm2 = self.load_layer_norm(&format!("{}.img_norm2", prefix), hidden_size)?;
        let txt_norm1 = self.load_layer_norm(&format!("{}.txt_norm1", prefix), hidden_size)?;
        let txt_norm2 = self.load_layer_norm(&format!("{}.txt_norm2", prefix), hidden_size)?;
        
        // TODO: Load modulation layers if they exist
        
        // Construct block manually
        // This is simplified - you'd need to match your actual block structure
        todo!("Complete double block construction")
    }
    
    fn load_single_block(&self, idx: usize, config: &FluxConfig) -> Result<FluxSingleBlockWithLoRA> {
        let prefix = format!("single_blocks.{}", idx);
        // Similar to double block but simpler
        todo!("Implement single block loading")
    }
    
    fn load_layer_norm(&self, prefix: &str, normalized_shape: usize) -> Result<LayerNorm> {
        let weight = self.get_tensor(&format!("{}.weight", prefix))?;
        
        let var_map = VarMap::new();
        var_map.set_one("weight", weight)?;
        
        let vb = VarBuilder::from_varmap(&var_map, self.dtype, &self.device);
        candle_nn::layer_norm(normalized_shape, 1e-6, vb)
    }
}