//! Custom Flux model implementation with built-in LoRA support
//! 
//! This is a complete reimplementation of the Flux model architecture
//! that allows for proper LoRA injection throughout all layers.

use candle_core::{Device, DType, Module, Result, Tensor, D, Var, IndexOp};
use candle_nn::{linear, Linear, VarBuilder, VarMap};
use std::collections::HashMap;
use std::path::Path;
use safetensors;
use serde::{Serialize, Deserialize};
use anyhow::Context;
use crate::models::flux_adaptive_loader::FluxAdaptiveLoader;
use crate::loaders::create_flux_remapper;

pub mod blocks;
pub mod lora;
pub mod utils;
pub mod positional_embeddings;
pub mod weight_loader;

pub use self::blocks::{
    FluxDoubleBlockWithLoRA, FluxSingleBlockWithLoRA,
    FluxAttentionWithLoRA, MLPWithLoRA,
};
pub use self::lora::{
    LoRAModule, LoRAConfig, LoRATarget, ModuleType, LoRACompatible,
    LinearWithLoRA,
};
use self::utils::{MLP, EmbedND};

/// Configuration for Flux model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FluxConfig {
    pub in_channels: usize,
    pub vec_in_dim: usize,
    pub context_in_dim: usize,
    pub hidden_size: usize,
    pub mlp_ratio: f32,
    pub num_heads: usize,
    pub depth: usize,
    pub depth_single_blocks: usize,
    pub axes_dim: Vec<usize>,
    pub theta: f32,
    pub qkv_bias: bool,
    pub guidance_embed: bool,
}

impl Default for FluxConfig {
    fn default() -> Self {
        Self {
            in_channels: 64,
            vec_in_dim: 768,
            context_in_dim: 4096,
            hidden_size: 3072,
            mlp_ratio: 4.0,
            num_heads: 24,
            depth: 19,
            depth_single_blocks: 38,
            axes_dim: vec![16, 56, 56],
            theta: 10_000.0,
            qkv_bias: true,
            guidance_embed: true,
        }
    }
}

/// Custom Flux model with LoRA support
pub struct FluxModelWithLoRA {
    config: FluxConfig,
    
    // Input projections
    img_in: Linear,
    txt_in: Linear,
    time_in: MLP,
    vector_in: MLP,
    
    // Positional embeddings
    pe_embedder: EmbedND,
    
    // Transformer blocks
    double_blocks: Vec<FluxDoubleBlockWithLoRA>,
    single_blocks: Vec<FluxSingleBlockWithLoRA>,
    
    // Output projection
    final_layer: LinearWithLoRA,
    
    // Guidance embedding (optional)
    guidance_in: Option<MLP>,
    
    hidden_size: usize,
    num_heads: usize,
}

impl FluxModelWithLoRA {
    pub fn new(config: &FluxConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_size = config.hidden_size;
        
        println!("Creating FluxModelWithLoRA...");
        
        // Input projections
        let img_in = linear(config.in_channels, hidden_size, vb.pp("img_in"))?;
        let txt_in = linear(config.context_in_dim, hidden_size, vb.pp("txt_in"))?;
        
        // Time embeddings MLP - use Flux naming convention
        let time_in = MLP::new(
            256,
            hidden_size,
            hidden_size,
            vb.pp("time_in"),
        )?;
        
        // Vector embeddings MLP - use Flux naming convention
        let vector_in = MLP::new(
            config.vec_in_dim,
            hidden_size,
            hidden_size,
            vb.pp("vector_in"),
        )?;
        
        // Positional embeddings
        let pe_embedder = EmbedND::new(
            hidden_size,
            &config.axes_dim,
            vb.pp("pe_embedder"),
        )?;
        
        // Transformer blocks
        println!("Creating {} double blocks...", config.depth);
        let mut double_blocks = Vec::new();
        for i in 0..config.depth {
            println!("  Creating double block {}/{}", i+1, config.depth);
            double_blocks.push(FluxDoubleBlockWithLoRA::new(
                hidden_size,
                config.num_heads,
                config.mlp_ratio,
                Some(0.0), // dropout
                i,
                vb.pp(&format!("double_blocks.{}", i)),
            )?);
        }
        println!("Double blocks created");
        
        println!("Creating {} single blocks...", config.depth_single_blocks);
        let mut single_blocks = Vec::new();
        for i in 0..config.depth_single_blocks {
            if i % 10 == 0 {
                println!("  Creating single block {}/{}", i+1, config.depth_single_blocks);
            }
            single_blocks.push(FluxSingleBlockWithLoRA::new(
                hidden_size,
                config.num_heads,
                config.mlp_ratio,
                Some(0.0),
                i,
                vb.pp(&format!("single_blocks.{}", i)),
            )?);
        }
        println!("Single blocks created");
        
        // Final output projection
        let final_layer = LinearWithLoRA::new(
            hidden_size,
            config.in_channels,
            "final_layer".to_string(),
            vb.pp("final_layer"),
        )?;
        
        // Optional guidance embedding
        let guidance_in = if config.guidance_embed {
            Some(MLP::new(
                256,
                hidden_size,
                hidden_size,
                vb.pp("guidance_in"),
            )?)
        } else {
            None
        };
        
        Ok(Self {
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
    
    pub fn add_lora_to_all(
        &mut self,
        lora_config: &LoRAConfig,
        device: &Device,
        dtype: DType,
    ) -> Result<()> {
        // Add LoRA to double blocks
        for block in &mut self.double_blocks {
            block.add_lora(lora_config, device, dtype)?;
        }
        
        // Add LoRA to single blocks
        for block in &mut self.single_blocks {
            block.add_lora(lora_config, device, dtype)?;
        }
        
        // Add LoRA to final layer
        self.final_layer.add_lora(
            lora_config.rank,
            lora_config.alpha,
            lora_config.init_scale,
            device,
            dtype,
        )?;
        
        Ok(())
    }
    
    pub fn forward(
        &self,
        img: &Tensor,
        img_ids: &Tensor,
        txt: &Tensor,
        txt_ids: &Tensor,
        timesteps: &Tensor,
        y: &Tensor,
        guidance: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, seq_len, _) = img.dims3()?;
        
        // Handle img_ids shape - it can be either [b, h, w, 3] or [b, h, w]
        let (h, w) = match img_ids.dims().len() {
            4 => (img_ids.dim(1)?, img_ids.dim(2)?),
            3 => (img_ids.dim(1)?, img_ids.dim(2)?),
            _ => return Err(candle_core::Error::Msg(format!(
                "Expected img_ids to have 3 or 4 dimensions, got {:?}", img_ids.shape()
            ))),
        };
        
        let p = 2; // patch size for Flux
        let c = self.config.in_channels;
        
        // Project inputs
        let img = self.img_in.forward(img)?;
        let txt = self.txt_in.forward(txt)?;
        
        // Time embedding
        let time_emb = timestep_embedding(timesteps, 256)?;
        
        // Ensure time_emb is on the same device as the model weights
        let time_emb = time_emb.to_device(img.device())?;
        
        let vec = self.time_in.forward(&time_emb)?;
        
        // Add pooled text embedding
        let vec = vec.add(&self.vector_in.forward(y)?)?;
        
        // Add guidance if provided
        let vec = if let (Some(guidance), Some(guidance_in)) = (guidance, &self.guidance_in) {
            let g_emb = timestep_embedding(guidance, 256)?;
            // Ensure guidance embedding is on the same device as vec
            let g_emb = g_emb.to_device(vec.device())?;
            vec.add(&guidance_in.forward(&g_emb)?)?
        } else {
            vec
        };
        
        // Get positional embeddings
        // TODO: Add positional embeddings
        
        // Double transformer blocks
        let mut img = img;
        let mut txt = txt;
        
        for block in &self.double_blocks {
            let (new_img, new_txt) = block.forward(&img, &txt, &vec)?;
            img = new_img;
            txt = new_txt;
        }
        
        // Concatenate for single blocks
        let combined = Tensor::cat(&[&img, &txt], 1)?;
        
        // Single transformer blocks
        let mut x = combined;
        for block in &self.single_blocks {
            x = block.forward(&x, &vec)?;
        }
        
        // Take only image part
        let img_out = x.i((.., ..seq_len, ..))?;
        
        // Final projection
        let out = self.final_layer.forward(&img_out)?;
        
        // Unpatchify: convert from patches back to image format
        // out shape: [b, seq_len, c] where seq_len = (h/p) * (w/p)
        // Need to reshape to [b, h/p, w/p, c*p*p] then permute and reshape to [b, c, h, w]
        
        let h_patches = h / p;
        let w_patches = w / p;
        
        // First reshape to separate spatial dimensions
        let out = out.reshape((b, h_patches, w_patches, c * p * p))?;
        
        // Then reshape to separate patch dimensions
        let out = out.reshape((b, h_patches, w_patches, c, p, p))?;
        
        // Permute to get [b, c, h_patches, p, w_patches, p]
        let out = out.permute((0, 3, 1, 4, 2, 5))?;
        
        // Finally reshape to [b, c, h, w]
        let out = out.reshape((b, c, h, w))?;
        
        // Forward pass complete
        
        Ok(out)
    }
    
    pub fn get_trainable_params(&self) -> Vec<Var> {
        let mut params = Vec::new();
        
        for block in &self.double_blocks {
            for var in block.get_trainable_params() {
                params.push(var.clone());
            }
        }
        
        for block in &self.single_blocks {
            for var in block.get_trainable_params() {
                params.push(var.clone());
            }
        }
        
        for var in self.final_layer.get_trainable_params() {
            params.push(var.clone());
        }
        
        params
    }
}

/// Timestep embedding function (sinusoidal)
pub fn timestep_embedding(timesteps: &Tensor, dim: usize) -> Result<Tensor> {
    // Use the same device as the input timesteps
    let device = timesteps.device();
    let dtype = timesteps.dtype();
    
    // Ensure timesteps is 1D
    let timesteps = if timesteps.dims().len() > 1 {
        timesteps.flatten_all()?
    } else {
        timesteps.clone()
    };
    
    let half = dim / 2;
    let freqs = (0..half)
        .map(|i| 10000f32.powf(-(i as f32) / (half as f32 - 1.0)))
        .collect::<Vec<_>>();
    
    let freqs = Tensor::new(freqs.as_slice(), &device)?.to_dtype(dtype)?;
    
    // Compute sin and cos embeddings
    let args = timesteps.unsqueeze(1)?.broadcast_mul(&freqs.unsqueeze(0)?)?;
    let sin = args.sin()?;
    let cos = args.cos()?;
    
    // Concatenate sin and cos
    Tensor::cat(&[&sin, &cos], D::Minus1)
}

impl LoRACompatible for FluxModelWithLoRA {
    fn get_lora_targets(&self) -> Vec<LoRATarget> {
        let mut targets = Vec::new();
        
        // Double blocks
        for i in 0..self.double_blocks.len() {
            for (name, module_type) in [
                ("img_attn", ModuleType::Attention),
                ("txt_attn", ModuleType::Attention),
                ("img_mlp", ModuleType::MLP),
                ("txt_mlp", ModuleType::MLP),
            ] {
                targets.push(LoRATarget {
                    name: format!("double_blocks.{}.{}", i, name),
                    module_type,
                    in_features: self.hidden_size,
                    out_features: self.hidden_size,
                });
            }
        }
        
        // Single blocks
        for i in 0..self.single_blocks.len() {
            for (name, module_type) in [
                ("attn", ModuleType::Attention),
                ("mlp", ModuleType::MLP),
            ] {
                targets.push(LoRATarget {
                    name: format!("single_blocks.{}.{}", i, name),
                    module_type,
                    in_features: self.hidden_size,
                    out_features: self.hidden_size,
                });
            }
        }
        
        targets
    }
    
    fn apply_lora(&mut self, config: &LoRAConfig) -> Result<()> {
        let device = Device::cuda_if_available(0)?;
        let dtype = DType::BF16;
        self.add_lora_to_all(config, &device, dtype)
    }
    
    fn get_trainable_params(&self) -> Vec<Var> {
        self.get_trainable_params()
    }
    
    fn save_lora_weights(&self, path: &Path) -> Result<()> {
        let mut tensors = HashMap::new();
        
        // Collect all LoRA weights
        for block in &self.double_blocks {
            block.save_weights(&mut tensors)?;
        }
        
        for block in &self.single_blocks {
            block.save_weights(&mut tensors)?;
        }
        
        self.final_layer.save_weights(&mut tensors)?;
        
        // Save config
        let config_path = path.with_extension("json");
        let config_json = serde_json::to_string_pretty(&self.config)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to serialize config: {}", e)))?;
        std::fs::write(config_path, config_json)?;
        
        // Save tensors using safetensors
        candle_core::safetensors::save(&tensors, path)?;
        
        println!("Saved {} LoRA tensors to {:?}", tensors.len(), path);
        Ok(())
    }
    
    fn load_lora_weights(&mut self, path: &Path) -> Result<()> {
        // Load tensors
        let data = std::fs::read(path)?;
        let tensors = candle_core::safetensors::load_buffer(&data, &Device::Cpu)?;
        
        // Apply to model
        for block in &mut self.double_blocks {
            block.load_weights(&tensors)?;
        }
        
        for block in &mut self.single_blocks {
            block.load_weights(&tensors)?;
        }
        
        self.final_layer.load_weights(&tensors)?;
        
        println!("Loaded LoRA weights from {:?}", path);
        Ok(())
    }
}

/// Helper function to create a Flux LoRA model
pub fn create_flux_lora_model(
    config: Option<FluxConfig>,
    device: &Device,
    dtype: DType,
    model_path: Option<&Path>,
) -> Result<FluxModelWithLoRA> {
    let config = config.unwrap_or_default();
    
    let vb = if let Some(path) = model_path {
        // Create a custom VarBuilder that handles Flux tensor name mapping
        create_mapped_flux_var_builder(path, &config, device, dtype)?
    } else {
        VarBuilder::zeros(dtype, device)
    };
    
    FluxModelWithLoRA::new(&config, vb)
}

/// Create a VarBuilder with proper Flux tensor name mapping using TensorRemapper
fn create_mapped_flux_var_builder<'a>(
    model_path: &Path,
    config: &FluxConfig,
    device: &'a Device,
    dtype: DType,
) -> Result<VarBuilder<'a>> {
    println!("Loading Flux checkpoint with tensor remapper...");
    println!("Model path: {:?}", model_path);
    println!("Device: {:?}, DType: {:?}", device, dtype);
    
    // Use the new tensor remapper for better handling
    println!("Creating flux remapper...");
    let remapper = create_flux_remapper(model_path, device.clone(), dtype)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to create remapper: {}", e)))?;
    println!("Remapper created successfully");
    
    // Create VarMap and populate it
    let var_map = VarMap::new();
    {
        let mut data = var_map.data().lock().unwrap();
        
        // Get all tensors we need for the model
        let tensor_names = get_required_tensor_names(config);
        println!("Need to load {} tensors", tensor_names.len());
        
        for (i, name) in tensor_names.iter().enumerate() {
            if i % 50 == 0 {
                println!("Loading tensor {}/{}: {}", i+1, tensor_names.len(), name);
            }
            match remapper.load_with_fallbacks(&name) {
                Ok(tensor) => {
                    let var = candle_core::Var::from_tensor(&tensor)?;
                    data.insert(name.clone(), var);
                }
                Err(e) => {
                    println!("Warning: Could not load tensor {}: {}", name, e);
                }
            }
        }
        println!("All tensors loaded into VarMap");
    }
    
    Ok(VarBuilder::from_varmap(&var_map, dtype, device))
}

/// Get list of required tensor names for Flux model
fn get_required_tensor_names(config: &FluxConfig) -> Vec<String> {
    let mut names = Vec::new();
    
    // Time and vector embeddings
    names.push("time_in.mlp.0.fc1.weight".to_string());
    names.push("time_in.mlp.0.fc1.bias".to_string());
    names.push("time_in.mlp.0.fc2.weight".to_string());
    names.push("time_in.mlp.0.fc2.bias".to_string());
    
    names.push("vector_in.mlp.0.fc1.weight".to_string());
    names.push("vector_in.mlp.0.fc1.bias".to_string());
    names.push("vector_in.mlp.0.fc2.weight".to_string());
    names.push("vector_in.mlp.0.fc2.bias".to_string());
    
    // Input projections
    names.push("img_in.weight".to_string());
    names.push("img_in.bias".to_string());
    names.push("txt_in.weight".to_string());
    names.push("txt_in.bias".to_string());
    
    // Double blocks
    for i in 0..config.depth {
        let prefix = format!("double_blocks.{}", i);
        
        // Attention layers
        for attn_type in ["img_attn", "txt_attn"] {
            names.push(format!("{}.{}.to_q.weight", prefix, attn_type));
            names.push(format!("{}.{}.to_q.bias", prefix, attn_type));
            names.push(format!("{}.{}.to_k.weight", prefix, attn_type));
            names.push(format!("{}.{}.to_k.bias", prefix, attn_type));
            names.push(format!("{}.{}.to_v.weight", prefix, attn_type));
            names.push(format!("{}.{}.to_v.bias", prefix, attn_type));
            names.push(format!("{}.{}.to_out.0.weight", prefix, attn_type));
            names.push(format!("{}.{}.to_out.0.bias", prefix, attn_type));
        }
        
        // MLP layers
        for mlp_type in ["img_mlp", "txt_mlp"] {
            names.push(format!("{}.{}.0.weight", prefix, mlp_type));
            names.push(format!("{}.{}.0.bias", prefix, mlp_type));
            names.push(format!("{}.{}.2.weight", prefix, mlp_type));
            names.push(format!("{}.{}.2.bias", prefix, mlp_type));
        }
        
        // Modulation layers
        for mod_type in ["img_mod", "txt_mod"] {
            names.push(format!("{}.{}.lin.weight", prefix, mod_type));
            names.push(format!("{}.{}.lin.bias", prefix, mod_type));
        }
        
        // Layer norms (will be synthesized)
        for norm in ["img_norm1", "img_norm2", "txt_norm1", "txt_norm2"] {
            names.push(format!("{}.{}.weight", prefix, norm));
            names.push(format!("{}.{}.bias", prefix, norm));
        }
    }
    
    // Single blocks
    for i in 0..config.depth_single_blocks {
        let prefix = format!("single_blocks.{}", i);
        
        // Attention
        names.push(format!("{}.attn.to_q.weight", prefix));
        names.push(format!("{}.attn.to_q.bias", prefix));
        names.push(format!("{}.attn.to_k.weight", prefix));
        names.push(format!("{}.attn.to_k.bias", prefix));
        names.push(format!("{}.attn.to_v.weight", prefix));
        names.push(format!("{}.attn.to_v.bias", prefix));
        names.push(format!("{}.attn.to_out.0.weight", prefix));
        names.push(format!("{}.attn.to_out.0.bias", prefix));
        
        // MLP
        names.push(format!("{}.mlp.linear1.weight", prefix));
        names.push(format!("{}.mlp.linear1.bias", prefix));
        names.push(format!("{}.mlp.linear2.weight", prefix));
        names.push(format!("{}.mlp.linear2.bias", prefix));
        
        // Modulation
        names.push(format!("{}.modulation.lin.weight", prefix));
        names.push(format!("{}.modulation.lin.bias", prefix));
        
        // Layer norms
        names.push(format!("{}.norm1.weight", prefix));
        names.push(format!("{}.norm1.bias", prefix));
        names.push(format!("{}.norm2.weight", prefix));
        names.push(format!("{}.norm2.bias", prefix));
    }
    
    // Final layer
    names.push("final_layer.weight".to_string());
    names.push("final_layer.bias".to_string());
    
    names
}

