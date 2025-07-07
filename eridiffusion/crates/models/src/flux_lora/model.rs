//! Flux model with built-in LoRA support
//! 
//! This implements the complete Flux architecture with LoRA adapters
//! integrated into the model structure for proper gradient flow.

use candle_core::{Tensor, Module, Result, Device, DType, D};
use candle_nn::{VarBuilder, Conv2d, Conv2dConfig, Linear, linear, LayerNorm, Embedding};
use eridiffusion_networks::LoRALayerConfig;
use eridiffusion_core::{ModelInputs, ModelOutput};
use crate::flux_lora::{FluxDoubleBlockWithLoRA, FluxSingleBlockWithLoRA, DoubleBlockConfig, SingleBlockConfig};
use crate::DiffusionModel;
use std::sync::Arc;

/// Flux model configuration
#[derive(Debug, Clone)]
pub struct FluxConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub patch_size: usize,
    pub image_size: usize,
    pub num_double_blocks: usize,
    pub num_single_blocks: usize,
    pub mlp_ratio: f32,
    pub guidance_embed: bool,
    pub text_hidden_size: usize,
    pub max_seq_length: usize,
}

impl Default for FluxConfig {
    fn default() -> Self {
        Self {
            in_channels: 16,        // VAE latent channels
            out_channels: 16,
            hidden_size: 3072,      // Flux large hidden size
            num_heads: 24,
            patch_size: 2,          // 2x2 patches
            image_size: 64,         // For 1024px images (1024/8/2 = 64)
            num_double_blocks: 19,  // Flux default
            num_single_blocks: 38,  // Flux default
            mlp_ratio: 4.0,
            guidance_embed: true,
            text_hidden_size: 4096, // T5-XXL hidden size
            max_seq_length: 512,
        }
    }
}

/// Time/guidance embedding module
struct FluxTimeEmbedding {
    linear1: Linear,
    linear2: Linear,
}

impl FluxTimeEmbedding {
    fn new(hidden_size: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            linear1: linear(256, hidden_size, vb.pp("0"))?,
            linear2: linear(hidden_size, hidden_size, vb.pp("2"))?,
        })
    }
}

impl Module for FluxTimeEmbedding {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Sinusoidal timestep embeddings
        let t_emb = timestep_embedding(x, 256)?;
        let t_emb = self.linear1.forward(&t_emb)?;
        let t_emb = t_emb.silu()?;
        self.linear2.forward(&t_emb)
    }
}

/// Flux model with integrated LoRA support
pub struct FluxModelWithLoRA {
    // Input layers
    img_in: Conv2d,
    txt_in: Linear,
    time_in: FluxTimeEmbedding,
    guidance_in: Option<Linear>,
    pos_embed: Embedding,
    
    // Transformer blocks
    double_blocks: Vec<FluxDoubleBlockWithLoRA>,
    single_blocks: Vec<FluxSingleBlockWithLoRA>,
    
    // Output layers
    final_norm: LayerNorm,
    proj_out: Linear,
    
    // Configuration
    config: FluxConfig,
    lora_config: Option<LoRALayerConfig>,
}

impl FluxModelWithLoRA {
    /// Create a new Flux model with LoRA support
    pub fn new(
        config: FluxConfig,
        lora_config: Option<LoRALayerConfig>,
        vb: VarBuilder,
    ) -> Result<Self> {
        // Input layers
        let img_in = Conv2d::new(
            config.in_channels,
            config.hidden_size,
            config.patch_size,
            Conv2dConfig {
                stride: config.patch_size,
                padding: 0,
                ..Default::default()
            },
            vb.pp("img_in"),
        )?;
        
        let txt_in = linear(config.text_hidden_size, config.hidden_size, vb.pp("txt_in"))?;
        let time_in = FluxTimeEmbedding::new(config.hidden_size, vb.pp("time_in"))?;
        
        let guidance_in = if config.guidance_embed {
            Some(linear(config.hidden_size, config.hidden_size, vb.pp("guidance_in"))?)
        } else {
            None
        };
        
        // Position embeddings
        let num_patches = (config.image_size / config.patch_size).pow(2);
        let pos_embed = Embedding::new(num_patches, config.hidden_size, vb.pp("pos_embed"))?;
        
        // Create transformer blocks
        let mut double_blocks = Vec::new();
        for i in 0..config.num_double_blocks {
            let block_config = DoubleBlockConfig {
                hidden_size: config.hidden_size,
                num_heads: config.num_heads,
                mlp_ratio: config.mlp_ratio,
                qkv_bias: true,
                lora_config: lora_config.clone(),
            };
            
            let block = FluxDoubleBlockWithLoRA::new(
                block_config,
                i,
                vb.pp(&format!("double_blocks.{}", i)),
            )?;
            double_blocks.push(block);
        }
        
        let mut single_blocks = Vec::new();
        for i in 0..config.num_single_blocks {
            let block_config = SingleBlockConfig {
                hidden_size: config.hidden_size,
                num_heads: config.num_heads,
                mlp_ratio: config.mlp_ratio,
                lora_config: lora_config.clone(),
            };
            
            let block = FluxSingleBlockWithLoRA::new(
                block_config,
                i,
                vb.pp(&format!("single_blocks.{}", i)),
            )?;
            single_blocks.push(block);
        }
        
        // Output layers
        let final_norm = candle_nn::layer_norm(config.hidden_size, 1e-6, vb.pp("final_norm"))?;
        let proj_out = linear(config.hidden_size, config.patch_size * config.patch_size * config.out_channels, vb.pp("proj_out"))?;
        
        Ok(Self {
            img_in,
            txt_in,
            time_in,
            guidance_in,
            pos_embed,
            double_blocks,
            single_blocks,
            final_norm,
            proj_out,
            config,
            lora_config,
        })
    }
    
    /// Get all trainable LoRA parameters
    pub fn trainable_parameters(&self) -> Vec<&candle_core::Var> {
        let mut params = Vec::new();
        
        // Collect from double blocks
        for block in &self.double_blocks {
            params.extend(block.trainable_parameters());
        }
        
        // Collect from single blocks
        for block in &self.single_blocks {
            params.extend(block.trainable_parameters());
        }
        
        params
    }
    
    /// Load pretrained weights and add LoRA
    pub fn from_pretrained(
        model_path: &std::path::Path,
        config: FluxConfig,
        lora_config: Option<LoRALayerConfig>,
        device: &Device,
    ) -> Result<Self> {
        let vb = unsafe { 
            candle_nn::VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, device)?
        };
        
        Self::new(config, lora_config, vb)
    }
    
    /// Forward pass through the model
    pub fn forward(&self, inputs: &ModelInputs) -> Result<ModelOutput> {
        let latents = &inputs.latents;
        let timestep = &inputs.timestep;
        let encoder_hidden_states = inputs.encoder_hidden_states.as_ref()
            .ok_or_else(|| candle_core::Error::Msg("Flux requires text embeddings".into()))?;
        let pooled_projections = inputs.pooled_projections.as_ref();
        
        // Patchify and embed images
        let img_emb = self.embed_image(latents)?;
        
        // Embed text
        let txt_emb = self.txt_in.forward(encoder_hidden_states)?;
        
        // Time and guidance embeddings
        let time_emb = self.time_in.forward(timestep)?;
        let mut vec = time_emb;
        
        // Add pooled text embeddings if provided
        if let Some(pooled) = pooled_projections {
            vec = vec.add(pooled)?;
        }
        
        // Add guidance embedding if configured
        if let Some(guidance_scale) = inputs.guidance_scale {
            if let Some(guidance_in) = &self.guidance_in {
                let guidance_emb = Tensor::new(&[guidance_scale], vec.device())?
                    .unsqueeze(0)?
                    .repeat((vec.dim(0)?, 1))?;
                let guidance_emb = guidance_in.forward(&guidance_emb)?;
                vec = vec.add(&guidance_emb)?;
            }
        }
        
        // Process through double blocks
        let (mut img, mut txt) = (img_emb, txt_emb);
        for block in &self.double_blocks {
            let (new_img, new_txt) = block.forward(&img, &txt, &vec)?;
            img = new_img;
            txt = new_txt;
        }
        
        // Concatenate for single blocks
        let mut x = Tensor::cat(&[&img, &txt], 1)?;
        
        // Process through single blocks
        for block in &self.single_blocks {
            x = block.forward(&x, &vec)?;
        }
        
        // Extract image features
        let img_seq_len = img.dim(1)?;
        let img_out = x.narrow(1, 0, img_seq_len)?;
        
        // Final norm and projection
        let x = self.final_norm.forward(&img_out)?;
        let x = self.proj_out.forward(&x)?;
        
        // Unpatchify
        let output = self.unpatchify(&x)?;
        
        Ok(ModelOutput {
            sample: output,
            ..Default::default()
        })
    }
    
    /// Embed and patchify image latents
    fn embed_image(&self, x: &Tensor) -> Result<Tensor> {
        let (batch_size, channels, height, width) = x.dims4()?;
        
        // Conv2d patchification
        let x = self.img_in.forward(x)?; // [B, hidden, h/patch, w/patch]
        
        // Reshape to sequence
        let x = x.permute((0, 2, 3, 1))?; // [B, h/patch, w/patch, hidden]
        let h_patches = height / self.config.patch_size;
        let w_patches = width / self.config.patch_size;
        let x = x.reshape(&[batch_size, h_patches * w_patches, self.config.hidden_size])?;
        
        // Add position embeddings
        let positions = Tensor::arange(0, x.dim(1)? as i64, x.device())?;
        let pos_emb = self.pos_embed.forward(&positions)?;
        x.broadcast_add(&pos_emb)
    }
    
    /// Unpatchify output back to image format
    fn unpatchify(&self, x: &Tensor) -> Result<Tensor> {
        let batch_size = x.dim(0)?;
        let seq_len = x.dim(1)?;
        let patch_size = self.config.patch_size;
        
        // Calculate spatial dimensions
        let h_patches = (seq_len as f64).sqrt() as usize;
        let w_patches = h_patches;
        
        // Reshape to image format
        let x = x.reshape(&[
            batch_size,
            h_patches,
            w_patches,
            patch_size,
            patch_size,
            self.config.out_channels,
        ])?;
        
        // Permute to standard image format
        x.permute((0, 5, 1, 3, 2, 4))?
            .reshape(&[
                batch_size,
                self.config.out_channels,
                h_patches * patch_size,
                w_patches * patch_size,
            ])
    }
}

/// Create sinusoidal timestep embeddings
fn timestep_embedding(timesteps: &Tensor, dim: usize) -> Result<Tensor> {
    let half_dim = dim / 2;
    let device = timesteps.device();
    
    // Create frequency bands
    let freqs = Tensor::arange(0, half_dim as i64, device)?
        .to_dtype(DType::F32)?
        .affine(-std::f64::consts::LN_2 * 10000.0f64.ln() / (half_dim - 1) as f64, 0.0)?
        .exp()?;
    
    // Apply to timesteps
    let args = timesteps.to_dtype(DType::F32)?.unsqueeze(1)?
        .broadcast_mul(&freqs.unsqueeze(0)?)?;
    
    // Create sinusoidal embeddings
    let cos = args.cos()?;
    let sin = args.sin()?;
    
    Tensor::cat(&[&cos, &sin], 1)
}

// Implement DiffusionModel trait
impl DiffusionModel for FluxModelWithLoRA {
    fn forward(&self, inputs: &ModelInputs) -> Result<ModelOutput> {
        self.forward(inputs)
    }
    
    fn architecture(&self) -> eridiffusion_core::ModelArchitecture {
        eridiffusion_core::ModelArchitecture::Flux
    }
    
    fn in_channels(&self) -> usize {
        self.config.in_channels
    }
    
    fn out_channels(&self) -> usize {
        self.config.out_channels
    }
    
    fn train_mode(&mut self, _mode: bool) {
        // LoRA layers don't need special train mode handling
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_flux_model_creation() -> Result<()> {
        let device = Device::Cpu;
        let config = FluxConfig {
            num_double_blocks: 2,  // Reduced for testing
            num_single_blocks: 2,
            ..Default::default()
        };
        
        let lora_config = LoRALayerConfig {
            rank: 16,
            alpha: 16.0,
            dropout: 0.0,
            use_bias: false,
        };
        
        let vs = candle_nn::VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&vs, DType::F32, &device);
        
        // Initialize all required weights
        // ... (weight initialization would go here in real implementation)
        
        let model = FluxModelWithLoRA::new(config, Some(lora_config), vb)?;
        
        // Check we have LoRA parameters
        let params = model.trainable_parameters();
        assert!(params.len() > 0, "Model should have trainable LoRA parameters");
        
        Ok(())
    }
}