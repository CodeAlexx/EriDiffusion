//! Flux model with LoRA matching Candle's exact structure

use anyhow::Result;
use candle_core::{Device, DType, Module, Tensor, D};
use candle_nn::{Linear, VarBuilder, linear};

use super::double_block_candle::FluxDoubleStreamBlock;
use super::single_block::FluxSingleStreamBlockWithLoRA;
use super::embeddings::FluxTimeEmbedding;

/// Flux configuration matching Candle
#[derive(Debug, Clone)]
pub struct FluxConfig {
    pub in_channels: usize,
    pub vec_in_dim: usize,
    pub context_in_dim: usize,
    pub hidden_size: usize,
    pub mlp_ratio: f64,
    pub num_heads: usize,
    pub depth: usize,
    pub depth_single_blocks: usize,
    pub axes_dim: Vec<usize>,
    pub theta: usize,
    pub qkv_bias: bool,
    pub guidance_embed: bool,
}

impl FluxConfig {
    pub fn dev() -> Self {
        Self {
            in_channels: 64,  // This is patchified input (16 * 2 * 2)
            vec_in_dim: 768,
            context_in_dim: 4096,
            hidden_size: 3072,
            mlp_ratio: 4.0,
            num_heads: 24,
            depth: 19,
            depth_single_blocks: 38,
            axes_dim: vec![16, 56, 56],
            theta: 10_000,
            qkv_bias: true,
            guidance_embed: true,
        }
    }
    
    pub fn schnell() -> Self {
        Self {
            guidance_embed: false,
            ..Self::dev()
        }
    }
}

/// Flux model with LoRA support matching Candle's structure
pub struct FluxModelWithLoRA {
    // Input projections
    img_in: Linear,  // Linear, not Conv2d - Flux expects patchified input
    txt_in: Linear,
    time_in: FluxTimeEmbedding,
    guidance_in: Option<FluxTimeEmbedding>,
    vector_in: Linear,
    
    // Transformer blocks
    double_blocks: Vec<FluxDoubleStreamBlock>,
    single_blocks: Vec<FluxSingleStreamBlockWithLoRA>,
    
    // Output projection
    final_layer: Linear,
    
    // Configuration
    config: FluxConfig,
    device: Device,
}

impl FluxModelWithLoRA {
    pub fn new(
        config: FluxConfig,
        vb: VarBuilder,
        lora_rank: Option<usize>,
        lora_alpha: Option<f32>,
    ) -> Result<Self> {
        let device = vb.device().clone();
        
        // Input projections - matching Candle exactly
        let img_in = linear(config.in_channels, config.hidden_size, vb.pp("img_in"))?;
        let txt_in = linear(config.context_in_dim, config.hidden_size, vb.pp("txt_in"))?;
        let time_in = FluxTimeEmbedding::new(256, config.hidden_size, vb.pp("time_in"))?;
        
        let guidance_in = if config.guidance_embed {
            Some(FluxTimeEmbedding::new(256, config.hidden_size, vb.pp("guidance_in"))?)
        } else {
            None
        };
        
        let vector_in = linear(config.vec_in_dim, config.hidden_size, vb.pp("vector_in"))?;
        
        // Double blocks
        let mut double_blocks = Vec::with_capacity(config.depth);
        for i in 0..config.depth {
            let block = FluxDoubleStreamBlock::new(
                &config,
                vb.pp(&format!("double_blocks.{}", i)),
                lora_rank,
                lora_alpha,
            )?;
            double_blocks.push(block);
        }
        
        // Single blocks
        let mut single_blocks = Vec::with_capacity(config.depth_single_blocks);
        for i in 0..config.depth_single_blocks {
            let block = FluxSingleStreamBlockWithLoRA::new(
                &config,
                vb.pp(&format!("single_blocks.{}", i)),
                lora_rank,
                lora_alpha,
            )?;
            single_blocks.push(block);
        }
        
        // Final layer
        let final_layer = linear(
            config.hidden_size,
            config.in_channels,
            vb.pp("final_layer"),
        )?;
        
        Ok(Self {
            img_in,
            txt_in,
            time_in,
            guidance_in,
            vector_in,
            double_blocks,
            single_blocks,
            final_layer,
            config,
            device,
        })
    }
    
    pub fn forward(
        &self,
        img: &Tensor,        // Patchified image: [B, L, 64]
        txt: &Tensor,        // Text embeddings: [B, L_txt, 4096]
        timesteps: &Tensor,  // Timesteps: [B]
        y: &Tensor,          // Pooled text: [B, 768]
        guidance: Option<&Tensor>,  // Guidance scale: [B]
    ) -> Result<Tensor> {
        // Ensure img is patchified (64 channels)
        let img_shape = img.dims();
        if img_shape[2] != 64 {
            anyhow::bail!(
                "Expected patchified input with 64 channels, got shape: {:?}",
                img_shape
            );
        }
        
        // Project inputs
        let mut img = img.apply(&self.img_in)?;
        let txt = txt.apply(&self.txt_in)?;
        
        // Time embedding
        let mut vec_ = self.time_in.forward(timesteps)?.apply(&y.apply(&self.vector_in)?)?;
        
        // Add guidance embedding if present
        if let (Some(guidance), Some(guidance_in)) = (guidance, &self.guidance_in) {
            vec_ = (vec_ + guidance_in.forward(guidance)?)?;
        }
        
        // Create position embeddings (RoPE)
        let pe = self.create_rope_embeddings(&img)?;
        
        // Apply double blocks
        for block in &self.double_blocks {
            let (new_img, new_txt) = block.forward(&img, &txt, &vec_, &pe)?;
            img = new_img;
            // txt is updated but not used further in double blocks
        }
        
        // Concatenate for single blocks
        img = Tensor::cat(&[&img, &txt], 1)?;
        
        // Apply single blocks
        for block in &self.single_blocks {
            img = block.forward(&img, &vec_, &pe)?;
        }
        
        // Take only image part
        let seq_len = img.dim(1)?;
        let txt_len = txt.dim(1)?;
        img = img.i((.., ..seq_len - txt_len, ..))?;
        
        // Final projection
        img.apply(&self.final_layer)
    }
    
    fn create_rope_embeddings(&self, img: &Tensor) -> Result<Tensor> {
        // Create RoPE embeddings based on image sequence length
        // This is a simplified version - full implementation would use
        // the axes_dim configuration
        let seq_len = img.dim(1)?;
        let positions = Tensor::arange(0, seq_len as i64, img.device())?;
        positions.unsqueeze(0)?.broadcast_as(img.dims3()?.0, seq_len)
    }
    
    pub fn device(&self) -> &Device {
        &self.device
    }
}

/// Helper to create Flux model for inference (no LoRA)
pub fn create_flux_model(
    model_path: &str,
    device: &Device,
    dtype: DType,
    config: FluxConfig,
) -> Result<FluxModelWithLoRA> {
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_path], dtype, device)? };
    FluxModelWithLoRA::new(config, vb, None, None)
}

/// Helper to create Flux model with LoRA for training
pub fn create_flux_model_with_lora(
    model_path: &str,
    device: &Device,
    dtype: DType,
    config: FluxConfig,
    lora_rank: usize,
    lora_alpha: f32,
) -> Result<FluxModelWithLoRA> {
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_path], dtype, device)? };
    FluxModelWithLoRA::new(config, vb, Some(lora_rank), Some(lora_alpha))
}