//! Minimal Flux model for testing on limited VRAM
//! 
//! This is a reduced version of Flux with:
//! - Fewer transformer blocks (4 double, 8 single instead of 19/38)
//! - Smaller hidden dimension (1536 instead of 3072)
//! - Reduced attention heads (12 instead of 24)

use candle_core::{Device, DType, Module, Result, Tensor, D};
use candle_nn::{linear, Linear, VarBuilder};
use serde::{Serialize, Deserialize};

use crate::models::flux_custom::{
    FluxDoubleBlockWithLoRA, FluxSingleBlockWithLoRA,
    LinearWithLoRA, utils::{EmbedND, MLP},
};
use crate::models::flux_custom::lora::{LoRAConfig, LoRACompatible};

/// Configuration for minimal Flux model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinimalFluxConfig {
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

impl Default for MinimalFluxConfig {
    fn default() -> Self {
        Self {
            in_channels: 64,          // Same as full Flux
            vec_in_dim: 768,          // Same as full Flux
            context_in_dim: 4096,     // Same as full Flux
            hidden_size: 1536,        // Reduced from 3072
            mlp_ratio: 4.0,          // Same as full Flux
            num_heads: 12,           // Reduced from 24
            depth: 4,                // Reduced from 19
            depth_single_blocks: 8,  // Reduced from 38
            axes_dim: vec![16, 56, 56],
            theta: 10_000.0,
            qkv_bias: true,
            guidance_embed: true,
        }
    }
}

/// Minimal Flux model with LoRA support
pub struct MinimalFluxModelWithLoRA {
    config: MinimalFluxConfig,
    
    // Input projections
    img_in: Linear,
    txt_in: Linear,
    time_in: MLP,
    vector_in: MLP,
    
    // Positional embeddings
    pe_embedder: EmbedND,
    
    // Transformer blocks (reduced count)
    double_blocks: Vec<FluxDoubleBlockWithLoRA>,
    single_blocks: Vec<FluxSingleBlockWithLoRA>,
    
    // Output projection
    final_layer: LinearWithLoRA,
    
    // Guidance embedding (optional)
    guidance_in: Option<MLP>,
    
    hidden_size: usize,
    num_heads: usize,
}

impl MinimalFluxModelWithLoRA {
    pub fn new(config: &MinimalFluxConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_size = config.hidden_size;
        
        println!("Creating MinimalFluxModel with reduced parameters:");
        println!("  Hidden size: {} (vs 3072 in full)", hidden_size);
        println!("  Double blocks: {} (vs 19 in full)", config.depth);
        println!("  Single blocks: {} (vs 38 in full)", config.depth_single_blocks);
        println!("  Attention heads: {} (vs 24 in full)", config.num_heads);
        
        // Input projections
        let img_in = linear(config.in_channels, hidden_size, vb.pp("img_in"))?;
        let txt_in = linear(config.context_in_dim, hidden_size, vb.pp("txt_in"))?;
        
        let time_in = MLP::new(256, hidden_size, hidden_size, vb.pp("time_in"))?;
        let vector_in = MLP::new(config.vec_in_dim, hidden_size, hidden_size, vb.pp("vector_in"))?;
        
        // Positional embeddings
        let pe_embedder = EmbedND::new(hidden_size, &config.axes_dim, vb.pp("pe_embedder"))?;
        
        // Transformer blocks
        let mut double_blocks = Vec::new();
        for i in 0..config.depth {
            double_blocks.push(FluxDoubleBlockWithLoRA::new(
                hidden_size,
                config.num_heads,
                config.mlp_ratio,
                Some(0.0),
                i,
                vb.pp(&format!("double_blocks.{}", i)),
            )?);
        }
        
        let mut single_blocks = Vec::new();
        for i in 0..config.depth_single_blocks {
            single_blocks.push(FluxSingleBlockWithLoRA::new(
                hidden_size,
                config.num_heads,
                config.mlp_ratio,
                Some(0.0),
                i,
                vb.pp(&format!("single_blocks.{}", i)),
            )?);
        }
        
        // Final output projection
        let final_layer = LinearWithLoRA::new(
            hidden_size,
            config.in_channels,
            "final_layer".to_string(),
            vb.pp("final_layer"),
        )?;
        
        // Guidance embedding
        let guidance_in = if config.guidance_embed {
            Some(MLP::new(256, hidden_size, hidden_size, vb.pp("guidance_in"))?)
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
    
    /// Add LoRA to all applicable layers
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
    
    /// Forward pass (simplified)
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
        // This is a placeholder - full implementation would match Flux forward pass
        // For now, just return a dummy output of the right shape
        Ok(img.clone())
    }
    
    /// Estimate memory usage
    pub fn estimate_memory_gb(&self) -> f32 {
        let params_per_double_block = self.hidden_size * self.hidden_size * 16; // Rough estimate
        let params_per_single_block = self.hidden_size * self.hidden_size * 8;  // Rough estimate
        
        let total_params = 
            self.hidden_size * 1000 +  // Input projections
            self.config.depth * params_per_double_block +
            self.config.depth_single_blocks * params_per_single_block +
            self.hidden_size * self.config.in_channels; // Output projection
            
        // Assuming BF16 (2 bytes per param)
        let memory_bytes = total_params * 2;
        memory_bytes as f32 / 1e9
    }
}

/// Create a minimal Flux config that fits in limited VRAM
pub fn create_minimal_config_for_vram(vram_gb: f32) -> MinimalFluxConfig {
    // Rough estimation: we need space for model + gradients + optimizer states + activations
    // Model takes about 1/4 of total VRAM in practice
    let model_budget_gb = vram_gb / 4.0;
    
    if model_budget_gb >= 6.0 {
        // Can use default minimal config
        MinimalFluxConfig::default()
    } else if model_budget_gb >= 3.0 {
        // Need even smaller model
        MinimalFluxConfig {
            hidden_size: 1024,
            num_heads: 8,
            depth: 2,
            depth_single_blocks: 4,
            ..MinimalFluxConfig::default()
        }
    } else {
        // Tiny model for testing
        MinimalFluxConfig {
            hidden_size: 512,
            num_heads: 4,
            depth: 1,
            depth_single_blocks: 2,
            ..MinimalFluxConfig::default()
        }
    }
}