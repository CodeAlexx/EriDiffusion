//! LoRA-enabled transformer layers for SDXL
//! This module provides spatial transformers with integrated LoRA support

use candle_core::{Device, DType, Module, Result, Tensor, D};
use candle_nn as nn;
use super::lora_attention::{BasicTransformerBlockWithLoRA, CrossAttentionWithLoRA, LoRAAttentionConfig};
use super::sdxl_lora_layer::LoRALinear;

/// Configuration for SpatialTransformer
#[derive(Debug, Clone, Copy)]
pub struct SpatialTransformerConfig {
    pub depth: usize,
    pub num_groups: usize,
    pub context_dim: Option<usize>,
    pub sliced_attention_size: Option<usize>,
    pub use_linear_projection: bool,
}

impl Default for SpatialTransformerConfig {
    fn default() -> Self {
        Self {
            depth: 1,
            num_groups: 32,
            context_dim: None,
            sliced_attention_size: None,
            use_linear_projection: false,
        }
    }
}

/// Projection layer type
enum Proj {
    Conv2d(nn::Conv2d),
    Linear(nn::Linear),
}

impl Module for Proj {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Proj::Conv2d(conv) => conv.forward(xs),
            Proj::Linear(linear) => linear.forward(xs),
        }
    }
}

/// SpatialTransformer with LoRA support
/// This is the main transformer block used in SDXL's UNet
pub struct SpatialTransformerWithLoRA {
    norm: nn::GroupNorm,
    proj_in: Proj,
    transformer_blocks: Vec<BasicTransformerBlockWithLoRA>,
    proj_out: Proj,
    config: SpatialTransformerConfig,
}

impl SpatialTransformerWithLoRA {
    pub fn new(
        vs: nn::VarBuilder,
        in_channels: usize,
        n_heads: usize,
        d_head: usize,
        use_flash_attn: bool,
        config: SpatialTransformerConfig,
        lora_config: Option<&LoRAAttentionConfig>,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        let inner_dim = n_heads * d_head;
        
        // Group normalization
        let norm = nn::group_norm(config.num_groups, in_channels, 1e-6, vs.pp("norm"))?;
        
        // Input projection
        let proj_in = if config.use_linear_projection {
            Proj::Linear(nn::linear(in_channels, inner_dim, vs.pp("proj_in"))?)
        } else {
            Proj::Conv2d(nn::conv2d(
                in_channels,
                inner_dim,
                1,
                Default::default(),
                vs.pp("proj_in"),
            )?)
        };
        
        // Transformer blocks with LoRA
        let mut transformer_blocks = Vec::new();
        let vs_tb = vs.pp("transformer_blocks");
        
        for index in 0..config.depth {
            let block = BasicTransformerBlockWithLoRA::new(
                vs_tb.pp(index.to_string()),
                inner_dim,
                n_heads,
                d_head,
                config.context_dim,
                config.sliced_attention_size,
                use_flash_attn,
                lora_config,
                device.clone(),
                dtype,
            )?;
            transformer_blocks.push(block);
        }
        
        // Output projection
        let proj_out = if config.use_linear_projection {
            Proj::Linear(nn::linear(inner_dim, in_channels, vs.pp("proj_out"))?)
        } else {
            Proj::Conv2d(nn::conv2d(
                inner_dim,
                in_channels,
                1,
                Default::default(),
                vs.pp("proj_out"),
            )?)
        };
        
        Ok(Self {
            norm,
            proj_in,
            transformer_blocks,
            proj_out,
            config,
        })
    }
    
    pub fn forward(&self, xs: &Tensor, context: Option<&Tensor>) -> Result<Tensor> {
        let (batch, channel, height, width) = xs.dims4()?;
        let residual = xs;
        
        // Normalize
        let xs = self.norm.forward(xs)?;
        
        // Project and reshape for transformer
        let xs = match &self.proj_in {
            Proj::Conv2d(_) => {
                let xs = self.proj_in.forward(&xs)?;
                let inner_dim = xs.dim(1)?;
                // Reshape from (B, C, H, W) to (B, H*W, C)
                xs.permute((0, 2, 3, 1))?
                    .reshape((batch, height * width, inner_dim))?
            }
            Proj::Linear(_) => {
                // First reshape to (B, H*W, C)
                let xs = xs.permute((0, 2, 3, 1))?
                    .reshape((batch, height * width, channel))?;
                self.proj_in.forward(&xs)?
            }
        };
        
        // Apply transformer blocks
        let mut xs = xs;
        for block in &self.transformer_blocks {
            xs = block.forward(&xs, context)?;
        }
        
        // Project back and reshape
        let xs = match &self.proj_out {
            Proj::Conv2d(_) => {
                let inner_dim = xs.dim(2)?;
                // Reshape from (B, H*W, C) to (B, C, H, W)
                let xs = xs.reshape((batch, height, width, inner_dim))?
                    .permute((0, 3, 1, 2))?;
                self.proj_out.forward(&xs)?
            }
            Proj::Linear(_) => {
                let xs = self.proj_out.forward(&xs)?;
                // Reshape from (B, H*W, C) to (B, C, H, W)
                xs.reshape((batch, height, width, channel))?
                    .permute((0, 3, 1, 2))?
            }
        };
        
        // Add residual connection
        xs + residual
    }
    
    /// Get all LoRA layers from all transformer blocks
    pub fn get_lora_layers(&self) -> Vec<(&str, &LoRALinear)> {
        let mut all_layers = Vec::new();
        
        for (block_idx, block) in self.transformer_blocks.iter().enumerate() {
            for (layer_name, layer) in block.get_lora_layers() {
                // Add block index to distinguish between blocks
                all_layers.push((layer_name, layer));
            }
        }
        
        all_layers
    }
    
    /// Get the number of transformer blocks
    pub fn depth(&self) -> usize {
        self.transformer_blocks.len()
    }
}

/// Helper to collect all LoRA layers from a model with spatial transformers
pub struct LoRALayerCollector<'a> {
    layers: Vec<(String, &'a LoRALinear)>,
}

impl<'a> LoRALayerCollector<'a> {
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
        }
    }
    
    /// Add a spatial transformer's LoRA layers with a prefix
    pub fn add_spatial_transformer(
        &mut self,
        prefix: &str,
        transformer: &'a SpatialTransformerWithLoRA,
    ) {
        for (layer_name, layer) in transformer.get_lora_layers() {
            let full_name = format!("{}.{}", prefix, layer_name);
            self.layers.push((full_name, layer));
        }
    }
    
    /// Get all collected layers
    pub fn into_layers(self) -> Vec<(String, &'a LoRALinear)> {
        self.layers
    }
}