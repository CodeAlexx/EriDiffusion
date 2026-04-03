use crate::loaders::WeightLoader;
use flame_core::{DType, Shape, Tensor};
use flame_core::device::Device;
use flame_core::optimizers::{Adam, SGD};
use crate::ops::Conv2d;
use crate::ops::Linear;
use crate::ops::GroupNorm;
use super::{CrossAttentionWithLoRA, LoRAAttentionConfig};
use std::collections::HashMap;
use super::lora_attention::{BasicTransformerBlockWithLoRA, sdxl_lora_layer::LoRALinear};
use flame_core::{Result};

pub struct LoRALayerCollector<'a> {
    layers: Vec<(String, &'a LoRALinear)>,
}

// LoRA-enabled transformer layers for SDXL
// This module provides spatial transformers with integrated LoRA support

/// Configuration for SpatialTransformer
#[derive(Debug, Clone, Copy)]
pub struct SpatialTransformerConfig {
    pub depth: usize,
    pub num_groups: usize,
    pub context_dim: Option<usize>,
    pub use_linear_projection: bool,
    pub attention_type: AttentionType,
}

impl Default for SpatialTransformerConfig {
    fn default() -> Self {
        Self {
            depth: 1,
            num_groups: 32,
            context_dim: None,
            use_linear_projection: false,
            attention_type: AttentionType::SelfAttention,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AttentionType {
    SelfAttention,
    CrossAttention,
}

/// Projection layer (Linear or Conv2d)
pub enum Proj {
    Linear(Linear),
    Conv2d(Conv2d),
}

impl Proj {
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Proj::Linear(linear) => linear.forward(xs),
            Proj::Conv2d(conv) => conv.forward(xs),
        }
    }
}

/// SpatialTransformer with LoRA support
/// This is the main transformer block used in SDXL's UNet
pub struct SpatialTransformerWithLoRA {
    norm: GroupNorm,
    proj_in: Proj,
    transformer_blocks: Vec<BasicTransformerBlockWithLoRA>,
    proj_out: Proj,
    config: SpatialTransformerConfig,
}

impl SpatialTransformerWithLoRA {
    pub fn new(
        loader: &WeightLoader,
        in_channels: usize,
        n_heads: usize,
        d_head: usize,
        use_flash_attn: bool,
        config: SpatialTransformerConfig,
        lora_config: Option<&LoRAAttentionConfig>,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let inner_dim = n_heads * d_head;

        // Group normalization
        let norm = GroupNorm::new(
            config.num_groups,
            in_channels,
            1e-6,
            true,
            device.cuda_device().clone(),
        )?;

        // Input projection
        let proj_in = if config.use_linear_projection {
            // TODO: Load weights from pre-trained model
            Proj::Linear(Linear::new(in_channels, inner_dim, true, &device.cuda_device())?)
        } else {
            {
                // Create Conv2d with proper constructor
                let conv = Conv2d::new(in_channels, inner_dim, 1, 1, 0, device.cuda_device().clone())?;
                // TODO: Load weights from loader.tensor("proj_in.weight") and bias
                Proj::Conv2d(conv)
            }
        };

        // Transformer blocks with LoRA
        let mut transformer_blocks = Vec::new();
        
        for index in 0..config.depth {
            let tb_loader = loader.pp(&format!("transformer_blocks.{}", index));
            let block = BasicTransformerBlockWithLoRA::new(
                &tb_loader,
                inner_dim,
                n_heads,
                d_head,
                config.context_dim,
                use_flash_attn,
                lora_config,
                device.clone(),
                dtype,
            )?;
            transformer_blocks.push(block);
        }

        // Output projection
        let proj_out = if config.use_linear_projection {
            // TODO: Load weights from pre-trained model  
            Proj::Linear(Linear::new(inner_dim, in_channels, true, &device.cuda_device())?)
        } else {
            {
                // Create Conv2d with proper constructor
                let conv = Conv2d::new(inner_dim, in_channels, 1, 1, 0, device.cuda_device().clone())?;
                // TODO: Load weights from loader.tensor("proj_out.weight") and bias
                Proj::Conv2d(conv)
            }
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
        let dims = xs.shape().dims();
        let (batch, channel, height, width) = (dims[0], dims[1], dims[2], dims[3]);
        let residual = xs.clone();

        // Normalize
        let xs = self.norm.forward(xs)?;

        // Project in
        let inner = self.proj_in.forward(&xs)?;

        // Reshape for transformer: (b, c, h, w) -> (b, h*w, c)
        let inner_dims = inner.shape().dims();
        let inner_dim = inner_dims[1];
        let seq_len = height * width;
        let hidden_states = inner
            .reshape((batch, inner_dim, seq_len))?
            .transpose_dims(1, 2)?;

        // Apply transformer blocks
        let mut hidden_states = hidden_states;
        for block in &self.transformer_blocks {
            hidden_states = block.forward(&hidden_states, context)?;
        }

        // Reshape back: (b, h*w, c) -> (b, c, h, w)
        let hidden_states = hidden_states
            .transpose_dims(1, 2)?
            .reshape((batch, inner_dim, height, width))?;

        // Project out
        let output = self.proj_out.forward(&hidden_states)?;

        // Add residual
        output.add(&residual)
    }

    /// Collect all LoRA layers for saving/loading
    pub fn collect_lora_layers(&self) -> Vec<(String, &LoRALinear)> {
        let mut layers = Vec::new();

        for (block_idx, block) in self.transformer_blocks.iter().enumerate() {
            let block_prefix = format!("transformer_blocks.{}", block_idx);
            let block_layers = block.collect_lora_layers();
            
            for (name, layer) in block_layers {
                layers.push((format!("{}.{}", block_prefix, name), layer));
            }
        }

        layers
    }

    /// Get trainable parameters (LoRA only)
    pub fn trainable_parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        
        for block in &self.transformer_blocks {
            params.extend(block.trainable_parameters());
        }
        
        params
    }
}

/// Create a spatial transformer block for SDXL
pub fn spatial_transformer_sdxl(
    loader: &WeightLoader,
    in_channels: usize,
    n_heads: usize,
    d_head: usize,
    depth: usize,
    context_dim: usize,
    use_linear_projection: bool,
    use_flash_attn: bool,
    lora_config: Option<&LoRAAttentionConfig>,
    device: Device,
    dtype: DType,
) -> Result<SpatialTransformerWithLoRA> {
    let config = SpatialTransformerConfig {
        depth,
        num_groups: 32,
        context_dim: Some(context_dim),
        use_linear_projection,
        attention_type: AttentionType::CrossAttention,
    };

    SpatialTransformerWithLoRA::new(
        loader,
        in_channels,
        n_heads,
        d_head,
        use_flash_attn,
        config,
        lora_config,
        device,
        dtype,
    )
}