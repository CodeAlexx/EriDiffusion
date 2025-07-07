//! Flux Double Block with LoRA support
//! 
//! Implements the double-stream transformer blocks used in Flux,
//! with built-in LoRA adaptation for parameter-efficient fine-tuning.

use candle_core::{Tensor, Module, Result, Device, DType, D};
use candle_nn::{VarBuilder, LayerNorm, LayerNormConfig};
use eridiffusion_networks::{AttentionWithLoRA, FeedForwardWithLoRA, LoRALayerConfig, Activation};
use crate::flux::modulation::{Modulation, apply_modulation};

// Constants for Flux
const FLUX_NORM_GROUPS: usize = 32;  // Flux uses 32 groups for normalization
const FLUX_NORM_EPS: f64 = 1e-6;

// TODO: Import GroupNorm from eridiffusion ops module once available in models crate

/// Flux Double Block configuration
pub struct DoubleBlockConfig {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub mlp_ratio: f32,
    pub qkv_bias: bool,
    pub lora_config: Option<LoRALayerConfig>,
}

/// Flux Double Block with LoRA support
/// 
/// Processes both image and text streams with cross-modal interactions
pub struct FluxDoubleBlockWithLoRA {
    // Image stream components
    img_norm1: LayerNorm,
    img_modulation1: Modulation,
    img_attn: AttentionWithLoRA,
    
    img_norm2: LayerNorm,
    img_modulation2: Modulation,
    img_mlp: FeedForwardWithLoRA,
    
    // Text stream components
    txt_norm1: LayerNorm,
    txt_modulation1: Modulation,
    txt_attn: AttentionWithLoRA,
    
    txt_norm2: LayerNorm,
    txt_modulation2: Modulation,
    txt_mlp: FeedForwardWithLoRA,
    
    // Configuration
    config: DoubleBlockConfig,
    block_idx: usize,
}

impl FluxDoubleBlockWithLoRA {
    pub fn new(
        config: DoubleBlockConfig,
        block_idx: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let mlp_hidden_dim = (hidden_size as f32 * config.mlp_ratio) as usize;
        
        // Layer normalization configuration
        let ln_config = LayerNormConfig {
            eps: 1e-6,
            remove_mean: true,
            affine: true,
        };
        
        // Image stream components
        let img_norm1 = LayerNorm::new_with_config(hidden_size, ln_config, vb.pp("img_norm1"))?;
        let img_modulation1 = Modulation::new(hidden_size, vb.pp("img_mod1"))?;
        let img_attn = AttentionWithLoRA::new(
            hidden_size,
            config.num_heads,
            config.lora_config.as_ref(),
            vb.pp("img_attn"),
        )?;
        
        let img_norm2 = LayerNorm::new_with_config(hidden_size, ln_config, vb.pp("img_norm2"))?;
        let img_modulation2 = Modulation::new(hidden_size, vb.pp("img_mod2"))?;
        let img_mlp = FeedForwardWithLoRA::new(
            hidden_size,
            mlp_hidden_dim,
            config.lora_config.as_ref(),
            Activation::Gelu,
            vb.pp("img_mlp"),
        )?;
        
        // Text stream components
        let txt_norm1 = LayerNorm::new_with_config(hidden_size, ln_config, vb.pp("txt_norm1"))?;
        let txt_modulation1 = Modulation::new(hidden_size, vb.pp("txt_mod1"))?;
        let txt_attn = AttentionWithLoRA::new(
            hidden_size,
            config.num_heads,
            config.lora_config.as_ref(),
            vb.pp("txt_attn"),
        )?;
        
        let txt_norm2 = LayerNorm::new_with_config(hidden_size, ln_config, vb.pp("txt_norm2"))?;
        let txt_modulation2 = Modulation::new(hidden_size, vb.pp("txt_mod2"))?;
        let txt_mlp = FeedForwardWithLoRA::new(
            hidden_size,
            mlp_hidden_dim,
            config.lora_config.as_ref(),
            Activation::Gelu,
            vb.pp("txt_mlp"),
        )?;
        
        Ok(Self {
            img_norm1,
            img_modulation1,
            img_attn,
            img_norm2,
            img_modulation2,
            img_mlp,
            txt_norm1,
            txt_modulation1,
            txt_attn,
            txt_norm2,
            txt_modulation2,
            txt_mlp,
            config,
            block_idx,
        })
    }
    
    /// Get all trainable LoRA parameters
    pub fn trainable_parameters(&self) -> Vec<&candle_core::Var> {
        let mut params = Vec::new();
        params.extend(self.img_attn.trainable_parameters());
        params.extend(self.img_mlp.trainable_parameters());
        params.extend(self.txt_attn.trainable_parameters());
        params.extend(self.txt_mlp.trainable_parameters());
        params
    }
    
    /// Forward pass through the double block
    pub fn forward(
        &self,
        img: &Tensor,
        txt: &Tensor,
        vec: &Tensor,  // Modulation vector from time/guidance embedding
    ) -> Result<(Tensor, Tensor)> {
        // Process image stream
        let img_out = self.forward_img_stream(img, txt, vec)?;
        
        // Process text stream
        let txt_out = self.forward_txt_stream(txt, img, vec)?;
        
        Ok((img_out, txt_out))
    }
    
    /// Forward pass for image stream
    fn forward_img_stream(
        &self,
        img: &Tensor,
        txt: &Tensor,
        vec: &Tensor,
    ) -> Result<Tensor> {
        // Self-attention with modulation
        let (shift1, scale1, gate1) = self.img_modulation1.forward(vec)?;
        let img_norm = self.img_norm1.forward(img)?;
        let img_modulated = apply_modulation(&img_norm, &shift1, &scale1)?;
        
        // Concatenate image and text for cross-attention
        let combined = Tensor::cat(&[&img_modulated, txt], 1)?;
        let attn_out = self.img_attn.forward(&combined)?;
        
        // Extract image part
        let img_seq_len = img.dim(1)?;
        let img_attn_out = attn_out.narrow(1, 0, img_seq_len)?;
        
        // Gated residual connection
        let img = img.add(&(gate1.unsqueeze(1)?.broadcast_mul(&img_attn_out)?))?;
        
        // MLP with modulation
        let (shift2, scale2, gate2) = self.img_modulation2.forward(vec)?;
        let img_norm2 = self.img_norm2.forward(&img)?;
        let img_modulated2 = apply_modulation(&img_norm2, &shift2, &scale2)?;
        let mlp_out = self.img_mlp.forward(&img_modulated2)?;
        
        // Gated residual connection
        img.add(&(gate2.unsqueeze(1)?.broadcast_mul(&mlp_out)?))
    }
    
    /// Forward pass for text stream
    fn forward_txt_stream(
        &self,
        txt: &Tensor,
        img: &Tensor,
        vec: &Tensor,
    ) -> Result<Tensor> {
        // Self-attention with modulation
        let (shift1, scale1, gate1) = self.txt_modulation1.forward(vec)?;
        let txt_norm = self.txt_norm1.forward(txt)?;
        let txt_modulated = apply_modulation(&txt_norm, &shift1, &scale1)?;
        
        // Concatenate text and image for cross-attention
        let combined = Tensor::cat(&[&txt_modulated, img], 1)?;
        let attn_out = self.txt_attn.forward(&combined)?;
        
        // Extract text part
        let txt_seq_len = txt.dim(1)?;
        let txt_attn_out = attn_out.narrow(1, 0, txt_seq_len)?;
        
        // Gated residual connection
        let txt = txt.add(&(gate1.unsqueeze(1)?.broadcast_mul(&txt_attn_out)?))?;
        
        // MLP with modulation
        let (shift2, scale2, gate2) = self.txt_modulation2.forward(vec)?;
        let txt_norm2 = self.txt_norm2.forward(&txt)?;
        let txt_modulated2 = apply_modulation(&txt_norm2, &shift2, &scale2)?;
        let mlp_out = self.txt_mlp.forward(&txt_modulated2)?;
        
        // Gated residual connection
        txt.add(&(gate2.unsqueeze(1)?.broadcast_mul(&mlp_out)?))
    }
}

/// Single Block configuration
pub struct SingleBlockConfig {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub mlp_ratio: f32,
    pub lora_config: Option<LoRALayerConfig>,
}

/// Flux Single Block with LoRA support
/// 
/// Processes concatenated image-text features
pub struct FluxSingleBlockWithLoRA {
    norm1: LayerNorm,
    modulation1: Modulation,
    attn: AttentionWithLoRA,
    
    norm2: LayerNorm,
    modulation2: Modulation,
    mlp: FeedForwardWithLoRA,
    
    config: SingleBlockConfig,
    block_idx: usize,
}

impl FluxSingleBlockWithLoRA {
    pub fn new(
        config: SingleBlockConfig,
        block_idx: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let mlp_hidden_dim = (hidden_size as f32 * config.mlp_ratio) as usize;
        
        let ln_config = LayerNormConfig {
            eps: 1e-6,
            remove_mean: true,
            affine: true,
        };
        
        Ok(Self {
            norm1: LayerNorm::new_with_config(hidden_size, ln_config, vb.pp("norm1"))?,
            modulation1: Modulation::new(hidden_size, vb.pp("mod1"))?,
            attn: AttentionWithLoRA::new(
                hidden_size,
                config.num_heads,
                config.lora_config.as_ref(),
                vb.pp("attn"),
            )?,
            norm2: LayerNorm::new_with_config(hidden_size, ln_config, vb.pp("norm2"))?,
            modulation2: Modulation::new(hidden_size, vb.pp("mod2"))?,
            mlp: FeedForwardWithLoRA::new(
                hidden_size,
                mlp_hidden_dim,
                config.lora_config.as_ref(),
                Activation::Gelu,
                vb.pp("mlp"),
            )?,
            config,
            block_idx,
        })
    }
    
    pub fn trainable_parameters(&self) -> Vec<&candle_core::Var> {
        let mut params = Vec::new();
        params.extend(self.attn.trainable_parameters());
        params.extend(self.mlp.trainable_parameters());
        params
    }
    
    pub fn forward(&self, x: &Tensor, vec: &Tensor) -> Result<Tensor> {
        // Self-attention with modulation
        let (shift1, scale1, gate1) = self.modulation1.forward(vec)?;
        let x_norm = self.norm1.forward(x)?;
        let x_modulated = apply_modulation(&x_norm, &shift1, &scale1)?;
        let attn_out = self.attn.forward(&x_modulated)?;
        
        // Gated residual
        let x = x.add(&(gate1.unsqueeze(1)?.broadcast_mul(&attn_out)?))?;
        
        // MLP with modulation
        let (shift2, scale2, gate2) = self.modulation2.forward(vec)?;
        let x_norm2 = self.norm2.forward(&x)?;
        let x_modulated2 = apply_modulation(&x_norm2, &shift2, &scale2)?;
        let mlp_out = self.mlp.forward(&x_modulated2)?;
        
        // Gated residual
        x.add(&(gate2.unsqueeze(1)?.broadcast_mul(&mlp_out)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_double_block_forward() -> Result<()> {
        let device = Device::Cpu;
        let batch_size = 2;
        let img_seq_len = 256;
        let txt_seq_len = 77;
        let hidden_size = 1024;
        
        let config = DoubleBlockConfig {
            hidden_size,
            num_heads: 16,
            mlp_ratio: 4.0,
            qkv_bias: true,
            lora_config: Some(LoRALayerConfig {
                rank: 16,
                alpha: 16.0,
                dropout: 0.0,
                use_bias: false,
            }),
        };
        
        // Create var builder
        let vs = candle_nn::VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&vs, DType::F32, &device);
        
        // Initialize required weights
        // ... (weight initialization code)
        
        let block = FluxDoubleBlockWithLoRA::new(config, 0, vb)?;
        
        // Test inputs
        let img = Tensor::randn(0.0, 1.0, &[batch_size, img_seq_len, hidden_size], &device)?;
        let txt = Tensor::randn(0.0, 1.0, &[batch_size, txt_seq_len, hidden_size], &device)?;
        let vec = Tensor::randn(0.0, 1.0, &[batch_size, hidden_size], &device)?;
        
        let (img_out, txt_out) = block.forward(&img, &txt, &vec)?;
        
        assert_eq!(img_out.shape().dims(), &[batch_size, img_seq_len, hidden_size]);
        assert_eq!(txt_out.shape().dims(), &[batch_size, txt_seq_len, hidden_size]);
        
        // Check LoRA parameters
        let params = block.trainable_parameters();
        assert!(params.len() > 0, "Should have trainable LoRA parameters");
        
        Ok(())
    }
}