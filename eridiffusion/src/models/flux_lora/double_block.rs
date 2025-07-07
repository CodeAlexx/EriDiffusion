//! Flux Double Block with LoRA support
//! 
//! Implements the double-stream transformer blocks used in Flux,
//! with built-in LoRA adaptation for parameter-efficient fine-tuning.

use candle_core::{Tensor, Module, Result, Device, DType, D};
use candle_nn::VarBuilder;
use super::lora_layers::{FeedForwardWithLoRA, Activation};
use super::lora_config::LoRALayerConfig;
use super::modulation::{Modulation, Modulation2, ModulationOut, ModulationParams, apply_modulation};
use super::norm_wrapper::FluxNorm;
use super::attention_flux::FluxAttentionWithLoRA;
use super::lora_layers::AttentionWithLoRA;
use crate::ops::{GroupNorm, group_norm, get_2d_positions};
use std::cell::RefCell;

// Constants for Flux
pub const FLUX_NORM_GROUPS: usize = 32;  // Flux uses 32 groups for normalization
pub const FLUX_NORM_EPS: f64 = 1e-6;

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
    // Modulation layers (one per stream, outputs 2 sets of params each)
    img_mod: Modulation2,
    txt_mod: Modulation2,
    
    // Normalization layers (created as ones, not loaded)
    img_norm1: FluxNorm,
    img_norm2: FluxNorm,
    txt_norm1: FluxNorm,
    txt_norm2: FluxNorm,
    
    // Attention layers
    img_attn: FluxAttentionWithLoRA,
    txt_attn: FluxAttentionWithLoRA,
    
    // MLP layers
    img_mlp: FeedForwardWithLoRA,
    txt_mlp: FeedForwardWithLoRA,
    
    // Configuration
    config: DoubleBlockConfig,
    block_idx: usize,
    
    // Image dimensions for RoPE position generation (interior mutability for const forward)
    img_dims: RefCell<(usize, usize)>,
}

impl FluxDoubleBlockWithLoRA {
    pub fn new(
        config: DoubleBlockConfig,
        block_idx: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let mlp_hidden_dim = (hidden_size as f32 * config.mlp_ratio) as usize;
        
        // Modulation layers - one per stream, outputs 2 sets each
        let img_mod = Modulation2::new(hidden_size, vb.pp("img_mod"))?;
        let txt_mod = Modulation2::new(hidden_size, vb.pp("txt_mod"))?;
        
        // Normalization layers (created as ones, not loaded from model)
        let img_norm1 = FluxNorm::new(hidden_size, vb.pp("img_norm1"))?;
        let img_norm2 = FluxNorm::new(hidden_size, vb.pp("img_norm2"))?;
        let txt_norm1 = FluxNorm::new(hidden_size, vb.pp("txt_norm1"))?;
        let txt_norm2 = FluxNorm::new(hidden_size, vb.pp("txt_norm2"))?;
        
        // Attention layers with Flux structure
        let img_attn = FluxAttentionWithLoRA::new(
            hidden_size,
            config.num_heads,
            config.qkv_bias,
            config.lora_config.as_ref(),
            vb.pp("img_attn"),
        )?;
        
        let txt_attn = FluxAttentionWithLoRA::new(
            hidden_size,
            config.num_heads,
            config.qkv_bias,
            config.lora_config.as_ref(),
            vb.pp("txt_attn"),
        )?;
        
        // MLP layers
        let img_mlp = FeedForwardWithLoRA::new(
            hidden_size,
            mlp_hidden_dim,
            config.lora_config.as_ref(),
            Activation::Gelu,
            vb.pp("img_mlp"),
        )?;
        
        let txt_mlp = FeedForwardWithLoRA::new(
            hidden_size,
            mlp_hidden_dim,
            config.lora_config.as_ref(),
            Activation::Gelu,
            vb.pp("txt_mlp"),
        )?;
        
        Ok(Self {
            img_mod,
            txt_mod,
            img_norm1,
            img_norm2,
            txt_norm1,
            txt_norm2,
            img_attn,
            txt_attn,
            img_mlp,
            txt_mlp,
            config,
            block_idx,
            img_dims: RefCell::new((64, 64)),  // Default, will be updated dynamically
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
    
    /// Update image dimensions for RoPE position generation
    pub fn set_image_dimensions(&self, height: usize, width: usize) {
        *self.img_dims.borrow_mut() = (height, width);
    }
    
    /// Forward pass through the double block (matching candle structure)
    pub fn forward(
        &self,
        img: &Tensor,
        txt: &Tensor,
        vec: &Tensor,  // Modulation vector from time/guidance embedding
        pe: &Tensor,   // Position embeddings
    ) -> Result<(Tensor, Tensor)> {
        // Get modulation parameters for both streams
        let (img_mod1, img_mod2) = self.img_mod.forward(vec)?;
        let (txt_mod1, txt_mod2) = self.txt_mod.forward(vec)?;
        
        // Apply first modulation and attention for image
        let img_modulated = img.apply(&self.img_norm1)?;
        let img_modulated = img_mod1.scale_shift(&img_modulated)?;
        let (img_q, img_k, img_v) = self.img_attn.qkv(&img_modulated)?;
        
        // Apply first modulation and attention for text
        let txt_modulated = txt.apply(&self.txt_norm1)?;
        let txt_modulated = txt_mod1.scale_shift(&txt_modulated)?;
        let (txt_q, txt_k, txt_v) = self.txt_attn.qkv(&txt_modulated)?;
        
        // Concatenate for cross-attention
        let q = Tensor::cat(&[&txt_q, &img_q], 2)?;
        let k = Tensor::cat(&[&txt_k, &img_k], 2)?;
        let v = Tensor::cat(&[&txt_v, &img_v], 2)?;
        
        // Apply attention with RoPE
        let attn = crate::ops::attention(&q, &k, &v, pe)?;
        let txt_attn = attn.narrow(1, 0, txt.dim(1)?)?;
        let img_attn = attn.narrow(1, txt.dim(1)?, attn.dim(1)? - txt.dim(1)?)?;
        
        // Apply gating and residual for image
        let img = (img + img_mod1.gate(&img_attn.apply(self.img_attn.proj())?)?)?;
        let img = (&img + img_mod2.gate(
            &img_mod2.scale_shift(&img.apply(&self.img_norm2)?)?.apply(&self.img_mlp)?
        )?)?;
        
        // Apply gating and residual for text
        let txt = (txt + txt_mod1.gate(&txt_attn.apply(self.txt_attn.proj())?)?)?;
        let txt = (&txt + txt_mod2.gate(
            &txt_mod2.scale_shift(&txt.apply(&self.txt_norm2)?)?.apply(&self.txt_mlp)?
        )?)?;
        
        Ok((img, txt))
    }
    
    /// Forward pass for image stream
    fn forward_img_stream(
        &self,
        img: &Tensor,
        txt: &Tensor,
        vec: &Tensor,
    ) -> Result<Tensor> {
        // Self-attention with modulation
        let (mod1, mod2) = self.img_mod.forward(vec)?;
        let img_norm = self.img_norm1.forward(img)?;
        let img_modulated = mod1.scale_shift(&img_norm)?;
        
        // Generate 2D positions for image patches
        let batch_size = img.dim(0)?;
        let img_seq_len = img.dim(1)?;
        let device = img.device();
        
        // For Flux, images are patchified into 2x2 patches
        // So a 64x64 latent becomes 32x32 patches = 1024 positions
        let (img_height, img_width) = *self.img_dims.borrow();
        let positions = get_2d_positions(img_height / 2, img_width / 2, device)?;
        
        // Concatenate image and text for cross-attention
        let combined = Tensor::cat(&[&img_modulated, txt], 1)?;
        
        // Apply attention (RoPE handling would be done internally)
        // For now, just use standard forward
        let attn_out = self.img_attn.forward(&combined)?;
        
        // Extract image part
        let img_attn_out = attn_out.narrow(1, 0, img_seq_len)?;
        
        // Gated residual connection
        let img = img.add(&mod1.gate(&img_attn_out)?)?;
        
        // MLP with modulation
        let img_norm2 = self.img_norm2.forward(&img)?;
        let img_modulated2 = mod2.scale_shift(&img_norm2)?;
        let mlp_out = self.img_mlp.forward(&img_modulated2)?;
        
        // Gated residual connection
        img.add(&mod2.gate(&mlp_out)?)
    }
    
    /// Forward pass for text stream
    fn forward_txt_stream(
        &self,
        txt: &Tensor,
        img: &Tensor,
        vec: &Tensor,
    ) -> Result<Tensor> {
        // Self-attention with modulation
        let (mod1, mod2) = self.txt_mod.forward(vec)?;
        let txt_norm = self.txt_norm1.forward(txt)?;
        let txt_modulated = mod1.scale_shift(&txt_norm)?;
        
        // Concatenate text and image for cross-attention
        let combined = Tensor::cat(&[&txt_modulated, img], 1)?;
        let attn_out = self.txt_attn.forward(&combined)?;
        
        // Extract text part
        let txt_seq_len = txt.dim(1)?;
        let txt_attn_out = attn_out.narrow(1, 0, txt_seq_len)?;
        
        // Gated residual connection
        let txt = txt.add(&mod1.gate(&txt_attn_out)?)?;
        
        // MLP with modulation
        let txt_norm2 = self.txt_norm2.forward(&txt)?;
        let txt_modulated2 = mod2.scale_shift(&txt_norm2)?;
        let mlp_out = self.txt_mlp.forward(&txt_modulated2)?;
        
        // Gated residual connection
        txt.add(&mod2.gate(&mlp_out)?)
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
    norm1: FluxNorm,
    modulation1: Modulation,
    attn: AttentionWithLoRA,
    
    norm2: FluxNorm,
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
        
        Ok(Self {
            norm1: FluxNorm::new(hidden_size, vb.pp("norm1"))?,
            modulation1: Modulation::new(hidden_size, hidden_size, false, vb.pp("mod1"))?,
            attn: AttentionWithLoRA::new(
                hidden_size,
                config.num_heads,
                config.lora_config.as_ref(),
                vb.pp("attn"),
            )?,
            norm2: FluxNorm::new(hidden_size, vb.pp("norm2"))?,
            modulation2: Modulation::new(hidden_size, hidden_size, false, vb.pp("mod2"))?,
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
        let mod_params = self.modulation1.forward(vec)?;
        let (shift1, scale1, gate1) = match mod_params {
            ModulationParams::Single { shift, scale, gate } => (shift, scale, gate),
            _ => panic!("Expected Single modulation params"),
        };
        let x_norm = self.norm1.forward(x)?;
        let x_modulated = apply_modulation(&x_norm, &shift1, &scale1)?;
        let attn_out = self.attn.forward(&x_modulated)?;
        
        // Gated residual
        let x = x.add(&(gate1.unsqueeze(1)?.broadcast_mul(&attn_out)?))?;
        
        // MLP with modulation
        let mod_params = self.modulation2.forward(vec)?;
        let (shift2, scale2, gate2) = match mod_params {
            ModulationParams::Single { shift, scale, gate } => (shift, scale, gate),
            _ => panic!("Expected Single modulation params"),
        };
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