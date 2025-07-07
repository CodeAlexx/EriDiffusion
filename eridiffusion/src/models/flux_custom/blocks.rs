//! Flux transformer blocks with LoRA support

use candle_core::{Device, DType, Module, Result, Tensor, D, IndexOp, ModuleT};
use candle_nn::{linear, Linear, VarBuilder, Activation};
use candle_core::Var;
use std::collections::HashMap;
use crate::models::flux_custom::lora::{LoRAConfig, LoRAModule, LinearWithLoRA};
use crate::models::flux_lora::modulation::{Modulation1, Modulation2, ModulationOut};

/// Flux-style attention block with LoRA support
pub struct FluxAttentionWithLoRA {
    qkv: LinearWithLoRA,  // Combined QKV projection
    proj: LinearWithLoRA,  // Output projection (replaces to_out)
    heads: usize,
    dim_head: usize,
    dropout: Option<candle_nn::Dropout>,
}

impl FluxAttentionWithLoRA {
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        dropout: Option<f32>,
        name_prefix: String,
        vb: VarBuilder,
    ) -> Result<Self> {
        let dim_head = hidden_size / num_heads;
        
        // QKV dimension is 3 * hidden_size for combined projection
        let qkv_dim = 3 * hidden_size;
        
        Ok(Self {
            qkv: LinearWithLoRA::new(
                hidden_size,
                qkv_dim,
                format!("{}.qkv", name_prefix),
                vb.pp("qkv"),
            )?,
            proj: LinearWithLoRA::new(
                hidden_size,
                hidden_size,
                format!("{}.proj", name_prefix),
                vb.pp("proj"),
            )?,
            heads: num_heads,
            dim_head,
            dropout: dropout.map(candle_nn::Dropout::new),
        })
    }
    
    pub fn add_lora(&mut self, config: &LoRAConfig, device: &Device, dtype: DType) -> Result<()> {
        self.qkv.add_lora(config.rank, config.alpha, config.init_scale, device, dtype)?;
        self.proj.add_lora(config.rank, config.alpha, config.init_scale, device, dtype)?;
        Ok(())
    }
    
    pub fn forward(&self, x: &Tensor, context: Option<&Tensor>) -> Result<Tensor> {
        let (b, seq_len, _) = x.dims3()?;
        let input = context.unwrap_or(x);
        
        // Apply combined QKV projection
        let qkv = self.qkv.forward(input)?;
        
        // Split into Q, K, V
        let qkv = qkv.reshape((b, seq_len, 3, self.heads, self.dim_head))?;
        let q = qkv.i((.., .., 0, .., ..))?.transpose(1, 2)?.contiguous()?; // [b, heads, seq_len, dim_head]
        let k = qkv.i((.., .., 1, .., ..))?.transpose(1, 2)?.contiguous()?;
        let v = qkv.i((.., .., 2, .., ..))?.transpose(1, 2)?.contiguous()?;
        
        // Compute attention scores
        let scale = (self.dim_head as f64).sqrt();
        let scores = q.matmul(&k.transpose(2, 3)?)?;
        let scores = (scores * (1.0 / scale))?;
        
        // Apply softmax
        let attn = candle_nn::ops::softmax_last_dim(&scores)?;
        
        // Apply dropout if enabled
        let attn = if let Some(dropout) = &self.dropout {
            dropout.forward_t(&attn, true)?
        } else {
            attn
        };
        
        // Apply attention to values
        let out = attn.matmul(&v)?;
        
        // Reshape back
        let out = out.transpose(1, 2)?
            .contiguous()?
            .reshape((b, seq_len, self.heads * self.dim_head))?;
        
        // Output projection
        self.proj.forward(&out)
    }
    
    pub fn get_trainable_params(&self) -> Vec<&Var> {
        let mut params = Vec::new();
        params.extend(self.qkv.get_trainable_params());
        params.extend(self.proj.get_trainable_params());
        params
    }
    
    pub fn save_weights(&self, tensors: &mut HashMap<String, Tensor>) -> Result<()> {
        self.qkv.save_weights(tensors)?;
        self.proj.save_weights(tensors)?;
        Ok(())
    }
    
    pub fn load_weights(&mut self, tensors: &HashMap<String, Tensor>) -> Result<()> {
        self.qkv.load_weights(tensors)?;
        self.proj.load_weights(tensors)?;
        Ok(())
    }
}

/// MLP block with LoRA support
pub struct MLPWithLoRA {
    fc1: LinearWithLoRA,
    fc2: LinearWithLoRA,
    activation: Activation,
    dropout: Option<candle_nn::Dropout>,
}

impl MLPWithLoRA {
    pub fn new(
        hidden_size: usize,
        mlp_ratio: f32,
        activation: Activation,
        dropout: Option<f32>,
        name_prefix: String,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mlp_hidden = (hidden_size as f32 * mlp_ratio) as usize;
        
        // Determine naming based on block type
        let is_single_block = name_prefix.contains("single_blocks");
        let (fc1_name, fc2_name) = if is_single_block {
            ("linear1", "linear2")  // Single blocks use linear1/linear2
        } else {
            ("0", "2")  // Double blocks use numbered layers 0 and 2
        };
        
        Ok(Self {
            fc1: LinearWithLoRA::new(
                hidden_size,
                mlp_hidden,
                format!("{}.{}", name_prefix, fc1_name),
                vb.pp(fc1_name),
            )?,
            fc2: LinearWithLoRA::new(
                mlp_hidden,
                hidden_size,
                format!("{}.{}", name_prefix, fc2_name),
                vb.pp(fc2_name),
            )?,
            activation,
            dropout: dropout.map(candle_nn::Dropout::new),
        })
    }
    
    pub fn add_lora(&mut self, config: &LoRAConfig, device: &Device, dtype: DType) -> Result<()> {
        self.fc1.add_lora(config.rank, config.alpha, config.init_scale, device, dtype)?;
        self.fc2.add_lora(config.rank, config.alpha, config.init_scale, device, dtype)?;
        Ok(())
    }
    
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.fc1.forward(x)?;
        let h = self.activation.forward(&h)?;
        
        let h = if let Some(dropout) = &self.dropout {
            dropout.forward_t(&h, true)?
        } else {
            h
        };
        
        self.fc2.forward(&h)
    }
    
    pub fn get_trainable_params(&self) -> Vec<&Var> {
        let mut params = Vec::new();
        params.extend(self.fc1.get_trainable_params());
        params.extend(self.fc2.get_trainable_params());
        params
    }
    
    pub fn save_weights(&self, tensors: &mut HashMap<String, Tensor>) -> Result<()> {
        self.fc1.save_weights(tensors)?;
        self.fc2.save_weights(tensors)?;
        Ok(())
    }
    
    pub fn load_weights(&mut self, tensors: &HashMap<String, Tensor>) -> Result<()> {
        self.fc1.load_weights(tensors)?;
        self.fc2.load_weights(tensors)?;
        Ok(())
    }
}

/// Double transformer block (processes both image and text)
pub struct FluxDoubleBlockWithLoRA {
    img_mod: Modulation2,
    txt_mod: Modulation2,
    img_attn: FluxAttentionWithLoRA,
    txt_attn: FluxAttentionWithLoRA,
    img_mlp: MLPWithLoRA,
    txt_mlp: MLPWithLoRA,
    img_norm1: candle_nn::LayerNorm,
    img_norm2: candle_nn::LayerNorm,
    txt_norm1: candle_nn::LayerNorm,
    txt_norm2: candle_nn::LayerNorm,
    block_idx: usize,
}

impl FluxDoubleBlockWithLoRA {
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        mlp_ratio: f32,
        dropout: Option<f32>,
        block_idx: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let name_prefix = format!("double_blocks.{}", block_idx);
        
        Ok(Self {
            img_mod: Modulation2::new(hidden_size, vb.pp("img_mod"))?,
            txt_mod: Modulation2::new(hidden_size, vb.pp("txt_mod"))?,
            img_attn: FluxAttentionWithLoRA::new(
                hidden_size,
                num_heads,
                dropout,
                format!("{}.img_attn", name_prefix),
                vb.pp("img_attn"),
            )?,
            txt_attn: FluxAttentionWithLoRA::new(
                hidden_size,
                num_heads,
                dropout,
                format!("{}.txt_attn", name_prefix),
                vb.pp("txt_attn"),
            )?,
            img_mlp: MLPWithLoRA::new(
                hidden_size,
                mlp_ratio,
                Activation::Gelu,
                dropout,
                format!("{}.img_mlp", name_prefix),
                vb.pp("img_mlp"),
            )?,
            txt_mlp: MLPWithLoRA::new(
                hidden_size,
                mlp_ratio,
                Activation::Gelu,
                dropout,
                format!("{}.txt_mlp", name_prefix),
                vb.pp("txt_mlp"),
            )?,
            img_norm1: candle_nn::layer_norm(hidden_size, 1e-6, vb.pp("img_norm1"))?,
            img_norm2: candle_nn::layer_norm(hidden_size, 1e-6, vb.pp("img_norm2"))?,
            txt_norm1: candle_nn::layer_norm(hidden_size, 1e-6, vb.pp("txt_norm1"))?,
            txt_norm2: candle_nn::layer_norm(hidden_size, 1e-6, vb.pp("txt_norm2"))?,
            block_idx,
        })
    }
    
    pub fn add_lora(&mut self, config: &LoRAConfig, device: &Device, dtype: DType) -> Result<()> {
        self.img_attn.add_lora(config, device, dtype)?;
        self.txt_attn.add_lora(config, device, dtype)?;
        self.img_mlp.add_lora(config, device, dtype)?;
        self.txt_mlp.add_lora(config, device, dtype)?;
        Ok(())
    }
    
    pub fn forward(&self, img: &Tensor, txt: &Tensor, vec: &Tensor) -> Result<(Tensor, Tensor)> {
        // Get modulation parameters
        let (img_mod1, img_mod2) = self.img_mod.forward(vec)?;
        let (txt_mod1, txt_mod2) = self.txt_mod.forward(vec)?;
        
        // Image stream
        let img_modulated = img_mod1.scale_shift(&self.img_norm1.forward(img)?)?;
        let img_attn_out = self.img_attn.forward(&img_modulated, None)?;
        let img = img.add(&img_mod1.gate(&img_attn_out)?)?;
        
        let img_modulated = img_mod2.scale_shift(&self.img_norm2.forward(&img)?)?;
        let img_mlp_out = self.img_mlp.forward(&img_modulated)?;
        let img = img.add(&img_mod2.gate(&img_mlp_out)?)?;
        
        // Text stream
        let txt_modulated = txt_mod1.scale_shift(&self.txt_norm1.forward(txt)?)?;
        let txt_attn_out = self.txt_attn.forward(&txt_modulated, None)?;
        let txt = txt.add(&txt_mod1.gate(&txt_attn_out)?)?;
        
        let txt_modulated = txt_mod2.scale_shift(&self.txt_norm2.forward(&txt)?)?;
        let txt_mlp_out = self.txt_mlp.forward(&txt_modulated)?;
        let txt = txt.add(&txt_mod2.gate(&txt_mlp_out)?)?;
        
        Ok((img, txt))
    }
    
    pub fn get_trainable_params(&self) -> Vec<&Var> {
        let mut params = Vec::new();
        params.extend(self.img_attn.get_trainable_params());
        params.extend(self.txt_attn.get_trainable_params());
        params.extend(self.img_mlp.get_trainable_params());
        params.extend(self.txt_mlp.get_trainable_params());
        params
    }
    
    pub fn save_weights(&self, tensors: &mut HashMap<String, Tensor>) -> Result<()> {
        self.img_attn.save_weights(tensors)?;
        self.txt_attn.save_weights(tensors)?;
        self.img_mlp.save_weights(tensors)?;
        self.txt_mlp.save_weights(tensors)?;
        Ok(())
    }
    
    pub fn load_weights(&mut self, tensors: &HashMap<String, Tensor>) -> Result<()> {
        self.img_attn.load_weights(tensors)?;
        self.txt_attn.load_weights(tensors)?;
        self.img_mlp.load_weights(tensors)?;
        self.txt_mlp.load_weights(tensors)?;
        Ok(())
    }
}

/// Single transformer block
pub struct FluxSingleBlockWithLoRA {
    modulation: Modulation1,
    attn: FluxAttentionWithLoRA,
    mlp: MLPWithLoRA,
    norm1: candle_nn::LayerNorm,
    norm2: candle_nn::LayerNorm,
    block_idx: usize,
}

impl FluxSingleBlockWithLoRA {
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        mlp_ratio: f32,
        dropout: Option<f32>,
        block_idx: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let name_prefix = format!("single_blocks.{}", block_idx);
        
        Ok(Self {
            modulation: Modulation1::new(hidden_size, vb.pp("modulation"))?,
            attn: FluxAttentionWithLoRA::new(
                hidden_size,
                num_heads,
                dropout,
                format!("{}.attn", name_prefix),
                vb.pp("attn"),
            )?,
            mlp: MLPWithLoRA::new(
                hidden_size,
                mlp_ratio,
                Activation::Gelu,
                dropout,
                format!("{}.mlp", name_prefix),
                vb.pp("mlp"),
            )?,
            norm1: candle_nn::layer_norm(hidden_size, 1e-6, vb.pp("norm1"))?,
            norm2: candle_nn::layer_norm(hidden_size, 1e-6, vb.pp("norm2"))?,
            block_idx,
        })
    }
    
    pub fn add_lora(&mut self, config: &LoRAConfig, device: &Device, dtype: DType) -> Result<()> {
        self.attn.add_lora(config, device, dtype)?;
        self.mlp.add_lora(config, device, dtype)?;
        Ok(())
    }
    
    pub fn forward(&self, x: &Tensor, vec: &Tensor) -> Result<Tensor> {
        // Get modulation parameters
        let mod_params = self.modulation.forward(vec)?;
        
        // First norm + attention
        let x_modulated = mod_params.scale_shift(&self.norm1.forward(x)?)?;
        let attn_out = self.attn.forward(&x_modulated, None)?;
        let x = x.add(&mod_params.gate(&attn_out)?)?;
        
        // Second norm + MLP
        let mod_params = self.modulation.forward(vec)?;
        let x_modulated = mod_params.scale_shift(&self.norm2.forward(&x)?)?;
        let mlp_out = self.mlp.forward(&x_modulated)?;
        x.add(&mod_params.gate(&mlp_out)?)
    }
    
    pub fn get_trainable_params(&self) -> Vec<&Var> {
        let mut params = Vec::new();
        params.extend(self.attn.get_trainable_params());
        params.extend(self.mlp.get_trainable_params());
        params
    }
    
    pub fn save_weights(&self, tensors: &mut HashMap<String, Tensor>) -> Result<()> {
        self.attn.save_weights(tensors)?;
        self.mlp.save_weights(tensors)?;
        Ok(())
    }
    
    pub fn load_weights(&mut self, tensors: &HashMap<String, Tensor>) -> Result<()> {
        self.attn.load_weights(tensors)?;
        self.mlp.load_weights(tensors)?;
        Ok(())
    }
}