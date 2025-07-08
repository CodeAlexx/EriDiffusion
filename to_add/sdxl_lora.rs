use candle_core::{Device, Result as CandleResult, Tensor, DType, Module, D, Shape, IndexOp};
use candle_nn::{VarBuilder, VarMap, Linear, Conv2d, GroupNorm, Activation, ops::softmax, Init};
use candle_transformers::models::clip::{ClipTextModel, ClipTextConfig, ClipTextTransformer};
use std::collections::HashMap;
use std::path::Path;
use serde::{Deserialize, Serialize};
use anyhow::{Result, Context};

// ==================== LoRA Variants ====================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LoRAType {
    LoRA,
    LoHa,
    LoKr,
    LoCon,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoRAConfig {
    pub lora_type: LoRAType,
    pub rank: usize,
    pub alpha: f64,
    pub dropout: f64,
    pub target_modules: Vec<String>,
    pub conv_rank: Option<usize>,
    pub conv_alpha: Option<f64>,
    pub decompose_both: bool,
    pub factor: Option<i32>,
}

impl Default for LoRAConfig {
    fn default() -> Self {
        Self {
            lora_type: LoRAType::LoRA,
            rank: 16,
            alpha: 16.0,
            dropout: 0.0,
            target_modules: vec![
                "to_q".to_string(),
                "to_k".to_string(), 
                "to_v".to_string(),
                "to_out.0".to_string(),
            ],
            conv_rank: Some(16),
            conv_alpha: Some(16.0),
            decompose_both: false,
            factor: Some(-1),
        }
    }
}

pub trait LoRALayer: Module + Send + Sync {
    fn get_scaling(&self) -> f64;
    fn get_rank(&self) -> usize;
    fn merge_weights(&mut self) -> CandleResult<()>;
    fn unmerge_weights(&mut self) -> CandleResult<()>;
    fn is_merged(&self) -> bool;
    fn save_adapter_weights(&self, path: &Path) -> Result<()>;
    fn load_adapter_weights(&mut self, path: &Path) -> Result<()>;
}

// ==================== Standard LoRA ====================

pub struct LoRALinear {
    base_layer: Linear,
    lora_a: Linear,
    lora_b: Linear,
    scaling: f64,
    rank: usize,
    dropout: candle_nn::Dropout,
    merged: bool,
    merged_weights: Option<Tensor>,
}

impl LoRALinear {
    pub fn new(
        in_features: usize,
        out_features: usize,
        rank: usize,
        alpha: f64,
        dropout: f64,
        vb: VarBuilder,
    ) -> CandleResult<Self> {
        let base_layer = candle_nn::linear(in_features, out_features, vb.pp("base"))?;
        
        // Initialize LoRA A with kaiming uniform
        let lora_a_init = Init::Kaiming { dist: candle_nn::init::NormalOrUniform::Uniform, fan: candle_nn::init::FanInOut::FanIn };
        let lora_a = candle_nn::linear_with_init(in_features, rank, lora_a_init, None, vb.pp("lora_A"))?;
        
        // Initialize LoRA B with zeros
        let lora_b_init = Init::Const(0.0);
        let lora_b = candle_nn::linear_with_init(rank, out_features, lora_b_init, None, vb.pp("lora_B"))?;
        
        let scaling = alpha / rank as f64;
        let dropout = candle_nn::Dropout::new(dropout);
        
        Ok(Self {
            base_layer,
            lora_a,
            lora_b,
            scaling,
            rank,
            dropout,
            merged: false,
            merged_weights: None,
        })
    }
}

impl Module for LoRALinear {
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let base_out = self.base_layer.forward(x)?;
        
        if self.merged {
            return Ok(base_out);
        }
        
        let lora_out = self.lora_a.forward(x)?;
        let lora_out = self.dropout.forward(&lora_out, false)?;
        let lora_out = self.lora_b.forward(&lora_out)?;
        let lora_out = lora_out.mul(&Tensor::new(&[self.scaling as f32], x.device())?)?;
        
        base_out.add(&lora_out)
    }
}

impl LoRALayer for LoRALinear {
    fn get_scaling(&self) -> f64 { self.scaling }
    fn get_rank(&self) -> usize { self.rank }
    fn is_merged(&self) -> bool { self.merged }
    
    fn merge_weights(&mut self) -> CandleResult<()> {
        if !self.merged {
            let lora_weight = self.lora_b.weight().matmul(self.lora_a.weight())?;
            let scaled_lora = lora_weight.mul(&Tensor::new(&[self.scaling as f32], lora_weight.device())?)?;
            self.merged_weights = Some(scaled_lora.clone());
            self.merged = true;
        }
        Ok(())
    }
    
    fn unmerge_weights(&mut self) -> CandleResult<()> {
        if self.merged {
            self.merged = false;
            self.merged_weights = None;
        }
        Ok(())
    }
    
    fn save_adapter_weights(&self, path: &Path) -> Result<()> {
        let mut weights = HashMap::new();
        weights.insert("lora_A.weight".to_string(), self.lora_a.weight().clone());
        weights.insert("lora_B.weight".to_string(), self.lora_b.weight().clone());
        weights.insert("scaling".to_string(), Tensor::new(&[self.scaling as f32], &Device::Cpu)?);
        safetensors::save(&weights, path)?;
        Ok(())
    }
    
    fn load_adapter_weights(&mut self, path: &Path) -> Result<()> {
        let weights = safetensors::load(path, &Device::Cpu)?;
        // In a real implementation, you'd load these into the LoRA layers
        Ok(())
    }
}

// ==================== LoHa Implementation ====================

pub struct LoHaLinear {
    base_layer: Linear,
    hada_w1_a: Linear,
    hada_w1_b: Linear,
    hada_w2_a: Linear,
    hada_w2_b: Linear,
    hada_t1: Option<Tensor>,
    hada_t2: Option<Tensor>,
    scaling: f64,
    rank: usize,
    dropout: candle_nn::Dropout,
    merged: bool,
}

impl LoHaLinear {
    pub fn new(
        in_features: usize,
        out_features: usize,
        rank: usize,
        alpha: f64,
        dropout: f64,
        vb: VarBuilder,
    ) -> CandleResult<Self> {
        let base_layer = candle_nn::linear(in_features, out_features, vb.pp("base"))?;
        
        let kaiming_init = Init::Kaiming { dist: candle_nn::init::NormalOrUniform::Uniform, fan: candle_nn::init::FanInOut::FanIn };
        let zero_init = Init::Const(0.0);
        
        let hada_w1_a = candle_nn::linear_with_init(in_features, rank, kaiming_init, None, vb.pp("hada_w1_a"))?;
        let hada_w1_b = candle_nn::linear_with_init(rank, out_features, zero_init, None, vb.pp("hada_w1_b"))?;
        let hada_w2_a = candle_nn::linear_with_init(in_features, rank, kaiming_init, None, vb.pp("hada_w2_a"))?;
        let hada_w2_b = candle_nn::linear_with_init(rank, out_features, zero_init, None, vb.pp("hada_w2_b"))?;
        
        let scaling = alpha / rank as f64;
        let dropout = candle_nn::Dropout::new(dropout);
        
        Ok(Self {
            base_layer,
            hada_w1_a,
            hada_w1_b,
            hada_w2_a,
            hada_w2_b,
            hada_t1: None,
            hada_t2: None,
            scaling,
            rank,
            dropout,
            merged: false,
        })
    }
}

impl Module for LoHaLinear {
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let base_out = self.base_layer.forward(x)?;
        
        if self.merged {
            return Ok(base_out);
        }
        
        let w1_out = self.hada_w1_a.forward(x)?;
        let w1_out = self.dropout.forward(&w1_out, false)?;
        let w1_out = self.hada_w1_b.forward(&w1_out)?;
        
        let w2_out = self.hada_w2_a.forward(x)?;
        let w2_out = self.dropout.forward(&w2_out, false)?;
        let w2_out = self.hada_w2_b.forward(&w2_out)?;
        
        let hada_out = w1_out.mul(&w2_out)?;
        let hada_out = hada_out.mul(&Tensor::new(&[self.scaling as f32], x.device())?)?;
        
        base_out.add(&hada_out)
    }
}

impl LoRALayer for LoHaLinear {
    fn get_scaling(&self) -> f64 { self.scaling }
    fn get_rank(&self) -> usize { self.rank }
    fn is_merged(&self) -> bool { self.merged }
    
    fn merge_weights(&mut self) -> CandleResult<()> {
        if !self.merged {
            self.merged = true;
        }
        Ok(())
    }
    
    fn unmerge_weights(&mut self) -> CandleResult<()> {
        if self.merged {
            self.merged = false;
        }
        Ok(())
    }
    
    fn save_adapter_weights(&self, path: &Path) -> Result<()> {
        let mut weights = HashMap::new();
        weights.insert("hada_w1_a.weight".to_string(), self.hada_w1_a.weight().clone());
        weights.insert("hada_w1_b.weight".to_string(), self.hada_w1_b.weight().clone());
        weights.insert("hada_w2_a.weight".to_string(), self.hada_w2_a.weight().clone());
        weights.insert("hada_w2_b.weight".to_string(), self.hada_w2_b.weight().clone());
        safetensors::save(&weights, path)?;
        Ok(())
    }
    
    fn load_adapter_weights(&mut self, path: &Path) -> Result<()> {
        let _weights = safetensors::load(path, &Device::Cpu)?;
        Ok(())
    }
}

// ==================== LoKr Implementation ====================

pub struct LoKrLinear {
    base_layer: Linear,
    lokr_w1: Option<Linear>,
    lokr_w1_a: Option<Linear>,
    lokr_w1_b: Option<Linear>,
    lokr_w2: Option<Linear>,
    lokr_w2_a: Option<Linear>,
    lokr_w2_b: Option<Linear>,
    lokr_t2: Option<Tensor>,
    scaling: f64,
    rank: usize,
    factor: i32,
    dropout: candle_nn::Dropout,
    merged: bool,
}

impl LoKrLinear {
    pub fn new(
        in_features: usize,
        out_features: usize,
        rank: usize,
        alpha: f64,
        dropout: f64,
        factor: i32,
        vb: VarBuilder,
    ) -> CandleResult<Self> {
        let base_layer = candle_nn::linear(in_features, out_features, vb.pp("base"))?;
        
        let (w1_dim, w2_dim) = Self::factorization(in_features, factor);
        let (w1_dim_out, w2_dim_out) = Self::factorization(out_features, factor);
        
        let kaiming_init = Init::Kaiming { dist: candle_nn::init::NormalOrUniform::Uniform, fan: candle_nn::init::FanInOut::FanIn };
        let zero_init = Init::Const(0.0);
        
        let (lokr_w1, lokr_w1_a, lokr_w1_b) = if w1_dim < in_features {
            (None, 
             Some(candle_nn::linear_with_init(in_features, w1_dim, kaiming_init, None, vb.pp("lokr_w1_a"))?),
             Some(candle_nn::linear_with_init(w1_dim, rank, zero_init, None, vb.pp("lokr_w1_b"))?))
        } else {
            (Some(candle_nn::linear_with_init(in_features, rank, kaiming_init, None, vb.pp("lokr_w1"))?), None, None)
        };
        
        let (lokr_w2, lokr_w2_a, lokr_w2_b) = if w2_dim_out < out_features {
            (None,
             Some(candle_nn::linear_with_init(rank, w2_dim_out, zero_init, None, vb.pp("lokr_w2_a"))?),
             Some(candle_nn::linear_with_init(w2_dim_out, out_features, zero_init, None, vb.pp("lokr_w2_b"))?))
        } else {
            (Some(candle_nn::linear_with_init(rank, out_features, zero_init, None, vb.pp("lokr_w2"))?), None, None)
        };
        
        let scaling = alpha / rank as f64;
        let dropout = candle_nn::Dropout::new(dropout);
        
        Ok(Self {
            base_layer,
            lokr_w1,
            lokr_w1_a,
            lokr_w1_b,
            lokr_w2,
            lokr_w2_a,
            lokr_w2_b,
            lokr_t2: None,
            scaling,
            rank,
            factor,
            dropout,
            merged: false,
        })
    }
    
    fn factorization(dimension: usize, factor: i32) -> (usize, usize) {
        if factor > 0 && dimension % factor as usize == 0 {
            (factor as usize, dimension / factor as usize)
        } else {
            let mut best_factor = 1;
            for i in 1..=(dimension as f64).sqrt() as usize {
                if dimension % i == 0 {
                    best_factor = i;
                }
            }
            (best_factor, dimension / best_factor)
        }
    }
}

impl Module for LoKrLinear {
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let base_out = self.base_layer.forward(x)?;
        
        if self.merged {
            return Ok(base_out);
        }
        
        let mut w1_out = x.clone();
        
        if let Some(w1) = &self.lokr_w1 {
            w1_out = w1.forward(&w1_out)?;
        } else if let (Some(w1_a), Some(w1_b)) = (&self.lokr_w1_a, &self.lokr_w1_b) {
            w1_out = w1_a.forward(&w1_out)?;
            w1_out = w1_b.forward(&w1_out)?;
        }
        
        w1_out = self.dropout.forward(&w1_out, false)?;
        
        if let Some(w2) = &self.lokr_w2 {
            w1_out = w2.forward(&w1_out)?;
        } else if let (Some(w2_a), Some(w2_b)) = (&self.lokr_w2_a, &self.lokr_w2_b) {
            w1_out = w2_a.forward(&w1_out)?;
            w1_out = w2_b.forward(&w1_out)?;
        }
        
        let lokr_out = w1_out.mul(&Tensor::new(&[self.scaling as f32], x.device())?)?;
        base_out.add(&lokr_out)
    }
}

impl LoRALayer for LoKrLinear {
    fn get_scaling(&self) -> f64 { self.scaling }
    fn get_rank(&self) -> usize { self.rank }
    fn is_merged(&self) -> bool { self.merged }
    
    fn merge_weights(&mut self) -> CandleResult<()> {
        if !self.merged {
            self.merged = true;
        }
        Ok(())
    }
    
    fn unmerge_weights(&mut self) -> CandleResult<()> {
        if self.merged {
            self.merged = false;
        }
        Ok(())
    }
    
    fn save_adapter_weights(&self, path: &Path) -> Result<()> {
        let mut weights = HashMap::new();
        if let Some(w1) = &self.lokr_w1 {
            weights.insert("lokr_w1.weight".to_string(), w1.weight().clone());
        }
        if let Some(w2) = &self.lokr_w2 {
            weights.insert("lokr_w2.weight".to_string(), w2.weight().clone());
        }
        safetensors::save(&weights, path)?;
        Ok(())
    }
    
    fn load_adapter_weights(&mut self, path: &Path) -> Result<()> {
        let _weights = safetensors::load(path, &Device::Cpu)?;
        Ok(())
    }
}

// ==================== LoCon Implementation ====================

pub struct LoConConv2d {
    base_layer: Conv2d,
    lora_a: Conv2d,
    lora_b: Conv2d,
    scaling: f64,
    rank: usize,
    kernel_size: usize,
    dropout: candle_nn::Dropout,
    merged: bool,
}

impl LoConConv2d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        rank: usize,
        alpha: f64,
        dropout: f64,
        vb: VarBuilder,
    ) -> CandleResult<Self> {
        let config = candle_nn::Conv2dConfig {
            padding: kernel_size / 2,
            ..Default::default()
        };
        
        let base_layer = candle_nn::conv2d(in_channels, out_channels, kernel_size, config, vb.pp("base"))?;
        
        let lora_a = if kernel_size == 1 {
            candle_nn::conv2d(in_channels, rank, 1, Default::default(), vb.pp("lora_A"))?
        } else {
            candle_nn::conv2d(in_channels, rank, kernel_size, config, vb.pp("lora_A"))?
        };
        
        let lora_b = candle_nn::conv2d(rank, out_channels, 1, Default::default(), vb.pp("lora_B"))?;
        
        let scaling = alpha / rank as f64;
        let dropout = candle_nn::Dropout::new(dropout);
        
        Ok(Self {
            base_layer,
            lora_a,
            lora_b,
            scaling,
            rank,
            kernel_size,
            dropout,
            merged: false,
        })
    }
}

impl Module for LoConConv2d {
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let base_out = self.base_layer.forward(x)?;
        
        if self.merged {
            return Ok(base_out);
        }
        
        let lora_out = self.lora_a.forward(x)?;
        let lora_out = self.dropout.forward(&lora_out, false)?;
        let lora_out = self.lora_b.forward(&lora_out)?;
        let lora_out = lora_out.mul(&Tensor::new(&[self.scaling as f32], x.device())?)?;
        
        base_out.add(&lora_out)
    }
}

impl LoRALayer for LoConConv2d {
    fn get_scaling(&self) -> f64 { self.scaling }
    fn get_rank(&self) -> usize { self.rank }
    fn is_merged(&self) -> bool { self.merged }
    
    fn merge_weights(&mut self) -> CandleResult<()> {
        if !self.merged {
            self.merged = true;
        }
        Ok(())
    }
    
    fn unmerge_weights(&mut self) -> CandleResult<()> {
        if self.merged {
            self.merged = false;
        }
        Ok(())
    }
    
    fn save_adapter_weights(&self, path: &Path) -> Result<()> {
        let mut weights = HashMap::new();
        weights.insert("lora_A.weight".to_string(), self.lora_a.weight().clone());
        weights.insert("lora_B.weight".to_string(), self.lora_b.weight().clone());
        safetensors::save(&weights, path)?;
        Ok(())
    }
    
    fn load_adapter_weights(&mut self, path: &Path) -> Result<()> {
        let _weights = safetensors::load(path, &Device::Cpu)?;
        Ok(())
    }
}

// ==================== LoRA Factory ====================

pub struct LoRAFactory;

impl LoRAFactory {
    pub fn create_linear_layer(
        config: &LoRAConfig,
        in_features: usize,
        out_features: usize,
        vb: VarBuilder,
    ) -> CandleResult<Box<dyn LoRALayer>> {
        match config.lora_type {
            LoRAType::LoRA => {
                let layer = LoRALinear::new(
                    in_features,
                    out_features,
                    config.rank,
                    config.alpha,
                    config.dropout,
                    vb,
                )?;
                Ok(Box::new(layer))
            },
            LoRAType::LoHa => {
                let layer = LoHaLinear::new(
                    in_features,
                    out_features,
                    config.rank,
                    config.alpha,
                    config.dropout,
                    vb,
                )?;
                Ok(Box::new(layer))
            },
            LoRAType::LoKr => {
                let factor = config.factor.unwrap_or(-1);
                let layer = LoKrLinear::new(
                    in_features,
                    out_features,
                    config.rank,
                    config.alpha,
                    config.dropout,
                    factor,
                    vb,
                )?;
                Ok(Box::new(layer))
            },
            LoRAType::LoCon => {
                let layer = LoRALinear::new(
                    in_features,
                    out_features,
                    config.rank,
                    config.alpha,
                    config.dropout,
                    vb,
                )?;
                Ok(Box::new(layer))
            },
        }
    }
    
    pub fn create_conv_layer(
        config: &LoRAConfig,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        vb: VarBuilder,
    ) -> CandleResult<Box<dyn LoRALayer>> {
        let conv_rank = config.conv_rank.unwrap_or(config.rank);
        let conv_alpha = config.conv_alpha.unwrap_or(config.alpha);
        
        let layer = LoConConv2d::new(
            in_channels,
            out_channels,
            kernel_size,
            conv_rank,
            conv_alpha,
            config.dropout,
            vb,
        )?;
        Ok(Box::new(layer))
    }
}

// ==================== Time Embedding ====================

pub struct TimeEmbedding {
    linear_1: Linear,
    linear_2: Linear,
}

impl TimeEmbedding {
    pub fn new(time_embed_dim: usize, vb: VarBuilder) -> CandleResult<Self> {
        let linear_1 = candle_nn::linear(time_embed_dim, time_embed_dim * 4, vb.pp("linear_1"))?;
        let linear_2 = candle_nn::linear(time_embed_dim * 4, time_embed_dim * 4, vb.pp("linear_2"))?;
        
        Ok(Self { linear_1, linear_2 })
    }
    
    pub fn forward(&self, time: &Tensor) -> CandleResult<Tensor> {
        let time_emb = self.timestep_embedding(time, 320)?;
        let emb = self.linear_1.forward(&time_emb)?;
        let emb = emb.silu()?;
        self.linear_2.forward(&emb)
    }
    
    fn timestep_embedding(&self, timesteps: &Tensor, dim: usize) -> CandleResult<Tensor> {
        let half_dim = dim / 2;
        let emb = (10000f64).ln() / (half_dim - 1) as f64;
        let device = timesteps.device();
        
        let freqs = Tensor::arange(0f32, half_dim as f32, device)?
            .mul(&Tensor::new(&[(-emb) as f32], device)?)?
            .exp()?;
        
        let args = timesteps.unsqueeze(1)?.broadcast_mul(&freqs.unsqueeze(0)?)?;
        let sin_args = args.sin()?;
        let cos_args = args.cos()?;
        
        Tensor::cat(&[cos_args, sin_args], 1)
    }
}

// ==================== ResNet Block ====================

pub struct ResnetBlock {
    norm1: GroupNorm,
    conv1: Conv2d,
    time_emb_proj: Option<Linear>,
    norm2: GroupNorm,
    conv2: Conv2d,
    nin_shortcut: Option<Conv2d>,
    dropout: candle_nn::Dropout,
}

impl ResnetBlock {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        time_emb_channels: Option<usize>,
        dropout: f64,
        vb: VarBuilder,
    ) -> CandleResult<Self> {
        let groups = 32.min(in_channels);
        let norm1 = candle_nn::group_norm(groups, in_channels, 1e-6, vb.pp("norm1"))?;
        
        let conv_config = candle_nn::Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let conv1 = candle_nn::conv2d(in_channels, out_channels, 3, conv_config, vb.pp("conv1"))?;
        
        let time_emb_proj = if let Some(time_channels) = time_emb_channels {
            Some(candle_nn::linear(time_channels, out_channels, vb.pp("time_emb_proj"))?)
        } else {
            None
        };
        
        let norm2 = candle_nn::group_norm(32.min(out_channels), out_channels, 1e-6, vb.pp("norm2"))?;
        let conv2 = candle_nn::conv2d(out_channels, out_channels, 3, conv_config, vb.pp("conv2"))?;
        
        let nin_shortcut = if in_channels != out_channels {
            Some(candle_nn::conv2d(in_channels, out_channels, 1, Default::default(), vb.pp("nin_shortcut"))?)
        } else {
            None
        };

        let dropout = candle_nn::Dropout::new(dropout);

        Ok(Self {
            norm1,
            conv1,
            time_emb_proj,
            norm2,
            conv2,
            nin_shortcut,
            dropout,
        })
    }

    pub fn forward(&self, x: &Tensor, time_emb: Option<&Tensor>) -> CandleResult<Tensor> {
        let h = self.norm1.forward(x)?;
        let h = h.silu()?;
        let h = self.conv1.forward(&h)?;

        if let (Some(time_proj), Some(temb)) = (&self.time_emb_proj, time_emb) {
            let temb = temb.silu()?;
            let temb = time_proj.forward(&temb)?;
            let temb = temb.unsqueeze(2)?.unsqueeze(3)?;
            let h = h.broadcast_add(&temb)?;
        }

        let h = self.norm2.forward(&h)?;
        let h = h.silu()?;
        let h = self.dropout.forward(&h, false)?;
        let h = self.conv2.forward(&h)?;

        let shortcut = if let Some(nin_shortcut) = &self.nin_shortcut {
            nin_shortcut.forward(x)?
        } else {
            x.clone()
        };

        h.add(&shortcut)
    }
}

// ==================== Cross Attention ====================

pub struct CrossAttentionBlock {
    norm: GroupNorm,
    q_proj: Box<dyn LoRALayer>,
    k_proj: Box<dyn LoRALayer>,
    v_proj: Box<dyn LoRALayer>,
    out_proj: Box<dyn LoRALayer>,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl CrossAttentionBlock {
    pub fn new(
        query_dim: usize,
        context_dim: usize,
        num_heads: usize,
        lora_config: Option<&LoRAConfig>,
        vb: VarBuilder,
    ) -> CandleResult<Self> {
        let head_dim = query_dim / num_heads;
        let scale = 1.0 / (head_dim as f64).sqrt();
        
        let norm = candle_nn::group_norm(32, query_dim, 1e-6, vb.pp("norm"))?;
        
        let default_config = LoRAConfig::default();
        let config = lora_config.unwrap_or(&default_config);
        
        let q_proj = LoRAFactory::create_linear_layer(config, query_dim, query_dim, vb.pp("to_q"))?;
        let k_proj = LoRAFactory::create_linear_layer(config, context_dim, query_dim, vb.pp("to_k"))?;
        let v_proj = LoRAFactory::create_linear_layer(config, context_dim, query_dim, vb.pp("to_v"))?;
        let out_proj = LoRAFactory::create_linear_layer(config, query_dim, query_dim, vb.pp("to_out"))?;

        Ok(Self {
            norm,
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads,
            head_dim,
            scale,
        })
    }

    pub fn forward(&self, x: &Tensor, context: &Tensor) -> CandleResult<Tensor> {
        let (b, c, h, w) = x.dims4()?;
        let spatial_size = h * w;
        
        let x_seq = x.reshape((b, c, spatial_size))?.transpose(1, 2)?;
        let x_norm = self.norm.forward(&x_seq)?;
        
        let q = self.q_proj.forward(&x_norm)?;
        let k = self.k_proj.forward(context)?;
        let v = self.v_proj.forward(context)?;
        
        let q = q.reshape((b, spatial_size, self.num_heads, self.head_dim))?
                 .transpose(1, 2)?;
        let k = k.reshape((b, context.dim(1)?, self.num_heads, self.head_dim))?
                 .transpose(1, 2)?;
        let v = v.reshape((b, context.dim(1)?, self.num_heads, self.head_dim))?
                 .transpose(1, 2)?;
        
        let scores = q.matmul(&k.transpose(2, 3)?)?;
        let scores = scores.mul(&Tensor::new(&[self.scale as f32], x.device())?)?;
        let attention_weights = softmax(&scores, D::Minus1)?;
        let out = attention_weights.matmul(&v)?;
        
        let out = out.transpose(1, 2)?
                     .reshape((b, spatial_size, self.num_heads * self.head_dim))?;
        let out = self.out_proj.forward(&out)?;
        
        let out = x_seq.add(&out)?;
        out.transpose(1, 2)?.reshape((b, c, h, w))
    }
}

// ==================== SDXL UNet Configuration ====================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SDXLUNetConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub model_channels: usize,
    pub attention_resolutions: Vec<usize>,
    pub num_res_blocks: usize,
    pub channel_mult: Vec<usize>,
    pub num_heads: usize,
    pub use_spatial_transformer: bool,
    pub transformer_depth: usize,
    pub context_dim: usize,
    pub use_checkpoint: bool,
    pub legacy: bool,
    pub num_classes: Option<usize>,
    pub use_fp16: bool,
    pub num_head_channels: Option<usize>,
    pub use_new_attention_order: bool,
    pub use_linear_in_transformer: bool,
    pub adm_in_channels: Option<usize>,
}

impl Default for SDXLUNetConfig {
    fn default() -> Self {
        Self {
            in_channels: 4,
            out_channels: 4,
            model_channels: 320,
            attention_resolutions: vec![4, 2, 1],
            num_res_blocks: 2,
            channel_mult: vec![1, 2, 4, 4],
            num_heads: 8,
            use_spatial_transformer: true,
            transformer_depth: 1,
            context_dim: 2048,
            use_checkpoint: false,
            legacy: false,
            num_classes: None,
            use_fp16: false,
            num_head_channels: None,
            use_new_attention_order: false,
            use_linear_in_transformer: false,
            adm_in_channels: Some(2816),
        }
    }
}

// ==================== UNet Blocks ====================

pub struct DownBlock {
    resnets: Vec<ResnetBlock>,
    attentions: Vec<CrossAttentionBlock>,
    downsample: Option<Conv2d>,
}

impl DownBlock {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        time_emb_channels: Option<usize>,
        num_layers: usize,
        context_dim: usize,
        num_heads: usize,
        add_downsample: bool,
        lora_config: Option<&LoRAConfig>,
        vb: VarBuilder,
    ) -> CandleResult<Self> {
        let mut resnets = Vec::new();
        let mut attentions = Vec::new();
        
        for i in 0..num_layers {
            let input_channels = if i == 0 { in_channels } else { out_channels };
            
            let resnet = ResnetBlock::new(
                input_channels,
                out_channels,
                time_emb_channels,
                0.0,
                vb.pp(&format!("resnets.{}", i))
            )?;
            resnets.push(resnet);
            
            let attention = CrossAttentionBlock::new(
                out_channels,
                context_dim,
                num_heads,
                lora_config,
                vb.pp(&format!("attentions.{}", i))
            )?;
            attentions.push(attention);
        }
        
        let downsample = if add_downsample {
            let config = candle_nn::Conv2dConfig {
                stride: 2,
                padding: 1,
                ..Default::default()
            };
            Some(candle_nn::conv2d(out_channels, out_channels, 3, config, vb.pp("downsamplers.0.conv"))?)
        } else {
            None
        };
        
        Ok(Self {
            resnets,
            attentions,
            downsample,
        })
    }
    
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        time_emb: Option<&Tensor>,
        encoder_hidden_states: &Tensor,
    ) -> CandleResult<(Tensor, Vec<Tensor>)> {
        let mut output_states = Vec::new();
        let mut hidden_states = hidden_states.clone();
        
        for (resnet, attention) in self.resnets.iter().zip(self.attentions.iter()) {
            hidden_states = resnet.forward(&hidden_states, time_emb)?;
            hidden_states = attention.forward(&hidden_states, encoder_hidden_states)?;
            output_states.push(hidden_states.clone());
        }
        
        if let Some(downsample) = &self.downsample {
            hidden_states = downsample.forward(&hidden_states)?;
            output_states.push(hidden_states.clone());
        }
        
        Ok((hidden_states, output_states))
    }
}

pub struct MidBlock {
    resnet_1: ResnetBlock,
    attention: CrossAttentionBlock,
    resnet_2: ResnetBlock,
}

impl MidBlock {
    pub fn new(
        in_channels: usize,
        time_emb_channels: Option<usize>,
        context_dim: usize,
        num_heads: usize,
        lora_config: Option<&LoRAConfig>,
        vb: VarBuilder,
    ) -> CandleResult<Self> {
        let resnet_1 = ResnetBlock::new(
            in_channels,
            in_channels,
            time_emb_channels,
            0.0,
            vb.pp("resnets.0")
        )?;
        
        let attention = CrossAttentionBlock::new(
            in_channels,
            context_dim,
            num_heads,
            lora_config,
            vb.pp("attentions.0")
        )?;
        
        let resnet_2 = ResnetBlock::new(
            in_channels,
            in_channels,
            time_emb_channels,
            0.0,
            vb.pp("resnets.1")
        )?;
        
        Ok(Self {
            resnet_1,
            attention,
            resnet_2,
        })
    }
    
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        time_emb: Option<&Tensor>,
        encoder_hidden_states: &Tensor,
    ) -> CandleResult<Tensor> {
        let hidden_states = self.resnet_1.forward(hidden_states, time_emb)?;
        let hidden_states = self.attention.forward(&hidden_states, encoder_hidden_states)?;
        self.resnet_2.forward(&hidden_states, time_emb)
    }
}

pub struct UpBlock {
    resnets: Vec<ResnetBlock>,
    attentions: Vec<CrossAttentionBlock>,
    upsample: Option<Conv2d>,
    num_layers: usize,
}

impl UpBlock {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        time_emb_channels: Option<usize>,
        num_layers: usize,
        context_dim: usize,
        num_heads: usize,
        add_upsample: bool,
        lora_config: Option<&LoRAConfig>,
        vb: VarBuilder,
    ) -> CandleResult<Self> {
        let mut resnets = Vec::new();
        let mut attentions = Vec::new();
        
        for i in 0..num_layers {
            let input_channels = if i == 0 { in_channels } else { out_channels };
            
            let resnet = ResnetBlock::new(
                input_channels,
                out_channels,
                time_emb_channels,
                0.0,
                vb.pp(&format!("resnets.{}", i))
            )?;
            resnets.push(resnet);
            
            let attention = CrossAttentionBlock::new(
                out_channels,
                context_dim,
                num_heads,
                lora_config,
                vb.pp(&format!("attentions.{}", i))
            )?;
            attentions.push(attention);
        }
        
        let upsample = if add_upsample {
            let config = candle_nn::Conv2dConfig {
                padding: 1,
                ..Default::default()
            };
            Some(candle_nn::conv2d(out_channels, out_channels, 3, config, vb.pp("upsamplers.0.conv"))?)
        } else {
            None
        };
        
        Ok(Self {
            resnets,
            attentions,
            upsample,
            num_layers,
        })
    }
    
    pub fn get_num_layers(&self) -> usize {
        self.num_layers
    }
    
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        res_hidden_states_tuple: &[Tensor],
        time_emb: Option<&Tensor>,
        encoder_hidden_states: &Tensor,
    ) -> CandleResult<Tensor> {
        let mut hidden_states = hidden_states.clone();
        
        for (i, (resnet, attention)) in self.resnets.iter().zip(self.attentions.iter()).enumerate() {
            if i < res_hidden_states_tuple.len() {
                let res_hidden_states = &res_hidden_states_tuple[i];
                hidden_states = Tensor::cat(&[&hidden_states, res_hidden_states], 1)?;
            }
            
            hidden_states = resnet.forward(&hidden_states, time_emb)?;
            hidden_states = attention.forward(&hidden_states, encoder_hidden_states)?;
        }
        
        if let Some(upsample) = &self.upsample {
            let (_, _, h, w) = hidden_states.dims4()?;
            let hidden_states = hidden_states.upsample_nearest2d(h * 2, w * 2)?;
            hidden_states = upsample.forward(&hidden_states)?;
        }
        
        Ok(hidden_states)
    }
}

// ==================== SDXL UNet Model ====================

pub struct SDXLUNet {
    pub config: SDXLUNetConfig,
    pub lora_config: Option<LoRAConfig>,
    time_embedding: TimeEmbedding,
    label_emb: Option<Linear>,
    conv_in: Conv2d,
    down_blocks: Vec<DownBlock>,
    mid_block: MidBlock,
    up_blocks: Vec<UpBlock>,
    conv_norm_out: GroupNorm,
    conv_out: Conv2d,
    lora_layers: HashMap<String, Box<dyn LoRALayer>>,
}

impl SDXLUNet {
    pub fn new(config: SDXLUNetConfig, lora_config: Option<LoRAConfig>, vb: VarBuilder) -> CandleResult<Self> {
        let time_embed_dim = config.model_channels * 4;
        let time_embedding = TimeEmbedding::new(time_embed_dim, vb.pp("time_embedding"))?;
        
        let label_emb = if let Some(adm_channels) = config.adm_in_channels {
            Some(candle_nn::linear(adm_channels, time_embed_dim, vb.pp("label_emb"))?)
        } else {
            None
        };
        
        let conv_config = candle_nn::Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let conv_in = candle_nn::conv2d(
            config.in_channels,
            config.model_channels,
            3,
            conv_config,
            vb.pp("conv_in")
        )?;
        
        let mut down_blocks = Vec::new();
        let mut input_channels = config.model_channels;
        
        for (i, &mult) in config.channel_mult.iter().enumerate() {
            let output_channels = config.model_channels * mult;
            let is_final_block = i == config.channel_mult.len() - 1;
            
            let down_block = DownBlock::new(
                input_channels,
                output_channels,
                Some(time_embed_dim),
                config.num_res_blocks,
                config.context_dim,
                config.num_heads,
                !is_final_block,
                lora_config.as_ref(),
                vb.pp(&format!("down_blocks.{}", i))
            )?;
            
            down_blocks.push(down_block);
            input_channels = output_channels;
        }
        
        let mid_block = MidBlock::new(
            input_channels,
            Some(time_embed_dim),
            config.context_dim,
            config.num_heads,
            lora_config.as_ref(),
            vb.pp("mid_block")
        )?;
        
        let mut up_blocks = Vec::new();
        let mut reversed_mult = config.channel_mult.clone();
        reversed_mult.reverse();
        
        for (i, &mult) in reversed_mult.iter().enumerate() {
            let output_channels = config.model_channels * mult;
            let prev_output_channels = if i == 0 {
                input_channels
            } else {
                config.model_channels * reversed_mult[i - 1]
            };
            
            let is_final_block = i == reversed_mult.len() - 1;
            
            let up_block = UpBlock::new(
                input_channels + prev_output_channels,
                output_channels,
                Some(time_embed_dim),
                config.num_res_blocks + 1,
                config.context_dim,
                config.num_heads,
                !is_final_block,
                lora_config.as_ref(),
                vb.pp(&format!("up_blocks.{}", i))
            )?;
            
            up_blocks.push(up_block);
            input_channels = output_channels;
        }
        
        let conv_norm_out = candle_nn::group_norm(32, input_channels, 1e-6, vb.pp("conv_norm_out"))?;
        let conv_out = candle_nn::conv2d(
            input_channels,
            config.out_channels,
            3,
            conv_config,
            vb.pp("conv_out")
        )?;
        
        Ok(Self {
            config,
            lora_config,
            time_embedding,
            label_emb,
            conv_in,
            down_blocks,
            mid_block,
            up_blocks,
            conv_norm_out,
            conv_out,
            lora_layers: HashMap::new(),
        })
    }
    
    pub fn forward(
        &self,
        sample: &Tensor,
        timestep: &Tensor,
        encoder_hidden_states: &Tensor,
        added_cond_kwargs: Option<&HashMap<String, Tensor>>,
    ) -> CandleResult<Tensor> {
        let mut time_emb = self.time_embedding.forward(timestep)?;
        
        if let Some(kwargs) = added_cond_kwargs {
            if let (Some(text_embeds), Some(time_ids)) = (kwargs.get("text_embeds"), kwargs.get("time_ids")) {
                let added_emb = Tensor::cat(&[text_embeds, time_ids], 1)?;
                if let Some(label_emb) = &self.label_emb {
                    let label_emb = label_emb.forward(&added_emb)?;
                    time_emb = time_emb.add(&label_emb)?;
                }
            }
        }
        
        let mut sample = self.conv_in.forward(sample)?;
        
        let mut down_block_res_samples = vec![sample.clone()];
        for down_block in &self.down_blocks {
            let (new_sample, res_samples) = down_block.forward(&sample, Some(&time_emb), encoder_hidden_states)?;
            sample = new_sample;
            down_block_res_samples.extend(res_samples);
        }
        
        sample = self.mid_block.forward(&sample, Some(&time_emb), encoder_hidden_states)?;
        
        for up_block in &self.up_blocks {
            let res_samples: Vec<Tensor> = down_block_res_samples
                .drain(down_block_res_samples.len().saturating_sub(up_block.get_num_layers())..)
                .collect();
            sample = up_block.forward(&sample, &res_samples, Some(&time_emb), encoder_hidden_states)?;
        }
        
        let sample = self.conv_norm_out.forward(&sample)?;
        let sample = sample.silu()?;
        self.conv_out.forward(&sample)
    }
    
    pub fn apply_lora(&mut self, lora_config: &LoRAConfig) -> Result<()> {
        self.lora_config = Some(lora_config.clone());
        Ok(())
    }
    
    pub fn merge_lora_weights(&mut self) -> CandleResult<()> {
        for (_, layer) in self.lora_layers.iter_mut() {
            layer.merge_weights()?;
        }
        Ok(())
    }
    
    pub fn unmerge_lora_weights(&mut self) -> CandleResult<()> {
        for (_, layer) in self.lora_layers.iter_mut() {
            layer.unmerge_weights()?;
        }
        Ok(())
    }
    
    pub fn save_lora_weights(&self, path: &Path) -> Result<()> {
        for (name, layer) in &self.lora_layers {
            let layer_path = path.join(format!("{}.safetensors", name));
            layer.save_adapter_weights(&layer_path)?;
        }
        Ok(())
    }
    
    pub fn load_lora_weights(&mut self, path: &Path) -> Result<()> {
        for (name, layer) in self.lora_layers.iter_mut() {
            let layer_path = path.join(format!("{}.safetensors", name));
            if layer_path.exists() {
                layer.load_adapter_weights(&layer_path)?;
            }
        }
        Ok(())
    }
}

// ==================== DDPM Scheduler ====================

#[derive(Debug, Clone)]
pub struct DDPMSchedulerConfig {
    pub num_train_timesteps: u32,
    pub beta_start: f64,
    pub beta_end: f64,
    pub beta_schedule: String,
    pub variance_type: String,
    pub clip_sample: bool,
    pub prediction_type: String,
}

impl Default for DDPMSchedulerConfig {
    fn default() -> Self {
        Self {
            num_train_timesteps: 1000,
            beta_start: 0.00085,
            beta_end: 0.012,
            beta_schedule: "scaled_linear".to_string(),
            variance_type: "fixed_small".to_string(),
            clip_sample: false,
            prediction_type: "epsilon".to_string(),
        }
    }
}

pub struct DDPMScheduler {
    pub config: DDPMSchedulerConfig,
    pub timesteps: Vec<u32>,
    pub init_noise_sigma: f32,
    alphas: Tensor,
    betas: Tensor,
    alphas_cumprod: Tensor,
    device: Device,
}

impl DDPMScheduler {
    pub fn new(config: DDPMSchedulerConfig, device: &Device) -> CandleResult<Self> {
        let betas = Self::get_beta_schedule(&config, device)?;
        let alphas = (Tensor::ones_like(&betas)? - &betas)?;
        let alphas_cumprod = Self::cumprod(&alphas)?;
        
        Ok(Self {
            config,
            timesteps: Vec::new(),
            init_noise_sigma: 1.0,
            alphas,
            betas,
            alphas_cumprod,
            device: device.clone(),
        })
    }
    
    fn get_beta_schedule(config: &DDPMSchedulerConfig, device: &Device) -> CandleResult<Tensor> {
        match config.beta_schedule.as_str() {
            "linear" => {
                let betas = Tensor::arange(
                    config.beta_start as f32,
                    config.beta_end as f32,
                    device,
                )?;
                Ok(betas)
            },
            "scaled_linear" => {
                let start = (config.beta_start.sqrt()) as f32;
                let end = (config.beta_end.sqrt()) as f32;
                let betas = Tensor::arange(start, end, device)?;
                let betas = betas.sqr()?;
                Ok(betas)
            },
            _ => {
                let betas = Tensor::arange(
                    config.beta_start as f32,
                    config.beta_end as f32,
                    device,
                )?;
                Ok(betas)
            }
        }
    }
    
    fn cumprod(tensor: &Tensor) -> CandleResult<Tensor> {
        let mut result = Vec::new();
        let data = tensor.to_vec1::<f32>()?;
        let mut cumulative = 1.0f32;
        
        for value in data {
            cumulative *= value;
            result.push(cumulative);
        }
        
        Tensor::from_slice(&result, tensor.shape(), tensor.device())
    }
    
    pub fn set_timesteps(&mut self, num_inference_steps: usize) {
        let step_ratio = self.config.num_train_timesteps as f32 / num_inference_steps as f32;
        let timesteps: Vec<u32> = (0..num_inference_steps)
            .map(|i| ((num_inference_steps - 1 - i) as f32 * step_ratio).round() as u32)
            .collect();
        self.timesteps = timesteps;
    }
    
    pub fn scale_model_input(&self, sample: &Tensor, _timestep: u32) -> CandleResult<Tensor> {
        Ok(sample.clone())
    }
    
    pub fn step(
        &self,
        model_output: &Tensor,
        timestep: u32,
        sample: &Tensor,
        _extra_step_kwargs: &HashMap<String, f32>,
    ) -> CandleResult<Tensor> {
        let t = timestep as usize;
        let alpha_prod_t = self.alphas_cumprod.i(t)?;
        let alpha_prod_t_prev = if t > 0 {
            self.alphas_cumprod.i(t - 1)?
        } else {
            Tensor::ones_like(&alpha_prod_t)?
        };
        
        let beta_prod_t = (Tensor::ones_like(&alpha_prod_t)? - &alpha_prod_t)?;
        let beta_prod_t_prev = (Tensor::ones_like(&alpha_prod_t_prev)? - &alpha_prod_t_prev)?;
        
        let pred_original_sample = (sample - &beta_prod_t.sqrt()?.mul(model_output)?)?.div(&alpha_prod_t.sqrt()?)?;
        
        let pred_sample_direction = beta_prod_t_prev.sqrt()?.mul(model_output)?;
        let prev_sample = alpha_prod_t_prev.sqrt()?.mul(&pred_original_sample)?.add(&pred_sample_direction)?;
        
        Ok(prev_sample)
    }
    
    pub fn add_noise(&self, original_samples: &Tensor, noise: &Tensor, timesteps: &Tensor) -> CandleResult<Tensor> {
        let sqrt_alpha_prod = self.get_sqrt_alpha_prod(timesteps)?;
        let sqrt_one_minus_alpha_prod = self.get_sqrt_one_minus_alpha_prod(timesteps)?;
        
        let noisy_samples = sqrt_alpha_prod.mul(original_samples)?.add(&sqrt_one_minus_alpha_prod.mul(noise)?)?;
        Ok(noisy_samples)
    }
    
    pub fn get_alpha_prod_t(&self, timesteps: &Tensor) -> CandleResult<Tensor> {
        let indices = timesteps.to_vec1::<u32>()?;
        let mut alpha_prods = Vec::new();
        
        for &t in &indices {
            let alpha_prod = self.alphas_cumprod.i(t as usize)?;
            alpha_prods.push(alpha_prod.to_scalar::<f32>()?);
        }
        
        Tensor::from_slice(&alpha_prods, (indices.len(),), timesteps.device())
    }
    
    fn get_sqrt_alpha_prod(&self, timesteps: &Tensor) -> CandleResult<Tensor> {
        let alpha_prod = self.get_alpha_prod_t(timesteps)?;
        alpha_prod.sqrt()
    }
    
    fn get_sqrt_one_minus_alpha_prod(&self, timesteps: &Tensor) -> CandleResult<Tensor> {
        let alpha_prod = self.get_alpha_prod_t(timesteps)?;
        let one_minus_alpha = (Tensor::ones_like(&alpha_prod)? - alpha_prod)?;
        one_minus_alpha.sqrt()
    }
}

// ==================== Tests ====================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_lora_linear_creation() -> CandleResult<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        
        let lora = LoRALinear::new(512, 512, 16, 16.0, 0.1, vb)?;
        assert_eq!(lora.get_rank(), 16);
        assert_eq!(lora.get_scaling(), 1.0);
        Ok(())
    }
    
    #[test]
    fn test_lora_factory() -> CandleResult<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        
        let config = LoRAConfig::default();
        let layer = LoRAFactory::create_linear_layer(&config, 256, 256, vb)?;
        assert_eq!(layer.get_rank(), 16);
        Ok(())
    }
    
    #[test]
    fn test_sdxl_unet_config() {
        let config = SDXLUNetConfig::default();
        assert_eq!(config.in_channels, 4);
        assert_eq!(config.out_channels, 4);
        assert_eq!(config.model_channels, 320);
    }
    
    #[test]
    fn test_ddpm_scheduler() -> CandleResult<()> {
        let device = Device::Cpu;
        let config = DDPMSchedulerConfig::default();
        let mut scheduler = DDPMScheduler::new(config, &device)?;
        
        scheduler.set_timesteps(50);
        assert_eq!(scheduler.timesteps.len(), 50);
        Ok(())
    }
}

// ==================== VAE Components ====================

#[derive(Debug, Clone)]
pub struct VAEConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub down_block_types: Vec<String>,
    pub up_block_types: Vec<String>,
    pub block_out_channels: Vec<usize>,
    pub layers_per_block: usize,
    pub act_fn: String,
    pub latent_channels: usize,
    pub norm_num_groups: usize,
    pub sample_size: usize,
    pub scaling_factor: f64,
}

impl Default for VAEConfig {
    fn default() -> Self {
        Self {
            in_channels: 3,
            out_channels: 3,
            down_block_types: vec![
                "DownEncoderBlock2D".to_string(),
                "DownEncoderBlock2D".to_string(),
                "DownEncoderBlock2D".to_string(),
                "DownEncoderBlock2D".to_string(),
            ],
            up_block_types: vec![
                "UpDecoderBlock2D".to_string(),
                "UpDecoderBlock2D".to_string(),
                "UpDecoderBlock2D".to_string(),
                "UpDecoderBlock2D".to_string(),
            ],
            block_out_channels: vec![128, 256, 512, 512],
            layers_per_block: 2,
            act_fn: "silu".to_string(),
            latent_channels: 4,
            norm_num_groups: 32,
            sample_size: 512,
            scaling_factor: 0.18215,
        }
    }
}

pub struct VAEEncoder {
    conv_in: Conv2d,
    down_blocks: Vec<DownEncoderBlock>,
    mid_block: MidBlock,
    conv_norm_out: GroupNorm,
    conv_out: Conv2d,
    config: VAEConfig,
}

impl VAEEncoder {
    pub fn new(config: VAEConfig, vb: VarBuilder) -> CandleResult<Self> {
        let conv_in = candle_nn::conv2d(
            config.in_channels,
            config.block_out_channels[0],
            3,
            candle_nn::Conv2dConfig { padding: 1, ..Default::default() },
            vb.pp("conv_in")
        )?;
        
        let mut down_blocks = Vec::new();
        let mut output_channel = config.block_out_channels[0];
        
        for (i, &out_channels) in config.block_out_channels.iter().enumerate() {
            let input_channel = output_channel;
            output_channel = out_channels;
            
            let is_final_block = i == config.block_out_channels.len() - 1;
            let add_downsample = !is_final_block;
            
            let down_block = DownEncoderBlock::new(
                input_channel,
                output_channel,
                config.layers_per_block,
                add_downsample,
                vb.pp(&format!("down_blocks.{}", i))
            )?;
            down_blocks.push(down_block);
        }
        
        let mid_block = MidBlock::new(
            output_channel,
            None,
            512, // context_dim for VAE
            8,   // num_heads
            None, // no LoRA for VAE
            vb.pp("mid_block")
        )?;
        
        let conv_norm_out = candle_nn::group_norm(
            config.norm_num_groups,
            output_channel,
            1e-6,
            vb.pp("conv_norm_out")
        )?;
        
        let conv_out = candle_nn::conv2d(
            output_channel,
            config.latent_channels * 2, // mean and logvar
            3,
            candle_nn::Conv2dConfig { padding: 1, ..Default::default() },
            vb.pp("conv_out")
        )?;
        
        Ok(Self {
            conv_in,
            down_blocks,
            mid_block,
            conv_norm_out,
            conv_out,
            config,
        })
    }
    
    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let mut sample = self.conv_in.forward(x)?;
        
        for down_block in &self.down_blocks {
            sample = down_block.forward(&sample)?;
        }
        
        sample = self.mid_block.forward(&sample, None, &Tensor::zeros((1, 77, 512), sample.device())?)?;
        
        sample = self.conv_norm_out.forward(&sample)?;
        sample = sample.silu()?;
        sample = self.conv_out.forward(&sample)?;
        
        // Split into mean and logvar for VAE
        let (mean, logvar) = sample.chunk(2, 1)?;
        
        // Sample using reparameterization trick
        let std = (logvar * 0.5)?.exp()?;
        let eps = Tensor::randn(0f32, 1f32, mean.shape(), mean.device())?;
        let sample = mean.add(&std.mul(&eps)?)?;
        
        // Scale by scaling factor
        sample.mul(&Tensor::new(&[self.config.scaling_factor as f32], sample.device())?)
    }
}

pub struct VAEDecoder {
    conv_in: Conv2d,
    mid_block: MidBlock,
    up_blocks: Vec<UpDecoderBlock>,
    conv_norm_out: GroupNorm,
    conv_out: Conv2d,
    config: VAEConfig,
}

impl VAEDecoder {
    pub fn new(config: VAEConfig, vb: VarBuilder) -> CandleResult<Self> {
        let conv_in = candle_nn::conv2d(
            config.latent_channels,
            config.block_out_channels.last().unwrap().clone(),
            3,
            candle_nn::Conv2dConfig { padding: 1, ..Default::default() },
            vb.pp("conv_in")
        )?;
        
        let mid_block = MidBlock::new(
            *config.block_out_channels.last().unwrap(),
            None,
            512,
            8,
            None,
            vb.pp("mid_block")
        )?;
        
        let mut up_blocks = Vec::new();
        let mut reversed_block_out_channels = config.block_out_channels.clone();
        reversed_block_out_channels.reverse();
        
        let mut output_channel = *reversed_block_out_channels.first().unwrap();
        
        for (i, &out_channels) in reversed_block_out_channels.iter().enumerate() {
            let is_final_block = i == reversed_block_out_channels.len() - 1;
            let prev_output_channel = if i == 0 {
                output_channel
            } else {
                reversed_block_out_channels[i - 1]
            };
            output_channel = out_channels;
            
            let add_upsample = !is_final_block;
            
            let up_block = UpDecoderBlock::new(
                prev_output_channel,
                output_channel,
                config.layers_per_block,
                add_upsample,
                vb.pp(&format!("up_blocks.{}", i))
            )?;
            up_blocks.push(up_block);
        }
        
        let conv_norm_out = candle_nn::group_norm(
            config.norm_num_groups,
            output_channel,
            1e-6,
            vb.pp("conv_norm_out")
        )?;
        
        let conv_out = candle_nn::conv2d(
            output_channel,
            config.out_channels,
            3,
            candle_nn::Conv2dConfig { padding: 1, ..Default::default() },
            vb.pp("conv_out")
        )?;
        
        Ok(Self {
            conv_in,
            mid_block,
            up_blocks,
            conv_norm_out,
            conv_out,
            config,
        })
    }
    
    pub fn forward(&self, z: &Tensor) -> CandleResult<Tensor> {
        // Scale by inverse scaling factor
        let sample = z.div(&Tensor::new(&[self.config.scaling_factor as f32], z.device())?)?;
        
        let mut sample = self.conv_in.forward(&sample)?;
        
        sample = self.mid_block.forward(&sample, None, &Tensor::zeros((1, 77, 512), sample.device())?)?;
        
        for up_block in &self.up_blocks {
            sample = up_block.forward(&sample)?;
        }
        
        sample = self.conv_norm_out.forward(&sample)?;
        sample = sample.silu()?;
        sample = self.conv_out.forward(&sample)?;
        
        Ok(sample)
    }
}

pub struct DownEncoderBlock {
    resnets: Vec<ResnetBlock>,
    downsample: Option<Conv2d>,
}

impl DownEncoderBlock {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        num_layers: usize,
        add_downsample: bool,
        vb: VarBuilder,
    ) -> CandleResult<Self> {
        let mut resnets = Vec::new();
        
        for i in 0..num_layers {
            let input_channels = if i == 0 { in_channels } else { out_channels };
            
            let resnet = ResnetBlock::new(
                input_channels,
                out_channels,
                None, // no time embedding for VAE
                0.0,
                vb.pp(&format!("resnets.{}", i))
            )?;
            resnets.push(resnet);
        }
        
        let downsample = if add_downsample {
            Some(candle_nn::conv2d(
                out_channels,
                out_channels,
                3,
                candle_nn::Conv2dConfig { stride: 2, padding: 1, ..Default::default() },
                vb.pp("downsamplers.0.conv")
            )?)
        } else {
            None
        };
        
        Ok(Self {
            resnets,
            downsample,
        })
    }
    
    pub fn forward(&self, hidden_states: &Tensor) -> CandleResult<Tensor> {
        let mut hidden_states = hidden_states.clone();
        
        for resnet in &self.resnets {
            hidden_states = resnet.forward(&hidden_states, None)?;
        }
        
        if let Some(downsample) = &self.downsample {
            hidden_states = downsample.forward(&hidden_states)?;
        }
        
        Ok(hidden_states)
    }
}

pub struct UpDecoderBlock {
    resnets: Vec<ResnetBlock>,
    upsample: Option<Conv2d>,
}

impl UpDecoderBlock {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        num_layers: usize,
        add_upsample: bool,
        vb: VarBuilder,
    ) -> CandleResult<Self> {
        let mut resnets = Vec::new();
        
        for i in 0..num_layers {
            let input_channels = if i == 0 { in_channels } else { out_channels };
            
            let resnet = ResnetBlock::new(
                input_channels,
                out_channels,
                None,
                0.0,
                vb.pp(&format!("resnets.{}", i))
            )?;
            resnets.push(resnet);
        }
        
        let upsample = if add_upsample {
            Some(candle_nn::conv2d(
                out_channels,
                out_channels,
                3,
                candle_nn::Conv2dConfig { padding: 1, ..Default::default() },
                vb.pp("upsamplers.0.conv")
            )?)
        } else {
            None
        };
        
        Ok(Self {
            resnets,
            upsample,
        })
    }
    
    pub fn forward(&self, hidden_states: &Tensor) -> CandleResult<Tensor> {
        let mut hidden_states = hidden_states.clone();
        
        for resnet in &self.resnets {
            hidden_states = resnet.forward(&hidden_states, None)?;
        }
        
        if let Some(upsample) = &self.upsample {
            let (_, _, h, w) = hidden_states.dims4()?;
            hidden_states = hidden_states.upsample_nearest2d(h * 2, w * 2)?;
            hidden_states = upsample.forward(&hidden_states)?;
        }
        
        Ok(hidden_states)
    }
}

// ==================== Text Encoder Components ====================

#[derive(Debug, Clone)]
pub struct CLIPTokenizer {
    vocab_size: usize,
    max_length: usize,
    pad_token_id: u32,
    bos_token_id: u32,
    eos_token_id: u32,
    unk_token_id: u32,
}

impl CLIPTokenizer {
    pub fn new() -> Self {
        Self {
            vocab_size: 49408,
            max_length: 77,
            pad_token_id: 1,
            bos_token_id: 49406,
            eos_token_id: 49407,
            unk_token_id: 0,
        }
    }
    
    pub fn encode(&self, text: &str) -> Vec<u32> {
        // Simplified tokenization - in production use a proper tokenizer
        let mut tokens = vec![self.bos_token_id];
        
        // Simple word-based tokenization for demo
        let words: Vec<&str> = text.split_whitespace().collect();
        for word in words {
            // Hash-based token ID generation (simplified)
            let token_id = (word.len() * 1000 + word.chars().next().unwrap_or('a') as usize) % self.vocab_size;
            tokens.push(token_id as u32);
        }
        
        tokens.push(self.eos_token_id);
        
        // Pad or truncate to max_length
        while tokens.len() < self.max_length {
            tokens.push(self.pad_token_id);
        }
        tokens.truncate(self.max_length);
        
        tokens
    }
    
    pub fn encode_batch(&self, texts: &[String]) -> Vec<Vec<u32>> {
        texts.iter().map(|text| self.encode(text)).collect()
    }
}

pub struct CLIPTextEmbedding {
    token_embedding: candle_nn::Embedding,
    position_embedding: candle_nn::Embedding,
    max_position_embeddings: usize,
}

impl CLIPTextEmbedding {
    pub fn new(
        vocab_size: usize,
        embed_dim: usize,
        max_position_embeddings: usize,
        vb: VarBuilder,
    ) -> CandleResult<Self> {
        let token_embedding = candle_nn::embedding(vocab_size, embed_dim, vb.pp("token_embedding"))?;
        let position_embedding = candle_nn::embedding(max_position_embeddings, embed_dim, vb.pp("position_embedding"))?;
        
        Ok(Self {
            token_embedding,
            position_embedding,
            max_position_embeddings,
        })
    }
    
    pub fn forward(&self, input_ids: &Tensor) -> CandleResult<Tensor> {
        let (batch_size, seq_len) = input_ids.dims2()?;
        
        let token_embeds = self.token_embedding.forward(input_ids)?;
        
        let position_ids = Tensor::arange(0u32, seq_len as u32, input_ids.device())?
            .unsqueeze(0)?
            .expand((batch_size, seq_len))?;
        let position_embeds = self.position_embedding.forward(&position_ids)?;
        
        token_embeds.add(&position_embeds)
    }
}

pub struct CLIPTextEncoder {
    embeddings: CLIPTextEmbedding,
    encoder_layers: Vec<CLIPEncoderLayer>,
    final_layer_norm: candle_nn::LayerNorm,
    tokenizer: CLIPTokenizer,
}

impl CLIPTextEncoder {
    pub fn new(
        vocab_size: usize,
        embed_dim: usize,
        num_layers: usize,
        num_heads: usize,
        max_position_embeddings: usize,
        vb: VarBuilder,
    ) -> CandleResult<Self> {
        let embeddings = CLIPTextEmbedding::new(vocab_size, embed_dim, max_position_embeddings, vb.pp("embeddings"))?;
        
        let mut encoder_layers = Vec::new();
        for i in 0..num_layers {
            let layer = CLIPEncoderLayer::new(embed_dim, num_heads, vb.pp(&format!("encoder.layers.{}", i)))?;
            encoder_layers.push(layer);
        }
        
        let final_layer_norm = candle_nn::layer_norm(embed_dim, 1e-5, vb.pp("final_layer_norm"))?;
        let tokenizer = CLIPTokenizer::new();
        
        Ok(Self {
            embeddings,
            encoder_layers,
            final_layer_norm,
            tokenizer,
        })
    }
    
    pub fn encode_text(&self, texts: &[String]) -> CandleResult<(Tensor, Tensor)> {
        let token_batch = self.tokenizer.encode_batch(texts);
        let batch_size = token_batch.len();
        let seq_len = token_batch[0].len();
        
        let input_ids = Tensor::from_slice(
            &token_batch.into_iter().flatten().collect::<Vec<_>>(),
            (batch_size, seq_len),
            &Device::Cpu,
        )?;
        
        let hidden_states = self.embeddings.forward(&input_ids)?;
        let mut hidden_states = hidden_states;
        
        for layer in &self.encoder_layers {
            hidden_states = layer.forward(&hidden_states)?;
        }
        
        let last_hidden_state = self.final_layer_norm.forward(&hidden_states)?;
        
        // Get pooled output (last token)
        let pooled_output = last_hidden_state.i((.., seq_len - 1, ..))?;
        
        Ok((last_hidden_state, pooled_output))
    }
}

pub struct CLIPEncoderLayer {
    self_attn: CLIPAttention,
    mlp: CLIPMLP,
    layer_norm1: candle_nn::LayerNorm,
    layer_norm2: candle_nn::LayerNorm,
}

impl CLIPEncoderLayer {
    pub fn new(embed_dim: usize, num_heads: usize, vb: VarBuilder) -> CandleResult<Self> {
        let self_attn = CLIPAttention::new(embed_dim, num_heads, vb.pp("self_attn"))?;
        let mlp = CLIPMLP::new(embed_dim, vb.pp("mlp"))?;
        let layer_norm1 = candle_nn::layer_norm(embed_dim, 1e-5, vb.pp("layer_norm1"))?;
        let layer_norm2 = candle_nn::layer_norm(embed_dim, 1e-5, vb.pp("layer_norm2"))?;
        
        Ok(Self {
            self_attn,
            mlp,
            layer_norm1,
            layer_norm2,
        })
    }
    
    pub fn forward(&self, hidden_states: &Tensor) -> CandleResult<Tensor> {
        let residual = hidden_states.clone();
        let hidden_states = self.layer_norm1.forward(hidden_states)?;
        let hidden_states = self.self_attn.forward(&hidden_states)?;
        let hidden_states = residual.add(&hidden_states)?;
        
        let residual = hidden_states.clone();
        let hidden_states = self.layer_norm2.forward(&hidden_states)?;
        let hidden_states = self.mlp.forward(&hidden_states)?;
        residual.add(&hidden_states)
    }
}

pub struct CLIPAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl CLIPAttention {
    pub fn new(embed_dim: usize, num_heads: usize, vb: VarBuilder) -> CandleResult<Self> {
        let head_dim = embed_dim / num_heads;
        let scale = 1.0 / (head_dim as f64).sqrt();
        
        let q_proj = candle_nn::linear(embed_dim, embed_dim, vb.pp("q_proj"))?;
        let k_proj = candle_nn::linear(embed_dim, embed_dim, vb.pp("k_proj"))?;
        let v_proj = candle_nn::linear(embed_dim, embed_dim, vb.pp("v_proj"))?;
        let out_proj = candle_nn::linear(embed_dim, embed_dim, vb.pp("out_proj"))?;
        
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads,
            head_dim,
            scale,
        })
    }
    
    pub fn forward(&self, hidden_states: &Tensor) -> CandleResult<Tensor> {
        let (batch_size, seq_len, _) = hidden_states.dims3()?;
        
        let q = self.q_proj.forward(hidden_states)?;
        let k = self.k_proj.forward(hidden_states)?;
        let v = self.v_proj.forward(hidden_states)?;
        
        let q = q.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
                 .transpose(1, 2)?;
        let k = k.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
                 .transpose(1, 2)?;
        let v = v.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
                 .transpose(1, 2)?;
        
        let scores = q.matmul(&k.transpose(2, 3)?)?;
        let scores = scores.mul(&Tensor::new(&[self.scale as f32], hidden_states.device())?)?;
        let attention_weights = softmax(&scores, D::Minus1)?;
        let out = attention_weights.matmul(&v)?;
        
        let out = out.transpose(1, 2)?
                     .reshape((batch_size, seq_len, self.num_heads * self.head_dim))?;
        
        self.out_proj.forward(&out)
    }
}

pub struct CLIPMLP {
    fc1: Linear,
    fc2: Linear,
}

impl CLIPMLP {
    pub fn new(embed_dim: usize, vb: VarBuilder) -> CandleResult<Self> {
        let intermediate_size = embed_dim * 4;
        let fc1 = candle_nn::linear(embed_dim, intermediate_size, vb.pp("fc1"))?;
        let fc2 = candle_nn::linear(intermediate_size, embed_dim, vb.pp("fc2"))?;
        
        Ok(Self { fc1, fc2 })
    }
    
    pub fn forward(&self, hidden_states: &Tensor) -> CandleResult<Tensor> {
        let hidden_states = self.fc1.forward(hidden_states)?;
        let hidden_states = hidden_states.gelu()?;
        self.fc2.forward(&hidden_states)
    }
}

// ==================== SDXL Pipeline ====================

#[derive(Debug, Clone)]
pub struct SDXLPipelineConfig {
    pub num_inference_steps: usize,
    pub guidance_scale: f64,
    pub height: usize,
    pub width: usize,
    pub eta: f64,
    pub generator_seed: Option<u64>,
    pub negative_prompt: Option<String>,
}

impl Default for SDXLPipelineConfig {
    fn default() -> Self {
        Self {
            num_inference_steps: 50,
            guidance_scale: 7.5,
            height: 1024,
            width: 1024,
            eta: 0.0,
            generator_seed: None,
            negative_prompt: None,
        }
    }
}

pub struct SDXLPipeline {
    pub unet: SDXLUNet,
    pub text_encoder: CLIPTextEncoder,
    pub text_encoder_2: CLIPTextEncoder,
    pub vae_decoder: VAEDecoder,
    pub vae_encoder: VAEEncoder,
    pub scheduler: DDPMScheduler,
    pub device: Device,
}

impl SDXLPipeline {
    pub fn new(
        unet: SDXLUNet,
        text_encoder: CLIPTextEncoder,
        text_encoder_2: CLIPTextEncoder,
        vae_decoder: VAEDecoder,
        vae_encoder: VAEEncoder,
        scheduler: DDPMScheduler,
        device: Device,
    ) -> Self {
        Self {
            unet,
            text_encoder,
            text_encoder_2,
            vae_decoder,
            vae_encoder,
