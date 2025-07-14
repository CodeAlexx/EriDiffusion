//! Flux double block implementation matching Candle's exact structure

use anyhow::Result;
use candle_core::{Module, Tensor, D};
use candle_nn::{LayerNorm, Linear, VarBuilder, linear};

use super::attention_candle::FluxSelfAttentionWithLoRA;
use super::modulation::Modulation2;

/// MLP matching Candle's exact structure
#[derive(Debug, Clone)]
pub struct Mlp {
    lin1: Linear,  // Named "0" in the weights
    lin2: Linear,  // Named "2" in the weights
}

impl Mlp {
    pub fn new(in_sz: usize, mlp_sz: usize, vb: VarBuilder) -> Result<Self> {
        // Note: Candle uses "0" and "2" as the layer names
        let lin1 = linear(in_sz, mlp_sz, vb.pp("0"))?;
        let lin2 = linear(mlp_sz, in_sz, vb.pp("2"))?;
        Ok(Self { lin1, lin2 })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.lin1)?.gelu()?.apply(&self.lin2)
    }
}

/// MLP with LoRA support
pub struct MlpWithLoRA {
    base: Mlp,
    // LoRA modules for lin1 and lin2
    lin1_lora: Option<crate::networks::lora::LoRAModule>,
    lin2_lora: Option<crate::networks::lora::LoRAModule>,
}

impl MlpWithLoRA {
    pub fn new(
        in_sz: usize,
        mlp_sz: usize,
        vb: VarBuilder,
        lora_rank: Option<usize>,
        lora_alpha: Option<f32>,
    ) -> Result<Self> {
        let base = Mlp::new(in_sz, mlp_sz, vb)?;
        
        let (lin1_lora, lin2_lora) = if let Some(rank) = lora_rank {
            let alpha = lora_alpha.unwrap_or(rank as f32);
            let device = vb.device();
            let dtype = vb.dtype();
            
            let lin1_lora = Some(crate::networks::lora::LoRAModule::new(
                in_sz, mlp_sz, rank, alpha, device.clone(), dtype,
            )?);
            
            let lin2_lora = Some(crate::networks::lora::LoRAModule::new(
                mlp_sz, in_sz, rank, alpha, device.clone(), dtype,
            )?);
            
            (lin1_lora, lin2_lora)
        } else {
            (None, None)
        };
        
        Ok(Self { base, lin1_lora, lin2_lora })
    }
    
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // First linear + LoRA
        let mut h = xs.apply(&self.base.lin1)?;
        if let Some(lora) = &self.lin1_lora {
            h = (h + lora.forward(xs)?)?;
        }
        
        // GELU activation
        h = h.gelu()?;
        
        // Second linear + LoRA
        let mut out = h.apply(&self.base.lin2)?;
        if let Some(lora) = &self.lin2_lora {
            out = (out + lora.forward(&h)?)?;
        }
        
        Ok(out)
    }
}

fn layer_norm(dim: usize, vb: VarBuilder) -> Result<LayerNorm> {
    let ws = Tensor::ones(dim, vb.dtype(), vb.device())?;
    Ok(LayerNorm::new_no_bias(ws, 1e-6))
}

/// Flux DoubleStreamBlock matching Candle's exact structure
#[derive(Debug, Clone)]
pub struct FluxDoubleStreamBlock {
    img_mod: Modulation2,
    img_norm1: LayerNorm,
    img_attn: FluxSelfAttentionWithLoRA,
    img_norm2: LayerNorm,
    img_mlp: MlpWithLoRA,
    txt_mod: Modulation2,
    txt_norm1: LayerNorm,
    txt_attn: FluxSelfAttentionWithLoRA,
    txt_norm2: LayerNorm,
    txt_mlp: MlpWithLoRA,
}

impl FluxDoubleStreamBlock {
    pub fn new(
        cfg: &super::model::FluxConfig,
        vb: VarBuilder,
        lora_rank: Option<usize>,
        lora_alpha: Option<f32>,
    ) -> Result<Self> {
        let h_sz = cfg.hidden_size;
        let mlp_sz = (h_sz as f64 * cfg.mlp_ratio) as usize;
        
        // Image stream
        let img_mod = Modulation2::new(h_sz, vb.pp("img_mod"))?;
        let img_norm1 = layer_norm(h_sz, vb.pp("img_norm1"))?;
        let img_attn = FluxSelfAttentionWithLoRA::new(
            h_sz, cfg.num_heads, cfg.qkv_bias, vb.pp("img_attn"), lora_rank, lora_alpha
        )?;
        let img_norm2 = layer_norm(h_sz, vb.pp("img_norm2"))?;
        let img_mlp = MlpWithLoRA::new(h_sz, mlp_sz, vb.pp("img_mlp"), lora_rank, lora_alpha)?;
        
        // Text stream
        let txt_mod = Modulation2::new(h_sz, vb.pp("txt_mod"))?;
        let txt_norm1 = layer_norm(h_sz, vb.pp("txt_norm1"))?;
        let txt_attn = FluxSelfAttentionWithLoRA::new(
            h_sz, cfg.num_heads, cfg.qkv_bias, vb.pp("txt_attn"), lora_rank, lora_alpha
        )?;
        let txt_norm2 = layer_norm(h_sz, vb.pp("txt_norm2"))?;
        let txt_mlp = MlpWithLoRA::new(h_sz, mlp_sz, vb.pp("txt_mlp"), lora_rank, lora_alpha)?;
        
        Ok(Self {
            img_mod,
            img_norm1,
            img_attn,
            img_norm2,
            img_mlp,
            txt_mod,
            txt_norm1,
            txt_attn,
            txt_norm2,
            txt_mlp,
        })
    }
    
    pub fn forward(
        &self,
        img: &Tensor,
        txt: &Tensor,
        vec_: &Tensor,
        pe: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        // Get modulation parameters
        let (img_mod1, img_mod2) = self.img_mod.forward(vec_)?;
        let (txt_mod1, txt_mod2) = self.txt_mod.forward(vec_)?;
        
        // Image stream
        let img_modulated = img.apply(&img_mod1.scale_shift(&self.img_norm1))?;
        let img_attn_out = self.img_attn.forward(&img_modulated, pe)?;
        let img = (img + img_mod1.gate(&img_attn_out)?)?;
        
        let img_modulated = img.apply(&img_mod2.scale_shift(&self.img_norm2))?;
        let img_mlp_out = self.img_mlp.forward(&img_modulated)?;
        let img = (img + img_mod2.gate(&img_mlp_out)?)?;
        
        // Text stream
        let txt_modulated = txt.apply(&txt_mod1.scale_shift(&self.txt_norm1))?;
        let txt_attn_out = self.txt_attn.forward(&txt_modulated, pe)?;
        let txt = (txt + txt_mod1.gate(&txt_attn_out)?)?;
        
        let txt_modulated = txt.apply(&txt_mod2.scale_shift(&self.txt_norm2))?;
        let txt_mlp_out = self.txt_mlp.forward(&txt_modulated)?;
        let txt = (txt + txt_mod2.gate(&txt_mlp_out)?)?;
        
        Ok((img, txt))
    }
}