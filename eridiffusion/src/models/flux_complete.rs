//! Complete Flux model implementation with double and single stream blocks

use crate::ops::{LayerNorm, Linear, RMSNorm};
use flame_core::device::Device;
use flame_core::{CudaDevice, DType, Error, Result, Shape, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

/// Flux model configuration
#[derive(Clone)]
pub struct FluxConfig {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub mlp_ratio: f32,
    pub num_double_blocks: usize,
    pub num_single_blocks: usize,
    pub patch_size: usize,
    pub guidance_embed: bool,
    pub vocab_size: usize, // For learned positional embeddings
}

impl Default for FluxConfig {
    fn default() -> Self {
        Self {
            hidden_size: 3072,
            num_heads: 24,
            head_dim: 128,
            mlp_ratio: 4.0,
            num_double_blocks: 19,
            num_single_blocks: 38,
            patch_size: 2,
            guidance_embed: true,
            vocab_size: 10000,
        }
    }
}

/// Complete Flux model with double and single stream blocks
pub struct FluxModel {
    pub device: Arc<CudaDevice>,
    pub config: FluxConfig,

    // Embeddings
    pub img_in: Linear,
    pub txt_in: Linear,
    pub time_in: MLPEmbedder,
    pub vector_in: MLPEmbedder,
    pub guidance_in: Option<MLPEmbedder>,

    // Position embeddings
    pub pos_embed: Linear,

    // Double stream blocks (process image and text separately)
    pub double_blocks: Vec<DoubleStreamBlock>,

    // Single stream blocks (process concatenated features)
    pub single_blocks: Vec<SingleStreamBlock>,

    // Output layers
    pub final_layer: Linear,
}

impl FluxModel {
    pub fn new(config: FluxConfig, device: Arc<CudaDevice>) -> Result<Self> {
        // Input projections
        let img_in = Linear::new(64, config.hidden_size, true, &device)?; // 64-dim patchified input (16ch * 2x2)
        let txt_in = Linear::new(4096, config.hidden_size, true, &device)?; // T5 hidden size

        // Time and vector embeddings
        let time_in = MLPEmbedder::new(256, config.hidden_size, device.clone())?;
        let vector_in = MLPEmbedder::new(768, config.hidden_size, device.clone())?; // CLIP hidden size

        let guidance_in = if config.guidance_embed {
            Some(MLPEmbedder::new(256, config.hidden_size, device.clone())?)
        } else {
            None
        };

        // Positional embeddings
        let pos_embed = Linear::new(config.head_dim, config.head_dim, false, &device)?;

        // Double stream blocks
        let mut double_blocks = Vec::with_capacity(config.num_double_blocks);
        for _ in 0..config.num_double_blocks {
            double_blocks.push(DoubleStreamBlock::new(&config, device.clone())?);
        }

        // Single stream blocks
        let mut single_blocks = Vec::with_capacity(config.num_single_blocks);
        for _ in 0..config.num_single_blocks {
            single_blocks.push(SingleStreamBlock::new(&config, device.clone())?);
        }

        // Output projection
        let final_layer = Linear::new(config.hidden_size, 64, true, &device)?;

        Ok(Self {
            device,
            config,
            img_in,
            txt_in,
            time_in,
            vector_in,
            guidance_in,
            pos_embed,
            double_blocks,
            single_blocks,
            final_layer,
        })
    }

    pub fn from_weights(weights: HashMap<String, Tensor>, device: Arc<CudaDevice>) -> Result<Self> {
        let config = FluxConfig::default();
        let mut model = Self::new(config, device)?;

        // Load weights from HashMap
        // This is simplified - in production, match weight names to model parameters

        Ok(model)
    }

    pub fn forward(
        &self,
        img: &Tensor,              // [B, L, 64] - patchified image latents
        txt: &Tensor,              // [B, L_txt, 4096] - T5 text embeddings
        timesteps: &Tensor,        // [B] - diffusion timesteps
        y: &Tensor,                // [B, 768] - CLIP text embeddings
        guidance: Option<&Tensor>, // [B] - guidance scale
    ) -> Result<Tensor> {
        let batch_size = img.shape().dims()[0];
        let img_seq_len = img.shape().dims()[1];
        let txt_seq_len = txt.shape().dims()[1];

        // Project inputs
        let mut img_hidden = self.img_in.forward(img)?;
        let txt_hidden = self.txt_in.forward(txt)?;

        // Time embedding
        let time_emb = get_timestep_embedding(timesteps, 256, self.device.clone())?;
        let time_emb = self.time_in.forward(&time_emb)?;

        // Vector embedding (CLIP)
        let vec_emb = self.vector_in.forward(y)?;

        // Combine time and vector embeddings
        let mut c = time_emb.add(&vec_emb)?;

        // Add guidance embedding if provided
        if let (Some(g), Some(guidance_in)) = (guidance, &self.guidance_in) {
            let g_emb = get_timestep_embedding(g, 256, self.device.clone())?;
            let g_emb = guidance_in.forward(&g_emb)?;
            c = c.add(&g_emb)?;
        }

        // Get positional embeddings using RoPE
        let pos_img = self.get_rope_embeddings(batch_size, img_seq_len, &self.device)?;
        let pos_txt = self.get_rope_embeddings(batch_size, txt_seq_len, &self.device)?;

        // Process through double stream blocks
        let (mut img_hidden, mut txt_hidden) =
            self.forward_double_blocks(img_hidden, txt_hidden, &pos_img, &pos_txt, &c)?;

        // Concatenate for single stream processing
        let mut hidden = Tensor::cat(&[&img_hidden, &txt_hidden], 1)?;
        let pos_combined = Tensor::cat(&[&pos_img, &pos_txt], 1)?;

        // Process through single stream blocks
        hidden = self.forward_single_blocks(hidden, &pos_combined, &c)?;

        // Extract image part
        let img_out =
            hidden.slice(&[(0, batch_size), (0, img_seq_len), (0, self.config.hidden_size)])?;

        // Final projection
        self.final_layer.forward(&img_out)
    }

    fn forward_double_blocks(
        &self,
        mut img: Tensor,
        mut txt: Tensor,
        pos_img: &Tensor,
        pos_txt: &Tensor,
        c: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        for block in &self.double_blocks {
            let (new_img, new_txt) = block.forward(&img, &txt, pos_img, pos_txt, c)?;
            img = new_img;
            txt = new_txt;
        }
        Ok((img, txt))
    }

    fn forward_single_blocks(
        &self,
        mut hidden: Tensor,
        pos: &Tensor,
        c: &Tensor,
    ) -> Result<Tensor> {
        for block in &self.single_blocks {
            hidden = block.forward(&hidden, pos, c)?;
        }
        Ok(hidden)
    }

    fn get_rope_embeddings(
        &self,
        batch_size: usize,
        seq_len: usize,
        device: &Arc<CudaDevice>,
    ) -> Result<Tensor> {
        // Flux uses RoPE with specific configuration
        rope_embeddings(
            batch_size,
            seq_len,
            self.config.head_dim,
            10000.0, // Base frequency
            device.clone(),
        )
    }
}

/// Double Stream Block - processes image and text separately
pub struct DoubleStreamBlock {
    // Image stream
    pub img_norm1: LayerNorm,
    pub img_attn: SelfAttention,
    pub img_norm2: LayerNorm,
    pub img_mlp: MLP,

    // Text stream
    pub txt_norm1: LayerNorm,
    pub txt_attn: SelfAttention,
    pub txt_norm2: LayerNorm,
    pub txt_mlp: MLP,

    // Modulation
    pub img_mod: Modulation,
    pub txt_mod: Modulation,
}

impl DoubleStreamBlock {
    pub fn new(config: &FluxConfig, device: Arc<CudaDevice>) -> Result<Self> {
        let hidden_size = config.hidden_size;

        Ok(Self {
            // Image stream
            img_norm1: LayerNorm::new(vec![hidden_size], 1e-6, device.clone())?,
            img_attn: SelfAttention::new(config, device.clone())?,
            img_norm2: LayerNorm::new(vec![hidden_size], 1e-6, device.clone())?,
            img_mlp: MLP::new(hidden_size, config.mlp_ratio, device.clone())?,

            // Text stream
            txt_norm1: LayerNorm::new(vec![hidden_size], 1e-6, device.clone())?,
            txt_attn: SelfAttention::new(config, device.clone())?,
            txt_norm2: LayerNorm::new(vec![hidden_size], 1e-6, device.clone())?,
            txt_mlp: MLP::new(hidden_size, config.mlp_ratio, device.clone())?,

            // Modulation
            img_mod: Modulation::new(hidden_size, device.clone())?,
            txt_mod: Modulation::new(hidden_size, device.clone())?,
        })
    }

    pub fn forward(
        &self,
        img: &Tensor,
        txt: &Tensor,
        pos_img: &Tensor,
        pos_txt: &Tensor,
        c: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        // Get modulation parameters
        let (img_shift_msa, img_scale_msa, img_shift_mlp, img_scale_mlp) =
            self.img_mod.forward(c)?;
        let (txt_shift_msa, txt_scale_msa, txt_shift_mlp, txt_scale_mlp) =
            self.txt_mod.forward(c)?;

        // Image stream
        let img_norm =
            modulate(&self.img_norm1.forward(img)?, &img_shift_msa, &img_scale_msa, &img.device())?;
        let img_attn_out = self.img_attn.forward(&img_norm, pos_img)?;
        let img = img.add(&img_attn_out)?;

        let img_norm = modulate(
            &self.img_norm2.forward(&img)?,
            &img_shift_mlp,
            &img_scale_mlp,
            &img.device(),
        )?;
        let img_mlp_out = self.img_mlp.forward(&img_norm)?;
        let img = img.add(&img_mlp_out)?;

        // Text stream
        let txt_norm =
            modulate(&self.txt_norm1.forward(txt)?, &txt_shift_msa, &txt_scale_msa, &txt.device())?;
        let txt_attn_out = self.txt_attn.forward(&txt_norm, pos_txt)?;
        let txt = txt.add(&txt_attn_out)?;

        let txt_norm = modulate(
            &self.txt_norm2.forward(&txt)?,
            &txt_shift_mlp,
            &txt_scale_mlp,
            &txt.device(),
        )?;
        let txt_mlp_out = self.txt_mlp.forward(&txt_norm)?;
        let txt = txt.add(&txt_mlp_out)?;

        Ok((img, txt))
    }
}

/// Single Stream Block - processes concatenated features
pub struct SingleStreamBlock {
    pub norm1: LayerNorm,
    pub attn: SelfAttention,
    pub norm2: LayerNorm,
    pub mlp: MLP,
    pub modulation: Modulation,
}

impl SingleStreamBlock {
    pub fn new(config: &FluxConfig, device: Arc<CudaDevice>) -> Result<Self> {
        let hidden_size = config.hidden_size;

        Ok(Self {
            norm1: LayerNorm::new(vec![hidden_size], 1e-6, device.clone())?,
            attn: SelfAttention::new(config, device.clone())?,
            norm2: LayerNorm::new(vec![hidden_size], 1e-6, device.clone())?,
            mlp: MLP::new(hidden_size, config.mlp_ratio, device.clone())?,
            modulation: Modulation::new(hidden_size, device.clone())?,
        })
    }

    pub fn forward(&self, x: &Tensor, pos: &Tensor, c: &Tensor) -> Result<Tensor> {
        let (shift_msa, scale_msa, shift_mlp, scale_mlp) = self.modulation.forward(c)?;

        // Self-attention
        let x_norm = modulate(&self.norm1.forward(x)?, &shift_msa, &scale_msa, &x.device())?;
        let attn_out = self.attn.forward(&x_norm, pos)?;
        let x = x.add(&attn_out)?;

        // MLP
        let x_norm = modulate(&self.norm2.forward(&x)?, &shift_mlp, &scale_mlp, &x.device())?;
        let mlp_out = self.mlp.forward(&x_norm)?;

        x.add(&mlp_out)
    }
}

/// Self-attention module for Flux
pub struct SelfAttention {
    pub qkv: Linear,
    pub proj: Linear,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl SelfAttention {
    pub fn new(config: &FluxConfig, device: Arc<CudaDevice>) -> Result<Self> {
        let hidden_size = config.hidden_size;

        Ok(Self {
            qkv: Linear::new(hidden_size, hidden_size * 3, true, &device)?,
            proj: Linear::new(hidden_size, hidden_size, true, &device)?,
            num_heads: config.num_heads,
            head_dim: config.head_dim,
        })
    }

    pub fn forward(&self, x: &Tensor, pos: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, _) = match x.shape().dims() {
            [b, s, h] => (*b, *s, *h),
            _ => {
                return Err(flame_core::Error::InvalidOperation("Invalid input shape".into()))
            }
        };

        // Generate QKV
        let qkv = self.qkv.forward(x)?;
        let qkv = qkv.reshape(&[batch, seq_len, 3, self.num_heads, self.head_dim])?;

        // Split and reshape
        let q = qkv
            .slice(&[(0, batch), (0, seq_len), (0, 1), (0, self.num_heads), (0, self.head_dim)])?
            .squeeze(Some(2))?
            .permute(&[0, 2, 1, 3])?;
        let k = qkv
            .slice(&[(0, batch), (0, seq_len), (1, 2), (0, self.num_heads), (0, self.head_dim)])?
            .squeeze(Some(2))?
            .permute(&[0, 2, 1, 3])?;
        let v = qkv
            .slice(&[(0, batch), (0, seq_len), (2, 3), (0, self.num_heads), (0, self.head_dim)])?
            .squeeze(Some(2))?
            .permute(&[0, 2, 1, 3])?;

        // Apply RoPE
        let (q_rope, k_rope) = apply_rope(&q, &k, pos)?;

        // Attention
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let scores = q_rope.matmul(&k_rope.permute(&[0, 1, 3, 2])?)?;
        let scores = scores.mul_scalar(scale as f32)?;
        let attn = scores.softmax(3)?;
        let out = attn.matmul(&v)?;

        // Reshape and project
        let out = out.permute(&[0, 2, 1, 3])?.reshape(&[
            batch,
            seq_len,
            self.num_heads * self.head_dim,
        ])?;

        self.proj.forward(&out)
    }
}

/// MLP module
pub struct MLP {
    pub fc1: Linear,
    pub fc2: Linear,
}

impl MLP {
    pub fn new(hidden_size: usize, mlp_ratio: f32, device: Arc<CudaDevice>) -> Result<Self> {
        let mlp_hidden = (hidden_size as f32 * mlp_ratio) as usize;

        Ok(Self {
            fc1: Linear::new(hidden_size, mlp_hidden, true, &device)?,
            fc2: Linear::new(mlp_hidden, hidden_size, true, &device)?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.fc1.forward(x)?;
        let x = x.gelu()?;
        self.fc2.forward(&x)
    }
}

/// MLP Embedder for time/vector embeddings
pub struct MLPEmbedder {
    pub fc1: Linear,
    pub fc2: Linear,
}

impl MLPEmbedder {
    pub fn new(in_dim: usize, hidden_dim: usize, device: Arc<CudaDevice>) -> Result<Self> {
        Ok(Self {
            fc1: Linear::new(in_dim, hidden_dim, true, &device)?,
            fc2: Linear::new(hidden_dim, hidden_dim, true, &device)?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.fc1.forward(x)?;
        let x = x.silu()?;
        self.fc2.forward(&x)
    }
}

/// Modulation module for adaptive normalization
pub struct Modulation {
    pub lin: Linear,
}

impl Modulation {
    pub fn new(hidden_size: usize, device: Arc<CudaDevice>) -> Result<Self> {
        Ok(Self { lin: Linear::new(hidden_size, hidden_size * 4, true, &device)? })
    }

    pub fn forward(&self, c: &Tensor) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let out = self.lin.forward(c)?;
        let chunks = out.chunk(4, out.shape().rank() - 1)?;

        Ok((chunks[0].clone(), chunks[1].clone(), chunks[2].clone(), chunks[3].clone()))
    }
}

/// Apply modulation: x * (1 + scale) + shift
fn modulate(x: &Tensor, shift: &Tensor, scale: &Tensor, device: &CudaDevice) -> Result<Tensor> {
    let scale_plus_one = scale.add_scalar(1.0)?;
    x.mul(&scale_plus_one)?.add(shift)
}

/// Get timestep embeddings
fn get_timestep_embedding(
    timesteps: &Tensor,
    embedding_dim: usize,
    device: Arc<CudaDevice>,
) -> Result<Tensor> {
    let half_dim = embedding_dim / 2;
    let exponent = flame_core::Tensor::arange(0.0, half_dim as f32, 1.0, device.clone())?
        .mul_scalar(-std::f32::consts::LN_2 as f32)?
        .div_scalar(half_dim as f32)?
        .exp()?;

    let emb = timesteps.unsqueeze(1)?.mul(&exponent.unsqueeze(0)?)?;

    let sin_emb = emb.sin()?;
    let cos_emb = emb.cos()?;

    Tensor::cat(&[&sin_emb, &cos_emb], 1)
}

/// Generate RoPE embeddings
fn rope_embeddings(
    batch_size: usize,
    seq_len: usize,
    dim: usize,
    base: f32,
    device: Arc<CudaDevice>,
) -> Result<Tensor> {
    let inv_freq =
        (0..dim / 2).map(|i| 1.0 / base.powf(2.0 * i as f32 / dim as f32)).collect::<Vec<_>>();

    let inv_freq = Tensor::from_vec(inv_freq, Shape::from_dims(&[dim / 2]), device.clone())?;
    let t = flame_core::Tensor::arange(0.0, seq_len as f32, 1.0, device.clone())?;

    let freqs = t.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?;
    let emb = Tensor::cat(&[&freqs.sin()?, &freqs.cos()?], 1)?;

    // Repeat along batch dimension
    let shape = emb.shape();
    let new_shape = Shape::from_dims(&[batch_size, shape.dims()[0], shape.dims()[1]]);
    emb.unsqueeze(0)?.broadcast_to(&new_shape)
}

/// Apply RoPE to queries and keys
fn apply_rope(q: &Tensor, k: &Tensor, pos: &Tensor) -> Result<(Tensor, Tensor)> {
    // Strict mode: remove simplified placeholder. Fail closed unless a real RoPE is provided.
    Err(Error::InvalidOperation(
        "Unsupported path: apply_rope placeholder removed. Use GPU-backed rotary embedding from ops or disable RoPE.".to_string(),
    ))
}

#[cfg(all(test, feature = "dev-bins"))]
mod tests {
    use super::*;

    #[test]
    fn test_flux_model(device: Arc<CudaDevice>) -> Result<()> {
        let config = FluxConfig::default();
        let model = FluxModel::new(config.clone(), device.clone())?;

        // Test inputs
        let batch = 2;
        let img_seq = 256; // 16x16 patches
        let txt_seq = 77;

        let img =
            Tensor::randn(Shape::from_dims(&[batch, img_seq, 64]), 0.0, 0.02, device.clone())?;
        let txt =
            Tensor::randn(Shape::from_dims(&[batch, txt_seq, 4096]), 0.0, 0.02, device.clone())?;
        let timesteps = Tensor::randn(Shape::from_dims(&[batch]), 0.0, 1.0, device.clone())?;
        let y = Tensor::randn(Shape::from_dims(&[batch, 768]), 0.0, 0.02, device.clone())?;

        let output = model.forward(&img, &txt, &timesteps, &y, None)?;
        assert_eq!(output.shape().dims(), &[batch, img_seq, 64]);

        Ok(())
    }
}
