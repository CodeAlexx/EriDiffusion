//! Positional embeddings for Flux model including RoPE and 2D sinusoidal embeddings

use candle_core::{Device, DType, Result, Tensor, D};
use std::f32::consts::PI;

/// Rotary Position Embedding (RoPE) for Flux
pub struct RotaryEmbedding {
    cos_cached: Tensor,
    sin_cached: Tensor,
    dim: usize,
    max_seq_len: usize,
    base: f32,
}

impl RotaryEmbedding {
    pub fn new(
        dim: usize,
        max_seq_len: usize,
        base: f32,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let inv_freq = Self::compute_inv_freq(dim, base, device)?;
        let (cos_cached, sin_cached) = Self::compute_cache(
            max_seq_len,
            &inv_freq,
            device,
            dtype,
        )?;
        
        Ok(Self {
            cos_cached,
            sin_cached,
            dim,
            max_seq_len,
            base,
        })
    }
    
    fn compute_inv_freq(dim: usize, base: f32, _device: &Device) -> Result<Tensor> {
        // Always use cached device
        let device = crate::trainers::cached_device::get_single_device()
            .map_err(|e| candle_core::Error::Msg(format!("Failed to get cached device: {}", e)))?;
        
        let half_dim = dim / 2;
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / base.powf(i as f32 / half_dim as f32))
            .collect();
        
        Tensor::from_vec(inv_freq, half_dim, &device)
    }
    
    fn compute_cache(
        max_seq_len: usize,
        inv_freq: &Tensor,
        _device: &Device,
        dtype: DType,
    ) -> Result<(Tensor, Tensor)> {
        // Always use cached device
        let device = crate::trainers::cached_device::get_single_device()
            .map_err(|e| candle_core::Error::Msg(format!("Failed to get cached device: {}", e)))?;
        
        let t = Tensor::arange(0, max_seq_len as i64, &device)?
            .to_dtype(DType::F32)?;
        
        // [seq_len, dim/2]
        let freqs = t.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?;
        
        // [seq_len, dim]
        let freqs = Tensor::cat(&[&freqs, &freqs], 1)?;
        
        let cos = freqs.cos()?.to_dtype(dtype)?;
        let sin = freqs.sin()?.to_dtype(dtype)?;
        
        Ok((cos, sin))
    }
    
    pub fn apply_rope(
        &self,
        q: &Tensor,
        k: &Tensor,
        seq_len: usize,
    ) -> Result<(Tensor, Tensor)> {
        let cos = self.cos_cached.narrow(0, 0, seq_len)?;
        let sin = self.sin_cached.narrow(0, 0, seq_len)?;
        
        let q_rot = Self::rotate_embeddings(q, &cos, &sin)?;
        let k_rot = Self::rotate_embeddings(k, &cos, &sin)?;
        
        Ok((q_rot, k_rot))
    }
    
    fn rotate_embeddings(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        // x: [batch, seq_len, n_heads, head_dim] or [batch, n_heads, seq_len, head_dim]
        let ndim = x.dims().len();
        let seq_dim = if ndim == 4 { 2 } else { 1 };
        
        // Split into two halves
        let half_dim = x.dim(D::Minus1)? / 2;
        let x1 = x.narrow(D::Minus1, 0, half_dim)?;
        let x2 = x.narrow(D::Minus1, half_dim, half_dim)?;
        
        // Rotate
        let cos = cos.unsqueeze(1)?; // Add head dimension
        let sin = sin.unsqueeze(1)?;
        
        let rotated = Tensor::cat(&[
            (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?,
            (x1.broadcast_mul(&sin)? + x2.broadcast_mul(&cos)?)?
        ], D::Minus1)?;
        
        Ok(rotated)
    }
}

/// 2D Sinusoidal Position Embeddings for image patches
pub struct SinusoidalPosEmbed2D {
    embed_dim: usize,
    temperature: f32,
}

impl SinusoidalPosEmbed2D {
    pub fn new(embed_dim: usize, temperature: f32) -> Self {
        Self { embed_dim, temperature }
    }
    
    pub fn forward(
        &self,
        h: usize,
        w: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Tensor> {
        let num_pos_feats = self.embed_dim / 2;
        
        // Create 2D grid
        let y_embed = Tensor::arange(0, h as i64, device)?
            .to_dtype(DType::F32)?
            .unsqueeze(1)?
            .repeat(&[1, w])?;
        
        let x_embed = Tensor::arange(0, w as i64, device)?
            .to_dtype(DType::F32)?
            .unsqueeze(0)?
            .repeat(&[h, 1])?;
        
        // Normalize to [0, 1]
        let eps = 1e-6;
        let y_embed = (&y_embed / (h as f64 - 1.0 + eps))?;
        let x_embed = (&x_embed / (w as f64 - 1.0 + eps))?;
        
        // Create sin/cos features
        let dim_t = Tensor::arange(0, num_pos_feats as i64, device)?
            .to_dtype(DType::F32)?;
        let dim_t = ((2.0 * &dim_t)? / num_pos_feats as f64)?;
        let dim_t = (self.temperature as f64 * dim_t.exp()?)?;
        
        // [H, W, num_pos_feats]
        let pos_x = x_embed.unsqueeze(2)?.broadcast_mul(&dim_t.unsqueeze(0)?.unsqueeze(0)?)?;
        let pos_y = y_embed.unsqueeze(2)?.broadcast_mul(&dim_t.unsqueeze(0)?.unsqueeze(0)?)?;
        
        // Apply sin and cos
        let pos_x_sin = pos_x.sin()?;
        let pos_x_cos = pos_x.cos()?;
        let pos_y_sin = pos_y.sin()?;
        let pos_y_cos = pos_y.cos()?;
        
        // Concatenate all features [H, W, embed_dim]
        let pos = Tensor::cat(&[pos_y_sin, pos_y_cos, pos_x_sin, pos_x_cos], 2)?;
        
        // Flatten to [H*W, embed_dim]
        pos.reshape(&[h * w, self.embed_dim])?
            .to_dtype(dtype)
    }
}

/// Learnable position embeddings
pub struct LearnedPosEmbed {
    embeddings: Tensor,
}

impl LearnedPosEmbed {
    pub fn new(
        num_patches: usize,
        embed_dim: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let embeddings = Tensor::randn(
            0f32,
            0.02f32,
            &[num_patches, embed_dim],
            device,
        )?.to_dtype(dtype)?;
        
        Ok(Self { embeddings })
    }
    
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [batch, seq_len, dim]
        let seq_len = x.dim(1)?;
        let pos_embed = self.embeddings.narrow(0, 0, seq_len)?;
        x.broadcast_add(&pos_embed.unsqueeze(0)?)
    }
}

/// Flux-specific positional embedding that combines 2D position info
pub struct FluxPositionalEmbedding {
    rope: Option<RotaryEmbedding>,
    pos_embed_2d: SinusoidalPosEmbed2D,
    use_rope: bool,
}

impl FluxPositionalEmbedding {
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        max_seq_len: usize,
        use_rope: bool,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let head_dim = hidden_size / num_heads;
        
        let rope = if use_rope {
            Some(RotaryEmbedding::new(
                head_dim,
                max_seq_len,
                10_000.0,
                device,
                dtype,
            )?)
        } else {
            None
        };
        
        let pos_embed_2d = SinusoidalPosEmbed2D::new(hidden_size, 10_000.0);
        
        Ok(Self {
            rope,
            pos_embed_2d,
            use_rope,
        })
    }
    
    pub fn get_2d_embeddings(
        &self,
        h: usize,
        w: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Tensor> {
        self.pos_embed_2d.forward(h, w, device, dtype)
    }
    
    pub fn apply_rope_if_enabled(
        &self,
        q: &Tensor,
        k: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        if let Some(rope) = &self.rope {
            let seq_len = q.dim(2)?; // Assuming [batch, heads, seq_len, head_dim]
            rope.apply_rope(q, k, seq_len)
        } else {
            Ok((q.clone(), k.clone()))
        }
    }
}