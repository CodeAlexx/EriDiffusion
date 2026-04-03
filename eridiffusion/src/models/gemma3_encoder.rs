//! Gemma 3 12B text encoder for LTX-2 inference.
//!
//! Pipeline:
//! 1. Tokenize text with Gemma 3 tokenizer (external)
//! 2. Forward through all 48 transformer layers (collect ALL hidden states)
//! 3. Stack 49 hidden states (embed + 48 layers) per token -> 49*3840 = 188160
//! 4. Project to 3840 via text_embedding_projection
//! 5. Normalize: 8.0 * (x - mean) / (max - min + eps)
//! 6. Video connector processes through 2 transformer blocks with RoPE + registers
//! 7. Output: (B, seq_len, 3840) ready for DiT caption_projection (3840 -> 4096)
//!
//! Architecture notes:
//! - Gemma 3 uses RMSNorm with additive weight: x * (1 + w) instead of x * w
//! - QK norms on attention (multiplicative, not additive)
//! - Sliding window (local) vs global attention alternating by layer
//! - Global layers use rope_theta=1_000_000, local layers use rope_theta=10_000
//! - GQA: 16 query heads, 8 key/value heads, head_dim=256
//! - MLP uses gelu(tanh approx) gating: down(gelu(gate(x)) * up(x))

use crate::ops::{Linear, RMSNorm};
use flame_core::{DType, Error, Result, Shape, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Gemma 3 12B Config
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Gemma3Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta_global: f64,
    pub rope_theta_local: f64,
    pub head_dim: usize,
    /// Sliding window pattern: Some(window_size) for local, None for global.
    /// Pattern repeats: [1024, 1024, 1024, 1024, 1024, None] (5 local, 1 global).
    pub sliding_window_pattern: Vec<Option<usize>>,
}

impl Default for Gemma3Config {
    fn default() -> Self {
        Self {
            vocab_size: 262208,
            hidden_size: 3840,
            intermediate_size: 15360,
            num_hidden_layers: 48,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            max_position_embeddings: 131072,
            rms_norm_eps: 1e-6,
            rope_theta_global: 1_000_000.0,
            rope_theta_local: 10_000.0,
            head_dim: 256,
            sliding_window_pattern: vec![
                Some(1024),
                Some(1024),
                Some(1024),
                Some(1024),
                Some(1024),
                None,
            ],
        }
    }
}

impl Gemma3Config {
    /// Whether layer `idx` uses global (full) attention.
    pub fn is_global_layer(&self, idx: usize) -> bool {
        let pattern_idx = idx % self.sliding_window_pattern.len();
        self.sliding_window_pattern[pattern_idx].is_none()
    }

    /// RoPE theta for the given layer index.
    pub fn rope_theta_for_layer(&self, idx: usize) -> f64 {
        if self.is_global_layer(idx) {
            self.rope_theta_global
        } else {
            self.rope_theta_local
        }
    }
}

// ---------------------------------------------------------------------------
// RoPE cache builder
// ---------------------------------------------------------------------------

/// Build cos/sin cache for RoPE. Returns (cos, sin) each of shape (seq_len, head_dim/2).
fn build_rope_cache(
    seq_len: usize,
    head_dim: usize,
    theta: f64,
    device: &Arc<flame_core::CudaDevice>,
) -> Result<(Tensor, Tensor)> {
    let half_dim = head_dim / 2;
    let mut cos_vals = vec![0.0f32; seq_len * half_dim];
    let mut sin_vals = vec![0.0f32; seq_len * half_dim];

    for pos in 0..seq_len {
        for i in 0..half_dim {
            let freq = 1.0 / theta.powf(2.0 * i as f64 / head_dim as f64);
            let angle = pos as f64 * freq;
            let idx = pos * half_dim + i;
            cos_vals[idx] = angle.cos() as f32;
            sin_vals[idx] = angle.sin() as f32;
        }
    }

    let shape = Shape::from_dims(&[seq_len, half_dim]);
    let cos = Tensor::from_vec_dtype(cos_vals, shape.clone(), device.clone(), DType::BF16)?;
    let sin = Tensor::from_vec_dtype(sin_vals, shape, device.clone(), DType::BF16)?;
    Ok((cos, sin))
}

/// Apply RoPE to tensor x of shape (B, H, T, D).
/// cos/sin are (seq_len, D/2) -- we slice to T and broadcast.
fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    let t = dims[2];
    let d = dims[3];
    let half_d = d / 2;

    // Slice cos/sin to (T, half_d) then reshape to (1, 1, T, half_d) for broadcast
    let cos_t = cos.narrow(0, 0, t)?.reshape(&[1, 1, t, half_d])?;
    let sin_t = sin.narrow(0, 0, t)?.reshape(&[1, 1, t, half_d])?;

    // Split x into first half and second half along last dim
    let x1 = x.narrow(3, 0, half_d)?;
    let x2 = x.narrow(3, half_d, half_d)?;

    // RoPE: [x1*cos - x2*sin, x2*cos + x1*sin]
    let r1 = x1.mul(&cos_t)?.sub(&x2.mul(&sin_t)?)?;
    let r2 = x2.mul(&cos_t)?.add(&x1.mul(&sin_t)?)?;

    // Concatenate along last dim
    Tensor::cat(&[&r1, &r2], 3)
}

// ---------------------------------------------------------------------------
// Gemma 3 Additive RMSNorm
// ---------------------------------------------------------------------------

/// Gemma 3 uses an additive RMSNorm variant: output = x * rsqrt(mean(x^2) + eps) * (1 + weight).
/// Standard flame_core RMSNorm uses multiplicative weight (x * w), so we wrap it
/// and pre-add 1.0 to the weight at load time for the additive layers.
///
/// For QK norms (non-additive), we use the standard RMSNorm directly.
pub struct Gemma3RMSNorm {
    inner: RMSNorm,
}

impl Gemma3RMSNorm {
    /// Create a new additive RMSNorm. Weight will be loaded separately.
    pub fn new(
        dim: usize,
        eps: f32,
        device: &Arc<flame_core::CudaDevice>,
    ) -> Result<Self> {
        let inner = RMSNorm::new(vec![dim], eps, true, device.clone())?;
        Ok(Self { inner })
    }

    /// Load weight and apply the additive offset: effective_weight = 1.0 + raw_weight.
    pub fn load_weight_additive(&mut self, raw_weight: &Tensor) -> Result<()> {
        let one_plus_w = raw_weight.add_scalar(1.0)?;
        self.inner.copy_weight_from(&one_plus_w)
    }

    /// Load weight directly (for non-additive norms like QK norms).
    pub fn load_weight_multiplicative(&mut self, weight: &Tensor) -> Result<()> {
        self.inner.copy_weight_from(weight)
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.inner.forward(x)
    }
}

// ---------------------------------------------------------------------------
// GQA repeat-interleave helper
// ---------------------------------------------------------------------------

/// Repeat-interleave along dim 1 for GQA expansion.
/// Input (B, H_kv, T, D) -> Output (B, H_kv * repeats, T, D)
/// Each head is duplicated `repeats` times in order: [h0, h0, h1, h1, ...]
fn repeat_interleave_dim1(x: &Tensor, repeats: usize) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    let (b, h, t, d) = (dims[0], dims[1], dims[2], dims[3]);
    // (B, H, T, D) -> (B, H, 1, T, D) -> expand -> (B, H, repeats, T, D) -> (B, H*repeats, T, D)
    let x = x.reshape(&[b, h, 1, t, d])?;
    let x = x.expand(&[b, h, repeats, t, d])?;
    x.reshape(&[b, h * repeats, t, d])
}

// ---------------------------------------------------------------------------
// Gemma 3 Attention
// ---------------------------------------------------------------------------

pub struct Gemma3Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: Gemma3RMSNorm,
    k_norm: Gemma3RMSNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    is_global: bool,
}

impl Gemma3Attention {
    pub fn new(
        config: &Gemma3Config,
        layer_idx: usize,
        device: &Arc<flame_core::CudaDevice>,
    ) -> Result<Self> {
        let h = config.hidden_size;
        let qd = config.num_attention_heads * config.head_dim;
        let kvd = config.num_key_value_heads * config.head_dim;

        let q_proj = Linear::new_zeroed(h, qd, false, device)?;
        let k_proj = Linear::new_zeroed(h, kvd, false, device)?;
        let v_proj = Linear::new_zeroed(h, kvd, false, device)?;
        let o_proj = Linear::new_zeroed(qd, h, false, device)?;

        // QK norms use multiplicative (not additive) RMSNorm
        let q_norm = Gemma3RMSNorm::new(config.head_dim, config.rms_norm_eps, device)?;
        let k_norm = Gemma3RMSNorm::new(config.head_dim, config.rms_norm_eps, device)?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            is_global: config.is_global_layer(layer_idx),
        })
    }

    /// Forward pass. cos/sin are pre-sliced for the correct theta (global or local).
    pub fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        _attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let (b, t) = (dims[0], dims[1]);

        // Project Q, K, V
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape to (B, T, num_heads, head_dim) then transpose to (B, num_heads, T, head_dim)
        let q = q
            .reshape(&[b, t, self.num_heads, self.head_dim])?
            .transpose_dims(1, 2)?;
        let k = k
            .reshape(&[b, t, self.num_kv_heads, self.head_dim])?
            .transpose_dims(1, 2)?;
        let v = v
            .reshape(&[b, t, self.num_kv_heads, self.head_dim])?
            .transpose_dims(1, 2)?;

        // QK norm (applied per-head, so shape is fine as (B, H, T, D))
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        // Apply RoPE
        let q = apply_rope(&q, cos, sin)?;
        let k = apply_rope(&k, cos, sin)?;

        // GQA: repeat-interleave K,V heads to match Q count.
        // repeat_interleave(dim=1) duplicates each head in-place:
        //   [h0, h1, ..., h7] -> [h0, h0, h1, h1, ..., h7, h7]
        // We achieve this via reshape + expand + reshape:
        //   (B, kv_heads, T, D) -> (B, kv_heads, 1, T, D) -> expand to (B, kv_heads, repeat, T, D)
        //   -> reshape (B, kv_heads*repeat, T, D)
        let (k, v) = if self.num_kv_heads < self.num_heads {
            let repeat = self.num_heads / self.num_kv_heads;
            let k = repeat_interleave_dim1(&k, repeat)?;
            let v = repeat_interleave_dim1(&v, repeat)?;
            (k, v)
        } else {
            (k, v)
        };

        // Scaled dot-product attention via flame_core sdpa
        let out = flame_core::sdpa::forward(&q, &k, &v, _attention_mask)?;

        // Reshape back: (B, H, T, D) -> (B, T, H*D)
        let out = out
            .transpose_dims(1, 2)?
            .reshape(&[b, t, self.num_heads * self.head_dim])?;

        self.o_proj.forward(&out)
    }
}

// ---------------------------------------------------------------------------
// Gemma 3 MLP (gated GeGLU with tanh approx)
// ---------------------------------------------------------------------------

pub struct Gemma3MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Gemma3MLP {
    pub fn new(
        config: &Gemma3Config,
        device: &Arc<flame_core::CudaDevice>,
    ) -> Result<Self> {
        let h = config.hidden_size;
        let ff = config.intermediate_size;
        Ok(Self {
            gate_proj: Linear::new_zeroed(h, ff, false, device)?,
            up_proj: Linear::new_zeroed(h, ff, false, device)?,
            down_proj: Linear::new_zeroed(ff, h, false, device)?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // gelu(gate(x)) * up(x) -> down
        let gate = self.gate_proj.forward(x)?.gelu()?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&gate.mul(&up)?)
    }
}

// ---------------------------------------------------------------------------
// Gemma 3 Transformer Layer
// ---------------------------------------------------------------------------

pub struct Gemma3Layer {
    self_attn: Gemma3Attention,
    mlp: Gemma3MLP,
    input_layernorm: Gemma3RMSNorm,
    post_attention_layernorm: Gemma3RMSNorm,
    pre_feedforward_layernorm: Gemma3RMSNorm,
    post_feedforward_layernorm: Gemma3RMSNorm,
    is_global: bool,
}

impl Gemma3Layer {
    pub fn new(
        config: &Gemma3Config,
        layer_idx: usize,
        device: &Arc<flame_core::CudaDevice>,
    ) -> Result<Self> {
        Ok(Self {
            self_attn: Gemma3Attention::new(config, layer_idx, device)?,
            mlp: Gemma3MLP::new(config, device)?,
            input_layernorm: Gemma3RMSNorm::new(config.hidden_size, config.rms_norm_eps, device)?,
            post_attention_layernorm: Gemma3RMSNorm::new(
                config.hidden_size,
                config.rms_norm_eps,
                device,
            )?,
            pre_feedforward_layernorm: Gemma3RMSNorm::new(
                config.hidden_size,
                config.rms_norm_eps,
                device,
            )?,
            post_feedforward_layernorm: Gemma3RMSNorm::new(
                config.hidden_size,
                config.rms_norm_eps,
                device,
            )?,
            is_global: config.is_global_layer(layer_idx),
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Pre-norm self-attention with post-norm
        let h = self.input_layernorm.forward(x)?;
        let h = self.self_attn.forward(&h, cos, sin, attention_mask)?;
        let h = self.post_attention_layernorm.forward(&h)?;
        let x = x.add(&h)?;

        // Pre-norm FFN with post-norm
        let h = self.pre_feedforward_layernorm.forward(&x)?;
        let h = self.mlp.forward(&h)?;
        let h = self.post_feedforward_layernorm.forward(&h)?;
        x.add(&h)
    }

    /// Whether this layer uses global attention.
    pub fn is_global(&self) -> bool {
        self.is_global
    }
}

// ---------------------------------------------------------------------------
// Gemma 3 Model (language model backbone)
// ---------------------------------------------------------------------------

pub struct Gemma3Model {
    pub config: Gemma3Config,
    embed_tokens: flame_core::embedding::Embedding,
    layers: Vec<Gemma3Layer>,
    norm: Gemma3RMSNorm,
}

impl Gemma3Model {
    /// Create an uninitialized model (weights must be loaded via `load_weights`).
    pub fn new(
        config: Gemma3Config,
        device: &Arc<flame_core::CudaDevice>,
    ) -> Result<Self> {
        let embed_tokens = flame_core::embedding::Embedding::new(
            config.vocab_size,
            config.hidden_size,
            device.clone(),
        )?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(Gemma3Layer::new(&config, i, device)?);
        }

        let norm = Gemma3RMSNorm::new(config.hidden_size, config.rms_norm_eps, device)?;

        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
        })
    }

    /// Forward pass returning ALL hidden states (embed output + each layer output).
    ///
    /// Returns a Vec of 49 tensors, each (B, T, 3840). This is required by LTX-2
    /// which concatenates all hidden states for its text projection.
    pub fn forward(&self, input_ids: &Tensor, _attention_mask: Option<&Tensor>) -> Result<Vec<Tensor>> {
        let dims = input_ids.shape().dims().to_vec();
        let t = dims[1];
        let device = input_ids.device();

        // Build RoPE caches for both global and local theta
        let (cos_global, sin_global) =
            build_rope_cache(t, self.config.head_dim, self.config.rope_theta_global, device)?;
        let (cos_local, sin_local) =
            build_rope_cache(t, self.config.head_dim, self.config.rope_theta_local, device)?;

        // TODO: Build causal + padding attention mask from _attention_mask if provided.
        // For inference with single prompts this is typically not needed (SDPA handles causal).
        let causal_mask: Option<&Tensor> = None;

        // Embed tokens and scale (Gemma 3 multiplies embeddings by sqrt(hidden_size))
        let mut x = self.embed_tokens.forward(input_ids)?;
        let scale = (self.config.hidden_size as f32).sqrt();
        x = x.mul_scalar(scale)?;

        let mut all_hidden_states = Vec::with_capacity(self.config.num_hidden_layers + 1);
        all_hidden_states.push(x.clone());

        for layer in &self.layers {
            let (cos, sin) = if layer.is_global() {
                (&cos_global, &sin_global)
            } else {
                (&cos_local, &sin_local)
            };
            x = layer.forward(&x, cos, sin, causal_mask)?;
            all_hidden_states.push(x.clone());
        }

        // Apply final norm to last hidden state only
        let last_idx = all_hidden_states.len() - 1;
        all_hidden_states[last_idx] = self.norm.forward(&all_hidden_states[last_idx])?;

        Ok(all_hidden_states)
    }

    /// Load Gemma 3 weights from a state dict (keys should have `model.` prefix stripped).
    ///
    /// Expected key format (after stripping `model.` prefix):
    /// - `embed_tokens.weight`
    /// - `layers.{i}.self_attn.{q,k,v,o}_proj.weight`
    /// - `layers.{i}.self_attn.{q,k}_norm.weight`
    /// - `layers.{i}.mlp.{gate,up,down}_proj.weight`
    /// - `layers.{i}.input_layernorm.weight`
    /// - `layers.{i}.post_attention_layernorm.weight`
    /// - `layers.{i}.pre_feedforward_layernorm.weight`
    /// - `layers.{i}.post_feedforward_layernorm.weight`
    /// - `norm.weight`
    pub fn load_weights(&mut self, weights: &HashMap<String, Tensor>) -> Result<()> {
        // Helper to get a weight, stripping `model.` prefix if present
        let get = |key: &str| -> Result<&Tensor> {
            weights
                .get(key)
                .or_else(|| weights.get(&format!("model.{key}")))
                .ok_or_else(|| Error::InvalidOperation(format!("Missing weight: {key}")))
        };

        // Embedding
        let embed_w = get("embed_tokens.weight")?;
        // Embedding weight is stored as [vocab, dim] -- copy directly
        // flame_core Embedding stores weight as the lookup table
        let embed_bf16 = if embed_w.dtype() != DType::BF16 {
            embed_w.to_dtype(DType::BF16)?
        } else {
            embed_w.clone()
        };
        self.embed_tokens.weight = embed_bf16;

        // Per-layer weights
        for i in 0..self.config.num_hidden_layers {
            let prefix = format!("layers.{i}");
            let layer = &mut self.layers[i];

            // Attention projections
            layer
                .self_attn
                .q_proj
                .copy_weight_from(&get(&format!("{prefix}.self_attn.q_proj.weight"))?.to_dtype(DType::BF16)?)?;
            layer
                .self_attn
                .k_proj
                .copy_weight_from(&get(&format!("{prefix}.self_attn.k_proj.weight"))?.to_dtype(DType::BF16)?)?;
            layer
                .self_attn
                .v_proj
                .copy_weight_from(&get(&format!("{prefix}.self_attn.v_proj.weight"))?.to_dtype(DType::BF16)?)?;
            layer
                .self_attn
                .o_proj
                .copy_weight_from(&get(&format!("{prefix}.self_attn.o_proj.weight"))?.to_dtype(DType::BF16)?)?;

            // QK norms (multiplicative, NOT additive)
            layer.self_attn.q_norm.load_weight_multiplicative(
                &get(&format!("{prefix}.self_attn.q_norm.weight"))?.to_dtype(DType::BF16)?,
            )?;
            layer.self_attn.k_norm.load_weight_multiplicative(
                &get(&format!("{prefix}.self_attn.k_norm.weight"))?.to_dtype(DType::BF16)?,
            )?;

            // MLP
            layer
                .mlp
                .gate_proj
                .copy_weight_from(&get(&format!("{prefix}.mlp.gate_proj.weight"))?.to_dtype(DType::BF16)?)?;
            layer
                .mlp
                .up_proj
                .copy_weight_from(&get(&format!("{prefix}.mlp.up_proj.weight"))?.to_dtype(DType::BF16)?)?;
            layer
                .mlp
                .down_proj
                .copy_weight_from(&get(&format!("{prefix}.mlp.down_proj.weight"))?.to_dtype(DType::BF16)?)?;

            // Layer norms (additive: effective_weight = 1 + raw_weight)
            layer.input_layernorm.load_weight_additive(
                &get(&format!("{prefix}.input_layernorm.weight"))?.to_dtype(DType::BF16)?,
            )?;
            layer.post_attention_layernorm.load_weight_additive(
                &get(&format!("{prefix}.post_attention_layernorm.weight"))?.to_dtype(DType::BF16)?,
            )?;
            layer.pre_feedforward_layernorm.load_weight_additive(
                &get(&format!("{prefix}.pre_feedforward_layernorm.weight"))?.to_dtype(DType::BF16)?,
            )?;
            layer.post_feedforward_layernorm.load_weight_additive(
                &get(&format!("{prefix}.post_feedforward_layernorm.weight"))?.to_dtype(DType::BF16)?,
            )?;
        }

        // Final norm (additive)
        self.norm
            .load_weight_additive(&get("norm.weight")?.to_dtype(DType::BF16)?)?;

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// LTX-2 Hidden State Packing + Normalization
// ---------------------------------------------------------------------------

/// Pack all 49 hidden states and normalize for LTX-2 text projection.
///
/// Input: 49 tensors each (B, T, 3840)
/// Output: (B, T, 188160) normalized and ready for text_embedding_projection
///
/// Normalization (per-batch, padding-aware):
///   out = 8.0 * (x - mean) / (max - min + eps)
pub fn pack_hidden_states_for_ltx2(
    all_hidden: &[Tensor],
    attention_mask: Option<&Tensor>,
) -> Result<Tensor> {
    if all_hidden.len() != 49 {
        return Err(Error::InvalidOperation(format!(
            "LTX-2 requires exactly 49 hidden states, got {}",
            all_hidden.len()
        )));
    }

    let dims = all_hidden[0].shape().dims().to_vec();
    let (b, t, d) = (dims[0], dims[1], dims[2]); // d = 3840

    // Stack: list of (B, T, 3840) -> (B, 49, T, 3840) via stack on axis=1
    let stacked = Tensor::stack(all_hidden, 1)?; // (B, 49, T, 3840)

    // Rearrange to (B, T, 3840, 49) for per-token packing
    // stack gave (B, 49, T, 3840), we need to transpose dims 1 and 2, then 2 and 3
    // Actually we want (B, T, 3840, 49):
    //   (B, 49, T, 3840) -> transpose(1,2) -> (B, T, 49, 3840) -> transpose(2,3) -> (B, T, 3840, 49)
    let out = stacked.transpose_dims(1, 2)?.transpose_dims(2, 3)?;
    // out shape: (B, T, 3840, 49)

    // Normalize
    let eps = 1e-6f32;
    let out = if let Some(_mask) = attention_mask {
        // TODO: Implement padding-aware normalization.
        // For inference with single prompts (no padding), the unmasked path is sufficient.
        // Full implementation would:
        //   1. Compute masked mean over valid positions only
        //   2. Compute masked min/max over valid positions only
        //   3. Apply 8.0 * (x - mean) / (range + eps)
        //   4. Zero out padded positions
        log::warn!(
            "Gemma3 pack_hidden_states: padding-aware normalization not yet implemented, \
             falling back to unmasked normalization"
        );
        normalize_unmasked(&out, eps)?
    } else {
        normalize_unmasked(&out, eps)?
    };

    // Flatten last two dims: (B, T, 3840, 49) -> (B, T, 188160)
    out.reshape(&[b, t, d * 49])
}

/// Unmasked normalization: 8.0 * (x - mean) / (max - min + eps)
fn normalize_unmasked(x: &Tensor, eps: f32) -> Result<Tensor> {
    // x shape: (B, T, D, 49)
    // Compute mean over dims 1 and 2 with keepdim
    let mean = x.mean_dim(&[1, 2], true)?; // (B, 1, 1, 49)

    // max over dims 1 and 2 sequentially (no multi-dim max in flame_core)
    let x_max = x.max_dim(1, true)?.max_dim(2, true)?; // (B, 1, 1, 49)

    // min via neg -> max -> neg (no min_dim in flame_core)
    let x_neg = x.neg()?;
    let x_min = x_neg.max_dim(1, true)?.max_dim(2, true)?.neg()?; // (B, 1, 1, 49)

    let range = x_max.sub(&x_min)?.add_scalar(eps)?;
    x.sub(&mean)?.div(&range)?.mul_scalar(8.0)
}

// ---------------------------------------------------------------------------
// Gemma3Encoder — full text encoding pipeline for LTX-2
// ---------------------------------------------------------------------------

/// Complete Gemma 3 text encoder for LTX-2 inference.
///
/// Encapsulates:
/// - Gemma 3 12B language model (all 48 layers)
/// - Text embedding projection (188160 -> 3840)
/// - Video embeddings connector (2 transformer blocks, handled externally or as TODO)
///
/// Usage:
/// ```ignore
/// let mut encoder = Gemma3Encoder::new(Gemma3Config::default(), &device)?;
/// encoder.load_gemma_weights(&gemma_weights)?;
/// encoder.load_text_projection(&ltx2_weights)?;
/// let packed = encoder.encode(&input_ids, None)?;
/// // packed: (B, T, 3840) -- pass to video_embeddings_connector then DiT
/// ```
pub struct Gemma3Encoder {
    model: Gemma3Model,
    text_projection: Option<Linear>,
    device: Arc<flame_core::CudaDevice>,
}

impl Gemma3Encoder {
    pub fn new(
        config: Gemma3Config,
        device: &Arc<flame_core::CudaDevice>,
    ) -> Result<Self> {
        let model = Gemma3Model::new(config, device)?;
        Ok(Self {
            model,
            text_projection: None,
            device: device.clone(),
        })
    }

    /// Load Gemma 3 weights from a HashMap (e.g., loaded from safetensors).
    /// Keys should match HuggingFace format: `model.layers.{i}.self_attn.q_proj.weight`, etc.
    /// The `model.` prefix is stripped automatically.
    pub fn load_gemma_weights(&mut self, weights: &HashMap<String, Tensor>) -> Result<()> {
        self.model.load_weights(weights)
    }

    /// Load the text_embedding_projection from the LTX-2 checkpoint.
    /// Looks for key `text_embedding_projection.aggregate_embed.weight`.
    pub fn load_text_projection(&mut self, ltx2_weights: &HashMap<String, Tensor>) -> Result<()> {
        let key = "text_embedding_projection.aggregate_embed.weight";
        let alt_key = "model.diffusion_model.text_embedding_projection.aggregate_embed.weight";
        let proj_w = ltx2_weights
            .get(key)
            .or_else(|| ltx2_weights.get(alt_key))
            .ok_or_else(|| Error::InvalidOperation(format!("Missing weight: {key}")))?;

        let proj_w_bf16 = proj_w.to_dtype(DType::BF16)?;
        let out_features = proj_w_bf16.shape().dims()[0]; // 3840
        let in_features = proj_w_bf16.shape().dims()[1]; // 188160

        let mut proj = Linear::new_zeroed(in_features, out_features, false, &self.device)?;
        proj.copy_weight_from(&proj_w_bf16)?;
        self.text_projection = Some(proj);
        Ok(())
    }

    /// Encode token IDs through the full pipeline (steps 1-5).
    ///
    /// Input:
    ///   - input_ids: (B, T) i32 token indices
    ///   - attention_mask: Optional (B, T) float mask (1.0 = valid, 0.0 = pad)
    ///
    /// Output:
    ///   - (B, T, 3840) projected and normalized embeddings
    ///
    /// After this, the output should go through the video_embeddings_connector
    /// (loaded separately as part of the LTX-2 DiT) and then the DiT's
    /// caption_projection (3840 -> 4096).
    pub fn encode(
        &self,
        input_ids: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let proj = self.text_projection.as_ref().ok_or_else(|| {
            Error::InvalidOperation(
                "Text projection not loaded. Call load_text_projection() first.".into(),
            )
        })?;

        // Step 2: Forward through Gemma (all 49 hidden states)
        let all_hidden = self.model.forward(input_ids, attention_mask)?;

        // Steps 3-4: Pack hidden states and normalize
        let packed = pack_hidden_states_for_ltx2(&all_hidden, attention_mask)?;
        // packed: (B, T, 188160)

        // Step 5: Project to 3840
        let projected = proj.forward(&packed)?;

        Ok(projected)
    }

    /// Access the underlying model for CPU offloading or other management.
    pub fn model(&self) -> &Gemma3Model {
        &self.model
    }

    /// Access the underlying model mutably.
    pub fn model_mut(&mut self) -> &mut Gemma3Model {
        &mut self.model
    }
}

// ---------------------------------------------------------------------------
// Video Embeddings Connector (stub -- depends on LTX-2 DiT blocks)
// ---------------------------------------------------------------------------

// The EmbeddingsConnector (2 self-attention + FFN blocks with RoPE and learnable
// registers) is architecturally part of the LTX-2 DiT, not the Gemma encoder.
// It uses the same CrossAttention and FeedForward blocks as the DiT transformer.
//
// When implementing the full LTX-2 pipeline in Rust, the connector should be
// implemented alongside the DiT blocks (it shares their attention/FFN code).
//
// For now, the Gemma3Encoder.encode() returns the projected (B, T, 3840) output
// which should be fed into the connector externally.
//
// TODO: Implement EmbeddingsConnector when LTX-2 DiT blocks are ported to Rust.
// Key details:
//   - inner_dim=3840, num_heads=30, head_dim=128
//   - 2 transformer blocks (self-attn + FFN with RMSNorm)
//   - 128 learnable register tokens (tiled to fill padding positions)
//   - RoPE with theta=10000, 1D frequency grid
//   - Weights live under `video_embeddings_connector.*` in the LTX-2 checkpoint
