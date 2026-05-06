//! Qwen2.5-VL text encoder (text-only path) for Kandinsky-5 inference.
//!
//! Loads a Qwen2.5-VL-7B-Instruct model from safetensors and extracts the
//! last hidden state from the language model. Only the text-only forward path
//! is implemented (no vision tokens).
//!
//! Architecture (7B variant):
//!   - 28 layers, hidden_size=3584, num_heads=28, num_kv_heads=4 (GQA 7:1)
//!   - head_dim=128, intermediate_size=18944
//!   - SwiGLU MLP: gate_proj + up_proj -> SiLU(gate) * up -> down_proj
//!   - RMSNorm (eps=1e-6), RoPE (theta=1e6)
//!   - Q, K, and V projections ALL have biases (`attention_bias=True`);
//!     only `o_proj` is bias-free.
//!   - No QK norm (unlike Qwen3)
//!
//! All BF16, F32 only for RoPE angles and normalization internals.

use flame_core::attention::sdpa as flame_sdpa;
use flame_core::{CudaDevice, DType, Result, Shape, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

/// Qwen2.5-VL model configuration, auto-detected from weight shapes.
#[derive(Debug, Clone)]
pub struct Qwen25VLConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f64,
    pub max_seq_len: usize,
}

impl Default for Qwen25VLConfig {
    fn default() -> Self {
        Self {
            vocab_size: 152064,
            hidden_size: 3584,
            num_layers: 28,
            intermediate_size: 18944,
            num_heads: 28,
            num_kv_heads: 4,
            head_dim: 128,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            max_seq_len: 512,
        }
    }
}

/// Qwen2.5-VL text encoder — pure flame-core implementation (text-only path).
///
/// Holds all weights as a flat `HashMap<String, Tensor>` and runs the
/// forward pass by indexing into it directly. For Kandinsky-5 inference,
/// returns the last hidden state from the final layer + final norm.
pub struct Qwen25VLEncoder {
    weights: HashMap<String, Tensor>,
    config: Qwen25VLConfig,
    device: Arc<CudaDevice>,
}

impl Qwen25VLEncoder {
    /// Create from pre-loaded weight tensors and config.
    ///
    /// All tensors must already be on the target CUDA device and in BF16.
    /// 2D weight matrices are pre-transposed at construction for faster matmul.
    pub fn new(
        weights: HashMap<String, Tensor>,
        config: Qwen25VLConfig,
        device: Arc<CudaDevice>,
    ) -> Self {
        // Pre-transpose all 2D weight matrices [out, in] -> [in, out]
        // Skip layernorm, norm, embed weights, and bias tensors.
        let mut weights = weights;
        let keys: Vec<String> = weights.keys().cloned().collect();
        for key in &keys {
            if key.ends_with(".weight")
                && !key.contains("layernorm")
                && !key.contains("norm")
                && !key.contains("embed")
            {
                let w = &weights[key];
                if w.shape().dims().len() == 2 {
                    if let Ok(wt) = flame_core::bf16_elementwise::transpose2d_bf16(w) {
                        weights.insert(key.clone(), wt);
                    }
                }
            }
        }
        Self {
            weights,
            config,
            device,
        }
    }

    /// Auto-detect configuration from weight tensor shapes.
    ///
    /// Examines `model.embed_tokens.weight`, `model.layers.0.self_attn.q_proj.weight`,
    /// etc. to infer all hyperparameters.
    pub fn config_from_weights(weights: &HashMap<String, Tensor>) -> Result<Qwen25VLConfig> {
        let embed_key = "model.embed_tokens.weight";
        let embed_w = weights.get(embed_key).ok_or_else(|| {
            flame_core::Error::InvalidInput(format!(
                "Cannot find {embed_key} in weights. First 10 keys: {:?}",
                weights.keys().take(10).collect::<Vec<_>>()
            ))
        })?;

        let vocab_size = embed_w.shape().dims()[0];
        let hidden_size = embed_w.shape().dims()[1];

        // Count layers
        let mut num_layers = 0;
        while weights.contains_key(&format!("model.layers.{num_layers}.self_attn.q_proj.weight"))
        {
            num_layers += 1;
        }

        // Intermediate size from gate_proj
        let gate_key = "model.layers.0.mlp.gate_proj.weight";
        let intermediate_size = weights
            .get(gate_key)
            .map(|t| t.shape().dims()[0])
            .unwrap_or(hidden_size * 4);

        // Head counts from projection shapes
        let head_dim = 128; // Standard for Qwen2.5-VL
        let q_key = "model.layers.0.self_attn.q_proj.weight";
        let k_key = "model.layers.0.self_attn.k_proj.weight";
        let num_heads = weights
            .get(q_key)
            .map(|t| t.shape().dims()[0] / head_dim)
            .unwrap_or(28);
        let num_kv_heads = weights
            .get(k_key)
            .map(|t| t.shape().dims()[0] / head_dim)
            .unwrap_or(4);

        Ok(Qwen25VLConfig {
            vocab_size,
            hidden_size,
            num_layers,
            intermediate_size,
            num_heads,
            num_kv_heads,
            head_dim,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            max_seq_len: 512,
        })
    }

    /// Get a reference to a weight tensor, returning an error if missing.
    fn w(&self, key: &str) -> Result<&Tensor> {
        self.weights.get(key).ok_or_else(|| {
            flame_core::Error::InvalidInput(format!("Missing weight key: {key}"))
        })
    }

    // -----------------------------------------------------------------------
    // Linear projection helpers
    // -----------------------------------------------------------------------

    /// Matmul for [B, N, C] x [C, out] -> [B, N, out].
    /// Weight is already pre-transposed at construction time.
    fn linear(x: &Tensor, weight_t: &Tensor) -> Result<Tensor> {
        let shape = x.shape().dims().to_vec();
        if shape.len() == 2 {
            return x.matmul(weight_t);
        }
        let b = shape[0];
        let n = shape[1];
        let c = shape[2];
        let x_2d = x.reshape(&[b * n, c])?;
        let out_2d = x_2d.matmul(weight_t)?;
        let out_dim = out_2d.shape().dims()[1];
        out_2d.reshape(&[b, n, out_dim])
    }

    /// Linear projection with bias: x @ weight_t + bias.
    fn linear_bias(x: &Tensor, weight_t: &Tensor, bias: &Tensor) -> Result<Tensor> {
        let out = Self::linear(x, weight_t)?;
        out.add(bias)
    }

    // -----------------------------------------------------------------------
    // RoPE — 1D causal (single sequence position)
    // -----------------------------------------------------------------------

    /// Build 1D RoPE cos/sin tables for a padded Qwen2.5-VL input.
    ///
    /// HF constructs position_ids for text-only prompts via
    /// `Qwen2_5_VLModel.get_rope_index`:
    ///   `position_ids = cumsum(attention_mask) - 1`
    ///   `position_ids.masked_fill_(attention_mask == 0, 1)`
    /// so real tokens map to `[0, 1, ..., real_len-1]` and every pad token
    /// gets dummy position 1. This diverges from a naive sequential
    /// `arange(seq_len)` once padding is present — using sequential IDs
    /// at pad positions gives those positions *different* K/V values than
    /// HF, and although the causal+pad mask normally keeps real-token
    /// outputs intact, any path that touches pad positions (or any parity
    /// metric that averages over the full sequence) will disagree.
    ///
    /// Returns (cos, sin), each shaped `[1, 1, seq_len, head_dim / 2]` in BF16.
    fn build_rope_cache(
        seq_len: usize,
        real_len: usize,
        head_dim: usize,
        theta: f64,
        device: &Arc<CudaDevice>,
    ) -> Result<(Tensor, Tensor)> {
        let half = head_dim / 2;

        let positions: Vec<f32> = (0..seq_len)
            .map(|i| if i < real_len { i as f32 } else { 1.0 })
            .collect();
        let pos = Tensor::from_vec(
            positions,
            Shape::from_dims(&[seq_len]),
            device.clone(),
        )?;

        let freq_idx = Tensor::arange(0.0, head_dim as f32, 2.0, device.clone())?;
        let log_theta = (theta as f32).ln();
        let scale = -log_theta / head_dim as f32;
        let log_freqs = freq_idx.mul_scalar(scale)?.exp()?;

        let pos_col = pos.reshape(&[seq_len, 1])?;
        let freq_row = log_freqs.reshape(&[1, half])?;
        let angles = pos_col.matmul(&freq_row)?;

        let cos = angles
            .cos()?
            .unsqueeze(0)?
            .unsqueeze(0)?
            .to_dtype(DType::BF16)?;
        let sin = angles
            .sin()?
            .unsqueeze(0)?
            .unsqueeze(0)?
            .to_dtype(DType::BF16)?;
        Ok((cos, sin))
    }

    /// Apply rotary position embeddings to a single tensor.
    ///
    /// `x`: `[B, H, N, D]`, `pe_cos`/`pe_sin`: `[1, 1, N, D/2]`.
    /// Fused RoPE — single CUDA kernel, no intermediates.
    fn apply_rope(x: &Tensor, pe_cos: &Tensor, pe_sin: &Tensor) -> Result<Tensor> {
        // Qwen2.5-VL uses half-split RoPE (HF rotate_half convention)
        flame_core::bf16_ops::rope_halfsplit_bf16(x, pe_cos, pe_sin)
    }

    // -----------------------------------------------------------------------
    // GQA head repeat
    // -----------------------------------------------------------------------

    /// Repeat KV heads to match Q head count for GQA.
    ///
    /// `x`: `[B, kv_heads, seq, head_dim]` -> `[B, kv_heads * n_rep, seq, head_dim]`
    fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
        if n_rep == 1 {
            return Ok(x.clone());
        }
        let dims = x.shape().dims();
        let b = dims[0];
        let h_kv = dims[1];
        let n = dims[2];
        let d = dims[3];

        let copies: Vec<Tensor> = (0..n_rep).map(|_| x.clone()).collect();
        let stacked = Tensor::stack(&copies, 2)?;
        stacked.reshape(&[b, h_kv * n_rep, n, d])
    }

    // -----------------------------------------------------------------------
    // Causal mask
    // -----------------------------------------------------------------------

    /// Build a combined causal + padding mask `[1, 1, seq_len, seq_len]`
    /// in BF16. **Multiplicative / boolean** convention, matching flame's
    /// SDPA (`crate::sdpa::forward_bf16_fallback`): the kernel internally
    /// computes `scores += (1.0 - mask) * -1e9`, so `mask = 1.0` is
    /// allowed and `mask = 0.0` is blocked. A float additive mask would
    /// be interpreted inside-out.
    ///
    /// `real_len`: number of real (non-pad) tokens at the start of the
    /// sequence. Attention from any query row `i` to key column `j` is
    /// allowed iff `j <= i && j < real_len` (causal + padding combined).
    fn build_causal_mask(
        seq_len: usize,
        real_len: usize,
        device: &Arc<CudaDevice>,
    ) -> Result<Tensor> {
        let mut data = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                if j <= i && j < real_len {
                    data[i * seq_len + j] = 1.0;
                }
            }
        }
        let mask_f32 = Tensor::from_vec(
            data,
            Shape::from_dims(&[1, 1, seq_len, seq_len]),
            device.clone(),
        )?;
        mask_f32.to_dtype(DType::BF16)
    }

    // -----------------------------------------------------------------------
    // RMSNorm helper (weight-only, no bias)
    // -----------------------------------------------------------------------

    /// Apply RMSNorm: reshape to 2D, normalize, reshape back.
    fn rms_norm(x: &Tensor, scale: &Tensor, eps: f32) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let hidden = *dims.last().unwrap();
        let batch: usize = dims[..dims.len() - 1].iter().product();
        let x_2d = x.reshape(&[batch, hidden])?;
        let out_2d = flame_core::cuda_ops_bf16::rms_norm_bf16(&x_2d, Some(scale), eps)?;
        out_2d.reshape(&dims)
    }

    // -----------------------------------------------------------------------
    // Single transformer layer
    // -----------------------------------------------------------------------

    /// Execute one transformer layer.
    fn layer_forward(
        &self,
        layer_idx: usize,
        hidden: &Tensor,
        pe_cos: &Tensor,
        pe_sin: &Tensor,
        attn_mask: &Tensor,
    ) -> Result<Tensor> {
        let cfg = &self.config;
        let h = cfg.num_heads;
        let h_kv = cfg.num_kv_heads;
        let d = cfg.head_dim;
        let n_rep = h / h_kv;
        let prefix = format!("model.layers.{layer_idx}");

        let dims = hidden.shape().dims().to_vec();
        let b = dims[0];
        let n = dims[1];

        // --- Self-attention ---

        // 1. RMSNorm (input_layernorm)
        let norm_w = self.w(&format!("{prefix}.input_layernorm.weight"))?;
        let normed = Self::rms_norm(hidden, norm_w, cfg.rms_norm_eps)?;

        // 2. Q, K, V projections — all three have biases in Qwen2.5-VL
        let q_w = self.w(&format!("{prefix}.self_attn.q_proj.weight"))?;
        let q_b = self.w(&format!("{prefix}.self_attn.q_proj.bias"))?;
        let k_w = self.w(&format!("{prefix}.self_attn.k_proj.weight"))?;
        let k_b = self.w(&format!("{prefix}.self_attn.k_proj.bias"))?;
        let v_w = self.w(&format!("{prefix}.self_attn.v_proj.weight"))?;
        let v_b = self.w(&format!("{prefix}.self_attn.v_proj.bias"))?;

        let q = Self::linear_bias(&normed, q_w, q_b)?;
        let k = Self::linear_bias(&normed, k_w, k_b)?;
        let v = Self::linear_bias(&normed, v_w, v_b)?;

        // 3. Reshape to [B, seq, heads, head_dim], permute to [B, heads, seq, head_dim]
        let q = q.reshape(&[b, n, h, d])?.permute(&[0, 2, 1, 3])?;
        let k = k.reshape(&[b, n, h_kv, d])?.permute(&[0, 2, 1, 3])?;
        let v = v.reshape(&[b, n, h_kv, d])?.permute(&[0, 2, 1, 3])?;

        // 4. RoPE on Q, K (no QK norm in Qwen2.5-VL, unlike Qwen3)
        let q = Self::apply_rope(&q, pe_cos, pe_sin)?;
        let k = Self::apply_rope(&k, pe_cos, pe_sin)?;

        // 5. GQA: repeat K,V from 4 heads to 28 heads
        let k = Self::repeat_kv(&k, n_rep)?;
        let v = Self::repeat_kv(&v, n_rep)?;

        // 6. SDPA with causal mask
        let attn_out = flame_sdpa(&q, &k, &v, Some(attn_mask))?;

        // 7. Reshape + O projection (no bias on O)
        let attn_out = attn_out
            .permute(&[0, 2, 1, 3])?
            .reshape(&[b, n, h * d])?;

        let o_w = self.w(&format!("{prefix}.self_attn.o_proj.weight"))?;
        let attn_out = Self::linear(&attn_out, o_w)?;

        // 8. Residual add
        let hidden = hidden.add(&attn_out)?;

        // --- MLP ---

        // 9. RMSNorm (post_attention_layernorm)
        let post_norm_w = self.w(&format!("{prefix}.post_attention_layernorm.weight"))?;
        let normed2 = Self::rms_norm(&hidden, post_norm_w, cfg.rms_norm_eps)?;

        // 10. SwiGLU MLP: silu(gate) * up -> down
        let gate_w = self.w(&format!("{prefix}.mlp.gate_proj.weight"))?;
        let up_w = self.w(&format!("{prefix}.mlp.up_proj.weight"))?;
        let down_w = self.w(&format!("{prefix}.mlp.down_proj.weight"))?;

        let gate = Self::linear(&normed2, gate_w)?;
        let up = Self::linear(&normed2, up_w)?;
        let mlp_out = gate.silu()?.mul(&up)?;
        let mlp_out = Self::linear(&mlp_out, down_w)?;

        // 11. Residual add
        hidden.add(&mlp_out)
    }

    // -----------------------------------------------------------------------
    // Embedding lookup
    // -----------------------------------------------------------------------

    /// Gather embedding rows for a list of token IDs.
    ///
    /// Returns `[1, seq_len, hidden_size]` in BF16.
    fn embed_tokens(&self, token_ids: &[i32]) -> Result<Tensor> {
        let embed_w = self.w("model.embed_tokens.weight")?;
        let seq_len = token_ids.len();

        let ids_tensor = Tensor::from_vec(
            token_ids.iter().map(|&id| id as f32).collect(),
            Shape::from_dims(&[seq_len]),
            self.device.clone(),
        )?
        .to_dtype(DType::I32)?;

        let selected = embed_w.index_select0(&ids_tensor)?;
        selected.unsqueeze(0)
    }

    // -----------------------------------------------------------------------
    // Full forward pass
    // -----------------------------------------------------------------------

    /// Full forward + capture of (embedding, per-layer outputs, final
    /// hidden). Used by `bin/qwen25vl_parity` to locate where a
    /// parity-test divergence starts. `max_layer` is inclusive; pass
    /// `usize::MAX` for "all layers".
    pub fn encode_with_intermediates(
        &self,
        token_ids: &[i32],
    ) -> Result<(Tensor, Vec<Tensor>, Tensor)> {
        let cfg = &self.config;
        let seq_len = token_ids.len();
        let pad_id = 151643i32;
        let real_len = token_ids
            .iter()
            .position(|&id| id == pad_id)
            .unwrap_or(seq_len);

        let hidden = self.embed_tokens(token_ids)?;
        let embed = hidden.clone();

        let (pe_cos, pe_sin) =
            Self::build_rope_cache(seq_len, real_len, cfg.head_dim, cfg.rope_theta, &self.device)?;
        let attn_mask = Self::build_causal_mask(seq_len, real_len, &self.device)?;

        let mut x = hidden;
        let mut layer_outs = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            x = self.layer_forward(i, &x, &pe_cos, &pe_sin, &attn_mask)?;
            layer_outs.push(x.clone());
        }
        let final_norm_w = self.w("model.norm.weight")?;
        let final_hidden = Self::rms_norm(&x, final_norm_w, cfg.rms_norm_eps)?;
        Ok((embed, layer_outs, final_hidden))
    }

    /// Dump every substep of layer 0 into a `HashMap` keyed by stage
    /// name. Used by `bin/qwen25vl_layer0_parity` to locate the first
    /// diverging op when the layer-0 output is wrong.
    ///
    /// Keys (in execution order):
    ///   `embed_out`       `[1, N, H]`
    ///   `normed_input`    `[1, N, H]`   (after `input_layernorm`)
    ///   `q_raw`, `k_raw`, `v_raw`       post-projection, pre-reshape, with bias
    ///   `q_heads`, `k_heads`, `v_heads` after `[B,N,h*d] -> [B,h,N,d]`
    ///   `q_roped`, `k_roped`            after `apply_rope`
    ///   `k_repeated`, `v_repeated`      after `repeat_kv(n_rep)`
    ///   `attn_sdpa`       `[1, H_q, N, D]`  after flame_sdpa
    ///   `attn_merge`      `[1, N, H]`       after permute + merge heads
    ///   `attn_o_out`      `[1, N, H]`       after o_proj
    ///   `after_attn`      `[1, N, H]`       after residual add (= first block output)
    ///   `normed_post`     `[1, N, H]`       after `post_attention_layernorm`
    ///   `gate_raw`, `up_raw`            MLP projections
    ///   `mlp_pre_down`    `silu(gate) * up`
    ///   `mlp_out`         after `down_proj`
    ///   `layer_0_out`     `[1, N, H]`       after MLP residual add
    pub fn layer0_substep_probe(
        &self,
        token_ids: &[i32],
    ) -> Result<HashMap<String, Tensor>> {
        let cfg = &self.config;
        let seq_len = token_ids.len();
        let pad_id = 151643i32;
        let real_len = token_ids
            .iter()
            .position(|&id| id == pad_id)
            .unwrap_or(seq_len);

        let hidden = self.embed_tokens(token_ids)?;
        let (pe_cos, pe_sin) =
            Self::build_rope_cache(seq_len, real_len, cfg.head_dim, cfg.rope_theta, &self.device)?;
        let attn_mask = Self::build_causal_mask(seq_len, real_len, &self.device)?;

        let mut out: HashMap<String, Tensor> = HashMap::new();
        out.insert("embed_out".into(), hidden.clone());

        let h = cfg.num_heads;
        let h_kv = cfg.num_kv_heads;
        let d = cfg.head_dim;
        let n_rep = h / h_kv;
        let dims = hidden.shape().dims().to_vec();
        let b = dims[0];
        let n = dims[1];
        let prefix = "model.layers.0";

        // --- Self-attention ---
        let norm_w = self.w(&format!("{prefix}.input_layernorm.weight"))?;
        let normed = Self::rms_norm(&hidden, norm_w, cfg.rms_norm_eps)?;
        out.insert("normed_input".into(), normed.clone());

        let q_w = self.w(&format!("{prefix}.self_attn.q_proj.weight"))?;
        let q_b = self.w(&format!("{prefix}.self_attn.q_proj.bias"))?;
        let k_w = self.w(&format!("{prefix}.self_attn.k_proj.weight"))?;
        let k_b = self.w(&format!("{prefix}.self_attn.k_proj.bias"))?;
        let v_w = self.w(&format!("{prefix}.self_attn.v_proj.weight"))?;
        let v_b = self.w(&format!("{prefix}.self_attn.v_proj.bias"))?;

        let q_raw = Self::linear_bias(&normed, q_w, q_b)?;
        let k_raw = Self::linear_bias(&normed, k_w, k_b)?;
        let v_raw = Self::linear_bias(&normed, v_w, v_b)?;
        out.insert("q_raw".into(), q_raw.clone());
        out.insert("k_raw".into(), k_raw.clone());
        out.insert("v_raw".into(), v_raw.clone());

        let q_heads = q_raw.reshape(&[b, n, h, d])?.permute(&[0, 2, 1, 3])?;
        let k_heads = k_raw.reshape(&[b, n, h_kv, d])?.permute(&[0, 2, 1, 3])?;
        let v_heads = v_raw.reshape(&[b, n, h_kv, d])?.permute(&[0, 2, 1, 3])?;
        out.insert("q_heads".into(), q_heads.clone());
        out.insert("k_heads".into(), k_heads.clone());
        out.insert("v_heads".into(), v_heads.clone());

        let q_roped = Self::apply_rope(&q_heads, &pe_cos, &pe_sin)?;
        let k_roped = Self::apply_rope(&k_heads, &pe_cos, &pe_sin)?;
        out.insert("q_roped".into(), q_roped.clone());
        out.insert("k_roped".into(), k_roped.clone());

        let k_rep = Self::repeat_kv(&k_roped, n_rep)?;
        let v_rep = Self::repeat_kv(&v_heads, n_rep)?;
        out.insert("k_repeated".into(), k_rep.clone());
        out.insert("v_repeated".into(), v_rep.clone());

        let attn_sdpa = flame_sdpa(&q_roped, &k_rep, &v_rep, Some(&attn_mask))?;
        out.insert("attn_sdpa".into(), attn_sdpa.clone());

        let attn_merge = attn_sdpa.permute(&[0, 2, 1, 3])?.reshape(&[b, n, h * d])?;
        out.insert("attn_merge".into(), attn_merge.clone());

        let o_w = self.w(&format!("{prefix}.self_attn.o_proj.weight"))?;
        let attn_o_out = Self::linear(&attn_merge, o_w)?;
        out.insert("attn_o_out".into(), attn_o_out.clone());

        let after_attn = hidden.add(&attn_o_out)?;
        out.insert("after_attn".into(), after_attn.clone());

        // --- MLP ---
        let post_norm_w = self.w(&format!("{prefix}.post_attention_layernorm.weight"))?;
        let normed2 = Self::rms_norm(&after_attn, post_norm_w, cfg.rms_norm_eps)?;
        out.insert("normed_post".into(), normed2.clone());

        let gate_w = self.w(&format!("{prefix}.mlp.gate_proj.weight"))?;
        let up_w = self.w(&format!("{prefix}.mlp.up_proj.weight"))?;
        let down_w = self.w(&format!("{prefix}.mlp.down_proj.weight"))?;

        let gate_raw = Self::linear(&normed2, gate_w)?;
        let up_raw = Self::linear(&normed2, up_w)?;
        out.insert("gate_raw".into(), gate_raw.clone());
        out.insert("up_raw".into(), up_raw.clone());

        let mlp_pre_down = gate_raw.silu()?.mul(&up_raw)?;
        out.insert("mlp_pre_down".into(), mlp_pre_down.clone());

        let mlp_out = Self::linear(&mlp_pre_down, down_w)?;
        out.insert("mlp_out".into(), mlp_out.clone());

        let layer_0 = after_attn.add(&mlp_out)?;
        out.insert("layer_0_out".into(), layer_0);

        Ok(out)
    }

    /// Run forward pass and return the last hidden state.
    ///
    /// # Returns
    /// Tensor of shape `[1, seq_len, 3584]` — the final hidden state after
    /// the last transformer layer and final RMSNorm, suitable for Kandinsky-5
    /// conditioning.
    pub fn encode(&self, token_ids: &[i32]) -> Result<Tensor> {
        let cfg = &self.config;
        let seq_len = token_ids.len();

        // Detect real (non-pad) token count.
        // Qwen2.5-VL pad token id = 151643
        let pad_id = 151643i32;
        let real_len = token_ids
            .iter()
            .position(|&id| id == pad_id)
            .unwrap_or(seq_len);

        // 1. Embed tokens
        let mut hidden = self.embed_tokens(token_ids)?;

        // 2. Build RoPE cache for seq_len
        let (pe_cos, pe_sin) =
            Self::build_rope_cache(seq_len, real_len, cfg.head_dim, cfg.rope_theta, &self.device)?;

        // Build causal + padding mask
        let attn_mask = Self::build_causal_mask(seq_len, real_len, &self.device)?;

        // 3. Run through all layers
        for i in 0..cfg.num_layers {
            hidden = self.layer_forward(i, &hidden, &pe_cos, &pe_sin, &attn_mask)?;
        }

        // 4. Final RMSNorm
        let final_norm_w = self.w("model.norm.weight")?;
        hidden = Self::rms_norm(&hidden, final_norm_w, cfg.rms_norm_eps)?;

        // 5. Return last hidden state [1, seq_len, hidden_size]
        Ok(hidden)
    }

    /// Get the output hidden dimension.
    pub fn output_dim(&self) -> usize {
        self.config.hidden_size
    }

    /// Get a reference to the config.
    pub fn config(&self) -> &Qwen25VLConfig {
        &self.config
    }

    /// Get read-only access to weights (for debugging).
    pub fn weights(&self) -> &HashMap<String, Tensor> {
        &self.weights
    }
}

// ---------------------------------------------------------------------------
// Weight key listing (for validation)
// ---------------------------------------------------------------------------

/// Return all expected weight keys for a Qwen2.5-VL language model with the
/// given layer count (text-only path — excludes vision encoder weights).
pub fn expected_weight_keys(num_layers: usize) -> Vec<String> {
    let mut keys = vec!["model.embed_tokens.weight".to_string()];

    for i in 0..num_layers {
        let p = format!("model.layers.{i}");
        keys.extend([
            format!("{p}.self_attn.q_proj.weight"),
            format!("{p}.self_attn.q_proj.bias"),
            format!("{p}.self_attn.k_proj.weight"),
            format!("{p}.self_attn.k_proj.bias"),
            format!("{p}.self_attn.v_proj.weight"),
            format!("{p}.self_attn.v_proj.bias"),
            format!("{p}.self_attn.o_proj.weight"),
            format!("{p}.mlp.gate_proj.weight"),
            format!("{p}.mlp.up_proj.weight"),
            format!("{p}.mlp.down_proj.weight"),
            format!("{p}.input_layernorm.weight"),
            format!("{p}.post_attention_layernorm.weight"),
        ]);
    }

    keys.push("model.norm.weight".to_string());
    keys
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = Qwen25VLConfig::default();
        assert_eq!(cfg.vocab_size, 152064);
        assert_eq!(cfg.hidden_size, 3584);
        assert_eq!(cfg.num_layers, 28);
        assert_eq!(cfg.num_heads, 28);
        assert_eq!(cfg.num_kv_heads, 4);
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.intermediate_size, 18944);
    }

    #[test]
    fn test_gqa_ratio() {
        let cfg = Qwen25VLConfig::default();
        assert_eq!(cfg.num_heads / cfg.num_kv_heads, 7);
    }

    #[test]
    fn test_output_dim() {
        let cfg = Qwen25VLConfig::default();
        assert_eq!(cfg.hidden_size, 3584);
    }

    #[test]
    fn test_expected_weight_keys_count() {
        let keys = expected_weight_keys(28);
        // 1 (embed) + 28 * 12 (per-layer: 4 attn weights + 3 qkv biases +
        // 2 layernorm + 3 mlp) + 1 (final norm) = 338
        assert_eq!(keys.len(), 1 + 28 * 12 + 1);
    }

    #[test]
    fn test_expected_weight_keys_has_biases() {
        let keys = expected_weight_keys(28);
        // Qwen2.5-VL has `attention_bias=True` — Q, K, V all have biases.
        assert!(keys.contains(&"model.layers.0.self_attn.q_proj.bias".to_string()));
        assert!(keys.contains(&"model.layers.0.self_attn.k_proj.bias".to_string()));
        assert!(keys.contains(&"model.layers.0.self_attn.v_proj.bias".to_string()));
        // o_proj alone is bias-free.
        assert!(!keys.contains(&"model.layers.0.self_attn.o_proj.bias".to_string()));
    }

    #[test]
    fn test_expected_weight_keys_no_qk_norm() {
        let keys = expected_weight_keys(28);
        // Qwen2.5-VL does not have QK norm (unlike Qwen3)
        assert!(!keys.contains(&"model.layers.0.self_attn.q_norm.weight".to_string()));
        assert!(!keys.contains(&"model.layers.0.self_attn.k_norm.weight".to_string()));
    }
}
