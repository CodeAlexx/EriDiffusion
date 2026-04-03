//! Qwen3 text encoder for Klein/Flux inference using flame-core tensor ops.
//!
//! Loads a Qwen3 causal LM from safetensors and extracts hidden states at
//! layers [8, 17, 26] (0-indexed), stacking them along the hidden dimension
//! to produce Klein's joint_attention_dim embedding:
//!   - Klein 4B: hidden_size=2560, joint_dim = 3 * 2560 = 7680
//!   - Klein 9B: hidden_size=4096, joint_dim = 3 * 4096 = 12288
//!
//! Architecture per layer:
//!   input -> RMSNorm -> QKV proj -> QK norm -> RoPE -> GQA attention -> residual
//!         -> RMSNorm -> gate+up -> SiLU(gate)*up -> down -> residual
//!
//! All BF16, F32 only for RoPE angles and normalization internals.

use crate::ops::RMSNorm;
use flame_core::attention::sdpa as flame_sdpa;
use flame_core::{CudaDevice, DType, Result, Shape, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

/// Default extraction layers for Klein models (0-indexed layer indices).
/// These correspond to hidden_states[9], [18], [27] in the transformers
/// convention where index 0 = embedding output.
const KLEIN_EXTRACT_LAYERS: [usize; 3] = [8, 17, 26];

/// Qwen3 model configuration, auto-detected from weight shapes.
#[derive(Debug, Clone)]
pub struct Qwen3Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f64,
    pub extract_layers: Vec<usize>,
}

impl Default for Qwen3Config {
    fn default() -> Self {
        Self {
            vocab_size: 151936,
            hidden_size: 2560,
            num_layers: 36,
            intermediate_size: 6912,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            extract_layers: KLEIN_EXTRACT_LAYERS.to_vec(),
        }
    }
}

/// Qwen3 text encoder for Klein — pure flame-core implementation.
///
/// Holds all weights as a flat `HashMap<String, Tensor>` and runs the
/// forward pass by indexing into it directly, matching the Python blueprint.
pub struct Qwen3Encoder {
    weights: HashMap<String, Tensor>,
    config: Qwen3Config,
    device: Arc<CudaDevice>,
}

impl Qwen3Encoder {
    /// Create from pre-loaded weight tensors and auto-detected config.
    ///
    /// All tensors must already be on the target CUDA device and in BF16.
    pub fn new(
        weights: HashMap<String, Tensor>,
        config: Qwen3Config,
        device: Arc<CudaDevice>,
    ) -> Self {
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
    pub fn config_from_weights(weights: &HashMap<String, Tensor>) -> Result<Qwen3Config> {
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
        while weights.contains_key(&format!("model.layers.{num_layers}.self_attn.q_proj.weight")) {
            num_layers += 1;
        }

        // Intermediate size from gate_proj
        let gate_key = "model.layers.0.mlp.gate_proj.weight";
        let intermediate_size = weights
            .get(gate_key)
            .map(|t| t.shape().dims()[0])
            .unwrap_or(hidden_size * 4);

        // Head counts from projection shapes
        let head_dim = 128; // Standard for Qwen3
        let q_key = "model.layers.0.self_attn.q_proj.weight";
        let k_key = "model.layers.0.self_attn.k_proj.weight";
        let num_heads = weights
            .get(q_key)
            .map(|t| t.shape().dims()[0] / head_dim)
            .unwrap_or(32);
        let num_kv_heads = weights
            .get(k_key)
            .map(|t| t.shape().dims()[0] / head_dim)
            .unwrap_or(8);

        Ok(Qwen3Config {
            vocab_size,
            hidden_size,
            num_layers,
            intermediate_size,
            num_heads,
            num_kv_heads,
            head_dim,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            extract_layers: KLEIN_EXTRACT_LAYERS.to_vec(),
        })
    }

    /// Get a reference to a weight tensor, returning an error if missing.
    fn w(&self, key: &str) -> Result<&Tensor> {
        self.weights.get(key).ok_or_else(|| {
            flame_core::Error::InvalidInput(format!("Missing weight key: {key}"))
        })
    }

    // -----------------------------------------------------------------------
    // Linear projection helper
    // -----------------------------------------------------------------------

    /// flame-core Linear::forward handles arbitrary rank, but we need a
    /// simple weight-only matmul for [B, N, C] x [out, C]^T -> [B, N, out].
    /// We flatten to 2D, matmul with transposed weight, then reshape back.
    fn linear_3d(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
        let shape = x.shape().dims().to_vec();
        if shape.len() == 2 {
            // x: [M, C], weight: [out, C] -> x @ weight^T = [M, out]
            let wt = weight.transpose_dims(0, 1)?;
            return x.matmul(&wt);
        }
        let b = shape[0];
        let n = shape[1];
        let c = shape[2];
        let x_2d = x.reshape(&[b * n, c])?;
        let wt = weight.transpose_dims(0, 1)?;
        let out_2d = x_2d.matmul(&wt)?;
        let out_dim = out_2d.shape().dims()[1];
        out_2d.reshape(&[b, n, out_dim])
    }

    // -----------------------------------------------------------------------
    // RoPE — 1D causal (single sequence position)
    // -----------------------------------------------------------------------

    /// Build 1D RoPE cos/sin tables for causal attention.
    ///
    /// Returns (cos, sin), each shaped `[1, 1, seq_len, head_dim / 2]` in BF16.
    fn build_rope_1d(
        seq_len: usize,
        head_dim: usize,
        theta: f64,
        device: &Arc<CudaDevice>,
    ) -> Result<(Tensor, Tensor)> {
        let half = head_dim / 2;

        // Position indices: [seq_len] in F32
        let pos = Tensor::arange(0.0, seq_len as f32, 1.0, device.clone())?;

        // Frequency indices: exp(-log(theta) * arange(0, head_dim, 2) / head_dim)
        let freq_idx = Tensor::arange(0.0, head_dim as f32, 2.0, device.clone())?;
        let log_theta = (theta as f32).ln();
        let scale = -log_theta / head_dim as f32;
        let log_freqs = freq_idx.mul_scalar(scale)?.exp()?; // [half]

        // Outer product: [seq_len, 1] * [1, half] -> [seq_len, half]
        let pos_col = pos.reshape(&[seq_len, 1])?;
        let freq_row = log_freqs.reshape(&[1, half])?;
        let angles = pos_col.matmul(&freq_row)?; // [seq_len, half]

        let cos = angles.cos()?.unsqueeze(0)?.unsqueeze(0)?.to_dtype(DType::BF16)?;
        let sin = angles.sin()?.unsqueeze(0)?.unsqueeze(0)?.to_dtype(DType::BF16)?;
        Ok((cos, sin)) // [1, 1, seq_len, half]
    }

    /// Apply rotary position embeddings to a single tensor.
    ///
    /// `x`: `[B, H, N, D]` (H can be any head count).
    /// `pe_cos`, `pe_sin`: `[1, 1, N, D/2]`.
    fn apply_rope_single(
        x: &Tensor,
        pe_cos: &Tensor,
        pe_sin: &Tensor,
    ) -> Result<Tensor> {
        let dims = x.shape().dims();
        let b = dims[0];
        let h = dims[1];
        let n = dims[2];
        let d = dims[3];
        let half = d / 2;

        let cos_f = pe_cos.to_dtype(DType::F32)?; // [1, 1, N, half]
        let sin_f = pe_sin.to_dtype(DType::F32)?;
        let cos_flat = cos_f.reshape(&[1, n, half])?;
        let sin_flat = sin_f.reshape(&[1, n, half])?;

        let x_f = x.to_dtype(DType::F32)?;
        // Reshape to [B*H, N, half, 2] then split even/odd
        let x_4d = x_f.reshape(&[b * h, n, half, 2])?;
        let x_even = x_4d.narrow(3, 0, 1)?.squeeze_dim(3)?; // [B*H, N, half]
        let x_odd = x_4d.narrow(3, 1, 1)?.squeeze_dim(3)?;

        // RoPE rotation: new_even = even*cos - odd*sin, new_odd = even*sin + odd*cos
        let new_even = x_even.mul(&cos_flat)?.sub(&x_odd.mul(&sin_flat)?)?;
        let new_odd = x_even.mul(&sin_flat)?.add(&x_odd.mul(&cos_flat)?)?;

        // Interleave back: stack on dim 3 -> [B*H, N, half, 2], flatten last two
        let stacked = Tensor::stack(&[new_even, new_odd], 3)?; // [B*H, N, half, 2]
        let flat = stacked.flatten_from(2)?; // [B*H, N, D]
        flat.reshape(&[b, h, n, d])?.to_dtype(x.dtype())
    }

    // -----------------------------------------------------------------------
    // GQA head repeat
    // -----------------------------------------------------------------------

    /// Repeat KV heads to match Q head count for GQA.
    ///
    /// `x`: `[B, H_kv, N, D]` -> `[B, H_kv * n_rep, N, D]`
    fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
        if n_rep == 1 {
            return Ok(x.clone());
        }
        let dims = x.shape().dims();
        let b = dims[0];
        let h_kv = dims[1];
        let n = dims[2];
        let d = dims[3];

        // Stack n_rep copies on dim 2: [B, H_kv, n_rep, N, D] then reshape
        let copies: Vec<Tensor> = (0..n_rep).map(|_| x.clone()).collect();
        let stacked = Tensor::stack(&copies, 2)?; // [B, H_kv, n_rep, N, D]
        stacked.reshape(&[b, h_kv * n_rep, n, d])
    }

    // -----------------------------------------------------------------------
    // Causal mask
    // -----------------------------------------------------------------------

    /// Build a lower-triangular causal mask `[1, 1, seq_len, seq_len]` in BF16.
    /// 1.0 for allowed positions, 0.0 for masked.
    fn build_causal_mask(
        seq_len: usize,
        device: &Arc<CudaDevice>,
    ) -> Result<Tensor> {
        let mut data = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..=i.min(seq_len - 1) {
                data[i * seq_len + j] = 1.0;
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
    /// Uses flame-core's RMSNorm struct internally.
    fn rms_norm_apply(
        x: &Tensor,
        weight: &Tensor,
        eps: f32,
        device: &Arc<CudaDevice>,
    ) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let hidden = *dims.last().unwrap();
        let batch: usize = dims[..dims.len() - 1].iter().product();

        // Create an RMSNorm layer, copy weight in
        let mut norm = RMSNorm::new(vec![hidden], eps, true, device.clone())?;
        norm.copy_weight_from(weight)?;

        // Flatten to [batch, hidden], forward, reshape back
        let x_2d = x.reshape(&[batch, hidden])?;
        let out_2d = norm.forward(&x_2d)?;
        out_2d.reshape(&dims)
    }

    // -----------------------------------------------------------------------
    // Single transformer layer
    // -----------------------------------------------------------------------

    /// Execute one transformer layer.
    ///
    /// Architecture:
    ///   input -> RMSNorm -> QKV proj -> QK norm -> RoPE -> GQA attention -> residual
    ///         -> RMSNorm -> gate+up -> SiLU(gate)*up -> down -> residual
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
        let _c = dims[2];

        // --- Self-attention ---

        // Input layernorm (RMSNorm)
        let norm_w = self.w(&format!("{prefix}.input_layernorm.weight"))?;
        let normed = Self::rms_norm_apply(hidden, norm_w, cfg.rms_norm_eps, &self.device)?;

        // QKV projections
        let q_w = self.w(&format!("{prefix}.self_attn.q_proj.weight"))?;
        let k_w = self.w(&format!("{prefix}.self_attn.k_proj.weight"))?;
        let v_w = self.w(&format!("{prefix}.self_attn.v_proj.weight"))?;

        let q = Self::linear_3d(&normed, q_w)?; // [B, N, H*D]
        let k = Self::linear_3d(&normed, k_w)?; // [B, N, H_kv*D]
        let v = Self::linear_3d(&normed, v_w)?; // [B, N, H_kv*D]

        // Reshape to [B, H, N, D]
        let mut q = q.reshape(&[b, n, h, d])?.permute(&[0, 2, 1, 3])?;
        let mut k = k.reshape(&[b, n, h_kv, d])?.permute(&[0, 2, 1, 3])?;
        let v = v.reshape(&[b, n, h_kv, d])?.permute(&[0, 2, 1, 3])?;

        // QK norm (per-head RMSNorm) — Qwen3 has q_norm and k_norm
        let q_norm_key = format!("{prefix}.self_attn.q_norm.weight");
        let k_norm_key = format!("{prefix}.self_attn.k_norm.weight");
        if let (Some(q_norm_w), Some(k_norm_w)) =
            (self.weights.get(&q_norm_key), self.weights.get(&k_norm_key))
        {
            // q: [B, H, N, D] -> flatten to [B*H*N, D], normalize, reshape back
            let q_flat = q.reshape(&[b * h * n, d])?;
            let mut qn = RMSNorm::new(vec![d], cfg.rms_norm_eps, true, self.device.clone())?;
            qn.copy_weight_from(q_norm_w)?;
            let q_normed = qn.forward(&q_flat)?;
            q = q_normed.reshape(&[b, h, n, d])?;

            let k_flat = k.reshape(&[b * h_kv * n, d])?;
            let mut kn = RMSNorm::new(vec![d], cfg.rms_norm_eps, true, self.device.clone())?;
            kn.copy_weight_from(k_norm_w)?;
            let k_normed = kn.forward(&k_flat)?;
            k = k_normed.reshape(&[b, h_kv, n, d])?;
        }

        // RoPE (applied separately since Q and K may have different head counts)
        let q = Self::apply_rope_single(&q, pe_cos, pe_sin)?;
        let k = Self::apply_rope_single(&k, pe_cos, pe_sin)?;

        // GQA: repeat KV heads to match Q
        let k = Self::repeat_kv(&k, n_rep)?;
        let v = Self::repeat_kv(&v, n_rep)?;

        // Causal attention via SDPA with mask
        let attn_out = flame_sdpa(&q, &k, &v, Some(attn_mask))?; // [B, H, N, D]

        // Reshape back: [B, H, N, D] -> [B, N, H*D]
        let attn_out = attn_out.permute(&[0, 2, 1, 3])?.reshape(&[b, n, h * d])?;

        // Output projection
        let o_w = self.w(&format!("{prefix}.self_attn.o_proj.weight"))?;
        let attn_out = Self::linear_3d(&attn_out, o_w)?;

        // Residual
        let hidden = hidden.add(&attn_out)?;

        // --- MLP ---

        // Post-attention layernorm (RMSNorm)
        let post_norm_w = self.w(&format!("{prefix}.post_attention_layernorm.weight"))?;
        let normed2 = Self::rms_norm_apply(&hidden, post_norm_w, cfg.rms_norm_eps, &self.device)?;

        // Gate + up projection, SiLU activation, down projection
        let gate_w = self.w(&format!("{prefix}.mlp.gate_proj.weight"))?;
        let up_w = self.w(&format!("{prefix}.mlp.up_proj.weight"))?;
        let down_w = self.w(&format!("{prefix}.mlp.down_proj.weight"))?;

        let gate = Self::linear_3d(&normed2, gate_w)?;
        let up = Self::linear_3d(&normed2, up_w)?;
        let mlp_out = gate.silu()?.mul(&up)?;
        let mlp_out = Self::linear_3d(&mlp_out, down_w)?;

        // Residual
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

        // Build I32 index tensor and use index_select0
        let ids_tensor = Tensor::from_vec(
            token_ids.iter().map(|&id| id as f32).collect(),
            Shape::from_dims(&[seq_len]),
            self.device.clone(),
        )?
        .to_dtype(DType::I32)?;

        let selected = embed_w.index_select0(&ids_tensor)?; // [seq_len, hidden_size]
        selected.unsqueeze(0) // [1, seq_len, hidden_size]
    }

    // -----------------------------------------------------------------------
    // Full forward pass
    // -----------------------------------------------------------------------

    /// Run forward pass and return stacked hidden states for Klein.
    ///
    /// # Arguments
    /// * `token_ids` - Token IDs (i32), length = max_length.
    ///
    /// # Returns
    /// Tensor of shape `[1, seq_len, 3 * hidden_size]` — hidden states from
    /// layers 8, 17, 26 (0-indexed) stacked along the last dimension.
    pub fn encode(&self, token_ids: &[i32]) -> Result<Tensor> {
        let cfg = &self.config;
        let seq_len = token_ids.len();

        // 1. Token embedding lookup
        let mut hidden = self.embed_tokens(token_ids)?;

        // 2. Build RoPE tables
        let (pe_cos, pe_sin) =
            Self::build_rope_1d(seq_len, cfg.head_dim, cfg.rope_theta, &self.device)?;

        // 3. Build causal attention mask
        let attn_mask = Self::build_causal_mask(seq_len, &self.device)?;

        // 4. Forward through all layers, collecting hidden states
        let mut collected: HashMap<usize, Tensor> = HashMap::new();
        for i in 0..cfg.num_layers {
            hidden = self.layer_forward(i, &hidden, &pe_cos, &pe_sin, &attn_mask)?;
            if cfg.extract_layers.contains(&i) {
                collected.insert(i, hidden.clone());
            }
        }

        // 5. Stack extracted hidden states: [1, seq_len, 3 * hidden_size]
        //    Order: layer 8, layer 17, layer 26
        //    We extract intermediate states BEFORE final norm, matching the
        //    PyTorch reference behavior.
        let selected: Vec<Tensor> = cfg
            .extract_layers
            .iter()
            .map(|&idx| {
                collected.remove(&idx).ok_or_else(|| {
                    flame_core::Error::InvalidInput(format!(
                        "Extract layer {idx} not collected — model has {} layers",
                        cfg.num_layers
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;

        // Stack on dim 1: [1, 3, seq_len, hidden_size]
        let stacked = Tensor::stack(&selected, 1)?;
        let stacked_dims = stacked.shape().dims();
        let b = stacked_dims[0];
        let num_extracts = stacked_dims[1];
        let s = stacked_dims[2];
        let d = stacked_dims[3];

        // Permute to [1, seq_len, 3, hidden_size] then reshape to [1, seq_len, 3*hidden_size]
        stacked
            .permute(&[0, 2, 1, 3])?
            .reshape(&[b, s, num_extracts * d])
    }

    /// Get the expected output hidden dimension (3 * hidden_size for Klein).
    pub fn output_dim(&self) -> usize {
        self.config.extract_layers.len() * self.config.hidden_size
    }

    /// Get a reference to the config.
    pub fn config(&self) -> &Qwen3Config {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Weight key listing (for documentation and validation)
// ---------------------------------------------------------------------------

/// Return all expected weight keys for a Qwen3 model with the given layer count.
///
/// Useful for validating loaded safetensors against expected keys.
pub fn expected_weight_keys(num_layers: usize) -> Vec<String> {
    let mut keys = vec!["model.embed_tokens.weight".to_string()];

    for i in 0..num_layers {
        let p = format!("model.layers.{i}");
        keys.extend([
            format!("{p}.self_attn.q_proj.weight"),
            format!("{p}.self_attn.k_proj.weight"),
            format!("{p}.self_attn.v_proj.weight"),
            format!("{p}.self_attn.o_proj.weight"),
            format!("{p}.self_attn.q_norm.weight"),
            format!("{p}.self_attn.k_norm.weight"),
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
        let cfg = Qwen3Config::default();
        assert_eq!(cfg.vocab_size, 151936);
        assert_eq!(cfg.hidden_size, 2560);
        assert_eq!(cfg.num_layers, 36);
        assert_eq!(cfg.num_heads, 32);
        assert_eq!(cfg.num_kv_heads, 8);
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.extract_layers, vec![8, 17, 26]);
        assert_eq!(cfg.rms_norm_eps, 1e-6);
        assert!((cfg.rope_theta - 1_000_000.0).abs() < 1.0);
    }

    #[test]
    fn test_gqa_rep_count() {
        let cfg = Qwen3Config::default();
        // 32 Q heads / 8 KV heads = 4 repeats
        assert_eq!(cfg.num_heads / cfg.num_kv_heads, 4);
    }

    #[test]
    fn test_output_dim() {
        let cfg = Qwen3Config::default();
        // 3 extract layers * 2560 = 7680
        let expected = cfg.extract_layers.len() * cfg.hidden_size;
        assert_eq!(expected, 7680);
    }

    #[test]
    fn test_expected_weight_keys_count() {
        let keys = expected_weight_keys(36);
        // 1 (embed) + 36 * 11 (per-layer) + 1 (final norm) = 398
        assert_eq!(keys.len(), 1 + 36 * 11 + 1);
    }

    #[test]
    fn test_expected_weight_keys_format() {
        let keys = expected_weight_keys(2);
        assert!(keys.contains(&"model.embed_tokens.weight".to_string()));
        assert!(keys.contains(&"model.layers.0.self_attn.q_proj.weight".to_string()));
        assert!(keys.contains(&"model.layers.0.self_attn.k_proj.weight".to_string()));
        assert!(keys.contains(&"model.layers.0.self_attn.v_proj.weight".to_string()));
        assert!(keys.contains(&"model.layers.0.self_attn.o_proj.weight".to_string()));
        assert!(keys.contains(&"model.layers.0.self_attn.q_norm.weight".to_string()));
        assert!(keys.contains(&"model.layers.0.self_attn.k_norm.weight".to_string()));
        assert!(keys.contains(&"model.layers.0.mlp.gate_proj.weight".to_string()));
        assert!(keys.contains(&"model.layers.0.mlp.up_proj.weight".to_string()));
        assert!(keys.contains(&"model.layers.0.mlp.down_proj.weight".to_string()));
        assert!(keys.contains(&"model.layers.0.input_layernorm.weight".to_string()));
        assert!(keys.contains(&"model.layers.0.post_attention_layernorm.weight".to_string()));
        assert!(keys.contains(&"model.layers.1.self_attn.q_proj.weight".to_string()));
        assert!(keys.contains(&"model.norm.weight".to_string()));
    }

    #[test]
    fn test_klein_extract_layers_within_bounds() {
        // All extract layers must be < 36 (default layer count)
        for &layer in &KLEIN_EXTRACT_LAYERS {
            assert!(layer < 36, "Extract layer {layer} >= 36");
        }
    }

    #[test]
    fn test_rope_half_dim() {
        let cfg = Qwen3Config::default();
        // RoPE operates on half the head dim
        assert_eq!(cfg.head_dim / 2, 64);
    }
}
