//! Qwen3 text encoder for Klein/ZImage inference using flame-core tensor ops.
//!
//! Loads a Qwen3 causal LM from safetensors and extracts hidden states at
//! layers [8, 17, 26] (0-indexed), stacking them along the hidden dimension
//! to produce Klein's joint_attention_dim embedding:
//!   - Klein 4B: hidden_size=2560, joint_dim = 3 * 2560 = 7680
//!   - Klein 9B: hidden_size=4096, joint_dim = 3 * 4096 = 12288
//!
//! For ZImage, use extract_layers = [26] (single layer) to get [1, seq_len, 2560].
//!
//! Architecture per layer:
//!   input -> RMSNorm -> QKV proj -> QK norm -> RoPE -> GQA attention -> residual
//!         -> RMSNorm -> gate+up -> SiLU(gate)*up -> down -> residual
//!
//! All BF16, F32 only for RoPE angles and normalization internals.
//!
//! Tokenization: use the `tokenizers` crate with the model's `tokenizer.json`.

use flame_core::attention::sdpa as flame_sdpa;
// RMSNorm used via flame_core::cuda_ops_bf16::rms_norm_bf16 directly (no object allocation)
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

impl Qwen3Config {
    /// Config for ZImage: single hidden state from layer 26.
    pub fn zimage() -> Self {
        Self {
            extract_layers: vec![26],
            ..Self::default()
        }
    }

    /// Config for the **text branch** of Qwen3-VL (used by NucleusAI/Nucleus-Image).
    ///
    /// Architecturally identical to the Qwen3 text model (same `q_proj`/
    /// `k_proj`/`v_proj`/`o_proj`/`q_norm`/`k_norm` plus SwiGLU MLP). The
    /// only differences vs `Qwen3Config::default()`:
    /// - `hidden_size` 4096 (vs 2560)
    /// - `intermediate_size` 12288 (vs 6912)
    /// - `rope_theta` 5_000_000.0 (vs 1_000_000.0)
    ///
    /// MRoPE caveat: Qwen3-VL uses MRoPE with `mrope_section=[24,20,20]`
    /// and `mrope_interleaved=True` for image-aware positions. For
    /// **text-only** inputs (no image tokens), `position_ids` has shape
    /// `(B, S)` and is broadcast to all 3 axes equally, so the per-axis
    /// `freqs[t]==freqs[h]==freqs[w]` and `apply_interleaved_mrope`'s
    /// stride-3 overwrite is a no-op. The remaining math collapses to
    /// standard HF half-split 1D RoPE — exactly what `Qwen3Encoder` does.
    /// So this config is correct for T2I prompt encoding without
    /// modification; vision-tower path is intentionally out of scope.
    ///
    /// `extract_layers` defaults to the final layer (`num_layers - 1`).
    /// Use `Qwen3Encoder::output_dim()` to read the resulting hidden dim.
    pub fn qwen3_vl_text() -> Self {
        Self {
            vocab_size: 151_936,
            hidden_size: 4096,
            num_layers: 36,
            intermediate_size: 12288,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            rms_norm_eps: 1e-6,
            rope_theta: 5_000_000.0,
            extract_layers: vec![35],
        }
    }
}

/// Qwen3 text encoder — pure flame-core implementation.
///
/// Holds all weights as a flat `HashMap<String, Tensor>` and runs the
/// forward pass by indexing into it directly.
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
        // Pre-transpose all 2D weight matrices [out, in] -> [in, out]
        // Eliminates ~200 GPU transpose ops per encode call.
        let mut weights = weights;
        let keys: Vec<String> = weights.keys().cloned().collect();
        for key in &keys {
            if key.ends_with(".weight") && !key.contains("layernorm") && !key.contains("norm") && !key.contains("embed") {
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

    /// Matmul for [B, N, C] x [C, out] -> [B, N, out].
    /// Weight is already pre-transposed at construction time.
    fn linear_3d(x: &Tensor, weight_t: &Tensor) -> Result<Tensor> {
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

        let pos = Tensor::arange(0.0, seq_len as f32, 1.0, device.clone())?;

        let freq_idx = Tensor::arange(0.0, head_dim as f32, 2.0, device.clone())?;
        let log_theta = (theta as f32).ln();
        let scale = -log_theta / head_dim as f32;
        let log_freqs = freq_idx.mul_scalar(scale)?.exp()?;

        let pos_col = pos.reshape(&[seq_len, 1])?;
        let freq_row = log_freqs.reshape(&[1, half])?;
        let angles = pos_col.matmul(&freq_row)?;

        let cos = angles.cos()?.unsqueeze(0)?.unsqueeze(0)?.to_dtype(DType::BF16)?;
        let sin = angles.sin()?.unsqueeze(0)?.unsqueeze(0)?.to_dtype(DType::BF16)?;
        Ok((cos, sin))
    }

    /// Apply rotary position embeddings to a single tensor.
    ///
    /// `x`: `[B, H, N, D]`, `pe_cos`/`pe_sin`: `[1, 1, N, D/2]`.
    /// Fused RoPE — single CUDA kernel, no intermediates.
    fn apply_rope_single(
        x: &Tensor,
        pe_cos: &Tensor,
        pe_sin: &Tensor,
    ) -> Result<Tensor> {
        // Qwen3 uses half-split RoPE (HF rotate_half convention)
        flame_core::bf16_ops::rope_halfsplit_bf16(x, pe_cos, pe_sin)
    }

    // -----------------------------------------------------------------------
    // GQA head repeat
    // -----------------------------------------------------------------------

    /// Repeat KV heads to match Q head count for GQA.
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

    /// Build a combined causal + padding mask `[1, 1, seq_len, seq_len]` in BF16.
    ///
    /// `real_len`: number of real (non-pad) tokens at the start of the sequence.
    /// Matches HuggingFace's causal mask convention:
    /// - Causal: position i can only attend to positions j <= i
    /// - Pad columns are always 0 (nobody attends TO pad tokens)
    /// - Pad rows CAN attend to real tokens (matching HF behavior)
    fn build_causal_mask(
        seq_len: usize,
        real_len: usize,
        device: &Arc<CudaDevice>,
    ) -> Result<Tensor> {
        let mut data = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                // Causal: j <= i, AND j must be a real token (not pad)
                // Any row (including pad rows) can attend to real columns
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
    /// Uses flame_core::cuda_ops_bf16::rms_norm_bf16 directly — no object allocation.
    fn rms_norm_apply(
        x: &Tensor,
        weight: &Tensor,
        eps: f32,
        _device: &Arc<CudaDevice>,
    ) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let hidden = *dims.last().unwrap();
        let batch: usize = dims[..dims.len() - 1].iter().product();
        let x_2d = x.reshape(&[batch, hidden])?;
        let out_2d = flame_core::cuda_ops_bf16::rms_norm_bf16(&x_2d, Some(weight), eps)?;
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

        let norm_w = self.w(&format!("{prefix}.input_layernorm.weight"))?;
        let normed = Self::rms_norm_apply(hidden, norm_w, cfg.rms_norm_eps, &self.device)?;

        let q_w = self.w(&format!("{prefix}.self_attn.q_proj.weight"))?;
        let k_w = self.w(&format!("{prefix}.self_attn.k_proj.weight"))?;
        let v_w = self.w(&format!("{prefix}.self_attn.v_proj.weight"))?;

        let q = Self::linear_3d(&normed, q_w)?;
        let k = Self::linear_3d(&normed, k_w)?;
        let v = Self::linear_3d(&normed, v_w)?;

        let mut q = q.reshape(&[b, n, h, d])?.permute(&[0, 2, 1, 3])?;
        let mut k = k.reshape(&[b, n, h_kv, d])?.permute(&[0, 2, 1, 3])?;
        let v = v.reshape(&[b, n, h_kv, d])?.permute(&[0, 2, 1, 3])?;

        // QK norm (per-head RMSNorm) — direct kernel, no object allocation
        let q_norm_key = format!("{prefix}.self_attn.q_norm.weight");
        let k_norm_key = format!("{prefix}.self_attn.k_norm.weight");
        if let (Some(q_norm_w), Some(k_norm_w)) =
            (self.weights.get(&q_norm_key), self.weights.get(&k_norm_key))
        {
            let q_flat = q.reshape(&[b * h * n, d])?;
            let q_normed = flame_core::cuda_ops_bf16::rms_norm_bf16(&q_flat, Some(q_norm_w), cfg.rms_norm_eps)?;
            q = q_normed.reshape(&[b, h, n, d])?;

            let k_flat = k.reshape(&[b * h_kv * n, d])?;
            let k_normed = flame_core::cuda_ops_bf16::rms_norm_bf16(&k_flat, Some(k_norm_w), cfg.rms_norm_eps)?;
            k = k_normed.reshape(&[b, h_kv, n, d])?;
        }

        let q = Self::apply_rope_single(&q, pe_cos, pe_sin)?;
        let k = Self::apply_rope_single(&k, pe_cos, pe_sin)?;

        let k = Self::repeat_kv(&k, n_rep)?;
        let v = Self::repeat_kv(&v, n_rep)?;

        let attn_out = flame_sdpa(&q, &k, &v, Some(attn_mask))?;

        let attn_out = attn_out.permute(&[0, 2, 1, 3])?.reshape(&[b, n, h * d])?;

        let o_w = self.w(&format!("{prefix}.self_attn.o_proj.weight"))?;
        let attn_out = Self::linear_3d(&attn_out, o_w)?;

        let hidden = hidden.add(&attn_out)?;

        // --- MLP ---

        let post_norm_w = self.w(&format!("{prefix}.post_attention_layernorm.weight"))?;
        let normed2 = Self::rms_norm_apply(&hidden, post_norm_w, cfg.rms_norm_eps, &self.device)?;

        let gate_w = self.w(&format!("{prefix}.mlp.gate_proj.weight"))?;
        let up_w = self.w(&format!("{prefix}.mlp.up_proj.weight"))?;
        let down_w = self.w(&format!("{prefix}.mlp.down_proj.weight"))?;

        let gate = Self::linear_3d(&normed2, gate_w)?;
        let up = Self::linear_3d(&normed2, up_w)?;
        let mlp_out = gate.silu()?.mul(&up)?;
        let mlp_out = Self::linear_3d(&mlp_out, down_w)?;

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

    /// Run forward pass and return stacked hidden states.
    ///
    /// # Returns
    /// Tensor of shape `[1, seq_len, extract_layers.len() * hidden_size]` —
    /// hidden states from configured extract layers stacked along the last dim.
    ///
    /// For Klein (default): layers [8, 17, 26] → `[1, seq_len, 7680]`
    /// For ZImage: layer [26] → `[1, seq_len, 2560]`
    pub fn encode(&self, token_ids: &[i32]) -> Result<Tensor> {
        let cfg = &self.config;
        let seq_len = token_ids.len();

        // Detect real (non-pad) token count. Pad token = 151643.
        let pad_id = 151643i32;
        let real_len = token_ids.iter().position(|&id| id == pad_id).unwrap_or(seq_len);

        let mut hidden = self.embed_tokens(token_ids)?;

        let (pe_cos, pe_sin) =
            Self::build_rope_1d(seq_len, cfg.head_dim, cfg.rope_theta, &self.device)?;

        let attn_mask = Self::build_causal_mask(seq_len, real_len, &self.device)?;

        let mut collected: HashMap<usize, Tensor> = HashMap::new();
        for i in 0..cfg.num_layers {
            hidden = self.layer_forward(i, &hidden, &pe_cos, &pe_sin, &attn_mask)?;
            if cfg.extract_layers.contains(&i) {
                collected.insert(i, hidden.clone());
            }
        }

        // Single extract layer: return directly without stacking
        if cfg.extract_layers.len() == 1 {
            let idx = cfg.extract_layers[0];
            return collected.remove(&idx).ok_or_else(|| {
                flame_core::Error::InvalidInput(format!(
                    "Extract layer {idx} not collected — model has {} layers",
                    cfg.num_layers
                ))
            });
        }

        // Multiple extract layers: stack along hidden dim
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

        let stacked = Tensor::stack(&selected, 1)?;
        let stacked_dims = stacked.shape().dims();
        let b = stacked_dims[0];
        let num_extracts = stacked_dims[1];
        let s = stacked_dims[2];
        let d = stacked_dims[3];

        stacked
            .permute(&[0, 2, 1, 3])?
            .reshape(&[b, s, num_extracts * d])
    }

    /// Get the expected output hidden dimension.
    pub fn output_dim(&self) -> usize {
        self.config.extract_layers.len() * self.config.hidden_size
    }

    /// Get a reference to the config.
    pub fn config(&self) -> &Qwen3Config {
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

/// Return all expected weight keys for a Qwen3 model with the given layer count.
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
    }

    #[test]
    fn test_zimage_config() {
        let cfg = Qwen3Config::zimage();
        assert_eq!(cfg.extract_layers, vec![26]);
        assert_eq!(cfg.hidden_size, 2560);
    }

    #[test]
    fn test_output_dim_klein() {
        let cfg = Qwen3Config::default();
        assert_eq!(cfg.extract_layers.len() * cfg.hidden_size, 7680);
    }

    #[test]
    fn test_qwen3_vl_text_config() {
        let cfg = Qwen3Config::qwen3_vl_text();
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.intermediate_size, 12288);
        assert_eq!(cfg.num_layers, 36);
        assert_eq!(cfg.rope_theta, 5_000_000.0);
        assert_eq!(cfg.extract_layers, vec![35]);
    }

    /// Phase 6.7: parity vs `transformers.Qwen3VLTextModel` on a small toy
    /// config. Confirms the existing `Qwen3Encoder` correctly handles a
    /// Qwen3-VL text checkpoint — MRoPE collapses to 1D RoPE for text-only
    /// inputs, and the layer / norm layout matches Qwen3 exactly.
    ///
    /// Generate the fixture with:
    /// ```
    /// python3 inference-flame/scripts/generate_qwen3vl_text_small.py
    /// ```
    #[test]
    fn qwen3_vl_text_parity_vs_transformers() {
        let fixture_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/pytorch_fixtures/nucleus/qwen3vl_text_small.safetensors");
        if !fixture_path.exists() {
            eprintln!(
                "fixture missing — generate with `python3 scripts/generate_qwen3vl_text_small.py`"
            );
            return;
        }

        let device = flame_core::CudaDevice::new(0).expect("cuda dev 0");
        let device: std::sync::Arc<flame_core::CudaDevice> = device;

        let mut map = flame_core::serialization::load_file(&fixture_path, &device)
            .expect("load fixture");

        // Pull off inputs, expected, and the final norm before we hand the
        // remaining model weights to the encoder.
        let input_ids_tensor = map
            .remove("inputs.input_ids")
            .expect("fixture missing inputs.input_ids");
        let expected = map
            .remove("expected.last_hidden_state")
            .expect("fixture missing expected.last_hidden_state");
        let final_norm_w = map
            .remove("model.norm.weight")
            .expect("fixture missing model.norm.weight");

        // Convert i32 token IDs to host slice (fixture stored as int32 cast
        // from torch.long). `Tensor::to_vec_f32` doesn't apply for I32, so we
        // peek at the underlying f32 representation that load_file produced.
        // The fixture script saved torch.int32; flame's load_file reads it as
        // DType::I32 (f32-bytes-relabeled per FLAME_CONVENTIONS). So
        // to_vec_f32 returns the int values as f32. Cast back to i32.
        let token_ids_f32 = input_ids_tensor
            .to_vec_f32()
            .expect("input_ids to_vec_f32");
        let token_ids: Vec<i32> = token_ids_f32.into_iter().map(|x| x as i32).collect();

        let cfg = Qwen3Config {
            vocab_size: 128,
            hidden_size: 128,
            num_layers: 4,
            intermediate_size: 384,
            num_heads: 4,
            num_kv_heads: 1,
            head_dim: 32,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            extract_layers: vec![3], // last layer
        };

        // Drop fixture metadata keys before constructing the encoder.
        let _ = map
            .keys()
            .filter(|k| k.starts_with("meta."))
            .cloned()
            .collect::<Vec<_>>()
            .into_iter()
            .map(|k| map.remove(&k))
            .count();

        let encoder = Qwen3Encoder::new(map, cfg.clone(), device.clone());
        let last_layer_pre_norm = encoder.encode(&token_ids).expect("encode");

        // Apply final RMSNorm (model.norm) to match transformers' last_hidden_state.
        let dims = last_layer_pre_norm.shape().dims().to_vec();
        let (b, s, h) = (dims[0], dims[1], dims[2]);
        let flat = last_layer_pre_norm.reshape(&[b * s, h]).expect("flatten");
        let normed = flame_core::cuda_ops_bf16::rms_norm_bf16(
            &flat,
            Some(&final_norm_w),
            cfg.rms_norm_eps,
        )
        .expect("final rms_norm");
        let normed = normed.reshape(&[b, s, h]).expect("reshape");

        let got = normed.to_vec_f32().expect("got vec");
        let exp = expected.to_vec_f32().expect("expected vec");
        assert_eq!(got.len(), exp.len(), "shape mismatch");

        let mut max_abs = 0f32;
        let mut sum_abs = 0f64;
        for (a, b) in got.iter().zip(exp.iter()) {
            let d = (a - b).abs();
            if d > max_abs {
                max_abs = d;
            }
            sum_abs += d as f64;
        }
        let mean_abs = (sum_abs / got.len() as f64) as f32;
        eprintln!(
            "qwen3_vl_text_small_parity: max_abs={max_abs:.4e} mean_abs={mean_abs:.4e}"
        );
        // BF16 scatter-add accumulation through 4 layers — mostly tight, max
        // can spike on near-zero outputs.
        assert!(mean_abs < 1e-3, "mean_abs {mean_abs} exceeds 1e-3");
        assert!(max_abs < 1e-1, "max_abs {max_abs} exceeds 1e-1");
    }

    #[test]
    fn test_output_dim_zimage() {
        let cfg = Qwen3Config::zimage();
        assert_eq!(cfg.extract_layers.len() * cfg.hidden_size, 2560);
    }

    #[test]
    fn test_expected_weight_keys_count() {
        let keys = expected_weight_keys(36);
        // 1 (embed) + 36 * 11 (per-layer) + 1 (final norm) = 398
        assert_eq!(keys.len(), 1 + 36 * 11 + 1);
    }

    #[test]
    fn test_klein_extract_layers_within_bounds() {
        for &layer in &KLEIN_EXTRACT_LAYERS {
            assert!(layer < 36, "Extract layer {layer} >= 36");
        }
    }
}
