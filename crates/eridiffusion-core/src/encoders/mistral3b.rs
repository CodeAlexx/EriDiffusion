//! Mistral-3 3B text encoder for ERNIE-Image — pure flame-core, all weights on GPU.
//!
//! Loads the Mistral-3 3B language model from ERNIE-Image's text_encoder checkpoint
//! and returns hidden_states[-2] (output after layer 24, 0-indexed) as the text
//! conditioning for the ERNIE-Image DiT:
//!   output shape = [1, seq_len, 3072]
//!
//! Architecture per layer:
//!   input -> RMSNorm -> QKV proj -> YaRN RoPE -> GQA attention -> residual
//!         -> RMSNorm -> gate+up -> SiLU(gate)*up -> down -> residual
//!
//! Key differences from the 24B Mistral encoder (mistral_encoder.rs):
//!   - hidden_size=3072 (not 5120), 26 layers (not 40), intermediate=9216 (not 32768)
//!   - YaRN RoPE (factor=16, original_max_pos=16384) instead of standard RoPE
//!   - Returns hidden_states[-2] (single tensor) instead of stacking 3 layers
//!   - No BlockOffloader needed — 7.2GB fits in 24GB VRAM
//!   - Skips vision_tower.* and multi_modal_projector.* keys during loading
//!
//! Weight prefix: `language_model.model.` (stripped on load).

use flame_core::attention::sdpa as flame_sdpa;
use flame_core::serialization::load_file_filtered;
use flame_core::{CudaDevice, DType, Result, Shape, Tensor};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// Weight key prefix in the ERNIE-Image text encoder safetensors.
const KEY_PREFIX: &str = "language_model.model.";

/// Prefixes to skip during loading (vision encoder + multimodal projector).
const SKIP_PREFIXES: [&str; 2] = ["vision_tower.", "multi_modal_projector."];

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Mistral-3 3B config, matching the ERNIE-Image text encoder checkpoint.
#[derive(Debug, Clone)]
pub struct Mistral3bConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f64,
    /// YaRN scaling factor.
    pub rope_factor: f64,
    /// Original max position embeddings before YaRN scaling.
    pub rope_original_max_pos: usize,
    /// YaRN beta_fast parameter (controls high-frequency boundary).
    pub rope_beta_fast: f64,
    /// YaRN beta_slow parameter (controls low-frequency boundary).
    pub rope_beta_slow: f64,
    /// Return hidden_states[-2]: 0-indexed layer whose output we return.
    /// For 26 layers: layer 24 (0-indexed) = hidden_states[25] = second-to-last.
    pub extract_layer: usize,
    pub max_seq_len: usize,
}

impl Default for Mistral3bConfig {
    fn default() -> Self {
        Self {
            vocab_size: 131072,
            hidden_size: 3072,
            num_layers: 26,
            intermediate_size: 9216,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            rms_norm_eps: 1e-5,
            rope_theta: 1_000_000.0,
            rope_factor: 16.0,
            rope_original_max_pos: 16384,
            rope_beta_fast: 32.0,
            rope_beta_slow: 1.0,
            extract_layer: 24, // hidden_states[-2] for 26-layer model
            max_seq_len: 512,
        }
    }
}

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

/// Mistral-3 3B encoder — all weights resident on GPU (7.2GB BF16).
pub struct Mistral3bEncoder {
    weights: HashMap<String, Tensor>,
    config: Mistral3bConfig,
    device: Arc<CudaDevice>,
}

impl Mistral3bEncoder {
    /// Load from a single BF16 safetensors file.
    ///
    /// Filters out vision_tower and multi_modal_projector keys.
    /// Strips `language_model.model.` prefix and pre-transposes linear weights.
    pub fn load(safetensors_path: &str, device: &Arc<CudaDevice>) -> Result<Self> {
        let config = Mistral3bConfig::default();

        let raw_weights = load_file_filtered(Path::new(safetensors_path), device, |key| {
            // Only load language_model.model.* keys
            key.starts_with(KEY_PREFIX)
                && !SKIP_PREFIXES.iter().any(|p| key.starts_with(p))
        })?;

        // Strip prefix
        let mut weights: HashMap<String, Tensor> = raw_weights
            .into_iter()
            .filter_map(|(k, v)| {
                k.strip_prefix(KEY_PREFIX).map(|s| (s.to_string(), v))
            })
            .collect();

        log::info!(
            "[Mistral3b] Loaded {} weights (prefix-stripped)",
            weights.len()
        );

        // Pre-transpose all 2D linear weights [out, in] -> [in, out]
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

        // Validate essential weights exist
        let required = ["embed_tokens.weight", "norm.weight"];
        for key in &required {
            if !weights.contains_key(*key) {
                return Err(flame_core::Error::InvalidInput(format!(
                    "Missing required weight: {KEY_PREFIX}{key}"
                )));
            }
        }

        log::info!(
            "[Mistral3b] Ready: {} layers, hidden={}, YaRN factor={}",
            config.num_layers,
            config.hidden_size,
            config.rope_factor
        );

        Ok(Self {
            weights,
            config,
            device: device.clone(),
        })
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn w(&self, key: &str) -> Result<&Tensor> {
        self.weights.get(key).ok_or_else(|| {
            flame_core::Error::InvalidInput(format!("Missing weight: {key}"))
        })
    }

    /// Matmul for [B, N, C] x [C, out] -> [B, N, out].
    /// Weight is already pre-transposed.
    fn linear_3d(x: &Tensor, weight_t: &Tensor) -> Result<Tensor> {
        let shape = x.shape().dims().to_vec();
        let b = shape[0];
        let n = shape[1];
        let c = shape[2];
        let x_2d = x.reshape(&[b * n, c])?;
        let out_2d = x_2d.matmul(weight_t)?;
        let out_dim = out_2d.shape().dims()[1];
        out_2d.reshape(&[b, n, out_dim])
    }

    fn rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let hidden = *dims.last().unwrap();
        let batch: usize = dims[..dims.len() - 1].iter().product();
        let x_2d = x.reshape(&[batch, hidden])?;
        let out_2d = flame_core::cuda_ops_bf16::rms_norm_bf16(&x_2d, Some(weight), eps)?;
        out_2d.reshape(&dims)
    }

    /// Build YaRN-scaled RoPE cos/sin tables [1, 1, seq_len, head_dim/2].
    ///
    /// YaRN applies frequency-dependent scaling:
    ///   - High frequencies (short wavelength): keep original
    ///   - Low frequencies (long wavelength): scale by 1/factor
    ///   - Medium frequencies: smooth interpolation between the two
    fn build_yarn_rope(
        seq_len: usize,
        config: &Mistral3bConfig,
        device: &Arc<CudaDevice>,
    ) -> Result<(Tensor, Tensor)> {
        let dim = config.head_dim;
        let half = dim / 2;
        let theta = config.rope_theta;
        let factor = config.rope_factor;
        let orig_max_pos = config.rope_original_max_pos as f64;
        let two_pi = 2.0 * std::f64::consts::PI;

        // Compute boundary factors from beta parameters
        let low_freq_factor = (orig_max_pos / (two_pi / config.rope_beta_fast)).round().max(1.0);
        let high_freq_factor = (orig_max_pos / (two_pi / config.rope_beta_slow)).round().max(1.0);

        let low_freq_wavelen = orig_max_pos / low_freq_factor;
        let high_freq_wavelen = orig_max_pos / high_freq_factor;

        // Build scaled inv_freq on CPU, then upload
        let mut inv_freqs = vec![0.0f32; half];
        for i in 0..half {
            let base_freq = 1.0 / theta.powf(2.0 * i as f64 / dim as f64);
            let wavelength = two_pi / base_freq;

            let scaled_freq = if wavelength < high_freq_wavelen {
                // High frequency: keep original
                base_freq
            } else if wavelength > low_freq_wavelen {
                // Low frequency: divide by factor
                base_freq / factor
            } else {
                // Smooth interpolation
                let smooth = (orig_max_pos / wavelength - low_freq_factor)
                    / (high_freq_factor - low_freq_factor);
                (1.0 - smooth) * (base_freq / factor) + smooth * base_freq
            };
            inv_freqs[i] = scaled_freq as f32;
        }

        let freq_tensor =
            Tensor::from_vec(inv_freqs, Shape::from_dims(&[1, half]), device.clone())?;
        let pos = Tensor::arange(0.0, seq_len as f32, 1.0, device.clone())?;
        let pos_col = pos.reshape(&[seq_len, 1])?;
        let angles = pos_col.matmul(&freq_tensor)?;

        let cos = angles.cos()?.unsqueeze(0)?.unsqueeze(0)?.to_dtype(DType::BF16)?;
        let sin = angles.sin()?.unsqueeze(0)?.unsqueeze(0)?.to_dtype(DType::BF16)?;
        Ok((cos, sin))
    }

    fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        flame_core::bf16_ops::rope_halfsplit_bf16(x, cos, sin)
    }

    fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
        if n_rep == 1 {
            return Ok(x.clone());
        }
        let dims = x.shape().dims();
        let (b, h_kv, n, d) = (dims[0], dims[1], dims[2], dims[3]);
        let copies: Vec<Tensor> = (0..n_rep).map(|_| x.clone()).collect();
        let stacked = Tensor::stack(&copies, 2)?;
        stacked.reshape(&[b, h_kv * n_rep, n, d])
    }

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
        Tensor::from_vec(
            data,
            Shape::from_dims(&[1, 1, seq_len, seq_len]),
            device.clone(),
        )?
        .to_dtype(DType::BF16)
    }

    // -----------------------------------------------------------------------
    // Single transformer layer
    // -----------------------------------------------------------------------

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
        let prefix = format!("layers.{layer_idx}");

        let dims = hidden.shape().dims().to_vec();
        let b = dims[0];
        let n = dims[1];

        let w = |suffix: &str| -> Result<&Tensor> {
            let key = format!("{prefix}.{suffix}");
            self.weights.get(&key).ok_or_else(|| {
                flame_core::Error::InvalidInput(format!("Missing weight: {key}"))
            })
        };

        // --- Self-attention ---

        let normed = Self::rms_norm(hidden, w("input_layernorm.weight")?, cfg.rms_norm_eps)?;

        let q = Self::linear_3d(&normed, w("self_attn.q_proj.weight")?)?;
        let k = Self::linear_3d(&normed, w("self_attn.k_proj.weight")?)?;
        let v = Self::linear_3d(&normed, w("self_attn.v_proj.weight")?)?;

        let q = q.reshape(&[b, n, h, d])?.permute(&[0, 2, 1, 3])?;
        let k = k.reshape(&[b, n, h_kv, d])?.permute(&[0, 2, 1, 3])?;
        let v = v.reshape(&[b, n, h_kv, d])?.permute(&[0, 2, 1, 3])?;

        // No QK norm in Mistral-3

        let q = Self::apply_rope(&q, pe_cos, pe_sin)?;
        let k = Self::apply_rope(&k, pe_cos, pe_sin)?;

        let k = Self::repeat_kv(&k, n_rep)?;
        let v = Self::repeat_kv(&v, n_rep)?;

        let attn_out = flame_sdpa(&q, &k, &v, Some(attn_mask))?;
        let attn_out = attn_out.permute(&[0, 2, 1, 3])?.reshape(&[b, n, h * d])?;
        let attn_out = Self::linear_3d(&attn_out, w("self_attn.o_proj.weight")?)?;
        let hidden = hidden.add(&attn_out)?;

        // --- MLP ---

        let normed2 =
            Self::rms_norm(&hidden, w("post_attention_layernorm.weight")?, cfg.rms_norm_eps)?;

        let gate = Self::linear_3d(&normed2, w("mlp.gate_proj.weight")?)?;
        let up = Self::linear_3d(&normed2, w("mlp.up_proj.weight")?)?;
        let mlp_out = gate.silu()?.mul(&up)?;
        let mlp_out = Self::linear_3d(&mlp_out, w("mlp.down_proj.weight")?)?;

        hidden.add(&mlp_out)
    }

    // -----------------------------------------------------------------------
    // Full forward pass
    // -----------------------------------------------------------------------

    /// Encode tokens and return hidden_states[-2] (second-to-last hidden state).
    ///
    /// For a 26-layer model, hidden_states[-2] = output of layer 24 (0-indexed).
    ///
    /// Pads with id=0 (legacy default — kept for back-compat with non-ERNIE callers).
    /// **For ERNIE prefer [`Self::encode_with_pad`] with pad_id=11**, the value
    /// declared in `text_encoder/config.json` (HF tokenizer.pad_token_id).
    ///
    /// Returns: `[1, seq_len, 3072]` BF16.
    pub fn encode(&self, token_ids: &[i32], max_len: usize) -> Result<Tensor> {
        self.encode_with_pad(token_ids, max_len, 0)
    }

    /// Same as [`Self::encode`] but takes an explicit pad token id.
    /// ERNIE callers should pass 11 (per `text_encoder/config.json:32`).
    pub fn encode_with_pad(
        &self,
        token_ids: &[i32],
        max_len: usize,
        pad_id: i32,
    ) -> Result<Tensor> {
        let cfg = &self.config;

        // Pad or truncate to max_len
        let mut ids = token_ids.to_vec();
        ids.truncate(max_len);
        let real_len = ids.len();
        while ids.len() < max_len {
            ids.push(pad_id);
        }
        let seq_len = ids.len();

        log::info!(
            "[Mistral3b] Encoding: seq_len={}, real_tokens={}",
            seq_len,
            real_len
        );

        // 1. Embed tokens
        let embed_w = self.w("embed_tokens.weight")?;
        let ids_tensor = Tensor::from_vec(
            ids.iter().map(|&id| id as f32).collect(),
            Shape::from_dims(&[seq_len]),
            self.device.clone(),
        )?
        .to_dtype(DType::I32)?;
        let mut hidden = embed_w.index_select0(&ids_tensor)?.unsqueeze(0)?;

        // 2. Build YaRN RoPE and causal mask
        let (pe_cos, pe_sin) = Self::build_yarn_rope(seq_len, cfg, &self.device)?;
        let attn_mask = Self::build_causal_mask(seq_len, real_len, &self.device)?;

        // 3. Forward through 26 layers, capture extract_layer output
        let mut result: Option<Tensor> = None;

        for i in 0..cfg.num_layers {
            hidden = self.layer_forward(i, &hidden, &pe_cos, &pe_sin, &attn_mask)?;

            if i == cfg.extract_layer {
                result = Some(hidden.clone());
                log::info!("[Mistral3b] Captured layer {} output (hidden_states[-2])", i);
            }

            if i % 10 == 0 || i == cfg.num_layers - 1 {
                log::info!("[Mistral3b] Layer {}/{} done", i + 1, cfg.num_layers);
            }
        }

        let output = result.ok_or_else(|| {
            flame_core::Error::InvalidInput(format!(
                "Extract layer {} not reached — model has {} layers",
                cfg.extract_layer, cfg.num_layers
            ))
        })?;

        log::info!("[Mistral3b] Output: {:?}", output.shape());
        Ok(output)
    }

    /// Get config reference.
    pub fn config(&self) -> &Mistral3bConfig {
        &self.config
    }

    /// Expected output dim: 3072.
    pub fn output_dim(&self) -> usize {
        self.config.hidden_size
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = Mistral3bConfig::default();
        assert_eq!(cfg.vocab_size, 131072);
        assert_eq!(cfg.hidden_size, 3072);
        assert_eq!(cfg.num_layers, 26);
        assert_eq!(cfg.num_heads, 32);
        assert_eq!(cfg.num_kv_heads, 8);
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.intermediate_size, 9216);
        assert_eq!(cfg.rms_norm_eps, 1e-5);
        assert_eq!(cfg.rope_theta, 1e6);
        assert_eq!(cfg.rope_factor, 16.0);
        assert_eq!(cfg.rope_original_max_pos, 16384);
    }

    #[test]
    fn test_extract_layer() {
        let cfg = Mistral3bConfig::default();
        // hidden_states[-2] for 26 layers = layer 24 (0-indexed)
        assert_eq!(cfg.extract_layer, 24);
        assert!(cfg.extract_layer < cfg.num_layers);
    }

    #[test]
    fn test_output_dim() {
        let cfg = Mistral3bConfig::default();
        assert_eq!(cfg.hidden_size, 3072);
    }

    #[test]
    fn test_gqa_ratio() {
        let cfg = Mistral3bConfig::default();
        assert_eq!(cfg.num_heads / cfg.num_kv_heads, 4);
    }

    #[test]
    fn test_q_proj_dim() {
        let cfg = Mistral3bConfig::default();
        // q_proj: [num_heads * head_dim, hidden_size] = [4096, 3072]
        assert_eq!(cfg.num_heads * cfg.head_dim, 4096);
        // k_proj: [num_kv_heads * head_dim, hidden_size] = [1024, 3072]
        assert_eq!(cfg.num_kv_heads * cfg.head_dim, 1024);
    }

    #[test]
    fn test_yarn_boundary_factors() {
        let cfg = Mistral3bConfig::default();
        let two_pi = 2.0 * std::f64::consts::PI;
        let orig = cfg.rope_original_max_pos as f64;

        let low_freq_factor = (orig / (two_pi / cfg.rope_beta_fast)).round().max(1.0);
        let high_freq_factor = (orig / (two_pi / cfg.rope_beta_slow)).round().max(1.0);

        // Sanity: low_freq_factor > high_freq_factor (more freqs kept original)
        assert!(low_freq_factor > high_freq_factor);
        // Both should be positive integers
        assert!(low_freq_factor >= 1.0);
        assert!(high_freq_factor >= 1.0);
    }
}
