//! T5-XXL encoder for FLUX 1 — pure flame-core implementation.
//!
//! Produces [B, 512, 4096] hidden states from text input.
//!
//! Architecture (T5 v1.1 XXL encoder-only):
//!   - 24 layers, d_model=4096, 64 heads, d_kv=64
//!   - Gated-GELU FFN: gelu(wi_0(x)) * wi_1(x) → wo
//!   - T5 LayerNorm (RMSNorm variant: no bias, no mean subtraction)
//!   - Relative position bias (computed from layer 0, shared across all layers)
//!   - NO position embeddings (positions encoded via relative bias)
//!   - NO biases on attention or FFN projections
//!
//! Weight key format:
//!   encoder.embed_tokens.weight                              [32128, 4096]
//!   encoder.block.{i}.layer.0.SelfAttention.{q,k,v,o}.weight [4096, 4096]
//!   encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight [32, 64]
//!   encoder.block.{i}.layer.0.layer_norm.weight              [4096]
//!   encoder.block.{i}.layer.1.DenseReluDense.wi_0.weight     [10240, 4096]  (gate)
//!   encoder.block.{i}.layer.1.DenseReluDense.wi_1.weight     [10240, 4096]  (up)
//!   encoder.block.{i}.layer.1.DenseReluDense.wo.weight       [4096, 10240]  (down)
//!   encoder.block.{i}.layer.1.layer_norm.weight              [4096]
//!   encoder.final_layer_norm.weight                          [4096]
//!
//! ⚠️ STANDALONE — does NOT connect to any inference pipeline.

use flame_core::attention::sdpa_with_bias as flame_sdpa_with_bias;
use flame_core::serialization::load_file_filtered;
use flame_core::{CudaDevice, DType, Result, Shape, Tensor};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct T5Config {
    pub vocab_size: usize,
    pub d_model: usize,
    pub num_layers: usize,
    pub d_ff: usize,
    pub num_heads: usize,
    pub d_kv: usize,
    pub relative_attention_num_buckets: usize,
    pub relative_attention_max_distance: usize,
    pub layer_norm_eps: f32,
    pub max_seq_len: usize,
}

impl Default for T5Config {
    fn default() -> Self {
        Self {
            vocab_size: 32128,
            d_model: 4096,
            num_layers: 24,
            d_ff: 10240,
            num_heads: 64,
            d_kv: 64,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            layer_norm_eps: 1e-6,
            max_seq_len: 512,
        }
    }
}

// ---------------------------------------------------------------------------
// T5 Encoder (resident)
// ---------------------------------------------------------------------------

/// T5-XXL encoder. Loads all 220 weights resident on GPU as BF16 (~9.7GB).
///
/// T5-XXL fits on a 24GB card (~9.7GB BF16) before the DiT loads, and we
/// drop it before the DiT BlockOffloader is constructed, so resident loading
/// is safe. No block offloading needed.
pub struct T5Encoder {
    weights: HashMap<String, Tensor>,
    config: T5Config,
    device: Arc<CudaDevice>,
}

impl T5Encoder {
    /// Load T5-XXL from safetensors. Casts every tensor to BF16 (loader
    /// upcasts F16 → F32; BF16 kernels need BF16).
    pub fn load(
        safetensors_path: &str,
        device: &Arc<CudaDevice>,
    ) -> Result<Self> {
        let config = T5Config::default();

        let raw = flame_core::serialization::load_file(
            Path::new(safetensors_path),
            device,
        )?;

        let mut weights: HashMap<String, Tensor> = HashMap::with_capacity(raw.len());
        for (k, v) in raw {
            let t = if v.dtype() == DType::BF16 {
                v
            } else {
                v.to_dtype(DType::BF16)?
            };
            weights.insert(k, t);
        }

        // Some checkpoints use `shared.weight` as the embed table.
        if !weights.contains_key("encoder.embed_tokens.weight") {
            if let Some(sw) = weights.get("shared.weight").cloned() {
                weights.insert("encoder.embed_tokens.weight".to_string(), sw);
            }
        }

        log::info!("[T5] Loaded: {} resident weights (BF16)", weights.len());

        Ok(Self { weights, config, device: device.clone() })
    }

    /// T5 LayerNorm (RMSNorm variant): norm(x) * weight
    /// T5 uses standard RMSNorm without the Gemma (1+weight) formulation.
    fn t5_layer_norm(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let hidden = *dims.last().unwrap();
        let batch: usize = dims[..dims.len() - 1].iter().product();
        let x_2d = x.reshape(&[batch, hidden])?;
        let out = flame_core::cuda_ops_bf16::rms_norm_bf16(&x_2d, Some(weight), eps)?;
        out.reshape(&dims)
    }

    /// Linear without bias: [B, N, C] x [out, in]^T → [B, N, out]
    fn linear(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
        let prof = std::env::var("FLUX1_LIN_PROF").ok().as_deref() == Some("1");
        let dev = x.device();

        if prof { let _ = dev.synchronize(); }
        let t0 = std::time::Instant::now();
        let wt = flame_core::bf16_elementwise::transpose2d_bf16(weight)?;
        if prof { let _ = dev.synchronize(); }
        let trans_us = t0.elapsed().as_micros();

        let shape = x.shape().dims().to_vec();
        let (b, n, c) = (shape[0], shape[1], shape[2]);
        let x_2d = x.reshape(&[b * n, c])?;

        if prof { let _ = dev.synchronize(); }
        let t1 = std::time::Instant::now();
        let out = x_2d.matmul(&wt)?;
        if prof { let _ = dev.synchronize(); }
        let mm_us = t1.elapsed().as_micros();

        if prof {
            static mut COUNT: usize = 0;
            unsafe {
                if COUNT < 8 {
                    eprintln!(
                        "[LIN] x={:?} w={:?} wt_storage={:?}: transpose={}us matmul={}us",
                        x.shape().dims(), weight.shape().dims(), wt.dtype(),
                        trans_us, mm_us
                    );
                    COUNT += 1;
                }
            }
        }

        let out_dim = out.shape().dims()[1];
        out.reshape(&[b, n, out_dim])
    }

    /// Compute T5 relative position bias.
    ///
    /// ## Reference:
    /// ```python
    /// # Bucket positions using logarithmic binning
    /// # relative_attention_bias.weight: [num_buckets, num_heads] = [32, 64]
    /// # Output: [1, num_heads, seq_len, seq_len]
    /// ```
    fn compute_relative_bias(
        bias_weight: &Tensor, // [num_buckets=32, num_heads=64]
        seq_len: usize,
        num_buckets: usize,
        max_distance: usize,
        device: &Arc<CudaDevice>,
    ) -> Result<Tensor> {
        let num_heads = bias_weight.shape().dims()[1];

        // ## HF reference (modeling_t5.py::compute_bias):
        // ```python
        // context_position = torch.arange(query_length)[:, None]
        // memory_position = torch.arange(key_length)[None, :]
        // relative_position = memory_position - context_position   # (Q, K)
        // relative_position_bucket = self._relative_position_bucket(
        //     relative_position,
        //     bidirectional=(not self.is_decoder),   # True for encoder
        //     num_buckets=self.relative_attention_num_buckets,  # 32
        //     max_distance=self.relative_attention_max_distance, # 128
        // )
        // values = self.relative_attention_bias(relative_position_bucket)  # (Q, K, H)
        // values = values.permute([2, 0, 1]).unsqueeze(0)                  # (1, H, Q, K)
        // ```
        //
        // T5.H2 fix: pass ORIGINAL num_buckets (32) to the bucket fn; halving
        // happens inside the bidirectional branch per HF.
        let mut bias_data = vec![0i32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                let relative_position = j as i64 - i as i64;
                let bucket = t5_relative_position_bucket(
                    relative_position,
                    true, // bidirectional (encoder)
                    num_buckets,
                    max_distance,
                );
                bias_data[i * seq_len + j] = bucket as i32;
            }
        }

        // Gather from bias_weight: [seq*seq] indices into [32, 64]
        let indices = Tensor::from_vec(
            bias_data.iter().map(|&i| i as f32).collect(),
            Shape::from_dims(&[seq_len * seq_len]),
            device.clone(),
        )?.to_dtype(DType::I32)?;

        // bias_weight: [32, 64] → gather rows → [seq*seq, 64]
        let gathered = bias_weight.index_select0(&indices)?; // [seq*seq, num_heads]

        // Reshape to [seq, seq, heads] → permute to [1, heads, seq, seq]
        let reshaped = gathered.reshape(&[seq_len, seq_len, num_heads])?;
        let permuted = reshaped.permute(&[2, 0, 1])?; // [heads, seq, seq]
        permuted.unsqueeze(0)?.to_dtype(DType::BF16) // [1, heads, seq, seq]
    }

    /// Single T5 encoder layer.
    ///
    /// ## PyTorch reference:
    /// ```python
    /// # Self-attention
    /// normed = layer_norm(hidden)
    /// q, k, v = attn_q(normed), attn_k(normed), attn_v(normed)
    /// attn_out = self_attention(q, k, v, position_bias)
    /// hidden = hidden + attn_out
    /// # FFN (gated-gelu)
    /// normed = layer_norm(hidden)
    /// gate = gelu(wi_0(normed))
    /// up = wi_1(normed)
    /// hidden = hidden + wo(gate * up)
    /// ```
    fn layer_forward(
        &self,
        hidden: &Tensor,
        weights: &HashMap<String, Tensor>,
        layer_idx: usize,
        position_bias: &Tensor, // [1, heads, seq, seq]
    ) -> Result<Tensor> {
        let cfg = &self.config;
        let h = cfg.num_heads;
        let d = cfg.d_kv;
        let prefix = format!("encoder.block.{layer_idx}");

        let dims = hidden.shape().dims().to_vec();
        let (b, n) = (dims[0], dims[1]);

        let w = |suffix: &str| -> Result<&Tensor> {
            let key = format!("{prefix}.{suffix}");
            weights.get(&key).ok_or_else(|| {
                flame_core::Error::InvalidInput(format!("Missing T5 weight: {key}"))
            })
        };

        // --- Self-attention ---
        let normed = Self::t5_layer_norm(hidden, w("layer.0.layer_norm.weight")?, cfg.layer_norm_eps)?;

        let q = Self::linear(&normed, w("layer.0.SelfAttention.q.weight")?)?;
        let k = Self::linear(&normed, w("layer.0.SelfAttention.k.weight")?)?;
        let v = Self::linear(&normed, w("layer.0.SelfAttention.v.weight")?)?;

        // Reshape: [B, N, heads*d_kv] → [B, heads, N, d_kv]
        let q = q.reshape(&[b, n, h, d])?.permute(&[0, 2, 1, 3])?;
        let k = k.reshape(&[b, n, h, d])?.permute(&[0, 2, 1, 3])?;
        let v = v.reshape(&[b, n, h, d])?.permute(&[0, 2, 1, 3])?;

        // ## BFL/HF T5 reference (transformers/models/t5/modeling_t5.py):
        // ```python
        // scores = torch.matmul(query_states, key_states.transpose(3, 2))  # raw, NO 1/sqrt(d) scaling
        // scores += position_bias_masked                                   # additive float bias
        // attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        // attn_output = torch.matmul(attn_weights, value_states)
        // ```
        //
        // T5 does NOT scale Q·K^T — the scaling is absorbed into q_proj weight init
        // (Mesh TF style). Pass `Some(1.0)` so flame skips the default 1/sqrt(d_kv).
        // position_bias is an ADDITIVE float (not a binary mask) — use sdpa_with_bias.
        let attn_out = flame_sdpa_with_bias(&q, &k, &v, Some(position_bias), Some(1.0))?;

        let attn_out = attn_out.permute(&[0, 2, 1, 3])?.reshape(&[b, n, h * d])?;
        let attn_out = Self::linear(&attn_out, w("layer.0.SelfAttention.o.weight")?)?;
        let hidden = hidden.add(&attn_out)?;

        // --- Gated-GELU FFN ---
        let normed2 = Self::t5_layer_norm(&hidden, w("layer.1.layer_norm.weight")?, cfg.layer_norm_eps)?;

        let gate = Self::linear(&normed2, w("layer.1.DenseReluDense.wi_0.weight")?)?;
        let up = Self::linear(&normed2, w("layer.1.DenseReluDense.wi_1.weight")?)?;
        let ffn_out = gate.gelu()?.mul(&up)?;
        let ffn_out = Self::linear(&ffn_out, w("layer.1.DenseReluDense.wo.weight")?)?;

        hidden.add(&ffn_out)
    }

    /// Encode text → [1, 512, 4096] hidden states.
    ///
    /// Pads/truncates `token_ids` to `config.max_seq_len` (512 for FLUX 1) to match
    /// BFL's `padding="max_length", max_length=512` tokenizer contract. T5 pad_token_id
    /// is 0. No attention mask is applied — BFL passes `attention_mask=None`.
    pub fn encode(&mut self, token_ids: &[i32]) -> Result<Tensor> {
        // Use the caller-supplied length: caller already padded/truncated.
        // Re-padding to cfg.max_seq_len here was a bug for SD3 (cached
        // [1, 589, 4096] vs upstream Python's [1, 154, 4096]).
        self.encode_with_len(token_ids, token_ids.len())
    }

    /// Same as `encode` but explicitly truncates/pads to `target_len`.
    pub fn encode_with_len(&mut self, token_ids: &[i32], target_len: usize) -> Result<Tensor> {
        let cfg = self.config.clone();
        let max_len = target_len.min(cfg.max_seq_len);

        let mut padded: Vec<i32> = token_ids.to_vec();
        if padded.len() > max_len {
            padded.truncate(max_len);
        } else {
            padded.resize(max_len, 0);
        }
        let seq_len = max_len;

        log::info!("[T5] Encoding: seq_len={} (padded from {})", seq_len, token_ids.len());

        // 1. Token embeddings (no position embeddings — T5 uses relative bias)
        let embed_w = self.weights.get("encoder.embed_tokens.weight")
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing embed_tokens".into()))?;
        let ids = Tensor::from_vec(
            padded.iter().map(|&id| id as f32).collect(),
            Shape::from_dims(&[seq_len]),
            self.device.clone(),
        )?.to_dtype(DType::I32)?;
        let mut hidden = embed_w.index_select0(&ids)?.unsqueeze(0)?; // [1, seq, 4096]

        // 2. Compute relative position bias from layer 0 weight (shared across layers)
        let bias_key = "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight";
        let bias_weight = self.weights.get(bias_key)
            .ok_or_else(|| flame_core::Error::InvalidInput(format!("Missing {bias_key}")))?;

        let position_bias = Self::compute_relative_bias(
            bias_weight,
            seq_len,
            cfg.relative_attention_num_buckets,
            cfg.relative_attention_max_distance,
            &self.device,
        )?;

        // 3. Forward through all layers (resident weights)
        for i in 0..cfg.num_layers {
            hidden = self.layer_forward(&hidden, &self.weights, i, &position_bias)?;
            if (i + 1) % 6 == 0 || i == cfg.num_layers - 1 {
                log::info!("[T5] Layer {}/{} done", i + 1, cfg.num_layers);
            }
        }

        // 4. Final layer norm
        let final_norm_w = self.weights.get("encoder.final_layer_norm.weight")
            .ok_or_else(|| flame_core::Error::InvalidInput("Missing final_layer_norm".into()))?;
        hidden = Self::t5_layer_norm(&hidden, final_norm_w, cfg.layer_norm_eps)?;

        log::info!("[T5] Output: {:?}", hidden.shape());
        Ok(hidden)
    }

    pub fn config(&self) -> &T5Config {
        &self.config
    }
}

/// T5 relative position bucket computation.
///
/// ## HF reference (transformers/models/t5/modeling_t5.py::_relative_position_bucket):
/// ```python
/// relative_buckets = 0
/// if bidirectional:
///     num_buckets //= 2
///     relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
///     relative_position = torch.abs(relative_position)
/// else:
///     relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
/// # now relative_position is in the range [0, inf)
///
/// max_exact = num_buckets // 2
/// is_small = relative_position < max_exact
///
/// relative_position_if_large = max_exact + (
///     torch.log(relative_position.float() / max_exact)
///     / math.log(max_distance / max_exact)
///     * (num_buckets - max_exact)
/// ).to(torch.long)
/// relative_position_if_large = torch.min(
///     relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
/// )
///
/// relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
/// return relative_buckets
/// ```
fn t5_relative_position_bucket(
    relative_position: i64,
    bidirectional: bool,
    num_buckets: usize,
    max_distance: usize,
) -> usize {
    let mut relative_buckets: usize = 0;
    let mut nb = num_buckets;

    let rp: usize = if bidirectional {
        nb /= 2;
        if relative_position > 0 {
            relative_buckets += nb;
        }
        relative_position.unsigned_abs() as usize
    } else {
        // relative_position = -min(relative_position, 0) = max(0, -relative_position)
        if relative_position < 0 {
            (-relative_position) as usize
        } else {
            0
        }
    };

    let max_exact = nb / 2;
    let is_small = rp < max_exact;

    let bucket = if is_small {
        rp
    } else {
        let val = max_exact as f64
            + ((rp as f64 / max_exact as f64).ln()
                / (max_distance as f64 / max_exact as f64).ln())
                * ((nb - max_exact) as f64);
        (val as usize).min(nb - 1)
    };

    relative_buckets + bucket
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_t5_config() {
        let cfg = T5Config::default();
        assert_eq!(cfg.d_model, 4096);
        assert_eq!(cfg.num_layers, 24);
        assert_eq!(cfg.num_heads * cfg.d_kv, 4096);
        assert_eq!(cfg.d_ff, 10240);
    }

    #[test]
    fn test_relative_position_bucket() {
        // Verified against HF transformers._relative_position_bucket
        // with num_buckets=32, max_distance=128, bidirectional=True.
        // For bidirectional: nb becomes 16 internally, max_exact=8.
        // rp=0: small, bucket = 0 (+0 for non-positive) = 0
        assert_eq!(t5_relative_position_bucket(0, true, 32, 128), 0);
        // rp=1 (j>i, positive): +16 (positive offset), rp=1, small → 16+1 = 17
        assert_eq!(t5_relative_position_bucket(1, true, 32, 128), 17);
        // rp=-1 (j<i): +0, rp=1, small → 1
        assert_eq!(t5_relative_position_bucket(-1, true, 32, 128), 1);
        // rp=7 (j<i, negative): small (7<8) → 7
        assert_eq!(t5_relative_position_bucket(-7, true, 32, 128), 7);
        // rp=8 (j<i, negative): not small → log path
        let b8 = t5_relative_position_bucket(-8, true, 32, 128);
        assert!(b8 >= 8 && b8 < 16, "bucket for -8 = {}", b8);
        // Far distance clamps to nb-1 = 15
        assert_eq!(t5_relative_position_bucket(-1000, true, 32, 128), 15);
        assert_eq!(t5_relative_position_bucket(1000, true, 32, 128), 16 + 15);
    }
}
