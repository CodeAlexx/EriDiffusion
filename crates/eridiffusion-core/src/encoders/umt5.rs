//! UMT5-XXL encoder for Wan2.2 — HF-native weight key layout.
//!
//! The original `wan/modules/t5.py::umt5_xxl` class builds a flat module
//! with keys like `blocks.{i}.attn.q.weight`, but the community releases
//! (including `ai-toolkit/umt5_xxl_encoder`) ship the weights in HF
//! `T5EncoderModel` layout:
//!
//! ```text
//!   shared.weight                                                [256384, 4096]
//!   encoder.block.{i}.layer.0.SelfAttention.{q,k,v,o}.weight    [4096, 4096]
//!   encoder.block.{i}.layer.0.SelfAttention.relative_attention_bias.weight
//!                                                                [32, 64]    (per-layer for UMT5!)
//!   encoder.block.{i}.layer.0.layer_norm.weight                 [4096]
//!   encoder.block.{i}.layer.1.DenseReluDense.wi_0.weight        [10240, 4096]  (gate)
//!   encoder.block.{i}.layer.1.DenseReluDense.wi_1.weight        [10240, 4096]  (up)
//!   encoder.block.{i}.layer.1.DenseReluDense.wo.weight          [4096, 10240]  (down)
//!   encoder.block.{i}.layer.1.layer_norm.weight                 [4096]
//!   encoder.final_layer_norm.weight                             [4096]
//! ```
//!
//! UMT5 differs from T5 v1.1 in one observable way here: **every encoder
//! layer carries its own `relative_attention_bias` table**. The existing
//! `t5_encoder.rs` (FLUX path) reads it only from layer 0 (T5 v1.1 shares),
//! which is wrong for UMT5. This module recomputes the bias each layer.
//!
//! Other specifics:
//! - Vocab 256,384 (multilingual).
//! - Attention uses bidirectional rel-pos bucketing (encoder only).
//! - T5 does NOT scale Q·K^T by `1/sqrt(d_kv)`; we pass `scale=1.0` to SDPA.
//! - No attention mask inside the encoder — padding tokens contribute to
//!   self-attention; the DiT's cross-attention downstream handles the pad
//!   rows as zeros after the text_embedding MLP, matching the reference.
//!
//! Tokenization is handled outside (via the `tokenizers` crate with the
//! `tokenizer.json` that ships next to the UMT5 weights).

use flame_core::attention::sdpa_with_bias as flame_sdpa_with_bias;
use flame_core::{CudaDevice, DType, Error, Result, Shape, Tensor};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Umt5Config {
    pub vocab_size: usize,          // 256384
    pub d_model: usize,             // 4096
    pub num_layers: usize,          // 24
    pub d_ff: usize,                // 10240
    pub num_heads: usize,           // 64
    pub d_kv: usize,                // 64
    pub num_buckets: usize,         // 32
    pub max_distance: usize,        // 128
    pub layer_norm_eps: f32,
    pub text_len: usize,            // 512
}

impl Default for Umt5Config {
    fn default() -> Self {
        Self {
            vocab_size: 256_384,
            d_model: 4096,
            num_layers: 24,
            d_ff: 10240,
            num_heads: 64,
            d_kv: 64,
            num_buckets: 32,
            max_distance: 128,
            layer_norm_eps: 1e-6,
            text_len: 512,
        }
    }
}

// ---------------------------------------------------------------------------
// Custom BF16-native loader — host-side F16 → BF16 conversion.
// ---------------------------------------------------------------------------

fn load_t5_safetensors_as_bf16(
    path: &Path,
    device: &Arc<CudaDevice>,
) -> Result<HashMap<String, Tensor>> {
    use serde_json::Value;

    let file = std::fs::File::open(path)
        .map_err(|e| Error::Io(format!("opening {}: {e}", path.display())))?;
    let mmap = unsafe { memmap2::Mmap::map(&file) }
        .map_err(|e| Error::Io(format!("mmap {}: {e}", path.display())))?;

    if mmap.len() < 8 {
        return Err(Error::Io("safetensors too small".into()));
    }
    let header_size = u64::from_le_bytes(mmap[..8].try_into().unwrap()) as usize;
    let header_end = 8 + header_size;
    let data_start = header_end;

    let metadata: Value = serde_json::from_slice(&mmap[8..header_end])
        .map_err(|e| Error::Io(format!("parsing header: {e}")))?;
    let metadata_obj = metadata
        .as_object()
        .ok_or_else(|| Error::InvalidInput("metadata not an object".into()))?;

    let mut out: HashMap<String, Tensor> = HashMap::new();
    for (name, info) in metadata_obj {
        if name == "__metadata__" || name == "spiece_model" {
            continue;
        }
        let dtype_str = info["dtype"].as_str().unwrap_or("F32");

        // Skip non-float tensors (e.g. uint8 spiece blobs if any other
        // key sneaks them in).
        if !matches!(dtype_str, "F32" | "BF16" | "F16") {
            continue;
        }

        let shape_vec: Vec<usize> = info["shape"]
            .as_array()
            .ok_or_else(|| Error::InvalidInput(format!("missing shape for {name}")))?
            .iter()
            .map(|v| v.as_u64().unwrap_or(0) as usize)
            .collect();

        let offsets = info["data_offsets"]
            .as_array()
            .ok_or_else(|| Error::InvalidInput(format!("missing data_offsets for {name}")))?;
        let start = data_start + offsets[0].as_u64().unwrap_or(0) as usize;
        let end = data_start + offsets[1].as_u64().unwrap_or(0) as usize;
        let bytes = &mmap[start..end];

        // Host-side convert to BF16 bits, then upload BF16 directly.
        let num_elems: usize = shape_vec.iter().product();
        let mut bf16_bits = vec![0u16; num_elems];
        match dtype_str {
            "BF16" => {
                for (dst, chunk) in bf16_bits.iter_mut().zip(bytes.chunks_exact(2)) {
                    *dst = u16::from_le_bytes([chunk[0], chunk[1]]);
                }
            }
            "F16" => {
                for (dst, chunk) in bf16_bits.iter_mut().zip(bytes.chunks_exact(2)) {
                    let f16_bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    let f = half::f16::from_bits(f16_bits).to_f32();
                    *dst = half::bf16::from_f32(f).to_bits();
                }
            }
            "F32" => {
                for (dst, chunk) in bf16_bits.iter_mut().zip(bytes.chunks_exact(4)) {
                    let f = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    *dst = half::bf16::from_f32(f).to_bits();
                }
            }
            _ => unreachable!(),
        }

        let mut tensor = Tensor::zeros_dtype(
            Shape::from_dims(&shape_vec),
            DType::BF16,
            device.clone(),
        )?;
        tensor.copy_from_bf16_slice(&bf16_bits)?;
        out.insert(name.clone(), tensor);
    }

    // Alias `shared.weight` → `encoder.embed_tokens.weight` if needed.
    if !out.contains_key("encoder.embed_tokens.weight") {
        if let Some(sw) = out.get("shared.weight").cloned() {
            out.insert("encoder.embed_tokens.weight".to_string(), sw);
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

pub struct Umt5Encoder {
    weights: HashMap<String, Tensor>,
    config: Umt5Config,
    device: Arc<CudaDevice>,
}

impl Umt5Encoder {
    /// Load from a safetensors file in HF `T5EncoderModel` format.
    ///
    /// Reads the mmap'd file directly and converts every weight to BF16 on
    /// the host side before uploading. `flame_core::serialization::load_file`
    /// upcasts F16 → F32 on GPU, which for an 11 GB UMT5 checkpoint means
    /// ~22 GB of F32 at peak — right at the 3090 Ti's ceiling. Converting
    /// in host RAM and uploading BF16 directly keeps peak GPU memory at
    /// ~11 GB (one resident BF16 copy).
    pub fn load(path: &Path, device: &Arc<CudaDevice>) -> Result<Self> {
        let weights = load_t5_safetensors_as_bf16(path, device)?;
        log::info!("[UMT5] Loaded: {} weights", weights.len());
        Ok(Self {
            weights,
            config: Umt5Config::default(),
            device: device.clone(),
        })
    }

    pub fn config(&self) -> &Umt5Config {
        &self.config
    }

    // -----------------------------------------------------------------
    // Low-level helpers
    // -----------------------------------------------------------------

    fn t5_layer_norm(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let hidden = *dims.last().unwrap();
        let batch: usize = dims[..dims.len() - 1].iter().product();
        let x_2d = x.reshape(&[batch, hidden])?;
        let out = flame_core::cuda_ops_bf16::rms_norm_bf16(&x_2d, Some(weight), eps)?;
        out.reshape(&dims)
    }

    fn linear_nobias(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
        // Weight is stored as `[out, in]`; transpose for matmul.
        let wt = flame_core::bf16_elementwise::transpose2d_bf16(weight)?;
        let dims = x.shape().dims().to_vec();
        let (b, n, c) = (dims[0], dims[1], dims[2]);
        let x_2d = x.reshape(&[b * n, c])?;
        let out = x_2d.matmul(&wt)?;
        let out_dim = out.shape().dims()[1];
        out.reshape(&[b, n, out_dim])
    }

    fn get(&self, key: &str) -> Result<&Tensor> {
        self.weights.get(key).ok_or_else(|| {
            flame_core::Error::InvalidInput(format!("Missing UMT5 weight: {key}"))
        })
    }

    // -----------------------------------------------------------------
    // Bidirectional relative-position bucketing (HF T5 encoder variant).
    //
    // Direct port of `transformers.models.t5.modeling_t5.T5Attention.
    // _relative_position_bucket(bidirectional=True)` — identical to the
    // Wan reference at `wan/modules/t5.py` (bidirectional branch).
    // -----------------------------------------------------------------

    fn relative_position_bucket(
        rel_pos: i64,
        num_buckets: usize,
        max_dist: usize,
    ) -> usize {
        // bidirectional encoder branch
        let num_buckets = num_buckets / 2;
        let rel_buckets = if rel_pos > 0 { num_buckets } else { 0 };
        let rel_pos = rel_pos.unsigned_abs() as usize;

        let max_exact = num_buckets / 2;
        if rel_pos < max_exact {
            rel_buckets + rel_pos
        } else {
            let rp = (rel_pos as f64 / max_exact as f64).ln()
                / (max_dist as f64 / max_exact as f64).ln();
            let large = max_exact + (rp * (num_buckets - max_exact) as f64) as usize;
            rel_buckets + large.min(num_buckets - 1)
        }
    }

    /// Build `[1, H, L, L]` relative-position bias from a `[num_buckets, num_heads]`
    /// table. UMT5 uses a **distinct** table per layer.
    fn compute_bias(&self, pos_embed: &Tensor, seq_len: usize) -> Result<Tensor> {
        let cfg = &self.config;
        let num_heads = cfg.num_heads;

        let mut bucket_ids = vec![0i32; seq_len * seq_len];
        for q in 0..seq_len {
            for k in 0..seq_len {
                let rel = k as i64 - q as i64;
                let b = Self::relative_position_bucket(rel, cfg.num_buckets, cfg.max_distance);
                bucket_ids[q * seq_len + k] = b as i32;
            }
        }
        let idx = Tensor::from_vec(
            bucket_ids.iter().map(|&i| i as f32).collect(),
            Shape::from_dims(&[seq_len * seq_len]),
            self.device.clone(),
        )?
        .to_dtype(DType::I32)?;

        let gathered = pos_embed.index_select0(&idx)?; // [L*L, H]
        let reshaped = gathered.reshape(&[seq_len, seq_len, num_heads])?;
        let permuted = reshaped.permute(&[2, 0, 1])?; // [H, L, L]
        permuted.unsqueeze(0)?.to_dtype(DType::BF16)
    }

    // -----------------------------------------------------------------
    // Block forward (single encoder layer)
    // -----------------------------------------------------------------

    fn layer_forward(&self, hidden: &Tensor, layer_idx: usize) -> Result<Tensor> {
        let cfg = &self.config;
        let h = cfg.num_heads;
        let d = cfg.d_kv;
        let prefix = format!("encoder.block.{layer_idx}");

        let w = |suf: &str| -> Result<&Tensor> {
            self.weights
                .get(&format!("{prefix}.{suf}"))
                .ok_or_else(|| {
                    flame_core::Error::InvalidInput(format!(
                        "Missing UMT5 weight: {prefix}.{suf}"
                    ))
                })
        };

        let dims = hidden.shape().dims().to_vec();
        let (b, n) = (dims[0], dims[1]);

        // Per-layer rel-pos bias.
        let pos_embed = w("layer.0.SelfAttention.relative_attention_bias.weight")?;
        let bias = self.compute_bias(pos_embed, n)?;
        if layer_idx == 0 && std::env::var("UMT5_DEBUG").is_ok() {
            let v = bias.to_dtype(DType::F32)?.to_vec1::<f32>()?;
            // bias shape [1, H, L, L]. Print head 0, query 0, first 5 keys.
            // stride: H*L*L per batch, L*L per head, L per query
            let ll = n * n;
            log::info!(
                "[UMT5 dbg] L0 bias[0,0,0,0..5]: {:.4} {:.4} {:.4} {:.4} {:.4}",
                v[0], v[1], v[2], v[3], v[4]
            );
            // head 1, q=0, first 5 keys:
            log::info!(
                "[UMT5 dbg] L0 bias[0,1,0,0..5]: {:.4} {:.4} {:.4} {:.4} {:.4}",
                v[ll], v[ll + 1], v[ll + 2], v[ll + 3], v[ll + 4]
            );
        }

        // --- Self-attention (pre-norm) ---
        let normed = Self::t5_layer_norm(hidden, w("layer.0.layer_norm.weight")?, cfg.layer_norm_eps)?;
        if layer_idx == 0 && std::env::var("UMT5_DEBUG").is_ok() {
            let v = normed.to_dtype(DType::F32)?.to_vec1::<f32>()?;
            log::info!(
                "[UMT5 dbg] L0 normed    tok0 first5: {:.4} {:.4} {:.4} {:.4} {:.4}",
                v[0], v[1], v[2], v[3], v[4]
            );
        }

        let q = Self::linear_nobias(&normed, w("layer.0.SelfAttention.q.weight")?)?;
        let k = Self::linear_nobias(&normed, w("layer.0.SelfAttention.k.weight")?)?;
        let v = Self::linear_nobias(&normed, w("layer.0.SelfAttention.v.weight")?)?;
        if layer_idx == 0 && std::env::var("UMT5_DEBUG").is_ok() {
            let vq = q.to_dtype(DType::F32)?.to_vec1::<f32>()?;
            let vk = k.to_dtype(DType::F32)?.to_vec1::<f32>()?;
            let vv = v.to_dtype(DType::F32)?.to_vec1::<f32>()?;
            log::info!(
                "[UMT5 dbg] L0 q tok0 first5: {:.4} {:.4} {:.4} {:.4} {:.4}",
                vq[0], vq[1], vq[2], vq[3], vq[4]
            );
            log::info!(
                "[UMT5 dbg] L0 k tok0 first5: {:.4} {:.4} {:.4} {:.4} {:.4}",
                vk[0], vk[1], vk[2], vk[3], vk[4]
            );
            log::info!(
                "[UMT5 dbg] L0 v tok0 first5: {:.4} {:.4} {:.4} {:.4} {:.4}",
                vv[0], vv[1], vv[2], vv[3], vv[4]
            );
        }

        let q = q.reshape(&[b, n, h, d])?.permute(&[0, 2, 1, 3])?;
        let k = k.reshape(&[b, n, h, d])?.permute(&[0, 2, 1, 3])?;
        let v = v.reshape(&[b, n, h, d])?.permute(&[0, 2, 1, 3])?;

        // T5: NO 1/sqrt(d) scaling. Bias is additive float (per-layer).
        // Mirror torch UMT5Attention exactly:
        //   scores = Q @ K^T       (BF16 inputs, F32 accumulator → BF16 result)
        //   scores += bias         (BF16)
        //   attn = softmax(scores.float(), dim=-1).type_as(scores)
        //   out = attn @ V
        // CRITICAL: doing Q@K^T in F32 (which appears more "precise") actually
        // diverges from torch's BF16-quantized scores. The model expects the
        // BF16 quantization step — without it, attention sharpness differs and
        // outputs drift catastrophically by layer 18. This was the Helios UMT5
        // bug.
        let sdpa_raw = {
            let bh = b * h;
            let q3 = q.reshape(&[bh, n, d])?;          // BF16
            let k3 = k.reshape(&[bh, n, d])?;
            let v3 = v.reshape(&[bh, n, d])?;
            let k3_t = k3.permute(&[0, 2, 1])?.contiguous()?;   // [bh, d, n] BF16
            let scores3 = q3.bmm(&k3_t)?;              // BF16 (F32 accum internally per cuBLAS)
            if layer_idx == 0 && std::env::var("UMT5_DEBUG").is_ok() {
                let sv = scores3.to_vec1::<f32>()?;
                // bh=b*h. head 0 = bh index 0. q_row=0, first 5 keys.
                log::info!(
                    "[UMT5 dbg] L0 scores_pre_bias[bh=0,q=0,k=0..5]: {:.4} {:.4} {:.4} {:.4} {:.4}",
                    sv[0], sv[1], sv[2], sv[3], sv[4]
                );
                // head 1 (bh=1), q=0
                let h1q0 = 1 * n * n;
                log::info!(
                    "[UMT5 dbg] L0 scores_pre_bias[bh=1,q=0,k=0..5]: {:.4} {:.4} {:.4} {:.4} {:.4}",
                    sv[h1q0], sv[h1q0 + 1], sv[h1q0 + 2], sv[h1q0 + 3], sv[h1q0 + 4]
                );
            }
            // scores BF16, bias BF16 — keep in BF16 to match torch's exact
            // arithmetic before the F32 softmax cast.
            let scores4 = scores3.reshape(&[b, h, n, n])?;
            let scores4 = scores4.add(&bias)?;
            // Now cast to F32 for the softmax (torch: softmax(scores.float(), dim=-1)).
            let scores4 = scores4.to_dtype(DType::F32)?;
            if layer_idx == 0 && std::env::var("UMT5_DEBUG").is_ok() {
                let sv = scores4.to_vec1::<f32>()?;
                log::info!(
                    "[UMT5 dbg] L0 scores_post_bias[0,0,0,0..{}]: {:?}",
                    n,
                    &sv[..n]
                );
                let max_v = sv[..n].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let min_v = sv[..n].iter().cloned().fold(f32::INFINITY, f32::min);
                log::info!("[UMT5 dbg] L0 scores_post_bias row max={max_v}  min={min_v}");
            }
            // Compute softmax manually in F32 to bypass any fast-path
            // numerical mismatch. Python: F.softmax(x.float(), dim=-1).
            let attn4 = {
                let max4 = flame_core::cuda_ops::GpuOps::max_dim(&scores4, 3, true)?;
                let shifted = scores4.sub(&max4)?;
                let exp4 = shifted.exp()?;
                let sum4 = flame_core::cuda_ops::GpuOps::sum_dim_keepdim(&exp4, 3)?;
                exp4.div(&sum4)?
            };
            if layer_idx == 0 && std::env::var("UMT5_DEBUG").is_ok() {
                let sv = attn4.to_vec1::<f32>()?;
                log::info!(
                    "[UMT5 dbg] L0 attn_weights[0,0,0,0..5]: {:.4} {:.4} {:.4} {:.4} {:.4}",
                    sv[0], sv[1], sv[2], sv[3], sv[4]
                );
                let row_sum: f32 = sv[..n].iter().sum();
                log::info!("[UMT5 dbg] L0 attn_weights row sum (head 0 q 0): {row_sum:.4}");
            }
            // attn weights BF16 (torch: .type_as(scores) → BF16). Multiply
            // attn @ V in BF16 with F32 accumulator.
            let attn3 = attn4.to_dtype(DType::BF16)?.reshape(&[bh, n, n])?;
            let out3 = attn3.bmm(&v3)?;                // BF16 (V is BF16)
            out3.reshape(&[b, h, n, d])?
        };
        if layer_idx == 0 && std::env::var("UMT5_DEBUG").is_ok() {
            // sdpa_raw shape: [B, H, L, D]
            let v = sdpa_raw.to_dtype(DType::F32)?.to_vec1::<f32>()?;
            log::info!(
                "[UMT5 dbg] L0 sdpa[0,0,0,0..5]: {:.4} {:.4} {:.4} {:.4} {:.4}",
                v[0], v[1], v[2], v[3], v[4]
            );
            // sdpa for (B=0, H=1, L=0)
            let l_len = n;
            let d_dim = d;
            let offs = 1 * l_len * d_dim;
            log::info!(
                "[UMT5 dbg] L0 sdpa[0,1,0,0..5]: {:.4} {:.4} {:.4} {:.4} {:.4}",
                v[offs], v[offs + 1], v[offs + 2], v[offs + 3], v[offs + 4]
            );
        }
        let attn_pre = sdpa_raw.permute(&[0, 2, 1, 3])?.reshape(&[b, n, h * d])?;
        if layer_idx == 0 && std::env::var("UMT5_DEBUG").is_ok() {
            let v = attn_pre.to_dtype(DType::F32)?.to_vec1::<f32>()?;
            log::info!(
                "[UMT5 dbg] L0 pre-o tok0 first5: {:.4} {:.4} {:.4} {:.4} {:.4}",
                v[0], v[1], v[2], v[3], v[4]
            );
        }
        let attn = Self::linear_nobias(&attn_pre, w("layer.0.SelfAttention.o.weight")?)?;
        if layer_idx == 0 && std::env::var("UMT5_DEBUG").is_ok() {
            let vh = hidden.to_dtype(DType::F32)?.to_vec1::<f32>()?;
            let va = attn.to_dtype(DType::F32)?.to_vec1::<f32>()?;
            log::info!(
                "[UMT5 dbg] L0 pre-attn  tok0 first5: {:.4} {:.4} {:.4} {:.4} {:.4}",
                vh[0], vh[1], vh[2], vh[3], vh[4]
            );
            log::info!(
                "[UMT5 dbg] L0 attn_out  tok0 first5: {:.4} {:.4} {:.4} {:.4} {:.4}",
                va[0], va[1], va[2], va[3], va[4]
            );
        }
        let hidden = hidden.add(&attn)?;
        if layer_idx == 0 && std::env::var("UMT5_DEBUG").is_ok() {
            let vh = hidden.to_dtype(DType::F32)?.to_vec1::<f32>()?;
            log::info!(
                "[UMT5 dbg] L0 post-attn tok0 first5: {:.4} {:.4} {:.4} {:.4} {:.4}",
                vh[0], vh[1], vh[2], vh[3], vh[4]
            );
        }

        // --- Gated-GELU FFN (pre-norm) ---
        let normed2 = Self::t5_layer_norm(&hidden, w("layer.1.layer_norm.weight")?, cfg.layer_norm_eps)?;
        let gate = Self::linear_nobias(&normed2, w("layer.1.DenseReluDense.wi_0.weight")?)?.gelu()?;
        let up = Self::linear_nobias(&normed2, w("layer.1.DenseReluDense.wi_1.weight")?)?;
        let ffn_hidden = up.mul(&gate)?;
        let ffn_out = Self::linear_nobias(&ffn_hidden, w("layer.1.DenseReluDense.wo.weight")?)?;
        hidden.add(&ffn_out)
    }

    // -----------------------------------------------------------------
    // Public encode
    // -----------------------------------------------------------------

    /// Encode a list of token ids → `[1, text_len, 4096]` BF16.
    ///
    /// Matches Wan's `T5EncoderModel.__call__`: the encoder runs on the
    /// **unpadded** sequence length, not text_len. The unpadded hidden
    /// states are then zero-padded up to `text_len` (the Wan DiT's
    /// `text_len=512`) **after** the final norm. Encoding the padding
    /// tokens through all 24 layers pollutes the attention
    /// distribution (because the per-layer relative position bias covers
    /// 512 positions and softmax normalizes over the padding keys).
    pub fn encode(&mut self, token_ids: &[i32]) -> Result<Tensor> {
        let cfg = self.config.clone();
        let max_len = cfg.text_len;

        // Truncate if longer than text_len, but do NOT pad up.
        let real_ids: Vec<i32> = if token_ids.len() > max_len {
            token_ids[..max_len].to_vec()
        } else {
            token_ids.to_vec()
        };
        let seq_len = real_ids.len().max(1);
        let padded = real_ids;

        log::info!("[UMT5] Encoding seq_len={}", seq_len);

        // Token embeddings.
        let embed = self.get("encoder.embed_tokens.weight")?;
        let ids = Tensor::from_vec(
            padded.iter().map(|&id| id as f32).collect(),
            Shape::from_dims(&[seq_len]),
            self.device.clone(),
        )?
        .to_dtype(DType::I32)?;
        let mut hidden = embed.index_select0(&ids)?.unsqueeze(0)?; // [1, L, d_model]
        if std::env::var("UMT5_DEBUG").is_ok() {
            let v = hidden.to_dtype(DType::F32)?.to_vec1::<f32>()?;
            log::info!(
                "[UMT5 dbg] embed token0 first5: {:.4} {:.4} {:.4} {:.4} {:.4}",
                v[0], v[1], v[2], v[3], v[4]
            );
        }

        // Transformer blocks (each layer recomputes its own rel-pos bias).
        // When UMT5_DUMP_PER_LAYER is set, save each layer's output for
        // bisection vs diffusers reference.
        let mut per_layer_dump: std::collections::HashMap<String, Tensor> = std::collections::HashMap::new();
        let dump_per_layer = std::env::var("UMT5_DUMP_PER_LAYER").is_ok();
        if dump_per_layer {
            per_layer_dump.insert("raw_embed".to_string(), hidden.clone());
        }
        for i in 0..cfg.num_layers {
            hidden = self.layer_forward(&hidden, i)?;
            if dump_per_layer {
                per_layer_dump.insert(format!("layer_{:02}", i), hidden.clone());
            }
            if std::env::var("UMT5_DEBUG").is_ok() && i < 3 {
                let v = hidden.to_dtype(DType::F32)?.to_vec1::<f32>()?;
                log::info!(
                    "[UMT5 dbg] after layer {}  token0 first5: {:.4} {:.4} {:.4} {:.4} {:.4}",
                    i, v[0], v[1], v[2], v[3], v[4]
                );
            }
            if (i + 1) % 6 == 0 || i == cfg.num_layers - 1 {
                log::info!("[UMT5] Layer {}/{} done", i + 1, cfg.num_layers);
            }
        }
        if dump_per_layer {
            let path = std::env::var("UMT5_DUMP_PER_LAYER").unwrap();
            flame_core::serialization::save_file(&per_layer_dump, &path)
                .map_err(|e| Error::Io(format!("save_file: {e}")))?;
            log::info!("[UMT5] saved per-layer outputs to {path}");
        }

        // Final norm.
        let final_w = self.get("encoder.final_layer_norm.weight")?;
        hidden = Self::t5_layer_norm(&hidden, final_w, cfg.layer_norm_eps)?;

        // Zero-pad to text_len (mirrors Wan DiT's padding step).
        if seq_len < max_len {
            let pad_shape = [1, max_len - seq_len, cfg.d_model];
            let pad = Tensor::zeros_dtype(
                Shape::from_dims(&pad_shape),
                DType::BF16,
                self.device.clone(),
            )?;
            hidden = Tensor::cat(&[&hidden, &pad], 1)?;
        }

        log::info!("[UMT5] Output: {:?}", hidden.shape());
        Ok(hidden)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bucket_signs_differ_bidirectional() {
        let bwd = Umt5Encoder::relative_position_bucket(-3, 32, 128);
        let fwd = Umt5Encoder::relative_position_bucket(3, 32, 128);
        assert_ne!(bwd, fwd);
        assert!(bwd < 16);
        assert!(fwd >= 16);
    }
    #[test]
    fn bucket_zero_is_zero() {
        assert_eq!(Umt5Encoder::relative_position_bucket(0, 32, 128), 0);
    }
}
