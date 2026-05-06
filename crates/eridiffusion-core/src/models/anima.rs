//! Anima — circlestone-labs/Anima T2I (Cosmos-Predict2) DiT, training port.
//!
//! Forward ported from `inference-flame/src/models/anima.rs`. Training-specific
//! adaptations:
//!
//! - `cuda_ops_bf16::rms_norm_bf16` (the inference fused kernel) is **inference-only**
//!   in flame-core; its backward at this hidden size silently produces zero or
//!   direction-randomized grads (cf. `feedback_flame_core_backward_precision.md`
//!   and the same fix applied across z-image / klein trainers). We substitute
//!   `primitive_rms_norm` (F32 internal, autograd-recorded primitive op chain)
//!   wherever the inference path called the fused kernel.
//! - LoRA injection uses the bundle's `LoRALinear::forward_delta` (same pattern
//!   as `zimage.rs::add_lora_delta`) instead of the inference-time `LoraStack`
//!   — gradients flow directly into `lora_a` / `lora_b` Parameters.
//! - The transformer-block residual stream is kept in F32 (matches the
//!   inference port's "x_f32" pattern: at this scale BF16 residuals saturate
//!   and forward output collapses).
//! - `padding_mask` is hard-zero per kohya `anima_train_network.py:303`
//!   (`torch.zeros(bs, 1, h_lat, w_lat)`), built inside `forward`. Since it is
//!   constant zero we skip the cat at the patchifier and instead pad by
//!   prepending the 1-channel zero band before linearising — matches numerics.
//!
//! ## Data layout
//!
//! Trainer feeds `[B, 16, H, W]` (4D NCHW) latents via `TrainableModel::forward`.
//! Internally we lift to `[B, 1, H, W, 16]` (T=1, C-last) which is the layout
//! the ported forward operates in. Output is reshaped back to `[B, 16, H, W]`.
//!
//! ## LoRA target list
//!
//! kohya `networks/lora_anima.py` targets every Linear inside `Block`, plus
//! `PatchEmbed`, `TimestepEmbedding`, `FinalLayer` (and optionally
//! `LLMAdapterTransformerBlock` when `train_llm_adapter=True`, default `False`).
//!
//! For Phase A training we cover the per-block 10 attention/MLP slots — these
//! are what every Anima LoRA in the wild trains and what the inference loader
//! supports. AdaLN modulation, x_embedder, t_embedder, final_layer, and the
//! LLM adapter linears are TODO and explicitly listed at the bottom of this
//! file.
//!
//! Save key naming follows ai-toolkit / PEFT (`diffusion_model.<...>.lora_A.weight`)
//! exactly as Z-Image does, so trained LoRAs route through the inference-flame
//! `LoraStack` `AiToolkit` mapper without conversion.

use std::collections::HashMap;
use std::sync::Arc;
use cudarc::driver::CudaDevice;
use flame_core::{parameter::Parameter, DType, Shape, Tensor};
use crate::config::TrainConfig;
use crate::lora::LoRALinear;
use crate::models::TrainableModel;
use crate::Result;

// ── Anima preview config (matches anima_utils.load_anima_model) ──
pub const HIDDEN: usize = 2048;             // model_channels
pub const HEADS: usize = 16;
pub const HEAD_DIM: usize = HIDDEN / HEADS; // 128
pub const NUM_BLOCKS: usize = 28;
pub const MLP_RATIO: f32 = 4.0;
pub const FFN: usize = (HIDDEN as f32 * MLP_RATIO) as usize; // 8192
pub const IN_CHANNELS: usize = 16;
pub const OUT_CHANNELS: usize = 16;
pub const PATCH_SPATIAL: usize = 2;
pub const PATCH_TEMPORAL: usize = 1;
pub const CONCAT_PADDING_MASK: bool = true;
pub const CROSSATTN_EMB_CHANNELS: usize = 1024; // Qwen3-0.6B hidden_size
pub const ADALN_LORA_DIM: usize = 256;
pub const NORM_EPS: f32 = 1e-6;

/// LLM Adapter blocks present in the preview checkpoint.
pub const ADAPTER_BLOCKS: usize = 6;
pub const ADAPTER_DIM: usize = 1024;
pub const ADAPTER_HEADS: usize = 16;
pub const ADAPTER_HEAD_DIM: usize = 64;
pub const ROPE_THETA: f32 = 10000.0;

/// Per-channel VAE normalization for `qwen_image_vae.safetensors`. Source:
/// `kohya / sd-scripts-anima / library / qwen_image_autoencoder_kl.py:877-912`
/// (`AutoencoderKLQwenImage.__init__` defaults).
///
/// Encode (training-time) — `(z - mean) / std`.
/// Decode (sample-time)   — `z * std + mean`.
pub const QWEN_VAE_LATENT_MEAN: [f32; 16] = [
    -0.7571, -0.7089, -0.9113,  0.1075, -0.1745,  0.9653, -0.1517,  1.5508,
     0.4134, -0.0715,  0.5517, -0.3632, -0.1922, -0.9497,  0.2503, -0.2921,
];
pub const QWEN_VAE_LATENT_STD: [f32; 16] = [
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
];

/// LoRA target slots per Anima Block.
///   0..3 self-attn:  q_proj, k_proj, v_proj, output_proj  (HIDDEN→HIDDEN)
///   4..7 cross-attn: q_proj (HIDDEN→HIDDEN), k_proj/v_proj (1024→HIDDEN), output_proj (HIDDEN→HIDDEN)
///   8..9 mlp:        layer1 (HIDDEN→FFN), layer2 (FFN→HIDDEN)
pub const LORA_SLOTS_PER_BLOCK: usize = 10;
pub const LORA_SLOT_KEYS: [&str; LORA_SLOTS_PER_BLOCK] = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.output_proj",
    "cross_attn.q_proj",
    "cross_attn.k_proj",
    "cross_attn.v_proj",
    "cross_attn.output_proj",
    "mlp.layer1",
    "mlp.layer2",
];
const LORA_SHAPES: [(usize, usize); LORA_SLOTS_PER_BLOCK] = [
    (HIDDEN, HIDDEN),
    (HIDDEN, HIDDEN),
    (HIDDEN, HIDDEN),
    (HIDDEN, HIDDEN),
    (HIDDEN, HIDDEN),
    (CROSSATTN_EMB_CHANNELS, HIDDEN),
    (CROSSATTN_EMB_CHANNELS, HIDDEN),
    (HIDDEN, HIDDEN),
    (HIDDEN, FFN),
    (FFN, HIDDEN),
];

/// Indexable enum for LoRA target lookup; mirrors the `LoraTarget` pattern in
/// `zimage.rs`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AnimaLoraTarget {
    SaQ, SaK, SaV, SaOut,
    CaQ, CaK, CaV, CaOut,
    MlpL1, MlpL2,
}
impl AnimaLoraTarget {
    pub fn slot(self) -> usize {
        match self {
            Self::SaQ => 0, Self::SaK => 1, Self::SaV => 2, Self::SaOut => 3,
            Self::CaQ => 4, Self::CaK => 5, Self::CaV => 6, Self::CaOut => 7,
            Self::MlpL1 => 8, Self::MlpL2 => 9,
        }
    }
}

pub struct AnimaLoraBundle {
    pub adapters: Vec<LoRALinear>, // length = NUM_BLOCKS * LORA_SLOTS_PER_BLOCK
}

impl AnimaLoraBundle {
    pub fn new(rank: usize, alpha: f32, device: Arc<CudaDevice>, seed: u64) -> Result<Self> {
        let mut adapters = Vec::with_capacity(NUM_BLOCKS * LORA_SLOTS_PER_BLOCK);
        for block_idx in 0..NUM_BLOCKS {
            for (slot_idx, &(in_f, out_f)) in LORA_SHAPES.iter().enumerate() {
                let s = seed + (block_idx * LORA_SLOTS_PER_BLOCK + slot_idx) as u64;
                adapters.push(LoRALinear::new(in_f, out_f, rank, alpha, device.clone(), s)?);
            }
        }
        Ok(Self { adapters })
    }

    pub fn parameters(&self) -> Vec<Parameter> {
        let mut p = Vec::with_capacity(self.adapters.len() * 2);
        for l in &self.adapters {
            p.extend(l.parameters());
        }
        p
    }

    pub fn named_parameters(&self) -> Vec<(String, Parameter)> {
        let mut out = Vec::with_capacity(self.adapters.len() * 2);
        for (i, adapter) in self.adapters.iter().enumerate() {
            let block_idx = i / LORA_SLOTS_PER_BLOCK;
            let slot = i % LORA_SLOTS_PER_BLOCK;
            let prefix = format!("diffusion_model.blocks.{block_idx}.{}", LORA_SLOT_KEYS[slot]);
            out.push((format!("{prefix}.lora_A.weight"), adapter.lora_a().clone()));
            out.push((format!("{prefix}.lora_B.weight"), adapter.lora_b().clone()));
        }
        out
    }

    /// Slot lookup by (block, target).
    pub fn get(&self, block: usize, target: AnimaLoraTarget) -> &LoRALinear {
        &self.adapters[block * LORA_SLOTS_PER_BLOCK + target.slot()]
    }

    pub fn save(&self, path: &std::path::Path) -> Result<()> {
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        for (i, adapter) in self.adapters.iter().enumerate() {
            let block_idx = i / LORA_SLOTS_PER_BLOCK;
            let slot = i % LORA_SLOTS_PER_BLOCK;
            let prefix = format!("diffusion_model.blocks.{block_idx}.{}", LORA_SLOT_KEYS[slot]);
            tensors.insert(format!("{prefix}.lora_A.weight"), adapter.lora_a().tensor()?);
            tensors.insert(format!("{prefix}.lora_B.weight"), adapter.lora_b().tensor()?);
        }
        flame_core::serialization::save_file(&tensors, path)
            .map_err(|e| crate::EriDiffusionError::Safetensors(format!("save_file: {e}")))?;
        Ok(())
    }

    pub fn load(&self, path: &std::path::Path, device: &Arc<CudaDevice>) -> Result<()> {
        let source = flame_core::serialization::load_file(path, device)
            .map_err(|e| crate::EriDiffusionError::Safetensors(format!("load_file: {e}")))?;
        for (i, adapter) in self.adapters.iter().enumerate() {
            let block_idx = i / LORA_SLOTS_PER_BLOCK;
            let slot = i % LORA_SLOTS_PER_BLOCK;
            let prefix = format!("diffusion_model.blocks.{block_idx}.{}", LORA_SLOT_KEYS[slot]);
            adapter.load_tensors(&prefix, &source)?;
        }
        Ok(())
    }
}

pub struct AnimaModel {
    pub config: TrainConfig,
    pub device: Arc<CudaDevice>,
    /// All base weights in BF16, keys with `net.` stripped.
    pub weights: HashMap<String, Tensor>,
    pub bundle: AnimaLoraBundle,
    pub is_lora: bool,
}

impl AnimaModel {
    pub fn load(
        weight_path: &std::path::Path,
        config: &TrainConfig,
        device: Arc<CudaDevice>,
    ) -> Result<Self> {
        let raw = flame_core::serialization::load_file(weight_path, &device)?;
        // Many Anima checkpoints prefix every key with `net.`. We keep the
        // prefix in the map so both styles of weight key work; the `w()` helper
        // tries the literal key, then `net.<key>` as fallback.
        let mut weights: HashMap<String, Tensor> = HashMap::with_capacity(raw.len());
        for (k, v) in raw {
            weights.insert(k, v.to_dtype(DType::BF16)?);
        }
        log::info!("Anima: {} tensors loaded from {}", weights.len(), weight_path.display());

        if !config.is_lora() {
            return Err(crate::EriDiffusionError::Model(
                "AnimaModel: only LoRA mode is implemented (set training_method = LoRA)".into(),
            ));
        }
        let rank = config.lora_rank as usize;
        let alpha = config.lora_alpha as f32;
        let bundle = AnimaLoraBundle::new(rank, alpha, device.clone(), 42)?;
        log::info!(
            "Anima LoRA: {} adapters across {} blocks (rank={}, alpha={})",
            bundle.adapters.len(), NUM_BLOCKS, rank, alpha,
        );
        Ok(Self {
            config: config.clone(),
            device,
            weights,
            bundle,
            is_lora: true,
        })
    }

    // ─── Weight lookup ──────────────────────────────────────────────────────
    fn w(&self, key: &str) -> Result<&Tensor> {
        // Try literal first, then `net.<key>` (preview checkpoints prefix
        // every key with `net.`).
        if let Some(t) = self.weights.get(key) { return Ok(t); }
        let alt = format!("net.{key}");
        self.weights.get(&alt).ok_or_else(|| {
            crate::EriDiffusionError::Model(format!("Anima: missing weight `{key}` (also tried `{alt}`)"))
        })
    }

    // ─── Linear helpers (autograd-aware via plain matmul) ───────────────────

    /// Plain `x @ W^T` with `.contiguous()` on the transpose so flame-core's
    /// BF16 matmul backward sees a contiguous layout (same fix used in
    /// zimage.rs::block_linear_no_bias).
    fn linear_no_bias(&self, x: &Tensor, weight_key: &str) -> Result<Tensor> {
        let weight = self.w(weight_key)?;
        let dims = x.shape().dims().to_vec();
        let in_f = *dims.last().unwrap();
        let batch: usize = dims[..dims.len() - 1].iter().product();
        let out_f = weight.shape().dims()[0];
        let x_2d = x.reshape(&[batch, in_f])?;
        let wt = weight.transpose()?.contiguous()?;
        let out_2d = x_2d.matmul(&wt)?;
        let mut out_shape = dims[..dims.len() - 1].to_vec();
        out_shape.push(out_f);
        out_2d.reshape(&out_shape).map_err(Into::into)
    }

    fn linear_with_bias(&self, x: &Tensor, w_key: &str, b_key: &str) -> Result<Tensor> {
        let out = self.linear_no_bias(x, w_key)?;
        let bias = self.w(b_key)?;
        let dims = out.shape().dims().to_vec();
        let last = *dims.last().unwrap();
        let batch: usize = dims[..dims.len() - 1].iter().product();
        let bias_1d = bias.reshape(&[1, last])?;
        let out_2d = out.reshape(&[batch, last])?;
        out_2d.add(&bias_1d)?.reshape(&dims).map_err(Into::into)
    }

    /// Linear with optional LoRA delta applied after the base matmul.
    /// `slot_block` = `Some((block, target))` enables LoRA; `None` skips.
    fn linear_lora(
        &self,
        x: &Tensor,
        weight_key: &str,
        slot_block: Option<(usize, AnimaLoraTarget)>,
    ) -> Result<Tensor> {
        let base = self.linear_no_bias(x, weight_key)?;
        if let Some((block, target)) = slot_block {
            let lora = self.bundle.get(block, target);
            let delta = lora.forward_delta(x).map_err(crate::EriDiffusionError::from)?;
            base.add(&delta).map_err(Into::into)
        } else {
            Ok(base)
        }
    }

    // ─── RMSNorm — F32-internal primitive chain (autograd-correct) ──────────

    fn rms_norm(&self, x: &Tensor, weight_key: &str, eps: f32) -> Result<Tensor> {
        let w = self.w(weight_key)?;
        primitive_rms_norm(x, w, eps).map_err(Into::into)
    }

    /// `x: [B, S, H, D]`, weight `[D]`. Norm over last dim per head.
    fn rms_norm_per_head(&self, x: &Tensor, weight_key: &str) -> Result<Tensor> {
        let w = self.w(weight_key)?;
        let dims = x.shape().dims().to_vec();
        let (b, s, h, d) = (dims[0], dims[1], dims[2], dims[3]);
        let flat = x.reshape(&[b * s * h, d])?;
        let normed = primitive_rms_norm(&flat, w, NORM_EPS)?;
        normed.reshape(&[b, s, h, d]).map_err(Into::into)
    }

    /// `x: [B, H, S, D]`, weight `[D]`.
    fn rms_norm_per_head_bhsd(&self, x: &Tensor, weight_key: &str) -> Result<Tensor> {
        let w = self.w(weight_key)?;
        let dims = x.shape().dims().to_vec();
        let (b, h, s, d) = (dims[0], dims[1], dims[2], dims[3]);
        let flat = x.reshape(&[b * h * s, d])?;
        let normed = primitive_rms_norm(&flat, w, NORM_EPS)?;
        normed.reshape(&[b, h, s, d]).map_err(Into::into)
    }

    // ─── Timestep embedder ──────────────────────────────────────────────────

    /// Returns (t_cond [B, 2048], base_adaln [B, 6144]).
    fn prepare_timestep(&self, t: &Tensor) -> Result<(Tensor, Tensor)> {
        let dim = HIDDEN;
        let half = dim / 2;
        let max_period: f32 = 10000.0;

        let t_data = t.to_dtype(DType::F32)?.to_vec()?;
        let batch = t_data.len();
        let mut emb_data = vec![0.0f32; batch * dim];
        for b in 0..batch {
            let t_val = t_data[b];
            for i in 0..half {
                let freq = (-f32::ln(max_period) * (i as f32) / (half as f32)).exp();
                let angle = t_val * freq;
                emb_data[b * dim + i] = angle.cos();
                emb_data[b * dim + half + i] = angle.sin();
            }
        }
        let emb = Tensor::from_vec_dtype(
            emb_data, Shape::from_dims(&[batch, dim]),
            self.device.clone(), DType::BF16,
        )?;
        // hidden = SiLU(Linear(emb))
        let hidden = self.linear_no_bias(&emb, "t_embedder.1.linear_1.weight")?.silu()?;
        let base_adaln = self.linear_no_bias(&hidden, "t_embedder.1.linear_2.weight")?;
        let t_cond = self.rms_norm(&emb, "t_embedding_norm.weight", 1e-6)?;
        Ok((t_cond, base_adaln))
    }

    // ─── AdaLN-LoRA ─────────────────────────────────────────────────────────

    fn adaln_modulation(
        &self,
        t_cond: &Tensor,
        base_adaln: &Tensor,
        prefix: &str,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let t_silu = t_cond.silu()?;
        let h = self.linear_no_bias(&t_silu, &format!("{prefix}.1.weight"))?;
        let mod_out = self.linear_no_bias(&h, &format!("{prefix}.2.weight"))?;
        let mod_out = mod_out.add(base_adaln)?;
        let dim = HIDDEN;
        let shift = mod_out.narrow(1, 0, dim)?;
        let scale = mod_out.narrow(1, dim, dim)?;
        let gate = mod_out.narrow(1, 2 * dim, dim)?;
        Ok((shift, scale, gate))
    }

    fn final_adaln_modulation(
        &self,
        t_cond: &Tensor,
        base_adaln: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let t_silu = t_cond.silu()?;
        let h = self.linear_no_bias(&t_silu, "final_layer.adaln_modulation.1.weight")?;
        let mod_out = self.linear_no_bias(&h, "final_layer.adaln_modulation.2.weight")?;
        let dim = HIDDEN;
        let adaln_slice = base_adaln.narrow(1, 0, 2 * dim)?;
        let mod_out = mod_out.add(&adaln_slice)?;
        let shift = mod_out.narrow(1, 0, dim)?;
        let scale = mod_out.narrow(1, dim, dim)?;
        Ok((shift, scale))
    }

    /// `(1 + scale) * LN(x) + shift` with primitive ops (autograd-correct).
    /// `x: [B, S, D]`, `shift/scale: [B, D]`.
    fn apply_adaln(&self, x: &Tensor, shift: &Tensor, scale: &Tensor) -> Result<Tensor> {
        let normed = primitive_layer_norm(x, NORM_EPS)?;
        let scale_3d = scale.unsqueeze(1)?.add_scalar(1.0)?;
        let shift_3d = shift.unsqueeze(1)?;
        normed.mul(&scale_3d)?.add(&shift_3d).map_err(Into::into)
    }

    // ─── Self-attention with 3D RoPE ────────────────────────────────────────

    fn self_attention(
        &self,
        x: &Tensor,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
        block: usize,
    ) -> Result<Tensor> {
        let prefix = format!("blocks.{block}.self_attn");
        let dims = x.shape().dims().to_vec();
        let (b, seq) = (dims[0], dims[1]);

        let q = self.linear_lora(x, &format!("{prefix}.q_proj.weight"), Some((block, AnimaLoraTarget::SaQ)))?;
        let k = self.linear_lora(x, &format!("{prefix}.k_proj.weight"), Some((block, AnimaLoraTarget::SaK)))?;
        let v = self.linear_lora(x, &format!("{prefix}.v_proj.weight"), Some((block, AnimaLoraTarget::SaV)))?;

        let q = q.reshape(&[b, seq, HEADS, HEAD_DIM])?.permute(&[0, 2, 1, 3])?;
        let k = k.reshape(&[b, seq, HEADS, HEAD_DIM])?.permute(&[0, 2, 1, 3])?;
        let v = v.reshape(&[b, seq, HEADS, HEAD_DIM])?.permute(&[0, 2, 1, 3])?;

        let q = self.rms_norm_per_head_bhsd(&q, &format!("{prefix}.q_norm.weight"))?;
        let k = self.rms_norm_per_head_bhsd(&k, &format!("{prefix}.k_norm.weight"))?;

        // 3D RoPE — `rope_halfsplit_bf16` IS autograd-aware (records Op via
        // `is_recording()` check at bf16_ops.rs:770).
        let q = q.contiguous()?;
        let k = k.contiguous()?;
        let q = flame_core::bf16_ops::rope_halfsplit_bf16(&q, rope_cos, rope_sin)?;
        let k = flame_core::bf16_ops::rope_halfsplit_bf16(&k, rope_cos, rope_sin)?;

        let out = flame_core::attention::sdpa(&q, &k, &v, None)?;
        let out = out.permute(&[0, 2, 1, 3])?.reshape(&[b, seq, HEADS * HEAD_DIM])?;
        self.linear_lora(&out, &format!("{prefix}.output_proj.weight"), Some((block, AnimaLoraTarget::SaOut)))
    }

    // ─── Cross-attention (no RoPE) ──────────────────────────────────────────

    fn cross_attention(
        &self,
        x: &Tensor,
        context: &Tensor,
        block: usize,
    ) -> Result<Tensor> {
        let prefix = format!("blocks.{block}.cross_attn");
        let dims = x.shape().dims().to_vec();
        let (b, seq_img) = (dims[0], dims[1]);
        let seq_txt = context.shape().dims()[1];

        let q = self.linear_lora(x, &format!("{prefix}.q_proj.weight"), Some((block, AnimaLoraTarget::CaQ)))?;
        let k = self.linear_lora(context, &format!("{prefix}.k_proj.weight"), Some((block, AnimaLoraTarget::CaK)))?;
        let v = self.linear_lora(context, &format!("{prefix}.v_proj.weight"), Some((block, AnimaLoraTarget::CaV)))?;

        let q = q.reshape(&[b, seq_img, HEADS, HEAD_DIM])?;
        let k = k.reshape(&[b, seq_txt, HEADS, HEAD_DIM])?;
        let v = v.reshape(&[b, seq_txt, HEADS, HEAD_DIM])?;

        let q = self.rms_norm_per_head(&q, &format!("{prefix}.q_norm.weight"))?;
        let k = self.rms_norm_per_head(&k, &format!("{prefix}.k_norm.weight"))?;

        let q = q.permute(&[0, 2, 1, 3])?;
        let k = k.permute(&[0, 2, 1, 3])?;
        let v = v.permute(&[0, 2, 1, 3])?;

        let out = flame_core::attention::sdpa(&q, &k, &v, None)?;
        let out = out.permute(&[0, 2, 1, 3])?.reshape(&[b, seq_img, HEADS * HEAD_DIM])?;
        self.linear_lora(&out, &format!("{prefix}.output_proj.weight"), Some((block, AnimaLoraTarget::CaOut)))
    }

    // ─── GELU MLP ───────────────────────────────────────────────────────────

    fn mlp(&self, x: &Tensor, block: usize) -> Result<Tensor> {
        let prefix = format!("blocks.{block}.mlp");
        let h = self.linear_lora(x, &format!("{prefix}.layer1.weight"), Some((block, AnimaLoraTarget::MlpL1)))?;
        let h = h.gelu()?;
        self.linear_lora(&h, &format!("{prefix}.layer2.weight"), Some((block, AnimaLoraTarget::MlpL2)))
    }

    // ─── Transformer block ──────────────────────────────────────────────────

    fn transformer_block(
        &self,
        x: &Tensor,
        context: &Tensor,
        t_cond: &Tensor,
        base_adaln: &Tensor,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
        block: usize,
    ) -> Result<Tensor> {
        // BF16 residual stream — flame-core convention. The previous F32
        // residual + per-sub-block BF16↔F32 round-tripping was a cargo-cult
        // import from the inference port; no precision is preserved by
        // casting BF16 outputs up to F32, adding, then casting back down.
        // Per-op precision (rms_norm, layer_norm, attention softmax) lives
        // inside those ops via `primitive_*`. Keeps activation cost ~3×
        // smaller and removes the F32 ↔ BF16 cast tape.
        let mut x = x.clone();

        // Self-attention.
        let (shift_sa, scale_sa, gate_sa) = self.adaln_modulation(
            t_cond, base_adaln,
            &format!("blocks.{block}.adaln_modulation_self_attn"),
        )?;
        let x_mod = self.apply_adaln(&x, &shift_sa, &scale_sa)?;
        let attn_out = self.self_attention(&x_mod, rope_cos, rope_sin, block)?;
        let gate_sa_3d = gate_sa.unsqueeze(1)?;
        x = x.add(&attn_out.mul(&gate_sa_3d)?)?;

        // Cross-attention.
        let (shift_ca, scale_ca, gate_ca) = self.adaln_modulation(
            t_cond, base_adaln,
            &format!("blocks.{block}.adaln_modulation_cross_attn"),
        )?;
        let x_mod = self.apply_adaln(&x, &shift_ca, &scale_ca)?;
        let cross_out = self.cross_attention(&x_mod, context, block)?;
        let gate_ca_3d = gate_ca.unsqueeze(1)?;
        x = x.add(&cross_out.mul(&gate_ca_3d)?)?;

        // MLP.
        let (shift_mlp, scale_mlp, gate_mlp) = self.adaln_modulation(
            t_cond, base_adaln,
            &format!("blocks.{block}.adaln_modulation_mlp"),
        )?;
        let x_mod = self.apply_adaln(&x, &shift_mlp, &scale_mlp)?;
        let mlp_out = self.mlp(&x_mod, block)?;
        let gate_mlp_3d = gate_mlp.unsqueeze(1)?;
        x = x.add(&mlp_out.mul(&gate_mlp_3d)?)?;

        Ok(x)
    }

    // ─── Final layer ────────────────────────────────────────────────────────

    fn final_layer(&self, x: &Tensor, t_cond: &Tensor, base_adaln: &Tensor) -> Result<Tensor> {
        let (shift, scale) = self.final_adaln_modulation(t_cond, base_adaln)?;
        let x_mod = self.apply_adaln(x, &shift, &scale)?;
        self.linear_no_bias(&x_mod, "final_layer.linear.weight")
    }

    // ─── Patchify / Unpatchify (5D NHWC C-last) ─────────────────────────────

    /// `x: [B, T, H, W, C]` (C=16). Pads with one zero channel for
    /// padding_mask, then patchify: `[B, T*nH*nW, (C+1)*pH*pW]`.
    fn patchify(&self, x: &Tensor) -> Result<(Tensor, usize, usize, usize)> {
        let dims = x.shape().dims().to_vec();
        let (b, t, h, w, c) = (dims[0], dims[1], dims[2], dims[3], dims[4]);
        let ph = PATCH_SPATIAL;
        let nh = h / ph;
        let nw = w / ph;
        // Append zero padding-mask channel.
        let mask = Tensor::zeros_dtype(
            Shape::from_dims(&[b, t, h, w, 1]), DType::BF16, self.device.clone(),
        )?;
        let x_padded = Tensor::cat(&[x, &mask], 4)?;
        let c_pad = c + 1; // 17
        let x_r = x_padded.reshape(&[b, t, nh, ph, nw, ph, c_pad])?;
        // einops: "b c (t r) (h m) (w n) -> b t h w (c r m n)"
        let x_p = x_r.permute(&[0, 1, 2, 4, 6, 3, 5])?.contiguous()?;
        let num_patches = t * nh * nw;
        let patch_dim = ph * ph * c_pad; // 68
        let x_flat = x_p.reshape(&[b, num_patches, patch_dim])?;
        Ok((x_flat, t, nh, nw))
    }

    fn unpatchify(&self, x: &Tensor, t: usize, nh: usize, nw: usize) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let b = dims[0];
        let ph = PATCH_SPATIAL;
        let c = IN_CHANNELS;
        let x_r = x.reshape(&[b, t, nh, nw, ph, ph, c])?;
        let x_p = x_r.permute(&[0, 1, 2, 4, 3, 5, 6])?.contiguous()?;
        x_p.reshape(&[b, t, nh * ph, nw * ph, c]).map_err(Into::into)
    }

    fn patch_embed(&self, patches: &Tensor) -> Result<Tensor> {
        self.linear_no_bias(patches, "x_embedder.proj.1.weight")
    }

    // ─── LLM Adapter ────────────────────────────────────────────────────────
    //
    // The LLM adapter has no LoRA targets in Phase A (kohya defaults
    // `train_llm_adapter=False`), so all linears here are base-only.

    fn llm_adapter(&self, token_ids: &Tensor, llm_hidden: &Tensor) -> Result<Tensor> {
        let prefix = "llm_adapter";
        let b = token_ids.shape().dims()[0];
        let seq_len = token_ids.shape().dims()[1];
        let dim = ADAPTER_DIM;
        let num_heads = ADAPTER_HEADS;
        let head_dim = ADAPTER_HEAD_DIM;

        let embed_w = self.w(&format!("{prefix}.embed.weight"))?;
        let x = embedding_lookup(embed_w, token_ids, dim, &self.device)?;

        let (rope_cos, rope_sin) = build_1d_rope(seq_len, head_dim, ROPE_THETA, &self.device)?;

        let mut x = x;
        for j in 0..ADAPTER_BLOCKS {
            let bp = format!("{prefix}.blocks.{j}");

            // Self-attn.
            let x_norm = self.rms_norm(&x, &format!("{bp}.norm_self_attn.weight"), 1e-6)?;
            let q = self.linear_no_bias(&x_norm, &format!("{bp}.self_attn.q_proj.weight"))?;
            let k = self.linear_no_bias(&x_norm, &format!("{bp}.self_attn.k_proj.weight"))?;
            let v = self.linear_no_bias(&x_norm, &format!("{bp}.self_attn.v_proj.weight"))?;

            let q = q.reshape(&[b, seq_len, num_heads, head_dim])?;
            let k = k.reshape(&[b, seq_len, num_heads, head_dim])?;
            let v = v.reshape(&[b, seq_len, num_heads, head_dim])?;

            let q = self.rms_norm_per_head(&q, &format!("{bp}.self_attn.q_norm.weight"))?;
            let k = self.rms_norm_per_head(&k, &format!("{bp}.self_attn.k_norm.weight"))?;

            let q = apply_rope_cossin(&q, &rope_cos, &rope_sin)?;
            let k = apply_rope_cossin(&k, &rope_cos, &rope_sin)?;

            let q = q.permute(&[0, 2, 1, 3])?;
            let k = k.permute(&[0, 2, 1, 3])?;
            let v = v.permute(&[0, 2, 1, 3])?;

            let attn = flame_core::attention::sdpa(&q, &k, &v, None)?;
            let attn = attn.permute(&[0, 2, 1, 3])?.reshape(&[b, seq_len, num_heads * head_dim])?;
            let attn = self.linear_no_bias(&attn, &format!("{bp}.self_attn.o_proj.weight"))?;
            x = x.add(&attn)?;

            // Cross-attn against Qwen3.
            let x_norm = self.rms_norm(&x, &format!("{bp}.norm_cross_attn.weight"), 1e-6)?;
            let q = self.linear_no_bias(&x_norm, &format!("{bp}.cross_attn.q_proj.weight"))?;
            let k = self.linear_no_bias(llm_hidden, &format!("{bp}.cross_attn.k_proj.weight"))?;
            let v = self.linear_no_bias(llm_hidden, &format!("{bp}.cross_attn.v_proj.weight"))?;

            let seq_llm = llm_hidden.shape().dims()[1];
            let q = q.reshape(&[b, seq_len, num_heads, head_dim])?;
            let k = k.reshape(&[b, seq_llm, num_heads, head_dim])?;
            let v = v.reshape(&[b, seq_llm, num_heads, head_dim])?;

            let q = self.rms_norm_per_head(&q, &format!("{bp}.cross_attn.q_norm.weight"))?;
            let k = self.rms_norm_per_head(&k, &format!("{bp}.cross_attn.k_norm.weight"))?;

            let (q_cos, q_sin) = build_1d_rope(seq_len, head_dim, ROPE_THETA, &self.device)?;
            let (k_cos, k_sin) = build_1d_rope(seq_llm, head_dim, ROPE_THETA, &self.device)?;
            let q = apply_rope_cossin(&q, &q_cos, &q_sin)?;
            let k = apply_rope_cossin(&k, &k_cos, &k_sin)?;

            let q = q.permute(&[0, 2, 1, 3])?;
            let k = k.permute(&[0, 2, 1, 3])?;
            let v = v.permute(&[0, 2, 1, 3])?;
            let cross = flame_core::attention::sdpa(&q, &k, &v, None)?;
            let cross = cross.permute(&[0, 2, 1, 3])?.reshape(&[b, seq_len, num_heads * head_dim])?;
            let cross = self.linear_no_bias(&cross, &format!("{bp}.cross_attn.o_proj.weight"))?;
            x = x.add(&cross)?;

            // MLP (with bias).
            let x_norm = self.rms_norm(&x, &format!("{bp}.norm_mlp.weight"), 1e-6)?;
            let h = self.linear_with_bias(
                &x_norm,
                &format!("{bp}.mlp.0.weight"),
                &format!("{bp}.mlp.0.bias"),
            )?;
            let h = h.gelu()?;
            let mlp_out = self.linear_with_bias(
                &h,
                &format!("{bp}.mlp.2.weight"),
                &format!("{bp}.mlp.2.bias"),
            )?;
            x = x.add(&mlp_out)?;
        }

        let x = self.linear_no_bias(&x, &format!("{prefix}.out_proj.weight"))?;
        let x = self.rms_norm(&x, &format!("{prefix}.norm.weight"), 1e-6)?;
        Ok(x)
    }

    // ─── Public forward ─────────────────────────────────────────────────────

    /// Trainer-facing forward.
    ///
    /// `noisy`:        `[B, 16, H, W]` BF16 (already noised, scaled latent)
    /// `timestep`:     `[B]` BF16/F32, in `[0, 1]` (kohya divides by 1000 before
    ///                 calling Anima — see `anima_train_network.py:279`)
    /// `cap_feats`:    `[B, seq, 1024]` Qwen3-0.6B last_hidden_state
    /// `cap_mask`:     `[B, seq]` 1.0 at valid tokens (currently unused; kohya
    ///                 doesn't apply a mask at the cross-attn input either)
    /// `t5_input_ids`: `[B, t5_seq]` F32 (i32 cast to F32 for safetensors I/O)
    /// `t5_attn_mask`: `[B, t5_seq]` (currently unused at this layer)
    ///
    /// Returns `[B, 16, H, W]` predicted velocity (rectified-flow target = `noise - clean`).
    pub fn forward(
        &mut self,
        noisy: &Tensor,
        timestep: &Tensor,
        cap_feats: &Tensor,
        _cap_mask: Option<&Tensor>,
        t5_input_ids: Option<&Tensor>,
        _t5_attn_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let in_dims = noisy.shape().dims().to_vec();
        if in_dims.len() != 4 {
            return Err(crate::EriDiffusionError::Model(format!(
                "AnimaModel::forward expected [B,16,H,W], got {:?}", in_dims
            )));
        }
        let (b, c, h, w) = (in_dims[0], in_dims[1], in_dims[2], in_dims[3]);
        if c != IN_CHANNELS {
            return Err(crate::EriDiffusionError::Model(format!(
                "AnimaModel::forward expected 16 channels, got {c}"
            )));
        }
        // 4D NCHW → 5D NTHWC (T=1, C-last).
        let x = noisy
            .reshape(&[b, c, 1, h, w])?           // [B, C, 1, H, W]
            .permute(&[0, 2, 3, 4, 1])?            // [B, 1, H, W, C]
            .contiguous()?;

        // 1. Timestep conditioning.
        let timestep_bf16 = timestep.to_dtype(DType::BF16)?;
        let (t_cond, base_adaln) = self.prepare_timestep(&timestep_bf16)?;

        // 2. Patchify + embed.
        let (patches, t_frames, nh, nw) = self.patchify(&x)?;
        let mut x_hidden = self.patch_embed(&patches)?;

        // 3. Build 3D RoPE cos/sin tables for self-attn (T=1, image case).
        let (rope_cos, rope_sin) = build_3d_rope_cossin(t_frames, nh, nw, HEAD_DIM, &self.device)?;

        // 4. Encode context via LLM Adapter (T5 token IDs + Qwen3 hidden).
        let context = if let Some(t5_ids_f32) = t5_input_ids {
            self.llm_adapter(t5_ids_f32, cap_feats)?
        } else {
            // Fallback: pass Qwen3 hidden directly (won't match training distribution
            // but lets us forward when the cache lacks T5 ids — used only by the
            // placeholder smoke path).
            cap_feats.clone()
        };

        // 5. 28 transformer blocks.
        for i in 0..NUM_BLOCKS {
            x_hidden = self.transformer_block(
                &x_hidden, &context, &t_cond, &base_adaln,
                &rope_cos, &rope_sin, i,
            )?;
        }

        // 6. Final layer.
        let x_out = self.final_layer(&x_hidden, &t_cond, &base_adaln)?;

        // 7. Unpatchify → [B, 1, H, W, 16] → [B, 16, H, W].
        let out_5d = self.unpatchify(&x_out, t_frames, nh, nw)?;
        let out_4d = out_5d.permute(&[0, 4, 1, 2, 3])?
            .reshape(&[b, c, h, w])?;
        Ok(out_4d)
    }

    pub fn named_parameters(&self) -> Vec<(String, Parameter)> {
        self.bundle.named_parameters()
    }
}

impl TrainableModel for AnimaModel {
    fn forward(
        &mut self,
        noisy: &Tensor,
        timestep: &Tensor,
        context: &[Tensor],
        _pooled: Option<&Tensor>,
    ) -> Result<Tensor> {
        let cap_feats = context.first().ok_or_else(|| {
            crate::EriDiffusionError::Model("Anima needs cap_feats in context[0]".into())
        })?.clone();
        let cap_mask = context.get(1).cloned();
        let t5_ids = context.get(2).cloned();
        let t5_mask = context.get(3).cloned();
        AnimaModel::forward(
            self, noisy, timestep, &cap_feats,
            cap_mask.as_ref(), t5_ids.as_ref(), t5_mask.as_ref(),
        )
    }

    fn parameters(&self) -> Vec<Parameter> {
        self.bundle.parameters()
    }

    fn post_optimizer_step(&mut self) {}

    fn save_weights(&self, path: &str) -> Result<()> {
        if !self.is_lora {
            return Err(crate::EriDiffusionError::Model(
                "AnimaModel::save_weights: non-LoRA path not implemented".into(),
            ));
        }
        self.bundle.save(std::path::Path::new(path))
    }

    fn load_weights(&mut self, path: &str) -> Result<()> {
        if !self.is_lora {
            return Err(crate::EriDiffusionError::Model(
                "AnimaModel::load_weights: non-LoRA path not implemented".into(),
            ));
        }
        self.bundle.load(std::path::Path::new(path), &self.device)
    }
}

// ─── Standalone helpers ─────────────────────────────────────────────────────

/// F32-internal RMSNorm built from primitive autograd ops. Same pattern used
/// throughout zimage.rs to bypass the inference-only `cuda_ops_bf16::rms_norm_bf16`.
fn primitive_rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> flame_core::Result<Tensor> {
    let out_dtype = x.dtype();
    let x_f32 = if out_dtype == DType::F32 { x.clone() } else { x.to_dtype(DType::F32)? };
    let weight_f32 = if weight.dtype() == DType::F32 {
        weight.clone()
    } else {
        weight.to_dtype(DType::F32)?
    };
    let sq = x_f32.mul(&x_f32)?;
    let dims = sq.shape().dims().to_vec();
    let last = dims.len() - 1;
    let n = dims[last] as f32;
    let mean_sq = sq.sum_dim_keepdim(last)?.mul_scalar(1.0 / n)?;
    let inv_rms = mean_sq.add_scalar(eps)?.rsqrt()?;
    let normed = x_f32.mul(&inv_rms)?;
    let scaled = normed.mul(&weight_f32)?;
    if out_dtype == DType::F32 { Ok(scaled) } else { scaled.to_dtype(out_dtype) }
}

/// F32-internal LayerNorm (no scale/bias). Pair with adaLN's `(1+scale)*y+shift`.
fn primitive_layer_norm(x: &Tensor, eps: f32) -> flame_core::Result<Tensor> {
    let out_dtype = x.dtype();
    let x_f32 = if out_dtype == DType::F32 { x.clone() } else { x.to_dtype(DType::F32)? };
    let dims = x_f32.shape().dims().to_vec();
    let last = dims.len() - 1;
    let n = dims[last] as f32;
    let mean = x_f32.sum_dim_keepdim(last)?.mul_scalar(1.0 / n)?;
    let centered = x_f32.sub(&mean)?;
    let sq = centered.mul(&centered)?;
    let var = sq.sum_dim_keepdim(last)?.mul_scalar(1.0 / n)?;
    let inv_std = var.add_scalar(eps)?.rsqrt()?;
    let normed = centered.mul(&inv_std)?;
    if out_dtype == DType::F32 { Ok(normed) } else { normed.to_dtype(out_dtype) }
}

/// Embedding lookup. Indices are stored as F32 in safetensors I/O.
fn embedding_lookup(
    weight: &Tensor,
    indices: &Tensor,
    dim: usize,
    device: &Arc<CudaDevice>,
) -> flame_core::Result<Tensor> {
    let idx_dims = indices.shape().dims().to_vec();
    let b = idx_dims[0];
    let s = idx_dims[1];
    let idx_f32 = indices.to_dtype(DType::F32)?;
    let idx_flat = idx_f32.reshape(&[b * s])?;
    let idx_data = idx_flat.to_vec()?;
    let vocab = weight.shape().dims()[0];
    let weight_f32 = weight.to_dtype(DType::F32)?;
    let weight_flat = weight_f32.reshape(&[vocab * dim])?;
    let weight_data = weight_flat.to_vec()?;
    let mut out_data = vec![0.0f32; b * s * dim];
    for i in 0..(b * s) {
        let idx = idx_data[i] as usize;
        if idx < vocab {
            let src = idx * dim;
            let dst = i * dim;
            out_data[dst..dst + dim].copy_from_slice(&weight_data[src..src + dim]);
        }
    }
    Tensor::from_vec_dtype(
        out_data, Shape::from_dims(&[b, s, dim]),
        device.clone(), DType::BF16,
    )
}

fn build_1d_rope(
    seq_len: usize,
    head_dim: usize,
    theta: f32,
    device: &Arc<CudaDevice>,
) -> flame_core::Result<(Tensor, Tensor)> {
    let half = head_dim / 2;
    let mut cos_data = vec![0.0f32; seq_len * half];
    let mut sin_data = vec![0.0f32; seq_len * half];
    for pos in 0..seq_len {
        for i in 0..half {
            let freq = 1.0 / theta.powf(2.0 * (i as f32) / (head_dim as f32));
            let angle = (pos as f32) * freq;
            cos_data[pos * half + i] = angle.cos();
            sin_data[pos * half + i] = angle.sin();
        }
    }
    let cos = Tensor::from_vec_dtype(
        cos_data, Shape::from_dims(&[seq_len, half]),
        device.clone(), DType::BF16,
    )?;
    let sin = Tensor::from_vec_dtype(
        sin_data, Shape::from_dims(&[seq_len, half]),
        device.clone(), DType::BF16,
    )?;
    Ok((cos, sin))
}

/// 1D RoPE half-split, `x: [B, S, H, D]`, cos/sin `[S, D/2]`.
fn apply_rope_cossin(
    x: &Tensor,
    rope_cos: &Tensor,
    rope_sin: &Tensor,
) -> flame_core::Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    let (b, s, h, d) = (dims[0], dims[1], dims[2], dims[3]);
    let half = d / 2;
    let x1 = x.narrow(3, 0, half)?;
    let x2 = x.narrow(3, half, half)?;
    let cos = rope_cos.reshape(&[1, s, 1, half])?;
    let sin = rope_sin.reshape(&[1, s, 1, half])?;
    let new_x1 = x1.mul(&cos)?.sub(&x2.mul(&sin)?)?;
    let new_x2 = x2.mul(&cos)?.add(&x1.mul(&sin)?)?;
    let f1 = new_x1.reshape(&[b * s * h, half])?;
    let f2 = new_x2.reshape(&[b * s * h, half])?;
    let result = Tensor::cat(&[&f1, &f2], 1)?;
    result.reshape(&[b, s, h, d])
}

/// Build 3D RoPE cos/sin tables for the fused kernel.
/// Returns `(cos, sin)` each `[1, 1, S, D/2]` where `S = T*nH*nW`.
fn build_3d_rope_cossin(
    t_frames: usize,
    nh: usize,
    nw: usize,
    head_dim: usize,
    device: &Arc<CudaDevice>,
) -> flame_core::Result<(Tensor, Tensor)> {
    let half_d = head_dim / 2;
    let total_seq = t_frames * nh * nw;
    let full_d = half_d * 2;
    let dim_h: usize = full_d / 6 * 2; // 42
    let dim_w: usize = dim_h;           // 42
    let dim_t: usize = full_d - 2 * dim_h; // 44
    let bins_t = dim_t / 2;
    let bins_h = dim_h / 2;
    let bins_w = dim_w / 2;

    let base_theta: f64 = 10000.0;
    let h_ntk = 4.0f64.powf(dim_h as f64 / (dim_h as f64 - 2.0));
    let w_ntk = 4.0f64.powf(dim_w as f64 / (dim_w as f64 - 2.0));
    let t_ntk = 1.0f64.powf(dim_t as f64 / (dim_t as f64 - 2.0));
    let theta_h = (base_theta * h_ntk) as f32;
    let theta_w = (base_theta * w_ntk) as f32;
    let theta_t = (base_theta * t_ntk) as f32;

    let freqs_t: Vec<f32> = (0..bins_t)
        .map(|i| 1.0 / theta_t.powf((2 * i) as f32 / dim_t as f32))
        .collect();
    let freqs_h: Vec<f32> = (0..bins_h)
        .map(|i| 1.0 / theta_h.powf((2 * i) as f32 / dim_h as f32))
        .collect();
    let freqs_w: Vec<f32> = (0..bins_w)
        .map(|i| 1.0 / theta_w.powf((2 * i) as f32 / dim_w as f32))
        .collect();

    // Build [S, half_d] cos/sin (the kernel uses half-split internally).
    let mut cos_data = vec![0.0f32; total_seq * half_d];
    let mut sin_data = vec![0.0f32; total_seq * half_d];
    for tf in 0..t_frames {
        for ih in 0..nh {
            for iw in 0..nw {
                let seq_idx = tf * nh * nw + ih * nw + iw;
                let base = seq_idx * half_d;
                let mut off = 0;
                for (fi, &freq) in freqs_t.iter().enumerate() {
                    let angle = (tf as f32) * freq;
                    cos_data[base + off + fi] = angle.cos();
                    sin_data[base + off + fi] = angle.sin();
                }
                off += bins_t;
                for (fi, &freq) in freqs_h.iter().enumerate() {
                    let angle = (ih as f32) * freq;
                    cos_data[base + off + fi] = angle.cos();
                    sin_data[base + off + fi] = angle.sin();
                }
                off += bins_h;
                for (fi, &freq) in freqs_w.iter().enumerate() {
                    let angle = (iw as f32) * freq;
                    cos_data[base + off + fi] = angle.cos();
                    sin_data[base + off + fi] = angle.sin();
                }
            }
        }
    }
    let cos = Tensor::from_vec_dtype(
        cos_data, Shape::from_dims(&[1, 1, total_seq, half_d]),
        device.clone(), DType::BF16,
    )?;
    let sin = Tensor::from_vec_dtype(
        sin_data, Shape::from_dims(&[1, 1, total_seq, half_d]),
        device.clone(), DType::BF16,
    )?;
    Ok((cos, sin))
}

// ─── TODO (Phase B) ─────────────────────────────────────────────────────────
// 1. Add LoRA targets for AdaLN modulation linears (3 × 2 per block = 168 modules)
//    plus PatchEmbed (1) + TimestepEmbedding (2) + FinalLayer (3) — full kohya
//    list is ~454 modules, we currently train 280. Per-target slot/in_features
//    differ; would extend `LORA_SHAPES` and the bundle indexing.
// 2. LLM-Adapter LoRA targets (when `train_llm_adapter=True`) — 6 blocks ×
//    (3 self_attn + 3 cross_attn + 2 mlp) = 48 modules.
// 3. Validate end-to-end loss-curve parity vs kohya `anima_train_network.py`
//    on a 100-step sweep (cf. `feedback_zimage_trainer_not_converged.md` —
//    forward parity ≠ training correctness).
// 4. Wire VAE per-channel normalization (`QWEN_VAE_LATENT_MEAN/STD`) into
//    `prepare_anima` (currently saves raw z) and into `sample_anima` decode.
