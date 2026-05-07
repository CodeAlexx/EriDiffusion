//! Flux 1 DiT model — correct implementation ported from flame-diffusion flux1-trainer.
//! Architecture constants match BFL/flux-1-dev.

use std::collections::HashMap;
use std::sync::Arc;
use cudarc::driver::CudaDevice;
use flame_core::{parameter::Parameter, DType, Shape, Tensor};
use flame_core::autograd::AutogradContext;
use crate::config::TrainConfig;
use crate::lora::LoRALinear;
use crate::models::TrainableModel;
use crate::Result;

pub const NUM_DOUBLE: usize = 19;
pub const NUM_SINGLE: usize = 38;
pub const DIM: usize = 3072;
pub const NUM_HEADS: usize = 24;
pub const HEAD_DIM: usize = 128;
pub const IN_CHANNELS: usize = 64;
pub const T5_DIM: usize = 4096;
pub const MLP_HIDDEN: usize = 12288;
pub const TIMESTEP_DIM: usize = 256;
pub const VECTOR_DIM: usize = 768;
pub const ROPE_AXES: [usize; 3] = [16, 56, 56];
pub const ROPE_THETA: f64 = 10000.0;
pub const NORM_EPS: f32 = 1e-6;

// ── LoRA targets ────────────────────────────────────────────────────
//
// Audit fix FLUX_VERIFY §H4 / §H5 / SKEPTIC §H4 / §H5: the OT-Python wrapper
// hooks **every** `nn.Linear` in the diffusers `FluxTransformer2DModel` whose
// dotted path matches the `attn-mlp` filter. BFL's fused QKV (`img_attn.qkv`,
// `txt_attn.qkv`) appears in diffusers as 3 separate linears (`to_q`, `to_k`,
// `to_v`) and BFL's fused `linear1` in single blocks splits into
// `attn.{to_q,to_k,to_v}` + `proj_mlp`. So the canonical OT adapter granularity
// per double block is 12 LoRAs (4 attn + 4 attn + 4 MLP) and per single block
// is 5 LoRAs (3 QKV + proj_mlp + proj_out).
//
// Pre-fix the ED-v2 impl had only 4 doubles/block + 2 singles/block (152 total
// vs canonical 418), and the single `Out` adapter wrapped a phantom `DIM→DIM`
// matrix when `linear2` is actually `5*DIM→DIM` (silent shape add over `attn`
// only — MLP-up half got no LoRA correction). This module now matches the
// canonical layout exactly.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DoubleLoraTarget {
    // Attention path (img/txt symmetric).
    ImgQ, ImgK, ImgV, ImgProj,
    TxtQ, TxtK, TxtV, TxtProj,
    // MLP path (Linear → GELU → Linear).
    ImgMlp0, ImgMlp2,
    TxtMlp0, TxtMlp2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SingleLoraTarget {
    // Q/K/V split from BFL fused linear1[:3*DIM].
    Q, K, V,
    // MLP-up half from BFL fused linear1[3*DIM:7*DIM] (DIM → 4*DIM).
    ProjMlp,
    // BFL `linear2` (5*DIM → DIM) — wraps the full fused [attn ‖ mlp_out].
    ProjOut,
}

// ── Model struct ────────────────────────────────────────────────────

pub struct FluxModel {
    pub config: TrainConfig,
    pub device: Arc<CudaDevice>,

    pub shared_weights: HashMap<String, Tensor>,
    pub double_block_weights: Vec<HashMap<String, Tensor>>,
    pub single_block_weights: Vec<HashMap<String, Tensor>>,

    // LoRA
    pub bundle: Option<FluxLoraBundle>,
    pub fft_params: Option<HashMap<String, Parameter>>,
    pub is_full_finetune: bool,

    pub has_guidance: bool,

    /// When Some, double/single block weights are streamed from these shards
    /// per layer instead of being resident. LoRA-only.
    /// Mirrors `ErnieModel::enable_offload`.
    pub offload_shards: Option<Vec<std::path::PathBuf>>,

    /// Default guidance value passed to the model at training time.
    /// 1.0 for Schnell, 3.5 for Dev (matches sd-scripts and EriDiffusion defaults).
    pub guidance_value: f32,
}

#[derive(Clone)]
pub struct FluxLoraBundle {
    pub double_adapters: HashMap<(usize, DoubleLoraTarget), LoRALinear>,
    pub single_adapters: HashMap<(usize, SingleLoraTarget), LoRALinear>,
}

impl FluxLoraBundle {
    /// Build the canonical 418-adapter LoRA bundle.
    /// `seed` is **fixed** (and ignored beyond the single seed=42 invariant) —
    /// each adapter is initialised from the same seed; the autograd graph and
    /// per-adapter shape differentiate them. Audit fix FLUX_VERIFY §H7 /
    /// SKEPTIC §H10 (`feedback_default_seed_42.md`).
    pub fn new(rank: usize, alpha: f32, device: Arc<CudaDevice>, seed: u64) -> Result<Self> {
        let mut da = HashMap::new();
        let mut sa = HashMap::new();
        // Double blocks: 12 adapters/block × 19 blocks = 228.
        for i in 0..NUM_DOUBLE {
            // Attention: Q/K/V split from BFL fused img_attn.qkv (3*DIM → 3 separate DIM→DIM).
            da.insert((i, DoubleLoraTarget::ImgQ),    LoRALinear::new(DIM, DIM, rank, alpha, device.clone(), seed)?);
            da.insert((i, DoubleLoraTarget::ImgK),    LoRALinear::new(DIM, DIM, rank, alpha, device.clone(), seed)?);
            da.insert((i, DoubleLoraTarget::ImgV),    LoRALinear::new(DIM, DIM, rank, alpha, device.clone(), seed)?);
            da.insert((i, DoubleLoraTarget::ImgProj), LoRALinear::new(DIM, DIM, rank, alpha, device.clone(), seed)?);
            da.insert((i, DoubleLoraTarget::TxtQ),    LoRALinear::new(DIM, DIM, rank, alpha, device.clone(), seed)?);
            da.insert((i, DoubleLoraTarget::TxtK),    LoRALinear::new(DIM, DIM, rank, alpha, device.clone(), seed)?);
            da.insert((i, DoubleLoraTarget::TxtV),    LoRALinear::new(DIM, DIM, rank, alpha, device.clone(), seed)?);
            da.insert((i, DoubleLoraTarget::TxtProj), LoRALinear::new(DIM, DIM, rank, alpha, device.clone(), seed)?);
            // MLP up + down (DIM → MLP_HIDDEN → DIM).
            da.insert((i, DoubleLoraTarget::ImgMlp0), LoRALinear::new(DIM, MLP_HIDDEN, rank, alpha, device.clone(), seed)?);
            da.insert((i, DoubleLoraTarget::ImgMlp2), LoRALinear::new(MLP_HIDDEN, DIM, rank, alpha, device.clone(), seed)?);
            da.insert((i, DoubleLoraTarget::TxtMlp0), LoRALinear::new(DIM, MLP_HIDDEN, rank, alpha, device.clone(), seed)?);
            da.insert((i, DoubleLoraTarget::TxtMlp2), LoRALinear::new(MLP_HIDDEN, DIM, rank, alpha, device.clone(), seed)?);
        }
        // Single blocks: 5 adapters/block × 38 blocks = 190.
        // BFL `linear1` is fused [Q | K | V | proj_mlp] = 7*DIM output (3*DIM + 4*DIM).
        // BFL `linear2` is `5*DIM → DIM` (input is cat([attn, mlp_out]) where mlp_out is 4*DIM).
        for i in 0..NUM_SINGLE {
            sa.insert((i, SingleLoraTarget::Q),       LoRALinear::new(DIM, DIM, rank, alpha, device.clone(), seed)?);
            sa.insert((i, SingleLoraTarget::K),       LoRALinear::new(DIM, DIM, rank, alpha, device.clone(), seed)?);
            sa.insert((i, SingleLoraTarget::V),       LoRALinear::new(DIM, DIM, rank, alpha, device.clone(), seed)?);
            sa.insert((i, SingleLoraTarget::ProjMlp), LoRALinear::new(DIM, 4*DIM, rank, alpha, device.clone(), seed)?);
            sa.insert((i, SingleLoraTarget::ProjOut), LoRALinear::new(5*DIM, DIM, rank, alpha, device.clone(), seed)?);
        }
        Ok(Self { double_adapters: da, single_adapters: sa })
    }

    pub fn parameters(&self) -> Vec<Parameter> {
        let mut p = Vec::new();
        for l in self.double_adapters.values() { p.extend(l.parameters()); }
        for l in self.single_adapters.values() { p.extend(l.parameters()); }
        p
    }

    /// Canonical (name, Parameter) pairs for full-checkpoint save/resume.
    /// Names match exactly what `<FluxModel as TrainableModel>::save_weights`
    /// writes: `double_blocks.{i}.{double_target_suffix(target)}.lora_{A,B}.weight`
    /// then `single_blocks.{i}.{single_target_suffix(target)}.lora_{A,B}.weight`.
    /// Order is deterministic via sorted (block_idx, target_idx) keys. The
    /// `.alpha` scalars that `save_weights` emits are NOT Parameters and are
    /// intentionally skipped (alpha is restored from CkptHeader on load).
    pub fn named_parameters(&self) -> Vec<(String, Parameter)> {
        let mut out = Vec::with_capacity(
            (self.double_adapters.len() + self.single_adapters.len()) * 2,
        );
        let mut dkeys: Vec<(usize, DoubleLoraTarget)> = self.double_adapters.keys().copied().collect();
        dkeys.sort_by_key(|(i, t)| (*i, *t as usize));
        for (i, target) in dkeys {
            let lora = &self.double_adapters[&(i, target)];
            let prefix = format!("double_blocks.{}.{}", i, double_target_suffix(target));
            out.push((format!("{prefix}.lora_A.weight"), lora.lora_a().clone()));
            out.push((format!("{prefix}.lora_B.weight"), lora.lora_b().clone()));
        }
        let mut skeys: Vec<(usize, SingleLoraTarget)> = self.single_adapters.keys().copied().collect();
        skeys.sort_by_key(|(i, t)| (*i, *t as usize));
        for (i, target) in skeys {
            let lora = &self.single_adapters[&(i, target)];
            let prefix = format!("single_blocks.{}.{}", i, single_target_suffix(target));
            out.push((format!("{prefix}.lora_A.weight"), lora.lora_a().clone()));
            out.push((format!("{prefix}.lora_B.weight"), lora.lora_b().clone()));
        }
        out
    }
}

impl FluxModel {
    pub fn load(
        model_path: &std::path::Path, config: &TrainConfig, device: Arc<CudaDevice>,
    ) -> Result<Self> {
        log::info!("[Flux] loading from {}", model_path.display());
        let all = flame_core::serialization::load_file(model_path, &device)?;
        log::info!("[Flux] {} weight tensors", all.len());

        let has_guidance = all.contains_key("guidance_in.in_layer.weight");
        log::info!("[Flux] guidance_in: {} ({})", has_guidance, if has_guidance {"Dev"} else {"Schnell"});

        let mut shared = HashMap::new();
        let mut db: Vec<_> = (0..NUM_DOUBLE).map(|_| HashMap::new()).collect();
        let mut sb: Vec<_> = (0..NUM_SINGLE).map(|_| HashMap::new()).collect();

        for (key, t) in &all {
            if let Some(rest) = key.strip_prefix("double_blocks.") {
                if let Some(idx) = parse_block_idx(rest, NUM_DOUBLE) { db[idx].insert(key.clone(), t.clone()); continue; }
            }
            if let Some(rest) = key.strip_prefix("single_blocks.") {
                if let Some(idx) = parse_block_idx(rest, NUM_SINGLE) { sb[idx].insert(key.clone(), t.clone()); continue; }
            }
            shared.insert(key.clone(), t.clone());
        }
        drop(all);

        log::info!("[Flux] {} shared, {} double blocks, {} single blocks", shared.len(), db.len(), sb.len());

        let is_fft = !config.is_lora();
        let (bundle, fft_params) = if is_fft {
            let mut params = HashMap::new();
            for (k, t) in &shared { params.insert(k.clone(), Parameter::new(t.to_dtype(DType::F32)?.requires_grad_(true))); }
            for block in &db { for (k, t) in block { params.insert(k.clone(), Parameter::new(t.to_dtype(DType::F32)?.requires_grad_(true))); } }
            for block in &sb { for (k, t) in block { params.insert(k.clone(), Parameter::new(t.to_dtype(DType::F32)?.requires_grad_(true))); } }
            (None, Some(params))
        } else {
            // SEED=42 (single fixed seed — see FluxLoraBundle::new for rationale).
            let b = FluxLoraBundle::new(config.lora_rank as usize, config.lora_alpha as f32, device.clone(), 42u64)?;
            (Some(b), None)
        };

        // Audit fix FLUX_VERIFY §H7 / SKEPTIC §H7: OT canonical training-time
        // guidance is `config.transformer.guidance_scale` (TrainConfig default
        // 1.0 — see `/home/alex/upstream Python/modules/util/config/TrainConfig.py:289`). 3.5
        // is the *inference* default; the sampler binary overrides this field
        // explicitly when generating images. Hardcoding 3.5 here at training
        // time shifted the guidance MLP's input distribution away from where
        // BFL distillation expects it.
        let guidance_value = 1.0;
        Ok(Self { config: config.clone(), device, shared_weights: shared, double_block_weights: db, single_block_weights: sb, bundle, fft_params, is_full_finetune: is_fft, has_guidance, offload_shards: None, guidance_value })
    }

    /// Drop double/single block weights from VRAM and remember source shards
    /// so each block's forward re-loads just that block's tensors. LoRA-only.
    /// Mirrors `ErnieModel::enable_offload`.
    pub fn enable_offload(&mut self, shards: Vec<std::path::PathBuf>) -> Result<()> {
        if self.is_full_finetune {
            return Err(crate::EriDiffusionError::Model("Flux offload requires LoRA mode".into()));
        }
        let mut dropped = 0usize;
        for block in &mut self.double_block_weights {
            dropped += block.len();
            block.clear();
        }
        for block in &mut self.single_block_weights {
            dropped += block.len();
            block.clear();
        }
        log::info!("[Flux] offload: dropped {} per-block weight tensors", dropped);
        flame_core::cuda_alloc_pool::clear_pool_cache();
        flame_core::trim_cuda_mempool(0);
        self.offload_shards = Some(shards);
        Ok(())
    }

    fn stage_double_block(&mut self, idx: usize) -> Result<()> {
        let shards = match &self.offload_shards { None => return Ok(()), Some(s) => s.clone() };
        let prefix = format!("double_blocks.{}.", idx);
        let mut block = HashMap::new();
        for shard in &shards {
            let part = flame_core::serialization::load_file_filtered(
                shard, &self.device, |k| k.starts_with(&prefix),
            )?;
            for (k, v) in part {
                block.insert(k, v.to_dtype(DType::BF16)?);
            }
        }
        self.double_block_weights[idx] = block;
        Ok(())
    }
    fn evict_double_block(&mut self, idx: usize) {
        if self.offload_shards.is_none() { return; }
        self.double_block_weights[idx].clear();
    }
    fn stage_single_block(&mut self, idx: usize) -> Result<()> {
        let shards = match &self.offload_shards { None => return Ok(()), Some(s) => s.clone() };
        let prefix = format!("single_blocks.{}.", idx);
        let mut block = HashMap::new();
        for shard in &shards {
            let part = flame_core::serialization::load_file_filtered(
                shard, &self.device, |k| k.starts_with(&prefix),
            )?;
            for (k, v) in part {
                block.insert(k, v.to_dtype(DType::BF16)?);
            }
        }
        self.single_block_weights[idx] = block;
        Ok(())
    }
    fn evict_single_block(&mut self, idx: usize) {
        if self.offload_shards.is_none() { return; }
        self.single_block_weights[idx].clear();
    }

    // ── Primitives ──────────────────────────────────────────────────

    fn dw(&self, idx: usize, suffix: &str) -> Result<&Tensor> {
        self.double_block_weights[idx].get(&format!("double_blocks.{}.{}", idx, suffix))
            .ok_or_else(|| crate::EriDiffusionError::Model(format!("missing DW: {}.{}", idx, suffix)))
    }
    fn singw(&self, idx: usize, suffix: &str) -> Result<&Tensor> {
        self.single_block_weights[idx].get(&format!("single_blocks.{}.{}", idx, suffix))
            .ok_or_else(|| crate::EriDiffusionError::Model(format!("missing SW: {}.{}", idx, suffix)))
    }
    fn sw(&self, key: &str) -> Result<&Tensor> {
        self.shared_weights.get(key)
            .ok_or_else(|| crate::EriDiffusionError::Model(format!("missing shared: {}", key)))
    }

    /// Linear with bias: x @ weight^T + bias. Autograd-recording, handles 3D input.
    fn linear(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let in_feat = *dims.last().unwrap();
        let batch: usize = dims[..dims.len()-1].iter().product();
        let out_feat = weight.shape().dims()[0];
        let x_2d = x.reshape(&[batch, in_feat])?;
        let wt = weight.transpose()?;
        let out_2d = x_2d.matmul(&wt)?.add(bias)?;
        let mut out_shape = dims[..dims.len()-1].to_vec();
        out_shape.push(out_feat);
        Ok(out_2d.reshape(&out_shape)?)
    }

    /// MLP embedder: Linear → SiLU → Linear
    fn mlp_embedder(&self, x: &Tensor, w1k: &str, b1k: &str, w2k: &str, b2k: &str) -> Result<Tensor> {
        let h = Self::linear(x, self.sw(w1k)?, self.sw(b1k)?)?;
        let h = h.silu()?;
        Self::linear(&h, self.sw(w2k)?, self.sw(b2k)?)
    }

    /// Sinusoidal timestep embedding. BFL model.py: `timestep_embedding`.
    ///
    /// Caller passes `t ∈ [0, 1)` (sigma directly); this function multiplies
    /// by `time_factor=1000` exactly once before forming the sinusoid arguments.
    /// Audit fix FLUX_VERIFY §H1 / SKEPTIC §H1: previously the trainer also
    /// passed `t ∈ [0, 1000)` so the multiply produced `t * 1_000_000`,
    /// wrapping the sinusoid into the wrong frequency band entirely.
    /// Mirrors the Klein fix at `models/klein.rs::timestep_embedding`.
    fn timestep_embedding(t: &Tensor, dim: usize, device: &Arc<CudaDevice>) -> Result<Tensor> {
        let t_f32 = t.to_dtype(DType::F32)?;
        let t_vec = t_f32.to_vec()?;
        let b = t_vec.len();
        let half = dim / 2;
        let mut data = vec![0f32; b * dim];
        for (bi, &tv) in t_vec.iter().enumerate() {
            let scaled = tv * 1000.0; // BFL `time_factor`
            for j in 0..half {
                let freq = (-(10000.0f64.ln()) * (j as f64) / (half as f64)).exp() as f32;
                let angle = scaled * freq;
                data[bi * dim + j] = angle.cos();
                data[bi * dim + half + j] = angle.sin();
            }
        }
        Tensor::from_slice(&data, Shape::from_dims(&[b, dim]), device.clone())?
            .to_dtype(DType::BF16).map_err(Into::into)
    }

    /// Per-head RMSNorm: reshape to [B*H, HEAD_DIM], norm, reshape back.
    fn rms_norm_per_head(x: &Tensor, scale: &Tensor) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let batch: usize = dims[..dims.len()-1].iter().product();
        let full_dim = *dims.last().unwrap();
        let x_heads = x.reshape(&[batch * NUM_HEADS, HEAD_DIM])?;
        let normed = flame_core::norm::rms_norm(&x_heads, &[HEAD_DIM], Some(scale), NORM_EPS)?;
        normed.reshape(&dims).map_err(Into::into)
    }

    /// 3-axis RoPE: ids [N, 3], works with bfs4d::rope_fused_bf16.
    fn build_rope(ids: &Tensor) -> Result<(Tensor, Tensor)> {
        let ids_f32 = ids.to_dtype(DType::F32)?.to_vec()?;
        let n = ids.shape().dims()[0];
        let half_dim = HEAD_DIM / 2;
        let mut cos_data = vec![0f32; n * half_dim];
        let mut sin_data = vec![0f32; n * half_dim];
        let mut offset = 0;
        for (axis, &axis_dim) in ROPE_AXES.iter().enumerate() {
            let half_ax = axis_dim / 2;
            for (i, row) in ids_f32.chunks(3).enumerate() {
                let pos = row[axis];
                for j in 0..half_ax {
                    let freq = (ROPE_THETA.powf(-(2.0 * j as f64) / axis_dim as f64)) as f32;
                    let angle = pos * freq;
                    cos_data[i * half_dim + offset + j] = angle.cos();
                    sin_data[i * half_dim + offset + j] = angle.sin();
                }
            }
            offset += half_ax;
        }
        let cos = Tensor::from_slice(&cos_data, Shape::from_dims(&[1, 1, n, half_dim]), flame_core::global_cuda_device())?;
        let sin = Tensor::from_slice(&sin_data, Shape::from_dims(&[1, 1, n, half_dim]), flame_core::global_cuda_device())?;
        Ok((cos.to_dtype(DType::BF16)?, sin.to_dtype(DType::BF16)?))
    }

    fn apply_rope(q: &Tensor, k: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<(Tensor, Tensor)> {
        Ok((flame_core::bf16_ops::rope_fused_bf16(q, cos, sin)?, flame_core::bf16_ops::rope_fused_bf16(k, cos, sin)?))
    }

    // ── Double block forward ────────────────────────────────────────

    /// Apply 3 split-QKV LoRAs to a fused QKV output. Each adapter sees the
    /// same input (`x_in`) and produces a `[B, N, DIM]` delta; deltas are
    /// concatenated along the last axis to match the fused base output of
    /// shape `[B, N, 3*DIM]`. Audit fix FLUX_VERIFY §H4 / SKEPTIC §H4 / §H5
    /// (Klein parity — see `models/klein.rs::linear_with_split_qkv_lora`).
    fn add_split_qkv_lora(
        base: &Tensor,
        x_in: &Tensor,
        lora_q: Option<&LoRALinear>,
        lora_k: Option<&LoRALinear>,
        lora_v: Option<&LoRALinear>,
    ) -> Result<Tensor> {
        if lora_q.is_none() && lora_k.is_none() && lora_v.is_none() {
            return Ok(base.clone());
        }
        // Each delta is [B, N, DIM]; missing adapter → zeros of matching shape.
        let zeros_dim = || -> Result<Tensor> {
            let mut shape = base.shape().dims().to_vec();
            *shape.last_mut().unwrap() = DIM;
            Ok(Tensor::zeros_dtype(Shape::from_dims(&shape), base.dtype(), base.device().clone())?)
        };
        let dq = match lora_q { Some(a) => a.forward_delta(x_in)?, None => zeros_dim()? };
        let dk = match lora_k { Some(a) => a.forward_delta(x_in)?, None => zeros_dim()? };
        let dv = match lora_v { Some(a) => a.forward_delta(x_in)?, None => zeros_dim()? };
        let delta = Tensor::cat(&[&dq, &dk, &dv], 2)?.contiguous()?;
        base.add(&delta).map_err(Into::into)
    }

    fn double_block_forward(&self, img: &Tensor, txt: &Tensor, vec: &Tensor, cos: &Tensor, sin: &Tensor, idx: usize) -> Result<(Tensor, Tensor)> {
        let dims = img.shape().dims().to_vec();
        let (b, n_img) = (dims[0], dims[1]);
        let n_txt = txt.shape().dims()[1];

        // Inference dispatch: use fused BF16 kernels (qkv_split_permute_bf16,
        // attn_split_txt_img_bf16, gate_residual_fused_bf16) when no grad tape
        // is being recorded. These kernels auto-dispatch to autograd-recording
        // primitives during training; the explicit is_inference branch is only
        // needed for gate_residual_fused_bf16 which requires a [B, DIM] gate.
        // Training path below is bit-identical to pre-iflame code.
        let is_inference = !AutogradContext::is_recording();

        // Modulation
        let img_mod = Self::linear(&vec.silu()?, self.dw(idx, "img_mod.lin.weight")?, self.dw(idx, "img_mod.lin.bias")?)?;
        let img_mods = img_mod.unsqueeze(1)?.chunk(6, 2)?;
        let (img_s1, img_scale1, img_g1): (_,&Tensor,&Tensor) = (&img_mods[0], &img_mods[1], &img_mods[2]);
        let (img_s2, img_scale2, img_g2) = (&img_mods[3], &img_mods[4], &img_mods[5]);

        let txt_mod = Self::linear(&vec.silu()?, self.dw(idx, "txt_mod.lin.weight")?, self.dw(idx, "txt_mod.lin.bias")?)?;
        let txt_mods = txt_mod.unsqueeze(1)?.chunk(6, 2)?;
        let (txt_s1, txt_scale1, txt_g1) = (&txt_mods[0], &txt_mods[1], &txt_mods[2]);
        let (txt_s2, txt_scale2, txt_g2) = (&txt_mods[3], &txt_mods[4], &txt_mods[5]);

        let bundle = self.bundle.as_ref();

        // --- Img attention --- (split Q/K/V LoRA; H4/H5)
        let img_norm = flame_core::layer_norm::layer_norm(img, &[DIM], None, None, NORM_EPS)?;
        let img_mod_in = img_norm.mul(&img_scale1.add_scalar(1.0)?)?.add(img_s1)?;
        let img_qkv = Self::linear(&img_mod_in, self.dw(idx, "img_attn.qkv.weight")?, self.dw(idx, "img_attn.qkv.bias")?)?;
        let img_qkv = Self::add_split_qkv_lora(
            &img_qkv, &img_mod_in,
            bundle.and_then(|b| b.double_adapters.get(&(idx, DoubleLoraTarget::ImgQ))),
            bundle.and_then(|b| b.double_adapters.get(&(idx, DoubleLoraTarget::ImgK))),
            bundle.and_then(|b| b.double_adapters.get(&(idx, DoubleLoraTarget::ImgV))),
        )?;

        // --- Txt attention --- (split Q/K/V LoRA)
        let txt_norm = flame_core::layer_norm::layer_norm(txt, &[DIM], None, None, NORM_EPS)?;
        let txt_mod_in = txt_norm.mul(&txt_scale1.add_scalar(1.0)?)?.add(txt_s1)?;
        let txt_qkv = Self::linear(&txt_mod_in, self.dw(idx, "txt_attn.qkv.weight")?, self.dw(idx, "txt_attn.qkv.bias")?)?;
        let txt_qkv = Self::add_split_qkv_lora(
            &txt_qkv, &txt_mod_in,
            bundle.and_then(|b| b.double_adapters.get(&(idx, DoubleLoraTarget::TxtQ))),
            bundle.and_then(|b| b.double_adapters.get(&(idx, DoubleLoraTarget::TxtK))),
            bundle.and_then(|b| b.double_adapters.get(&(idx, DoubleLoraTarget::TxtV))),
        )?;

        // QKV split → [B, H, N, D]. `qkv_split_permute_bf16` auto-dispatches:
        // fused CUDA kernel in inference, primitives (narrow+reshape+permute)
        // during training so autograd records each op. Both paths are safe.
        let (img_q, img_k, img_v) = flame_core::bf16_ops::qkv_split_permute_bf16(&img_qkv, NUM_HEADS, HEAD_DIM)?;
        let (txt_q, txt_k, txt_v) = flame_core::bf16_ops::qkv_split_permute_bf16(&txt_qkv, NUM_HEADS, HEAD_DIM)?;

        // QK norm (operates on [B, H, N, D] from qkv_split_permute_bf16)
        let img_q = Self::rms_norm_per_head(&img_q, self.dw(idx, "img_attn.norm.query_norm.scale")?)?;
        let img_k = Self::rms_norm_per_head(&img_k, self.dw(idx, "img_attn.norm.key_norm.scale")?)?;
        let txt_q = Self::rms_norm_per_head(&txt_q, self.dw(idx, "txt_attn.norm.query_norm.scale")?)?;
        let txt_k = Self::rms_norm_per_head(&txt_k, self.dw(idx, "txt_attn.norm.key_norm.scale")?)?;

        // Joint attention. `.contiguous()` after each cat — H9 / GOTCHAS §2.4:
        // Tensor::cat may return non-contiguous views; downstream BF16 SDPA /
        // rope_fused_bf16 read as if contig and silently garble (cos≈0.99,
        // max_abs ~1.8). See `feedback_flame_core_cat_contig.md`.
        let q = Tensor::cat(&[&txt_q, &img_q], 2)?.contiguous()?;
        let k = Tensor::cat(&[&txt_k, &img_k], 2)?.contiguous()?;
        let v = Tensor::cat(&[&txt_v, &img_v], 2)?.contiguous()?;
        let (q, k) = Self::apply_rope(&q, &k, cos, sin)?;

        // SDPA → [B, H, N_total, D]
        let attn = flame_core::attention::sdpa(&q, &k, &v, None)?;

        // Split attn back into txt/img streams. `attn_split_txt_img_bf16`
        // auto-dispatches: fused kernel in inference, narrow+permute+reshape
        // during training. Returns ([B, n_txt, DIM], [B, n_img, DIM]).
        let (txt_attn, img_attn) = flame_core::bf16_ops::attn_split_txt_img_bf16(&attn, n_txt, n_img)?;

        // Output proj + gate + residual
        let img_proj = Self::linear(&img_attn, self.dw(idx, "img_attn.proj.weight")?, self.dw(idx, "img_attn.proj.bias")?)?;
        let img_proj = if let Some(lora) = bundle.and_then(|b| b.double_adapters.get(&(idx, DoubleLoraTarget::ImgProj))) {
            img_proj.add(&lora.forward_delta(&img_attn)?)?
        } else { img_proj };

        // gate_residual_fused_bf16 (inference): squeeze [B,1,DIM] gate → [B,DIM].
        // Training path uses mul+add directly (bit-identical to pre-iflame code).
        let img = if is_inference {
            flame_core::bf16_ops::gate_residual_fused_bf16(img, &img_g1.squeeze(Some(1))?, &img_proj)?
        } else {
            img.add(&img_g1.mul(&img_proj)?)?
        };

        let txt_proj = Self::linear(&txt_attn, self.dw(idx, "txt_attn.proj.weight")?, self.dw(idx, "txt_attn.proj.bias")?)?;
        let txt_proj = if let Some(lora) = bundle.and_then(|b| b.double_adapters.get(&(idx, DoubleLoraTarget::TxtProj))) {
            txt_proj.add(&lora.forward_delta(&txt_attn)?)?
        } else { txt_proj };
        let txt = if is_inference {
            flame_core::bf16_ops::gate_residual_fused_bf16(txt, &txt_g1.squeeze(Some(1))?, &txt_proj)?
        } else {
            txt.add(&txt_g1.mul(&txt_proj)?)?
        };

        // --- GELU MLP --- (img + txt MLP up/down LoRAs added per H5)
        let img_norm2 = flame_core::layer_norm::layer_norm(&img, &[DIM], None, None, NORM_EPS)?;
        let img_mlp_in = img_norm2.mul(&img_scale2.add_scalar(1.0)?)?.add(img_s2)?;
        let img_mlp_h_base = Self::linear(&img_mlp_in, self.dw(idx, "img_mlp.0.weight")?, self.dw(idx, "img_mlp.0.bias")?)?;
        let img_mlp_h = if let Some(lora) = bundle.and_then(|b| b.double_adapters.get(&(idx, DoubleLoraTarget::ImgMlp0))) {
            img_mlp_h_base.add(&lora.forward_delta(&img_mlp_in)?)?
        } else { img_mlp_h_base };
        let img_mlp_h = img_mlp_h.gelu()?;
        let img_mlp_out_base = Self::linear(&img_mlp_h, self.dw(idx, "img_mlp.2.weight")?, self.dw(idx, "img_mlp.2.bias")?)?;
        let img_mlp_out = if let Some(lora) = bundle.and_then(|b| b.double_adapters.get(&(idx, DoubleLoraTarget::ImgMlp2))) {
            img_mlp_out_base.add(&lora.forward_delta(&img_mlp_h)?)?
        } else { img_mlp_out_base };
        let img = if is_inference {
            flame_core::bf16_ops::gate_residual_fused_bf16(&img, &img_g2.squeeze(Some(1))?, &img_mlp_out)?
        } else {
            img.add(&img_g2.mul(&img_mlp_out)?)?
        };

        let txt_norm2 = flame_core::layer_norm::layer_norm(&txt, &[DIM], None, None, NORM_EPS)?;
        let txt_mlp_in = txt_norm2.mul(&txt_scale2.add_scalar(1.0)?)?.add(txt_s2)?;
        let txt_mlp_h_base = Self::linear(&txt_mlp_in, self.dw(idx, "txt_mlp.0.weight")?, self.dw(idx, "txt_mlp.0.bias")?)?;
        let txt_mlp_h = if let Some(lora) = bundle.and_then(|b| b.double_adapters.get(&(idx, DoubleLoraTarget::TxtMlp0))) {
            txt_mlp_h_base.add(&lora.forward_delta(&txt_mlp_in)?)?
        } else { txt_mlp_h_base };
        let txt_mlp_h = txt_mlp_h.gelu()?;
        let txt_mlp_out_base = Self::linear(&txt_mlp_h, self.dw(idx, "txt_mlp.2.weight")?, self.dw(idx, "txt_mlp.2.bias")?)?;
        let txt_mlp_out = if let Some(lora) = bundle.and_then(|b| b.double_adapters.get(&(idx, DoubleLoraTarget::TxtMlp2))) {
            txt_mlp_out_base.add(&lora.forward_delta(&txt_mlp_h)?)?
        } else { txt_mlp_out_base };
        let txt = if is_inference {
            flame_core::bf16_ops::gate_residual_fused_bf16(&txt, &txt_g2.squeeze(Some(1))?, &txt_mlp_out)?
        } else {
            txt.add(&txt_g2.mul(&txt_mlp_out)?)?
        };

        Ok((img, txt))
    }

    // ── Single block forward ────────────────────────────────────────

    fn single_block_forward(&self, x: &Tensor, vec: &Tensor, cos: &Tensor, sin: &Tensor, _txt_len: usize, idx: usize) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let (b, n) = (dims[0], dims[1]);
        let bundle = self.bundle.as_ref();

        // Inference dispatch (see double_block_forward comment).
        let is_inference = !AutogradContext::is_recording();

        // Modulation: Linear(vec.silu()) → 3*DIM
        let m = Self::linear(&vec.silu()?, self.singw(idx, "modulation.lin.weight")?, self.singw(idx, "modulation.lin.bias")?)?;
        let mc = m.unsqueeze(1)?.chunk(3, 2)?;
        let (shift, scale, gate) = (&mc[0], &mc[1], &mc[2]);

        let x_norm = flame_core::layer_norm::layer_norm(x, &[DIM], None, None, NORM_EPS)?;
        let x_mod = x_norm.mul(&scale.add_scalar(1.0)?)?.add(shift)?;

        // linear1: [B, N, 7*DIM] → QKV (3*DIM) + MLP (4*DIM)
        let l1 = Self::linear(&x_mod, self.singw(idx, "linear1.weight")?, self.singw(idx, "linear1.bias")?)?;
        let qkv = l1.narrow(2, 0, 3*DIM)?;
        let mlp_in_base = l1.narrow(2, 3*DIM, 4*DIM)?;

        // Q/K/V split LoRAs (H4/H5: 3 separate adapters on the 3 slices of the
        // fused QKV output — Klein parity).
        let qkv = Self::add_split_qkv_lora(
            &qkv, &x_mod,
            bundle.and_then(|b| b.single_adapters.get(&(idx, SingleLoraTarget::Q))),
            bundle.and_then(|b| b.single_adapters.get(&(idx, SingleLoraTarget::K))),
            bundle.and_then(|b| b.single_adapters.get(&(idx, SingleLoraTarget::V))),
        )?;

        // ProjMlp LoRA on the MLP-up half of the fused linear1 (H5: previously
        // the entire MLP-up branch had no LoRA correction).
        let mlp_in = if let Some(lora) = bundle.and_then(|b| b.single_adapters.get(&(idx, SingleLoraTarget::ProjMlp))) {
            mlp_in_base.add(&lora.forward_delta(&x_mod)?)?
        } else { mlp_in_base };

        // QKV split → [B, H, N, D]. Auto-dispatches to primitives during training.
        let (q, k, v) = flame_core::bf16_ops::qkv_split_permute_bf16(&qkv, NUM_HEADS, HEAD_DIM)?;

        let q = Self::rms_norm_per_head(&q, self.singw(idx, "norm.query_norm.scale")?)?;
        let k = Self::rms_norm_per_head(&k, self.singw(idx, "norm.key_norm.scale")?)?;

        let (q, k) = Self::apply_rope(&q, &k, cos, sin)?;
        let attn = flame_core::attention::sdpa(&q, &k, &v, None)?;
        let attn = attn.permute(&[0, 2, 1, 3])?.reshape(&[b, n, DIM])?;

        // GELU MLP
        let mlp_out = mlp_in.gelu()?;
        // H9: explicit `.contiguous()` after cat — `linear2` is `5*DIM → DIM`
        // and reads the fused buffer as a flat `[B, N, 5*DIM]` 3D matmul input.
        let fused = Tensor::cat(&[&attn, &mlp_out], 2)?.contiguous()?;
        let l2 = Self::linear(&fused, self.singw(idx, "linear2.weight")?, self.singw(idx, "linear2.bias")?)?;
        // H4: `ProjOut` LoRA wraps the **full fused 5*DIM input** mapping to
        // DIM, matching BFL's actual `linear2` shape. Pre-fix this was a
        // phantom `DIM→DIM` adapter receiving only `attn` (1/5 of the input)
        // — silently shape-checked, MLP-up half got no correction.
        let l2 = if let Some(lora) = bundle.and_then(|b| b.single_adapters.get(&(idx, SingleLoraTarget::ProjOut))) {
            l2.add(&lora.forward_delta(&fused)?)?
        } else { l2 };

        // Gated residual. Inference: gate_residual_fused_bf16 with squeezed
        // [B,DIM] gate. Training: mul+add (bit-identical to pre-iflame code).
        if is_inference {
            flame_core::bf16_ops::gate_residual_fused_bf16(x, &gate.squeeze(Some(1))?, &l2)
                .map_err(Into::into)
        } else {
            x.add(&gate.mul(&l2)?).map_err(Into::into)
        }
    }

    // ── Full forward ─────────────────────────────────────────────────

    pub fn forward(
        &mut self,
        img: &Tensor,
        txt: &Tensor,
        timesteps: &Tensor,
        img_ids: &Tensor,
        txt_ids: &Tensor,
        guidance: Option<&Tensor>,
        vector: &Tensor,
    ) -> Result<Tensor> {
        let n_img = img.shape().dims()[1];
        let n_txt = txt.shape().dims()[1];

        // Input projections
        let img = Self::linear(img, self.sw("img_in.weight")?, self.sw("img_in.bias")?)?;
        let txt = Self::linear(txt, self.sw("txt_in.weight")?, self.sw("txt_in.bias")?)?;

        // Time + guidance + vector embeddings
        let t_emb = Self::timestep_embedding(timesteps, TIMESTEP_DIM, &self.device)?;
        let mut vec = self.mlp_embedder(&t_emb, "time_in.in_layer.weight", "time_in.in_layer.bias", "time_in.out_layer.weight", "time_in.out_layer.bias")?;

        if let Some(g) = guidance {
            if self.has_guidance {
                let g_emb = Self::timestep_embedding(g, TIMESTEP_DIM, &self.device)?;
                let gv = self.mlp_embedder(&g_emb, "guidance_in.in_layer.weight", "guidance_in.in_layer.bias", "guidance_in.out_layer.weight", "guidance_in.out_layer.bias")?;
                vec = vec.add(&gv)?;
            }
        }

        let vv = self.mlp_embedder(vector, "vector_in.in_layer.weight", "vector_in.in_layer.bias", "vector_in.out_layer.weight", "vector_in.out_layer.bias")?;
        vec = vec.add(&vv)?;

        // RoPE — `build_rope` reads `all_ids` via `to_vec()` so non-contiguous
        // input is harmless here (CPU-side gather). Kept idiomatic for clarity.
        let all_ids = Tensor::cat(&[txt_ids, img_ids], 0)?;
        let (cos, sin) = Self::build_rope(&all_ids)?;
        let (cos, sin) = (cos.to_dtype(DType::BF16)?, sin.to_dtype(DType::BF16)?);

        // Double blocks
        // Note: Flux double blocks return (img, txt) — gradient checkpointing
        // via flame_core::AutogradContext::checkpoint() returns a single Tensor,
        // so we'd need to fuse along a fresh axis to checkpoint. Left as a
        // future optimization — VRAM headroom is gained via `enable_offload`
        // (per-block weight streaming) instead.
        let (mut img, mut txt) = (img, txt);
        for i in 0..NUM_DOUBLE {
            self.stage_double_block(i)?;
            let (ni, nt) = self.double_block_forward(&img, &txt, &vec, &cos, &sin, i)?;
            self.evict_double_block(i);
            img = ni; txt = nt;
        }

        // Merge + single blocks. H9: `.contiguous()` after cat — the merged
        // joint sequence feeds into single_block_forward whose first op is a
        // BF16 layer_norm + linear matmul.
        let mut merged = Tensor::cat(&[&txt, &img], 1)?.contiguous()?;
        for i in 0..NUM_SINGLE {
            self.stage_single_block(i)?;
            merged = self.single_block_forward(&merged, &vec, &cos, &sin, n_txt, i)?;
            self.evict_single_block(i);
        }

        // Extract img + final layer
        let img_out = merged.narrow(1, n_txt, n_img)?;
        let i_norm = flame_core::layer_norm::layer_norm(&img_out, &[DIM], None, None, NORM_EPS)?;
        let i_linear = Self::linear(&i_norm, self.sw("final_layer.linear.weight")?, self.sw("final_layer.linear.bias")?)?;

        Ok(i_linear)
    }

    // ── Helpers ──────────────────────────────────────────────────────

    fn dw_optional(&self, idx: usize, suffix: &str) -> Result<Option<&Tensor>> {
        match self.double_block_weights[idx].get(&format!("double_blocks.{}.{}", idx, suffix)) {
            Some(t) => Ok(Some(t)),
            None => Ok(None),
        }
    }
    fn sw_optional(&self, key: &str) -> Result<Option<&Tensor>> {
        Ok(self.shared_weights.get(key))
    }
}

// ── Standalone helpers ──────────────────────────────────────────────

fn parse_block_idx(rest: &str, max: usize) -> Option<usize> {
    rest.find('.').and_then(|d| rest[..d].parse::<usize>().ok()).filter(|&i| i < max)
}

// ── TrainableModel trait ────────────────────────────────────────────

impl TrainableModel for FluxModel {
    /// Training forward — accepts pre-cached data from CachedDataset.
    /// `noisy`:    packed latents [B, N_img, 64] (already pack_latents'd at cache time)
    /// `context[0]`: T5-XXL embeddings [B, N_txt, 4096]
    /// `context[1]`: img_ids [N_img, 3]  — pre-computed at cache time
    /// `context[2]`: txt_ids [N_txt, 3]  — pre-computed at cache time
    /// `pooled`:   CLIP-L pooled [B, 768]
    ///
    /// Position IDs MUST be supplied — generating zeros here would silently break
    /// RoPE for image tokens (Flux uses row/col coords on axis 1/2).
    fn forward(&mut self, noisy: &Tensor, timestep: &Tensor, context: &[Tensor], pooled: Option<&Tensor>) -> Result<Tensor> {
        let t5 = context.first()
            .ok_or_else(|| crate::EriDiffusionError::Model("Flux requires T5 embeddings (context[0])".into()))?;
        let img_ids = context.get(1)
            .ok_or_else(|| crate::EriDiffusionError::Model("Flux requires img_ids (context[1])".into()))?;
        let txt_ids = context.get(2)
            .ok_or_else(|| crate::EriDiffusionError::Model("Flux requires txt_ids (context[2])".into()))?;

        let b = noisy.shape().dims()[0];
        let guidance_val = self.guidance_value;
        let guidance = Tensor::from_vec(
            vec![guidance_val; b],
            Shape::from_dims(&[b]),
            self.device.clone(),
        )?;

        let default_pool = Tensor::zeros(
            Shape::from_dims(&[b, VECTOR_DIM]),
            self.device.clone(),
        )?;
        let vector = pooled.unwrap_or(&default_pool);

        let guidance_opt = if self.has_guidance { Some(&guidance) } else { None };
        FluxModel::forward(self, noisy, t5, timestep, img_ids, txt_ids, guidance_opt, vector)
    }

    fn parameters(&self) -> Vec<Parameter> {
        if let Some(ref fft) = self.fft_params { return fft.values().cloned().collect(); }
        if let Some(ref b) = self.bundle { return b.parameters(); }
        Vec::new()
    }

    fn post_optimizer_step(&mut self) {
        if let Some(ref b) = self.bundle {
            for l in b.double_adapters.values() { l.refresh_cache(); }
            for l in b.single_adapters.values() { l.refresh_cache(); }
        }
        if let Some(ref fft) = self.fft_params {
            // Sync FFT weights back to BF16 for forward pass
            for (key, param) in fft {
                if let Ok(t) = param.tensor() {
                    let t = t.to_dtype(DType::BF16).unwrap_or(t);
                    let key = key.clone();
                    if let Some(rest) = key.strip_prefix("double_blocks.") {
                        if let Some(idx) = parse_block_idx(rest, NUM_DOUBLE) {
                            self.double_block_weights[idx].insert(key, t);
                            continue;
                        }
                    }
                    if let Some(rest) = key.strip_prefix("single_blocks.") {
                        if let Some(idx) = parse_block_idx(rest, NUM_SINGLE) {
                            self.single_block_weights[idx].insert(key, t);
                            continue;
                        }
                    }
                    self.shared_weights.insert(key, t);
                }
            }
        }
    }

    /// Save LoRA adapters in the diffusers/PEFT-style key naming PLUS a
    /// per-module `.alpha` scalar tensor. Audit fix FLUX_VERIFY §H5 / §H6 /
    /// SKEPTIC §H6: ecosystem loaders (kohya, ComfyUI, A1111, diffusers
    /// `load_lora_weights`) read `.alpha` to compute `scale = alpha / rank`.
    /// Without it they fall back to `scale = 1.0`, so a checkpoint trained
    /// with `alpha=1, rank=16` (in-trainer scale = 1/16) is silently amplified
    /// 16× on load → blown-out output.
    ///
    /// Per-block diffusers-style key paths used (matches `convert_flux_lora`
    /// expectations after Q/K/V split):
    ///   double_blocks.{i}.img_attn.{to_q,to_k,to_v,proj}
    ///   double_blocks.{i}.txt_attn.{to_q,to_k,to_v,proj}
    ///   double_blocks.{i}.{img,txt}_mlp.{0,2}
    ///   single_blocks.{i}.attn.{to_q,to_k,to_v}
    ///   single_blocks.{i}.{proj_mlp,proj_out}
    ///
    /// `LoRALinear::save_tensors` writes `<prefix>.lora_A.weight` and
    /// `<prefix>.lora_B.weight` (PEFT diffusers convention — same matrices as
    /// kohya `lora_down`/`lora_up`, different suffix). The `.alpha` companion
    /// is always emitted (matches SDXL's recently-landed pattern).
    fn save_weights(&self, path: &str) -> Result<()> {
        let mut tensors = HashMap::new();
        let bundle = match &self.bundle {
            Some(b) => b,
            None => {
                return Err(crate::EriDiffusionError::Model(
                    "Flux save_weights requires LoRA mode".into(),
                ));
            }
        };
        let emit_alpha = |prefix: &str, alpha: f32, out: &mut HashMap<String, Tensor>| -> Result<()> {
            let alpha_t = Tensor::from_vec(
                vec![alpha],
                Shape::from_dims(&[]),
                self.device.clone(),
            )
            .and_then(|t| t.to_dtype(DType::BF16))
            .map_err(|e| crate::EriDiffusionError::Lora(format!(
                "alpha tensor for {prefix}: {e}")))?;
            out.insert(format!("{prefix}.alpha"), alpha_t);
            Ok(())
        };
        for (&(idx, target), lora) in &bundle.double_adapters {
            let prefix = format!("double_blocks.{}.{}", idx, double_target_suffix(target));
            lora.save_tensors(&prefix, &mut tensors)?;
            emit_alpha(&prefix, lora.alpha, &mut tensors)?;
        }
        for (&(idx, target), lora) in &bundle.single_adapters {
            let prefix = format!("single_blocks.{}.{}", idx, single_target_suffix(target));
            lora.save_tensors(&prefix, &mut tensors)?;
            emit_alpha(&prefix, lora.alpha, &mut tensors)?;
        }
        let p = std::path::Path::new(path);
        flame_core::serialization::save_tensors(&tensors, p, flame_core::serialization::SerializationFormat::SafeTensors)?;
        Ok(())
    }

    fn load_weights(&mut self, path: &str) -> Result<()> {
        let bundle = self.bundle.as_ref()
            .ok_or_else(|| crate::EriDiffusionError::Model("Flux load_weights requires LoRA mode".into()))?;
        let source = flame_core::serialization::load_file(
            std::path::Path::new(path), &self.device,
        ).map_err(|e| crate::EriDiffusionError::Safetensors(format!("load_file: {e}")))?;
        for (&(idx, target), lora) in &bundle.double_adapters {
            let prefix = format!("double_blocks.{}.{}", idx, double_target_suffix(target));
            lora.load_tensors(&prefix, &source)?;
        }
        for (&(idx, target), lora) in &bundle.single_adapters {
            let prefix = format!("single_blocks.{}.{}", idx, single_target_suffix(target));
            lora.load_tensors(&prefix, &source)?;
        }
        Ok(())
    }
}

/// Diffusers-style suffix for each per-double-block LoRA target. Q/K/V are
/// split out from BFL's fused `img_attn.qkv` / `txt_attn.qkv` to match the
/// adapter granularity that `convert_flux_lora.py` expects.
fn double_target_suffix(t: DoubleLoraTarget) -> &'static str {
    match t {
        DoubleLoraTarget::ImgQ => "img_attn.to_q",
        DoubleLoraTarget::ImgK => "img_attn.to_k",
        DoubleLoraTarget::ImgV => "img_attn.to_v",
        DoubleLoraTarget::ImgProj => "img_attn.proj",
        DoubleLoraTarget::TxtQ => "txt_attn.to_q",
        DoubleLoraTarget::TxtK => "txt_attn.to_k",
        DoubleLoraTarget::TxtV => "txt_attn.to_v",
        DoubleLoraTarget::TxtProj => "txt_attn.proj",
        DoubleLoraTarget::ImgMlp0 => "img_mlp.0",
        DoubleLoraTarget::ImgMlp2 => "img_mlp.2",
        DoubleLoraTarget::TxtMlp0 => "txt_mlp.0",
        DoubleLoraTarget::TxtMlp2 => "txt_mlp.2",
    }
}

/// Diffusers-style suffix for each per-single-block LoRA target. Q/K/V split
/// from BFL `linear1[:3*DIM]`; `proj_mlp` is BFL `linear1[3*DIM:]`; `proj_out`
/// is BFL `linear2`.
fn single_target_suffix(t: SingleLoraTarget) -> &'static str {
    match t {
        SingleLoraTarget::Q => "attn.to_q",
        SingleLoraTarget::K => "attn.to_k",
        SingleLoraTarget::V => "attn.to_v",
        SingleLoraTarget::ProjMlp => "proj_mlp",
        SingleLoraTarget::ProjOut => "proj_out",
    }
}
