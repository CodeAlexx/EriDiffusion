//! SDXL UNet — training model with LoRA adapters on attention projections.
//!
//! Reference: sd-scripts `sdxl_original_unet.py`, inference-flame
//! `sdxl_unet.rs`, flame-diffusion `sdxl-trainer/src/model.rs`. Forward
//! ported from the flame-diffusion trainer (proven autograd-clean path);
//! LoRA injection rebuilt on top of ED-v2 `crate::lora::LoRALinear`.
//!
//! ## Architecture
//!
//! ```text
//! timestep   → sinusoidal(320)  → time_embed (Linear×2 + SiLU) → [B, 1280]
//! y [B,2816] → label_emb (Linear×2 + SiLU) → [B, 1280]
//! emb = time_embed + label_emb
//!
//! x [B,4,H,W] → conv_in(320) → 9 input blocks → middle block → 9 output blocks → out
//!
//! Input blocks:  [0]=conv_in,
//!                [1-2]=ResBlock(320),               td=0
//!                [3]=Downsample(320),
//!                [4-5]=ResBlock(640) + ST(td=2),
//!                [6]=Downsample(640),
//!                [7-8]=ResBlock(1280) + ST(td=10)
//! Middle block:  ResBlock(1280) + ST(td=10) + ResBlock(1280)
//! Output blocks: [0-2]=ResBlock(1280) + ST(td=10),
//!                [3-5]=ResBlock(640)  + ST(td=2),
//!                [6-8]=ResBlock(320),               td=0
//! Upsample lives at the END of output blocks 2 and 5.
//! ```
//!
//! Note: this layout (downsample as a separate "block 3"/"block 6" rather
//! than tagged onto the previous ResBlock) follows sd-scripts
//! `sdxl_original_unet.py` and the flame-diffusion sdxl-trainer. The
//! inference-flame `sdxl_unet.rs` uses an equivalent flat descriptor list;
//! both produce identical key paths in the LDM checkpoint.
//!
//! ## LoRA targets
//!
//! sd-scripts default ("attn-mlp" preset → in ED-v2 minimal port we cover
//! attention only): every `attn1` (self-attn) and `attn2` (cross-attn) inside
//! every transformer block, projections `to_q / to_k / to_v / to_out.0`.
//! Conv layers and ResBlocks stay frozen (kohya conv-LoRAs are NOT supported
//! here yet). Match keys: `{block_prefix}.transformer_blocks.{j}.attn{1,2}.to_*`.

use std::collections::HashMap;
use std::sync::Arc;

use cudarc::driver::CudaDevice;
use flame_core::cuda_ops::GpuOps;
use flame_core::{parameter::Parameter, DType, Shape, Tensor};

use crate::config::TrainConfig;
use crate::lora::LoRALinear;
use crate::models::TrainableModel;
use crate::Result;

// ---------------------------------------------------------------------------
// SDXL constants (from #sdxl 1.0 LoRA preset + sd-scripts sdxl_original_unet)
// ---------------------------------------------------------------------------
pub const IN_CHANNELS: usize = 4;
pub const OUT_CHANNELS: usize = 4;
pub const MODEL_CHANNELS: usize = 320;
pub const TIME_EMBED_DIM: usize = 1280;
pub const CONTEXT_DIM: usize = 2048;     // CLIP-L 768 + CLIP-G 1280
pub const ADM_IN_CHANNELS: usize = 2816; // CLIP-G pool 1280 + size_ids 1536
pub const HEAD_DIM: usize = 64;
pub const GN_GROUPS: usize = 32;
pub const GN_EPS: f32 = 1e-5;
pub const NORM_EPS: f32 = 1e-5;

/// Transformer depth per input block (0-8). Blocks 3, 6 are pure downsamples.
const TD_INPUT: [usize; 9] = [0, 0, 0, 0, 2, 2, 0, 10, 10];
/// Transformer depth for middle block.
const TD_MIDDLE: usize = 10;
/// Transformer depth per output block (0-8).
const TD_OUTPUT: [usize; 9] = [10, 10, 10, 2, 2, 2, 0, 0, 0];

// ---------------------------------------------------------------------------
// LoRA target enumeration
// ---------------------------------------------------------------------------

/// Build the full ordered list of LoRA target prefixes for the SDXL UNet,
/// each accompanied by its `(in_features, out_features)`. Order is the same
/// the trainer hits at runtime (input → middle → output), so the LoRA
/// adapter index in `SDXLModel.lora_adapters` matches the
/// `target_prefixes` index.
///
/// SDXL audit H3: expanded to upstream Python's `attn-mlp` preset coverage
/// (`LAYER_PRESETS["attn-mlp"]=["attentions"]` → substring `attentions`
/// matches every `Linear` under `*.attentions.*`). Per SpatialTransformer:
///   - `proj_in` (ch → ch), `proj_out` (ch → ch)               (2 projections)
///   - per BasicTransformerBlock × td:
///     - attn1 q/k/v/o (4)
///     - attn2 q/k/v/o (4 — k,v from CONTEXT_DIM=2048)
///     - ff.net.0.proj (GeGLU: ch → 8·ch), ff.net.2 (4·ch → ch)  (2)
/// FF is the largest single weight per block and is where most style/
/// identity transfer for SDXL LoRAs lives in the kohya/sd-scripts ecosystem.
/// Without FF coverage, the previous 640-adapter total left LoRAs visibly
/// weaker than equivalent upstream Python LoRAs.
const FF_MULT: usize = 4; // diffusers default for SDXL FeedForward (GeGLU 8x intermediate)
fn enumerate_lora_targets() -> Vec<(String, usize, usize)> {
    let mut out: Vec<(String, usize, usize)> = Vec::new();

    // Each SpatialTransformer (proj_in, td×BasicTransformerBlock, proj_out).
    let mut push_block = |block_prefix: &str, td: usize, ch: usize| {
        // proj_in / proj_out (Linear because SDXL uses use_linear_in_transformer=True).
        out.push((format!("{block_prefix}.proj_in"),  ch, ch));

        for j in 0..td {
            for attn_idx in [1usize, 2] {
                // attn1 is self-attn → all from x (ch → ch).
                // attn2 is cross-attn → q from x (ch→ch), k/v from context (CONTEXT_DIM→ch).
                let (k_in, v_in) = if attn_idx == 1 { (ch, ch) } else { (CONTEXT_DIM, CONTEXT_DIM) };
                let pre = format!("{block_prefix}.transformer_blocks.{j}.attn{attn_idx}");
                out.push((format!("{pre}.to_q"),     ch,   ch));
                out.push((format!("{pre}.to_k"),     k_in, ch));
                out.push((format!("{pre}.to_v"),     v_in, ch));
                out.push((format!("{pre}.to_out.0"), ch,   ch));
            }
            // FeedForward (GeGLU). net.0.proj projects ch → 2 * (FF_MULT*ch);
            // chunked, gated, then net.2 projects (FF_MULT*ch) → ch.
            let inner = FF_MULT * ch;
            let ff_pre = format!("{block_prefix}.transformer_blocks.{j}.ff");
            out.push((format!("{ff_pre}.net.0.proj"), ch,    inner * 2));
            out.push((format!("{ff_pre}.net.2"),       inner, ch));
        }

        out.push((format!("{block_prefix}.proj_out"), ch, ch));
    };

    // Input blocks 1-8 (block 0 is conv_in). Channels follow [320,320,320,640,640,640,1280,1280].
    let in_ch = [320usize, 320, 320, 640, 640, 640, 1280, 1280];
    for (i, &td) in TD_INPUT.iter().enumerate().skip(1) {
        if td > 0 { push_block(&format!("input_blocks.{i}.1"), td, in_ch[i - 1]); }
    }
    // Middle block — ST at sub-index 1.
    push_block("middle_block.1", TD_MIDDLE, 1280);
    // Output blocks 0-8.
    for (i, &td) in TD_OUTPUT.iter().enumerate() {
        if td > 0 {
            let ch = match i { 0..=2 => 1280, 3..=5 => 640, _ => 320 };
            push_block(&format!("output_blocks.{i}.1"), td, ch);
        }
    }

    out
}

// ---------------------------------------------------------------------------
// SDXLModel
// ---------------------------------------------------------------------------

pub struct SDXLModel {
    pub config: TrainConfig,
    pub device: Arc<CudaDevice>,
    pub weights: HashMap<String, Tensor>,
    pub lora_adapters: Vec<LoRALinear>,
    /// Parallel array to `lora_adapters`: the base weight key prefix the
    /// adapter targets (e.g. "input_blocks.4.1.transformer_blocks.0.attn1.to_q").
    /// Used for save/load to write `<prefix>.lora_A.weight` / `.lora_B.weight`.
    pub lora_target_prefixes: Vec<String>,
    /// Reverse lookup `prefix → adapter_idx`, populated at construction.
    lora_index_by_prefix: HashMap<String, usize>,
    pub parameters: Vec<Parameter>,
    pub is_lora: bool,
}

impl SDXLModel {
    pub fn load(paths: &[std::path::PathBuf], config: &TrainConfig, device: Arc<CudaDevice>) -> Result<Self> {
        // Load (one or many) safetensors files — SDXL ships as a single file
        // typically, but support shards for parity with other trainers.
        let mut weights: HashMap<String, Tensor> = HashMap::new();
        for p in paths {
            let part = flame_core::serialization::load_file(p, &device)
                .map_err(|e| crate::EriDiffusionError::Safetensors(format!("load {}: {e}", p.display())))?;
            for (k, v) in part {
                let k = k.strip_prefix("model.diffusion_model.").unwrap_or(&k).to_string();
                let v = if v.dtype() == DType::BF16 { v } else { v.to_dtype(DType::BF16)? };
                weights.insert(k, v);
            }
        }
        log::info!("SDXL: {} UNet tensors loaded", weights.len());

        let is_lora = config.is_lora();
        let mut lora_adapters: Vec<LoRALinear> = Vec::new();
        let mut lora_target_prefixes: Vec<String> = Vec::new();
        let mut lora_index_by_prefix: HashMap<String, usize> = HashMap::new();
        let mut parameters: Vec<Parameter> = Vec::new();

        if is_lora {
            let rank = config.lora_rank as usize;
            let alpha = config.lora_alpha as f32;
            let targets = enumerate_lora_targets();
            log::info!("SDXL LoRA: rank={} alpha={} targets={}", rank, alpha, targets.len());
            let seed_base = 42u64;
            for (idx, (prefix, in_f, out_f)) in targets.into_iter().enumerate() {
                let lora = LoRALinear::new(in_f, out_f, rank, alpha, device.clone(), seed_base + idx as u64)
                    .map_err(|e| crate::EriDiffusionError::Lora(format!("LoRA new {prefix}: {e}")))?;
                lora_index_by_prefix.insert(prefix.clone(), idx);
                lora_adapters.push(lora);
                lora_target_prefixes.push(prefix);
            }
            for l in &lora_adapters { parameters.extend(l.parameters()); }
        } else {
            // Full fine-tune mode (uncommon but supported by preset
            // `#sdxl 1.0.json`). Promote every tensor to F32 trainable.
            for (_, t) in &weights {
                parameters.push(Parameter::new(t.to_dtype(DType::F32)?.requires_grad_(true)));
            }
        }

        Ok(Self {
            config: config.clone(),
            device,
            weights,
            lora_adapters,
            lora_target_prefixes,
            lora_index_by_prefix,
            parameters,
            is_lora,
        })
    }

    fn w(&self, key: &str) -> Result<&Tensor> {
        self.weights.get(key)
            .ok_or_else(|| crate::EriDiffusionError::Model(format!("missing weight: {key}")))
    }

    // -----------------------------------------------------------------------
    // Linear helpers
    // -----------------------------------------------------------------------

    /// Autograd-aware linear: `out = x @ W^T + bias`. Use for any path whose
    /// output must propagate `requires_grad` (which is everything in SDXL —
    /// the time/label MLPs feed into the residual chain that the LoRA
    /// gradients have to flow back through).
    fn linear(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let in_feat = *dims.last().unwrap();
        let batch: usize = dims[..dims.len()-1].iter().product();
        let out_feat = weight.shape().dims()[0];
        let x_2d = x.reshape(&[batch, in_feat])?;
        let wt = weight.transpose()?;
        let mut out = x_2d.matmul(&wt)?;
        if let Some(b) = bias {
            out = out.add(b)?;
        }
        let mut shape = dims[..dims.len()-1].to_vec();
        shape.push(out_feat);
        out.reshape(&shape).map_err(Into::into)
    }

    /// LoRA-aware attention projection. `prefix` is the base weight prefix
    /// without ".weight" (e.g. "input_blocks.4.1.transformer_blocks.0.attn1.to_q").
    fn attn_proj(&self, x: &Tensor, prefix: &str) -> Result<Tensor> {
        let weight = self.w(&format!("{prefix}.weight"))?;
        let bias = self.weights.get(&format!("{prefix}.bias"));
        let base = Self::linear(x, weight, bias)?;

        if self.is_lora {
            if let Some(&idx) = self.lora_index_by_prefix.get(prefix) {
                let adapter = &self.lora_adapters[idx];
                let x_3d = ensure_3d(x)?;
                let delta = adapter.forward_delta(&x_3d)
                    .map_err(|e| crate::EriDiffusionError::Lora(format!("LoRA delta {prefix}: {e}")))?;
                // The base may be 2D or 3D depending on the call site; the
                // delta comes back 3D. Reshape it to match `base` before add.
                let base_dims = base.shape().dims().to_vec();
                let delta_reshaped = delta.reshape(&base_dims)?;
                return base.add(&delta_reshaped).map_err(Into::into);
            }
        }
        Ok(base)
    }

    /// Conv2d (frozen, no LoRA).
    fn conv2d(&self, x: &Tensor, weight_key: &str, bias_key: &str, stride: usize, padding: usize) -> Result<Tensor> {
        let weight = self.w(weight_key)?;
        let bias = self.weights.get(bias_key);
        flame_core::cuda_conv2d::conv2d(x, weight, bias, stride, padding).map_err(Into::into)
    }

    /// GroupNorm on NCHW: convert to NHWC for the kernel, convert back.
    fn group_norm(&self, x: &Tensor, weight_key: &str, bias_key: &str) -> Result<Tensor> {
        let weight = self.w(weight_key)?;
        let bias = self.w(bias_key)?;
        let nhwc = GpuOps::permute_nchw_to_nhwc(x)?;
        let out_nhwc = flame_core::group_norm::group_norm(&nhwc, GN_GROUPS, Some(weight), Some(bias), GN_EPS)?;
        GpuOps::permute_nhwc_to_nchw(&out_nhwc).map_err(Into::into)
    }

    // -----------------------------------------------------------------------
    // Timestep embedding
    // -----------------------------------------------------------------------

    fn timestep_embedding(t: &Tensor, dim: usize) -> Result<Tensor> {
        let t_f32 = t.to_dtype(DType::F32)?;
        let t_vec = t_f32.to_vec()?;
        let b = t_vec.len();
        let half = dim / 2;
        let mut data = vec![0.0f32; b * dim];
        for (bi, &tv) in t_vec.iter().enumerate() {
            for j in 0..half {
                let freq = (-(10000.0f64.ln()) * (j as f64) / (half as f64)).exp() as f32;
                let angle = tv * freq;
                data[bi * dim + j] = angle.cos();
                data[bi * dim + half + j] = angle.sin();
            }
        }
        let device = t.device().clone();
        Tensor::from_vec(data, Shape::from_dims(&[b, dim]), device)?
            .to_dtype(DType::BF16).map_err(Into::into)
    }

    // -----------------------------------------------------------------------
    // ResBlock
    // -----------------------------------------------------------------------

    fn resblock(&self, x: &Tensor, emb: &Tensor, prefix: &str) -> Result<Tensor> {
        let h = self.group_norm(x, &format!("{prefix}.in_layers.0.weight"), &format!("{prefix}.in_layers.0.bias"))?;
        let h = h.silu()?;
        let h = self.conv2d(&h, &format!("{prefix}.in_layers.2.weight"), &format!("{prefix}.in_layers.2.bias"), 1, 1)?;

        let emb_h = emb.silu()?;
        let emb_out = Self::linear(&emb_h,
            self.w(&format!("{prefix}.emb_layers.1.weight"))?,
            Some(self.w(&format!("{prefix}.emb_layers.1.bias"))?))?;
        let c = h.shape().dims()[1];
        let emb_bc = emb_out.narrow(1, 0, c)?.reshape(&[emb_out.shape().dims()[0], c, 1, 1])?;
        let h = h.add(&emb_bc)?;

        let h = self.group_norm(&h, &format!("{prefix}.out_layers.0.weight"), &format!("{prefix}.out_layers.0.bias"))?;
        let h = h.silu()?;
        let h = self.conv2d(&h, &format!("{prefix}.out_layers.3.weight"), &format!("{prefix}.out_layers.3.bias"), 1, 1)?;

        let residual = if self.weights.contains_key(&format!("{prefix}.skip_connection.weight")) {
            self.conv2d(x, &format!("{prefix}.skip_connection.weight"), &format!("{prefix}.skip_connection.bias"), 1, 0)?
        } else {
            x.clone()
        };

        // F32 residual accumulation — same pattern as the inference-flame
        // SDXL UNet; keeps BF16 truncation error from compounding through
        // 30+ skip connections.
        residual.to_dtype(DType::F32)?.add(&h.to_dtype(DType::F32)?)?
            .to_dtype(DType::BF16).map_err(Into::into)
    }

    // -----------------------------------------------------------------------
    // SpatialTransformer + BasicTransformerBlock + attention + GEGLU
    // -----------------------------------------------------------------------

    fn spatial_transformer(&self, x: &Tensor, context: &Tensor, prefix: &str, td: usize) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let (b, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);

        let x_norm = self.group_norm(x, &format!("{prefix}.norm.weight"), &format!("{prefix}.norm.bias"))?;
        // SDXL uses use_linear_in_transformer=true (proj_in/proj_out are
        // Linear, not Conv2d 1x1). NCHW → [B, H*W, C].
        let x_flat = x_norm.permute(&[0, 2, 3, 1])?.reshape(&[b, h * w, c])?;
        // SDXL audit H3: proj_in / proj_out routed through attn_proj so the
        // per-module LoRA delta is added when training.
        let mut h_state = self.attn_proj(&x_flat, &format!("{prefix}.proj_in"))?;

        for j in 0..td {
            h_state = self.basic_transformer_block(&h_state, context,
                &format!("{prefix}.transformer_blocks.{j}"))?;
        }

        let out = self.attn_proj(&h_state, &format!("{prefix}.proj_out"))?;

        let out = out.reshape(&[b, h, w, c])?.permute(&[0, 3, 1, 2])?.contiguous()?;
        x.add(&out).map_err(Into::into)
    }

    fn basic_transformer_block(&self, x: &Tensor, context: &Tensor, prefix: &str) -> Result<Tensor> {
        let c = *x.shape().dims().last().unwrap();

        let x_norm1 = flame_core::layer_norm::layer_norm(x, &[c],
            Some(self.w(&format!("{prefix}.norm1.weight"))?),
            Some(self.w(&format!("{prefix}.norm1.bias"))?), NORM_EPS)?;
        let attn1_out = self.attention(&x_norm1, &x_norm1, &format!("{prefix}.attn1"))?;
        let x = x.add(&attn1_out)?;

        let x_norm2 = flame_core::layer_norm::layer_norm(&x, &[c],
            Some(self.w(&format!("{prefix}.norm2.weight"))?),
            Some(self.w(&format!("{prefix}.norm2.bias"))?), NORM_EPS)?;
        let attn2_out = self.attention(&x_norm2, context, &format!("{prefix}.attn2"))?;
        let x = x.add(&attn2_out)?;

        let x_norm3 = flame_core::layer_norm::layer_norm(&x, &[c],
            Some(self.w(&format!("{prefix}.norm3.weight"))?),
            Some(self.w(&format!("{prefix}.norm3.bias"))?), NORM_EPS)?;
        let ff_out = self.feed_forward(&x_norm3, &format!("{prefix}.ff"))?;
        x.add(&ff_out).map_err(Into::into)
    }

    fn attention(&self, x: &Tensor, context: &Tensor, prefix: &str) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let (b, seq_q) = (dims[0], dims[1]);
        let inner_dim = *dims.last().unwrap();
        let num_heads = inner_dim / HEAD_DIM;
        let seq_kv = context.shape().dims()[1];

        let q = self.attn_proj(x,       &format!("{prefix}.to_q"))?;
        let k = self.attn_proj(context, &format!("{prefix}.to_k"))?;
        let v = self.attn_proj(context, &format!("{prefix}.to_v"))?;

        let q = q.reshape(&[b, seq_q,  num_heads, HEAD_DIM])?.permute(&[0, 2, 1, 3])?;
        let k = k.reshape(&[b, seq_kv, num_heads, HEAD_DIM])?.permute(&[0, 2, 1, 3])?;
        let v = v.reshape(&[b, seq_kv, num_heads, HEAD_DIM])?.permute(&[0, 2, 1, 3])?;

        let out = flame_core::attention::sdpa(&q, &k, &v, None)?;
        let out = out.permute(&[0, 2, 1, 3])?.reshape(&[b, seq_q, inner_dim])?;
        self.attn_proj(&out, &format!("{prefix}.to_out.0"))
    }

    fn feed_forward(&self, x: &Tensor, prefix: &str) -> Result<Tensor> {
        // GEGLU: net.0.proj is a 2x-wide projection, split, gelu(gate) * value.
        // SDXL audit H3: route net.0.proj and net.2 through attn_proj so the
        // FF LoRA targets receive their delta. FF accounts for the bulk of
        // a transformer block's weight volume; this is where most SDXL LoRA
        // identity transfer lives.
        let geglu = self.attn_proj(x, &format!("{prefix}.net.0.proj"))?;
        let chunks = geglu.chunk(2, 2)?;
        let h = chunks[0].gelu()?.mul(&chunks[1])?;
        self.attn_proj(&h, &format!("{prefix}.net.2"))
    }

    // -----------------------------------------------------------------------
    // Down / Up
    // -----------------------------------------------------------------------

    fn downsample(&self, x: &Tensor, prefix: &str) -> Result<Tensor> {
        self.conv2d(x, &format!("{prefix}.op.weight"), &format!("{prefix}.op.bias"), 2, 1)
    }

    fn upsample(&self, x: &Tensor, prefix: &str) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let (_b, _c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
        let up = x.upsample_nearest2d(h * 2, w * 2)?;
        self.conv2d(&up, &format!("{prefix}.conv.weight"), &format!("{prefix}.conv.bias"), 1, 1)
    }

    // -----------------------------------------------------------------------
    // Full UNet forward
    // -----------------------------------------------------------------------

    /// SDXL UNet forward pass.
    /// - `x`        : `[B, 4, H, W]` BF16 noisy latents
    /// - `timesteps`: `[B]` integer timesteps in `[0, 1000)` (BF16 or F32)
    /// - `context`  : `[B, 77, 2048]` BF16 dual-encoder text context
    /// - `y`        : `[B, 2816]` BF16 ADM input (CLIP-G pool 1280 + size_ids 1536)
    pub fn forward(
        &mut self,
        x: &Tensor,
        timesteps: &Tensor,
        context: &Tensor,
        y: &Tensor,
    ) -> Result<Tensor> {
        // Time + label embed
        let t_emb = Self::timestep_embedding(timesteps, MODEL_CHANNELS)?;
        let emb = Self::linear(&t_emb,
            self.w("time_embed.0.weight")?, Some(self.w("time_embed.0.bias")?))?;
        let emb = emb.silu()?;
        let emb = Self::linear(&emb,
            self.w("time_embed.2.weight")?, Some(self.w("time_embed.2.bias")?))?;

        let label = Self::linear(y,
            self.w("label_emb.0.0.weight")?, Some(self.w("label_emb.0.0.bias")?))?;
        let label = label.silu()?;
        let label = Self::linear(&label,
            self.w("label_emb.0.2.weight")?, Some(self.w("label_emb.0.2.bias")?))?;

        let emb = emb.add(&label)?;

        // --- Input blocks ---
        let mut hs: Vec<Tensor> = Vec::with_capacity(9);

        // Block 0: conv_in
        let mut h = self.conv2d(x, "input_blocks.0.0.weight", "input_blocks.0.0.bias", 1, 1)?;
        hs.push(h.clone());

        // Blocks 1-8
        for n in 1..=8 {
            let prefix = format!("input_blocks.{n}");
            let td = TD_INPUT[n];

            if n == 3 || n == 6 {
                // Pure stride-2 downsample (no ResBlock here in this layout).
                h = self.downsample(&h, &format!("{prefix}.0"))?;
            } else {
                h = self.resblock(&h, &emb, &format!("{prefix}.0"))?;
                if td > 0 {
                    h = self.spatial_transformer(&h, context, &format!("{prefix}.1"), td)?;
                }
            }
            hs.push(h.clone());
        }

        // --- Middle ---
        h = self.resblock(&h, &emb, "middle_block.0")?;
        h = self.spatial_transformer(&h, context, "middle_block.1", TD_MIDDLE)?;
        h = self.resblock(&h, &emb, "middle_block.2")?;

        // --- Output ---
        for n in 0..9 {
            let prefix = format!("output_blocks.{n}");
            let td = TD_OUTPUT[n];

            let skip = hs.pop().ok_or_else(||
                crate::EriDiffusionError::Model("ran out of skip connections".into()))?;
            h = Tensor::cat(&[&h, &skip], 1)?;

            h = self.resblock(&h, &emb, &format!("{prefix}.0"))?;
            if td > 0 {
                h = self.spatial_transformer(&h, context, &format!("{prefix}.1"), td)?;
            }
            // Upsample lives at the end of blocks 2 and 5; sub-index depends on
            // whether a SpatialTransformer was inserted at .1.
            if n == 2 || n == 5 {
                let up_idx = if td > 0 { 2 } else { 1 };
                h = self.upsample(&h, &format!("{prefix}.{up_idx}"))?;
            }
        }

        // --- Final ---
        let h = self.group_norm(&h, "out.0.weight", "out.0.bias")?;
        let h = h.silu()?;
        self.conv2d(&h, "out.2.weight", "out.2.bias", 1, 1)
    }
}

fn ensure_3d(t: &Tensor) -> Result<Tensor> {
    let dims = t.shape().dims();
    match dims.len() {
        2 => t.unsqueeze(0).map_err(Into::into),
        3 => Ok(t.clone()),
        _ => Err(crate::EriDiffusionError::Model(format!("expected 2D or 3D, got {}D", dims.len()))),
    }
}

// ---------------------------------------------------------------------------
// TrainableModel impl
// ---------------------------------------------------------------------------

impl TrainableModel for SDXLModel {
    fn forward(
        &mut self,
        noisy: &Tensor,
        timestep: &Tensor,
        context: &[Tensor],
        pooled: Option<&Tensor>,
    ) -> Result<Tensor> {
        let ctx = context.first().ok_or_else(||
            crate::EriDiffusionError::Model("SDXL needs concat(CLIP-L, CLIP-G) context".into()))?;
        let y = pooled.ok_or_else(||
            crate::EriDiffusionError::Model("SDXL needs pooled `y` (concat(CLIP-G pool, size_ids))".into()))?;
        SDXLModel::forward(self, noisy, timestep, ctx, y)
    }

    fn parameters(&self) -> Vec<Parameter> {
        self.parameters.clone()
    }

    fn post_optimizer_step(&mut self) {
        // LoRALinear caches nothing today, but mirror the contract:
        for l in &self.lora_adapters { l.refresh_cache(); }
    }

    /// Save LoRA adapters to safetensors using the diffusers/PEFT convention
    /// (`<prefix>.lora_A.weight` / `.lora_B.weight`) PLUS a per-module
    /// `<prefix>.alpha` scalar tensor (SDXL audit H6).
    ///
    /// All four LoRA loaders in inference-flame and the wider ecosystem
    /// (ComfyUI, A1111, kohya, sd-scripts converters) read `.alpha` to
    /// compute `scale = alpha / rank`. Without it they fall back to
    /// `scale = 1.0`, which at the default `alpha=1.0 / rank=16` is 16×
    /// too strong — visible as color shift / over-amplified style. The
    /// `.alpha` tensor is a 0-dim scalar containing the alpha value.
    ///
    /// NOTE: SDXL community tooling (kohya, sd-scripts, ComfyUI loaders)
    /// typically also expects the kohya naming
    /// `lora_unet_<dotted-path>.lora_down.weight / .lora_up.weight / .alpha`.
    /// The current ED-v2 convention writes the diffusers-style keys
    /// (`lora_A` / `lora_B`) with the `.alpha` companion; a kohya converter
    /// can rewrite these prefixes if needed.
    fn save_weights(&self, path: &str) -> Result<()> {
        if !self.is_lora {
            return Err(crate::EriDiffusionError::Model(
                "save_weights for full-FT SDXL not implemented yet".into(),
            ));
        }
        let mut out = HashMap::new();
        for (i, adapter) in self.lora_adapters.iter().enumerate() {
            let prefix = &self.lora_target_prefixes[i];
            adapter.save_tensors(prefix, &mut out)
                .map_err(|e| crate::EriDiffusionError::Lora(format!("save {prefix}: {e}")))?;
            // SDXL audit H6: companion `.alpha` scalar (BF16 to match
            // the rest of the LoRA tensors). 0-dim; loaders that expect
            // 1-dim length-1 either work or do an .item() — both are
            // standard.
            let alpha_t = Tensor::from_vec(
                vec![adapter.alpha],
                flame_core::Shape::from_dims(&[]),
                self.device.clone(),
            )
            .and_then(|t| t.to_dtype(DType::BF16))
            .map_err(|e| crate::EriDiffusionError::Lora(format!(
                "alpha tensor for {prefix}: {e}")))?;
            out.insert(format!("{prefix}.alpha"), alpha_t);
        }
        flame_core::serialization::save_file(&out, std::path::Path::new(path))
            .map_err(|e| crate::EriDiffusionError::Safetensors(format!("save_file: {e}")))?;
        Ok(())
    }

    fn load_weights(&mut self, path: &str) -> Result<()> {
        if !self.is_lora {
            return Err(crate::EriDiffusionError::Model(
                "load_weights for full-FT SDXL not implemented yet".into(),
            ));
        }
        let source = flame_core::serialization::load_file(
            std::path::Path::new(path), &self.device,
        ).map_err(|e| crate::EriDiffusionError::Safetensors(format!("load_file: {e}")))?;
        for (i, adapter) in self.lora_adapters.iter().enumerate() {
            let prefix = &self.lora_target_prefixes[i];
            adapter.load_tensors(prefix, &source)
                .map_err(|e| crate::EriDiffusionError::Lora(format!("load {prefix}: {e}")))?;
        }
        Ok(())
    }
}

impl SDXLModel {
    /// Canonical (name, Parameter) pairs for full-checkpoint save/resume.
    /// Mirrors `<SDXLModel as TrainableModel>::save_weights` exactly: the i-th
    /// adapter is paired with `lora_target_prefixes[i]`, emitted as
    /// `<prefix>.lora_A.weight` / `<prefix>.lora_B.weight`. The `.alpha` scalars
    /// that `save_weights` also writes are NOT Parameters and are intentionally
    /// skipped (alpha is restored from CkptHeader on load).
    pub fn named_parameters(&self) -> Vec<(String, Parameter)> {
        let mut out = Vec::with_capacity(self.lora_adapters.len() * 2);
        for (i, adapter) in self.lora_adapters.iter().enumerate() {
            let prefix = &self.lora_target_prefixes[i];
            out.push((format!("{prefix}.lora_A.weight"), adapter.lora_a().clone()));
            out.push((format!("{prefix}.lora_B.weight"), adapter.lora_b().clone()));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lora_target_count_matches_attn_mlp_preset() {
        // SDXL audit H3: upstream Python `attn-mlp` preset substring-matches
        // `"attentions"`, which catches every `Linear` under `*.attentions.*`:
        //   per BasicTransformerBlock: 4 (attn1) + 4 (attn2) + 2 (FF GeGLU) = 10
        //   per SpatialTransformer:    + 2 (proj_in, proj_out)
        //   td slots: input td=[2,2,10,10] (sum 24), middle td=10,
        //             output td=[10,10,10,2,2,2] (sum 36)
        //             ⇒ total td slots = 70  (NB: prior test asserted 80 due
        //             to an arithmetic error in the comment — corrected here)
        //   ST counts: input STs=4, middle=1, output STs=6 ⇒ 11 STs total
        //   Total adapters = 70 * 10 + 11 * 2 = 722
        let targets = enumerate_lora_targets();
        let total_td: usize = TD_INPUT.iter().sum::<usize>() + TD_MIDDLE + TD_OUTPUT.iter().sum::<usize>();
        let st_slots = TD_INPUT.iter().filter(|&&t| t > 0).count() + 1
            + TD_OUTPUT.iter().filter(|&&t| t > 0).count();
        assert_eq!(total_td, 70, "td slot sum (was 80 in prior comment, actual 70)");
        assert_eq!(st_slots, 11);
        assert_eq!(targets.len(), total_td * 10 + st_slots * 2);
        // Sanity: ensure FF and proj_in/out targets are present.
        assert!(targets.iter().any(|(p, _, _)| p.ends_with(".ff.net.0.proj")));
        assert!(targets.iter().any(|(p, _, _)| p.ends_with(".ff.net.2")));
        assert!(targets.iter().any(|(p, _, _)| p.ends_with(".proj_in")));
        assert!(targets.iter().any(|(p, _, _)| p.ends_with(".proj_out")));
    }
}
