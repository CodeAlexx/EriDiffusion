//! LTX-2 video DiT — T2V LoRA training port.
//!
//! Source of truth: `diffusers/models/transformers/transformer_ltx2.py`
//! (1350 LoC, audited 2026-05-05 — see `docs/LTX2_PORT_AUDIT_*.md`).
//!
//! ## Audio excision strategy (T2V-only)
//!
//! Per LTX2_PORT_AUDIT_AITOOLKIT.md §8.2 risk #1, naively dropping audio
//! sub-modules silently desyncs per-block modulation tensor shapes. The
//! safe excision is:
//!
//! 1. **Audio inputs** (`audio_hidden_states`, `audio_encoder_hidden_states`)
//!    are **never plumbed in**: T2V doesn't have them.
//! 2. **Audio output** is discarded (we only return the video output).
//! 3. **Audio sub-modules within each block** (audio_attn1, audio_attn2,
//!    audio_norm1/2/3, audio_ff, audio_to_video_attn, video_to_audio_attn)
//!    are **not invoked**. Reasoning: the only audio→video coupling is the
//!    `audio_to_video_attn` sub-module; with audio inputs at zero, that
//!    cross-attn produces zero output, which is gated by `a2v_gate` and
//!    added to video (zero contribution). So skipping it preserves block
//!    output bit-for-bit vs. the full block run.
//! 4. **Modulation tables**: the per-block `scale_shift_table[6]` (video
//!    self-attn + FF) and the global `time_embed` 6-mod-param output are
//!    consumed unchanged. The cross-attn modulation tables that gate the
//!    a2v / v2a paths are loaded into the model state but unused.
//! 5. **Audio time embedding** and `av_cross_attn_*` global modulation
//!    layers are NOT instantiated (no audio_hidden_states means no
//!    cross-attn that needs gating).
//!
//! ## Architecture summary
//!
//! - 48 layers, hidden=4096 (32 heads × 128 head_dim), inner_dim=4096.
//! - Patchify: spatial+temporal patch_size=1 → flatten `[B, 128, F, H, W]`
//!   to `[B, F*H*W, 128]`, then `proj_in: 128 → 4096`.
//! - 3D RoPE on (frame, h, w) axes, `rope_theta=10000`, dual flavors
//!   (interleaved / split). T2V-only port supports `interleaved` (LTX-2.0).
//! - QK-norm: RMSNorm across the full inner_dim before SDPA.
//! - AdaLN-Single: PixArt-style sin/cos timestep → SiLU → Linear → 6×dim
//!   modulation; applied to video self-attn (3) + video FF (3).
//! - Caption projection: PixArtAlphaTextProjection (Linear → GELU-tanh → Linear)
//!   from 3840 → 4096.
//! - Output: LayerNorm(no-affine) → AdaLN final shift/scale (2 mod params)
//!   → Linear(4096 → 128).
//!
//! ## Hardcoded structural constants
//! - `NUM_LAYERS = 48`, `INNER_DIM = 4096`, `HEAD_DIM = 128`, `HEADS = 32`,
//!   `IN_CHANNELS = 128`, `CAPTION_CHANNELS = 3840`, `NORM_EPS = 1e-6`.
//!
//! ## TODOs flagged for the Verify pass
//! - 3D RoPE: full pixel-coord pre-processing matching diffusers
//!   `prepare_video_coords` (scale by VAE factors, causal_offset clamp,
//!   /fps) is implemented; the `interleaved` flavor with
//!   `repeat_interleave(2)` is implemented; `split` is NOT — emit error.
//! - AdaLN modulation cast to F32 (audit risk #3) — done at the
//!   `silu().linear()` step; subsequent multiplies stay BF16 by default.
//! - Block-level checkpointing not yet wired (clone the ERNIE pattern in
//!   a follow-up).

use std::collections::HashMap;
use std::sync::Arc;

use cudarc::driver::CudaDevice;
use flame_core::{parameter::Parameter, DType, Shape, Tensor};

use crate::config::TrainConfig;
use crate::lora::LoRALinear;
use crate::models::TrainableModel;
use crate::Result;

// ---------------------------------------------------------------------------
// Public constants
// ---------------------------------------------------------------------------

pub const NUM_LAYERS: usize = 48;
pub const HEADS: usize = 32;
pub const HEAD_DIM: usize = 128;
pub const INNER_DIM: usize = HEADS * HEAD_DIM; // 4096
pub const IN_CHANNELS: usize = 128;
pub const OUT_CHANNELS: usize = 128;
pub const CAPTION_CHANNELS: usize = 3840;
pub const FFN_MULT: usize = 4;
pub const FFN_DIM: usize = INNER_DIM * FFN_MULT; // 16384
pub const NORM_EPS: f32 = 1e-6;
pub const ROPE_THETA: f32 = 10000.0;
pub const VAE_SCALE_F: usize = 8;
pub const VAE_SCALE_HW: usize = 32;
pub const CAUSAL_OFFSET: usize = 1;
/// Per-block LoRA slots: video Q/K/V/out + video FF (gate/up/down equivalent;
/// LTX-2 actually uses a standard PixArt FFN of Linear→GELU→Linear, so
/// FF has 2 linears: ff.0 and ff.2). 4 attn linears + 2 FF linears = 6.
pub const LORA_SLOTS_PER_BLOCK: usize = 6;

const LORA_SLOT_KEYS: [&str; LORA_SLOTS_PER_BLOCK] = [
    "attn1.to_q",
    "attn1.to_k",
    "attn1.to_v",
    "attn1.to_out.0",
    "ff.net.0.proj",
    "ff.net.2",
];

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

pub struct Ltx2Model {
    pub config: TrainConfig,
    pub device: Arc<CudaDevice>,
    pub weights: HashMap<String, Tensor>,
    pub lora_adapters: Vec<LoRALinear>,
    pub parameters: Vec<Parameter>,
    pub is_lora: bool,
    /// Number of latent video frames per training sample. Defaults to 1
    /// (image-as-frame bootstrap). Set explicitly for true video.
    pub num_frames: usize,
}

impl Ltx2Model {
    pub fn load(
        ckpt_paths: &[std::path::PathBuf],
        config: &TrainConfig,
        device: Arc<CudaDevice>,
    ) -> Result<Self> {
        let mut weights = HashMap::new();
        for p in ckpt_paths {
            let part = flame_core::serialization::load_file(p, &device)?;
            for (k, v) in part {
                weights.insert(k, v.to_dtype(DType::BF16)?);
            }
        }
        log::info!("LTX-2: {} tensors loaded from {} shard(s)", weights.len(), ckpt_paths.len());

        let is_lora = config.is_lora();
        let mut lora_adapters = Vec::new();
        let mut parameters = Vec::new();
        if is_lora {
            let rank = config.lora_rank as usize;
            let alpha = config.lora_alpha as f32;
            for i in 0..NUM_LAYERS {
                let s = 42u64 + (i as u64) * 16;
                // attn1.to_q/to_k/to_v/to_out: 4096 → 4096
                lora_adapters.push(LoRALinear::new(INNER_DIM, INNER_DIM, rank, alpha, device.clone(), s)?);
                lora_adapters.push(LoRALinear::new(INNER_DIM, INNER_DIM, rank, alpha, device.clone(), s + 1)?);
                lora_adapters.push(LoRALinear::new(INNER_DIM, INNER_DIM, rank, alpha, device.clone(), s + 2)?);
                lora_adapters.push(LoRALinear::new(INNER_DIM, INNER_DIM, rank, alpha, device.clone(), s + 3)?);
                // ff.net.0.proj: 4096 → 16384
                lora_adapters.push(LoRALinear::new(INNER_DIM, FFN_DIM, rank, alpha, device.clone(), s + 4)?);
                // ff.net.2: 16384 → 4096
                lora_adapters.push(LoRALinear::new(FFN_DIM, INNER_DIM, rank, alpha, device.clone(), s + 5)?);
            }
            for l in &lora_adapters { parameters.extend(l.parameters()); }
        } else {
            return Err(crate::EriDiffusionError::Model(
                "LTX-2 only supports LoRA training in this port".into(),
            ));
        }

        Ok(Self {
            config: config.clone(),
            device,
            weights,
            lora_adapters,
            parameters,
            is_lora,
            num_frames: 1,
        })
    }

    fn w(&self, key: &str) -> Result<&Tensor> {
        self.weights.get(key).ok_or_else(|| crate::EriDiffusionError::Model(
            format!("LTX-2 missing weight: {key}"),
        ))
    }

    /// Try to load a weight; if missing, return a zero/identity tensor of
    /// the requested shape. Used for graceful pre-port runs against
    /// checkpoints that haven't been audited for key naming yet.
    fn w_or_zeros(&self, key: &str, shape: &[usize]) -> Result<Tensor> {
        if let Some(t) = self.weights.get(key) {
            Ok(t.clone())
        } else {
            log::warn!("LTX-2: weight '{key}' missing, using zeros");
            Tensor::zeros_dtype(Shape::from_dims(shape), DType::BF16, self.device.clone())
                .map_err(Into::into)
        }
    }

    fn linear(&self, x: &Tensor, w_key: &str, bias_key: Option<&str>) -> Result<Tensor> {
        let w = self.w(w_key)?;
        let mut out = x.matmul(&w.transpose()?)?;
        if let Some(bk) = bias_key {
            if let Some(b) = self.weights.get(bk) {
                out = out.add(b)?;
            }
        }
        Ok(out)
    }

    fn linear_lora(&self, x: &Tensor, w_key: &str, bias_key: Option<&str>, adapter_idx: usize) -> Result<Tensor> {
        let base = self.linear(x, w_key, bias_key)?;
        if self.is_lora {
            if let Some(adapter) = self.lora_adapters.get(adapter_idx) {
                let delta = adapter.forward_delta(x)?;
                return base.add(&delta).map_err(Into::into);
            }
        }
        Ok(base)
    }

    fn rms_norm_full(&self, x: &Tensor, scale_key: &str) -> Result<Tensor> {
        // RMSNorm over the inner_dim axis (last). Scale optional — LTX-2
        // sets `elementwise_affine=False` on most norms, so weight may
        // not exist; in that case pass None.
        let scale = self.weights.get(scale_key);
        flame_core::norm::rms_norm(x, &[INNER_DIM], scale, NORM_EPS)
            .map_err(Into::into)
    }

    /// QK-norm: `rms_norm_across_heads`. The diffusers RMSNorm for QK in
    /// LTX-2 normalizes across the full inner_dim (= heads * head_dim).
    /// Same shape contract as `rms_norm_full`.
    fn qk_norm(&self, x: &Tensor, scale_key: &str) -> Result<Tensor> {
        let scale = self.weights.get(scale_key);
        flame_core::norm::rms_norm(x, &[INNER_DIM], scale, NORM_EPS)
            .map_err(Into::into)
    }

    /// PixArt sinusoidal timestep embedding with `flip_sin_to_cos = True`.
    /// Diffusers `Timesteps(embedding_dim, flip_sin_to_cos=True, downscale_freq_shift=0)`.
    fn timestep_sin_emb(&self, t: &Tensor, dim: usize) -> Result<Tensor> {
        let t_v = t.to_vec()?;
        let half = dim / 2;
        let mut data = vec![0f32; t_v.len() * dim];
        for (bi, &tv) in t_v.iter().enumerate() {
            for j in 0..half {
                let f = (-(10000.0f32).ln() * (j as f32) / (half as f32)).exp();
                let arg = tv * f;
                // flip_sin_to_cos=True: cos first, then sin.
                data[bi * dim + j] = arg.cos();
                data[bi * dim + half + j] = arg.sin();
            }
        }
        Tensor::from_vec(data, Shape::from_dims(&[t_v.len(), dim]), self.device.clone())?
            .to_dtype(DType::BF16)
            .map_err(Into::into)
    }

    /// AdaLN-Single output: 6 × inner_dim modulation parameters from a
    /// (B,) timestep tensor. **Modulation cast to F32** at the linear
    /// step to mitigate audit risk #3 (BF16 modulation overflow); the
    /// downstream multiplies happen in BF16 after a final cast.
    fn ada_modulation(&self, timestep: &Tensor, num_mod_params: usize) -> Result<Tensor> {
        let temb = self.timestep_sin_emb(timestep, INNER_DIM)?;
        // PixArtAlphaCombinedTimestepSizeEmbeddings: linear → silu → linear.
        // In LTX-2 base config, size embedding is disabled
        // (`use_additional_conditions=False`), so just timestep MLP.
        let h1 = self.linear(&temb, "time_embed.emb.timestep_embedder.linear_1.weight",
                              Some("time_embed.emb.timestep_embedder.linear_1.bias"))?;
        let h1 = h1.silu()?;
        let h2 = self.linear(&h1, "time_embed.emb.timestep_embedder.linear_2.weight",
                              Some("time_embed.emb.timestep_embedder.linear_2.bias"))?;

        // Final SiLU then linear to (num_mod_params * inner_dim).
        let act = h2.silu()?;
        // Audit risk #3: cast act to F32 before the modulation linear.
        let act_f32 = act.to_dtype(DType::F32)?;
        // Per-LTX2AdaLayerNormSingle, weight name is `time_embed.linear`.
        let w = self.w("time_embed.linear.weight")?.to_dtype(DType::F32)?;
        let b = self.w_or_zeros("time_embed.linear.bias", &[num_mod_params * INNER_DIM])?
            .to_dtype(DType::F32)?;
        let mod_out = act_f32.matmul(&w.transpose()?)?.add(&b)?;
        // Result shape: [B, num_mod_params * INNER_DIM]
        // Recast to BF16 for downstream block math.
        mod_out.to_dtype(DType::BF16).map_err(Into::into)
    }

    /// 3D RoPE for video tokens. Mirrors diffusers `LTX2AudioVideoRotaryPosEmbed`.
    /// Returns interleaved cos/sin tensors of shape `[1, 1, n_tokens, INNER_DIM]`
    /// (broadcastable over (B, heads)).
    ///
    /// ## CPU-side implementation
    /// 3D RoPE math is integer-grid-driven and small; CPU computation is
    /// fine and avoids new GPU kernels. Output materialized to BF16.
    fn build_video_rope(&self, num_frames: usize, h_lat: usize, w_lat: usize, fps: f32) -> Result<(Tensor, Tensor)> {
        // 1. Build per-token (frame, h, w) coords in pixel space.
        //    pixel_coord = latent_coord * vae_scale; for the first frame,
        //    causal_offset shift + clamp at 0; then divide temporal by fps.
        let n_tokens = num_frames * h_lat * w_lat;
        let mut coords = vec![[0f32; 3]; n_tokens];
        let mut idx = 0;
        for f in 0..num_frames {
            for hi in 0..h_lat {
                for wi in 0..w_lat {
                    // Patch boundaries [start, end); use midpoint = start + 0.5
                    // (since patch_size=1, end = start + 1 → midpoint = start + 0.5).
                    let f_mid_lat = f as f32 + 0.5;
                    let h_mid_lat = hi as f32 + 0.5;
                    let w_mid_lat = wi as f32 + 0.5;

                    // Latent → pixel space.
                    let mut f_pix = f_mid_lat * VAE_SCALE_F as f32;
                    let h_pix = h_mid_lat * VAE_SCALE_HW as f32;
                    let w_pix = w_mid_lat * VAE_SCALE_HW as f32;

                    // Causal offset on temporal axis: shift by causal_offset - vae_scale_F,
                    // clamp at 0.
                    f_pix = (f_pix + CAUSAL_OFFSET as f32 - VAE_SCALE_F as f32).max(0.0);

                    // Temporal axis scaled by fps → seconds.
                    f_pix /= fps;

                    coords[idx] = [f_pix, h_pix, w_pix];
                    idx += 1;
                }
            }
        }

        // 2. Compute pow_indices. Each axis gets `inner_dim / (3 * 2)` freqs
        //    (3 axes, 2 components per axis = cos + sin → "num_rope_elems = 6").
        let num_pos_dims = 3usize;
        let num_rope_elems = num_pos_dims * 2;
        let freqs_per_axis = INNER_DIM / num_rope_elems; // 4096 / 6 = 682 (with remainder 4)
        let pad_per_freq = INNER_DIM - freqs_per_axis * num_rope_elems; // 4

        // pow_indices: theta ^ linspace(0, 1, freqs_per_axis), then * pi/2.
        let mut pow_indices = Vec::with_capacity(freqs_per_axis);
        for j in 0..freqs_per_axis {
            let lin = if freqs_per_axis > 1 { j as f32 / (freqs_per_axis - 1) as f32 } else { 0.0 };
            pow_indices.push(ROPE_THETA.powf(lin) * std::f32::consts::FRAC_PI_2);
        }

        // 3. Per-token, per-axis: freq_per_axis values = (grid * 2 - 1) * pow_indices.
        //    grid = coord / max_position. For video: max_positions = (20, 2048, 2048).
        let max_positions = [20.0_f32, 2048.0, 2048.0];

        // Output layout: cos and sin, each [n_tokens, INNER_DIM].
        // For interleaved: per-axis (freqs_per_axis) values are repeated
        // pairwise (interleave(2)) giving 2*freqs_per_axis = INNER_DIM/3 per
        // axis. Then concat across 3 axes → 3 * (INNER_DIM/3) = INNER_DIM.
        // With remainder pad_per_freq: pad with cos=1, sin=0 at the front.
        let mut cos_data = vec![0f32; n_tokens * INNER_DIM];
        let mut sin_data = vec![0f32; n_tokens * INNER_DIM];
        for (ti, c) in coords.iter().enumerate() {
            let row = ti * INNER_DIM;
            // Padding lane: the front pad_per_freq positions get (cos=1, sin=0).
            for p in 0..pad_per_freq {
                cos_data[row + p] = 1.0;
                sin_data[row + p] = 0.0;
            }
            let mut col = pad_per_freq;
            for axis in 0..3 {
                let grid = c[axis] / max_positions[axis];
                let scaled = grid * 2.0 - 1.0;
                for &pi in &pow_indices {
                    let arg = scaled * pi;
                    let cv = arg.cos();
                    let sv = arg.sin();
                    // Interleaved: emit cv,cv  sv,sv (two slots per freq).
                    cos_data[row + col] = cv;
                    cos_data[row + col + 1] = cv;
                    sin_data[row + col] = sv;
                    sin_data[row + col + 1] = sv;
                    col += 2;
                }
            }
        }
        let cos = Tensor::from_vec(
            cos_data,
            Shape::from_dims(&[1, 1, n_tokens, INNER_DIM]),
            self.device.clone(),
        )?
        .to_dtype(DType::BF16)?;
        let sin = Tensor::from_vec(
            sin_data,
            Shape::from_dims(&[1, 1, n_tokens, INNER_DIM]),
            self.device.clone(),
        )?
        .to_dtype(DType::BF16)?;
        Ok((cos, sin))
    }

    /// Apply interleaved RoPE to a tensor of shape `[B, H, T, D]` where D
    /// is the head_dim×heads (= inner_dim) flattened across heads.
    ///
    /// **Note**: diffusers applies RoPE to the linear-output Q/K of shape
    /// `[B, T, inner_dim]` *before* the unflatten-to-heads step, then
    /// unflattens. The RoPE freq layout matches the flat inner_dim. That's
    /// what we do here too — caller passes Q/K shape `[B, T, INNER_DIM]`,
    /// our cos/sin is `[1, 1, T, INNER_DIM]` (broadcast over batch + a
    /// trivial unsqueezed head axis we skip by working pre-unflatten).
    fn apply_rope(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        // x: [B, T, INNER_DIM]; reshape to [B, T, INNER_DIM/2, 2] then
        // unbind last → x_real, x_imag. Rotated = stack([-x_imag, x_real]).
        let dims = x.shape().dims().to_vec();
        if dims.len() != 3 || dims[2] != INNER_DIM {
            return Err(crate::EriDiffusionError::Model(format!(
                "apply_rope expects [B, T, INNER_DIM], got {:?}", dims
            )));
        }
        let (b, t, _) = (dims[0], dims[1], dims[2]);
        let half = INNER_DIM / 2;
        let xv = x.reshape(&[b, t, half, 2])?.contiguous()?;
        let x_real = xv.narrow(3, 0, 1)?.contiguous()?.reshape(&[b, t, half])?;
        let x_imag = xv.narrow(3, 1, 1)?.contiguous()?.reshape(&[b, t, half])?;

        // Rotated stacked: [-x_imag, x_real] → flatten to [B, T, INNER_DIM].
        let neg_imag = x_imag.mul_scalar(-1.0)?;
        let stacked = Tensor::cat(
            &[&neg_imag.reshape(&[b, t, half, 1])?, &x_real.reshape(&[b, t, half, 1])?],
            3,
        )?
        .contiguous()?
        .reshape(&[b, t, INNER_DIM])?;

        // cos, sin are [1, 1, T, INNER_DIM]; reshape to [1, T, INNER_DIM] for broadcast.
        let cos_b = cos.reshape(&[1, t, INNER_DIM])?.to_dtype(DType::BF16)?;
        let sin_b = sin.reshape(&[1, t, INNER_DIM])?.to_dtype(DType::BF16)?;
        let prod_cos = x.mul(&cos_b)?;
        let prod_sin = stacked.mul(&sin_b)?;
        prod_cos.add(&prod_sin).map_err(Into::into)
    }

    /// Reshape Q/K/V of `[B, T, INNER_DIM]` to `[B, HEADS, T, HEAD_DIM]` for SDPA.
    fn to_heads(&self, x: &Tensor, b: usize, t: usize) -> Result<Tensor> {
        x.reshape(&[b, t, HEADS, HEAD_DIM])?.permute(&[0, 2, 1, 3])
            .map_err(Into::into)
    }

    /// Inverse of `to_heads` after SDPA: `[B, HEADS, T, HEAD_DIM]` → `[B, T, INNER_DIM]`.
    fn from_heads(&self, x: &Tensor, b: usize, t: usize) -> Result<Tensor> {
        x.permute(&[0, 2, 1, 3])?.contiguous()?.reshape(&[b, t, INNER_DIM])
            .map_err(Into::into)
    }

    /// Forward.
    ///
    /// Inputs:
    /// - `latent`: `[B, IN_CHANNELS=128, F, H, W]` BF16.
    /// - `text_emb`: `[B, T_text, CAPTION_CHANNELS=3840]` BF16.
    /// - `timestep`: `[B]` F32 (will be passed through directly to `time_embed`,
    ///   pre-multiplied by `timestep_scale_multiplier=1000` by caller).
    /// - `fps`: video frame rate (used by RoPE coord builder; default 24.0).
    ///
    /// Output: `[B, OUT_CHANNELS=128, F, H, W]` BF16 (velocity prediction).
    pub fn forward(
        &mut self,
        latent: &Tensor,
        text_emb: &Tensor,
        timestep: &Tensor,
        fps: f32,
    ) -> Result<Tensor> {
        let dims = latent.shape().dims().to_vec();
        if dims.len() != 5 || dims[1] != IN_CHANNELS {
            return Err(crate::EriDiffusionError::Model(format!(
                "LTX-2 forward expects [B,128,F,H,W], got {:?}", dims
            )));
        }
        let (b, _c, f, h_lat, w_lat) = (dims[0], dims[1], dims[2], dims[3], dims[4]);
        let n_tokens = f * h_lat * w_lat;

        // 1. Patchify [B, C, F, H, W] → [B, F*H*W, C].
        // permute to [B, F, H, W, C] then flatten (F*H*W).
        let x = latent.permute(&[0, 2, 3, 4, 1])?.contiguous()?
            .reshape(&[b, n_tokens, IN_CHANNELS])?;
        // proj_in: 128 → 4096
        let mut hidden = self.linear(&x, "proj_in.weight", Some("proj_in.bias"))?;

        // 2. Caption projection: [B, T_text, 3840] → [B, T_text, 4096].
        // PixArtAlphaTextProjection: Linear → GELU(tanh) → Linear.
        let cap_h1 = self.linear(text_emb, "caption_projection.linear_1.weight",
                                  Some("caption_projection.linear_1.bias"))?;
        let cap_h1 = cap_h1.gelu()?;
        let encoder_hidden_states = self.linear(&cap_h1, "caption_projection.linear_2.weight",
                                                  Some("caption_projection.linear_2.bias"))?;

        // 3. AdaLN-Single global modulation (6 mod params × inner_dim, plus
        //    embedded_timestep for output layer).
        let mod_full = self.ada_modulation(timestep, 6)?; // [B, 6 * 4096]
        let mod_chunks = mod_full.chunk(6, 1)?;
        let shift_msa = mod_chunks[0].unsqueeze(1)?;
        let scale_msa = mod_chunks[1].unsqueeze(1)?;
        let gate_msa = mod_chunks[2].unsqueeze(1)?;
        let shift_mlp = mod_chunks[3].unsqueeze(1)?;
        let scale_mlp = mod_chunks[4].unsqueeze(1)?;
        let gate_mlp = mod_chunks[5].unsqueeze(1)?;

        // Embedded timestep for output layer (just the post-MLP F32 cast):
        // re-derive by running the MLP again without the final mod-linear.
        let temb_for_out = {
            let temb = self.timestep_sin_emb(timestep, INNER_DIM)?;
            let h1 = self.linear(&temb, "time_embed.emb.timestep_embedder.linear_1.weight",
                                  Some("time_embed.emb.timestep_embedder.linear_1.bias"))?;
            let h1 = h1.silu()?;
            self.linear(&h1, "time_embed.emb.timestep_embedder.linear_2.weight",
                          Some("time_embed.emb.timestep_embedder.linear_2.bias"))?
        };

        // 4. Build 3D RoPE for video self-attention.
        let (cos_b, sin_b) = self.build_video_rope(f, h_lat, w_lat, fps)?;

        // 5. Run 48 transformer blocks (T2V slice — see audio excision strategy
        //    in module docs).
        for i in 0..NUM_LAYERS {
            hidden = self.block_forward_t2v(
                hidden,
                &encoder_hidden_states,
                &shift_msa, &scale_msa, &gate_msa,
                &shift_mlp, &scale_mlp, &gate_mlp,
                &cos_b, &sin_b,
                i,
                b, n_tokens,
            )?;
        }

        // 6. Output layer: LayerNorm(no-affine) → AdaLN final shift/scale
        //    (2 mod params from `scale_shift_table[2, dim] + temb_for_out`).
        let x_n = flame_core::layer_norm::layer_norm(&hidden, &[INNER_DIM], None, None, NORM_EPS)?;

        // scale_shift_table is a parameter [2, 4096]; add the embedded_timestep.
        // diffusers: scale_shift_values = self.scale_shift_table[None, None] +
        //            embedded_timestep[:, :, None]
        // → shape [B, T_text(or 1), 2, dim] → unbind dim=2 → shift, scale.
        // For our T2V slice we treat embedded_timestep as [B, dim] (no per-text-token
        // expansion since we're not doing audio dual-stream). Reshape to [B, 1, 1, dim]
        // for broadcast.
        let sst = self.w("scale_shift_table")?
            .reshape(&[1, 1, 2, INNER_DIM])?
            .to_dtype(DType::BF16)?;
        let temb_b = temb_for_out.reshape(&[b, 1, 1, INNER_DIM])?;
        let scale_shift = sst.add(&temb_b)?;
        let shift = scale_shift.narrow(2, 0, 1)?.squeeze_dim(2)?;  // [B, 1, INNER_DIM]
        let scale = scale_shift.narrow(2, 1, 1)?.squeeze_dim(2)?;
        let x_out = x_n.mul(&scale.add_scalar(1.0)?)?.add(&shift)?;

        // proj_out: 4096 → 128
        let projected = self.linear(&x_out, "proj_out.weight", Some("proj_out.bias"))?;
        // [B, n_tokens, 128] → [B, F, H, W, C] → [B, C, F, H, W]
        let unpacked = projected
            .reshape(&[b, f, h_lat, w_lat, OUT_CHANNELS])?
            .permute(&[0, 4, 1, 2, 3])?
            .contiguous()?;
        Ok(unpacked)
    }

    /// One block, T2V slice (audio sub-modules excised — see module docs
    /// for safety justification). Implements:
    ///   1. video self-attn with RoPE + AdaLN modulation
    ///   2. video cross-attn-text (no RoPE on text side)
    ///   3. video FF (gelu-approximate) with AdaLN modulation
    #[allow(clippy::too_many_arguments)]
    fn block_forward_t2v(
        &self,
        x: Tensor,
        text_proj: &Tensor,
        shift_msa: &Tensor, scale_msa: &Tensor, gate_msa: &Tensor,
        shift_mlp: &Tensor, scale_mlp: &Tensor, gate_mlp: &Tensor,
        cos_b: &Tensor, sin_b: &Tensor,
        layer_idx: usize,
        b: usize, t_video: usize,
    ) -> Result<Tensor> {
        let lora_base = layer_idx * LORA_SLOTS_PER_BLOCK;
        let pre = format!("transformer_blocks.{layer_idx}");

        // ── 1. Video self-attn ──
        let r = x.clone();
        let n = self.rms_norm_full(&x, &format!("{pre}.norm1.weight"))?;
        // Per-block scale_shift_table [6, dim]; first 3 rows = self-attn (shift, scale, gate).
        // We use the GLOBAL ada modulation for now (no per-block table addition) —
        // diffusers does `scale_shift_table[None, None] + temb.reshape(B, T, 6, -1)`.
        // That per-block table is small (6 × 4096); add it.
        let sst = self.w(&format!("{pre}.scale_shift_table"))?
            .reshape(&[1, 1, 6, INNER_DIM])?
            .to_dtype(DType::BF16)?;
        // global mods are already chunked; per-block table has 6 separate slots.
        let block_shift_msa = sst.narrow(2, 0, 1)?.squeeze_dim(2)?;
        let block_scale_msa = sst.narrow(2, 1, 1)?.squeeze_dim(2)?;
        let block_gate_msa = sst.narrow(2, 2, 1)?.squeeze_dim(2)?;
        let block_shift_mlp = sst.narrow(2, 3, 1)?.squeeze_dim(2)?;
        let block_scale_mlp = sst.narrow(2, 4, 1)?.squeeze_dim(2)?;
        let block_gate_mlp = sst.narrow(2, 5, 1)?.squeeze_dim(2)?;

        let s_msa = block_shift_msa.add(shift_msa)?;
        let sc_msa = block_scale_msa.add(scale_msa)?;
        let g_msa = block_gate_msa.add(gate_msa)?;
        let s_mlp = block_shift_mlp.add(shift_mlp)?;
        let sc_mlp = block_scale_mlp.add(scale_mlp)?;
        let g_mlp = block_gate_mlp.add(gate_mlp)?;

        let m = n.mul(&sc_msa.add_scalar(1.0)?)?.add(&s_msa)?;

        let q = self.linear_lora(&m, &format!("{pre}.attn1.to_q.weight"), Some(&format!("{pre}.attn1.to_q.bias")), lora_base)?;
        let k = self.linear_lora(&m, &format!("{pre}.attn1.to_k.weight"), Some(&format!("{pre}.attn1.to_k.bias")), lora_base + 1)?;
        let v = self.linear_lora(&m, &format!("{pre}.attn1.to_v.weight"), Some(&format!("{pre}.attn1.to_v.bias")), lora_base + 2)?;
        let q_n = self.qk_norm(&q, &format!("{pre}.attn1.norm_q.weight"))?;
        let k_n = self.qk_norm(&k, &format!("{pre}.attn1.norm_k.weight"))?;

        // RoPE on Q, K; not on V.
        let q_r = self.apply_rope(&q_n, cos_b, sin_b)?;
        let k_r = self.apply_rope(&k_n, cos_b, sin_b)?;

        let qh = self.to_heads(&q_r, b, t_video)?;
        let kh = self.to_heads(&k_r, b, t_video)?;
        let vh = self.to_heads(&v, b, t_video)?;
        let attn_out = flame_core::attention::sdpa(&qh, &kh, &vh, None)?;
        let attn_out = self.from_heads(&attn_out, b, t_video)?;
        let out = self.linear_lora(
            &attn_out,
            &format!("{pre}.attn1.to_out.0.weight"),
            Some(&format!("{pre}.attn1.to_out.0.bias")),
            lora_base + 3,
        )?;
        let x = r.add(&g_msa.mul(&out)?)?;

        // ── 2. Video cross-attn to text ──
        let r2 = x.clone();
        let n2 = self.rms_norm_full(&x, &format!("{pre}.norm2.weight"))?;
        // attn2 is cross-attn: K, V come from `text_proj`.
        let q2 = self.linear(&n2, &format!("{pre}.attn2.to_q.weight"),
                              Some(&format!("{pre}.attn2.to_q.bias")))?;
        let k2 = self.linear(text_proj, &format!("{pre}.attn2.to_k.weight"),
                              Some(&format!("{pre}.attn2.to_k.bias")))?;
        let v2 = self.linear(text_proj, &format!("{pre}.attn2.to_v.weight"),
                              Some(&format!("{pre}.attn2.to_v.bias")))?;
        let q2_n = self.qk_norm(&q2, &format!("{pre}.attn2.norm_q.weight"))?;
        let k2_n = self.qk_norm(&k2, &format!("{pre}.attn2.norm_k.weight"))?;

        let t_text = text_proj.shape().dims()[1];
        let q2h = self.to_heads(&q2_n, b, t_video)?;
        let k2h = self.to_heads(&k2_n, b, t_text)?;
        let v2h = self.to_heads(&v2, b, t_text)?;
        let attn2_out = flame_core::attention::sdpa(&q2h, &k2h, &v2h, None)?;
        let attn2_out = self.from_heads(&attn2_out, b, t_video)?;
        let out2 = self.linear(&attn2_out, &format!("{pre}.attn2.to_out.0.weight"),
                                Some(&format!("{pre}.attn2.to_out.0.bias")))?;
        let x = r2.add(&out2)?;

        // ── 3. Video FF ──
        let r3 = x.clone();
        let n3 = self.rms_norm_full(&x, &format!("{pre}.norm3.weight"))?;
        let m3 = n3.mul(&sc_mlp.add_scalar(1.0)?)?.add(&s_mlp)?;
        // ff.net.0.proj: 4096 → 16384 → GELU(tanh) → ff.net.2: 16384 → 4096
        let ff1 = self.linear_lora(&m3, &format!("{pre}.ff.net.0.proj.weight"),
                                     Some(&format!("{pre}.ff.net.0.proj.bias")), lora_base + 4)?;
        // GELU tanh-approx: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        // flame's `Tensor::gelu` is the exact GELU; for parity we hand-roll
        // the tanh approximation here.
        let ff1 = gelu_tanh(&ff1)?;
        let ff2 = self.linear_lora(&ff1, &format!("{pre}.ff.net.2.weight"),
                                     Some(&format!("{pre}.ff.net.2.bias")), lora_base + 5)?;
        let x = r3.add(&g_mlp.mul(&ff2)?)?;

        Ok(x)
    }
}

/// GELU tanh-approx (matches PyTorch `gelu(approximate='tanh')`).
fn gelu_tanh(x: &Tensor) -> Result<Tensor> {
    // 0.044715, sqrt(2/pi) ≈ 0.7978845608.
    let cube = x.mul(x)?.mul(x)?.mul_scalar(0.044715)?;
    let arg = x.add(&cube)?.mul_scalar(0.7978845608)?;
    let t = arg.tanh()?;
    let inner = t.add_scalar(1.0)?;
    x.mul(&inner)?.mul_scalar(0.5).map_err(Into::into)
}

impl TrainableModel for Ltx2Model {
    fn forward(&mut self, noisy: &Tensor, timestep: &Tensor, context: &[Tensor], _p: Option<&Tensor>) -> Result<Tensor> {
        let txt = context.first().ok_or_else(|| crate::EriDiffusionError::Model(
            "LTX-2 needs text embeddings in context[0]".into(),
        ))?;
        // FPS not exposed via TrainableModel; default 24.0 (LTX-2 standard).
        Ltx2Model::forward(self, noisy, txt, timestep, 24.0)
    }
    fn parameters(&self) -> Vec<Parameter> { self.parameters.clone() }
    fn post_optimizer_step(&mut self) {}

    fn save_weights(&self, path: &str) -> Result<()> {
        if !self.is_lora {
            return Err(crate::EriDiffusionError::Model(
                "save_weights for non-LoRA LTX-2 not implemented".into(),
            ));
        }
        let mut out = HashMap::new();
        for (i, adapter) in self.lora_adapters.iter().enumerate() {
            let layer_idx = i / LORA_SLOTS_PER_BLOCK;
            let slot = i % LORA_SLOTS_PER_BLOCK;
            let prefix = format!("transformer_blocks.{layer_idx}.{}", LORA_SLOT_KEYS[slot]);
            adapter.save_tensors(&prefix, &mut out)?;
        }
        flame_core::serialization::save_file(&out, std::path::Path::new(path))
            .map_err(|e| crate::EriDiffusionError::Safetensors(format!("save_file: {e}")))?;
        Ok(())
    }

    fn load_weights(&mut self, path: &str) -> Result<()> {
        if !self.is_lora {
            return Err(crate::EriDiffusionError::Model(
                "load_weights for non-LoRA LTX-2 not implemented".into(),
            ));
        }
        let source = flame_core::serialization::load_file(
            std::path::Path::new(path),
            &self.device,
        )
        .map_err(|e| crate::EriDiffusionError::Safetensors(format!("load_file: {e}")))?;
        for (i, adapter) in self.lora_adapters.iter().enumerate() {
            let layer_idx = i / LORA_SLOTS_PER_BLOCK;
            let slot = i % LORA_SLOTS_PER_BLOCK;
            let prefix = format!("transformer_blocks.{layer_idx}.{}", LORA_SLOT_KEYS[slot]);
            adapter.load_tensors(&prefix, &source)?;
        }
        Ok(())
    }
}
