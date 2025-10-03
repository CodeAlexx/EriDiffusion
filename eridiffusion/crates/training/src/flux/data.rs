//! Flux data contracts (config + batch + loader hooks). GPU-only, BF16 storage.
//! Provides: FluxDataConfig, FluxBatch, Synthetic loader, and plug-points for real VAE/text.

use std::collections::HashMap;

use eridiffusion_common_vae::{decode as vae_decode, VaePolicy, VaeSpec};
use eridiffusion_core::Device;
use eridiffusion_models::{
    common_text::attn_mask_from_lengths,
    devtensor::{shape3, shape4, to_device_dtype, zeros_on, BF16},
};
use flame_core::{Result as CoreResult, Tensor};
use serde_json::Value;

/// Minimal training config for Flux. Extend as needed.
#[derive(Clone, Debug)]
pub struct FluxDataConfig {
    pub height: i32,     // image height (px)
    pub width: i32,      // image width  (px)
    pub latent_div: i32, // SD/Flux default 8
    pub batch_size: i32,
    pub fixed_tokens: Option<i32>, // if Some, use fixed T (e.g., 77); else use seq_lens
}

impl Default for FluxDataConfig {
    fn default() -> Self {
        Self { height: 256, width: 256, latent_div: 8, batch_size: 1, fixed_tokens: Some(77) }
    }
}

/// Metadata for samples in a batch (paths, captions, arbitrary JSON metadata).
#[derive(Clone, Debug)]
pub struct FluxSampleRecord {
    pub caption: String,
    pub metadata: HashMap<String, Value>,
}

#[derive(Clone, Debug, Default)]
pub struct FluxBatchStats {
    pub samples: usize,
    pub bytes_latents: u64,
    pub bytes_text: u64,
    pub bytes_pooled: u64,
}

impl FluxBatchStats {
    pub fn total_bytes(&self) -> u64 {
        self.bytes_latents.saturating_add(self.bytes_text).saturating_add(self.bytes_pooled)
    }

    pub fn total_mb(&self) -> f32 {
        self.total_bytes() as f32 / (1024.0 * 1024.0)
    }
}

/// A single training batch for Flux.
#[derive(Clone, Debug)]
pub struct FluxBatch {
    /// Latents [B,4,H/latent_div,W/latent_div] (BF16 on device)
    pub latents: Tensor,
    /// Text context [B,T,hidden] (BF16)
    pub text_ctx: Tensor,
    /// Optional pooled/conditioning tensors (BF16)
    pub pooled: Option<Tensor>,
    /// Optional auxiliary conditioning (e.g., Flux time IDs)
    pub time_ids: Option<Tensor>,
    /// Attention mask [B,1,1,T]
    pub attn_mask: Tensor,
    /// Per-sample token lengths (ignored if fixed_tokens = Some)
    pub seq_lens: Vec<i32>,
    /// Sample metadata for diagnostics/checkpointing
    pub records: Vec<FluxSampleRecord>,
    /// Optional telemetry for host→device transfer sizes
    pub telemetry: Option<FluxBatchStats>,
}

/// Synthetic data: zeros latents + fixed text length. Useful for plumbing tests.
pub fn make_synthetic_batch(cfg: &FluxDataConfig, device: &Device) -> CoreResult<FluxBatch> {
    let b = cfg.batch_size as i64;
    let h = (cfg.height / cfg.latent_div) as i64;
    let w = (cfg.width / cfg.latent_div) as i64;
    let latents = zeros_on(shape4(b, 4, h, w), device, BF16)?;
    let t = cfg.fixed_tokens.unwrap_or(77);
    let seq_lens = vec![t; b as usize];
    let text_ctx = zeros_on(shape3(b, t as i64, 4096), device, BF16)?;
    let attn_mask = attn_mask_from_lengths(&seq_lens, t, device)?;
    Ok(FluxBatch {
        latents,
        text_ctx,
        pooled: None,
        time_ids: None,
        attn_mask,
        seq_lens,
        records: Vec::new(),
        telemetry: None,
    })
}

/// Builder hook for *real* data when you have a VAE + text encoder available.
/// Provide precomputed latents or call your VAE.decode as needed.
pub fn make_real_batch_from_latents(
    latents: Tensor,    // [B,4,H/8,W/8] on device (any dtype)
    text_ctx: Tensor,   // [B,T,H]
    seq_lens: Vec<i32>, // per-sample token lengths
    device: &Device,
    enforce_bf16: bool,
) -> CoreResult<FluxBatch> {
    let max_len = seq_lens.iter().copied().max().unwrap_or(0);
    let latents = if enforce_bf16 { to_device_dtype(&latents, device, BF16)? } else { latents };
    let text_ctx = if enforce_bf16 { to_device_dtype(&text_ctx, device, BF16)? } else { text_ctx };
    let attn_mask = attn_mask_from_lengths(&seq_lens, max_len, device)?;
    Ok(FluxBatch {
        latents,
        text_ctx,
        pooled: None,
        time_ids: None,
        attn_mask,
        seq_lens,
        records: Vec::new(),
        telemetry: None,
    })
}

/// Example hook if you insist on decoding *from* VAE latents (training usually encodes images → latents).
pub fn make_batch_via_vae_decode(
    spec: &VaeSpec,
    latents: &Tensor,
    seq_lens: Vec<i32>,
    device: &Device,
    policy: VaePolicy,
) -> CoreResult<FluxBatch> {
    // Decode for visualization or validation; not typical for training step input.
    let _ = vae_decode(spec, latents, policy)?; // side-effect: validates shapes/dtypes via real decoder.
    let max_len = seq_lens.iter().copied().max().unwrap_or(0);
    let latents = to_device_dtype(latents, device, BF16)?;
    let text_ctx = zeros_on(shape3(seq_lens.len() as i64, max_len as i64, 4096), device, BF16)?;
    let attn_mask = attn_mask_from_lengths(&seq_lens, max_len, device)?;
    Ok(FluxBatch {
        latents,
        text_ctx,
        pooled: None,
        time_ids: None,
        attn_mask,
        seq_lens,
        records: Vec::new(),
        telemetry: None,
    })
}
