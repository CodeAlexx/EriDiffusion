//! LTX-2 3D Causal Video VAE.
//!
//! Status: **Structural skeleton with stubbed deep encoder/decoder math.**
//!
//! What is real:
//!   - Constants matching `AutoencoderKLLTX2Video` config (channels=128,
//!     spatial 32× / temporal 8× compression, `latents_mean`/`latents_std`).
//!   - `CausalConv3d` wrapper with left-pad-only temporal padding (mirrors
//!     diffusers `LTX2VideoCausalConv3d.forward`, audit risk #3).
//!   - Per-channel normalize/denormalize using the loaded statistics buffer.
//!   - Image-as-frame encode bootstrap: takes a `[1, 3, H, W]` image,
//!     returns a `[1, 128, 1, H/32, W/32]` latent shape with **pass-through
//!     content** (zeros mean component, gaussian residual). This unblocks
//!     end-to-end training pipeline wiring; it does NOT produce
//!     train-quality latents.
//!
//! What is stubbed (TODO for Verify/follow-up):
//!   - The 4-stage encoder (`compress_space_res`, `compress_time_res`,
//!     `compress_all_res` × 2) — ~900 Rust LoC, see audit §5.2.
//!   - Decoder mirror, including ramp-blended temporal tiling.
//!   - PixelNorm vs GroupNorm switch (per-config).
//!   - `DualConv3d` (factorized 2+1 conv) for the spatial-only blocks.
//!   - Conv3d `.contiguous()` workaround per `feedback_flame_conv3d_contiguous`.
//!
//! ## Audit risks preserved
//!
//! 1. **Per-channel `latents_mean`/`latents_std` buffer is loaded and
//!    applied at the public encode/decode boundary.** Audit risk #1.
//! 2. **Causal pad uses left-pad-only on temporal axis** (first frame
//!    repeated). Audit risk #3 about `cat`-then-Conv3d contiguity is
//!    documented at the call site.
//! 3. **Latent layout is `[B, C, F, H, W]`**, channels at dim 1
//!    (audit "honourable mention" — easy bug spot).

use cudarc::driver::CudaDevice;
use flame_core::conv3d_simple::Conv3d;
use flame_core::{DType, Shape, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

use crate::Result;

// ---------------------------------------------------------------------------
// Public constants (verbatim from diffusers config).
// ---------------------------------------------------------------------------

/// `latent_channels` — both in and out for the VAE bottleneck.
pub const LTX2_LATENT_CHANNELS: usize = 128;
/// Spatial compression (H/32, W/32).
pub const LTX2_SPATIAL_COMPRESSION: usize = 32;
/// Temporal compression: F' = 1 + (F - 1) / 8.
pub const LTX2_TEMPORAL_COMPRESSION: usize = 8;
/// Frames must satisfy `(F - 1) % 8 == 0`.
pub const LTX2_FRAME_DIVISIBILITY: usize = 8;
/// Bucket divisibility for spatial dims (32 px).
pub const LTX2_BUCKET_DIVISIBILITY: usize = 32;

// ---------------------------------------------------------------------------
// CausalConv3d
// ---------------------------------------------------------------------------

/// LTX-2 causal Conv3d with left-pad-only temporal padding.
///
/// ## Audit risk: cat-then-Conv3d contiguity
/// `Tensor::cat` does not guarantee contiguous output, and flame's Conv3d
/// silently reads garbage at the seam if the input is not contiguous
/// (per project memory `feedback_flame_conv3d_contiguous`). This wrapper
/// calls `.contiguous()` after the cat as a defensive measure.
pub struct CausalConv3d {
    pub conv: Conv3d,
    pub kernel_size: (usize, usize, usize),
    pub causal: bool,
}

impl CausalConv3d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize, usize),
        stride: (usize, usize, usize),
        device: Arc<CudaDevice>,
    ) -> Result<Self> {
        // Spatial padding = kernel_size // 2 on H/W; time pad = 0 (we do it
        // manually below by repeating the first frame).
        let padding = (0, kernel_size.1 / 2, kernel_size.2 / 2);
        let conv = Conv3d::new(
            in_channels,
            out_channels,
            kernel_size,
            Some(stride),
            Some(padding),
            None, None, true,
            device,
        )
        .map_err(|e| crate::EriDiffusionError::Model(format!("CausalConv3d Conv3d::new: {e:?}")))?;
        Ok(Self { conv, kernel_size, causal: true })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Causal time-axis pad: first frame repeated `time_kernel_size - 1` times,
        // concatenated to the front. Mirrors diffusers `LTX2VideoCausalConv3d.forward`.
        let time_k = self.kernel_size.0;
        let pad_left_n = time_k - 1;
        let padded = if pad_left_n == 0 {
            x.contiguous()?
        } else {
            // narrow gives a view; contiguous materializes; repeat along time axis.
            let first = x.narrow(2, 0, 1)?.contiguous()?;
            let pad = first.repeat_axis_device(2, pad_left_n)?;
            // Cat may produce a strided view on some shapes — defensive contiguous.
            // (See feedback_flame_conv3d_contiguous.md.)
            Tensor::cat(&[&pad, x], 2)?.contiguous()?
        };
        self.conv.forward(&padded)
            .map_err(|e| crate::EriDiffusionError::Model(format!("CausalConv3d::forward: {e:?}")))
    }
}

// ---------------------------------------------------------------------------
// LTX-2 VAE
// ---------------------------------------------------------------------------

/// LTX-2 3D causal video VAE. T2V LoRA training only needs:
///   - encode at cache time (image-as-frame bootstrap or video frames)
///   - decode at preview/sample time
/// VAE itself is frozen.
pub struct Ltx2Vae {
    pub device: Arc<CudaDevice>,
    /// Per-channel mean of latents, shape `[128]`. Used to normalize at encode
    /// and de-normalize at decode. **Critical** — see audit risk #1.
    pub latents_mean: Tensor,
    /// Per-channel std of latents, shape `[128]`.
    pub latents_std: Tensor,
    pub weights: HashMap<String, Tensor>,
    /// True if `latents_mean`/`latents_std` were loaded from disk; False if
    /// we fell back to (0, 1).
    pub stats_loaded: bool,
}

impl Ltx2Vae {
    pub fn load(ckpt: &std::path::Path, device: Arc<CudaDevice>) -> Result<Self> {
        let weights = if ckpt.exists() {
            flame_core::serialization::load_file(ckpt, &device)?
        } else {
            log::warn!("Ltx2Vae::load: {} does not exist; using empty weights (stub).", ckpt.display());
            HashMap::new()
        };

        // Try canonical key names; fall back to (0, 1).
        let mut stats_loaded = false;
        let latents_mean = match weights.get("latents_mean").cloned() {
            Some(t) => { stats_loaded = true; t.to_dtype(DType::F32)? },
            None => Tensor::zeros_dtype(
                Shape::from_dims(&[LTX2_LATENT_CHANNELS]), DType::F32, device.clone(),
            )?,
        };
        let latents_std = match weights.get("latents_std").cloned() {
            Some(t) => t.to_dtype(DType::F32)?,
            None => {
                // Defensive: ones, not zeros, so divide-by-std doesn't produce inf.
                Tensor::from_vec(
                    vec![1.0f32; LTX2_LATENT_CHANNELS],
                    Shape::from_dims(&[LTX2_LATENT_CHANNELS]),
                    device.clone(),
                )?
            }
        };
        if !stats_loaded {
            log::warn!(
                "Ltx2Vae::load: per-channel latents_mean/latents_std NOT loaded — \
                 falling back to (0, 1). Real training will silently learn the \
                 wrong target distribution. AUDIT RISK #1."
            );
        }
        Ok(Self { device, latents_mean, latents_std, weights, stats_loaded })
    }

    /// Image-as-frame encode bootstrap.
    ///
    /// Input: `[1, 3, H, W]` BF16 image in [-1, 1].
    /// Output: `[1, 128, 1, H/32, W/32]` BF16 latent (post-normalize).
    ///
    /// **STUB**: This does NOT run the real 4-stage compression encoder.
    /// It returns zeros + small-σ Gaussian noise of the correct shape so
    /// the rest of the pipeline (cache writer, dataloader, train loop)
    /// can be exercised end-to-end. **Replace with real encode for any
    /// real training run.** See audit §5.2.
    pub fn encode_image_as_frame(&self, image: &Tensor) -> Result<Tensor> {
        let dims = image.shape().dims();
        if dims.len() != 4 || dims[1] != 3 {
            return Err(crate::EriDiffusionError::Model(format!(
                "encode_image_as_frame expects [1,3,H,W], got {:?}", dims
            )));
        }
        let (b, h, w) = (dims[0], dims[2], dims[3]);
        if h % LTX2_SPATIAL_COMPRESSION != 0 || w % LTX2_SPATIAL_COMPRESSION != 0 {
            return Err(crate::EriDiffusionError::Model(format!(
                "image H={} W={} must each be divisible by {}", h, w, LTX2_SPATIAL_COMPRESSION
            )));
        }
        let h_lat = h / LTX2_SPATIAL_COMPRESSION;
        let w_lat = w / LTX2_SPATIAL_COMPRESSION;
        let shape = Shape::from_dims(&[b, LTX2_LATENT_CHANNELS, 1, h_lat, w_lat]);
        log::warn!(
            "Ltx2Vae::encode_image_as_frame — STUB encoder, producing noise of shape {:?}. \
             Replace with real 4-stage causal-3D encode for real training.",
            shape.dims()
        );
        let raw = Tensor::randn(shape, 0.0, 1.0, self.device.clone())?
            .to_dtype(DType::BF16)?;
        // Apply per-channel normalize so the cached latent has the same stats
        // a real-encoded one would (caller uses these to scale targets).
        self.normalize(&raw)
    }

    /// Decode video latents `[B, 128, F', H/32, W/32]` to pixel video
    /// `[B, 3, F, H, W]` where `F = 1 + (F' - 1) * 8`.
    ///
    /// **STUB**: This is a placeholder that upsamples spatially (nearest-
    /// neighbor by replication) just enough to produce the right output
    /// shape. Output is meaningless gray-scale. Replace with full decoder
    /// (~900 LoC).
    pub fn decode_video(&self, latents: &Tensor) -> Result<Tensor> {
        let dims = latents.shape().dims();
        if dims.len() != 5 || dims[1] != LTX2_LATENT_CHANNELS {
            return Err(crate::EriDiffusionError::Model(format!(
                "decode_video expects [B,128,F',h,w], got {:?}", dims
            )));
        }
        let (b, _c, fp, hp, wp) = (dims[0], dims[1], dims[2], dims[3], dims[4]);
        let f = 1 + (fp - 1) * LTX2_TEMPORAL_COMPRESSION;
        let h = hp * LTX2_SPATIAL_COMPRESSION;
        let w = wp * LTX2_SPATIAL_COMPRESSION;
        let shape = Shape::from_dims(&[b, 3, f, h, w]);
        log::warn!(
            "Ltx2Vae::decode_video — STUB decoder, producing blank video of shape {:?}. \
             Replace with full causal-3D decoder for real previews.",
            shape.dims()
        );
        let zeros = Tensor::zeros_dtype(shape, DType::BF16, self.device.clone())?;
        Ok(zeros)
    }

    /// Apply per-channel normalize: `(x - mean) / std`. Operates on the
    /// channel axis (dim 1) of a `[B, C, F, H, W]` latent.
    pub fn normalize(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        if dims.len() != 5 || dims[1] != LTX2_LATENT_CHANNELS {
            return Err(crate::EriDiffusionError::Model(format!(
                "normalize expects [B,128,F,H,W], got {:?}", dims
            )));
        }
        let mean_b = self
            .latents_mean
            .reshape(&[1, LTX2_LATENT_CHANNELS, 1, 1, 1])?
            .to_dtype(x.dtype())?;
        let std_b = self
            .latents_std
            .reshape(&[1, LTX2_LATENT_CHANNELS, 1, 1, 1])?
            .to_dtype(x.dtype())?;
        let centered = x.sub(&mean_b)?;
        // Avoid full elementwise divide: multiply by reciprocal.
        let std_recip = std_b.reciprocal()?;
        centered.mul(&std_recip).map_err(Into::into)
    }

    /// Inverse of `normalize`: `x * std + mean`. Used pre-decode.
    pub fn denormalize(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        if dims.len() != 5 || dims[1] != LTX2_LATENT_CHANNELS {
            return Err(crate::EriDiffusionError::Model(format!(
                "denormalize expects [B,128,F,H,W], got {:?}", dims
            )));
        }
        let mean_b = self
            .latents_mean
            .reshape(&[1, LTX2_LATENT_CHANNELS, 1, 1, 1])?
            .to_dtype(x.dtype())?;
        let std_b = self
            .latents_std
            .reshape(&[1, LTX2_LATENT_CHANNELS, 1, 1, 1])?
            .to_dtype(x.dtype())?;
        x.mul(&std_b)?.add(&mean_b).map_err(Into::into)
    }
}
