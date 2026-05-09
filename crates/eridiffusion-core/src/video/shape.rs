//! `[B, C, F, H, W]` reshape utilities.
//!
//! Cross-model: every video DiT (Wan 2.2, LTX-2, Anima video, future
//! ports) shuffles between image-shaped `[B, 3, H, W]`, video-shaped
//! `[B, 3, F, H, W]`, and per-frame `[B, 3, H, W]` tensors. Centralizing
//! the squeeze/unsqueeze/single-frame-as-video conversions here keeps
//! each `train_*.rs`/`sample_*.rs` from re-deriving the indices.

use flame_core::{Result, Tensor};

/// Insert a temporal dim of length 1 at position 2: `[B, C, H, W]` →
/// `[B, C, 1, H, W]`. Used to feed an image into a video VAE encoder.
pub fn image_to_video(image_4d: &Tensor) -> Result<Tensor> {
    let dims = image_4d.shape().dims();
    if dims.len() != 4 {
        return Err(flame_core::Error::InvalidInput(format!(
            "image_to_video expects [B, C, H, W], got {:?}",
            dims
        )));
    }
    image_4d.unsqueeze(2)
}

/// Inverse of `image_to_video`: `[B, C, 1, H, W]` → `[B, C, H, W]`.
pub fn video_single_frame_to_image(video_5d: &Tensor) -> Result<Tensor> {
    let dims = video_5d.shape().dims();
    if dims.len() != 5 {
        return Err(flame_core::Error::InvalidInput(format!(
            "video_single_frame_to_image expects [B, C, F, H, W], got {:?}",
            dims
        )));
    }
    if dims[2] != 1 {
        return Err(flame_core::Error::InvalidInput(format!(
            "video_single_frame_to_image: expected F=1 single-frame video, got F={}",
            dims[2]
        )));
    }
    video_5d.squeeze(Some(2))
}

/// Squeeze the leading batch dim from a 5D video tensor:
/// `[1, C, F, H, W]` → `[C, F, H, W]`. Used at training time when the
/// model's `forward` takes per-sample 4D tensors (Wan 2.2 convention).
pub fn squeeze_batch_5d(video_5d: &Tensor) -> Result<Tensor> {
    let dims = video_5d.shape().dims();
    if dims.len() != 5 {
        return Err(flame_core::Error::InvalidInput(format!(
            "squeeze_batch_5d expects [1, C, F, H, W], got {:?}",
            dims
        )));
    }
    if dims[0] != 1 {
        return Err(flame_core::Error::InvalidInput(format!(
            "squeeze_batch_5d: expected B=1, got B={}",
            dims[0]
        )));
    }
    video_5d.squeeze(Some(0))
}

/// Inverse: `[C, F, H, W]` → `[1, C, F, H, W]`.
pub fn unsqueeze_batch_4d(video_4d: &Tensor) -> Result<Tensor> {
    let dims = video_4d.shape().dims();
    if dims.len() != 4 {
        return Err(flame_core::Error::InvalidInput(format!(
            "unsqueeze_batch_4d expects [C, F, H, W], got {:?}",
            dims
        )));
    }
    video_4d.unsqueeze(0)
}

/// Compute latent `(F', H', W')` from pixel `(F, H, W)` for a given VAE.
///
/// `vae_spatial_stride`: e.g. 8 for Wan2.1 / Flux VAE, 16 for Wan2.2 VAE.
/// `vae_temporal_factor`: e.g. 4 for Wan VAEs, 1 for image-only VAEs.
///
/// Wan-style temporal compression: `F' = 1 + (F - 1) / temporal_factor`
/// (the first frame stays as a single latent frame; subsequent frames are
/// grouped). For image-only VAEs (`temporal_factor=1`), this is `F' = F`.
pub fn latent_dims(
    pixel: (usize, usize, usize),
    vae_spatial_stride: usize,
    vae_temporal_factor: usize,
) -> (usize, usize, usize) {
    let (f, h, w) = pixel;
    let f_lat = if vae_temporal_factor <= 1 {
        f
    } else {
        1 + (f.saturating_sub(1)) / vae_temporal_factor
    };
    (f_lat, h / vae_spatial_stride, w / vae_spatial_stride)
}

/// Inverse: latent `(F', H', W')` → pixel `(F, H, W)` for a video VAE.
/// Mirrors `latent_dims` arithmetic.
pub fn pixel_dims(
    latent: (usize, usize, usize),
    vae_spatial_stride: usize,
    vae_temporal_factor: usize,
) -> (usize, usize, usize) {
    let (f_lat, h_lat, w_lat) = latent;
    let f_pix = if vae_temporal_factor <= 1 {
        f_lat
    } else {
        1 + (f_lat.saturating_sub(1)) * vae_temporal_factor
    };
    (
        f_pix,
        h_lat * vae_spatial_stride,
        w_lat * vae_spatial_stride,
    )
}

/// Number of pixel frames for a target latent-frame count.
/// Convenience: `pixel_frames(f_lat, 4)` = `1 + (f_lat - 1) * 4`.
pub fn pixel_frames(f_lat: usize, vae_temporal_factor: usize) -> usize {
    if vae_temporal_factor <= 1 {
        f_lat
    } else {
        1 + (f_lat.saturating_sub(1)) * vae_temporal_factor
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latent_dims_wan22_5b() {
        // 5B: VAE 16× spatial, 4× temporal.
        let (f, h, w) = latent_dims((9, 256, 256), 16, 4);
        assert_eq!(f, 1 + 8 / 4); // 3 latent frames
        assert_eq!((h, w), (16, 16));
    }

    #[test]
    fn latent_dims_wan21_14b() {
        // 14B: VAE 8× spatial, 4× temporal.
        let (f, h, w) = latent_dims((9, 256, 256), 8, 4);
        assert_eq!((f, h, w), (3, 32, 32));
    }

    #[test]
    fn latent_image_only_vae() {
        // Flux VAE: spatial 8, temporal 1.
        let (f, h, w) = latent_dims((1, 512, 512), 8, 1);
        assert_eq!((f, h, w), (1, 64, 64));
    }

    #[test]
    fn pixel_frames_roundtrip() {
        for tf in [1usize, 4] {
            for fl in [1usize, 3, 17] {
                let fp = pixel_frames(fl, tf);
                let back = if tf <= 1 { fp } else { 1 + (fp - 1) / tf };
                assert_eq!(back, fl, "tf={} fl={} fp={}", tf, fl, fp);
            }
        }
    }
}
