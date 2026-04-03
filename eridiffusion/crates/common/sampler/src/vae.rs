use anyhow::Result;
use eridiffusion_common_vae as native_vae;
use flame_core::{DType, Tensor};
use image::{ImageBuffer, Rgb};
use std::path::Path;

pub use native_vae::{VaeKind, VaePolicy, VaeSpec};

/// images: NHWC bf16 [B,H,W,3] → latents: NHWC bf16 [B,H/ld,W/ld,C]
pub fn vae_encode(spec: &VaeSpec, images: &Tensor, policy: VaePolicy) -> Result<Tensor> {
    let latents = native_vae::encode(spec, images, policy)?;

    super::log_stats_gpu_only("vae_encode", &latents);

    #[cfg(feature = "vae_debug_stats")]
    {
        log_stats_once("vae_encode", &latents)?;
    }

    Ok(latents)
}

#[cfg(feature = "vae_debug_stats")]
fn log_stats_once(tag: &str, t: &Tensor) -> Result<()> {
    let v = t.to_vec()?;
    if v.is_empty() {
        return Ok(());
    }
    let (mut mn, mut mx, mut sum, mut sq) = (v[0], v[0], 0.0f32, 0.0f32);
    for &x in v.iter().take(1_000_000) {
        mn = mn.min(x);
        mx = mx.max(x);
        sum += x;
        sq += x * x;
    }
    let n = v.len().min(1_000_000) as f32;
    let mean = sum / n;
    let var = (sq / n) - (mean * mean);
    let std = var.max(0.0).sqrt();
    eprintln!("[vae.stats] {tag} mean={mean:.4} std={std:.4} min={mn:.4} max={mx:.4}");
    Ok(())
}

#[cfg(not(feature = "vae_debug_stats"))]
fn log_stats_once(_tag: &str, _t: &Tensor) -> Result<()> {
    Ok(())
}

/// Decode NHWC latents with the native VAE and save as PNG ([-1,1] → [0,255]).
pub fn vae_decode_to_png(spec: &VaeSpec, latents: &Tensor, policy: VaePolicy, out_path: &str) -> Result<()> {
    let decoded = native_vae::decode(spec, latents, policy)?;
    let img = decoded.to_dtype(DType::F32)?;
    let dims = img.shape().dims().to_vec();
    anyhow::ensure!(dims.len() == 4 && dims[0] == 1 && dims[3] == 3, "expected [1,H,W,3], got {:?}", dims);

    let h = dims[1];
    let w = dims[2];
    let data = img.to_vec()?;
    let mut buf = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(w as u32, h as u32);
    for y in 0..h {
        for x in 0..w {
            let base = (y * w + x) * 3;
            let r = ((data[base] * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8;
            let g = ((data[base + 1] * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8;
            let b = ((data[base + 2] * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8;
            buf.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
        }
    }
    buf.save(Path::new(out_path))?;
    Ok(())
}
