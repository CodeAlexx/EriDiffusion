// GPU-only VAE preprocessor: CPU image decode/resize → GPU NHWC BF16 → VAE encode → NHWC BF16 latents
// No Candle. Uses Flame tensors and your shared VAE crate.

use anyhow::{Context, Result, bail};
use flame_core::{Tensor, Shape, DType};
use eridiffusion_common_vae::{VaeSpec, VaePolicy, VaeKind, encode as vae_encode};
use image::{GenericImageView, ImageBuffer, Rgb, imageops::FilterType};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Controls CPU-side image prep and GPU encode batching.
#[derive(Debug, Clone)]
pub struct PreprocessConfig {
    /// Target size (H, W) after optional crop & resize (in pixels).
    pub target_size: (u32, u32),
    /// If true, center-crop to square before resize (common for 1:1 datasets).
    pub center_crop: bool,
    /// Map RGB from [0,255] → [0,1] → [-1,1] if true (SD pipelines expect [-1,1]).
    pub normalize_to_neg1_1: bool,
    /// Number of images per GPU micro-batch for encode.
    pub microbatch: usize,
}

impl Default for PreprocessConfig {
    fn default() -> Self {
        Self {
            target_size: (1024, 1024),
            center_crop: true,
            normalize_to_neg1_1: true,
            microbatch: 4,
        }
    }
}

/// Metadata describing one processed sample (useful for manifests).
#[derive(Debug, Clone)]
pub struct SampleInfo {
    pub path: PathBuf,
    pub orig_hw: (u32, u32),
    pub crop_xy: (u32, u32),
    pub out_hw: (u32, u32),
}

/// Output of a preprocessing+encode pass.
#[derive(Debug)]
pub struct LatentBatch {
    /// NHWC BF16 latents: [B, H/ld, W/ld, C] (C typically 4).
    pub latents: Tensor,
    /// Per-sample info aligned with the batch dimension.
    pub info: Vec<SampleInfo>,
}

pub struct VaePreprocessor {
    /// CUDA device handle for tensor creation.
    cuda: Arc<flame_core::CudaDevice>,
    /// VAE spec (kind/path/div/channels/scale).
    vae: VaeSpec,
    /// Optional height tiling for extremely large resolutions (still GPU-only).
    tile_h: Option<i64>,
}

impl VaePreprocessor {
    /// Create a GPU-only preprocessor for a given CUDA device index and VAE spec.
    pub fn new(cuda_index: usize, vae: VaeSpec) -> Result<Self> {
        let cuda = flame_core::CudaDevice::new(cuda_index)?;
        Ok(Self { cuda, vae, tile_h: None })
    }

    /// Optional GPU tiling along H in decode stages of the VAE (not used in encode).
    pub fn with_tile_h(mut self, tile_h: Option<i64>) -> Self {
        self.tile_h = tile_h;
        self
    }

    /// Encode a list of image paths into latents (NHWC BF16) using the VAE on GPU.
    /// CPU work: decode + resize. Everything else is on GPU.
    pub fn encode_paths(&self, paths: &[PathBuf], cfg: &PreprocessConfig) -> Result<LatentBatch> {
        if paths.is_empty() {
            bail!("encode_paths: empty input list");
        }

        // CPU decode + preprocess each image
        let mut infos = Vec::with_capacity(paths.len());
        let mut imgs_bf16: Vec<Tensor> = Vec::with_capacity(paths.len());

        for p in paths {
            let (rgb, orig_hw, crop_xy) = load_preprocess_rgb(p, cfg)
                .with_context(|| format!("decode/resize failed for {:?}", p))?;
            let (h, w) = (rgb.height() as usize, rgb.width() as usize);

            // Flatten CPU image to f32 vector (NHWC)
            let mut host_f32: Vec<f32> = Vec::with_capacity(h * w * 3);
            for (_y, row) in rgb.rows().enumerate() {
                for pixel in row {
                    let r = pixel[0] as f32 / 255.0;
                    let g = pixel[1] as f32 / 255.0;
                    let b = pixel[2] as f32 / 255.0;
                    host_f32.push(r);
                    host_f32.push(g);
                    host_f32.push(b);
                }
            }
            if cfg.normalize_to_neg1_1 {
                for v in &mut host_f32 { *v = *v * 2.0 - 1.0; }
            }

            // Upload to GPU as F32 NHWC, then cast to BF16
            let shape = Shape::from_dims(&[1, h, w, 3]);
            let img_f32 = Tensor::from_slice(&host_f32, shape, self.cuda.clone())?;
            let img_bf16 = img_f32.to_dtype(DType::BF16)?;
            imgs_bf16.push(img_bf16);

            infos.push(SampleInfo { path: p.clone(), orig_hw, crop_xy, out_hw: (w as u32, h as u32) });
        }

        // Micro-batched encode on GPU
        let mut latent_chunks: Vec<Tensor> = Vec::new();
        for chunk in imgs_bf16.chunks(cfg.microbatch.max(1)) {
            let batch = stack_nhwc(chunk)?;
            let lat = vae_encode(&self.vae, &batch, VaePolicy::GpuFirst)
                .context("vae_encode failed")?;
            latent_chunks.push(lat);
        }
        let latents = concat_nhwc(latent_chunks, /*dim=*/0)?;

        Ok(LatentBatch { latents, info: infos })
    }
}

/* ---------- helpers ---------- */

/// CPU load, optional center-crop to square, then resize to target (H,W). Returns RgbImage.
fn load_preprocess_rgb(path: &Path, cfg: &PreprocessConfig) -> Result<(ImageBuffer<Rgb<u8>, Vec<u8>>, (u32,u32), (u32,u32))> {
    let img = image::open(path)
        .with_context(|| format!("failed to open {:?}", path))?;
    let (orig_w, orig_h) = img.dimensions();

    // Convert to RGB8 (drop alpha if present)
    let mut rgb = img.to_rgb8();

    // Center-crop to square if requested
    let (crop_x, crop_y);
    if cfg.center_crop {
        let min_side = orig_w.min(orig_h);
        crop_x = (orig_w - min_side) / 2;
        crop_y = (orig_h - min_side) / 2;
        rgb = image::imageops::crop(&mut rgb, crop_x, crop_y, min_side, min_side).to_image();
    } else {
        crop_x = 0; crop_y = 0;
    }

    // Resize to target
    let (t_h, t_w) = cfg.target_size;
    let out = image::imageops::resize(&rgb, t_w, t_h, FilterType::Lanczos3);

    Ok((out, (orig_h, orig_w), (crop_x, crop_y)))
}

/// Stack a slice of NHWC tensors along batch dim (0).
fn stack_nhwc(xs: &[Tensor]) -> Result<Tensor> {
    anyhow::ensure!(!xs.is_empty(), "stack_nhwc: empty");
    // Verify common shape/dtype
    let base = &xs[0];
    let dims = base.shape().dims().to_vec();
    anyhow::ensure!(dims.len() == 4, "stack_nhwc expects [B,H,W,C]");
    let mut items: Vec<Tensor> = Vec::with_capacity(xs.len());
    for x in xs {
        anyhow::ensure!(x.dtype() == base.dtype(), "dtype mismatch in stack");
        let d = x.shape().dims().to_vec();
        anyhow::ensure!(d[1] == dims[1] && d[2] == dims[2] && d[3] == dims[3], "shape mismatch in stack");
        items.push(x.clone());
    }
    // Flame stack along batch dimension
    Tensor::stack(&items, 0).map_err(Into::into)
}

/// Concatenate NHWC tensors along dim (0)
fn concat_nhwc(chunks: Vec<Tensor>, dim: i64) -> Result<Tensor> {
    anyhow::ensure!(!chunks.is_empty(), "concat_nhwc: empty");
    // Flame cat across batch
    let refs: Vec<&Tensor> = chunks.iter().collect();
    Tensor::cat(&refs, dim as usize).map_err(Into::into)
}

// Minimal normalizer placeholder to satisfy users
pub struct VAENormalizer;
impl VAENormalizer {
    pub fn new(_arch: eridiffusion_core::ModelArchitecture) -> Self { Self }
}

// --- YAML → VaeSpec loader (optional helper) ---
use serde::Deserialize;
use std::fs::File;
use std::io::BufReader;

/// Load a VAE spec from a modelPath.yaml-like file:
/// models:
///   <model_id>:
///     vae:
///       kind: sdxl|sd35|flux
///       path: /abs/path/to/vae.safetensors
///       latent_div: 8
///       latent_channels: 4
///       latent_scale: 0.18215
pub fn load_vae_spec_from_model_paths<P: AsRef<std::path::Path>>(
    model_paths_yaml: P,
    model_id: &str,
)
-> Result<VaeSpec> {
    let f = File::open(&model_paths_yaml)
        .with_context(|| format!("open {:?}", model_paths_yaml.as_ref()))?;
    let reader = BufReader::new(f);
    let mp: ModelPaths = serde_yaml::from_reader(reader)
        .with_context(|| format!("parse {:?}", model_paths_yaml.as_ref()))?;

    let m = mp.models.get(model_id)
        .ok_or_else(|| anyhow::anyhow!(format!("model_id '{}' not found in modelPaths", model_id)))?;

    let v = m.vae.as_ref()
        .ok_or_else(|| anyhow::anyhow!(format!("models.{}.vae missing in modelPaths", model_id)))?;

    let kind = match v.kind.as_str() {
        "sdxl" => VaeKind::Sdxl,
        "sd35" => VaeKind::Sd35,
        "flux" => VaeKind::Flux,
        other  => anyhow::bail!("unsupported vae.kind '{}'", other),
    };

    anyhow::ensure!(!v.path.trim().is_empty(), "vae.path is empty for model_id '{}'", model_id);
    anyhow::ensure!(v.latent_div > 0, "vae.latent_div must be > 0");
    anyhow::ensure!(v.latent_channels > 0, "vae.latent_channels must be > 0");
    anyhow::ensure!(v.latent_scale.is_finite() && v.latent_scale > 0.0, "vae.latent_scale must be > 0");

    Ok(VaeSpec {
        kind,
        path: v.path.clone(),
        latent_div: v.latent_div,
        latent_channels: v.latent_channels,
        latent_scale: v.latent_scale,
    })
}

#[derive(Debug, Deserialize)]
struct ModelPaths {
    models: std::collections::HashMap<String, ModelEntry>,
}

#[derive(Debug, Deserialize)]
struct ModelEntry {
    #[allow(dead_code)]
    arch: Option<String>,
    vae: Option<VaeEntry>,
}

#[derive(Debug, Deserialize)]
struct VaeEntry {
    kind: String,
    path: String,
    latent_div: usize,
    latent_channels: usize,
    latent_scale: f32,
}
