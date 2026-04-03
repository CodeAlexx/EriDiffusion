//! WAN 2.2 video dataset (images-only bootstrap): builds synthetic frame sequences from images.

use eridiffusion_core::{Device, Error, Result};
use eridiffusion_models::devtensor::{shape1, tensor_from_vec_on, zeros_on, BF16};
use flame_core::{Shape, Tensor};
use image::GenericImageView;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VideoDataConfig {
    pub dataset_dir: String,
    pub buckets: Vec<(usize, usize)>, // spatial buckets
    pub time_buckets: Vec<usize>,     // allowed frame counts (e.g., [49, 73])
    pub frames: usize,                // target frames if time_buckets empty
    pub downscale_only: bool,         // never upscale images
    pub caption_trigger_word: Option<String>,
}

pub struct Wan22VideoDataset {
    cfg: VideoDataConfig,
    device: Device,
    files: Vec<(PathBuf, Option<PathBuf>)>, // (image, caption)
}

#[derive(Clone, Debug)]
pub struct VideoBatch {
    pub latents: Tensor,   // [B, T, H/8, W/8, 4] BF16/F32 zeros (bootstrap)
    pub timesteps: Tensor, // [B]
    pub prompts: Vec<String>,
    pub h: usize,
    pub w: usize,
    pub t: usize,
    pub images: Tensor, // [B, H, W, 3] BF16 normalized [-1,1]
}

impl Wan22VideoDataset {
    pub fn new(cfg: &VideoDataConfig, device: Device) -> Result<Self> {
        if cfg.buckets.is_empty() {
            return Err(Error::Config("video dataset: buckets required".into()));
        }
        let root = Path::new(&cfg.dataset_dir);
        if !root.exists() {
            return Err(Error::DataError(format!("dataset_dir not found: {}", root.display())));
        }
        // Collect images + same-name captions
        let mut files = Vec::new();
        for e in WalkDir::new(root).min_depth(1).max_depth(6) {
            let e = match e {
                Ok(v) => v,
                Err(_) => continue,
            };
            if !e.file_type().is_file() {
                continue;
            }
            let p = e.path();
            let ext = p.extension().and_then(|s| s.to_str()).unwrap_or("").to_ascii_lowercase();
            if matches!(ext.as_str(), "jpg" | "jpeg" | "png" | "webp" | "bmp" | "tiff") {
                let cap = p.with_extension("txt");
                files.push((p.to_path_buf(), if cap.exists() { Some(cap) } else { None }));
            }
        }
        if files.is_empty() {
            return Err(Error::DataError("no images found for video dataset".into()));
        }
        Ok(Self { cfg: cfg.clone(), device, files })
    }

    pub fn num_files(&self) -> usize {
        self.files.len()
    }

    /// Summarize how many files would land in each (H,W) bucket (downscale-only policy assumed).
    pub fn buckets_summary(&self) -> Vec<(usize, usize, usize)> {
        let mut counts: std::collections::BTreeMap<(usize, usize), usize> =
            std::collections::BTreeMap::new();
        for (img_path, _) in &self.files {
            if let Ok(img) = image::open(img_path) {
                let (w0, h0) = img.dimensions();
                let (bh, bw, _t) = self.pick_bucket(w0, h0);
                *counts.entry((bh, bw)).or_insert(0) += 1;
            }
        }
        let mut out: Vec<(usize, usize, usize)> =
            counts.into_iter().map(|((h, w), c)| (h, w, c)).collect();
        out.sort_by_key(|(h, w, _)| (*h, *w));
        out
    }

    /// Pick a spatial bucket closest in aspect; pick time from config
    fn pick_bucket(&self, img_w: u32, img_h: u32) -> (usize, usize, usize) {
        let ar = (img_w as f32) / (img_h as f32);
        let mut best = 0usize;
        let mut best_diff = f32::MAX;
        for (i, &(bh, bw)) in self.cfg.buckets.iter().enumerate() {
            let bar = (bw as f32) / (bh as f32);
            let d = (ar - bar).abs();
            if d < best_diff {
                best_diff = d;
                best = i;
            }
        }
        let (bh, bw) = self.cfg.buckets[best];
        let t = if !self.cfg.time_buckets.is_empty() {
            self.cfg.time_buckets[0]
        } else {
            self.cfg.frames.max(1)
        };
        (bh, bw, t)
    }

    /// Next batch as synthetic sequences (B=1 for bootstrap)
    pub fn next_batch(&mut self, index: usize) -> Result<VideoBatch> {
        if self.files.is_empty() {
            return Err(Error::DataError("empty dataset".into()));
        }
        let idx = index % self.files.len();
        let (img_path, cap_path_opt) = &self.files[idx];
        let img = image::open(img_path)
            .map_err(|e| Error::DataError(format!("image open failed: {}", e)))?;
        let (w0, h0) = img.dimensions();
        let (bh, bw, t) = self.pick_bucket(w0, h0);
        // Downscale-only guard
        if self.cfg.downscale_only && (w0 < bw as u32 || h0 < bh as u32) {
            return Err(Error::DataError(format!(
                "upscaling forbidden: {} ({}x{}) -> bucket {}x{}",
                img_path.display(),
                w0,
                h0,
                bw,
                bh
            )));
        }
        // Produce zeros latents [B=1, T, H/8, W/8, C=4]
        let lh = (bh / 8).max(1);
        let lw = (bw / 8).max(1);
        let latents =
            zeros_on(flame_core::Shape::from_dims(&[1usize, t, lh, lw, 4]), &self.device, BF16)?;
        let timesteps = zeros_on(shape1(1), &self.device, BF16)?;
        // Prepare image tensor [1,H,W,3] BF16 normalized to [-1,1]
        use image::imageops::FilterType;
        let resized = img.resize_exact(bw as u32, bh as u32, FilterType::Lanczos3);
        let rgb = resized.to_rgb8();
        let (rw, rh) = rgb.dimensions();
        let mut data: Vec<f32> = Vec::with_capacity((rw * rh * 3) as usize);
        for p in rgb.pixels() {
            data.push((p[0] as f32 / 127.5) - 1.0);
            data.push((p[1] as f32 / 127.5) - 1.0);
            data.push((p[2] as f32 / 127.5) - 1.0);
        }
        let images = tensor_from_vec_on(
            data,
            Shape::from_dims(&[1usize, rh as usize, rw as usize, 3usize]),
            &self.device,
            BF16,
        )?;
        // Prompt
        let mut text = cap_path_opt
            .as_ref()
            .and_then(|p| std::fs::read_to_string(p).ok())
            .unwrap_or_else(|| "".into());
        if let Some(trig) = &self.cfg.caption_trigger_word {
            text = text.replace("[trigger]", trig);
        }
        if text.trim().is_empty() {
            text = "a short video".into();
        }
        Ok(VideoBatch { latents, timesteps, prompts: vec![text], h: bh, w: bw, t, images })
    }
}
