//! WAN 2.2 image+caption dataset with downscale-only bucketing (skeleton)

use eridiffusion_core::{Device, Error, Result};
use eridiffusion_models::devtensor::{shape1, shape4, zeros_on};
use flame_core::DType;
use flame_core::Tensor;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DataConfig {
    pub dataset_dir: String,
    pub buckets: Vec<(usize, usize)>,
    pub min_size: usize,
    pub max_size: usize,
    pub caption_trigger_word: Option<String>,
    pub cache_latents: bool,
}

pub struct Wan22ImageDataset {
    cfg: DataConfig,
    device: Device,
    idx: usize,
    files: Vec<(PathBuf, Option<PathBuf>)>,
    by_bucket: Vec<Vec<usize>>, // indices per bucket
}

impl Wan22ImageDataset {
    pub fn new(cfg: &DataConfig, device: Device) -> Result<Self> {
        if cfg.buckets.is_empty() {
            return Err(Error::Config("wan22 dataset: at least one bucket required".into()));
        }
        let root = Path::new(&cfg.dataset_dir);
        if !root.exists() {
            return Err(Error::DataError(format!("dataset_dir not found: {}", root.display())));
        }
        // Scan files
        let mut files: Vec<(PathBuf, Option<PathBuf>)> = Vec::new();
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
            return Err(Error::DataError("no images found".into()));
        }
        // Simple initial bucket assignment by aspect
        let mut by_bucket: Vec<Vec<usize>> = vec![Vec::new(); cfg.buckets.len()];
        for (i, (img_path, _)) in files.iter().enumerate() {
            if let Ok(img) = image::open(img_path) {
                let thumbnail = image::imageops::thumbnail(&img, 64, 64);
                let (w, h) = (thumbnail.width() as f32, thumbnail.height() as f32);
                let ar = w / h;
                let mut best = 0usize;
                let mut best_diff = f32::MAX;
                for (bi, &(bh, bw)) in cfg.buckets.iter().enumerate() {
                    let bar = (bw as f32) / (bh as f32);
                    let diff = (ar - bar).abs();
                    if diff < best_diff {
                        best_diff = diff;
                        best = bi;
                    }
                }
                by_bucket[best].push(i);
            }
        }
        Ok(Self { cfg: cfg.clone(), device, idx: 0, files, by_bucket })
    }

    pub fn next_batch(&mut self) -> Result<Batch> {
        // Pick the first non-empty bucket
        let mut pick = None;
        for (bi, v) in self.by_bucket.iter().enumerate() {
            if !v.is_empty() {
                pick = Some(bi);
                break;
            }
        }
        let bi = pick.unwrap_or(0);
        let (bh, bw) = self.cfg.buckets[bi];
        let mut indices = self.by_bucket[bi].clone();
        if indices.is_empty() {
            indices.push(self.idx.min(self.files.len() - 1));
        }
        let b = indices.len().min(1); // default batch size 1 for now
                                      // Build latents zeros [B,4,H/8,W/8]
        let latents = zeros_on(
            shape4(b as i64, 4, (bh / 8) as i64, (bw / 8) as i64),
            &self.device,
            DType::BF16,
        )?;
        let timesteps = zeros_on(shape1(b as i64), &self.device, DType::BF16)?;
        let target = zeros_on(latents.shape().clone(), &self.device, DType::BF16)?;
        // Captions/prompts
        let mut prompts = Vec::with_capacity(b);
        for i in 0..b {
            let (_img, cap) = &self.files[indices[i]];
            let mut text = cap
                .as_ref()
                .and_then(|p| std::fs::read_to_string(p).ok())
                .unwrap_or_else(|| "".into());
            if let Some(trig) = &self.cfg.caption_trigger_word {
                text = text.replace("[trigger]", trig);
            }
            if text.trim().is_empty() {
                text = "a photo".into();
            }
            prompts.push(text);
        }
        Ok(Batch { latents, timesteps, target, prompts })
    }

    pub fn num_files(&self) -> usize {
        self.files.len()
    }

    pub fn buckets_summary(&self) -> Vec<((usize, usize), usize)> {
        self.cfg.buckets.iter().enumerate().map(|(i, &hw)| (hw, self.by_bucket[i].len())).collect()
    }
}

pub struct Batch {
    pub latents: Tensor,
    pub timesteps: Tensor,
    pub target: Tensor,
    pub prompts: Vec<String>,
}
