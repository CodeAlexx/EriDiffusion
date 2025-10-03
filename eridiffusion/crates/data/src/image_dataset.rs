//! Simple image+caption dataset (Flame-only)

use crate::types::DataLoaderBatch;
use eridiffusion_core::{Error, Device, Result};
use flame_core::{Tensor, Shape};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::PathBuf;
use walkdir::WalkDir;
use image::GenericImageView;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetConfig {
    pub root_dir: PathBuf,
    pub caption_ext: String, // e.g., "txt"
    pub resolution: usize,   // square resize/crop
    pub center_crop: bool,
    pub random_flip: bool,
}

pub struct ImageDataset {
    cfg: DatasetConfig,
    image_paths: Vec<PathBuf>,
    captions: Vec<String>,
    device: Device,
}

impl ImageDataset {
    pub fn new(cfg: DatasetConfig, device: Device) -> Result<Self> {
        let mut image_paths = Vec::new();
        let mut captions = Vec::new();
        let rd = &cfg.root_dir;
        if !rd.exists() { return Err(Error::DataError(format!("root_dir not found: {}", rd.display()))); }
        for entry in WalkDir::new(rd).follow_links(false).min_depth(0) {
            let entry = match entry { Ok(e) => e, Err(_) => continue };
            if !entry.file_type().is_file() { continue; }
            let p = entry.path().to_path_buf();
            let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
            match ext.as_str() {
                "jpg" | "jpeg" | "png" | "webp" | "bmp" | "tiff" => {
                    let txt = p.with_extension(&cfg.caption_ext);
                    let cap = if txt.exists() { std::fs::read_to_string(&txt).unwrap_or_default().trim().to_string() } else { String::new() };
                    image_paths.push(p); captions.push(cap);
                }
                _ => {}
            }
        }
        if image_paths.is_empty() {
            let abs = rd.canonicalize().unwrap_or(rd.clone());
            return Err(Error::DataError(format!("no images found in {}", abs.display())));
        }
        Ok(Self { cfg, image_paths, captions, device })
    }

    pub fn len(&self) -> usize { self.image_paths.len() }

    pub fn get_item(&self, index: usize) -> Result<DataLoaderBatch> {
        if index >= self.len() { return Err(Error::InvalidInput("index out of bounds".into())); }
        let img = image::open(&self.image_paths[index])
            .map_err(|e| Error::DataError(format!("image open failed: {}", e)))?;

        let (proc, _crop) = self.process_image(img)?;
        let chw = self.image_to_tensor(proc)?;

        let mut meta = std::collections::HashMap::new();
        meta.insert("image_path".to_string(), Value::String(self.image_paths[index].to_string_lossy().to_string()));
        Ok(DataLoaderBatch::new(chw, vec![self.captions[index].clone()], None, None, meta))
    }

    fn process_image(&self, img: image::DynamicImage) -> Result<(image::DynamicImage, (u32,u32))> {
        use image::imageops::FilterType;
        let (w,h) = img.dimensions();
        let res = self.cfg.resolution as u32;
        let scale = if self.cfg.center_crop { (res as f32 / w.min(h) as f32).max(1.0) } else { (res as f32 / w.max(h) as f32).min(1.0) };
        let nw = (w as f32 * scale) as u32; let nh = (h as f32 * scale) as u32;
        let mut img = img.resize_exact(nw, nh, FilterType::Lanczos3);
        let crop_x = if self.cfg.center_crop { (nw.saturating_sub(res))/2 } else { 0 };
        let crop_y = if self.cfg.center_crop { (nh.saturating_sub(res))/2 } else { 0 };
        img = img.crop(crop_x, crop_y, res, res);
        if self.cfg.random_flip { if rand::random::<bool>() { img = img.fliph(); } }
        Ok((img,(crop_x,crop_y)))
    }

    fn image_to_tensor(&self, img: image::DynamicImage) -> Result<Tensor> {
        let rgb = img.to_rgb8();
        let (w,h) = rgb.dimensions();
        let data: Vec<f32> = rgb.pixels().flat_map(|p| [ (p[0] as f32/127.5)-1.0, (p[1] as f32/127.5)-1.0, (p[2] as f32/127.5)-1.0 ]).collect();
        let dev = self.device.to_flame_cuda()?;
        let chw = Tensor::from_vec(data, Shape::from_dims(&[3, h as usize, w as usize]), dev )?;
        Ok(chw)
    }
}
