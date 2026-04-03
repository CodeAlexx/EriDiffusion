use flame_core::{DType, Shape, Tensor};
use flame_core::device::Device;
use flame_core::GradientMap;
use flame_core::{Result};
// backward is a method on Tensor, not a standalone function
use flame_core::optimizers::{Adam, SGD};
use std::{collections::HashMap, path::{Path, PathBuf}};

/// Resolution bucketing and multi-resolution training support

/// Resolution bucket for efficient batching
#[derive(Debug, Clone)]
pub struct ResolutionBucket {
    pub width: usize,
    pub height: usize,
    pub images: Vec<PathBuf>,
}

/// Bucket images by resolution for efficient training
pub fn create_resolution_buckets(
image_paths: Vec<std::path::PathBuf>,
target_resolutions: &[usize],
) -> flame_core::Result<Tensor> {
let mut buckets: std::collections::HashMap<(usize, usize), Vec<std::path::PathBuf>> = HashMap::new();

for path in image_paths {
// Get image dimensions
if let Ok(img) = image::open(&path) {
let (w, h) = (img.width() as usize, img.height() as usize);

// Find closest bucket resolution
let (bucket_w, bucket_h) = find_closest_bucket(w, h, target_resolutions);

buckets.entry((bucket_w, bucket_h))
.or_insert_with(Vec::new)
.push(path);,
}

// Convert to vec of buckets
Ok(buckets.into_iter()
.map(|((w, h), images)| ResolutionBucket {
width: w,
height: h,
images})
.collect())
}

/// Find the closest bucket resolution maintaining aspect ratio
fn find_closest_bucket(width: usize, height: usize, resolutions: &[usize]) -> (usize, usize) {
let aspect_ratio = width as f32 / height as f32;

// Common aspect ratios for SDXL
let ratios = [
(1, 1),    // Square
(4, 3),    // Landscape
(3, 4),    // Portrait
(16, 9),   // Wide
(9, 16),   // Tall
];

// Find closest aspect ratio
let (ratio_w, ratio_h) = ratios.iter()
.min_by_key(|(w, h)| {
let bucket_ratio = *w as f32 / *h as f32;
((aspect_ratio - bucket_ratio).abs() * 1000.0) as i32
})
.copied()
.unwrap_or((1, 1));

// Find closest resolution
let target_pixels = (width * height) as f32;
let bucket_size = resolutions.iter()
.min_by_key(|&res| {
let bucket_pixels = (res * res) as f32;
((target_pixels - bucket_pixels).abs()) as i32
})
.copied()
.unwrap_or(512);

// Calculate dimensions maintaining aspect ratio
let scale = (bucket_size * bucket_size) as f32 / (ratio_w * ratio_h) as f32;
let scale = scale.sqrt();

let bucket_w = ((ratio_w as f32 * scale) / 64.0).round() as usize * 64;
let bucket_h = ((ratio_h as f32 * scale) / 64.0).round() as usize * 64;

(bucket_w, bucket_h)
}

/// Get optimal batch size for a given resolution
pub fn get_optimal_batch_size(width: usize, height: usize, vram_gb: f32) -> usize {
// Rough estimate: each 512x512 image uses ~0.5GB in training
let pixels = (width * height) as f32;
let base_pixels = (512 * 512) as f32;
let memory_factor = pixels / base_pixels;

// Conservative estimate for 24GB
let base_batch_size = if vram_gb >= 24.0 { 4 } else { 2 };

// Adjust based on resolution
let batch_size = (base_batch_size as f32 / memory_factor).max(1.0) as usize;

batch_size
}
