use flame_core::{DType, Result, Shape, Tensor};
use flame_core::device::Device;
use flame_core::optimizers::{Adam, SGD};

// FLAME uses flame_core::device::Device instead of Device

/// Save tensor as image file
pub fn save_tensor_image(tensor: &Tensor, path: &Path) -> flame_core::Result<()> {
// Ensure tensor is on CPU and in correct format
let tensor = tensor;

// Get dimensions
let dims = tensor.shape();
let (c, h, w) = match dims.rank() {
3 => (dims[0], dims[1], dims[2]),
4 => {
if dims[0] != 1 {
return Err(Error::Msg("Batch size must be 1 for image saving");
}
(dims[1], dims[2], dims[3])
}
_ => return Err(Error::Msg("Invalid tensor shape for image".into()))};

if c != 3 {
return Err(Error::Msg("Image must have 3 channels (RGB)");
}

// Convert to u8 if needed
let tensor = if tensor.dtype() != DType::U8 {
tensor.to_dtype(DType::U8)?
} else {
tensor.clone()
};

// Permute from CHW to HWC
let tensor = tensor.permute((1, 2, 0))?;

// Get raw data
let raw_pixels = tensor.flatten_all()?.to_vec1::<u8>()?;

// Create image buffer
let img = image::ImageBuffer::<image::Rgb<u8>, Vec<u8>>::from_raw(
// Removed broken import: w as u32,
h as u32,
raw_pixels,
).ok_or_else(|| Error::Msg("Failed to create image buffer".into()))?;

// Save based on extension
let ext = path.extension()
.and_then(|e| e.to_str())
.unwrap_or("jpg");

match ext.to_lowercase().as_str() {
"png" => {
img.save_with_format(path, image::ImageFormat::Png)
.map_err(|e| Error::Msg(format!("Failed to save PNG: {}", e)))?;
}
_ => {
// Default to JPEG
let rgb_img = image::DynamicImage::ImageRgb8(img);
rgb_img.save_with_format(path, image::ImageFormat::Jpeg)
.map_err(|e| Error::Msg(format!("Failed to save JPEG: {}", e)))?;
}

Ok(())
}

/// Create output directory structure
pub fn ensure_output_dir(base_dir: &Path, lora_name: &str) -> flame_core::Result<Tensor> {
let output_dir = base_dir.join(lora_name).join("samples");
std::fs::create_dir_all(&output_dir)
.map_err(|e| Error::Msg(format!("Failed to create output directory: {}", e)))?;
Ok(output_dir)
}

/// Generate timestep schedule for inference
pub fn get_timestep_schedule(num_steps: usize, scheduler_type: &str) -> Vec<usize> {
match scheduler_type {
"ddim" => {
// Standard DDIM schedule
let step_ratio = 1000 / num_steps;
(0..num_steps)
.map(|s| s * step_ratio)
.rev()
.collect()
}
"euler" => {
// Euler schedule
(0..num_steps).rev().collect()
}
_ => {
// Default to DDIM
let step_ratio = 1000 / num_steps;
(0..num_steps)
.map(|s| s * step_ratio)
.rev()
.collect()
}
}
}
}
