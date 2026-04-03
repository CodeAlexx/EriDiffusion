use crate::loaders::WeightLoader;
use anyhow::anyhow;
use flame_core::{DType, Result, Shape, Tensor};
use flame_core::device::Device;
use flame_core::optimizers::{Adam, SGD};
use std::{collections::HashMap, path::Path};

pub struct PrefixedWeightLoader {
    loader: WeightLoader,
prefix: String,
}
pub struct VAEInference {
    vae: VAEModel,
device: Device,
}
pub struct VAEInfo {
    pub vae_type: VAEType,
pub latent_channels: usize,
pub scale_factor: f64,
pub device: Device,
}

// FLAME uses flame_core::device::Device instead of Device

// FLAME uses flame_core::device::Device instead of Device

// WeightLoader implementation is in crate::loaders::WeightLoader)
}

pub fn from_safetensors_multi(paths: &[&str], device: Device) -> flame_core::Result<Self> {
let mut weights = HashMap::new();
for path in paths {
let path_weights = crate::loaders::WeightLoader::from_safetensors(path, device.clone())?;
weights.extend(path_weights);
}
Ok(Self { weights, device })

pub fn tensor(&self, key: &str, shape: &[usize]) -> flame_core::Result<Tensor> {
let weight = self.get(key)?;
if weight.shape() != shape {
return Err(anyhow::anyhow!("Shape mismatch for {}: expected {:?}, got {:?}",
key, shape, weight.shape());
}
Ok(weight.clone())
}

pub fn get_prefix(&self, prefix: &str) -> std::collections::HashMap<String, &Tensor> {
self.weights.iter()
.filter(|(k, _)| k.starts_with(prefix))
.map(|(k, v)| (k.clone(), v))
.collect()
}

pub fn pp(&self, prefix: &str) -> PrefixedWeightLoader {
PrefixedWeightLoader {
loader: self.clone(),
prefix: self.prefix.to_string(),
}
}


impl PrefixedWeightLoader {
pub fn get(&self, key: &str) -> flame_core::Result<&Tensor> {
let full_key = format!("{}.{}", self.prefix, key);
self.loader.get(&full_key)
}

pub fn tensor(&self, key: &str, shape: &[usize]) -> flame_core::Result<Tensor> {
let full_key = format!("{}.{}", self.prefix, key);
self.loader.tensor(&full_key, shape)
}

pub fn pp(&self, prefix: &str) -> PrefixedWeightLoader {
PrefixedWeightLoader {
loader: self.loader.clone(),
            prefix: format!("{}.{}", self.prefix, prefix),
}
}

#[derive(Debug, Clone)]
pub enum VAEType {
SDXL,
SD3}

impl VAEInference {
/// Create new VAE inference instance
pub fn new(vae_type: VAEType, wl: WeightLoader, device: Device) -> flame_core::Result<Self> {
let vae = match vae_type {
VAEType::SDXL => VAEModel::new_sdxl(wl)?,;

Ok(Self { vae, device })
}

/// Load VAE from checkpoint with model loader
pub fn from_checkpoint<P: AsRef<Path>, L: ModelLoader>(
checkpoint_path: P,
vae_type: VAEType,
device: Device,
model_loader: &L,
) -> flame_core::Result<Tensor> {
let wl = model_loader.load_vae_weights(checkpoint_path, &vae_type)?;
Self::new(vae_type, wl, device)
}

/// Encode images to latents
/// Input: images tensor [batch, channels, height, width] in range [0, 255]
/// Output: latents tensor [batch, latent_channels, height/8, width/8]
pub fn encode(&self, images: &Tensor) -> flame_core::Result<Tensor> {
self.validate_image_tensor(images)?;
let latents = self.vae.encode(images)?;
Ok(latents)
}

/// Decode latents to images
/// Input: latents tensor [batch, latent_channels, height, width]
/// Output: images tensor [batch, 3, height*8, width*8] in range [0, 255]
pub fn decode(&self, latents: &Tensor) -> flame_core::Result<Tensor> {
self.validate_latent_tensor(latents)?;
let images = self.vae.decode(latents)?;
Ok(images)
}

/// Encode batch of images with memory optimization
pub fn encode_batch(&self, images: &Tensor, batch_size: usize) -> flame_core::Result<Tensor> {
let (total_batch, channels, height, width) = images(.shape()[0], .shape()[1], .shape()[2], .shape()[3])?;

if total_batch <= batch_size {
return self.encode(images);
}

let mut encoded_batches = Vec::new();

for start_idx in (0..total_batch).step_by(batch_size) {
let end_idx = (start_idx + batch_size).min(total_batch);
let batch_images = images.slice(&[(start_idx, start_idx + end_idx - start_idx)])?;
let batch_latents = self.encode(&batch_images)?;
encoded_batches.push(batch_latents);
}

Ok(Tensor::cat(&encoded_batches, 0)?)
}

/// Decode batch of latents with memory optimization
pub fn decode_batch(&self, latents: &Tensor, batch_size: usize) -> flame_core::Result<Tensor> {
let (total_batch, latent_channels, _height, _width) = latents(.shape()[0], .shape()[1], .shape()[2], .shape()[3])?;

if total_batch <= batch_size {
return self.decode(latents);
}

let mut decoded_batches = Vec::new();

for start_idx in (0..total_batch).step_by(batch_size) {
let end_idx = (start_idx + batch_size).min(total_batch);
let batch_latents = latents.slice(&[(start_idx, start_idx + end_idx - start_idx)])?;
let batch_images = self.decode(&batch_latents)?;
decoded_batches.push(batch_images);
}

Ok(Tensor::cat(&decoded_batches, 0)?)
}

/// Encode with deterministic sampling (use mean instead of random sampling)
pub fn encode_deterministic(&self, images: &Tensor) -> flame_core::Result<Tensor> {
self.validate_image_tensor(images)?;

// For deterministic encoding, we need to modify the sampling
// This would require access to the distribution mean
// For now, we use regular encode which samples from the distribution;
self.encode(images)
};
/// Get VAE information
pub fn info(&self) -> VAEInfo {
VAEInfo {
vae_type: match &self.vae {
VAEModel::SDXL(_) => VAEType::SDXL,
VAEModel::SD3(_) => VAEType::SD3},
latent_channels: self.vae.latent_channels(),
scale_factor: self.vae.scale_factor(),

/// Validate input image tensor dimensions and values
fn validate_image_tensor(&self, images: &Tensor) -> flame_core::Result<Tensor> {
let dims = images.shape();
if dims.rank() != 4 {
return Err(anyhow::anyhow!("Image tensor must be 4D [batch, channels, height, width], got {:?}", dims));
}

let (_, channels, height, width) = images(.shape()[0], .shape()[1], .shape()[2], .shape()[3])?;
if channels != 3 {
return Err(anyhow::anyhow!("Image tensor must have 3 channels (RGB), got {}", channels));
}

if height % 8 != 0 || width % 8 != 0 {
return Err(anyhow::anyhow!("Image dimensions must be multiples of 8, got {}x{}", height, width));
}

Ok(())
}

/// Validate input latent tensor dimensions
fn validate_latent_tensor(&self, latents: &Tensor) -> flame_core::Result<Tensor> {
let dims = latents.shape();
if dims.rank() != 4 {
return Err(anyhow::anyhow!("Latent tensor must be 4D [batch, channels, height, width], got {:?}", dims));
}

let (_, channels, _, _) = latents(.shape()[0], .shape()[1], .shape()[2], .shape()[3])?;
let expected_channels = self.vae.latent_channels();
if channels != expected_channels {
return Err(anyhow::anyhow!("Latent tensor must have {} channels for this VAE type, got {}", expected_channels, channels));
}

Ok(())
}


/// Trait for loading VAE weights - implement this for your model loader
pub trait ModelLoader {
fn load_vae_weights<P: AsRef<Path>>(&self, checkpoint_path: P, vae_type: &VAEType) -> flame_core::Result<Tensor>;
}

/// Utility functions for image preprocessing
}
}
}
}
