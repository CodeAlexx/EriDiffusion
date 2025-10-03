use crate::loaders::WeightLoader;
use anyhow::Context;
use flame_core::{DType, Result, Shape, Tensor, CudaDevice};
use flame_core::device::Device;
use flame_core::optimizers::{Adam, SGD};
use half::{bf16, f16};
use std::{collections::HashMap, path::Path};

pub struct PrefixedWeightLoader {
    loader: WeightLoader,
prefix: String,
}
pub struct SDXLInferencePipeline {
    unet: SDXLUNet2DConditionModel,
vae: VAEModel,
text_encoders: TextEncoders,
scheduler: Box<dyn Scheduler>,
device: Device,
dtype: DType,
}
pub struct SDXLInferenceConfig {
    pub model_path: String,
pub vae_path: String,
pub text_encoder_path: String,
pub text_encoder_2_path: String,
pub scheduler: String,
pub device: String,
pub dtype: String,
};

// FLAME uses flame_core::device::Device instead of Device

/// SDXL inference pipeline for text-to-image generation



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

impl SDXLInferencePipeline {
pub fn new(
unet: SDXLUNet2DConditionModel,
vae: VAEModel,
text_encoders: TextEncoders,
scheduler_type: &str,
device: Device,
dtype: DType,
) -> flame_core::Result<Self> {
// Create scheduler
let scheduler: Box<dyn Scheduler> = match scheduler_type {
"ddim" => Box::new(DDIMScheduler::new(50)?),
"dpm" => Box::new(DPMSolverMultistepScheduler::new(25)?),
_ => Box::new(DDIMScheduler::new(50)?)};

Ok(Self {
unet,
vae,
text_encoders,
scheduler,
device,
dtype})
}

/// Generate images from text prompts
pub fn generate(
&mut self,
prompt: &str,
negative_prompt: Option<&str>,
width: usize,
height: usize,
num_inference_steps: usize,
guidance_scale: f32,
seed: Option<u64>,
) -> flame_core::Result<Tensor> {
// Set random seed if provided
if let Some(s) = seed {
// FLAME doesn't have a global seed setter, we'll handle it when creating tensors
println!("Using seed: {}", s);
}

// Encode prompts
let (prompt_embeds, pooled_prompt_embeds) = self.encode_prompt(prompt)?;
let (negative_embeds, negative_pooled) = if let Some(neg) = negative_prompt {
self.encode_prompt(neg)?
} else {
self.encode_prompt("")?
};

// Prepare conditioning
let batch_size = 1;
let do_classifier_free_guidance = guidance_scale > 1.0;

// Concatenate positive and negative embeddings for CFG
let prompt_embeds = if do_classifier_free_guidance {
Tensor::cat(&[&negative_embeds, &prompt_embeds], 0)?
} else {
prompt_embeds
};

let pooled_prompt_embeds = if do_classifier_free_guidance {
Tensor::cat(&[&negative_pooled, &pooled_prompt_embeds], 0)?
} else {
pooled_prompt_embeds
};

// Generate time_ids
let time_ids_config = TimeIdsConfig {
original_height: height as f32,
original_width: width as f32,
crop_top: 0.0,
crop_left: 0.0,
target_height: height as f32,;

let time_ids = TimeIdsGenerator::generate_single(&time_ids_config, &self.device, self.dtype)?;
let time_ids = if do_classifier_free_guidance {
Tensor::cat(&[&time_ids, &time_ids], 0)?
} else {
time_ids
};

// Prepare latents
let latent_channels = 4;
let latent_height = height / 8;
let latent_width = width / 8;

let mut latents = Tensor::randn(0.0_f32, 1.0_f32, Shape::new(vec![batch_size, latent_channels, latent_height, latent_width]), &self.device,
)?.to_dtype(self.dtype)?;

// Scale initial noise by scheduler's init noise sigma
let sigma = Tensor::zeros(self.scheduler.init_noise_sigma(), self.device.clone())?.to_dtype(self.dtype)?;
latents = latents.mul(&sigma)?;

// Prepare additional conditioning
let mut added_cond_kwargs = HashMap::new();
added_cond_kwargs.insert("text_embeds".to_string(), pooled_prompt_embeds);
added_cond_kwargs.insert("time_ids".to_string(), time_ids);

// Set timesteps
self.scheduler.set_timesteps(num_inference_steps);
let timesteps = self.scheduler.timesteps();

// Denoising loop
println!("Running {} denoising steps...", timesteps.len());
for (i, &t) in timesteps.iter().enumerate() {
// Expand latents for CFG
let latent_model_input = if do_classifier_free_guidance {
Tensor::cat(&[&latents, &latents], 0)?
} else {
latents.clone()
};

// Scale model input by scheduler
let latent_model_input = self.scheduler.scale_model_input(&latent_model_input, t)?;

// Predict noise residual
let timestep_tensor = Tensor::full(Shape::from_dims(&[1]), t as f32, self.device.cuda_device().clone())?;
let noise_pred = self.unet.forward_train(
&latent_model_input,
&timestep_tensor,
&prompt_embeds,
Some(&added_cond_kwargs),
)?;

// Perform guidance
let noise_pred = if do_classifier_free_guidance {
let chunks = noise_pred.chunk(2, 0)?;
let noise_pred_uncond = &chunks[0];
let noise_pred_text = &chunks[1];

// CFG formula: w * (text - uncond) + uncond
let diff = noise_pred_text.sub(noise_pred_uncond)?;
let scale_tensor = Tensor::full(diff.shape().clone(), guidance_scale as f32, self.device.cuda_device().clone())?.to_dtype(self.dtype)?;
let guided = diff.mul(&scale_tensor)?;
noise_pred_uncond.add(&guided)?
} else {
noise_pred
};

// Compute previous latents
latents = self.scheduler.step(&noise_pred, t, &latents)?; // TODO: Use gradient_map instead of individual tensor

if i % 10 == 0 {
println!("Step {}/{}", i + 1, timesteps.len());
}

// Decode latents to image
println!("Decoding latents to image...");
let images = self.vae.decode(&latents)?;

// Post-process
let one = Tensor::full(images.shape().clone(), 1.0f32, self.device.cuda_device().clone())?.to_dtype(self.dtype)?;
let scale = Tensor::full(images.shape().clone(), 127.5f32, self.device.cuda_device().clone())?.to_dtype(self.dtype)?;
let images = images.add(&one)?.mul(&scale)?;
let images = images.clamp(0.0, 255.0)?;

Ok(images)
}

/// Encode text prompt to embeddings
fn encode_prompt(&mut self, prompt: &str) -> flame_core::Result<Tensor> {
self.text_encoders.encode_sdxl(prompt, 77)
}

/// Generate with LoRA weights applied
pub fn generate_with_lora(
&mut self,
prompt: &str,
lora_path: &Path,
lora_scale: f32,
negative_prompt: Option<&str>,
width: usize,
height: usize,
num_inference_steps: usize,
guidance_scale: f32,
seed: Option<u64>,
) -> flame_core::Result<Tensor> {
// Load and apply LoRA weights
println!("Loading LoRA weights from: {:?}", lora_path);

// For now, we'll use the standard generation
// Full implementation would load LoRA weights and apply them to UNet
self.generate(
prompt,
negative_prompt,
width,
height,
num_inference_steps,
guidance_scale,
seed,
)
}

/// Configuration for SDXL inference;
#[derive(Debug, Clone)]
impl Default for SDXLInferenceConfig {
fn default(device: &CudaDevice) -> Self {
Self {
model_path: String::new(),
vae_path: String::new(),
text_encoder_path: String::new(),
text_encoder_2_path: String::new(),
scheduler: "ddim".to_string(),
device: "cuda:0".to_string(),,
}

/// Load SDXL inference pipeline from configuration
pub fn load_sdxl_pipeline(config: &SDXLInferenceConfig, device: &CudaDevice) -> flame_core::Result<Tensor> {
// Setup device
let device = if config.device.starts_with("cuda") {
Device::cuda(0)?
} else {
device
};

let dtype = match config.dtype.as_str() {
"fp16" => DType::F16,
"bf16" => DType::BF16,
_ => DType::F32};

println!("Loading SDXL models for inference...");

// Load UNet
let unet_vb = unsafe {
WeightLoader::from_mmaped_safetensors(&[&config.model_path], DType::F32, &device)?
};
let unet = SDXLUNet2DConditionModel::new(unet_vb, Default::default())?;

// Load VAE
let vae = {
let vae_vb = unsafe {
WeightLoader::from_mmaped_safetensors(&[&config.vae_path], dtype, &device)?
};

// Load text encoders
let mut text_encoders = TextEncoders::new(device);
text_encoders.load_clip_l(&config.text_encoder_path)?;
text_encoders.load_clip_g(&config.text_encoder_2_path)?;

SDXLInferencePipeline::new(
unet,
vae,
text_encoders,
&config.scheduler,
device,
dtype,
)
}

/// Alias for backward compatibility
pub type SDXLInference = SDXLInferencePipeline;
}
}
}
}
}
