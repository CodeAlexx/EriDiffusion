use flame_core::{DType, Error, Result, Shape, Tensor, CudaDevice};
use flame_core::device::Device;
use flame_core::optimizers::{Adam, SGD};
use half::{bf16, f16};
use safetensors::{SafeTensors, tensor::Dtype as SafeDtype};
use std::{collections::HashMap, path::PathBuf};
use std::path::Path;

pub struct MemoryConfig {
    pub max_gpu_memory_gb: f32,
pub cpu_offload: bool,
pub sequential_offload: bool,
pub vae_tiling: bool,
pub vae_tile_size: usize,
pub attention_slicing: bool,
}
pub struct UnifiedInferenceConfig {
    pub model_type: ModelType,
pub memory_config: MemoryConfig,
pub device: Device,
pub dtype: DType,
pub lora_path: Option<std::path::PathBuf>,
pub lora_scale: f32,
}
pub struct UnifiedInferencePipeline {
    config: UnifiedInferenceConfig,
}
struct SDXLOffloadedPipeline {
    unet: super::sdxl_cpu_offload_inference::CPUOffloadedSDXLInference,
vae: crate::trainers::sdxl_vae_native::SDXLVAENative,
text_encoders: crate::trainers::text_encoders::TextEncoders,
lora_collection: Option<super::sdxl_lora_inference::LoRAInferenceCollection>,
config: UnifiedInferenceConfig,
}
struct SD35OffloadedPipeline {
    mmdit: super::sd35_cpu_offload_inference::CPUOffloadedSD35Inference,
vae: crate::trainers::sdxl_vae_native::SDXLVAENative, // SD3.5 uses similar VAE,
text_encoders: crate::trainers::text_encoders::TextEncoders,
config: UnifiedInferenceConfig,
}
struct FluxOffloadedPipeline {
    flux: super::flux_cpu_offload_inference::CPUOffloadedFluxInference,
vae: crate::trainers::sdxl_vae_native::SDXLVAENative, // Use SDXL VAE for now,
text_encoder: crate::trainers::text_encoders::TextEncoders,
config: UnifiedInferenceConfig,
}

// Unified inference framework for all diffusion models with CPU offloading
// Supports SDXL, SD3.5, and Flux with automatic optimization

// FLAME uses flame_core::device::Device instead of Device

/// Model type for automatic pipeline selection
#[derive(Debug, Clone)]
pub enum ModelType {
SDXL,
SD35Medium,
SD35Large,
FluxSchnell,
FluxDev}

// Extension trait for Tensor to add missing methods





trait TensorExt {
    fn sum_dim(&self, dim: usize) -> flame_core::Result<Tensor>;
    fn add_scalar(&self, scalar: f32) -> flame_core::Result<Tensor>;
    fn mul_scalar(&self, scalar: f32) -> flame_core::Result<Tensor>;
    fn square(&self) -> flame_core::Result<Tensor>;
}

impl TensorExt for Tensor {
    fn sum_dim(&self, dim: usize, device: &CudaDevice) -> flame_core::Result<Tensor> {
        // Sum along dimension
        self.sum_dim(dim)?
    }

    fn add_scalar(&self, scalar: f32, device: &CudaDevice) -> flame_core::Result<Tensor> {
        // Add scalar to all elements
        let scalar_tensor = Tensor::full(self.shape().clone(), scalar, self.device().clone())?;
        self.add(&scalar_tensor)
    }

    fn mul_scalar(&self, scalar: f32, device: &CudaDevice) -> flame_core::Result<Tensor> {
        // Multiply all elements by scalar
        let scalar_tensor = Tensor::full(self.shape().clone(), scalar, self.device().clone())?;
        self.mul(&scalar_tensor)
    }

    fn square(&self) -> flame_core::Result<Tensor> {
        // Element-wise square
        self.mul(self)
    }
}

impl ModelType {
/// Detect model type from path or config
pub fn from_path(path: &Path) -> flame_core::Result<Self> {
let path_str = path.to_string_lossy().to_lowercase();

if path_str.contains("sdxl") {
Ok(ModelType::SDXL)
} else if path_str.contains("sd3.5") || path_str.contains("sd35") {
if path_str.contains("medium") {
Ok(ModelType::SD35Medium)
} else {
Ok(ModelType::SD35Large)
}
} else if path_str.contains("flux") {
if path_str.contains("schnell") {
Ok(ModelType::FluxSchnell)
} else {
Ok(ModelType::FluxDev)
}
} else {
Err(flame_core::Error::InvalidOperation("Could not determine model type from path".into()))
}
}

/// Get recommended memory settings for 24GB GPU
pub fn memory_config(&self) -> MemoryConfig {
match self {
ModelType::SDXL => MemoryConfig {
max_gpu_memory_gb: 20.0,
cpu_offload: true,
sequential_offload: false,
vae_tiling: true,
vae_tile_size: 512,
attention_slicing: false,
},
ModelType::SD35Medium | ModelType::SD35Large => MemoryConfig {
max_gpu_memory_gb: 18.0,
cpu_offload: true,
sequential_offload: true, // SD3.5 is larger
vae_tiling: true,
vae_tile_size: 512,
attention_slicing: false,
},
ModelType::FluxSchnell | ModelType::FluxDev => MemoryConfig {
max_gpu_memory_gb: 16.0,
cpu_offload: true,
sequential_offload: true, // Flux is huge
vae_tiling: true,
vae_tile_size: 256, // Smaller tiles for Flux
attention_slicing: true,
},
}
}
}

/// Memory configuration for inference

/// Unified inference configuration

/// Unified inference pipeline that handles all models

/// Trait for model-specific inference pipelines
pub trait InferencePipeline: Send + Sync {
/// Generate image from prompt
fn generate(
&mut self,
prompt: &str,
negative_prompt: &str,
width: usize,
height: usize,
steps: usize,
guidance_scale: f32,
_seed: u64,
) -> flame_core::Result<Tensor>;

/// Get model info
fn model_info(&self, device: &CudaDevice) -> String;,
}

impl UnifiedInferencePipeline {
/// Create new unified pipeline with automatic model detection
pub fn new(
model_path: &Path,
device: Device,
dtype: DType,
lora_path: Option<std::path::PathBuf>,
) -> flame_core::Result<Self> {
let model_type = ModelType::from_path(model_path)?;
let memory_config = model_type.memory_config();

let config = UnifiedInferenceConfig {
model_type: model_type.clone(),
memory_config,
device: device,
dtype,
lora_path,;

// Create appropriate pipeline based on model type
let pipeline: Box<dyn InferencePipeline> = match model_type {
ModelType::SDXL => {
Box::new(SDXLOffloadedPipeline::new(model_path, &config)?)
}
ModelType::SD35Medium | ModelType::SD35Large => {
Box::new(SD35OffloadedPipeline::new(model_path, &config)?)
}
ModelType::FluxSchnell | ModelType::FluxDev => {
Box::new(FluxOffloadedPipeline::new(model_path, &config)?)
};

Ok(Self { config, pipeline })
}

/// Generate image with automatic optimization
pub fn generate(
&mut self,
prompt: &str,
negative_prompt: &str,
width: usize,
height: usize,
steps: usize,
guidance_scale: f32,
_seed: u64,
) -> flame_core::Result<Tensor> {
println!("Generating with {}", self.pipeline.model_info());
self.pipeline.generate(
prompt,
negative_prompt,
width,
height,
steps,
guidance_scale,
_seed,
)
}

/// SDXL pipeline with CPU offloading

impl SDXLOffloadedPipeline {
fn new(model_path: &Path, config: &UnifiedInferenceConfig, device: &CudaDevice) -> flame_core::Result<Tensor> {
println!("Loading SDXL with CPU offloading...");

// Load UNet weights
let unet_weights = load_model_weights(model_path, &config.device, config.dtype)?;

// Create offloaded UNet
let unet = super::sdxl_cpu_offload_inference::create_offloaded_pipeline(
unet_weights,
&config.device,
config.dtype,
config.memory_config.max_gpu_memory_gb,
)?;

// Load VAE
let vae_path = model_path.parent()
.ok_or_else(|| anyhow::flame_core::Error::InvalidOperation("Invalid model path".into()))?
.join("vae");

let vae_weights = if vae_path.exists() {
println!("Loading VAE from: {}", vae_path.display());
load_model_weights(&vae_path, &config.device, config.dtype)?
} else {
// Try to load VAE from the main model file
println!("Loading VAE from main model file");
let all_weights = load_model_weights(model_path, &config.device, config.dtype)?;

// Extract VAE weights
let mut vae_weights = HashMap::new();
for (name, tensor) in all_weights {
if name.starts_with("vae.") || name.starts_with("first_stage_model.") {
let vae_name = name.strip_prefix("vae.").or_else(|| name.strip_prefix("first_stage_model."))
.unwrap_or(&name);
vae_weights.insert(vae_name.to_string(), tensor);
}

if vae_weights.is_empty() {
return Err(flame_core::Error::InvalidOperation("No VAE weights found in model".into()));
}

vae_weights
};

let vae = crate::trainers::sdxl_vae_native::SDXLVAENative::new(vae_weights, config.device, config.dtype)?;

// Load text encoders
let mut text_encoders = crate::trainers::text_encoders::TextEncoders::new(config.device);

// Try to load CLIP models from standard locations
let model_dir = model_path.parent()
.ok_or_else(|| anyhow::flame_core::Error::InvalidOperation("Invalid model path".into()))?;

// Check for CLIP-L
let clip_l_paths = vec![
model_dir.join("text_encoder"),
model_dir.join("clip_l"),
std::path::PathBuf::from("/home/alex/SwarmUI/Models/text_encoder/clip_l"),
std::path::PathBuf::from("/home/alex/SwarmUI/Models/CLIP/clip_l"),
];

for path in clip_l_paths {
if path.exists() {
println!("Loading CLIP-L from: {}", path.display());
if let Err(e) = text_encoders.load_clip_l(path.to_str().unwrap_or("")) {
println!("Warning: Failed to load CLIP-L from {}: {}", path.display(), e);
} else {
break;
}
}

// Check for CLIP-G (SDXL specific)
let clip_g_paths = vec![
model_dir.join("text_encoder_2"),
model_dir.join("clip_g"),
std::path::PathBuf::from("/home/alex/SwarmUI/Models/text_encoder/clip_g"),
std::path::PathBuf::from("/home/alex/SwarmUI/Models/CLIP/clip_g"),
];

for path in clip_g_paths {
if path.exists() {
println!("Loading CLIP-G from: {}", path.display());
if let Err(e) = text_encoders.load_clip_g(path.to_str().unwrap_or("")) {
println!("Warning: Failed to load CLIP-G from {}: {}", path.display(), e);
} else {
break;
}
}

// Load LoRA if provided
let lora_collection = if let Some(lora_path) = &config.lora_path {
Some(super::sdxl_lora_inference::LoRAInferenceCollection::load(
lora_path,
&config.device,
config.dtype,
)?)
} else {
None
};

Ok(Self {
unet,
vae,
text_encoders,
lora_collection,
})
}
}

impl InferencePipeline for SDXLOffloadedPipeline {
fn generate(
&mut self,
prompt: &str,
negative_prompt: &str,
width: usize,
height: usize,
steps: usize,
guidance_scale: f32,
_seed: u64,
) -> flame_core::Result<Tensor> {
// Encode text
let (prompt_embeds, _) = self.text_encoders.encode_sdxl(prompt, 77)?;
let (neg_embeds, _) = self.text_encoders.encode_sdxl(negative_prompt, 77)?;

// Initialize latents
let latent_shape = [1, 4, height / 8, width / 8];
let mut latents = Tensor::randn(shape, device.clone())?;
latents = (latents * 14.6146)?;

// Denoising loop
let timesteps: Vec<i64> = (0..steps)
.map(|i| (1000 - (i + 1) * 1000 / steps) as i64)
.rev()
.collect();

for (idx, &t) in timesteps.iter().enumerate() {
let t_tensor = Tensor::zeros(Shape::new(vec![t]), self.config.device)?.to_dtype(self.config.dtype)?;

// Prepare for CFG
let latent_input = if guidance_scale > 1.0 {
Tensor::cat(&[&latents, &latents], 0)?
} else {
latents.clone()
};

let context = if guidance_scale > 1.0 {
Tensor::cat(&[&neg_embeds, &prompt_embeds], 0)?
} else {
prompt_embeds.clone()
};

// Forward pass
let noise_pred = self.unet.forward(
&latent_input,
&t_tensor,
&context,
self.lora_collection.as_ref().unwrap_or(&super::sdxl_lora_inference::LoRAInferenceCollection {
adapters: HashMap::new(),
rank: 16,),
)?;

// Apply guidance
let noise_pred = apply_guidance(noise_pred, guidance_scale)?;

// Update latents
latents = ddim_step(&latents, &noise_pred, t, idx, &timesteps)?;
}

// Decode with VAE
let images = if self.config.memory_config.vae_tiling {
decode_latents_tiled(&self.vae, &latents, self.config.memory_config.vae_tile_size)?
} else {
self.vae.decode(&latents)?
};

// Post-process
Ok(postprocess_images(images)?)
}

fn model_info(&self) -> String {
format!("SDXL with CPU offloading ({})",
if self.lora_collection.is_some() { "with LoRA" } else { "no LoRA" })
}

/// SD3.5 pipeline with CPU offloading

impl SD35OffloadedPipeline {
fn new(model_path: &Path, config: &UnifiedInferenceConfig, device: &CudaDevice) -> flame_core::Result<Tensor> {
println!("Loading SD3.5 with CPU offloading...");

// Load MMDiT weights
let weights = load_model_weights(model_path, &config.device, config.dtype)?;

// Create offloaded MMDiT
let model_size = match config.model_type {
ModelType::SD35Medium => "medium",
ModelType::SD35Large => "large",
_ => "large"};

let mmdit = super::sd35_cpu_offload_inference::CPUOffloadedSD35Inference::new(
weights,
&config.device,
config.dtype,
config.memory_config.max_gpu_memory_gb,
model_size,
)?;

// Load VAE (SD3.5 uses a similar VAE to SDXL with 16 channels)
// TODO: Load actual VAE weights
let vae_weights = std::collections::HashMap::new();
let vae = crate::trainers::sdxl_vae_native::SDXLVAENative::new(vae_weights, config.device, config.dtype)?;

// Load text encoders (SD3.5 uses CLIP-L, CLIP-G, and T5)
let mut text_encoders = crate::trainers::text_encoders::TextEncoders::new(config.device);

// Try to load text encoders from standard locations
let model_dir = model_path.parent()
.ok_or_else(|| anyhow::flame_core::Error::InvalidOperation("Invalid model path".into()))?;

// Load CLIP-L
let clip_l_paths = vec![
model_dir.join("text_encoders/clip_l"),
std::path::PathBuf::from("/home/alex/SwarmUI/Models/text_encoder/stable-diffusion-3/clip_l"),
];

for path in clip_l_paths {
if path.exists() {
println!("Loading SD3.5 CLIP-L from: {}", path.display());
if text_encoders.load_clip_l(path.to_str().unwrap_or("")).is_ok() {
break;
}
}

// Load CLIP-G
let clip_g_paths = vec![
model_dir.join("text_encoders/clip_g"),
std::path::PathBuf::from("/home/alex/SwarmUI/Models/text_encoder/stable-diffusion-3/clip_g"),
];

for path in clip_g_paths {
if path.exists() {
println!("Loading SD3.5 CLIP-G from: {}", path.display());
if text_encoders.load_clip_g(path.to_str().unwrap_or("")).is_ok() {
break;
}
}

// Load T5-XXL (SD3.5 specific)
let t5_paths = vec![
model_dir.join("text_encoders/t5xxl"),
std::path::PathBuf::from("/home/alex/SwarmUI/Models/text_encoder/stable-diffusion-3/t5xxl"),
];

for path in t5_paths {
if path.exists() {
println!("Loading T5-XXL from: {}", path.display());
if text_encoders.load_t5(path.to_str().unwrap_or("")).is_ok() {
break;
}
}

Ok(Self {
mmdit,
vae,
text_encoders,

impl InferencePipeline for SD35OffloadedPipeline {
fn generate(
&mut self,
prompt: &str,
negative_prompt: &str,
width: usize,
height: usize,
steps: usize,
guidance_scale: f32,
_seed: u64,
) -> flame_core::Result<Tensor> {
// Encode text with SD3.5's triple text encoders
let (prompt_embeds, pooled) = self.text_encoders.encode(prompt, 256)?;

let (neg_embeds, neg_pooled) = if negative_prompt.is_empty() {
// Use empty prompt for negative
self.text_encoders.encode("", 256)?
} else {
self.text_encoders.encode(negative_prompt, 256)?
};

// Initialize latents (16 channels for SD3.5)
let latent_shape = [1, 16, height / 8, width / 8];
let mut latents = Tensor::randn(shape, device.clone())?;

// Flow matching schedule
let sigmas = flow_matching_sigmas(steps);

for (idx, &sigma) in sigmas.iter().enumerate() {
let timestep = Tensor::zeros(Shape::new(vec![sigma]), self.config.device)?.to_dtype(self.config.dtype)?;

// Prepare for CFG
let latent_input = if guidance_scale > 1.0 {
Tensor::cat(&[&latents, &latents], 0)?
} else {
latents.clone()
};

let context = if guidance_scale > 1.0 {
Tensor::cat(&[&neg_embeds, &prompt_embeds], 0)?
} else {
prompt_embeds.clone()
};

let pooled_input = if guidance_scale > 1.0 {
Tensor::cat(&[&neg_pooled, &pooled], 0)?
} else {
pooled.clone()
};

// Forward pass
let noise_pred = self.mmdit.forward(
&latent_input,
&timestep,
&context,
&pooled_input,
)?;

// Apply guidance
let noise_pred = apply_guidance(noise_pred, guidance_scale)?;

// Update latents (flow matching)
latents = flow_matching_step(&latents, &noise_pred, sigma, idx, &sigmas)?;
}

// Decode with VAE
let images = if self.config.memory_config.vae_tiling {
// TODO: Implement VAE tiling for SD3.5
self.vae.decode(&latents)?
} else {
self.vae.decode(&latents)?
};

Ok(postprocess_images(images)?)
}

fn model_info(&self) -> String {
"SD3.5 with CPU offloading".to_string()
}

/// Flux pipeline with CPU offloading

impl FluxOffloadedPipeline {
fn new(model_path: &Path, config: &UnifiedInferenceConfig, device: &CudaDevice) -> flame_core::Result<Tensor> {
println!("Loading Flux with layer-wise CPU offloading...");

// Load weights and create offloaded Flux
let weights = load_model_weights(model_path, &config.device, config.dtype)?;
let is_schnell = matches!(config.model_type, ModelType::FluxSchnell);

let flux = super::flux_cpu_offload_inference::CPUOffloadedFluxInference::new(
weights,
&config.device,
config.dtype,
config.memory_config.max_gpu_memory_gb,
is_schnell,
)?;

// Load VAE
let vae_path = model_path.parent()
.ok_or_else(|| anyhow::flame_core::Error::InvalidOperation("Invalid model path".into()))?
.join("vae");

let vae_weights = if vae_path.exists() {
println!("Loading VAE from: {}", vae_path.display());
load_model_weights(&vae_path, &config.device, config.dtype)?
} else {
// Try to load VAE from the main model file
println!("Loading VAE from main model file");
let all_weights = load_model_weights(model_path, &config.device, config.dtype)?;

// Extract VAE weights
let mut vae_weights = HashMap::new();
for (name, tensor) in all_weights {
if name.starts_with("vae.") || name.starts_with("first_stage_model.") {
let vae_name = name.strip_prefix("vae.").or_else(|| name.strip_prefix("first_stage_model."))
.unwrap_or(&name);
vae_weights.insert(vae_name.to_string(), tensor);
}

if vae_weights.is_empty() {
return Err(flame_core::Error::InvalidOperation("No VAE weights found in model".into()));
}

vae_weights
};

let vae = crate::trainers::sdxl_vae_native::SDXLVAENative::new(vae_weights, config.device, config.dtype)?;

// Load text encoder
let mut text_encoder = crate::trainers::text_encoders::TextEncoders::new(config.device);

// Try to load text encoders for Flux
let model_dir = model_path.parent()
.ok_or_else(|| anyhow::flame_core::Error::InvalidOperation("Invalid model path".into()))?;

// Load CLIP for Flux
let clip_paths = vec![
model_dir.join("text_encoder/clip_l"),
std::path::PathBuf::from("/home/alex/SwarmUI/Models/text_encoder/flux/clip_l"),
];

for path in clip_paths {
if path.exists() {
println!("Loading Flux CLIP from: {}", path.display());
if text_encoder.load_clip_l(path.to_str().unwrap_or("")).is_ok() {
break;
}
}

// Load T5-XXL for Flux
let t5_paths = vec![
model_dir.join("text_encoder/t5xxl"),
std::path::PathBuf::from("/home/alex/SwarmUI/Models/text_encoder/flux/t5xxl"),
];

for path in t5_paths {
if path.exists() {
println!("Loading Flux T5-XXL from: {}", path.display());
if text_encoder.load_t5(path.to_str().unwrap_or("")).is_ok() {
break;
}
}

Ok(Self {
flux,
vae,
text_encoder,

impl InferencePipeline for FluxOffloadedPipeline {
fn generate(
&mut self,
prompt: &str,
negative_prompt: &str,
width: usize,
height: usize,
steps: usize,
guidance_scale: f32,
_seed: u64,
) -> flame_core::Result<Tensor> {
// Encode text with Flux's T5-XXL and CLIP
let (text_embeds, pooled) = self.text_encoder.encode(prompt, 256)?;

let (neg_embeds, _) = if negative_prompt.is_empty() {
// Use empty prompt for negative
self.text_encoder.encode("", 256)?
} else {
self.text_encoder.encode(negative_prompt, 256)?
};

// Initialize latents (patchified for Flux)
let latent_shape = [1, (height / 8) * (width / 8), 64]; // 2x2 patches of 16 channels
let mut latents = Tensor::randn(shape, device.clone())?;

// Flux uses shifted sigmoid schedule
let timesteps = flux_timesteps(steps);

for (idx, &t) in timesteps.iter().enumerate() {
let timestep = Tensor::zeros(Shape::new(vec![t]), self.config.device)?.to_dtype(self.config.dtype)?;

// Prepare inputs
let img_input = if guidance_scale > 1.0 {
Tensor::cat(&[&latents, &latents], 0)?
} else {
latents.clone()
};

let txt_input = if guidance_scale > 1.0 {
Tensor::cat(&[&neg_embeds, &text_embeds], 0)?
} else {
text_embeds.clone()
};

// Forward pass with layer-wise offloading
let noise_pred = self.flux.forward(
&img_input,
&txt_input,
&timestep,
&pooled,
Some(guidance_scale),
)?;

// Update latents
latents = flux_sampling_step(&latents, &noise_pred, t, idx, &timesteps)?;
}

// Unpatchify and decode
let latents_2d = unpatchify_flux_latents(&latents, height / 8, width / 8)?;

let images = if self.config.memory_config.vae_tiling {
// TODO: Implement VAE tiling for Flux
self.vae.decode(&latents_2d)?
} else {
self.vae.decode(&latents_2d)?
};

Ok(postprocess_images(images)?)
}

fn model_info(&self) -> String {
format!("Flux {} with layer-wise CPU offloading",
match self.config.model_type {
ModelType::FluxSchnell => "Schnell",
ModelType::FluxDev => "Dev",
_ => "Unknown"})
}

// Helper functions

fn load_model_weights(
path: &Path,
device: &Device,
dtype: DType,
) -> flame_core::Result<Tensor> {

let data = std::fs::read(path).map_err(|e| flame_core::Error::InvalidOperation(e.to_string()))?;
let tensors = SafeTensors::deserialize(&data)?;
let mut weights = HashMap::new();

for (name, tensor_view) in tensors.tensors() {
let shape: Vec<usize> = tensor_view.shape().to_vec();
let tensor = convert_safetensor(tensor_view, shape.dims(), device, dtype)?;
weights.insert(name.to_string(), tensor);
}

Ok(weights)
}

fn convert_safetensor(
tensor_view: safetensors::tensor::TensorView<'_>,
shape: &[usize],
device: &Device,
dtype: DType,
) -> flame_core::Result<Tensor> {
let data = tensor_view.data();

match tensor_view.dtype() {
SafeDtype::F16 => {
let data: Vec<half::f16> = data.chunks_exact(2)
.map(|chunk| half::f16::from_le_bytes([chunk[0], chunk[1]]))
.collect();
Ok(Tensor::from_vec(data, shape, device.cuda_device().clone())?.to_dtype(dtype)?)
}
SafeDtype::F32 => {
let data: Vec<f32> = data.chunks_exact(4)
.map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
.collect();
Ok(Tensor::from_vec(data, shape, device.cuda_device().clone())?.to_dtype(dtype)?)
}
SafeDtype::BF16 => {
let data: Vec<half::bf16> = data.chunks_exact(2)
.map(|chunk| half::bf16::from_le_bytes([chunk[0], chunk[1]]))
.collect();
Ok(Tensor::from_vec(data, shape, device.cuda_device().clone())?.to_dtype(dtype)?)
}
_ => Err(anyhow::flame_core::Error::InvalidOperation("Unsupported tensor dtype".into()))}

fn apply_guidance(noise_pred: Tensor, guidance_scale: f32) -> flame_core::Result<Tensor> {
if guidance_scale > 1.0 {
let chunks = noise_pred.chunk(2, 0)?;
let noise_pred_uncond = &chunks[0];
let noise_pred_cond = &chunks[1];
let diff = (noise_pred_cond - noise_pred_uncond)?;
Ok((noise_pred_uncond + (diff * guidance_scale as f64)?)?)
} else {
Ok(noise_pred)
}

fn ddim_step(
latents: &Tensor,
noise_pred: &Tensor,
t: i64,
idx: usize,
timesteps: &[i64],
) -> flame_core::Result<Tensor> {
let alpha_prod_t = ((1000 - t) as f32 / 1000.0).powi(2);
let alpha_prod_t_prev = if idx < timesteps.len() - 1 {
((1000 - timesteps[idx + 1]) as f32 / 1000.0).powi(2)
} else {
1.0
};

let beta_prod_t = 1.0 - alpha_prod_t;
let beta_prod_t_prev = 1.0 - alpha_prod_t_prev;

let pred_x0 = ((latents - (noise_pred * (beta_prod_t.sqrt() as f64))?)?
/ (alpha_prod_t.sqrt() as f64))?;
let dir_xt = (noise_pred * (beta_prod_t_prev.sqrt() as f64))?;
Ok(((pred_x0 * (alpha_prod_t_prev.sqrt() as f64))? + dir_xt)?)
}

fn flow_matching_sigmas(steps: usize) -> Vec<f32> {
(0..steps)
.map(|i| 1.0 - (i as f32 / steps as f32))
.collect()
}

fn flow_matching_step(
latents: &Tensor,
velocity: &Tensor,
sigma: f32,
idx: usize,
sigmas: &[f32],
) -> flame_core::Result<Tensor> {
let dt = if idx < sigmas.len() - 1 {
sigmas[idx] - sigmas[idx + 1]
} else {
sigmas[idx]
};

Ok((latents + (velocity * dt as f64)?)?)
}

fn flux_timesteps(steps: usize) -> Vec<f32> {
// Shifted sigmoid schedule
(0..steps)
.map(|i| {
let t = i as f32 / steps as f32;
1.0 / (1.0 + (-12.0 * (t -.5)).exp())
})
.collect()
}

fn flux_sampling_step(
latents: &Tensor,
pred: &Tensor,
t: f32,
idx: usize,
timesteps: &[f32],
) -> flame_core::Result<Tensor> {
// Flux uses flow matching
let velocity = pred;
let dt = if idx < timesteps.len() - 1 {
timesteps[idx + 1] - t
} else {
-t
};

Ok((latents + (velocity * dt as f64)?)?)
}

fn unpatchify_flux_latents(
latents: &Tensor,
height: usize,
width: usize,
) -> flame_core::Result<Tensor> {
// Convert from [B, H*W, 64] to [B, 16, H, W]
let (batch, _, _) = latents(.shape()[0], .shape()[1], .shape()[2])?;

// Reshape to patches
let latents = latents.reshape((batch, height, width, 2, 2, 16))?;

// Transpose to get [B, 16, H, 2, W, 2]
let latents = latents.permute((0, 5, 1, 3, 2, 4))?;

// Reshape to final size
Ok(latents.reshape((batch, 16, height * 2, width * 2))?)
}

fn decode_latents_tiled(
vae: &crate::trainers::sdxl_vae_native::SDXLVAENative,
latents: &Tensor,
tile_size: usize,
) -> flame_core::Result<Tensor> {
// TODO: Implement proper VAE tiling
// For now, just decode normally
vae.decode(latents)
}

fn postprocess_images(images: Tensor) -> flame_core::Result<Tensor> {
// Convert from [-1, 1] to [0, 255]
let images = ((images + 1.0)? * 127.5)?;
Ok(images.clamp(0.0, 255.0)?)
}

/// High-level API for easy inference
pub fn generate_image(
model_path: &Path,
prompt: &str,
negative_prompt: &str,
width: usize,
height: usize,
steps: usize,
guidance_scale: f32,
_seed: u64,
lora_path: Option<std::path::PathBuf>,
) -> flame_core::Result<Tensor> {
let device = Device::cuda(0)?;
let dtype = DType::F16; // Use F16 for memory efficiency

let mut pipeline = UnifiedInferencePipeline::new(
model_path,
device,
dtype,
lora_path,
)?;

pipeline.generate(
prompt,
negative_prompt,
width,
height,
steps,
guidance_scale,
_seed,
)
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
