use flame_core::device::Device;
use flame_core::{DType, Result, Shape, Tensor};
use std::{collections::HashMap, path::Path};

pub mod flux;
pub mod sd35;
pub mod sdxl;
pub use flux::generate_flux_image;
pub mod flame_inference;
pub mod flux_sampling;
pub mod mmdit_streaming;
pub mod numerical_accuracy;
#[cfg(feature = "experimental")]
pub mod qwen_image;
pub mod sd35_sampling;
pub mod sd35_simple;
pub mod sdxl_sampling;
pub mod klein_sampling;
pub mod zimage_sampling;
pub mod ltx2_sampling;
pub mod unified_sampling; // Experimental placeholder for Qwen Image inference

// FLAME modules - DISABLED: Consolidating with non-flame versions
// pub mod flame_utils;
// pub mod flame_model_loader;
// pub mod flame_weight_converter;
// pub mod flame_inference_pipeline;
// pub mod flame_sampler;
// pub mod flame_scheduler;

// Re-export key types
pub use flame_inference::{
    CLIPTextEncoder, FlowMatchScheduler, FluxPipeline, FluxScheduler, SD35Pipeline, T5TextEncoder,
    TextEncoderType, TextEncoders,
};

/// Trait for unified diffusion model inference
pub trait DiffusionInference {
    /// Load model weights and initialize
    fn load_model(&mut self, config: &ModelConfig) -> flame_core::Result<()>;

    /// Encode text prompt to embeddings
    fn encode_prompt(&mut self, prompt: &str) -> flame_core::Result<Tensor>;

    /// Run denoising process
    fn denoise(
        &mut self,
        latents: &Tensor,
        text_embeds: &Tensor,
        steps: usize,
        cfg_scale: f64,
    ) -> flame_core::Result<Tensor>;

    /// Decode latents to image using VAE
    fn decode_vae(&self, latents: &Tensor) -> flame_core::Result<Tensor>;

    /// Apply LoRA weights to the model
    fn apply_lora(
        &mut self,
        lora_weights: &std::collections::HashMap<String, Tensor>,
        scale: f32,
    ) -> flame_core::Result<()>;
}

/// Common configuration for model inference
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub unet_path: String,
    pub vae_path: String,
    pub clip_path: String,
    pub clip2_path: Option<String>,
    pub t5_path: Option<String>,
    pub tokenizer_path: String,
    pub tokenizer2_path: Option<String>,
    pub t5_tokenizer_path: Option<String>,
    pub height: usize,
    pub width: usize,
    pub use_flash_attn: bool,
    pub num_inference_steps: usize,
}

/// Sampling configuration from YAML
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    pub prompts: Vec<String>,
    pub sample_every: usize,
    pub sample_steps: usize,
    pub cfg_scale: f64,
    pub seed: u64,
    pub use_lora: bool,
    pub lora_scale: f32,
}
