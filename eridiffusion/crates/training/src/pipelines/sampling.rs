//! Sampling utilities (minimal compile-safe stubs)

use std::path::PathBuf;

use eridiffusion_core::{Device, DiffusionModel};
use eridiffusion_models::{TextEncoder, VAE};

/// Sampling configuration
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    pub num_inference_steps: usize,
    pub guidance_scale: f32,
    pub eta: f32,
    pub generator_seed: Option<u64>,
    pub output_dir: PathBuf,
    pub sample_prompts: Vec<String>,
    pub negative_prompt: Option<String>,
    pub height: usize,
    pub width: usize,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            num_inference_steps: 50,
            guidance_scale: 7.5,
            eta: 0.0,
            generator_seed: None,
            output_dir: PathBuf::from("samples"),
            sample_prompts: vec![],
            negative_prompt: None,
            height: 1024,
            width: 1024,
        }
    }
}

/// Sampler for generating images during training
pub struct TrainingSampler {
    #[allow(dead_code)]
    config: SamplingConfig,
    #[allow(dead_code)]
    device: Device,
}

impl TrainingSampler {
    pub fn new(config: SamplingConfig, device: Device) -> Self {
        Self { config, device }
    }

    pub async fn sample_sd3(
        &self,
        _model: &dyn DiffusionModel,
        _vae: &dyn VAE,
        _text_encoder: &dyn TextEncoder,
        _step: usize,
    ) -> anyhow::Result<Vec<PathBuf>> {
        Ok(Vec::new())
    }

    pub async fn sample_sdxl(
        &self,
        _model: &dyn DiffusionModel,
        _vae: &dyn VAE,
        _text_encoder: &dyn TextEncoder,
        _step: usize,
    ) -> anyhow::Result<Vec<PathBuf>> {
        Ok(Vec::new())
    }
}
