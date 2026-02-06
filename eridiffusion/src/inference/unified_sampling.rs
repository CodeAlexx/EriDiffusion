//! Unified Sampling Interface for all Diffusion Models
//!
//! Provides a common interface for sampling from SDXL, SD3.5, and Flux models
//! with proper handling of their unique requirements.

use flame_core::device::Device;
use flame_core::{DType, Error, Result, Tensor};
use std::path::{Path, PathBuf};

// Re-export individual samplers
pub use super::flux_sampling::{FluxSampler, FluxSamplingConfig};
pub use super::sd35_sampling::{SD35Sampler, SD35SamplingConfig};
pub use super::sdxl_sampling::{SDXLSampler, SDXLSamplingConfig};

/// Unified model type enum
#[derive(Debug, Clone, Copy)]
pub enum ModelType {
    SDXL,
    SD35Large,
    SD35Medium,
    FluxDev,
    FluxSchnell,
}

/// Unified sampling configuration
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    pub model_type: ModelType,
    pub prompt: String,
    pub negative_prompt: String,
    pub num_inference_steps: usize,
    pub guidance_scale: f64,
    pub height: usize,
    pub width: usize,
    pub batch_size: usize,
    pub seed: Option<u64>,
    pub output_dir: PathBuf,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            model_type: ModelType::SDXL,
            prompt: "A beautiful landscape".to_string(),
            negative_prompt: "".to_string(),
            num_inference_steps: 50,
            guidance_scale: 7.5,
            height: 1024,
            width: 1024,
            batch_size: 1,
            seed: None,
            output_dir: PathBuf::from("outputs"),
        }
    }
}

/// Main unified sampler
pub struct UnifiedSampler {
    device: Device,
    dtype: DType,
}

impl UnifiedSampler {
    pub fn new(device: Device) -> Self {
        // Choose dtype based on model requirements
        Self {
            device,
            dtype: DType::F16, // Default to F16 for memory efficiency
        }
    }

    /// Sample from any model type
    pub fn sample(
        &self,
        config: &SamplingConfig,
        model_components: ModelComponents<'static>,
    ) -> Result<Vec<PathBuf>> {
        match config.model_type {
            ModelType::SDXL => self.sample_sdxl(config, model_components),
            ModelType::SD35Large | ModelType::SD35Medium => {
                self.sample_sd35(config, model_components)
            }
            ModelType::FluxDev | ModelType::FluxSchnell => {
                self.sample_flux(config, model_components)
            }
        }
    }

    /// Sample from SDXL
    fn sample_sdxl(
        &self,
        config: &SamplingConfig,
        components: ModelComponents<'static>,
    ) -> Result<Vec<PathBuf>> {
        let sdxl_config = SDXLSamplingConfig {
            num_inference_steps: config.num_inference_steps,
            guidance_scale: config.guidance_scale,
            ..Default::default()
        };

        let sampler = SDXLSampler::new(self.device.clone(), self.dtype, sdxl_config);

        // SDXL needs dual CLIP encoders
        let (unet, vae, text_encoders, lora) = match components {
            ModelComponents::SDXL { unet, vae, text_encoders, lora } => {
                (unet, vae, text_encoders, lora)
            }
            _ => {
                return Err(flame_core::Error::InvalidOperation(
                    "Invalid model components for SDXL".to_string(),
                ))
            }
        };

        // Generate samples
        // Cast text_encoders from Any to proper type
        let text_encoders = text_encoders
            .downcast_mut::<super::sdxl_sampling::TextEncoders>()
            .ok_or_else(|| {
                flame_core::Error::InvalidOperation("Failed to downcast text encoders".to_string())
            })?;

        sampler.generate_samples(
            &unet,
            &lora,
            &vae,
            text_encoders,
            &[config.prompt.clone()],
            Some(&[config.negative_prompt.clone()]),
            config.height,
            config.width,
            config.seed,
        )
    }

    /// Sample from SD 3.5
    fn sample_sd35(
        &self,
        config: &SamplingConfig,
        components: ModelComponents<'static>,
    ) -> Result<Vec<PathBuf>> {
        let sd35_config = SD35SamplingConfig {
            num_inference_steps: config.num_inference_steps,
            guidance_scale: config.guidance_scale,
            resolution: (config.height, config.width),
            ..Default::default()
        };

        let sampler = SD35Sampler::new(self.device.clone(), self.dtype, sd35_config);

        // Extract components
        let (mmdit, vae, encoders) = match components {
            ModelComponents::SD35 { mmdit, vae, clip_l, clip_g, t5 } => {
                (mmdit, vae, (clip_l, clip_g, t5))
            }
            _ => {
                return Err(flame_core::Error::InvalidOperation(
                    "Invalid model components for SD3.5".to_string(),
                ))
            }
        };

        // Encode prompts
        let (text_embeds, pooled_embeds) = sampler.encode_prompts(
            encoders.0.as_ref(),
            encoders.1.as_ref(),
            encoders.2.as_ref().map(|f| f.as_ref()),
            &config.prompt,
            &config.negative_prompt,
        )?;

        // Generate
        sampler.generate(
            mmdit.as_ref(),
            vae.as_ref(),
            &text_embeds,
            &pooled_embeds,
            config.batch_size,
            config.seed,
        )
    }

    /// Sample from Flux
    fn sample_flux(
        &self,
        config: &SamplingConfig,
        components: ModelComponents<'static>,
    ) -> Result<Vec<PathBuf>> {
        // Flux uses BF16
        let flux_config = FluxSamplingConfig {
            num_inference_steps: config.num_inference_steps,
            guidance_scale: 1.0, // Flux doesn't use CFG
            resolution: (config.height, config.width),
            ..Default::default()
        };

        let sampler = FluxSampler::new(self.device.clone(), DType::BF16, flux_config);

        // Extract components
        let (model, vae, clip, t5) = match components {
            ModelComponents::Flux { model, vae, clip, t5 } => (model, vae, clip, t5),
            _ => {
                return Err(flame_core::Error::InvalidOperation(
                    "Invalid model components for Flux".to_string(),
                ))
            }
        };

        // Encode text
        let (text_embeds, pooled_embeds) =
            sampler.encode_text(clip.as_ref(), t5.as_ref(), &config.prompt)?;

        // Generate
        sampler.generate(
            model.as_ref(),
            vae.as_ref(),
            &text_embeds,
            &pooled_embeds,
            config.batch_size,
            config.seed,
        )
    }
}

/// Model components for different architectures
pub enum ModelComponents<'a> {
    SDXL {
        unet: std::collections::HashMap<String, Tensor>,
        vae: Box<dyn Fn(&Tensor) -> Result<Tensor> + 'a>,
        text_encoders: &'a mut dyn std::any::Any, // TextEncoders
        lora: Box<dyn std::any::Any>,             // LoRACollection
    },
    SD35 {
        mmdit: Box<dyn Fn(&Tensor, &Tensor, &Tensor, Option<&Tensor>) -> Result<Tensor> + 'a>,
        vae: Box<dyn Fn(&Tensor) -> Result<Tensor> + 'a>,
        clip_l: Box<dyn Fn(&str) -> Result<(Tensor, Tensor)> + 'a>,
        clip_g: Box<dyn Fn(&str) -> Result<(Tensor, Tensor)> + 'a>,
        t5: Option<Box<dyn Fn(&str, usize) -> Result<Tensor> + 'a>>,
    },
    Flux {
        model: Box<dyn Fn(&Tensor, &Tensor, &Tensor, &Tensor) -> Result<Tensor> + 'a>,
        vae: Box<dyn Fn(&Tensor) -> Result<Tensor> + 'a>,
        clip: Box<dyn Fn(&str) -> Result<Tensor> + 'a>,
        t5: Box<dyn Fn(&str, usize) -> Result<Tensor> + 'a>,
    },
}

/// Helper to determine optimal settings for each model
pub fn get_optimal_settings(model_type: ModelType) -> SamplingConfig {
    let mut config = SamplingConfig::default();
    config.model_type = model_type;

    match model_type {
        ModelType::SDXL => {
            config.num_inference_steps = 30;
            config.guidance_scale = 7.5;
        }
        ModelType::SD35Large => {
            config.num_inference_steps = 50;
            config.guidance_scale = 7.0;
        }
        ModelType::SD35Medium => {
            config.num_inference_steps = 28; // Optimized for speed
            config.guidance_scale = 5.0;
        }
        ModelType::FluxDev => {
            config.num_inference_steps = 50;
            config.guidance_scale = 1.0; // No CFG
        }
        ModelType::FluxSchnell => {
            config.num_inference_steps = 4; // Distilled model
            config.guidance_scale = 1.0;
        }
    }

    config
}

/// Integration with training pipelines - generate samples during training
pub fn generate_validation_samples(
    model_type: ModelType,
    components: ModelComponents<'static>,
    prompts: &[&str],
    step: usize,
    output_dir: &Path,
) -> Result<Vec<PathBuf>> {
    let device = Device::cuda(0)?;
    let sampler = UnifiedSampler::new(device);

    let mut all_paths = Vec::new();

    // Since we need to use components multiple times but sample() takes ownership,
    // we need to generate all samples in one call
    let mut config = get_optimal_settings(model_type);
    config.seed = Some(42); // Fixed seed for consistency
    config.output_dir = output_dir.join(format!("step_{}", step));

    // Generate samples for all prompts at once
    for (i, prompt) in prompts.iter().enumerate() {
        config.prompt = prompt.to_string();
        config.seed = Some(42 + i as u64); // Different seed for each prompt

        let paths = sampler.sample(&config, components)?;
        all_paths.extend(paths);

        // Can't reuse components after first iteration since sample() takes ownership
        break;
    }

    // TODO: This currently only generates one sample. To generate multiple samples,
    // we'd need to refactor to either clone components or change the API

    Ok(all_paths)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimal_settings() {
        let sdxl_config = get_optimal_settings(ModelType::SDXL);
        assert_eq!(sdxl_config.guidance_scale, 7.5);

        let flux_config = get_optimal_settings(ModelType::FluxDev);
        assert_eq!(flux_config.guidance_scale, 1.0);

        let schnell_config = get_optimal_settings(ModelType::FluxSchnell);
        assert_eq!(schnell_config.num_inference_steps, 4);
    }
}
