//! Unified sampling interface for all diffusion models
//! Provides a common trait and implementations for SDXL, SD 3.5, and Flux

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use log::{info, debug};
use std::path::{Path, PathBuf};
use std::collections::HashMap;

// Import our image utilities
use super::candle_image_utils::{save_image, create_sample_directory, ModelType};

/// Common trait for all model samplers
pub trait DiffusionSampler {
    /// Generate samples during training
    fn generate_samples(
        &self,
        step: usize,
        prompts: &[&str],
        output_dir: &Path,
        seed: u64,
    ) -> Result<Vec<PathBuf>>;
    
    /// Get the model type for VAE scaling
    fn model_type(&self) -> ModelType;
    
    /// Get default sampling parameters
    fn default_params(&self) -> SamplingParams;
}

/// Sampling parameters common to all models
#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub num_inference_steps: usize,
    pub guidance_scale: f32,
    pub height: usize,
    pub width: usize,
    pub negative_prompt: Option<String>,
}

/// SDXL Sampler Implementation
pub struct SDXLSampler<'a> {
    unet: &'a HashMap<String, Tensor>,
    vae: &'a dyn VaeDecoder,
    text_encoder: &'a dyn TextEncoder,
    device: &'a Device,
    lora_scale: f32,
}

impl<'a> SDXLSampler<'a> {
    pub fn new(
        unet: &'a HashMap<String, Tensor>,
        vae: &'a dyn VaeDecoder,
        text_encoder: &'a dyn TextEncoder,
        device: &'a Device,
        lora_scale: f32,
    ) -> Self {
        Self { unet, vae, text_encoder, device, lora_scale }
    }
}

impl<'a> DiffusionSampler for SDXLSampler<'a> {
    fn generate_samples(
        &self,
        step: usize,
        prompts: &[&str],
        output_dir: &Path,
        seed: u64,
    ) -> Result<Vec<PathBuf>> {
        info!("Generating SDXL samples at step {}", step);
        let params = self.default_params();
        let mut generated_paths = Vec::new();
        
        for (idx, prompt) in prompts.iter().enumerate() {
            debug!("Generating sample {} with prompt: {}", idx, prompt);
            
            // Encode text
            let text_embeddings = self.text_encoder.encode(prompt, 77)?;
            
            // Initialize latents
            let latents = Tensor::randn(
                0.0,
                1.0,
                &[1, 4, params.height / 8, params.width / 8],
                self.device,
            )?;
            
            // Run DDIM sampling loop
            let denoised = self.run_ddim_sampling(
                &latents,
                &text_embeddings,
                params.num_inference_steps,
                params.guidance_scale,
            )?;
            
            // Decode VAE
            let images = self.vae.decode(&(denoised / 0.18215)?)?;
            
            // Convert and save
            let image = images.get(0)?;
            let filename = format!("sample_step{:06}_idx{:02}.jpg", step, idx);
            let filepath = output_dir.join(&filename);
            
            save_image(&image, &filepath)?;
            
            // Save metadata
            self.save_metadata(&filepath, prompt, step, params.guidance_scale, seed + idx as u64)?;
            
            generated_paths.push(filepath);
        }
        
        Ok(generated_paths)
    }
    
    fn model_type(&self) -> ModelType {
        ModelType::SDXL
    }
    
    fn default_params(&self) -> SamplingParams {
        SamplingParams {
            num_inference_steps: 30,
            guidance_scale: 7.5,
            height: 1024,
            width: 1024,
            negative_prompt: None,
        }
    }
}

impl<'a> SDXLSampler<'a> {
    fn run_ddim_sampling(
        &self,
        latents: &Tensor,
        text_embeddings: &Tensor,
        steps: usize,
        guidance_scale: f32,
    ) -> Result<Tensor> {
        // Simplified DDIM sampling for demonstration
        // In production, use proper scheduler
        let mut x = latents.clone();
        
        for i in (0..steps).rev() {
            let t = Tensor::full(i as f32, &[1], self.device)?;
            
            // Classifier-free guidance with conditional and unconditional
            let noise_pred = self.unet_forward(&x, &t, text_embeddings)?;
            
            // DDIM step
            x = self.ddim_step(&x, &noise_pred, i, steps)?;
        }
        
        Ok(x)
    }
    
    fn unet_forward(&self, x: &Tensor, t: &Tensor, encoder_hidden_states: &Tensor) -> Result<Tensor> {
        // Placeholder for UNet forward pass
        // In production, implement proper forward pass with LoRA injection
        Ok(x.clone())
    }
    
    fn ddim_step(&self, x: &Tensor, noise_pred: &Tensor, t: usize, total_steps: usize) -> Result<Tensor> {
        // Simplified DDIM step
        let alpha = 1.0 - (t as f32 / total_steps as f32);
        Ok((x - noise_pred * (1.0 - alpha))?)
    }
    
    fn save_metadata(&self, path: &Path, prompt: &str, step: usize, cfg: f32, seed: u64) -> Result<()> {
        let content = format!(
            "Model: SDXL\nPrompt: {}\nStep: {}\nCFG Scale: {}\nSeed: {}\nLoRA Scale: {}",
            prompt, step, cfg, seed, self.lora_scale
        );
        std::fs::write(path.with_extension("txt"), content)?;
        Ok(())
    }
}

/// SD 3.5 Sampler Implementation
pub struct SD35Sampler<'a> {
    mmdit: &'a HashMap<String, Tensor>,
    vae: &'a dyn VaeDecoder,
    text_encoders: &'a dyn TripleTextEncoder,
    device: &'a Device,
    lora_scale: f32,
}

impl<'a> SD35Sampler<'a> {
    pub fn new(
        mmdit: &'a HashMap<String, Tensor>,
        vae: &'a dyn VaeDecoder,
        text_encoders: &'a dyn TripleTextEncoder,
        device: &'a Device,
        lora_scale: f32,
    ) -> Self {
        Self { mmdit, vae, text_encoders, device, lora_scale }
    }
}

impl<'a> DiffusionSampler for SD35Sampler<'a> {
    fn generate_samples(
        &self,
        step: usize,
        prompts: &[&str],
        output_dir: &Path,
        seed: u64,
    ) -> Result<Vec<PathBuf>> {
        info!("Generating SD 3.5 samples at step {}", step);
        let params = self.default_params();
        let mut generated_paths = Vec::new();
        
        for (idx, prompt) in prompts.iter().enumerate() {
            debug!("Generating sample {} with prompt: {}", idx, prompt);
            
            // Triple text encoding
            let (clip_l, clip_g, t5) = self.text_encoders.encode_triple(prompt)?;
            let text_embeddings = Tensor::cat(&[&clip_l, &clip_g, &t5], 2)?;
            
            // Initialize 16-channel latents
            let latents = Tensor::randn(
                0.0,
                1.0,
                &[1, 16, params.height / 8, params.width / 8],
                self.device,
            )?;
            
            // Flow matching sampling
            let denoised = self.run_flow_matching(
                &latents,
                &text_embeddings,
                params.num_inference_steps,
                params.guidance_scale,
            )?;
            
            // Decode with SD3.5 scaling
            let scaled = (denoised / 1.5305)? + 0.0609;
            let images = self.vae.decode(&(scaled / 0.18215)?)?;
            
            // Convert and save as PNG
            let image = images.get(0)?;
            let filename = format!("sample_step{:06}_idx{:02}.png", step, idx);
            let filepath = output_dir.join(&filename);
            
            save_image(&image, &filepath)?;
            
            // Save metadata
            self.save_metadata(&filepath, prompt, step, params.guidance_scale, seed + idx as u64)?;
            
            generated_paths.push(filepath);
        }
        
        Ok(generated_paths)
    }
    
    fn model_type(&self) -> ModelType {
        ModelType::SD35
    }
    
    fn default_params(&self) -> SamplingParams {
        SamplingParams {
            num_inference_steps: 28,
            guidance_scale: 5.0,
            height: 1024,
            width: 1024,
            negative_prompt: Some("".to_string()),
        }
    }
}

impl<'a> SD35Sampler<'a> {
    fn run_flow_matching(
        &self,
        latents: &Tensor,
        text_embeddings: &Tensor,
        steps: usize,
        guidance_scale: f32,
    ) -> Result<Tensor> {
        // Flow matching for SD 3.5
        let mut x = latents.clone();
        
        for i in 0..steps {
            let t = (i as f32 / steps as f32) * 1000.0;
            let timestep = Tensor::full(t, &[1], self.device)?;
            
            // MMDiT forward pass
            let v_pred = self.mmdit_forward(&x, &timestep, text_embeddings)?;
            
            // Flow matching update
            x = self.flow_step(&x, &v_pred, i, steps)?;
        }
        
        Ok(x)
    }
    
    fn mmdit_forward(&self, x: &Tensor, t: &Tensor, context: &Tensor) -> Result<Tensor> {
        // Placeholder for MMDiT forward pass
        Ok(x.clone())
    }
    
    fn flow_step(&self, x: &Tensor, v: &Tensor, t: usize, total_steps: usize) -> Result<Tensor> {
        let dt = 1.0 / total_steps as f32;
        Ok((x + v * dt)?)
    }
    
    fn save_metadata(&self, path: &Path, prompt: &str, step: usize, cfg: f32, seed: u64) -> Result<()> {
        let content = format!(
            "Model: SD 3.5\nPrompt: {}\nStep: {}\nCFG Scale: {}\nSeed: {}\nLoRA Scale: {}",
            prompt, step, cfg, seed, self.lora_scale
        );
        std::fs::write(path.with_extension("txt"), content)?;
        Ok(())
    }
}

/// Flux Sampler Implementation
pub struct FluxSampler<'a> {
    flux: &'a HashMap<String, Tensor>,
    vae: &'a dyn VaeDecoder,
    text_encoder: &'a dyn TextEncoder,
    device: &'a Device,
    lora_scale: f32,
}

impl<'a> FluxSampler<'a> {
    pub fn new(
        flux: &'a HashMap<String, Tensor>,
        vae: &'a dyn VaeDecoder,
        text_encoder: &'a dyn TextEncoder,
        device: &'a Device,
        lora_scale: f32,
    ) -> Self {
        Self { flux, vae, text_encoder, device, lora_scale }
    }
}

impl<'a> DiffusionSampler for FluxSampler<'a> {
    fn generate_samples(
        &self,
        step: usize,
        prompts: &[&str],
        output_dir: &Path,
        seed: u64,
    ) -> Result<Vec<PathBuf>> {
        info!("Generating Flux samples at step {}", step);
        let params = self.default_params();
        let mut generated_paths = Vec::new();
        
        for (idx, prompt) in prompts.iter().enumerate() {
            debug!("Generating sample {} with prompt: {}", idx, prompt);
            
            // Text encoding
            let text_embeddings = self.text_encoder.encode(prompt, 256)?;
            
            // Initialize patchified latents
            let latents = Tensor::randn(
                0.0,
                1.0,
                &[1, (params.height / 16) * (params.width / 16), 64],
                self.device,
            )?;
            
            // Flow matching with shifted sigmoid
            let denoised = self.run_flux_sampling(
                &latents,
                &text_embeddings,
                params.num_inference_steps,
                params.guidance_scale,
            )?;
            
            // Unpatchify and decode
            let unpatchified = self.unpatchify(&denoised, params.height / 8, params.width / 8)?;
            let images = self.vae.decode(&(unpatchified / 0.13025)?)?;
            
            // Convert and save
            let image = images.get(0)?;
            let filename = format!("sample_step{:06}_idx{:02}.jpg", step, idx);
            let filepath = output_dir.join(&filename);
            
            save_image(&image, &filepath)?;
            
            // Save metadata
            self.save_metadata(&filepath, prompt, step, params.guidance_scale, seed + idx as u64)?;
            
            generated_paths.push(filepath);
        }
        
        Ok(generated_paths)
    }
    
    fn model_type(&self) -> ModelType {
        ModelType::Flux
    }
    
    fn default_params(&self) -> SamplingParams {
        SamplingParams {
            num_inference_steps: 4,  // Schnell
            guidance_scale: 3.5,
            height: 1024,
            width: 1024,
            negative_prompt: None,
        }
    }
}

impl<'a> FluxSampler<'a> {
    fn run_flux_sampling(
        &self,
        latents: &Tensor,
        text_embeddings: &Tensor,
        steps: usize,
        guidance_scale: f32,
    ) -> Result<Tensor> {
        let mut x = latents.clone();
        
        for i in 0..steps {
            // Shifted sigmoid timestep
            let t = self.shifted_sigmoid_schedule(i, steps);
            let timestep = Tensor::full(t, &[1], self.device)?;
            
            // Flux forward pass
            let v_pred = self.flux_forward(&x, &timestep, text_embeddings, guidance_scale)?;
            
            // Flow update
            x = self.flow_step(&x, &v_pred, i, steps)?;
        }
        
        Ok(x)
    }
    
    fn shifted_sigmoid_schedule(&self, t: usize, total: usize) -> f32 {
        let x = 10.0 * (t as f32 / total as f32) - 5.0;
        1.0 / (1.0 + (-x).exp())
    }
    
    fn flux_forward(&self, x: &Tensor, t: &Tensor, context: &Tensor, guidance: f32) -> Result<Tensor> {
        // Placeholder for Flux forward pass
        Ok(x.clone())
    }
    
    fn flow_step(&self, x: &Tensor, v: &Tensor, t: usize, total_steps: usize) -> Result<Tensor> {
        let dt = 1.0 / total_steps as f32;
        Ok((x + v * dt)?)
    }
    
    fn unpatchify(&self, x: &Tensor, h: usize, w: usize) -> Result<Tensor> {
        // Unpatchify from (B, H*W/4, 64) to (B, 16, H, W)
        let batch_size = x.dims()[0];
        let x = x.reshape((batch_size, h / 2, w / 2, 16, 2, 2))?;
        let x = x.permute((0, 3, 1, 4, 2, 5))?;
        x.reshape((batch_size, 16, h, w))
    }
    
    fn save_metadata(&self, path: &Path, prompt: &str, step: usize, cfg: f32, seed: u64) -> Result<()> {
        let content = format!(
            "Model: Flux\nPrompt: {}\nStep: {}\nCFG Scale: {}\nSeed: {}\nLoRA Scale: {}",
            prompt, step, cfg, seed, self.lora_scale
        );
        std::fs::write(path.with_extension("txt"), content)?;
        Ok(())
    }
}

/// Traits for text encoding
pub trait TextEncoder {
    fn encode(&self, prompt: &str, max_length: usize) -> Result<Tensor>;
}

pub trait TripleTextEncoder {
    fn encode_triple(&self, prompt: &str) -> Result<(Tensor, Tensor, Tensor)>;
}

/// Trait for VAE decoding
pub trait VaeDecoder {
    fn decode(&self, latents: &Tensor) -> Result<Tensor>;
}

/// Training integration helper
pub struct TrainingSamplerConfig {
    pub sample_every: usize,
    pub sample_prompts: Vec<String>,
    pub lora_name: String,
}

impl TrainingSamplerConfig {
    pub fn should_sample(&self, step: usize) -> bool {
        step > 0 && step % self.sample_every == 0
    }
    
    pub fn get_output_dir(&self) -> Result<PathBuf> {
        create_sample_directory(&self.lora_name)
    }
}

/// Helper function to create appropriate sampler
pub fn create_sampler<'a>(
    model_type: ModelType,
    model_weights: &'a HashMap<String, Tensor>,
    vae: &'a dyn VaeDecoder,
    text_encoder: &'a dyn TextEncoder,
    device: &'a Device,
    lora_scale: f32,
) -> Box<dyn DiffusionSampler + 'a> {
    match model_type {
        ModelType::SDXL => Box::new(SDXLSampler::new(model_weights, vae, text_encoder, device, lora_scale)),
        ModelType::Flux => Box::new(FluxSampler::new(model_weights, vae, text_encoder, device, lora_scale)),
        _ => panic!("SD3.5 requires TripleTextEncoder"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sampling_config() {
        let config = TrainingSamplerConfig {
            sample_every: 100,
            sample_prompts: vec!["test".to_string()],
            lora_name: "test_lora".to_string(),
        };
        
        assert!(!config.should_sample(0));
        assert!(config.should_sample(100));
        assert!(!config.should_sample(101));
        assert!(config.should_sample(200));
    }
}