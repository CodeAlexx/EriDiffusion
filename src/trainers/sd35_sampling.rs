//! SD 3.5 sampling/inference implementation for training validation
//! 
//! This module provides sampling functionality during training to monitor
//! progress and generate validation images.

use anyhow::{Result, Context};
use candle_core::{Tensor, Device, DType, D};
use candle_transformers::models::stable_diffusion::vae::AutoEncoderKL;
use image::{DynamicImage, ImageBuffer, Rgb};
use std::path::{Path, PathBuf};
use std::fs;
use std::collections::HashMap;

use crate::models::sd35_mmdit::SD35MMDiTWithLoRA;
use crate::trainers::text_encoders::TextEncoders;

/// Sampling configuration
pub struct SD35SamplingConfig {
    /// Number of denoising steps
    pub num_inference_steps: usize,
    /// Guidance scale (classifier-free guidance strength)
    pub guidance_scale: f32,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Output image size
    pub width: usize,
    pub height: usize,
    /// Validation prompts
    pub prompts: Vec<String>,
    /// Negative prompt (optional)
    pub negative_prompt: Option<String>,
    /// Use linear timesteps (recommended for SD 3.5)
    pub linear_timesteps: bool,
    /// SNR gamma for loss weighting
    pub snr_gamma: Option<f32>,
}

impl Default for SD35SamplingConfig {
    fn default() -> Self {
        Self {
            num_inference_steps: 50,
            guidance_scale: 7.0,
            seed: Some(42),
            width: 1024,
            height: 1024,
            prompts: vec![
                "a beautiful landscape with mountains and lakes".to_string(),
                "a portrait of a person in dramatic lighting".to_string(),
                "a futuristic cityscape at night".to_string(),
            ],
            negative_prompt: Some("blurry, low quality".to_string()),
            linear_timesteps: true,
            snr_gamma: Some(5.0),
        }
    }
}

/// SD 3.5 sampler for generating images during training
pub struct SD35Sampler {
    config: SD35SamplingConfig,
    device: Device,
}

impl SD35Sampler {
    pub fn new(config: SD35SamplingConfig, device: Device) -> Self {
        Self { config, device }
    }

    /// Generate samples using the current model state
    pub fn generate_samples(
        &self,
        model: &SD35MMDiTWithLoRA,
        vae: &AutoEncoderKL,
        text_encoders: &mut TextEncoders,
        step: usize,
        output_dir: &Path,
    ) -> Result<Vec<PathBuf>> {
        // Create output directory
        fs::create_dir_all(output_dir)?;
        
        let mut generated_paths = Vec::new();
        let batch_size = self.config.prompts.len();
        
        // Encode text prompts with triple encoding (CLIP-L, CLIP-G, T5-XXL)
        let mut all_text_embeds = Vec::new();
        let mut all_pooled_embeds = Vec::new();
        
        for prompt in &self.config.prompts {
            // SD 3.5 uses triple text encoding
            let (clip_l_embeds, clip_l_pooled) = text_encoders.encode_clip_l(
                &[prompt.clone()],
                77,  // Max token length for CLIP
            )?;
            
            let (clip_g_embeds, clip_g_pooled) = text_encoders.encode_clip_g(
                &[prompt.clone()],
                77,
            )?;
            
            let t5_embeds = text_encoders.encode_t5(
                &[prompt.clone()],
                154,  // Max token length for T5 in SD 3.5
            )?;
            
            // Concatenate embeddings (CLIP-L, CLIP-G, T5)
            let combined_embeds = Tensor::cat(&[&clip_l_embeds, &clip_g_embeds, &t5_embeds], 2)?;
            all_text_embeds.push(combined_embeds);
            
            // Use CLIP-G pooled embeddings
            all_pooled_embeds.push(clip_g_pooled);
        }
        
        // Stack embeddings
        let text_embeds = Tensor::cat(&all_text_embeds, 0)?;
        let pooled_embeds = Tensor::cat(&all_pooled_embeds, 0)?;
        
        // Generate latents
        let latents = self.generate_latents(
            model,
            &text_embeds,
            &pooled_embeds,
            batch_size,
        )?;
        
        // Decode latents to images
        for (i, prompt) in self.config.prompts.iter().enumerate() {
            let latent = latents.narrow(0, i, 1)?;
            let image = self.decode_latent(vae, &latent)?;
            
            // Save image
            let filename = format!("step_{:06}_sample_{:02}.png", step, i);
            let filepath = output_dir.join(&filename);
            image.save(&filepath)?;
            
            println!("Generated: {} - {}", filename, prompt);
            generated_paths.push(filepath);
        }
        
        Ok(generated_paths)
    }

    /// Generate latents using flow matching
    fn generate_latents(
        &self,
        model: &SD35MMDiTWithLoRA,
        text_embeds: &Tensor,
        pooled_embeds: &Tensor,
        batch_size: usize,
    ) -> Result<Tensor> {
        // Initialize random noise
        let shape = [
            batch_size,
            16,  // SD 3.5 uses 16 latent channels
            self.config.height / 8,  // VAE downscaling factor
            self.config.width / 8,
        ];
        
        let mut latents = if let Some(seed) = self.config.seed {
            // TODO: Implement proper seeding
            Tensor::randn(0.0f32, 1.0f32, &shape, &self.device)?
        } else {
            Tensor::randn(0.0f32, 1.0f32, &shape, &self.device)?
        };
        
        // Create timestep schedule
        let timesteps = if self.config.linear_timesteps {
            self.create_linear_timestep_schedule(self.config.num_inference_steps)?
        } else {
            self.create_cosine_timestep_schedule(self.config.num_inference_steps)?
        };
        
        // Denoising loop
        for (i, &t) in timesteps.iter().enumerate() {
            // Create timestep tensor
            let timestep = Tensor::new(&[t], &self.device)?
                .broadcast_as((batch_size,))?;
            
            // Prepare y (pooled projections)
            let y = &pooled_embeds;
            
            // Model forward pass
            let noise_pred = model.forward(
                &latents,
                &timestep,
                &text_embeds,
                y,
            )?;
            
            // Apply classifier-free guidance if needed
            let noise_pred = if self.config.guidance_scale > 1.0 {
                self.apply_guidance(&noise_pred, self.config.guidance_scale)?
            } else {
                noise_pred
            };
            
            // Update latents using flow matching
            latents = self.flow_step(&latents, &noise_pred, t, i, timesteps.len())?;
        }
        
        Ok(latents)
    }
    
    /// Create linear timestep schedule for SD 3.5
    fn create_linear_timestep_schedule(&self, num_steps: usize) -> Result<Vec<f32>> {
        let mut timesteps = Vec::with_capacity(num_steps);
        
        for i in 0..num_steps {
            let t = 1.0 - (i as f32 / (num_steps - 1) as f32);
            timesteps.push(t);
        }
        
        Ok(timesteps)
    }
    
    /// Create cosine timestep schedule (alternative)
    fn create_cosine_timestep_schedule(&self, num_steps: usize) -> Result<Vec<f32>> {
        let mut timesteps = Vec::with_capacity(num_steps);
        
        for i in 0..num_steps {
            let u = i as f32 / (num_steps - 1) as f32;
            let alpha = (u * std::f32::consts::PI / 2.0).cos();
            timesteps.push(alpha);
        }
        
        timesteps.reverse();
        Ok(timesteps)
    }
    
    /// Apply classifier-free guidance
    fn apply_guidance(&self, noise_pred: &Tensor, scale: f32) -> Result<Tensor> {
        // For proper CFG, we would need conditional and unconditional predictions
        // For now, just scale the prediction
        Ok(noise_pred.affine(scale as f64, 0.0)?)
    }
    
    /// Flow matching step
    fn flow_step(
        &self,
        x_t: &Tensor,
        velocity: &Tensor,
        t_curr: f32,
        step: usize,
        total_steps: usize,
    ) -> Result<Tensor> {
        // Calculate step size
        let t_next = if step < total_steps - 1 {
            1.0 - ((step + 1) as f32 / (total_steps - 1) as f32)
        } else {
            0.0
        };
        
        let dt = t_next - t_curr;
        
        // Euler step: x_{t+1} = x_t + dt * v_t
        Ok(x_t.add(&velocity.affine(dt as f64, 0.0)?)?)
    }
    
    /// Decode latent to image using VAE
    fn decode_latent(&self, vae: &AutoEncoderKL, latent: &Tensor) -> Result<DynamicImage> {
        // Scale latent by VAE scaling factor (SD 3.5 uses different scaling)
        let scaled_latent = (latent / 0.18215)?;
        
        // Decode
        let decoded = vae.decode(&scaled_latent)?;
        
        // Convert from [-1, 1] to [0, 255]
        let decoded = ((decoded + 1.0)? * 127.5)?;
        let decoded = decoded.clamp(0.0, 255.0)?;
        
        // Convert to image
        tensor_to_image(&decoded)
    }
}

/// Convert tensor to image
fn tensor_to_image(tensor: &Tensor) -> Result<DynamicImage> {
    let (_, channels, height, width) = tensor.dims4()?;
    
    // Ensure we have RGB channels
    if channels != 3 {
        anyhow::bail!("Expected 3 channels, got {}", channels);
    }
    
    // Convert to CPU and get data
    let tensor_cpu = tensor.to_device(&Device::Cpu)?;
    let data = tensor_cpu.to_vec3::<f32>()?;
    
    // Create image buffer
    let mut img = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(width as u32, height as u32);
    
    for y in 0..height {
        for x in 0..width {
            let r = data[0][y][x].round() as u8;
            let g = data[1][y][x].round() as u8;
            let b = data[2][y][x].round() as u8;
            img.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
        }
    }
    
    Ok(DynamicImage::ImageRgb8(img))
}

/// Extension implementation to add sampling to SD35LoRATrainer
impl super::sd35_lora::SD35LoRATrainer {
    /// Generate validation samples during training
    pub fn generate_samples(
        &mut self,
        step: usize,
        output_dir: &Path,
        config: Option<SD35SamplingConfig>,
    ) -> Result<Vec<PathBuf>> {
        let sampling_config = config.unwrap_or_default();
        
        // Create sampler
        let sampler = SD35Sampler::new(sampling_config, self.device.clone());
        
        // Note: This is a simplified implementation
        // In a real implementation, we'd need to:
        // 1. Create MMDiT model with loaded weights
        // 2. Apply LoRA adapters
        // 3. Load VAE if not already loaded
        // 4. Get text encoders
        
        println!("SD 3.5 sampling functionality not yet fully integrated");
        println!("This requires:");
        println!("  - MMDiT model wrapper with LoRA support");
        println!("  - Triple text encoding (CLIP-L, CLIP-G, T5-XXL)");
        println!("  - 16-channel VAE support");
        
        Ok(Vec::new())
    }
}