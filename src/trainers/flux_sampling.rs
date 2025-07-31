//! Flux sampling/inference implementation for training validation
//! 
//! This module provides sampling functionality during training to monitor
//! progress and generate validation images.

use anyhow::{Result, Context};
use flame::{Tensor, Device, DType, D};
use candle_transformers::models::stable_diffusion::vae::AutoEncoderKL;
use image::{DynamicImage, ImageBuffer, Rgb};
use std::path::{Path, PathBuf};
use std::fs;

use crate::models::flux_custom::FluxModelWithLoRA;
use crate::trainers::text_encoders::TextEncoders;
use crate::eridiffusion_core::{ModelInputs, ModelOutput, DiffusionModel};

/// Sampling configuration
pub struct FluxSamplingConfig {
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
}

impl Default for FluxSamplingConfig {
    fn default() -> Self {
        Self {
            num_inference_steps: 28,  // Flux default
            guidance_scale: 3.5,      // Flux default
            seed: Some(42),
            width: 1024,
            height: 1024,
            prompts: vec![
                "a beautiful landscape with mountains and lakes".to_string(),
                "a portrait of a person in dramatic lighting".to_string(),
                "a futuristic cityscape at night".to_string(),
            ],
            negative_prompt: None,
        }
    }
}

/// Flux sampler for generating images during training
pub struct FluxSampler {
    config: FluxSamplingConfig,
    device: Device,
}

impl FluxSampler {
    pub fn new(config: FluxSamplingConfig, device: Device) -> Self {
        Self { config, device }
    }

    /// Generate samples using the current model state
    pub fn generate_samples(
        &self,
        model: &FluxModelWithLoRA,
        vae: &AutoEncoderKL,
        text_encoders: &mut TextEncoders,
        step: usize,
        output_dir: &Path,
    ) -> Result<Vec<PathBuf>> {
        // Create output directory
        fs::create_dir_all(output_dir)?;
        
        let mut generated_paths = Vec::new();
        let batch_size = self.config.prompts.len();
        
        // Encode text prompts
        let (text_embeds, pooled_embeds) = text_encoders.encode_batch(
            &self.config.prompts,
            512,  // Max sequence length
        )?;
        
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
        model: &FluxModelWithLoRA,
        text_embeds: &Tensor,
        pooled_embeds: &Tensor,
        batch_size: usize,
    ) -> Result<Tensor> {
        // Initialize random noise
        // For Flux, we need to account for the 2x2 patches
        let h = self.config.height / 8;  // VAE downscaling
        let w = self.config.width / 8;
        
        // Adjust for patch size (2x2)
        let h_patches = h / 2;
        let w_patches = w / 2;
        
        let shape = [
            batch_size,
            16,  // Flux uses 16 latent channels
            h,
            w,
        ];
        
        let mut img = if let Some(seed) = self.config.seed {
            // For reproducibility, we'd need to set a global seed
            // TODO: Implement proper seeding
            Tensor::randn(0.0f32, 1.0f32, &shape, &self.device)?
        } else {
            Tensor::randn(0.0f32, 1.0f32, &shape, &self.device)?
        };
        
        // Create position embeddings for image
        let img_ids = self.create_img_ids(h_patches, w_patches, batch_size)?;
        
        // Create position embeddings for text
        let txt_seq_len = text_embeds.dim(1)?;
        let txt_ids = Tensor::zeros((batch_size, txt_seq_len, 3), DType::F32, &self.device)?;
        
        // Create timestep schedule (shifted sigmoid for Flux)
        let timesteps = self.create_timestep_schedule(self.config.num_inference_steps)?;
        
        // Create guidance tensor
        let guidance = Tensor::new(&[self.config.guidance_scale], &self.device)?
            .broadcast_as((batch_size,))?;
        
        // Denoising loop
        for (i, &t) in timesteps.iter().enumerate() {
            // Patchify the image: [B, C, H, W] -> [B, H*W/4, C*4]
            let img_patches = self.patchify(&img)?;
            
            // Create timestep tensor
            let timestep = Tensor::new(&[t], &self.device)?
                .broadcast_as((batch_size,))?;
            
            // Model forward pass
            let velocity_pred = model.forward(
                &img_patches,
                &img_ids,
                text_embeds,
                &txt_ids,
                &timestep,
                pooled_embeds,
                Some(&guidance),
            )?;
            
            // Unpatchify the velocity prediction: [B, H*W/4, C*4] -> [B, C, H, W]
            let velocity = self.unpatchify(&velocity_pred, h, w)?;
            
            // Update latents using flow matching
            img = self.flow_step(&img, &velocity, t, i, timesteps.len())?;
        }
        
        Ok(img)
    }
    
    /// Patchify image tensor for Flux processing
    fn patchify(&self, img: &Tensor) -> Result<Tensor> {
        let (bs, c, h, w) = img.dims4()?;
        
        // Reshape to extract 2x2 patches
        // [B, C, H, W] -> [B, C, H/2, 2, W/2, 2]
        let img = img.reshape((bs, c, h / 2, 2, w / 2, 2))?;
        
        // Permute to group patches
        // [B, C, H/2, 2, W/2, 2] -> [B, H/2, W/2, C, 2, 2]
        let img = img.permute((0, 2, 4, 1, 3, 5))?;
        
        // Flatten patches
        // [B, H/2, W/2, C, 2, 2] -> [B, H/2 * W/2, C * 4]
        let img = img.reshape((bs, (h / 2) * (w / 2), c * 4))?;
        
        Ok(img)
    }
    
    /// Unpatchify model output back to image format
    fn unpatchify(&self, x: &Tensor, h: usize, w: usize) -> Result<Tensor> {
        let (b, _seq_len, c_times_4) = x.dims3()?;
        let c = c_times_4 / 4;
        let h_patches = h / 2;
        let w_patches = w / 2;
        
        // Reshape to separate spatial and patch dimensions
        // [B, H/2 * W/2, C * 4] -> [B, H/2, W/2, C, 2, 2]
        let x = x.reshape((b, h_patches, w_patches, c, 2, 2))?;
        
        // Permute to standard image format
        // [B, H/2, W/2, C, 2, 2] -> [B, C, H/2, 2, W/2, 2]
        let x = x.permute((0, 3, 1, 4, 2, 5))?;
        
        // Merge patch dimensions
        // [B, C, H/2, 2, W/2, 2] -> [B, C, H, W]
        let x = x.reshape((b, c, h, w))?;
        
        Ok(x)
    }
    
    /// Create position IDs for image patches
    fn create_img_ids(&self, h: usize, w: usize, batch_size: usize) -> Result<Tensor> {
        // Create 3D position embeddings [batch, seq_len, 3]
        // where 3 dimensions are [batch_idx, y_pos, x_pos]
        
        let mut ids = Vec::new();
        
        for y in 0..h {
            for x in 0..w {
                ids.push(vec![0u32, y as u32, x as u32]);
            }
        }
        
        let ids_tensor = Tensor::new(ids.as_slice(), &self.device)?
            .to_dtype(DType::F32)?
            .unsqueeze(0)?  // Add batch dimension
            .repeat((batch_size, 1, 1))?;
        
        Ok(ids_tensor)
    }

    /// Create timestep schedule for Flux (shifted sigmoid)
    fn create_timestep_schedule(&self, num_steps: usize) -> Result<Vec<f32>> {
        let shift = 1.15;  // Flux shift parameter
        let mut timesteps = Vec::with_capacity(num_steps);
        
        for i in 0..num_steps {
            // Linear spacing in sigmoid space
            let u = (i as f32) / (num_steps - 1) as f32;
            // Apply shifted sigmoid transform
            let x = (u * 2.0 - 1.0) * shift;
            let t = 1.0 / (1.0 + (-x).exp());
            timesteps.push(1.0 - t);  // Reverse for denoising
        }
        
        timesteps.reverse();  // Start from noise
        Ok(timesteps)
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
            let next_idx = step + 1;
            let u_next = (next_idx as f32) / (total_steps - 1) as f32;
            let x_next = (u_next * 2.0 - 1.0) * 1.15;
            1.0 - 1.0 / (1.0 + (-x_next).exp())
        } else {
            0.0
        };
        
        let dt = t_next - t_curr;
        
        // Euler step: x_{t+1} = x_t + dt * v_t
        Ok(x_t.add(&velocity.affine(dt as f64, 0.0)?)?)
    }

    /// Decode latent to image using VAE
    fn decode_latent(&self, vae: &AutoEncoderKL, latent: &Tensor) -> Result<DynamicImage> {
        // Scale latent by VAE scaling factor
        let scaled_latent = (latent / 0.13025)?;
        
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

/// Extension implementation to add sampling to FluxLoRATrainer
impl super::flux_lora::FluxLoRATrainer {
    /// Generate validation samples during training
    pub fn generate_samples(
        &mut self,
        step: usize,
        output_dir: &Path,
        config: Option<FluxSamplingConfig>,
    ) -> Result<Vec<PathBuf>> {
        let sampling_config = config.unwrap_or_default();
        
        // Ensure models are loaded
        let model = self.flux_model.as_ref()
            .or(self.model.as_ref())
            .context("No Flux model loaded for sampling")?;
            
        let vae = self.vae.as_ref()
            .or(self.vae_cpu.as_ref())
            .context("No VAE loaded for sampling")?;
            
        let text_encoders = self.text_encoders.as_mut()
            .context("No text encoders loaded for sampling")?;
        
        // Create sampler
        let device = candle_core::Device::cuda_if_available(0)?;
        let sampler = FluxSampler::new(sampling_config, device);
        
        // Generate samples
        let samples_dir = output_dir.join(format!("samples_step_{:06}", step));
        sampler.generate_samples(
            model,
            vae,
            text_encoders,
            step,
            &samples_dir,
        )
    }
}