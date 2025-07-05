//! Sampling utilities for training pipelines

use eridiffusion_core::{Result, Error, Device, ModelArchitecture, ModelInputs};
use eridiffusion_models::{DiffusionModel, VAE, TextEncoder};
use candle_core::{Tensor, DType};
use std::path::PathBuf;
use rand::{SeedableRng, rngs::StdRng, Rng};
// Image saving functionality implemented without external crate

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
            sample_prompts: vec![
                "lady at the beach".to_string(),
                "a professional photograph of an astronaut riding a horse".to_string(),
                "a painting of a sunset over mountains in the style of van gogh".to_string(),
                "a detailed digital art of a cyberpunk city at night".to_string(),
            ],
            negative_prompt: Some("blurry, bad quality, distorted".to_string()),
            height: 1024,
            width: 1024,
        }
    }
}

/// Sampler for generating images during training
pub struct TrainingSampler {
    config: SamplingConfig,
    device: Device,
}

impl TrainingSampler {
    /// Create new sampler
    pub fn new(config: SamplingConfig, device: Device) -> Self {
        Self { config, device }
    }
    
    /// Generate samples for SD3/SD3.5 using flow matching
    pub async fn sample_sd3(
        &self,
        model: &dyn DiffusionModel,
        vae: &dyn VAE,
        text_encoder: &dyn TextEncoder,
        step: usize,
    ) -> Result<Vec<PathBuf>> {
        let mut saved_paths = Vec::new();
        
        // Create output directory
        let step_dir = self.config.output_dir.join(format!("step_{}", step));
        std::fs::create_dir_all(&step_dir)?;
        
        // Encode prompts
        let (text_embeds, pooled_embeds) = text_encoder.encode(&self.config.sample_prompts)?;
        
        // Encode negative prompt if provided
        let (neg_embeds, neg_pooled) = if let Some(neg_prompt) = &self.config.negative_prompt {
            let neg_prompts = vec![neg_prompt.clone(); self.config.sample_prompts.len()];
            text_encoder.encode(&neg_prompts)?
        } else {
            // Use empty embeddings
            let batch_size = self.config.sample_prompts.len();
            let hidden_size = text_embeds.dims()[2];
            let seq_len = text_embeds.dims()[1];
            
            let candle_device = to_candle_device(&self.device)?;
            let zeros = Tensor::zeros(&[batch_size, seq_len, hidden_size], DType::F32, &candle_device)?;
            let pooled_zeros = pooled_embeds.as_ref().map(|p| {
                let pooled_size = p.dims()[1];
                Tensor::zeros(&[batch_size, pooled_size], DType::F32, &candle_device).ok()
            }).flatten();
            
            (zeros, pooled_zeros)
        };
        
        // Generate latents for each prompt
        for (i, prompt) in self.config.sample_prompts.iter().enumerate() {
            // Get embeddings for this prompt
            let prompt_embeds = text_embeds.narrow(0, i, 1)?;
            let prompt_pooled = pooled_embeds.as_ref().map(|p| p.narrow(0, i, 1).ok()).flatten();
            let neg_embed = neg_embeds.narrow(0, i, 1)?;
            let neg_pooled = neg_pooled.as_ref().map(|p| p.narrow(0, i, 1).ok()).flatten();
            
            // Generate latent
            let latent = self.generate_sd3_latent(
                model,
                &prompt_embeds,
                prompt_pooled.as_ref(),
                &neg_embed,
                neg_pooled.as_ref(),
            )?;
            
            // Decode to image
            let image = vae.decode(&latent)?;
            
            // Save image
            let path = step_dir.join(format!("sample_{}.ppm", i));
            save_tensor_as_image(&image, &path)?;
            saved_paths.push(path.clone());
            
            // Also save with prompt as filename (sanitized)
            let safe_prompt = prompt
                .chars()
                .map(|c| if c.is_alphanumeric() || c == ' ' { c } else { '_' })
                .collect::<String>()
                .trim()
                .chars()
                .take(50)
                .collect::<String>();
            let prompt_path = step_dir.join(format!("{}_{}.ppm", i, safe_prompt));
            std::fs::copy(&path, prompt_path)?;
        }
        
        Ok(saved_paths)
    }
    
    /// Generate a single SD3 latent using flow matching
    fn generate_sd3_latent(
        &self,
        model: &dyn DiffusionModel,
        text_embeds: &Tensor,
        pooled_embeds: Option<&Tensor>,
        neg_embeds: &Tensor,
        neg_pooled: Option<&Tensor>,
    ) -> Result<Tensor> {
        let latent_channels = match model.architecture() {
            ModelArchitecture::SD3 | ModelArchitecture::SD35 => 16,
            _ => 4,
        };
        
        // Calculate latent dimensions
        let latent_height = self.config.height / 8;
        let latent_width = self.config.width / 8;
        
        // Initialize random latent
        let mut latent = if let Some(seed) = self.config.generator_seed {
            // Use seeded random
            let _rng = StdRng::seed_from_u64(seed);
            Tensor::randn(
                0.0f32,
                1.0f32,
                (1, latent_channels, latent_height, latent_width),
                &to_candle_device(&self.device)?,
            )?
        } else {
            Tensor::randn(
                0.0f32,
                1.0f32,
                (1, latent_channels, latent_height, latent_width),
                &to_candle_device(&self.device)?,
            )?
        };
        
        // SD3 uses flow matching - sample along the probability flow ODE
        let num_steps = self.config.num_inference_steps;
        let shift = 3.0; // SD3.5 time shift
        
        for i in 0..num_steps {
            // Calculate timestep (from 1 to 0)
            let t = 1.0 - (i as f32 / (num_steps - 1) as f32);
            let timestep = Tensor::new(&[t * 1000.0], &to_candle_device(&self.device)?)?;
            
            // Apply time shift for SD3.5
            let shifted_t = (t * 1000.0 + shift) / (1000.0 + shift);
            let shifted_timestep = Tensor::new(&[shifted_t * 1000.0], &to_candle_device(&self.device)?)?;
            
            // Prepare conditional and unconditional inputs
            let latent_input = if self.config.guidance_scale > 1.0 {
                // Classifier-free guidance - concatenate conditional and unconditional
                Tensor::cat(&[&latent, &latent], 0)?
            } else {
                latent.clone()
            };
            
            let embeds_input = if self.config.guidance_scale > 1.0 {
                Tensor::cat(&[text_embeds, neg_embeds], 0)?
            } else {
                text_embeds.clone()
            };
            
            let pooled_input = if let (Some(pooled), Some(neg_pooled)) = (pooled_embeds, neg_pooled) {
                if self.config.guidance_scale > 1.0 {
                    Some(Tensor::cat(&[pooled, neg_pooled], 0)?)
                } else {
                    Some(pooled.clone())
                }
            } else {
                None
            };
            
            // Create model inputs
            let mut additional = std::collections::HashMap::new();
            if let Some(pooled) = pooled_input {
                additional.insert("pooled_projections".to_string(), pooled);
            }
            
            let inputs = eridiffusion_core::ModelInputs {
                latents: latent_input,
                timestep: if self.config.guidance_scale > 1.0 {
                    Tensor::cat(&[&shifted_timestep, &shifted_timestep], 0)?
                } else {
                    shifted_timestep
                },
                encoder_hidden_states: Some(embeds_input),
                attention_mask: None,
                guidance_scale: Some(self.config.guidance_scale),
                pooled_projections: None,
                additional,
            };
            
            // Model prediction
            let output = model.forward(&inputs)?;
            let velocity_pred = output.sample;
            
            // Apply classifier-free guidance
            let velocity = if self.config.guidance_scale > 1.0 {
                let cond_velocity = velocity_pred.narrow(0, 0, 1)?;
                let uncond_velocity = velocity_pred.narrow(0, 1, 1)?;
                
                // guidance = uncond + scale * (cond - uncond)
                let diff = cond_velocity.sub(&uncond_velocity)?;
                uncond_velocity.add(&diff.affine(self.config.guidance_scale as f64, 0.0)?)?
            } else {
                velocity_pred
            };
            
            // Update latent using Euler method
            let dt = 1.0 / num_steps as f32;
            latent = (latent - velocity.affine(dt as f64, 0.0)?)?;
        }
        
        Ok(latent)
    }
    
    /// Generate samples for Flux using flow matching
    pub async fn sample_flux(
        &self,
        model: &dyn DiffusionModel,
        vae: &dyn VAE,
        text_encoder: &dyn TextEncoder,
        step: usize,
    ) -> Result<Vec<PathBuf>> {
        let mut saved_paths = Vec::new();
        
        // Create output directory
        let step_dir = self.config.output_dir.join(format!("step_{}", step));
        std::fs::create_dir_all(&step_dir)?;
        
        // Encode prompts (Flux uses T5 and CLIP embeddings)
        let (t5_embeds, clip_embeds) = text_encoder.encode(&self.config.sample_prompts)?;
        
        // Generate latents for each prompt
        for (i, prompt) in self.config.sample_prompts.iter().enumerate() {
            // Get embeddings for this prompt
            let prompt_t5 = t5_embeds.narrow(0, i, 1)?;
            let prompt_clip = clip_embeds.as_ref().map(|c| c.narrow(0, i, 1).ok()).flatten();
            
            // Generate latent
            let latent = self.generate_flux_latent(
                model,
                &prompt_t5,
                prompt_clip.as_ref(),
            )?;
            
            // Decode to image
            let image = vae.decode(&latent)?;
            
            // Save image
            let path = step_dir.join(format!("flux_sample_{}.ppm", i));
            save_tensor_as_image(&image, &path)?;
            saved_paths.push(path.clone());
            
            // Also save with prompt as filename (sanitized)
            let safe_prompt = prompt
                .chars()
                .map(|c| if c.is_alphanumeric() || c == ' ' { c } else { '_' })
                .collect::<String>()
                .trim()
                .chars()
                .take(50)
                .collect::<String>();
            let prompt_path = step_dir.join(format!("flux_{}_{}.ppm", i, safe_prompt));
            std::fs::copy(&path, prompt_path)?;
        }
        
        Ok(saved_paths)
    }
    
    /// Generate a single Flux latent using flow matching
    fn generate_flux_latent(
        &self,
        model: &dyn DiffusionModel,
        t5_embeds: &Tensor,
        clip_embeds: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Flux uses 64 channels in latent space (16 * 2 * 2 from patchification)
        let latent_channels = 16;
        
        // Calculate latent dimensions (Flux uses 2x2 patches)
        let latent_height = self.config.height / 16; // VAE downscale by 8, then patchify by 2
        let latent_width = self.config.width / 16;
        
        // Initialize random latent
        let mut latent = if let Some(seed) = self.config.generator_seed {
            let _rng = StdRng::seed_from_u64(seed);
            Tensor::randn(
                0.0f32,
                1.0f32,
                (1, latent_channels, latent_height, latent_width),
                &to_candle_device(&self.device)?,
            )?
        } else {
            Tensor::randn(
                0.0f32,
                1.0f32,
                (1, latent_channels, latent_height, latent_width),
                &to_candle_device(&self.device)?,
            )?
        };
        
        // Patchify latents for Flux (2x2 patches)
        let patchified = self.patchify_for_flux(&latent)?;
        
        // Flux uses a shifted sigmoid schedule
        let num_steps = self.config.num_inference_steps;
        let timesteps = self.get_flux_timesteps(num_steps, latent_height * latent_width);
        
        // Determine guidance scale based on model variant
        let guidance = match model.architecture() {
            ModelArchitecture::FluxSchnell => 1.0, // No guidance for Schnell
            ModelArchitecture::FluxDev => self.config.guidance_scale.max(3.5), // Default 3.5 for Dev
            _ => self.config.guidance_scale,
        };
        
        // Prepare model-specific inputs
        let img_ids = self.prepare_flux_img_ids(latent_height, latent_width)?;
        let txt_ids = Tensor::zeros((1, t5_embeds.dims()[1], 3), DType::F32, &to_candle_device(&self.device)?)?;
        
        // Flow matching sampling loop
        let mut x = patchified;
        for window in timesteps.windows(2) {
            let (t_curr, t_prev) = match window {
                [a, b] => (a, b),
                _ => continue,
            };
            
            let t_vec = Tensor::new(&[*t_curr as f32], &to_candle_device(&self.device)?)?;
            
            // Create model inputs for Flux
            let mut additional = std::collections::HashMap::new();
            additional.insert("img_ids".to_string(), img_ids.clone());
            additional.insert("txt_ids".to_string(), txt_ids.clone());
            if let Some(clip) = clip_embeds {
                additional.insert("vec".to_string(), clip.clone());
            }
            additional.insert("guidance".to_string(), Tensor::new(&[guidance], &to_candle_device(&self.device)?)?);
            
            let inputs = eridiffusion_core::ModelInputs {
                latents: x.clone(),
                timestep: t_vec,
                encoder_hidden_states: Some(t5_embeds.clone()),
                attention_mask: None,
                guidance_scale: Some(guidance),
                pooled_projections: clip_embeds.cloned(),
                additional,
            };
            
            // Model prediction (velocity for flow matching)
            let output = model.forward(&inputs)?;
            let velocity = output.sample;
            
            // Update using Euler method
            let dt = t_prev - t_curr;
            x = x.add(&velocity.affine(dt as f64, 0.0)?)?;
        }
        
        // Unpatchify back to standard latent format
        self.unpatchify_from_flux(&x, latent_height, latent_width)
    }
    
    /// Patchify latents for Flux (2x2 patches)
    fn patchify_for_flux(&self, latent: &Tensor) -> Result<Tensor> {
        let (b, c, h, w) = latent.dims4()?;
        // Reshape to patches: (b, c, h, ph, w, pw) -> (b, h, w, c, ph, pw)
        let patched = latent
            .reshape((b, c, h / 2, 2, w / 2, 2))?
            .permute((0, 2, 4, 1, 3, 5))?
            .reshape((b, h / 2 * w / 2, c * 4))?;
        Ok(patched)
    }
    
    /// Unpatchify from Flux format back to standard latents
    fn unpatchify_from_flux(&self, x: &Tensor, height: usize, width: usize) -> Result<Tensor> {
        let (b, _hw, c_ph_pw) = x.dims3()?;
        Ok(x.reshape((b, height, width, c_ph_pw / 4, 2, 2))?
            .permute((0, 3, 1, 4, 2, 5))?
            .reshape((b, c_ph_pw / 4, height * 2, width * 2))?)
    }
    
    /// Prepare image position IDs for Flux
    fn prepare_flux_img_ids(&self, height: usize, width: usize) -> Result<Tensor> {
        let device = &to_candle_device(&self.device)?;
        let img_ids = Tensor::stack(
            &[
                Tensor::full(0u32, (height, width), device)?,
                Tensor::arange(0u32, height as u32, device)?
                    .reshape(((), 1))?
                    .broadcast_as((height, width))?,
                Tensor::arange(0u32, width as u32, device)?
                    .reshape((1, ()))?
                    .broadcast_as((height, width))?,
            ],
            2,
        )?
        .to_dtype(DType::F32)?;
        let img_ids = img_ids.reshape((1, height * width, 3))?;
        Ok(img_ids)
    }
    
    /// Get Flux timesteps with shift based on resolution
    fn get_flux_timesteps(&self, num_steps: usize, image_seq_len: usize) -> Vec<f64> {
        // Flux uses a shifted sigmoid schedule
        let timesteps: Vec<f64> = (0..=num_steps)
            .map(|v| v as f64 / num_steps as f64)
            .rev()
            .collect();
            
        // Apply shift based on resolution (following Flux paper)
        let (x1, x2) = (256.0, 4096.0);
        let (y1, y2) = (0.5, 1.15);
        let m = (y2 - y1) / (x2 - x1);
        let b = y1 - m * x1;
        let mu = m * image_seq_len as f64 + b;
        
        timesteps
            .into_iter()
            .map(|t| {
                let e = mu.exp();
                e / (e + (1.0 / t - 1.0).powf(1.0))
            })
            .collect()
    }
    
    /// Generate samples for SDXL
    pub async fn sample_sdxl(
        &self,
        model: &dyn DiffusionModel,
        vae: &dyn VAE,
        text_encoder: &dyn TextEncoder,
        prompt: &str,
        negative_prompt: Option<&str>,
        num_steps: usize,
        guidance_scale: f32,
        seed: u64,
        step: usize,
    ) -> Result<Vec<PathBuf>> {
        // SDXL uses noise prediction with DDIM/DDPM scheduler
        let mut images = Vec::new();
        let device = model.device();
        
        // Initialize noise scheduler (DDIM for faster sampling)
        let num_train_timesteps = 1000;
        let timesteps = (0..num_steps)
            .map(|i| (num_train_timesteps - 1) - (i * num_train_timesteps / num_steps))
            .collect::<Vec<_>>();
        
        // Text encoding for SDXL (dual CLIP)
        let (context, pooled) = text_encoder.encode(&[prompt.to_string()])?;
        
        // Generate latents
        let latent_channels = 4;
        let latent_height = 128; // 1024 / 8
        let latent_width = 128;
        
        let mut rng = StdRng::seed_from_u64(seed);
        // Generate random latents
        let shape = vec![1, latent_channels, latent_height, latent_width];
        let num_elements: usize = shape.iter().product();
        let mut rng_values = vec![0.0f32; num_elements];
        for val in rng_values.iter_mut() {
            let u1: f32 = rng.gen();
            let u2: f32 = rng.gen();
            // Box-Muller transform for normal distribution
            *val = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        }
        let mut latents = Tensor::from_vec(rng_values, shape.as_slice(), &to_candle_device(&device)?)?;
        
        // Sampling loop
        for &t in timesteps.iter() {
            let t_tensor = Tensor::new(&[t as f32], &to_candle_device(&device)?)?;
            
            // Predict noise
            let noise_pred = model.forward(&ModelInputs {
                latents: latents.clone(),
                timestep: t_tensor,
                encoder_hidden_states: Some(context.clone()),
                pooled_projections: pooled.clone(),
                attention_mask: None,
                guidance_scale: Some(self.config.guidance_scale),
                additional: Default::default(),
            })?;
            
            // DDIM step
            let alpha_t = ((1000 - t) as f32 / 1000.0).sqrt();
            let alpha_t_prev = if t > num_train_timesteps / num_steps {
                ((1000 - t + num_train_timesteps / num_steps) as f32 / 1000.0).sqrt()
            } else {
                1.0
            };
            
            let beta_t = 1.0 - alpha_t * alpha_t;
            let beta_t_prev = 1.0 - alpha_t_prev * alpha_t_prev;
            
            // Compute x0 prediction
            let noise_scaled = noise_pred.sample.affine(beta_t.sqrt() as f64, 0.0)?;
            let pred_x0 = latents.sub(&noise_scaled)?
                .affine(1.0 / alpha_t as f64, 0.0)?;
            
            // Compute variance
            let variance = (beta_t_prev / beta_t).sqrt() * (1.0 - alpha_t * alpha_t / alpha_t_prev / alpha_t_prev).sqrt();
            
            // Update latents
            latents = (pred_x0.affine(alpha_t_prev as f64, 0.0)? 
                + noise_pred.sample.affine(variance as f64, 0.0)?)?;
        }
        
        // Decode latents to images
        let scaled_latents = latents.affine(1.0 / 0.18215, 0.0)?;
        let image = vae.decode(&scaled_latents)?;
        
        // Save image
        let filename = format!("sdxl_step{:06}_{:02}.png", step, 0);
        let output_path = self.config.output_dir.join(filename);
        save_tensor_as_image(&image, &output_path)?;
        images.push(output_path);
        
        Ok(images)
    }
}

/// Save tensor as image
/// Convert eridiffusion_core::Device to candle_core::Device
fn to_candle_device(device: &Device) -> Result<candle_core::Device> {
    device.to_candle()
}

/// Save tensor as image
fn save_tensor_as_image(tensor: &Tensor, path: &PathBuf) -> Result<()> {
    use std::io::Write;
    
    // Assume tensor is [1, 3, H, W] in range [-1, 1]
    let tensor = tensor.squeeze(0)?; // Remove batch dimension
    let (channels, height, width) = (tensor.dims()[0], tensor.dims()[1], tensor.dims()[2]);
    
    if channels != 3 {
        return Err(Error::InvalidShape(format!(
            "Expected 3 channels, got {}",
            channels
        )));
    }
    
    // Convert to [0, 255] range
    let tensor = ((tensor + 1.0)? * 127.5)?;
    let tensor = tensor.clamp(0.0, 255.0)?;
    
    // Convert to u8 data
    let data = tensor.flatten_all()?.to_vec1::<f32>()?;
    
    // Create PPM image format (simple text-based format)
    let mut ppm_data = format!("P3\n{} {}\n255\n", width, height);
    
    for y in 0..height {
        for x in 0..width {
            let r = data[y * width + x] as u8;
            let g = data[height * width + y * width + x] as u8;
            let b = data[2 * height * width + y * width + x] as u8;
            ppm_data.push_str(&format!("{} {} {} ", r, g, b));
        }
        ppm_data.push('\n');
    }
    
    // Change extension to .ppm
    let ppm_path = path.with_extension("ppm");
    let mut file = std::fs::File::create(&ppm_path)?;
    file.write_all(ppm_data.as_bytes())?;
    
    Ok(())
}