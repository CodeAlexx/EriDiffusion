//! SD 3.5 Sampling Implementation with Flow Matching
//!
//! Implements Stable Diffusion 3.5 inference using:
//! - MMDiT (Multimodal Diffusion Transformer) architecture
//! - Triple text encoding (CLIP-L, CLIP-G, T5-XXL)
//! - Flow matching with rectified flow
//! (-16i64) as usize-channel VAE

use flame_core::device::Device;
use flame_core::{CudaDevice, DType, Error, Result, Shape, Tensor};
use image::{ImageBuffer, RgbImage};
use log::{debug, info};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// SD 3.5 Sampling Configuration
#[derive(Debug, Clone)]
pub struct SD35SamplingConfig {
    pub num_inference_steps: usize,
    pub guidance_scale: f64,
    pub shift: f64, // Shift parameter for flow matching (typically 3.0)
    pub use_t5: bool,
    pub t5_max_length: usize,
    pub resolution: (usize, usize),
}

impl Default for SD35SamplingConfig {
    fn default() -> Self {
        Self {
            num_inference_steps: 50,
            guidance_scale: 7.0,
            shift: 3.0,
            use_t5: true,
            t5_max_length: 256, // Shorter than Flux
            resolution: (1024, 1024),
        }
    }
}

/// Flow Matching Scheduler for SD 3.5
pub struct FlowMatchScheduler {
    num_inference_steps: usize,
    timesteps: Vec<f32>,
    shift: f64,
}

impl FlowMatchScheduler {
    pub fn new(num_inference_steps: usize, shift: f64) -> Self {
        // Create linear timesteps from 1.0 to 0.0
        let timesteps: Vec<f32> = (0..num_inference_steps)
            .map(|i| 1.0 - (i as f32 / (num_inference_steps - 1) as f32))
            .collect();

        Self { num_inference_steps, timesteps, shift }
    }

    /// Apply time shift for better sampling quality
    fn apply_shift(&self, t: f32) -> f32 {
        let shift = self.shift as f32;
        // Shifted sigmoid: t' = sigmoid(shift * (2t - 1))
        let t_shifted = 1.0 / (1.0 + (-shift * (2.0 * t - 1.0)).exp());
        t_shifted
    }

    /// Single flow matching step
    pub fn step(
        &self,
        model_output: &Tensor,
        timestep: f32,
        sample: &Tensor,
        _generator: Option<&mut StdRng>,
    ) -> Result<Tensor> {
        // Apply shift to timestep
        let t = self.apply_shift(timestep);
        let t_prev = if timestep > 0.0 {
            self.apply_shift(timestep - 1.0 / self.num_inference_steps as f32)
        } else {
            0.0
        };

        // Flow matching update: x_t = (1-t) * x_0 + t * epsilon
        // Rearranging: x_0 = (x_t - t * v_theta) / (1 - t)
        // Where v_theta is the velocity prediction

        // For numerical stability near t=1
        if t > 0.999 {
            return Ok(sample.clone());
        }

        // Update rule for rectified flow
        let dt = t_prev - t;
        let velocity = model_output;

        // x_{t-1} = x_t + velocity * dt
        let updated = sample.add(&velocity.mul_scalar(dt as f32)?)?;

        Ok(updated)
    }
}

/// SD 3.5 Sampler
pub struct SD35Sampler {
    device: Device,
    dtype: DType,
    config: SD35SamplingConfig,
}

impl SD35Sampler {
    pub fn new(device: Device, dtype: DType, config: SD35SamplingConfig) -> Self {
        Self { device, dtype, config }
    }

    /// Encode prompts using triple encoder system
    pub fn encode_prompts(
        &self,
        clip_l: &dyn Fn(&str) -> Result<(Tensor, Tensor)>,
        clip_g: &dyn Fn(&str) -> Result<(Tensor, Tensor)>,
        t5: Option<&dyn Fn(&str, usize) -> Result<Tensor>>,
        prompt: &str,
        negative_prompt: &str,
    ) -> Result<(Tensor, Tensor)> {
        info!("Encoding prompts with triple encoder system...");

        // Encode with CLIP-L
        let (clip_l_embeds, clip_l_pooled) = clip_l(prompt)?;
        let (clip_l_neg_embeds, clip_l_neg_pooled) = clip_l(negative_prompt)?;

        // Encode with CLIP-G
        let (clip_g_embeds, clip_g_pooled) = clip_g(prompt)?;
        let (clip_g_neg_embeds, clip_g_neg_pooled) = clip_g(negative_prompt)?;

        // Encode with T5 if available
        let text_embeds = if self.config.use_t5 & t5.is_some() {
            let t5_encoder = t5.unwrap();
            let t5_embeds = t5_encoder(prompt, self.config.t5_max_length)?;
            let t5_neg_embeds = t5_encoder(negative_prompt, self.config.t5_max_length)?;

            // Pad CLIP embeddings to match T5 dimension (4096)
            let clip_l_padded =
                self.pad_to_dimension(&clip_l_embeds, 4096, self.device.cuda_device_arc())?;
            let clip_g_padded =
                self.pad_to_dimension(&clip_g_embeds, 4096, self.device.cuda_device_arc())?;
            let clip_l_neg_padded =
                self.pad_to_dimension(&clip_l_neg_embeds, 4096, self.device.cuda_device_arc())?;
            let clip_g_neg_padded =
                self.pad_to_dimension(&clip_g_neg_embeds, 4096, self.device.cuda_device_arc())?;

            // Concatenate all embeddings
            let pos_embeds = Tensor::cat(&[&clip_l_padded, &clip_g_padded, &t5_embeds], 1)?;
            let neg_embeds =
                Tensor::cat(&[&clip_l_neg_padded, &clip_g_neg_padded, &t5_neg_embeds], 1)?;

            Tensor::cat(&[&neg_embeds, &pos_embeds], 0)?
        } else {
            // Use only CLIP encoders
            // Concatenate along the last dimension (dimension 1 for 2D tensors)
            let pos_embeds = Tensor::cat(&[&clip_l_embeds, &clip_g_embeds], 1)?;
            // Concatenate along the last dimension (dimension 1 for 2D tensors)
            let neg_embeds = Tensor::cat(&[&clip_l_neg_embeds, &clip_g_neg_embeds], 1)?;

            // Project to expected dimension
            Tensor::cat(&[&neg_embeds, &pos_embeds], 0)?
        };

        // Concatenate pooled embeddings
        // Concatenate pooled embeddings along the last dimension
        let pooled_pos = Tensor::cat(&[&clip_l_pooled, &clip_g_pooled], 1)?;
        let pooled_neg = Tensor::cat(&[&clip_l_neg_pooled, &clip_g_neg_pooled], 1)?;
        let pooled_embeds = Tensor::cat(&[&pooled_neg, &pooled_pos], 0)?;

        Ok((text_embeds, pooled_embeds))
    }

    /// Pad tensor to target dimension
    fn pad_to_dimension(
        &self,
        tensor: &Tensor,
        target_dim: usize,
        device: Arc<CudaDevice>,
    ) -> Result<Tensor> {
        let shape = tensor.shape();
        let dims = shape.dims();
        let current_dim = dims[dims.len() - 1];

        if current_dim >= target_dim {
            return Ok(tensor.narrow(shape.rank() - 1, 0, target_dim)?);
        }

        let padding = target_dim - current_dim;
        let pad_tensor =
            Tensor::zeros(Shape::from_dims(&[dims[0], dims[1], padding]), device.clone())?
                .to_dtype(tensor.dtype())?;

        // Concatenate along the last dimension (dimension 1 for 2D tensors)
        Tensor::cat(&[tensor, &pad_tensor], 1)
    }

    /// Initialize latents for SD 3.5 (16 channels)
    fn prepare_latents(
        &self,
        batch_size: usize,
        height: usize,
        width: usize,
        generator: Option<&mut StdRng>,
    ) -> Result<Tensor> {
        let shape = vec![batch_size, 16, height / 8, width / 8]; // 16 channels for SD3.5

        let latents = if let Some(gen) = generator {
            let values: Vec<f32> =
                (0..shape.iter().product()).map(|_| gen.gen::<f32>() * 2.0 - 1.0).collect();
            Tensor::from_vec(values, Shape::from_dims(&shape), self.device.cuda_device_arc())?
        } else {
            Tensor::randn(Shape::from_dims(&shape), 0.0, 1.0, self.device.cuda_device_arc())?
        };

        latents.to_dtype(self.dtype)
    }

    /// Generate samples using SD 3.5
    pub fn generate(
        &self,
        mmdit_forward: &dyn Fn(&Tensor, &Tensor, &Tensor, &Tensor) -> Result<Tensor>,
        vae_decode: &dyn Fn(&Tensor) -> Result<Tensor>,
        text_embeds: &Tensor,
        pooled_embeds: &Tensor,
        batch_size: usize,
        seed: Option<u64>,
    ) -> Result<Vec<PathBuf>> {
        let (height, width) = self.config.resolution;

        // Initialize random generator
        let mut generator = seed.map(|s| StdRng::seed_from_u64(s));

        // Initialize latents
        info!("Initializing latents...");
        let mut latents = self.prepare_latents(batch_size, height, width, generator.as_mut())?;

        // Create scheduler
        let scheduler = FlowMatchScheduler::new(self.config.num_inference_steps, self.config.shift);

        // Denoising loop
        info!("Running flow matching for {} steps...", self.config.num_inference_steps);
        for (i, &timestep) in scheduler.timesteps.iter().enumerate() {
            // Expand latents for CFG
            let latent_model_input = if self.config.guidance_scale > 1.0 {
                Tensor::cat(&[&latents, &latents], 0)?
            } else {
                latents.clone()
            };

            // Create timestep embedding
            let t = Tensor::zeros(Shape::new(vec![1]), self.device.cuda_device_arc())?
                .add_scalar(timestep)?
                .to_dtype(self.dtype)?
                .unsqueeze(0)?;
            let t = if self.config.guidance_scale > 1.0 { Tensor::cat(&[&t, &t], 0)? } else { t };

            // Forward pass through MMDiT
            let model_output = mmdit_forward(&latent_model_input, &t, text_embeds, pooled_embeds)?;

            // Perform guidance if needed
            let model_output = if self.config.guidance_scale > 1.0 {
                let chunks = model_output.chunk(2, 0)?;
                let uncond_output = &chunks[0];
                let cond_output = &chunks[1];
                uncond_output.add(
                    &cond_output
                        .sub(&uncond_output)?
                        .mul_scalar(self.config.guidance_scale as f32)?,
                )?
            } else {
                model_output
            };

            // Scheduler step
            latents = scheduler.step(&model_output, timestep, &latents, generator.as_mut())?; // TODO: Use gradient_map instead of individual tensor

            if (i + 1) % 10 == 0 {
                info!("Completed step {}/{}", i + 1, self.config.num_inference_steps);
            }
        }

        // VAE decode
        info!("Decoding latents...");
        // SD3.5 VAE scaling
        let latents = latents.mul_scalar(1.0 / 1.5305 as f32)?.add_scalar(0.0609)?;
        let images = vae_decode(&latents)?;

        // Save images
        let output_dir = Path::new("outputs/sd35");
        std::fs::create_dir_all(output_dir)
            .map_err(|e| {
                flame_core::Error::InvalidOperation(format!(
                    "Failed to create directory: {}",
                    e
                ))
            })
            .map_err(|e| flame_core::Error::InvalidOperation(e.to_string()))?;

        let mut saved_paths = Vec::new();
        for i in 0..batch_size {
            let image = images.get(i)?;
            let path = output_dir.join(format!("sd35_sample_{}.png", i));
            self.save_image(&image, &path)?;
            saved_paths.push(path);
        }

        Ok(saved_paths)
    }

    /// Save image tensor to file
    fn save_image(&self, image: &Tensor, path: &Path) -> Result<()> {
        // Convert from [-1, 1] to [0, 255]
        let image = image.clamp(-1.0, 1.0)?.mul_scalar(127.5)?.add_scalar(127.5)?;

        // Ensure CHW format and get dimensions
        let shape = image.shape();
        let dims = shape.dims();
        let (c, h, w) = match shape.rank() {
            3 => (dims[0], dims[1], dims[2]),
            4 => (dims[1], dims[2], dims[3]), // Remove batch dim
            _ => {
                return Err(flame_core::Error::InvalidOperation(
                    "Invalid image dimensions".to_string(),
                ))
            }
        };

        // Convert to HWC and flatten
        let image_data = image.permute(&[1, 2, 0])?.flatten_all()?.to_vec1::<f32>()?;

        // Convert to u8
        let image_data: Vec<u8> = image_data.iter().map(|&x| x.clamp(0.0, 255.0) as u8).collect();

        // Create and save image
        let img = RgbImage::from_raw(w as u32, h as u32, image_data)
            .ok_or_else(|| anyhow::anyhow!("Failed to create image"))?;
        img.save(path).map_err(|e| {
            flame_core::Error::InvalidOperation(format!("Failed to save image: {}", e))
        })?;

        info!("Saved SD3.5 sample to: {}", path.display());
        Ok(())
    }
}

/// Example usage function
pub fn sample_sd35(
    device: Device,
    mmdit_forward: impl Fn(&Tensor, &Tensor, &Tensor, &Tensor) -> Result<Tensor>,
    vae_decode: impl Fn(&Tensor) -> Result<Tensor>,
    clip_l: impl Fn(&str) -> Result<(Tensor, Tensor)>,
    clip_g: impl Fn(&str) -> Result<(Tensor, Tensor)>,
    t5: Option<impl Fn(&str, usize) -> Result<Tensor>>,
    prompt: &str,
    seed: Option<u64>,
) -> Result<Vec<PathBuf>> {
    let config = SD35SamplingConfig::default();
    let sampler = SD35Sampler::new(device, DType::F16, config);

    // Encode prompts
    let (text_embeds, pooled_embeds) = sampler.encode_prompts(
        &clip_l,
        &clip_g,
        t5.as_ref().map(|f| f as &dyn Fn(&str, usize) -> Result<Tensor>),
        prompt,
        "", // negative prompt
    )?;

    // Generate samples
    sampler.generate(
        &mmdit_forward,
        &vae_decode,
        &text_embeds,
        &pooled_embeds,
        1, // batch size
        seed,
    )
}
