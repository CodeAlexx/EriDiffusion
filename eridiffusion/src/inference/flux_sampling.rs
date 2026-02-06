//! Flux Model Sampling Implementation
//!
//! Implements Flux-dev inference with:
//! (-16i64) as usize-channel VAE with 2x2 patchification  
//! - Shifted sigmoid timestep scheduling
//! - Double and single stream blocks
//! - T5-XXL + CLIP-L text encoding
//! - NO classifier-free guidance (guidance scale = 1.0)

use flame_core::device::Device;
use flame_core::{DType, Error, Result, Shape, Tensor};
use image::{ImageBuffer, RgbImage};
use log::{debug, info};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::path::{Path, PathBuf};

/// Flux Sampling Configuration
#[derive(Debug, Clone)]
pub struct FluxSamplingConfig {
    pub num_inference_steps: usize,
    pub shift: f64,          // Shift parameter (typically 3.0)
    pub guidance_scale: f64, // Should be 1.0 for Flux
    pub resolution: (usize, usize),
    pub t5_max_length: usize, // 512 for Flux
}

impl Default for FluxSamplingConfig {
    fn default() -> Self {
        Self {
            num_inference_steps: 50,
            shift: 3.0,
            guidance_scale: 1.0, // NO CFG for Flux
            resolution: (1024, 1024),
            t5_max_length: 512,
        }
    }
}

/// Flux Scheduler with shifted sigmoid
pub struct FluxScheduler {
    num_inference_steps: usize,
    timesteps: Vec<f32>,
    shift: f64,
}

impl FluxScheduler {
    pub fn new(num_inference_steps: usize, shift: f64) -> Self {
        // Create timesteps from 1.0 to 0.0
        let timesteps: Vec<f32> = (0..num_inference_steps)
            .map(|i| {
                let t = (num_inference_steps - 1 - i) as f32 / (num_inference_steps - 1) as f32;
                t
            })
            .collect();

        Self { num_inference_steps, timesteps, shift }
    }

    /// Get the timesteps
    pub fn timesteps(&self) -> &[f32] {
        &self.timesteps
    }

    /// Apply shifted sigmoid transformation
    fn shift_timestep(&self, t: f32) -> f32 {
        // Flux uses: t' = sigmoid(shift * t) / sigmoid(shift)
        let shift = self.shift as f32;
        let sigmoid_shift = 1.0 / (1.0 + (-shift).exp());
        let t_shifted = 1.0 / (1.0 + (-shift * t).exp());
        t_shifted / sigmoid_shift
    }

    /// Single denoising step for Flux
    pub fn step(
        &self,
        model_output: &Tensor,
        timestep_idx: usize,
        sample: &Tensor,
        _generator: Option<&mut StdRng>,
    ) -> Result<Tensor> {
        let t = self.timesteps[timestep_idx];
        let t_shifted = self.shift_timestep(t);

        // Get previous timestep
        let t_prev = if timestep_idx < self.timesteps.len() - 1 {
            self.timesteps[timestep_idx + 1]
        } else {
            0.0
        };
        let t_prev_shifted = self.shift_timestep(t_prev);

        // Flow matching update
        // In Flux: x_t = alpha_t * x_0 + sigma_t * eps
        // where alpha_t = cos(pi/2 * t_shifted), sigma_t = sin(pi/2 * t_shifted)

        let alpha_t = (std::f32::consts::PI / 2.0 * t_shifted).cos();
        let sigma_t = (std::f32::consts::PI / 2.0 * t_shifted).sin();
        let alpha_prev = (std::f32::consts::PI / 2.0 * t_prev_shifted).cos();
        let sigma_prev = (std::f32::consts::PI / 2.0 * t_prev_shifted).sin();

        // Predict x_0 from model output (velocity prediction)
        let pred_x0 =
            sample.sub(&model_output.mul_scalar(sigma_t as f32)?)?.div_scalar(alpha_t.max(1e-5))?;

        // Clip prediction
        let pred_x0 = pred_x0.clamp(-1.0, 1.0)?;

        // Get predicted noise
        let pred_eps =
            sample.sub(&pred_x0.mul_scalar(alpha_t as f32)?)?.div_scalar(sigma_t.max(1e-5))?;

        // Compute x_{t-1}
        let x_prev =
            pred_x0.mul_scalar(alpha_prev as f32)?.add(&pred_eps.mul_scalar(sigma_prev as f32)?)?;

        Ok(x_prev)
    }
}

/// Flux Sampler
pub struct FluxSampler {
    device: Device,
    dtype: DType,
    config: FluxSamplingConfig,
}

impl FluxSampler {
    pub fn new(device: Device, dtype: DType, config: FluxSamplingConfig) -> Self {
        Self { device, dtype, config }
    }

    /// Encode text using CLIP-L and T5-XXL
    pub fn encode_text(
        &self,
        clip_encoder: &dyn Fn(&str) -> Result<Tensor>,
        t5_encoder: &dyn Fn(&str, usize) -> Result<Tensor>,
        prompt: &str,
    ) -> Result<(Tensor, Tensor)> {
        info!("Encoding text with CLIP-L and T5-XXL...");

        // CLIP encoding (max_length=77)
        let clip_embeds = clip_encoder(prompt)?;

        // T5 encoding (max_length=512 for Flux)
        let t5_embeds = t5_encoder(prompt, self.config.t5_max_length)?;

        // Concatenate along sequence dimension
        // CLIP: [1, 77, 768], T5: [1, 512, 4096]
        // Need to project or pad to same dimension
        let text_embeds = Tensor::cat(&[&clip_embeds, &t5_embeds], 1)?;

        // For Flux, we also need pooled CLIP output for conditioning
        // Use the last token of CLIP as pooled representation
        let pooled = clip_embeds.slice(&[(76, 76)])?.squeeze(Some(1))?;

        Ok((text_embeds, pooled))
    }

    /// Patchify latents for Flux (2x2 patches)
    fn patchify(&self, latents: &Tensor) -> Result<Tensor> {
        let dims = latents.shape().dims();
        let (b, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);

        // Ensure dimensions are divisible by 2
        if h % 2 != 0 || w % 2 != 0 {
            return Err(flame_core::Error::InvalidOperation(
                "Height and width must be divisible by 2 for patchification".to_string(),
            ));
        }

        // Reshape to extract 2x2 patches
        // [B, C, H, W] -> [B, C, H/2, 2, W/2, 2]
        let reshaped = latents.reshape(&[b, c, h / 2, 2, w / 2, 2])?;

        // Permute to [B, H/2, W/2, C, 2, 2]
        let permuted = reshaped.permute(&[0, 2, 4, 1, 5, 3])?;

        // Flatten patches: [B, H/2, W/2, C*4]
        let patched = permuted.reshape(&[b, h / 2, w / 2, c * 4])?;

        // Return as [B, H/2*W/2, C*4] for transformer input
        patched.reshape(&[b, (h / 2) * (w / 2), c * 4])
    }

    /// Unpatchify back to latent space
    fn unpatchify(&self, patches: &Tensor, h: usize, w: usize) -> Result<Tensor> {
        let shape = patches.shape();
        let dims = shape.dims();
        let b = dims[0];
        let c = dims[2] / 4; // Original channels before patchification

        // Reshape from [B, H/2*W/2, C*4] to [B, H/2, W/2, C*4]
        let reshaped = patches.reshape(&[b, h / 2, w / 2, c * 4])?;

        // Further reshape to [B, H/2, W/2, C, 2, 2]
        let reshaped = reshaped.reshape(&[b, h / 2, w / 2, c, 2, 2])?;

        // Permute back to [B, C, H/2, 2, W/2, 2]
        let permuted = reshaped.permute(&[0, 3, 1, 4, 2, 5])?;

        // Reshape to original [B, C, H, W]
        permuted.reshape(&[b, c, h, w])
    }

    /// Initialize latents for Flux
    fn prepare_latents(
        &self,
        batch_size: usize,
        height: usize,
        width: usize,
        generator: Option<&mut StdRng>,
    ) -> Result<Tensor> {
        // 16 channels for Flux VAE, 8x downscaling
        let shape = vec![batch_size, 16, height / 8, width / 8];

        let latents = if let Some(gen) = generator {
            let values: Vec<f32> =
                (0..shape.iter().product()).map(|_| gen.gen::<f32>() * 2.0 - 1.0).collect();
            Tensor::from_vec(values, Shape::from_dims(&shape), self.device.cuda_device_arc())?
        } else {
            Tensor::randn(Shape::from_dims(&shape), 0.0f32, 1.0f32, self.device.cuda_device_arc())?
        };

        latents.to_dtype(self.dtype)
    }

    /// Generate samples using Flux
    pub fn generate(
        &self,
        flux_forward: &dyn Fn(&Tensor, &Tensor, &Tensor, &Tensor) -> Result<Tensor>,
        vae_decode: &dyn Fn(&Tensor) -> Result<Tensor>,
        text_embeds: &Tensor,
        pooled_embeds: &Tensor,
        batch_size: usize,
        seed: Option<u64>,
    ) -> Result<Vec<PathBuf>> {
        let (height, width) = self.config.resolution;

        // Initialize generator
        let mut generator = seed.map(|s| StdRng::seed_from_u64(s));

        // Initialize latents
        info!("Initializing Flux latents...");
        let latents = self.prepare_latents(batch_size, height, width, generator.as_mut())?;

        // Patchify latents for transformer input
        let mut latents_patched = self.patchify(&latents)?;

        // Create scheduler
        let scheduler = FluxScheduler::new(self.config.num_inference_steps, self.config.shift);

        // Guidance embedding for Flux (typically 3.5 for dev)
        let guidance_scale = Tensor::zeros(Shape::from_dims(&[1]), self.device.cuda_device_arc())?
            .add_scalar(3.5f32)?
            .to_dtype(self.dtype)?
            .unsqueeze(0)?;
        let guidance_scale = {
            let mut copies = vec![];
            for _ in 0..batch_size {
                copies.push(&guidance_scale);
            }
            Tensor::cat(&copies, 0)?
        };

        // Denoising loop
        info!("Running Flux denoising for {} steps...", self.config.num_inference_steps);
        for (i, &timestep) in scheduler.timesteps.iter().enumerate() {
            // Create timestep embedding
            let t = Tensor::zeros(Shape::from_dims(&[1]), self.device.cuda_device_arc())?
                .add_scalar(timestep)?
                .to_dtype(self.dtype)?
                .unsqueeze(0)?;
            let t = {
                let mut copies = vec![];
                for _ in 0..batch_size {
                    copies.push(&t);
                }
                Tensor::cat(&copies, 0)?
            };

            // Forward pass through Flux model
            // Note: Flux doesn't use CFG, so no need to duplicate inputs
            let model_output = flux_forward(&latents_patched, &t, text_embeds, &guidance_scale)?;

            // Unpatchify for scheduler step
            let latents_shape = self.unpatchify(&latents_patched, height / 8, width / 8)?;
            let output_shape = self.unpatchify(&model_output, height / 8, width / 8)?;

            // Scheduler step
            let latents_next =
                scheduler.step(&output_shape, i, &latents_shape, generator.as_mut())?; // TODO: Use gradient_map instead of individual tensor

            // Patchify again for next iteration
            latents_patched = self.patchify(&latents_next)?;

            if (i + 1) % 10 == 0 {
                info!("Completed step {}/{}", i + 1, self.config.num_inference_steps);
            }
        }

        // Final unpatchify
        let final_latents = self.unpatchify(&latents_patched, height / 8, width / 8)?;

        // VAE decode with Flux-specific scaling
        info!("Decoding Flux latents...");
        let scaled_latents = final_latents.div_scalar(0.3611)?;
        let images = vae_decode(&scaled_latents)?;

        // Save images
        let output_dir = Path::new("outputs/flux");
        std::fs::create_dir_all(output_dir)
            .map_err(|e| {
                flame_core::Error::InvalidOperation(format!("Failed to create directory: {}", e))
            })
            .map_err(|e| flame_core::Error::InvalidOperation(e.to_string()))?;

        let mut saved_paths = Vec::new();
        for i in 0..batch_size {
            let image = images.slice(&[(i, i + 1)])?.squeeze(Some(0))?;
            let path = output_dir.join(format!("flux_sample_{}.png", i));
            self.save_image(&image, &path)?;
            saved_paths.push(path);
        }

        Ok(saved_paths)
    }

    /// Save image tensor to file
    fn save_image(&self, image: &Tensor, path: &Path) -> Result<()> {
        // Convert from [-1, 1] to [0, 255]
        let image = image.clamp(-1.0, 1.0)?.mul_scalar(127.5f32)?.add_scalar(127.5f32)?;

        // Get dimensions
        let shape = image.shape();
        let dims = shape.dims();
        let (c, h, w) = match shape.rank() {
            3 => (dims[0], dims[1], dims[2]),
            4 => (dims[1], dims[2], dims[3]), // Skip batch
            _ => {
                return Err(flame_core::Error::InvalidOperation(
                    "Invalid image tensor rank".to_string(),
                ))
            }
        };

        // Ensure we have RGB
        let rgb_image = if c == 1 {
            // Grayscale to RGB
            Tensor::cat(&[&image, &image, &image], 0)?
        } else if c == 3 {
            image.clone()
        } else {
            return Err(flame_core::Error::InvalidOperation(
                "Image must have 1 or 3 channels".to_string(),
            ));
        };

        // Convert to HWC and flatten
        let image_data = rgb_image.permute(&[1, 2, 0])?.flatten_all()?.to_vec1::<f32>()?;

        // Convert to u8
        let image_data: Vec<u8> = image_data.iter().map(|&x| x.clamp(0.0, 255.0) as u8).collect();

        // Save image
        let img = RgbImage::from_raw(w as u32, h as u32, image_data)
            .ok_or_else(|| anyhow::anyhow!("Failed to create image"))?;
        img.save(path).map_err(|e| {
            flame_core::Error::InvalidOperation(format!("Failed to save image: {}", e))
        })?;

        info!("Saved Flux sample to: {}", path.display());
        Ok(())
    }
}

/// Helper function to create RoPE embeddings for Flux
pub fn create_rope_embeddings(
    seq_len: usize,
    dim: usize,
    theta: f32,
    device: &Device,
) -> Result<Tensor> {
    // Create position indices
    let pos = Tensor::arange(0.0, seq_len as f32, 1.0, device.cuda_device_arc())?;

    // Create frequency bands
    let inv_freq =
        (0..dim / 2).map(|i| 1.0 / theta.powf(2.0 * i as f32 / dim as f32)).collect::<Vec<f32>>();
    let inv_freq =
        Tensor::from_vec(inv_freq, Shape::from_dims(&[dim / 2]), device.cuda_device_arc())?;

    // Compute angles
    let angles = pos.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?;

    // Create sin and cos embeddings
    let sin = angles.sin()?;
    let cos = angles.cos()?;

    // Interleave sin and cos
    // Stack along the last dimension
    let ndims = cos.shape().dims().len();
    let rope = Tensor::stack(&[cos.clone(), sin.clone()], ndims)?.flatten_from(1)?;

    Ok(rope)
}

/// Example usage
pub fn sample_flux(
    device: Device,
    flux_model: impl Fn(&Tensor, &Tensor, &Tensor, &Tensor) -> Result<Tensor>,
    vae_decode: impl Fn(&Tensor) -> Result<Tensor>,
    clip_encoder: impl Fn(&str) -> Result<Tensor>,
    t5_encoder: impl Fn(&str, usize) -> Result<Tensor>,
    prompt: &str,
    seed: Option<u64>,
) -> Result<Vec<PathBuf>> {
    let config = FluxSamplingConfig::default();
    let sampler = FluxSampler::new(device, DType::BF16, config);

    // Encode text
    let (text_embeds, pooled_embeds) = sampler.encode_text(&clip_encoder, &t5_encoder, prompt)?;

    // Generate
    sampler.generate(
        &flux_model,
        &vae_decode,
        &text_embeds,
        &pooled_embeds,
        1, // batch size
        seed,
    )
}
