//! SD3/SD3.5 inference pipeline

use eridiffusion_core::{Device, Result, Error, ModelInputs};
use eridiffusion_models::{
    sd3::SD3Model, 
    text_encoder::TextEncoder, 
    vae::VAE, 
    ModelOutput,
    adapters::{SD3TextEncoderAdapter, ClipLAdapter, ClipGAdapter, T5Adapter, SD3VAEAdapter},
    sd3_candle::{Which, StableDiffusion3TripleClipWithTokenizer, build_sd3_vae_autoencoder, sd3_vae_vb_rename}
};
use crate::SD35ModelVariant;
use candle_core::{Tensor, DType, D};
use image::{ImageBuffer, Rgb};
use std::sync::{Arc, Mutex};
use tokio::sync::Mutex as TokioMutex;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use candle_nn::VarBuilder;

/// SD3 pipeline configuration
#[derive(Debug, Clone)]
pub struct SD3PipelineConfig {
    pub model_variant: SD35ModelVariant,
    pub scheduler: Scheduler,
    pub guidance_scale: f32,
    pub num_inference_steps: usize,
}

/// Scheduler types
#[derive(Debug, Clone, Copy)]
pub enum Scheduler {
    FlowMatch,
    DDIM,
    EulerDiscrete,
}

/// SD3 inference pipeline
pub struct SD3Pipeline {
    config: SD3PipelineConfig,
    model: Arc<TokioMutex<SD3Model>>,
    clip_l: Arc<Box<dyn TextEncoder>>,
    clip_g: Arc<Box<dyn TextEncoder>>,
    t5: Arc<Box<dyn TextEncoder>>,
    vae: Arc<Box<dyn VAE>>,
    device: Device,
}

impl SD3Pipeline {
    /// Create new SD3 pipeline
    pub fn new(
        config: SD3PipelineConfig,
        model: SD3Model,
        clip_l: Box<dyn TextEncoder>,
        clip_g: Box<dyn TextEncoder>,
        t5: Box<dyn TextEncoder>,
        vae: Box<dyn VAE>,
        device: Device,
    ) -> Result<Self> {
        Ok(Self {
            config,
            model: Arc::new(TokioMutex::new(model)),
            clip_l: Arc::new(clip_l),
            clip_g: Arc::new(clip_g),
            t5: Arc::new(t5),
            vae: Arc::new(vae),
            device,
        })
    }
    
    /// Create SD3 pipeline from model files
    pub fn from_files(
        config: SD3PipelineConfig,
        model_file: &Path,
        clip_g_file: Option<&Path>,
        clip_l_file: Option<&Path>,
        t5_file: Option<&Path>,
        device: Device,
    ) -> Result<Self> {
        // Convert to candle device
        let candle_device = match &device {
            Device::Cpu => candle_core::Device::Cpu,
            Device::Cuda(ordinal) => candle_core::Device::new_cuda(*ordinal)?,
        };
        
        // Determine which variant
        let which = match config.model_variant {
            SD35ModelVariant::Medium => Which::V3_5Medium,
            SD35ModelVariant::Large => Which::V3_5Large,
            SD35ModelVariant::LargeTurbo => Which::V3_5LargeTurbo,
        };
        
        // Create SD3 model
        let mut model = SD3Model::new(which, candle_device.clone())?;
        
        // Load model weights
        model.load_from_files(
            model_file,
            clip_g_file,
            clip_l_file,
            t5_file,
        )?;
        
        // Create text encoder adapters
        let text_encoder = if which.is_3_5() {
            // For SD3.5, load from separate files
            if let (Some(clip_g), Some(clip_l), Some(t5)) = (clip_g_file, clip_l_file, t5_file) {
                Arc::new(Mutex::new(StableDiffusion3TripleClipWithTokenizer::new_split(
                    &clip_g.to_path_buf(),
                    &clip_l.to_path_buf(),
                    &t5.to_path_buf(),
                    &candle_device,
                ).map_err(|e| Error::Model(format!("Failed to load text encoders: {}", e)))?))
            } else {
                return Err(Error::Model("SD3.5 requires separate CLIP-G, CLIP-L and T5 files".to_string()));
            }
        } else {
            // For SD3, load from combined file
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&[model_file], DType::F16, &candle_device)
                    .map_err(|e| Error::Model(format!("Failed to load model file: {}", e)))?
            };
            Arc::new(Mutex::new(StableDiffusion3TripleClipWithTokenizer::new(
                vb.pp("text_encoders")
            ).map_err(|e| Error::Model(format!("Failed to load text encoders: {}", e)))?))
        };
        
        // Create individual encoder adapters
        let clip_l = Box::new(ClipLAdapter::new(text_encoder.clone(), candle_device.clone()));
        let clip_g = Box::new(ClipGAdapter::new(text_encoder.clone(), candle_device.clone()));
        let t5 = Box::new(T5Adapter::new(text_encoder.clone(), candle_device.clone()));
        
        // Create VAE
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[model_file], DType::F16, &candle_device)
                .map_err(|e| Error::Model(format!("Failed to load model file: {}", e)))?
        };
        let vb_vae = vb.rename_f(sd3_vae_vb_rename).pp("first_stage_model");
        let vae_model = build_sd3_vae_autoencoder(vb_vae)
            .map_err(|e| Error::Model(format!("Failed to load VAE: {}", e)))?;
        let vae = Box::new(SD3VAEAdapter::new(Arc::new(Mutex::new(vae_model))));
        
        // Create pipeline
        Self::new(config, model, clip_l, clip_g, t5, vae, device)
    }
    
    /// Generate image from text prompt
    pub async fn generate(
        &self,
        prompt: &str,
        width: usize,
        height: usize,
        guidance_scale: f32,
        num_steps: usize,
        seed: Option<u64>,
    ) -> Result<Tensor> { // TODO: Return image when image crate is added
        // Get time shift based on scheduler
        let time_shift = match self.config.scheduler {
            Scheduler::FlowMatch => 3.0, // Default for flow matching
            _ => 1.0,
        };
        
        // Call the SD3Model's generate method
        let mut model = self.model.lock().await;
        let image = model.generate(
            prompt,
            "", // uncond prompt
            width,
            height,
            num_steps,
            guidance_scale as f64,
            time_shift,
            seed,
            false, // use_slg
        )?;
        
        Ok(image)
    }
    
    /// Generate and save image to file
    pub async fn generate_to_file(
        &self,
        prompt: &str,
        width: usize,
        height: usize,
        guidance_scale: f32,
        num_steps: usize,
        seed: Option<u64>,
        output_path: &Path,
    ) -> Result<()> {
        let image = self.generate(prompt, width, height, guidance_scale, num_steps, seed).await?;
        self.save_image(&image, output_path)?;
        Ok(())
    }
    
    /// Save tensor as image
    pub fn save_image(&self, tensor: &Tensor, path: &Path) -> Result<()> {
        // Ensure tensor is on CPU
        let tensor = tensor.to_device(&candle_core::Device::Cpu)?;
        
        // Get dimensions (B, C, H, W)
        let (_, channels, height, width) = tensor.dims4()
            .map_err(|e| Error::Runtime(format!("Invalid tensor shape: {}", e)))?;
        
        if channels != 3 {
            return Err(Error::Runtime(format!(
                "Expected 3 channels, got {}",
                channels
            )));
        }
        
        // Convert to u8 and get data
        let data = tensor.to_dtype(DType::U8)?;
        let data_vec = data.flatten_all()?.to_vec1::<u8>()
            .map_err(|e| Error::Runtime(format!("Failed to convert tensor: {}", e)))?;
        
        // Create image buffer
        let mut img = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(width as u32, height as u32);
        
        // Copy data (tensor is in CHW format, need to convert to HWC)
        for y in 0..height {
            for x in 0..width {
                let r = data_vec[0 * height * width + y * width + x];
                let g = data_vec[1 * height * width + y * width + x];
                let b = data_vec[2 * height * width + y * width + x];
                img.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
            }
        }
        
        // Save image
        img.save(path)
            .map_err(|e| Error::Runtime(format!("Failed to save image: {}", e)))?;
        
        Ok(())
    }
    
    /// Encode text prompt with triple encoders
    async fn encode_prompt(&self, prompt: &str) -> Result<TextEmbeddings> {
        // Encode with CLIP-L
        let clip_l_output = self.clip_l.encode(&[prompt.to_string()])?;
        
        // Encode with CLIP-G
        let clip_g_output = self.clip_g.encode(&[prompt.to_string()])?;
        
        // Encode with T5
        let t5_output = self.t5.encode(&[prompt.to_string()])?;
        
        // Pad CLIP embeddings to 2048 each
        let clip_l_padded = self.pad_embeddings(&clip_l_output.0, 2048)?;
        let clip_g_padded = self.pad_embeddings(&clip_g_output.0, 2048)?;
        
        // Concatenate all embeddings
        let encoder_hidden_states = Tensor::cat(&[
            &clip_l_padded,
            &clip_g_padded,
            &t5_output.0,
        ], 2)?; // Concatenate along sequence dimension
        
        // Concatenate pooled outputs from CLIP models
        let pooled_output = if let (Some(ref pl), Some(ref pg)) = (clip_l_output.1.as_ref(), clip_g_output.1.as_ref()) {
            Some(Tensor::cat(&[pl, pg], 1)?)
        } else {
            None
        };
        
        let candle_device = match &self.device {
            Device::Cpu => candle_core::Device::Cpu,
            Device::Cuda(ordinal) => candle_core::Device::new_cuda(*ordinal)?,
        };
        
        let pooled = pooled_output.unwrap_or_else(|| {
            // Create pooled output from encoder hidden states if missing
            // Take the mean of the sequence dimension
            encoder_hidden_states.mean(1).unwrap_or_else(|_| {
                Tensor::zeros(&[1, 2048], DType::F32, &candle_device).unwrap()
            })
        });
        
        Ok(TextEmbeddings {
            encoder_hidden_states,
            pooled_output: pooled,
            attention_mask: None,
        })
    }
    
    /// Pad embeddings to target dimension
    fn pad_embeddings(&self, embeddings: &Tensor, target_dim: usize) -> Result<Tensor> {
        let current_dim = embeddings.dim(D::Minus1)?;
        
        if current_dim >= target_dim {
            // Truncate if larger
            Ok(embeddings.narrow(D::Minus1, 0, target_dim)?)
        } else {
            // Pad with zeros
            let padding = target_dim - current_dim;
            let candle_device = match &self.device {
                Device::Cpu => candle_core::Device::Cpu,
                Device::Cuda(ordinal) => candle_core::Device::new_cuda(*ordinal)?,
            };
            let zeros = Tensor::zeros(
                (embeddings.dim(0)?, embeddings.dim(1)?, padding),
                embeddings.dtype(),
                &candle_device,
            )?;
            Ok(Tensor::cat(&[embeddings, &zeros], D::Minus1)?)
        }
    }
    
    /// Get timestep schedule based on scheduler type
    fn get_timestep_schedule(&self, num_steps: usize) -> Result<Vec<f32>> {
        match self.config.scheduler {
            Scheduler::FlowMatch => {
                // Flow matching uses continuous timesteps from 1 to 0
                let mut timesteps = Vec::with_capacity(num_steps);
                for i in 0..num_steps {
                    let t = 1.0 - (i as f32 / (num_steps - 1) as f32);
                    timesteps.push(t);
                }
                Ok(timesteps)
            }
            Scheduler::DDIM => {
                // DDIM uses discrete timesteps
                let mut timesteps = Vec::with_capacity(num_steps);
                let step_size = 1000 / num_steps;
                for i in 0..num_steps {
                    timesteps.push((1000 - i * step_size) as f32);
                }
                Ok(timesteps)
            }
            Scheduler::EulerDiscrete => {
                // Euler uses logarithmic spacing
                let mut timesteps = Vec::with_capacity(num_steps);
                let sigma_max = 14.615f32;
                let sigma_min = 0.0292f32;
                
                for i in 0..num_steps {
                    let t = (num_steps - 1 - i) as f32 / (num_steps - 1) as f32;
                    let sigma = sigma_max.powf(t) * sigma_min.powf(1.0 - t);
                    timesteps.push(sigma);
                }
                Ok(timesteps)
            }
        }
    }
    
    /// Perform scheduler step
    fn scheduler_step(
        &self,
        latents: &Tensor,
        noise_pred: &Tensor,
        timestep: f32,
        step: usize,
        total_steps: usize,
    ) -> Result<Tensor> {
        match self.config.scheduler {
            Scheduler::FlowMatch => {
                // Flow matching ODE step
                let dt = 1.0 / total_steps as f32;
                let velocity = noise_pred;
                
                // Euler step: x_{t-dt} = x_t - dt * v(x_t, t)
                Ok(latents.sub(&velocity.affine(dt as f64, 0.0)?)?)
            }
            Scheduler::DDIM => {
                // DDIM step
                let alpha_prod_t = self.alpha_prod_from_timestep(timestep);
                let alpha_prod_t_prev = if step == total_steps - 1 {
                    1.0
                } else {
                    self.alpha_prod_from_timestep(timestep - 1000.0 / total_steps as f32)
                };
                
                let beta_prod_t = 1.0 - alpha_prod_t;
                let beta_prod_t_prev = 1.0 - alpha_prod_t_prev;
                
                // Compute x0 prediction
                let pred_x0 = (latents - &noise_pred.affine(beta_prod_t.sqrt() as f64, 0.0)?)?.affine(1.0 / alpha_prod_t.sqrt() as f64, 0.0)?;
                
                // Compute variance
                let variance = (beta_prod_t_prev / beta_prod_t) * (1.0 - alpha_prod_t / alpha_prod_t_prev);
                let std_dev = variance.sqrt();
                
                // Compute direction
                let dir_xt = &noise_pred.affine(beta_prod_t_prev.sqrt() as f64, 0.0)?;
                
                // Compute previous sample
                let alpha_prev_sqrt = alpha_prod_t_prev.sqrt();
                let prev_sample = pred_x0.affine(alpha_prev_sqrt as f64, 0.0)?.add(dir_xt)?;
                
                // Add noise for non-final steps
                if step < total_steps - 1 {
                    let noise = Tensor::randn_like(latents, 0.0, 1.0)?;
                    Ok(prev_sample.add(&noise.affine(std_dev as f64, 0.0)?)?)
                } else {
                    Ok(prev_sample)
                }
            }
            Scheduler::EulerDiscrete => {
                // Euler discrete step
                let sigma = timestep;
                let sigma_next = if step == total_steps - 1 {
                    0.0
                } else {
                    self.get_timestep_schedule(total_steps)?[step + 1]
                };
                
                let dt = sigma_next - sigma;
                
                // Euler step
                Ok(latents.add(&noise_pred.affine(dt as f64, 0.0)?)?)
            }
        }
    }
    
    /// Get alpha product from timestep (for DDIM)
    fn alpha_prod_from_timestep(&self, timestep: f32) -> f32 {
        // Simple linear schedule
        let alpha = 1.0 - (timestep / 1000.0);
        alpha.max(0.0001)
    }
    
    /// Decode latents to image
    async fn decode_latents(&self, latents: &Tensor) -> Result<Tensor> {
        // Scale latents
        let scaling_factor = 1.0 / 1.5305; // SD3 VAE scaling factor
        let scaled_latents = (latents * scaling_factor)?;
        
        // Decode with VAE
        let decoded = self.vae.decode(&scaled_latents)?;
        
        // Convert to image
        self.tensor_to_image(&decoded)
    }
    
    /// Convert tensor to RGB image
    fn tensor_to_image(&self, tensor: &Tensor) -> Result<Tensor> {
        // Ensure tensor is on CPU
        let tensor = tensor.to_device(&candle_core::Device::Cpu)?;
        
        // Get dimensions
        let (_, channels, height, width) = tensor.dims4()?;
        
        if channels != 3 {
            return Err(Error::Runtime(format!(
                "Expected 3 channels, got {}",
                channels
            )));
        }
        
        // Convert to f32 and scale to 0-255
        let data = tensor.to_dtype(DType::F32)?;
        let data = ((data + 1.0)? * 127.5)?
            .clamp(0.0, 255.0)?;
        
        // Return the processed tensor
        // When image crate is added, we can convert to actual image
        Ok(data)
    }
}

/// Text embeddings structure
#[derive(Clone)]
pub struct TextEmbeddings {
    pub encoder_hidden_states: Tensor,
    pub pooled_output: Tensor,
    pub attention_mask: Option<Tensor>,
}

/// Create default SD3 pipeline configuration
impl Default for SD3PipelineConfig {
    fn default() -> Self {
        Self {
            model_variant: SD35ModelVariant::Large,
            scheduler: Scheduler::FlowMatch,
            guidance_scale: 7.5,
            num_inference_steps: 50,
        }
    }
}