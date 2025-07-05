//! Inference pipeline

use eridiffusion_core::{
    DiffusionModel, NetworkAdapter, ModelInputs, ModelOutput,
    Device, Result, Error,
};
use eridiffusion_models::{vae::VAE, text_encoder::TextEncoder};
use candle_core::{Tensor, DType};
use serde::{Serialize, Deserialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::HashMap;

/// Inference configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    pub num_inference_steps: usize,
    pub guidance_scale: f32,
    pub eta: f32,
    pub generator_seed: Option<u64>,
    pub scheduler: SchedulerType,
    pub output_type: OutputType,
    pub return_intermediates: bool,
    pub clip_sample: bool,
    pub clip_sample_range: f32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SchedulerType {
    DDIM,
    DDPM,
    PNDM,
    LMSDiscrete,
    EulerDiscrete,
    EulerAncestral,
    DPMSolver,
    UniPC,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputType {
    Latent,
    Tensor,
    PIL,
    Numpy,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            num_inference_steps: 50,
            guidance_scale: 7.5,
            eta: 0.0,
            generator_seed: None,
            scheduler: SchedulerType::DDIM,
            output_type: OutputType::Tensor,
            return_intermediates: false,
            clip_sample: false,
            clip_sample_range: 1.0,
        }
    }
}

/// Inference pipeline
pub struct InferencePipeline {
    model: Arc<Box<dyn DiffusionModel>>,
    vae: Option<Arc<Box<dyn VAE>>>,
    text_encoder: Option<Arc<Box<dyn TextEncoder>>>,
    adapters: Vec<Box<dyn NetworkAdapter>>,
    scheduler: Box<dyn Scheduler>,
    config: InferenceConfig,
    device: Device,
    cache: Arc<RwLock<InferenceCache>>,
}

/// Inference cache
struct InferenceCache {
    text_embeddings: HashMap<String, Tensor>,
    adapter_outputs: HashMap<String, Tensor>,
    latents: Option<Tensor>,
}

impl InferencePipeline {
    /// Create new inference pipeline
    pub fn new(
        model: Box<dyn DiffusionModel>,
        config: InferenceConfig,
        device: Device,
    ) -> Result<Self> {
        let scheduler = create_scheduler(config.scheduler, &config)?;
        
        Ok(Self {
            model: Arc::new(model),
            vae: None,
            text_encoder: None,
            adapters: Vec::new(),
            scheduler,
            config,
            device,
            cache: Arc::new(RwLock::new(InferenceCache {
                text_embeddings: HashMap::new(),
                adapter_outputs: HashMap::new(),
                latents: None,
            })),
        })
    }
    
    /// Add network adapter
    pub fn add_adapter(&mut self, adapter: Box<dyn NetworkAdapter>) {
        self.adapters.push(adapter);
    }
    
    /// Generate images from text
    pub async fn text_to_image(
        &self,
        prompt: &str,
        negative_prompt: Option<&str>,
        width: usize,
        height: usize,
        batch_size: usize,
    ) -> Result<InferenceOutput> {
        // Encode prompts
        let text_embeddings = self.encode_prompt(prompt, negative_prompt, batch_size).await?;
        
        // Generate latents
        let latents = self.prepare_latents(batch_size, height / 8, width / 8).await?;
        
        // Run inference
        self.run_inference(latents, text_embeddings, None).await
    }
    
    /// Generate images from image
    pub async fn image_to_image(
        &self,
        image: &Tensor,
        prompt: &str,
        negative_prompt: Option<&str>,
        strength: f32,
        batch_size: usize,
    ) -> Result<InferenceOutput> {
        // Encode prompts
        let text_embeddings = self.encode_prompt(prompt, negative_prompt, batch_size).await?;
        
        // Encode image to latents
        let init_latents = self.encode_image(image).await?;
        
        // Calculate starting timestep based on strength
        let init_timestep = (self.config.num_inference_steps as f32 * strength) as usize;
        
        // Add noise to latents
        let latents = self.add_noise(&init_latents, init_timestep).await?;
        
        // Run inference
        self.run_inference(latents, text_embeddings, Some(init_timestep)).await
    }
    
    /// Inpaint image
    pub async fn inpaint(
        &self,
        image: &Tensor,
        mask: &Tensor,
        prompt: &str,
        negative_prompt: Option<&str>,
        batch_size: usize,
    ) -> Result<InferenceOutput> {
        // Encode prompts
        let text_embeddings = self.encode_prompt(prompt, negative_prompt, batch_size).await?;
        
        // Prepare masked latents
        let masked_latents = self.prepare_masked_latents(image, mask).await?;
        
        // Run inference
        self.run_inference(masked_latents, text_embeddings, None).await
    }
    
    /// Run inference loop
    async fn run_inference(
        &self,
        mut latents: Tensor,
        text_embeddings: Tensor,
        start_timestep: Option<usize>,
    ) -> Result<InferenceOutput> {
        let timesteps = self.scheduler.get_timesteps(self.config.num_inference_steps);
        let start_idx = start_timestep.unwrap_or(0);
        
        let mut intermediates = Vec::new();
        
        // Apply adapters if any
        let adapter_outputs = self.apply_adapters(&latents).await?;
        
        // Inference loop
        for (i, &timestep) in timesteps[start_idx..].iter().enumerate() {
            // Expand latents for classifier-free guidance
            let latent_model_input = if self.config.guidance_scale > 1.0 {
                Tensor::cat(&[&latents, &latents], 0)?
            } else {
                latents.clone()
            };
            
            // Create model inputs
            let inputs = ModelInputs {
                latents: latent_model_input,
                timestep: Tensor::from_vec(
                    vec![timestep as f32],
                    &[1],
                    &self.device.to_candle()?,
                )?,
                encoder_hidden_states: Some(text_embeddings.clone()),
                pooled_projections: None,
                attention_mask: None,
                guidance_scale: Some(self.config.guidance_scale),
                additional: adapter_outputs.clone(),
            };
            
            // Predict noise
            let noise_pred = self.model.forward(&inputs)?;
            
            // Perform guidance
            let noise_pred = if self.config.guidance_scale > 1.0 {
                let chunks = noise_pred.sample.chunk(2, 0)?;
                let noise_pred_uncond = &chunks[0];
                let noise_pred_text = &chunks[1];
                let diff = (noise_pred_text - noise_pred_uncond)?;
                let guided = (noise_pred_uncond + diff * self.config.guidance_scale as f64)?;
                guided
            } else {
                noise_pred.sample
            };
            
            // Scheduler step
            latents = self.scheduler.step(&noise_pred, timestep, &latents)?;
            
            // Clip sample if configured
            if self.config.clip_sample {
                let range = self.config.clip_sample_range;
                latents = latents.clamp(-range as f64, range as f64)?;
            }
            
            // Save intermediate if requested
            if self.config.return_intermediates && i % 10 == 0 {
                intermediates.push(latents.clone());
            }
        }
        
        // Decode latents
        let images = self.decode_latents(&latents).await?;
        
        Ok(InferenceOutput {
            images,
            latents: if self.config.output_type == OutputType::Latent {
                Some(latents)
            } else {
                None
            },
            intermediates: if self.config.return_intermediates {
                Some(intermediates)
            } else {
                None
            },
        })
    }
    
    /// Encode text prompt
    async fn encode_prompt(
        &self,
        prompt: &str,
        negative_prompt: Option<&str>,
        batch_size: usize,
    ) -> Result<Tensor> {
        // Check cache
        let cache_key = format!("{}|{}", prompt, negative_prompt.unwrap_or(""));
        {
            let cache = self.cache.read().await;
            if let Some(cached) = cache.text_embeddings.get(&cache_key) {
                return Ok(cached.clone());
            }
        }
        
        // Encode text using the text encoder
        let text_encoder = self.text_encoder.as_ref()
            .ok_or_else(|| Error::Model("Text encoder not loaded".to_string()))?;
        
        let (embeddings, pooled) = text_encoder.encode(&[prompt.to_string()])?;
        
        // Get dimensions from embeddings
        let dims = embeddings.dims();
        let seq_len = if dims.len() >= 2 { dims[1] } else { text_encoder.get_max_length() };
        let embedding_dim = if dims.len() >= 3 { dims[2] } else { text_encoder.get_hidden_size() };
        
        // Encode negative prompt
        let negative_embeddings = Tensor::zeros(
            &[1, seq_len, embedding_dim],
            DType::F32,
            &self.device.to_candle()?,
        )?;
        
        // Concatenate for classifier-free guidance
        let text_embeddings = if self.config.guidance_scale > 1.0 {
            Tensor::cat(&[&negative_embeddings, &embeddings], 0)?
        } else {
            embeddings
        };
        
        // Repeat for batch
        let text_embeddings = text_embeddings.repeat(&[batch_size, 1, 1])?;
        
        // Cache result
        {
            let mut cache = self.cache.write().await;
            cache.text_embeddings.insert(cache_key, text_embeddings.clone());
        }
        
        Ok(text_embeddings)
    }
    
    /// Prepare random latents
    async fn prepare_latents(
        &self,
        batch_size: usize,
        height: usize,
        width: usize,
    ) -> Result<Tensor> {
        let shape = &[batch_size, 4, height, width]; // 4 channels for latent space
        let device = self.device.to_candle()?;
        
        // Generate random latents
        let latents = if let Some(seed) = self.config.generator_seed {
            // Set random seed
            use rand::{SeedableRng, rngs::StdRng};
            let _rng = StdRng::seed_from_u64(seed);
            candle_core::Tensor::randn(0.0f32, 1.0, shape, &device)?
        } else {
            candle_core::Tensor::randn(0.0f32, 1.0, shape, &device)?
        };
        
        // Scale by scheduler init noise sigma
        let init_noise_sigma = self.scheduler.init_noise_sigma();
        Ok((latents * init_noise_sigma as f64)?)
    }
    
    /// Encode image to latents
    async fn encode_image(&self, image: &Tensor) -> Result<Tensor> {
        // Use VAE encoder if available
        if let Some(vae) = &self.vae {
            vae.encode(image)
        } else {
            Err(Error::Runtime("VAE not loaded".into()))
        }
    }
    
    /// Decode latents to image
    async fn decode_latents(&self, latents: &Tensor) -> Result<Tensor> {
        // Use VAE decoder if available
        if let Some(vae) = &self.vae {
            vae.decode(latents)
        } else {
            Err(Error::Runtime("VAE not loaded".into()))
        }
    }
    
    /// Add noise to latents
    async fn add_noise(&self, latents: &Tensor, timestep: usize) -> Result<Tensor> {
        let noise = Tensor::randn(
            0.0f32,
            1.0,
            latents.shape(),
            latents.device(),
        )?;
        
        self.scheduler.add_noise(latents, &noise, timestep)
    }
    
    /// Prepare masked latents for inpainting
    async fn prepare_masked_latents(
        &self,
        image: &Tensor,
        mask: &Tensor,
    ) -> Result<Tensor> {
        // Encode image
        let latents = self.encode_image(image).await?;
        
        // Resize mask to latent size
        let latent_mask = self.resize_mask(mask, latents.dims()[2], latents.dims()[3])?;
        
        // Apply mask
        let masked_latents = (latents * (1.0 - &latent_mask)?)?;
        
        Ok(masked_latents)
    }
    
    /// Apply network adapters
    async fn apply_adapters(&self, latents: &Tensor) -> Result<HashMap<String, Tensor>> {
        let mut outputs = HashMap::new();
        
        // Network adapters modify weights during model forward pass
        // They don't produce separate outputs, so we return empty map
        // The actual adapter application happens inside the model's forward method
        // when it checks for active adapters and applies them to the appropriate layers
        
        Ok(outputs)
    }
    
    /// Resize mask
    fn resize_mask(&self, mask: &Tensor, height: usize, width: usize) -> Result<Tensor> {
        let current_shape = mask.dims();
        if current_shape.len() < 2 {
            return Err(Error::Runtime("Mask must have at least 2 dimensions".into()));
        }
        
        let current_height = current_shape[current_shape.len() - 2];
        let current_width = current_shape[current_shape.len() - 1];
        
        if current_height == height && current_width == width {
            return Ok(mask.clone());
        }
        
        // Simple nearest neighbor resize
        let height_scale = current_height as f32 / height as f32;
        let width_scale = current_width as f32 / width as f32;
        
        let batch_size = if current_shape.len() > 2 { current_shape[0] } else { 1 };
        let channels = if current_shape.len() > 3 { current_shape[1] } else { 1 };
        
        // Create indices for gathering
        let mut y_indices = Vec::with_capacity(height);
        let mut x_indices = Vec::with_capacity(width);
        
        for y in 0..height {
            y_indices.push(((y as f32 * height_scale) as usize).min(current_height - 1));
        }
        
        for x in 0..width {
            x_indices.push(((x as f32 * width_scale) as usize).min(current_width - 1));
        }
        
        // Create tensors for indices (convert to i64)
        let y_indices_i64: Vec<i64> = y_indices.into_iter().map(|x| x as i64).collect();
        let x_indices_i64: Vec<i64> = x_indices.into_iter().map(|x| x as i64).collect();
        let y_idx = Tensor::from_vec(y_indices_i64, height, mask.device())?;
        let x_idx = Tensor::from_vec(x_indices_i64, width, mask.device())?;
        
        // Gather along height dimension
        let mask_h = mask.index_select(&y_idx, current_shape.len() - 2)?;
        
        // Gather along width dimension
        let resized = mask_h.index_select(&x_idx, current_shape.len() - 1)?;
        
        Ok(resized)
    }
}

/// Inference output
#[derive(Debug)]
pub struct InferenceOutput {
    pub images: Tensor,
    pub latents: Option<Tensor>,
    pub intermediates: Option<Vec<Tensor>>,
}

/// Scheduler trait
trait Scheduler: Send + Sync {
    fn get_timesteps(&self, num_steps: usize) -> Vec<usize>;
    fn step(&self, noise_pred: &Tensor, timestep: usize, sample: &Tensor) -> Result<Tensor>;
    fn add_noise(&self, sample: &Tensor, noise: &Tensor, timestep: usize) -> Result<Tensor>;
    fn init_noise_sigma(&self) -> f32;
}

/// Create scheduler
fn create_scheduler(scheduler_type: SchedulerType, config: &InferenceConfig) -> Result<Box<dyn Scheduler>> {
    match scheduler_type {
        SchedulerType::DDIM => Ok(Box::new(DDIMScheduler::new(config))),
        SchedulerType::PNDM => Ok(Box::new(PNDMScheduler::new(config))),
        SchedulerType::EulerDiscrete => Ok(Box::new(EulerDiscreteScheduler::new(config))),
        _ => Ok(Box::new(DDIMScheduler::new(config))), // Default to DDIM
    }
}

/// DDIM Scheduler
struct DDIMScheduler {
    num_train_timesteps: usize,
    beta_start: f32,
    beta_end: f32,
    alphas_cumprod: Vec<f32>,
    eta: f32,
}

impl DDIMScheduler {
    fn new(config: &InferenceConfig) -> Self {
        let num_train_timesteps = 1000;
        let beta_start = 0.00085;
        let beta_end = 0.012;
        
        // Linear beta schedule
        let betas: Vec<f32> = (0..num_train_timesteps)
            .map(|i| {
                beta_start + (beta_end - beta_start) * i as f32 / (num_train_timesteps - 1) as f32
            })
            .collect();
        
        // Calculate alphas_cumprod
        let mut alphas_cumprod = Vec::new();
        let mut cumprod = 1.0;
        for beta in &betas {
            cumprod *= 1.0 - beta;
            alphas_cumprod.push(cumprod);
        }
        
        Self {
            num_train_timesteps,
            beta_start,
            beta_end,
            alphas_cumprod,
            eta: config.eta,
        }
    }
}

impl Scheduler for DDIMScheduler {
    fn get_timesteps(&self, num_steps: usize) -> Vec<usize> {
        let step_ratio = self.num_train_timesteps / num_steps;
        (0..num_steps)
            .map(|i| (num_steps - 1 - i) * step_ratio)
            .collect()
    }
    
    fn step(&self, noise_pred: &Tensor, timestep: usize, sample: &Tensor) -> Result<Tensor> {
        // DDIM step
        let alpha_prod_t = self.alphas_cumprod[timestep];
        let alpha_prod_t_prev = if timestep > 0 {
            self.alphas_cumprod[timestep - 1]
        } else {
            1.0
        };
        
        let beta_prod_t = 1.0 - alpha_prod_t;
        let beta_prod_t_prev = 1.0 - alpha_prod_t_prev;
        
        // Compute predicted original sample
        let pred_original_sample = ((sample - noise_pred.affine(beta_prod_t.sqrt() as f64, 0.0)?)? 
            .affine(1.0 / alpha_prod_t.sqrt() as f64, 0.0))?;
        
        // Compute variance
        let variance = (beta_prod_t_prev / beta_prod_t) * 
            (1.0 - alpha_prod_t / alpha_prod_t_prev);
        let std_dev_t = self.eta * variance.sqrt();
        
        // Compute direction
        let pred_sample_direction = noise_pred.affine((1.0 - alpha_prod_t_prev - std_dev_t * std_dev_t).sqrt() as f64, 0.0)?;
        
        // Compute previous sample
        let prev_sample = (pred_original_sample.affine(alpha_prod_t_prev.sqrt() as f64, 0.0)? + pred_sample_direction)?;
        
        // Add noise
        if self.eta > 0.0 && timestep > 0 {
            let noise = Tensor::randn(0.0f32, 1.0, sample.shape(), sample.device())?;
            Ok((prev_sample + noise.affine(std_dev_t as f64, 0.0)?)?)
        } else {
            Ok(prev_sample)
        }
    }
    
    fn add_noise(&self, sample: &Tensor, noise: &Tensor, timestep: usize) -> Result<Tensor> {
        let alpha_prod = self.alphas_cumprod[timestep];
        let noisy = (sample.affine(alpha_prod.sqrt() as f64, 0.0)? + 
                   noise.affine((1.0 - alpha_prod).sqrt() as f64, 0.0)?)?;
        Ok(noisy)
    }
    
    fn init_noise_sigma(&self) -> f32 {
        1.0
    }
}

/// PNDM Scheduler
struct PNDMScheduler {
    ddim: DDIMScheduler,
    ets: Vec<Tensor>,
}

impl PNDMScheduler {
    fn new(config: &InferenceConfig) -> Self {
        Self {
            ddim: DDIMScheduler::new(config),
            ets: Vec::new(),
        }
    }
}

impl Scheduler for PNDMScheduler {
    fn get_timesteps(&self, num_steps: usize) -> Vec<usize> {
        self.ddim.get_timesteps(num_steps)
    }
    
    fn step(&self, noise_pred: &Tensor, timestep: usize, sample: &Tensor) -> Result<Tensor> {
        // Simplified PNDM - would implement full algorithm
        self.ddim.step(noise_pred, timestep, sample)
    }
    
    fn add_noise(&self, sample: &Tensor, noise: &Tensor, timestep: usize) -> Result<Tensor> {
        self.ddim.add_noise(sample, noise, timestep)
    }
    
    fn init_noise_sigma(&self) -> f32 {
        self.ddim.init_noise_sigma()
    }
}

/// Euler Discrete Scheduler  
struct EulerDiscreteScheduler {
    ddim: DDIMScheduler,
    sigmas: Vec<f32>,
}

impl EulerDiscreteScheduler {
    fn new(config: &InferenceConfig) -> Self {
        let ddim = DDIMScheduler::new(config);
        
        // Calculate sigmas
        let sigmas: Vec<f32> = ddim.alphas_cumprod.iter()
            .map(|&alpha| ((1.0 - alpha) / alpha).sqrt())
            .collect();
        
        Self { ddim, sigmas }
    }
}

impl Scheduler for EulerDiscreteScheduler {
    fn get_timesteps(&self, num_steps: usize) -> Vec<usize> {
        self.ddim.get_timesteps(num_steps)
    }
    
    fn step(&self, noise_pred: &Tensor, timestep: usize, sample: &Tensor) -> Result<Tensor> {
        let sigma = self.sigmas[timestep];
        let sigma_next = if timestep > 0 {
            self.sigmas[timestep - 1]
        } else {
            0.0
        };
        
        // Euler step
        let pred_original_sample = (sample - noise_pred.affine(sigma as f64, 0.0)?)?;
        let derivative = (sample - &pred_original_sample)?.affine(1.0 / sigma as f64, 0.0)?;
        let dt = sigma_next - sigma;
        
        Ok((sample + derivative.affine(dt as f64, 0.0)?)?)
    }
    
    fn add_noise(&self, sample: &Tensor, noise: &Tensor, timestep: usize) -> Result<Tensor> {
        self.ddim.add_noise(sample, noise, timestep)
    }
    
    fn init_noise_sigma(&self) -> f32 {
        self.sigmas[0]
    }
}