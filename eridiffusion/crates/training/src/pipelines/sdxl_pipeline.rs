//! SDXL training pipeline with UNet and traditional DDPM

use std::{collections::HashMap, sync::Arc};

use eridiffusion_core::{Device, DiffusionModel, Error, ModelArchitecture, ModelInputs};
use eridiffusion_data::DataLoaderBatch;
use eridiffusion_models::{
    devtensor::{ones_on, shape1, shape2, tensor_from_vec_on},
    TextEncoder, VAE,
};
use flame_core::{DType, Shape, Tensor};
use tracing::{debug, warn};

use super::{PipelineConfig, PipelineUtils, PreparedBatch, PromptEmbeds, TrainingPipeline};
use crate::{
    loss::{masked_eps_loss, masked_l1_loss},
    policy,
    tensor_utils::scalar_f32,
};

/// SDXL training pipeline
pub struct SDXLPipeline {
    config: PipelineConfig,
    vae: Option<Arc<dyn VAE + Send + Sync>>,
    text_encoders: Option<SDXLTextEncoders>,
    noise_scheduler: DDPMScheduler,
    use_refiner: bool,
}

/// Text encoders for SDXL
struct SDXLTextEncoders {
    clip_l: Arc<dyn TextEncoder + Send + Sync>, // CLIP-L: 768 dim
    clip_g: Arc<dyn TextEncoder + Send + Sync>, // OpenCLIP-G: 1280 dim
}

/// DDPM noise scheduler for SDXL
#[derive(Debug, Clone)]
pub struct DDPMScheduler {
    num_train_timesteps: usize,
    beta_start: f32,
    beta_end: f32,
    beta_schedule: String,
    prediction_type: String,
    rescale_betas_zero_snr: bool,

    // Precomputed values
    betas: Option<Tensor>,
    alphas: Option<Tensor>,
    alphas_cumprod: Option<Tensor>,
}

impl Default for DDPMScheduler {
    fn default() -> Self {
        Self {
            num_train_timesteps: 1000,
            beta_start: 0.00085,
            beta_end: 0.012,
            beta_schedule: "scaled_linear".to_string(),
            prediction_type: "epsilon".to_string(),
            rescale_betas_zero_snr: false,
            betas: None,
            alphas: None,
            alphas_cumprod: None,
        }
    }
}

impl DDPMScheduler {
    pub fn new() -> Self {
        Self::default()
    }

    /// Initialize the scheduler with precomputed values
    pub fn init(&mut self, device: &Device) -> anyhow::Result<()> {
        let num_steps = self.num_train_timesteps;

        // Compute beta schedule
        let betas = match self.beta_schedule.as_str() {
            "linear" => {
                let step = (self.beta_end - self.beta_start) / (num_steps - 1) as f32;
                let betas: Vec<f32> =
                    (0..num_steps).map(|i| self.beta_start + i as f32 * step).collect();
                tensor_from_vec_on(betas, shape1(num_steps as i64), device, DType::F32)
                    .map_err(|e| anyhow::anyhow!("scheduler betas linear failed: {e}"))?
            }
            "scaled_linear" => {
                let step = (self.beta_end.sqrt() - self.beta_start.sqrt()) / (num_steps - 1) as f32;
                let betas: Vec<f32> = (0..num_steps)
                    .map(|i| {
                        let beta_sqrt = self.beta_start.sqrt() + i as f32 * step;
                        beta_sqrt * beta_sqrt
                    })
                    .collect();
                tensor_from_vec_on(betas, shape1(num_steps as i64), device, DType::F32)
                    .map_err(|e| anyhow::anyhow!("scheduler betas scaled_linear failed: {e}"))?
            }
            _ => {
                return Err(
                    Error::Config(format!("Unknown beta schedule: {}", self.beta_schedule)).into()
                )
            }
        };

        // Rescale betas for zero SNR if enabled
        let betas = if self.rescale_betas_zero_snr {
            self.rescale_zero_terminal_snr(&betas)?
        } else {
            betas
        };

        // Compute alphas and cumulative products
        let ones = ones_on(betas.shape().clone(), device, DType::F32)
            .map_err(|e| anyhow::anyhow!("scheduler ones alloc failed: {e}"))?;
        let alphas = ones.sub(&betas)?;
        let alphas_cumprod = cumprod_device_0(&alphas)?;

        self.betas = Some(betas);
        self.alphas = Some(alphas);
        self.alphas_cumprod = Some(alphas_cumprod);

        Ok(())
    }

    /// Sample random timesteps for training
    pub fn sample_timesteps(&self, batch_size: usize, device: &Device) -> anyhow::Result<Tensor> {
        let timesteps: Vec<i32> = (0..batch_size)
            .map(|_| rand::random::<usize>() % self.num_train_timesteps)
            .map(|t| t as i32)
            .collect();
        let ts_f32: Vec<f32> = timesteps.iter().map(|&x| x as f32).collect();
        Ok(tensor_from_vec_on(ts_f32, shape1(batch_size as i64), device, DType::F32)
            .map_err(|e| Error::Training(format!("Failed to create timesteps: {e}")))?)
    }

    /// Get alpha_cumprod for given timesteps
    pub fn get_alpha_cumprod(&self, timesteps: &Tensor) -> anyhow::Result<Tensor> {
        let alphas_cumprod_orig = self
            .alphas_cumprod
            .as_ref()
            .ok_or_else(|| Error::Training("Scheduler not initialized".into()))?;

        // Safety clamps on alpha_cumprod (avoid 0/1 extremes)
        let dev = alphas_cumprod_orig.device().clone();
        let eps = crate::tensor_utils::scalar_f32(1e-8, dev.clone())?;
        let cap_hi = crate::tensor_utils::scalar_f32(0.999, dev.clone())?;
        debug_assert_eq!(alphas_cumprod_orig.shape().dims().len(), 1);
        debug_assert!({
            let dt = alphas_cumprod_orig.dtype();
            dt == DType::F32 || dt == DType::BF16
        });
        let alphas_cumprod = alphas_cumprod_orig.maximum(&eps)?.minimum(&cap_hi)?;

        // Prepare indices: round and clamp timesteps to valid range [0, len-1]
        let zero = crate::tensor_utils::scalar_f32(0.0, dev.clone())?;
        let max_idx_f = (self.num_train_timesteps.saturating_sub(1)) as f32;
        let max_idx = crate::tensor_utils::scalar_f32(max_idx_f, dev.clone())?;
        let idx = timesteps
            .round()? // f32
            .maximum(&zero)?
            .minimum(&max_idx)?;

        // Use GPU index_select along dim 0 (I32 indices)
        let idx_i32 = idx.to_dtype(DType::I32)?;
        Ok(alphas_cumprod.index_select(0, &idx_i32)?)
    }

    /// Rescale betas for zero terminal SNR
    fn rescale_zero_terminal_snr(&self, betas: &Tensor) -> anyhow::Result<Tensor> {
        // Convert betas to alphas
        let dev = betas.device().clone();
        let ones = Tensor::ones(betas.shape().clone(), dev.clone())?;
        let alphas = ones.sub(betas)?;

        // Compute cumulative product (device-side)
        let alphas_cumprod = cumprod_device_0(&alphas)?;

        // Get final alpha_cumprod (last element) via narrow
        let n = alphas_cumprod.shape().dims()[0];
        let final_alpha_cumprod = alphas_cumprod.narrow(0, n - 1, 1)?;

        // Rescale to ensure zero terminal SNR
        let rescale_factor = final_alpha_cumprod.sqrt()?;
        let alphas_cumprod_rescaled = alphas_cumprod.div(&rescale_factor)?;

        // Convert back to betas: alpha_t = alpha_bar_t / alpha_bar_{t-1}
        let n = alphas_cumprod_rescaled.shape().dims()[0];
        let head = crate::tensor_utils::scalar_f32(1.0, alphas_cumprod_rescaled.device().clone())?; // [1]
        let tail = alphas_cumprod_rescaled.narrow(0, 0, n - 1)?; // [n-1]
        let prev = Tensor::cat(&[&head, &tail], 0)?; // [n]
        let alphas_rescaled = alphas_cumprod_rescaled.div(&prev)?;
        let betas_rescaled = ones.sub(&alphas_rescaled)?;

        Ok(betas_rescaled)
    }

    // Device-side cumprod is used; no host fallback needed.
}

/// Compute cumulative product along dim 0 on device using narrow + cat.
fn cumprod_device_0(x: &Tensor) -> anyhow::Result<Tensor> {
    let n = x.shape().dims()[0];
    if n == 0 {
        return Ok(x.clone());
    }
    let mut out: Vec<Tensor> = Vec::with_capacity(n);
    // First element
    let mut running = x.narrow(0, 0, 1)?;
    out.push(running.clone());
    for i in 1..n {
        let xi = x.narrow(0, i, 1)?;
        running = running.mul(&xi)?;
        out.push(running.clone());
    }
    Ok(Tensor::cat(&out.iter().collect::<Vec<_>>(), 0)?)
}

impl SDXLPipeline {
    pub fn new(config: PipelineConfig) -> anyhow::Result<Self> {
        let mut noise_scheduler = DDPMScheduler::new();
        noise_scheduler.init(&config.device)?;

        Ok(Self { config, vae: None, text_encoders: None, noise_scheduler, use_refiner: false })
    }

    /// Set VAE for latent encoding
    pub fn with_vae(mut self, vae: Arc<dyn VAE + Send + Sync>) -> Self {
        self.vae = Some(vae);
        self
    }

    /// Set text encoders for SDXL
    pub fn with_text_encoders(
        mut self,
        clip_l: Arc<dyn TextEncoder + Send + Sync>,
        clip_g: Arc<dyn TextEncoder + Send + Sync>,
    ) -> Self {
        self.text_encoders = Some(SDXLTextEncoders { clip_l, clip_g });
        self
    }

    /// Enable refiner model
    pub fn with_refiner(mut self, use_refiner: bool) -> Self {
        self.use_refiner = use_refiner;
        self
    }

    /// Add noise using DDPM schedule
    fn add_noise_ddpm(
        &self,
        original_samples: &Tensor,
        noise: &Tensor,
        timesteps: &Tensor,
    ) -> anyhow::Result<Tensor> {
        let alpha_cumprod = self.noise_scheduler.get_alpha_cumprod(timesteps)?;
        let sqrt_alpha_cumprod = alpha_cumprod.sqrt()?;
        let one = crate::tensor_utils::scalar_f32(1.0, alpha_cumprod.device().clone())?;
        let sqrt_one_minus_alpha_cumprod = one.sub(&alpha_cumprod)?.sqrt()?;

        // Expand to match sample dimensions
        let sqrt_alpha_cumprod = sqrt_alpha_cumprod.unsqueeze(1)?.unsqueeze(2)?.unsqueeze(3)?;
        let sqrt_one_minus_alpha_cumprod =
            sqrt_one_minus_alpha_cumprod.unsqueeze(1)?.unsqueeze(2)?.unsqueeze(3)?;

        // Apply noise: sqrt(alpha_cumprod) * x + sqrt(1 - alpha_cumprod) * noise
        let noisy_samples = original_samples
            .mul(&sqrt_alpha_cumprod)?
            .add(&noise.mul(&sqrt_one_minus_alpha_cumprod)?)?;

        Ok(noisy_samples)
    }

    /// Get conditioning for SDXL (includes time embeddings and other conditioning)
    fn get_conditioning(
        &self,
        batch: &PreparedBatch,
        timesteps: &Tensor,
    ) -> anyhow::Result<HashMap<String, Tensor>> {
        let mut conditioning = HashMap::new();

        // Time embeddings
        let time_embeds = PipelineUtils::get_timestep_embedding(timesteps, 320, 10000.0)?;
        conditioning.insert("time_embeds".to_string(), time_embeds);

        // Original size conditioning
        if let Some(original_sizes) = batch.metadata.get("original_sizes") {
            if let Ok(sizes) = serde_json::from_value::<Vec<(u32, u32)>>(original_sizes.clone()) {
                let size_embeds = self.encode_size_conditioning(&sizes, timesteps.device())?;
                conditioning.insert("original_size_embeds".to_string(), size_embeds);
            }
        }

        // Crop coordinates conditioning
        if let Some(crop_coords) = batch.metadata.get("crop_coords") {
            if let Ok(coords) = serde_json::from_value::<Vec<(u32, u32)>>(crop_coords.clone()) {
                let crop_embeds = self.encode_size_conditioning(&coords, timesteps.device())?;
                conditioning.insert("crop_coords_embeds".to_string(), crop_embeds);
            }
        }

        // Target size conditioning (usually same as original for training)
        if let Some(original_sizes) = batch.metadata.get("original_sizes") {
            if let Ok(sizes) = serde_json::from_value::<Vec<(u32, u32)>>(original_sizes.clone()) {
                let target_embeds = self.encode_size_conditioning(&sizes, timesteps.device())?;
                conditioning.insert("target_size_embeds".to_string(), target_embeds);
            }
        }

        Ok(conditioning)
    }

    /// Encode size conditioning for SDXL
    fn encode_size_conditioning(
        &self,
        sizes: &[(u32, u32)],
        device: &std::sync::Arc<flame_core::CudaDevice>,
    ) -> anyhow::Result<Tensor> {
        // SDXL uses [W,H] ordering for size/crop embeddings
        let flattened: Vec<f32> =
            sizes.iter().flat_map(|(h, w)| vec![*w as f32, *h as f32]).collect();
        let device_enum = Device::Cuda(device.ordinal());
        let tensor =
            tensor_from_vec_on(flattened, shape2(sizes.len() as i64, 2), &device_enum, DType::F32)
                .map_err(|e| anyhow::anyhow!("encode_size_conditioning failed: {e}"))?;
        Ok(tensor)
    }
}

impl TrainingPipeline for SDXLPipeline {
    fn architecture(&self) -> ModelArchitecture {
        ModelArchitecture::SDXL
    }

    fn prepare_batch(&self, batch: &DataLoaderBatch) -> anyhow::Result<PreparedBatch> {
        // Encode images to latents if VAE is available
        let latents = if let Some(vae) = &self.vae {
            let b = batch.images.shape().dims()[0];
            debug!("Encoding batch of {} images to latents", b);
            vae.encode(&batch.images)?
        } else {
            // Assume pre-encoded latents are provided
            batch.images.clone()
        };

        // Verify latent shape for SDXL (4 channels)
        let latent_channels = latents.shape().dims()[1];
        if latent_channels != 4 {
            warn!(
                "Expected 4 latent channels for SDXL, got {}. Continuing anyway.",
                latent_channels
            );
        }

        // Apply input perturbation
        let latents =
            PipelineUtils::apply_input_perturbation(&latents, self.config.input_perturbation)?;

        // Get batch size and image dimensions
        let batch_size = batch.images.shape().dims()[0];
        let height = batch.images.shape().dims()[2];
        let width = batch.images.shape().dims()[3];

        // Extract metadata with proper SDXL conditioning
        let mut metadata = HashMap::new();
        metadata.insert(
            "original_sizes".to_string(),
            serde_json::json!(vec![(width, height); batch_size]),
        );
        metadata.insert("crop_coords".to_string(), serde_json::json!(vec![(0, 0); batch_size]));
        metadata.insert(
            "target_sizes".to_string(),
            serde_json::json!(vec![(width, height); batch_size]),
        );

        Ok(PreparedBatch {
            images: batch.images.clone(),
            latents: Some(latents),
            captions: batch.captions.clone(),
            metadata,
        })
    }

    fn encode_images(&self, images: &Tensor) -> anyhow::Result<Tensor> {
        if let Some(vae) = &self.vae {
            vae.encode(images)
        } else {
            Ok(images.clone())
        }
    }

    fn encode_prompts(
        &self,
        prompts: &[String],
        _model: &dyn DiffusionModel,
    ) -> anyhow::Result<PromptEmbeds> {
        if let Some(encoders) = &self.text_encoders {
            debug!("Encoding {} prompts with SDXL text encoders", prompts.len());

            // Encode with CLIP-L (768 dim)
            let (_clip_l_embeds, clip_l_pooled) = encoders.clip_l.encode(prompts)?;

            // Encode with OpenCLIP-G (1280 dim)
            let (clip_g_embeds, clip_g_pooled) = encoders.clip_g.encode(prompts)?;

            // For SDXL, we typically use only the CLIP-G embeddings for the main conditioning
            // and concatenate pooled embeddings for the added conditioning
            let encoder_hidden_states = clip_g_embeds;

            // Concatenate pooled embeddings if available
            let pooled_projections = match (clip_l_pooled.as_ref(), clip_g_pooled.as_ref()) {
                (Some(l_pooled), Some(g_pooled)) => Some(Tensor::cat(&[l_pooled, g_pooled], 1)?),
                (None, Some(g_pooled)) => Some(g_pooled.clone()),
                (Some(l_pooled), None) => Some(l_pooled.clone()),
                (None, None) => None,
            };

            Ok(PromptEmbeds { encoder_hidden_states, pooled_projections, attention_mask: None })
        } else {
            Err(Error::Config(
                "Text encoders not configured. SDXL requires CLIP-L and OpenCLIP-G encoders."
                    .to_string(),
            )
            .into())
        }
    }

    fn add_noise(
        &self,
        latents: &Tensor,
        noise: &Tensor,
        timesteps: &Tensor,
    ) -> anyhow::Result<Tensor> {
        // Apply noise offset first
        let noise = PipelineUtils::apply_noise_offset(noise, self.config.noise_offset)?;

        // Use DDPM noise addition
        self.add_noise_ddpm(latents, &noise, timesteps)
    }

    fn compute_loss(
        &self,
        model: &dyn DiffusionModel,
        noisy_latents: &Tensor,
        noise: &Tensor,
        timesteps: &Tensor,
        prompt_embeds: &PromptEmbeds,
        batch: &PreparedBatch,
    ) -> anyhow::Result<Tensor> {
        // Get model inputs with SDXL-specific conditioning
        let inputs = self.get_model_inputs(noisy_latents, timesteps, prompt_embeds, batch)?;
        let output = model.forward(&inputs)?;
        let model_pred = self.postprocess_outputs(&output)?;

        // Determine the target based on prediction type
        let target = match self.noise_scheduler.prediction_type.as_str() {
            "epsilon" => noise.clone(),
            "v_prediction" => {
                if let Some(ref latents) = batch.latents {
                    let alpha_cumprod = self.noise_scheduler.get_alpha_cumprod(timesteps)?;
                    let sqrt_alpha_cumprod =
                        alpha_cumprod.sqrt()?.unsqueeze(1)?.unsqueeze(2)?.unsqueeze(3)?;
                    let sqrt_one_minus_alpha_cumprod =
                        crate::tensor_utils::scalar_f32(1.0, alpha_cumprod.device().clone())?
                            .sub(&alpha_cumprod)?
                            .sqrt()?
                            .unsqueeze(1)?
                            .unsqueeze(2)?
                            .unsqueeze(3)?;

                    // v = sqrt(alpha_cumprod) * noise - sqrt(1 - alpha_cumprod) * latents
                    sqrt_alpha_cumprod
                        .mul(noise)?
                        .sub(&sqrt_one_minus_alpha_cumprod.mul(latents)?)?
                } else {
                    return Err(Error::Training("No latents found for v_prediction".into()).into());
                }
            }
            "sample" => {
                if let Some(ref latents) = batch.latents {
                    latents.clone()
                } else {
                    return Err(Error::Training(
                        "No latents found for sample prediction".to_string(),
                    )
                    .into());
                }
            }
            _ => {
                return Err(Error::Config(format!(
                    "Unknown prediction type: {}",
                    self.noise_scheduler.prediction_type
                ))
                .into())
            }
        };

        // Compute loss
        let loss = match self.config.loss_type.as_str() {
            "mse" => masked_eps_loss(&model_pred, &target, None)?,
            "mae" | "l1" => masked_l1_loss(&model_pred, &target, None)?,
            "huber" => {
                // Huber with delta=1.0
                let diff = model_pred.sub(&target)?;
                let abs_diff = diff.abs()?;
                // mask for quadratic region |x| <= 1
                let one = scalar_f32(1.0, abs_diff.device().clone())?;
                let quad_mask = abs_diff.le(&one)?.to_dtype(DType::F32)?;
                // linear mask = 1 - quad_mask
                let ones = Tensor::ones(quad_mask.shape().clone(), quad_mask.device().clone())?;
                let lin_mask = ones.sub(&quad_mask)?;
                // quadratic loss: 0.5 * x^2 in quad region
                let quad_loss = diff.square()?.affine(0.5, 0.0)?.mul(&quad_mask)?;
                // linear loss: |x| - 0.5 in linear region
                let lin_loss = abs_diff.affine(1.0, -0.5)?.mul(&lin_mask)?;
                policy::reduce_mean_fp32_keepdim(&quad_loss.add(&lin_loss)?)?
            }
            _ => {
                return Err(
                    Error::Config(format!("Unknown loss type: {}", self.config.loss_type)).into()
                )
            }
        };

        // Apply SNR weighting if configured
        let weighted_loss = self.apply_snr_weight(&loss, timesteps)?;

        Ok(weighted_loss)
    }

    fn get_model_inputs(
        &self,
        noisy_latents: &Tensor,
        timesteps: &Tensor,
        prompt_embeds: &PromptEmbeds,
        batch: &PreparedBatch,
    ) -> anyhow::Result<ModelInputs> {
        // Get additional conditioning for SDXL
        let conditioning = self.get_conditioning(batch, timesteps)?;

        Ok(ModelInputs {
            latents: noisy_latents.clone(),
            timestep: timesteps.clone(),
            encoder_hidden_states: Some(prompt_embeds.encoder_hidden_states.clone()),
            attention_mask: match &prompt_embeds.attention_mask {
                Some(m) => Some(m.clone()),
                None => None,
            },
            guidance_scale: Some(1.0),
            pooled_projections: match &prompt_embeds.pooled_projections {
                Some(p) => Some(p.clone()),
                None => None,
            },
            additional: conditioning,
        })
    }

    fn apply_snr_weight(&self, loss: &Tensor, timesteps: &Tensor) -> anyhow::Result<Tensor> {
        if let Some(snr_gamma) = self.config.snr_gamma {
            // Compute SNR = alpha_bar / (1 - alpha_bar) for given timesteps
            let alpha_bar = self.noise_scheduler.get_alpha_cumprod(timesteps)?; // [B]
            let one = scalar_f32(1.0, alpha_bar.device().clone())?;
            let denom = one.sub(&alpha_bar)?;
            let snr = alpha_bar.div(&denom)?;
            let cap = scalar_f32(snr_gamma, snr.device().clone())?;
            let weights = snr.minimum(&cap)?.div(&snr)?;
            return Ok(loss.mul(&weights)?);
        }
        Ok(loss.clone())
    }

    fn compute_prior_loss(
        &self,
        model: &dyn DiffusionModel,
        batch: &PreparedBatch,
    ) -> anyhow::Result<Option<Tensor>> {
        if !self.config.prior_preservation {
            return Ok(None);
        }

        // TODO: Implement prior preservation for SDXL if needed
        // This would involve generating samples with class prompts
        // and computing a regularization loss

        Ok(None)
    }

    fn get_scheduler_config(&self) -> HashMap<String, serde_json::Value> {
        let mut config = HashMap::new();
        config.insert(
            "num_train_timesteps".to_string(),
            serde_json::json!(self.noise_scheduler.num_train_timesteps),
        );
        config.insert("beta_start".to_string(), serde_json::json!(self.noise_scheduler.beta_start));
        config.insert("beta_end".to_string(), serde_json::json!(self.noise_scheduler.beta_end));
        config.insert(
            "beta_schedule".to_string(),
            serde_json::json!(self.noise_scheduler.beta_schedule),
        );
        config.insert(
            "prediction_type".to_string(),
            serde_json::json!(self.noise_scheduler.prediction_type),
        );
        config
    }
}
