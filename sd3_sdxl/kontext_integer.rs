//! Flux Kontext training integration with existing pipeline

use eridiffusion_core::{Result, Error, Device, ModelArchitecture};
use eridiffusion_models::{DiffusionModel, VAE, TextEncoder};
use candle_core::{Tensor, DType};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{info, debug};
use serde::{Serialize, Deserialize};

use crate::{
    TrainingPipeline, PipelineConfig, PreparedBatch, PromptEmbeds,
    DatasetManager, FluxKontextDataset, FluxKontextBatch,
    LatentCache, VAEPreprocessor, VAEConfig,
};

/// Flux Kontext training pipeline
pub struct FluxKontextTrainingPipeline {
    config: FluxKontextPipelineConfig,
    vae_preprocessor: VAEPreprocessor,
    latent_cache: Option<Arc<LatentCache>>,
    device: Device,
    flow_matching_config: FlowMatchingConfig,
}

/// Flux Kontext specific pipeline configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FluxKontextPipelineConfig {
    pub base: PipelineConfig,
    
    // Flow matching parameters
    pub base_image_seq_len: usize,
    pub base_shift: f32,
    pub max_image_seq_len: usize,
    pub max_shift: f32,
    pub shift: f32,
    pub use_dynamic_shifting: bool,
    
    // Control parameters
    pub control_conditioning_scale: f32,
    pub control_guidance_start: f32,
    pub control_guidance_end: f32,
    
    // Text encoder parameters
    pub max_sequence_length: usize,
    pub use_t5_xxl: bool,
    pub use_clip: bool,
    
    // Training parameters
    pub pack_latents: bool,
    pub use_position_ids: bool,
    pub guidance_scale: f32,
}

impl Default for FluxKontextPipelineConfig {
    fn default() -> Self {
        Self {
            base: PipelineConfig::default(),
            
            // Flow matching (from Python implementation)
            base_image_seq_len: 256,
            base_shift: 0.5,
            max_image_seq_len: 4096,
            max_shift: 1.15,
            shift: 3.0,
            use_dynamic_shifting: true,
            
            // Control conditioning
            control_conditioning_scale: 1.0,
            control_guidance_start: 0.0,
            control_guidance_end: 1.0,
            
            // Text encoders
            max_sequence_length: 512,
            use_t5_xxl: true,
            use_clip: true,
            
            // Training
            pack_latents: true,
            use_position_ids: true,
            guidance_scale: 1.0,
        }
    }
}

/// Flow matching configuration
#[derive(Debug, Clone)]
pub struct FlowMatchingConfig {
    pub num_train_timesteps: usize,
    pub beta_schedule: String,
    pub prediction_type: String,
}

impl Default for FlowMatchingConfig {
    fn default() -> Self {
        Self {
            num_train_timesteps: 1000,
            beta_schedule: "linear".to_string(),
            prediction_type: "v_prediction".to_string(),
        }
    }
}

impl FluxKontextTrainingPipeline {
    /// Create new Flux Kontext training pipeline
    pub fn new(config: FluxKontextPipelineConfig) -> Result<Self> {
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        
        // Create VAE preprocessor for Flux
        let vae_config = VAEConfig::flux_kontext();
        let vae_preprocessor = VAEPreprocessor::new(
            // You'd pass actual VAE here, for now we'll create a placeholder
            Arc::new(MockFluxVAE::new(vae_config)),
            ModelArchitecture::FluxKontext,
        )?;
        
        // Create latent cache if enabled
        let latent_cache = if config.base.cache_latents {
            Some(Arc::new(LatentCache::new(
                std::path::PathBuf::from(".cache/flux_kontext"),
                ModelArchitecture::FluxKontext,
                device.clone(),
                None,
            )?))
        } else {
            None
        };
        
        let flow_matching_config = FlowMatchingConfig::default();
        
        Ok(Self {
            config,
            vae_preprocessor,
            latent_cache,
            device,
            flow_matching_config,
        })
    }
    
    /// Prepare Flux Kontext batch for training
    pub fn prepare_flux_kontext_batch(
        &self,
        batch: FluxKontextBatch,
    ) -> Result<FluxKontextPreparedBatch> {
        // Process main images to latents
        let main_latents = if let Some(cache) = &self.latent_cache {
            // Try to get from cache first
            // For now, encode directly
            self.vae_preprocessor.encode_batch(&batch.base.images)?
        } else {
            self.vae_preprocessor.encode_batch(&batch.base.images)?
        };
        
        // Process control images if present
        let control_latents = if let Some(control_images) = &batch.control_images {
            Some(self.vae_preprocessor.encode_batch(control_images)?)
        } else {
            None
        };
        
        // Combine latents for Flux Kontext (concatenate on channel dimension)
        let combined_latents = if let Some(control) = control_latents {
            // Ensure dimensions match
            let main_dims = main_latents.dims();
            let control_dims = control.dims();
            
            if main_dims[2..] != control_dims[2..] {
                return Err(Error::InvalidShape(format!(
                    "Main and control latent spatial dimensions don't match: {:?} vs {:?}",
                    &main_dims[2..], &control_dims[2..]
                )));
            }
            
            // Concatenate: [batch, 32, h, w] (16 + 16 channels)
            Tensor::cat(&[&main_latents, &control], 1)?
        } else {
            main_latents
        };
        
        // Pack latents for transformer if enabled
        let packed_latents = if self.config.pack_latents {
            self.pack_latents_for_transformer(&combined_latents)?
        } else {
            combined_latents
        };
        
        // Generate position IDs if enabled
        let (img_ids, txt_ids) = if self.config.use_position_ids {
            self.generate_position_ids(&packed_latents, &batch.base.captions)?
        } else {
            (None, None)
        };
        
        Ok(FluxKontextPreparedBatch {
            latents: packed_latents,
            captions: batch.base.captions,
            control_strengths: batch.control_strengths,
            has_control: batch.has_control,
            img_ids,
            txt_ids,
            original_size: extract_sizes_from_metadata(&batch.base.metadata, "original_size"),
            crop_coords: extract_sizes_from_metadata(&batch.base.metadata, "crop_coords"),
            metadata: batch.base.metadata,
        })
    }
    
    /// Pack latents for transformer input: [B, C, H, W] -> [B, seq_len, packed_dim]
    fn pack_latents_for_transformer(&self, latents: &Tensor) -> Result<Tensor> {
        let (batch, channels, height, width) = match latents.dims() {
            [b, c, h, w] => (*b, *c, *h, *w),
            _ => return Err(Error::InvalidShape("Expected BCHW format".into())),
        };
        
        let patch_size = 2; // Flux uses 2x2 patches
        
        if height % patch_size != 0 || width % patch_size != 0 {
            return Err(Error::InvalidShape(format!(
                "Height and width must be divisible by patch size {}", patch_size
            )));
        }
        
        let seq_h = height / patch_size;
        let seq_w = width / patch_size;
        let seq_len = seq_h * seq_w;
        let packed_dim = channels * patch_size * patch_size;
        
        // Reshape to patches: [B, C, H, W] -> [B, seq_len, packed_dim]
        let reshaped = latents
            .reshape(&[batch, channels, seq_h, patch_size, seq_w, patch_size])?
            .transpose(3, 4)? // [B, C, seq_h, seq_w, patch_size, patch_size]
            .reshape(&[batch, seq_len, packed_dim])?;
        
        Ok(reshaped)
    }
    
    /// Generate position IDs for transformer
    fn generate_position_ids(
        &self,
        packed_latents: &Tensor,
        captions: &[String],
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let batch_size = packed_latents.dims()[0];
        let seq_len = packed_latents.dims()[1];
        
        // Create image position IDs
        let height = (seq_len as f32).sqrt() as usize;
        let width = seq_len / height;
        
        let mut img_ids_data = Vec::with_capacity(seq_len * 3);
        for h in 0..height {
            for w in 0..width {
                img_ids_data.push(0.0f32); // type_id (0 for main image)
                img_ids_data.push(h as f32); // height_id
                img_ids_data.push(w as f32); // width_id
            }
        }
        
        let candle_device = match &self.device {
            Device::Cpu => candle_core::Device::Cpu,
            Device::Cuda(id) => candle_core::Device::new_cuda(*id)?,
        };
        
        let img_ids = Tensor::from_vec(
            img_ids_data,
            &[1, seq_len, 3],
            &candle_device,
        )?.broadcast_as(&[batch_size, seq_len, 3])?;
        
        // Create text position IDs (zeros for now)
        let max_text_len = self.config.max_sequence_length;
        let txt_ids = Tensor::zeros(
            &[batch_size, max_text_len, 3],
            DType::F32,
            &candle_device,
        )?;
        
        Ok((Some(img_ids), Some(txt_ids)))
    }
    
    /// Apply dynamic shifting for flow matching timesteps
    pub fn apply_dynamic_shifting(&self, timesteps: &Tensor, seq_len: usize) -> Result<Tensor> {
        if !self.config.use_dynamic_shifting {
            return Ok(timesteps.clone());
        }
        
        let device = timesteps.device();
        
        // Calculate shift factor based on sequence length
        let shift_factor = if seq_len <= self.config.base_image_seq_len {
            self.config.base_shift
        } else if seq_len >= self.config.max_image_seq_len {
            self.config.max_shift
        } else {
            let ratio = (seq_len - self.config.base_image_seq_len) as f32 /
                       (self.config.max_image_seq_len - self.config.base_image_seq_len) as f32;
            self.config.base_shift + ratio * (self.config.max_shift - self.config.base_shift)
        };
        
        // Apply shifting: t_shifted = t + shift_factor * (1 - t) * t
        let t_normalized = timesteps.broadcast_div(&Tensor::new(1000.0f32, device)?)?;
        let one_minus_t = Tensor::new(1.0f32, device)?.broadcast_sub(&t_normalized)?;
        let shift_term = t_normalized.mul(&one_minus_t)?.affine(shift_factor as f64, 0.0)?;
        let shifted = t_normalized.add(&shift_term)?;
        
        // Scale back to [0, 1000]
        Ok(shifted.affine(1000.0, 0.0)?)
    }
    
    /// Compute flow matching loss
    pub fn compute_flow_matching_loss(
        &self,
        model_output: &Tensor,
        target: &Tensor,
        timesteps: &Tensor,
    ) -> Result<Tensor> {
        // Flow matching loss is typically MSE between predicted and target velocity
        let diff = model_output.sub(target)?;
        let loss = diff.sqr()?.mean_all()?;
        
        // Apply any timestep weighting if needed
        Ok(loss)
    }
}

/// Prepared batch for Flux Kontext training
#[derive(Debug, Clone)]
pub struct FluxKontextPreparedBatch {
    pub latents: Tensor,
    pub captions: Vec<String>,
    pub control_strengths: Vec<f32>,
    pub has_control: Vec<bool>,
    pub img_ids: Option<Tensor>,
    pub txt_ids: Option<Tensor>,
    pub original_size: Vec<(u32, u32)>,
    pub crop_coords: Vec<(u32, u32)>,
    pub metadata: HashMap<String, Vec<serde_json::Value>>,
}

impl FluxKontextPreparedBatch {
    pub fn batch_size(&self) -> usize {
        self.latents.dims()[0]
    }
    
    pub fn to_device(&self, device: &Device) -> Result<Self> {
        let candle_device = match device {
            Device::Cpu => candle_core::Device::Cpu,
            Device::Cuda(id) => candle_core::Device::new_cuda(*id)?,
        };
        
        let latents = self.latents.to_device(&candle_device)?;
        let img_ids = self.img_ids.as_ref()
            .map(|ids| ids.to_device(&candle_device))
            .transpose()?;
        let txt_ids = self.txt_ids.as_ref()
            .map(|ids| ids.to_device(&candle_device))
            .transpose()?;
        
        Ok(Self {
            latents,
            captions: self.captions.clone(),
            control_strengths: self.control_strengths.clone(),
            has_control: self.has_control.clone(),
            img_ids,
            txt_ids,
            original_size: self.original_size.clone(),
            crop_coords: self.crop_coords.clone(),
            metadata: self.metadata.clone(),
        })
    }
}

/// Mock VAE for testing (you'd replace this with actual VAE implementation)
struct MockFluxVAE {
    config: VAEConfig,
}

impl MockFluxVAE {
    fn new(config: VAEConfig) -> Self {
        Self { config }
    }
}

impl VAE for MockFluxVAE {
    fn encode(&self, input: &Tensor) -> Result<Tensor> {
        let (batch, channels, height, width) = match input.dims() {
            [b, c, h, w] => (*b, *c, *h, *w),
            _ => return Err(Error::InvalidShape("Expected BCHW format".into())),
        };
        
        let latent_h = height / self.config.downsampling_factor;
        let latent_w = width / self.config.downsampling_factor;
        
        // Create placeholder latent (normally this would be actual VAE encoding)
        Tensor::randn(
            0.0f32,
            1.0,
            &[batch, self.config.latent_channels, latent_h, latent_w],
            input.device(),
        ).map_err(Error::from)
    }
    
    fn decode(&self, latent: &Tensor) -> Result<Tensor> {
        let (batch, channels, height, width) = match latent.dims() {
            [b, c, h, w] => (*b, *c, *h, *w),
            _ => return Err(Error::InvalidShape("Expected BCHW format".into())),
        };
        
        let image_h = height * self.config.downsampling_factor;
        let image_w = width * self.config.downsampling_factor;
        
        // Create placeholder image
        Tensor::randn(
            0.0f32,
            1.0,
            &[batch, 3, image_h, image_w],
            latent.device(),
        ).map_err(Error::from)
    }
    
    fn encode_deterministic(&self, input: &Tensor) -> Result<Tensor> {
        self.encode(input)
    }
    
    fn latent_channels(&self) -> usize {
        self.config.latent_channels
    }
    
    fn downsampling_factor(&self) -> usize {
        self.config.downsampling_factor
    }
}

/// Helper function to extract size tuples from metadata
fn extract_sizes_from_metadata(
    metadata: &HashMap<String, Vec<serde_json::Value>>,
    key: &str,
) -> Vec<(u32, u32)> {
    metadata.get(key)
        .map(|values| {
            values.iter()
                .map(|v| {
                    v.as_array()
                        .and_then(|arr| {
                            let w = arr.get(0)?.as_u64()? as u32;
                            let h = arr.get(1)?.as_u64()? as u32;
                            Some((w, h))
                        })
                        .unwrap_or((1024, 1024))
                })
                .collect()
        })
        .unwrap_or_else(|| vec![(1024, 1024); 1])
}

/// Integration with existing dataset manager
impl DatasetManager {
    /// Create Flux Kontext dataset manager
    pub fn flux_kontext(
        dataset_path: std::path::PathBuf,
        control_image_dir: Option<std::path::PathBuf>,
        vae: Option<Arc<dyn VAE>>,
    ) -> Result<Self> {
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        
        // Create Flux Kontext dataset config
        let flux_config = crate::FluxKontextDatasetConfig {
            control_image_dir,
            ..Default::default()
        };
        
        flux_config.base.root_dir = dataset_path;
        
        // Create Flux Kontext dataset
        let dataset = Box::new(FluxKontextDataset::new(flux_config)?);
        
        // Create latent cache for Flux
        let latent_cache = if vae.is_some() {
            let cache_dir = std::path::PathBuf::from(".cache/latents")
                .join("flux_kontext");
            Some(Arc::new(LatentCache::new(
                cache_dir,
                ModelArchitecture::FluxKontext,
                device.clone(),
                None,
            )?))
        } else {
            None
        };
        
        // Create Flux preprocessor
        let preprocessor = Box::new(crate::FluxKontextPreprocessor::new(
            control_image_dir.is_some()
        ));
        
        let resolution_config = crate::ResolutionConfig::flux_kontext();
        
        Ok(Self {
            architecture: ModelArchitecture::FluxKontext,
            dataset,
            vae,
            latent_cache,
            resolution_config,
            preprocessor,
            device,
            stats: Arc::new(tokio::sync::RwLock::new(crate::DatasetStats::default())),
        })
    }
}
