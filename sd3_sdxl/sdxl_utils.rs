//! SDXL utilities and helper functions

use eridiffusion_core::{Result, Error, Device};
use candle_core::{Tensor, DType};
use std::collections::HashMap;
use tracing::{debug, warn};

/// SDXL-specific utilities
pub struct SDXLUtils;

/// Aspect ratio bucketing for SDXL training
#[derive(Debug, Clone)]
pub struct AspectRatioBucket {
    pub width: u32,
    pub height: u32,
    pub area: u32,
}

/// Aspect ratio bucketing configuration
#[derive(Debug, Clone)]
pub struct BucketingConfig {
    pub min_resolution: u32,
    pub max_resolution: u32,
    pub step_size: u32,
    pub max_aspect_ratio: f32,
    pub base_area: u32,
}

impl Default for BucketingConfig {
    fn default() -> Self {
        Self {
            min_resolution: 512,
            max_resolution: 2048,
            step_size: 64,
            max_aspect_ratio: 4.0,
            base_area: 1024 * 1024, // 1MP base area
        }
    }
}

impl SDXLUtils {
    /// Generate aspect ratio buckets for SDXL training
    pub fn generate_aspect_ratio_buckets(config: &BucketingConfig) -> Vec<AspectRatioBucket> {
        let mut buckets = Vec::new();
        
        // Generate all possible width/height combinations
        let mut width = config.min_resolution;
        while width <= config.max_resolution {
            let mut height = config.min_resolution;
            while height <= config.max_resolution {
                let aspect_ratio = width as f32 / height as f32;
                
                // Check aspect ratio constraints
                if aspect_ratio >= 1.0 / config.max_aspect_ratio && 
                   aspect_ratio <= config.max_aspect_ratio {
                    
                    let area = width * height;
                    
                    // Keep areas reasonable (within 0.5x to 2x base area)
                    if area >= config.base_area / 2 && area <= config.base_area * 2 {
                        buckets.push(AspectRatioBucket {
                            width,
                            height,
                            area,
                        });
                    }
                }
                
                height += config.step_size;
            }
            width += config.step_size;
        }
        
        // Sort by area for consistency
        buckets.sort_by_key(|b| b.area);
        
        debug!("Generated {} aspect ratio buckets", buckets.len());
        buckets
    }
    
    /// Find the best bucket for given image dimensions
    pub fn find_best_bucket(
        image_width: u32,
        image_height: u32,
        buckets: &[AspectRatioBucket],
    ) -> Option<&AspectRatioBucket> {
        let image_aspect = image_width as f32 / image_height as f32;
        let image_area = image_width * image_height;
        
        buckets.iter()
            .min_by(|a, b| {
                let a_aspect_diff = (a.width as f32 / a.height as f32 - image_aspect).abs();
                let b_aspect_diff = (b.width as f32 / b.height as f32 - image_aspect).abs();
                
                let a_area_diff = (a.area as f32 / image_area as f32 - 1.0).abs();
                let b_area_diff = (b.area as f32 / image_area as f32 - 1.0).abs();
                
                // Weighted comparison: aspect ratio is more important
                let a_score = a_aspect_diff * 2.0 + a_area_diff;
                let b_score = b_aspect_diff * 2.0 + b_area_diff;
                
                a_score.partial_cmp(&b_score).unwrap_or(std::cmp::Ordering::Equal)
            })
    }
    
    /// Apply center crop and resize for SDXL
    pub fn preprocess_image(
        image: &Tensor,
        target_width: u32,
        target_height: u32,
    ) -> Result<(Tensor, (u32, u32), (u32, u32))> {
        let dims = image.dims();
        if dims.len() != 4 {
            return Err(Error::InvalidShape("Expected 4D tensor [B, C, H, W]".to_string()));
        }
        
        let batch_size = dims[0];
        let channels = dims[1];
        let orig_height = dims[2] as u32;
        let orig_width = dims[3] as u32;
        
        // Calculate crop coordinates for center crop
        let scale_w = target_width as f32 / orig_width as f32;
        let scale_h = target_height as f32 / orig_height as f32;
        let scale = scale_w.max(scale_h); // Use the larger scale to avoid black bars
        
        let new_width = (orig_width as f32 * scale) as u32;
        let new_height = (orig_height as f32 * scale) as u32;
        
        let crop_x = if new_width > target_width {
            (new_width - target_width) / 2
        } else {
            0
        };
        let crop_y = if new_height > target_height {
            (new_height - target_height) / 2
        } else {
            0
        };
        
        // For now, return the original image with metadata
        // In a real implementation, you'd apply the actual resize and crop operations
        let processed_image = image.clone();
        
        Ok((processed_image, (orig_width, orig_height), (crop_x, crop_y)))
    }
    
    /// Encode micro-conditioning for SDXL
    pub fn encode_micro_conditioning(
        original_sizes: &[(u32, u32)],
        crop_coords: &[(u32, u32)],
        target_sizes: &[(u32, u32)],
        device: &Device,
    ) -> Result<Tensor> {
        if original_sizes.len() != crop_coords.len() || original_sizes.len() != target_sizes.len() {
            return Err(Error::InvalidShape("All conditioning arrays must have same length".to_string()));
        }
        
        let batch_size = original_sizes.len();
        
        // Concatenate all conditioning info: [orig_w, orig_h, crop_x, crop_y, target_w, target_h]
        let conditioning_data: Vec<f32> = original_sizes.iter()
            .zip(crop_coords.iter())
            .zip(target_sizes.iter())
            .flat_map(|(((orig_w, orig_h), (crop_x, crop_y)), (target_w, target_h))| {
                vec![
                    *orig_w as f32,
                    *orig_h as f32,
                    *crop_x as f32,
                    *crop_y as f32,
                    *target_w as f32,
                    *target_h as f32,
                ]
            })
            .collect();
        
        Tensor::new(conditioning_data.as_slice(), device)?
            .reshape(&[batch_size, 6])
            .map_err(|e| Error::Training(format!("Failed to reshape conditioning: {}", e)))
    }
    
    /// Create timestep embeddings with SDXL-specific dimensions
    pub fn create_timestep_embeddings(
        timesteps: &Tensor,
        embedding_dim: usize,
    ) -> Result<Tensor> {
        let device = timesteps.device();
        let batch_size = timesteps.dims()[0];
        
        // Create sinusoidal embeddings
        let half_dim = embedding_dim / 2;
        let emb_scale = 10000.0f32;
        
        // Create frequency multipliers
        let freqs: Vec<f32> = (0..half_dim)
            .map(|i| {
                let exp = -(i as f32) * (emb_scale.ln() / half_dim as f32);
                exp.exp()
            })
            .collect();
        
        let freqs_tensor = Tensor::new(freqs.as_slice(), device)?
            .unsqueeze(0)?
            .broadcast_as(&[batch_size, half_dim])?;
        
        // Compute embeddings
        let timesteps_expanded = timesteps.to_dtype(DType::F32)?
            .unsqueeze(1)?
            .broadcast_as(&[batch_size, half_dim])?;
        
        let angles = timesteps_expanded.mul(&freqs_tensor)?;
        let sin_embeds = angles.sin()?;
        let cos_embeds = angles.cos()?;
        
        // Concatenate sin and cos
        Tensor::cat(&[&sin_embeds, &cos_embeds], 1)
    }
    
    /// Apply SDXL-specific data augmentations
    pub fn apply_augmentations(
        image: &Tensor,
        caption: &str,
        config: &AugmentationConfig,
    ) -> Result<(Tensor, String)> {
        let mut augmented_image = image.clone();
        let mut augmented_caption = caption.to_string();
        
        // Random horizontal flip
        if config.horizontal_flip && rand::random::<f32>() < config.flip_probability {
            // TODO: Implement actual horizontal flip
            debug!("Applied horizontal flip");
        }
        
        // Color jittering
        if config.color_jitter {
            // TODO: Implement color jittering
            debug!("Applied color jittering");
        }
        
        // Caption dropout for classifier-free guidance
        if config.caption_dropout && rand::random::<f32>() < config.caption_dropout_probability {
            augmented_caption = String::new();
            debug!("Applied caption dropout");
        }
        
        Ok((augmented_image, augmented_caption))
    }
    
    /// Compute CLIP score for image-text alignment
    pub fn compute_clip_score(
        image_features: &Tensor,
        text_features: &Tensor,
    ) -> Result<f32> {
        // Normalize features
        let image_norm = image_features.sqr()?.sum_keepdim(&[1])?.sqrt()?;
        let text_norm = text_features.sqr()?.sum_keepdim(&[1])?.sqrt()?;
        
        let image_normalized = image_features.broadcast_div(&image_norm)?;
        let text_normalized = text_features.broadcast_div(&text_norm)?;
        
        // Compute cosine similarity
        let similarity = image_normalized.mul(&text_normalized)?.sum(&[1])?;
        let mean_similarity = similarity.mean_all()?;
        
        mean_similarity.to_scalar::<f32>()
    }
    
    /// Create noise schedule for SDXL inference
    pub fn create_inference_schedule(
        num_inference_steps: usize,
        num_train_timesteps: usize,
        beta_start: f32,
        beta_end: f32,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        // Create beta schedule
        let betas = Self::create_beta_schedule(num_train_timesteps, beta_start, beta_end, device)?;
        
        // Compute alphas
        let ones = Tensor::ones(&[num_train_timesteps], DType::F32, device)?;
        let alphas = ones.sub(&betas)?;
        
        // Compute cumulative alphas
        let mut alphas_cumprod = Vec::new();
        let alphas_vec = alphas.to_vec1::<f32>()?;
        let mut running_product = 1.0f32;
        
        for alpha in alphas_vec {
            running_product *= alpha;
            alphas_cumprod.push(running_product);
        }
        
        let alphas_cumprod_tensor = Tensor::new(alphas_cumprod, device)?;
        
        // Create inference timesteps
        let step_size = num_train_timesteps / num_inference_steps;
        let inference_timesteps: Vec<i64> = (0..num_inference_steps)
            .map(|i| (num_train_timesteps - 1 - i * step_size) as i64)
            .collect();
        
        let timesteps_tensor = Tensor::new(inference_timesteps, device)?;
        
        Ok((timesteps_tensor, alphas_cumprod_tensor))
    }
    
    /// Create beta schedule
    fn create_beta_schedule(
        num_timesteps: usize,
        beta_start: f32,
        beta_end: f32,
        device: &Device,
    ) -> Result<Tensor> {
        // Scaled linear schedule (SDXL default)
        let step = (beta_end.sqrt() - beta_start.sqrt()) / (num_timesteps - 1) as f32;
        let betas: Vec<f32> = (0..num_timesteps)
            .map(|i| {
                let beta_sqrt = beta_start.sqrt() + i as f32 * step;
                beta_sqrt * beta_sqrt
            })
            .collect();
        
        Tensor::new(betas, device)
    }
}

/// Configuration for data augmentations
#[derive(Debug, Clone)]
pub struct AugmentationConfig {
    pub horizontal_flip: bool,
    pub flip_probability: f32,
    pub color_jitter: bool,
    pub color_jitter_strength: f32,
    pub caption_dropout: bool,
    pub caption_dropout_probability: f32,
    pub random_crop: bool,
    pub center_crop: bool,
}

impl Default for AugmentationConfig {
    fn default() -> Self {
        Self {
            horizontal_flip: true,
            flip_probability: 0.5,
            color_jitter: false,
            color_jitter_strength: 0.1,
            caption_dropout: true,
            caption_dropout_probability: 0.1,
            random_crop: false,
            center_crop: true,
        }
    }
}

/// SDXL model configurations
#[derive(Debug, Clone)]
pub struct SDXLModelConfig {
    pub base_model: BaseModelConfig,
    pub refiner_model: Option<RefinerModelConfig>,
    pub vae_config: VAEConfig,
    pub text_encoder_config: TextEncoderConfig,
}

#[derive(Debug, Clone)]
pub struct BaseModelConfig {
    pub unet_channels: usize,
    pub attention_head_dim: usize,
    pub transformer_layers_per_block: Vec<usize>,
    pub cross_attention_dim: usize,
    pub use_linear_projection: bool,
    pub upcast_attention: bool,
}

#[derive(Debug, Clone)]
pub struct RefinerModelConfig {
    pub unet_channels: usize,
    pub attention_head_dim: usize,
    pub transformer_layers_per_block: Vec<usize>,
    pub cross_attention_dim: usize,
    pub conditioning_embedding_channels: usize,
}

#[derive(Debug, Clone)]
pub struct VAEConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub latent_channels: usize,
    pub scaling_factor: f32,
}

#[derive(Debug, Clone)]
pub struct TextEncoderConfig {
    pub clip_l_dim: usize,
    pub clip_g_dim: usize,
    pub max_sequence_length: usize,
    pub use_pooled_projection: bool,
}

impl Default for SDXLModelConfig {
    fn default() -> Self {
        Self {
            base_model: BaseModelConfig {
                unet_channels: 320,
                attention_head_dim: 64,
                transformer_layers_per_block: vec![2, 2, 10, 2],
                cross_attention_dim: 2048,
                use_linear_projection: true,
                upcast_attention: false,
            },
            refiner_model: Some(RefinerModelConfig {
                unet_channels: 384,
                attention_head_dim: 64,
                transformer_layers_per_block: vec![2, 2, 2, 2],
                cross_attention_dim: 1280,
                conditioning_embedding_channels: 256,
            }),
            vae_config: VAEConfig {
                in_channels: 3,
                out_channels: 3,
                latent_channels: 4,
                scaling_factor: 0.13025,
            },
            text_encoder_config: TextEncoderConfig {
                clip_l_dim: 768,
                clip_g_dim: 1280,
                max_sequence_length: 77,
                use_pooled_projection: true,
            },
        }
    }
}

/// SDXL inference utilities
pub struct SDXLInference;

impl SDXLInference {
    /// Run SDXL base model inference
    pub fn run_base_inference(
        model: &dyn DiffusionModel,
        prompt_embeds: &PromptEmbeds,
        latents: &Tensor,
        timesteps: &[i64],
        guidance_scale: f32,
        device: &Device,
    ) -> Result<Tensor> {
        let mut current_latents = latents.clone();
        
        for &timestep in timesteps {
            let timestep_tensor = Tensor::new(&[timestep], device)?;
            
            // Duplicate latents for classifier-free guidance
            let latents_input = if guidance_scale > 1.0 {
                Tensor::cat(&[&current_latents, &current_latents], 0)?
            } else {
                current_latents.clone()
            };
            
            // Create model inputs
            let inputs = ModelInputs {
                latents: latents_input,
                timestep: timestep_tensor,
                encoder_hidden_states: Some(prompt_embeds.encoder_hidden_states.clone()),
                attention_mask: prompt_embeds.attention_mask.clone(),
                guidance_scale: Some(guidance_scale),
                pooled_projections: prompt_embeds.pooled_projections.clone(),
                additional: HashMap::new(),
            };
            
            // Forward pass
            let output = model.forward(&inputs)?;
            let mut noise_pred = output.sample;
            
            // Apply classifier-free guidance
            if guidance_scale > 1.0 {
                let batch_size = current_latents.dims()[0];
                let noise_pred_uncond = noise_pred.narrow(0, 0, batch_size)?;
                let noise_pred_text = noise_pred.narrow(0, batch_size, batch_size)?;
                
                noise_pred = noise_pred_uncond.add(
                    &noise_pred_text.sub(&noise_pred_uncond)?
                        .mul(&Tensor::new(guidance_scale, device)?)?
                )?;
            }
            
            // Apply scheduler step (simplified DDIM step)
            current_latents = Self::ddim_step(&current_latents, &noise_pred, timestep, device)?;
        }
        
        Ok(current_latents)
    }
    
    /// Run SDXL refiner inference
    pub fn run_refiner_inference(
        model: &dyn DiffusionModel,
        prompt_embeds: &PromptEmbeds,
        latents: &Tensor,
        timesteps: &[i64],
        guidance_scale: f32,
        device: &Device,
    ) -> Result<Tensor> {
        // Refiner inference is similar to base but typically uses fewer steps
        // and starts from partially denoised latents
        Self::run_base_inference(model, prompt_embeds, latents, timesteps, guidance_scale, device)
    }
    
    /// DDIM sampling step
    fn ddim_step(
        latents: &Tensor,
        noise_pred: &Tensor,
        timestep: i64,
        device: &Device,
    ) -> Result<Tensor> {
        // Simplified DDIM step - in practice you'd use proper scheduler
        let alpha = 0.99f32; // Simplified alpha value
        let alpha_tensor = Tensor::new(alpha, device)?;
        
        // x_{t-1} = sqrt(alpha) * (x_t - sqrt(1-alpha) * noise_pred) / sqrt(alpha_prev) + sqrt(1-alpha_prev) * noise_pred
        let one_minus_alpha = Tensor::new(1.0f32 - alpha, device)?;
        let sqrt_alpha = alpha_tensor.sqrt()?;
        let sqrt_one_minus_alpha = one_minus_alpha.sqrt()?;
        
        let pred_original = latents.sub(
            &noise_pred.mul(&sqrt_one_minus_alpha)?
        )?.div(&sqrt_alpha)?;
        
        // For simplicity, just return a step towards the prediction
        let step_size = 0.1f32;
        latents.add(&pred_original.sub(latents)?.mul(&Tensor::new(step_size, device)?)?)
    }
}

/// SDXL training callbacks and monitoring
pub trait SDXLCallback: Send + Sync {
    fn on_step_begin(&self, step: usize) -> Result<()> { Ok(()) }
    fn on_step_end(&self, step: usize, metrics: &TrainingMetrics) -> Result<()> { Ok(()) }
    fn on_epoch_begin(&self, epoch: usize) -> Result<()> { Ok(()) }
    fn on_epoch_end(&self, epoch: usize, avg_loss: f32) -> Result<()> { Ok(()) }
    fn on_validation(&self, step: usize, val_loss: f32) -> Result<()> { Ok(()) }
}

/// Logging callback for SDXL training
pub struct LoggingCallback {
    log_frequency: usize,
}

impl LoggingCallback {
    pub fn new(log_frequency: usize) -> Self {
        Self { log_frequency }
    }
}

impl SDXLCallback for LoggingCallback {
    fn on_step_end(&self, step: usize, metrics: &TrainingMetrics) -> Result<()> {
        if step % self.log_frequency == 0 {
            info!(
                "Step {}: Loss={:.6}, Base={:.6}, Refiner={:.6}, LR={:.2e}, Time={:.2}s",
                step,
                metrics.total_loss,
                metrics.base_loss,
                metrics.refiner_loss,
                metrics.learning_rate,
                metrics.step_time
            );
        }
        Ok(())
    }
    
    fn on_validation(&self, step: usize, val_loss: f32) -> Result<()> {
        info!("Validation at step {}: Loss={:.6}", step, val_loss);
        Ok(())
    }
}

/// Checkpoint callback for SDXL training
pub struct CheckpointCallback {
    save_frequency: usize,
    checkpoint_dir: String,
}

impl CheckpointCallback {
    pub fn new(save_frequency: usize, checkpoint_dir: String) -> Self {
        Self {
            save_frequency,
            checkpoint_dir,
        }
    }
}

impl SDXLCallback for CheckpointCallback {
    fn on_step_end(&self, step: usize, _metrics: &TrainingMetrics) -> Result<()> {
        if step % self.save_frequency == 0 {
            let checkpoint_path = format!("{}/checkpoint_step_{}", self.checkpoint_dir, step);
            info!("Saving checkpoint to {}", checkpoint_path);
            // TODO: Implement actual checkpoint saving
        }
        Ok(())
    }
}

/// Wandb integration callback (placeholder)
pub struct WandbCallback {
    project_name: String,
    run_name: String,
}

impl WandbCallback {
    pub fn new(project_name: String, run_name: String) -> Self {
        Self {
            project_name,
            run_name,
        }
    }
}

impl SDXLCallback for WandbCallback {
    fn on_step_end(&self, step: usize, metrics: &TrainingMetrics) -> Result<()> {
        // TODO: Log to wandb
        debug!("Would log metrics to wandb: step={}, loss={:.6}", step, metrics.total_loss);
        Ok(())
    }
}

/// SDXL data loading utilities
pub struct SDXLDataLoader;

impl SDXLDataLoader {
    /// Create batches with aspect ratio bucketing
    pub fn create_aspect_ratio_batches(
        image_paths: &[String],
        captions: &[String],
        buckets: &[AspectRatioBucket],
        batch_size: usize,
    ) -> Result<Vec<Vec<(String, String, AspectRatioBucket)>>> {
        if image_paths.len() != captions.len() {
            return Err(Error::Training("Image paths and captions length mismatch".to_string()));
        }
        
        // Group images by bucket
        let mut bucket_groups: HashMap<usize, Vec<(String, String)>> = HashMap::new();
        
        for (image_path, caption) in image_paths.iter().zip(captions.iter()) {
            // TODO: Get actual image dimensions
            let (width, height) = (1024, 1024); // Placeholder
            
            if let Some(bucket) = SDXLUtils::find_best_bucket(width, height, buckets) {
                let bucket_idx = buckets.iter().position(|b| 
                    b.width == bucket.width && b.height == bucket.height
                ).unwrap();
                
                bucket_groups.entry(bucket_idx)
                    .or_insert_with(Vec::new)
                    .push((image_path.clone(), caption.clone()));
            }
        }
        
        // Create batches within each bucket
        let mut batches = Vec::new();
        
        for (bucket_idx, mut items) in bucket_groups {
            // Shuffle items within bucket
            use rand::seq::SliceRandom;
            let mut rng = rand::thread_rng();
            items.shuffle(&mut rng);
            
            let bucket = buckets[bucket_idx].clone();
            
            // Create batches
            for chunk in items.chunks(batch_size) {
                let batch: Vec<(String, String, AspectRatioBucket)> = chunk.iter()
                    .map(|(path, caption)| (path.clone(), caption.clone(), bucket.clone()))
                    .collect();
                
                if batch.len() == batch_size {
                    batches.push(batch);
                }
            }
        }
        
        // Shuffle batches
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        batches.shuffle(&mut rng);
        
        info!("Created {} batches with aspect ratio bucketing", batches.len());
        Ok(batches)
    }
    
    /// Load and preprocess a batch of images
    pub fn load_batch(
        batch: &[(String, String, AspectRatioBucket)],
        device: &Device,
    ) -> Result<DataLoaderBatch> {
        let batch_size = batch.len();
        let bucket = &batch[0].2; // All items in batch have same bucket
        
        // TODO: Implement actual image loading
        // For now, create dummy tensors
        let images = Tensor::zeros(
            &[batch_size, 3, bucket.height as usize, bucket.width as usize],
            DType::F32,
            device,
        )?;
        
        let captions: Vec<String> = batch.iter().map(|(_, caption, _)| caption.clone()).collect();
        
        Ok(DataLoaderBatch {
            images,
            captions,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_aspect_ratio_bucketing() {
        let config = BucketingConfig::default();
        let buckets = SDXLUtils::generate_aspect_ratio_buckets(&config);
        
        assert!(!buckets.is_empty());
        
        // Test finding best bucket
        let best_bucket = SDXLUtils::find_best_bucket(1024, 1024, &buckets);
        assert!(best_bucket.is_some());
    }
    
    #[test]
    fn test_micro_conditioning() {
        let device = Device::Cpu;
        let original_sizes = vec![(1024, 1024), (512, 768)];
        let crop_coords = vec![(0, 0), (50, 100)];
        let target_sizes = vec![(1024, 1024), (512, 768)];
        
        let conditioning = SDXLUtils::encode_micro_conditioning(
            &original_sizes,
            &crop_coords,
            &target_sizes,
            &device,
        );
        
        assert!(conditioning.is_ok());
        let cond_tensor = conditioning.unwrap();
        assert_eq!(cond_tensor.dims(), &[2, 6]);
    }
    
    #[test]
    fn test_timestep_embeddings() {
        let device = Device::Cpu;
        let timesteps = Tensor::new(&[0i64, 100, 500, 999], &device).unwrap();
        let embeddings = SDXLUtils::create_timestep_embeddings(&timesteps, 320);
        
        assert!(embeddings.is_ok());
        let emb_tensor = embeddings.unwrap();
        assert_eq!(emb_tensor.dims(), &[4, 320]);
    }
    
    #[test]
    fn test_augmentation_config() {
        let config = AugmentationConfig::default();
        assert!(config.horizontal_flip);
        assert_eq!(config.flip_probability, 0.5);
    }
    
    #[test]
    fn test_model_configs() {
        let model_config = SDXLModelConfig::default();
        assert_eq!(model_config.base_model.unet_channels, 320);
        assert_eq!(model_config.vae_config.latent_channels, 4);
        assert_eq!(model_config.text_encoder_config.clip_l_dim, 768);
    }
}

// Re-export commonly used types
pub use eridiffusion_models::{ModelInputs, ModelOutput};
pub use eridiffusion_data::DataLoaderBatch;
use super::{TrainingMetrics, PromptEmbeds};
use eridiffusion_models::DiffusionModel;
