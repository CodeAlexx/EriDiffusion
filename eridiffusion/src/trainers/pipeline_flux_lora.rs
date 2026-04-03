//! Complete Flux training pipeline with LoRA and full fine-tuning support

use crate::inference::flux::{FluxConfig as InferenceConfig, FluxInference};
use crate::inference::ModelConfig;
use crate::loaders::WeightLoader;
use crate::models::{
    flux_lora_wrapper::FluxModelWithLoRA,
    flux_model_complete::{FluxModel, FluxModelConfig},
    flux_vae::{AutoEncoderConfig as VAEConfig, AutoencoderKL},
};
use crate::samplers::flame_schedulers::SchedulerStepOutput;
use crate::trainers::{
    adam8bit::Adam8bit,
    adam8bit_enhanced::Adam8bitConfig,
    flux_cache_manager::FluxCacheManager,
    flux_data_loader::{DatasetConfig, FluxDataLoader, TrainingSample},
    gradient_accumulator::GradientAccumulator,
    text_encoders::TextEncoders,
};
use flame_core::device::Device;
use flame_core::{DType, Parameter, Result, Shape, Tensor};
use log::info;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Flux training configuration
#[derive(Clone)]
pub struct FluxTrainingConfig {
    // Model configuration
    pub model_path: PathBuf,
    pub vae_path: PathBuf,
    pub text_encoder_paths: TextEncoderPaths,

    // Training configuration
    pub train_mode: TrainMode,
    pub batch_size: usize,
    pub gradient_accumulation_steps: usize,
    pub learning_rate: f64,
    pub warmup_steps: usize,
    pub max_train_steps: usize,
    pub checkpointing_steps: usize,

    // Optimization
    pub mixed_precision: bool,
    pub gradient_checkpointing: bool,
    pub use_8bit_adam: bool,
    pub max_grad_norm: f32,

    // LoRA configuration (if applicable)
    pub lora_rank: usize,
    pub lora_alpha: f32,
    pub lora_dropout: f32,
    pub lora_target_modules: Vec<String>,

    // Data configuration
    pub resolution: usize,
    pub center_crop: bool,
    pub random_flip: bool,
    pub caption_dropout_rate: f32,

    // Flux-specific
    pub guidance_scale: f32,
    pub bypass_guidance_embedding: bool,
    pub shift_schedule: f32, // Shifted sigmoid parameter

    // Logging
    pub logging_dir: PathBuf,
    pub report_to: Vec<String>,
    pub validation_prompts: Vec<String>,
    pub validation_steps: usize,

    // Caching configuration
    pub cache_latents_to_disk: bool,
    pub cache_dir: Option<PathBuf>,
    pub force_reencode: bool,
    pub dataset_name: String, // Dataset name for organizing cache files

    // Layer streaming configuration
    pub use_layer_streaming: Option<bool>,
    pub streaming_memory_limit_gb: Option<f32>,

    // INT8 quantization for base model (reduces 23GB to ~11GB)
    pub use_int8_base_model: Option<bool>,

    // ChromaXL configuration
    pub chroma_config: Option<ChromaXLConfig>,
}

/// ChromaXL-style layer-specific training configuration
#[derive(Clone)]
pub struct ChromaXLConfig {
    pub layer_pattern: String, // "early", "middle", "late", "attention", "chroma_default"
    pub layer_lr_multipliers: Option<HashMap<String, f32>>, // Layer-specific LR multipliers
    pub ramp_double_blocks: bool,
    pub ramp_target_lr: f32,
    pub ramp_warmup_steps: usize,
    pub ramp_type: String, // "linear" or "cosine"
}

#[derive(Clone)]
pub struct TextEncoderPaths {
    pub clip_l: PathBuf,
    pub t5_xxl: Option<PathBuf>, // Make T5-XXL optional
}

#[derive(Clone, PartialEq, Debug)]
pub enum TrainMode {
    LoRA,
    FullFineTune,
}

/// LoRA layer for Flux
#[derive(Clone)]
pub struct FluxLoRALayer {
    pub lora_down: Parameter,
    pub lora_up: Parameter,
    pub scale: f32,
    pub dropout: f32,
    pub rank: usize,
}

impl FluxLoRALayer {
    pub fn new(
        in_features: usize,
        out_features: usize,
        rank: usize,
        alpha: f32,
        dropout: f32,
        device: &Device,
    ) -> flame_core::Result<Self> {
        let scale = alpha / rank as f32;

        // Initialize LoRA matrices as Parameters for gradient tracking
        // Use a much smaller initialization scale to prevent extreme values
        // Standard LoRA initialization: down ~ N(0, 1/sqrt(r)), up ~ zeros
        let std = 1.0 / (rank as f32).sqrt();
        let lora_down = Parameter::randn(
            Shape::from_dims(&[rank, in_features]),
            0.0,
            std * 0.001, // Much smaller init for Flux to prevent explosion
            device.cuda_device_arc(),
        )?;

        // Initialize LoRA up with small random values to prevent zero-induced explosions
        // Zero initialization can cause numerical instability in attention/normalization layers
        let lora_up = Parameter::randn(
            Shape::from_dims(&[out_features, rank]),
            0.0,
            std * 0.0001, // Very small but non-zero initialization
            device.cuda_device_arc(),
        )?;

        // Assert: Initial LoRA weights should be small
        debug_assert!(
            {
                let down_tensor = lora_down.tensor()?;
                let up_tensor = lora_up.tensor()?;
                let down_max = down_tensor.max_all().unwrap_or(f32::INFINITY);
                let up_max = up_tensor.max_all().unwrap_or(f32::INFINITY);
                down_max.abs() < 0.1 && up_max.abs() < 0.01
            },
            "LoRA initialization out of expected range"
        );

        Ok(Self { lora_down, lora_up, scale, dropout, rank })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        base_output: &Tensor,
        training: bool,
    ) -> flame_core::Result<Tensor> {
        // Get tensor values from parameters while maintaining gradient graph
        let lora_down_tensor = self.lora_down.as_tensor()?;
        let lora_up_tensor = self.lora_up.as_tensor()?;

        // Apply dropout during training
        let lora_out = if self.dropout > 0.0 && training {
            // Apply dropout to input
            // FLAME doesn't have rand_like, create uniform random tensor
            let dropout_mask = Tensor::randn(x.shape().clone(), 0.5, 0.289, x.device().clone())?
                .add_scalar(0.5)?;
            // Create a mask where values < (1.0 - dropout) are 1.0, else 0.0
            let threshold = Tensor::full(
                dropout_mask.shape().clone(),
                1.0 - self.dropout,
                dropout_mask.device().clone(),
            )?;
            let dropout_binary = dropout_mask.le(&threshold)?;
            let x_dropout = x.mul(&dropout_binary)?;
            x_dropout.matmul(&lora_down_tensor.transpose_dims(0, 1)?)?
        } else {
            x.matmul(&lora_down_tensor.transpose_dims(0, 1)?)?
        };

        let lora_out = lora_out.matmul(&lora_up_tensor.transpose_dims(0, 1)?)?;
        let scaled_lora = lora_out.mul_scalar(self.scale as f32)?;

        // 🔒 Assert LoRA delta magnitude is reasonable
        debug_assert!(
            {
                let delta_mean =
                    scaled_lora.abs()?.mean()?.to_scalar::<f32>().unwrap_or(f32::INFINITY);
                if delta_mean > 100.0 {
                    eprintln!(
                        "❌ LoRA delta too large, possible misinit or accumulation: mean={:.2e}",
                        delta_mean
                    );
                    false
                } else {
                    true
                }
            },
            "LoRA delta magnitude check failed"
        );

        // Assert: Base model output should be reasonable
        debug_assert!(
            {
                let base_max = base_output.max_all().unwrap_or(f32::INFINITY);
                let base_min = base_output.min_all().unwrap_or(f32::NEG_INFINITY);
                base_max.is_finite()
                    && base_min.is_finite()
                    && base_max.abs() < 1e6
                    && base_min.abs() < 1e6
            },
            "Base model output out of range"
        );

        // Assert: LoRA contribution should be small
        debug_assert!(
            {
                let lora_max = scaled_lora.max_all().unwrap_or(f32::INFINITY);
                let lora_min = scaled_lora.min_all().unwrap_or(f32::NEG_INFINITY);
                lora_max.is_finite()
                    && lora_min.is_finite()
                    && lora_max.abs() < 100.0
                    && lora_min.abs() < 100.0
            },
            "LoRA output too large"
        );

        // CRITICAL: Detach base output to prevent gradients flowing into frozen base model
        // Only LoRA parameters should receive gradients
        base_output.detach()?.add(&scaled_lora)
    }
}

/// Training batch
pub struct TrainingBatch {
    pub images: Tensor,
    pub prompts: Vec<String>,
    pub timesteps: Option<Tensor>,
    pub image_paths: Vec<PathBuf>, // Add paths for cache lookups
    // Additional fields for cached data
    pub pixel_values: Option<Tensor>,                 // VAE latents
    pub encoder_hidden_states: Option<Tensor>,        // T5 embeddings
    pub pooled_encoder_hidden_states: Option<Tensor>, // CLIP pooled embeddings
}

/// Prediction type for the model
#[derive(Clone, Copy, Debug)]
pub enum PredictionType {
    Epsilon,
    VPrediction,
}

/// Simple noise scheduler for Flux
pub struct FluxNoiseScheduler {
    pub device: Device,
    pub shift: f32,
    pub prediction_type: PredictionType,
}

impl FluxNoiseScheduler {
    pub fn new(device: Device, shift: f32) -> Self {
        Self {
            device,
            shift,
            prediction_type: PredictionType::VPrediction, // Flux uses v-prediction
        }
    }

    /// Get sigma from timestep using shifted sigmoid
    fn sigma(&self, t: f32) -> f32 {
        let shifted_t = self.shift * (2.0 * t - 1.0);
        1.0 / (1.0 + (-shifted_t).exp())
    }

    /// Add noise using flow matching
    pub fn add_noise(
        &self,
        latents: &Tensor,
        noise: &Tensor,
        timesteps: &Tensor,
    ) -> flame_core::Result<(Tensor, Tensor)> {
        let batch_size = timesteps.shape().dims()[0];
        let mut noisy_latents = Vec::new();
        let mut velocities = Vec::new();

        let t_data = timesteps.to_vec1::<f32>()?;

        for i in 0..batch_size {
            let t = t_data[i] / 1000.0; // Normalize to [0, 1] for sigma calculation
            let sigma_t = self.sigma(t);

            // Flow matching: x_t = (1 - sigma_t) * x_0 + sigma_t * epsilon
            let latent_shape = latents.shape();
            let dims = latent_shape.dims();

            // Handle both 3D and 4D latents
            let latent_slice = if dims.len() == 3 {
                // 3D tensor: [channels, height, width] - take the whole tensor as a single batch
                if i == 0 {
                    latents.clone()
                } else {
                    continue; // Skip if batch_size > 1 for 3D tensor
                }
            } else {
                // 4D tensor: [batch, channels, height, width]
                latents.slice(&[(i, i + 1), (0, dims[1]), (0, dims[2]), (0, dims[3])])?
            };
            let noise_shape = noise.shape();
            let noise_dims = noise_shape.dims();

            // Handle both 3D and 4D noise tensors
            let noise_slice = if noise_dims.len() == 3 {
                // 3D tensor: [channels, height, width] - take the whole tensor as a single batch
                if i == 0 {
                    noise.clone()
                } else {
                    continue; // Skip if batch_size > 1 for 3D tensor
                }
            } else {
                // 4D tensor: [batch, channels, height, width]
                noise.slice(&[
                    (i, i + 1),
                    (0, noise_dims[1]),
                    (0, noise_dims[2]),
                    (0, noise_dims[3]),
                ])?
            };

            let scaled_latent = latent_slice.mul_scalar(1.0 - sigma_t)?;
            let scaled_noise = noise_slice.mul_scalar(sigma_t)?;
            let noisy = scaled_latent.add(&scaled_noise)?;

            // Velocity v = epsilon - x_0
            let velocity = noise_slice.sub(&latent_slice)?;

            noisy_latents.push(noisy);
            velocities.push(velocity);
        }

        let noisy_latents = Tensor::cat(&noisy_latents.iter().collect::<Vec<_>>(), 0)?;
        let velocities = Tensor::cat(&velocities.iter().collect::<Vec<_>>(), 0)?;

        Ok((noisy_latents, velocities))
    }

    /// Compute loss based on prediction type
    pub fn compute_loss(
        &self,
        model_pred: &Tensor,
        target: &Tensor,
        timesteps: &Tensor,
    ) -> flame_core::Result<Tensor> {
        match self.prediction_type {
            PredictionType::Epsilon => {
                // Standard MSE loss for epsilon prediction
                model_pred.sub(target)?.square()?.mean()
            }
            PredictionType::VPrediction => {
                // MSE loss for velocity prediction
                model_pred.sub(target)?.square()?.mean()
            }
        }
    }

    /// Sample timesteps for training
    pub fn sample_timesteps(&self, batch_size: usize) -> flame_core::Result<Tensor> {
        // Uniform sampling for flow matching
        // FLAME uses randn for random normal distribution
        Tensor::randn(Shape::from_dims(&[batch_size]), 0.0, 1.0, self.device.cuda_device().clone())
    }

    /// Get scheduler state for checkpointing
    pub fn state_dict(&self) -> HashMap<String, Tensor> {
        HashMap::new() // No state to save for this simple scheduler
    }

    pub fn step(
        &self,
        model_output: &Tensor,
        timestep: &Tensor,
        sample: &Tensor,
        generator: Option<&mut rand::rngs::StdRng>,
    ) -> flame_core::Result<SchedulerStepOutput> {
        // For inference - implement flow matching step
        let sigma_t = self.sigma(timestep.to_scalar::<f32>()?);

        // Update sample based on model prediction
        let pred_sample = match self.prediction_type {
            PredictionType::VPrediction => {
                // x_t - sigma_t * v_t
                sample.sub(&model_output.mul_scalar(sigma_t)?)?
            }
            PredictionType::Epsilon => {
                // (x_t - sigma_t * eps) / (1 - sigma_t)
                let numerator = sample.sub(&model_output.mul_scalar(sigma_t)?)?;
                numerator.div_scalar(1.0 - sigma_t)?
            }
        };

        Ok(SchedulerStepOutput { prev_sample: pred_sample, pred_original_sample: None })
    }
}

/// Data loader trait
pub trait DataLoader: Iterator<Item = Result<TrainingBatch>> {
    fn len(&self) -> usize;
    fn total_samples(&self) -> usize;
    fn enumerate(self) -> std::iter::Enumerate<Self>
    where
        Self: Sized,
    {
        Iterator::enumerate(self)
    }
}

/// Flux Training Pipeline
pub struct FluxTrainer {
    pub config: FluxTrainingConfig,
    pub device: Device,

    // Models (may be None if using caching)
    pub vae: Option<Arc<AutoencoderKL>>,
    pub flux_with_lora: Option<FluxModelWithLoRA>, // LoRA-wrapped model
    pub flux_base: Option<FluxModel>,              // Base model for full fine-tuning
    pub text_encoders: Option<Arc<TextEncoders>>,

    // LoRA layers (if applicable)
    pub lora_layers: Option<HashMap<String, FluxLoRALayer>>,

    // Optimizer
    pub optimizer: Adam8bit,

    // Gradient accumulator
    pub gradient_accumulator: GradientAccumulator,

    // Scheduler
    pub noise_scheduler: FluxNoiseScheduler,

    // Cache manager
    pub cache_manager: Option<FluxCacheManager>,

    // Stats
    pub global_step: usize,
    pub epoch: usize,

    // Last gradient map from backward pass
    pub last_grad_map: Option<flame_core::gradient::GradientMap>,
}

impl FluxTrainer {
    /// Create new Flux trainer with SimpleTuner-style memory management
    pub fn new(config: FluxTrainingConfig, device: Device) -> flame_core::Result<Self> {
        // Setup cache manager if enabled
        let cache_manager = if config.cache_latents_to_disk {
            let cache_dir =
                config.cache_dir.clone().unwrap_or_else(|| config.logging_dir.join("cache"));
            println!("Initializing cache manager at: {:?}", cache_dir);
            println!("  Dataset name: {}", config.dataset_name);
            Some(FluxCacheManager::with_dataset_name(
                cache_dir,
                device.clone(),
                true,
                config.dataset_name.clone(),
            )?)
        } else {
            None
        };

        // SimpleTuner-style loading: Don't load VAE and text encoders yet if caching is enabled
        let (vae, text_encoders) = if config.cache_latents_to_disk {
            println!("SimpleTuner-style loading enabled - VAE and text encoders will be loaded temporarily for encoding");
            (None, None)
        } else {
            // Traditional loading - keep everything in memory
            println!("Loading VAE from: {:?}", config.vae_path);
            let vae = Some(Arc::new(load_vae(&config.vae_path, &device)?));
            println!("✅ VAE loaded successfully");

            // Check if we should skip text encoders
            let skip_text = std::env::var("SKIP_T5_ENCODER").is_ok()
                || std::env::var("USE_DUMMY_EMBEDDINGS").is_ok();

            let text_encoders = if skip_text {
                println!("⚠️  Skipping text encoder loading - will use dummy embeddings");
                None
            } else {
                println!("Loading text encoders...");
                let encoders =
                    Some(Arc::new(load_text_encoders(&config.text_encoder_paths, &device)?));
                println!("✅ Text encoders loaded successfully");
                encoders
            };

            (vae, text_encoders)
        };

        // DEFER loading the Flux model until AFTER pre-encoding to save memory!
        // This is CRITICAL for 24GB GPUs - we need memory for text encoding first
        println!("⚠️  DEFERRING Flux model loading until after pre-encoding phase");
        println!("   This allows CLIP/T5 to encode text first, then free memory for Flux");

        // Initialize as None - will be loaded after pre-encoding
        let (flux_with_lora, flux_base, lora_layers) = (None, None, None);

        // Create optimizer
        let trainable_params = get_trainable_parameters(&flux_base, &lora_layers, &config);
        let optimizer = create_optimizer(trainable_params, &config)?;

        // Gradient accumulator
        let gradient_accumulator =
            GradientAccumulator::new(config.gradient_accumulation_steps, device.clone());

        // Noise scheduler
        let noise_scheduler = FluxNoiseScheduler::new(device.clone(), config.shift_schedule);

        Ok(Self {
            config,
            device,
            vae,
            flux_with_lora,
            flux_base,
            text_encoders,
            lora_layers,
            optimizer,
            gradient_accumulator,
            noise_scheduler,
            cache_manager,
            global_step: 0,
            epoch: 0,
            last_grad_map: None,
        })
    }

    /// Pre-encode all data (SimpleTuner style)
    pub fn pre_encode_data(&mut self, data_loader: &mut FluxDataLoader) -> flame_core::Result<()> {
        if let Some(ref cache_manager) = self.cache_manager {
            println!("\n=== SimpleTuner-style pre-encoding phase ===");

            // Phase 1: Encode all latents
            println!("\nPhase 1: Encoding latents with VAE");
            cache_manager.encode_all_latents(
                data_loader,
                &self.config.vae_path,
                self.config.force_reencode,
            )?;

            // CRITICAL: Force complete memory cleanup between phases
            println!("\n=== Forcing complete GPU memory cleanup before text encoding ===");
            self.device.synchronize()?;

            // Clear memory pool
            if let Ok(pool) =
                flame_core::memory_pool::MEMORY_POOL.get_pool(&self.device.cuda_device_arc())
            {
                if let Ok(mut pool_guard) = pool.lock() {
                    pool_guard.clear_cache();
                    let _ = pool_guard.force_cleanup();
                }
            }

            // Add delay to ensure memory is truly freed
            std::thread::sleep(std::time::Duration::from_secs(2));

            // Check memory status
            println!("GPU memory status before text encoding:");
            std::process::Command::new("nvidia-smi")
                .arg("--query-gpu=memory.used,memory.free")
                .arg("--format=csv,noheader")
                .status()
                .expect("Failed to run nvidia-smi");

            // Phase 2: Encode all text embeddings
            println!("\nPhase 2: Encoding text embeddings");
            let t5_path = if let Some(t5_path_buf) = &self.config.text_encoder_paths.t5_xxl {
                if t5_path_buf.exists() {
                    Some(t5_path_buf.as_path())
                } else {
                    None
                }
            } else {
                None
            };

            cache_manager.encode_all_text_embeddings(
                data_loader,
                &self.config.text_encoder_paths.clip_l,
                t5_path,
                self.config.force_reencode,
            )?;

            // Report cache statistics
            let (latent_count, embed_count) = cache_manager.get_stats()?;
            println!("\n✅ Pre-encoding complete!");
            println!("   - Cached latents: {}", latent_count);
            println!("   - Cached embeddings: {}", embed_count);

            // CRITICAL: Force complete memory cleanup before model loading
            println!("Forcing GPU memory cleanup...");
            self.device.synchronize()?;

            // Clear the memory manager's cache
            use crate::memory::manager::MemoryManager;
            MemoryManager::empty_cache()?;

            // Sleep to ensure CUDA actually releases memory
            std::thread::sleep(std::time::Duration::from_secs(2));

            println!("   - GPU memory has been forcefully cleared for model loading");
        }

        Ok(())
    }

    /// Load Flux model after pre-encoding phase
    fn load_flux_model_for_training(&mut self) -> flame_core::Result<()> {
        if self.flux_with_lora.is_some() || self.flux_base.is_some() {
            println!("Flux model already loaded");
            return Ok(());
        }

        println!("\n=== Loading Flux model for training ===");
        println!("Memory is now free after pre-encoding phase");
        println!("Loading Flux model from: {:?}", self.config.model_path);

        let flux = load_flux_model(&self.config.model_path, &self.device)?;
        println!("✅ Flux model loaded successfully");

        // Setup LoRA if needed
        let (flux_with_lora, flux_base, lora_layers) = if self.config.train_mode == TrainMode::LoRA
        {
            let (flux_with_lora_layers, lora_layers) = setup_flux_lora(
                flux,
                self.config.lora_rank,
                self.config.lora_alpha,
                self.config.lora_dropout,
                &self.config.lora_target_modules,
                &self.device,
            )?;
            let flux_with_lora = FluxModelWithLoRA::new(flux_with_lora_layers, lora_layers.clone());
            (Some(flux_with_lora), None, Some(lora_layers))
        } else {
            (None, Some(flux), None)
        };

        self.flux_with_lora = flux_with_lora;
        self.flux_base = flux_base;
        self.lora_layers = lora_layers;

        // Re-create optimizer with the now-loaded parameters
        let trainable_params =
            get_trainable_parameters(&self.flux_base, &self.lora_layers, &self.config);
        self.optimizer = create_optimizer(trainable_params, &self.config)?;

        Ok(())
    }

    /// Main training loop
    pub fn train(&mut self, data_loader: &mut FluxDataLoader) -> flame_core::Result<()> {
        println!("Starting Flux training...");

        // Load the Flux model NOW (after pre-encoding freed memory)
        self.load_flux_model_for_training()?;

        // Enable gradient checkpointing if requested
        if self.config.gradient_checkpointing {
            // Set up gradient checkpointing policy
            let mut checkpoint_manager =
                flame_core::gradient_checkpointing::CHECKPOINT_MANAGER.lock().unwrap();

            // Use CPU offload for better accuracy preservation during training
            checkpoint_manager
                .set_policy(flame_core::gradient_checkpointing::CheckpointPolicy::CPUOffload);

            // Initialize with the current device
            // Device is already the correct type
            checkpoint_manager.set_device(self.device.cuda_device().clone());

            println!("Gradient checkpointing enabled for Flux model");
            println!("  - Policy: CPU offload (preserves accuracy)");
            println!("  - This will reduce GPU memory usage at the cost of some speed");
        }

        let num_batches = data_loader.len();
        let num_update_steps_per_epoch = num_batches / self.config.gradient_accumulation_steps;
        let total_batch_size = self.config.batch_size * self.config.gradient_accumulation_steps;

        println!("***** Running training *****");
        println!("  Num examples = {}", data_loader.total_samples());
        println!("  Num batches each epoch = {}", num_batches);
        println!("  Instantaneous batch size = {}", self.config.batch_size);
        println!("  Total train batch size = {}", total_batch_size);
        println!("  Gradient Accumulation steps = {}", self.config.gradient_accumulation_steps);
        println!("  Total optimization steps = {}", self.config.max_train_steps);

        // Initialize progress tracking
        let mut accumulated_loss = 0.0;
        let mut loss_count = 0;

        // Training loop
        for epoch in 0..self.config.max_train_steps / num_update_steps_per_epoch + 1 {
            self.epoch = epoch;

            // Reset dataloader for new epoch
            data_loader.shuffle_dataset()?;

            println!(
                "\n========== Epoch {}/{} ==========",
                epoch + 1,
                self.config.max_train_steps / num_update_steps_per_epoch + 1
            );

            // Iterate over batches in this epoch
            let mut step = 0;
            while let Some(batch) = data_loader.next_batch_old()? {
                let batch_start = std::time::Instant::now();
                let loss = self.training_step(batch)?;
                step += 1;

                // Accumulate loss for averaging
                accumulated_loss += loss;
                loss_count += 1;

                // Accumulate gradients
                if step % self.config.gradient_accumulation_steps == 0 {
                    // Clip gradients
                    if self.config.max_grad_norm > 0.0 {
                        self.clip_gradients(self.config.max_grad_norm)?;
                    }

                    // Optimizer step - use the stored grad_map
                    if let Some(ref grad_map) = self.last_grad_map {
                        // FLAME optimizer step doesn't take grad_map
                        self.optimizer.step()?;
                        self.last_grad_map = None; // Clear after use
                    }
                    // Gradients handled by FLAME

                    self.global_step += 1;

                    // Calculate average loss
                    let avg_loss = accumulated_loss / loss_count as f32;

                    // Calculate progress
                    let progress =
                        (self.global_step as f32 / self.config.max_train_steps as f32) * 100.0;

                    // Print progress every step (like other trainers)
                    let elapsed = batch_start.elapsed();
                    let seconds_per_step = elapsed.as_secs_f32();

                    // Calculate ETA
                    let steps_remaining = self.config.max_train_steps - self.global_step;
                    let eta_seconds = (steps_remaining as f32 * seconds_per_step) as u64;
                    let eta_hours = eta_seconds / 3600;
                    let eta_minutes = (eta_seconds % 3600) / 60;
                    let eta_secs = eta_seconds % 60;

                    // Calculate samples/sec
                    let samples_per_sec = self.config.batch_size as f32 / seconds_per_step;

                    // Calculate gradient norm if available
                    let grad_norm = if let Some(lora_layers) = &self.lora_layers {
                        // Calculate average gradient norm across LoRA layers
                        let mut total_norm = 0.0;
                        let mut count = 0;
                        for (_, layer) in lora_layers {
                            if let Some(grad) = layer.lora_down.grad() {
                                total_norm += grad.square()?.sum()?.to_scalar::<f32>()?.sqrt();
                                count += 1;
                            }
                            if let Some(grad) = layer.lora_up.grad() {
                                total_norm += grad.square()?.sum()?.to_scalar::<f32>()?.sqrt();
                                count += 1;
                            }
                        }
                        if count > 0 {
                            total_norm / count as f32
                        } else {
                            0.0
                        }
                    } else {
                        0.0
                    };

                    println!(
                        "Steps: {:5}/{} ({:5.1}%) | Loss: {:.6} | Grad: {:.4} | LR: {:.2e} | {:.1} samples/s | {:.2}s/step | ETA: {:02}:{:02}:{:02}",
                        self.global_step,
                        self.config.max_train_steps,
                        progress,
                        avg_loss,
                        grad_norm,
                        self.config.learning_rate,
                        samples_per_sec,
                        seconds_per_step,
                        eta_hours,
                        eta_minutes,
                        eta_secs
                    );

                    // Reset accumulated loss every 10 steps for running average
                    if self.global_step % 10 == 0 {
                        accumulated_loss = 0.0;
                        loss_count = 0;
                    }

                    // Validation
                    if self.global_step % self.config.validation_steps == 0 {
                        println!("\n>>> Running validation at step {}...", self.global_step);
                        self.validate()?;
                        println!(">>> Validation complete\n");
                    }

                    // Checkpointing
                    if self.global_step % self.config.checkpointing_steps == 0 {
                        println!("\n>>> Saving checkpoint at step {}...", self.global_step);
                        self.save_checkpoint()?;
                        println!(">>> Checkpoint saved\n");
                    }

                    // Check if done
                    if self.global_step >= self.config.max_train_steps {
                        println!("\n=== Training complete! ===");
                        println!("Total steps: {}", self.global_step);
                        println!("Final loss: {:.6}", avg_loss);
                        return Ok(());
                    }
                }
            }
        }

        Ok(())
    }

    /// Single training step
    fn training_step(&mut self, batch: TrainingBatch) -> flame_core::Result<f32> {
        println!("  [training_step] Starting...");
        // Move batch to device
        let images = batch.images;
        let prompts = batch.prompts;
        println!(
            "  [training_step] Batch loaded: {} images, {} prompts",
            images.shape().dims()[0],
            prompts.len()
        );

        // Get latents - either from cache or encode on the fly
        let latents = if let Some(ref cache_manager) = self.cache_manager {
            // SimpleTuner style - load from cache using real image path
            let image_path = &batch.image_paths[0]; // Get the actual image path from batch

            if let Some(cached_latent) = cache_manager.load_latent(image_path)? {
                println!(
                    "  [training_step] ✅ Loaded cached latent for: {:?}",
                    image_path.file_name().unwrap_or_default()
                );
                cached_latent
            } else {
                // Fallback to encoding if not cached
                println!(
                    "  [training_step] ⚠️  Cache miss for: {:?}, falling back to VAE encoding",
                    image_path.file_name().unwrap_or_default()
                );
                let vae = self.vae.as_ref()
                    .ok_or_else(|| flame_core::Error::InvalidOperation(
                        format!("Cache miss for {:?} but VAE not loaded. Enable cache with force_reencode or disable caching.", image_path)
                    ))?;
                let mean = vae.encode(&images)?;
                mean
            }
        } else {
            // Traditional path - encode on the fly
            println!("  [training_step] VAE encoding starting (cuDNN enabled)...");
            let vae_start = std::time::Instant::now();
            let vae = self
                .vae
                .as_ref()
                .ok_or_else(|| flame_core::Error::InvalidOperation("VAE not loaded".into()))?;
            let mean = vae.encode(&images)?;
            println!("  [training_step] VAE encoding done in {:?}", vae_start.elapsed());
            mean
        };

        // Patchify latents for Flux
        println!("  [training_step] Patchifying latents...");
        let latents = patchify_latents(&latents)?;
        println!("  [training_step] Latents shape after patchify: {:?}", latents.shape());

        // Sample noise
        let noise = Tensor::randn(latents.shape().clone(), 0.0, 1.0, latents.device().clone())?;
        let timesteps = self.sample_timesteps(latents.shape().dims()[0])?;

        // Add noise using flow matching
        let (noisy_latents, velocity) =
            self.noise_scheduler.add_noise(&latents, &noise, &timesteps)?;

        // Get text embeddings - either from cache or encode on the fly
        let (clip_embeds, t5_embeds) = if let Some(ref cache_manager) = self.cache_manager {
            // SimpleTuner style - load from cache using real image path
            let image_path = &batch.image_paths[0]; // Get the actual image path from batch

            if let Some((clip_cached, t5_cached)) =
                cache_manager.load_text_embeddings(image_path)?
            {
                println!(
                    "  [training_step] ✅ Loaded cached text embeddings for: {:?}",
                    image_path.file_name().unwrap_or_default()
                );
                (clip_cached, t5_cached)
            } else {
                println!(
                    "  [training_step] ⚠️  Cache miss for text embeddings: {:?}",
                    image_path.file_name().unwrap_or_default()
                );
                // Fallback to encoding if not cached
                self.encode_prompts(&prompts)?
            }
        } else {
            // Traditional path - encode on the fly
            if self.text_encoders.is_some() {
                println!("  [training_step] Text encoding starting...");
                let text_start = std::time::Instant::now();
                let result = self.encode_prompts(&prompts)?;
                println!("  [training_step] Text encoding done in {:?}", text_start.elapsed());
                result
            } else {
                // Use dummy embeddings if text encoders not loaded
                println!("  [training_step] Using dummy embeddings (text encoders not loaded)");
                let batch_size = 1;
                let seq_len = 77; // CLIP sequence length
                let t5_seq_len = 256; // T5 sequence length
                let clip_dim = 768;
                let t5_dim = 4096;

                // Create dummy tensors with appropriate shapes
                let clip_embeds = Tensor::randn(
                    Shape::from_dims(&[batch_size, seq_len, clip_dim]),
                    0.0,
                    0.02,
                    self.device.cuda_device_arc(),
                )?;
                let t5_embeds = Tensor::randn(
                    Shape::from_dims(&[batch_size, t5_seq_len, t5_dim]),
                    0.0,
                    0.02,
                    self.device.cuda_device_arc(),
                )?;
                (clip_embeds, t5_embeds)
            }
        };

        // Forward pass with appropriate model
        let guidance_tensor = if !self.config.bypass_guidance_embedding {
            Some(Tensor::full(
                timesteps.shape().clone(),
                self.config.guidance_scale,
                timesteps.device().clone(),
            )?)
        } else {
            None
        };

        println!("  [training_step] Starting Flux forward pass...");
        let forward_start = std::time::Instant::now();
        let model_pred = if let Some(flux_with_lora) = &self.flux_with_lora {
            // Use LoRA-enabled model
            println!("  [training_step] Using LoRA-enabled Flux model");
            flux_with_lora.forward(
                &noisy_latents,
                &t5_embeds,
                &timesteps,
                &clip_embeds,
                guidance_tensor.as_ref(),
            )?
        } else if let Some(flux_base) = &self.flux_base {
            println!("  [training_step] Using base Flux model");
            // Use base model for full fine-tuning
            flux_base.forward(
                &noisy_latents,
                &t5_embeds,
                &timesteps,
                &clip_embeds,
                guidance_tensor.as_ref(),
            )?
        } else {
            return Err(flame_core::Error::InvalidOperation(
                "No model available for forward pass".into(),
            ));
        };
        println!("  [training_step] Forward pass done in {:?}", forward_start.elapsed());

        // Compute loss - Flux uses v-prediction
        let loss = match self.noise_scheduler.prediction_type {
            PredictionType::VPrediction => mse_loss(&model_pred, &velocity)?,
            PredictionType::Epsilon => mse_loss(&model_pred, &noise)?,
        };

        // Get scalar loss value for logging
        let loss_value = loss.to_scalar::<f32>()?;

        // Backward pass - keep the tensor for gradient computation
        println!("  [training_step] Starting backward pass...");
        let backward_start = std::time::Instant::now();
        let grad_map = loss.backward()?;
        println!("  [training_step] Backward pass done in {:?}", backward_start.elapsed());

        // Store grad_map for optimizer step
        self.last_grad_map = Some(grad_map);

        // Step the gradient accumulator counter
        self.gradient_accumulator.step();

        Ok(loss_value)
    }

    /// Sample timesteps
    fn sample_timesteps(&self, batch_size: usize) -> flame_core::Result<Tensor> {
        // Flux uses continuous timesteps in [0, 1]
        let timesteps =
            (0..batch_size).map(|_| rand::thread_rng().gen::<f32>()).collect::<Vec<_>>();

        Tensor::from_vec(
            timesteps,
            Shape::from_dims(&[batch_size]),
            self.device.cuda_device().clone(),
        )
    }

    /// Encode prompts
    fn encode_prompts(&self, prompts: &[String]) -> flame_core::Result<(Tensor, Tensor)> {
        let text_encoders = self.text_encoders.as_ref().ok_or_else(|| {
            flame_core::Error::InvalidOperation("Text encoders not loaded".into())
        })?;

        let mut clip_embeds = Vec::new();
        let mut t5_embeds = Vec::new();

        for prompt in prompts {
            let (clip, t5) = text_encoders.encode_flux(prompt)?;
            clip_embeds.push(clip);
            t5_embeds.push(t5);
        }

        let clip_batch = Tensor::cat(&clip_embeds.iter().collect::<Vec<_>>(), 0)?;
        let t5_batch = Tensor::cat(&t5_embeds.iter().collect::<Vec<_>>(), 0)?;

        Ok((clip_batch, t5_batch))
    }

    /// Validation step
    fn validate(&self) -> flame_core::Result<()> {
        println!("Running validation...");

        for (i, prompt) in self.config.validation_prompts.iter().enumerate() {
            match self.generate_sample(prompt, None) {
                Ok(image) => {
                    match self.save_image(&image, &format!("val_{}_{}.png", self.global_step, i)) {
                        Ok(()) => println!("✅ Saved validation sample {}", i),
                        Err(e) => println!("⚠️ Failed to save validation sample {}: {}", i, e),
                    }
                }
                Err(e) => {
                    println!("⚠️ Validation sample {} generation failed: {}", i, e);
                    // Continue with other samples instead of failing entirely
                }
            }
        }

        Ok(())
    }

    /// Generate sample image
    pub fn generate_sample(&self, prompt: &str, seed: Option<u64>) -> flame_core::Result<Tensor> {
        info!("Generating Flux sample at step {}", self.global_step);

        // Create inference config
        let inference_config = InferenceConfig {
            height: self.config.resolution,
            width: self.config.resolution,
            num_inference_steps: 20, // Flux schnell uses 20 steps
            guidance_scale: self.config.guidance_scale as f64,
            seed,
            bypass_guidance_embedding: self.config.bypass_guidance_embedding,
        };

        // Create model config for inference
        let model_config = ModelConfig {
            unet_path: self.config.model_path.to_string_lossy().to_string(),
            vae_path: self.config.vae_path.to_string_lossy().to_string(),
            clip_path: self.config.text_encoder_paths.clip_l.to_string_lossy().to_string(),
            clip2_path: None,
            t5_path: self
                .config
                .text_encoder_paths
                .t5_xxl
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
            tokenizer_path: "".to_string(), // TODO: Add tokenizer path to config
            tokenizer2_path: None,
            t5_tokenizer_path: None, // TODO: Add T5 tokenizer path to config
            height: inference_config.height,
            width: inference_config.width,
            use_flash_attn: false,
            num_inference_steps: inference_config.num_inference_steps,
        };

        // Create Flux inference instance
        let mut flux_inference = FluxInference::new(&model_config, &self.device)?;

        // Apply LoRA weights if available
        if let Some(lora_layers) = &self.lora_layers {
            // TODO: Apply LoRA weights to inference model
            // This requires implementing a method to merge LoRA weights into the model
        }

        // Skip actual inference for now - the trait doesn't have a generate method
        // and proper inference requires full implementation
        println!("Warning: Flux inference not yet fully implemented, skipping validation sample generation");

        // Return a placeholder tensor instead of trying to generate
        // This allows training to continue while inference is being implemented
        let placeholder = Tensor::zeros(
            Shape::from_dims(&[1, 3, 1024, 1024]),
            self.device.cuda_device().clone(),
        )?;
        Ok(placeholder)
    }

    /// Save checkpoint
    fn save_checkpoint(&self) -> flame_core::Result<()> {
        let checkpoint_path =
            self.config.logging_dir.join(format!("checkpoint-{}", self.global_step));
        std::fs::create_dir_all(&checkpoint_path)
            .map_err(|e| {
                flame_core::Error::InvalidOperation(format!("Failed to create directory: {}", e))
            })
            .map_err(|e| flame_core::Error::InvalidOperation(e.to_string()))?;

        if let Some(lora_layers) = &self.lora_layers {
            // Save LoRA weights
            save_lora_weights(lora_layers, &checkpoint_path)?;
        } else {
            // Save full model weights
            // TODO: Implement full model saving
        }

        // Save optimizer state
        // TODO: Implement optimizer state saving
        // self.optimizer.save_state(&checkpoint_path.join("optimizer.pt"))?;

        // Save training state
        let state = TrainingState {
            global_step: self.global_step,
            epoch: self.epoch,
            best_loss: 0.0, // Track if needed
        };
        save_training_state(&state, &checkpoint_path)?;

        println!("Saved checkpoint to {:?}", checkpoint_path);
        Ok(())
    }

    /// Clip gradients
    fn clip_gradients(&self, max_norm: f32) -> flame_core::Result<()> {
        let params = get_all_parameters(&self.flux_base, &self.lora_layers);
        let total_norm = compute_gradient_norm(&params)?;

        if total_norm > max_norm {
            let scale = max_norm / total_norm;
            for param in params {
                if let Some(grad) = param.grad() {
                    let scaled_grad = grad.mul_scalar(scale as f32)?;
                    param.set_grad(scaled_grad)?;
                }
            }
        }

        Ok(())
    }

    fn log_metrics(&self, loss: f32) -> flame_core::Result<()> {
        // Log to tensorboard/wandb if configured
        println!(
            "Step: {}, Loss: {:.4}, LR: {:.2e}",
            self.global_step, loss, self.config.learning_rate
        );
        Ok(())
    }

    fn save_image(&self, image: &Tensor, filename: &str) -> flame_core::Result<()> {
        // Save image to disk
        let path = self.config.logging_dir.join("samples").join(filename);
        std::fs::create_dir_all(path.parent().unwrap())
            .map_err(|e| flame_core::Error::InvalidOperation(e.to_string()))?;

        // Convert to PIL and save
        save_tensor_as_image(image, &path)?;

        Ok(())
    }
}

// Duplicate methods have been moved to the first impl block

/// Setup LoRA for Flux
fn setup_flux_lora(
    mut flux: FluxModel,
    rank: usize,
    alpha: f32,
    dropout: f32,
    target_modules: &[String],
    device: &Device,
) -> flame_core::Result<(FluxModel, HashMap<String, FluxLoRALayer>)> {
    let mut lora_layers = HashMap::new();

    // Flux model dimensions
    let hidden_size = 3072; // Flux uses 3072 hidden size
    let mlp_hidden = (hidden_size as f32 * 4.0) as usize; // MLPRatio is 4.0

    // Add LoRA to target modules
    for module_name in target_modules {
        let (in_features, out_features) = match module_name.as_str() {
            // Attention layers - Q, K, V projections
            "img_attn" | "txt_attn" => (hidden_size, hidden_size * 3), // QKV projection

            // MLP layers
            "img_mlp" | "txt_mlp" => {
                // MLP has two linear layers: hidden -> mlp_hidden -> hidden
                // We'll add LoRA to both, so create two layers
                let lora1 =
                    FluxLoRALayer::new(hidden_size, mlp_hidden, rank, alpha, dropout, device)?;
                lora_layers.insert(format!("{}.lin1", module_name), lora1);

                let lora2 =
                    FluxLoRALayer::new(mlp_hidden, hidden_size, rank, alpha, dropout, device)?;
                lora_layers.insert(format!("{}.lin2", module_name), lora2);

                continue; // Skip the main loop body
            }

            // Output projection for attention
            "img_attn.proj" | "txt_attn.proj" => (hidden_size, hidden_size),

            _ => {
                println!("Warning: Unknown module name: {}", module_name);
                continue;
            }
        };

        let lora = FluxLoRALayer::new(in_features, out_features, rank, alpha, dropout, device)?;
        lora_layers.insert(module_name.clone(), lora);
    }

    println!("Created {} LoRA layers for Flux model", lora_layers.len());
    for (name, layer) in &lora_layers {
        println!("  {} - rank: {}, scale: {:.4}", name, layer.rank, layer.scale);
    }

    Ok((flux, lora_layers))
}

/// Get trainable parameters
fn get_trainable_parameters(
    flux: &Option<FluxModel>,
    lora_layers: &Option<HashMap<String, FluxLoRALayer>>,
    config: &FluxTrainingConfig,
) -> Vec<Parameter> {
    if let Some(lora) = lora_layers {
        // Only LoRA parameters are trainable
        let mut params = Vec::new();
        for (name, layer) in lora.iter() {
            println!("Adding LoRA parameters for: {}", name);
            params.push(layer.lora_down.clone());
            params.push(layer.lora_up.clone());
        }
        println!("Total trainable LoRA parameters: {}", params.len());
        params
    } else {
        // For full fine-tuning, we would need to collect all model parameters
        // This requires the FluxModel to expose its parameters
        // For now, return empty as full fine-tuning is not implemented
        println!("Warning: Full fine-tuning not yet implemented for Flux");
        vec![]
    }
}

/// Create optimizer
fn create_optimizer(
    params: Vec<Parameter>,
    config: &FluxTrainingConfig,
) -> flame_core::Result<Adam8bit> {
    let optimizer = Adam8bit::with_params(
        config.learning_rate,
        0.9,   // beta1
        0.999, // beta2
        1e-8,  // eps
        0.01,  // weight_decay
    );

    Ok(optimizer)
}

/// Helper functions for patchifying/unpatchifying
fn patchify_latents(latents: &Tensor) -> flame_core::Result<Tensor> {
    let (batch, channels, height, width) = match latents.shape().dims() {
        [b, c, h, w] => (*b, *c, *h, *w),
        _ => return Err(flame_core::Error::InvalidOperation("Invalid latent shape".into())),
    };

    // Reshape to patches: [B, 16, H, W] -> [B, (H/2)*(W/2), 64]
    latents
        .reshape(&[batch, channels, height / 2, 2, width / 2, 2])?
        .permute(&[0, 2, 4, 3, 5, 1])?
        .reshape(&[batch, (height / 2) * (width / 2), 64])
}

fn unpatchify_latents(latents: &Tensor, height: usize, width: usize) -> flame_core::Result<Tensor> {
    let batch = latents.shape().dims()[0];

    latents
        .reshape(&[batch, height / 2, width / 2, 2, 2, 16])?
        .permute(&[0, 5, 1, 3, 2, 4])?
        .reshape(&[batch, 16, height, width])
}

/// Training state for checkpointing
#[derive(Serialize, Deserialize)]
struct TrainingState {
    global_step: usize,
    epoch: usize,
    best_loss: f32,
}

// Helper functions
fn load_vae(path: &Path, device: &Device) -> flame_core::Result<AutoencoderKL> {
    use crate::loaders::WeightLoader;

    // Load weights from safetensors (VAE is smaller, regular loading is OK)
    println!("  Loading VAE weights...");
    let weight_loader =
        WeightLoader::from_safetensors_with_dtype(path, device.clone(), DType::F16)?;
    println!("  VAE weights loaded: {} tensors", weight_loader.weights.len());

    // Create VAE with Flux configuration (16-channel)
    let vae = AutoencoderKL::new(&weight_loader, device.clone(), true)?; // Enable CPU offloading

    Ok(vae)
}

fn load_flux_model(path: &Path, device: &Device) -> flame_core::Result<FluxModel> {
    // Use memory-mapped loading for the Flux model
    println!("  🌊 Loading Flux model via memory-mapped streaming...");
    println!("  Model path: {:?}", path);

    // Check file size
    if let Ok(metadata) = std::fs::metadata(path) {
        let size_gb = metadata.len() as f64 / (1024.0 * 1024.0 * 1024.0);
        println!("  Model size: {:.2} GB", size_gb);

        // If model is too large, suggest using a smaller one
        if size_gb > 20.0 {
            println!(
                "  ⚠️  WARNING: Model is very large ({:.1} GB), may cause OOM in 24GB VRAM",
                size_gb
            );
            println!("  Consider using flux1-dev-kontext_fp8_scaled.safetensors (12GB) or flux.1-lite-8B.safetensors (16GB)");
        }
    }

    // Use memory-mapped streaming loader to avoid OOM
    println!("  Using memory-mapped file access to handle large models in 24GB VRAM...");
    let weight_loader =
        WeightLoader::from_safetensors_streaming(path, device.clone(), DType::BF16)?;
    println!(
        "  ✅ Weights loaded via memory mapping, total tensors: {}",
        weight_loader.weights.len()
    );

    // Detect model variant from weights
    let config = if weight_loader
        .weights
        .contains_key("model.diffusion_model.double_blocks.0.txt_mlp.0.weight")
    {
        println!("  Detected Flux-dev model (with guidance)");
        FluxModelConfig::flux_dev() // Has guidance
    } else if weight_loader.weights.contains_key("double_blocks.0.txt_mlp.0.weight") {
        println!("  Detected Flux-dev model (with guidance, no prefix)");
        FluxModelConfig::flux_dev() // Has guidance
    } else {
        println!("  Detected Flux-schnell model (no guidance)");
        FluxModelConfig::flux_schnell() // No guidance
    };

    // Create model with loaded weights
    println!("  Initializing Flux model architecture...");
    println!("    - Hidden size: {}", config.hidden_size);
    println!("    - Double blocks: {}", config.depth);
    println!("    - Single blocks: {}", config.depth_single_blocks);
    println!("    - Guidance embedding: {}", config.guidance_embed);

    let model = FluxModel::new(config, device.clone(), weight_loader.weights)?;
    println!("  ✅ Flux model initialized successfully with memory-mapped weights");

    Ok(model)
}

fn load_text_encoders(
    paths: &TextEncoderPaths,
    device: &Device,
) -> flame_core::Result<TextEncoders> {
    let mut encoders = TextEncoders::new(device.clone());

    // Load CLIP-L (small, regular loading is fine)
    println!("Loading CLIP-L from: {:?}", paths.clip_l);
    encoders.load_clip_l(&paths.clip_l.to_string_lossy())?;

    // Load T5-XXL only if path is provided
    if let Some(t5_path) = &paths.t5_xxl {
        println!("Loading T5-XXL from: {:?}", t5_path);
        encoders.load_t5(&t5_path.to_string_lossy())?;
    } else {
        println!("T5-XXL disabled in config - using CLIP-L only");
    }

    Ok(encoders)
}

fn save_lora_weights(
    lora_layers: &HashMap<String, FluxLoRALayer>,
    path: &Path,
) -> flame_core::Result<()> {
    use safetensors::{serialize, tensor::Dtype as SafeDtype};
    use std::collections::HashMap as StdHashMap;

    let mut tensors = StdHashMap::new();

    // Convert LoRA layers to safetensors format
    for (name, layer) in lora_layers {
        // Get tensor values
        let down_tensor = layer.lora_down.tensor()?;
        let up_tensor = layer.lora_up.tensor()?;

        // Convert to safetensors format
        let down_data = down_tensor.to_vec1::<f32>()?;
        let up_data = up_tensor.to_vec1::<f32>()?;

        let down_shape = down_tensor.shape().dims().to_vec();
        let up_shape = up_tensor.shape().dims().to_vec();

        // Add to tensors map with proper naming convention
        // Convert f32 vectors to byte slices
        let down_bytes = unsafe {
            std::slice::from_raw_parts(down_data.as_ptr() as *const u8, down_data.len() * 4)
        };
        let up_bytes =
            unsafe { std::slice::from_raw_parts(up_data.as_ptr() as *const u8, up_data.len() * 4) };

        tensors.insert(
            format!("{}.lora_down.weight", name),
            safetensors::tensor::TensorView::new(SafeDtype::F32, down_shape, down_bytes)
                .map_err(|e| flame_core::Error::InvalidOperation(e.to_string()))?,
        );
        tensors.insert(
            format!("{}.lora_up.weight", name),
            safetensors::tensor::TensorView::new(SafeDtype::F32, up_shape, up_bytes)
                .map_err(|e| flame_core::Error::InvalidOperation(e.to_string()))?,
        );
    }

    // Serialize to safetensors format
    let data = serialize(tensors, &None)
        .map_err(|e| flame_core::Error::InvalidOperation(e.to_string()))?;

    // Write to file
    std::fs::write(path.join("flux_lora.safetensors"), data)
        .map_err(|e| flame_core::Error::InvalidOperation(e.to_string()))?;

    println!("Saved LoRA weights to {:?}", path.join("flux_lora.safetensors"));
    Ok(())
}

fn save_training_state(state: &TrainingState, path: &Path) -> flame_core::Result<()> {
    let json = serde_json::to_string_pretty(state).map_err(|e| {
        flame_core::Error::InvalidOperation(format!("Failed to serialize state: {}", e))
    })?;

    std::fs::write(path.join("training_state.json"), json)
        .map_err(|e| flame_core::Error::InvalidOperation(e.to_string()))?;
    Ok(())
}

fn compute_gradient_norm(params: &[&Parameter]) -> flame_core::Result<f32> {
    let mut total_norm = 0.0f32;

    for param in params {
        if let Some(grad) = param.grad() {
            let grad_norm = grad.square()?.sum()?.to_scalar::<f32>()?;
            total_norm += grad_norm;
        }
    }

    Ok(total_norm.sqrt())
}

fn get_all_parameters<'a>(
    flux: &'a Option<FluxModel>,
    lora_layers: &'a Option<HashMap<String, FluxLoRALayer>>,
) -> Vec<&'a Parameter> {
    let mut params = Vec::new();

    if let Some(lora) = lora_layers {
        // Only LoRA parameters
        for lora_layer in lora.values() {
            params.push(&lora_layer.lora_down);
            params.push(&lora_layer.lora_up);
        }
    } else {
        // TODO: Collect all model parameters
    }

    params
}

fn save_tensor_as_image(tensor: &Tensor, path: &Path) -> flame_core::Result<()> {
    // Create parent directory
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| {
                flame_core::Error::InvalidOperation(format!("Failed to create directory: {}", e))
            })
            .map_err(|e| flame_core::Error::InvalidOperation(e.to_string()))?;
    }

    // Attempt to save tensor metadata at minimum
    let info_path = path.with_extension("txt");
    let info = format!(
        "Tensor Info:\nShape: {:?}\nDType: {:?}\nDevice: {:?}\n",
        tensor.shape(),
        tensor.dtype(),
        tensor.device()
    );

    std::fs::write(&info_path, info).map_err(|e| {
        flame_core::Error::InvalidOperation(format!("Failed to save tensor info: {}", e))
    })?;

    println!("Saved tensor info to: {:?}", info_path);

    // TODO: Implement actual image saving when image crate is integrated
    // For now, this is better than silent failure
    Err(flame_core::Error::InvalidOperation(
        "Image saving not implemented. Use external tools to convert tensor data.".to_string(),
    ))
}

fn mse_loss(pred: &Tensor, target: &Tensor) -> flame_core::Result<Tensor> {
    let diff = pred.sub(target)?;
    let squared = diff.mul(&diff)?; // square by multiplying with itself
    squared.mean()
}
