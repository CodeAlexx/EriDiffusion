use anyhow::{anyhow, Context};
use eridiffusion_core::Device as CoreDevice;
use flame_core::device::Device;
use flame_core::optimizers::{Adam, SGD};
use flame_core::{DType, Error, Result, Shape, Tensor};
use half::{bf16, f16};
use log::{debug, error, info, warn};
use rand::{Rng, SeedableRng};
use safetensors::{serialize, tensor::TensorView, Dtype as SafeDtype};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use std::{collections::HashMap, fs, path::PathBuf};

use super::adam8bit::Adam8bit;
use super::cpu_offload_manager::CPUOffloadManager;
use super::ddpm_scheduler::DDPMScheduler;
use super::memory_utils::*;
use super::sdxl_forward_sampling::forward_sdxl_sampling;
use super::sdxl_forward_sd_format_flash::forward_sdxl_sd_format_flash;
use super::sdxl_memory_efficient::*;
use super::sdxl_utils::*;
use super::sdxl_vae_native::SDXLVAENative;
use super::sdxl_vae_wrapper::SDXLVAEWrapper;
use super::text_encoders::TextEncoders;
use crate::config::trainer_config::{Config, ProcessConfig, SampleConfig};
// use super::enhanced_data_loader::EnhancedDataLoader; // Not implemented yet
use super::checkpoint_manager::CheckpointManager;
use super::gpu_gradient_checkpoint::GPUGradientCheckpoint as SDXLGradientCheckpoint;
use super::gradient_accumulator::GradientAccumulator;
use super::unified_sampling::SamplingConfig;
use crate::loaders::WeightLoader;
use eridiffusion_common_weights::strict_loader::StrictMmapLoader;

// Import from new modules
use super::lora::{LoRACollection, LoRAConfig, SimpleLoRA};
use super::models::{ModelLoader, ModelManager};
use super::sampling::{generate_samples, SDXLSampler, SamplerConfig};
use super::training::{compute_loss, LossType, TrainingLoop, TrainingState};
use crate::config::trainer_config::TreadConfig;
use crate::inference::sdxl::{SDXLConfig as InferenceConfig, SDXLInference};
use crate::trainers::adapters_util;
use eridiffusion_training::sdxl::registry::SdxlLayerRegistry;
use eridiffusion_training::sdxl::runtime::AttnChunkConfig;
use eridiffusion_training::sdxl::weights::SdxlWeightProvider;
use eridiffusion_training::sdxl::RuntimeMode;
use eridiffusion_training::streaming::WeightProvider;
use eridiffusion_training::telemetry::TelemetryCsv;
use eridiffusion_training::tread;
use eridiffusion_training::tread::TreadMetrics;

const SDXL_LYCO_ALLOW: &[&str] = &[
    "unet.*.st.tb*.xattn.to_q",
    "unet.*.st.tb*.xattn.to_k",
    "unet.*.st.tb*.xattn.to_v",
    "unet.*.st.tb*.xattn.to_out",
    "unet.*.st.tb*.ff.fc1",
    "unet.*.st.tb*.ff.fc2",
    "unet.*.b*.in_conv",
    "unet.*.b*.out_conv",
];

// SimpleLoRA and LoRACollection are now imported from the lora module

pub struct SDXLConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub model_channels: usize,
    pub num_res_blocks: usize,
    pub channel_mult: Vec<usize>,
    pub context_dim: usize,
    pub use_linear_projection: bool,
    pub num_heads: usize,
}

pub struct SDXLLoRATrainerFixed {
    // Core components
    device: Device,
    dtype: DType,

    // Model weights loaded directly
    unet_weights: Option<HashMap<String, Tensor>>,

    // Models
    vae_encoder: Option<SDXLVAENative>,
    text_encoders: Option<TextEncoders>,

    // Scheduler
    noise_scheduler: DDPMScheduler,

    // LoRA adapters
    lora_collection: Option<LoRACollection>,

    // Optimizer
    optimizer: Option<Adam8bit>,
    cpu_offload: Option<CPUOffloadManager>,

    // Training state
    global_step: usize,
    gradient_accumulator: GradientAccumulator,
    gradient_checkpoint: Option<SDXLGradientCheckpoint>,

    // Configuration
    config: ProcessConfig,
    output_dir: PathBuf,
    // Optional registry streaming for transformer blocks
    use_registry_streaming: bool,
    sdxl_registry: Option<Arc<SdxlLayerRegistry>>,
    sdxl_provider: Option<Arc<SdxlWeightProvider>>,

    // Telemetry
    telemetry: Option<TelemetryCsv>,
    last_tread_metrics: Option<TreadMetrics>,
    // LyCORIS adapters (optional)
    lyco_adapters: Option<adapters::adapter::AdapterSet>,
}

// SimpleLoRA and LoRACollection implementations are now in the lora module

impl SDXLLoRATrainerFixed {
    pub fn new(
        config: ProcessConfig,
        device: Device,
        dtype: DType,
        output_dir: PathBuf,
    ) -> flame_core::Result<Self> {
        // Map optional TREAD config to env so downstream modules can pick it up
        if let Some(tread) = config.tread.as_ref() {
            std::env::set_var("TREAD_ENABLED", if tread.enabled { "1" } else { "0" });
            if let Some(mask) = tread.mask.as_ref() {
                if let Some(k) = mask.k {
                    std::env::set_var("TREAD_K", k.to_string());
                }
                if let Some(kf) = mask.k_frac {
                    std::env::set_var("TREAD_K_FRAC", format!("{}", kf));
                }
                if let Some(ref t) = mask.r#type {
                    std::env::set_var("TREAD_MASK_TYPE", t);
                }
            }
            if !tread.schedule.is_empty() {
                let s = tread
                    .schedule
                    .iter()
                    .map(|p| format!("{}:{}", p.out, p.r#in))
                    .collect::<Vec<_>>()
                    .join(",");
                std::env::set_var("TREAD_SCHEDULE", s);
            }
            if let Some(rei) = tread.reinject.as_ref() {
                if let Some(ref m) = rei.mode {
                    std::env::set_var("TREAD_REINJECT_MODE", m);
                }
            }
            if let Some(loss) = tread.loss.as_ref() {
                if let Some(l) = loss.route_lambda {
                    std::env::set_var("TREAD_LAMBDA", format!("{}", l));
                }
            }
        }
        let noise_scheduler = DDPMScheduler::new(1000, 0.0001, 0.02, "linear", &device)?;
        let gradient_accumulator = GradientAccumulator::new(1, device.clone());
        // Optional registry streaming init
        let use_registry_streaming =
            std::env::var("USE_REGISTRY_STREAMING").ok().map(|v| v == "1").unwrap_or(false);
        let (sdxl_registry, sdxl_provider) = if use_registry_streaming {
            let model_path = config.model.name_or_path.clone();
            let loader = Arc::new(
                StrictMmapLoader::open(std::path::Path::new(&model_path)).map_err(|e| {
                    flame_core::Error::InvalidOperation(format!(
                        "open sdxl shard {}: {}",
                        model_path.display(),
                        e
                    ))
                })?,
            );
            let provider = Arc::new(SdxlWeightProvider::new(loader.clone(), device.clone()));
            let registry =
                Arc::new(SdxlLayerRegistry::build(provider.clone(), RuntimeMode::Streamed)?);
            (Some(registry), Some(provider))
        } else {
            (None, None)
        };

        // Telemetry CSV in output dir; rows adapt to #blocks if registry active
        let num_layers = sdxl_registry.as_ref().map(|r| r.block_count()).unwrap_or(0);
        let telemetry = TelemetryCsv::new(output_dir.join("telemetry.csv"), num_layers)
            .map_err(|e| flame_core::Error::InvalidOperation(e.to_string()))?;

        Ok(Self {
            device,
            dtype,
            unet_weights: None,
            vae_encoder: None,
            text_encoders: None,
            noise_scheduler,
            lora_collection: None,
            optimizer: None,
            cpu_offload: None,
            global_step: 0,
            gradient_accumulator,
            gradient_checkpoint: None,
            config,
            output_dir,
            use_registry_streaming,
            sdxl_registry,
            sdxl_provider,
            telemetry: Some(telemetry),
            last_tread_metrics: None,
            lyco_adapters: None,
        })
    }

    // FIXED: Return Result<()> instead of Result<Tensor>
    fn setup_device(device_str: &str) -> flame_core::Result<Device> {
        let device = if device_str.starts_with("cuda") {
            if let Some(idx_str) = device_str.strip_prefix("cuda:") {
                let idx = idx_str.parse::<usize>().map_err(|_| {
                    flame_core::Error::InvalidOperation(format!(
                        "Invalid CUDA device index: {}",
                        idx_str
                    ))
                })?;
                Device::cuda(idx)?
            } else {
                Device::cuda(0)?
            }
        } else {
            flame_core::device::Device::cuda(0)?
        };

        info!("Using device: CUDA");
        Ok(device)
    }

    // FIXED: Return Result<()> instead of Result<Tensor>
    fn ensure_unet_weights_loaded(&mut self) -> flame_core::Result<()> {
        if self.unet_weights.is_none() {
            let model_path = PathBuf::from(&self.config.model.name_or_path);
            info!("Loading UNet weights from: {}", model_path.display());

            let weights =
                crate::loaders::WeightLoader::from_safetensors(&model_path, self.device.clone())?;
            self.unet_weights = Some(weights.weights);
        }
        Ok(())
    }

    // FIXED: Return Result<()> instead of Result<Tensor>
    pub fn load_models(&mut self) -> flame_core::Result<()> {
        info!("Loading models...");

        // Ensure UNet weights are loaded
        self.ensure_unet_weights_loaded()?;

        // Load VAE encoder
        if self.vae_encoder.is_none() {
            info!("Loading VAE encoder...");
            let vae_path =
                self.config.model.vae_path.as_ref().map(PathBuf::from).unwrap_or_else(|| {
                    PathBuf::from("/home/alex/SwarmUI/Models/vae/sdxl_vae.safetensors")
                });

            let vae_weights = WeightLoader::from_safetensors(&vae_path, self.device.clone())?;
            // SDXLVAENative expects weights HashMap directly
            self.vae_encoder =
                Some(SDXLVAENative::new(vae_weights.weights, self.device.clone(), self.dtype)?);
        }

        // Load text encoders
        if self.text_encoders.is_none() {
            info!("Loading text encoders...");
            let clip_l_path =
                self.config.model.text_encoder_path.as_ref().cloned().unwrap_or_else(|| {
                    PathBuf::from("/home/alex/SwarmUI/Models/clip/clip_l.safetensors")
                });
            let clip_g_path =
                self.config.model.text_encoder_2_path.as_ref().cloned().unwrap_or_else(|| {
                    PathBuf::from("/home/alex/SwarmUI/Models/clip/clip_g.safetensors")
                });

            let mut text_encoders = TextEncoders::new(self.device.clone());
            text_encoders.load_clip_l(&clip_l_path.to_string_lossy())?;
            text_encoders.load_clip_g(&clip_g_path.to_string_lossy())?;
            self.text_encoders = Some(text_encoders);
        }

        Ok(())
    }

    // FIXED: Return Result<()> instead of Result<Tensor>
    pub fn init_lora_adapters(&mut self) -> flame_core::Result<()> {
        info!("Initializing LoRA adapters...");

        let rank = self.config.network.linear;
        let alpha = self.config.network.linear_alpha;

        let lora_config = LoRAConfig {
            rank,
            alpha,
            dropout: Some(0.0),
            target_modules: vec!["attn".to_string()],
            dtype: match self.dtype {
                DType::F16 => "f16".to_string(),
                DType::BF16 => "bf16".to_string(),
                _ => "f32".to_string(),
            },
        };
        let mut lora_collection = LoRACollection::new(lora_config, &self.device)?;

        // Get UNet config
        let unet_config = SDXLConfig {
            in_channels: 4,
            out_channels: 4,
            model_channels: 320,
            num_res_blocks: 2,
            channel_mult: vec![1, 2, 4],
            context_dim: 2048,
            use_linear_projection: true,
            num_heads: 8,
        };

        // Create LoRA adapters for attention layers
        self.ensure_unet_weights_loaded()?;

        if let Some(weights) = &self.unet_weights {
            info!("Creating LoRA adapters for attention layers...");

            // Pattern: Find all attention projection layers
            for (name, weight) in weights {
                if name.contains("attn")
                    & (name.contains("to_k")
                        || name.contains("to_v")
                        || name.contains("to_q")
                        || name.contains("to_out.0"))
                {
                    let shape = weight.shape();
                    if shape.rank() >= 2 {
                        let dims = shape.dims();
                        let out_features = dims[0];
                        let in_features = dims[1];

                        lora_collection.add_adapter(
                            &name,
                            in_features,
                            out_features,
                            &self.device,
                        )?;
                        debug!(
                            "Created LoRA adapter for {}: {}x{}",
                            name, in_features, out_features
                        );
                    }
                }
            }

            info!("Created {} LoRA adapters", lora_collection.adapters.len());
        }

        self.lora_collection = Some(lora_collection);
        Ok(())
    }

    pub fn init_optimizer(&mut self) -> flame_core::Result<()> {
        info!("Initializing optimizer...");
        // Prefer LyCORIS adapters if present; else use LoRA collection params
        let learning_rate = self.config.train.lr as f64;
        if let Some(ref set) = self.lyco_adapters {
            let params = adapters_util::to_parameters(set)
                .map_err(|e| Error::InvalidOperation(e.to_string()))?;
            info!(
                "Using LyCORIS adapters: {} targets; {} params",
                set.by_target.len(),
                params.len()
            );
            let opt = Adam8bit::with_params(learning_rate, 0.9, 0.999, 1e-8, 0.01);
            self.optimizer = Some(opt);
        } else if let Some(lora_collection) = &self.lora_collection {
            let _params = lora_collection.trainable_parameters();
            info!("Using LoRA adapters (legacy path)");
            let opt = Adam8bit::with_params(learning_rate, 0.9, 0.999, 1e-8, 0.01);
            self.optimizer = Some(opt);
        }

        Ok(())
    }

    fn try_attach_lycoris(&mut self) -> flame_core::Result<()> {
        if let Ok(dir) = std::env::var("LYCORIS_DIR") {
            // Build base shapes from loaded UNet weights if available
            if let Some(ref weights) = self.unet_weights {
                let device_core = CoreDevice::cuda(self.device.ordinal())
                    .map_err(|e| Error::InvalidOperation(e.to_string()))?;
                let bs = adapters_util::base_shapes_from_weights(weights);
                let set = adapters::loader::safetensors_lyco::load_lycoris_dir(
                    std::path::Path::new(&dir),
                    &device_core,
                    DType::BF16,
                    &bs,
                )
                .map_err(|e| Error::InvalidOperation(e.to_string()))?;
                let allow: Vec<String> = SDXL_LYCO_ALLOW.iter().map(|s| s.to_string()).collect();
                let filtered = adapters_util::filter_adapters(&set, &allow, &vec![]);
                info!("Attached LyCORIS adapters: {} targets", filtered.by_target.len());
                self.lyco_adapters = Some(filtered);
            } else {
                warn!("LYCORIS_DIR set but unet_weights are not loaded; skip attach");
            }
        }
        Ok(())
    }

    pub fn train(
        &mut self,
        mut dataloader: super::enhanced_data_loader::EnhancedDataLoader,
    ) -> flame_core::Result<()> {
        info!("Starting training...");

        let total_steps = self.config.train.steps;
        let gradient_accumulation_steps = self.config.train.gradient_accumulation_steps;
        let save_every = self.config.save.save_every;
        let sample_every = self.config.sample.sample_every;

        // Initialize gradient checkpointing if enabled
        if self.config.train.gradient_checkpointing {
            info!("Enabling gradient checkpointing...");
            self.gradient_checkpoint = Some(SDXLGradientCheckpoint::new(true, &self.device));
        }

        // Training loop
        for step in 0..total_steps {
            self.global_step = step;

            // Get batch
            let batch = dataloader.next_batch()?;

            // Convert DataBatch to HashMap for training_step
            let mut batch_map = HashMap::new();
            batch_map.insert("pixel_values".to_string(), batch.pixel_values);
            batch_map.insert("input_ids".to_string(), batch.input_ids);

            // Forward pass and accumulate gradients
            let step_start = Instant::now();
            let (loss_value, grad_map) = self.training_step(&batch_map)?;

            // Optimizer step every gradient_accumulation_steps
            if (step + 1) % gradient_accumulation_steps == 0 {
                self.optimizer_step(&grad_map)?;
                // Clear gradients after optimizer step
                self.gradient_accumulator.clear();
            }

            // Logging
            if step % 10 == 0 {
                info!("Step {}/{}: loss = {:.6}", step, total_steps, loss_value);
            }

            // Telemetry after each step
            if let Some(tel) = &mut self.telemetry {
                let m = self
                    .last_tread_metrics
                    .take()
                    .unwrap_or_else(|| eridiffusion_training::tread::TreadState::new().metrics());
                let route_loss_value: f32 = 0.0;
                let fw_sec = step_start.elapsed().as_secs_f32();
                let bw_sec = 0.0;
                let h2d_mb_s = 0.0;
                let d2h_mb_s = 0.0;
                let gpu_mem_peak = 0.0;
                let tokens_per_s = 0.0;
                let grad_norm = 0.0;
                let fused = eridiffusion_training::telemetry::fused_runtime_enabled();
                let _ = tel.write(
                    step as u64,
                    fw_sec,
                    bw_sec,
                    h2d_mb_s,
                    d2h_mb_s,
                    gpu_mem_peak,
                    tokens_per_s,
                    grad_norm,
                    loss_value,
                    fused,
                    m.route_lambda,
                    route_loss_value,
                    m.kept_avg,
                    &m.kept_frac,
                    m.shelves_mb,
                    m.route_miss,
                );
            }

            // Save checkpoint
            if step > 0 && step % save_every == 0 {
                self.save_checkpoint(step)?;
            }

            // Generate samples
            if step > 0 && step % sample_every == 0 {
                self.generate_samples(step)?;
            }
        }

        // Final save
        self.save_checkpoint(total_steps)?;

        info!("Training completed!");
        Ok(())
    }

    fn training_step(
        &mut self,
        batch: &HashMap<String, Tensor>,
    ) -> flame_core::Result<(f32, flame_core::gradient::GradientMap)> {
        // Get batch data
        let pixel_values = batch.get("pixel_values").ok_or_else(|| {
            flame_core::Error::InvalidOperation("Missing pixel_values in batch".into())
        })?;
        let input_ids = batch.get("input_ids").ok_or_else(|| {
            flame_core::Error::InvalidOperation("Missing input_ids in batch".into())
        })?;

        // Get batch size
        let batch_size = pixel_values.shape().dims()[0];

        // Encode images to latents
        let latents = if let Some(vae) = &self.vae_encoder {
            vae.encode(pixel_values).map_err(|e| {
                flame_core::Error::InvalidOperation(format!("VAE encode failed: {}", e))
            })?
        } else {
            return Err(flame_core::Error::InvalidOperation("VAE encoder not loaded".to_string()));
        };

        // Encode text
        let (encoder_hidden_states, pooled_output) =
            if let Some(text_encoders) = &mut self.text_encoders {
                // For SDXL we need both CLIP encoders
                // TODO: Convert input_ids tensor to strings for encoding
                // For now, use placeholder
                let texts = vec!["placeholder text".to_string(); batch_size];
                text_encoders.encode_batch(&texts, 77)?
            } else {
                return Err(flame_core::Error::InvalidOperation(
                    "Text encoders not loaded".to_string(),
                ));
            };

        // Sample noise
        let noise =
            Tensor::randn(latents.shape().clone(), 0.0f32, 1.0f32, latents.device().clone())?;

        // Sample timesteps
        // Create random timesteps by generating uniform random numbers and scaling
        let random_vals = Tensor::randn(
            Shape::from_dims(&[batch_size]),
            0.0f32,
            1.0f32,
            self.device.cuda_device().clone(),
        )?;
        // FLAME doesn't have floor(), convert to i64 directly which will truncate
        let timesteps = random_vals
            .abs()?
            .mul_scalar(self.noise_scheduler.num_train_timesteps() as f32)?
            .to_dtype(DType::I64)?;

        // Add noise to latents
        let mut noisy_latents = self.noise_scheduler.add_noise(&latents, &noise, &timesteps)?;

        // Prepare time embeddings
        let time_ids = compute_time_ids(
            pixel_values.shape().dims()[2], // height
            pixel_values.shape().dims()[3], // width
            batch_size,
            &self.device,
            self.dtype,
        )?;

        // Optional: stream transformer blocks via registry (IO validation)
        if self.use_registry_streaming {
            if let Some(provider) = &self.sdxl_provider {
                let warmup_blocks = std::env::var("SDXL_IO_WARMUP_COUNT")
                    .ok()
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(3);
                for idx in 0..warmup_blocks {
                    if let Err(e) = provider.prefetch_block(idx) {
                        warn!("[io][sdxl] prefetch block {} failed: {}", idx, e);
                        break;
                    }
                    if let Err(e) = provider.load_block_to_gpu(idx) {
                        warn!("[io][sdxl] load block {} failed: {}", idx, e);
                        break;
                    }
                    if let Err(e) = provider.release_block(idx as isize) {
                        warn!("[io][sdxl] release block {} failed: {}", idx, e);
                        break;
                    }
                }
            }

            if let Some(reg) = &self.sdxl_registry {
                let dims = noisy_latents.shape().dims().to_vec();
                let (b, _c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
                let latents_nhwc = noisy_latents.permute(&[0, 2, 3, 1])?.to_dtype(DType::BF16)?;
                let ctx = encoder_hidden_states.to_dtype(DType::BF16)?;
                let cond = reg.make_conditioning(&pooled_output, &timesteps, &time_ids)?;
                let attn_cfg = AttnChunkConfig::default();
                let streamed = reg
                    .forward_blocks(latents_nhwc, &ctx, &cond, attn_cfg)
                    .map_err(|e| flame_core::Error::InvalidOperation(e.to_string()))?;
                let restored = streamed
                    .permute(&[0, 3, 1, 2])?
                    .reshape(&[b, 4, h, w])?
                    .to_dtype(noisy_latents.dtype())?;
                noisy_latents = restored;
                self.last_tread_metrics = None;
            }
        }

        // Forward pass through UNet with LoRA
        let noise_pred =
            self.forward_with_lora(&noisy_latents, &timesteps, &encoder_hidden_states, &time_ids)?;

        // Compute loss
        let loss = match self.config.train.noise_scheduler.as_str() {
            "ddpm" => {
                // Simple MSE loss
                // FLAME doesn't have pow_scalar, use mul to square
                let diff = noise_pred.sub(&noise)?;
                diff.mul(&diff)?.mean()?
            }
            "flowmatch" => {
                // Flow matching loss
                let v_pred = noise_pred;
                let v_target = latents.sub(&noise)?;
                // FLAME doesn't have pow_scalar, use mul to square
                let diff = v_pred.sub(&v_target)?;
                diff.mul(&diff)?.mean()?
            }
            _ => {
                // Default to MSE
                // FLAME doesn't have pow_scalar, use mul to square
                let diff = noise_pred.sub(&noise)?;
                diff.mul(&diff)?.mean()?
            }
        };

        // Backward pass - returns GradientMap
        let grads = loss.backward()?;

        // Store gradients for accumulation
        // TODO: Implement proper gradient accumulation with FLAME's GradientMap
        // For now, we'll handle this in optimizer_step

        let loss_value = loss.to_scalar::<f32>()?;
        Ok((loss_value, grads))
    }

    fn forward_with_lora(
        &self,
        sample: &Tensor,
        timestep: &Tensor,
        encoder_hidden_states: &Tensor,
        time_ids: &Tensor,
    ) -> flame_core::Result<Tensor> {
        // Use the appropriate forward function based on config
        let use_flash_attention = self.config.train.gradient_checkpointing;

        if use_flash_attention {
            // forward_sdxl_sd_format_flash expects WeightLoader, not HashMap
            // For now, use forward_sdxl_sampling which accepts HashMap
            forward_sdxl_sampling(
                sample,
                timestep,
                encoder_hidden_states,
                self.unet_weights.as_ref().unwrap(),
                self.lora_collection.as_ref().unwrap(),
                None,           // pooled_proj
                Some(time_ids), // time_ids
            )
        } else {
            forward_sdxl_sampling(
                sample,
                timestep,
                encoder_hidden_states,
                self.unet_weights.as_ref().unwrap(),
                self.lora_collection.as_ref().unwrap(),
                None,           // pooled_proj
                Some(time_ids), // time_ids
            )
        }
    }

    fn optimizer_step(
        &mut self,
        grad_map: &flame_core::gradient::GradientMap,
    ) -> flame_core::Result<()> {
        if let Some(optimizer) = &mut self.optimizer {
            if let Some(lora_collection) = &mut self.lora_collection {
                // Update each LoRA adapter's parameters
                for (name, adapter) in &mut lora_collection.adapters {
                    // Update down projection
                    let down_name = format!("{}.down", name);
                    let down_tensor = adapter.down.tensor()?;
                    if let Some(grad) = grad_map.get(down_tensor.id()) {
                        let new_down = optimizer.update(&down_name, &down_tensor, grad)?;
                        // Update the parameter with the new tensor
                        adapter.down.apply_update(&down_tensor.sub(&new_down)?)?;
                    }

                    // Update up projection
                    let up_name = format!("{}.up", name);
                    let up_tensor = adapter.up.tensor()?;
                    if let Some(grad) = grad_map.get(up_tensor.id()) {
                        let new_up = optimizer.update(&up_name, &up_tensor, grad)?;
                        // Update the parameter with the new tensor
                        adapter.up.apply_update(&up_tensor.sub(&new_up)?)?;
                    }
                }

                // Increment optimizer step counter
                optimizer.step()?;
            }
        }
        Ok(())
    }

    fn save_checkpoint(&self, step: usize) -> flame_core::Result<()> {
        if let Some(lora_collection) = &self.lora_collection {
            // For now, just save the LoRA weights directly without CheckpointManager
            // TODO: Implement proper checkpoint management
            let checkpoint_path = self.output_dir.join(format!("checkpoint-{}.safetensors", step));
            lora_collection.save_weights(&checkpoint_path)?;
            info!("Saved checkpoint to {:?}", checkpoint_path);
        }
        Ok(())
    }

    fn generate_samples(&mut self, step: usize) -> flame_core::Result<()> {
        let sample_config = &self.config.sample;
        info!("Generating samples at step {}", step);

        // Create inference config
        let inference_config = InferenceConfig {
            height: 1024,
            width: 1024,
            num_inference_steps: sample_config.sample_steps,
            guidance_scale: sample_config.guidance_scale as f64,
            seed: Some(sample_config.seed),
        };

        // Create SDXL inference instance
        let mut sdxl_inference = SDXLInference::new(&self.device)?;

        // Apply LoRA weights if available
        if let Some(lora_collection) = &self.lora_collection {
            // Convert LoRA collection to weight format expected by inference
            let lora_weights = lora_collection.get_merged_weights()?;
            sdxl_inference.apply_lora(&lora_weights, 1.0)?;
        }

        // Generate samples for each prompt
        for (idx, prompt) in sample_config.prompts.iter().enumerate() {
            let negative_prompt = &sample_config.neg;

            // Run inference
            let image_tensor =
                sdxl_inference.generate(prompt, negative_prompt, &inference_config)?;

            // Save the image
            let filename = format!("sdxl_step_{:06}_sample_{:02}.png", step, idx);
            let output_path = self.output_dir.join(filename);

            super::unified_sampling::save_image(
                &image_tensor,
                &self.output_dir,
                "sdxl",
                step,
                idx,
                prompt,
            )?;
        }
        Ok(())
    }
}

// Re-export key types
pub use self::SDXLLoRATrainerFixed as SDXLLoRATrainer;
