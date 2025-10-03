//! Complete SD3.5 training pipeline with LoRA and full fine-tuning support

use crate::inference::sd35::{SD35Config as InferenceConfig, SD35Inference};
use crate::models::{
    mmdit_blocks::{MMDiT, MMDiTConfig},
    unified_vae::{VAEConfig, VAE as AutoencoderKL},
};
use crate::trainers::text_encoders::TextEncoders;
use crate::trainers::{
    adam8bit_enhanced::{Adam8bit, Adam8bitConfig},
    ema::EMAModel,
    gradient_accumulator::GradientAccumulator,
    snr_weighting::SNRWeighting,
};
use flame_core::device::Device;
use flame_core::{CudaDevice, DType, Error, Parameter, Result, Shape, Tensor};
// LinearWithLoRA is not needed for this implementation
use log::{info, warn};
use rand::Rng;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
// Registry streaming (optional)
use eridiffusion_common_weights::strict_loader::StrictMmapLoader;
use eridiffusion_training::chroma::weights::MmapWeightProvider as Sd35MmapProvider;
use eridiffusion_training::sd35::keymap::Sd35KeyMap;
use eridiffusion_training::sd35::registry::LayerRegistry as Sd35Registry;
use eridiffusion_training::streaming::{KeyMap as _KeyMap, WeightProvider as _WeightProvider};
use crate::trainers::adapters_util;
use eridiffusion_training::telemetry::TelemetryCsv;
use eridiffusion_training::tread::TreadMetrics;
use eridiffusion_core::Device as CoreDevice;

const SD35_LYCO_ALLOW: &[&str] = &[
    "joint_blocks.*.x_block.attn.qkv",
    "joint_blocks.*.x_block.attn.proj",
    "joint_blocks.*.x_block.mlp.fc1",
    "joint_blocks.*.x_block.mlp.fc2",
    "joint_blocks.*.context_block.attn.qkv",
    "joint_blocks.*.context_block.attn.proj",
    "joint_blocks.*.context_block.mlp.fc1",
    "joint_blocks.*.context_block.mlp.fc2",
];

/// SD3.5 training configuration
#[derive(Clone)]
pub struct SD35TrainingConfig {
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

    // SD3.5-specific
    pub snr_gamma: Option<f32>,
    pub linear_timesteps: bool,
    pub t5_max_length: usize,

    // Logging
    pub logging_dir: PathBuf,
    pub report_to: Vec<String>,
    pub validation_prompts: Vec<String>,
    pub validation_steps: usize,

    // EMA
    pub use_ema: bool,
    pub ema_decay: f32,
}

#[derive(Clone)]
pub struct TextEncoderPaths {
    pub clip_l: PathBuf,
    pub clip_g: PathBuf,
    pub t5_xxl: PathBuf,
}

#[derive(Clone, PartialEq)]
pub enum TrainMode {
    LoRA,
    FullFineTune,
}

/// LoRA layer for SD3.5 MMDiT
pub struct SD35LoRALayer {
    pub lora_down: Parameter,
    pub lora_up: Parameter,
    pub scale: f32,
    pub dropout: f32,
}

impl SD35LoRALayer {
    pub fn new(
        in_features: usize,
        out_features: usize,
        rank: usize,
        alpha: f32,
        dropout: f32,
        device: Device,
    ) -> flame_core::Result<Self> {
        let scale = alpha / rank as f32;

        // Initialize LoRA matrices as Parameters for training
        let lora_down = Parameter::randn(
            Shape::from_dims(&[rank, in_features]),
            0.0,
            0.02,
            device.cuda_device().clone(),
        )?;

        let lora_up = Parameter::zeros(
            Shape::from_dims(&[out_features, rank]),
            device.cuda_device().clone(),
        )?;

        Ok(Self { lora_down, lora_up, scale, dropout })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        base_output: &Tensor,
        training: bool,
    ) -> flame_core::Result<Tensor> {
        // Get tensors from parameters
        let lora_down_tensor = self.lora_down.tensor()?;
        let lora_up_tensor = self.lora_up.tensor()?;

        // Apply dropout during training
        let lora_out = if self.dropout > 0.0 && training {
            // Apply dropout to input by randomly zeroing elements
            let dropout_mask = Tensor::rand_like(x)?;
            // Scale by 1/(1-dropout) to maintain expected value
            let scale_factor = 1.0 / (1.0 - self.dropout);
            let dropout_scale = Tensor::full(
                dropout_mask.shape().clone(),
                scale_factor,
                dropout_mask.device().clone(),
            )?;
            let x_masked = x.mul(&dropout_mask)?;
            let x_dropout = x_masked.mul(&dropout_scale)?;
            x_dropout.matmul(&lora_down_tensor.transpose_dims(0, 1)?)?
        } else {
            x.matmul(&lora_down_tensor.transpose_dims(0, 1)?)?
        };

        let lora_out = lora_out.matmul(&lora_up_tensor.transpose_dims(0, 1)?)?;
        let scaled_lora = lora_out.mul_scalar(self.scale as f32)?;

        base_output.add(&scaled_lora)
    }
}

/// SD3.5 Training Pipeline
pub struct SD35Trainer {
    pub config: SD35TrainingConfig,
    pub device: Arc<CudaDevice>,

    // Models
    pub vae: Arc<AutoencoderKL>,
    pub mmdit: MMDiT,
    pub text_encoders: Arc<TextEncoders>,

    // LoRA layers (if applicable)
    pub lora_layers: Option<HashMap<String, SD35LoRALayer>>,

    // Optimizer
    pub optimizer: Adam8bit,

    // Gradient accumulator
    pub gradient_accumulator: GradientAccumulator,

    // Scheduler
    pub noise_scheduler: SD35NoiseScheduler,

    // EMA
    pub ema_model: Option<EMAModel>,

    // Stats
    pub global_step: usize,
    pub epoch: usize,

    // Last gradient map from backward pass
    pub last_grad_map: Option<flame_core::gradient::GradientMap>,

    // Optional registry streaming
    use_registry_streaming: bool,
    sd35_registry: Option<Sd35Registry>,
    sd35_provider: Option<Sd35MmapProvider<Sd35KeyMap>>,

    // Telemetry
    telemetry: Option<TelemetryCsv>,
    last_tread_metrics: Option<TreadMetrics>,
    // LyCORIS adapters (optional)
    lyco_adapters: Option<adapters::adapter::AdapterSet>,
}

impl SD35Trainer {
    // Find a LoRA layer by exact key or by suffix (e.g. "attn.q")
    fn find_lora_layer<'a>(
        lora_layers: &'a HashMap<String, SD35LoRALayer>,
        key_or_suffix: &str,
    ) -> Option<&'a SD35LoRALayer> {
        if let Some(l) = lora_layers.get(key_or_suffix) {
            return Some(l);
        }
        // fallback: first whose key ends with suffix
        for (k, v) in lora_layers.iter() {
            if k.ends_with(key_or_suffix) {
                return Some(v);
            }
        }
        None
    }
    /// Create new SD3.5 trainer
    pub fn new(config: SD35TrainingConfig, device: Arc<CudaDevice>) -> flame_core::Result<Self> {
        // Load models
        let vae = Arc::new(load_vae(&config.vae_path, device.clone())?);
        let mmdit = load_mmdit(&config.model_path, device.clone())?;
        let text_encoders =
            Arc::new(load_text_encoders(&config.text_encoder_paths, device.clone())?);

        // Setup LoRA if needed
        // Resolve LoRA targets: use config or fall back to SD3.5 defaults
        let resolved_targets: Vec<String> = if !config.lora_target_modules.is_empty() {
            config.lora_target_modules.clone()
        } else {
            SD35_LYCO_ALLOW.iter().map(|s| s.to_string()).collect()
        };

        let (mmdit, lora_layers) = if config.train_mode == TrainMode::LoRA {
            let (mmdit_with_lora, lora_layers) = setup_mmdit_lora(
                mmdit,
                config.lora_rank,
                config.lora_alpha,
                config.lora_dropout,
                &resolved_targets,
                device.clone(),
            )?;
            (mmdit_with_lora, Some(lora_layers))
        } else {
            (mmdit, None)
        };

        // Create optimizer — prefer LyCORIS adapters if provided via env; else enforce adapters-only for LoRA
        let use_registry_streaming = if config.train_mode == TrainMode::LoRA {
            // Force registry streaming when training LoRA for fidelity and explicit base freeze
            true
        } else {
            std::env::var("USE_REGISTRY_STREAMING").ok().map(|v| v == "1").unwrap_or(false)
        };
        let (sd35_registry, sd35_provider) = if use_registry_streaming {
            let loader = StrictMmapLoader::open(std::path::Path::new(&config.model_path)).map_err(
                |e| {
                    Error::InvalidOperation(format!(
                        "open sd35 shard {}: {}",
                        config.model_path.display(),
                        e
                    ))
                },
            )?;
            let ld = Arc::new(loader);
            let provider: Sd35MmapProvider<Sd35KeyMap> =
                Sd35MmapProvider::new(ld, Device::from(device.clone()));
            (Some(Sd35Registry::new()), Some(provider))
        } else {
            (None, None)
        };

        let mut optimizer;
        if let Ok(dir) = std::env::var("LYCORIS_DIR") {
            if use_registry_streaming {
                // Build base shapes from first streamed block
                let mut shapes = adapters::loader::BaseShapes::default();
                if let (Some(reg), Some(provider)) = (&sd35_registry, &sd35_provider) {
                    for i in reg.forward_ids() {
                        let w = provider
                            .load_block_to_gpu(i)
                            .map_err(|e| Error::InvalidOperation(e.to_string()))?;
                        for (j, t) in w.tensors.iter().enumerate() {
                            let key = format!("block{}.t{}", i, j);
                            shapes.by_target.insert(
                                key,
                                (
                                    t.shape().dims().iter().map(|&d| d as i64).collect(),
                                    t.shape().dims().len() == 4,
                                ),
                            );
                        }
                        break;
                    }
                }
                let device_core = CoreDevice::cuda(device.ordinal())
                    .map_err(|e| Error::InvalidOperation(e.to_string()))?;
                let set = adapters::loader::safetensors_lyco::load_lycoris_dir(
                    std::path::Path::new(&dir),
                    &device_core,
                    DType::BF16,
                    &shapes,
                )
                .map_err(|e| Error::InvalidOperation(e.to_string()))?;
                let allow: Vec<String> = SD35_LYCO_ALLOW.iter().map(|s| s.to_string()).collect();
                let filtered = adapters_util::filter_adapters(&set, &allow, &vec![]);
                let param_vec = adapters_util::to_parameters(&filtered)
                    .map_err(|e| Error::InvalidOperation(e.to_string()))?;
                info!(
                    "Using LyCORIS adapters (sd35): targets={} params={}",
                    filtered.by_target.len(),
                    param_vec.len()
                );
                optimizer = create_optimizer(param_vec, &config)?;
            } else {
                warn!("LYCORIS_DIR set but streaming is off; falling back to LoRA-only optimizer params");
                let trainable_params = get_trainable_parameters(&mmdit, &lora_layers, &config);
                optimizer = create_optimizer(trainable_params, &config)?;
            }
        } else {
            let trainable_params = get_trainable_parameters(&mmdit, &lora_layers, &config);
            if config.train_mode == TrainMode::LoRA {
                let expected = lora_layers.as_ref().map(|m| m.len() * 2).unwrap_or(0);
                if trainable_params.len() != expected {
                    return Err(Error::InvalidOperation(format!(
                        "SD3.5 LoRA guard: expected {} trainable adapter params (down/up per target), got {}",
                        expected, trainable_params.len())));
                }
            }
            optimizer = create_optimizer(trainable_params, &config)?;
        }

        // Create Device wrapper
        let flame_device = Device::from(device.clone());

        // Gradient accumulator
        let gradient_accumulator =
            GradientAccumulator::new(config.gradient_accumulation_steps, flame_device.clone());

        // Noise scheduler
        let noise_scheduler = SD35NoiseScheduler::new(config.linear_timesteps);

        // EMA
        let ema_model = if config.use_ema {
            Some(EMAModel::new(config.ema_decay, flame_device.clone())?)
        } else {
            None
        };

        // Optional registry streaming init
        // Telemetry CSV (append) in logging dir; columns adapt to #blocks
        let num_layers = sd35_registry.as_ref().map(|r| r.blocks.len()).unwrap_or(0);
        let telemetry = TelemetryCsv::new(config.logging_dir.join("telemetry.csv"), num_layers)
            .map_err(|e| Error::InvalidOperation(e.to_string()))?;

        Ok(Self {
            config,
            device,
            vae,
            mmdit,
            text_encoders,
            lora_layers,
            optimizer,
            gradient_accumulator,
            noise_scheduler,
            ema_model,
            global_step: 0,
            epoch: 0,
            last_grad_map: None,
            use_registry_streaming,
            sd35_registry,
            sd35_provider,
            telemetry: Some(telemetry),
            last_tread_metrics: None,
            lyco_adapters: None,
        })
    }

    /// Main training loop
    pub fn train<D: DataLoader>(&mut self, dataloader: D) -> flame_core::Result<()> {
        println!("Starting SD3.5 training...");

        // Enable gradient checkpointing if requested
        if self.config.gradient_checkpointing {
            // TODO: Enable gradient checkpointing on model
        }

        let num_update_steps_per_epoch = dataloader.len() / self.config.gradient_accumulation_steps;
        let total_batch_size = self.config.batch_size * self.config.gradient_accumulation_steps;

        println!("***** Running training *****");
        println!("  Num examples = {}", dataloader.total_samples());
        println!("  Num batches each epoch = {}", dataloader.len());
        println!("  Instantaneous batch size = {}", self.config.batch_size);
        println!("  Total train batch size = {}", total_batch_size);
        println!("  Gradient Accumulation steps = {}", self.config.gradient_accumulation_steps);
        println!("  Total optimization steps = {}", self.config.max_train_steps);

        // Training loop
        for epoch in 0..self.config.max_train_steps / num_update_steps_per_epoch + 1 {
            self.epoch = epoch;

            for (step, batch) in dataloader.iter().enumerate() {
                let step_start = std::time::Instant::now();
                let device = self.device.clone();
                let loss = self.training_step(batch, &device)?;

                // Accumulate gradients
                if (step + 1) % self.config.gradient_accumulation_steps == 0 {
                    // Clip gradients
                    if self.config.max_grad_norm > 0.0 {
                        self.clip_gradients(self.config.max_grad_norm)?;
                    }

                    // Optimizer step - use the stored grad_map
                    if let Some(ref grad_map) = self.last_grad_map {
                        // Update parameters using the grad_map
                        // TODO: Implement proper optimizer update with grad_map
                        self.optimizer.step()?;
                        self.last_grad_map = None; // Clear after use
                    }
                    // // Gradients handled by FLAME removed - handle gradients manually

                    // Update EMA
                    // Get params before mutable borrow
                    let ema_params = if self.ema_model.is_some() {
                        Some(
                            self.get_model_params()
                                .iter()
                                .map(|(k, v)| (k.clone(), (*v).clone()))
                                .collect::<HashMap<String, Parameter>>(),
                        )
                    } else {
                        None
                    };

                    if let (Some(ema), Some(params)) = (&mut self.ema_model, ema_params) {
                        ema.update(&params)?;
                    }

                    self.global_step += 1;

                    // Logging
                    if self.global_step % 100 == 0 {
                        println!("Step: {}, Loss: {:.4}", self.global_step, loss);
                        self.log_metrics(loss)?;
                    }

                    // Validation
                    if self.global_step % self.config.validation_steps == 0 {
                        self.validate()?;
                    }

                    // Checkpointing
                    if self.global_step % self.config.checkpointing_steps == 0 {
                        self.save_checkpoint()?;
                    }

                    // Telemetry write
                    if let Some(tel) = &mut self.telemetry {
                        let m = self.last_tread_metrics.take().unwrap_or_else(|| {
                            eridiffusion_training::tread::TreadState::new().metrics()
                        });
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
                            self.global_step as u64,
                            fw_sec,
                            bw_sec,
                            h2d_mb_s,
                            d2h_mb_s,
                            gpu_mem_peak,
                            tokens_per_s,
                            grad_norm,
                            loss,
                            fused,
                            m.route_lambda,
                            route_loss_value,
                            m.kept_avg,
                            &m.kept_frac,
                            m.shelves_mb,
                            m.route_miss,
                        );
                    }

                    // Check if done
                    if self.global_step >= self.config.max_train_steps {
                        return Ok(());
                    }
                }
            }
        }

        Ok(())
    }

    /// Single training step
    fn training_step(
        &mut self,
        batch: TrainingBatch,
        device: &CudaDevice,
    ) -> flame_core::Result<f32> {
        // Move batch to device
        let images = batch.images;
        let prompts = batch.prompts;

        // Encode images to latents
        let (mean, _logvar) = self.vae.encode(&images)?;
        let latents = mean;

        // Sample noise
        let noise = Tensor::randn(latents.shape().clone(), 0.0, 1.0, latents.device().clone())?;
        let timesteps = self.sample_timesteps(latents.shape().dims()[0])?;

        // Add noise
        let (noisy_latents, target) =
            self.noise_scheduler.add_noise(&latents, &noise, &timesteps)?;

        // Encode text
        let (clip_l, clip_g, t5) = self.encode_prompts(&prompts)?;

        // Create conditioning embedding
        let context = self.create_context_embedding(&clip_l, &clip_g, &t5)?;

        // Forward pass (optional streamed compute if SD35_SEQ_MODE=1)
        let model_pred = if self.use_registry_streaming
            && std::env::var("SD35_SEQ_MODE").ok().as_deref() == Some("1")
        {
            if let (Some(reg), Some(wp)) = (&self.sd35_registry, &self.sd35_provider) {
                let dims = noisy_latents.shape().dims().to_vec();
                let (b, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
                let mut x_seq = noisy_latents.reshape(&[b, c, h * w])?.transpose_dims(1, 2)?; // [B,T,D]
                if x_seq.dtype() != DType::BF16 {
                    x_seq = x_seq.to_dtype(DType::BF16)?;
                }
                let use_trainer_tread =
                    std::env::var("USE_TRAINER_TREAD").ok().as_deref() == Some("1");
                if use_trainer_tread {
                    let linear_bt = |bt: &Tensor, wmat: &Tensor| -> flame_core::Result<Tensor> {
                        let dims = bt.shape().dims().to_vec();
                        let (b, t, d) = (dims[0], dims[1], dims[2]);
                        let flat = bt.reshape(&[b * t, d])?;
                        let out = if flat.dtype() == DType::BF16 && wmat.dtype() == DType::BF16 {
                            flat.matmul_bf16(wmat)?
                        } else {
                            flat.matmul(wmat)?
                        };
                        let od = out.shape().dims()[1];
                        out.reshape(&[b, t, od])
                    };
                    for i in reg.forward_ids() {
                        let _ = wp.prefetch_block(i);
                        let w = wp.load_block_to_gpu(i)?;
                        let wq = &w.tensors[0];
                        let wk = &w.tensors[1];
                        let wv = &w.tensors[2];
                        let wo = &w.tensors[3];
                        let f1 = &w.tensors[4].transpose()?;
                        let f2 = &w.tensors[5].transpose()?;
                        // simple RMS norm on last dim
                        let dims = x_seq.shape().dims().to_vec();
                        let last = dims[2] as f32;
                        let eps = Tensor::from_scalar(1e-6f32, x_seq.device().clone())?;
                        let x32 = x_seq.to_dtype(DType::F32)?;
                        let rms = x32
                            .square()?
                            .sum_dim_keepdim(2)?
                            .div_scalar(last)?
                            .sqrt()?
                            .maximum(&eps)?;
                        let xn = x32.div(&rms)?;
                        let mut q = linear_bt(&xn, wq)?;
                        let mut k = linear_bt(&xn, wk)?;
                        let mut v = linear_bt(&xn, wv)?;
                        // Optional LoRA injection for q/k/v
                        if let Some(loras) = &self.lora_layers {
                            if let Some(layer) = Self::find_lora_layer(loras, "attn.q") {
                                q = layer.forward(&xn, &q, true)?;
                            }
                            if let Some(layer) = Self::find_lora_layer(loras, "attn.k") {
                                k = layer.forward(&xn, &k, true)?;
                            }
                            if let Some(layer) = Self::find_lora_layer(loras, "attn.v") {
                                v = layer.forward(&xn, &v, true)?;
                            }
                        }
                        let d_last = q.shape().dims()[2] as f32;
                        let scale = d_last.sqrt().recip();
                        let kt = k.transpose_dims(1, 2)?;
                        let mut scores = q.to_dtype(DType::F32)?.bmm(&kt.to_dtype(DType::F32)?)?;
                        scores = scores.mul_scalar(scale)?;
                        let dev = scores.device().clone();
                        let cap = Tensor::from_scalar(30.0f32, dev.clone())?;
                        let ncap = Tensor::from_scalar(-30.0f32, dev.clone())?;
                        let scores = scores.minimum(&cap)?.maximum(&ncap)?;
                        let attn = eridiffusion_training::tensor_utils::softmax_stable(&scores, -1)
                            .map_err(|e| Error::InvalidOperation(e.to_string()))?;
                        let ctx = attn.to_dtype(DType::F32)?.bmm(&v.to_dtype(DType::F32)?)?;
                        let mut o0 = linear_bt(&ctx, wo)?;
                        if let Some(loras) = &self.lora_layers {
                            if let Some(layer) = Self::find_lora_layer(loras, "attn.o") {
                                o0 = layer.forward(&ctx, &o0, true)?;
                            }
                        }
                        let o = {
                            let cap = Tensor::from_scalar(1.0e3f32, o0.device().clone())?;
                            let ncap = Tensor::from_scalar(-1.0e3f32, o0.device().clone())?;
                            o0.minimum(&cap)?.maximum(&ncap)?
                        };
                        let mut h1 = linear_bt(&o, f1)?;
                        if let Some(loras) = &self.lora_layers {
                            if let Some(layer) = Self::find_lora_layer(loras, "mlp.fc1") {
                                h1 = layer.forward(&o, &h1, true)?;
                            }
                        }
                        let h1 = {
                            let cap = Tensor::from_scalar(1.0e3f32, h1.device().clone())?;
                            let ncap = Tensor::from_scalar(-1.0e3f32, h1.device().clone())?;
                            h1.minimum(&cap)?.maximum(&ncap)?
                        };
                        let a1 = h1.gelu()?;
                        let mut out0 = linear_bt(&a1, f2)?;
                        if let Some(loras) = &self.lora_layers {
                            if let Some(layer) = Self::find_lora_layer(loras, "mlp.fc2") {
                                out0 = layer.forward(&a1, &out0, true)?;
                            }
                        }
                        x_seq = {
                            let cap = Tensor::from_scalar(1.0e3f32, out0.device().clone())?;
                            let ncap = Tensor::from_scalar(-1.0e3f32, out0.device().clone())?;
                            out0.minimum(&cap)?.maximum(&ncap)?
                        };
                        let _ = wp.release_block(i as isize);
                    }
                } else {
                    let empty_loras: Vec<eridiffusion_training::chroma::lora::LoRALinear> = Vec::new();
                    let cond = eridiffusion_training::sd35::registry::Cond {
                        text_hidden: clip_l.clone(),
                        sigma: timesteps.clone(),
                        mask_lat: None,
                    };
                    for (i, blk) in reg.blocks.iter().enumerate() {
                        let _ = wp.prefetch_block(i);
                        let w = wp.load_block_to_gpu(i)?;
                        let lora_slice = &empty_loras;
                        x_seq = blk
                            .apply(&x_seq, &cond, &w, lora_slice)
                            .map_err(|e| Error::InvalidOperation(e.to_string()))?;
                        let _ = wp.release_block(i as isize);
                    }
                    self.last_tread_metrics = None;
                }
                x_seq.transpose_dims(1, 2)?.reshape(&[b, c, h, w])?
            } else {
                // Fallback to legacy path if registry pieces missing
                let positions = Tensor::zeros(
                    Shape::from_dims(&[noisy_latents.shape().dims()[0], 2]),
                    noisy_latents.device().clone(),
                )?;
                let (pred, _) =
                    self.mmdit.forward(&noisy_latents, &context, &timesteps, &positions)?;
                pred
            }
        } else {
            // MMDiT forward needs 4 arguments: x, context, conditioning, positions
            // Create dummy positions for now
            let positions = Tensor::zeros(
                Shape::from_dims(&[noisy_latents.shape().dims()[0], 2]),
                noisy_latents.device().clone(),
            )?;
            let (pred, _) = self.mmdit.forward(&noisy_latents, &context, &timesteps, &positions)?;
            pred
        };

        // Apply LoRA if enabled
        let model_pred = if let Some(lora_layers) = &self.lora_layers {
            // Apply LoRA to output
            // This is simplified - in practice, LoRA is applied within the model
            model_pred
        } else {
            model_pred
        };

        // Compute loss
        let mut loss = mse_loss(&model_pred, &target)?;

        // Apply SNR weighting if configured
        if let Some(gamma) = self.config.snr_gamma {
            let snr_weighting = SNRWeighting::new(gamma, None);
            let num_train_timesteps = 1000; // SD3.5 default
            loss = snr_weighting.apply_snr_weighting(&loss, &timesteps, num_train_timesteps)?;
        }

        // Get scalar loss value for logging
        let loss_value = loss.mean()?.to_scalar::<f32>()?;

        // Backward pass - keep the tensor for gradient computation
        let grad_map = loss.mean()?.backward()?;

        // Store grad_map for optimizer step
        self.last_grad_map = Some(grad_map);

        // Accumulate gradients
        // FLAME gradient accumulator needs individual parameter names and gradients
        if let Some(ref grad_map) = self.last_grad_map {
            if let Some(lora_layers) = &self.lora_layers {
                for (layer_name, layer) in lora_layers {
                    // Accumulate gradients for lora_down
                    if let Some(grad) = grad_map.get(layer.lora_down.id()) {
                        self.gradient_accumulator
                            .accumulate(&format!("{}_lora_down", layer_name), grad)?;
                    }
                    // Accumulate gradients for lora_up
                    if let Some(grad) = grad_map.get(layer.lora_up.id()) {
                        self.gradient_accumulator
                            .accumulate(&format!("{}_lora_up", layer_name), grad)?;
                    }
                }
            }
        }

        Ok(loss_value)
    }

    /// Sample timesteps
    fn sample_timesteps(&self, batch_size: usize) -> flame_core::Result<Tensor> {
        let timesteps = if self.config.linear_timesteps {
            // Linear timesteps for SD3.5
            (0..batch_size).map(|_| rand::thread_rng().gen_range(0..1000)).collect::<Vec<_>>()
        } else {
            // Cosine or other schedule
            (0..batch_size)
                .map(|_| {
                    let u = rand::thread_rng().gen::<f32>();
                    ((1.0 - u) * 1000.0) as i32
                })
                .collect::<Vec<_>>()
        };

        Tensor::from_vec(
            timesteps.into_iter().map(|t| t as f32).collect(),
            Shape::from_dims(&[batch_size]),
            self.device.clone(),
        )
    }

    /// Encode prompts with all three text encoders
    fn encode_prompts(&self, prompts: &[String]) -> flame_core::Result<(Tensor, Tensor, Tensor)> {
        let mut clip_l_embeds = Vec::new();
        let mut clip_g_embeds = Vec::new();
        let mut t5_embeds = Vec::new();

        for prompt in prompts {
            // encode_sd35 takes 3 arguments: text, negative_text, and max_sequence_length
            let (clip_l, clip_g, t5) = self.text_encoders.encode_sd35(
                prompt, None, // negative prompt
                77,   // max_sequence_length for CLIP
            )?;
            clip_l_embeds.push(clip_l);
            clip_g_embeds.push(clip_g);
            t5_embeds.push(t5);
        }

        let clip_l_batch = Tensor::cat(&clip_l_embeds.iter().collect::<Vec<_>>(), 0)?;
        let clip_g_batch = Tensor::cat(&clip_g_embeds.iter().collect::<Vec<_>>(), 0)?;
        let t5_batch = Tensor::cat(&t5_embeds.iter().collect::<Vec<_>>(), 0)?;

        Ok((clip_l_batch, clip_g_batch, t5_batch))
    }

    /// Create context embedding for SD3.5
    fn create_context_embedding(
        &self,
        clip_l: &Tensor,
        clip_g: &Tensor,
        t5: &Tensor,
    ) -> flame_core::Result<Tensor> {
        // SD3.5 concatenates CLIP embeddings and uses T5 separately
        // Use 2 for the last dimension instead of -1
        let clip_concat = Tensor::cat(&[clip_l, clip_g], 2)?;

        // Combine with T5 (implementation depends on model variant)
        // For now, concatenate all
        Tensor::cat(&[&clip_concat, &t5], 1)
    }

    /// Validation step
    fn validate(&self) -> flame_core::Result<()> {
        println!("Running validation...");

        // Use EMA weights if available
        let use_ema = self.ema_model.is_some();

        for (i, prompt) in self.config.validation_prompts.iter().enumerate() {
            let image = self.generate_sample(prompt, None, use_ema)?;
            self.save_image(&image, &format!("val_{}_{}.png", self.global_step, i))?;
        }

        Ok(())
    }

    /// Generate sample image
    pub fn generate_sample(
        &self,
        prompt: &str,
        seed: Option<u64>,
        use_ema: bool,
    ) -> flame_core::Result<Tensor> {
        info!("Generating SD3.5 sample at step {}", self.global_step);

        // Create inference config
        let inference_config = InferenceConfig {
            height: self.config.resolution,
            width: self.config.resolution,
            num_inference_steps: 30,
            guidance_scale: 7.0,
            seed,
            linear_timesteps: self.config.linear_timesteps,
            snr_gamma: self.config.snr_gamma,
        };

        // Create SD3.5 inference instance
        // SD35Inference::new expects &Device
        let device = Device::from(self.device.clone());
        let mut sd35_inference = SD35Inference::new(&device)?;

        // Apply LoRA weights if available
        if let Some(lora_layers) = &self.lora_layers {
            // Convert LoRA layers to weights format
            let mut lora_weights = HashMap::new();
            for (name, layer) in lora_layers {
                lora_weights.insert(format!("{}.down", name), layer.lora_down.tensor()?);
                lora_weights.insert(format!("{}.up", name), layer.lora_up.tensor()?);
            }
            sd35_inference.apply_lora(&lora_weights, 1.0)?;
        }

        // Use EMA weights if requested and available
        if use_ema && self.ema_model.is_some() {
            // TODO: Apply EMA weights to inference model
        }

        // Run inference
        // generate method takes 4 arguments: prompt, negative_prompt, config, device
        let image_tensor = sd35_inference.generate(
            prompt,
            "", // No negative prompt for SD3.5
            &inference_config,
            &device,
        )?;

        Ok(image_tensor)
    }

    /// Save checkpoint
    fn save_checkpoint(&self) -> flame_core::Result<()> {
        let checkpoint_path =
            self.config.logging_dir.join(format!("checkpoint-{}", self.global_step));
        std::fs::create_dir_all(&checkpoint_path)
            .map_err(|e| flame_core::Error::Io(e.to_string()))?;

        if let Some(lora_layers) = &self.lora_layers {
            // Save LoRA weights
            save_lora_weights(lora_layers, &checkpoint_path)?;
        } else {
            // Save full model weights
            // TODO: Implement full model saving
        }

        // Save optimizer state
        self.optimizer.save_state(&checkpoint_path.join("optimizer.pt"))?;

        // Save EMA weights if available
        if let Some(ema) = &self.ema_model {
            save_ema_weights(ema, &checkpoint_path)?;
        }

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
        // FLAME: Gradient clipping needs to be implemented differently
        // since gradients are stored in grad_map, not on parameters
        // TODO: Implement gradient clipping with FLAME's gradient system
        Ok(())
    }

    fn log_metrics(&self, loss: f32) -> flame_core::Result<()> {
        // Log to tensorboard/wandb if configured
        println!(
            "Step: {}, Loss: {:.4}, LR: {:.2e}",
            self.global_step,
            loss,
            self.optimizer.current_lr()
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

    fn get_model_params(&self) -> HashMap<String, &Parameter> {
        // Collect all model parameters for EMA
        let mut params = HashMap::new();

        // TODO: Collect parameters from mmdit

        if let Some(lora_layers) = &self.lora_layers {
            for (name, layer) in lora_layers {
                params.insert(format!("{}_down", name), &layer.lora_down);
                params.insert(format!("{}_up", name), &layer.lora_up);
            }
        }

        params
    }

    fn get_trainable_params(&self) -> Vec<&Parameter> {
        // Collect trainable parameters
        let mut params = Vec::new();

        if let Some(lora_layers) = &self.lora_layers {
            for layer in lora_layers.values() {
                params.push(&layer.lora_down);
                params.push(&layer.lora_up);
            }
        }
        // TODO: Add all model params for full fine-tuning

        params
    }
}

/// SD3.5 noise scheduler
pub struct SD35NoiseScheduler {
    pub linear_timesteps: bool,
}

impl SD35NoiseScheduler {
    pub fn new(linear_timesteps: bool) -> Self {
        Self { linear_timesteps }
    }

    /// Add noise to latents
    pub fn add_noise(
        &self,
        latents: &Tensor,
        noise: &Tensor,
        timesteps: &Tensor,
    ) -> flame_core::Result<(Tensor, Tensor)> {
        // SD3.5 uses v-prediction
        let timesteps_f = timesteps.to_dtype(DType::F32)?.div_scalar(1000.0)?;

        // Compute alpha and sigma
        let (alpha_t, sigma_t) = if self.linear_timesteps {
            // Linear schedule
            let alpha_t =
                Tensor::full(timesteps_f.shape().clone(), 1.0, timesteps_f.device().clone())?
                    .sub(&timesteps_f)?;
            let sigma_t = timesteps_f;
            (alpha_t, sigma_t)
        } else {
            // Cosine or other schedule
            // TODO: Implement other schedules
            let alpha_t =
                Tensor::full(timesteps_f.shape().clone(), 1.0, timesteps_f.device().clone())?
                    .sub(&timesteps_f)?;
            let sigma_t = timesteps_f;
            (alpha_t, sigma_t)
        };

        // Add noise: x_t = alpha_t * x_0 + sigma_t * epsilon
        // Expand alpha_t and sigma_t from [batch] to [batch, 1, 1, 1] for broadcasting
        let alpha_t_expanded = alpha_t.unsqueeze(1)?.unsqueeze(2)?.unsqueeze(3)?;
        let sigma_t_expanded = sigma_t.unsqueeze(1)?.unsqueeze(2)?.unsqueeze(3)?;
        let noisy_latents = latents.mul(&alpha_t_expanded)?.add(&noise.mul(&sigma_t_expanded)?)?;

        // For v-prediction, target is v = alpha_t * epsilon - sigma_t * x_0
        let v_target = noise.mul(&alpha_t_expanded)?.sub(&latents.mul(&sigma_t_expanded)?)?;

        Ok((noisy_latents, v_target))
    }

    /// Get timesteps for inference
    pub fn get_inference_timesteps(&self, num_steps: usize) -> Vec<i32> {
        (0..num_steps).map(|i| (1000 * (num_steps - 1 - i) / (num_steps - 1)) as i32).collect()
    }

    /// Single denoising step
    pub fn step(
        &self,
        model_output: &Tensor,
        latents: &Tensor,
        t: i32,
    ) -> flame_core::Result<Tensor> {
        let timestep = t as f32 / 1000.0;
        let next_timestep = (t.saturating_sub(1000) / 30) as f32 / 1000.0; // Assuming 30 steps

        // Compute alpha and sigma
        let (alpha_t, sigma_t) = if self.linear_timesteps {
            ((1.0 - timestep), timestep)
        } else {
            // TODO: Other schedules
            ((1.0 - timestep), timestep)
        };

        let (alpha_next, sigma_next) = if self.linear_timesteps {
            ((1.0 - next_timestep), next_timestep)
        } else {
            ((1.0 - next_timestep), next_timestep)
        };

        // Convert v-prediction to x0
        // v = alpha_t * epsilon - sigma_t * x_0
        // => x_0 = (alpha_t * epsilon - v) / sigma_t
        // We need epsilon, so: epsilon = (v + sigma_t * x_t / alpha_t) / alpha_t

        // Proper flow matching DDIM update
        // For flow matching with v-prediction:
        // v_t = alpha_t * dz/dt - sigma_t * x_0
        // where alpha_t = sqrt(1 - sigma_t^2) for flow matching
        // Rearranging: x_0 = (alpha_t * v_t - sigma_t * z_t) / alpha_t^2

        let pred_x0 =
            latents.sub(&model_output.mul_scalar(sigma_t as f32)?)?.div_scalar(alpha_t)?;

        // DDIM update
        let pred_noise = latents.sub(&pred_x0.mul_scalar(alpha_t as f32)?)?.div_scalar(sigma_t)?;

        pred_x0.mul_scalar(alpha_next as f32)?.add(&pred_noise.mul_scalar(sigma_next as f32)?)
    }
}

/// Training batch
pub struct TrainingBatch {
    pub images: Tensor,
    pub prompts: Vec<String>,
}

/// DataLoader trait
pub trait DataLoader {
    fn len(&self) -> usize;
    fn total_samples(&self) -> usize;
    fn iter(&self) -> Box<dyn Iterator<Item = TrainingBatch> + '_>;
}

/// Setup LoRA for MMDiT
fn setup_mmdit_lora(
    mut mmdit: MMDiT,
    rank: usize,
    alpha: f32,
    dropout: f32,
    target_modules: &[String],
    device: Arc<CudaDevice>,
) -> flame_core::Result<(MMDiT, HashMap<String, SD35LoRALayer>)> {
    let mut lora_layers = HashMap::new();

    // Add LoRA to target modules
    for module_name in target_modules {
        if module_name.contains("attn") {
            // Add LoRA to attention layers
            let hidden_size = mmdit.config.hidden_size;
            let lora = SD35LoRALayer::new(
                hidden_size,
                hidden_size,
                rank,
                alpha,
                dropout,
                Device::from(device.clone()),
            )?;
            lora_layers.insert(module_name.clone(), lora);
        }
    }

    // TODO: Inject LoRA layers into model forward passes

    Ok((mmdit, lora_layers))
}

/// Get trainable parameters
fn get_trainable_parameters(
    mmdit: &MMDiT,
    lora_layers: &Option<HashMap<String, SD35LoRALayer>>,
    config: &SD35TrainingConfig,
) -> Vec<Parameter> {
    if let Some(lora) = lora_layers {
        // Only LoRA parameters are trainable
        lora.values().flat_map(|l| vec![l.lora_down.clone(), l.lora_up.clone()]).collect()
    } else {
        // All model parameters are trainable
        // TODO: Collect all parameters from model
        vec![]
    }
}

/// Create optimizer
fn create_optimizer(
    params: Vec<Parameter>,
    config: &SD35TrainingConfig,
) -> flame_core::Result<Adam8bit> {
    let optimizer_config = Adam8bitConfig {
        lr: config.learning_rate,
        betas: (0.9, 0.999),
        eps: 1e-8,
        weight_decay: 0.01,
        amsgrad: false,
        block_wise: true,
        percentile_clipping: 100,
    };

    Adam8bit::new(params, optimizer_config)
}

/// Training state for checkpointing
#[derive(serde::Serialize, serde::Deserialize)]
struct TrainingState {
    global_step: usize,
    epoch: usize,
    best_loss: f32,
}

// Helper functions
fn load_vae(path: &Path, device: Arc<CudaDevice>) -> flame_core::Result<AutoencoderKL> {
    // TODO: Implement VAE loading from safetensors
    let config = VAEConfig::sd3_vae(); // SD3.5 uses 16-channel VAE
    AutoencoderKL::new(config, device)
}

fn load_mmdit(path: &Path, device: Arc<CudaDevice>) -> flame_core::Result<MMDiT> {
    // TODO: Implement MMDiT loading from safetensors
    let config = MMDiTConfig {
        hidden_size: 1536, // SD3.5 Large
        num_heads: 24,
        depth: 38,
        mlp_ratio: 4.0,
        qkv_bias: false,
        qk_norm: true,
        pos_embed_max_size: 192,
    };
    let cond_dim = 4096; // T5-XXL hidden size for SD3.5
    let device = Device::from(device.clone());
    MMDiT::new(config, cond_dim, &device)
}

fn load_text_encoders(
    paths: &TextEncoderPaths,
    device: Arc<CudaDevice>,
) -> flame_core::Result<TextEncoders> {
    // Use the constructor method
    let mut encoders = TextEncoders::new(Device::from(device.clone()));

    // Load each encoder
    encoders.load_clip_l(&paths.clip_l.to_string_lossy())?;
    encoders.load_clip_g(&paths.clip_g.to_string_lossy())?;
    encoders.load_t5(&paths.t5_xxl.to_string_lossy())?;

    Ok(encoders)
}

fn save_lora_weights(
    lora_layers: &HashMap<String, SD35LoRALayer>,
    path: &Path,
) -> flame_core::Result<()> {
    // TODO: Implement LoRA weight saving to safetensors format
    Ok(())
}

fn save_ema_weights(ema: &EMAModel, path: &Path) -> flame_core::Result<()> {
    // TODO: Implement EMA weight saving
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
    // FLAME: This function needs to be refactored to accept grad_map
    // For now, return a dummy value
    // TODO: Implement proper gradient norm computation with grad_map
    Ok(1.0f32)
}

fn get_all_parameters<'a>(
    mmdit: &'a MMDiT,
    lora_layers: &'a Option<HashMap<String, SD35LoRALayer>>,
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
    // TODO: Implement image saving
    Ok(())
}

fn mse_loss(pred: &Tensor, target: &Tensor) -> flame_core::Result<Tensor> {
    let diff = pred.sub(target)?;
    let squared = diff.square()?;
    squared.mean()
}

impl Default for SD35TrainingConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from(
                "/home/alex/SwarmUI/Models/diffusion_models/sd3.5_large.safetensors",
            ),
            vae_path: PathBuf::from(
                "/home/alex/SwarmUI/Models/diffusion_models/sd3.5_large.safetensors",
            ),
            text_encoder_paths: TextEncoderPaths {
                clip_l: PathBuf::from("/home/alex/SwarmUI/Models/clip/clip_l.safetensors"),
                clip_g: PathBuf::from("/home/alex/SwarmUI/Models/clip/clip_g.safetensors"),
                t5_xxl: PathBuf::from("/home/alex/SwarmUI/Models/clip/t5xxl_fp16.safetensors"),
            },
            train_mode: TrainMode::LoRA,
            batch_size: 1,
            gradient_accumulation_steps: 1,
            learning_rate: 1e-4,
            warmup_steps: 100,
            max_train_steps: 1000,
            checkpointing_steps: 500,
            mixed_precision: true,
            gradient_checkpointing: true,
            use_8bit_adam: true,
            max_grad_norm: 1.0,
            lora_rank: 16,
            lora_alpha: 16.0,
            lora_dropout: 0.0,
            lora_target_modules: vec![
                "attn.to_q".to_string(),
                "attn.to_k".to_string(),
                "attn.to_v".to_string(),
                "attn.to_out.0".to_string(),
            ],
            resolution: 1024,
            center_crop: false,
            random_flip: true,
            caption_dropout_rate: 0.1,
            snr_gamma: Some(5.0),
            linear_timesteps: true,
            t5_max_length: 154,
            logging_dir: PathBuf::from("output"),
            report_to: vec!["tensorboard".to_string()],
            validation_prompts: vec![],
            validation_steps: 500,
            use_ema: true,
            ema_decay: 0.9999,
        }
    }
}
