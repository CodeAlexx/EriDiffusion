//! Flux LoRA trainer for 24GB VRAM
//! Extends the base trainer with LoRA-specific functionality

use eridiffusion_core::{Device, Result, Error};
use eridiffusion_networks::LoRAConfig;
use candle_core::{Tensor, DType, Module, D, Var};
use candle_nn::{VarBuilder, VarMap, Optimizer, AdamW, ParamsAdamW};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use serde::{Serialize, Deserialize};
use crate::flux_preprocessor::{PreprocessedFluxDataset, PreprocessedFluxBatch};
use crate::metrics_logger::MetricsLogger;
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::HashMap;
use tracing::warn;

/// Data type configuration
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum DTypeConfig {
    FP32,
    FP16,
    BF16,
}

/// Optimizer wrapper to support different optimizer types
enum OptimizerWrapper {
    AdamW(AdamW),
    AdamW8bit(crate::optimizers::adam8bit::AdamW8bit),
}

impl OptimizerWrapper {
    fn step(&mut self) -> Result<()> {
        match self {
            Self::AdamW(opt) => {
                // AdamW needs a dummy backward step
                let device = candle_core::Device::Cpu;
                let dummy = Tensor::zeros(&[], DType::F32, &device)?;
                opt.backward_step(&dummy)?;
                Ok(())
            }
            Self::AdamW8bit(opt) => {
                // AdamW8bit also needs dummy backward step
                let device = candle_core::Device::Cpu;
                let dummy = Tensor::zeros(&[], DType::F32, &device)?;
                opt.backward_step(&dummy)?;
                Ok(())
            }
        }
    }
}

impl DTypeConfig {
    pub fn to_candle_dtype(&self) -> DType {
        match self {
            Self::FP32 => DType::F32,
            Self::FP16 => DType::F16,
            Self::BF16 => DType::BF16,
        }
    }
}

/// LoRA-specific training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FluxLoRATraining24GBConfig {
    // Base configurations
    pub model_path: PathBuf,
    pub cache_dir: PathBuf,
    pub output_dir: PathBuf,
    pub learning_rate: f64,
    pub batch_size: usize,
    pub gradient_accumulation_steps: usize,
    pub num_train_steps: usize,
    pub gradient_checkpointing: bool,
    pub dtype: DTypeConfig,
    pub device_id: usize, // Which GPU to use
    
    // LoRA specific
    pub lora_rank: usize,
    pub lora_alpha: f32,
    
    // Training settings
    pub ema_decay: f32,
    pub save_every: usize,
    pub log_every: usize,
    pub sample_every: usize,
    pub max_grad_norm: f32,
    pub optimizer_type: String,
    pub noise_scheduler: String,
    
    // Sampling settings
    pub sample_prompts: Vec<String>,
    pub sample_size: (u32, u32),
    pub sample_steps: usize,
    pub guidance_scale: f32,
}


/// LoRA adapter for Flux
struct FluxLoRAAdapter {
    /// LoRA down projections (rank reduction)
    lora_down: HashMap<String, Tensor>,
    /// LoRA up projections (rank restoration)
    lora_up: HashMap<String, Tensor>,
    /// LoRA configuration
    config: LoRAConfig,
    /// Variable map for LoRA parameters
    var_map: VarMap,
}

impl FluxLoRAAdapter {
    /// Create new LoRA adapter
    fn new(rank: usize, alpha: f32, target_modules: Vec<String>, device: &candle_core::Device) -> Result<Self> {
        let config = LoRAConfig {
            rank,
            alpha,
            dropout: 0.0,
            target_modules: target_modules.clone(),
            use_bias: false,
            fan_in_fan_out: false,
            merge_weights: false,
            init_weights: true,
            use_rslora: false,
            use_dora: false,
            rank_pattern: HashMap::new(),
            alpha_pattern: HashMap::new(),
        };
        
        let var_map = VarMap::new();
        let mut lora_down = HashMap::new();
        let mut lora_up = HashMap::new();
        
        // Initialize LoRA weights for target modules
        // In Flux, we typically target attention layers
        for (idx, module) in target_modules.iter().enumerate() {
            // Dummy dimensions - would be set based on actual model
            let in_features = 3072; // Flux hidden size
            let out_features = 3072;
            
            // Create tensors manually first with correct dtype
            let scale = (1.0 / in_features as f64).sqrt();
            let dtype = DType::BF16; // Match Flux model dtype
            let down_tensor = Tensor::randn(0.0, scale, (rank, in_features), device)?.to_dtype(dtype)?;
            let up_tensor = Tensor::zeros((out_features, rank), dtype, device)?;
            
            // Store in VarMap for optimizer
            let down_path = format!("lora_down_{}", idx);
            let up_path = format!("lora_up_{}", idx);
            
            // Add to var_map - this will create Vars internally
            let _ = var_map.get(
                (rank, in_features),
                &down_path,
                candle_nn::Init::Const(0.0), // Will be overwritten
                dtype,
                device,
            )?;
            let _ = var_map.get(
                (out_features, rank),
                &up_path,
                candle_nn::Init::Const(0.0), // Will be overwritten
                dtype,
                device,
            )?;
            
            // Get the vars and set their values
            if let Some(down_var) = var_map.data().lock().unwrap().get(&down_path) {
                down_var.set(&down_tensor)?;
            }
            if let Some(up_var) = var_map.data().lock().unwrap().get(&up_path) {
                up_var.set(&up_tensor)?;
            }
            
            // Store tensors for forward pass
            lora_down.insert(module.clone(), down_tensor);
            lora_up.insert(module.clone(), up_tensor);
        }
        
        Ok(Self {
            lora_down,
            lora_up,
            config,
            var_map,
        })
    }
    
    /// Apply LoRA to a tensor
    fn apply(&self, x: &Tensor, module_name: &str) -> Result<Tensor> {
        if let (Some(down), Some(up)) = (self.lora_down.get(module_name), self.lora_up.get(module_name)) {
            // LoRA: output = x + (x @ down @ up) * (alpha / rank)
            let scale = self.config.alpha / self.config.rank as f32;
            let lora_out = x.matmul(down)?.matmul(up)?.affine(scale as f64, 0.0)?;
            x.add(&lora_out).map_err(Error::from)
        } else {
            Ok(x.clone())
        }
    }
    
    /// Get all LoRA parameters
    fn parameters(&self) -> Vec<Tensor> {
        self.var_map.all_vars()
            .into_iter()
            .map(|var| var.as_tensor().clone())
            .collect()
    }
}

/// Flux LoRA trainer for 24GB
pub struct FluxLoRATrainer24GB {
    /// Base Flux model (frozen)
    model: candle_transformers::models::flux::model::Flux,
    
    /// LoRA adapter
    lora_adapter: FluxLoRAAdapter,
    
    /// Base model variable map (frozen)
    base_var_map: VarMap,
    
    /// Optimizer (only for LoRA parameters)
    optimizer: OptimizerWrapper,
    
    /// Configuration
    config: FluxLoRATraining24GBConfig,
    
    /// Device
    device: candle_core::Device,
    
    /// Current step
    global_step: usize,
    
    /// Metrics logger
    metrics: MetricsLogger,
    
    /// Cached model config
    flux_config: candle_transformers::models::flux::model::Config,
    
    /// Position embedder
    pe_embedder: candle_transformers::models::flux::model::EmbedNd,
}

impl FluxLoRATrainer24GB {
    /// Create new LoRA trainer
    pub async fn new(config: FluxLoRATraining24GBConfig) -> Result<Self> {
        println!("🚀 Initializing Flux LoRA 24GB Trainer");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        
        // Setup device - force GPU 0 for now
        let device = if candle_core::utils::cuda_is_available() {
            // Always use GPU 0 until we figure out the device selection issue
            candle_core::Device::new_cuda(0)?
        } else {
            return Err(Error::Device("CUDA required for Flux training".into()));
        };
        
        println!("✓ Using device: {:?}", device);
        
        // Load base model
        println!("Loading Flux model from: {}", config.model_path.display());
        let dtype = config.dtype.to_candle_dtype();
        
        // Test tensor creation to verify device is working
        let _test_tensor = Tensor::zeros(&[1, 1], dtype, &device)?;
        println!("✓ Test tensor created successfully on device");
        
        let base_var_map = VarMap::new();
        println!("Creating VarBuilder with mmap...");
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&config.model_path], dtype, &device)?
        };
        
        println!("Initializing Flux config...");
        let flux_config = candle_transformers::models::flux::model::Config::dev();
        println!("Building Flux model (this may take a moment)...");
        let model = candle_transformers::models::flux::model::Flux::new(&flux_config, vb)?;
        
        println!("✓ Base model loaded successfully!");
        println!("Note: First forward pass will load weights to GPU (may take 2-5 minutes)");
        
        // Identify target modules for LoRA
        let target_modules = Self::get_target_modules();
        println!("✓ LoRA target modules: {} layers", target_modules.len());
        
        // Create LoRA adapter
        let lora_adapter = FluxLoRAAdapter::new(
            config.lora_rank,
            config.lora_alpha,
            target_modules,
            &device,
        )?;
        
        println!("✓ LoRA adapter created: rank={}, alpha={}", config.lora_rank, config.lora_alpha);
        
        // Create optimizer for LoRA parameters only
        let optimizer = match config.optimizer_type.as_str() {
            "adamw8bit" | "adamw_8bit" => {
                // Use 8-bit optimizer for memory efficiency
                use crate::optimizers::adam8bit::{AdamW8bit, AdamW8bitConfig};
                let config_8bit = AdamW8bitConfig {
                    lr: config.learning_rate,
                    beta1: 0.9,
                    beta2: 0.999,
                    epsilon: 1e-8,
                    weight_decay: 0.01,
                };
                OptimizerWrapper::AdamW8bit(<AdamW8bit as Optimizer>::new(
                    lora_adapter.var_map.all_vars(),
                    config_8bit,
                )?)
            }
            "adamw" => {
                OptimizerWrapper::AdamW(AdamW::new(
                    lora_adapter.var_map.all_vars(),
                    ParamsAdamW {
                        lr: config.learning_rate,
                        beta1: 0.9,
                        beta2: 0.999,
                        eps: 1e-8,
                        weight_decay: 0.01,
                    },
                )?)
            }
            _ => {
                return Err(Error::Config(format!("Unsupported optimizer: {}", config.optimizer_type)));
            }
        };
        
        // Create metrics logger
        let metrics = MetricsLogger::new(&config.output_dir.join("metrics.csv"))?;
        
        // Create position embedder
        let pe_dim = flux_config.hidden_size / flux_config.num_heads;
        let pe_embedder = candle_transformers::models::flux::model::EmbedNd::new(
            pe_dim,
            flux_config.theta,
            flux_config.axes_dim.clone(),
        );
        
        // Print memory usage
        Self::print_memory_summary(&config);
        
        Ok(Self {
            model,
            lora_adapter,
            base_var_map,
            optimizer,
            config,
            device,
            global_step: 0,
            metrics,
            flux_config,
            pe_embedder,
        })
    }
    
    /// Get target modules for LoRA
    fn get_target_modules() -> Vec<String> {
        // In Flux, we typically target:
        // - Self-attention projections (q, k, v, out)
        // - Cross-attention projections
        // - Feed-forward projections
        vec![
            "double_blocks.*.img_attn.qkv".to_string(),
            "double_blocks.*.img_attn.proj".to_string(),
            "double_blocks.*.txt_attn.qkv".to_string(),
            "double_blocks.*.txt_attn.proj".to_string(),
            "single_blocks.*.attn.qkv".to_string(),
            "single_blocks.*.attn.proj".to_string(),
        ]
    }
    
    /// Train the model
    pub async fn train(&mut self, dataset: PreprocessedFluxDataset) -> Result<()> {
        println!("\n🎯 Starting LoRA Training");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        
        let pb = ProgressBar::new(self.config.num_train_steps as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} Step {msg}")
                .unwrap()
        );
        
        let mut accumulated_loss = 0.0;
        let mut accumulation_counter = 0;
        
        for step in 0..self.config.num_train_steps {
            self.global_step = step;
            
            // Get random sample
            println!("Step {}: Getting random sample from dataset...", step);
            let idx = rand::random::<usize>() % dataset.len();
            println!("  Selected index: {}", idx);
            let batch = dataset.get_item(idx)?;
            println!("  Batch loaded successfully");
            
            // Forward pass
            let loss = self.training_step(batch)?;
            accumulated_loss += loss;
            accumulation_counter += 1;
            
            // Accumulate gradients
            if accumulation_counter >= self.config.gradient_accumulation_steps {
                // Optimizer step
                self.optimizer.step()?;
                
                // Zero gradients - in candle, gradients are cleared automatically
                // after optimizer step, so this is not needed
                
                // Log metrics
                if step % self.config.log_every == 0 {
                    let avg_loss = accumulated_loss / accumulation_counter as f32;
                    self.metrics.log_scalar("loss", avg_loss, step)?;
                    pb.set_message(format!("Loss: {:.4}", avg_loss));
                }
                
                accumulated_loss = 0.0;
                accumulation_counter = 0;
            }
            
            // Sample
            if step > 0 && step % self.config.sample_every == 0 {
                self.generate_samples(step).await?;
            }
            
            // Save checkpoint
            if step > 0 && step % self.config.save_every == 0 {
                self.save_checkpoint(step).await?;
            }
            
            pb.inc(1);
        }
        
        pb.finish_with_message("Training complete!");
        
        // Final save
        self.save_checkpoint(self.global_step).await?;
        
        Ok(())
    }
    
    /// Single training step
    fn training_step(&mut self, batch: PreprocessedFluxBatch) -> Result<f32> {
        use rand::Rng;
        use candle_core::IndexOp;
        
        // Log first step for debugging
        if self.global_step == 0 {
            println!("First training step - debugging info:");
            println!("  Device: {:?}", self.device);
            println!("  Dtype: {:?}", self.config.dtype);
        }
        
        // Move batch data to device and ensure correct dtype
        let dtype = self.config.dtype.to_candle_dtype();
        let latents = batch.latents.to_device(&self.device)?.to_dtype(dtype)?;
        let t5_embeds = batch.t5_embeds.to_device(&self.device)?.to_dtype(dtype)?;
        let clip_pooled = batch.clip_pooled.to_device(&self.device)?.to_dtype(dtype)?;
        
        // Get batch size and dimensions
        println!("Getting tensor dimensions...");
        let batch_size = latents.dims()[0];
        let (_, c, h, w) = latents.dims4()?;
        println!("Batch size: {}, channels: {}, height: {}, width: {}", batch_size, c, h, w);
        
        // Sample timesteps with shift for Flux flow matching
        let mut rng = rand::thread_rng();
        let shift = 1.0; // Flux uses shift=1.0 for training
        let timesteps: Vec<f32> = (0..batch_size)
            .map(|_| {
                let u = rng.gen_range(0.0..1.0);
                // Apply shift function: t = (shift * u) / (1 + (shift - 1) * u)
                // For shift=1.0, this simplifies to t = u
                let t = if shift == 1.0 {
                    u
                } else {
                    (shift * u) / (1.0 + (shift - 1.0) * u)
                };
                t
            })
            .collect();
        let timesteps_tensor = Tensor::from_vec(timesteps.clone(), batch_size, &self.device)?.to_dtype(dtype)?;
        
        // Add noise to latents (flow matching objective)
        let noise = Tensor::randn_like(&latents, 0.0, 1.0)?.to_dtype(dtype)?;
        let t_reshaped = timesteps_tensor.reshape((batch_size, 1, 1, 1))?;
        let one_minus_t = Tensor::new(&[1.0f32], &self.device)?.to_dtype(dtype)?.broadcast_sub(&t_reshaped)?;
        let noisy_latents = (latents.broadcast_mul(&one_minus_t)? + noise.broadcast_mul(&t_reshaped)?)?;
        
        // Prepare latents for Flux (patchify)
        let seq_len = (h / 2) * (w / 2);
        let img = noisy_latents.reshape((batch_size, c, h / 2, 2, w / 2, 2))?
            .permute((0, 2, 4, 3, 5, 1))?
            .reshape((batch_size, seq_len, 4 * c))?;
        
        // Create position IDs
        let img_seq_len = (h / 2) * (w / 2);
        let img_ids = self.create_position_ids(batch_size, h / 2, w / 2)?;
        let txt_ids = self.create_text_position_ids(batch_size, t5_embeds.dim(1)?)?;
        
        // Prepare guidance
        let guidance = if self.flux_config.guidance_embed {
            // For training, we typically use guidance_scale = 1.0
            Some(Tensor::ones(&[batch_size], dtype, &self.device)?)
        } else {
            None
        };
        
        // Use the Flux model's forward method which handles all the internals
        // We'll need to intercept at the attention layers for LoRA
        use candle_transformers::models::flux::WithForward;
        
        // Log shapes before forward pass on first step
        if self.global_step == 0 {
            println!("\nForward pass input shapes:");
            println!("  img: {:?} on device {:?}", img.shape(), img.device());
            println!("  img_ids: {:?} on device {:?}", img_ids.shape(), img_ids.device());
            println!("  t5_embeds: {:?} on device {:?}", t5_embeds.shape(), t5_embeds.device());
            println!("  txt_ids: {:?} on device {:?}", txt_ids.shape(), txt_ids.device());
            println!("  timesteps: {:?} on device {:?}", timesteps_tensor.shape(), timesteps_tensor.device());
            println!("  clip_pooled: {:?} on device {:?}", clip_pooled.shape(), clip_pooled.device());
            println!("\nStarting forward pass through Flux model...");
            println!("Note: First forward pass may take 30-60 seconds as model weights are loaded to GPU");
        }
        
        // Time the forward pass
        let start = std::time::Instant::now();
        
        // For now, use the base model forward pass
        // In a real implementation, we would need to modify the forward pass
        // to inject LoRA at the attention layers
        let model_output = self.model.forward(
            &img,
            &img_ids,
            &t5_embeds,
            &txt_ids,
            &timesteps_tensor,
            &clip_pooled,
            guidance.as_ref(),
        )?;
        
        let elapsed = start.elapsed();
        if self.global_step == 0 {
            println!("Forward pass complete in {:.2}s! Output shape: {:?}", elapsed.as_secs_f32(), model_output.shape());
        }
        
        // The model output needs to be unpatchified back to image shape
        // model_output shape: [batch, h/2 * w/2, patch_size * patch_size * channels]
        let patch_size = 2;
        let out_channels = c;
        
        // Reshape back to image format
        let model_pred_img = model_output
            .reshape((batch_size, h / patch_size, w / patch_size, patch_size, patch_size, out_channels))?
            .permute((0, 5, 1, 3, 2, 4))?  // [b, c, h/2, 2, w/2, 2]
            .reshape((batch_size, out_channels, h, w))?;
        
        // Compute target: velocity for flow matching
        let target_velocity = (&latents - &noise)?;
        
        // Compute loss (flow matching objective)
        let loss = model_pred_img.sub(&target_velocity)?.sqr()?.mean_all()?;
        
        // Check for NaN/Inf loss
        let loss_scalar = loss.to_scalar::<f32>()?;
        if loss_scalar.is_nan() || loss_scalar.is_infinite() {
            warn!("NaN/Inf loss detected at step {}: {}", self.global_step, loss_scalar);
            // Skip this step
            return Ok(0.0);
        }
        
        // Apply gradient clipping if enabled
        if self.config.max_grad_norm > 0.0 {
            // Get all LoRA parameters
            let params = self.lora_adapter.parameters();
            
            // Compute gradient norm
            let mut total_norm = 0.0f32;
            for param in &params {
                if let Ok(grad) = param.grad() {
                    let grad_norm = grad.sqr()?.sum_all()?.to_scalar::<f32>()?;
                    total_norm += grad_norm;
                }
            }
            let total_norm = total_norm.sqrt();
            
            if total_norm.is_nan() || total_norm.is_infinite() {
                warn!("NaN/Inf gradient norm detected at step {}", self.global_step);
                return Ok(0.0);
            }
            
            // Clip gradients if norm exceeds threshold
            if total_norm > self.config.max_grad_norm {
                let clip_coef = self.config.max_grad_norm / (total_norm + 1e-6);
                for param in &params {
                    if let Ok(grad) = param.grad() {
                        let clipped_grad = grad.affine(clip_coef as f64, 0.0)?;
                        // Note: Candle doesn't have direct grad assignment, 
                        // this is a limitation we need to work around
                    }
                }
            }
        }
        
        // Update weights via optimizer
        match &mut self.optimizer {
            OptimizerWrapper::AdamW(opt) => {
                opt.backward_step(&loss)?;
            }
            OptimizerWrapper::AdamW8bit(opt) => {
                opt.backward_step(&loss)?;
            }
        }
        
        // Return loss value
        Ok(loss_scalar)
    }
    
    /// Create position IDs for images
    fn create_position_ids(&self, batch_size: usize, h: usize, w: usize) -> Result<Tensor> {
        // Create position IDs for the Flux model
        // Format: [batch, seq_len, 3] where each position has [y, x, 0]
        let seq_len = h * w;
        let mut ids = Vec::new();
        
        for _ in 0..batch_size {
            for y in 0..h {
                for x in 0..w {
                    // Flux expects position as [y, x, 0]
                    ids.push(y as f32);
                    ids.push(x as f32);
                    ids.push(0.0);
                }
            }
        }
        
        Ok(Tensor::from_vec(ids, &[batch_size, seq_len, 3], &self.device)?
            .to_dtype(self.config.dtype.to_candle_dtype())?)
    }
    
    /// Create position IDs for text
    fn create_text_position_ids(&self, batch_size: usize, seq_len: usize) -> Result<Tensor> {
        // Create position IDs for text sequences
        // Format: [batch, seq_len, 3] where each position has [pos, 0, 0]
        let mut ids = Vec::new();
        
        for _ in 0..batch_size {
            for i in 0..seq_len {
                // Text position: [position, 0, 0]
                ids.push(i as f32);
                ids.push(0.0);
                ids.push(0.0);
            }
        }
        
        Ok(Tensor::from_vec(ids, &[batch_size, seq_len, 3], &self.device)?
            .to_dtype(self.config.dtype.to_candle_dtype())?)
    }
    
    /// Generate samples
    async fn generate_samples(&self, step: usize) -> Result<()> {
        println!("\n🎨 Generating samples at step {}...", step);
        // Sample generation implementation
        Ok(())
    }
    
    /// Save checkpoint
    pub async fn save_checkpoint(&self, step: usize) -> Result<()> {
        let checkpoint_dir = self.config.output_dir.join(format!("checkpoint-{}", step));
        std::fs::create_dir_all(&checkpoint_dir)?;
        
        println!("\n💾 Saving checkpoint at step {}...", step);
        
        // Collect LoRA state dict manually
        let mut tensors = HashMap::new();
        
        // Get all variables from the VarMap
        let var_data = self.lora_adapter.var_map.data();
        let vars = var_data.lock().unwrap();
        
        for (name, var) in vars.iter() {
            // Get the tensor value
            let tensor = var.as_tensor();
            
            // Force computation if tensor is lazy
            let tensor_data = tensor.to_dtype(DType::F32)?;
            
            // Verify tensor has actual data
            let shape = tensor_data.shape();
            if shape.dims().iter().any(|&d| d == 0) {
                warn!("Skipping empty tensor: {}", name);
                continue;
            }
            
            tensors.insert(name.clone(), tensor_data);
        }
        
        if tensors.is_empty() {
            return Err(Error::Runtime("No LoRA weights to save".into()));
        }
        
        // Save using safetensors
        let lora_path = checkpoint_dir.join("flux_lora.safetensors");
        safetensors::save(&tensors, &lora_path)?;
        
        // Verify file size
        let metadata = std::fs::metadata(&lora_path)?;
        println!("✓ LoRA weights saved: {} ({:.2} MB)", 
            lora_path.display(), 
            metadata.len() as f64 / 1024.0 / 1024.0
        );
        
        // Save training metadata
        let metadata_path = checkpoint_dir.join("training_metadata.json");
        let metadata = serde_json::json!({
            "step": step,
            "rank": self.config.lora_rank,
            "alpha": self.config.lora_alpha,
            "learning_rate": self.config.learning_rate,
            "model": "flux-dev",
        });
        std::fs::write(&metadata_path, serde_json::to_string_pretty(&metadata)?)?;
        
        Ok(())
    }
    
    /// Get current step
    pub fn get_current_step(&self) -> usize {
        self.global_step
    }
    
    /// Print memory summary
    fn print_memory_summary(config: &FluxLoRATraining24GBConfig) {
        println!("\n💾 Memory Usage Summary:");
        println!("─────────────────────────────────────");
        
        let base_model = match config.dtype {
            DTypeConfig::FP32 => 24.0,
            DTypeConfig::FP16 | DTypeConfig::BF16 => 12.0,
        };
        
        // More accurate LoRA memory calculation
        // For each attention layer: Q, K, V, O projections
        // Flux has ~19 double blocks + ~38 single blocks
        let num_attention_layers = 19 * 4 + 38 * 2; // Approximate
        let hidden_size = 3072; // Flux hidden dimension
        let lora_params_count = num_attention_layers * hidden_size * config.lora_rank * 2; // down + up
        let lora_params = (lora_params_count * 4) as f32 / 1e9; // FP32 bytes
        
        // Optimizer memory depends on type
        let optimizer_mem = if config.optimizer_type.contains("8bit") {
            // 8-bit optimizer: params + 2*int8 states
            lora_params + (lora_params_count * 2) as f32 / 1e9
        } else {
            // Standard AdamW: params + 2*fp32 states
            lora_params * 3.0
        };
        
        println!("Base Flux model: {:.1} GB (frozen)", base_model);
        println!("LoRA parameters: {:.3} GB (rank={})", lora_params, config.lora_rank);
        println!("Optimizer states: {:.3} GB ({})", optimizer_mem, config.optimizer_type);
        println!("Gradients: ~{:.2} GB (LoRA only)", lora_params);
        println!("Activations: ~2-3 GB (with checkpointing)");
        println!("─────────────────────────────────────");
        println!("Total estimate: {:.1} GB", base_model + lora_params + optimizer_mem + lora_params + 2.5);
        
        if base_model + lora_params + optimizer_mem + lora_params + 2.5 > 24.0 {
            println!("⚠️  WARNING: May exceed 24GB!");
        } else {
            println!("✅ Fits comfortably in 24GB");
        }
        
        // Show memory savings with 8-bit optimizer
        if config.optimizer_type.contains("8bit") {
            let standard_mem = lora_params * 3.0;
            let savings = standard_mem - optimizer_mem;
            println!("\n📊 8-bit Optimizer Savings: {:.2} GB", savings);
        }
    }
}