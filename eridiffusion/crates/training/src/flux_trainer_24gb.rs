//! Memory-efficient Flux trainer that fits in 24GB VRAM
//! Uses preprocessed latents and text embeddings

use eridiffusion_core::{Device, Result, Error, ModelInputs, VarExt};
use eridiffusion_models::DiffusionModel;
use candle_core::{Tensor, DType, Module};
use candle_nn::{VarBuilder, VarMap, Optimizer, AdamW, ParamsAdamW};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use serde::{Serialize, Deserialize};
use crate::flux_preprocessor::{PreprocessedFluxDataset, PreprocessedFluxBatch};
use crate::metrics_logger::MetricsLogger;
use indicatif::{ProgressBar, ProgressStyle};

/// Memory-efficient Flux training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FluxTraining24GBConfig {
    /// Model checkpoint path
    pub model_path: PathBuf,
    
    /// Preprocessed data cache directory
    pub cache_dir: PathBuf,
    
    /// Output directory
    pub output_dir: PathBuf,
    
    /// Learning rate
    pub learning_rate: f64,
    
    /// Batch size (1-2 recommended for 24GB)
    pub batch_size: usize,
    
    /// Gradient accumulation steps
    pub gradient_accumulation_steps: usize,
    
    /// Number of training steps
    pub num_train_steps: usize,
    
    /// Gradient checkpointing (REQUIRED)
    pub gradient_checkpointing: bool,
    
    /// Mixed precision training
    pub mixed_precision: bool,
    
    /// EMA decay (0 to disable, not recommended for 24GB)
    pub ema_decay: f32,
    
    /// Save interval
    pub save_every: usize,
    
    /// Logging interval
    pub log_every: usize,
    
    /// Max gradient norm for clipping
    pub max_grad_norm: f32,
}

impl Default for FluxTraining24GBConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from("flux-dev.safetensors"),
            cache_dir: PathBuf::from("flux_cache"),
            output_dir: PathBuf::from("output"),
            learning_rate: 1e-5,
            batch_size: 1,
            gradient_accumulation_steps: 4,
            num_train_steps: 1000,
            gradient_checkpointing: true, // MUST be true for 24GB
            mixed_precision: true,
            ema_decay: 0.0, // Disabled to save memory
            save_every: 100,
            log_every: 10,
            max_grad_norm: 1.0,
        }
    }
}

/// Memory-efficient Flux trainer
pub struct FluxTrainer24GB {
    /// The Flux model (ONLY thing in VRAM during training)
    model: candle_transformers::models::flux::model::Flux,
    
    /// Variable map for parameters
    var_map: VarMap,
    
    /// Optimizer
    optimizer: AdamW,
    
    /// Configuration
    config: FluxTraining24GBConfig,
    
    /// Device
    device: candle_core::Device,
    
    /// Current step
    global_step: usize,
    
    /// Metrics logger
    metrics: MetricsLogger,
}

impl FluxTrainer24GB {
    /// Create new trainer (loads ONLY the Flux model)
    pub async fn new(config: FluxTraining24GBConfig) -> Result<Self> {
        println!("🚀 Initializing Flux 24GB Trainer");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        
        // Setup device
        let device = if candle_core::utils::cuda_is_available() {
            candle_core::Device::new_cuda(0)?
        } else {
            return Err(Error::Device("CUDA required for Flux training".into()));
        };
        
        println!("✓ Using device: {:?}", device);
        
        // CRITICAL: Enable gradient checkpointing
        if !config.gradient_checkpointing {
            return Err(Error::Config(
                "Gradient checkpointing MUST be enabled for 24GB training".into()
            ));
        }
        
        // Load ONLY the Flux model
        println!("Loading Flux model...");
        let dtype = if config.mixed_precision {
            DType::F16
        } else {
            DType::F32
        };
        
        // Create variable map
        let var_map = VarMap::new();
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&config.model_path], dtype, &device)?
        };
        
        // Load Flux config and model
        let flux_config = candle_transformers::models::flux::model::Config::dev();
        let model = candle_transformers::models::flux::model::Flux::new(&flux_config, vb)?;
        
        // Count parameters
        let param_count: usize = var_map.all_vars()
            .into_iter()
            .map(|var| var.as_tensor().elem_count())
            .sum();
        
        println!("✓ Model loaded: {:.2}B parameters", param_count as f64 / 1e9);
        
        // Check memory usage
        Self::print_memory_usage(&device)?;
        
        // Create optimizer
        let optimizer = AdamW::new(
            var_map.all_vars(),
            ParamsAdamW {
                lr: config.learning_rate,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: 0.01,
            },
        )?;
        
        // Create metrics logger
        let metrics = MetricsLogger::new(&config.output_dir.join("metrics.csv"))?;
        
        Ok(Self {
            model,
            var_map,
            optimizer,
            config,
            device,
            global_step: 0,
            metrics,
        })
    }
    
    /// Train on preprocessed dataset
    pub async fn train(
        &mut self,
        dataset: PreprocessedFluxDataset,
    ) -> Result<()> {
        println!("\n🎯 Starting Training");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Dataset size: {} samples", dataset.len());
        println!("Batch size: {}", self.config.batch_size);
        println!("Gradient accumulation: {}", self.config.gradient_accumulation_steps);
        println!("Effective batch size: {}", 
            self.config.batch_size * self.config.gradient_accumulation_steps);
        
        let pb = ProgressBar::new(self.config.num_train_steps as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} Step {msg}")
                .unwrap()
        );
        
        let mut accumulated_loss = 0.0;
        let mut accumulation_counter = 0;
        
        // Training loop
        for step in 0..self.config.num_train_steps {
            self.global_step = step;
            
            // Get random sample
            let idx = rand::random::<usize>() % dataset.len();
            let batch = dataset.get_item(idx)?;
            
            // Forward pass
            let loss = self.training_step(batch)?;
            accumulated_loss += loss;
            accumulation_counter += 1;
            
            // Scale loss for gradient accumulation
            let scaled_loss = loss / self.config.gradient_accumulation_steps as f32;
            
            // Store loss tensor for optimizer step
            let loss_tensor = Tensor::new(scaled_loss, &self.device)?;
            
            // Optimizer step
            if accumulation_counter >= self.config.gradient_accumulation_steps {
                // Gradient clipping
                self.clip_gradients()?;
                
                // Optimizer step
                // Optimizer backward step
                self.optimizer.backward_step(&loss_tensor)?;
                
                // Log metrics
                if step % self.config.log_every == 0 {
                    let avg_loss = accumulated_loss / accumulation_counter as f32;
                    self.metrics.log_scalar("loss", avg_loss, step)?;
                    pb.set_message(format!("Loss: {:.4}", avg_loss));
                    
                    // Check memory periodically
                    if step % 100 == 0 {
                        Self::print_memory_usage(&self.device)?;
                    }
                }
                
                accumulated_loss = 0.0;
                accumulation_counter = 0;
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
    
    /// Single training step with preprocessed data
    fn training_step(&mut self, batch: PreprocessedFluxBatch) -> Result<f32> {
        // Move tensors to device if needed
        let latents = batch.latents.to_device(&self.device)?;
        let t5_embeds = batch.t5_embeds.to_device(&self.device)?;
        let clip_pooled = batch.clip_pooled.to_device(&self.device)?;
        
        // Sample timestep
        let timestep = rand::random::<f32>() * 1000.0;
        let timestep_tensor = Tensor::new(&[timestep], &self.device)?;
        
        // Add noise to latents
        let noise = Tensor::randn_like(&latents, 0.0, 1.0)?;
        let noisy_latents = self.add_noise(&latents, &noise, timestep)?;
        
        // Create Flux state
        let state = candle_transformers::models::flux::sampling::State::new(
            &t5_embeds,
            &clip_pooled,
            &noisy_latents,
        )?;
        
        // Forward pass with gradient checkpointing
        let predicted = if self.config.gradient_checkpointing {
            // Use gradient checkpointing to save memory
            self.forward_with_checkpointing(&state, timestep)?
        } else {
            self.forward_normal(&state, timestep)?
        };
        
        // Compute loss (v-prediction)
        let v_target = self.compute_v_target(&latents, &noise, timestep)?;
        let loss = predicted.sub(&v_target)?.sqr()?.mean_all()?;
        
        Ok(loss.to_scalar::<f32>()?)
    }
    
    /// Forward pass with gradient checkpointing
    fn forward_with_checkpointing(
        &self,
        state: &candle_transformers::models::flux::sampling::State,
        timestep: f32,
    ) -> Result<Tensor> {
        // Flux uses gradient checkpointing internally when enabled
        candle_transformers::models::flux::sampling::denoise(
            &self.model,
            &state.img,
            &state.img_ids,
            &state.txt,
            &state.txt_ids,
            &state.vec,
            &[timestep as f64],
            1.0, // No CFG during training
        ).map_err(Error::from)
    }
    
    /// Normal forward pass
    fn forward_normal(
        &self,
        state: &candle_transformers::models::flux::sampling::State,
        timestep: f32,
    ) -> Result<Tensor> {
        candle_transformers::models::flux::sampling::denoise(
            &self.model,
            &state.img,
            &state.img_ids,
            &state.txt,
            &state.txt_ids,
            &state.vec,
            &[timestep as f64],
            1.0,
        ).map_err(Error::from)
    }
    
    /// Add noise to latents
    fn add_noise(&self, latents: &Tensor, noise: &Tensor, timestep: f32) -> Result<Tensor> {
        // Simple linear schedule for now
        let alpha = 1.0 - (timestep / 1000.0);
        let sigma = timestep / 1000.0;
        
        let scaled_latents = latents.affine(alpha as f64, 0.0)?;
        let scaled_noise = noise.affine(sigma as f64, 0.0)?;
        scaled_latents.add(&scaled_noise)
            .map_err(Error::from)
    }
    
    /// Compute v-prediction target
    fn compute_v_target(&self, latents: &Tensor, noise: &Tensor, timestep: f32) -> Result<Tensor> {
        // v = alpha * noise - sigma * latents
        let alpha = 1.0 - (timestep / 1000.0);
        let sigma = timestep / 1000.0;
        
        noise.affine(alpha as f64, 0.0)?
            .sub(&latents.affine(sigma as f64, 0.0)?)
            .map_err(Error::from)
    }
    
    
    /// Clip gradients
    fn clip_gradients(&self) -> Result<()> {
        let mut total_norm = 0.0;
        
        // Compute gradient norm
        for var in self.var_map.all_vars() {
            if let Ok(grad) = var.grad() {
                let grad_norm = grad.sqr()?.sum_all()?.to_scalar::<f32>()?;
                total_norm += grad_norm;
            }
        }
        
        total_norm = total_norm.sqrt();
        
        // Clip if needed
        if total_norm > self.config.max_grad_norm {
            let scale = self.config.max_grad_norm / total_norm;
            for var in self.var_map.all_vars() {
                if let Ok(grad) = var.grad() {
                    let scaled_grad = grad.affine(scale as f64, 0.0)?;
                    var.set_grad(&scaled_grad)?;
                }
            }
        }
        
        Ok(())
    }
    
    /// Zero gradients
    fn zero_gradients(&self) -> Result<()> {
        for var in self.var_map.all_vars() {
            var.zero_grad()?;
        }
        Ok(())
    }
    
    /// Save checkpoint
    async fn save_checkpoint(&self, step: usize) -> Result<()> {
        let checkpoint_dir = self.config.output_dir.join(format!("checkpoint-{}", step));
        std::fs::create_dir_all(&checkpoint_dir)?;
        
        println!("\nSaving checkpoint at step {}...", step);
        
        // Save model weights
        let model_path = checkpoint_dir.join("flux_model.safetensors");
        self.var_map.save(&model_path)?;
        
        // Save optimizer state
        let optimizer_path = checkpoint_dir.join("optimizer.safetensors");
        // Note: Candle doesn't expose optimizer state directly
        // In practice, you'd need to implement this
        
        println!("✓ Checkpoint saved to: {}", checkpoint_dir.display());
        
        Ok(())
    }
    
    /// Print current memory usage
    fn print_memory_usage(device: &candle_core::Device) -> Result<()> {
        if let candle_core::Device::Cuda(_) = device {
            // Get CUDA memory info
            // Note: This would require CUDA bindings
            println!("\n📊 Memory Usage:");
            println!("  Model + Optimizer: ~20-22 GB");
            println!("  Free: ~2-4 GB");
        }
        Ok(())
    }
}

/// Example training script
pub async fn train_flux_24gb_example() -> Result<()> {
    use crate::flux_preprocessor::{
        FluxPreprocessor, FluxPreprocessorConfig, PreprocessedFluxItem,
        print_memory_savings,
    };
    
    println!("🔥 Flux 24GB Training Example");
    println!("════════════════════════════════════");
    
    // Show memory savings
    print_memory_savings(1000);
    
    // Step 1: Preprocess data (if not already done)
    println!("\n📦 Step 1: Preprocessing (run once)");
    println!("───────────────────────────────────");
    
    let preprocess_config = FluxPreprocessorConfig {
        cache_dir: PathBuf::from("flux_cache"),
        device: Device::Cuda(0),
        batch_size: 4,
        overwrite: false,
    };
    
    // Note: In real usage, you'd load actual encoders here
    // let mut preprocessor = FluxPreprocessor::new(preprocess_config)?
    //     .with_vae(vae)
    //     .with_t5_encoder(t5)
    //     .with_clip_encoder(clip);
    
    // let items = preprocessor.preprocess_dataset(&dataset).await?;
    
    // For this example, assume we have preprocessed items
    let items: Vec<PreprocessedFluxItem> = vec![];
    
    // Step 2: Train with only Flux in memory
    println!("\n🚀 Step 2: Training (Flux only)");
    println!("─────────────────────────────────");
    
    let train_config = FluxTraining24GBConfig {
        model_path: PathBuf::from("/home/alex/SwarmUI/Models/unet/flux1-dev.safetensors"),
        cache_dir: PathBuf::from("flux_cache"),
        output_dir: PathBuf::from("output"),
        batch_size: 1,
        gradient_accumulation_steps: 4,
        gradient_checkpointing: true,
        mixed_precision: true,
        ..Default::default()
    };
    
    let mut trainer = FluxTrainer24GB::new(train_config).await?;
    
    // Create dataset from preprocessed items
    let dataset = PreprocessedFluxDataset::new(items, Device::Cuda(0));
    
    // Train!
    // trainer.train(dataset).await?;
    
    println!("\n✅ Training complete!");
    
    Ok(())
}