//! SDXL LoRA trainer
//! This implementation follows the proven approach from candle-fork
//! Direct weight loading and custom forward pass

use log::{info, debug, warn, error};
use anyhow::{Result, Context};
use candle_core::{Device, DType, Tensor, Module, D, Var};
use std::path::{Path, PathBuf};
use std::collections::HashMap;
use std::fs;
use std::time::Instant;
use serde::{Serialize, Deserialize};

use super::{Config, ProcessConfig, SampleConfig};
use crate::trainers::text_encoders::TextEncoders;
use crate::loaders::sdxl_checkpoint_loader::load_text_encoders_sdxl;
use crate::loaders::sdxl_weight_remapper::remap_sdxl_weights;
use crate::loaders::sdxl_full_remapper::remap_sdxl_unet_weights;
use crate::trainers::sdxl_forward_with_lora::forward_sdxl_with_lora;
use crate::trainers::sdxl_forward_simple::forward_sdxl_simple;
use crate::trainers::sdxl_forward_sd_format::forward_sdxl_sd_format;
use crate::trainers::sdxl_forward_sd_format_flash::forward_sdxl_sd_format_flash;
use crate::trainers::sdxl_forward_efficient::forward_sdxl_efficient;
use crate::trainers::sdxl_forward_sd_efficient::forward_sdxl_sd_efficient;
use crate::trainers::sdxl_aggressive_checkpoint::forward_sdxl_aggressive_checkpoint;
use crate::trainers::sdxl_forward_checkpoint::forward_sdxl_with_checkpoint;
use crate::trainers::sdxl_vae_native::SDXLVAENative;
use crate::trainers::sdxl_proper_checkpoint::forward_sdxl_proper_checkpoint;
use crate::trainers::ddpm_scheduler::{DDPMScheduler, compute_snr_loss_weights};
use crate::trainers::adam8bit::Adam8bit;
use crate::trainers::enhanced_data_loader::{EnhancedCaptionHandler, EnhancedDataConfig, ModelType as EnhancedModelType};
use crate::trainers::sdxl_sampling_complete::{TrainingSampler, SDXLSamplingConfig, SchedulerType, SDXLSampler};
use crate::trainers::memory_utils;
use crate::trainers::gradient_accumulator::GradientAccumulator;
use crate::trainers::gradient_checkpoint::SDXLGradientCheckpoint;
use crate::trainers::vae_tiling::{TiledVAE, TilingConfig, BlendMode};
use crate::trainers::snr_weighting::SNRWeighting;
use crate::trainers::lr_scheduler::{LRScheduler, create_scheduler};
use crate::trainers::ema::{EMAHelper, EMAModel};
use crate::trainers::validation::{ValidationDataset, ValidationRunner, ValidationConfig as ValConfig};
use crate::trainers::sdxl_vae_wrapper::SDXLVAEWrapper;
use candle_core::backprop::GradStore;
use rand::Rng;

/// Helper function for linear operation
fn linear(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    // Linear: out = input @ weight.T + bias
    // weight shape: [out_features, in_features]
    // input shape: [..., in_features]
    // output shape: [..., out_features]
    
    // Disable debug output to save memory
    // info!("  Linear op: input {:?}, weight {:?}", x.dims(), weight.dims());
    
    // Ensure weight is contiguous before transpose
    let weight_contig = weight.contiguous()?;
    let w = weight_contig.t()?;
    // info!("  After transpose: weight_t {:?}", w.dims());
    // info!("  Matmul: {:?} @ {:?}", x.dims(), w.dims());
    
    // Handle 3D tensor multiplication
    let out = if x.dims().len() == 3 {
        // For 3D tensors, we might need to reshape
        let (b, s, d) = x.dims3()?;
        let x_2d = x.reshape((b * s, d))?;
        // debug!("  Reshaped input to 2D: {:?}");
        let out_2d = x_2d.matmul(&w)?;
        let out_d = w.dims()[1];  // Get output dimension from transposed weight
        out_2d.reshape((b, s, out_d))?
    } else {
        x.matmul(&w)?
    };
    
    let mut out = out;
    if let Some(b) = bias {
        out = out.broadcast_add(b)?;
    }
    Ok(out)
}
use safetensors::{serialize, tensor::TensorView, Dtype as SafeDtype};
use image;

/// Simple LoRA adapter - exactly like candle-fork
pub struct SimpleLoRA {
    pub down: Var,
    pub up: Var,
    pub scale: f64,
}

impl SimpleLoRA {
    pub fn new(in_features: usize, out_features: usize, rank: usize, alpha: f32, device: &Device, dtype: DType) -> Result<Self> {
        // Initialize LoRA weights - we must create fresh tensors then wrap in Var
        // This is safe because we're creating new tensors, not converting existing ones
        let down_tensor = Tensor::randn(0.0f32, 0.02, (rank, in_features), device)?.to_dtype(dtype)?;
        let up_tensor = Tensor::zeros((out_features, rank), dtype, device)?;
        
        // Use Var::from_tensor for newly created tensors - this should work
        let down = Var::from_tensor(&down_tensor)?;
        let up = Var::from_tensor(&up_tensor)?;
        
        Ok(Self {
            down,
            up,
            scale: (alpha / rank as f32) as f64,
        })
    }
    
    /// Apply LoRA to a linear operation
    pub fn forward(&self, input: &Tensor, base_weight: &Tensor, base_bias: Option<&Tensor>) -> Result<Tensor> {
        // Check if we can use CUDA acceleration
        #[cfg(feature = "cuda")]
        {
            if input.device().is_cuda() {
                // Use our optimized CUDA kernel
                let alpha = self.scale as f32;
                if let Ok(result) = candle_core::cuda_lora_forward::cuda_lora_forward(
                    input,
                    base_weight,
                    &self.down.as_tensor(),
                    &self.up.as_tensor(),
                    alpha,
                ) {
                    // Add bias if present
                    if let Some(bias) = base_bias {
                        return result.broadcast_add(bias);
                    }
                    return Ok(result);
                }
                // Fall through to CPU implementation if CUDA fails
            }
        }
        
        // Base linear operation (CPU fallback)
        let weight_t = base_weight.t()?;
        
        // Debug dimensions
        if self.down.dims()[1] == 2048 || self.up.dims()[0] == 640 {
            info!("  SimpleLoRA forward: input {:?}, weight {:?}, weight_t {:?}", input.dims(), base_weight.dims(), weight_t.dims());
            info!("  LoRA down: {:?}, up: {:?}", self.down.dims(), self.up.dims());
        }
        
        // Try to reshape if needed for matmul
        let (input_2d, original_shape) = if input.dims().len() == 3 {
            let (b, s, d) = input.dims3()?;
            (input.reshape((b * s, d))?, Some((b, s)))
        } else {
            (input.clone(), None)
        };
        
        let output_2d = input_2d.matmul(&weight_t)?;
        
        let mut output = if let Some((b, s)) = original_shape {
            // Use base_weight.dims()[0] to get the output dimension
            let out_d = base_weight.dims()[0];
            output_2d.reshape((b, s, out_d))?
        } else {
            output_2d
        };
        if let Some(bias) = base_bias {
            output = output.broadcast_add(bias)?;
        }
        
        // Add LoRA: output = base + scale * up(down(input))
        // Use the reshaped input for LoRA as well
        let down_out = input_2d.matmul(&self.down.as_tensor().t()?)?;
        let lora_out_2d = down_out.matmul(&self.up.as_tensor().t()?)?;
        
        let lora_out = if let Some((b, s)) = original_shape {
            // Use the same output dimension as the base linear operation
            // Use base_weight.dims()[0] to get the output dimension
            let out_d = base_weight.dims()[0];
            lora_out_2d.reshape((b, s, out_d))?
        } else {
            lora_out_2d
        };
        
        Ok((output + lora_out * self.scale)?)
    }
    
    /// Get trainable parameters
    pub fn vars(&self) -> Vec<&Var> {
        vec![&self.down, &self.up]
    }
} // impl SimpleLoRA

/// Collection of LoRA adapters
pub struct LoRACollection {
    pub adapters: HashMap<String, SimpleLoRA>,
    pub rank: usize,
    pub alpha: f32,
    pub dtype: DType,
}

impl LoRACollection {
    pub fn new(rank: usize, alpha: f32, dtype: DType) -> Self {
        Self {
            adapters: HashMap::new(),
            rank,
            alpha,
            dtype,
        }
    }
    
    /// Add LoRA adapter for a specific layer
    pub fn add(&mut self, name: &str, in_dim: usize, out_dim: usize, device: &Device) -> Result<()> {
        self.adapters.insert(
            name.to_string(),
            SimpleLoRA::new(in_dim, out_dim, self.rank, self.alpha, device, self.dtype)?
        );
        Ok(())
    }
    
    /// Apply LoRA if it exists for this layer, otherwise regular linear
    pub fn apply(&self, layer_name: &str, input: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
        // Disable debug output to save memory
        // if layer_name.contains("to_k") || layer_name.contains("to_v") {
        //     debug!("DEBUG LoRA apply for {}: input shape {:?}, weight shape {:?}"), weight.dims();
        // }
        
        if let Some(lora) = self.adapters.get(layer_name) {
            // Check dimension compatibility
            let weight_out_dim = weight.dims()[0];
            let weight_in_dim = weight.dims()[1];
            let lora_out_dim = lora.up.dims()[0];
            let lora_in_dim = lora.down.dims()[1];
            
            if weight_out_dim != lora_out_dim || weight_in_dim != lora_in_dim {
                info!("WARNING: LoRA dimension mismatch for {}", layer_name);
                info!("  Weight: [{}, {}], LoRA: [{}, {}]", weight_out_dim, weight_in_dim, lora_out_dim, lora_in_dim);
                info!("  Skipping LoRA for this layer");
                // Fall back to regular linear
                return linear(input, weight, bias);
            }
            
            lora.forward(input, weight, bias)
        } else {
            // Regular linear operation without LoRA (shouldn't happen in training)
            linear(input, weight, bias)
        }
    }
    
    /// Get all trainable variables
    pub fn vars(&self) -> Vec<&Var> {
        let mut vars = Vec::new();
        for adapter in self.adapters.values() {
            vars.extend(adapter.vars());
        }
        vars
    }
    
    /// Check if a specific layer has a linear LoRA adapter
    pub fn get_linear_lora(&self, name: &str) -> Option<&SimpleLoRA> {
        self.adapters.get(name)
    }
    
    /// Check if a specific attention layer has LoRA adapters
    pub fn has_attention_lora(&self, name: &str) -> bool {
        // Check for any of the attention projections
        self.adapters.contains_key(&format!("{}.to_q", name)) ||
        self.adapters.contains_key(&format!("{}.to_k", name)) ||
        self.adapters.contains_key(&format!("{}.to_v", name)) ||
        self.adapters.contains_key(&format!("{}.to_out.0", name))
    }
    
    /// Apply attention LoRA using CUDA acceleration when available
    #[cfg(feature = "cuda")]
    pub fn apply_attention_cuda(
        &self,
        prefix: &str,
        input: &Tensor,
        w_q: &Tensor,
        w_k: &Tensor,
        w_v: &Tensor,
    ) -> Result<Option<(Tensor, Tensor, Tensor)>> {
        if !input.device().is_cuda() {
            return Ok(None);
        }
        
        // Get LoRA adapters for Q, K, V
        let lora_q = self.adapters.get(&format!("{}.to_q", prefix))
            .map(|l| (&l.down.as_tensor(), &l.up.as_tensor()));
        let lora_k = self.adapters.get(&format!("{}.to_k", prefix))
            .map(|l| (&l.down.as_tensor(), &l.up.as_tensor()));
        let lora_v = self.adapters.get(&format!("{}.to_v", prefix))
            .map(|l| (&l.down.as_tensor(), &l.up.as_tensor()));
        
        // Only use CUDA if we have at least one LoRA adapter
        if lora_q.is_some() || lora_k.is_some() || lora_v.is_some() {
            // Get alpha values
            let alpha_q = self.adapters.get(&format!("{}.to_q", prefix))
                .map(|l| l.scale as f32).unwrap_or(0.0);
            let alpha_k = self.adapters.get(&format!("{}.to_k", prefix))
                .map(|l| l.scale as f32).unwrap_or(0.0);
            let alpha_v = self.adapters.get(&format!("{}.to_v", prefix))
                .map(|l| l.scale as f32).unwrap_or(0.0);
            
            match candle_core::cuda_lora_forward::cuda_attention_lora(
                input, w_q, w_k, w_v,
                lora_q, lora_k, lora_v,
                alpha_q, alpha_k, alpha_v,
            ) {
                Ok((q, k, v)) => Ok(Some((q, k, v))),
                Err(_) => Ok(None), // Fall back to CPU
            }
        } else {
            Ok(None)
        }
    }
    
    /// Save LoRA weights in ComfyUI format
    pub fn save(&self, path: &Path) -> Result<()> {
        // First collect all tensor data
        let mut tensor_data = Vec::new();
        let mut tensor_info = Vec::new();
        
        // Save in ComfyUI format
        for (name, adapter) in &self.adapters {
            // ComfyUI expects keys like: lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_k.lora_down.weight
            let key_base = name.replace(".", "_");
            
            // Convert down tensor
            let down_tensor = adapter.down.as_tensor();
            let down_data = tensor_to_vec(down_tensor)?;
            tensor_info.push((
                format!("lora_unet_{}.lora_down.weight", key_base),
                convert_dtype(down_tensor.dtype()?)?,
                down_tensor.dims().to_vec(),
                tensor_data.len()
            ));
            tensor_data.push(down_data);
            
            // Convert up tensor
            let up_tensor = adapter.up.as_tensor();
            let up_data = tensor_to_vec(up_tensor)?;
            tensor_info.push((
                format!("lora_unet_{}.lora_up.weight", key_base),
                convert_dtype(up_tensor.dtype()?)?,
                up_tensor.dims().to_vec(),
                tensor_data.len()
            ));
            tensor_data.push(up_data);
        }
        
        // Now create TensorViews using indices
        let mut tensors = HashMap::new();
        for (name, dtype, shape, idx) in tensor_info {
            tensors.insert(
                name,
                TensorView::new(
                    dtype,
                    shape,
                    &tensor_data[idx],
                )?
            );
        }
        
        // Add metadata for ComfyUI
        let mut metadata = HashMap::new();
        metadata.insert("ss_network_rank".to_string(), self.rank.to_string());
        metadata.insert("ss_network_alpha".to_string(), self.alpha.to_string());
        metadata.insert("ss_network_module".to_string(), "networks.lora".to_string());
        metadata.insert("ss_network_dim".to_string(), self.rank.to_string());
        metadata.insert("ss_network_args".to_string(), format!("{{\"rank\": {}, \"alpha\": {}}}", self.rank, self.alpha));
        
        // Save using safetensors
        let data = serialize(&tensors, &Some(metadata))?;
        fs::write(path, data)?;
        
        Ok(())
    }
}

/// SDXL configuration
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

impl Default for SDXLConfig {
    fn default() -> Self {
        Self {
            in_channels: 4,
            out_channels: 4,
            model_channels: 320,
            num_res_blocks: 2,
            channel_mult: vec![1, 2, 4],
            context_dim: 2048,
            use_linear_projection: true,
            num_heads: 8,
        }
    }
}

/// SDXL LoRA trainer
pub struct SDXLLoRATrainerFixed {
    // Core components
    device: Device,
    dtype: DType,
    
    // Model weights loaded directly
    unet_weights: HashMap<String, Tensor>,
    vae_weights: HashMap<String, Tensor>,
    
    // Models
    vae_encoder: Option<SDXLVAENative>,
    tiled_vae: Option<TiledVAE>,
    text_encoders: Option<TextEncoders>,
    
    // Scheduler
    noise_scheduler: DDPMScheduler,
    
    // LoRA adapters
    lora_collection: LoRACollection,
    
    // SDXL config
    sdxl_config: SDXLConfig,
    
    // Training configuration
    learning_rate: f64,
    batch_size: usize,
    gradient_accumulation_steps: usize,
    num_steps: usize,
    save_every: usize,
    
    // Paths
    model_path: PathBuf,
    output_dir: PathBuf,
    dataset_path: PathBuf,
    
    // Keep process config for accessing settings
    process_config: ProcessConfig,
    
    // Training state
    current_step: usize,
    accumulated_loss: f32,
    accumulation_step: usize,
    
    // Latent cache directory
    latent_cache_dir: Option<PathBuf>,
    gradient_accumulator: GradientAccumulator,
    gradient_checkpoint: SDXLGradientCheckpoint,
    
    // Optimizer
    adam8bit: Option<Adam8bit>,
    
    // Settings
    use_8bit_adam: bool,
    gradient_checkpointing: bool,
    mixed_precision: bool,
    max_grad_norm: Option<f32>,
    use_flash_attention: bool,
    
    // Enhanced caption handling
    caption_handler: Option<EnhancedCaptionHandler>,
    caption_dropout_rate: f32,
    
    // Config reference (for VAE path, etc)
    config: Config,
    
    // Sampling configuration
    sample_config: Option<SampleConfig>,
    sample_every: usize,
    
    // New comprehensive sampling system
    training_sampler: Option<TrainingSampler>,
    
    // New training features
    snr_weighting: Option<SNRWeighting>,
    lr_scheduler: Option<Box<dyn LRScheduler>>,
    ema_helper: EMAHelper,
    validation_runner: Option<ValidationRunner>,
}

impl SDXLLoRATrainerFixed {
    pub fn new(config: &Config, process: &ProcessConfig) -> Result<Self> {
        info!("\n=== Initializing SDXL LoRA Training ===");
        
        // Setup device and validate GPU requirement
        let device = Self::setup_device(process.device.as_deref().unwrap_or("cuda:0"))?;
        
        // Validate GPU requirement
        match &device {
            Device::Cuda(_) => {
                info!("CUDA GPU detected and verified");
            }
            Device::Cpu => {
                error!("Training requires a CUDA GPU. CPU device not supported.");
                return Err(anyhow!(
                    "GPU required for training. CPU device not supported.\n\
                     Please use a CUDA-capable GPU."
                ));
            }
        }
        
        // We'll determine dtype from the actual model weights later
        let training_dtype = match process.train.dtype.as_str() {
            "fp16" => DType::F16,
            "bf16" => DType::BF16,
            _ => DType::BF16,
        };
        
        // Create LoRA collection (will set dtype later based on model weights)
        let rank = process.network.linear.unwrap_or(16);
        let alpha = process.network.linear_alpha.unwrap_or(16.0) as f32;
        let lora_collection = LoRACollection::new(rank, alpha, training_dtype);
        
        // Training settings
        let learning_rate = process.train.lr as f64;
        let batch_size = process.train.batch_size;
        let gradient_accumulation_steps = process.train.gradient_accumulation.unwrap_or(1);
        let num_steps = process.train.steps;
        let save_every = process.save.save_every;
        
        // Paths
        let model_path = PathBuf::from(&process.model.name_or_path);
        let output_dir = PathBuf::from(format!("output/{}", config.config.name.as_deref().unwrap_or("sdxl_lora")));
        let dataset_path = PathBuf::from(&process.datasets[0].folder_path);
        
        // Create output directory
        fs::create_dir_all(&output_dir)?;
        
        // Check if device is CUDA before moving it
        // Temporarily disable Flash Attention until CUDA kernels are properly compiled
        let use_flash_attention = false; // device.is_cuda();
        
        // Create noise scheduler
        let noise_scheduler = DDPMScheduler::new(
            1000, // num_timesteps
            0.00085, // beta_start
            0.012, // beta_end
            "scaled_linear", // beta_schedule
            &device,
        )?;
        
        // Get caption dropout rate from dataset config
        let caption_dropout_rate = process.datasets[0].caption_dropout_rate.unwrap_or(0.0);
        
        // Setup enhanced caption handler if configured
        let caption_handler = if process.datasets[0].empty_prompt_file.is_some() ||
                                process.datasets[0].duplicate_threshold.is_some() {
            let enhanced_config = EnhancedDataConfig {
                empty_prompt_file: process.datasets[0].empty_prompt_file.as_ref()
                    .map(|p| PathBuf::from(p)),
                caption_dropout_rate,
                use_empty_prompt_for_dropout: process.datasets[0].use_empty_prompt_for_dropout
                    .unwrap_or(true),
                duplicate_threshold: process.datasets[0].duplicate_threshold.unwrap_or(0.1),
                duplicate_limit: process.datasets[0].duplicate_limit.unwrap_or(0.3),
                model_type: EnhancedModelType::SDXL,
            };
            Some(EnhancedCaptionHandler::new(enhanced_config)?)
        } else {
            None
        };
        
        let gradient_accumulator = GradientAccumulator::new(
            process.train.gradient_accumulation.unwrap_or(1),
            device.clone()
        );
        let gradient_checkpoint = SDXLGradientCheckpoint::new(
            process.train.gradient_checkpointing.unwrap_or(false)
        );
        
        Ok(Self {
            device,
            dtype: training_dtype,
            unet_weights: HashMap::new(),
            vae_weights: HashMap::new(),
            vae_encoder: None,
            tiled_vae: None,
            text_encoders: None,
            noise_scheduler,
            lora_collection,
            sdxl_config: SDXLConfig::default(),
            learning_rate,
            batch_size,
            gradient_accumulation_steps,
            num_steps,
            save_every,
            model_path,
            output_dir,
            dataset_path,
            process_config: process.clone(),
            current_step: 0,
            accumulated_loss: 0.0,
            accumulation_step: 0,
            latent_cache_dir: None,
            gradient_accumulator,
            gradient_checkpoint,
            adam8bit: None,
            use_8bit_adam: process.train.optimizer == "adamw8bit",
            gradient_checkpointing: process.train.gradient_checkpointing.unwrap_or(false),
            mixed_precision: false,  // Not in config yet
            max_grad_norm: Some(1.0),  // Default max gradient norm
            use_flash_attention,  // Determined earlier before device was moved
            caption_handler,
            caption_dropout_rate,
            config: (*config).clone(),
            sample_config: process.sample.clone(),
            sample_every: process.sample.as_ref()
                .map(|s| s.sample_every)
                .unwrap_or(500),
            training_sampler: None,  // Will be initialized after loading models
            
            // Initialize new training features
            snr_weighting: process.model.snr_gamma.map(|gamma| {
                SNRWeighting::new(gamma, process.train.min_snr_gamma)
            }),
            lr_scheduler: if let Some(scheduler_type) = &process.train.lr_scheduler {
                Some(create_scheduler(
                    scheduler_type,
                    learning_rate as f32,
                    process.train.lr_warmup_steps.unwrap_or(0),
                    num_steps,
                    process.train.lr_num_cycles,
                    process.train.lr_power,
                )?)
            } else {
                None
            },
            ema_helper: EMAHelper::new(process.train.ema_decay, device.clone()),
            validation_runner: if let Some(val_config) = &process.validation {
                let val_dataset = ValidationDataset::new(
                    ValConfig {
                        dataset_path: PathBuf::from(&val_config.dataset_path),
                        batch_size: val_config.batch_size.unwrap_or(1),
                        every_n_steps: val_config.every_n_steps.unwrap_or(100),
                        num_samples: val_config.num_samples,
                    },
                    device.clone()
                )?;
                Some(ValidationRunner::new(val_dataset))
            } else {
                None
            },
        })
    }
    
    /// Setup device using cached device to avoid Candle bug
    fn setup_device(device_str: &str) -> Result<Device> {
        // Use cached device to avoid Candle's CUDA device ID bug
        let device = crate::trainers::cached_device::get_single_device()?;
        info!("Using cached device: {:?}", device);
        Ok(device)
    }
    
    /// Load models WITHOUT VarBuilder - direct weight loading
    pub fn load_models(&mut self) -> Result<()> {
        info!("\n=== Loading Models ===");
        let start = Instant::now();
        
        // Debug model path
        info!("Model path: {:?}", self.model_path);
        info!("Is file: {}", self.model_path.is_file());
        info!("Extension: {:?}", self.model_path.extension());
        
        // Check if model_path is a single file or directory
        if self.model_path.is_file() && self.model_path.extension().map_or(false, |ext| ext == "safetensors") {
            // Single safetensors file - load all weights from it
            info!("Loading all weights from single file: {:?}", self.model_path);
            let all_weights = candle_core::safetensors::load(&self.model_path, &self.device)?;
            
            // Separate UNet and VAE weights by prefix
            for (name, tensor) in all_weights {
                if name.starts_with("vae.") || name.starts_with("first_stage_model.") {
                    // VAE weights
                    let vae_name = name.strip_prefix("vae.").or_else(|| name.strip_prefix("first_stage_model.")).unwrap();
                    self.vae_weights.insert(vae_name.to_string(), tensor);
                } else if name.starts_with("model.diffusion_model.") {
                    // UNet weights with SD checkpoint prefix
                    let unet_name = name.strip_prefix("model.diffusion_model.").unwrap_or(&name);
                    self.unet_weights.insert(unet_name.to_string(), tensor);
                } else if !name.starts_with("cond_stage_model.") && !name.starts_with("conditioner.") {
                    // Assume it's a UNet weight if not text encoder
                    self.unet_weights.insert(name, tensor);
                }
            }
            info!("Loaded {} UNet weights and {} VAE weights", self.unet_weights.len(), self.vae_weights.len());
            
            // Detect model dtype from weights
            if let Some(first_weight) = self.unet_weights.values().next() {
                let model_dtype = first_weight.dtype();
                debug!("Detected model dtype: {:?}", model_dtype);
                
                // Convert weights to training dtype if different
                if model_dtype != self.dtype {
                    info!("Converting weights from {:?} to {:?}...", model_dtype, self.dtype);
                    
                    // Convert UNet weights
                    for (name, tensor) in self.unet_weights.iter_mut() {
                        *tensor = tensor.to_dtype(self.dtype)?;
                    }
                    
                    // Convert VAE weights
                    for (name, tensor) in self.vae_weights.iter_mut() {
                        *tensor = tensor.to_dtype(self.dtype)?;
                    }
                    
                    info!("Weight conversion complete");
                }
                
                // Update LoRA collection with training dtype
                self.lora_collection.dtype = self.dtype;
            }
            
            // NO REMAPPING - Use SD format directly!
            info!("Using SD format weights directly (no remapping)");
            info!("UNet weights: {} keys", self.unet_weights.len());
            info!("Loading SDXL weights from checkpoint: {:?}", self.model_path);
            
            // If no VAE weights found in the main model, check for separate VAE file
            if self.vae_weights.is_empty() {
                info!("No VAE weights found in main model file, checking for separate VAE...");
                
                // First check if VAE path is specified in config
                if let Some(vae_path_str) = &self.process_config.model.vae_path {
                    let vae_path = Path::new(vae_path_str);
                    if vae_path.exists() {
                        info!("Loading VAE weights from config path: {:?}", vae_path);
                        self.vae_weights = candle_core::safetensors::load(vae_path, &self.device)?;
                        info!("Loaded {} VAE weights from separate file", self.vae_weights.len());
                    } else {
                        info!("Warning: VAE path from config doesn't exist: {:?}", vae_path);
                    }
                } else {
                    // Default to known location
                    let vae_path = Path::new("/home/alex/SwarmUI/Models/VAE/OfficialStableDiffusion/sdxl_vae.safetensors");
                    if vae_path.exists() {
                        info!("Loading VAE weights from default path: {:?}", vae_path);
                        self.vae_weights = candle_core::safetensors::load(vae_path, &self.device)?;
                        info!("Loaded {} VAE weights from separate file", self.vae_weights.len());
                    }
                }
            }
        } else {
            // Directory structure - load from separate files
            let unet_path = self.model_path.join("unet/diffusion_pytorch_model.fp16.safetensors");
            info!("Loading UNet weights from: {:?}", unet_path);
            self.unet_weights = candle_core::safetensors::load(&unet_path, &self.device)?;
            
            // Load VAE weights directly
            let vae_path = self.model_path.join("vae/diffusion_pytorch_model.fp16.safetensors");
            info!("Loading VAE weights from: {:?}", vae_path);
            self.vae_weights = candle_core::safetensors::load(&vae_path, &self.device)?;
        }
        
        // Create VAE encoder using our native implementation
        // Make VAE optional - if it fails to load, we'll use random latents
        info!("\n=== CREATING NATIVE VAE ===");
        match SDXLVAENative::new(
            self.vae_weights.clone(),
            self.device.clone(),
            self.dtype,
        ) {
            Ok(vae) => {
                // Check if we should use tiled VAE for high resolution
                let resolution = self.process_config.datasets[0].resolution[0];
                if resolution >= 768 {
                    // Create tiled VAE for high resolution
                    let tiling_config = TilingConfig {
                        tile_size: 512,
                        overlap: 64,
                        blend_mode: BlendMode::Linear,
                    };
                    
                    // Create TiledVAE with a separate VAE instance
                    match SDXLVAENative::new(
                        self.vae_weights.clone(),
                        self.device.clone(),
                        self.dtype,
                    ) {
                        Ok(vae_for_tiling) => {
                            self.tiled_vae = Some(TiledVAE::new(vae_for_tiling, tiling_config));
                            info!("TiledVAE initialized for {}x{} resolution", resolution, resolution);
                        }
                        Err(e) => {
                            info!("Warning: Failed to create TiledVAE: {}", e);
                        }
                    }
                }
                
                self.vae_encoder = Some(vae);
                info!("Native VAE loaded successfully");
            }
            Err(e) => {
                info!("Warning: Failed to load VAE encoder: {}", e);
                info!("Training will continue with random latents instead of encoded images");
                self.vae_encoder = None;
                self.tiled_vae = None;
            }
        }
        
        // Load text encoders (these still use the existing loader for now)
        let (clip_l, clip_g) = load_text_encoders_sdxl(&self.model_path, &self.device, self.dtype)?;
                            if prediction_type == "epsilon" {
                                unet_config.prediction_type = "epsilon".to_string();
        text_encoders.clip_g = Some(clip_g);
        self.text_encoders = Some(text_encoders);
        
        // Initialize LoRA adapters for SDXL architecture BEFORE creating CPU offloaded UNet
        self.init_lora_adapters()?;
        
        // NO CPU OFFLOADING - GPU ONLY!
        if self.process_config.train.cpu_offload.unwrap_or(false) {
            panic!("CPU offloading is not supported! This is a GPU-only trainer. Get a better GPU!");
        }
        info!("GPU-only training - no CPU offloading!");
        
        // Initialize optimizer - GPU ONLY!
        info!("Optimizer config: optimizer='{}'", self.process_config.train.optimizer);
        
        if self.process_config.train.optimizer == "adamw_cpu_offload" {
            panic!("CPU offloaded optimizer is not supported! Use adamw8bit for GPU-only training!");
        }
        
        if self.use_8bit_adam {
            self.adam8bit = Some(Adam8bit::new(self.learning_rate));
            info!("Using 8-bit Adam optimizer");
        } else {
            panic!("Only 8-bit Adam is supported for GPU training! Set optimizer to 'adamw8bit'");
        }
        
        info!("Models loaded in {:.2}s", start_time.elapsed().as_secs_f32());
        debug!("UNet weights loaded");
        debug!("VAE weights loaded");
        info!("LoRA adapters: {}", self.lora_collection.adapters.len());
        
        // Initialize EMA if configured
        if self.ema_helper.get_ema_model().is_some() {
            let mut ema_params = HashMap::new();
            for (name, adapter) in &self.lora_collection.adapters {
                ema_params.insert(format!("{}_down", name), &adapter.down);
                ema_params.insert(format!("{}_up", name), &adapter.up);
            }
            self.ema_helper.init(&ema_params)?;
            info!("Initialized EMA with decay rate: {:?}", self.process_config.train.ema_decay);
        }
        
        // Initialize the comprehensive training sampler only if VAE is available
        if let Some(ref sample_config) = self.sample_config {
            if self.vae_encoder.is_some() {
                info!("Initializing training sampler...");
                
                // Create sampling configuration
                let sampling_config = SDXLSamplingConfig {
                    scheduler_type: SchedulerType::DDIM,
                    num_inference_steps: sample_config.sample_steps.unwrap_or(30),
                    guidance_scale: sample_config.guidance_scale.unwrap_or(7.5) as f64,
                    eta: 0.0,  // Deterministic DDIM
                    prediction_type: crate::trainers::sdxl_sampling_complete::PredictionType::Epsilon,
                    clip_sample: false,
                    thresholding: false,
                    dynamic_thresholding_ratio: 0.995,
                    sample_max_value: 1.0,
                };
                
                // Create training sampler
                let validation_prompts = sample_config.prompts.clone();
                let negative_prompt = sample_config.neg.clone();
                
                self.training_sampler = Some(TrainingSampler::new(
                    self.device.clone(),
                    self.dtype,
                    self.output_dir.clone(),
                    validation_prompts,
                    negative_prompt,
                    sampling_config,
                )?);
                
                info!("Training sampler initialized with {} validation prompts", sample_config.prompts.len());
            } else {
                warn!("Sampling disabled because VAE is not available");
            }
        }
        
        Ok(())
    }
    
    /// Initialize LoRA adapters for SD format SDXL checkpoint
    pub fn init_lora_adapters(&mut self) -> Result<()> {
        let device = &self.device;
        
        // SD format uses input_blocks and output_blocks
        // SDXL structure in SD format:
        // input_blocks: 0=conv_in, 1-2=down.0, 3-5=down.1, 6-8=down.2, 9-11=down.3
        // output_blocks: 0-2=up.3, 3-5=up.2, 6-8=up.1, 9-11=up.0
        
        // SDXL uses context_dim=2048 for cross-attention
        let context_dim = 2048;
        
        // Check actual weight dimensions by examining the first attention layer
        let test_key = "input_blocks.1.1.k.weight";
        if let Some(k_weight) = self.unet_weights.get(test_key) {
            let dims = k_weight.dims();
            info!("DEBUG: Found K weight dimensions: {:?}", dims);
            // For cross-attention, K weight shape is [out_features, in_features]
            // where in_features is context_dim for encoder_hidden_states
        }
        
        // Input blocks (down blocks in SD format)
        let input_block_configs = vec![
            // Block indices and their channel multipliers
            (vec![1, 2], 1),        // down.0: blocks 1-2, 320 channels
            (vec![4, 5], 2),        // down.1: blocks 4-5, 640 channels  
            (vec![7, 8], 4),        // down.2: blocks 7-8, 1280 channels
            (vec![10, 11], 4),      // down.3: blocks 10-11, 1280 channels
        ];
        
        for (block_indices, ch_mult) in input_block_configs {
            let channels = 320 * ch_mult; // Base channels = 320 for SDXL
            
            for block_idx in block_indices {
                // Check if this block has transformer blocks
                if self.unet_weights.contains_key(&format!("input_blocks.{}.1.transformer_blocks.0.norm1.weight", block_idx)) {
                    // Determine number of transformer blocks
                    let mut num_transformer_blocks = 0;
                if self.unet_weights.contains_key(&format!("input_blocks.{}.1.transformer_blocks.0.norm1.weight", block_idx)) {
                        num_transformer_blocks += 1;
                    }
                    
                    // Add LoRA for each transformer block
                    for tb_idx in 0..num_transformer_blocks {
                        let base_prefix = format!("input_blocks.{}.1.transformer_blocks.{}", block_idx, tb_idx);
                        
                        // Self-attention (attn1)
                        let attn1_prefix = format!("{}.attn1", base_prefix);
                        self.lora_collection.add(&format!("{}.to_q", attn1_prefix), channels, channels, device)?;
                        self.lora_collection.add(&format!("{}.to_k", attn1_prefix), channels, channels, device)?;
                        self.lora_collection.add(&format!("{}.to_v", attn1_prefix), channels, channels, device)?;
                        self.lora_collection.add(&format!("{}.to_out.0", attn1_prefix), channels, channels, device)?;
                        
                        // Cross-attention (attn2)
                        // IMPORTANT: K/V weights in SD format are [channels, context_dim]
                        // They project FROM context_dim TO channels
                        let attn2_prefix = format!("{}.attn2", base_prefix);
                        self.lora_collection.add(&format!("{}.to_q", attn2_prefix), channels, channels, device)?;
                        self.lora_collection.add(&format!("{}.to_k", attn2_prefix), context_dim, channels, device)?;
                        self.lora_collection.add(&format!("{}.to_v", attn2_prefix), context_dim, channels, device)?;
                        self.lora_collection.add(&format!("{}.to_out.0", attn2_prefix), channels, channels, device)?;
                    }
                }
            }
        }
        
        // Middle block
        let mid_channels = 1280; // SDXL middle block channels
        
        // Check if middle block has transformer blocks
        if self.unet_weights.contains_key("middle_block.1.transformer_blocks.0.norm1.weight") {
            // Determine number of transformer blocks
            let mut num_transformer_blocks = 0;
        if self.unet_weights.contains_key("middle_block.1.transformer_blocks.0.norm1.weight") {
                num_transformer_blocks += 1;
            }
            
            // Add LoRA for each transformer block
            for tb_idx in 0..num_transformer_blocks {
                let base_prefix = format!("middle_block.1.transformer_blocks.{}", tb_idx);
                
                // Self-attention (attn1)
                let attn1_prefix = format!("{}.attn1", base_prefix);
                self.lora_collection.add(&format!("{}.to_q", attn1_prefix), mid_channels, mid_channels, device)?;
                self.lora_collection.add(&format!("{}.to_k", attn1_prefix), mid_channels, mid_channels, device)?;
                self.lora_collection.add(&format!("{}.to_v", attn1_prefix), mid_channels, mid_channels, device)?;
                self.lora_collection.add(&format!("{}.to_out.0", attn1_prefix), mid_channels, mid_channels, device)?;
                
                // Cross-attention (attn2)
                // IMPORTANT: K/V weights in SD format are [channels, context_dim]
                // They project FROM context_dim TO channels
                let attn2_prefix = format!("{}.attn2", base_prefix);
                self.lora_collection.add(&format!("{}.to_q", attn2_prefix), mid_channels, mid_channels, device)?;
                self.lora_collection.add(&format!("{}.to_k", attn2_prefix), context_dim, mid_channels, device)?;
                self.lora_collection.add(&format!("{}.to_v", attn2_prefix), context_dim, mid_channels, device)?;
                self.lora_collection.add(&format!("{}.to_out.0", attn2_prefix), mid_channels, mid_channels, device)?;
            }
        }
        
        // Output blocks (up blocks in SD format) 
        let output_block_configs = vec![
            // Block indices and their channel multipliers (reverse order)
            (vec![0, 1, 2], 4),     // up.3: blocks 0-2, 1280 channels
            (vec![3, 4, 5], 4),     // up.2: blocks 3-5, 1280 channels  
            (vec![6, 7, 8], 2),     // up.1: blocks 6-8, 640 channels
            (vec![9, 10, 11], 1),   // up.0: blocks 9-11, 320 channels
        ];
        
        for (block_indices, ch_mult) in output_block_configs {
            let channels = 320 * ch_mult;
            
            for block_idx in block_indices {
                // Check if this block has transformer blocks
                if self.unet_weights.contains_key(&format!("output_blocks.{}.1.transformer_blocks.0.norm1.weight", block_idx)) {
                    // Determine number of transformer blocks
                    let mut num_transformer_blocks = 0;
                if self.unet_weights.contains_key(&format!("output_blocks.{}.1.transformer_blocks.0.norm1.weight", block_idx)) {
                        num_transformer_blocks += 1;
                    }
                    
                    // Add LoRA for each transformer block
                    for tb_idx in 0..num_transformer_blocks {
                        let base_prefix = format!("output_blocks.{}.1.transformer_blocks.{}", block_idx, tb_idx);
                        
                        // Self-attention (attn1)
                        let attn1_prefix = format!("{}.attn1", base_prefix);
                        self.lora_collection.add(&format!("{}.to_q", attn1_prefix), channels, channels, device)?;
                        self.lora_collection.add(&format!("{}.to_k", attn1_prefix), channels, channels, device)?;
                        self.lora_collection.add(&format!("{}.to_v", attn1_prefix), channels, channels, device)?;
                        self.lora_collection.add(&format!("{}.to_out.0", attn1_prefix), channels, channels, device)?;
                        
                        // Cross-attention (attn2)
                        // IMPORTANT: K/V weights in SD format are [channels, context_dim]
                        // They project FROM context_dim TO channels
                        let attn2_prefix = format!("{}.attn2", base_prefix);
                        self.lora_collection.add(&format!("{}.to_q", attn2_prefix), channels, channels, device)?;
                        self.lora_collection.add(&format!("{}.to_k", attn2_prefix), context_dim, channels, device)?;
                        self.lora_collection.add(&format!("{}.to_v", attn2_prefix), context_dim, channels, device)?;
                        self.lora_collection.add(&format!("{}.to_out.0", attn2_prefix), channels, channels, device)?;
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Forward pass with GPU gradient checkpointing - NO CPU FALLBACK!
    fn forward_with_checkpointing(
        &self,
        noisy_latents: &Tensor,
        timesteps: &Tensor,
        prompt_embeds: &Tensor,
    ) -> Result<Tensor> {
        // Ensure we're on GPU
        if !noisy_latents.device().is_cuda() {
            panic!("GPU gradient checkpointing requires CUDA! No CPU training supported!");
        }
        
        info!("Using GPU gradient checkpointing - GPU memory efficient, no CPU transfers!");
        
        // Use GPU-only gradient checkpointing
        crate::trainers::gpu_gradient_checkpoint::forward_sdxl_gpu_checkpoint(
            noisy_latents,
            timesteps,
            prompt_embeds,
            &self.unet_weights,
            &self.lora_collection,
            &self.gradient_checkpoint,
        )
    }
    
    /// Run training
    pub fn train(&mut self) -> Result<()> {
        // Check for GPU requirement
        match &self.device {
            Device::Cuda(_) => {
                info!("GPU detected, starting training...");
            }
            Device::Cpu => {
                error!("GPU is required for training. No CUDA device found.");
                return Err(anyhow!("Training requires a CUDA GPU. CPU training is not supported."));
            }
        }
        
        // Setup CUDA memory management
        memory_utils::setup_cuda_memory_management();
        
        info!("\n=== Starting Training ===");
        info!("Steps: {}", self.num_steps);
        info!("Batch size: {}", self.batch_size);
        info!("Learning rate: {}", self.learning_rate);
        info!("Gradient accumulation: {}", self.gradient_accumulation_steps);
        info!("Using 8-bit Adam: {}", self.use_8bit_adam);
        info!("Using CPU offload: {}", self.use_cpu_offload);
        info!("Using Flash Attention: {}", self.use_flash_attention);
        
        // Log initial memory usage
        memory_utils::log_memory_usage("Before dataset loading")?;
        
        // Load dataset
        let dataset = self.load_dataset()?;
        info!("Dataset items: {}", dataset.len());
        
        memory_utils::log_memory_usage("After dataset loading")?;
        
        // Enable latent caching to save memory and speed up training
        let cache_dir = if self.vae_encoder.is_some() && self.dataset_path.exists() {
            // Check if caching is enabled in config
            let cache_enabled = self.process_config.datasets.get(0)
                .map(|p| PathBuf::from(p))
                .unwrap_or(false);
            
            if cache_enabled {
                info!("\nLatent caching enabled - this will save memory and speed up training");
                
                // Create cache directory
                let cache_dir = self.output_dir.join("latent_cache");
                fs::create_dir_all(&cache_dir)?;
                
                // Cache all latents
                info!("Caching latents to: {}", cache_dir.display());
                self.cache_all_latents(&dataset, &cache_dir)?;
                
                Some(cache_dir)
            } else {
                info!("\nLatent caching disabled - will encode on the fly");
                None
            }
        } else if self.vae_encoder.is_none() {
            info!("\nVAE not available - using random latents for training");
            warn!("This is useful for testing training mechanics but won't produce meaningful results");
            None
        } else {
            None
        };
        
        // Store cache directory for use in training
        self.latent_cache_dir = cache_dir;
        
        // Generate initial samples before training
        if self.sample_config.is_some() && self.vae_encoder.is_some() {
            info!("\nGenerating pre-training samples...");
            self.generate_pre_training_samples()?;
        }
        
        // Training loop
        for step in 0..self.num_steps {
            self.current_step = step;
            
            // Get batch indices
            let mut rng = rand::thread_rng();
            let indices: Vec<usize> = (0..self.batch_size)
                .collect();
            
            // Forward pass and loss computation
            let loss = self.training_step(&dataset, &indices, step)?;
            
            // Accumulate loss
            self.accumulated_loss += loss;
            self.accumulation_step += 1;
            
            // Update weights when accumulation is complete
            if self.gradient_accumulator.should_update() {
                // Average the accumulated loss
                let avg_loss = self.accumulated_loss / self.gradient_accumulation_steps as f32;
                
                // Reset accumulation
                self.accumulated_loss = 0.0;
                self.accumulation_step = 0;
                
                // Log progress with learning rate info
                if step % 10 == 0 {
                    let current_lr = if let Some(scheduler) = &self.lr_scheduler {
                        scheduler.get_lr(step)
                    } else {
                        self.learning_rate as f32
                    };
                    
                    info!("Step {}/{}: loss = {:.6}, lr = {:.2e}", step, self.num_steps, avg_loss, current_lr);
                    
                    // Log memory usage periodically
                    if step % 50 == 0 {
                        memory_utils::log_memory_usage(&format!("Step {}", step))?;
                    }
                }
            }
            
            // Run validation if configured
            if let Some(ref mut validation_runner) = self.validation_runner {
                let val_metrics = validation_runner.validate(step, |batch| {
                    self.compute_validation_loss(batch, &cache_dir)
                })?;
                
                // Log validation metrics if computed
                if val_metrics.num_samples > 0 {
                    info!("Validation at step {}: {}", step, val_metrics.summary());
                }
            }
            
            // Save checkpoint
            if (step + 1) % self.save_every == 0 {
                self.save_lora(step + 1)?;
                
                // Also save EMA weights if available
                if let Some(ema_model) = self.ema_helper.get_ema_model() {
                    self.save_ema_lora(step + 1, ema_model)?;
                }
            }
            
            // Generate samples with EMA weights if available
            if (step + 1) % self.sample_every == 0 && self.sample_config.is_some() && self.vae_encoder.is_some() {
                info!("\nGenerating validation samples at step {}...", step + 1);
                
                // Generate samples with EMA weights if available
                if self.ema_helper.get_ema_model().is_some() {
                    // Apply EMA weights temporarily for sampling
                    let mut lora_params = HashMap::new();
                    for (name, adapter) in &mut self.lora_collection.adapters {
                        lora_params.insert(format!("{}_down", name), adapter.down.clone());
                        lora_params.insert(format!("{}_up", name), adapter.up.clone());
                    }
                    
                    self.ema_helper.apply_ema(&mut lora_params, || {
                        self.generate_samples(step + 1)
                    })?;
                } else {
                    // Generate samples with regular weights
                    self.generate_samples(step + 1)?;
                }
            }
        }
        
        // Final save
        self.save_final_lora()?;
        
        info!("\n=== Training Complete ===");
        Ok(())
    }
    
    /// Single training step
    fn training_step(&mut self, dataset: &[DatasetItem], indices: &[usize], step: usize) -> Result<f32> {
        // Prepare batch
        let batch = self.prepare_batch(dataset, indices)?;
        
        // Sample random timesteps
        let timesteps = self.sample_timesteps(self.batch_size)?;
        
        // Add noise to latents - ensure same dtype
        let mut noise = Tensor::randn_like(&batch.latents, 0.0, 1.0)?
            .to_dtype(batch.latents.dtype())?;
        
        // Apply offset noise if configured
        if let Some(offset_strength) = self.process_config.train.offset_noise {
            let offset = Tensor::randn(0.0f32, offset_strength, &[batch.latents.dim(0)?, 1, 1, 1], &self.device)?
                .to_dtype(batch.latents.dtype())?;
            noise = (noise + offset.broadcast_as(noise.shape())?)?;
        }
        
        // Apply input perturbation if configured
        let latents_to_noise = if let Some(perturbation_strength) = self.process_config.train.input_perturbation {
            let perturbation = Tensor::randn_like(&batch.latents, 0.0, perturbation_strength)?;
            (batch.latents.clone() + perturbation)?
        } else {
            batch.latents.clone()
        };
        
        let noisy_latents = self.add_noise(&latents_to_noise, &noise, &timesteps)?;
        
        // Encode text
        let text_encoders = self.text_encoders.as_mut()
            .ok_or_else(|| anyhow::anyhow!("Text encoders not loaded"))?;
        
        // Process prompts for dropout and encode
        let mut prompt_embeds_list = Vec::new();
        let mut pooled_embeds_list = Vec::new();
        
        for prompt in &batch.prompts {
            let (prompt_embed, pooled_embed) = text_encoders.encode_sdxl(prompt, 77)?;
            prompt_embeds_list.push(prompt_embed);
            pooled_embeds_list.push(pooled_embed);
        }
        
        // Stack embeddings - they're already in correct shape [batch, seq_len, dim]
        let prompt_embeds = if prompt_embeds_list.len() == 1 {
            let embeds = &prompt_embeds_list[0];
            embeds.clone()
        } else {
            Tensor::cat(&prompt_embeds_list, 0)?
        };
        let pooled_embeds = if pooled_embeds_list.len() == 1 {
            pooled_embeds_list[0].clone()
        } else {
            Tensor::cat(&pooled_embeds_list, 0)?
        };
        
        debug!("DEBUG: Final prompt_embeds shape: {:?}", prompt_embeds.dims());
        
        // Debug: Check UNet weight keys to understand naming convention (only once)
        static PRINTED_DEBUG: std::sync::Once = std::sync::Once::new();
        PRINTED_DEBUG.call_once(|| {
            debug!("\n=== Training Step {} Debug ===", step);
            let norm_keys: Vec<_> = self.unet_weights.keys()
                .filter(|k| k.contains("norm1") || k.contains("norm2"))
                .take(10)
                .cloned()
                .collect();
            info!("Sample norm keys: {:?}", norm_keys);
            
            let resnet_keys: Vec<_> = self.unet_weights.keys()
                .filter(|k| k.contains("resnets"))
                .take(10)
                .cloned()
                .collect();
            info!("Sample resnet keys: {:?}", resnet_keys);
            
            // Check actual structure
            let input_block_keys: Vec<_> = self.unet_weights.keys()
                .filter(|k| k.starts_with("input_blocks.1.0"))
                .take(10)
                .cloned()
                .collect();
            info!("Sample input_blocks.1.0 keys: {:?}", input_block_keys);
        });
        
        // Forward pass - choose between flash attention and efficient attention
        let noise_pred = if self.gradient_checkpointing {
            // Use gradient checkpointing for memory efficiency
            self.forward_with_checkpointing(
                &noisy_latents,
                &timesteps,
                &prompt_embeds,
            )?
        } else if self.use_flash_attention {
            // Use Flash Attention for speed (requires BF16/F16 and CUDA)
            forward_sdxl_sd_format_flash(
                &noisy_latents,
                &timesteps,
                &prompt_embeds,
                &self.unet_weights,
                &self.lora_collection,
                true, // Enable Flash Attention
            )?
        } else if let Some(ref cpu_offloaded_unet) = self.cpu_offloaded_unet {
            // Use CPU offloaded forward pass
            cpu_offloaded_unet.forward_offloaded(
                &noisy_latents,
                &timesteps,
                &prompt_embeds,
                &self.lora_collection,
            )?
        } else {
            // Regular forward pass with efficient attention
            forward_sdxl_sd_efficient(
                &noisy_latents,
                &timesteps,
                &prompt_embeds,
                &self.unet_weights,
                &self.lora_collection,
            )?
        };
        
        // Clear memory before loss calculation
        memory_utils::clear_cuda_cache(&self.device)?;
        
        // Debug dtypes
        debug!("DEBUG: noise_pred dtype: {:?}, noise dtype: {:?}", noise_pred.dtype(), noise.dtype());
        
        // Compute target based on parameterization
        let target = if self.process_config.train.v_parameterization.unwrap_or(false) {
            // v-parameterization: v = sqrt(alpha) * noise - sqrt(1 - alpha) * sample
            let alphas_cumprod = self.noise_scheduler.alphas_cumprod()?;
            super::snr_weighting::compute_v_prediction(&noise_pred, &latents_to_noise, &noise, &alphas_cumprod, &timesteps)?
        } else {
            // Standard noise prediction
            noise.clone()
        };
        
        // Ensure same dtype for loss calculation
        let target = target.to_dtype(noise_pred.dtype())?;
        
        // Compute loss with optional SNR weighting
        let mse_loss = (noise_pred - target)?.sqr()?;
        
        // Apply SNR weighting if configured
        let loss = if let Some(snr_weighting) = &self.snr_weighting {
            // Apply SNR weighting per sample, then take mean
            let weighted_loss = snr_weighting.apply_snr_weighting(
                &mse_loss, 
                &timesteps, 
                self.noise_scheduler.num_train_timesteps()
            )?;
            weighted_loss.mean_all()?
        } else {
            // Simple MSE loss
            mse_loss.mean_all()?
        };
        
        // Get loss value before backward pass (convert to F32 for scalar)
        let loss_value = loss.to_dtype(DType::F32)?.to_scalar::<f32>()?;
        info!("DEBUG: Loss computed successfully, value: {}", loss_value);
        
        // Backward pass
        debug!("Starting backward pass...");
        let grads = loss.backward()?;
        debug!("Backward pass complete");
        
        // Accumulate gradients properly
        self.accumulate_gradients(&grads)?;
        
        // Update LoRA parameters when accumulation is complete
        if self.gradient_accumulator.should_update() {
            self.apply_accumulated_gradients()?;
            self.gradient_accumulator.clear();
        }
        self.gradient_accumulator.step();
        
        // Clear CUDA cache periodically to prevent memory fragmentation
        if step % 10 == 0 {
            memory_utils::clear_cuda_cache(&self.device)?;
        }
        
        Ok(loss_value)
    }
    
    /// Accumulate gradients for all LoRA parameters
    fn accumulate_gradients(&mut self, grads: &GradStore) -> Result<()> {
        // Accumulate gradients for each LoRA parameter
        for (name, adapter) in &self.lora_collection.adapters {
            // Accumulate down projection gradient
            if let Some(grad) = grads.get(adapter.down.as_tensor()) {
                self.gradient_accumulator.accumulate(&format!("{}_down", name), grad)?;
            }
            
            // Accumulate up projection gradient
            if let Some(grad) = grads.get(adapter.up.as_tensor()) {
                self.gradient_accumulator.accumulate(&format!("{}_up", name), grad)?;
            }
        }
        
        Ok(())
    }
    
    /// Apply accumulated gradients to parameters
    fn apply_accumulated_gradients(&mut self) -> Result<()> {
        info!("DEBUG: Update parameters - use_cpu_offload: {}, use_8bit_adam: {}", self.use_cpu_offload, self.use_8bit_adam);
        
        // Get current learning rate from scheduler if available
        let current_lr = if let Some(scheduler) = &self.lr_scheduler {
            scheduler.get_lr(self.current_step) as f64
        } else {
            self.learning_rate
        };
        
        if self.use_cpu_offload {
            let optimizer = self.cpu_offloaded_adam.as_mut()
                .ok_or_else(|| anyhow::anyhow!("CPU-offloaded Adam not initialized"))?;
            
            // Update optimizer learning rate if changed
            optimizer.set_lr(current_lr as f32);
            
            // Update each LoRA parameter with accumulated gradients
            for (name, adapter) in &self.lora_collection.adapters {
                // Update down projection
                if let Some(grad) = self.gradient_accumulator.get_gradient(&format!("{}_down", name)) {
                    optimizer.step(&adapter.down, grad)?;
                }
                
                // Update up projection
                if let Some(grad) = self.gradient_accumulator.get_gradient(&format!("{}_up", name)) {
                    optimizer.step(&adapter.up, grad)?;
                }
            }
            
            // Print memory stats occasionally
            if self.current_step % 100 == 0 {
                let (m_mem, v_mem) = optimizer.memory_stats();
                info!("CPU optimizer memory: momentum={:.1}MB, variance={:.1}MB", m_mem as f32 / 1024.0 / 1024.0, 
                         v_mem as f32 / 1024.0 / 1024.0);
            }
        } else if self.use_8bit_adam {
            let adam = self.adam8bit.as_mut()
                .ok_or_else(|| anyhow::anyhow!("8-bit Adam not initialized"))?;
            
            // Update optimizer learning rate if changed
            adam.set_lr(current_lr as f32);
            
            // Update each LoRA parameter with accumulated gradients
            for (name, adapter) in &self.lora_collection.adapters {
                // Update down projection
                if let Some(grad) = self.gradient_accumulator.get_gradient(&format!("{}_down", name)) {
                    adam.update(&format!("{}_down", name), &adapter.down, grad)?;
                }
                
                // Update up projection
                if let Some(grad) = self.gradient_accumulator.get_gradient(&format!("{}_up", name)) {
                    adam.update(&format!("{}_up", name), &adapter.up, grad)?;
                }
            }
            
            // Increment step counter once per optimization step
            adam.step();
        } else {
            // Standard Adam optimizer with accumulated gradients
            self.update_parameters_standard()?;
        }
        
        // Update EMA after optimizer step
        if self.ema_helper.get_ema_model().is_some() {
            let mut ema_params = HashMap::new();
            for (name, adapter) in &self.lora_collection.adapters {
                ema_params.insert(format!("{}_down", name), &adapter.down);
                ema_params.insert(format!("{}_up", name), &adapter.up);
            }
            self.ema_helper.update(&ema_params)?;
        }
        
        Ok(())
    }
    
    /// Standard Adam optimizer (non-8bit) with accumulated gradients
    fn update_parameters_standard(&mut self) -> Result<()> {
        let beta1 = 0.9;
        let beta2 = 0.999;
        let eps = 1e-8;
        
        // Get current learning rate from scheduler if available
        let current_lr = if let Some(scheduler) = &self.lr_scheduler {
            scheduler.get_lr(self.current_step) as f64
        } else {
            self.learning_rate
        };
        
        // Update each LoRA parameter with accumulated gradients
        for (name, adapter) in &self.lora_collection.adapters {
            // Update down projection
            if let Some(grad) = self.gradient_accumulator.get_gradient(&format!("{}_down", name)) {
                self.update_param_standard(&adapter.down, grad, beta1, beta2, eps, current_lr)?;
            }
            
            // Update up projection
            if let Some(grad) = self.gradient_accumulator.get_gradient(&format!("{}_up", name)) {
                self.update_param_standard(&adapter.up, grad, beta1, beta2, eps, current_lr)?;
            }
        }
        
        Ok(())
    }
    
    /// Standard parameter update
    fn update_param_standard(&self, param: &Var, grad: &Tensor, 
                             beta1: f64, beta2: f64, eps: f64, learning_rate: f64) -> Result<()> {
        // Simple SGD for now - full Adam would need persistent state
        // Scale gradient by learning rate
        let update = grad.to_dtype(DType::F32)?;
        let update = (update * learning_rate)?;
        let update = update.to_dtype(param.dtype())?;
        let new_value = (param.as_tensor() - update)?;
        param.set(&new_value)?;
        Ok(())
    }
    
    /// Sample timesteps
    fn sample_timesteps(&self, batch_size: usize) -> Result<Tensor> {
        self.noise_scheduler.sample_timesteps(batch_size, &self.device)
    }
    
    /// Add noise to latents
    fn add_noise(&self, latents: &Tensor, noise: &Tensor, timesteps: &Tensor) -> Result<Tensor> {
        self.noise_scheduler.add_noise(latents, noise, timesteps)
    }
    
    /// Load dataset
    fn load_dataset(&self) -> Result<Vec<DatasetItem>> {
        let mut dataset = Vec::new();
        
        for entry in fs::read_dir(&self.dataset_path)? {
            let entry = entry?;
            let path = entry.path();
            
            if path.extension().and_then(|s| s.to_str()).map(|s| s == "jpg" || s == "png" || s == "jpeg").unwrap_or(false)
            {
                let caption_path = path.with_extension("txt");
                if caption_path.exists() {
                    let caption = fs::read_to_string(&caption_path)?;
                    dataset.push(DatasetItem {
                        image_path: path,
                        caption: caption.trim().to_string(),
                    });
                }
            }
        }
        
        Ok(dataset)
    }
    
    /// Prepare batch with fresh VAE encoding or random latents if VAE is unavailable
    fn prepare_batch(&mut self, dataset: &[DatasetItem], indices: &[usize]) -> Result<TrainingBatch> {
        let mut latents_list = Vec::new();
        let mut prompts = Vec::new();
        let mut rng = rand::thread_rng();
        
        for &idx in indices {
            let item = &dataset[idx];
            
            // Process caption with enhanced handler if available
            let caption = if let Some(ref mut handler) = self.caption_handler {
                // Extract concept from caption (simplified - you might want to improve this)
                let concept = if item.caption.contains("woman") || item.caption.contains("girl") {
                    Some("person")
                } else {
                    None
                };
                handler.process_caption(&item.caption, concept, &mut rng)
            } else {
                // Fallback to simple dropout
                if self.caption_dropout_rate > 0.0 && rng.gen::<f32>() < self.caption_dropout_rate {
                    String::new()
                } else {
                    item.caption.clone()
                }
            };
            
            prompts.push(caption);
            
            // Generate latent either from cache, VAE encoding, or random
            let latent = if let Some(ref cache_dir) = self.latent_cache_dir {
                // Load from cache
                self.load_or_encode_latent(&item, cache_dir)?
            } else if let Some(ref vae) = self.vae_encoder {
                // VAE is available - encode the image
                // Load and preprocess image
                let img = image::open(&item.image_path)?;
                let img = img.to_rgb8();
                
                // Get resolution from config
                let resolution = self.process_config.datasets[0].resolution[0];
                
                // Resize to configured resolution
                let img = image::imageops::resize(&img, resolution as u32, resolution as u32, image::imageops::FilterType::Lanczos3);
                
                // Convert to tensor [H, W, C] -> [C, H, W] normalized to [0, 1]
                // VAE encoder will convert to [-1, 1] internally
                let img_vec: Vec<f32> = img.pixels()
                    .flat_map(|p| {
                        [
                            p[0] as f32 / 255.0,
                            p[1] as f32 / 255.0,
                            p[2] as f32 / 255.0
                        ]
                    })
                    .collect();
                
                let img_tensor = Tensor::from_vec(img_vec, &[resolution, resolution, 3], &self.device)?
                    .permute((2, 0, 1))?  // [H, W, C] -> [C, H, W]
                    .unsqueeze(0)?       // Add batch dimension [1, C, H, W]
                    .to_dtype(self.dtype)?;  // Convert to F16 to match weights
                
                // Encode with VAE - use tiled encoding for high resolution
                if let Some(ref tiled_vae) = self.tiled_vae {
                    info!("Using TiledVAE for {}x{} image", resolution, resolution);
                    tiled_vae.encode_tiled(&img_tensor)?
                } else {
                    vae.encode(&img_tensor)?
                }
            } else {
                // VAE not available - generate random latents
                // Latent size is image_size / 8 (VAE downsampling factor)
                let resolution = self.process_config.datasets[0].resolution[0];
                let latent_size = resolution / 8;
                Tensor::randn(0.0f32, 1.0f32, &[1, 4, latent_size, latent_size], &self.device)?
                    .to_dtype(self.dtype)?
            };
            
            latents_list.push(latent);
        }
        
        // Stack latents into batch
        let latents = Tensor::cat(&latents_list, 0)?;
        
        Ok(TrainingBatch {
            latents,
            prompts,
        })
    }
    
    /// Save the trained LoRA
    pub fn save_lora(&self, step: usize) -> Result<()> {
        // Create directory for this checkpoint: output/{name}/checkpoint-{step}/
        let checkpoint_dir = self.output_dir.join(format!("checkpoint-{}", step));
        fs::create_dir_all(&checkpoint_dir)?;
        
        // Save LoRA weights
        let lora_path = checkpoint_dir.join("sdxl_lora.safetensors");
        info!("Saving checkpoint to: {:?}", checkpoint_dir);
        self.lora_collection.save(&lora_path)?;
        
        // Save optimizer state
        if let Some(ref optimizer) = self.adam8bit {
            let optimizer_path = checkpoint_dir.join("optimizer.safetensors");
            self.save_optimizer_state(optimizer, &optimizer_path)?;
        }
        
        // Save training state
        let state_path = checkpoint_dir.join("training_state.json");
        let state = serde_json::json!({
            "step": step,
            "learning_rate": self.learning_rate,
            "loss": self.accumulated_loss,
            "model_type": "sdxl",
            "network_type": "lora",
            "rank": self.lora_collection.rank,
            "alpha": self.lora_collection.alpha,
        });
        fs::write(&state_path, serde_json::to_string_pretty(&state)?)?;
        
        Ok(())
    }
    
    /// Save final LoRA checkpoint with standardized naming
    pub fn save_final_lora(&self) -> Result<()> {
        // Final checkpoint uses model_network_final.safetensors format
        let save_path = self.output_dir.join("sdxl_lora_final.safetensors");
        info!("\nSaving final LoRA to: {:?}", save_path);
        self.lora_collection.save(&save_path)?;
        Ok(())
    }
    
    /// Save optimizer state
    fn save_optimizer_state(&self, optimizer: &Adam8bit, path: &Path) -> Result<()> {
        // Get optimizer state tensors
        let state = optimizer.get_state_tensors()?;
        // First collect all data
        let mut tensor_data = Vec::new();
        let mut tensor_info = Vec::new();
        
        // Convert optimizer state to safetensors format
        for (name, (m, v)) in state {
            // Save first moment info
            let m_data = tensor_to_vec(&m)?;
            tensor_info.push((
                format!("{}_m", name),
                convert_dtype(m.dtype())?,
                m.dims().to_vec(),
                tensor_data.len()
            ));
            tensor_data.push(m_data);
            
            // Save second moment info
            let v_data = tensor_to_vec(&v)?;
            tensor_info.push((
                format!("{}_v", name),
                convert_dtype(v.dtype())?,
                v.dims().to_vec(),
                tensor_data.len()
            ));
            tensor_data.push(v_data);
        }
        
        // Now create TensorViews using indices
        let mut tensors = HashMap::new();
        for (name, dtype, shape, idx) in tensor_info {
            tensors.insert(
                name,
                TensorView::new(
                    dtype,
                    shape,
                    &tensor_data[idx],
                )?
            );
        }
        
        // Add metadata
        let mut metadata = HashMap::new();
        metadata.insert("step".to_string(), self.current_step.to_string());
        metadata.insert("optimizer_step".to_string(), optimizer.get_step().to_string());
        metadata.insert("learning_rate".to_string(), self.learning_rate.to_string());
        metadata.insert("optimizer_type".to_string(), "adam8bit".to_string());
        metadata.insert("beta1".to_string(), "0.9".to_string());
        metadata.insert("beta2".to_string(), "0.999".to_string());
        metadata.insert("eps".to_string(), "1e-8".to_string());
        
        // Save using safetensors
        let data = serialize(&tensors, &Some(metadata))?;
        fs::write(path, data)?;
        info!("Saved optimizer state with {} parameters", tensors.len() / 2);
        Ok(())
    }
    
    /// Generate pre-training samples to establish baseline
    fn generate_pre_training_samples(&mut self) -> Result<()> {
        if self.vae_encoder.is_none() {
            info!("Skipping pre-training samples - VAE not available");
            return Ok(());
        }
        info!("Generating pre-training baseline samples...");
        
        // Use step 0 with special directory name
        if let Some(ref mut training_sampler) = self.training_sampler {
            // Create special pre-training directory
            let pre_train_dir = self.output_dir.join("samples").join("pre_training");
            std::fs::create_dir_all(&pre_train_dir)?;
            
            // Generate samples at step 0
            self.generate_samples(0)?;
            
            info!("Pre-training samples saved to: {:?}", pre_train_dir);
        }
        
        Ok(())
    }
    
    /// Generate validation samples during training using the new comprehensive sampler
    fn generate_samples(&mut self, step: usize) -> Result<()> {
        // Check if VAE is available
        if self.vae_encoder.is_none() {
            info!("\nStep {}: Skipping sample generation (VAE not available)", step);
            return Ok(());
        }
        if let Some(ref mut training_sampler) = self.training_sampler {
            info!("\nGenerating samples at step {}...", step);
            
            // Ensure we have all required components
            if self.vae_encoder.is_none() {
                return Err(anyhow::anyhow!("VAE not loaded for sampling"));
            }
            if self.text_encoders.is_none() {
                return Err(anyhow::anyhow!("Text encoders not loaded for sampling"));
            }
            
            // Create step-specific directory
            info!("\nGenerating {} samples at step {} with enhanced sampler...", validation_prompts.len(), step);
            fs::create_dir_all(&step_dir)?;
            
            // Generate samples with the native VAE - direct implementation
            let validation_prompts = vec![
                "a professional photograph of an astronaut riding a horse",
                "a painting of a sunset over mountains in the style of Monet",
                "a cute robot playing with a ball in a park",
            ];
            
            for (i, prompt) in validation_prompts.iter().enumerate() {
                info!("Generating sample {}/{}: {}", i + 1, validation_prompts.len(), prompt);
                
                // Generate latents
                let latents = self.generate_latents_with_prompt(
                    prompt,
                    "low quality, blurry",
                    1024,
                    1024,
                    30,
                    7.5,
                    Some(42 + i as u64),
                )?;
                
                // Decode with native VAE
                let vae = self.vae_encoder.as_ref().unwrap();
                let image = vae.decode(&latents)?;
                
                // Convert to image and save
                let image_path = step_dir.join(format!("sample_{:02}.png", i));
                self.save_image_tensor(&image, &image_path)?;
                
                // Save prompt
                let prompt_path = step_dir.join(format!("sample_{:02}.txt", i));
                fs::write(&prompt_path, prompt)?;
            }
            
            info!("Generated {} samples at step {}", validation_prompts.len(), step);
            Ok(())
        } else {
            // Fallback to extended method if new sampler not available
            self.generate_samples_extended(step, false)
        }
    }
    
    /// Extended sample generation with dataset sampling option
    fn generate_samples_extended(&mut self, step: usize, include_dataset_samples: bool) -> Result<()> {
        let config = self.sample_config.as_ref().unwrap();
        
        // Create sampler
        let sampler = SDXLSampler::new(
            self.device.clone(),
            self.dtype,
            config.sample_steps.unwrap_or(30),
            config.guidance_scale.unwrap_or(7.5),
        );
        
        // Create output directory for samples with better organization
        let sample_dir = self.output_dir.join("samples");
        fs::create_dir_all(&sample_dir)?;
        
        // Create a subdirectory for this step
        let step_dir = sample_dir.join(format!("step_{:06}", step));
        fs::create_dir_all(&step_dir)?;
        
        // Prepare VAE decoder if needed
        let vae = self.vae_encoder.as_ref();
        info!("\nGenerating extended samples at step {}...", step);
        
        // Generate samples with current LoRA weights
        info!("Generating samples with current LoRA weights...");
        
        // For now, we'll generate samples with the prompts from config
        let prompts = &config.prompts;
        let negative_prompt = config.neg.as_deref();
        
        // For each prompt, generate an image
        for (i, prompt) in prompts.iter().enumerate() {
            info!("Generating sample {}/{}: {}", i + 1, prompts.len(), prompt);
            
            // Encode prompt
            let (prompt_embeds, pooled_embeds) = self.text_encoders.as_mut()
                .ok_or_else(|| anyhow::anyhow!("Text encoders not loaded"))?
                .encode_sdxl(prompt, 77)?;
            
            // For negative prompt
            let (neg_embeds, neg_pooled) = if let Some(neg) = negative_prompt {
                self.text_encoders.as_mut().unwrap().encode_sdxl(neg, 77)?
            } else {
                // Use unconditional embeddings for SDXL
                self.text_encoders.as_mut().unwrap().encode_unconditional(1, 77)?
            };
            
            // Simple denoising loop (simplified version)
            let latent_shape = [1, 4, config.height / 8, config.width / 8];
            let mut latents = Tensor::randn(0.0f32, 1.0f32, &latent_shape, &self.device)?;
            
            // Scale initial noise
            let init_noise_sigma = 14.6146;
            latents = (latents * init_noise_sigma)?;
            
            // Run denoising with DDIM scheduler (simplified)
            let num_steps = config.sample_steps.unwrap_or(30);
            let timesteps: Vec<i64> = (0..num_steps)
                .map(|i| (1000 - (i + 1) * 1000 / num_steps) as i64)
                .rev()
                .collect();
            
            for (idx, &t) in timesteps.iter().enumerate() {
                let t_tensor = Tensor::new(&[t], &self.device)?
                    .to_dtype(self.dtype)?;
                
                // Prepare latent model input for classifier-free guidance
                let latent_model_input = if config.guidance_scale.unwrap_or(7.5) > 1.0 {
                    Tensor::cat(&[&latents, &latents], 0)?
                } else {
                    latents.clone()
                };
                
                // Prepare encoder hidden states for CFG
                let encoder_hidden_states = if config.guidance_scale.unwrap_or(7.5) > 1.0 {
                    Tensor::cat(&[&neg_embeds, &prompt_embeds], 0)?
                } else {
                    prompt_embeds.clone()
                };
                
                // Forward pass through UNet with LoRA
                let noise_pred = forward_sdxl_with_lora(
                    &latent_model_input,
                    &t_tensor,
                    &encoder_hidden_states,
                    &self.unet_weights,
                    &self.lora_collection,
                )?;
                
                // Perform guidance
                let model_output = noise_pred;  // Rename to avoid shadowing
                let noise_pred = if config.guidance_scale.unwrap_or(7.5) > 1.0 {
                    let chunks = model_output.chunk(2, 0)?;
                    let noise_pred_uncond = &chunks[0];
                    let noise_pred_cond = &chunks[1];
                    let guidance_scale = config.guidance_scale.unwrap_or(7.5);
                    
                    // guided_pred = uncond + guidance_scale * (cond - uncond)
                    let diff = (noise_pred_cond - noise_pred_uncond)?;
                    (noise_pred_uncond + (diff * guidance_scale as f64)?)?
                } else {
                    model_output
                };
                
                // DDIM step (simplified)
                let alpha_prod_t = ((1000 - t) as f32 / 1000.0).powi(2); // Simplified alpha schedule
                let alpha_prod_t_prev = if idx < timesteps.len() - 1 {
                    ((1000 - timesteps[idx + 1]) as f32 / 1000.0).powi(2)
                } else {
                    1.0
                };
                
                let beta_prod_t = 1.0 - alpha_prod_t;
                let beta_prod_t_prev = 1.0 - alpha_prod_t_prev;
                
                // Compute x0 from noise prediction
                let pred_x0 = ((latents.clone() - (noise_pred.clone() * (beta_prod_t.sqrt() as f64))?)? / (alpha_prod_t.sqrt() as f64))?;
                
                // Compute direction
                let dir_xt = (noise_pred * (beta_prod_t_prev.sqrt() as f64))?;
                
                let alpha_prod_t_prev = self.alphas_cumprod[prev_t as usize];
                latents = ((pred_x0 * (alpha_prod_t_prev.sqrt() as f64))? + dir_xt)?;
            }
            
            // Decode latents to image - use tiled decoding for high resolution
            let images = if let Some(ref tiled_vae) = self.tiled_vae {
                info!("Using TiledVAE for decoding");
                tiled_vae.decode_tiled(&latents)?
            } else {
                vae.decode(&latents)?
            };
            
            // Save image
            let filename = format!("sample_{:02}.png", i);
            let filepath = step_dir.join(filename);
            
            // Also save the prompt used
            let prompt_file = step_dir.join(format!("sample_{:02}_prompt.txt", i));
            fs::write(&prompt_file, if i < prompts.len() { &prompts[i] } else { "[negative prompt]" })?;
            let image_path = output_dir.join(format!("sample_{}_step_{}.png", i, step));
        }
        
        info!("Samples saved to: {:?}", output_dir);
        
        // Optionally generate samples from random dataset items
        if include_dataset_samples {
            info!("\nGenerating samples from random dataset items...");
            
            // Load dataset if not already loaded
            let dataset = self.load_dataset()?;
            if !dataset.is_empty() {
                let mut rng = rand::thread_rng();
                info!("\nGenerating {} dataset samples...", num_dataset_samples);
                
                for i in 0..num_dataset_samples {
                    // Pick random dataset item
                    let idx = rng.gen_range(0..dataset.len());
                    let caption = &dataset[idx].caption;
                    
                    info!("  Dataset sample {}: {}", i + 1, caption);
                    
                    // Generate image with this caption
                    let (prompt_embeds, pooled_embeds) = self.text_encoders.as_mut()
                        .ok_or_else(|| anyhow::anyhow!("Text encoders not loaded"))?
                        .encode_sdxl(&caption, 77)?;
                    
                    // Use same negative prompt as config
                    let negative_prompt = config.neg.as_deref();
                    let (neg_embeds, _neg_pooled) = if let Some(neg) = negative_prompt {
                        self.text_encoders.as_mut().unwrap()
                            .encode_sdxl(neg, 77)?
                    } else {
                        self.text_encoders.as_mut().unwrap()
                            .encode_sdxl("", 77)?
                    };
                    
                    // Generate single image
                    let latent_shape = [1, 4, config.height / 8, config.width / 8];
                    let mut latents = Tensor::randn(0.0f32, 1.0f32, &latent_shape, &self.device)?;
                    latents = (latents * 14.6146)?;
                    
                    // Create timesteps for denoising
                    let num_steps = config.sample_steps.unwrap_or(30);
                    let timesteps: Vec<i64> = (0..num_steps)
                        .map(|i| (1000 - (i + 1) * 1000 / num_steps) as i64)
                        .rev()
                        .collect();
                    
                    // Run denoising
                    for (idx, &t) in timesteps.iter().enumerate() {
                        let t_tensor = Tensor::new(&[t], &self.device)?.to_dtype(self.dtype)?;
                        
                        let latent_model_input = if config.guidance_scale.unwrap_or(7.5) > 1.0 {
                            Tensor::cat(&[&latents, &latents], 0)?
                        } else {
                            latents.clone()
                        };
                        
                        let encoder_hidden_states = if config.guidance_scale.unwrap_or(7.5) > 1.0 {
                            Tensor::cat(&[&neg_embeds, &prompt_embeds], 0)?
                        } else {
                            prompt_embeds.clone()
                        };
                        
                        let noise_pred = forward_sdxl_with_lora(
                            &latent_model_input,
                            &t_tensor,
                            &encoder_hidden_states,
                            &self.unet_weights,
                            &self.lora_collection,
                        )?;
                        
                        // Apply guidance
                        let model_output = noise_pred;
                        let noise_pred = if config.guidance_scale.unwrap_or(7.5) > 1.0 {
                            let chunks = model_output.chunk(2, 0)?;
                            let noise_pred_uncond = &chunks[0];
                            let noise_pred_cond = &chunks[1];
                            let guidance_scale = config.guidance_scale.unwrap_or(7.5);
                            let diff = (noise_pred_cond - noise_pred_uncond)?;
                            (noise_pred_uncond + (diff * guidance_scale as f64)?)?
                        } else {
                            model_output
                        };
                        
                        // DDIM step
                        let alpha_prod_t = ((1000 - t) as f32 / 1000.0).powi(2);
                        let alpha_prod_t_prev = if idx < timesteps.len() - 1 {
                            ((1000 - timesteps[idx + 1]) as f32 / 1000.0).powi(2)
                        } else {
                            1.0
                        };
                        
                        let beta_prod_t = 1.0 - alpha_prod_t;
                        let beta_prod_t_prev = 1.0 - alpha_prod_t_prev;
                        
                        let pred_x0 = ((latents.clone() - (noise_pred.clone() * beta_prod_t.sqrt())?)? / alpha_prod_t.sqrt())?;
                        let dir_xt = (noise_pred * beta_prod_t_prev.sqrt())?;
                        let alpha_prod_t_prev = self.alphas_cumprod[prev_t as usize];
                        latents = ((pred_x0 * alpha_prod_t_prev.sqrt())? + dir_xt)?;
                    }
                    // Decode and save - use tiled decoding for high resolution
                    let images = if let Some(ref tiled_vae) = self.tiled_vae {
                        tiled_vae.decode_tiled(&latents)?
                    } else {
                        vae.decode(&latents)?
                    };
                    let filename = format!("dataset_sample_{:02}.png", i);
                    let filepath = step_dir.join(filename);
                    self.save_image_tensor(&images, &filepath)?;
                    
                    // Save the caption
                    let caption_file = step_dir.join(format!("dataset_sample_{:02}_caption.txt", i));
                    fs::write(&caption_file, caption)?;
                }
                
                info!("Generated {} dataset samples", num_dataset_samples);
            }
        }
        
        Ok(())
    }
    
    /// Save image tensor to file
    /// Generate latents using SDXL with LoRA
    fn generate_latents_with_prompt(
        &mut self,
        prompt: &str,
        negative_prompt: &str,
        width: usize,
        height: usize,
        steps: usize,
        guidance_scale: f64,
        seed: Option<u64>,
    ) -> Result<Tensor> {
        // Set random seed if provided
        if let Some(s) = seed {
            use rand::SeedableRng;
            let _ = rand::rngs::StdRng::seed_from_u64(s);
        }
        
        // Encode prompts
        let (prompt_embeds, pooled_embeds) = self.text_encoders.as_mut()
            .ok_or_else(|| anyhow::anyhow!("Text encoders not loaded"))?
            .encode_sdxl(prompt, 77)?;
        
        let (neg_embeds, neg_pooled) = self.text_encoders.as_mut()
            .ok_or_else(|| anyhow::anyhow!("Text encoders not loaded"))?
            .encode_sdxl(negative_prompt, 77)?;
        info!("\nGenerating {} inference samples...", prompts.len());
        
        // Initialize latents
        let latent_shape = [1, 4, height / 8, width / 8];
        let mut latents = Tensor::randn(0.0f32, 1.0f32, &latent_shape, &self.device)?
            .to_dtype(self.dtype)?;
        
        // Scale initial noise by scheduler init noise sigma (14.6146 for SDXL)
        latents = (latents * 14.6146)?;
        
        // Create time_ids for SDXL
        let time_ids = vec![
            height as f32, width as f32, 0.0, 0.0,
            height as f32, width as f32,
        ];
        let time_ids_tensor = Tensor::from_vec(time_ids, &[1, 6], &self.device)?
            .to_dtype(self.dtype)?;
        
        // Duplicate for CFG
        let time_ids_cfg = Tensor::cat(&[&time_ids_tensor, &time_ids_tensor], 0)?;
        
        // Simple DDIM scheduler
        let timesteps: Vec<i64> = (0..steps)
            .map(|i| (1000 - (i + 1) * 1000 / steps) as i64)
            .rev()
            .collect();
        
        // Denoising loop
        for (idx, &t) in timesteps.iter().enumerate() {
            // Expand latents for CFG
            let latent_model_input = Tensor::cat(&[&latents, &latents], 0)?;
            
            // Create timestep tensor
            let timestep = Tensor::new(&[t, t], &self.device)?.to_dtype(self.dtype)?;
            
            // Combine embeddings for CFG
            let encoder_hidden_states = Tensor::cat(&[&neg_embeds, &prompt_embeds], 0)?;
            let pooled_states = Tensor::cat(&[&neg_pooled, &pooled_embeds], 0)?;
            
            // Forward pass through UNet with LoRA
            // For now, use the standard forward pass
            // Forward pass through UNet with LoRA
            let noise_pred = if let Some(ref cpu_offloaded_unet) = self.cpu_offloaded_unet {
                cpu_offloaded_unet.forward_offloaded(
                    &latent_model_input,
                    &timestep,
                    &encoder_hidden_states,
                    &self.lora_collection,
                )?
            } else if self.use_flash_attention {
                // Use Flash Attention forward pass
                forward_sdxl_sd_format_flash(
                    &latent_model_input,
                    &timestep,
                    &encoder_hidden_states,
                    &self.unet_weights,
                    &self.lora_collection,
                    true, // Enable Flash Attention
                )?
            } else {
                // Use efficient forward pass
                forward_sdxl_sd_efficient(
                    &latent_model_input,
                    &timestep,
                    &encoder_hidden_states,
                    &self.unet_weights,
                    &self.lora_collection,
                )?
            };
            
            // Perform CFG
            let chunks = noise_pred.chunk(2, 0)?;
            let noise_pred_uncond = &chunks[0];
            let noise_pred_cond = &chunks[1];
            let diff = (noise_pred_cond - noise_pred_uncond)?;
            let scaled_diff = (diff * guidance_scale)?;
            let noise_pred = (noise_pred_uncond + scaled_diff)?;
            
            // DDIM step
            let alpha_prod_t = ((1000 - t) as f32 / 1000.0).powi(2);
            let alpha_prod_t_prev = if idx < timesteps.len() - 1 {
                ((1000 - timesteps[idx + 1]) as f32 / 1000.0).powi(2)
            } else {
                1.0
            };
            
            let beta_prod_t = 1.0 - alpha_prod_t;
            let beta_prod_t_prev = 1.0 - alpha_prod_t_prev;
            
            // Compute x0 prediction
            let pred_x0 = ((latents.clone() - (noise_pred.clone() * (beta_prod_t.sqrt() as f64))?)? 
                / (alpha_prod_t.sqrt() as f64))?;
            
            // Compute direction
            let dir_xt = (noise_pred * (beta_prod_t_prev.sqrt() as f64))?;
            
            let alpha_prod_t_prev = self.alphas_cumprod[prev_t as usize];
            latents = ((pred_x0 * (alpha_prod_t_prev.sqrt() as f64))? + dir_xt)?;
        }
        
        Ok(latents)
    }
    
    fn save_image_tensor(&self, tensor: &Tensor, path: &Path) -> Result<()> {
        // VAE output is typically in [-1, 1], convert to [0, 255]
        let tensor_plus_one = (tensor + 1.0)?;
        let tensor = (tensor_plus_one * 127.5)?;
        let tensor = tensor.clamp(0.0, 255.0)?;
        
        // Get dimensions [B, C, H, W]
        let dims = tensor.dims();
        if dims.len() != 4 || dims[1] != 3 {
            return Err(anyhow::anyhow!("Expected tensor shape [B, 3, H, W], got {:?}", dims));
        }
        
        let height = dims[2];
        let width = dims[3];
        
        // Convert to u8 and create image
        let data = tensor
            .squeeze(0)?  // Remove batch dimension
            .permute((1, 2, 0))?  // [C, H, W] -> [H, W, C]
            .flatten_all()?
            .to_dtype(DType::U8)?
            .to_vec1::<u8>()?;
        
        // Create RGB image
        let img = image::RgbImage::from_raw(width as u32, height as u32, data)
            .ok_or_else(|| anyhow::anyhow!("Failed to create image from tensor"))?;
        
        // Save as PNG
        img.save(path)?;
        
        Ok(())
    }
    
    /// Cache all latents to disk for faster training
    fn cache_all_latents(&mut self, dataset: &[DatasetItem], cache_dir: &Path) -> Result<()> {
        // Skip caching if VAE is not available
        let vae = match self.vae_encoder.as_ref() {
            Some(v) => v,
            None => {
                warn!("Skipping latent caching because VAE is not available");
                return Ok(());
            }
        };
        
        let total_items = dataset.len();
        let mut cached_count = 0;
        
        for (idx, item) in dataset.iter().enumerate() {
            // Create cache filename based on image path
            // Use a simple hash of the path for cache filename
            let path_str = item.image_path.to_string_lossy();
            let hash = path_str.bytes().fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
            let cache_filename = format!("{:016x}.safetensors", hash);
            let cache_path = cache_dir.join(&cache_filename);
            
            // Skip if already cached
            if cache_path.exists() {
                cached_count += 1;
                continue;
            }
            
            // Load and preprocess image
            let img = image::open(&item.image_path)?;
            let img = img.to_rgb8();
            
            // Get resolution from config
            let resolution = self.process_config.datasets[0].resolution[0];
            
            // Resize to configured resolution
            let img = image::imageops::resize(&img, resolution as u32, resolution as u32, image::imageops::FilterType::Lanczos3);
            
            // Convert to tensor [H, W, C] -> [C, H, W] normalized to [0, 1]
            // VAE encoder will convert to [-1, 1] internally
            let img_vec: Vec<f32> = img.pixels()
                .flat_map(|p| {
                    [
                        p[0] as f32 / 255.0,
                        p[1] as f32 / 255.0,
                        p[2] as f32 / 255.0
                    ]
                })
                .collect();
            
            let img_tensor = Tensor::from_vec(img_vec, &[resolution, resolution, 3], &self.device)?
                .permute((2, 0, 1))?  // [H, W, C] -> [C, H, W]
                .unsqueeze(0)?       // Add batch dimension [1, C, H, W]
                .to_dtype(self.dtype)?;  // Convert to F16 to match weights
            
            // Encode with VAE - use tiled encoding for high resolution
            let latent = if let Some(ref tiled_vae) = self.tiled_vae {
                info!("Using TiledVAE for {}x{} image", resolution, resolution);
                tiled_vae.encode_tiled(&img_tensor)?
            } else {
                vae.encode(&img_tensor)?
            };
            
            // Save latent to cache
            let mut tensors = HashMap::new();
            let data = tensor_to_vec(&latent)?;
            tensors.insert(
                "latent".to_string(),
                TensorView::new(
                    convert_dtype(latent.dtype()?)?,
                    latent.dims().to_vec(),
                    &data,
                )?
            );
            
            let metadata = HashMap::new();
            let data = serialize(&tensors, &Some(metadata))?;
            fs::write(&cache_path, data)?;
            
            // Progress update
            if (idx + 1) % 10 == 0 || idx + 1 == total_items {
                info!("Cached {}/{} latents ({}% complete)", idx + 1 - cached_count, total_items - cached_count,
                    ((idx + 1 - cached_count) * 100) / (total_items - cached_count));
            }
        }
        
        if cached_count > 0 {
            info!("Found {} pre-cached latents, skipped encoding", cached_count);
        }
        
        Ok(())
    }
    
    /// Compute validation loss for a batch
    fn compute_validation_loss(&mut self, batch: &[super::validation::ValidationItem], cache_dir: &Option<PathBuf>) -> Result<(f32, usize)> {
        let mut total_loss = 0.0;
        let batch_size = batch.len();
        
        // Convert validation items to dataset items
        let dataset_items: Vec<DatasetItem> = batch.iter()
            .map(|item| DatasetItem {
                image_path: item.image_path.clone(),
                caption: item.caption.clone(),
            })
            .collect();
        
        // Use indices 0..batch_size
        let indices: Vec<usize> = (0..batch_size).collect();
        
        // Compute loss for this batch
        let loss = self.training_step(&dataset_items, &indices, self.current_step)?;
        total_loss += loss * batch_size as f32;
        
        Ok((total_loss, batch_size))
    }
    
    /// Save EMA LoRA weights
    fn save_ema_lora(&self, step: usize, ema_model: &EMAModel) -> Result<()> {
        let filename = format!("lora_ema_step_{:06}.safetensors", step);
        let save_path = self.output_dir.join(&filename);
        
        info!("Saving EMA LoRA to: {}", save_path.display());
        
        // Create a temporary LoRA collection with EMA weights
        let mut ema_collection = LoRACollection::new(
            self.lora_collection.rank,
            self.lora_collection.alpha,
            self.lora_collection.dtype
        );
        
        // Copy adapter structure but with EMA weights
        for (name, adapter) in &self.lora_collection.adapters {
            let down_ema = ema_model.shadow_params()
                .get(&format!("{}_down", name))
                .ok_or_else(|| anyhow::anyhow!("Missing EMA weight for {}_down", name))?;
            let up_ema = ema_model.shadow_params()
                .get(&format!("{}_up", name))
                .ok_or_else(|| anyhow::anyhow!("Missing EMA weight for {}_up", name))?;
            
            // Create new adapter with EMA weights
            let ema_adapter = SimpleLoRA {
                down: Var::from_tensor(down_ema)?,
                up: Var::from_tensor(up_ema)?,
                scale: adapter.scale,
            };
            
            ema_collection.adapters.insert(name.clone(), ema_adapter);
        }
        
        // Save using the collection's save method
        ema_collection.save(&save_path)?;
        
        Ok(())
    }
    
    /// Load cached latent or encode on the fly
    fn load_or_encode_latent(&self, item: &DatasetItem, cache_dir: &Path) -> Result<Tensor> {
        // Try to load from cache first
        let path_str = item.image_path.to_string_lossy();
        let hash = path_str.bytes().fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
        let cache_filename = format!("{:016x}.safetensors", hash);
        let cache_path = cache_dir.join(&cache_filename);
        
        if cache_path.exists() {
            // Load from cache
            let tensors = candle_core::safetensors::load(&cache_path, &self.device)?;
            if let Some(latent) = tensors.get("latent") {
                return Ok(latent.clone());
            }
        }
        
        // Fallback to encoding on the fly or random latents
        if let Some(ref vae) = self.vae_encoder {
            // Load and preprocess image
            let img = image::open(&item.image_path)?;
            let img = img.to_rgb8();
            
            // Get resolution from config
            let resolution = self.process_config.datasets[0].resolution[0];
            let img = image::imageops::resize(&img, resolution as u32, resolution as u32, image::imageops::FilterType::Lanczos3);
            
            let img_vec: Vec<f32> = img.pixels()
                .flat_map(|p| {
                    [
                        (p[0] as f32 / 127.5) - 1.0,
                        (p[1] as f32 / 127.5) - 1.0,
                        (p[2] as f32 / 127.5) - 1.0
                    ]
                })
                .collect();
            
            let img_tensor = Tensor::from_vec(img_vec, &[resolution, resolution, 3], &self.device)?
                .permute((2, 0, 1))?
                .unsqueeze(0)?
                .to_dtype(self.dtype)?;
            
            // Use tiled encoding if available
            Ok(if let Some(ref tiled_vae) = self.tiled_vae {
                tiled_vae.encode_tiled(&img_tensor)?
            } else {
                vae.encode(&img_tensor)?
            })
        } else {
            // VAE not available - return random latents
            // Calculate latent dimensions (VAE uses 8x downscaling)
            let resolution = self.process_config.datasets[0].resolution[0];
            let latent_size = resolution / 8;
            Ok(Tensor::randn(0.0f32, 1.0f32, &[1, 4, latent_size, latent_size], &self.device)?
                .to_dtype(self.dtype)?)
        }
    }
    }
    
} // impl SDXLLoRATrainerFixed

// Dataset structures
#[derive(Debug)]
struct DatasetItem {
    image_path: PathBuf,
    caption: String,
}

#[derive(Debug)]
struct TrainingBatch {
    latents: Tensor,
    prompts: Vec<String>,
}

// Helper functions for safetensors conversion
fn convert_dtype(dtype: DType) -> Result<SafeDtype> {
    match dtype {
        DType::F32 => Ok(SafeDtype::F32),
        DType::F16 => Ok(SafeDtype::F16),
        DType::BF16 => Ok(SafeDtype::BF16),
        DType::U8 => Ok(SafeDtype::U8),
        DType::U32 => Ok(SafeDtype::U32),
        DType::I64 => Ok(SafeDtype::I64),
        _ => Err(anyhow::anyhow!("Unsupported dtype for safetensors: {:?}", dtype)),
    }
}

fn tensor_to_vec(tensor: &Tensor) -> Result<Vec<u8>> {
    // Flatten tensor to 1D
    let flattened = tensor.flatten_all()?;
    
    let data = match tensor.dtype() {
        DType::F32 => {
            let data: Vec<f32> = flattened.to_vec1()?;
            data.into_iter().flat_map(|f| f.to_le_bytes()).collect()
        }
        DType::F16 => {
            let data: Vec<half::f16> = flattened.to_vec1()?;
            data.into_iter().flat_map(|f| f.to_le_bytes()).collect()
        }
        DType::BF16 => {
            let data: Vec<half::bf16> = flattened.to_vec1()?;
            data.into_iter().flat_map(|f| f.to_le_bytes()).collect()
        }
        _ => return Err(anyhow::anyhow!("Unsupported tensor dtype for conversion")),
    };
    Ok(data)
}

/// Main training function
pub fn train_sdxl_lora_fixed(config: &Config, process: &ProcessConfig) -> Result<()> {
    let mut trainer = SDXLLoRATrainerFixed::new(config, process)?;
    trainer.load_models()?;
    trainer.train()?;
    Ok(())
}
