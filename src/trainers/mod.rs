pub mod sdxl_lora_trainer_fixed;
pub mod sdxl_sampling_complete;
pub mod sd35_lora;
pub mod flux_lora;
pub mod flux_sampling;
pub mod flux_data_loader;
pub mod device_debug;
pub mod text_encoders;
pub mod adam8bit;
pub mod ddpm_scheduler;
pub mod enhanced_data_loader;
pub mod sdxl_forward_with_lora;
pub mod sdxl_forward_sampling;
pub mod sdxl_vae_native;
pub mod sdxl_vae_wrapper;
pub mod memory_utils;
pub mod sampling_utils;

// Re-export key types
pub use sdxl_lora_trainer_fixed::SDXLLoRATrainerFixed;
pub use sd35_lora::SD35LoRATrainer;
pub use flux_lora::FluxLoRATrainer;
pub use sdxl_sampling_complete::{SDXLSampler, TrainingSampler, SDXLSamplingConfig, SchedulerType};
pub use text_encoders::TextEncoders;
pub use adam8bit::Adam8bit;

use anyhow::{Result, Context};
use serde::{Serialize, Deserialize};
use std::path::PathBuf;
use std::fs;
use candle_core;

#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    pub job: String,
    pub config: ConfigData,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub meta: Option<MetaConfig>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ConfigData {
    pub name: Option<String>,
    pub process: Vec<ProcessConfig>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ProcessConfig {
    #[serde(rename = "type")]
    pub process_type: Option<String>,
    pub device: Option<String>,
    pub low_vram: Option<bool>,
    pub trigger_word: Option<String>,
    pub model: ModelConfig,
    pub network: NetworkConfig,
    pub save: SaveConfig,
    pub datasets: Vec<DatasetConfig>,
    pub train: TrainConfig,
    pub sample: Option<SampleConfig>,
    pub logging: Option<LoggingConfig>,
    pub advanced: Option<AdvancedConfig>,
    pub validation: Option<ValidationConfig>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelConfig {
    pub name_or_path: String,
    pub is_sdxl: Option<bool>,
    pub is_flux: Option<bool>,
    pub is_v3: Option<bool>,
    pub vae_path: Option<String>,
    pub text_encoder_path: Option<String>,
    pub text_encoder_2_path: Option<String>,
    pub snr_gamma: Option<f32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct NetworkConfig {
    #[serde(rename = "type")]
    pub type_: String,
    pub linear: Option<usize>,
    pub linear_alpha: Option<usize>,
    pub conv: Option<usize>,
    pub conv_alpha: Option<usize>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SaveConfig {
    pub dtype: String,
    pub save_every: usize,
    pub max_step_saves_to_keep: usize,
    pub push_to_hub: Option<bool>,
    pub hf_repo_id: Option<String>,
    pub hf_private: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DatasetConfig {
    pub folder_path: String,
    pub caption_ext: String,
    pub caption_dropout_rate: f32,
    pub shuffle_tokens: bool,
    pub cache_latents_to_disk: bool,
    pub resolution: Vec<usize>,
    pub duplicate_threshold: Option<f32>,
    pub use_enhanced_loader: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TrainConfig {
    pub batch_size: usize,
    pub steps: usize,
    pub gradient_accumulation: usize,
    pub train_unet: bool,
    pub train_text_encoder: bool,
    pub gradient_checkpointing: bool,
    pub noise_scheduler: String,
    pub optimizer: String,
    pub lr: f32,
    pub lr_scheduler: Option<String>,
    pub lr_scheduler_num_cycles: Option<usize>,
    pub lr_warmup_steps: Option<usize>,
    pub dtype: String,
    pub xformers: Option<bool>,
    pub min_snr_gamma: Option<f32>,
    pub max_grad_norm: Option<f32>,
    pub seed: Option<u64>,
    pub cpu_offload: Option<bool>,
    pub ema_decay: Option<f32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SampleConfig {
    pub sampler: String,
    pub sample_every: usize,
    pub sample_steps: usize,
    pub guidance_scale: f32,
    pub prompts: Vec<String>,
    pub neg: Option<String>,
    pub width: usize,
    pub height: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LoggingConfig {
    pub log_every: usize,
    pub log_grad_norm: bool,
    pub use_wandb: bool,
    pub wandb_project: Option<String>,
    pub wandb_run_name: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AdvancedConfig {
    pub vae_tiling: bool,
    pub vae_tile_size: usize,
    pub mixed_precision: String,
    pub empty_cache_steps: usize,
    pub lora_bias: String,
    pub lora_dropout: f32,
    pub attention_mode: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ValidationConfig {
    pub prompts: Vec<String>,
    pub every_n_steps: usize,
    pub batch_size: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MetaConfig {
    pub author: String,
    pub version: String,
    pub description: String,
}

pub fn load_config(path: &PathBuf) -> Result<Config> {
    let config_str = fs::read_to_string(path)
        .with_context(|| format!("Failed to read config file: {}", path.display()))?;
    
    let config: Config = serde_yaml::from_str(&config_str)
        .with_context(|| "Failed to parse YAML config")?;
    
    Ok(config)
}

#[derive(Debug)]
enum ModelType {
    SDXL,
    SD35,
    Flux,
}

fn detect_model_type(process_config: &ProcessConfig) -> Result<ModelType> {
    // Check explicit flags first
    if process_config.model.is_sdxl.unwrap_or(false) {
        return Ok(ModelType::SDXL);
    }
    if process_config.model.is_v3.unwrap_or(false) {
        return Ok(ModelType::SD35);
    }
    if process_config.model.is_flux.unwrap_or(false) {
        return Ok(ModelType::Flux);
    }
    
    // Check by model path/name
    let model_path = process_config.model.name_or_path.to_lowercase();
    if model_path.contains("sdxl") || model_path.contains("sd_xl") {
        return Ok(ModelType::SDXL);
    }
    if model_path.contains("sd3") || model_path.contains("sd35") || model_path.contains("sd_3") {
        return Ok(ModelType::SD35);
    }
    if model_path.contains("flux") {
        return Ok(ModelType::Flux);
    }
    
    // Default to SDXL
    println!("Warning: Could not determine model type from config, defaulting to SDXL");
    Ok(ModelType::SDXL)
}

pub fn train_from_config(config_path: PathBuf) -> Result<()> {
    // Check for GPU requirement first
    let device = candle_core::Device::cuda_if_available(0)?;
    if !device.is_cuda() {
        eprintln!("ERROR: GPU is required for training. No CUDA device found.");
        eprintln!("This trainer follows industry standards and requires a CUDA-capable GPU.");
        eprintln!("CPU training is not supported.");
        return Err(anyhow::anyhow!("Training requires a CUDA GPU. CPU training is not supported."));
    }
    println!("GPU detected and verified for training");
    
    // Load YAML configuration
    let config_str = fs::read_to_string(&config_path)
        .with_context(|| format!("Failed to read config file: {}", config_path.display()))?;
    
    let config: Config = serde_yaml::from_str(&config_str)
        .with_context(|| "Failed to parse YAML config")?;
    
    // Find the sd_trainer process config
    let process_config = config.config.process
        .iter()
        .find(|p| p.process_type == Some("sd_trainer".to_string()))
        .ok_or_else(|| anyhow::anyhow!("No 'sd_trainer' process found in config"))?;
    
    // Detect model type
    let model_type = detect_model_type(process_config)?;
    
    println!("\nDetected model type: {:?}", model_type);
    println!("Model path: {}", process_config.model.name_or_path);
    println!("Network type: {}", process_config.network.type_);
    
    // Route to appropriate trainer
    match model_type {
        ModelType::SDXL => {
            println!("\nStarting SDXL training...");
            match process_config.network.type_.as_str() {
                "lora" => {
                    let mut trainer = SDXLLoRATrainerFixed::new(&config, process_config)?;
                    trainer.load_models()?;
                    trainer.train()?;
                }
                _ => {
                    return Err(anyhow::anyhow!(
                        "Unsupported network type '{}' for SDXL", 
                        process_config.network.type_
                    ));
                }
            }
        }
        ModelType::SD35 => {
            println!("\nStarting SD 3.5 training...");
            match process_config.network.type_.as_str() {
                "lora" => {
                    let mut trainer = SD35LoRATrainer::new(&config, process_config)?;
                    trainer.train()?;
                }
                _ => {
                    return Err(anyhow::anyhow!(
                        "Unsupported network type '{}' for SD 3.5", 
                        process_config.network.type_
                    ));
                }
            }
        }
        ModelType::Flux => {
            println!("\nStarting Flux training...");
            match process_config.network.type_.as_str() {
                "lora" => {
                    let mut trainer = FluxLoRATrainer::new(&config, process_config)?;
                    trainer.train()?;
                }
                _ => {
                    return Err(anyhow::anyhow!(
                        "Unsupported network type '{}' for Flux", 
                        process_config.network.type_
                    ));
                }
            }
        }
    }
    
    println!("\nTraining completed successfully!");
    Ok(())
}