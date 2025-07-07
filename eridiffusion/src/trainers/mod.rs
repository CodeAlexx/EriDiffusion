use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::fs;

pub mod sd35_lokr;
pub mod rms_norm_patch;
pub mod mmdit_loader;
pub mod cuda_rms_norm;
pub mod rms_norm_fix;
pub mod mmdit_patch;
pub mod text_encoders;
pub mod text_encoder_cached;
pub mod simple_text_encoder;
pub mod embedded_tokenizers;
pub mod proper_text_encoder;
pub mod real_tokenizers;
pub mod sampling;
pub mod flux_lora;
pub mod flux_data_loader;
pub mod flux_sampling;
pub mod flux_lora_cpu;
pub mod flux_int8_loader;
pub mod flux_quantized_loader;
pub mod quanto_var_builder;
pub mod device_fix;
pub mod force_device_zero;
pub mod single_device_enforcer;
pub mod cached_device;
pub mod flux_single_pass_loader;
pub mod flux_lazy_loader;
pub mod flux_lazy_layers;
pub mod flux_cpu_offload;
pub mod flux_incremental_loader;
pub mod flux_memory_efficient;
pub mod flux_cpu_offloaded_model;
pub mod flux_selective_loader;
pub mod device_debug;
pub mod flux_init_weights;
pub mod flux_efficient_loader;
pub mod flux_layerwise_offload;
pub mod gradient_checkpointing;
pub mod optimizer_cpu_offload;
pub mod flux_lora_only_loader;

// Export function is defined below, no need to re-export

#[derive(Debug, Deserialize, Serialize)]
struct Config {
    job: String,
    config: JobConfig,
}

#[derive(Debug, Deserialize, Serialize)]
struct JobConfig {
    name: String,
    process: Vec<ProcessConfig>,
}

#[derive(Debug, Deserialize, Serialize)]
struct ProcessConfig {
    #[serde(rename = "type")]
    process_type: String,
    device: Option<String>,
    trigger_word: Option<String>,
    network: NetworkConfig,
    save: SaveConfig,
    datasets: Vec<DatasetConfig>,
    train: TrainConfig,
    model: ModelConfig,
    sample: Option<SampleConfig>,
}

#[derive(Debug, Deserialize, Serialize)]
struct NetworkConfig {
    #[serde(rename = "type")]
    network_type: String,
    linear: Option<usize>,
    linear_alpha: Option<f32>,
    lokr_factor: Option<usize>,
    lokr_full_rank: Option<bool>,
}

#[derive(Debug, Deserialize, Serialize)]
struct SaveConfig {
    dtype: String,
    save_every: usize,
    max_step_saves_to_keep: Option<usize>,
}

#[derive(Debug, Deserialize, Serialize)]
struct DatasetConfig {
    folder_path: String,
    caption_ext: String,
    caption_dropout_rate: Option<f32>,
    shuffle_tokens: Option<bool>,
    cache_latents_to_disk: Option<bool>,
    resolution: Vec<usize>,
}

#[derive(Debug, Deserialize, Serialize)]
struct TrainConfig {
    batch_size: usize,
    steps: usize,
    gradient_accumulation: Option<usize>,
    train_unet: Option<bool>,
    train_text_encoder: Option<bool>,
    gradient_checkpointing: Option<bool>,
    noise_scheduler: String,
    optimizer: String,
    lr: f32,
    linear_timesteps: Option<bool>,
    bypass_guidance_embedding: Option<bool>,
    dtype: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct ModelConfig {
    name_or_path: String,
    is_v3: Option<bool>,
    is_flux: Option<bool>,
    quantize: Option<bool>,
    max_grad_norm: Option<f32>,
    t5_max_length: Option<usize>,
    snr_gamma: Option<f32>,
    // Text encoder paths - if not provided, will try to auto-detect
    clip_l_path: Option<String>,
    clip_g_path: Option<String>,
    t5_path: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
struct SampleConfig {
    sampler: String,
    sample_every: usize,
    width: usize,
    height: usize,
    prompts: Vec<String>,
    neg: Option<String>,
    seed: Option<u64>,
    guidance_scale: Option<f32>,
    sample_steps: Option<usize>,
}

#[derive(Debug)]
enum ModelType {
    SD35,
    SDXL,
    Flux,
    SD15,
    SD21,
}

fn detect_model_type(config: &ProcessConfig) -> Result<ModelType> {
    // First check explicit flags
    if config.model.is_v3.unwrap_or(false) {
        return Ok(ModelType::SD35);
    }
    
    if config.model.is_flux.unwrap_or(false) {
        return Ok(ModelType::Flux);
    }
    
    // Check by model path/name
    let model_path = &config.model.name_or_path.to_lowercase();
    
    if model_path.contains("sd3.5") || model_path.contains("sd35") || model_path.contains("sd_3.5") {
        return Ok(ModelType::SD35);
    }
    
    if model_path.contains("flux") {
        return Ok(ModelType::Flux);
    }
    
    if model_path.contains("sdxl") || model_path.contains("sd_xl") {
        return Ok(ModelType::SDXL);
    }
    
    if model_path.contains("sd2") || model_path.contains("v2") {
        return Ok(ModelType::SD21);
    }
    
    if model_path.contains("sd1") || model_path.contains("v1-5") || model_path.contains("v1.5") {
        return Ok(ModelType::SD15);
    }
    
    // Check by training config hints
    if config.train.linear_timesteps.unwrap_or(false) {
        // SD3.5 uses linear timesteps
        return Ok(ModelType::SD35);
    }
    
    if config.train.bypass_guidance_embedding.unwrap_or(false) {
        // Flux uses bypass_guidance_embedding
        return Ok(ModelType::Flux);
    }
    
    // Default to SDXL as it's the most common
    println!("Warning: Could not determine model type from config, defaulting to SDXL");
    Ok(ModelType::SDXL)
}

pub fn train_from_config(config_path: PathBuf) -> Result<()> {
    // Load YAML configuration
    let config_str = fs::read_to_string(&config_path)
        .with_context(|| format!("Failed to read config file: {}", config_path.display()))?;
    
    let config: Config = serde_yaml::from_str(&config_str)
        .with_context(|| "Failed to parse YAML config")?;
    
    // Find the sd_trainer process config
    let process_config = config.config.process
        .iter()
        .find(|p| p.process_type == "sd_trainer")
        .ok_or_else(|| anyhow::anyhow!("No 'sd_trainer' process found in config"))?;
    
    // Detect model type
    let model_type = detect_model_type(process_config)?;
    
    println!("\nDetected model type: {:?}", model_type);
    println!("Model path: {}", process_config.model.name_or_path);
    println!("Network type: {}", process_config.network.network_type);
    
    // Route to appropriate trainer
    match model_type {
        ModelType::SD35 => {
            println!("\nStarting SD 3.5 training...");
            match process_config.network.network_type.as_str() {
                "lokr" | "lokr_full_rank" => {
                    sd35_lokr::train_sd35_lokr(&config, process_config)?;
                }
                "lora" => {
                    // For now, treat LoRA as LoKr with factor=1
                    println!("Note: Using LoKr implementation for LoRA (factor=1)");
                    sd35_lokr::train_sd35_lokr(&config, process_config)?;
                }
                _ => {
                    return Err(anyhow::anyhow!(
                        "Unsupported network type '{}' for SD 3.5", 
                        process_config.network.network_type
                    ));
                }
            }
        }
        ModelType::SDXL => {
            println!("\nStarting SDXL training...");
            match process_config.network.network_type.as_str() {
                "lora" | "lokr" | "lokr_full_rank" => {
                    // Use SD3.5 trainer for SDXL temporarily
                    println!("Note: Using SD3.5 LoKr implementation for SDXL");
                    sd35_lokr::train_sd35_lokr(&config, process_config)?;
                }
                _ => {
                    return Err(anyhow::anyhow!(
                        "Unsupported network type '{}' for SDXL", 
                        process_config.network.network_type
                    ));
                }
            }
        }
        ModelType::Flux => {
            println!("\nStarting Flux training...");
            match process_config.network.network_type.as_str() {
                "lora" => {
                    flux_lora::train_flux_lora(&config, process_config)?;
                }
                "lokr" | "lokr_full_rank" => {
                    println!("Note: LoKr not yet implemented for Flux, using LoRA instead");
                    flux_lora::train_flux_lora(&config, process_config)?;
                }
                _ => {
                    return Err(anyhow::anyhow!(
                        "Unsupported network type '{}' for Flux", 
                        process_config.network.network_type
                    ));
                }
            }
        }
        ModelType::SD15 | ModelType::SD21 => {
            return Err(anyhow::anyhow!(
                "SD 1.5/2.1 training not yet implemented in unified trainer. Use legacy diffusers-rs."
            ));
        }
    }
    
    println!("\n=== Training Complete ===");
    Ok(())
}