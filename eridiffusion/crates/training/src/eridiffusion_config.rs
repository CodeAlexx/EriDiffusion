//! AI-Toolkit compatible configuration parser for Flux training
//! Parses the exact same YAML format as the Python version

use serde::{Deserialize, Serialize};
use serde_yaml::Value;
use std::path::{Path, PathBuf};
use std::collections::HashMap;
use eridiffusion_core::{Result, Error};

/// Root configuration structure matching AI-Toolkit format
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AIToolkitConfig {
    pub job: String,
    pub config: JobConfig,
    #[serde(default)]
    pub meta: Option<HashMap<String, Value>>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct JobConfig {
    pub name: String,
    pub process: Vec<ProcessConfig>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type")]
pub enum ProcessConfig {
    #[serde(rename = "sd_trainer")]
    SDTrainer(TrainerConfig),
}

/// Main trainer configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TrainerConfig {
    pub training_folder: PathBuf,
    #[serde(default)]
    pub performance_log_every: Option<usize>,
    pub device: String,
    #[serde(default)]
    pub trigger_word: Option<String>,
    pub network: NetworkConfig,
    pub save: SaveConfig,
    pub datasets: Vec<DatasetConfig>,
    pub train: TrainConfig,
    pub model: ModelConfig,
    pub sample: SampleConfig,
}

/// Network configuration (LoRA, DoRA, etc.)
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type")]
pub enum NetworkConfig {
    #[serde(rename = "lora")]
    LoRA {
        linear: u32,
        linear_alpha: u32,
        #[serde(default)]
        conv: Option<u32>,
        #[serde(default)]
        conv_alpha: Option<u32>,
    },
    #[serde(rename = "dora")]
    DoRA {
        linear: u32,
        linear_alpha: u32,
    },
}

/// Save configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SaveConfig {
    pub dtype: String, // float16, bfloat16, float32
    pub save_every: usize,
    #[serde(default = "default_max_saves")]
    pub max_step_saves_to_keep: usize,
    #[serde(default)]
    pub push_to_hub: bool,
    #[serde(default)]
    pub hf_repo_id: Option<String>,
    #[serde(default)]
    pub hf_private: Option<bool>,
}

fn default_max_saves() -> usize { 4 }

/// Dataset configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DatasetConfig {
    pub folder_path: PathBuf,
    #[serde(default = "default_caption_ext")]
    pub caption_ext: String,
    #[serde(default)]
    pub caption_dropout_rate: f32,
    #[serde(default)]
    pub shuffle_tokens: bool,
    #[serde(default = "default_true")]
    pub cache_latents_to_disk: bool,
    pub resolution: ResolutionConfig,
}

fn default_caption_ext() -> String { "txt".to_string() }
fn default_true() -> bool { true }

/// Resolution configuration - can be single value or array
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum ResolutionConfig {
    Single(u32),
    Multiple(Vec<u32>),
}

/// Training configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TrainConfig {
    pub batch_size: usize,
    pub steps: usize,
    #[serde(default = "default_grad_accum")]
    pub gradient_accumulation_steps: usize,
    #[serde(default = "default_true")]
    pub train_unet: bool,
    #[serde(default)]
    pub train_text_encoder: bool,
    #[serde(default = "default_true")]
    pub gradient_checkpointing: bool,
    pub noise_scheduler: String, // flowmatch, ddpm, etc.
    pub optimizer: String, // adamw8bit, adamw, prodigy, etc.
    pub lr: f64,
    #[serde(default)]
    pub skip_first_sample: bool,
    #[serde(default)]
    pub disable_sampling: bool,
    #[serde(default)]
    pub linear_timesteps: bool,
    #[serde(default)]
    pub ema_config: Option<EMAConfig>,
    pub dtype: String, // bf16, fp16, fp32
}

fn default_grad_accum() -> usize { 1 }

/// EMA configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EMAConfig {
    pub use_ema: bool,
    #[serde(default = "default_ema_decay")]
    pub ema_decay: f32,
}

fn default_ema_decay() -> f32 { 0.999 }

/// Model configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelConfig {
    pub name_or_path: String,
    #[serde(default)]
    pub is_flux: bool,
    #[serde(default)]
    pub is_v3: bool,
    #[serde(default)]
    pub is_sd3: bool,
    #[serde(default)]
    pub is_auraflow: bool,
    #[serde(default)]
    pub quantize: bool,
    #[serde(default)]
    pub low_vram: bool,
}

/// Sample configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SampleConfig {
    pub sampler: String,
    pub sample_every: usize,
    pub width: u32,
    pub height: u32,
    pub prompts: Vec<String>,
    #[serde(default)]
    pub neg: String,
    #[serde(default = "default_seed")]
    pub seed: u64,
    #[serde(default)]
    pub walk_seed: bool,
    pub guidance_scale: f32,
    pub sample_steps: usize,
}

fn default_seed() -> u64 { 42 }

impl AIToolkitConfig {
    /// Load configuration from YAML file
    pub fn from_yaml_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path.as_ref())?;
        Self::from_yaml_str(&content)
    }
    
    /// Parse configuration from YAML string
    pub fn from_yaml_str(yaml: &str) -> Result<Self> {
        serde_yaml::from_str(yaml)
            .map_err(|e| Error::Config(format!("Failed to parse YAML: {}", e)))
    }
    
    /// Get the trainer configuration (assumes single process)
    pub fn get_trainer_config(&self) -> Result<&TrainerConfig> {
        match self.config.process.first() {
            Some(ProcessConfig::SDTrainer(config)) => Ok(config),
            _ => Err(Error::Config("No sd_trainer process found".into())),
        }
    }
    
    /// Convert to our internal Flux training config
    pub fn to_flux_config(&self) -> Result<crate::flux_trainer_24gb::FluxTraining24GBConfig> {
        let trainer = self.get_trainer_config()?;
        
        // Parse device
        let device_id = if trainer.device.starts_with("cuda:") {
            trainer.device.strip_prefix("cuda:")
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(0)
        } else {
            0
        };
        
        // Check if cache_latents_to_disk is enabled
        let use_cache = trainer.datasets.iter()
            .all(|d| d.cache_latents_to_disk);
        
        if !use_cache {
            return Err(Error::Config(
                "cache_latents_to_disk must be true for 24GB training".into()
            ));
        }
        
        // Get cache directory
        let cache_dir = trainer.training_folder.join(&self.config.name).join("cache");
        
        // Get model path - this would need to be resolved from name_or_path
        let model_path = self.resolve_model_path(&trainer.model.name_or_path)?;
        
        Ok(crate::flux_trainer_24gb::FluxTraining24GBConfig {
            model_path,
            cache_dir,
            output_dir: trainer.training_folder.join(&self.config.name),
            learning_rate: trainer.train.lr,
            batch_size: trainer.train.batch_size,
            gradient_accumulation_steps: trainer.train.gradient_accumulation_steps,
            num_train_steps: trainer.train.steps,
            gradient_checkpointing: trainer.train.gradient_checkpointing,
            mixed_precision: matches!(trainer.train.dtype.as_str(), "bf16" | "fp16"),
            ema_decay: trainer.train.ema_config
                .as_ref()
                .filter(|e| e.use_ema)
                .map(|e| e.ema_decay)
                .unwrap_or(0.0),
            save_every: trainer.save.save_every,
            log_every: trainer.performance_log_every.unwrap_or(10),
            max_grad_norm: 1.0, // Default, could be added to config
        })
    }
    
    /// Get LoRA configuration
    pub fn get_lora_config(&self) -> Result<(u32, u32)> {
        let trainer = self.get_trainer_config()?;
        match &trainer.network {
            NetworkConfig::LoRA { linear, linear_alpha, .. } => Ok((*linear, *linear_alpha)),
            NetworkConfig::DoRA { linear, linear_alpha } => Ok((*linear, *linear_alpha)),
        }
    }
    
    /// Resolve model path from name or local path
    fn resolve_model_path(&self, name_or_path: &str) -> Result<PathBuf> {
        // Check if it's a local path
        let path = Path::new(name_or_path);
        if path.exists() {
            return Ok(path.to_path_buf());
        }
        
        // Check common model locations
        let home = std::env::var("HOME").unwrap_or_else(|_| "/home/alex".to_string());
        let model_dirs = [
            format!("{}/SwarmUI/Models/unet", home),
            format!("{}/models", home),
            format!("{}/ComfyUI/models/unet", home),
        ];
        
        // Map HuggingFace names to local files
        let model_map = [
            ("black-forest-labs/FLUX.1-dev", "flux1-dev.safetensors"),
            ("black-forest-labs/FLUX.1-schnell", "flux1-schnell.safetensors"),
        ];
        
        // Try to find the model
        for (hf_name, filename) in &model_map {
            if name_or_path == *hf_name {
                for dir in &model_dirs {
                    let full_path = PathBuf::from(dir).join(filename);
                    if full_path.exists() {
                        return Ok(full_path);
                    }
                }
            }
        }
        
        Err(Error::Config(format!("Model not found: {}", name_or_path)))
    }
    
    /// Replace template variables in strings
    pub fn process_templates(&mut self) {
        let name = self.config.name.clone();
        
        // Process meta
        if let Some(meta) = &mut self.meta {
            for value in meta.values_mut() {
                if let Value::String(s) = value {
                    *s = s.replace("[name]", &name);
                }
            }
        }
        
        // Process prompts
        if let Ok(trainer) = self.get_trainer_config() {
            if let Some(trigger) = &trainer.trigger_word {
                // Note: In actual implementation, we'd need mutable access
                // to replace [trigger] in prompts
            }
        }
    }
}

/// Dataset preprocessing status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessingStatus {
    pub total_images: usize,
    pub processed_images: usize,
    pub cache_dir: PathBuf,
    pub is_complete: bool,
}

impl PreprocessingStatus {
    /// Check if preprocessing is needed
    pub fn check_cache_status(cache_dir: &Path, dataset_dir: &Path) -> Result<Self> {
        // Count images in dataset
        let total_images = std::fs::read_dir(dataset_dir)?
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path().extension()
                    .and_then(|ext| ext.to_str())
                    .map(|ext| matches!(ext.to_lowercase().as_str(), "jpg" | "jpeg" | "png"))
                    .unwrap_or(false)
            })
            .count();
        
        // Count processed files in cache
        let processed_images = if cache_dir.exists() {
            std::fs::read_dir(cache_dir)?
                .filter_map(|e| e.ok())
                .filter(|e| e.path().extension()
                    .map(|ext| ext == "safetensors")
                    .unwrap_or(false))
                .count() / 3 // Each image has 3 files (latents, t5, clip)
        } else {
            0
        };
        
        Ok(Self {
            total_images,
            processed_images,
            cache_dir: cache_dir.to_path_buf(),
            is_complete: processed_images >= total_images && total_images > 0,
        })
    }
}

/// Example usage
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_parse_flux_config() {
        let yaml = std::fs::read_to_string("/home/alex/diffusers-rs/config/train_lora_flux_24gb.yaml")
            .expect("Failed to read config");
        
        let config = AIToolkitConfig::from_yaml_str(&yaml)
            .expect("Failed to parse config");
        
        assert_eq!(config.job, "extension");
        assert_eq!(config.config.name, "my_first_flux_lora_v1");
        
        let trainer = config.get_trainer_config().unwrap();
        assert_eq!(trainer.device, "cuda:0");
        assert_eq!(trainer.train.batch_size, 1);
        assert_eq!(trainer.train.steps, 2000);
        
        // Check LoRA config
        let (rank, alpha) = config.get_lora_config().unwrap();
        assert_eq!(rank, 16);
        assert_eq!(alpha, 16);
    }
}