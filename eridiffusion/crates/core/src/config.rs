//! Configuration system for eridiffusion

use crate::{Result, Error, ModelArchitecture, NetworkType};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Base configuration trait
pub trait Config: Serialize + for<'de> Deserialize<'de> + Send + Sync {
    /// Validate the configuration
    fn validate(&self) -> Result<()>;
    
    /// Merge with another configuration
    fn merge(&mut self, other: &Value) -> Result<()>;
    
    /// Get as JSON value
    fn as_value(&self) -> Result<Value> {
        serde_json::to_value(self)
            .map_err(|e| Error::Config(format!("Failed to serialize config: {}", e)))
    }
}

/// Configuration loader
pub struct ConfigLoader {
    search_paths: Vec<PathBuf>,
    environment_prefix: String,
}

impl ConfigLoader {
    /// Create a new config loader
    pub fn new() -> Self {
        Self {
            search_paths: vec![
                PathBuf::from("."),
                PathBuf::from("config"),
                dirs::config_dir()
                    .unwrap_or_default()
                    .join("eridiffusion"),
            ],
            environment_prefix: "AI_TOOLKIT".to_string(),
        }
    }
    
    /// Add a search path
    pub fn add_search_path(&mut self, path: PathBuf) {
        self.search_paths.push(path);
    }
    
    /// Load configuration from file
    pub fn load<T: Config>(&self, filename: &str) -> Result<T> {
        // Try to find the file in search paths
        let file_path = self.find_config_file(filename)?;
        
        // Read the file
        let content = std::fs::read_to_string(&file_path)
            .map_err(|e| Error::Config(format!("Failed to read config file: {}", e)))?;
        
        // Parse based on extension
        let config: T = match file_path.extension().and_then(|s| s.to_str()) {
            Some("yaml") | Some("yml") => serde_yaml::from_str(&content)
                .map_err(|e| Error::Config(format!("Failed to parse YAML: {}", e)))?,
            Some("json") => serde_json::from_str(&content)
                .map_err(|e| Error::Config(format!("Failed to parse JSON: {}", e)))?,
            Some("toml") => toml::from_str(&content)
                .map_err(|e| Error::Config(format!("Failed to parse TOML: {}", e)))?,
            _ => return Err(Error::Config("Unsupported config format".to_string())),
        };
        
        // Apply environment overrides
        let config = self.apply_env_overrides(config)?;
        
        // Validate
        config.validate()?;
        
        Ok(config)
    }
    
    /// Find config file in search paths
    fn find_config_file(&self, filename: &str) -> Result<PathBuf> {
        for path in &self.search_paths {
            let file_path = path.join(filename);
            if file_path.exists() {
                return Ok(file_path);
            }
            
            // Try with common extensions
            for ext in &["yaml", "yml", "json", "toml"] {
                let file_path = path.join(format!("{}.{}", filename, ext));
                if file_path.exists() {
                    return Ok(file_path);
                }
            }
        }
        
        Err(Error::Config(format!("Config file not found: {}", filename)))
    }
    
    /// Apply environment variable overrides
    fn apply_env_overrides<T: Config>(&self, mut config: T) -> Result<T> {
        let mut value = config.as_value()?;
        
        // Collect environment variables with our prefix
        for (key, env_value) in std::env::vars() {
            if key.starts_with(&self.environment_prefix) {
                let path = key[self.environment_prefix.len()..]
                    .trim_start_matches('_')
                    .to_lowercase()
                    .replace('_', ".");
                
                self.set_value_at_path(&mut value, &path, &env_value)?;
            }
        }
        
        // Deserialize back
        serde_json::from_value(value)
            .map_err(|e| Error::Config(format!("Failed to apply env overrides: {}", e)))
    }
    
    /// Set a value at a dotted path
    fn set_value_at_path(&self, value: &mut Value, path: &str, new_value: &str) -> Result<()> {
        let parts: Vec<&str> = path.split('.').collect();
        let mut current = value;
        
        for (i, part) in parts.iter().enumerate() {
            if i == parts.len() - 1 {
                // Last part - set the value
                if let Value::Object(map) = current {
                    // Try to parse as appropriate type
                    let parsed_value = if let Ok(b) = new_value.parse::<bool>() {
                        Value::Bool(b)
                    } else if let Ok(n) = new_value.parse::<i64>() {
                        Value::Number(serde_json::Number::from(n))
                    } else if let Ok(f) = new_value.parse::<f64>() {
                        Value::Number(serde_json::Number::from_f64(f).unwrap())
                    } else {
                        Value::String(new_value.to_string())
                    };
                    
                    map.insert(part.to_string(), parsed_value);
                }
            } else {
                // Navigate deeper
                if let Value::Object(map) = current {
                    current = map.entry(part.to_string())
                        .or_insert_with(|| Value::Object(serde_json::Map::new()));
                }
            }
        }
        
        Ok(())
    }
}

/// Universal configuration structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniversalConfig {
    /// Job configuration
    pub job: JobConfig,
    
    /// Model configuration
    pub model: ModelConfig,
    
    /// Network adapter configuration
    pub network: Option<NetworkConfig>,
    
    /// Training configuration
    pub training: Option<TrainingConfig>,
    
    /// Dataset configuration
    pub dataset: Option<DatasetConfig>,
    
    /// Inference configuration
    pub inference: Option<InferenceConfig>,
    
    /// Extension configurations
    pub extensions: HashMap<String, Value>,
}

/// Job configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobConfig {
    pub name: String,
    pub job_type: JobType,
    pub description: Option<String>,
    pub priority: JobPriority,
    pub tags: Vec<String>,
}

/// Job types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobType {
    Training,
    Inference,
    Conversion,
    Evaluation,
    Custom,
}

/// Job priority
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobPriority {
    Low,
    Normal,
    High,
    Critical,
}

/// Model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub architecture: ModelArchitecture,
    pub pretrained_model_name_or_path: String,
    pub revision: Option<String>,
    pub variant: Option<String>,
    pub torch_dtype: Option<String>,
    pub use_safetensors: bool,
}

/// Network adapter configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    pub network_type: NetworkType,
    pub rank: usize,
    pub alpha: f32,
    pub dropout: f32,
    pub target_modules: Vec<String>,
    pub use_bias: bool,
    pub init_strategy: InitStrategy,
}

/// Initialization strategy
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InitStrategy {
    Normal,
    Xavier,
    Kaiming,
    Zero,
    PiSSA,
}

/// Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub num_epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub gradient_accumulation_steps: usize,
    pub max_grad_norm: f32,
    pub warmup_steps: usize,
    pub save_steps: usize,
    pub logging_steps: usize,
    pub validation_steps: Option<usize>,
    pub checkpointing_steps: Option<usize>,
    pub resume_from_checkpoint: Option<PathBuf>,
    pub output_dir: PathBuf,
    pub mixed_precision: MixedPrecisionConfig,
    pub optimizer: OptimizerConfig,
    pub scheduler: SchedulerConfig,
}

/// Mixed precision configuration
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MixedPrecisionConfig {
    No,
    Fp16,
    Bf16,
    Fp8,
}

/// Optimizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OptimizerConfig {
    AdamW {
        betas: (f32, f32),
        eps: f32,
        weight_decay: f32,
    },
    Adam8bit {
        betas: (f32, f32),
        eps: f32,
    },
    Prodigy {
        betas: (f32, f32),
        beta3: f32,
        weight_decay: f32,
        eps: f32,
        decouple: bool,
        use_bias_correction: bool,
        safeguard_warmup: bool,
    },
    Adafactor {
        scale_parameter: bool,
        relative_step: bool,
        warmup_init: bool,
        weight_decay: f32,
    },
}

/// Learning rate scheduler configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SchedulerConfig {
    Constant,
    Linear {
        num_warmup_steps: usize,
    },
    Cosine {
        num_warmup_steps: usize,
        num_cycles: f32,
    },
    CosineWithRestarts {
        num_warmup_steps: usize,
        num_cycles: usize,
    },
    Polynomial {
        num_warmup_steps: usize,
        power: f32,
    },
}

/// Dataset configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetConfig {
    pub dataset_name_or_path: String,
    pub caption_column: String,
    pub image_column: String,
    pub conditioning_image_column: Option<String>,
    pub resolution: usize,
    pub random_flip: bool,
    pub center_crop: bool,
    pub augmentations: Vec<AugmentationConfig>,
    pub cache_dir: Option<PathBuf>,
}

/// Augmentation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AugmentationConfig {
    ColorJitter {
        brightness: f32,
        contrast: f32,
        saturation: f32,
        hue: f32,
    },
    RandomRotation {
        degrees: f32,
    },
    RandomCrop {
        size: (usize, usize),
    },
}

/// Inference configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    pub num_inference_steps: usize,
    pub guidance_scale: f32,
    pub negative_prompt: Option<String>,
    pub num_images_per_prompt: usize,
    pub seed: Option<u64>,
    pub scheduler: InferenceScheduler,
}

/// Inference scheduler types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InferenceScheduler {
    DDPM,
    DDIM,
    DPMSolverMultistep,
    DPMSolverSinglestep,
    EulerDiscrete,
    EulerAncestralDiscrete,
    HeunDiscrete,
    LCM,
    FlowMatchEuler,
}

impl Config for UniversalConfig {
    fn validate(&self) -> Result<()> {
        // Validate job config
        if self.job.name.is_empty() {
            return Err(Error::Config("Job name cannot be empty".to_string()));
        }
        
        // Validate model config
        if self.model.pretrained_model_name_or_path.is_empty() {
            return Err(Error::Config("Model path cannot be empty".to_string()));
        }
        
        // Validate training config if present
        if let Some(ref training) = self.training {
            if training.batch_size == 0 {
                return Err(Error::Config("Batch size must be greater than 0".to_string()));
            }
            if training.learning_rate <= 0.0 {
                return Err(Error::Config("Learning rate must be positive".to_string()));
            }
        }
        
        Ok(())
    }
    
    fn merge(&mut self, other: &Value) -> Result<()> {
        // Convert self to Value
        let mut self_value = self.as_value()?;
        
        // Merge with other
        if let (Value::Object(ref mut self_map), Value::Object(other_map)) = (&mut self_value, other) {
            for (k, v) in other_map {
                self_map.insert(k.clone(), v.clone());
            }
        }
        
        // Convert back
        *self = serde_json::from_value(self_value)
            .map_err(|e| Error::Config(format!("Failed to merge config: {}", e)))?;
        
        Ok(())
    }
}
