//! Builder patterns for complex type construction

use crate::{Result, Error, Device, ModelArchitecture};
use crate::dtype::Precision;
use std::path::PathBuf;
use std::collections::HashMap;

/// Builder for model configuration
#[derive(Default)]
pub struct ModelConfigBuilder {
    architecture: Option<ModelArchitecture>,
    device: Option<Device>,
    precision: Option<Precision>,
    model_path: Option<String>,
    compile: Option<bool>,
    use_flash_attention: Option<bool>,
    custom_config: HashMap<String, serde_json::Value>,
}

impl ModelConfigBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn architecture(mut self, arch: ModelArchitecture) -> Self {
        self.architecture = Some(arch);
        self
    }
    
    pub fn device(mut self, device: Device) -> Self {
        self.device = Some(device);
        self
    }
    
    pub fn precision(mut self, precision: Precision) -> Self {
        self.precision = Some(precision);
        self
    }
    
    pub fn model_path(mut self, path: impl Into<String>) -> Self {
        self.model_path = Some(path.into());
        self
    }
    
    pub fn compile(mut self, compile: bool) -> Self {
        self.compile = Some(compile);
        self
    }
    
    pub fn use_flash_attention(mut self, use_flash: bool) -> Self {
        self.use_flash_attention = Some(use_flash);
        self
    }
    
    pub fn custom<T: serde::Serialize>(mut self, key: &str, value: T) -> Result<Self> {
        let json_value = serde_json::to_value(value)
            .map_err(|e| Error::Serialization(e.to_string()))?;
        self.custom_config.insert(key.to_string(), json_value);
        Ok(self)
    }
    
    pub fn build(self) -> Result<ModelConfig> {
        Ok(ModelConfig {
            architecture: self.architecture
                .ok_or_else(|| Error::Config("Architecture not specified".to_string()))?,
            device: self.device.unwrap_or_else(|| Device::cuda_if_available().unwrap_or(Device::Cpu)),
            precision: self.precision.unwrap_or(Precision::Float16),
            model_path: self.model_path
                .ok_or_else(|| Error::Config("Model path not specified".to_string()))?,
            compile: self.compile.unwrap_or(false),
            use_flash_attention: self.use_flash_attention.unwrap_or(true),
            custom_config: self.custom_config,
        })
    }
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub architecture: ModelArchitecture,
    pub device: Device,
    pub precision: Precision,
    pub model_path: String,
    pub compile: bool,
    pub use_flash_attention: bool,
    pub custom_config: HashMap<String, serde_json::Value>,
}

/// Builder for training configuration
#[derive(Default)]
pub struct TrainingConfigBuilder {
    // Model settings
    model_config: Option<ModelConfig>,
    
    // Training hyperparameters
    learning_rate: Option<f32>,
    batch_size: Option<usize>,
    num_epochs: Option<usize>,
    gradient_accumulation_steps: Option<usize>,
    
    // Optimizer settings
    optimizer_type: Option<String>,
    weight_decay: Option<f32>,
    beta1: Option<f32>,
    beta2: Option<f32>,
    epsilon: Option<f32>,
    
    // Scheduler settings
    scheduler_type: Option<String>,
    warmup_steps: Option<usize>,
    
    // Other settings
    output_dir: Option<PathBuf>,
    checkpoint_steps: Option<usize>,
    logging_steps: Option<usize>,
    eval_steps: Option<usize>,
    seed: Option<u64>,
    mixed_precision: Option<bool>,
    gradient_checkpointing: Option<bool>,
}

impl TrainingConfigBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn model(mut self, config: ModelConfig) -> Self {
        self.model_config = Some(config);
        self
    }
    
    pub fn learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = Some(lr);
        self
    }
    
    pub fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = Some(size);
        self
    }
    
    pub fn epochs(mut self, epochs: usize) -> Self {
        self.num_epochs = Some(epochs);
        self
    }
    
    pub fn gradient_accumulation(mut self, steps: usize) -> Self {
        self.gradient_accumulation_steps = Some(steps);
        self
    }
    
    pub fn optimizer(mut self, optimizer_type: impl Into<String>) -> Self {
        self.optimizer_type = Some(optimizer_type.into());
        self
    }
    
    pub fn adam_params(mut self, beta1: f32, beta2: f32, epsilon: f32) -> Self {
        self.beta1 = Some(beta1);
        self.beta2 = Some(beta2);
        self.epsilon = Some(epsilon);
        self
    }
    
    pub fn weight_decay(mut self, decay: f32) -> Self {
        self.weight_decay = Some(decay);
        self
    }
    
    pub fn scheduler(mut self, scheduler_type: impl Into<String>) -> Self {
        self.scheduler_type = Some(scheduler_type.into());
        self
    }
    
    pub fn warmup_steps(mut self, steps: usize) -> Self {
        self.warmup_steps = Some(steps);
        self
    }
    
    pub fn output_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.output_dir = Some(dir.into());
        self
    }
    
    pub fn checkpointing(mut self, steps: usize) -> Self {
        self.checkpoint_steps = Some(steps);
        self
    }
    
    pub fn logging(mut self, steps: usize) -> Self {
        self.logging_steps = Some(steps);
        self
    }
    
    pub fn evaluation(mut self, steps: usize) -> Self {
        self.eval_steps = Some(steps);
        self
    }
    
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
    
    pub fn mixed_precision(mut self, enabled: bool) -> Self {
        self.mixed_precision = Some(enabled);
        self
    }
    
    pub fn gradient_checkpointing(mut self, enabled: bool) -> Self {
        self.gradient_checkpointing = Some(enabled);
        self
    }
    
    pub fn build(self) -> Result<TrainingConfig> {
        // Validate required fields
        let model_config = self.model_config
            .ok_or_else(|| Error::Config("Model configuration not specified".to_string()))?;
        
        Ok(TrainingConfig {
            model_config,
            learning_rate: self.learning_rate.unwrap_or(1e-4),
            batch_size: self.batch_size.unwrap_or(1),
            num_epochs: self.num_epochs.unwrap_or(10),
            gradient_accumulation_steps: self.gradient_accumulation_steps.unwrap_or(1),
            optimizer_type: self.optimizer_type.unwrap_or_else(|| "adamw".to_string()),
            weight_decay: self.weight_decay.unwrap_or(0.01),
            beta1: self.beta1.unwrap_or(0.9),
            beta2: self.beta2.unwrap_or(0.999),
            epsilon: self.epsilon.unwrap_or(1e-8),
            scheduler_type: self.scheduler_type.unwrap_or_else(|| "cosine".to_string()),
            warmup_steps: self.warmup_steps.unwrap_or(500),
            output_dir: self.output_dir.unwrap_or_else(|| PathBuf::from("./output")),
            checkpoint_steps: self.checkpoint_steps.unwrap_or(1000),
            logging_steps: self.logging_steps.unwrap_or(100),
            eval_steps: self.eval_steps.unwrap_or(500),
            seed: self.seed.unwrap_or(42),
            mixed_precision: self.mixed_precision.unwrap_or(true),
            gradient_checkpointing: self.gradient_checkpointing.unwrap_or(false),
        })
    }
}

#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub model_config: ModelConfig,
    pub learning_rate: f32,
    pub batch_size: usize,
    pub num_epochs: usize,
    pub gradient_accumulation_steps: usize,
    pub optimizer_type: String,
    pub weight_decay: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub scheduler_type: String,
    pub warmup_steps: usize,
    pub output_dir: PathBuf,
    pub checkpoint_steps: usize,
    pub logging_steps: usize,
    pub eval_steps: usize,
    pub seed: u64,
    pub mixed_precision: bool,
    pub gradient_checkpointing: bool,
}

/// Builder for inference configuration
pub struct InferenceConfigBuilder {
    model_config: Option<ModelConfig>,
    scheduler: Option<String>,
    num_inference_steps: Option<usize>,
    guidance_scale: Option<f32>,
    eta: Option<f32>,
    seed: Option<u64>,
    batch_size: Option<usize>,
    compile_model: Option<bool>,
}

impl Default for InferenceConfigBuilder {
    fn default() -> Self {
        Self {
            model_config: None,
            scheduler: None,
            num_inference_steps: None,
            guidance_scale: None,
            eta: None,
            seed: None,
            batch_size: None,
            compile_model: None,
        }
    }
}

impl InferenceConfigBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn model(mut self, config: ModelConfig) -> Self {
        self.model_config = Some(config);
        self
    }
    
    pub fn scheduler(mut self, scheduler: impl Into<String>) -> Self {
        self.scheduler = Some(scheduler.into());
        self
    }
    
    pub fn steps(mut self, steps: usize) -> Self {
        self.num_inference_steps = Some(steps);
        self
    }
    
    pub fn guidance_scale(mut self, scale: f32) -> Self {
        self.guidance_scale = Some(scale);
        self
    }
    
    pub fn eta(mut self, eta: f32) -> Self {
        self.eta = Some(eta);
        self
    }
    
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
    
    pub fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = Some(size);
        self
    }
    
    pub fn compile(mut self, compile: bool) -> Self {
        self.compile_model = Some(compile);
        self
    }
    
    pub fn build(self) -> Result<InferenceConfig> {
        let model_config = self.model_config
            .ok_or_else(|| Error::Config("Model configuration not specified".to_string()))?;
        
        Ok(InferenceConfig {
            model_config,
            scheduler: self.scheduler.unwrap_or_else(|| "ddim".to_string()),
            num_inference_steps: self.num_inference_steps.unwrap_or(25),
            guidance_scale: self.guidance_scale.unwrap_or(7.5),
            eta: self.eta.unwrap_or(0.0),
            seed: self.seed,
            batch_size: self.batch_size.unwrap_or(1),
            compile_model: self.compile_model.unwrap_or(false),
        })
    }
}

#[derive(Debug, Clone)]
pub struct InferenceConfig {
    pub model_config: ModelConfig,
    pub scheduler: String,
    pub num_inference_steps: usize,
    pub guidance_scale: f32,
    pub eta: f32,
    pub seed: Option<u64>,
    pub batch_size: usize,
    pub compile_model: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_model_config_builder() {
        let config = ModelConfigBuilder::new()
            .architecture(ModelArchitecture::SD15)
            .device(Device::Cpu)
            .precision(Precision::Float32)
            .model_path("test_model")
            .compile(true)
            .use_flash_attention(false)
            .build()
            .unwrap();
        
        assert_eq!(config.architecture, ModelArchitecture::SD15);
        assert_eq!(config.compile, true);
        assert_eq!(config.use_flash_attention, false);
    }
    
    #[test]
    fn test_training_config_builder() {
        let model_config = ModelConfigBuilder::new()
            .architecture(ModelArchitecture::SDXL)
            .model_path("sdxl")
            .build()
            .unwrap();
        
        let training_config = TrainingConfigBuilder::new()
            .model(model_config)
            .learning_rate(1e-5)
            .batch_size(4)
            .epochs(10)
            .optimizer("adamw")
            .adam_params(0.9, 0.999, 1e-8)
            .mixed_precision(true)
            .build()
            .unwrap();
        
        assert_eq!(training_config.learning_rate, 1e-5);
        assert_eq!(training_config.batch_size, 4);
        assert_eq!(training_config.num_epochs, 10);
    }
}