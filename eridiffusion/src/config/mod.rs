//! Configuration module for parsing YAML training configs

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

pub mod trainer_config;

pub use trainer_config::{ModelConfig, NetworkConfig, SampleConfig, TrainConfig, TrainingConfig};

/// Parse a YAML configuration file
pub fn parse_config(path: &str) -> Result<TrainingConfig, Box<dyn std::error::Error>> {
    let content = std::fs::read_to_string(path)?;
    let config: TrainingConfig = serde_yaml::from_str(&content)?;
    Ok(config)
}

/// Determine the model architecture from the config
pub fn get_model_arch(config: &TrainingConfig) -> String {
    // Check explicit arch field first
    if let Some(arch) = &config.config.process[0].model.arch {
        return arch.clone();
    }

    // Otherwise infer from model flags
    if config.config.process[0].model.is_flux.unwrap_or(false) {
        return "flux".to_string();
    }

    if config.config.process[0].model.is_v3.unwrap_or(false) {
        return "sd3".to_string();
    }

    if config.config.process[0].model.is_sdxl.unwrap_or(false) {
        return "sdxl".to_string();
    }

    // Default to SDXL if nothing specified
    "sdxl".to_string()
}
