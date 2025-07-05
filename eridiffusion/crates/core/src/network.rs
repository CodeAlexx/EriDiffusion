//! Network adapter traits and types

use crate::{Result, Device};
use async_trait::async_trait;
use candle_core::{Tensor, Var};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::fmt;

/// Network adapter types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NetworkType {
    /// Standard LoRA
    LoRA,
    /// DoRA (Weight-Decomposed LoRA)
    DoRA,
    /// LoCoN (LoRA for Convolution)
    LoCoN,
    /// LoKr (Kronecker Product)
    LoKr,
    /// GLoRA (Generalized LoRA)
    GLoRA,
    /// LoRM (Low-Rank Modification)
    LoRM,
    /// iLoRA (Iterative LoRA)
    ILoRA,
    /// Custom adapter
    Custom(String),
}

impl fmt::Display for NetworkType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NetworkType::LoRA => write!(f, "LoRA"),
            NetworkType::DoRA => write!(f, "DoRA"),
            NetworkType::LoCoN => write!(f, "LoCoN"),
            NetworkType::LoKr => write!(f, "LoKr"),
            NetworkType::GLoRA => write!(f, "GLoRA"),
            NetworkType::LoRM => write!(f, "LoRM"),
            NetworkType::ILoRA => write!(f, "iLoRA"),
            NetworkType::Custom(name) => write!(f, "Custom({})", name),
        }
    }
}

/// Base trait for network adapters
#[async_trait]
pub trait NetworkAdapter: Send + Sync {
    /// Get adapter type
    fn adapter_type(&self) -> NetworkType;
    
    /// Get metadata
    fn metadata(&self) -> &NetworkMetadata;
    
    /// Get target modules
    fn target_modules(&self) -> &[String];
    
    /// Get trainable parameters
    fn trainable_parameters(&self) -> Vec<&Var>;
    
    /// Get all parameters as owned tensors (cloned)
    fn parameters(&self) -> HashMap<String, Tensor>;
    
    /// Apply adapter to a layer
    fn apply_to_layer(&self, layer_name: &str, input: &Tensor) -> Result<Tensor>;
    
    /// Merge weights with base model
    fn merge_weights(&mut self, scale: f32) -> Result<()>;
    
    /// Save adapter weights
    async fn save_weights(&self, path: &Path) -> Result<()>;
    
    /// Load adapter weights
    async fn load_weights(&mut self, path: &Path) -> Result<()>;
    
    /// Get memory usage
    fn memory_usage(&self) -> usize;
    
    /// Set device
    fn to_device(&mut self, device: &Device) -> Result<()>;
}

/// Network adapter metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkMetadata {
    pub name: String,
    pub network_type: NetworkType,
    pub version: String,
    pub base_model: String,
    pub rank: Option<usize>,
    pub alpha: Option<f32>,
    pub target_modules: Vec<String>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub config: HashMap<String, serde_json::Value>,
}

impl Default for NetworkMetadata {
    fn default() -> Self {
        Self {
            name: "unnamed".to_string(),
            network_type: NetworkType::LoRA,
            version: "1.0.0".to_string(),
            base_model: "unknown".to_string(),
            rank: None,
            alpha: None,
            target_modules: vec![],
            created_at: chrono::Utc::now(),
            config: HashMap::new(),
        }
    }
}

/// Configuration for network adapters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    pub network_type: NetworkType,
    pub rank: usize,
    pub alpha: f32,
    pub dropout: f32,
    pub target_modules: Vec<String>,
    pub use_bias: bool,
    pub fan_in_fan_out: bool,
    pub modules_to_save: Vec<String>,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            network_type: NetworkType::LoRA,
            rank: 16,
            alpha: 16.0,
            dropout: 0.0,
            target_modules: vec![],
            use_bias: false,
            fan_in_fan_out: false,
            modules_to_save: vec![],
        }
    }
}

/// Module pattern matcher for automatic target module detection
#[derive(Debug, Clone)]
pub struct ModuleMatcher {
    patterns: Vec<String>,
}

impl ModuleMatcher {
    pub fn new(patterns: Vec<String>) -> Self {
        Self { patterns }
    }
    
    /// Check if a module name matches any pattern
    pub fn matches(&self, module_name: &str) -> bool {
        self.patterns.iter().any(|pattern| {
            if pattern.contains('*') {
                // Simple wildcard matching
                let parts: Vec<&str> = pattern.split('*').collect();
                if parts.len() == 2 {
                    module_name.starts_with(parts[0]) && module_name.ends_with(parts[1])
                } else {
                    module_name.contains(&pattern.replace("*", ""))
                }
            } else {
                module_name == pattern
            }
        })
    }
    
    /// Get default patterns for an architecture
    pub fn default_patterns(architecture: crate::ModelArchitecture) -> Vec<String> {
        use crate::ModelArchitecture;
        
        match architecture {
            ModelArchitecture::SDXL | ModelArchitecture::SD15 => vec![
                "*.to_q".to_string(),
                "*.to_k".to_string(),
                "*.to_v".to_string(),
                "*.to_out.0".to_string(),
                "*.ff.net.0".to_string(),
                "*.ff.net.2".to_string(),
            ],
            ModelArchitecture::SD3 | ModelArchitecture::SD35 => vec![
                "*.attn.to_q".to_string(),
                "*.attn.to_k".to_string(),
                "*.attn.to_v".to_string(),
                "*.attn.to_out.0".to_string(),
                "*.ff.linear_1".to_string(),
                "*.ff.linear_2".to_string(),
            ],
            ModelArchitecture::Flux | ModelArchitecture::FluxSchnell | ModelArchitecture::FluxDev => vec![
                "*.self_attn.qkv".to_string(),
                "*.self_attn.o".to_string(),
                "*.mlp.fc1".to_string(),
                "*.mlp.fc2".to_string(),
            ],
            _ => vec![
                "*.attention.*.query".to_string(),
                "*.attention.*.key".to_string(),
                "*.attention.*.value".to_string(),
                "*.attention.*.out".to_string(),
            ],
        }
    }
}