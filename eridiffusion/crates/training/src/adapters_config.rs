use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AdaptersConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub kind: String,          // "lycoris"
    #[serde(default)]
    pub path: Option<String>,  // dir or .safetensors
    #[serde(default)]
    pub apply_to: Vec<String>, // allowlist globs
    #[serde(default)]
    pub denylist: Vec<String>, // denylist globs
    #[serde(default = "default_weight")] 
    pub weight: f32,
}

fn default_weight() -> f32 { 1.0 }

