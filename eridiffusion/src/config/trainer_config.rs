//! Training configuration structures that match the YAML format

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Root configuration structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    #[serde(default)]
    pub job: Option<String>,
    pub config: Config,
    #[serde(default)]
    pub meta: HashMap<String, String>,
}

/// Main config section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub name: String,
    pub process: Vec<ProcessConfig>,
}

/// Process configuration (supports multiple but we typically use one)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessConfig {
    #[serde(rename = "type")]
    pub process_type: String,
    pub training_folder: String,
    #[serde(default = "default_device")]
    pub device: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trigger_word: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub performance_log_every: Option<usize>,
    pub network: NetworkConfig,
    pub save: SaveConfig,
    pub datasets: Vec<DatasetConfig>,
    pub train: TrainConfig,
    pub model: ModelConfig,
    pub sample: SampleConfig,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tread: Option<TreadConfig>,
}

/// Network configuration (LoRA, LoKR, etc)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    #[serde(rename = "type")]
    pub network_type: String,
    pub linear: usize,     // This is lora_rank
    pub linear_alpha: f32, // This is lora_alpha
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conv: Option<usize>, // For LoKR
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conv_alpha: Option<f32>,

    // ChromaXL-specific fields
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ramp_double_blocks: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ramp_target_lr: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ramp_warmup_steps: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ramp_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub network_kwargs: Option<NetworkKwargs>,
}

/// Network kwargs for layer-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkKwargs {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lr_if_contains: Option<HashMap<String, f32>>,
}

/// Save configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SaveConfig {
    pub dtype: String,
    pub save_every: usize,
    pub max_step_saves_to_keep: usize,
    #[serde(default)]
    pub push_to_hub: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hf_repo_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hf_private: Option<bool>,
}

/// Dataset configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
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
    pub resolution: Vec<usize>,
    #[serde(default)]
    pub force_recache: bool,
}

/// Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainConfig {
    pub batch_size: usize,
    pub steps: usize,
    #[serde(default = "default_one")]
    pub gradient_accumulation_steps: usize,
    #[serde(default = "default_true")]
    pub train_unet: bool,
    #[serde(default)]
    pub train_text_encoder: bool,
    #[serde(default = "default_true")]
    pub gradient_checkpointing: bool,
    pub noise_scheduler: String,
    pub optimizer: String,
    pub lr: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skip_first_sample: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub disable_sampling: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub linear_timesteps: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ema_config: Option<EMAConfig>,
    pub dtype: String,
    #[serde(default = "default_true")]
    pub bypass_guidance_embedding: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub use_layer_streaming: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub streaming_memory_limit_gb: Option<f32>,
}

/// TREAD routing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreadConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub mask: Option<TreadMask>,
    #[serde(default)]
    pub schedule: Vec<TreadPair>,
    #[serde(default)]
    pub reinject: Option<TreadReinject>,
    #[serde(default)]
    pub loss: Option<TreadLoss>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreadMask {
    #[serde(default)]
    pub r#type: Option<String>, // attn_topk|topk_norm|random|uniform
    #[serde(default)]
    pub k: Option<usize>,
    #[serde(default)]
    pub k_frac: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreadPair {
    pub out: usize,
    pub r#in: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreadReinject {
    #[serde(default)]
    pub mode: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreadLoss {
    #[serde(default)]
    pub route_lambda: Option<f32>,
}

/// EMA configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EMAConfig {
    pub use_ema: bool,
    pub ema_decay: f32,
}

/// Model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub name_or_path: PathBuf,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arch: Option<String>, // Explicit architecture
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_flux: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_v3: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_sdxl: Option<bool>,
    #[serde(default)]
    pub quantize: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub low_vram: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vae_path: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text_encoder_path: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text_encoder_2_path: Option<PathBuf>,
    // Flux-specific aliases that map to the generic fields
    #[serde(skip_serializing_if = "Option::is_none", alias = "clip_l_path")]
    pub clip_l_path: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none", alias = "t5_path")]
    pub t5_path: Option<PathBuf>,
}

/// Sampling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleConfig {
    pub sampler: String,
    pub sample_every: usize,
    pub width: usize,
    pub height: usize,
    pub prompts: Vec<String>,
    #[serde(default)]
    pub neg: String,
    pub seed: u64,
    #[serde(default)]
    pub walk_seed: bool,
    pub guidance_scale: f32,
    pub sample_steps: usize,
}

// Default functions for serde
fn default_device() -> String {
    "cuda:0".to_string()
}

fn default_caption_ext() -> String {
    "txt".to_string()
}

fn default_true() -> bool {
    true
}

fn default_one() -> usize {
    1
}
