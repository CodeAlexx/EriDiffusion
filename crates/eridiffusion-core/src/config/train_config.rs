use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use super::enums::*;

// For serde default = "path" references — keeps the module clean.
fn default_true() -> bool { true }
fn default_false() -> bool { false }
fn default_one() -> f64 { 1.0 }
fn default_one_u64() -> u64 { 1 }
fn default_none_f64() -> Option<f64> { None }
fn default_none_u64() -> Option<u64> { None }
fn default_zero() -> f64 { 0.0 }
fn default_zero_u64() -> u64 { 0 }
fn default_empty() -> String { String::new() }
fn default_full() -> String { "full".to_string() }
fn default_workspace() -> String { "workspace/run".to_string() }
fn default_cache() -> String { "workspace-cache/run".to_string() }
fn default_lr() -> f64 { 3e-6 }
fn default_lr_opt() -> Option<f64> { None }
fn default_wd() -> f64 { 0.01 }
fn default_eps() -> f64 { 1e-8 }
fn default_b1() -> f64 { 0.9 }
fn default_b2() -> f64 { 0.999 }
fn default_clip() -> f64 { 1.0 }
fn default_ema_decay() -> f64 { 0.999 }
fn default_rank() -> u64 { 16 }
fn default_alpha() -> f64 { 1.0 }
fn default_warmup() -> f64 { 200.0 }
fn default_epochs() -> u64 { 100 }
fn default_backup_mins() -> u64 { 30 }
fn default_save_sample10() -> u64 { 10 }
fn default_resolution() -> String { "1024".to_string() }
fn default_optimizer() -> String { "adamw".to_string() }
fn default_one_f32() -> f32 { 1.0 }
fn default_ema_power() -> f32 { 0.6667 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainConfig {
    #[serde(default = "default_zero_u64")]
    pub __version: u64,

    // ── Identity ──
    #[serde(default)]
    pub model_type: ModelType,
    #[serde(default)]
    pub training_method: TrainingMethod,

    // ── Paths ──
    #[serde(default = "default_workspace")]
    pub workspace_dir: String,
    #[serde(default = "default_cache")]
    pub cache_dir: String,
    #[serde(default = "default_empty")]
    pub base_model_name: String,
    #[serde(default)]
    pub output_model_destination: String,
    #[serde(default)]
    pub concept_file_name: String,

    // ── Hyperparameters ──
    #[serde(default = "default_lr")]
    pub learning_rate: f64,
    #[serde(default = "default_one_u64")]
    pub batch_size: u64,
    #[serde(default = "default_one_u64")]
    pub gradient_accumulation_steps: u64,
    #[serde(default = "default_epochs")]
    pub epochs: u64,

    // ── Model parts ──
    #[serde(default)]
    pub unet: ModelPartConfig,
    #[serde(default)]
    pub transformer: ModelPartConfig,
    #[serde(default)]
    pub text_encoder: ModelPartConfig,
    #[serde(default)]
    pub text_encoder_2: ModelPartConfig,
    #[serde(default)]
    pub text_encoder_3: ModelPartConfig,
    #[serde(default)]
    pub vae: ModelPartConfig,

    // ── LoRA ──
    #[serde(default)]
    pub peft_type: PeftType,
    #[serde(default = "default_rank")]
    pub lora_rank: u64,
    #[serde(default = "default_alpha")]
    pub lora_alpha: f64,
    #[serde(default = "default_false")]
    pub lora_decompose: bool,
    #[serde(default)]
    pub lora_weight_dtype: DataType,
    #[serde(default = "default_empty")]
    pub lora_model_name: String,
    #[serde(default)]
    pub layer_filter: String,

    // ── Optimizer ──
    #[serde(default)]
    pub optimizer: TrainOptimizerConfig,

    // ── Scheduler ──
    #[serde(default)]
    pub learning_rate_scheduler: LrScheduler,
    #[serde(default = "default_warmup")]
    pub learning_rate_warmup_steps: f64,
    #[serde(default = "default_one")]
    pub learning_rate_cycles: f64,

    // ── EMA ──
    #[serde(default)]
    pub ema: EmAMode,
    #[serde(default = "default_ema_decay")]
    pub ema_decay: f64,
    #[serde(default = "default_one_u64")]
    pub ema_update_step_interval: u64,

    // ── Noise / timestep ──
    #[serde(default)]
    pub timestep_distribution: TimestepDistribution,
    #[serde(default = "default_one")]
    pub timestep_shift: f64,
    #[serde(default = "default_one")]
    pub max_noising_strength: f64,
    #[serde(default = "default_zero")]
    pub min_noising_strength: f64,
    #[serde(default = "default_zero")]
    pub offset_noise_weight: f64,
    #[serde(default = "default_false")]
    pub force_v_prediction: bool,

    // ── Loss ──
    #[serde(default = "default_one")]
    pub mse_strength: f64,
    #[serde(default = "default_zero")]
    pub mae_strength: f64,
    #[serde(default)]
    pub loss_weight_fn: LossWeight,
    #[serde(default = "default_zero")]
    pub dropout_probability: f64,

    // ── Gradient ──
    #[serde(default = "default_clip")]
    pub clip_grad_norm: f64,
    #[serde(default)]
    pub gradient_checkpointing: GradientCheckpointing,

    // ── Sampling ──
    #[serde(default = "default_false")]
    pub validation: bool,
    #[serde(default = "default_one_u64")]
    pub validate_after: u64,
    #[serde(default = "default_save_sample10")]
    pub sample_after: u64,
    #[serde(default)]
    pub samples_to_tensorboard: bool,

    // ── Checkpointing ──
    #[serde(default = "default_backup_mins")]
    pub backup_after: u64,
    #[serde(default = "default_zero_u64")]
    pub save_every: u64,

    // ── DType / device ──
    #[serde(default)]
    pub train_dtype: DataType,
    #[serde(default)]
    pub output_dtype: DataType,
    #[serde(default)]
    pub output_model_format: ModelFormat,
    #[serde(default = "default_empty")]
    pub train_device: String,

    // ── Debug ──
    #[serde(default = "default_false")]
    pub debug_mode: bool,

    // ── Phase 1 (multi-feature rollout) — default-off ──
    /// MIN-SNR-γ loss weighting. `None` = no weighting (default).
    #[serde(default)]
    pub min_snr_gamma: Option<f32>,
    /// Probability of replacing caption with cached unconditional embedding
    /// per training step. `0.0` = never drop (default).
    #[serde(default)]
    pub caption_dropout_probability: f32,
    /// Bernoulli gate for the offset-noise modifier. Only takes effect if
    /// `offset_noise_weight > 0`. Default `1.0` keeps current behavior:
    /// when offset noise is enabled, it always fires.
    #[serde(default = "default_one_f32")]
    pub noise_offset_probability: f32,
    /// γ for input-perturbation noise modifier. `0.0` = no perturbation (default).
    #[serde(default)]
    pub gamma_input_perturbation: f32,
    /// Huber-loss strength (combined with `mse_strength` and `mae_strength`).
    /// `0.0` = no Huber term (default).
    #[serde(default)]
    pub huber_strength: f32,
    /// Floor multiplier on the LR schedule (e.g. `0.1` keeps LR ≥ 0.1·base_lr
    /// at the bottom of cosine decay). `0.0` = no floor (default).
    #[serde(default)]
    pub lr_min_factor: f32,

    // ── Phase 2 ──
    /// Held-out validation cache directory. `None` = no validation (default).
    #[serde(default)]
    pub validation_dataset_dir: Option<PathBuf>,
    /// Run validation every N training steps. `0` = disabled (default).
    #[serde(default)]
    pub validation_every_steps: u64,
    /// Per-backend sampling weights for multi-concept dataset mixing.
    /// Empty = single backend, identical to today's behavior.
    #[serde(default)]
    pub multi_backend_weights: Vec<f32>,
    /// JSON file describing N validation prompts × M seeds for periodic
    /// sample rendering. `None` = single-prompt path (default).
    #[serde(default)]
    pub validation_prompts_file: Option<PathBuf>,

    // ── Phase 3 ──
    /// Foreground-mask weight for masked-loss weighting. `0.0` = unmasked (default).
    #[serde(default)]
    pub masked_loss_weight: f32,
    /// EMA `inv_gamma` for power-decay schedule.
    #[serde(default = "default_one_f32")]
    pub ema_inv_gamma: f32,
    /// EMA `power` for power-decay schedule.
    #[serde(default = "default_ema_power")]
    pub ema_power: f32,
    /// Skip EMA updates until this step.
    #[serde(default)]
    pub ema_update_after_step: u64,
    /// EMA decay floor.
    #[serde(default)]
    pub ema_min_decay: f32,
    /// Swap EMA shadow weights INTO the live parameters at sample/checkpoint
    /// time so renders + saves use the EMA-averaged weights. Default `false`
    /// preserves prior behavior (samples render against the live training
    /// weights). Caller must also have EMA active for this to do anything.
    #[serde(default)]
    pub ema_validation_swap: bool,

    // ── Phase 4 ──
    /// TREAD route pattern, e.g. `"12-23"`. `None` = no token routing (default).
    #[serde(default)]
    pub tread_route_pattern: Option<String>,
    /// TREAD keep ratio: fraction of tokens that route through the routed
    /// block range. `1.0` (default) → no routing, byte-identical to
    /// non-TREAD forward. Phase 4 ships the CLI surface; Phase 4.5 wires
    /// it into model forward.
    #[serde(default = "default_tread_keep_ratio")]
    pub tread_keep_ratio: f32,
}

fn default_tread_keep_ratio() -> f32 { 1.0 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPartConfig {
    #[serde(default = "default_true")]
    pub train: bool,
    #[serde(default = "default_empty")]
    pub model_name: String,
    #[serde(default = "default_none_f64")]
    pub learning_rate: Option<f64>,
    #[serde(default = "default_zero")]
    pub dropout_probability: f64,
    #[serde(default = "default_true")]
    pub train_embedding: bool,
}

impl Default for ModelPartConfig {
    fn default() -> Self {
        Self { train: true, model_name: String::new(), learning_rate: None, dropout_probability: 0.0, train_embedding: true }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainOptimizerConfig {
    #[serde(default = "default_optimizer")]
    pub name: String,
    #[serde(default = "default_lr_opt")]
    pub learning_rate: Option<f64>,
    #[serde(default = "default_wd")]
    pub weight_decay: f64,
    #[serde(default = "default_eps")]
    pub eps: f64,
    #[serde(default = "default_b1")]
    pub beta1: f64,
    #[serde(default = "default_b2")]
    pub beta2: f64,
}

impl Default for TrainOptimizerConfig {
    fn default() -> Self {
        Self { name: "adamw".into(), learning_rate: None, weight_decay: 0.01, eps: 1e-8, beta1: 0.9, beta2: 0.999 }
    }
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            __version: 10,
            model_type: ModelType::default(),
            training_method: TrainingMethod::default(),
            workspace_dir: "workspace/run".into(),
            cache_dir: "workspace-cache/run".into(),
            base_model_name: String::new(),
            output_model_destination: String::new(),
            concept_file_name: String::new(),
            learning_rate: 3e-6,
            batch_size: 1,
            gradient_accumulation_steps: 1,
            epochs: 100,
            unet: ModelPartConfig::default(),
            transformer: ModelPartConfig::default(),
            text_encoder: ModelPartConfig::default(),
            text_encoder_2: ModelPartConfig::default(),
            text_encoder_3: ModelPartConfig::default(),
            vae: ModelPartConfig::default(),
            peft_type: PeftType::Lora,
            lora_rank: 16,
            lora_alpha: 1.0,
            lora_decompose: false,
            lora_weight_dtype: DataType::Float32,
            lora_model_name: String::new(),
            layer_filter: String::new(),
            optimizer: TrainOptimizerConfig::default(),
            learning_rate_scheduler: LrScheduler::Constant,
            learning_rate_warmup_steps: 200.0,
            learning_rate_cycles: 1.0,
            ema: EmAMode::Off,
            ema_decay: 0.999,
            ema_update_step_interval: 5,
            timestep_distribution: TimestepDistribution::Uniform,
            timestep_shift: 1.0,
            max_noising_strength: 1.0,
            min_noising_strength: 0.0,
            offset_noise_weight: 0.0,
            force_v_prediction: false,
            mse_strength: 1.0,
            mae_strength: 0.0,
            loss_weight_fn: LossWeight::Constant,
            dropout_probability: 0.0,
            clip_grad_norm: 1.0,
            gradient_checkpointing: GradientCheckpointing::On,
            validation: false,
            validate_after: 1,
            sample_after: 10,
            samples_to_tensorboard: true,
            backup_after: 30,
            save_every: 0,
            train_dtype: DataType::Float16,
            output_dtype: DataType::Float32,
            output_model_format: ModelFormat::Safetensors,
            train_device: String::new(),
            debug_mode: false,
            // Phase 1 — defaults preserve existing behavior
            min_snr_gamma: None,
            caption_dropout_probability: 0.0,
            noise_offset_probability: 1.0,
            gamma_input_perturbation: 0.0,
            huber_strength: 0.0,
            lr_min_factor: 0.0,
            // Phase 2
            validation_dataset_dir: None,
            validation_every_steps: 0,
            multi_backend_weights: Vec::new(),
            validation_prompts_file: None,
            // Phase 3
            masked_loss_weight: 0.0,
            ema_inv_gamma: 1.0,
            ema_power: 0.6667,
            ema_update_after_step: 0,
            ema_min_decay: 0.0,
            ema_validation_swap: false,
            // Phase 4
            tread_route_pattern: None,
            tread_keep_ratio: 1.0,
        }
    }
}

impl TrainConfig {
    pub fn from_json_path(path: &str) -> crate::Result<Self> {
        let f = std::fs::File::open(path)?;
        Ok(serde_json::from_reader(f)?)
    }

    pub fn from_json_str(s: &str) -> crate::Result<Self> {
        Ok(serde_json::from_str(s)?)
    }

    pub fn to_json_pretty(&self) -> crate::Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    pub fn is_lora(&self) -> bool { self.training_method == TrainingMethod::Lora }
    pub fn is_fine_tune(&self) -> bool { self.training_method == TrainingMethod::FineTune }
    pub fn is_flow_matching(&self) -> bool {
        matches!(self.model_type,
            ModelType::FluxDev1 | ModelType::Flux2 | ModelType::StableDiffusion3 |
            ModelType::StableDiffusion35 | ModelType::Sana | ModelType::ZImage |
            ModelType::Qwen | ModelType::HunyuanVideo
        )
    }
}
