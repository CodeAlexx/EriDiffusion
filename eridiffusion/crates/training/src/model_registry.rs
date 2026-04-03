use eridiffusion_core::{Error, Result};
use serde::{Deserialize, Serialize};

#[cfg(all(feature = "real", any(feature = "sdxl", feature = "sd35")))]
use crate::pipeline::{adapter::ModelAdapter, orchestrator::Orchestrator};
/// Optional block specification for registry-driven models
#[derive(Debug, Clone)]
pub struct BlockSpec {
    pub name: &'static str,
    pub keys: &'static [&'static str],
    pub route_out: bool,
    pub route_in: bool,
}

pub mod models {
    pub mod flux_loader;
    pub mod hidream_loader;
    pub mod hunyuan_loader;
    pub mod ltx_loader;
    pub mod lumina_loader;
    pub mod omnigen2_loader;
    pub mod qwen_image_loader;
    pub mod sd3_loader;
    pub mod sdxl_loader;
    pub mod wan22_loader;
}

// Optional nested config groups to hydrate flat TrainCfg without breaking existing schema
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PathsNested {
    #[serde(default)]
    pub flux_weights: String,
    #[serde(default)]
    pub vae: String,
    #[serde(default)]
    pub t5: String,
    #[serde(default)]
    pub clip: String,
    #[serde(default)]
    pub t5_tokenizer: String,
    #[serde(default)]
    pub clip_tokenizer: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DataNested {
    #[serde(default)]
    pub root: String,
    #[serde(default)]
    pub batch_size: usize,
    #[serde(default)]
    pub prefetch: usize,
    #[serde(default)]
    pub manifest: String,
    #[serde(default)]
    pub manifest_root: String,
    #[serde(default)]
    pub manifest_index: String,
    #[serde(default)]
    pub manifest_validate: bool,
    #[serde(default)]
    pub manifest_shuffle: Option<bool>,
    #[serde(default)]
    pub manifest_repeat: Option<usize>,
    #[serde(default)]
    pub manifest_sdxl: String,
    #[serde(default)]
    pub manifest_sdxl_root: String,
    #[serde(default)]
    pub manifest_sdxl_index: String,
    #[serde(default)]
    pub manifest_sdxl_validate: bool,
    #[serde(default)]
    pub manifest_sdxl_shuffle: Option<bool>,
    #[serde(default)]
    pub manifest_sdxl_repeat: Option<usize>,
    #[serde(default)]
    pub manifest_sd35: String,
    #[serde(default)]
    pub manifest_sd35_root: String,
    #[serde(default)]
    pub manifest_sd35_index: String,
    #[serde(default)]
    pub manifest_sd35_validate: bool,
    #[serde(default)]
    pub manifest_sd35_shuffle: Option<bool>,
    #[serde(default)]
    pub manifest_sd35_repeat: Option<usize>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LoraNested {
    pub rank: usize,
    #[serde(default)]
    pub alpha: Option<f32>,
    #[serde(default)]
    pub zero_init: Option<bool>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TrainNested {
    pub steps: usize,
    pub grad_accum_steps: usize,
    #[serde(default)]
    pub lr: Option<f64>,
    pub ckpt_every: usize,
    #[serde(default)]
    pub bf16: bool,
    #[serde(default)]
    pub clip_grad_norm: Option<f64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScheduleNested {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub steps: usize,
    #[serde(default)]
    pub sigma_min: Option<f32>,
    #[serde(default)]
    pub sigma_max: Option<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelKind {
    Flux,
    SDXL,
    SD3,
    QwenImageV1,
    Wan22,
    HiDreamI1,
    Omnigen2,
    Ltx,
    Hunyuan,
    Lumina,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainCfg {
    pub model_kind: ModelKind,
    #[serde(default)]
    #[cfg(all(feature = "real", any(feature = "sdxl", feature = "sd35")))]
    pub recipe: Option<Vec<crate::pipeline::stages::StageName>>, // optional override
    #[serde(default)]
    #[cfg(not(feature = "real"))]
    pub recipe: Option<Vec<String>>, // placeholder in synthetic mode
    #[serde(default)]
    pub weights: Option<String>,
    #[serde(default)]
    pub bf16_params: bool,
    #[serde(default = "default_steps")]
    pub steps: usize,
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    #[serde(default = "default_grad_accum_steps")]
    pub grad_accum_steps: usize,
    #[serde(default = "default_lr")]
    pub lr: f64,
    #[serde(default = "default_seed")]
    pub seed: u64,
    #[serde(default)]
    /// Optional checkpoint interval; if set, real trainer paths may use this to emit checkpoints
    pub ckpt_every: Option<usize>,
    #[serde(default)]
    pub optimizer: Option<String>,
    #[serde(default)]
    pub offload_pinned_threshold_mb: Option<usize>,
    // Flux-specific paths
    #[serde(default)]
    pub flux_model_path: Option<String>,
    #[serde(default)]
    pub flux_vae_path: Option<String>,
    #[serde(default)]
    pub flux_t5_path: Option<String>,
    #[serde(default)]
    pub flux_clip_path: Option<String>,
    #[serde(default)]
    pub flux_t5_tokenizer: Option<String>,
    #[serde(default)]
    pub flux_clip_tokenizer: Option<String>,
    #[serde(default)]
    pub flux_variant: Option<String>,
    #[serde(default)]
    pub flux_manifest: Option<String>,
    #[serde(default)]
    pub flux_manifest_root: Option<String>,
    #[serde(default)]
    pub flux_manifest_index: Option<String>,
    #[serde(default)]
    pub flux_manifest_validate: Option<bool>,
    #[serde(default)]
    pub flux_manifest_shuffle: Option<bool>,
    #[serde(default)]
    pub flux_manifest_repeat: Option<usize>,
    #[serde(default)]
    pub sdxl_manifest: Option<String>,
    #[serde(default)]
    pub sdxl_manifest_root: Option<String>,
    #[serde(default)]
    pub sdxl_manifest_index: Option<String>,
    #[serde(default)]
    pub sdxl_manifest_validate: Option<bool>,
    #[serde(default)]
    pub sdxl_manifest_shuffle: Option<bool>,
    #[serde(default)]
    pub sdxl_manifest_repeat: Option<usize>,
    #[serde(default)]
    pub sd35_manifest: Option<String>,
    #[serde(default)]
    pub sd35_manifest_root: Option<String>,
    #[serde(default)]
    pub sd35_manifest_index: Option<String>,
    #[serde(default)]
    pub sd35_manifest_validate: Option<bool>,
    #[serde(default)]
    pub sd35_manifest_shuffle: Option<bool>,
    #[serde(default)]
    pub sd35_manifest_repeat: Option<usize>,
    // Optional LoRA hyperparams (flat)
    #[serde(default)]
    pub lora_rank: Option<usize>,
    #[serde(default)]
    pub lora_alpha: Option<f32>,
    #[serde(default)]
    pub lora_zero_init: Option<bool>,
    // Optional data convenience (flat)
    #[serde(default)]
    pub data_root: Option<String>,
    #[serde(default)]
    pub prefetch: Option<usize>,
    // Optional schedule hints (unused by minimal loop today)
    #[serde(default)]
    pub schedule_name: String,
    #[serde(default)]
    pub schedule_steps: usize,
    #[serde(default)]
    pub sigma_min: Option<f32>,
    #[serde(default)]
    pub sigma_max: Option<f32>,
    #[serde(default)]
    pub clip_grad_norm: Option<f64>,
    // Nested groups accepted in YAML; hydrated into flat fields on load
    #[serde(default)]
    pub paths_nested: Option<PathsNested>,
    #[serde(default)]
    pub data_nested: Option<DataNested>,
    #[serde(default)]
    pub lora_nested: Option<LoraNested>,
    #[serde(default)]
    pub train_nested: Option<TrainNested>,
    #[serde(default)]
    pub schedule_nested: Option<ScheduleNested>,
    // SDXL-specific paths
    #[serde(default)]
    pub sdxl_unet_path: Option<String>,
    #[serde(default)]
    pub sdxl_vae_path: Option<String>,
    #[serde(default)]
    pub sdxl_clip_l_path: Option<String>,
    #[serde(default)]
    pub sdxl_clip_g_path: Option<String>,
    #[serde(default)]
    pub sdxl_clip_l_tokenizer: Option<String>,
    #[serde(default)]
    pub sdxl_clip_g_tokenizer: Option<String>,
    // SD3.5-specific paths
    #[serde(default)]
    pub sd35_model_path: Option<String>,
    #[serde(default)]
    pub sd35_vae_path: Option<String>,
    #[serde(default)]
    pub sd35_clip_l_path: Option<String>,
    #[serde(default)]
    pub sd35_clip_g_path: Option<String>,
    #[serde(default)]
    pub sd35_t5_path: Option<String>,
    #[serde(default)]
    pub sd35_clip_l_tokenizer: Option<String>,
    #[serde(default)]
    pub sd35_clip_g_tokenizer: Option<String>,
    #[serde(default)]
    pub sd35_t5_tokenizer: Option<String>,

    // Mode gates (baseline-only for now)
    #[serde(default)]
    pub mode: ModeCfg,
}

#[derive(Debug, Clone)]
pub struct ModelBundle {
    pub name: String,
}

pub fn build_model(cfg: &TrainCfg) -> Result<ModelBundle> {
    match cfg.model_kind {
        ModelKind::Flux => models::flux_loader::load_flux(cfg),
        ModelKind::SDXL => models::sdxl_loader::load_sdxl(cfg),
        ModelKind::SD3 => models::sd3_loader::load_sd3(cfg),
        ModelKind::QwenImageV1 => models::qwen_image_loader::load_qwen_image(cfg),
        ModelKind::Wan22 => models::wan22_loader::load_wan22(cfg),
        ModelKind::HiDreamI1 => models::hidream_loader::load_hidream(cfg),
        ModelKind::Omnigen2 => models::omnigen2_loader::load_omnigen2(cfg),
        ModelKind::Ltx => models::ltx_loader::load_ltx(cfg),
        ModelKind::Hunyuan => models::hunyuan_loader::load_hunyuan(cfg),
        ModelKind::Lumina => models::lumina_loader::load_lumina(cfg),
    }
}

fn default_steps() -> usize {
    10
}
fn default_batch_size() -> usize {
    1
}
fn default_grad_accum_steps() -> usize {
    1
}
fn default_lr() -> f64 {
    1e-4
}
fn default_seed() -> u64 {
    1337
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModeCfg {
    #[serde(default = "ModeCfg::default_baseline")]
    pub baseline: bool,
    #[serde(default)]
    pub tread: bool,
    #[serde(default)]
    pub srpo: bool,
}

impl Default for ModeCfg {
    fn default() -> Self {
        Self { baseline: true, tread: false, srpo: false }
    }
}

impl ModeCfg {
    fn default_baseline() -> bool {
        true
    }
}

impl TrainCfg {
    /// Hydrate optional nested groups into flat fields while preserving explicit flat values.
    pub fn hydrate_nested(&mut self) {
        if let Some(p) = &self.paths_nested {
            if self.flux_model_path.is_none() && !p.flux_weights.is_empty() {
                self.flux_model_path = Some(p.flux_weights.clone());
            }
            if self.flux_vae_path.is_none() && !p.vae.is_empty() {
                self.flux_vae_path = Some(p.vae.clone());
            }
            if self.flux_t5_path.is_none() && !p.t5.is_empty() {
                self.flux_t5_path = Some(p.t5.clone());
            }
            if self.flux_clip_path.is_none() && !p.clip.is_empty() {
                self.flux_clip_path = Some(p.clip.clone());
            }
            if self.flux_t5_tokenizer.is_none() && !p.t5_tokenizer.is_empty() {
                self.flux_t5_tokenizer = Some(p.t5_tokenizer.clone());
            }
            if self.flux_clip_tokenizer.is_none() && !p.clip_tokenizer.is_empty() {
                self.flux_clip_tokenizer = Some(p.clip_tokenizer.clone());
            }
        }
        if let Some(d) = &self.data_nested {
            if self.data_root.is_none() && !d.root.is_empty() {
                self.data_root = Some(d.root.clone());
            }
            if self.batch_size == default_batch_size() && d.batch_size > 0 {
                self.batch_size = d.batch_size;
            }
            if self.prefetch.is_none() && d.prefetch > 0 {
                self.prefetch = Some(d.prefetch);
            }
            if self.flux_manifest.is_none() && !d.manifest.is_empty() {
                self.flux_manifest = Some(d.manifest.clone());
            }
            if self.flux_manifest_root.is_none() && !d.manifest_root.is_empty() {
                self.flux_manifest_root = Some(d.manifest_root.clone());
            }
            if self.flux_manifest_index.is_none() && !d.manifest_index.is_empty() {
                self.flux_manifest_index = Some(d.manifest_index.clone());
            }
            if self.flux_manifest_validate.is_none() {
                self.flux_manifest_validate = Some(d.manifest_validate);
            }
            if self.flux_manifest_shuffle.is_none() {
                self.flux_manifest_shuffle = d.manifest_shuffle;
            }
            if self.flux_manifest_repeat.is_none() {
                self.flux_manifest_repeat = d.manifest_repeat;
            }
            if self.sdxl_manifest.is_none() && !d.manifest_sdxl.is_empty() {
                self.sdxl_manifest = Some(d.manifest_sdxl.clone());
            }
            if self.sdxl_manifest_root.is_none() && !d.manifest_sdxl_root.is_empty() {
                self.sdxl_manifest_root = Some(d.manifest_sdxl_root.clone());
            }
            if self.sdxl_manifest_index.is_none() && !d.manifest_sdxl_index.is_empty() {
                self.sdxl_manifest_index = Some(d.manifest_sdxl_index.clone());
            }
            if self.sdxl_manifest_validate.is_none() {
                self.sdxl_manifest_validate = Some(d.manifest_sdxl_validate);
            }
            if self.sdxl_manifest_shuffle.is_none() {
                self.sdxl_manifest_shuffle = d.manifest_sdxl_shuffle;
            }
            if self.sdxl_manifest_repeat.is_none() {
                self.sdxl_manifest_repeat = d.manifest_sdxl_repeat;
            }
            if self.sd35_manifest.is_none() && !d.manifest_sd35.is_empty() {
                self.sd35_manifest = Some(d.manifest_sd35.clone());
            }
            if self.sd35_manifest_root.is_none() && !d.manifest_sd35_root.is_empty() {
                self.sd35_manifest_root = Some(d.manifest_sd35_root.clone());
            }
            if self.sd35_manifest_index.is_none() && !d.manifest_sd35_index.is_empty() {
                self.sd35_manifest_index = Some(d.manifest_sd35_index.clone());
            }
            if self.sd35_manifest_validate.is_none() {
                self.sd35_manifest_validate = Some(d.manifest_sd35_validate);
            }
            if self.sd35_manifest_shuffle.is_none() {
                self.sd35_manifest_shuffle = d.manifest_sd35_shuffle;
            }
            if self.sd35_manifest_repeat.is_none() {
                self.sd35_manifest_repeat = d.manifest_sd35_repeat;
            }
        }
        if let Some(l) = &self.lora_nested {
            if self.lora_rank.is_none() {
                self.lora_rank = Some(l.rank);
            }
            if self.lora_alpha.is_none() {
                self.lora_alpha = Some(l.alpha.unwrap_or(l.rank as f32));
            }
            if self.lora_zero_init.is_none() {
                self.lora_zero_init = Some(l.zero_init.unwrap_or(true));
            }
        }
        if let Some(tn) = &self.train_nested {
            if self.steps == default_steps() && tn.steps > 0 {
                self.steps = tn.steps;
            }
            if self.grad_accum_steps == default_grad_accum_steps() && tn.grad_accum_steps > 0 {
                self.grad_accum_steps = tn.grad_accum_steps;
            }
            if self.lr == default_lr() && tn.lr.unwrap_or(0.0) > 0.0 {
                self.lr = tn.lr.unwrap_or(self.lr);
            }
            if self.ckpt_every.is_none() && tn.ckpt_every > 0 {
                self.ckpt_every = Some(tn.ckpt_every);
            }
            if !self.bf16_params && tn.bf16 {
                self.bf16_params = true;
            }
            if self.clip_grad_norm.is_none() {
                self.clip_grad_norm = tn.clip_grad_norm;
            }
        }
        if let Some(s) = &self.schedule_nested {
            if self.schedule_name.is_empty() {
                self.schedule_name = s.name.clone();
            }
            if self.schedule_steps == 0 {
                self.schedule_steps = s.steps;
            }
            if self.sigma_min.is_none() {
                self.sigma_min = s.sigma_min;
            }
            if self.sigma_max.is_none() {
                self.sigma_max = s.sigma_max;
            }
        }
    }
    pub fn validate_modes(&self) -> anyhow::Result<()> {
        let m = &self.mode;
        let count = (m.baseline as u8) + (m.tread as u8) + (m.srpo as u8);
        if count != 1 {
            return Err(anyhow::anyhow!(
                "config.mode must set exactly one of baseline|tread|srpo to true"
            ));
        }
        Ok(())
    }
    /// Minimal schema guard for baseline trainer config
    pub fn validate(&self) -> eridiffusion_core::Result<()> {
        use eridiffusion_core::Error;
        if self.batch_size == 0 {
            return Err(Error::InvalidInput("batch_size must be > 0".into()));
        }
        if self.steps == 0 {
            return Err(Error::InvalidInput("steps must be > 0".into()));
        }
        let lr = self.lr;
        if !(1e-7..1e-2).contains(&lr) {
            return Err(Error::InvalidInput(format!(
                "lr out of range: {} (expected 1e-7..1e-2)",
                lr
            )));
        }
        Ok(())
    }

    /// Quality-of-life normalization: for Flux model, allow generic `weights` to satisfy
    /// `flux_model_path`. If both are provided and differ, raise a clear error.
    pub fn apply_flux_path_fallback(&mut self) -> Result<()> {
        match self.model_kind {
            ModelKind::Flux => {
                match (&self.flux_model_path, &self.weights) {
                    (None, Some(w)) => {
                        self.flux_model_path = Some(w.clone());
                    }
                    (Some(a), Some(b)) if a != b => {
                        return Err(Error::InvalidInput(format!(
                            "Flux config conflict: both flux_model_path and weights set differently:\n  flux_model_path={}\n  weights={}", a, b
                        )));
                    }
                    _ => {}
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }
}

/// Build an orchestrator for the given config (SDXL/SD35/Flux only)
#[cfg(all(feature = "real", any(feature = "sdxl", feature = "sd35")))]
pub fn build_orchestrator(cfg: &TrainCfg) -> anyhow::Result<Orchestrator<'static>> {
    let adapter: Box<dyn ModelAdapter> = match cfg.model_kind {
        ModelKind::SDXL => Box::new(crate::adapters::sdxl::SdxlAdapter::new(cfg)?),
        ModelKind::SD3 => Box::new(crate::adapters::sd35::Sd35Adapter::new(cfg)?),
        ModelKind::Flux => Box::new(crate::adapters::flux::FluxAdapter::new(cfg)?),
        _ => return Err(anyhow::anyhow!("Unsupported model_kind for orchestrator")),
    };
    Ok(Orchestrator::new(adapter, cfg))
}

#[cfg(not(all(feature = "real", any(feature = "sdxl", feature = "sd35"))))]
pub fn build_orchestrator(_cfg: &TrainCfg) -> anyhow::Result<()> {
    Err(anyhow::anyhow!("orchestrator not available in synthetic build"))
}
