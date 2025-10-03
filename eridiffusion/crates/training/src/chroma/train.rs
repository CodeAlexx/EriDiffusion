use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Arc,
};

use common_ema::EMAModel;
use ctrlc;
use eridiffusion_common_text::{
    clip_l::ClipL as ClipLEmb, openclip_g::OpenClipG as OpenClipEmb, HfTokenizer,
};
use eridiffusion_common_vae::{VaeKind, VaePolicy, VaeSpec};
use eridiffusion_common_weights::strict_loader::StrictMmapLoader;
use eridiffusion_core::{Device as EriDevice, Error, Result};
use eridiffusion_data::image_dataset::{DatasetConfig as ImgDsCfg, ImageDataset};
use eridiffusion_models::devtensor::{randn_on, randn_scaled_on, shape2, shape3, zeros_on};
use flame_core::{CudaDevice, DType, Device as FlameDevice, Parameter, Shape, Tensor};
use safetensors::Dtype as SafeDtype;

// use common_ema::ema::EMAContext; // replaced by EMAHelper
use crate::chroma::keymap::ChromaKeyMap;
use crate::{
    checkpoint_safetensors::{serialize_tensors, tensor_to_bf16_bytes, OwnedTensorView},
    chroma::{
        lora::LoraHandles,
        registry::{Cond, LayerRegistry},
        weights::MmapWeightProvider,
    },
    flux::data_loader::{FluxDataLoader, FluxPrecomputedCfg},
    mixed_precision::{GradScaler, MixedPrecisionConfig},
    optimizer::{
        self, clip_grads_global_norm_fp32_tensors, Optimizer, OptimizerConfig, OptimizerType,
    },
    policy,
    streaming::WeightProvider,
};

#[inline]
fn guard(tag: &str, t: &Tensor) -> Result<()> {
    // Use tensor-level tripwire (no host staging unless TRIPWIRES=1)
    use eridiffusion_core::TensorDebugExt;
    t.debug_check(tag)
}
#[inline]
fn is_finite_tensor(t: &Tensor) -> bool {
    if let Ok(v) = t.to_vec() {
        v.iter().all(|x| x.is_finite())
    } else {
        false
    }
}

fn zero_if_non_finite_(g: &mut Tensor) -> bool {
    if let Ok(v) = g.to_dtype(DType::F32).and_then(|u| u.to_vec()) {
        if v.iter().any(|x| !x.is_finite()) {
            if let Ok(z) = g.zeros_like() {
                *g = z;
            }
            return true;
        }
    }
    false
}

fn find_latest_ema_file(dir: &Path) -> Option<PathBuf> {
    let mut best: Option<(u64, PathBuf)> = None;
    if let Ok(rd) = std::fs::read_dir(dir) {
        for entry in rd.flatten() {
            let p = entry.path();
            let name = match p.file_name().and_then(|s| Some(s.to_string_lossy().to_string())) {
                Some(s) => s,
                None => continue,
            };
            if !name.starts_with("lora_ema_step") || !name.ends_with(".safetensors") {
                continue;
            }
            if let Some(num_str) =
                name.strip_prefix("lora_ema_step").and_then(|s| s.strip_suffix(".safetensors"))
            {
                let ns = num_str.trim_start_matches('0');
                if let Ok(num) = if ns.is_empty() { Ok(0u64) } else { ns.parse::<u64>() } {
                    if best.as_ref().map(|(n, _)| num > *n).unwrap_or(true) {
                        best = Some((num, p));
                    }
                }
            }
        }
    }
    best.map(|(_, p)| p)
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct OptimizerCfg {
    pub name: String,
    pub lr: f32,
    pub betas: (f32, f32),
    pub weight_decay: f32,
    pub lr_if_contains: Option<HashMap<String, f32>>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct TrainCfg {
    pub batch_size: usize,
    pub steps: usize,
    pub optimizer: OptimizerCfg,
    #[serde(default)]
    pub change_layer_every: Option<usize>,
    #[serde(default)]
    pub trained_single_blocks: Option<usize>,
    #[serde(default)]
    pub trained_double_blocks: Option<usize>,
    #[serde(default)]
    pub warmup_steps: Option<usize>,
    #[serde(default)]
    pub offload_param_count: Option<usize>,
}
#[derive(Debug, Clone, serde::Deserialize)]
pub struct IO {
    pub out_dir: String,
    pub save_every: usize,
}
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ModelStreaming {
    pub shard_path: String,
}
#[derive(Debug, Clone, serde::Deserialize)]
pub struct AdapterCfg {
    pub rank: usize,
    pub alpha: f32,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub train: TrainCfg,
    pub io: IO,
    pub model: ModelStreaming,
    pub adapters: AdapterCfg,
    #[serde(default)]
    pub run: Option<RunCfg>,
    #[serde(default)]
    pub scheduler: Option<SchedCfg>,
    #[serde(default)]
    pub checkpoint: Option<CheckpointCfg>,
    #[serde(default)]
    pub ema: Option<EmaCfg>,
    #[serde(default)]
    pub nan_safety: Option<NanSafetyCfg>,
    #[serde(default)]
    pub noise: Option<NoiseCfg>,
    #[serde(default)]
    pub precomputed: Option<PrecomputedCfg>,
}

#[derive(Debug, Clone, Default, serde::Deserialize)]
pub struct RunCfg {
    pub seed: Option<u64>,
    pub deterministic: Option<bool>,
    pub val_every: Option<usize>,
    pub max_blocks_per_step: Option<usize>,
}

#[derive(Debug, Clone, Default, serde::Deserialize)]
pub struct SchedCfg {
    pub kind: Option<String>,
    pub warmup: Option<u32>,
    pub total: Option<u32>,
}

#[derive(Debug, Clone, Default, serde::Deserialize)]
pub struct CheckpointCfg {
    pub resume: Option<String>,
    pub save_ema: Option<bool>,
}

#[derive(Debug, Clone, Default, serde::Deserialize)]
pub struct EmaCfg {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub decay: Option<f32>,
    #[serde(default)]
    pub half_life_steps: Option<usize>,
    #[serde(default)]
    pub window_size: Option<usize>,
    #[serde(default = "default_dtype")]
    pub dtype: String,
    #[serde(default = "default_true")]
    pub use_bias_correction: bool,
    #[serde(default = "one_f32")]
    pub power: f32,
}
fn default_dtype() -> String {
    "bf16".into()
}
fn default_true() -> bool {
    true
}
fn one_f32() -> f32 {
    1.0
}

#[derive(Debug, Clone, Default, serde::Deserialize)]
pub struct NoiseCfg {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "noise_min")]
    pub sigma_min: f32,
    #[serde(default = "noise_max")]
    pub sigma_max: f32,
}
fn noise_min() -> f32 {
    1.0e-2
}
fn noise_max() -> f32 {
    1.0
}

#[derive(Debug, Clone, Default, serde::Deserialize)]
pub struct PrecomputedCfg {
    pub manifest: String,
    #[serde(default)]
    pub root: Option<String>,
    #[serde(default)]
    pub index: Option<String>,
    #[serde(default = "default_true")]
    pub validate: bool,
    #[serde(default = "default_true")]
    pub shuffle: bool,
    #[serde(default)]
    pub repeat: Option<usize>,
}

fn choose_ema_decay(cfg: &EmaCfg) -> f32 {
    if let Some(d) = cfg.decay {
        return d;
    }
    // Approximate: half-life h → decay beta = 0.5^(1/h)
    if let Some(n) = cfg.half_life_steps {
        return 0.5_f32.powf(1.0f32 / n.max(1) as f32);
    }
    // Window w → beta ≈ (w-1)/w
    if let Some(w) = cfg.window_size {
        return ((w.max(1) as f32) - 1.0f32) / (w.max(1) as f32);
    }
    0.999
}

#[derive(Debug, Clone, Default, serde::Deserialize)]
pub struct GroupBudget {
    pub attn: Option<f32>,
    pub mlp: Option<f32>,
    pub emb: Option<f32>,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BudgetAction {
    SkipStep,
    ClipToBudget,
}

impl Default for BudgetAction {
    fn default() -> Self {
        BudgetAction::SkipStep
    }
}

#[derive(Debug, Clone, Default, serde::Deserialize)]
pub struct NanSafetyCfg {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "one_f32")]
    pub grad_clip_norm: f32,
    #[serde(default)]
    pub loss_clip: f32,
    #[serde(default = "half_f32")]
    pub lr_backoff: f32,
    #[serde(default = "half_f32")]
    pub scaler_backoff: f32,
    #[serde(default)]
    pub reset_opt_on_nan: bool,
    #[serde(default)]
    pub zero_bad_grads: bool,
    #[serde(default)]
    pub exclude_wd_on: Vec<String>,
    #[serde(default)]
    pub group_grad_norm_budget: Option<GroupBudget>,
    #[serde(default = "default_budget_action")]
    pub on_budget_violation: BudgetAction,
}
fn half_f32() -> f32 {
    0.5
}
fn default_budget_action() -> BudgetAction {
    BudgetAction::SkipStep
}

struct LRSched {
    base: f32,
    warmup: u32,
    total: u32,
    step: u32,
    kind: String,
}
impl LRSched {
    fn new(base: f32, warmup: u32, total: u32, kind: String) -> Self {
        Self { base, warmup, total, step: 0, kind }
    }
    fn lr(&self) -> f32 {
        let s = self.step.min(self.total);
        if s < self.warmup && self.warmup > 0 {
            self.base * (s as f32 / self.warmup as f32)
        } else {
            match self.kind.as_str() {
                "linear" => {
                    let t = (s.saturating_sub(self.warmup)) as f32
                        / (self.total.saturating_sub(self.warmup).max(1)) as f32;
                    self.base * (1.0 - t)
                }
                _ => {
                    // cosine
                    let t = (s.saturating_sub(self.warmup)) as f32
                        / (self.total.saturating_sub(self.warmup).max(1)) as f32;
                    let cs = 0.5 * (1.0 + (std::f32::consts::PI * (1.0 - t)).cos());
                    self.base * cs
                }
            }
        }
    }
    fn step(&mut self) {
        self.step = self.step.saturating_add(1);
    }
    fn backoff(&mut self, factor: f32) {
        self.base = (self.base * factor).max(1e-8);
    }
}

struct LayerRotator {
    every: usize,
    cur: usize,
    n_double: usize,
}
impl LayerRotator {
    fn new(every: usize, n_double: usize) -> Self {
        Self { every, cur: 0, n_double }
    }
    fn apply(&mut self, step: usize, lora: &mut LoraHandles) {
        if self.every == 0 {
            return;
        }
        if step % self.every != 0 {
            return;
        }
        let n = lora.blocks_len().max(1);
        self.cur = (self.cur + 1) % n;
        lora.set_all_trainable(false);
        for i in 0..self.n_double.max(1) {
            let idx = (self.cur + i) % n;
            lora.set_block_trainable(idx, true);
        }
    }
}

pub fn train_loop(cfg: &Config) -> Result<()> {
    // Device init per guidelines: Arc<CudaDevice> + wrappers where needed
    let cuda: Arc<CudaDevice> =
        CudaDevice::new(0).map_err(|e| Error::Device(format!("CUDA device: {}", e)))?;
    let flame_dev: FlameDevice = FlameDevice::from(cuda.clone());
    let eri_dev: EriDevice = EriDevice::Cuda(0);
    policy::assert_gpu_only(&eri_dev)?;

    // Optional: prewarm key CUDA kernels to avoid first-usage JIT stalls
    if std::env::var("PREWARM_KERNELS").ok().as_deref() == Some("1") {
        let x = randn_on(shape2(4, 4), &eri_dev, DType::F32, Some(0))?;
        let _ = x.gelu()?; // GELU forward kernel
        let _ = x.tanh()?; // Tanh activation
        let _ = x.exp()?; // Exp kernel
        let w = randn_scaled_on(shape2(4, 4), &eri_dev, DType::F32, 0.0, 0.1, Some(1))?;
        let _ = x.matmul(&w)?; // GEMM path
        let _ = x.sum_dim_keepdim(1)?; // Reduction keepdim
                                       // A tiny backward to populate tape and exercise kernels
        let y = x.gelu()?;
        let l = y.square()?.sum()?;
        let _ = flame_core::autograd::backward(&l, false);
        flame_core::autograd::AutogradContext::clear();
        println!("[prewarm] kernels warmed");
    }

    // Seeding & determinism
    let seed = cfg.run.as_ref().and_then(|r| r.seed).unwrap_or_else(|| {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u64
    });
    eridiffusion_core::seed_all(seed)?;
    // Device-level hook (no-op today)
    flame_dev.set_seed(seed).map_err(|e| Error::Device(format!("seed: {}", e)))?;
    println!("[seed] {}", seed);

    // Real dataset wrapper using ImageDataset
    struct Batch {
        latent_image: Tensor,
        latent_mask: Option<Tensor>,
        text_hidden: Tensor,
    }
    struct Dataset {
        inner: ImageDataset,
        idx: usize,
        device: EriDevice,
    }
    impl Dataset {
        fn new(
            root: &std::path::Path,
            caption_ext: &str,
            resolution: usize,
            device: EriDevice,
        ) -> Result<Self> {
            let cfg = ImgDsCfg {
                root_dir: root.to_path_buf(),
                caption_ext: caption_ext.to_string(),
                resolution,
                center_crop: true,
                random_flip: true,
            };
            let inner = ImageDataset::new(cfg, device.clone())?;
            Ok(Self { inner, idx: 0, device })
        }
        fn next_batch(
            &mut self,
            bs: usize,
        ) -> Result<(Tensor /*images NHWC BF16*/, Vec<String>)> {
            let len = self.inner.len();
            let mut images: Vec<Tensor> = Vec::with_capacity(bs);
            let mut caps: Vec<String> = Vec::with_capacity(bs);
            let mut sample_paths: Vec<String> = Vec::new();
            let start_idx = self.idx;
            for _ in 0..bs {
                let item = self.inner.get_item(self.idx % len)?;
                self.idx = (self.idx + 1) % len;
                if let Some(v) = item.metadata.get("image_path") {
                    sample_paths.push(v.to_string());
                }
                images.push(item.images);
                if let Some(c) = item.captions.get(0) {
                    caps.push(c.clone());
                } else {
                    caps.push(String::new());
                }
            }
            // Stack to [B,C,H,W]
            let chw = Tensor::stack(&images, 0)?;
            // NHWC and BF16 for VAE
            let nhwc = chw.permute(&[0, 2, 3, 1])?.to_dtype(DType::BF16)?;
            let io_logs: bool = std::env::var("IO_LOGS").ok().map(|v| v != "0").unwrap_or(true);
            if io_logs {
                // Print a richer preview of the current batch
                let preview_k = std::env::var("PRINT_BATCH_SAMPLES_K")
                    .ok()
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(8);
                let preview = sample_paths
                    .iter()
                    .take(preview_k)
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                println!(
                    "[data] batch idx={}..{} bs={} files=[{}]",
                    start_idx, self.idx, bs, preview
                );
            }
            Ok((nhwc, caps))
        }
    }

    // --- Helpers: text encode and VAE encode ---
    fn encode_text_clip(
        device: &EriDevice,
        tokenizer: &HfTokenizer,
        encoder_kind: &str,
        encoder_clip: Option<&ClipLEmb>,
        encoder_openclip: Option<&OpenClipEmb>,
        captions: &[String],
    ) -> Result<Tensor> {
        let (ids, _lengths, _pad) = tokenizer
            .encode_batch(captions)
            .map_err(|e| Error::Training(format!("tokenize: {}", e)))?;
        // ids: I32 [B,seq] on cuda:0 by tokenizer implementation
        let hidden = match encoder_kind {
            "openclip_g" => {
                let enc = encoder_openclip
                    .ok_or_else(|| Error::Training("openclip encoder not loaded".into()))?;
                enc.forward(&ids).map_err(|e| Error::Training(e.to_string()))?
            }
            _ => {
                let enc = encoder_clip
                    .ok_or_else(|| Error::Training("clip-l encoder not loaded".into()))?;
                enc.forward(&ids).map_err(|e| Error::Training(e.to_string()))?
            }
        };
        // Ensure [B,77,1024]: pad or narrow along last dim as needed
        let dims = hidden.shape().dims().to_vec();
        let (b, seq, d) = (dims[0], dims[1], dims[2]);
        let target_seq = 77usize;
        let target_d = 1024usize;
        let h_seq = if seq != target_seq {
            let take = seq.min(target_seq);
            let mut h = hidden.narrow(1, 0, take)?;
            if take < target_seq {
            let pad = zeros_on(
                    shape3(b as i64, (target_seq - take) as i64, d as i64),
                    device,
                    DType::BF16,
                )?;
                h = Tensor::cat(&[&h, &pad], 1)?;
            }
            h
        } else {
            hidden
        };
        let d_now = h_seq.shape().dims()[2];
        let out = if d_now == target_d {
            h_seq
        } else if d_now > target_d {
            h_seq.narrow(2, 0, target_d)?
        } else {
            let pad = zeros_on(
                shape3(b as i64, target_seq as i64, (target_d - d_now) as i64),
                device,
                DType::BF16,
            )?;
            Tensor::cat(&[&h_seq, &pad], 2)?
        };
        Ok(out)
    }

    fn encode_images_to_latents(
        vae_spec: &VaeSpec,
        images_nhwc_bf16: &Tensor,
        step: usize,
    ) -> Result<Tensor> {
        let mut latents =
            eridiffusion_common_vae::encode(vae_spec, images_nhwc_bf16, VaePolicy::GpuFirst)
                .map_err(|e| Error::Training(format!("vae encode: {}", e)))?; // [B,H',W',C]
                                                                              // Apply latent scale in FP32 for numerical stability
        let scale = vae_spec.latent_scale;
        if (scale - 1.0f32).abs() > 1e-8 {
            latents = latents.to_dtype(DType::F32)?.mul_scalar(scale)?;
        } else {
            latents = latents.to_dtype(DType::F32)?;
        }
        // Tripwire: print mean/std for first steps
        if step <= 20 {
            let l32 = &latents;
            let numel = l32.shape().elem_count() as f32;
            let mean = l32.sum()?.div_scalar(numel)?;
            let mean_v = mean.to_dtype(DType::F32)?.to_vec().unwrap_or_else(|_| vec![f32::NAN]);
            let m = *mean_v.get(0).unwrap_or(&f32::NAN);
            let sq_mean = l32.square()?.sum()?.div_scalar(numel)?;
            let sq_v = sq_mean.to_dtype(DType::F32)?.to_vec().unwrap_or_else(|_| vec![f32::NAN]);
            let m2 = *sq_v.get(0).unwrap_or(&f32::NAN);
            let s = (m2 - m * m).max(0.0).sqrt();
            println!("[vae] step={} lat mean={:.3e} std={:.3e} scale={:.6}", step, m, s, scale);
        }
        Ok(latents)
    }

    #[inline]
    fn latents_to_btd_with_pad(latents_bhwc_f32: &Tensor, d: usize) -> Result<Tensor> {
        // Flatten spatial and channel dims, pad to a multiple of d, then reshape to [B, T, d]
        let dims = latents_bhwc_f32.shape().dims().to_vec();
        let (b, h, w, c) = (dims[0], dims[1], dims[2], dims[3]);
        let total = h * w * c;
        let seq = (total + d - 1) / d; // ceil division
        let need = seq * d - total;
        let flat = latents_bhwc_f32.reshape(&[b, total])?; // [B, total]
        let padded = if need > 0 {
            let pad = Tensor::zeros_dtype(
                Shape::from_dims(&[b, need]),
                DType::F32,
                latents_bhwc_f32.device().clone(),
            )?;
            Tensor::cat(&[&flat, &pad], 1)?
        } else {
            flat
        };
        Ok(padded.reshape(&[b, seq, d])?)
    }

    let reg = LayerRegistry::new();
    // Perf/safety knobs via env
    let sync_every: usize =
        std::env::var("SYNC_EVERY_N_BLOCKS").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let io_logs: bool = std::env::var("IO_LOGS").ok().map(|v| v != "0").unwrap_or(true);
    let km_blocks = <ChromaKeyMap as crate::streaming::KeyMap>::block_count();
    let mut lora = LoraHandles::new(
        km_blocks,
        1024,
        1024,
        cfg.adapters.rank,
        cfg.adapters.alpha,
        flame_dev.clone(),
    )?;
    // Ensure we start with trainable LoRA params; rotator will narrow later if configured
    lora.set_all_trainable(true);
    // Diagnostic: count trainable LoRA params before building optimizer
    {
        let named = lora.named_parameters();
        let total = named.len();
        let trainables = named.values().filter(|p| p.requires_grad()).count();
        println!("[diag] trainable LoRA params = {}/{}", trainables, total);
        assert!(trainables > 0, "No trainable params — check set_requires_grad and rotation");
    }
    // LoRA-only grad guard: ensure only adapters carry grads (placeholder)
    let adapter_params = lora.all_params();

    // Optimizer over LoRA adapter params (build tensor views for API)
    let param_view_tensors: Vec<Tensor> =
        adapter_params.iter().filter_map(|p| p.as_tensor().ok()).collect();
    let param_refs: Vec<&Tensor> = param_view_tensors.iter().collect();
    let mut opt_cfg = OptimizerConfig::default();
    // Map name → type (fallback AdamW)
    let opt_ty = match cfg.train.optimizer.name.to_lowercase().as_str() {
        "adam" => OptimizerType::Adam,
        "adamw" => OptimizerType::AdamW,
        "sgd" => OptimizerType::SGD,
        "lion" => OptimizerType::Lion,
        "adafactor" => OptimizerType::AdaFactor,
        "prodigy" | "prodigyopt" => OptimizerType::ProdigyOpt,
        "radam_schedulefree" | "radam" => OptimizerType::RAdamScheduleFree,
        _ => OptimizerType::AdamW,
    };
    opt_cfg.optimizer_type = opt_ty;
    opt_cfg.lr = cfg.train.optimizer.lr as f64;
    opt_cfg.weight_decay = cfg.train.optimizer.weight_decay as f64;
    opt_cfg.beta1 = cfg.train.optimizer.betas.0 as f64;
    opt_cfg.beta2 = cfg.train.optimizer.betas.1 as f64;
    let mut optimizer: Box<dyn Optimizer> =
        optimizer::create_optimizer(opt_cfg.clone(), &param_refs)?;
    // Wire WD exclusion names (interim) if provided
    let names_ordered = lora.param_names_in_order();
    let _ = optimizer.set_param_names(names_ordered);
    if let Some(ns) = &cfg.nan_safety {
        if !ns.exclude_wd_on.is_empty() {
            let _ = optimizer.set_wd_exclusion_patterns(ns.exclude_wd_on.clone());
        }
    }

    // Scheduler (optional)
    let mut sched = {
        let base = cfg.train.optimizer.lr;
        let sc = cfg.scheduler.clone().unwrap_or_default();
        let warmup = cfg.train.warmup_steps.unwrap_or(sc.warmup.unwrap_or(0) as usize) as u32;
        let total = sc.total.unwrap_or(cfg.train.steps as u32);
        let kind = sc.kind.unwrap_or_else(|| "cosine".to_string());
        LRSched::new(base, warmup, total, kind)
    };

    // Optional layer rotation for LoRA blocks
    let mut rot = LayerRotator::new(
        cfg.train.change_layer_every.unwrap_or(0),
        cfg.train.trained_double_blocks.unwrap_or(0),
    );

    // GradScaler (mixed precision) with dynamic scaling enabled by default
    let mut mp_cfg = MixedPrecisionConfig::default();
    mp_cfg.enabled = true;
    let grad_scaler = GradScaler::new(mp_cfg.clone());

    // Local Tokio runtime for GradScaler async calls
    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| Error::Training(format!("tokio runtime: {}", e)))?;

    // SIGINT graceful save
    let out_dir_sig = cfg.io.out_dir.clone();
    let lora_for_sig = Arc::new(lora.all_params());
    let (tx_sig, rx_sig) = std::sync::mpsc::channel();
    let params_for_sig = Arc::clone(&lora_for_sig);
    let _ = ctrlc::set_handler(move || {
        let param_count = params_for_sig.len();
        let _ = tx_sig.send(());
        let _ = println!(
            "[signal] SIGINT received; will save checkpoint and exit after staging {param_count} adapter tensors"
        );
        // fire and forget; train loop handles save and exit
    });
    println!(
        "[signal] prepared {} adapter parameters for emergency checkpoints in {}",
        lora_for_sig.len(),
        out_dir_sig
    );

    // Loader + streaming provider (with diagnostics around file access)
    {
        let p = std::path::Path::new(&cfg.model.shard_path);
        println!("[weights] opening shard: {}", p.display());
        if !p.exists() {
            return Err(Error::Training(format!("[weights] file not found: {}", p.display())));
        }
        match std::fs::metadata(p) {
            Ok(m) => {
                println!(
                    "[weights] shard exists: size={} bytes, modified={:?}",
                    m.len(),
                    m.modified().ok()
                );
            }
            Err(e) => {
                println!("[weights] ERROR: cannot stat shard: {}", e);
            }
        }
    }
    let ld = Arc::new(
        StrictMmapLoader::open(std::path::Path::new(&cfg.model.shard_path)).map_err(|e| {
            Error::Training(format!("[weights] failed to open {}: {}", cfg.model.shard_path, e))
        })?,
    );
    println!("[weights] mmap loader ready");
    let wp: MmapWeightProvider<ChromaKeyMap> =
        MmapWeightProvider::new(ld.clone(), flame_dev.clone());
    println!("[weights] streaming provider ready");

    // Optimizer groups from lr_if_contains (adapters only) + diagnostics for fallback
    let params = lora.all_params();
    let disable_overrides =
        std::env::var("DISABLE_LR_OVERRIDES").ok().map(|v| v == "1").unwrap_or(false);
    let mut groups: Vec<(f32, Vec<Parameter>)> = vec![(cfg.train.optimizer.lr, Vec::new())];
    if !disable_overrides {
        if let Some(map) = &cfg.train.optimizer.lr_if_contains {
            for (_pat, _lr) in map {
                groups.push((*_lr, Vec::new()));
            }
        }
        for (idx, p) in params.iter().enumerate() {
            let key = format!("double_blocks$${}$$", idx % km_blocks); // simple stable mapping, matches YAML patterns
            let mut placed = false;
            if let Some(map) = &cfg.train.optimizer.lr_if_contains {
                for (gi, (pat, _)) in map.iter().enumerate() {
                    if key.contains(pat) {
                        groups[gi + 1].1.push(p.clone());
                        placed = true;
                        break;
                    }
                }
            }
            if !placed {
                groups[0].1.push(p.clone());
            }
        }
        println!(
            "Param LR groups: default={} {}",
            groups[0].1.len(),
            cfg.train
                .optimizer
                .lr_if_contains
                .as_ref()
                .map(|m| format!(", custom={}", m.len()))
                .unwrap_or_default()
        );
    } else {
        for p in params.into_iter() {
            groups[0].1.push(p);
        }
        println!("Param LR groups (overrides disabled): default={}", groups[0].1.len());
    }

    // Additional per-param LR diagnostic and safe fallback if no patterns match
    {
        let base_lr = cfg.train.optimizer.lr;
        let named = lora.named_parameters();
        let pats = cfg.train.optimizer.lr_if_contains.clone().unwrap_or_default();
        let mut hit: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut custom: Vec<(&str, f32)> = Vec::new();
        for (pat, lr) in &pats {
            custom.push((pat.as_str(), *lr));
        }
        let mut per_param_lr: Vec<(String, Parameter, f32)> = Vec::new();
        for (name, p) in &named {
            let mut lr = base_lr;
            for (pat, lr_p) in &custom {
                if name.contains(pat) {
                    lr = *lr_p;
                    hit.insert(name.clone());
                    break;
                }
            }
            per_param_lr.push((name.clone(), p.clone(), lr));
        }
        if hit.is_empty() {
            println!("[lr] no custom patterns matched — using base LR for all params");
        }
        let default_cnt = named.len().saturating_sub(hit.len());
        println!("[diag] LR groups: default_params={}, custom_matches={}", default_cnt, hit.len());
    }

    // Initialize dataset from YAML data.sources[0] (with fallbacks)
    struct DataSources {
        dataset: Option<Dataset>,
        manifest_loader: Option<FluxDataLoader>,
        manifest_stats: Option<crate::flux_preprocessor::DatasetStats>,
    }

    let data_sources = if let Some(pre) = cfg.precomputed.as_ref() {
        let manifest_path = PathBuf::from(&pre.manifest);
        let root_dir = pre.root.as_ref().map(PathBuf::from);
        if std::env::var("IO_LOGS").ok().map(|v| v != "0").unwrap_or(true) {
            println!("[data] manifest mode: {} (root={:?})", manifest_path.display(), root_dir);
        }
        let pre_cfg = FluxPrecomputedCfg {
            manifest_path: manifest_path.clone(),
            root_dir,
            batch_size: cfg.train.batch_size,
            shuffle: pre.shuffle,
            max_epochs: pre.repeat,
            enforce_bf16: true,
            validate_on_load: pre.validate,
            cache_index: pre.index.as_ref().map(PathBuf::from),
            device: eri_dev.clone(),
        };
        let loader = FluxDataLoader::from_precomputed(pre_cfg)?;
        let stats = loader.precomputed_stats();
        if let Some(stats) = stats.as_ref() {
            let to_gb = |bytes: u64| bytes as f64 / (1024.0 * 1024.0 * 1024.0);
            println!(
                "[data] manifest entries={} latents={:.2}GB text={:.2}GB clip={:.2}GB hash={}",
                stats.len,
                to_gb(stats.total_latent_bytes),
                to_gb(stats.total_t5_bytes),
                to_gb(stats.total_clip_bytes),
                stats.manifest_hash.as_deref().unwrap_or("<unknown>"),
            );
        }
        DataSources { dataset: None, manifest_loader: Some(loader), manifest_stats: stats }
    } else {
        let (mut ds_root, ds_caption_ext, ds_res) = {
            // Fallbacks if fields are missing
            let mut folder = std::path::PathBuf::from(".");
            let mut capext = String::from("txt");
            let mut res = 512usize;
            let cfg_path = std::env::var("CHROMA_CONFIG_PATH")
                .unwrap_or_else(|_| "eridiffusion/configs/my_chroma_lora.yaml".to_string());
            if let Ok(txt) = std::fs::read_to_string(&cfg_path) {
                println!("[config] CHROMA_CONFIG_PATH={}", cfg_path);
                if let Ok(doc) = serde_yaml::from_str::<serde_yaml::Value>(&txt) {
                    if let Some(src0) = doc
                        .get("data")
                        .and_then(|d| d.get("sources"))
                        .and_then(|s| s.as_sequence())
                        .and_then(|v| v.get(0))
                    {
                        if let Some(s) = src0.get("folder_path").and_then(|v| v.as_str()) {
                            folder = std::path::PathBuf::from(s);
                        }
                        if let Some(s) = src0.get("caption_ext").and_then(|v| v.as_str()) {
                            capext = s.to_string();
                        }
                        if let Some(n) = src0.get("resolution").and_then(|v| v.as_u64()) {
                            res = n as usize;
                        }
                    } else if let Some(df) =
                        doc.get("data").and_then(|d| d.get("folder_path")).and_then(|v| v.as_str())
                    {
                        folder = std::path::PathBuf::from(df);
                    }
                }
            }
            (folder, capext, res)
        };
        ds_root = ds_root.canonicalize().unwrap_or(ds_root);
        if std::env::var("IO_LOGS").ok().map(|v| v != "0").unwrap_or(true) {
            println!("[dataset] root_dir = {}", ds_root.display());
            println!(
                "[data] init folder={} captions_ext={} res={}",
                ds_root.display(),
                ds_caption_ext,
                ds_res
            );
        }
        let dataset = Dataset::new(&ds_root, &ds_caption_ext, ds_res, eri_dev.clone())
            .map_err(|e| Error::DataError(format!("dataset init: {}", e)))?;
        if io_logs {
            println!("[dataset] total_images={}", dataset.inner.len());
        }
        DataSources { dataset: Some(dataset), manifest_loader: None, manifest_stats: None }
    };

    // Load VAE + Text tokenizer/encoder from YAML (best-effort)
    // Use CHROMA_CONFIG_PATH if provided to stay consistent with CLI selection
    let (vae_spec_opt, text_tok_opt, text_enc_kind, clip_path_opt, openclip_path_opt) = {
        let mut vae_spec: Option<VaeSpec> = None;
        let mut tok_path: Option<String> = None;
        let mut enc_kind = String::from("clip_l");
        let mut clip_p: Option<String> = None;
        let mut openclip_p: Option<String> = None;
        let cfg_path = std::env::var("CHROMA_CONFIG_PATH")
            .unwrap_or_else(|_| "eridiffusion/configs/my_chroma_lora.yaml".to_string());
        if let Ok(txt) = std::fs::read_to_string(&cfg_path) {
            if let Ok(doc) = serde_yaml::from_str::<serde_yaml::Value>(&txt) {
                if let Some(v) =
                    doc.get("vae").and_then(|v| v.get("weights")).and_then(|v| v.as_str())
                {
                    let scale = doc
                        .get("vae")
                        .and_then(|v| v.get("scale"))
                        .and_then(|x| x.as_f64())
                        .unwrap_or(1.0) as f32;
                    // Optional kind: "sdxl" | "flux" | "sd35" (default flux for Chroma/Flux Schnell)
                    let kind_str = doc
                        .get("vae")
                        .and_then(|v| v.get("kind"))
                        .and_then(|x| x.as_str())
                        .unwrap_or("flux");
                    let kind = match kind_str.to_ascii_lowercase().as_str() {
                        "sdxl" => VaeKind::Sdxl,
                        "sd35" => VaeKind::Sd35,
                        _ => VaeKind::Flux,
                    };
                    // Loud guard on typical scales
                    match kind {
                        VaeKind::Sdxl => {
                            if (scale - 0.18215).abs() > 1e-4 {
                                println!("[vae] WARNING: SDXL VAE detected but vae.scale={:.6}. Typical is 0.18215.", scale);
                            }
                        }
                        VaeKind::Flux => {
                            if (scale - 1.0).abs() > 1e-6 {
                                println!("[vae] WARNING: Flux/Chroma VAE expected scale≈1.0 but got {:.6}.", scale);
                            }
                        }
                        VaeKind::Sd35 => {
                            // No strict default; leave informational
                            println!("[vae] Using SD3.5-style VAE; scale={:.6}", scale);
                        }
                    }
                    vae_spec = Some(VaeSpec {
                        kind,
                        path: v.to_string(),
                        latent_div: 8,
                        latent_channels: 4,
                        latent_scale: scale,
                    });
                }
                if let Some(tpath) =
                    doc.get("text").and_then(|t| t.get("tokenizer_path")).and_then(|v| v.as_str())
                {
                    tok_path = Some(tpath.to_string());
                }
                if let Some(kind) =
                    doc.get("text").and_then(|t| t.get("model")).and_then(|v| v.as_str())
                {
                    enc_kind = kind.to_string();
                }
                if let Some(cp) =
                    doc.get("text").and_then(|t| t.get("clip_l_path")).and_then(|v| v.as_str())
                {
                    clip_p = Some(cp.to_string());
                }
                if let Some(op) =
                    doc.get("text").and_then(|t| t.get("openclip_g_path")).and_then(|v| v.as_str())
                {
                    openclip_p = Some(op.to_string());
                }
            }
        }
        (vae_spec, tok_path, enc_kind, clip_p, openclip_p)
    };
    // Text tokenizer/encoders are optional; skip if path is empty or missing
    let tokenizer_opt = if let Some(tp) = &text_tok_opt {
        if tp.trim().is_empty() || !std::path::Path::new(tp).exists() {
            println!("[text] tokenizer path missing/empty → disabled");
            None
        } else {
            Some(
                HfTokenizer::from_path(tp.as_str(), 77)
                    .map_err(|e| Error::Training(e.to_string()))?,
            )
        }
    } else {
        None
    };
    let mut clip_enc: Option<ClipLEmb> = None;
    let mut openclip_enc: Option<OpenClipEmb> = None;
    if let Some(cp) = &clip_path_opt {
        if cp.trim().is_empty() || !std::path::Path::new(cp).exists() {
            println!("[text] clip_l_path missing/empty → disabled");
        } else {
            clip_enc = Some(
                ClipLEmb::from_weights_auto(cp.as_str(), &flame_dev, 77)
                    .map_err(|e| Error::Training(e.to_string()))?,
            );
        }
    }
    if let Some(op) = &openclip_path_opt {
        if op.trim().is_empty() || !std::path::Path::new(op).exists() {
            println!("[text] openclip_g_path missing/empty → disabled");
        } else {
            openclip_enc = Some(
                OpenClipEmb::from_weights_auto(op.as_str(), &flame_dev, 77)
                    .map_err(|e| Error::Training(e.to_string()))?,
            );
        }
    }
    std::fs::create_dir_all(&cfg.io.out_dir).ok();
    // Progress/ETA helpers
    // Resume support
    let mut start_step: usize = 1;
    if let Some(cp) = &cfg.checkpoint {
        let resume_dir = cp
            .resume
            .as_ref()
            .map(|s| std::path::PathBuf::from(s))
            .unwrap_or_else(|| std::path::PathBuf::from(&cfg.io.out_dir));
        let state_path = resume_dir.join("trainer_state.json");
        if state_path.exists() {
            if let Ok(txt) = std::fs::read_to_string(&state_path) {
                if let Ok(doc) = serde_json::from_str::<serde_json::Value>(&txt) {
                    if let Some(s) = doc.get("global_step").and_then(|v| v.as_u64()) {
                        start_step = (s as usize).saturating_add(1);
                        println!(
                            "[resume] starting from step {} (loaded from {})",
                            start_step,
                            state_path.display()
                        );
                    }
                }
            }
        }
    }
    let total_steps = cfg.train.steps as usize;
    let run_start = std::time::Instant::now();
    let mut ema_step_secs: f32 = 0.0; // exponential moving average of step seconds
    let ema_alpha: f32 = 0.1;
    // Shared EMA model (if enabled)
    let mut ema: Option<EMAModel> = if cfg.ema.as_ref().map(|e| e.enabled).unwrap_or(false) {
        let cfg_ema = cfg.ema.as_ref().unwrap();
        let decay = choose_ema_decay(cfg_ema);
        println!("[ema] decay = {:.9}", decay);
        let mut m = common_ema::EMAModel::new(decay, flame_dev.clone());
        let named = lora.named_parameters();
        m.init_from_params(&named).map_err(|e| Error::Training(e.to_string()))?;
        // Try resume EMA state from latest file
        let out_dir = std::path::Path::new(&cfg.io.out_dir);
        if let Some(p) = find_latest_ema_file(out_dir) {
            let _ = m.load_state(&p); // best-effort
        }
        Some(m)
    } else {
        None
    };

    let mut data_sources = data_sources;
    for step in start_step..=cfg.train.steps {
        let iter_start = std::time::Instant::now();
        let (features_3d, text_hidden, captions) =
            if let Some(loader) = data_sources.manifest_loader.as_mut() {
                let batch = loop {
                    match loader.next()? {
                        Some(b) => break b,
                        None => continue,
                    }
                };
                let captions: Vec<String> = if batch.records.is_empty() {
                    let b = batch.latents.shape().dims()[0];
                    vec![String::new(); b]
                } else {
                    batch.records.iter().map(|r| r.caption.clone()).collect()
                };
                let latents_f32 = batch.latents.to_dtype(DType::F32)?;
                let latents_bhwc = latents_f32.permute(&[0, 2, 3, 1])?;
                let features = latents_to_btd_with_pad(&latents_bhwc, 3072usize)?;
                let text_hidden = batch.text_ctx.to_dtype(DType::BF16)?;
                (features, text_hidden, captions)
            } else {
                let dataset = data_sources.dataset.as_mut().expect("dataset unavailable");
                let (images_nhwc_bf16, captions) =
                    dataset.next_batch(cfg.train.batch_size)?;
                let features_3d = if let Some(spec) = &vae_spec_opt {
                    if io_logs {
                        let d = images_nhwc_bf16.shape().dims().to_vec();
                        println!("[vae] encode start NHWC=B{} H{} W{} C{}", d[0], d[1], d[2], d[3]);
                    }
                    let t0 = std::time::Instant::now();
                    let latents = encode_images_to_latents(spec, &images_nhwc_bf16, step)?;
                    if io_logs {
                        let dt = t0.elapsed().as_secs_f32();
                        let d = latents.shape().dims().to_vec();
                        println!(
                            "[vae] encode done -> B{} H{} W{} C{} in {:.3}s",
                            d[0], d[1], d[2], d[3], dt
                        );
                    }
                    latents_to_btd_with_pad(&latents, 3072usize)?
                } else {
                    let dims = images_nhwc_bf16.shape().dims().to_vec();
                    let (b, h, w, c) = (dims[0], dims[1], dims[2], dims[3]);
                    let d = 3072usize;
                    let total = h * w * c;
                    let seq = total / d;
                    images_nhwc_bf16.clone().reshape(&[b, seq, d])?
                };
                let text_hidden = if let (Some(tok), true) =
                    (tokenizer_opt.as_ref(), clip_enc.is_some() || openclip_enc.is_some())
                {
                    if io_logs {
                        println!(
                            "[text] encode start model={} B={} seq=77",
                            text_enc_kind,
                            captions.len()
                        );
                    }
                    let t0 = std::time::Instant::now();
                    let th = encode_text_clip(
                        &eri_dev,
                        tok,
                        &text_enc_kind,
                        clip_enc.as_ref(),
                        openclip_enc.as_ref(),
                        &captions,
                    )?;
                    if io_logs {
                        println!("[text] encode done in {:.3}s", t0.elapsed().as_secs_f32());
                    }
                    th
                } else {
                    zeros_on(shape3(captions.len() as i64, 77, 1024), &eri_dev, DType::BF16)?
                };
                (features_3d, text_hidden, captions)
            };
        debug_assert_eq!(captions.len(), features_3d.shape().dims()[0] as usize);
        if step <= 50 {
            guard("features_3d", &features_3d)?;
        }
        if step <= 50 {
            guard("text_hidden", &text_hidden)?;
        }
        // Choose 3D or 2D based on attention mode
        let attn_mode =
            std::env::var("CHROMA_ATTENTION_MODE").unwrap_or_else(|_| "btd".to_string());
        if io_logs {
            let d = features_3d.shape().dims().to_vec();
            println!("[attn] mode={} features B{} T{} D{}", attn_mode, d[0], d[1], d[2]);
        }
        let z = if attn_mode.to_lowercase() == "btd" {
            features_3d.clone()
        } else {
            let dims = features_3d.shape().dims().to_vec();
            let (b, t, d) = (dims[0], dims[1], dims[2]);
            features_3d.reshape(&[b * t, d])?
        };
        // Noise – bounded scheduler (optional), zeros by default for bring-up
        let nc = cfg.noise.clone().unwrap_or_default();
        let t01 = policy::sample_timesteps01(cfg.train.batch_size, &eri_dev)?; // [B,1]
        let sigma = policy::sigma_for_bounded(&t01, nc.sigma_min, nc.sigma_max)?; // [B,1]
        let z32 = z.to_dtype(DType::F32)?;
        let eps = if nc.enabled {
            randn_on(z.shape().clone(), &eri_dev, DType::F32, None)?
        } else {
            zeros_on(z.shape().clone(), &eri_dev, DType::F32)?
        };
        let zt = policy::add_noise(&z32, &eps, &sigma)?;
        let cond = Cond { text_hidden: text_hidden.clone(), sigma: sigma.clone(), mask_lat: None };
        let mut x = zt.clone();
        // Sequential streaming across all Chroma blocks with debug prints
        let mut count = 0usize;
        for i in reg.forward_ids() {
            // Check for SIGINT between blocks to allow fast, graceful shutdown
            if rx_sig.try_recv().is_ok() {
                // Flush device work to make sure any in-flight ops are done
                let _ = cuda.synchronize();
                // Capture grad scaler state if available
                let gs_pair = {
                    let scale = rt.block_on(async { grad_scaler.get_scale().await });
                    let growth = rt.block_on(async { grad_scaler.get_growth_tracker().await });
                    (scale, growth)
                };
                save_checkpoint(
                    &cfg.io.out_dir,
                    step,
                    &adapter_params,
                    Some(optimizer.state()),
                    ema.as_ref(),
                    Some(gs_pair),
                )?;
                println!(
                    "[signal] checkpoint saved at step {} → {}/lora_step{:04}.safetensors",
                    step, &cfg.io.out_dir, step
                );
                return Ok(());
            }
            if let Some(run) = &cfg.run {
                if let Some(maxb) = run.max_blocks_per_step {
                    if count >= maxb {
                        break;
                    }
                }
            }
            // Prefetch next block (i+1) per guidelines to smooth H2D
            if i + 1 < km_blocks {
                if io_logs {
                    println!("[io] prefetch {}", i + 1);
                }
                wp.prefetch_block(i + 1)?;
            }
            {
                let w_i = wp.load_block_to_gpu(i)?;
                if io_logs {
                    println!("[io] load {} ({} tensors)", i, w_i.tensors.len());
                }
                x = reg.blocks[i].apply(&x, &cond, &w_i, lora.for_block(i))?;
                // Explicitly drop weights for this block before proceeding
                drop(w_i);
            }
            // Release the previous block (i-1) promptly; current was dropped above
            wp.release_block(i as isize - 1)?;
            if step <= 50 {
                guard(&format!("blk{}.out", i).as_str(), &x)?;
            }
            if io_logs {
                println!("[io] release {}", i);
            }
            // Ensure all work and frees complete before next load (every N blocks)
            if (i + 1) % sync_every == 0 {
                cuda.synchronize().map_err(|e| Error::Device(format!("CUDA sync: {}", e)))?;
            }
            // Optional: allocator trim if enabled via env (best-effort placeholder)
            if std::env::var("TRIM_AFTER_BLOCK").ok().as_deref() == Some("1") {
                // No-op placeholder: implement mempool trim if exposed in the engine
            }
            count += 1;
        }
        let pred = x;
        if step <= 50 {
            guard("pred_eps", &pred)?;
        }
        // Fused, broadcast-safe MSE in FP32 on GPU
        let mut loss = policy::masked_eps_loss(&pred, &eps, None)?; // scalar [1]
                                                                    // Guard: rank-0/1 scalar on GPU
        debug_assert!(loss.shape().elem_count() == 1, "loss must be scalar-like");
        if step <= 50 {
            let pshape = pred.shape().dims().to_vec();
            println!(
                "[grad] mse fused pred shape={:?} loss.requires_grad={}",
                pshape,
                loss.requires_grad()
            );
        }
        if step < 5 {
            let ls = loss
                .to_dtype(DType::F32)
                .and_then(|t| t.to_vec())
                .ok()
                .and_then(|v| v.get(0).copied())
                .unwrap_or(f32::NAN);
            println!("[diag] step={} loss={:.3} base_lr={:.2e}", step, ls, cfg.train.optimizer.lr);
        }
        if cfg.nan_safety.as_ref().map(|n| n.enabled).unwrap_or(false) {
            // FP32 scalar read via to_vec()
            if let Ok(v) = loss.to_dtype(DType::F32).and_then(|t| t.to_vec()) {
                let val = *v.get(0).unwrap_or(&f32::NAN);
                // Warm-up loss clip: starts very high, decays to target over N steps
                let target_clip: f32 = cfg
                    .nan_safety
                    .as_ref()
                    .and_then(|n| if n.loss_clip > 0.0 { Some(n.loss_clip) } else { None })
                    .unwrap_or(1.0e8);
                let warmup_n: usize = cfg.train.warmup_steps.unwrap_or(500);
                let loss_clip_now: f32 = if warmup_n > 0 && step <= warmup_n {
                    let t = step as f32 / warmup_n.max(1) as f32;
                    let k = 0.5f32 * (1.0 + (std::f32::consts::PI * t).cos()); // 1→0
                    1.0e12f32 * k + target_clip * (1.0 - k)
                } else {
                    target_clip
                };
                let exceeded = val.abs() > loss_clip_now;
                if !val.is_finite() || exceeded {
                    println!(
                        "[nan] loss={} (clip={}) → skip step, backoff lr/scaler",
                        val, loss_clip_now
                    );
                    let back = cfg.nan_safety.as_ref().map(|n| n.scaler_backoff).unwrap_or(0.5);
                    let cur = rt.block_on(async { grad_scaler.get_scale().await });
                    let _ = rt.block_on(async { grad_scaler.set_scale(cur * back).await });
                    let lrb = cfg.nan_safety.as_ref().map(|n| n.lr_backoff).unwrap_or(0.5);
                    sched.backoff(lrb);
                    continue;
                }
            }
        }

        // Optional: rotate which LoRA layers are trainable this step
        rot.apply(step, &mut lora);

        // Optional: add tiny L2 on LoRA params to sanity-check grad wiring
        if std::env::var("ADD_PARAM_L2").ok().as_deref() == Some("1") {
            let mut reg = zeros_on(loss.shape().clone(), &eri_dev, DType::F32)?;
            for p in adapter_params.iter() {
                if let Ok(t) = p.as_tensor() {
                    reg = reg.add(&t.square()?)?;
                }
            }
            loss = loss.add(&reg.mul_scalar(1e-8)?)?;
        }

        // Optional: exit before backward for quick forward-only sanity
        if std::env::var("ONE_STEP_NO_BACKWARD").ok().as_deref() == Some("1") {
            println!("[one-step] forward complete; skipping backward by request");
            return Ok(());
        }

        // --- Backward + grad collection ---
        // Ensure loss is F32 scalar before scaling/backward
        let loss = loss.to_dtype(DType::F32)?;
        // Allow bypassing scaler if it causes device stalls
        let disable_scaler = std::env::var("DISABLE_SCALER").ok().as_deref() == Some("1");
        // Scale scalar loss for AMP (Flame backward expects a scalar)
        let scaled_loss = if disable_scaler {
            loss.clone()
        } else {
            rt.block_on(async { grad_scaler.scale(&loss).await })?
        };
        // Flush outstanding kernels before backward
        cuda.synchronize().map_err(|e| Error::Device(format!("CUDA sync pre-backward: {}", e)))?;
        // Backward: capture gradient map so we can wire grads back into Parameters
        // Default to debug mode for the first step to surface tape details unless explicitly disabled
        let use_dbg = std::env::var("BACKWARD_DEBUG").ok().map(|v| v == "1").unwrap_or(step <= 1);
        let grad_map = if use_dbg {
            flame_core::autograd::AutogradContext::backward_debug(&scaled_loss)?
        } else {
            flame_core::autograd::backward(&scaled_loss, false)?
        };
        // Debug: how many LoRA params have grads in the map
        {
            let mut hits = 0usize;
            for (_name, p) in lora.named_parameters() {
                if let Ok(t) = p.as_tensor() {
                    if grad_map.get(t.id()).is_some() {
                        hits += 1;
                    }
                }
            }
            println!("[gradmap] hits={}", hits);
        }

        // Collect grads for optimizer, setting Parameter grads from gradient map
        let mut grads: Vec<Tensor> = Vec::with_capacity(adapter_params.len());
        for p in adapter_params.iter() {
            if let Ok(t) = p.as_tensor() {
                if let Some(g) = grad_map.get(t.id()) {
                    // Ensure FP32 grad and set on parameter for any downstream logic
                    let g32 =
                        if g.dtype() != DType::F32 { g.to_dtype(DType::F32)? } else { g.clone() };
                    let _ = p.set_grad(g32.clone());
                    grads.push(g32);
                } else {
                    grads.push(Tensor::zeros_like(&t)?);
                }
            }
        }

        // Unscale + Inf/NaN check with GradScaler
        rt.block_on(async { grad_scaler.unscale_gradients(&mut grads).await })?;
        let ok = rt.block_on(async { grad_scaler.check_gradients(&grads).await })?;
        let mut found_inf_nonmp = false;
        if cfg.nan_safety.as_ref().map(|n| n.enabled && n.zero_bad_grads).unwrap_or(false) {
            for g in grads.iter_mut() {
                if zero_if_non_finite_(g) {
                    found_inf_nonmp = true;
                }
            }
        }
        if !ok || found_inf_nonmp {
            println!("[nan] found_inf → skip optimizer.step, backoff");
            let back = cfg.nan_safety.as_ref().map(|n| n.scaler_backoff).unwrap_or(0.5);
            let cur = rt.block_on(async { grad_scaler.get_scale().await });
            let _ = rt.block_on(async { grad_scaler.set_scale(cur * back).await });
            let lrb = cfg.nan_safety.as_ref().map(|n| n.lr_backoff).unwrap_or(0.5);
            sched.backoff(lrb);
            if cfg.nan_safety.as_ref().map(|n| n.reset_opt_on_nan).unwrap_or(false) {
                let _ = optimizer.set_state(crate::optimizer::OptimizerState::new());
            }
            flame_core::autograd::AutogradContext::clear();
            continue;
        }

        // Optional: one-step grad dump for diagnostics (env: ONE_STEP_GRAD_DUMP=1)
        if std::env::var("ONE_STEP_GRAD_DUMP").ok().as_deref() == Some("1") {
            // Ensure Parameter grads are set (already wired above). Dump norms per param.
            let named = lora.named_parameters();
            let mut entries: Vec<(String, f32)> = Vec::new();
            for (name, p) in named.iter() {
                if let Some(g) = p.grad() {
                    let g32 = if g.dtype() != DType::F32 { g.to_dtype(DType::F32)? } else { g };
                    let n = g32
                        .square()?
                        .sum()?
                        .to_dtype(DType::F32)?
                        .to_vec()
                        .unwrap_or_else(|_| vec![0.0]);
                    let norm = n.get(0).copied().unwrap_or(0.0).sqrt();
                    entries.push((name.clone(), norm));
                } else {
                    entries.push((name.clone(), 0.0));
                }
            }
            entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let topk = std::env::var("ONE_STEP_GRAD_TOPK")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(16);
            println!("[graddump] top {} param norms:", topk);
            for (i, (n, v)) in entries.iter().take(topk).enumerate() {
                println!("  {:02} {:>32} : {:.6}", i, n, v);
            }
            println!(
                "[graddump] nonzero={}/{}",
                entries.iter().filter(|(_, v)| *v > 0.0).count(),
                entries.len()
            );
            // Skip optimizer and exit after a single diagnostic iteration
            flame_core::autograd::AutogradContext::clear();
            return Ok(());
        }

        // Gradient clipping (global L2 in FP32)
        // Mixed-precision reminder: grads/optimizer states remain FP32; params/acts can be BF16.
        debug_assert!(
            grads.iter().all(|g| g.dtype() == DType::F32),
            "optimizer expects FP32 grads"
        );
        let clipn = cfg.nan_safety.as_ref().map(|n| n.grad_clip_norm).unwrap_or(0.0);
        if clipn > 0.0 {
            let _ = clip_grads_global_norm_fp32_tensors(&mut grads, clipn)?;
        }

        // Group grad-norm budgets (optional)
        if let Some(ns) = &cfg.nan_safety {
            if let Some(budgets) = &ns.group_grad_norm_budget {
                // Build a fresh named view for grad access
                let named = lora.named_parameters();
                // Compute per-group norms
                let mut sums: std::collections::HashMap<&'static str, f32> =
                    std::collections::HashMap::new();
                for (name, p) in named.iter() {
                    if let Some(g) = p.grad() {
                        // FP32 norm: sqrt(sum(g^2))
                        let g32 = if g.dtype() != DType::F32 { g.to_dtype(DType::F32)? } else { g };
                        let s = g32.square()?.sum()?;
                        let sv = s.to_dtype(DType::F32)?.to_vec()?;
                        let ss = *sv.get(0).unwrap_or(&0.0);
                        let grp = if name.contains(".attn.") {
                            "attn"
                        } else if name.contains(".mlp.") || name.contains(".fc") {
                            "mlp"
                        } else if name.contains("embed") || name.contains(".emb") {
                            "emb"
                        } else {
                            "other"
                        };
                        *sums.entry(grp).or_insert(0.0) += ss;
                    }
                }
                let norms: std::collections::HashMap<String, f32> =
                    sums.into_iter().map(|(k, v)| (k.to_string(), v.sqrt())).collect();
                let attn_ok = budgets
                    .attn
                    .map(|b| norms.get("attn").copied().unwrap_or(0.0) <= b)
                    .unwrap_or(true);
                let mlp_ok = budgets
                    .mlp
                    .map(|b| norms.get("mlp").copied().unwrap_or(0.0) <= b)
                    .unwrap_or(true);
                let emb_ok = budgets
                    .emb
                    .map(|b| norms.get("emb").copied().unwrap_or(0.0) <= b)
                    .unwrap_or(true);
                let violated = !(attn_ok && mlp_ok && emb_ok);
                if violated {
                    match ns.on_budget_violation {
                        BudgetAction::SkipStep => {
                            println!("[budget] violation → skip optimizer.step; backoff");
                            let back = ns.scaler_backoff;
                            let cur = rt.block_on(async { grad_scaler.get_scale().await });
                            let _ = rt.block_on(async { grad_scaler.set_scale(cur * back).await });
                            sched.backoff(ns.lr_backoff);
                            if ns.reset_opt_on_nan {
                                let _ =
                                    optimizer.set_state(crate::optimizer::OptimizerState::new());
                            }
                            flame_core::autograd::AutogradContext::clear();
                            continue;
                        }
                        BudgetAction::ClipToBudget => {
                            // Compute per-group scale factors
                            let attn_scale = budgets
                                .attn
                                .map(|b| {
                                    let cur = norms.get("attn").copied().unwrap_or(0.0);
                                    if cur > b && cur > 0.0 {
                                        b / cur
                                    } else {
                                        1.0
                                    }
                                })
                                .unwrap_or(1.0);
                            let mlp_scale = budgets
                                .mlp
                                .map(|b| {
                                    let cur = norms.get("mlp").copied().unwrap_or(0.0);
                                    if cur > b && cur > 0.0 {
                                        b / cur
                                    } else {
                                        1.0
                                    }
                                })
                                .unwrap_or(1.0);
                            let emb_scale = budgets
                                .emb
                                .map(|b| {
                                    let cur = norms.get("emb").copied().unwrap_or(0.0);
                                    if cur > b && cur > 0.0 {
                                        b / cur
                                    } else {
                                        1.0
                                    }
                                })
                                .unwrap_or(1.0);

                            // Apply scaling to parameter gradients
                            for (name, p) in named.iter() {
                                if let Some(g) = p.grad() {
                                    let grp = if name.contains(".attn.") {
                                        "attn"
                                    } else if name.contains(".mlp.") || name.contains(".fc") {
                                        "mlp"
                                    } else if name.contains("embed") || name.contains(".emb") {
                                        "emb"
                                    } else {
                                        "other"
                                    };
                                    let scale = match grp {
                                        "attn" => attn_scale,
                                        "mlp" => mlp_scale,
                                        "emb" => emb_scale,
                                        _ => 1.0,
                                    };
                                    if scale < 1.0 {
                                        let g32 = if g.dtype() != DType::F32 {
                                            g.to_dtype(DType::F32)?
                                        } else {
                                            g
                                        };
                                        let scaled = g32.mul_scalar(scale)?;
                                        // Set back on parameter
                                        p.set_grad(scaled)?;
                                    }
                                }
                            }

                            // Rebuild grads vector from updated parameter gradients to keep optimizer inputs in sync
                            grads.clear();
                            for p in adapter_params.iter() {
                                if let Some(g) = p.grad() {
                                    grads.push(g);
                                } else if let Ok(t) = p.as_tensor() {
                                    grads.push(Tensor::zeros_like(&t)?);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Grad diagnostics: global norm and nonzero grad count (first 10 steps and every 100 after)
        if step <= 10 || step % 100 == 0 {
            let mut total_sq: f32 = 0.0;
            let mut nonzero: usize = 0;
            for g in grads.iter() {
                let g32 = if g.dtype() != DType::F32 { g.to_dtype(DType::F32)? } else { g.clone() };
                let s = g32.mul(&g32)?.sum()?;
                let v = s.to_dtype(DType::F32)?.to_vec().unwrap_or_default();
                let ss = *v.get(0).unwrap_or(&0.0);
                if ss > 0.0 {
                    nonzero += 1;
                }
                total_sq += ss;
            }
            let gnorm = total_sq.sqrt();
            println!("[grad] global_norm={:.6} nonzero_grads={}/{}", gnorm, nonzero, grads.len());
            if nonzero == 0 {
                println!(
                    "[grad][warn] all adapter grads are zero — check trainable flags/rotation"
                );
            }
        }

        // Optimizer step
        let lr_now = sched.lr() as f64;
        optimizer.step(&param_refs, &grads, lr_now)?;
        sched.step();

        // EMA update (if configured)
        if let Some(ema) = ema.as_mut() {
            let named = lora.named_parameters();
            ema.update(&named).map_err(|e| Error::Training(e.to_string()))?;
        }

        // Clear grads for next step
        flame_core::autograd::AutogradContext::clear();

        // Update scaler for next iteration
        let _ = rt.block_on(async { grad_scaler.update_scale().await });
        if rx_sig.try_recv().is_ok() {
            let gs_pair = {
                let scale = rt.block_on(async { grad_scaler.get_scale().await });
                let growth = rt.block_on(async { grad_scaler.get_growth_tracker().await });
                (scale, growth)
            };
            save_checkpoint(
                &cfg.io.out_dir,
                step,
                &adapter_params,
                Some(optimizer.state()),
                ema.as_ref(),
                Some(gs_pair),
            )?;
            println!(
                "[signal] checkpoint saved at step {} → {}/lora_step{:04}.safetensors",
                step, &cfg.io.out_dir, step
            );
            return Ok(());
        }
        if step % cfg.io.save_every == 0 {
            let gs_pair = {
                let scale = rt.block_on(async { grad_scaler.get_scale().await });
                let growth = rt.block_on(async { grad_scaler.get_growth_tracker().await });
                (scale, growth)
            };
            save_checkpoint(
                &cfg.io.out_dir,
                step,
                &adapter_params,
                Some(optimizer.state()),
                ema.as_ref(),
                Some(gs_pair),
            )?;
            println!(
                "[chroma] saved checkpoint at step {} → {}/lora_step{:04}.safetensors",
                step, &cfg.io.out_dir, step
            );
        }

        // Optional validation (EMA swap if available)
        if let Some(run) = &cfg.run {
            if let Some(ve) = run.val_every {
                if ve > 0 && step % ve == 0 {
                    if let Some(dataset) = data_sources.dataset.as_mut() {
                        let (images_nhwc_bf16, captions) =
                            dataset.next_batch(cfg.train.batch_size)?;
                        let run_val =
                            |images_nhwc_bf16: &Tensor, captions: &Vec<String>| -> Result<f32> {
                                let features_3d: Tensor = if let Some(spec) = &vae_spec_opt {
                                    let latents =
                                        encode_images_to_latents(spec, images_nhwc_bf16, step)?;
                                    latents_to_btd_with_pad(&latents, 3072usize)?
                                } else {
                                    let dims = images_nhwc_bf16.shape().dims().to_vec();
                                    let (b, h, w, c) = (dims[0], dims[1], dims[2], dims[3]);
                                    let total = h * w * c;
                                    let seq = total / 3072usize;
                                    images_nhwc_bf16.clone().reshape(&[b, seq, 3072usize])?
                                };
                                let attn_mode = std::env::var("CHROMA_ATTENTION_MODE")
                                    .unwrap_or_else(|_| "btd".to_string());
                                let z = if attn_mode.to_lowercase() == "btd" {
                                    features_3d.clone()
                                } else {
                                    let dims = features_3d.shape().dims().to_vec();
                                    let (b, t, d) = (dims[0], dims[1], dims[2]);
                                    features_3d.reshape(&[b * t, d])?
                                };
                                let t = policy::sample_timesteps(cfg.train.batch_size, &eri_dev)?;
                                let sigma = policy::sigma_for(&t, false)?;
                                let eps = zeros_on(z.shape().clone(), &eri_dev, DType::F32)?;
                                let zt = z.to_dtype(DType::F32)?.add(&eps)?;
                                let cond = Cond {
                                    text_hidden: zeros_on(
                                        shape3(captions.len() as i64, 77, 1024),
                                        &eri_dev,
                                        DType::BF16,
                                    )?,
                                    sigma: sigma.clone(),
                                    mask_lat: None,
                                };
                                let mut xval = zt.clone();
                                let mut count = 0usize;
                                for i in reg.forward_ids() {
                                    if let Some(maxb) = run.max_blocks_per_step {
                                        if count >= maxb {
                                            break;
                                        }
                                    }
                                    if i + 1 < km_blocks {
                                        wp.prefetch_block(i + 1)?;
                                    }
                                    {
                                        let w_i = wp.load_block_to_gpu(i)?;
                                        xval = reg.blocks[i].apply(
                                            &xval,
                                            &cond,
                                            &w_i,
                                            lora.for_block(i),
                                        )?;
                                        drop(w_i);
                                    }
                                    wp.release_block(i as isize - 1)?;
                                    count += 1;
                                }
                                let predv = xval;
                                let vloss = policy::masked_eps_loss(&predv, &eps, None)?;
                                let v = vloss.to_dtype(DType::F32)?.to_vec()?;
                                Ok(*v.get(0).unwrap_or(&f32::NAN))
                            };
                        let v = if let Some(e) = ema.as_ref() {
                            let mut named = lora.named_parameters();
                            let out: anyhow::Result<f32> =
                                common_ema::EMAHelper::with_ema_params(e, &mut named, || {
                                    run_val(&images_nhwc_bf16, &captions)
                                        .map_err(|e| anyhow::anyhow!(e.to_string()))
                                });
                            out.map_err(|e| Error::Training(e.to_string()))?
                        } else {
                            run_val(&images_nhwc_bf16, &captions)?
                        };
                        if ema.is_some() {
                            println!("[val][ema] step={} loss={:.6}", step, v);
                        } else {
                            println!("[val] step={} loss={:.6}", step, v);
                        }
                    } else {
                        println!("[val] skipped (manifest mode has no raw image loader)");
                    }
                }
            }
        }
        let secs = iter_start.elapsed().as_secs_f32();
        // Update EMA for ETA calculation
        ema_step_secs =
            if step == 1 { secs } else { ema_alpha * secs + (1.0 - ema_alpha) * ema_step_secs };
        let remaining = if total_steps > step { (total_steps - step) as f32 } else { 0.0 };
        let eta_secs = ema_step_secs * remaining;
        let hrs = (eta_secs / 3600.0) as u32;
        let mins = ((eta_secs % 3600.0) / 60.0) as u32;
        let secs_i = (eta_secs % 60.0) as u32;
        let eta_str = format!("{:02}:{:02}:{:02}", hrs, mins, secs_i);
        let lr_disp = sched.lr();
        // Loss scalar for logging
        let loss_val = loss
            .to_dtype(DType::F32)
            .and_then(|t| t.to_vec())
            .ok()
            .and_then(|v| v.get(0).copied())
            .unwrap_or(f32::NAN);
        println!(
            "[chroma] step={}/{} loss={:.6} sec={:.3} avg={:.3} eta={} lr={:.5e}",
            step, total_steps, loss_val, secs, ema_step_secs, eta_str, lr_disp
        );
        // Clear autograd tape already done after step to avoid accumulation
    }
    let total_duration = run_start.elapsed();
    println!(
        "[chroma] training complete in {:.2?} across {} steps",
        total_duration,
        cfg.train.steps
    );
    Ok(())
}

fn save_checkpoint(
    out_dir: &str,
    step: usize,
    params: &[Parameter],
    opt_state: Option<&crate::optimizer::OptimizerState>,
    ema: Option<&EMAModel>,
    grad_scaler_state: Option<(f32, usize)>,
) -> Result<()> {
    let mut tensor_views: Vec<(String, OwnedTensorView)> = Vec::new();
    for (i, param) in params.iter().enumerate() {
        let tensor = param.as_tensor()?;
        let (shape, data) = tensor_to_bf16_bytes(&tensor)?;
        tensor_views.push((
            format!("lora_param_{}", i),
            OwnedTensorView::new(SafeDtype::BF16, shape, data),
        ));
    }
    let mut metadata = HashMap::new();
    metadata.insert("step".to_string(), step.to_string());
    metadata.insert("param_count".to_string(), params.len().to_string());
    let lora_path = Path::new(out_dir).join(format!("lora_step{:04}.safetensors", step));
    serialize_tensors(&lora_path, metadata, tensor_views)?;
    if let Some(ema_model) = ema {
        // Save EMA using built-in helper with metadata
        let dtype = "bf16";
        let mut p = std::path::PathBuf::from(out_dir);
        p.push(format!("lora_ema_step{:04}.safetensors", step));
        ema_model.save_state(&p, dtype).map_err(|e| Error::Training(e.to_string()))?;
    }
    // Trainer state json
    let mut state = serde_json::json!({"global_step": step});
    if let Some(os) = opt_state {
        // Save optimizer state as JSON alongside trainer_state
        let opt_json = serde_json::to_value(os).map_err(|e| Error::Serialization(e.to_string()))?;
        if let Some(obj) = state.as_object_mut() {
            obj.insert("optimizer_state".into(), opt_json);
        }
    }
    if let Some((scale, growth)) = grad_scaler_state {
        if let Some(obj) = state.as_object_mut() {
            obj.insert("grad_scaler".into(), serde_json::json!({"scale": scale, "growth": growth}));
        }
    }
    std::fs::write(format!("{}/trainer_state.json", out_dir), serde_json::to_vec_pretty(&state)?)?;
    Ok(())
}
