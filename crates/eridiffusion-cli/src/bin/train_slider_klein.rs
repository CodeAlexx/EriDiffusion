//! train_slider_klein — Klein 4B/9B Slider-LoRA training (Concept Sliders, Gandikota et al. 2023).
//!
//! Differs from `train_klein` in objective only: instead of supervising against
//! the rectified-flow target `noise - clean`, we supervise the LoRA so that
//! scaling it by α ∈ [-1, +1] interpolates between two prompts (positive ↔
//! negative). The dataset latent is reused as a generic noisy-input source;
//! its caption is ignored (the slider direction is fixed at startup from
//! `--slider-positive-prompt` / `--slider-negative-prompt`).
//!
//! Per-step pipeline:
//!   1. Load `latent` from cache (the cache caption embedding is ignored).
//!   2. Sample timestep + build `noisy = noise·σ + latent·(1-σ)`.
//!   3. Run FOUR forwards on `noisy`:
//!        a. Base + positive (no LoRA, no_grad)  → ε_pos   (DETACHED)
//!        b. Base + negative (no LoRA, no_grad)  → ε_neg   (DETACHED)
//!        c. Base+LoRA + positive                 → ε_pos_lora (autograd)
//!        d. Base+LoRA + negative                 → ε_neg_lora (autograd)
//!   4. target_pos = ε_pos + scale·(ε_pos - ε_neg);   target_neg = ε_neg - scale·(ε_pos - ε_neg)
//!      loss = mean((ε_pos_lora - target_pos)²) + mean((ε_neg_lora - target_neg)²)
//!   5. Backward; gradients flow only through the LoRA branches.
//!
//! LoRA toggle: `KleinModel.is_lora` is a public field; we flip it `false`
//! before the two base forwards and restore it before the two LoRA forwards.
//! When `is_lora=false`, `forward_inner` builds `lora = None` and calls the
//! pure base path. The base passes also run under `AutogradContext::no_grad`
//! so their activations don't bloat memory.
//!
//! Single seed=42; AdamW(lr=3e-5, beta=0.9/0.999, wd=0.01) — same defaults
//! as `train_klein`. Reference paper: <https://arxiv.org/abs/2311.12092>.

use clap::Parser;
use std::path::PathBuf;
use flame_core::{autograd::AutogradContext, DType, Shape, Tensor};
use flame_core::adam::AdamW;
use eridiffusion_core::config::{TrainConfig, TrainingMethod};
use eridiffusion_core::debug as dbg;
use eridiffusion_core::encoders::qwen3::Qwen3Encoder;
use eridiffusion_core::models::{klein::KleinModel, TrainableModel};
use eridiffusion_core::sampler::klein_sampler;
use eridiffusion_core::training::board::BoardWriter;
use eridiffusion_core::training::checkpoint::{self, CkptHeader};
use eridiffusion_core::training::features::{
    disk_check, multi_backend::MultiBackend,
    noise_modifiers, sample_library::SampleLibrary, slider, tread, validation::ValidationLoop,
};
use eridiffusion_core::training::features::health::GpuHealthMonitor;
use eridiffusion_core::training::features::lr_schedule;
use eridiffusion_core::training::features::webhook::WebhookClient;
use eridiffusion_core::training::training_features::OptimizerKind;

const NUM_TRAIN_TIMESTEPS: usize = 1000;
const LOGIT_NORMAL_BIAS: f32 = 0.0;
const LOGIT_NORMAL_SCALE: f32 = 1.0;
const TIMESTEP_SHIFT: f32 = 1.0;        // klein preset default
const SEED: u64 = 42;
const CLIP_GRAD_NORM: f32 = 1.0;        // klein preset default — essential for convergence

#[derive(Parser)]
struct Args {
    #[arg(long)] config: PathBuf,
    #[arg(long)] cache_dir: PathBuf,
    /// Klein transformer safetensors path. Either a directory of shards or a
    /// single-file checkpoint (e.g. `flux-2-klein-base-4b.safetensors`).
    #[arg(long)] transformer: PathBuf,
    #[arg(long, default_value = "100")] steps: usize,
    #[arg(long, default_value = "16")] rank: usize,
    #[arg(long, default_value = "16.0")] lora_alpha: f64,
    /// Klein 9B preset = 3e-5; 4B can usually take a touch higher.
    #[arg(long, default_value = "3e-5")] lr: f32,
    /// Linear LR warmup steps. OT preset `klein9b_lora_boxjana.json` says 100.
    /// Must be > 0 to avoid contaminated AdamW moments at step 0.
    #[arg(long, default_value = "100")] warmup_steps: usize,
    /// Per-step batch size — N cached samples are loaded and stacked along
    /// dim 0 each step. upstream Python's klein9b preset uses batch=2; ED-v2
    /// previously silently used batch=1 by ignoring the config field.
    #[arg(long, default_value = "1")] batch_size: usize,
    /// Resume from a saved LoRA checkpoint — overwrites freshly-init zeros
    /// after model load. Use to continue training. Optimizer state NOT resumed.
    #[arg(long, conflicts_with = "resume_full")] resume_lora: Option<PathBuf>,
    /// Full resume: LoRA weights + AdamW (m, v, t) + step counter. Refuses
    /// rank/alpha mismatch. `--steps N` is the TARGET total step.
    #[arg(long, conflicts_with = "resume_lora")] resume_full: Option<PathBuf>,
    /// Periodic + final save mode. Default `full` (LoRA + AdamW + step) for
    /// resumable runs. `weights` writes legacy weights-only files.
    #[arg(long, default_value = "full")] save_mode: String,
    #[arg(long, default_value = "output")] output_dir: PathBuf,
    /// Per-block weight streaming via BlockOffloader. Mirrors `train_flux`.
    /// Klein 9B (~17.5 GB BF16) + forward/backward activations OOMs on 24 GB
    /// without this; Klein 4B fits resident. Default off so 4B users keep
    /// resident-fast path; pass `--offload` for 9B on 24 GB cards.
    #[arg(long)] offload: bool,

    // ── Periodic save + sample (every N steps) ──────────────────────────
    /// Save a LoRA checkpoint AND render a sample image every N steps.
    /// `0` disables. Default 500 — matches user's iteration cadence.
    #[arg(long, default_value = "500")] sample_every: usize,
    /// Prompt for the periodic sample. Required if `--sample-every > 0`.
    #[arg(long, default_value = "")] sample_prompt: String,
    /// Negative / unconditional prompt for CFG.
    #[arg(long, default_value = "")] sample_neg_prompt: String,
    /// Klein VAE safetensors. Required if `--sample-every > 0`.
    #[arg(long)] sample_vae: Option<PathBuf>,
    /// Qwen3 weights (single file or sharded directory). Required if `--sample-every > 0`.
    #[arg(long)] sample_qwen3: Option<PathBuf>,
    /// Qwen3 tokenizer.json. Required if `--sample-every > 0`.
    #[arg(long)] sample_tokenizer: Option<PathBuf>,
    /// Sample resolution. Default 1024² — gives the actual visual quality the
    /// model is targeted for. Klein 4B fits 1024² inference comfortably on
    /// 24 GB even with training state still resident (model ~8 GB + VAE 0.5 GB
    /// + sample intermediates 4-6 GB ≈ 14 GB peak; train intermediates are
    /// dropped under no_grad scope during the sample call).
    #[arg(long, default_value = "1024")] sample_size: usize,
    /// Denoise steps for periodic sample. Klein is guidance-distilled-ish
    /// so default is short.
    #[arg(long, default_value = "20")] sample_steps: usize,
    /// CFG scale for periodic sample. 1.0 = single forward (no CFG).
    #[arg(long, default_value = "4.0")] sample_cfg: f32,
    /// Fixed seed for periodic sample (so visual progression is comparable across steps).
    #[arg(long, default_value = "42")] sample_seed: u64,

    // ── Phase 0 multi-feature rollout (default-off; Phase 1+ will consume) ──
    #[arg(long)] min_snr_gamma: Option<f32>,
    #[arg(long, default_value_t = 0.0)] caption_dropout_probability: f32,
    #[arg(long, default_value_t = 1.0)] noise_offset_probability: f32,
    #[arg(long, default_value_t = 0.0)] gamma_input_perturbation: f32,
    #[arg(long, default_value_t = 0.0)] huber_strength: f32,
    #[arg(long, default_value_t = 0.0)] lr_min_factor: f32,
    #[arg(long)] validation_dataset_dir: Option<PathBuf>,
    #[arg(long, default_value_t = 0)] validation_every_steps: u64,
    #[arg(long, num_args = 0..)] multi_backend_weights: Vec<f32>,
    /// Phase 2: N concept directories paired with `--multi-backend-weights`.
    /// When both have the same non-zero count, training samples are drawn
    /// across these dirs by weight instead of round-robin over `--cache-dir`.
    #[arg(long, num_args = 0..)] multi_backend_cache_dirs: Vec<PathBuf>,
    /// Phase 2: JSON file with N validation prompts × M seeds. When set the
    /// inline-sample step iterates over all (prompt, seed) pairs instead of
    /// the single `--sample-prompt` / `--sample-seed`.
    #[arg(long)] validation_prompts_file: Option<PathBuf>,
    /// Phase 2: log per-bucket sample counts at startup. Default on; pass
    /// `--no-bucket-report` style with `--bucket-report=false` to suppress.
    #[arg(long, default_value_t = true)] bucket_report: bool,
    #[arg(long, default_value_t = 0.0)] masked_loss_weight: f32,
    #[arg(long, default_value_t = 1.0)] ema_inv_gamma: f32,
    #[arg(long, default_value_t = 0.6667)] ema_power: f32,
    #[arg(long, default_value_t = 0)] ema_update_after_step: u64,
    #[arg(long, default_value_t = 0.0)] ema_min_decay: f32,
    /// Phase 3: swap EMA shadow weights into live params at sample/checkpoint
    /// time. Default false. No effect when EMA is not constructed.
    #[arg(long, default_value_t = false)] ema_validation_swap: bool,
    #[arg(long)] tread_route_pattern: Option<String>,
    /// Phase 4: TREAD token-keep ratio. `1.0` (default) = no routing,
    /// byte-identical to non-TREAD forward. Values in `(0, 1)` route a
    /// fraction of tokens. Phase 4 ships the CLI surface only; model
    /// integration (consuming `TreadStep` in `forward`) is Phase 4.5.
    #[arg(long, default_value_t = 1.0)] tread_keep_ratio: f32,
    /// Optimizer family. Phase 1 wires the CLI flag; non-AdamW dispatch lands
    /// in Phase 5. Selecting a non-AdamW optimizer logs a warning and falls
    /// back to AdamW for now.
    #[arg(long, default_value = "adamw")] optimizer: String,

    // ── Phase 6 multi-feature rollout ─────────────────────────────────────
    /// Per-backend repeat count (sample weight multiplier). Length must match
    /// `--multi-backend-weights`. Backend i is sampled with probability
    /// proportional to `weights[i] * repeats[i]`. Default empty = identity
    /// (no repeat scaling). Common pattern: weight identical concepts equally
    /// but boost a small style backend with `repeats 1 1 5`.
    #[arg(long, num_args = 0..)] multi_backend_repeats: Vec<u32>,
    /// Phase 6 plumbing only — caption tag-shuffle is a Phase 7+ feature
    /// (cache stores encoded text). When set the trainer logs a warning and
    /// proceeds. See `caption_aug.rs` for the shuffle helper.
    #[arg(long, default_value_t = false)] caption_tag_shuffle: bool,
    /// Reload the cache_files list at every epoch boundary. Useful when a
    /// separate process is regenerating the cache mid-training. Default
    /// `false`: never reload (byte-identical to the prior commit when the
    /// cache directory is static).
    #[arg(long, default_value_t = false)] cache_clear_each_epoch: bool,
    /// Phase 6 plumbing only — kept for symmetry with prepare_klein. Trainer
    /// reads pre-encoded latents; this flag is forwarded to the prep step in
    /// pipeline tooling and otherwise ignored at training time.
    #[arg(long, default_value_t = false)] cache_invalidate: bool,

    // ── Phase 7 multi-feature rollout ─────────────────────────────────────
    /// Spawn a background NVML poller that aborts training on sustained
    /// over-temperature (≥90 °C for 30 s) or any uncorrected ECC error.
    /// Default off → no NVML init, no thread, byte-identical to Phase 6.
    #[arg(long, default_value_t = false)] gpu_health_monitor: bool,
    /// CUDA device index that the health monitor watches. Default 0.
    #[arg(long, default_value_t = 0)] gpu_health_device: u32,
    /// Discord/Slack-compatible webhook URL. When set, posts JSON
    /// notifications at training start, each checkpoint save, completion,
    /// and on panic. Default unset → no notifications, no `ureq` calls.
    #[arg(long)] webhook_url: Option<String>,

    // ── Slider-LoRA (concept slider) — REQUIRED for this binary ───────────
    /// Positive concept prompt — the direction the slider pushes toward at
    /// positive α. Encoded once at startup and reused every step.
    #[arg(long)] slider_positive_prompt: String,
    /// Negative concept prompt — the direction the slider pushes away from at
    /// positive α. Encoded once at startup and reused every step.
    #[arg(long)] slider_negative_prompt: String,
    /// Magnitude of the slider direction. Default 2.0 follows the original
    /// Concept Sliders paper. Larger values train a stronger slider but
    /// risk overshoot and instability; smaller values are subtler.
    #[arg(long, default_value = "2.0")] slider_scale: f32,

    // ── Phase 5b: autograd v2 bridge opt-in ────────────────────────────────
    /// Route the backward pass through `AutogradContext::backward_v2`
    /// (`MatchInsertedDtype` policy → BF16 grads end-to-end). Default OFF
    /// preserves v3 byte-equivalence. See train_zimage.rs:269 for full doc.
    #[arg(long, default_value_t = false)] use_autograd_v2: bool,
}

/// LOGIT_NORMAL timestep sample. Returns continuous t in [0, 1000).
fn sample_timestep_logit_normal(rng: &mut rand::rngs::StdRng) -> f32 {
    use rand_distr::{Distribution, Normal};
    let normal = Normal::new(LOGIT_NORMAL_BIAS, LOGIT_NORMAL_SCALE).unwrap();
    let z = normal.sample(rng);
    let logit_normal = 1.0 / (1.0 + (-z).exp());
    let t = logit_normal * NUM_TRAIN_TIMESTEPS as f32;
    if (TIMESTEP_SHIFT - 1.0).abs() < 1e-6 {
        t
    } else {
        NUM_TRAIN_TIMESTEPS as f32 * TIMESTEP_SHIFT * t
            / ((TIMESTEP_SHIFT - 1.0) * t + NUM_TRAIN_TIMESTEPS as f32)
    }
}

fn collect_klein_shards(path: &std::path::Path) -> anyhow::Result<Vec<PathBuf>> {
    if path.is_file() {
        return Ok(vec![path.to_path_buf()]);
    }
    let mut shards: Vec<PathBuf> = std::fs::read_dir(path)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("safetensors"))
        .collect();
    shards.sort();
    if shards.is_empty() {
        anyhow::bail!("no klein safetensors at {:?}", path);
    }
    Ok(shards)
}

fn main() -> anyhow::Result<()> {
    use rand::SeedableRng;
    env_logger::init();
    let args = Args::parse();
    std::fs::create_dir_all(&args.output_dir)?;

    flame_core::config::set_default_dtype(DType::BF16);
    let device = flame_core::global_cuda_device();

    let mut config = TrainConfig::from_json_path(&args.config.to_string_lossy())?;
    config.training_method = TrainingMethod::Lora;
    config.lora_rank = args.rank as u64;
    config.lora_alpha = args.lora_alpha;
    config.learning_rate = args.lr as f64;

    // Phase 0 multi-feature rollout — plumb CLI args into config.
    // None of the training-loop code reads these yet; Phase 1+ wires them in.
    config.min_snr_gamma = args.min_snr_gamma;
    config.caption_dropout_probability = args.caption_dropout_probability;
    config.noise_offset_probability = args.noise_offset_probability;
    config.gamma_input_perturbation = args.gamma_input_perturbation;
    config.huber_strength = args.huber_strength;
    config.lr_min_factor = args.lr_min_factor;
    config.validation_dataset_dir = args.validation_dataset_dir.clone();
    config.validation_every_steps = args.validation_every_steps;
    config.multi_backend_weights = args.multi_backend_weights.clone();
    config.masked_loss_weight = args.masked_loss_weight;
    config.ema_inv_gamma = args.ema_inv_gamma;
    config.ema_power = args.ema_power;
    config.ema_update_after_step = args.ema_update_after_step;
    config.ema_min_decay = args.ema_min_decay;
    config.ema_validation_swap = args.ema_validation_swap;
    config.tread_route_pattern = args.tread_route_pattern.clone();
    config.tread_keep_ratio = args.tread_keep_ratio;
    let tread_ranges: Option<Vec<(usize, usize)>> = if config.tread_route_pattern.is_some()
        && config.tread_keep_ratio < 1.0
    {
        let pat = config.tread_route_pattern.as_ref().unwrap();
        let r = tread::TreadConfig::parse(pat)?;
        if r.is_empty() {
            log::warn!("[tread] route_pattern={:?} parsed to empty list — TREAD disabled", pat);
            None
        } else {
            log::info!(
                "[tread] WIRED — route_pattern={:?} keep_ratio={} ({} range(s) over single blocks)",
                pat,
                config.tread_keep_ratio,
                r.len()
            );
            Some(r)
        }
    } else {
        None
    };
    config.validation_prompts_file = args.validation_prompts_file.clone();

    // ── Slider prompt encoding (REQUIRED) + periodic sample setup ────────
    // The slider needs `slider_positive_prompt` / `slider_negative_prompt`
    // encoded once at startup; we also opportunistically encode the periodic-
    // sample prompts in the same Qwen3 lifetime to avoid loading it twice.
    // Klein 9B DiT (~18 GB) + Qwen3 8B (~16 GB) cannot coexist on 24 GB, so
    // Qwen3 is dropped before DiT load.
    let periodic = args.sample_every > 0;
    if periodic {
        let _ = args.sample_qwen3.as_ref()
            .ok_or_else(|| anyhow::anyhow!("--sample-every > 0 requires --sample-qwen3"))?;
        let _ = args.sample_tokenizer.as_ref()
            .ok_or_else(|| anyhow::anyhow!("--sample-every > 0 requires --sample-tokenizer"))?;
        let _ = args.sample_vae.as_ref()
            .ok_or_else(|| anyhow::anyhow!("--sample-every > 0 requires --sample-vae"))?;
    }
    // Slider always requires Qwen3 + tokenizer (cache caption is ignored).
    let qwen3_path = args.sample_qwen3.as_ref()
        .ok_or_else(|| anyhow::anyhow!("--sample-qwen3 is required (slider needs Qwen3 to encode positive/negative prompts)"))?;
    let tok_path = args.sample_tokenizer.as_ref()
        .ok_or_else(|| anyhow::anyhow!("--sample-tokenizer is required (slider needs tokenizer.json)"))?;

    log::info!("[slider-setup] loading Qwen3 + tokenizer to encode slider prompts (before DiT load)...");
    let qwen_w = klein_load_qwen3(qwen3_path, &device)?;
    let qcfg = Qwen3Encoder::config_from_weights(&qwen_w)?;
    let qwen = Qwen3Encoder::new(qwen_w, qcfg, device.clone());
    let tok = tokenizers::Tokenizer::from_file(tok_path)
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    let slider_pos_emb = klein_encode_prompt(&qwen, &tok, &args.slider_positive_prompt)?;
    let slider_neg_emb = klein_encode_prompt(&qwen, &tok, &args.slider_negative_prompt)?;
    log::info!(
        "[slider-setup] positive=\"{}\" → {:?}",
        args.slider_positive_prompt, slider_pos_emb.shape().dims()
    );
    log::info!(
        "[slider-setup] negative=\"{}\" → {:?}",
        args.slider_negative_prompt, slider_neg_emb.shape().dims()
    );
    log::info!("[slider-setup] slider_scale={}", args.slider_scale);

    let (sample_cap, sample_uncond, sample_vae_path) = if periodic {
        let vae_path = args.sample_vae.as_ref().unwrap().clone();
        let cap = klein_encode_prompt(&qwen, &tok, &args.sample_prompt)?;
        let unc = klein_encode_prompt(&qwen, &tok, &args.sample_neg_prompt)?;
        log::info!("[sample-setup] cap={:?} uncond={:?}", cap.shape().dims(), unc.shape().dims());
        (Some(cap), Some(unc), Some(vae_path))
    } else {
        (None, None, None)
    };
    drop(qwen);
    flame_core::cuda_alloc_pool::clear_pool_cache();
    flame_core::trim_cuda_mempool(0);
    log::info!("[slider-setup] Qwen3 dropped; DiT will load next.");

    let shards = collect_klein_shards(&args.transformer)?;
    log::info!("Loading Klein transformer from {} shard(s) (rank={} alpha={})",
        shards.len(), args.rank, args.lora_alpha);
    let mut model = KleinModel::load(&shards, &config, device.clone())?;
    if args.offload {
        model.enable_offload(shards.clone())?;
        log::info!("  block-offload enabled — per-block streaming from {} shard(s)", shards.len());
    }
    let params = model.parameters();
    log::info!("Loaded {} trainable LoRA tensors", params.len());
    if params.is_empty() {
        anyhow::bail!("No trainable parameters — TrainingMethod::Lora produced empty param list");
    }

    // Phase 1: optimizer dispatch is wired only at the CLI surface. Non-AdamW
    // selection logs a warning and falls back to AdamW; full dispatch lands
    // in Phase 5.
    match OptimizerKind::parse(&args.optimizer) {
        Ok(OptimizerKind::AdamW) => {}
        Ok(other) => log::warn!(
            "non-AdamW optimizer selected: {} — Phase 1 falls back to AdamW (full dispatch in Phase 5)",
            other.as_str()
        ),
        Err(e) => log::warn!("--optimizer parse: {} — falling back to AdamW", e),
    }
    let mut opt = AdamW::new(args.lr, 0.9, 0.999, 1e-8, 0.01);

    // Caption dropout startup check: if requested but no uncond source is
    // available (sample mode is off), disable the feature with a warning so
    // training still runs.
    let mut effective_caption_dropout_prob = args.caption_dropout_probability;
    if effective_caption_dropout_prob > 0.0 && !periodic {
        log::warn!(
            "caption_dropout_probability={:.3} but --sample-every is 0 (no unconditional embedding source) — feature disabled",
            effective_caption_dropout_prob
        );
        effective_caption_dropout_prob = 0.0;
    }

    if let Some(resume_path) = args.resume_lora.as_ref() {
        log::info!("Resuming LoRA weights only (no optimizer state) from {}", resume_path.display());
        model.load_weights(&resume_path.to_string_lossy())?;
    }

    // ── Full resume: weights + AdamW state + step counter ────────────────
    let mut start_step: usize = 0;
    if let Some(resume_path) = args.resume_full.as_ref() {
        log::info!("Full-resume from {}", resume_path.display());
        let loaded = checkpoint::load_full(resume_path, &device)?;
        let named = model.named_parameters();
        checkpoint::apply_lora_weights(&loaded, &named)?;
        checkpoint::apply_to_optimizer(&loaded, &mut opt, &named, args.rank, args.lora_alpha as f32)?;
        start_step = loaded.header.step as usize;
        if start_step >= args.steps {
            log::warn!("Resumed step ({start_step}) >= --steps ({}) — nothing to do.", args.steps);
            return Ok(());
        }
        log::info!("Continuing from step {start_step}/{}", args.steps);
    }

    let save_mode_full = match args.save_mode.as_str() {
        "full" => true,
        "weights" => false,
        other => anyhow::bail!("--save-mode must be `full` or `weights`, got `{other}`"),
    };

    // Phase 2: optional multi-backend mixing. When `--multi-backend-cache-dirs`
    // is provided WITH matching `--multi-backend-weights`, training samples
    // are drawn from N weighted backends. Default-off invariance is preserved
    // when either is empty: fallthrough is the existing single-cache_dir
    // round-robin code path with the same seeded RNG.
    let multi_backend: Option<MultiBackend> = if !args.multi_backend_cache_dirs.is_empty()
        && !args.multi_backend_weights.is_empty()
    {
        if args.multi_backend_cache_dirs.len() != args.multi_backend_weights.len() {
            anyhow::bail!(
                "--multi-backend-cache-dirs ({}) and --multi-backend-weights ({}) must have equal length",
                args.multi_backend_cache_dirs.len(),
                args.multi_backend_weights.len()
            );
        }
        // Phase 6: optional `--multi-backend-repeats` multiplies into the
        // categorical weight. Empty repeats vec = unchanged Phase 2 behavior.
        let mb = if args.multi_backend_repeats.is_empty() {
            MultiBackend::new(&args.multi_backend_cache_dirs, &args.multi_backend_weights)?
        } else {
            if args.multi_backend_repeats.len() != args.multi_backend_cache_dirs.len() {
                anyhow::bail!(
                    "--multi-backend-repeats ({}) must match --multi-backend-cache-dirs ({})",
                    args.multi_backend_repeats.len(),
                    args.multi_backend_cache_dirs.len()
                );
            }
            log::info!(
                "[multi-backend-repeats] {:?}",
                args.multi_backend_repeats
            );
            MultiBackend::new_with_repeats(
                &args.multi_backend_cache_dirs,
                &args.multi_backend_weights,
                &args.multi_backend_repeats,
            )?
        };
        log::info!(
            "[multi-backend] {} backends, {} total samples",
            mb.backends.len(),
            mb.total_files()
        );
        for (i, c) in mb.per_backend_counts().iter().enumerate() {
            log::info!(
                "  backend[{i}] dir={} files={} weight={:.4}",
                args.multi_backend_cache_dirs[i].display(),
                c,
                mb.weights[i]
            );
        }
        Some(mb)
    } else if !args.multi_backend_cache_dirs.is_empty() || !args.multi_backend_weights.is_empty() {
        anyhow::bail!(
            "multi-backend: must set BOTH --multi-backend-cache-dirs and --multi-backend-weights, or neither"
        );
    } else {
        None
    };

    // Phase 6 plumbing-only flags: log status without changing behavior.
    if args.caption_tag_shuffle {
        log::warn!(
            "[caption-tag-shuffle] enabled — Phase 6 records intent only. Cache stores encoded text; runtime re-encode lands in Phase 7+."
        );
    }
    if args.cache_invalidate {
        log::info!("[cache-invalidate] flag noted — trainer reads pre-encoded latents; this is consumed at prep-time.");
    }
    if args.cache_clear_each_epoch {
        log::info!("[cache-clear-each-epoch] enabled — cache_files will reload at each epoch boundary");
    }

    let mut cache_files: Vec<PathBuf> = std::fs::read_dir(&args.cache_dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map_or(false, |e| e == "safetensors"))
        .collect();
    cache_files.sort();
    if cache_files.is_empty() && multi_backend.is_none() {
        anyhow::bail!("No cached samples in {:?}", args.cache_dir);
    }
    log::info!("Found {} cached samples", cache_files.len());

    // Phase 2: bucket-report — distribution of latent (H, W) at startup.
    // Best-effort header parse; doesn't fail training if it can't read.
    if args.bucket_report {
        if let Some(ref mb) = multi_backend {
            let dist_per_backend = mb.bucket_distribution();
            for (bi, sizes) in dist_per_backend.iter().enumerate() {
                log::info!(
                    "[bucket-report] backend[{bi}] {} samples; size distribution:",
                    mb.backends[bi].len()
                );
                let mut sorted: Vec<_> = sizes.iter().collect();
                sorted.sort();
                for ((h, w), n) in sorted {
                    log::info!("  {h}×{w}: {n} samples");
                }
            }
        } else {
            let mut sizes: std::collections::HashMap<(usize, usize), usize> =
                std::collections::HashMap::new();
            for f in &cache_files {
                if let Some((h, w)) =
                    eridiffusion_core::training::features::multi_backend::read_latent_hw(f)
                {
                    *sizes.entry((h, w)).or_default() += 1;
                }
            }
            log::info!(
                "[bucket-report] {} samples; size distribution:",
                cache_files.len()
            );
            let mut sorted: Vec<_> = sizes.iter().collect();
            sorted.sort();
            for ((h, w), n) in sorted {
                log::info!("  {h}×{w}: {n} samples");
            }
        }
    }

    // Phase 2: validation harness — held-out cache + cadence. None at default.
    let validation_loop: Option<ValidationLoop> = if let (Some(dir), n) =
        (args.validation_dataset_dir.as_ref(), args.validation_every_steps)
    {
        if n > 0 {
            let v = ValidationLoop::new(dir, n)?;
            log::info!(
                "[validation] {} held-out samples, every {} steps",
                v.len(),
                n
            );
            Some(v)
        } else {
            None
        }
    } else {
        None
    };

    // Phase 2: optional sample-prompt library. When None, the trainer falls
    // back to the single `--sample-prompt` / `--sample-seed` path that's been
    // running since Phase 1.
    let sample_library: Option<SampleLibrary> = if let Some(p) = args.validation_prompts_file.as_ref() {
        let lib = SampleLibrary::from_file(p)?;
        log::info!(
            "[sample-library] loaded {} prompt(s) from {}",
            lib.len(),
            p.display()
        );
        Some(lib)
    } else {
        None
    };

    let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);

    // ── Step-0 baseline sample (LoRA-init = base model output) ───────────
    // SKIPPED for Klein 9B + --offload: the inline sample at training-time-resident
    // residency saturates 24 GB. After a failed sample, blocks are cached on GPU
    // and step 1 forward OOMs at "free=19 MiB". Run sample_klein on the base model
    // separately to get a step-0 reference, then resume training without the
    // inline step-0 hit.

    let board = BoardWriter::open(
        &args.output_dir,
        BoardWriter::new_session_id(),
        if start_step > 0 { Some(start_step as u64) } else { None },
    ).map_err(|e| log::warn!("board.db open failed: {e}")).ok();
    if let Some(b) = &board {
        log::info!("SerenityBoard: writing scalars to {}", b.db_path.display());
    }

    // Phase 7: optional GPU health monitor. Spawned lazily — when the flag is
    // off, NVML is never initialized, no thread is spawned, and byte invariance
    // is preserved.
    let health_handle = if args.gpu_health_monitor {
        match GpuHealthMonitor::new(args.gpu_health_device) {
            Ok(mon) => {
                log::info!(
                    "[health] GPU{} health monitor armed (≥90 °C/30 s OR any uncorrected ECC → abort)",
                    args.gpu_health_device
                );
                Some(mon.spawn())
            }
            Err(e) => {
                log::warn!("[health] NVML init failed ({e}) — continuing without health monitor");
                None
            }
        }
    } else {
        None
    };

    // Phase 7: optional webhook client. `Option::None` → never constructed,
    // never POSTs, no `ureq` traffic.
    let webhook = args.webhook_url.as_ref().map(|u| WebhookClient::new(u.clone()));
    if let Some(ref w) = webhook {
        w.send(&format!(
            "Training started: {} steps, batch={}, output={}",
            args.steps,
            args.batch_size,
            args.output_dir.display()
        ));
    }

    let t_start = std::time::Instant::now();
    let mut total_loss = 0f32;
    let dataset_len = cache_files.len();
    // Phase 6: track the last epoch index to detect crossings. `dataset_len`
    // is captured once for the bounds; reloads after that may change `cache_files.len()`.
    let mut last_epoch: Option<usize> = None;

    for step in start_step..args.steps {
        // Phase 7: GPU health gate. When the monitor is unset (default) this
        // load is never reached. When set, abort flips on temp/ECC fault.
        if let Some(ref h) = health_handle {
            if h.abort_flag.load(std::sync::atomic::Ordering::Relaxed) {
                log::error!("[health] aborting due to GPU fault");
                if let Some(b) = &board {
                    b.set_status("crashed");
                }
                if let Some(ref w) = webhook {
                    w.send(&format!(
                        "Training aborted at step {} due to GPU fault",
                        step
                    ));
                }
                anyhow::bail!("GPU health monitor triggered abort");
            }
        }

        // Phase 6: optional per-epoch cache reload. Default-off — when the
        // flag is `false` the `last_epoch` watcher fires zero times and the
        // cache_files Vec is identical to the Phase 5 path. When set, on
        // every epoch boundary we re-read the cache directory; useful when
        // a separate process is regenerating cache mid-training.
        if args.cache_clear_each_epoch && multi_backend.is_none() && !cache_files.is_empty() {
            let bs_for_epoch = args.batch_size.max(1);
            let cur_epoch = (step * bs_for_epoch) / cache_files.len();
            let crossed = match last_epoch {
                None => false, // first iteration — reference epoch, no reload
                Some(prev) => cur_epoch > prev,
            };
            if crossed {
                let mut reloaded: Vec<PathBuf> = std::fs::read_dir(&args.cache_dir)?
                    .filter_map(|e| e.ok().map(|e| e.path()))
                    .filter(|p| p.extension().map_or(false, |e| e == "safetensors"))
                    .collect();
                reloaded.sort();
                if !reloaded.is_empty() {
                    log::info!(
                        "[cache-clear-each-epoch] epoch {} reload: {} → {} samples",
                        cur_epoch,
                        cache_files.len(),
                        reloaded.len()
                    );
                    cache_files = reloaded;
                } else {
                    log::warn!(
                        "[cache-clear-each-epoch] epoch {} reload found 0 samples in {:?}; keeping previous list",
                        cur_epoch, args.cache_dir
                    );
                }
            }
            last_epoch = Some(cur_epoch);
        }

        // Stack `batch_size` cache files. upstream Python's klein9b preset uses
        // batch=2; the previous ED-v2 impl silently loaded one sample per
        // step regardless of config, breaking apples-to-apples comparison.
        // Per-element timesteps + per-element noise — matches upstream Python
        // `ModelSetupNoiseMixin._get_timestep_discrete(batch_size=...)`.
        let bs = args.batch_size.max(1);
        let mut latents = Vec::with_capacity(bs);
        let mut txts = Vec::with_capacity(bs);
        // Slider mode: no per-pixel mask path. Kept as an empty Vec so the
        // later `_ = masks` silence still type-checks.
        let masks: Vec<flame_core::Tensor> = Vec::new();
        for b in 0..bs {
            // Phase 2: when multi-backend is active, pick by weight; else fall
            // back to the historical (step * bs + b) % N round-robin which the
            // 5-step Klein 9B byte-invariance smoke depends on.
            let cache_path: PathBuf = if let Some(ref mb) = multi_backend {
                mb.pick(&mut rng).clone()
            } else {
                cache_files[(step * bs + b) % cache_files.len()].clone()
            };
            let sample = flame_core::serialization::load_file(&cache_path, &device)?;
            let l = sample.get("latent")
                .ok_or_else(|| anyhow::anyhow!("cache {} missing 'latent'", cache_path.display()))?
                .to_dtype(DType::BF16)?;
            let t = sample.get("text_embedding")
                .ok_or_else(|| anyhow::anyhow!("cache {} missing 'text_embedding'", cache_path.display()))?
                .to_dtype(DType::BF16)?;
            // masked_loss is unused in slider mode (no noise-vs-clean target).
            let _ = config.masked_loss_weight;
            latents.push(l);
            txts.push(t);
        }
        let latent = if bs == 1 {
            latents.into_iter().next().unwrap()
        } else {
            Tensor::cat(&latents.iter().collect::<Vec<_>>(), 0)?
        };
        let txt = if bs == 1 {
            txts.into_iter().next().unwrap()
        } else {
            Tensor::cat(&txts.iter().collect::<Vec<_>>(), 0)?
        };

        // Slider mode: the cache caption embedding is IGNORED. The slider
        // direction is fixed at startup from `--slider-positive-prompt` and
        // `--slider-negative-prompt`, encoded into `slider_pos_emb` and
        // `slider_neg_emb`. We tile each to match the batch size.
        let _ = txt; // silence unused — kept above for shape logging at step 0
        let tile_to_bs = |emb: &Tensor| -> anyhow::Result<Tensor> {
            let dims = emb.shape().dims();
            if dims[0] == bs {
                Ok(emb.clone())
            } else {
                let mut tgt = dims.to_vec();
                tgt[0] = bs;
                Ok(emb.broadcast_to(&Shape::from_dims(&tgt))?)
            }
        };
        let pos_txt = tile_to_bs(&slider_pos_emb)?;
        let neg_txt = tile_to_bs(&slider_neg_emb)?;

        // Per-batch-element timesteps. upstream Python samples shape [B] (line
        // 99 BaseFlux2Setup.py: `batch_size=batch['latent_image'].shape[0]`).
        let mut t_per_b: Vec<f32> = Vec::with_capacity(bs);
        let mut sigma_per_b: Vec<f32> = Vec::with_capacity(bs);
        let mut t_model_per_b: Vec<f32> = Vec::with_capacity(bs);
        for _ in 0..bs {
            let t_continuous = sample_timestep_logit_normal(&mut rng);
            let sigma_idx = (t_continuous.floor() as usize).min(NUM_TRAIN_TIMESTEPS - 1);
            let sigma = (sigma_idx + 1) as f32 / NUM_TRAIN_TIMESTEPS as f32;
            t_per_b.push(t_continuous);
            sigma_per_b.push(sigma);
            t_model_per_b.push(sigma_idx as f32 / NUM_TRAIN_TIMESTEPS as f32);
        }
        // For the noise/blend math we broadcast sigma over [B, C, H, W]
        // by multiplying each batch element separately and stacking.
        let noise = Tensor::randn(latent.shape().clone(), 0.0, 1.0, device.clone())?
            .to_dtype(DType::BF16)?;
        // Phase 1: noise modifiers — offset noise (per-channel constant added
        // to noise) + input perturbation (gaussian extra noise on noise). Both
        // are no-ops at default config (offset_noise_weight=0.0,
        // gamma_input_perturbation=0.0). Offset noise is part of the "clean"
        // noise distribution that the target supervises against; input
        // perturbation feeds the model input only and must NOT contaminate
        // target (SimpleTuner reference). Default-off: when
        // gamma_input_perturbation=0, perturbed_noise == clean_noise so byte
        // invariance is preserved.
        let clean_noise = noise_modifiers::maybe_apply_offset_noise(
            &noise,
            config.offset_noise_weight as f32,
            args.noise_offset_probability,
            &mut rng,
        )?;
        let perturbed_noise = noise_modifiers::maybe_apply_input_perturbation(
            &clean_noise,
            args.gamma_input_perturbation,
            &mut rng,
        )?;
        let noisy = if bs == 1 {
            perturbed_noise.mul_scalar(sigma_per_b[0])?
                .add(&latent.mul_scalar(1.0 - sigma_per_b[0])?)?
        } else {
            // Per-element scaling. Slice batch dim, scale each, re-stack.
            let mut pieces = Vec::with_capacity(bs);
            for b in 0..bs {
                let n_b = perturbed_noise.narrow(0, b, 1)?;
                let l_b = latent.narrow(0, b, 1)?;
                let s = sigma_per_b[b];
                pieces.push(n_b.mul_scalar(s)?.add(&l_b.mul_scalar(1.0 - s)?)?);
            }
            Tensor::cat(&pieces.iter().collect::<Vec<_>>(), 0)?
        };
        // Slider doesn't use a noise-vs-clean target. Keep `clean_noise` /
        // `latent` references silent for the unused-binding lints.
        let _ = clean_noise;
        let _ = masks;
        // timestep tensor shape [B] — model.forward broadcasts over batch.
        let timestep = Tensor::from_vec(
            t_model_per_b.clone(),
            Shape::from_dims(&[bs]),
            device.clone(),
        )?;
        let t_continuous = t_per_b[0]; // for OT_DEBUG line; per-step single number
        let sigma = sigma_per_b[0];
        let sigma_idx = (t_per_b[0].floor() as usize).min(NUM_TRAIN_TIMESTEPS - 1);

        if step == 0 {
            log::info!(
                "step 0 | batch={} latent={:?} pos_txt={:?} neg_txt={:?} sigma[0]={:.4} (idx={})",
                bs, latent.shape().dims(),
                pos_txt.shape().dims(), neg_txt.shape().dims(),
                sigma, sigma_idx
            );
        }

        // TREAD is incompatible with the slider 4-forward path (the routed
        // tokens differ between passes); explicitly disable it for clarity.
        let _ = &tread_ranges;
        let tread_step: Option<&tread::TreadStep> = None;

        // ── 4-forward slider step ─────────────────────────────────────────
        // Two base passes (no LoRA, no_grad) feed the detached target; two
        // with-LoRA passes carry the gradient. We toggle `model.is_lora`
        // to disable the LoRA delta on the base passes — the public field
        // is read at the start of each block in `forward_inner`.
        let saved_is_lora = model.is_lora;

        let (eps_pos, eps_neg) = {
            let _g = AutogradContext::no_grad();
            model.is_lora = false;
            let ep = model.forward_train(&noisy, &pos_txt, &timestep, tread_step)?;
            let ep_f = ep.to_dtype(DType::F32)?;
            drop(ep);
            AutogradContext::clear();
            flame_core::cuda_alloc_pool::clear_pool_cache();
            flame_core::trim_cuda_mempool(0);
            let en = model.forward_train(&noisy, &neg_txt, &timestep, tread_step)?;
            let en_f = en.to_dtype(DType::F32)?;
            drop(en);
            AutogradContext::clear();
            flame_core::cuda_alloc_pool::clear_pool_cache();
            flame_core::trim_cuda_mempool(0);
            (ep_f, en_f)
        };

        // Restore LoRA for the gradient-carrying passes.
        model.is_lora = saved_is_lora;
        let eps_pos_lora = model.forward_train(&noisy, &pos_txt, &timestep, tread_step)?;
        let eps_neg_lora = model.forward_train(&noisy, &neg_txt, &timestep, tread_step)?;
        if eps_pos_lora.shape().dims() != eps_pos.shape().dims() {
            anyhow::bail!(
                "slider shape mismatch: lora {:?} vs base {:?}",
                eps_pos_lora.shape().dims(),
                eps_pos.shape().dims()
            );
        }

        let loss = slider::slider_loss(
            &eps_pos_lora,
            &eps_neg_lora,
            &eps_pos,
            &eps_neg,
            args.slider_scale,
        )?;
        let loss_val = loss.to_vec()?[0];
        if !loss_val.is_finite() {
            anyhow::bail!("slider loss not finite at step {step}: {loss_val}");
        }
        total_loss += loss_val;

        // === Slider-mode per-step debug line ===
        let dbg_on = dbg::enabled("OT_DEBUG_STATS");
        if dbg_on {
            let pl = dbg::stats(&eps_pos_lora);
            let nl = dbg::stats(&eps_neg_lora);
            let pb = dbg::stats(&eps_pos);
            let nb = dbg::stats(&eps_neg);
            eprintln!(
                "[OT_DEBUG step={:5}] t={:.2} slider_loss={:.4} | pos_lora[std={:.3e}] neg_lora[std={:.3e}] pos_base[std={:.3e}] neg_base[std={:.3e}]",
                step, t_continuous, loss_val,
                pl.std, nl.std, pb.std, nb.std,
            );
        }

        // Phase 5b: Route (ii) bridge. `--use-autograd-v2` flips the
        // backward entry to construct a `MatchInsertedDtype` GradientMap.
        let grads = if args.use_autograd_v2 {
            #[cfg(feature = "autograd_v2")]
            {
                flame_core::AutogradContext::backward_v2(&loss)?
            }
            #[cfg(not(feature = "autograd_v2"))]
            {
                anyhow::bail!(
                    "--use-autograd-v2 set, but flame-core was built without the \
                     `autograd_v2` feature. Rebuild with `--features autograd_v2`."
                );
            }
        } else {
            loss.backward()?
        };

        // clip_grad_norm = 1.0 (klein preset default; ERNIE memory: convergence killer
        // if omitted).
        //
        // Fusion Sprint Phase 5: replaced the per-tensor `.to_vec()?[0]` loop
        // (N D2H syncs per step) with `flame_core::ops::grad_norm::global_l2_norm`,
        // which keeps the L2 reduction on device and does ONE D2H sync at the end
        // for the host-side scale. For Klein 9B LoRA (~200 LoRA tensors) that's a
        // 200× reduction in sync count.
        let grad_refs: Vec<&flame_core::Tensor> = params
            .iter()
            .filter_map(|p| grads.get(p.id()))
            .collect();
        let total_norm = flame_core::ops::grad_norm::global_l2_norm(&grad_refs)?
            .item()? as f32;
        if dbg_on {
            eprintln!("[OT_DEBUG step={:5}] grad_norm_pre_clip={:.4e}", step, total_norm);
        }
        let scale = if total_norm > CLIP_GRAD_NORM { CLIP_GRAD_NORM / total_norm } else { 1.0 };
        for param in &params {
            if let Some(g) = grads.get(param.id()) {
                let g_scaled = if scale < 1.0 {
                    g.mul_scalar(scale)?
                } else {
                    g.clone()
                };
                param.set_grad(g_scaled)?;
            }
        }
        if step < 5 || (step + 1) % 50 == 0 {
            log::debug!("grad_norm={:.4} (scale={:.4})", total_norm, scale);
        }

        // Apply linear warmup → scheduled LR. Step 0 uses lr/warmup, ramps to
        // base_lr at step `warmup_steps - 1`. Then dispatches by
        // `learning_rate_scheduler`. Default `Constant` is byte-identical to
        // the legacy `constant_with_warmup` Klein has used since launch —
        // see lr_schedule::tests::constant_lr_matches_legacy_constant_with_warmup.
        let cur_lr = lr_schedule::dispatch_lr(
            &config.learning_rate_scheduler,
            args.lr,
            step,
            args.steps,
            args.warmup_steps,
            config.lr_min_factor,
            config.learning_rate_cycles as f32,
        );
        {
            let _g = AutogradContext::no_grad();
            opt.set_lr(cur_lr);
            opt.step(&params)?;
            opt.zero_grad(&params);
        }
        AutogradContext::clear();

        let _ = total_loss;
        eridiffusion_core::training::progress::log_step(
            "Klein-slider-lora",
            step, args.steps, dataset_len, args.batch_size.max(1),
            loss_val, total_norm, cur_lr, t_start, board.as_ref(),
        );

        // Validation pass is disabled in slider mode — the held-out cache
        // would feed a noise-vs-clean target that is meaningless against
        // the slider objective. A slider-aware validation harness is a
        // Phase 5+ follow-up.
        let _ = &validation_loop;

        // ── Periodic save + inline sample (every N steps) ───────────────
        let step_num = step + 1;
        if periodic && step_num % args.sample_every == 0 && step_num < args.steps {
            let mid_ckpt = args.output_dir.join(format!("klein_lora_step{step_num}.safetensors"));
            // Phase 7: disk-space pre-check. 2 GB threshold covers Klein 9B
            // LoRA full save (~520 MB) + safety margin. On insufficient space
            // we LOG and SKIP the save (a partial-write checkpoint is worse
            // than no checkpoint).
            let mut skip_save = false;
            if let Err(e) = disk_check::check_free_space(
                &args.output_dir,
                2 * 1024 * 1024 * 1024,
            ) {
                log::warn!("[disk-check step {step_num}] {e} — skipping mid-save");
                skip_save = true;
            }
            if !skip_save {
                if save_mode_full {
                    let header = CkptHeader::from_adamw(
                        "train_klein", step_num as u64, &opt,
                        args.rank, args.lora_alpha as f32, SEED, String::new(),
                    );
                    let named = model.named_parameters();
                    if let Err(e) = checkpoint::save_full(&mid_ckpt, &named, &opt, &header) {
                        log::warn!("[mid-save step {step_num}] full save failed: {e}");
                    }
                } else if let Err(e) = model.save_weights(&mid_ckpt.to_string_lossy()) {
                    log::warn!("[mid-save step {step_num}] save_weights failed: {e}");
                } else {
                    log::info!("[mid-save step {step_num}] {}", mid_ckpt.display());
                }
            }
            // Phase 7: webhook checkpoint notification. Sends regardless of
            // skip — operators want to know the checkpoint cadence even on
            // disk-skip events (so they can free space and try again).
            if let Some(ref w) = webhook {
                let avg_so_far = if (step + 1) > start_step {
                    total_loss / ((step + 1 - start_step) as f32)
                } else {
                    0.0
                };
                let suffix = if skip_save { " (save SKIPPED — low disk)" } else { "" };
                w.send(&format!(
                    "Step {}/{}: avg loss {:.4}{}",
                    step_num, args.steps, avg_so_far, suffix
                ));
            }
            // Phase 2: belt+braces iteration tracker JSON. Belt+braces resume
            // fallback — not consumed by anything yet, but cheap to write.
            let avg_so_far = if (step + 1) > start_step {
                total_loss / ((step + 1 - start_step) as f32)
            } else {
                0.0
            };
            write_iteration_tracker(
                &args.output_dir,
                step_num,
                args.steps,
                t_start.elapsed().as_secs_f64(),
                avg_so_far,
                loss_val,
            );
            // Phase 2: validation prompts library. Constraint: the inline
            // sample path uses PRE-ENCODED `sample_cap`/`sample_uncond`
            // tensors because re-loading Qwen3 8B at each sample step would
            // OOM Klein 9B on 24 GB (see sample-setup comment). So we keep
            // the existing single primary prompt encoding, but iterate the
            // multi-SEED sweep when the library specifies seeds. Multi-PROMPT
            // requires deferring Qwen3 encode-on-demand, which is a follow-up.
            let cap = sample_cap.as_ref().unwrap();
            let unc = sample_uncond.as_ref().unwrap();
            let vae_path = sample_vae_path.as_ref().unwrap();
            let mut seeds_to_render: Vec<u64> = vec![args.sample_seed];
            if let Some(ref lib) = sample_library {
                // Use the FIRST library entry's seed sweep against the
                // existing primary cap/uncond. Per-prompt encoding deferred.
                if let Some(first) = lib.prompts.first() {
                    if !first.seeds.is_empty() {
                        seeds_to_render = first.seeds.clone();
                    }
                }
                if lib.prompts.len() > 1 {
                    log::info!(
                        "[sample-library] {} prompts loaded but only the first prompt's seed-sweep ({} seeds) is rendered: per-prompt encoding requires reloading Qwen3 (deferred follow-up)",
                        lib.prompts.len(),
                        seeds_to_render.len()
                    );
                }
            }
            for (si, seed) in seeds_to_render.iter().enumerate() {
                let sample_out = if seeds_to_render.len() == 1 {
                    args.output_dir.join(format!("sample_step{step_num}.png"))
                } else {
                    args.output_dir.join(format!("sample_step{step_num}_seed{si}.png"))
                };
                log::info!("[sample step={step_num} seed={seed}] → {}", sample_out.display());
                if let Err(e) = klein_inline_sample(
                    &mut model, cap, unc, vae_path, &sample_out,
                    args.sample_size, args.sample_steps, args.sample_cfg, *seed,
                    &device,
                ) {
                    log::warn!("[sample step={step_num} seed={seed}] failed: {e}");
                }
            }
        }
    }

    let trained = args.steps - start_step;
    let avg_loss = if trained > 0 { total_loss / trained as f32 } else { 0.0 };
    let wall_time = t_start.elapsed().as_secs_f64();
    log::info!("Training complete: {trained} new steps (total={}), avg loss={:.4}", args.steps, avg_loss);
    if let Some(b) = &board { b.set_status("completed"); }

    let ckpt = args.output_dir.join(format!("klein_lora_{}steps.safetensors", args.steps));
    // Phase 7: final-checkpoint disk-space pre-check. Skip + log on shortage.
    let mut final_skip_save = false;
    if let Err(e) = disk_check::check_free_space(
        &args.output_dir,
        2 * 1024 * 1024 * 1024,
    ) {
        log::warn!("[disk-check final] {e} — skipping final save");
        final_skip_save = true;
    }
    if !final_skip_save {
        if save_mode_full {
            let header = CkptHeader::from_adamw(
                "train_klein", args.steps as u64, &opt,
                args.rank, args.lora_alpha as f32, SEED, String::new(),
            );
            let named = model.named_parameters();
            if let Err(e) = checkpoint::save_full(&ckpt, &named, &opt, &header) {
                log::warn!("save_full failed: {e}");
            } else {
                log::info!("Saved checkpoint to {}", ckpt.display());
            }
        } else if let Err(e) = model.save_weights(&ckpt.to_string_lossy()) {
            log::warn!("save_weights failed: {e}");
        } else {
            log::info!("Saved checkpoint to {}", ckpt.display());
        }
    }
    // Phase 7: webhook completion notification.
    if let Some(ref w) = webhook {
        w.send(&format!(
            "Training complete: {} steps, avg loss {:.4}, took {:.1}s{}",
            args.steps,
            avg_loss,
            wall_time,
            if final_skip_save { " (final save SKIPPED — low disk)" } else { "" }
        ));
    }

    // Phase 2: write the final iteration tracker JSON sidecar.
    write_iteration_tracker(
        &args.output_dir,
        args.steps,
        args.steps,
        t_start.elapsed().as_secs_f64(),
        avg_loss,
        avg_loss,
    );

    // ── Final sample at the end of training ──────────────────────────────
    if periodic {
        let cap = sample_cap.as_ref().unwrap();
        let unc = sample_uncond.as_ref().unwrap();
        let vae_path = sample_vae_path.as_ref().unwrap();
        let sample_out = args.output_dir.join(format!("sample_step{}.png", args.steps));
        log::info!("[sample step={} FINAL] → {}", args.steps, sample_out.display());
        if let Err(e) = klein_inline_sample(
            &mut model, cap, unc, vae_path, &sample_out,
            args.sample_size, args.sample_steps, args.sample_cfg, args.sample_seed,
            &device,
        ) {
            log::warn!("[sample final] failed: {e}");
        }
    }
    Ok(())
}

// ── Periodic-sample helpers ──────────────────────────────────────────────

/// Klein chat template — must match `prepare_klein` and `sample_klein` verbatim.
/// The trailing `<think>\n\n</think>\n\n` block is REQUIRED — Klein was trained
/// to consume it on the assistant turn, and dropping it skews token positions
/// so the cached embeddings used for training and inline-sample don't share
/// a distribution.
const KLEIN_TEMPLATE_PRE: &str = "<|im_start|>user\n";
const KLEIN_TEMPLATE_POST: &str = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
const KLEIN_PAD_TOKEN_ID: i32 = 151643;
const KLEIN_TXT_PAD_LEN: usize = 512;

/// Phase 2: write a small JSON sidecar at `<output_dir>/last_iteration.json`
/// holding (step, total_steps, wall_time_secs, last_avg_loss, last_loss).
/// Belt+braces resume fallback. Best-effort — failures are logged not fatal.
fn write_iteration_tracker(
    output_dir: &std::path::Path,
    step: usize,
    total_steps: usize,
    wall_time_secs: f64,
    last_avg_loss: f32,
    last_loss: f32,
) {
    let path = output_dir.join("last_iteration.json");
    let body = serde_json::json!({
        "step": step,
        "total_steps": total_steps,
        "wall_time_secs": wall_time_secs,
        "last_avg_loss": last_avg_loss,
        "last_loss": last_loss,
    });
    if let Err(e) = std::fs::write(&path, serde_json::to_string_pretty(&body).unwrap_or_default()) {
        log::warn!("[iteration-tracker] write {}: {e}", path.display());
    }
}

fn klein_load_qwen3(
    path: &std::path::Path,
    device: &std::sync::Arc<flame_core::CudaDevice>,
) -> anyhow::Result<std::collections::HashMap<String, Tensor>> {
    if path.is_file() {
        return flame_core::serialization::load_file(path, device)
            .map_err(|e| anyhow::anyhow!("load_file: {e}"));
    }
    let mut all = std::collections::HashMap::new();
    for entry in std::fs::read_dir(path)? {
        let p = entry?.path();
        if p.extension().and_then(|e| e.to_str()) == Some("safetensors") {
            let part = flame_core::serialization::load_file(&p, device)
                .map_err(|e| anyhow::anyhow!("load_file {}: {e}", p.display()))?;
            all.extend(part);
        }
    }
    Ok(all)
}

/// Render one sample using the live in-training model state and pre-encoded
/// prompt embeddings. Loads + drops the VAE per call to bound VRAM.
fn klein_inline_sample(
    model: &mut KleinModel,
    cond: &Tensor,
    uncond: &Tensor,
    vae_path: &std::path::Path,
    out_path: &std::path::Path,
    size: usize,
    steps: usize,
    cfg: f32,
    seed: u64,
    device: &std::sync::Arc<flame_core::CudaDevice>,
) -> anyhow::Result<()> {
    use eridiffusion_core::encoders::vae::KleinVaeDecoder;
    let _no_grad = AutogradContext::no_grad();
    let h_lat = size / 16;
    let w_lat = size / 16;
    let n_img = h_lat * w_lat;
    let timesteps = klein_sampler::get_schedule(steps, n_img);

    // Seed the GLOBAL RNG that flame_core::Tensor::randn reads from.
    // (Previously we were creating a local StdRng and dropping it — `Tensor::randn`
    // never observes a local rng, so seeding had no effect on noise determinism.)
    flame_core::rng::set_seed(seed)
        .map_err(|e| anyhow::anyhow!("flame_core::rng::set_seed: {e}"))?;
    let latent_shape = Shape::from_dims(&[1, 128, h_lat, w_lat]);
    let mut latent = Tensor::randn(latent_shape, 0.0, 1.0, device.clone())?
        .to_dtype(DType::BF16)?;

    for step in 0..steps {
        let sigma = timesteps[step];
        let sigma_next = timesteps[step + 1];
        let t = klein_sampler::sigma_to_timestep(sigma);
        let t_tensor = Tensor::from_vec(vec![t], Shape::from_dims(&[1]), device.clone())?;
        let pred_cond = model.forward(&latent, cond, &t_tensor)?;
        let pred_uncond = model.forward(&latent, uncond, &t_tensor)?;
        let pred = pred_uncond.add(&pred_cond.sub(&pred_uncond)?.mul_scalar(cfg)?)?;
        latent = klein_sampler::euler_step(&latent, &pred, sigma, sigma_next)?;
    }

    let vae_weights = flame_core::serialization::load_file(vae_path, device)
        .map_err(|e| anyhow::anyhow!("vae load: {e}"))?;
    let dev = flame_core::Device::from(device.clone());
    let decoder = KleinVaeDecoder::load(&vae_weights, &dev)
        .map_err(|e| anyhow::anyhow!("vae decoder: {e}"))?;
    drop(vae_weights);
    let img = decoder.decode(&latent)?;

    let pixels: Vec<f32> = img.to_dtype(DType::F32)?.to_vec()?;
    let dims = img.shape().dims();
    let (c, h, w) = if dims.len() == 4 { (dims[1], dims[2], dims[3]) } else { (3, dims[0], dims[1]) };
    let mut buf = vec![0u8; h * w * 3];
    for y in 0..h {
        for x in 0..w {
            for ch in 0..c.min(3) {
                let idx = ch * h * w + y * w + x;
                let v = pixels.get(idx).copied().unwrap_or(0.0);
                buf[(y * w + x) * 3 + ch] = ((v.clamp(-1.0, 1.0) + 1.0) * 127.5) as u8;
            }
        }
    }
    image::save_buffer(out_path, &buf, w as u32, h as u32, image::ColorType::Rgb8)
        .map_err(|e| anyhow::anyhow!("save: {e}"))?;
    Ok(())
}

fn klein_encode_prompt(
    qwen: &Qwen3Encoder,
    tok: &tokenizers::Tokenizer,
    prompt: &str,
) -> anyhow::Result<Tensor> {
    let wrapped = format!("{KLEIN_TEMPLATE_PRE}{}{KLEIN_TEMPLATE_POST}", prompt.trim());
    let enc = tok.encode(wrapped.as_str(), false)
        .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
    let mut ids: Vec<i32> = enc.get_ids().iter().map(|&i| i as i32).collect();
    ids.resize(KLEIN_TXT_PAD_LEN, KLEIN_PAD_TOKEN_ID);
    let hidden = qwen.encode(&ids)?;
    Ok(hidden.to_dtype(DType::BF16)?)
}
