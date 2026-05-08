//! train_flux — FLUX.1 (Dev/Schnell) LoRA training binary.
//!
//! Mirrors `train_ernie.rs` per OT preset semantics (see `/home/alex/upstream Python/training_presets/#flux LoRA.json`):
//!   - learning_rate         = 0.0003
//!   - batch_size            = 4              (per-step here is 1; gradient acc not yet wired)
//!   - resolution            = 768            (dataset-side)
//!   - lora_rank / alpha     = 16 / 1.0       (preset doesn't pin these — common community defaults)
//!   - timestep_distribution = LOGIT_NORMAL   (preset L27)
//!   - dynamic_timestep_shifting = false      (preset L28)
//!   - timestep_shift        = 1.0            (default — overridden by per-resolution mu in sampling, NOT in training)
//!   - noising_weight        = 0.0            (TrainConfig default → LOGIT_NORMAL scale = 1.0)
//!   - noising_bias          = 0.0            (TrainConfig default → LOGIT_NORMAL bias  = 0.0)
//!   - clip_grad_norm        = 1.0
//!   - train_dtype           = BF16
//!   - optimizer             = AdamW (β=(0.9, 0.999), ε=1e-8, wd=0.01)
//!
//! Variant flag (`--variant dev|schnell`): controls guidance handling.
//! Dev model has `guidance_in.in_layer.weight` → injects guidance=3.5 at training time.
//! Schnell has no guidance embed → guidance is ignored.
//!
//! Cached sample format (produced by prepare_flux):
//!   latent     [1, 16, H/8, W/8]  BF16   (RAW VAE posterior — unscaled, unpacked)
//!   t5_embed   [1, 512, 4096]     BF16
//!   clip_pool  [1, 768]           BF16
//!
//! Audit fix FLUX_VERIFY §H3: pre-fix the cache stored already-scaled,
//! already-packed latents AND `img_ids`/`txt_ids`. We now match upstream Python's
//! contract — cache is RAW posterior; `(latent - SHIFT) * SCALE` and
//! `pack_latents` happen at training time; position IDs are recomputed per
//! step from latent shape.

use clap::{Parser, ValueEnum};
use flame_core::{adam::AdamW, autograd::AutogradContext, DType, Shape, Tensor};
use eridiffusion_core::config::{LrScheduler, TrainConfig, TrainingMethod};
use eridiffusion_core::encoders::flux_vae::{SCALE, SHIFT};
use eridiffusion_core::models::{flux::FluxModel, TrainableModel};
use eridiffusion_core::sampler::flux_sampler::{build_img_ids, build_txt_ids, pack_latents};
use eridiffusion_core::training::board::BoardWriter;
use eridiffusion_core::training::checkpoint::{self, CkptHeader};
use eridiffusion_core::training::ema::ParameterEma;
use eridiffusion_core::training::features::ema_advanced::EmaConfig;
use eridiffusion_core::training::features::{loss_weight, lr_schedule, noise_modifiers, timestep_bias};
use eridiffusion_core::training::training_features::OptimizerKind;
use std::path::PathBuf;

const NUM_TRAIN_TIMESTEPS: usize = 1000;
const LOGIT_NORMAL_BIAS: f32 = 0.0;          // TrainConfig.noising_bias default
const LOGIT_NORMAL_SCALE: f32 = 1.0;         // noising_weight + 1.0 = 0.0 + 1.0
const TIMESTEP_SHIFT: f32 = 1.0;             // preset default (dynamic shifting = false)
const SEED: u64 = 42;
const CLIP_GRAD_NORM: f32 = 1.0;

#[derive(Copy, Clone, ValueEnum, Debug)]
enum Variant { Dev, Schnell }

#[derive(Parser)]
struct Args {
    /// OT-format JSON config (optional). Falls back to TrainConfig::default() if absent.
    #[arg(long)] config: Option<PathBuf>,
    #[arg(long)] cache_dir: PathBuf,
    /// FLUX transformer dir (containing `flux1-dev.safetensors` or shards).
    #[arg(long)] transformer: PathBuf,
    #[arg(long, value_enum, default_value_t = Variant::Dev)] variant: Variant,
    #[arg(long, default_value = "100")] steps: usize,
    #[arg(long, default_value = "16")] rank: usize,
    /// LoRA alpha. Convention: alpha = rank (effective scale = 1.0). FLUX_VERIFY §H12 /
    /// SKEPTIC §H12. Pre-fix default was 1.0 with rank=16 → effective scale = 0.0625.
    #[arg(long, default_value = "16.0")] lora_alpha: f64,
    /// Preset learning rate (3e-4).
    #[arg(long, default_value = "3e-4")] lr: f32,
    #[arg(long, default_value = "output")] output_dir: PathBuf,
    /// Per-block weight streaming (drops resident block weights, reloads per layer).
    #[arg(long)] offload: bool,
    /// Save a LoRA checkpoint every N steps (0 = end-only).
    #[arg(long, default_value = "0")] save_every: usize,
    /// Resume LoRA weights only (no optimizer state).
    #[arg(long, conflicts_with = "resume_full")] resume_lora: Option<PathBuf>,
    /// Full resume: LoRA weights + AdamW (m, v, t) + step counter.
    #[arg(long, conflicts_with = "resume_lora")] resume_full: Option<PathBuf>,
    /// Periodic + final save mode. Default `full` (LoRA + AdamW + step) for
    /// resumable runs. `weights` writes legacy weights-only files.
    #[arg(long, default_value = "full")] save_mode: String,

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
    /// Phase 2: paired with --multi-backend-weights. Klein-only wiring; other
    /// trainers accept-and-warn until per-model wiring lands.
    #[arg(long, num_args = 0..)] multi_backend_cache_dirs: Vec<std::path::PathBuf>,
    /// Phase 2: validation prompt library JSON (Klein-only wiring; other
    /// trainers accept-and-warn).
    #[arg(long)] validation_prompts_file: Option<std::path::PathBuf>,
    #[arg(long, default_value_t = 0.0)] masked_loss_weight: f32,
    /// Master switch for EMA shadow. See train_klein.rs for full doc.
    /// Default off → byte-identical to pre-flag commits.
    #[arg(long, default_value_t = false)] ema: bool,
    #[arg(long, default_value_t = 1.0)] ema_inv_gamma: f32,
    #[arg(long, default_value_t = 0.6667)] ema_power: f32,
    #[arg(long, default_value_t = 0)] ema_update_after_step: u64,
    #[arg(long, default_value_t = 0.0)] ema_min_decay: f32,
    #[arg(long, default_value_t = 0.9999)] ema_max_decay: f32,
    /// Swap shadow → live params at sample/save time.
    #[arg(long, default_value_t = false)] ema_validation_swap: bool,
    /// Pyramid / multi-resolution noise: number of additional resolution
    /// levels to mix into the per-step training noise. `0` (default) is a
    /// no-op — byte-identical to no-multires.
    #[arg(long, default_value_t = 0)] multires_noise_iterations: usize,
    /// Per-level discount factor for `--multires-noise-iterations`.
    #[arg(long, default_value_t = 0.3)] multires_noise_discount: f32,
    /// Multi-distribution timestep bias strategy. `none` is byte-identical.
    #[arg(long, default_value = "none")] timestep_bias_strategy: String,
    #[arg(long, default_value_t = 0.0)] timestep_bias_multiplier: f32,
    #[arg(long, default_value_t = 0.0)] timestep_bias_range_min: f32,
    #[arg(long, default_value_t = 1.0)] timestep_bias_range_max: f32,
    #[arg(long)] tread_route_pattern: Option<String>,
    /// Phase 1: optimizer family CLI surface (Phase 5 wires full dispatch).
    #[arg(long, default_value = "adamw")] optimizer: String,

    // ── Phase 6 multi-feature rollout (plumb-only; multi-backend wired in Klein) ──
    #[arg(long, num_args = 0..)] multi_backend_repeats: Vec<u32>,
    #[arg(long, default_value_t = false)] caption_tag_shuffle: bool,
    #[arg(long, default_value_t = false)] cache_clear_each_epoch: bool,
    #[arg(long, default_value_t = false)] cache_invalidate: bool,
    /// Phase 5: LR scheduler family. Default `constant` + `warmup_steps=0` is
    /// byte-equivalent to prior fixed-LR behaviour.
    #[arg(long, default_value = "constant")] lr_scheduler: String,
    /// Phase 5: linear LR warmup steps. Default 0 keeps prior behaviour.
    #[arg(long, default_value_t = 0)] warmup_steps: usize,
    /// Phase 5: cosine-with-restarts cycle count.
    #[arg(long, default_value_t = 1.0)] lr_cycles: f32,
}

/// LOGIT_NORMAL timestep sample matching OT `_get_timestep_discrete`.
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

fn collect_shards(path: &std::path::Path) -> anyhow::Result<Vec<PathBuf>> {
    if path.is_file() { return Ok(vec![path.to_path_buf()]); }
    let mut shards: Vec<PathBuf> = std::fs::read_dir(path)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("safetensors"))
        .collect();
    shards.sort();
    if shards.is_empty() { anyhow::bail!("no safetensors at {:?}", path); }
    Ok(shards)
}

fn main() -> anyhow::Result<()> {
    use rand::SeedableRng;
    env_logger::init();
    let args = Args::parse();
    // Phase 2: Klein-only wiring of multi-backend + validation prompts library.
    // Other trainers accept-and-warn so configs/launchers aren't broken; full
    // wiring is a follow-up after the per-model encoder + sample paths are
    // consolidated.
    if !args.multi_backend_cache_dirs.is_empty() || !args.multi_backend_weights.is_empty() {
        log::warn!("--multi-backend-* flags are Klein-only in Phase 2; ignored here");
    }
    if args.validation_prompts_file.is_some() {
        log::warn!("--validation-prompts-file is Klein-only in Phase 2; ignored here");
    }
    std::fs::create_dir_all(&args.output_dir)?;

    flame_core::config::set_default_dtype(DType::BF16);
    let device = flame_core::global_cuda_device();

    let mut config = if let Some(cp) = &args.config {
        TrainConfig::from_json_path(&cp.to_string_lossy())?
    } else {
        TrainConfig::default()
    };
    config.training_method = TrainingMethod::Lora;
    config.lora_rank = args.rank as u64;
    config.lora_alpha = args.lora_alpha;
    config.learning_rate = args.lr as f64;

    // Phase 0 multi-feature rollout — plumb CLI args into config (default-off, unused yet).
    config.min_snr_gamma = args.min_snr_gamma;
    config.caption_dropout_probability = args.caption_dropout_probability;
    config.noise_offset_probability = args.noise_offset_probability;
    config.gamma_input_perturbation = args.gamma_input_perturbation;
    config.huber_strength = args.huber_strength;
    config.lr_min_factor = args.lr_min_factor;
    config.validation_dataset_dir = args.validation_dataset_dir.clone();
    config.validation_every_steps = args.validation_every_steps;
    config.multi_backend_weights = args.multi_backend_weights.clone();
    config.validation_prompts_file = args.validation_prompts_file.clone();
    config.masked_loss_weight = args.masked_loss_weight;
    config.ema_inv_gamma = args.ema_inv_gamma;
    config.ema_power = args.ema_power;
    config.ema_update_after_step = args.ema_update_after_step;
    config.ema_min_decay = args.ema_min_decay;
    config.tread_route_pattern = args.tread_route_pattern.clone();

    let shards = collect_shards(&args.transformer)?;
    log::info!("[Flux] loading transformer from {} shard(s) (variant={:?}, rank={}, alpha={})...",
        shards.len(), args.variant, args.rank, args.lora_alpha);

    // Per-shard load via FluxModel::load — pass the first shard (typical Flux release ships
    // a single `flux1-dev.safetensors`); auto-detection of has_guidance happens from the keys.
    let mut model = FluxModel::load(&shards[0], &config, device.clone())?;
    // Variant override: if the user explicitly says Schnell, force guidance off even if
    // the checkpoint accidentally has guidance keys (and vice-versa for Dev).
    match args.variant {
        Variant::Dev => {
            if !model.has_guidance {
                log::warn!("--variant dev but loaded checkpoint has no guidance_in. Treating as Schnell-shaped Dev (no guidance injection).");
            }
            // Audit fix FLUX_VERIFY §H7 — `load()` defaults `guidance_value` to
            // 1.0 (matches OT canonical training default). 3.5 is inference-only.
        }
        Variant::Schnell => {
            model.has_guidance = false;
            model.guidance_value = 1.0;
        }
    }
    if args.offload {
        model.enable_offload(shards.clone())?;
        log::info!("  block-offload enabled — per-block streaming from {} shards", shards.len());
    }

    let params = model.parameters();
    log::info!("[Flux] {} trainable LoRA tensors", params.len());
    if params.is_empty() {
        anyhow::bail!("No trainable parameters — check TrainingMethod::Lora setup");
    }

    // OT preset optimizer: AdamW(β=(0.9, 0.999), ε=1e-8, wd=0.01).
    match OptimizerKind::parse(&args.optimizer) {
        Ok(OptimizerKind::AdamW) => {}
        Ok(other) => log::warn!(
            "non-AdamW optimizer selected: {} — Phase 1 falls back to AdamW (full dispatch in Phase 5)",
            other.as_str()
        ),
        Err(e) => log::warn!("--optimizer parse: {} — falling back to AdamW", e),
    }
    let mut opt = AdamW::new(args.lr, 0.9, 0.999, 1e-8, 0.01);

    // EMA shadow (Phase 3). See train_klein.rs for the same pattern.
    let ema_cfg = EmaConfig {
        inv_gamma: args.ema_inv_gamma,
        power: args.ema_power,
        update_after_step: args.ema_update_after_step,
        min_decay: args.ema_min_decay,
        max_decay: args.ema_max_decay,
    };
    let mut ema: Option<ParameterEma> = if args.ema {
        let _g = AutogradContext::no_grad();
        let e = ParameterEma::new(&params, args.ema_max_decay)
            .map_err(|e| anyhow::anyhow!("EMA construction failed: {e}"))?;
        log::info!(
            "[ema] WIRED — {} shadow tensors, swap={}",
            e.len(),
            args.ema_validation_swap
        );
        Some(e)
    } else {
        None
    };

    // Timestep bias config — defaults are byte-identical (Strategy::None).
    let timestep_bias_cfg = {
        let strategy = timestep_bias::Strategy::parse(&args.timestep_bias_strategy)
            .map_err(|e| anyhow::anyhow!("--timestep-bias-strategy: {e}"))?;
        let cfg = timestep_bias::BiasConfig {
            strategy,
            multiplier: args.timestep_bias_multiplier,
            range_min: args.timestep_bias_range_min,
            range_max: args.timestep_bias_range_max,
        };
        if strategy != timestep_bias::Strategy::None {
            log::info!(
                "[timestep-bias] strategy={} multiplier={} range=[{}, {}]",
                strategy.as_str(),
                cfg.multiplier,
                cfg.range_min,
                cfg.range_max
            );
        }
        cfg
    };

    // Phase 1: caption_dropout requires an unconditional embedding source.
    // Flux trainer doesn't currently encode one (no inline sample mode);
    // warn + disable when requested.
    if args.caption_dropout_probability > 0.0 {
        log::warn!(
            "caption_dropout_probability={:.3} requested but Flux trainer has no inline encoder — feature disabled",
            args.caption_dropout_probability
        );
    }

    if let Some(resume_path) = args.resume_lora.as_ref() {
        log::info!("Resuming LoRA weights only (no optimizer state) from {}", resume_path.display());
        model.load_weights(&resume_path.to_string_lossy())?;
    }

    let mut start_step: usize = 0;
    if let Some(resume_path) = args.resume_full.as_ref() {
        log::info!("Full-resume from {}", resume_path.display());
        let loaded = checkpoint::load_full(resume_path, &device)?;
        let bundle = model.bundle.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Flux full-resume requires LoRA mode"))?;
        let named = bundle.named_parameters();
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

    let mut cache_files: Vec<PathBuf> = std::fs::read_dir(&args.cache_dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map_or(false, |e| e == "safetensors"))
        .collect();
    cache_files.sort();
    if cache_files.is_empty() {
        anyhow::bail!("No cached samples in {:?}", args.cache_dir);
    }
    log::info!("[Flux] {} cached samples", cache_files.len());

    // SEED=42 for both timestep RNG and noise RNG. Audit fix FLUX_VERIFY §H7 /
    // SKEPTIC §H10 / §M1 (`feedback_default_seed_42.md`): seed flame_core's
    // global RNG so `Tensor::randn` is deterministic too, not just our host RNG.
    flame_core::rng::set_seed(SEED)
        .map_err(|e| anyhow::anyhow!("flame_core set_seed: {e}"))?;
    let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);
    let board = BoardWriter::open(
        &args.output_dir,
        BoardWriter::new_session_id(),
        if start_step > 0 { Some(start_step as u64) } else { None },
    ).map_err(|e| log::warn!("board.db open failed: {e}")).ok();
    if let Some(b) = &board {
        log::info!("SerenityBoard: writing scalars to {}", b.db_path.display());
    }
    let t_start = std::time::Instant::now();
    let mut total_loss = 0f32;

    let sched: LrScheduler = lr_schedule::parse_cli_scheduler(&args.lr_scheduler);
    for step in start_step..args.steps {
        let cache_idx = step % cache_files.len();
        let sample = flame_core::serialization::load_file(&cache_files[cache_idx], &device)?;

        // Audit fix FLUX_VERIFY §H3 / SKEPTIC §H3: cache stores RAW VAE
        // posterior latent `[1, 16, H/8, W/8]` (no shift/scale, no patchify).
        // We apply `(raw - SHIFT) * SCALE` (correct BFL direction — H2) and
        // `pack_latents` at train time. Matches upstream Python contract
        // (`BaseFluxSetup.py:235`). Also: we no longer cache img_ids/txt_ids;
        // they are recomputed from latent shape per step (cheap, mirrors OT).
        let latent_raw = sample.get("latent")
            .ok_or_else(|| anyhow::anyhow!("missing 'latent'"))?.to_dtype(DType::BF16)?;
        let t5 = sample.get("t5_embed")
            .ok_or_else(|| anyhow::anyhow!("missing 't5_embed'"))?.to_dtype(DType::BF16)?;
        let clip_pool = sample.get("clip_pool")
            .ok_or_else(|| anyhow::anyhow!("missing 'clip_pool'"))?.to_dtype(DType::BF16)?;

        // VAE shift/scale (H2: encode = `(raw - shift) * scale` — multiply,
        // not divide; pre-fix prepare_flux divided which left latents at
        // ~7.7× the variance the BFL DiT was trained on).
        let latent_scaled = latent_raw
            .add_scalar(-SHIFT)?
            .mul_scalar(SCALE)?;
        let (latent, h_tok, w_tok) = pack_latents(&latent_scaled)?;
        let latent = latent.to_dtype(DType::BF16)?;
        let n_txt = t5.shape().dims()[1];
        let img_ids = build_img_ids(h_tok, w_tok, device.clone())?
            .to_dtype(DType::BF16)?;
        let txt_ids = build_txt_ids(n_txt, device.clone())?
            .to_dtype(DType::BF16)?;

        // Flow-matching: t in [0, 1000), sigma = (idx+1)/1000.
        let raw_t = sample_timestep_logit_normal(&mut rng);
        // Default-off: Strategy::None → returns raw_t unchanged.
        let t_continuous = timestep_bias::apply_bias(
            raw_t,
            NUM_TRAIN_TIMESTEPS as f32,
            &timestep_bias_cfg,
        );
        let sigma_idx = (t_continuous.floor() as usize).min(NUM_TRAIN_TIMESTEPS - 1);
        let sigma = (sigma_idx + 1) as f32 / NUM_TRAIN_TIMESTEPS as f32;

        let noise = Tensor::randn(latent.shape().clone(), 0.0, 1.0, device.clone())?
            .to_dtype(DType::BF16)?;
        // Pyramid / multi-resolution noise (additive). Default-off when
        // iterations == 0 → byte-identical.
        let noise = noise_modifiers::maybe_apply_multires_noise(
            &noise,
            args.multires_noise_iterations,
            args.multires_noise_discount,
            &mut rng,
        )?;
        // Phase 1: noise modifiers (default-off). Offset noise is part of the
        // clean noise distribution; input perturbation feeds the model input
        // only (target keeps the unperturbed noise). Default-off byte
        // invariance: gamma=0 → perturbed_noise == clean_noise.
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
        let noisy = perturbed_noise.mul_scalar(sigma)?
            .add(&latent.mul_scalar(1.0 - sigma)?)?;
        // Rectified-flow target: target = noise - clean.
        let target = clean_noise.sub(&latent)?;

        // Audit fix FLUX_VERIFY §H1 / SKEPTIC §H1: pass `t / 1000 ∈ [0, 1)` to
        // the model — `flux.rs::timestep_embedding` then multiplies by 1000
        // exactly once. Pre-fix the trainer passed `t ∈ [0, 1000)` and the
        // embedder multiplied again → `t * 1_000_000` in the sinusoid arg.
        // Mirrors Klein's recently-landed fix (`train_klein.rs:147`).
        let t_model = t_continuous / NUM_TRAIN_TIMESTEPS as f32;
        let timestep = Tensor::from_vec(
            vec![t_model],
            Shape::from_dims(&[1]),
            device.clone(),
        )?;

        if step == 0 {
            log::info!("step 0 | latent={:?} t5={:?} clip={:?} img_ids={:?} txt_ids={:?} sigma={:.4} t_model={:.4}",
                latent.shape().dims(), t5.shape().dims(), clip_pool.shape().dims(),
                img_ids.shape().dims(), txt_ids.shape().dims(), sigma, t_model);
        }

        let context = vec![t5, img_ids, txt_ids];
        let pred = <FluxModel as TrainableModel>::forward(
            &mut model, &noisy, &timestep, &context, Some(&clip_pool),
        )?;

        if pred.shape().dims() != target.shape().dims() {
            anyhow::bail!("pred {:?} != target {:?}", pred.shape().dims(), target.shape().dims());
        }

        // F32 mean-MSE (CONSTANT loss-weight, mse_strength=1.0).
        // Phase 1: combined loss + loss-weighting. Default-off invariant.
        let pred_f32 = pred.to_dtype(DType::F32)?;
        let target_f32 = target.to_dtype(DType::F32)?;
        let raw_loss = loss_weight::combined_loss(
            &pred_f32,
            &target_f32,
            config.mse_strength as f32,
            config.mae_strength as f32,
            args.huber_strength,
        )?;
        let loss = loss_weight::apply_loss_weight(
            &raw_loss,
            sigma,
            config.loss_weight_fn,
            args.min_snr_gamma,
            true,
        )?;
        let loss_val = loss.to_vec()?[0];
        total_loss += loss_val;

        let grads = loss.backward()?;

        // clip_grad_norm = 1.0 (preset default; matches OT BaseFluxSetup).
        // Fusion Sprint Phase 5: device-resident global L2 norm — one D2H per step.
        let grad_refs: Vec<&flame_core::Tensor> = params
            .iter()
            .filter_map(|p| grads.get(p.id()))
            .collect();
        let total_norm = flame_core::ops::grad_norm::global_l2_norm(&grad_refs)?
            .item()? as f32;
        let scale = if total_norm > CLIP_GRAD_NORM { CLIP_GRAD_NORM / total_norm } else { 1.0 };
        for p in &params {
            if let Some(g) = grads.get(p.id()) {
                let g_scaled = if scale < 1.0 { g.mul_scalar(scale)? } else { g.clone() };
                p.set_grad(g_scaled)?;
            }
        }

        // Phase 5: dispatch LR per scheduler. Default Constant + warmup_steps=0
        // is byte-equivalent to prior fixed-LR behaviour.
        let cur_lr = lr_schedule::dispatch_lr(
            &sched,
            args.lr,
            step,
            args.steps,
            args.warmup_steps,
            args.lr_min_factor,
            args.lr_cycles,
        );
        {
            let _g = AutogradContext::no_grad();
            opt.set_lr(cur_lr);
            opt.step(&params)?;
            opt.zero_grad(&params);
            if let Some(ref mut e) = ema {
                e.update_with_schedule(&params, &ema_cfg, (step + 1) as u64)
                    .map_err(|err| anyhow::anyhow!("EMA update failed at step {}: {err}", step + 1))?;
            }
        }
        AutogradContext::clear();
        model.post_optimizer_step();

        let _ = total_loss;
        eridiffusion_core::training::progress::log_step(
            step, args.steps, cache_files.len(), 1,
            loss_val, total_norm, cur_lr, t_start, board.as_ref(),
        );

        // Periodic save (full or weights-only).
        let step_num = step + 1;
        let save_now = args.save_every > 0 && step_num % args.save_every == 0 && step_num < args.steps;
        // EMA swap: when --ema --ema-validation-swap, save sees EMA-averaged weights.
        let ema_backup = if save_now && args.ema_validation_swap {
            if let Some(ref e) = ema {
                let _g = AutogradContext::no_grad();
                Some(e.swap_with_live(&params)
                    .map_err(|err| anyhow::anyhow!("EMA swap_with_live (mid) failed: {err}"))?)
            } else {
                None
            }
        } else {
            None
        };
        if save_now {
            let mid_ckpt = args.output_dir.join(format!("flux_lora_step{step_num}.safetensors"));
            if save_mode_full {
                let header = CkptHeader::from_adamw(
                    "train_flux", step_num as u64, &opt,
                    args.rank, args.lora_alpha as f32, SEED, String::new(),
                );
                let named = model.bundle.as_ref()
                    .map(|b| b.named_parameters())
                    .unwrap_or_default();
                if let Err(e) = checkpoint::save_full(&mid_ckpt, &named, &opt, &header) {
                    log::warn!("[save step {step_num}] full save failed: {e}");
                }
            } else if let Err(e) = model.save_weights(&mid_ckpt.to_string_lossy()) {
                log::warn!("[save step {step_num}] failed: {e}");
            } else {
                log::info!("[save step {step_num}] {}", mid_ckpt.display());
            }
        }
        if let (Some(backup), Some(ref e)) = (ema_backup, &ema) {
            let _g = AutogradContext::no_grad();
            e.restore_swapped(&params, backup)
                .map_err(|err| anyhow::anyhow!("EMA restore_swapped (mid) failed: {err}"))?;
        }
    }

    // Final EMA swap before final save. No restore — process exits.
    if args.ema_validation_swap {
        if let Some(ref e) = ema {
            let _g = AutogradContext::no_grad();
            let _ = e.swap_with_live(&params)
                .map_err(|err| anyhow::anyhow!("EMA swap_with_live (final) failed: {err}"))?;
            log::info!("[ema] swapped EMA shadow into live params for final save");
        }
    }

    let trained = args.steps - start_step;
    let avg_loss = if trained > 0 { total_loss / trained as f32 } else { 0.0 };
    log::info!("Training complete: {trained} new steps (total={}), avg loss={:.4}", args.steps, avg_loss);
    if let Some(b) = &board { b.set_status("completed"); }

    let ckpt = args.output_dir.join(format!("flux_lora_{}steps.safetensors", args.steps));
    if save_mode_full {
        let header = CkptHeader::from_adamw(
            "train_flux", args.steps as u64, &opt,
            args.rank, args.lora_alpha as f32, SEED, String::new(),
        );
        let named = model.bundle.as_ref()
            .map(|b| b.named_parameters())
            .unwrap_or_default();
        if let Err(e) = checkpoint::save_full(&ckpt, &named, &opt, &header) {
            log::warn!("save_full failed: {e}");
        } else {
            log::info!("Saved checkpoint to {}", ckpt.display());
        }
    } else if let Err(e) = model.save_weights(&ckpt.to_string_lossy()) {
        log::warn!("save_weights returned: {e}");
    } else {
        log::info!("Saved checkpoint to {}", ckpt.display());
    }
    Ok(())
}
