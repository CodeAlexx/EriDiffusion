//! train_wan22 — Wan 2.2 video DiT LoRA trainer with dual-expert dispatch.
//!
//! Wan 2.2 ships in two model families:
//!
//! 1. **TI2V-5B** — single transformer (no dual expert).
//! 2. **T2V/I2V-A14B** — TWO transformers ("high noise" + "low noise"),
//!    each ~14B params. Per training step, the active expert is chosen
//!    from the sampled timestep:
//!
//!    ```text
//!    if t_continuous >= --noise-boundary  →  high-noise expert
//!    else                                 →  low-noise expert
//!    ```
//!
//!    Each expert maintains its OWN LoRA bundle. Per-step gradient
//!    flows ONLY through the active expert's LoRA params; the other
//!    expert's params are skipped by the optimizer that step.
//!
//! ## Quant policy (Wan 2.2 only)
//! Per `feedback_wan22_quant_exception.md` the project-wide "no quant"
//! rule is relaxed for Wan 2.2: 28B params at FP16 don't fit on 24 GB,
//! so `--weight-dtype fp8_scaled` is permitted. Today flame-core has
//! no FP8 runtime DType — `fp8_scaled` upcasts to BF16 on load (so
//! disk savings only). True FP8-resident weights are an open flame-core
//! work item; surfaced here for the bug-fixer to flag.
//!
//! ## Status
//! The Wan 2.2 transformer forward is NOT yet ported (see
//! `crates/eridiffusion-core/src/models/wan22.rs` module docstring).
//! This binary builds, parses CLI, loads both experts, decides the
//! dispatch correctly, and wires the modern feature surface — but every
//! `forward` call hits a typed `not yet ported` error. Use
//! `--max-steps 0` to dry-run wiring.

use clap::Parser;
use std::path::PathBuf;

use eridiffusion_core::config::{LrScheduler, TrainConfig, TrainingMethod};
use eridiffusion_core::models::wan22::{Wan22Config, Wan22Model, Wan22Variant};
use eridiffusion_core::sampler::wan22_sampler::{
    self as wan22, Expert, DEFAULT_NOISE_BOUNDARY_T2V, DEFAULT_SHIFT_TI2V_5B,
};
use eridiffusion_core::training::board::BoardWriter;
use eridiffusion_core::training::ema::ParameterEma;
use eridiffusion_core::training::features::{
    ema_advanced::EmaConfig, loss_weight, lr_schedule, timestep_bias,
    validation::ValidationLoop,
};
use eridiffusion_core::training::training_features::OptimizerKind;
use flame_core::adam::AdamW;
use flame_core::autograd::AutogradContext;
use flame_core::{DType, Shape, Tensor};

const SEED: u64 = 42;
const NUM_TRAIN_TIMESTEPS: f32 = 1000.0;

#[derive(Parser)]
struct Args {
    #[arg(long)] config: PathBuf,
    #[arg(long)] cache_dir: PathBuf,

    // ── Wan 2.2 dual-expert checkpoints ─────────────────────────────────
    /// High-noise expert checkpoint. Used when `t_continuous >=
    /// --noise-boundary`. Required for 14B; ignored for 5B.
    #[arg(long)] high_noise: Option<PathBuf>,
    /// Low-noise expert checkpoint. Used when `t_continuous <
    /// --noise-boundary`. For 5B (single expert) this is the only
    /// checkpoint and must be set.
    #[arg(long)] low_noise: PathBuf,
    /// Continuous-`t` boundary for dual-expert dispatch. Default
    /// matches Wan 2.2 T2V-A14B (`0.875 * 1000 = 875` timesteps).
    /// Ignored when `--variant ti2v_5b`.
    #[arg(long, default_value_t = DEFAULT_NOISE_BOUNDARY_T2V)]
    noise_boundary: f32,

    /// Storage dtype for the frozen transformer weights. One of
    /// `bf16 | fp16 | fp8_scaled`. `fp8_scaled` accepts on-disk FP8
    /// (only meaningful gain is disk space until flame-core grows an
    /// FP8 runtime DType). LoRA params remain F32.
    #[arg(long, default_value = "bf16")] weight_dtype: String,

    /// Wan 2.2 variant: `ti2v_5b` (single expert, dim=3072) or
    /// `t2v_14b` (dual expert, dim=5120) or `i2v_14b` (dual, image-
    /// conditioned — out of scope for this port).
    #[arg(long, default_value = "t2v_14b")] variant: String,

    /// Wan VAE checkpoint path. Used by the sampler/preview path; the
    /// trainer itself consumes pre-cached latents.
    #[arg(long)] vae: Option<PathBuf>,

    // ── Training surface (mirrors Klein/LTX-2) ──────────────────────────
    #[arg(long, default_value = "2000")] steps: usize,
    #[arg(long, default_value = "16")] rank: usize,
    #[arg(long, default_value = "16.0")] lora_alpha: f64,
    #[arg(long, default_value = "5e-5")] lr: f32,
    #[arg(long, default_value = "1")] batch_size: usize,
    #[arg(long, default_value = "output")] output_dir: PathBuf,

    /// Time-shift (`sample_shift` in the Wan repo). Defaults to TI2V-5B
    /// official 5.0; T2V-A14B reference also uses 5.0.
    #[arg(long, default_value_t = DEFAULT_SHIFT_TI2V_5B)] shift: f32,

    /// `sigmoid_scale` for the logit-normal sampler. 1.0 reproduces the
    /// archive default.
    #[arg(long, default_value_t = 1.0)] sigmoid_scale: f32,

    /// `logit_normal | uniform` — see archive `schedule.rs`. Both
    /// then get `apply_time_shift(_, shift)` applied on top.
    #[arg(long, default_value = "logit_normal")] timestep_method: String,

    #[arg(long, default_value = "0")] save_every: usize,
    #[arg(long, default_value = "0")] sample_every: usize,
    #[arg(long, default_value = "")] sample_prompt: String,
    #[arg(long, default_value = "256")] sample_size: usize,
    #[arg(long, default_value = "20")] sample_steps: usize,
    #[arg(long, default_value = "5.0")] sample_cfg: f32,
    #[arg(long, default_value = "42")] sample_seed: u64,

    /// Resume training from a previous LoRA checkpoint pair.
    #[arg(long)] resume_high_lora: Option<PathBuf>,
    #[arg(long)] resume_low_lora: Option<PathBuf>,

    /// Optimizer family. AdamW8bit explicitly permitted for Wan 2.2
    /// per `feedback_wan22_quant_exception.md`. Phase 1 falls back to
    /// AdamW for unrecognised values.
    #[arg(long, default_value = "adamw")] optimizer: String,

    /// Hard upper bound — useful for smoke tests / dry-runs.
    #[arg(long)] max_steps: Option<usize>,

    // ── Modern feature surface (mirror Klein) ──────────────────────────
    #[arg(long)] min_snr_gamma: Option<f32>,
    #[arg(long, default_value_t = 0.0)] caption_dropout_probability: f32,
    #[arg(long)] null_text_cache: Option<PathBuf>,
    #[arg(long, default_value_t = 1.0)] noise_offset_probability: f32,
    #[arg(long, default_value_t = 0.0)] gamma_input_perturbation: f32,
    #[arg(long, default_value_t = 0.0)] huber_strength: f32,
    #[arg(long, default_value_t = 0.0)] lr_min_factor: f32,
    #[arg(long)] validation_dataset_dir: Option<PathBuf>,
    #[arg(long, default_value_t = 0)] validation_every_steps: u64,
    #[arg(long, num_args = 0..)] multi_backend_weights: Vec<f32>,
    #[arg(long, num_args = 0..)] multi_backend_cache_dirs: Vec<PathBuf>,
    #[arg(long)] validation_prompts_file: Option<PathBuf>,
    #[arg(long, default_value_t = 0.0)] masked_loss_weight: f32,

    /// EMA — default-off → byte-identical to no-EMA.
    #[arg(long, default_value_t = false)] ema: bool,
    #[arg(long, default_value_t = 1.0)] ema_inv_gamma: f32,
    #[arg(long, default_value_t = 0.6667)] ema_power: f32,
    #[arg(long, default_value_t = 0)] ema_update_after_step: u64,
    #[arg(long, default_value_t = 0.0)] ema_min_decay: f32,
    #[arg(long, default_value_t = 0.9999)] ema_max_decay: f32,
    #[arg(long, default_value_t = false)] ema_validation_swap: bool,

    /// Multi-resolution noise — 4D-only helper; Wan latents are 5D
    /// `[B, C, F, H, W]` so this emits a warn-and-skip (kept for CLI
    /// uniformity).
    #[arg(long, default_value_t = 0)] multires_noise_iterations: usize,
    #[arg(long, default_value_t = 0.3)] multires_noise_discount: f32,

    /// Timestep biasing — default `none` is byte-identical to no
    /// biasing.
    #[arg(long, default_value = "none")] timestep_bias_strategy: String,
    #[arg(long, default_value_t = 0.0)] timestep_bias_multiplier: f32,
    #[arg(long, default_value_t = 0.0)] timestep_bias_range_min: f32,
    #[arg(long, default_value_t = 1.0)] timestep_bias_range_max: f32,

    #[arg(long)] tread_route_pattern: Option<String>,

    #[arg(long, num_args = 0..)] multi_backend_repeats: Vec<u32>,
    #[arg(long, default_value_t = false)] caption_tag_shuffle: bool,
    #[arg(long, default_value_t = false)] cache_clear_each_epoch: bool,
    #[arg(long, default_value_t = false)] cache_invalidate: bool,
    #[arg(long, default_value = "constant")] lr_scheduler: String,
    #[arg(long, default_value_t = 0)] warmup_steps: usize,
    #[arg(long, default_value_t = 1.0)] lr_cycles: f32,
}

fn parse_weight_dtype(s: &str) -> anyhow::Result<DType> {
    match s.to_ascii_lowercase().as_str() {
        "bf16" | "bfloat16" => Ok(DType::BF16),
        "fp16" | "f16" | "float16" => Ok(DType::F16),
        // FP8 has no flame-core runtime DType today. Map to BF16 with
        // a warning so the on-disk FP8 files load (the loader upcasts
        // FP8 → F32 → BF16 during deserialization).
        "fp8" | "fp8_scaled" | "fp8_e4m3" => {
            log::warn!(
                "[wan22] --weight-dtype {} requested; flame-core has no FP8 \
                 runtime DType so weights are upcast to BF16 on load. Disk-side \
                 savings only.",
                s
            );
            Ok(DType::BF16)
        }
        other => anyhow::bail!("unknown --weight-dtype: {other}"),
    }
}

fn main() -> anyhow::Result<()> {
    use rand::SeedableRng;
    env_logger::init();
    let args = Args::parse();
    if !args.multi_backend_cache_dirs.is_empty() || !args.multi_backend_weights.is_empty() {
        log::warn!("--multi-backend-* flags are Klein-only in Phase 2; ignored here");
    }
    if args.validation_prompts_file.is_some() {
        log::warn!("--validation-prompts-file is Klein-only in Phase 2; ignored here");
    }
    std::fs::create_dir_all(&args.output_dir)?;

    flame_core::config::set_default_dtype(DType::BF16);
    let device = flame_core::global_cuda_device();

    // ── Variant + dual-expert wiring ────────────────────────────────────
    let variant = Wan22Variant::parse(&args.variant)
        .map_err(|e| anyhow::anyhow!("--variant: {e}"))?;
    let cfg = Wan22Config::for_variant(variant);
    let weight_dtype = parse_weight_dtype(&args.weight_dtype)?;

    let dual = variant.is_dual_expert();
    if dual && args.high_noise.is_none() {
        anyhow::bail!(
            "variant {} requires --high-noise (dual expert); only --low-noise was provided",
            variant.as_str()
        );
    }
    if !dual && args.high_noise.is_some() {
        log::warn!(
            "[wan22] variant {} is single-expert; --high-noise is ignored",
            variant.as_str()
        );
    }

    let steps = args.max_steps.unwrap_or(args.steps);

    // ── TrainConfig (re-uses common fields) ─────────────────────────────
    let mut config = TrainConfig::from_json_path(&args.config.to_string_lossy())?;
    config.training_method = TrainingMethod::Lora;
    config.lora_rank = args.rank as u64;
    config.lora_alpha = args.lora_alpha;
    config.learning_rate = args.lr as f64;
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
    config.ema_validation_swap = args.ema_validation_swap;
    config.tread_route_pattern = args.tread_route_pattern.clone();

    // ── Load expert(s) ──────────────────────────────────────────────────
    log::info!(
        "[wan22] variant={} dim={} layers={} dual={} boundary={:.4} weight_dtype={:?}",
        variant.as_str(),
        cfg.dim,
        cfg.num_layers,
        dual,
        args.noise_boundary,
        weight_dtype,
    );

    let mut low_model = Wan22Model::load(
        &args.low_noise,
        cfg.clone(),
        args.rank,
        args.lora_alpha as f32,
        weight_dtype,
        device.clone(),
        SEED,
        "low",
    )?;
    if let Some(p) = &args.resume_low_lora {
        log::info!("[wan22:low] resume LoRA <- {}", p.display());
        // LoRA bundles save as flat safetensors; loading is a per-file
        // hydrate. We surface a clear error if the format mismatches.
        let map = flame_core::serialization::load_file(p, &device)?;
        rehydrate_bundle(&mut low_model, &map)?;
    }

    let mut high_model: Option<Wan22Model> = if dual {
        let mut hm = Wan22Model::load(
            args.high_noise.as_ref().unwrap(),
            cfg.clone(),
            args.rank,
            args.lora_alpha as f32,
            weight_dtype,
            device.clone(),
            SEED ^ 0xA14B_A14B,
            "high",
        )?;
        if let Some(p) = &args.resume_high_lora {
            log::info!("[wan22:high] resume LoRA <- {}", p.display());
            let map = flame_core::serialization::load_file(p, &device)?;
            rehydrate_bundle(&mut hm, &map)?;
        }
        Some(hm)
    } else {
        None
    };

    // ── Optimizer per expert ────────────────────────────────────────────
    // Each expert gets its OWN AdamW: per-step we step ONLY the
    // optimizer of the active expert. This matches the gradient flow
    // (only one expert sees gradient per step).
    match OptimizerKind::parse(&args.optimizer) {
        Ok(OptimizerKind::AdamW) => {}
        Ok(other) => log::warn!(
            "non-AdamW optimizer {} — Phase 1 falls back to AdamW",
            other.as_str()
        ),
        Err(e) => log::warn!("--optimizer parse: {e} — falling back to AdamW"),
    }
    let mut opt_low = AdamW::new(args.lr, 0.9, 0.999, 1e-8, 0.01);
    let mut opt_high: Option<AdamW> = if dual {
        Some(AdamW::new(args.lr, 0.9, 0.999, 1e-8, 0.01))
    } else {
        None
    };

    let params_low = low_model.parameters();
    log::info!("[wan22:low] {} trainable LoRA tensors", params_low.len());
    let params_high: Vec<flame_core::parameter::Parameter> = if let Some(ref hm) = high_model {
        let p = hm.parameters();
        log::info!("[wan22:high] {} trainable LoRA tensors", p.len());
        p
    } else {
        Vec::new()
    };

    // ── Caption dropout (null text cache) ───────────────────────────────
    let mut effective_caption_dropout_prob = args.caption_dropout_probability;
    let null_text: Option<Tensor> = if effective_caption_dropout_prob > 0.0 {
        match args.null_text_cache.as_ref() {
            Some(p) => match flame_core::serialization::load_file(p, &device) {
                Ok(s) => {
                    let nt = s.get("text_embedding")
                        .ok_or_else(|| anyhow::anyhow!("--null-text-cache missing 'text_embedding'"))?
                        .to_dtype(DType::BF16)?;
                    log::info!(
                        "[caption-dropout] WIRED — prob={:.3} (null_text_embedding={:?})",
                        effective_caption_dropout_prob,
                        nt.shape().dims()
                    );
                    Some(nt)
                }
                Err(e) => {
                    log::warn!("[caption-dropout] failed to load --null-text-cache {}: {e} — feature disabled", p.display());
                    effective_caption_dropout_prob = 0.0;
                    None
                }
            },
            None => {
                log::warn!(
                    "caption_dropout_probability={:.3} requested but --null-text-cache not provided — feature disabled",
                    effective_caption_dropout_prob
                );
                effective_caption_dropout_prob = 0.0;
                None
            }
        }
    } else {
        None
    };

    // ── EMA per expert (default-off) ────────────────────────────────────
    let ema_cfg = EmaConfig {
        inv_gamma: args.ema_inv_gamma,
        power: args.ema_power,
        update_after_step: args.ema_update_after_step,
        min_decay: args.ema_min_decay,
        max_decay: args.ema_max_decay,
    };
    let mut ema_low: Option<ParameterEma> = if args.ema {
        let _g = AutogradContext::no_grad();
        Some(ParameterEma::new(&params_low, args.ema_max_decay)
            .map_err(|e| anyhow::anyhow!("EMA-low construction: {e}"))?)
    } else {
        None
    };
    let mut ema_high: Option<ParameterEma> = if args.ema && dual {
        let _g = AutogradContext::no_grad();
        Some(ParameterEma::new(&params_high, args.ema_max_decay)
            .map_err(|e| anyhow::anyhow!("EMA-high construction: {e}"))?)
    } else {
        None
    };
    if let Some(ref e) = ema_low {
        log::info!(
            "[ema:low] WIRED — {} shadow tensors, validation_swap={}",
            e.len(), args.ema_validation_swap
        );
    }
    if let Some(ref e) = ema_high {
        log::info!(
            "[ema:high] WIRED — {} shadow tensors",
            e.len()
        );
    }

    if args.multires_noise_iterations > 0 {
        log::warn!(
            "[multires-noise] Wan latents are 5D video; multires noise (4D-only helper) is skipped."
        );
    }

    // ── Timestep bias ───────────────────────────────────────────────────
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
                "[timestep-bias] strategy={} multiplier={} range=[{},{}]",
                strategy.as_str(), cfg.multiplier, cfg.range_min, cfg.range_max
            );
        }
        cfg
    };

    // ── Cache files ─────────────────────────────────────────────────────
    let mut cache_files: Vec<PathBuf> = std::fs::read_dir(&args.cache_dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map_or(false, |e| e == "safetensors"))
        .collect();
    cache_files.sort();
    if cache_files.is_empty() {
        anyhow::bail!("No cached samples in {:?}", args.cache_dir);
    }
    log::info!(
        "Found {} cached samples (batch_size={})",
        cache_files.len(), args.batch_size
    );

    // ── Validation harness ──────────────────────────────────────────────
    let validation_loop: Option<ValidationLoop> = if let (Some(dir), n) =
        (args.validation_dataset_dir.as_ref(), args.validation_every_steps)
    {
        if n > 0 {
            let v = ValidationLoop::new(dir, n)?;
            log::info!("[validation] {} held-out, every {} steps", v.len(), n);
            Some(v)
        } else { None }
    } else { None };
    let _ = validation_loop; // wired but eval pass below is a stub until forward lands

    // ── Per-step state ──────────────────────────────────────────────────
    let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);
    let board = BoardWriter::open(&args.output_dir, BoardWriter::new_session_id(), None)
        .map_err(|e| log::warn!("board.db open failed: {e}")).ok();
    if let Some(b) = &board {
        log::info!("SerenityBoard: writing scalars to {}", b.db_path.display());
    }
    let t_start = std::time::Instant::now();
    let mut total_loss = 0f32;
    let sched: LrScheduler = lr_schedule::parse_cli_scheduler(&args.lr_scheduler);

    let timestep_method = args.timestep_method.to_ascii_lowercase();
    let use_logit_normal = matches!(
        timestep_method.as_str(),
        "logit_normal" | "logitnormal"
    );

    // ── Training loop ───────────────────────────────────────────────────
    log::info!("[wan22] starting training: {} steps", steps);
    for step in 0..steps {
        // --- 1. Sample one batch (B=1 for first pass; archive matched)
        let mut batch_latents: Vec<Tensor> = Vec::with_capacity(args.batch_size);
        let mut batch_texts: Vec<Tensor> = Vec::with_capacity(args.batch_size);
        for bi in 0..args.batch_size {
            let cache_idx = (step * args.batch_size + bi) % cache_files.len();
            let sample = flame_core::serialization::load_file(&cache_files[cache_idx], &device)?;
            let latent = sample.get("latent")
                .ok_or_else(|| anyhow::anyhow!("cached sample missing 'latent'"))?
                .to_dtype(DType::BF16)?;
            let txt = sample.get("text_embedding")
                .ok_or_else(|| anyhow::anyhow!("cached sample missing 'text_embedding'"))?
                .to_dtype(DType::BF16)?;
            // Validate Wan video latent: 5D, H/W even (patch_size=(1,2,2)).
            let dims = latent.shape().dims();
            if dims.len() != 5 {
                anyhow::bail!(
                    "Wan22 latent must be 5D [B, C, F, H, W], got {:?} from {}",
                    dims, cache_files[cache_idx].display()
                );
            }
            if dims[3] % 2 != 0 || dims[4] % 2 != 0 {
                anyhow::bail!(
                    "Wan22 latent H/W must be even (patch_size=(1,2,2)), got H={}, W={} from {}",
                    dims[3], dims[4], cache_files[cache_idx].display()
                );
            }
            // Caption dropout
            let txt = if let Some(ref nt) = null_text {
                use rand::Rng;
                if rng.r#gen::<f32>() < effective_caption_dropout_prob {
                    nt.clone()
                } else {
                    txt
                }
            } else { txt };
            batch_latents.push(latent);
            batch_texts.push(txt);
        }
        let latent = if args.batch_size == 1 {
            batch_latents.pop().unwrap()
        } else {
            let refs: Vec<&Tensor> = batch_latents.iter().collect();
            Tensor::cat(&refs, 0)?.contiguous()?
        };
        let txt = if args.batch_size == 1 {
            batch_texts.pop().unwrap()
        } else {
            let refs: Vec<&Tensor> = batch_texts.iter().collect();
            Tensor::cat(&refs, 0)?.contiguous()?
        };

        // --- 2. Per-batch-element timestep + Wan time shift
        let mut t_continuous = Vec::with_capacity(args.batch_size);
        for _ in 0..args.batch_size {
            let raw_t = if use_logit_normal {
                wan22::sample_logit_normal_with_shift(&mut rng, args.shift, args.sigmoid_scale)
            } else {
                wan22::sample_uniform_with_shift(&mut rng, args.shift)
            };
            // timestep_bias works in 1000-step space.
            let t_in_steps = raw_t * NUM_TRAIN_TIMESTEPS;
            let t_biased = timestep_bias::apply_bias(
                t_in_steps,
                NUM_TRAIN_TIMESTEPS,
                &timestep_bias_cfg,
            );
            t_continuous.push((t_biased / NUM_TRAIN_TIMESTEPS).clamp(1.0e-4, 1.0 - 1.0e-4));
        }

        // --- 3. Dual-expert dispatch (uses first batch element's t)
        let chosen = if dual {
            wan22::expert_for_timestep(t_continuous[0], args.noise_boundary)
        } else {
            // 5B: only the low_model exists; route everything there.
            Expert::Low
        };

        // --- 4. Build noisy + velocity target (matches archive pipeline.rs)
        //   x_t = (1 - t) * x_1 + t * x_0   (x_1 = clean, x_0 = noise)
        //   target = x_0 - x_1
        let noise = Tensor::randn(latent.shape().clone(), 0.0, 1.0, device.clone())?
            .to_dtype(DType::BF16)?;
        let (noisy, target) = if args.batch_size == 1 {
            let t = t_continuous[0];
            let noisy = noise.mul_scalar(t)?.add(&latent.mul_scalar(1.0 - t)?)?;
            let target = noise.sub(&latent)?;
            (noisy, target)
        } else {
            let t_tensor = Tensor::from_vec(
                t_continuous.clone(),
                Shape::from_dims(&[args.batch_size, 1, 1, 1, 1]),
                device.clone(),
            )?.to_dtype(DType::BF16)?;
            let one_minus_t = t_tensor.mul_scalar(-1.0)?.add_scalar(1.0)?;
            let noisy = noise.mul(&t_tensor)?.add(&latent.mul(&one_minus_t)?)?;
            let target = noise.sub(&latent)?;
            (noisy, target)
        };

        // Wan timestep tensor: t * 1000 (matches archive prepare_inputs).
        let t_scaled: Vec<f32> = t_continuous.iter().map(|t| t * NUM_TRAIN_TIMESTEPS).collect();
        let timestep = Tensor::from_vec(
            t_scaled,
            Shape::from_dims(&[args.batch_size]),
            device.clone(),
        )?;

        if step == 0 {
            log::info!(
                "step 0 | latent={:?} text={:?} t={:.4} expert={:?}",
                latent.shape().dims(), txt.shape().dims(), t_continuous[0], chosen
            );
        }

        // --- 5. Forward through chosen expert (currently errors)
        let pred_res = match chosen {
            Expert::High => match high_model.as_mut() {
                Some(hm) => hm.forward(&noisy, &timestep, &txt),
                None => Err(eridiffusion_core::EriDiffusionError::Model(
                    "high-noise expert requested but not loaded".into(),
                )),
            },
            Expert::Low => low_model.forward(&noisy, &timestep, &txt),
        };
        let pred = match pred_res {
            Ok(p) => p,
            Err(e) => {
                // The forward is intentionally not yet ported. Surface a
                // clear error and bail so the dual-expert dispatch
                // remains testable via --max-steps 0 and the model loader
                // smoke-tests without silently producing zeros.
                anyhow::bail!(
                    "step {step} expert={:?}: {e}\n\
                     (see crates/eridiffusion-core/src/models/wan22.rs and \
                     flame-diffusion-archive/wan-trainer/src/forward_impl/ \
                     for the deferred forward port)",
                    chosen
                );
            }
        };

        if pred.shape().dims() != target.shape().dims() {
            anyhow::bail!(
                "predicted shape {:?} != target {:?}",
                pred.shape().dims(), target.shape().dims()
            );
        }

        // --- 6. Loss = mean MSE in F32 (with min-snr / loss_weight)
        let pred_f32 = pred.to_dtype(DType::F32)?;
        let target_f32 = target.to_dtype(DType::F32)?;
        let raw_loss = loss_weight::combined_loss(
            &pred_f32, &target_f32,
            config.mse_strength as f32,
            config.mae_strength as f32,
            args.huber_strength,
        )?;
        let loss = loss_weight::apply_loss_weight(
            &raw_loss,
            t_continuous[0],
            config.loss_weight_fn,
            args.min_snr_gamma,
            true,
        )?;
        let loss_val = loss.to_vec()?[0];
        total_loss += loss_val;

        // --- 7. Backward + clip-grad-norm + step (only the active expert)
        let grads = loss.backward()?;
        const CLIP_GRAD_NORM: f32 = 1.0;
        let active_params = match chosen {
            Expert::High => &params_high,
            Expert::Low  => &params_low,
        };
        let grad_refs: Vec<&flame_core::Tensor> = active_params
            .iter()
            .filter_map(|p| grads.get(p.id()))
            .collect();
        let total_norm = if grad_refs.is_empty() {
            0.0
        } else {
            flame_core::ops::grad_norm::global_l2_norm(&grad_refs)?.item()? as f32
        };
        let scale = if total_norm > CLIP_GRAD_NORM { CLIP_GRAD_NORM / total_norm } else { 1.0 };
        for param in active_params {
            if let Some(g) = grads.get(param.id()) {
                let g_scaled = if scale < 1.0 { g.mul_scalar(scale)? } else { g.clone() };
                param.set_grad(g_scaled)?;
            }
        }
        let cur_lr = lr_schedule::dispatch_lr(
            &sched, args.lr, step, steps,
            args.warmup_steps, args.lr_min_factor, args.lr_cycles,
        );
        {
            let _g = AutogradContext::no_grad();
            match chosen {
                Expert::High => {
                    if let Some(ref mut o) = opt_high {
                        o.set_lr(cur_lr);
                        o.step(&params_high)?;
                        o.zero_grad(&params_high);
                    }
                    if let Some(ref mut e) = ema_high {
                        e.update_with_schedule(&params_high, &ema_cfg, (step + 1) as u64)
                            .map_err(|err| anyhow::anyhow!("EMA-high update {step}: {err}"))?;
                    }
                    if let Some(ref hm) = high_model { hm.refresh_lora_cache(); }
                }
                Expert::Low => {
                    opt_low.set_lr(cur_lr);
                    opt_low.step(&params_low)?;
                    opt_low.zero_grad(&params_low);
                    if let Some(ref mut e) = ema_low {
                        e.update_with_schedule(&params_low, &ema_cfg, (step + 1) as u64)
                            .map_err(|err| anyhow::anyhow!("EMA-low update {step}: {err}"))?;
                    }
                    low_model.refresh_lora_cache();
                }
            }
        }
        AutogradContext::clear();

        eridiffusion_core::training::progress::log_step(
            step, steps, cache_files.len(), args.batch_size.max(1),
            loss_val, total_norm, cur_lr, t_start, board.as_ref(),
        );

        // --- 8. Periodic save
        let step_num = step + 1;
        let save_fires = args.save_every > 0 && step_num % args.save_every == 0 && step_num < steps;
        if save_fires {
            let p_low = args.output_dir.join(format!("wan22_low_lora_step{step_num}.safetensors"));
            if let Err(e) = low_model.save_weights(&p_low) {
                log::warn!("[mid-save low @ {step_num}] {e}");
            } else {
                log::info!("[mid-save low @ {step_num}] {}", p_low.display());
            }
            if let Some(ref hm) = high_model {
                let p_hi = args.output_dir.join(format!("wan22_high_lora_step{step_num}.safetensors"));
                if let Err(e) = hm.save_weights(&p_hi) {
                    log::warn!("[mid-save high @ {step_num}] {e}");
                } else {
                    log::info!("[mid-save high @ {step_num}] {}", p_hi.display());
                }
            }
        }
    }

    let avg_loss = if steps > 0 { total_loss / steps as f32 } else { 0.0 };
    log::info!("Training complete: {} steps, avg loss={:.4}", steps, avg_loss);
    if let Some(b) = &board { b.set_status("completed"); }

    let final_low = args.output_dir.join(format!("wan22_low_lora_{}steps.safetensors", steps));
    if let Err(e) = low_model.save_weights(&final_low) {
        log::warn!("save_weights low: {e}");
    } else {
        log::info!("Saved {}", final_low.display());
    }
    if let Some(ref hm) = high_model {
        let final_hi = args.output_dir.join(format!("wan22_high_lora_{}steps.safetensors", steps));
        if let Err(e) = hm.save_weights(&final_hi) {
            log::warn!("save_weights high: {e}");
        } else {
            log::info!("Saved {}", final_hi.display());
        }
    }
    Ok(())
}

/// Hydrate a Wan22Model's LoRA bundle from an on-disk safetensors map.
/// Format mirrors `Wan22LoraBundle::save`.
fn rehydrate_bundle(
    model: &mut Wan22Model,
    map: &std::collections::HashMap<String, Tensor>,
) -> anyhow::Result<()> {
    let mut hits = 0usize;
    let mut misses = 0usize;
    for ((idx, target), lora) in &model.lora.adapters {
        let prefix = format!(
            "lora_wan_blocks_{idx}_{}",
            target.key().replace('.', "_")
        );
        let a_key = format!("{prefix}.lora_A.weight");
        let b_key = format!("{prefix}.lora_B.weight");
        match (map.get(&a_key), map.get(&b_key)) {
            (Some(a), Some(b)) => {
                lora.lora_a().set_data(a.clone())?;
                lora.lora_b().set_data(b.clone())?;
                hits += 1;
            }
            _ => {
                misses += 1;
            }
        }
        let _ = (idx, target); // silence unused lint
    }
    log::info!("[wan22:{}] rehydrated {} adapters ({} missing)", model.expert_label, hits, misses);
    if hits == 0 {
        anyhow::bail!("no LoRA adapters matched in resume file (key prefix mismatch)");
    }
    Ok(())
}
