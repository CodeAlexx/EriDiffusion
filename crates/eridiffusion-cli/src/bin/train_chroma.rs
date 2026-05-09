//! train_chroma — Chroma (Lodestone Rock's FLUX-derived DiT) LoRA training.
//!
//! Chroma1-HD is a de-distilled FLUX.1 derivative: same VAE (16-ch LDM), same
//! T5-XXL text encoder, similar dual-stream (19) + single-stream (38) DiT
//! topology. Differences from Flux that matter for training:
//!   - **No CLIP-pool branch.** Modulation is produced by the
//!     `distilled_guidance_layer` ("approximator") MLP fed only the timestep.
//!     No `vector_in`, no clip pool, no guidance value injection.
//!   - **NOT distilled** — real CFG at sample time (`sample_chroma` does 2
//!     forwards per step). At training time we use unconditional/conditional
//!     pairs the same way Flux training does (caption_dropout via
//!     `--null-text-cache`).
//!   - **Forward signature**: `model.forward(latent_nchw, t5_embed, timestep)`
//!     — chroma's training model handles patchify + RoPE + unpatchify
//!     internally. No `pack_latents` / `build_img_ids` at training time.
//!
//! Cached sample format (produced by `prepare_chroma`):
//!   - `latent`:   [1, 16, H/8, W/8] BF16 — RAW Flux VAE posterior
//!   - `t5_embed`: [1, T5, 4096]    BF16 — T5-XXL hidden states
//!
//! Latent shift/scale is applied here (trainer side) — matches `train_flux`
//! and the EDv2 H3 audit fix. The archive trainer used pre-scaled caches; we
//! deliberately diverged at the prepare step.
//!
//! Modern feature surface (mirrors train_flux.rs Phase 0+):
//!   - EMA shadow + `--ema-validation-swap`
//!   - Multi-resolution noise (default off → byte-invariant)
//!   - Timestep bias (default `none` → byte-invariant)
//!   - Caption dropout via `--null-text-cache` (T5-only; no clip-pool swap)
//!   - Validation harness (held-out cache + side-RNG)
//!   - min-SNR loss weighting (sigma form, FM)
//!   - Combined MSE + MAE + Huber loss
//!   - LR scheduler family + warmup + cycles + lr_min_factor
//!   - Optimizer family CLI (Phase 1 fallback to AdamW)
//!
//! Quant rule: BF16/F32 throughout. NO FP8, NO AdamW8bit. Matches existing
//! FLUX-family trainers (Klein/Flux).

use clap::Parser;
use flame_core::{adam::AdamW, autograd::AutogradContext, DType, Shape, Tensor};
use eridiffusion_core::config::{LrScheduler, TrainConfig, TrainingMethod};
use eridiffusion_core::encoders::flux_vae::{SCALE, SHIFT};
use eridiffusion_core::lycoris::{LycorisAlgo, LycorisBundleConfig};
use eridiffusion_core::models::chroma::ChromaLoraBundle;
use eridiffusion_core::models::ChromaTrainingModel;
use eridiffusion_core::training::board::BoardWriter;
use eridiffusion_core::training::ema::ParameterEma;
use eridiffusion_core::training::features::ema_advanced::EmaConfig;
use eridiffusion_core::training::features::validation::ValidationLoop;
use eridiffusion_core::training::features::{loss_weight, lr_schedule, noise_modifiers, timestep_bias};
use eridiffusion_core::training::training_features::OptimizerKind;
use std::path::PathBuf;

const NUM_TRAIN_TIMESTEPS: usize = 1000;
const SEED: u64 = 42;
const CLIP_GRAD_NORM: f32 = 1.0;

#[derive(Parser)]
struct Args {
    /// OT-format JSON config (optional). Falls back to TrainConfig::default().
    #[arg(long)] config: Option<PathBuf>,
    #[arg(long)] cache_dir: PathBuf,
    /// Chroma transformer dir (containing `*.safetensors` shards) or single file.
    #[arg(long)] transformer: PathBuf,
    /// Training mode: `lora` (default) or `full`.
    #[arg(long, default_value = "lora")] mode: String,
    #[arg(long, default_value = "100")] steps: usize,
    #[arg(long, default_value = "16")] rank: usize,
    /// LoRA alpha. Convention: alpha = rank (effective scale = 1.0). Matches
    /// train_flux's H12 fix.
    #[arg(long, default_value = "16.0")] lora_alpha: f64,
    /// Default 1e-4. Chroma archive used 1e-4 in `boxjana_lora.toml`.
    #[arg(long, default_value = "1e-4")] lr: f32,
    #[arg(long, default_value = "output")] output_dir: PathBuf,
    /// Per-block weight streaming via BlockOffloader (required for 24GB VRAM
    /// at 1024² with 57 blocks resident).
    #[arg(long)] offload: bool,
    /// Save a LoRA checkpoint every N steps (0 = end-only).
    #[arg(long, default_value = "0")] save_every: usize,
    /// Resume LoRA weights only (no optimizer state).
    #[arg(long)] resume_lora: Option<PathBuf>,
    /// Save mode: `weights` (default) writes LoRA-only safetensors. `full`
    /// (LoRA + AdamW + step) is not yet wired for chroma — its bundle lacks
    /// `named_parameters()`. Use `weights` until that lands.
    #[arg(long, default_value = "weights")] save_mode: String,

    // ── Modern feature surface (mirror train_flux.rs Phase 0+) ──
    #[arg(long)] min_snr_gamma: Option<f32>,
    #[arg(long, default_value_t = 0.0)] caption_dropout_probability: f32,
    /// Path to a single cache file produced by `prepare_chroma` from an empty-
    /// caption sample. When `--caption-dropout-probability > 0`, the trainer
    /// loads `t5_embed` from this file and swaps it in with probability `p`
    /// per step. Chroma has no CLIP-pool branch, so only T5 swaps.
    #[arg(long)] null_text_cache: Option<PathBuf>,
    #[arg(long, default_value_t = 1.0)] noise_offset_probability: f32,
    #[arg(long, default_value_t = 0.0)] gamma_input_perturbation: f32,
    #[arg(long, default_value_t = 0.0)] huber_strength: f32,
    #[arg(long, default_value_t = 0.0)] lr_min_factor: f32,
    #[arg(long)] validation_dataset_dir: Option<PathBuf>,
    #[arg(long, default_value_t = 0)] validation_every_steps: u64,
    #[arg(long, default_value_t = 0.0)] masked_loss_weight: f32,

    // EMA
    #[arg(long, default_value_t = false)] ema: bool,
    #[arg(long, default_value_t = 1.0)] ema_inv_gamma: f32,
    #[arg(long, default_value_t = 0.6667)] ema_power: f32,
    #[arg(long, default_value_t = 0)] ema_update_after_step: u64,
    #[arg(long, default_value_t = 0.0)] ema_min_decay: f32,
    #[arg(long, default_value_t = 0.9999)] ema_max_decay: f32,
    #[arg(long, default_value_t = false)] ema_validation_swap: bool,

    // Multi-resolution noise (default-off byte invariant)
    #[arg(long, default_value_t = 0)] multires_noise_iterations: usize,
    #[arg(long, default_value_t = 0.3)] multires_noise_discount: f32,

    // Timestep bias (default-off byte invariant)
    #[arg(long, default_value = "none")] timestep_bias_strategy: String,
    #[arg(long, default_value_t = 0.0)] timestep_bias_multiplier: f32,
    #[arg(long, default_value_t = 0.0)] timestep_bias_range_min: f32,
    #[arg(long, default_value_t = 1.0)] timestep_bias_range_max: f32,

    /// Timestep distribution. `uniform` (default — matches Chroma archive's
    /// `let u: f32 = rng.gen()` byte-for-byte after FLUX shift remap),
    /// `logit_normal` (FLUX preset), `sigmoid`, `heavy_tail`, `cos_map`,
    /// `inverted_parabola`.
    #[arg(long, default_value = "uniform")] timestep_distribution: String,
    /// Distribution-specific weight knob (default 0.0 — uniform legacy).
    #[arg(long, default_value_t = 0.0)] noising_weight: f32,
    /// Distribution-specific bias knob (default 0.0).
    #[arg(long, default_value_t = 0.0)] noising_bias: f32,

    /// Optional resolution-shift override. Default `auto` mirrors the chroma
    /// archive's `shift_for_resolution(...)` formula: linear blend 1.0..3.0
    /// across token counts 256..4096.
    #[arg(long, default_value_t = 0.0)] timestep_shift: f32,

    /// Phase 1: optimizer family CLI. `adamw` (default) is wired; others
    /// log a warning and fall back to AdamW (full dispatch is a Phase 5 task).
    #[arg(long, default_value = "adamw")] optimizer: String,
    /// LR scheduler family: `constant` (default), `linear`, `cosine`,
    /// `cosine_with_restarts`, `polynomial`. Default + `warmup_steps=0` is
    /// byte-equivalent to fixed-LR.
    #[arg(long, default_value = "constant")] lr_scheduler: String,
    #[arg(long, default_value_t = 0)] warmup_steps: usize,
    #[arg(long, default_value_t = 1.0)] lr_cycles: f32,

    // ── LyCORIS algo selection (Phase 2b) ──
    //
    // `--algo lora` (default) keeps the legacy LoRALinear path — byte-identical
    // training to pre-Phase-2b. Other values select LyCORIS algos via
    // `LycorisBundleConfig::new_with_config`. `lora_alpha` and `rank` are
    // shared with the legacy CLI flags above (no separate `--lycoris-rank`).
    /// LyCORIS algo: `lora` (default, legacy path) | `locon` | `loha` | `lokr`
    /// | `full` | `oft`. `full` and `oft` build successfully but their
    /// `forward_delta` will error inside the chroma forward pass —
    /// chroma's `base + delta_on_input` call pattern is incompatible with
    /// Full/OFT semantics. Phase 2c will wire a `merge_into_base` path.
    #[arg(long, default_value = "lora")] algo: String,
    /// LoKr Kronecker split factor (ignored for non-LoKr).
    #[arg(long, default_value_t = 16)] lokr_factor: i32,
    /// OFT block size (ignored for non-OFT).
    #[arg(long, default_value_t = 32)] oft_block_size: usize,
    /// OFT Cayley-Neumann series term count (ignored for non-OFT).
    #[arg(long, default_value_t = 5)] oft_neumann_terms: usize,
    /// LoCon / LoHa / LoKr conv variant — Tucker decomposition for non-1×1
    /// kernels. Chroma is linear-only so this is currently a no-op.
    #[arg(long, default_value_t = false)] use_tucker: bool,
    /// LoKr only: factorize both W1 *and* W2 (default false: only W2).
    #[arg(long, default_value_t = false)] decompose_both: bool,
    /// Enable DoRA (weight-decomposed LoRA). Applies to LoCon/LoHa/LoKr
    /// (Full inherits, OFT errors).
    ///
    /// Phase 2b limitation: chroma's bundle ctor doesn't have access to the
    /// streamed block weights at construction time, so DoRA's magnitude is
    /// initialized from `||I||_2 = 1` rather than `||W_orig||_2`. The
    /// trainer should still converge but will spend the first few hundred
    /// steps adjusting the magnitude. Phase 2c will wire pre-load
    /// magnitude init.
    #[arg(long, default_value_t = false)] dora: bool,
    /// DoRA magnitude axis. Default `true` matches lycoris-upstream
    /// (norm over input dims, magnitude shape `[out, 1]`).
    #[arg(long, default_value_t = true)] dora_wd_on_out: bool,
    #[arg(long, default_value_t = 1e-6)] dora_eps: f32,
}

/// Resolution-dependent timestep shift (matches the chroma archive's
/// `shift_for_resolution` and FLUX/Klein convention).
fn shift_for_resolution(h_lat: usize, w_lat: usize) -> f32 {
    let tokens = (h_lat / 2) * (w_lat / 2);
    let t = ((tokens as f32 - 256.0) / (4096.0 - 256.0)).clamp(0.0, 1.0);
    1.0 + t * (3.0 - 1.0)
}

/// Sample a base timestep `u ∈ [0, 1]` from the configured distribution. With
/// `(uniform, 0, 0)` this matches `let u: f32 = rng.gen()` byte-for-byte —
/// the chroma archive's legacy default.
fn sample_base_u(
    rng: &mut rand::rngs::StdRng,
    distribution: &str,
    weight: f32,
    bias: f32,
) -> anyhow::Result<f32> {
    use rand::Rng;
    use rand_distr::{Distribution, Normal};
    match distribution {
        "uniform" => Ok(rng.r#gen::<f32>()),
        "logit_normal" => {
            // OT semantics: noising_weight + 1.0 = scale, noising_bias = mean
            let scale = (weight + 1.0).max(1.0e-6);
            let normal = Normal::new(bias, scale)
                .map_err(|e| anyhow::anyhow!("logit_normal Normal: {e}"))?;
            let z = normal.sample(rng);
            Ok(1.0 / (1.0 + (-z).exp()))
        }
        "sigmoid" => {
            // Match musubi's sigmoid: u ~ N(bias, weight+1) then sigmoid.
            // (For Chroma we keep this for parity with archive plumbing.)
            let scale = (weight + 1.0).max(1.0e-6);
            let normal = Normal::new(bias, scale)
                .map_err(|e| anyhow::anyhow!("sigmoid Normal: {e}"))?;
            let z = normal.sample(rng);
            Ok(1.0 / (1.0 + (-z).exp()))
        }
        other => anyhow::bail!(
            "--timestep-distribution `{other}` not yet wired in train_chroma; \
             supported: uniform, logit_normal, sigmoid"
        ),
    }
}

fn main() -> anyhow::Result<()> {
    use rand::SeedableRng;
    env_logger::init();
    let args = Args::parse();
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

    // Plumb CLI args into config (default-off, modern feature rollout).
    config.min_snr_gamma = args.min_snr_gamma;
    config.caption_dropout_probability = args.caption_dropout_probability;
    config.noise_offset_probability = args.noise_offset_probability;
    config.gamma_input_perturbation = args.gamma_input_perturbation;
    config.huber_strength = args.huber_strength;
    config.lr_min_factor = args.lr_min_factor;
    config.validation_dataset_dir = args.validation_dataset_dir.clone();
    config.validation_every_steps = args.validation_every_steps;
    config.masked_loss_weight = args.masked_loss_weight;
    config.ema_inv_gamma = args.ema_inv_gamma;
    config.ema_power = args.ema_power;
    config.ema_update_after_step = args.ema_update_after_step;
    config.ema_min_decay = args.ema_min_decay;

    log::info!(
        "[Chroma] loading transformer from {} (mode={}, rank={}, alpha={}, offload={})",
        args.transformer.display(),
        args.mode,
        args.rank,
        args.lora_alpha,
        args.offload,
    );

    // Phase 2b: parse the LyCORIS algo selector. `lora` (default) keeps the
    // legacy LoRALinear bundle constructed inside `ChromaTrainingModel::load`.
    // Anything else swaps the bundle in-place after model construction so we
    // don't have to re-plumb the per-trainer constructor signatures.
    //
    // NOTE: `LycorisAlgo::parse("lora")` aliases to `LycorisAlgo::LoCon`
    // (since LoCon-Linear is the canonical LoRA decomposition). For chroma
    // we need to distinguish LEGACY plain `LoRALinear` (byte-identical) from
    // the new `LycorisAdapter::LoCon` path, so re-map `"lora"` → `None` here
    // explicitly. Users who want the new LoCon path pass `--algo locon`.
    let algo_str = args.algo.trim().to_ascii_lowercase();
    let algo = if algo_str == "lora" || algo_str == "none" || algo_str.is_empty() {
        LycorisAlgo::None
    } else {
        LycorisAlgo::parse(&args.algo).map_err(|e| anyhow::anyhow!("--algo: {e}"))?
    };
    // Default storage (F32) inherited from `LycorisBundleConfig::default()`.
    // This matches the trainer-side AdamW state requirement; do NOT switch
    // to BF16/FP8 — chroma trainer is BF16/F32-only (see top-of-file rule).
    let lyc_config = LycorisBundleConfig {
        algo,
        rank: args.rank,
        alpha: args.lora_alpha as f32,
        factor: args.lokr_factor,
        block_size: args.oft_block_size,
        neumann_terms: args.oft_neumann_terms,
        use_tucker: args.use_tucker,
        decompose_both: args.decompose_both,
        use_scalar: false,
        dora: args.dora,
        dora_wd_on_out: args.dora_wd_on_out,
        dora_eps: args.dora_eps,
        ..LycorisBundleConfig::default()
    };

    let mut model = if args.offload {
        ChromaTrainingModel::load_swapped(
            &args.transformer,
            &args.mode,
            args.rank,
            args.lora_alpha as f32,
            device.clone(),
            SEED,
        )?
    } else {
        ChromaTrainingModel::load(
            &args.transformer,
            &args.mode,
            args.rank,
            args.lora_alpha as f32,
            device.clone(),
            SEED,
        )?
    };

    // If a LyCORIS algo other than the legacy plain LoRA was requested, swap
    // the bundle. Plain `--algo lora` (or `lora`/`none`) keeps the legacy
    // bundle as-is so this branch is byte-equivalent to the pre-Phase-2b
    // pipeline.
    if algo != LycorisAlgo::None && args.mode == "lora" {
        log::info!(
            "[Chroma] LyCORIS algo='{}' rank={} alpha={} factor={} block_size={} dora={}",
            algo.as_str(),
            lyc_config.rank,
            lyc_config.alpha,
            lyc_config.factor,
            lyc_config.block_size,
            lyc_config.dora,
        );
        if matches!(algo, LycorisAlgo::Full | LycorisAlgo::Oft) {
            log::warn!(
                "[Chroma] algo='{}' selected — bundle construction will succeed, but \
                 forward_delta will error inside chroma's `base + delta_on_input` call \
                 pattern. Phase 2c will wire merge-into-base for these algos.",
                algo.as_str()
            );
        }
        let new_bundle = ChromaLoraBundle::new_with_config(&lyc_config, device.clone(), SEED)
            .map_err(|e| anyhow::anyhow!("LyCORIS bundle construction: {e}"))?;
        model.bundle = Some(new_bundle);
    } else if algo == LycorisAlgo::None {
        // Explicit log: legacy path — no swap.
        log::info!("[Chroma] algo='lora' (legacy LoRALinear path, byte-identical)");
    }

    let params = model.parameters();
    log::info!("[Chroma] {} trainable tensors", params.len());
    if params.is_empty() {
        anyhow::bail!("No trainable parameters — check mode={}", args.mode);
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

    // EMA shadow.
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

    // Timestep bias config (default-off byte invariance).
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

    // Caption dropout: chroma has no clip-pool, so we only swap T5.
    let mut effective_caption_dropout_prob = args.caption_dropout_probability;
    let null_t5: Option<Tensor> = if effective_caption_dropout_prob > 0.0 {
        match args.null_text_cache.as_ref() {
            Some(p) => match flame_core::serialization::load_file(p, &device) {
                Ok(s) => {
                    let nt5 = s.get("t5_embed")
                        .ok_or_else(|| anyhow::anyhow!("--null-text-cache missing 't5_embed'"))?
                        .to_dtype(DType::BF16)?;
                    log::info!(
                        "[caption-dropout] WIRED — prob={:.3} (null_t5={:?})",
                        effective_caption_dropout_prob,
                        nt5.shape().dims()
                    );
                    Some(nt5)
                }
                Err(e) => {
                    log::warn!(
                        "[caption-dropout] failed to load --null-text-cache {}: {e} — feature disabled",
                        p.display()
                    );
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

    if let Some(resume_path) = args.resume_lora.as_ref() {
        log::info!("Resuming LoRA weights from {}", resume_path.display());
        // 2026-05-09: wired. `ChromaLoraBundle::load_weights` mutates the
        // existing bundle's adapters in-place via Parameter::set_data, so
        // the AdamW optimizer's parameter list (already built below) keeps
        // referencing the same Parameter IDs.
        match model.bundle.as_ref() {
            Some(bundle) => {
                bundle.load_weights(resume_path, device.clone())
                    .map_err(|e| anyhow::anyhow!("--resume-lora load: {e}"))?;
                log::info!("Resumed {} LoRA adapters", bundle.num_adapters());
            }
            None => anyhow::bail!("--resume-lora requires LoRA mode (--mode lora), not full"),
        }
    }

    let save_mode_full = match args.save_mode.as_str() {
        "full" => {
            anyhow::bail!(
                "--save-mode full not yet wired for chroma — its bundle lacks named_parameters(). \
                 Use --save-mode weights for v1."
            );
        }
        "weights" => false,
        other => anyhow::bail!("--save-mode must be `weights`, got `{other}`"),
    };
    let _ = save_mode_full; // reserved for future full-resume wiring

    let mut cache_files: Vec<PathBuf> = std::fs::read_dir(&args.cache_dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map_or(false, |e| e == "safetensors"))
        .collect();
    cache_files.sort();
    if cache_files.is_empty() {
        anyhow::bail!("No cached samples in {:?}", args.cache_dir);
    }
    log::info!("[Chroma] {} cached samples", cache_files.len());

    // Seed both flame_core RNG and host RNG from SEED=42 (matches train_flux).
    flame_core::rng::set_seed(SEED)
        .map_err(|e| anyhow::anyhow!("flame_core set_seed: {e}"))?;
    let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);

    let board = BoardWriter::open(
        &args.output_dir,
        BoardWriter::new_session_id(),
        None,
    ).map_err(|e| log::warn!("board.db open failed: {e}")).ok();
    if let Some(b) = &board {
        log::info!("SerenityBoard: writing scalars to {}", b.db_path.display());
    }
    let t_start = std::time::Instant::now();
    let mut total_loss = 0f32;

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

    let sched: LrScheduler = lr_schedule::parse_cli_scheduler(&args.lr_scheduler);

    for step in 0..args.steps {
        let cache_idx = step % cache_files.len();
        let sample = flame_core::serialization::load_file(&cache_files[cache_idx], &device)?;

        // RAW VAE posterior — apply `(raw - SHIFT) * SCALE` per step (same as
        // train_flux H3 fix). Chroma's DiT was trained on shift/scaled latents.
        let latent_raw = sample.get("latent")
            .ok_or_else(|| anyhow::anyhow!("missing 'latent' in {}", cache_files[cache_idx].display()))?
            .to_dtype(DType::BF16)?;
        let t5 = sample.get("t5_embed")
            .ok_or_else(|| anyhow::anyhow!("missing 't5_embed' in {}", cache_files[cache_idx].display()))?
            .to_dtype(DType::BF16)?;

        // Caption dropout (T5-only — chroma has no clip pool).
        let t5 = if let Some(ref nt5) = null_t5 {
            use rand::Rng;
            if rng.r#gen::<f32>() < effective_caption_dropout_prob {
                nt5.clone()
            } else {
                t5
            }
        } else {
            t5
        };

        // VAE shift/scale. (raw - shift) * scale — multiply, not divide.
        let latent = latent_raw
            .add_scalar(-SHIFT)?
            .mul_scalar(SCALE)?
            .to_dtype(DType::BF16)?;

        let lat_dims = latent.shape().dims().to_vec();
        if lat_dims.len() != 4 {
            anyhow::bail!("expected 4D latent [B, C, H, W], got {:?}", lat_dims);
        }
        let (h_lat, w_lat) = (lat_dims[2], lat_dims[3]);

        // Resolution-dependent shift (default behaviour). Override via --timestep-shift.
        let shift = if args.timestep_shift > 0.0 {
            args.timestep_shift
        } else {
            shift_for_resolution(h_lat, w_lat)
        };

        // Sample base u and apply FLUX shift remap.
        let u_base = sample_base_u(
            &mut rng,
            &args.timestep_distribution,
            args.noising_weight,
            args.noising_bias,
        )?;
        // Default-off: Strategy::None returns u_base unchanged (the bias module
        // is in [0, NUM_TRAIN_TIMESTEPS] units, so we lift to that range and
        // back).
        let u_t = timestep_bias::apply_bias(
            u_base * NUM_TRAIN_TIMESTEPS as f32,
            NUM_TRAIN_TIMESTEPS as f32,
            &timestep_bias_cfg,
        ) / NUM_TRAIN_TIMESTEPS as f32;
        // Apply FLUX shift remap: sigma = shift * u / (1 + (shift - 1) * u).
        let sigma = shift * u_t / (1.0 + (shift - 1.0) * u_t);

        // Clean noise + multires + offset + perturbation. All default-off
        // are byte-invariant.
        let noise = Tensor::randn(latent.shape().clone(), 0.0, 1.0, device.clone())?
            .to_dtype(DType::BF16)?;
        let noise = noise_modifiers::maybe_apply_multires_noise(
            &noise,
            args.multires_noise_iterations,
            args.multires_noise_discount,
            &mut rng,
        )?;
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

        // x_t = (1 - sigma) * latent + sigma * noise (FLUX flow-matching).
        let noisy = latent.mul_scalar(1.0 - sigma)?
            .add(&perturbed_noise.mul_scalar(sigma)?)?;
        // Rectified-flow target: noise - clean (matches archive `pipeline.rs`).
        let target = clean_noise.sub(&latent)?;

        // Chroma's forward expects sigma directly as the timestep input
        // (not `sigma * 1000`). Matches `pipeline.rs::prepare_inputs` and
        // `sample_chroma`'s denoise loop.
        let timestep = Tensor::from_vec(
            vec![sigma],
            Shape::from_dims(&[1]),
            device.clone(),
        )?.to_dtype(DType::BF16)?;

        if step == 0 {
            log::info!(
                "step 0 | latent={:?} t5={:?} sigma={:.4} shift={:.2}",
                latent.shape().dims(), t5.shape().dims(), sigma, shift,
            );
        }

        let pred = model.forward(&noisy, &t5, &timestep)?;

        if pred.shape().dims() != target.shape().dims() {
            anyhow::bail!("pred {:?} != target {:?}", pred.shape().dims(), target.shape().dims());
        }

        // Combined MSE+MAE+Huber + min-SNR weighting (default config keeps
        // mse_strength=1, mae=0, huber=0 → straight MSE; min-snr=None is no-op).
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
            true, // FM (sigma form)
        )?;
        let loss_val = loss.to_vec()?[0];
        total_loss += loss_val;

        let grads = loss.backward()?;

        // Global L2 grad clip = 1.0 (preset default).
        let grad_refs: Vec<&Tensor> = params
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
            // Refresh chroma's LoRA cache so the LoRA contribution is visible
            // to the next forward (matches archive pattern).
            model.refresh_lora_cache();
            if let Some(ref mut e) = ema {
                e.update_with_schedule(&params, &ema_cfg, (step + 1) as u64)
                    .map_err(|err| anyhow::anyhow!("EMA update failed at step {}: {err}", step + 1))?;
            }
        }
        AutogradContext::clear();
        // Flush GPU allocation pool (matches archive — chroma forward + bwd
        // accumulates a lot of intermediates).
        flame_core::cuda_alloc_pool::clear_pool_cache();
        device.synchronize().ok();

        let _ = total_loss;
        eridiffusion_core::training::progress::log_step(
            step, args.steps, cache_files.len(), 1,
            loss_val, total_norm, cur_lr, t_start, board.as_ref(),
        );

        // Validation eval pass (no_grad), side-RNG seeded as `SEED ^ (step+1)`.
        if let Some(ref vloop) = validation_loop {
            if vloop.should_run(step + 1) {
                let mut sum = 0.0_f32;
                let mut count = 0_usize;
                for vfile in &vloop.cache_files {
                    let _g = AutogradContext::no_grad();
                    let sample = match flame_core::serialization::load_file(vfile, &device) {
                        Ok(s) => s,
                        Err(e) => {
                            log::warn!("[validation] load {} failed: {e}", vfile.display());
                            continue;
                        }
                    };
                    let v_latent_raw = match sample.get("latent") {
                        Some(t) => t.to_dtype(DType::BF16)?,
                        None => continue,
                    };
                    let v_t5 = match sample.get("t5_embed") {
                        Some(t) => t.to_dtype(DType::BF16)?,
                        None => continue,
                    };
                    let v_latent = v_latent_raw.add_scalar(-SHIFT)?
                        .mul_scalar(SCALE)?
                        .to_dtype(DType::BF16)?;
                    let v_dims = v_latent.shape().dims().to_vec();
                    let (vh, vw) = (v_dims[2], v_dims[3]);
                    let v_shift = if args.timestep_shift > 0.0 {
                        args.timestep_shift
                    } else {
                        shift_for_resolution(vh, vw)
                    };
                    let mut vrng = rand::rngs::StdRng::seed_from_u64(SEED ^ (step as u64 + 1));
                    let v_u = sample_base_u(
                        &mut vrng,
                        &args.timestep_distribution,
                        args.noising_weight,
                        args.noising_bias,
                    )?;
                    let v_u_t = timestep_bias::apply_bias(
                        v_u * NUM_TRAIN_TIMESTEPS as f32,
                        NUM_TRAIN_TIMESTEPS as f32,
                        &timestep_bias_cfg,
                    ) / NUM_TRAIN_TIMESTEPS as f32;
                    let v_sigma = v_shift * v_u_t / (1.0 + (v_shift - 1.0) * v_u_t);
                    let v_noise = Tensor::randn(v_latent.shape().clone(), 0.0, 1.0, device.clone())?
                        .to_dtype(DType::BF16)?;
                    let v_noisy = v_latent.mul_scalar(1.0 - v_sigma)?
                        .add(&v_noise.mul_scalar(v_sigma)?)?;
                    let v_target = v_noise.sub(&v_latent)?;
                    let v_timestep = Tensor::from_vec(
                        vec![v_sigma],
                        Shape::from_dims(&[1]),
                        device.clone(),
                    )?.to_dtype(DType::BF16)?;
                    let v_pred = match model.forward(&v_noisy, &v_t5, &v_timestep) {
                        Ok(p) => p,
                        Err(e) => {
                            log::warn!("[validation] forward failed: {e}");
                            continue;
                        }
                    };
                    let v_loss = v_pred.to_dtype(DType::F32)?
                        .sub(&v_target.to_dtype(DType::F32)?)?
                        .square()?
                        .mean()?;
                    let v_loss_val = v_loss.to_vec()?[0];
                    if v_loss_val.is_finite() {
                        sum += v_loss_val;
                        count += 1;
                    }
                    AutogradContext::clear();
                }
                if count > 0 {
                    let val_avg = sum / count as f32;
                    log::info!(
                        "[validation step={}] loss/val = {:.4} ({} samples)",
                        step + 1,
                        val_avg,
                        count
                    );
                    if let Some(b) = &board {
                        b.log_scalar("loss/val", (step + 1) as u64, val_avg as f64);
                    }
                }
            }
        }

        // Periodic save (weights-only — full mode is not yet wired).
        let step_num = step + 1;
        let save_now = args.save_every > 0 && step_num % args.save_every == 0 && step_num < args.steps;
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
            let mid_ckpt = args.output_dir.join(format!("chroma_lora_step{step_num}.safetensors"));
            if let Err(e) = model.save_weights(&mid_ckpt) {
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

    // Final EMA swap. No restore — process exits.
    if args.ema_validation_swap {
        if let Some(ref e) = ema {
            let _g = AutogradContext::no_grad();
            let _ = e.swap_with_live(&params)
                .map_err(|err| anyhow::anyhow!("EMA swap_with_live (final) failed: {err}"))?;
            log::info!("[ema] swapped EMA shadow into live params for final save");
        }
    }

    let avg_loss = if args.steps > 0 { total_loss / args.steps as f32 } else { 0.0 };
    log::info!("Training complete: {} steps, avg loss={:.4}", args.steps, avg_loss);
    if let Some(b) = &board { b.set_status("completed"); }

    let suffix = if args.mode == "full" { "full" } else { "lora" };
    let ckpt = args.output_dir.join(format!("chroma_{suffix}_{}steps.safetensors", args.steps));
    if let Err(e) = model.save_weights(&ckpt) {
        log::warn!("save_weights returned: {e}");
    } else {
        log::info!("Saved checkpoint to {}", ckpt.display());
    }
    Ok(())
}
