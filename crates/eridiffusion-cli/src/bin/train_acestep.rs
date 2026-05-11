//! train_acestep — ACE-Step DiT LoRA training binary.
//!
//! Patterned after train_ernie.rs. Single-sample-per-step flow matching with
//! CFG dropout. Loads `.safetensors` (or `.pt`-style with same key set)
//! produced by ACE-Step's Python preprocessing pipeline. Each cache file
//! must contain: target_latents, attention_mask, encoder_hidden_states,
//! encoder_attention_mask, context_latents.
//!
//! ACE-Step is text-to-music — there is NO prepare/sample binary in EDv2
//! yet. Data comes from the upstream ACE-Step Python pipeline.

use clap::Parser;
use flame_core::{autograd::AutogradContext, DType, Shape, Tensor};
use flame_core::gradient_clip::GradientClipper;
use eridiffusion_core::lycoris::{LoraInitType, LycorisAlgo, LycorisBundleConfig};
use eridiffusion_core::models::AceStepLoRAModel;
use eridiffusion_core::training::board::BoardWriter;
use eridiffusion_core::training::checkpoint::{self, CkptHeader};
use eridiffusion_core::config::LrScheduler;
use eridiffusion_core::training::features::{
    caption_dropout, ema_advanced::EmaConfig, loss_weight, lr_schedule, noise_modifiers, timestep_bias,
    validation::ValidationLoop,
};
use eridiffusion_core::training::ema::ParameterEma;
use eridiffusion_core::training::schedule;
use eridiffusion_core::training::training_features::{Optimizer, OptimizerKind};
use eridiffusion_core::training::training_features::timestep_dist::{TimestepConfig, TimestepDistribution};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::str::FromStr as _;
use std::path::PathBuf;

const SEED_DEFAULT: u64 = 42;

#[derive(Parser)]
struct Args {
    /// ACE-Step DiT base safetensors checkpoint.
    #[arg(long)] model: PathBuf,
    /// Directory of preprocessed `.safetensors` (or `.pt`) sample files.
    #[arg(long)] cache_dir: PathBuf,
    #[arg(long, default_value = "100")] steps: usize,
    #[arg(long, default_value = "16")] rank: usize,
    #[arg(long, default_value = "16.0")] lora_alpha: f32,
    #[arg(long, default_value = "4e-4")] lr: f32,
    #[arg(long, default_value = "200")] warmup_steps: usize,
    /// Logit-normal timestep mu (configuration_acestep_v15.py default -0.4).
    /// Used only when `--timestep-distribution=auto` (default).
    #[arg(long, default_value = "-0.4")] timestep_mu: f32,
    /// Logit-normal timestep sigma (default 1.0).
    /// Used only when `--timestep-distribution=auto` (default).
    #[arg(long, default_value = "1.0")] timestep_sigma: f32,
    /// Unified OneTrainer timestep distribution. `auto` (default) keeps
    /// the byte-equivalent legacy `(timestep_mu, timestep_sigma)` logit-normal
    /// path. Other choices: `uniform`, `sigmoid`, `logit_normal`, `heavy_tail`,
    /// `cos_map`, `inverted_parabola`. When non-`auto`, `--timestep-mu` and
    /// `--timestep-sigma` are ignored in favor of `--noising-weight` /
    /// `--noising-bias`.
    #[arg(long, default_value = "auto")] timestep_distribution: String,
    /// Distribution-specific weight knob (default 0.0). Ignored when
    /// `--timestep-distribution=auto`.
    #[arg(long, default_value_t = 0.0)] noising_weight: f32,
    /// Distribution-specific bias knob (default 0.0). Ignored when
    /// `--timestep-distribution=auto`.
    #[arg(long, default_value_t = 0.0)] noising_bias: f32,
    /// CFG dropout ratio (modeling_acestep_v15_base.py default 0.15).
    #[arg(long, default_value = "0.15")] cfg_ratio: f32,
    /// Resume LoRA weights only.
    #[arg(long, conflicts_with = "resume_full")] resume_lora: Option<PathBuf>,
    /// Full resume: LoRA + AdamW + step.
    #[arg(long, conflicts_with = "resume_lora")] resume_full: Option<PathBuf>,
    /// Save mode: `full` (LoRA + AdamW + step) or `weights` (legacy).
    #[arg(long, default_value = "full")] save_mode: String,
    #[arg(long, default_value = "0")] save_every: usize,
    #[arg(long, default_value = "output")] output_dir: PathBuf,
    #[arg(long, default_value_t = SEED_DEFAULT)] seed: u64,

    // ── Phase 0 multi-feature rollout (default-off; Phase 1+ will consume) ──
    #[arg(long)] min_snr_gamma: Option<f32>,
    #[arg(long, default_value_t = 0.0)] caption_dropout_probability: f32,
    /// Path to a single cache file produced upstream from an empty-caption
    /// sample. When `--caption-dropout-probability > 0`, the trainer loads
    /// `encoder_hidden_states` from this file and swaps it in with probability
    /// `p` per step. This is independent of `--cfg-ratio` (which uses the
    /// model's internal `null_condition_emb`); both can be active and either
    /// firing produces a null-conditioned step. If unset and dropout > 0, the
    /// feature is disabled with a warning.
    #[arg(long)] null_text_cache: Option<PathBuf>,
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
    /// Master switch for EMA shadow. Default false; loss curves are byte-
    /// identical to `--ema=false` because the shadow is parallel — only
    /// `--ema-validation-swap` exposes it at sample/checkpoint time.
    #[arg(long, default_value_t = false)] ema: bool,
    #[arg(long, default_value_t = 1.0)] ema_inv_gamma: f32,
    #[arg(long, default_value_t = 0.6667)] ema_power: f32,
    #[arg(long, default_value_t = 0)] ema_update_after_step: u64,
    #[arg(long, default_value_t = 0.0)] ema_min_decay: f32,
    /// Upper clamp for the per-step computed decay. Default 0.9999 matches
    /// diffusers EMAModel.
    #[arg(long, default_value_t = 0.9999)] ema_max_decay: f32,
    /// Swap EMA shadow into live params at sample/checkpoint time. Default
    /// false. No effect when EMA is not constructed.
    #[arg(long, default_value_t = false)] ema_validation_swap: bool,
    /// Multi-resolution / pyramid noise iterations. 0 = disabled (byte-
    /// invariant). NOTE: ACE-Step trains on non-4D audio latents
    /// `[B, C, T]`; the helper short-circuits to a no-op for non-4D
    /// inputs, so this flag is effectively a documented no-op for ACE-Step.
    #[arg(long, default_value_t = 0)] multires_noise_iterations: usize,
    /// Per-level discount factor for `--multires-noise-iterations`.
    #[arg(long, default_value_t = 0.3)] multires_noise_discount: f32,
    /// Timestep bias strategy: `none` (default), `later`, `earlier`, `range`.
    #[arg(long, default_value = "none")] timestep_bias_strategy: String,
    /// Strength for `--timestep-bias-strategy later|earlier`.
    #[arg(long, default_value_t = 0.0)] timestep_bias_multiplier: f32,
    /// Lower bound for `--timestep-bias-strategy range`, fraction of
    /// NUM_TRAIN_TIMESTEPS in [0, 1].
    #[arg(long, default_value_t = 0.0)] timestep_bias_range_min: f32,
    /// Upper bound for `--timestep-bias-strategy range`, fraction in [0, 1].
    #[arg(long, default_value_t = 1.0)] timestep_bias_range_max: f32,
    #[arg(long)] tread_route_pattern: Option<String>,
    /// Phase 1: optimizer family CLI surface (Phase 5 wires full dispatch).
    #[arg(long, default_value = "adamw")] optimizer: String,

    // ── Phase 6 multi-feature rollout (plumb-only; multi-backend wired in Klein) ──
    #[arg(long, num_args = 0..)] multi_backend_repeats: Vec<u32>,
    #[arg(long, default_value_t = false)] caption_tag_shuffle: bool,
    #[arg(long, default_value_t = false)] cache_clear_each_epoch: bool,
    #[arg(long, default_value_t = false)] cache_invalidate: bool,
    /// Phase 5: LR scheduler family. Default `constant` is byte-equivalent to
    /// the legacy `constant_with_warmup` ACE-Step has used since launch.
    /// Accepted: constant, linear, cosine, cosine_with_restarts, polynomial, rex.
    #[arg(long, default_value = "constant")] lr_scheduler: String,
    /// Phase 5: cosine-with-restarts cycle count. Ignored for other schedulers.
    #[arg(long, default_value_t = 1.0)] lr_cycles: f32,

    // ── LyCORIS algo selection (Phase 2b) ──
    //
    // `--algo lora` (default) keeps the legacy plain `LoRALinear` path —
    // byte-identical to pre-Phase-2b training. Other values build a
    // `LycorisLinear` bundle via `AceStepLoRAModel::install_lycoris_bundle`.
    // `lora_alpha` and `rank` are shared with the existing CLI flags above
    // (no separate `--lycoris-rank`).
    /// LyCORIS algo: `lora` (default, legacy path) | `locon` | `loha` | `lokr`
    /// | `full` | `oft`. `full` and `oft` build successfully but their
    /// `forward_delta` will error inside ACE-Step's `base + delta_on_input`
    /// attention call pattern — Phase 2c will wire `merge_into_base`.
    #[arg(long, default_value = "lora")] algo: String,
    /// LoKr Kronecker split factor (ignored for non-LoKr).
    #[arg(long, default_value_t = 16)] lokr_factor: i32,
    /// OFT block size (ignored for non-OFT).
    #[arg(long, default_value_t = 32)] oft_block_size: usize,
    /// OFT Cayley-Neumann series term count (ignored for non-OFT).
    #[arg(long, default_value_t = 5)] oft_neumann_terms: usize,
    /// LoCon / LoHa / LoKr conv variant — Tucker decomposition for non-1×1
    /// kernels. ACE-Step's LoRA targets (Q/K/V/O) are linear-only so this is
    /// currently a no-op.
    #[arg(long, default_value_t = false)] use_tucker: bool,
    /// LoKr only: factorize both W1 *and* W2 (default false: only W2).
    #[arg(long, default_value_t = false)] decompose_both: bool,
    /// Enable DoRA (weight-decomposed LoRA). Applies to LoCon/LoHa/LoKr
    /// (Full inherits, OFT errors).
    ///
    /// Phase 2b limitation: ACE-Step's bundle ctor doesn't read the resident
    /// base weights at construction time, so DoRA's magnitude is initialized
    /// from `||I||_2 = 1` rather than `||W_orig||_2`. Trainer should still
    /// converge — first few hundred steps adjust the magnitude. Phase 2c
    /// will wire pre-load magnitude init.
    #[arg(long, default_value_t = false)] dora: bool,
    /// DoRA magnitude axis. Default `true` matches lycoris-upstream
    /// (norm over input dims, magnitude shape `[out, 1]`).
    #[arg(long, default_value_t = true)] dora_wd_on_out: bool,
    #[arg(long, default_value_t = 1e-6)] dora_eps: f32,
    /// PEFT/SimpleTuner `--lora_init_type`. Applies to LoCon (the LoRA path)
    /// only. Choices: `default | gaussian | pissa | olora | loftq`. The
    /// PISSA/OLoRA/LoftQ variants parse but error at adapter construction
    /// because flame-core does not yet expose SVD/QR.
    #[arg(long, default_value = "default")] lora_init_type: String,
    /// SimpleTuner-style `lycoris_config preset.json`. Optional; per-target
    /// `module_algo_map` overrides apply during adapter construction.
    #[arg(long)] lycoris_config: Option<PathBuf>,
    /// SimpleTuner-parity perturbed-normal LoKr init. No-op for ACE-Step in
    /// Phase 2b (base-weight-by-target lookup not plumbed — the resident
    /// `weights` map is keyed by string, and the per-target lookup helper
    /// would mirror `AceStepLoraTarget::suffix()`. Phase 2c will wire it).
    #[arg(long, default_value_t = 0.0)] init_lokr_norm: f32,
    /// Per-element dropout on the adapter delta (training only).
    #[arg(long, default_value_t = 0.0)] lora_dropout: f32,
    /// Per-rank Bernoulli on the down-projection intermediate.
    #[arg(long, default_value_t = 0.0)] rank_dropout: f32,
    /// Per-step Bernoulli on the entire adapter.
    #[arg(long, default_value_t = 0.0)] module_dropout: f32,
    /// Rescale rank-mask by `1/mean(mask)` to preserve expectation.
    #[arg(long, default_value_t = false)] rank_dropout_scale: bool,
}

/// Apply CFG dropout: with probability `cfg_ratio`, replace `encoder_hs` with
/// `null_emb` (broadcast to encoder_hs shape).
fn apply_cfg_dropout(
    encoder_hs: &Tensor,
    null_emb: &Tensor,
    cfg_ratio: f32,
    rng: &mut StdRng,
) -> flame_core::Result<Tensor> {
    if rng.r#gen::<f32>() < cfg_ratio {
        null_emb.broadcast_to(encoder_hs.shape())
    } else {
        Ok(encoder_hs.clone())
    }
}

fn main() -> anyhow::Result<()> {
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
    if args.masked_loss_weight > 0.0 {
        log::warn!(
            "[masked-loss] --masked-loss-weight={:.3} requested but ACE-Step's audio-latent cache schema has no `latent_mask` field; flag is a no-op for this trainer.",
            args.masked_loss_weight
        );
    }
    flame_core::config::set_default_dtype(DType::BF16);
    let device = flame_core::global_cuda_device();

    log::info!("Loading ACE-Step DiT (rank={} alpha={}) from {}...",
        args.rank, args.lora_alpha, args.model.display());
    let mut model = AceStepLoRAModel::from_safetensors(
        &args.model, args.rank, args.lora_alpha, device.clone(),
    )?;

    // Phase 2b: parse the LyCORIS algo selector. `--algo lora` (or `none` /
    // empty) keeps the legacy `LoRALinear` bundle that
    // `AceStepLoRAModel::from_safetensors` already built, so the default
    // path is byte-equivalent to pre-Phase-2b training. Anything else swaps
    // the bundle in-place via `install_lycoris_bundle`.
    //
    // NOTE: `LycorisAlgo::parse("lora")` aliases to `LycorisAlgo::LoCon`
    // (LoCon-Linear is the canonical LoRA decomposition). For ACE-Step we
    // need to distinguish LEGACY plain `LoRALinear` (byte-identical) from
    // the new `LycorisAdapter::LoCon` path, so re-map `"lora"` → `None`
    // explicitly. Users wanting the new LoCon adapter pass `--algo locon`.
    let algo_str = args.algo.trim().to_ascii_lowercase();
    let algo = if algo_str == "lora" || algo_str == "none" || algo_str.is_empty() {
        LycorisAlgo::None
    } else {
        LycorisAlgo::parse(&args.algo).map_err(|e| anyhow::anyhow!("--algo: {e}"))?
    };
    let lyc_config = LycorisBundleConfig {
        algo,
        rank: args.rank,
        alpha: args.lora_alpha,
        factor: args.lokr_factor,
        block_size: args.oft_block_size,
        neumann_terms: args.oft_neumann_terms,
        use_tucker: args.use_tucker,
        decompose_both: args.decompose_both,
        use_scalar: false,
        dora: args.dora,
        dora_wd_on_out: args.dora_wd_on_out,
        dora_eps: args.dora_eps,
        init_type: LoraInitType::parse(&args.lora_init_type)
            .map_err(|e| anyhow::anyhow!("--lora_init_type: {e}"))?,
        ..LycorisBundleConfig::default()
    };
    let lyc_config = lyc_config
        .with_optional_lycoris_config_file(args.lycoris_config.as_deref())?;

    if algo != LycorisAlgo::None {
        log::info!(
            "[ACE-Step] LyCORIS algo='{}' rank={} alpha={} factor={} block_size={} dora={}",
            algo.as_str(),
            lyc_config.rank,
            lyc_config.alpha,
            lyc_config.factor,
            lyc_config.block_size,
            lyc_config.dora,
        );
        if matches!(algo, LycorisAlgo::Full | LycorisAlgo::Oft) {
            log::warn!(
                "[ACE-Step] algo='{}' selected — bundle construction will succeed, but \
                 forward_delta will error inside ACE-Step's `base + delta_on_input` \
                 attention call pattern. Phase 2c will wire merge-into-base.",
                algo.as_str()
            );
        }
        model.install_lycoris_bundle(&lyc_config, device.clone(), SEED_DEFAULT)
            .map_err(|e| anyhow::anyhow!("LyCORIS bundle install: {e}"))?;
    } else {
        log::info!("[ACE-Step] algo='lora' (legacy LoRALinear path, byte-identical)");
    }

    // Phase 2c — perturbed-normal LoKr init.
    if matches!(algo, LycorisAlgo::LoKr) && args.init_lokr_norm > 0.0 {
        let skipped = model
            .apply_init_perturbed_normal(args.init_lokr_norm)
            .map_err(|e| anyhow::anyhow!("init_lokr_norm: {e}"))?;
        if skipped > 0 {
            log::warn!(
                "[ACE-Step] init_lokr_norm: {} slot(s) skipped (see warnings above)",
                skipped
            );
        }
    }
    // Reference-only: silence unused-warning for dropout flags wired in
    // Block 2 but not yet plumbed past `LycorisBundleConfig` (Phase 2c).
    let _ = (args.lora_dropout, args.rank_dropout,
             args.module_dropout, args.rank_dropout_scale);

    let params = model.parameters();
    log::info!("Loaded {} trainable LoRA tensors ({} layers, hidden={})",
        params.len(), model.config().num_layers, model.config().hidden_size);
    if params.is_empty() {
        anyhow::bail!("No trainable parameters — model produced empty param list");
    }

    // Cache file enumeration.
    std::fs::create_dir_all(&args.output_dir)?;
    let mut cache_files: Vec<PathBuf> = std::fs::read_dir(&args.cache_dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension().and_then(|s| s.to_str())
                .map(|e| e == "safetensors" || e == "pt")
                .unwrap_or(false)
        })
        .collect();
    cache_files.sort();
    if cache_files.is_empty() {
        anyhow::bail!("No cached samples in {}", args.cache_dir.display());
    }
    log::info!("Found {} cached samples", cache_files.len());

    // Phase 2: validation harness — held-out cache + cadence. None at default.
    let validation_loop: Option<ValidationLoop> = if let (Some(dir), n) =
        (args.validation_dataset_dir.as_ref(), args.validation_every_steps)
    {
        if n > 0 {
            let v = ValidationLoop::new(dir, n)
                .map_err(|e| anyhow::anyhow!("validation harness: {e}"))?;
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

    let opt_kind = OptimizerKind::parse(&args.optimizer)
        .map_err(|e| anyhow::anyhow!("--optimizer: {e}"))?;
    log::info!("[ACE-Step] optimizer={}", opt_kind.as_str());
    // Phase 1: caption_dropout. ACE-Step has no inline encoder, so the user
    // supplies a `--null-text-cache` produced upstream on a single
    // empty-caption sample. We load `encoder_hidden_states` once and swap it
    // in per-step with the configured probability. Without `--null-text-cache`,
    // the feature is disabled with a warning. Note: this is independent of
    // the existing `--cfg-ratio` path which uses the model's internal
    // `null_condition_emb()` — both can be active. The cached
    // `encoder_attention_mask` is not consumed by the training step (assigned
    // to `_encoder_mask`), so we do not swap it.
    let mut effective_caption_dropout_prob = args.caption_dropout_probability;
    let null_text: Option<Tensor> = if effective_caption_dropout_prob > 0.0 {
        match args.null_text_cache.as_ref() {
            Some(p) => match flame_core::serialization::load_file(p, &device) {
                Ok(s) => {
                    let nt_raw = s.get("encoder_hidden_states")
                        .ok_or_else(|| anyhow::anyhow!("--null-text-cache missing 'encoder_hidden_states'"))?
                        .to_dtype(DType::BF16)?;
                    // Match the per-step `pull("encoder_hidden_states")?.unsqueeze(0)`
                    // pattern: cached null is [T, C], we add the batch dim to
                    // produce [1, T, C] so the swap is shape-aligned.
                    let nt = if nt_raw.shape().dims().len() == 2 {
                        nt_raw.unsqueeze(0)?
                    } else {
                        nt_raw
                    };
                    log::info!(
                        "[caption-dropout] WIRED — prob={:.3} (null_encoder_hidden_states={:?})",
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
    let mut optimizer = Optimizer::new(opt_kind, args.lr, 0.9, 0.999, 1e-8, 0.01);
    let mut start_step = 0usize;

    // EMA shadow (Phase 3 advanced). Built from current live params before
    // resume_* mutates them — mirrors Klein's ordering for parity. Updated
    // after each opt.step via `update_with_schedule`. Optional swap into live
    // params at sample / checkpoint time when --ema-validation-swap is set.
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
            "[ema] WIRED — {} shadow tensors, inv_gamma={} power={} update_after_step={} min_decay={} max_decay={} validation_swap={}",
            e.len(),
            ema_cfg.inv_gamma,
            ema_cfg.power,
            ema_cfg.update_after_step,
            ema_cfg.min_decay,
            ema_cfg.max_decay,
            args.ema_validation_swap,
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

    // Unified OneTrainer timestep distribution dispatch (optional override).
    // `auto` keeps the legacy `(timestep_mu, timestep_sigma)` logit-normal
    // path for byte-equivalence with pre-flag runs.
    let unified_timestep_cfg: Option<TimestepConfig> = if args.timestep_distribution.eq_ignore_ascii_case("auto") {
        None
    } else {
        let dist = TimestepDistribution::from_str(&args.timestep_distribution)
            .map_err(|e| anyhow::anyhow!("--timestep-distribution: {e}"))?;
        Some(TimestepConfig {
            distribution: dist,
            noising_weight: args.noising_weight,
            noising_bias: args.noising_bias,
            min_strength: 0.0,
            max_strength: 1.0,
        })
    };

    if args.multires_noise_iterations > 0 {
        log::warn!(
            "[multires-noise] ACE-Step uses non-4D latent shape; multires noise helper short-circuits to no-op for non-4D inputs. Pass 0 to silence."
        );
    }

    // Resume.
    if let Some(path) = &args.resume_lora {
        log::info!("Resuming LoRA weights from {}", path.display());
        let loaded = checkpoint::load_full(path, &device)?;
        let named = model.named_parameters();
        checkpoint::apply_lora_weights(&loaded, &named)?;
    } else if let Some(path) = &args.resume_full {
        log::info!("Full-resume from {}", path.display());
        let loaded = checkpoint::load_full(path, &device)?;
        let named = model.named_parameters();
        checkpoint::apply_lora_weights(&loaded, &named)?;
        if let Optimizer::AdamW(ref mut adam) = optimizer {
            checkpoint::apply_to_optimizer(&loaded, adam, &named, args.rank, args.lora_alpha)?;
        } else {
            log::warn!(
                "[resume-full] non-AdamW resume not yet implemented for {:?}; LoRA weights restored, optimizer state reset",
                optimizer.kind()
            );
        }
        start_step = loaded.header.step as usize;
        if start_step >= args.steps {
            log::warn!("Resumed step ({start_step}) >= --steps ({}); nothing to do.", args.steps);
            return Ok(());
        }
        log::info!("Continuing from step {start_step}/{}", args.steps);
    }

    let null_emb = model.null_condition_emb().clone();
    let clipper = GradientClipper::clip_by_norm(1.0);

    let mut rng = StdRng::seed_from_u64(args.seed);
    let mut loss_sum = 0.0f32;
    let mut loss_count = 0usize;
    let board = BoardWriter::open(
        &args.output_dir,
        BoardWriter::new_session_id(),
        if start_step > 0 { Some(start_step as u64) } else { None },
    ).map_err(|e| log::warn!("board.db open failed: {e}")).ok();
    if let Some(b) = &board {
        log::info!("SerenityBoard: writing scalars to {}", b.db_path.display());
    }
    let t_start = std::time::Instant::now();

    log::info!("Training {} steps from step={}, lr={} warmup={} cfg_ratio={}",
        args.steps, start_step, args.lr, args.warmup_steps, args.cfg_ratio);

    let sched: LrScheduler = args.lr_scheduler.parse().unwrap_or_else(|e: String| {
        log::warn!("[lr_scheduler] {e} — falling back to Constant");
        LrScheduler::Constant
    });
    for step in start_step..args.steps {
        // Phase 5: dispatch via LrScheduler enum. Default `Constant` is
        // byte-equivalent to legacy `constant_with_warmup`.
        let current_lr = lr_schedule::dispatch_lr(
            &sched,
            args.lr,
            step,
            args.steps,
            args.warmup_steps,
            args.lr_min_factor,
            args.lr_cycles,
        );
        optimizer.set_lr(current_lr);

        // Sample one cache file.
        let sample_idx = rng.gen_range(0..cache_files.len());
        let tensors = flame_core::serialization::load_file(&cache_files[sample_idx], &device)?;
        let pull = |k: &str| -> flame_core::Result<Tensor> {
            tensors.get(k)
                .ok_or_else(|| flame_core::Error::InvalidInput(
                    format!("Missing '{k}' in {}", cache_files[sample_idx].display())))
                .map(|t| t.clone().to_dtype(DType::BF16).unwrap_or_else(|_| t.clone()))
        };
        let target_latents = pull("target_latents")?.unsqueeze(0)?;
        let _attention_mask = pull("attention_mask")?.unsqueeze(0)?;
        let encoder_hs = pull("encoder_hidden_states")?.unsqueeze(0)?;
        let _encoder_mask = pull("encoder_attention_mask")?.unsqueeze(0)?;
        let context_latents = pull("context_latents")?.unsqueeze(0)?;

        // Caption dropout (independent of --cfg-ratio): single Bernoulli per
        // step swaps encoder_hidden_states with null cache. Default-off
        // (prob == 0.0 OR null_text == None) draws no rng.
        let encoder_hs = if let Some(ref nt) = null_text {
            caption_dropout::maybe_drop_caption(&encoder_hs, nt, effective_caption_dropout_prob, &mut rng)?
        } else {
            encoder_hs
        };

        // CFG dropout: replace encoder_hs with null_emb at probability cfg_ratio.
        let encoder_hs = apply_cfg_dropout(&encoder_hs, &null_emb, args.cfg_ratio, &mut rng)?;

        // Flow-matching: x1 = noise, x0 = clean target, t = sigmoid(z * sigma + mu).
        let x1 = Tensor::randn(target_latents.shape().clone(), 0.0, 1.0, device.clone())?
            .to_dtype(DType::BF16)?;
        // Pyramid / multi-resolution noise (additive). NOTE: ACE-Step trains
        // on non-4D audio latents `[B, C, T]`; the helper short-circuits to
        // a no-op for non-4D inputs, so this is a documented no-op here. We
        // call it anyway to keep the CLI surface uniform across trainers.
        let x1 = noise_modifiers::maybe_apply_multires_noise(
            &x1,
            args.multires_noise_iterations,
            args.multires_noise_discount,
            &mut rng,
        )?;
        // Phase 1: noise modifiers (default-off). ACE-Step trainer doesn't
        // load TrainConfig JSON — `offset_noise_weight` defaults to 0.0.
        // Offset noise is part of the clean noise distribution; input
        // perturbation feeds model input only (target keeps unperturbed noise).
        let x1_clean = noise_modifiers::maybe_apply_offset_noise(
            &x1,
            0.0,
            args.noise_offset_probability,
            &mut rng,
        )?;
        let x1_perturbed = noise_modifiers::maybe_apply_input_perturbation(
            &x1_clean,
            args.gamma_input_perturbation,
            &mut rng,
        )?;
        let x0 = target_latents;
        let t_val = {
            let raw_t = if let Some(ref tcfg) = unified_timestep_cfg {
                tcfg.sample_one(&mut rng)
            } else {
                schedule::sample_timestep_logit_normal(&mut rng, args.timestep_mu, args.timestep_sigma)
            };
            // schedule helper already returns sigmoid(z*sigma+mu)-equivalent in (0,1).
            // Default-off: Strategy::None → returns raw_t unchanged. Use total=1.0
            // because ACE-Step's t lives in (0, 1) directly (no NUM_TRAIN_TIMESTEPS
            // scaling at the trainer surface — the model multiplies by 1000 internally).
            timestep_bias::apply_bias(raw_t, 1.0, &timestep_bias_cfg)
        };
        let t_tensor = Tensor::from_vec(vec![t_val], Shape::from_dims(&[1]), device.clone())?
            .to_dtype(DType::BF16)?;

        // x_t = t * x1 + (1 - t) * x0  (use perturbed for model input)
        let xt = x1_perturbed.mul_scalar(t_val)?.add(&x0.mul_scalar(1.0 - t_val)?)?;

        // Forward + flow-matching loss.
        AutogradContext::clear();
        let pred = model.forward(
            &xt,
            &t_tensor,
            &t_tensor, // r = t for ACE-Step training
            &encoder_hs,
            &context_latents,
        )?;
        // Target uses clean noise so perturbation contamination is excluded.
        let flow = x1_clean.sub(&x0)?;
        // Phase 1: combined loss + per-step weighting. Default-off invariant.
        // ACE-Step trainer doesn't load TrainConfig; mse=1.0 mae=0.0 inline.
        let pred_f32 = pred.to_dtype(DType::F32)?;
        let flow_f32 = flow.to_dtype(DType::F32)?;
        let raw_loss = loss_weight::combined_loss(
            &pred_f32,
            &flow_f32,
            1.0,
            0.0,
            args.huber_strength,
        )?;
        // ACE-Step `t_val` is the flow-matching sigma analog.
        let loss = loss_weight::apply_loss_weight(
            &raw_loss,
            t_val,
            eridiffusion_core::config::LossWeight::Constant,
            args.min_snr_gamma,
            true,
        )?;

        let loss_f32 = loss.to_dtype(DType::F32)?;
        let loss_val = {
            let v: Vec<f32> = loss_f32.to_vec()?;
            v.first().copied().unwrap_or(f32::NAN)
        };
        if !loss_val.is_finite() {
            anyhow::bail!("step {}: non-finite loss {}", step + 1, loss_val);
        }
        loss_sum += loss_val;
        loss_count += 1;

        // Backward.
        let grads = loss_f32.backward()?;

        // Grad-flow diagnostic.  Runs at step 1 — NOT step 0 — because every
        // LoRA-style algo (LoRA, LoCon, LoHa, LoKr) initializes one factor
        // at zero so `delta = factor_a @ factor_b = 0` at step 0.  Backward
        // through `delta * weight` then forces half the leaves to zero
        // gradient by mathematical construction.  Step 1 (after the first
        // optimizer step has driven the zero leaves off zero) is when the
        // assertion can distinguish "real bug" from "expected zero-init
        // pattern".  See flame-core/docs/TRAINER_DIAGNOSTICS.md.
        if step == 1 {
            let named = model.named_parameters();
            let named_refs: Vec<(&str, &flame_core::parameter::Parameter)> = named
                .iter()
                .map(|(n, p)| (n.as_str(), p))
                .collect();
            let report = flame_core::diagnostics::assert_grad_flow(&grads, &named_refs)?;
            if report.is_clean() {
                log::info!("[grad-flow] step 2 clean ({} params)", report.ok_count);
            } else {
                log::warn!("{}", report.summary());
            }
        }

        for param in &params {
            if let Some(g) = grads.get(param.id()) {
                let g = if g.dtype() == DType::F32 { g.clone() } else { g.to_dtype(DType::F32)? };
                param.set_grad(g)?;
            }
        }

        // Clip gradients.
        let grad_norm = {
            let mut grad_tensors: Vec<Tensor> = Vec::new();
            let mut owners: Vec<usize> = Vec::new();
            for (idx, param) in params.iter().enumerate() {
                if let Some(g) = param.grad() {
                    grad_tensors.push(g);
                    owners.push(idx);
                }
            }
            let mut grad_refs: Vec<&mut Tensor> = grad_tensors.iter_mut().collect();
            let norm = clipper.clip_grads(&mut grad_refs)?;
            for (owner, grad) in owners.into_iter().zip(grad_tensors.into_iter()) {
                params[owner].set_grad(grad)?;
            }
            norm
        };

        // Optimizer step.
        {
            let _guard = AutogradContext::no_grad();
            optimizer.step(&params)?;
            optimizer.zero_grad(&params);
            if let Some(ref mut e) = ema {
                // 1-based step → matches the schedule's `update_after_step`
                // semantics (step==update_after_step returns 0 / "skip").
                e.update_with_schedule(&params, &ema_cfg, (step + 1) as u64)
                    .map_err(|err| anyhow::anyhow!("EMA update failed at step {}: {err}", step + 1))?;
            }
        }
        AutogradContext::clear();

        let _ = loss_sum; let _ = loss_count;
        eridiffusion_core::training::progress::log_step(
            "AceStep-lora",
            step, args.steps, cache_files.len(), 1,
            loss_val, grad_norm, current_lr, t_start, board.as_ref(),
        );

        // Phase 2: validation eval pass (no_grad) every `validation_every_steps`.
        // step+1 because `step` here is 0-based; ValidationLoop::should_run
        // expects the 1-based completed-step number. Validation uses its OWN
        // run-side RNG (seed ^ (step+1)) so it does not perturb the
        // training-side seeded sequence — byte invariance with --validation-
        // every-steps=0 (default) is guaranteed because the harness Option is
        // None and this entire block is skipped.
        if let Some(ref vloop) = validation_loop {
            if vloop.should_run((step + 1) as usize) {
                let mut sum = 0.0_f32;
                let mut count = 0_usize;
                for vfile in &vloop.cache_files {
                    let _g = AutogradContext::no_grad();
                    let vtensors = match flame_core::serialization::load_file(vfile, &device) {
                        Ok(s) => s,
                        Err(e) => {
                            log::warn!("[validation] load {} failed: {e}", vfile.display());
                            continue;
                        }
                    };
                    let vpull = |k: &str| -> Option<Tensor> {
                        vtensors.get(k).map(|t| {
                            t.clone().to_dtype(DType::BF16).unwrap_or_else(|_| t.clone())
                        })
                    };
                    let v_target = match vpull("target_latents") {
                        Some(t) => match t.unsqueeze(0) { Ok(x) => x, Err(e) => {
                            log::warn!("[validation] {} target_latents unsqueeze: {e}", vfile.display());
                            continue;
                        }},
                        None => {
                            log::warn!("[validation] {} missing target_latents", vfile.display());
                            continue;
                        }
                    };
                    let v_encoder_hs = match vpull("encoder_hidden_states") {
                        Some(t) => match t.unsqueeze(0) { Ok(x) => x, Err(e) => {
                            log::warn!("[validation] {} encoder_hidden_states unsqueeze: {e}", vfile.display());
                            continue;
                        }},
                        None => {
                            log::warn!("[validation] {} missing encoder_hidden_states", vfile.display());
                            continue;
                        }
                    };
                    let v_context = match vpull("context_latents") {
                        Some(t) => match t.unsqueeze(0) { Ok(x) => x, Err(e) => {
                            log::warn!("[validation] {} context_latents unsqueeze: {e}", vfile.display());
                            continue;
                        }},
                        None => {
                            log::warn!("[validation] {} missing context_latents", vfile.display());
                            continue;
                        }
                    };
                    // Sample timestep + noise identically to training. Validation
                    // uses its OWN run-side RNG so it does not perturb the
                    // training-side seeded sequence (byte invariance).
                    let mut vrng = StdRng::seed_from_u64(args.seed ^ (step as u64 + 1));
                    let v_t_val = if let Some(ref tcfg) = unified_timestep_cfg {
                        tcfg.sample_one(&mut vrng)
                    } else {
                        schedule::sample_timestep_logit_normal(
                            &mut vrng, args.timestep_mu, args.timestep_sigma,
                        )
                    };
                    // Mirror training: validation uses CLEAN noise (no offset/
                    // perturbation/multires — multires is a no-op for non-4D
                    // anyway, and offset+perturbation are training-only random
                    // augmentations that should not influence the eval signal).
                    let v_x1 = match Tensor::randn(
                        v_target.shape().clone(),
                        0.0,
                        1.0,
                        device.clone(),
                    ) {
                        Ok(t) => match t.to_dtype(DType::BF16) {
                            Ok(x) => x,
                            Err(e) => {
                                log::warn!("[validation] noise dtype: {e}");
                                continue;
                            }
                        },
                        Err(e) => {
                            log::warn!("[validation] noise: {e}");
                            continue;
                        }
                    };
                    // x_t = t * x1 + (1 - t) * x0
                    let v_xt = match v_x1.mul_scalar(v_t_val)
                        .and_then(|x| v_target.mul_scalar(1.0 - v_t_val).and_then(|y| x.add(&y)))
                    {
                        Ok(x) => x,
                        Err(e) => {
                            log::warn!("[validation] xt mix: {e}");
                            continue;
                        }
                    };
                    let v_t_tensor = match Tensor::from_vec(
                        vec![v_t_val],
                        Shape::from_dims(&[1]),
                        device.clone(),
                    ).and_then(|t| t.to_dtype(DType::BF16)) {
                        Ok(t) => t,
                        Err(e) => {
                            log::warn!("[validation] t_tensor: {e}");
                            continue;
                        }
                    };
                    let v_pred = match model.forward(
                        &v_xt,
                        &v_t_tensor,
                        &v_t_tensor,
                        &v_encoder_hs,
                        &v_context,
                    ) {
                        Ok(p) => p,
                        Err(e) => {
                            log::warn!("[validation] forward failed: {e}");
                            continue;
                        }
                    };
                    let v_flow = match v_x1.sub(&v_target) {
                        Ok(t) => t,
                        Err(e) => {
                            log::warn!("[validation] flow target: {e}");
                            continue;
                        }
                    };
                    let v_loss = match v_pred.to_dtype(DType::F32)
                        .and_then(|p| v_flow.to_dtype(DType::F32).and_then(|f| p.sub(&f)))
                        .and_then(|d| d.square())
                        .and_then(|s| s.mean())
                    {
                        Ok(l) => l,
                        Err(e) => {
                            log::warn!("[validation] loss compute: {e}");
                            continue;
                        }
                    };
                    let v_loss_val = match v_loss.to_vec() {
                        Ok(v) => v.first().copied().unwrap_or(f32::NAN),
                        Err(e) => {
                            log::warn!("[validation] loss to_vec: {e}");
                            continue;
                        }
                    };
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

        // Periodic save.
        if args.save_every > 0 && (step + 1) % args.save_every == 0 && (step + 1) < args.steps {
            // EMA swap: when `--ema --ema-validation-swap`, the saved checkpoint
            // sees EMA-averaged weights. Restored at the end of this block so
            // optimizer moments stay consistent with the live tensors they
            // were taken against.
            let ema_backup = if args.ema_validation_swap {
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
            let path = args.output_dir.join(format!("acestep_lora_step{}.safetensors", step + 1));
            save_ckpt(&path, &model, &optimizer, args.rank, args.lora_alpha, args.seed, &args.save_mode, step + 1)?;
            if let (Some(backup), Some(ref e)) = (ema_backup, &ema) {
                let _g = AutogradContext::no_grad();
                e.restore_swapped(&params, backup)
                    .map_err(|err| anyhow::anyhow!("EMA restore_swapped (mid) failed: {err}"))?;
            }
        }
    }

    // Final EMA swap (covers the final save). No restore — process exits, no
    // further training. Skipped when --ema-validation-swap is off or no EMA
    // was constructed.
    if args.ema_validation_swap {
        if let Some(ref e) = ema {
            let _g = AutogradContext::no_grad();
            let _ = e.swap_with_live(&params)
                .map_err(|err| anyhow::anyhow!("EMA swap_with_live (final) failed: {err}"))?;
            log::info!("[ema] swapped EMA shadow into live params for final save");
        }
    }

    let final_path = args.output_dir.join(format!("acestep_lora_{}steps.safetensors", args.steps));
    save_ckpt(&final_path, &model, &optimizer, args.rank, args.lora_alpha, args.seed, &args.save_mode, args.steps)?;
    log::info!(
        "Training complete: {} steps (avg loss={:.4}). Saved to {}",
        args.steps,
        loss_sum / loss_count.max(1) as f32,
        final_path.display(),
    );
    if let Some(b) = &board { b.set_status("completed"); }
    Ok(())
}

/// Save in `full` mode (LoRA + AdamW state + step) or `weights` mode (legacy
/// safetensors with `save_lora` key scheme only). The `full` path uses
/// `model.named_parameters()` so resume can restore m/v by canonical name.
fn save_ckpt(
    path: &std::path::Path,
    model: &AceStepLoRAModel,
    optimizer: &Optimizer,
    rank: usize,
    alpha: f32,
    seed: u64,
    mode: &str,
    step: usize,
) -> anyhow::Result<()> {
    if mode == "weights" {
        model.save_lora(path)?;
        log::info!("[save] {} (weights only)", path.display());
        return Ok(());
    }
    let adam = match optimizer {
        Optimizer::AdamW(a) => a,
        _ => {
            log::warn!(
                "[save] full-state save not yet implemented for {:?}; saving weights only",
                optimizer.kind()
            );
            model.save_lora(path)?;
            return Ok(());
        }
    };
    let header = CkptHeader::from_adamw(
        "train_acestep",
        step as u64,
        adam,
        rank,
        alpha,
        seed,
        String::new(),
    );
    let named = model.named_parameters();
    checkpoint::save_full(path, &named, adam, &header)
        .map_err(|e| anyhow::anyhow!("save_full: {e}"))?;
    Ok(())
}
