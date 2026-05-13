//! train_anima — Anima LoRA training binary, mirroring train_ernie structure.
//!
//! Reference: kohya `anima_train_network.AnimaNetworkTrainer.get_noise_pred_and_target`
//! (`anima_train_network.py:254`) + `flux_train_utils.get_noisy_model_input_and_timesteps`.
//!
//! Pipeline per step:
//!   1. Load cached `latent` ([B,16,h,w]) + `text_embedding` ([B,seq,1024]) +
//!      `text_mask` + `t5_input_ids` + `t5_attn_mask` from prepare_anima.
//!   2. Sample timestep via SIGMOID (logit-normal with sigmoid_scale=1.0) by
//!      default, optionally with `--discrete-flow-shift` reweight.
//!      sigma = timestep / 1000.
//!   3. noisy = sigma * noise + (1 - sigma) * latent (rectified flow).
//!   4. target = noise - latent (rect-flow target).
//!   5. Forward → [B,16,h,w]; loss = MSE(F32) with optional weighting.
//!
//! Hard constraints (per CLAUDE.md / MEMORY.md):
//!   - BF16 throughout, NO quantization (no fp8 / AdamW8bit / int8 LoRA).
//!     We use plain F32 AdamW, ignoring kohya's fp8/8bit defaults.
//!   - Default seed = 42; --resume-full + --save-mode wired.
//!
//! ## STATUS
//! Forward is wired against the ported `AnimaModel` (see
//! `crates/eridiffusion-core/src/models/anima.rs:703-767`, ported from
//! `inference-flame::anima`). End-to-end loss-curve parity vs kohya
//! `anima_train_network.py` not yet validated.

use clap::Parser;
use std::path::PathBuf;
use flame_core::{autograd::AutogradContext, DType, Shape, Tensor};
use eridiffusion_core::config::{LrScheduler, TrainConfig, TrainingMethod};
use eridiffusion_core::debug as dbg;
use eridiffusion_core::lycoris::{LoraInitType, LycorisAlgo, LycorisBundleConfig};
use eridiffusion_core::models::{anima as anima_mod, anima::AnimaLoraBundle, AnimaModel, TrainableModel};
use eridiffusion_core::training::board::BoardWriter;
use eridiffusion_core::training::checkpoint::{self, CkptHeader};
use eridiffusion_core::training::ema::ParameterEma;
use eridiffusion_core::training::features::ema_advanced::EmaConfig;
use eridiffusion_core::training::features::{
    loss_weight as feat_loss_weight, lr_schedule, noise_modifiers, timestep_bias,
};
use eridiffusion_core::training::training_features::{Optimizer, OptimizerKind};
use eridiffusion_core::training::training_features::timestep_dist::{TimestepConfig, TimestepDistribution};
use std::str::FromStr as _;

/// Slot class names for the 16 LoRA modules per Anima block. Used by debug
/// gradient summaries. MUST match `anima::LORA_SLOT_KEYS` order.
const ANIMA_LORA_CLASSES: [&str; anima_mod::LORA_SLOTS_PER_BLOCK] = [
    "self_attn.q", "self_attn.k", "self_attn.v", "self_attn.out",
    "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.out",
    "mlp.layer1", "mlp.layer2",
    "adaln_sa.1", "adaln_sa.2",
    "adaln_ca.1", "adaln_ca.2",
    "adaln_mlp.1", "adaln_mlp.2",
];

const NUM_TRAIN_TIMESTEPS: usize = 1000;
const SEED: u64 = 42;

#[derive(Parser)]
struct Args {
    #[arg(long)] config: PathBuf,
    #[arg(long)] cache_dir: PathBuf,
    /// Single safetensors file (e.g. `anima-preview.safetensors` or
    /// `anima-preview3-base.safetensors`).
    #[arg(long)] dit_path: PathBuf,
    #[arg(long, default_value = "100")] steps: usize,
    #[arg(long, default_value = "16")] rank: usize,
    #[arg(long, default_value = "1.0")] lora_alpha: f64,
    /// Learning rate. Default `5e-5` matches `Anima_lora_configs.toml:23`
    /// (canonical recipe). The previous default of 3e-4 was 6× higher and
    /// in the "blow up early, stabilize ugly" regime for 28-block DiTs.
    #[arg(long, default_value = "5e-5")] lr: f32,

    /// Timestep sampling: "sigmoid" (kohya default), "uniform", "shift".
    #[arg(long, default_value = "sigmoid")] timestep_sampling: String,
    /// Sigmoid scale used by `sigmoid` sampling (kohya default 1.0).
    #[arg(long, default_value = "1.0")] sigmoid_scale: f32,
    /// Rectified-flow timestep shift (kohya `--discrete_flow_shift`).
    /// Default 3.0 matches `Anima_lora_configs.toml:37` (canonical Anima
    /// recipe; reference Python's CLI default is 1.0 but the shipped TOML
    /// always overrides it). Mid-sigma loss mass is the principal hyper-
    /// parameter for Anima rect-flow training; shift=1.0 trains a
    /// fundamentally different distribution from reference checkpoints.
    #[arg(long, default_value = "3.0")] discrete_flow_shift: f32,
    /// Loss weighting scheme: "none" (default), "sigma_sqrt", "cosmap".
    #[arg(long, default_value = "none")] weighting_scheme: String,

    /// Resume LoRA weights only (no optimizer state).
    #[arg(long, conflicts_with = "resume_full")] resume_lora: Option<PathBuf>,
    /// Full resume: LoRA weights + AdamW (m, v, t) + step counter.
    #[arg(long, conflicts_with = "resume_lora")] resume_full: Option<PathBuf>,
    /// Periodic + final save mode. Default `full` (LoRA + AdamW + step) for
    /// resumable runs. `weights` writes legacy weights-only files.
    #[arg(long, default_value = "full")] save_mode: String,
    #[arg(long, default_value = "output")] output_dir: PathBuf,

    // ── Periodic save (in-trainer sample rendering not yet wired; use sample_anima.rs) ──
    #[arg(long, default_value = "0")] save_every: usize,

    // ── Phase 0 multi-feature rollout (default-off; Phase 1+ will consume) ──
    #[arg(long)] min_snr_gamma: Option<f32>,
    #[arg(long, default_value_t = 0.0)] caption_dropout_probability: f32,
    /// Path to a single cache file produced by `prepare_anima` from an empty-
    /// caption sample. When `--caption-dropout-probability > 0`, the trainer
    /// loads `text_embedding` from this file and swaps it in (along with the
    /// optional `text_mask`) with probability `p` per step.
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
    /// Master switch for EMA shadow. See train_klein.rs for full doc.
    #[arg(long, default_value_t = false)] ema: bool,
    #[arg(long, default_value_t = 1.0)] ema_inv_gamma: f32,
    #[arg(long, default_value_t = 0.6667)] ema_power: f32,
    #[arg(long, default_value_t = 0)] ema_update_after_step: u64,
    #[arg(long, default_value_t = 0.0)] ema_min_decay: f32,
    #[arg(long, default_value_t = 0.9999)] ema_max_decay: f32,
    /// When true (with --ema), periodic save + final save observe EMA-averaged weights.
    #[arg(long, default_value_t = false)] ema_validation_swap: bool,
    /// Multi-resolution / pyramid noise iterations. 0 (default) = no-op,
    /// byte-identical to no-multires.
    #[arg(long, default_value_t = 0)] multires_noise_iterations: usize,
    /// Per-level discount factor for `--multires-noise-iterations`.
    #[arg(long, default_value_t = 0.3)] multires_noise_discount: f32,
    /// Timestep biasing strategy: "none" (default), "later", "earlier", "range".
    #[arg(long, default_value = "none")] timestep_bias_strategy: String,
    #[arg(long, default_value_t = 0.0)] timestep_bias_multiplier: f32,
    #[arg(long, default_value_t = 0.0)] timestep_bias_range_min: f32,
    #[arg(long, default_value_t = 1.0)] timestep_bias_range_max: f32,

    /// Unified OneTrainer timestep distribution. When set to a value other
    /// than `auto` (default) this overrides `--timestep-sampling`, allowing
    /// any of the 6 OT distributions (uniform / sigmoid / logit_normal /
    /// heavy_tail / cos_map / inverted_parabola). `auto` preserves the
    /// existing `--timestep-sampling` byte-equivalent path (sigmoid default).
    #[arg(long, default_value = "auto")] timestep_distribution: String,
    /// Distribution-specific weight knob (default 0.0).
    #[arg(long, default_value_t = 0.0)] noising_weight: f32,
    /// Distribution-specific bias knob (default 0.0).
    #[arg(long, default_value_t = 0.0)] noising_bias: f32,

    #[arg(long)] tread_route_pattern: Option<String>,
    /// Phase 1: optimizer family CLI surface (Phase 5 wires full dispatch).
    #[arg(long, default_value = "adamw")] optimizer: String,

    // ── Phase 6 multi-feature rollout (plumb-only; multi-backend wired in Klein) ──
    #[arg(long, num_args = 0..)] multi_backend_repeats: Vec<u32>,
    #[arg(long, default_value_t = false)] caption_tag_shuffle: bool,
    #[arg(long, default_value_t = false)] cache_clear_each_epoch: bool,
    #[arg(long, default_value_t = false)] cache_invalidate: bool,
    /// Phase 5: LR scheduler family. Default `cosine` matches
    /// `Anima_lora_configs.toml:lr_scheduler = "cosine"` (canonical recipe).
    #[arg(long, default_value = "cosine")] lr_scheduler: String,
    /// Phase 5: linear LR warmup steps. Default 100 matches
    /// `Anima_lora_configs.toml:lr_warmup_steps = 100`.
    #[arg(long, default_value_t = 100)] warmup_steps: usize,
    /// Phase 5: cosine-with-restarts cycle count.
    #[arg(long, default_value_t = 1.0)] lr_cycles: f32,

    // ── LyCORIS algo selection (Phase 2b) ──
    //
    // `--algo lora` (default) keeps the legacy LoRALinear path — byte-identical
    // training to pre-Phase-2b. Other values select LyCORIS algos via
    // `AnimaLoraBundle::new_with_config`. `lora_alpha` and `rank` are shared
    // with the legacy CLI flags above (no separate `--lycoris-rank`).
    /// LyCORIS algo: `lora` (default, legacy path) | `locon` | `loha` | `lokr`
    /// | `full` | `oft`. Same `base + delta_on_input` caveat as chroma — Full
    /// and OFT bundle-construct successfully but error inside forward_delta;
    /// Phase 2c will hoist merge-into-base.
    #[arg(long, default_value = "lora")] algo: String,
    /// LoKr Kronecker split factor (ignored for non-LoKr).
    #[arg(long, default_value_t = 16)] lokr_factor: i32,
    /// OFT/BOFT block size (ignored for non-OFT/BOFT).
    #[arg(long, default_value_t = 32)] oft_block_size: usize,
    /// OFT Cayley-Neumann series term count.
    #[arg(long, default_value_t = 5)] oft_neumann_terms: usize,
    /// Tucker decomposition for non-1×1 conv kernels (anima is linear-only).
    #[arg(long, default_value_t = false)] use_tucker: bool,
    /// LoKr only: factorize both W1 and W2 (default false: only W2).
    #[arg(long, default_value_t = false)] decompose_both: bool,
    /// Enable DoRA (weight-decomposed LoRA).
    #[arg(long, default_value_t = false)] dora: bool,
    /// DoRA magnitude axis (`true` = lycoris-upstream).
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
    /// SimpleTuner-parity: perturbed-normal LoKr init. Scale `>0` triggers
    /// `lokr_w1=1, lokr_w2 ~ N(μ_W, σ_W)·scale`. No-op when algo != lokr or
    /// value is 0.0.
    #[arg(long, default_value_t = 0.0)] init_lokr_norm: f32,
    /// SimpleTuner / ai-toolkit `network.conv` — per-LyCORIS rank for
    /// CONV-layer targets (separate from linear `--rank`). `0` (default)
    /// = fall back to linear rank. Inert when no conv targets are wired
    /// in the model bundle (current state on all EDv2 trainers).
    #[arg(long, default_value_t = 0)] conv_rank: usize,
    /// SimpleTuner / ai-toolkit `network.conv_alpha` — alpha for CONV
    /// targets. `0.0` (default) = fall back to linear `--lora-alpha`.
    #[arg(long, default_value_t = 0.0)] conv_alpha: f32,
    /// Per-element dropout on the adapter delta (training only).
    #[arg(long, default_value_t = 0.0)] lora_dropout: f32,
    /// Per-rank Bernoulli on the down-projection intermediate.
    #[arg(long, default_value_t = 0.0)] rank_dropout: f32,
    /// Per-step Bernoulli on the entire adapter.
    #[arg(long, default_value_t = 0.0)] module_dropout: f32,
    /// Rescale rank-mask by `1/mean(mask)` to preserve expectation.
    #[arg(long, default_value_t = false)] rank_dropout_scale: bool,

    // ── Phase 5b: autograd v2 bridge opt-in ────────────────────────────────
    /// Route the backward pass through `AutogradContext::backward_v2`
    /// (`MatchInsertedDtype` policy → BF16 grads end-to-end). Default OFF
    /// preserves v3 byte-equivalence. See train_zimage.rs:269 for full doc.
    #[arg(long, default_value_t = false)] use_autograd_v2: bool,
}

/// SIGMOID timestep sampling: t = sigmoid(scale * z), z ~ N(0,1).
/// Returns continuous timestep in [0, NUM_TRAIN_TIMESTEPS).
fn sample_timestep_sigmoid(rng: &mut rand::rngs::StdRng, scale: f32) -> f32 {
    use rand_distr::{Distribution, Normal};
    let normal = Normal::new(0.0f32, 1.0f32).unwrap();
    let z = normal.sample(rng);
    let t = 1.0 / (1.0 + (-(scale * z)).exp());
    t * NUM_TRAIN_TIMESTEPS as f32
}

/// UNIFORM timestep sampling: t ~ U[0, NUM_TRAIN_TIMESTEPS).
fn sample_timestep_uniform(rng: &mut rand::rngs::StdRng) -> f32 {
    use rand::Rng;
    rng.gen::<f32>() * NUM_TRAIN_TIMESTEPS as f32
}

/// Apply rectified-flow shift to a sigma in [0, 1].
fn apply_shift(sigma: f32, shift: f32) -> f32 {
    if (shift - 1.0).abs() < 1e-6 {
        sigma
    } else {
        shift * sigma / (1.0 + (shift - 1.0) * sigma)
    }
}

/// Per-sample loss weighting (kohya `compute_loss_weighting_for_anima`).
fn loss_weight(scheme: &str, sigma: f32) -> f32 {
    match scheme {
        "sigma_sqrt" => 1.0 / (sigma * sigma).max(1e-6),
        "cosmap" => {
            let bot = 1.0 - 2.0 * sigma + 2.0 * sigma * sigma;
            2.0 / (std::f32::consts::PI * bot)
        }
        _ => 1.0,
    }
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

    // NB: `--config` is JSON-only. The kohya `Anima_lora_configs.toml` format
    // (with `[anima_arguments]` / `[training_arguments]` sections, e.g.
    // `discrete_flow_shift = 3.0`, `learning_rate = 5e-5`, `lr_scheduler =
    // "cosine"`, `lr_warmup_steps = 100`) is NOT consumed here — the EDv2
    // CLI defaults reflect those canonical TOML values directly, so the
    // launcher should pass them as flags or rely on defaults rather than
    // through the TOML. If a user passes `--config foo.toml`, the JSON
    // parser will fail with a clear error, which surfaces the mismatch.
    let mut config = TrainConfig::from_json_path(&args.config.to_string_lossy())?;
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
    if args.masked_loss_weight > 0.0 {
        log::warn!(
            "[masked-loss] --masked-loss-weight={:.3} requested but Anima's prepare_anima cache schema has no `latent_mask` field; flag is a no-op for this trainer.",
            args.masked_loss_weight
        );
    }
    config.ema_inv_gamma = args.ema_inv_gamma;
    config.ema_power = args.ema_power;
    config.ema_update_after_step = args.ema_update_after_step;
    config.ema_min_decay = args.ema_min_decay;
    config.tread_route_pattern = args.tread_route_pattern.clone();

    log::info!("Loading Anima DiT (rank={} alpha={}) from {}...",
        args.rank, args.lora_alpha, args.dit_path.display());
    let mut model = AnimaModel::load(&args.dit_path, &config, device.clone())?;

    // Phase 2b: parse the LyCORIS algo selector. `lora` (default) keeps the
    // legacy LoRALinear bundle constructed inside `AnimaModel::load`. Anything
    // else swaps the bundle in-place after model construction.
    //
    // NOTE: `LycorisAlgo::parse("lora")` aliases to `LycorisAlgo::LoCon`, but
    // for anima we need LEGACY plain `LoRALinear` (byte-identical) to remain
    // the default path; re-map `"lora"` → `None` explicitly. Users who want
    // the new LoCon path pass `--algo locon`.
    let algo_str = args.algo.trim().to_ascii_lowercase();
    let algo = if algo_str == "lora" || algo_str == "none" || algo_str.is_empty() {
        LycorisAlgo::None
    } else {
        LycorisAlgo::parse(&args.algo).map_err(|e| anyhow::anyhow!("--algo: {e}"))?
    };
    let lyc_config = LycorisBundleConfig {
        algo,
        rank: args.rank,
        alpha: args.lora_alpha as f32,
        factor: args.lokr_factor,
        conv_rank: args.conv_rank,
        conv_alpha: args.conv_alpha,
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
            "[Anima] LyCORIS algo='{}' rank={} alpha={} factor={} block_size={} dora={}",
            algo.as_str(),
            lyc_config.rank, lyc_config.alpha,
            lyc_config.factor, lyc_config.block_size, lyc_config.dora,
        );
        if matches!(algo, LycorisAlgo::Full | LycorisAlgo::Oft) {
            log::warn!(
                "[Anima] algo='{}' selected — bundle construction will succeed, but \
                 forward_delta will error inside anima's `base + delta_on_input` call \
                 pattern. Phase 2c will wire merge-into-base for these algos.",
                algo.as_str()
            );
        }
        let new_bundle = AnimaLoraBundle::new_with_config(&lyc_config, device.clone(), SEED)
            .map_err(|e| anyhow::anyhow!("LyCORIS bundle construction: {e}"))?;
        model.bundle = new_bundle;

        // Optional: SimpleTuner-style perturbed-normal init for LoKr.
        if matches!(algo, LycorisAlgo::LoKr) && args.init_lokr_norm > 0.0 {
            match model.bundle.apply_init_perturbed_normal(&model.weights, args.init_lokr_norm) {
                Ok(skipped) if skipped > 0 => log::warn!(
                    "[init_lokr_norm] {} slot(s) skipped (see warnings above)", skipped,
                ),
                Ok(_) => log::info!("[init_lokr_norm] applied scale={}", args.init_lokr_norm),
                Err(e) => log::warn!("[init_lokr_norm] failed: {e}"),
            }
        }
    } else {
        log::info!("[Anima] algo='lora' (legacy LoRALinear path, byte-identical)");
    }

    let params = model.parameters();
    let adapter_count = model
        .bundle
        .lyc_adapters
        .as_ref()
        .map(|v| v.len())
        .unwrap_or(model.bundle.adapters.len());
    log::info!("Loaded {} trainable LoRA tensors ({} adapters across {} blocks)",
        params.len(), adapter_count, anima_mod::NUM_BLOCKS);
    if params.is_empty() {
        anyhow::bail!("No trainable parameters — TrainingMethod::Lora produced empty param list");
    }

    let opt_kind = OptimizerKind::parse(&args.optimizer)
        .map_err(|e| anyhow::anyhow!("--optimizer: {e}"))?;
    if matches!(opt_kind, OptimizerKind::AdamW8bit) {
        anyhow::bail!(
            "AdamW8bit is forbidden in EDv2 (no-quantization rule per CLAUDE.md \
             — applies to Anima as well as Z-Image / Klein). Use `--optimizer adamw` \
             (BF16 stochastic-round AdamW) or another non-quantized optimizer."
        );
    }
    log::info!("[Anima] optimizer={}", opt_kind.as_str());
    let effective_caption_dropout_prob = args.caption_dropout_probability;
    // Anima caption signal is FOUR fields: text_embedding (Qwen3 cap_feats),
    // text_mask (Qwen3 mask), t5_input_ids (LLM Adapter input), t5_attn_mask.
    // ALL four are caption-dependent (the LLM Adapter combines T5 IDs with
    // cap_feats inside the model), so a correct CFG-style dropout must swap
    // every field together under one Bernoulli draw — otherwise the
    // unconditional path leaks the conditional T5 token sequence.
    let null_text: Option<(Tensor, Option<Tensor>, Option<Tensor>, Option<Tensor>)> =
        if effective_caption_dropout_prob > 0.0 {
        match args.null_text_cache.as_ref() {
            Some(p) => match flame_core::serialization::load_file(p, &device) {
                Ok(s) => {
                    let nt = s.get("text_embedding")
                        .ok_or_else(|| anyhow::anyhow!("--null-text-cache missing 'text_embedding'"))?
                        .to_dtype(DType::BF16)?;
                    let nm = s.get("text_mask").cloned();
                    let nt5_ids = s.get("t5_input_ids").cloned();
                    let nt5_mask = s.get("t5_attn_mask").cloned();
                    log::info!(
                        "[caption-dropout] WIRED — prob={:.3} (null text={:?}, mask={}, t5_ids={}, t5_mask={})",
                        effective_caption_dropout_prob,
                        nt.shape().dims(),
                        nm.is_some(),
                        nt5_ids.is_some(),
                        nt5_mask.is_some(),
                    );
                    Some((nt, nm, nt5_ids, nt5_mask))
                }
                Err(e) => anyhow::bail!(
                    "[caption-dropout] failed to load --null-text-cache {}: {e}. \
                     Either provide a valid null-text cache or set \
                     --caption-dropout-probability 0 to disable.",
                    p.display()
                ),
            },
            None => anyhow::bail!(
                "--caption-dropout-probability={:.3} requires --null-text-cache. \
                 Run prepare_anima on an empty caption first, then pass that \
                 cache path. Failing loud rather than silently disabling CFG \
                 conditioning training.",
                effective_caption_dropout_prob
            ),
        }
    } else {
        None
    };
    let mut opt = Optimizer::new(opt_kind, args.lr, 0.9, 0.999, 1e-8, 0.01);

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

    // Unified OneTrainer timestep distribution dispatch (optional override).
    // When `--timestep-distribution` is `auto`, the legacy `--timestep-sampling`
    // path is used unchanged (default-off byte invariance). Otherwise we build
    // a `TimestepConfig` and let it drive the sampler.
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

    if let Some(resume_path) = args.resume_lora.as_ref() {
        log::info!("Resuming LoRA weights only (no optimizer state) from {}", resume_path.display());
        model.load_weights(&resume_path.to_string_lossy())?;
    }

    let mut start_step: usize = 0;
    if let Some(resume_path) = args.resume_full.as_ref() {
        log::info!("Full-resume from {}", resume_path.display());
        let loaded = checkpoint::load_full(resume_path, &device)?;
        let named = model.named_parameters();
        checkpoint::apply_lora_weights(&loaded, &named)?;
        if let Optimizer::AdamW(ref mut adam) = opt {
            checkpoint::apply_to_optimizer(&loaded, adam, &named, args.rank, args.lora_alpha as f32)?;
        } else {
            log::warn!(
                "[resume-full] non-AdamW resume not yet implemented for {:?}; LoRA weights restored, optimizer state reset",
                opt.kind()
            );
        }
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
    log::info!("Found {} cached samples", cache_files.len());

    // Cache-version sentinel. prepare_anima writes `_meta.json` with
    // `version >= 2` once T5 pad rows are zeroed in `text_embedding`. Pre-fix
    // caches (no sentinel, or version 1) had non-zero activations at pad
    // positions that leaked through all 28 cross-attn layers — silent quality
    // degradation. Bail loudly so users re-run prepare_anima.
    let meta_path = args.cache_dir.join("_meta.json");
    if !meta_path.exists() {
        anyhow::bail!(
            "Cache at {} has no `_meta.json` sentinel — likely a pre-fix cache that trains on non-zeroed T5 pad rows. \
             Re-run `prepare_anima` to regenerate the cache with the mask-zeroing fix.",
            args.cache_dir.display()
        );
    }
    let meta_str = std::fs::read_to_string(&meta_path)
        .map_err(|e| anyhow::anyhow!("read {}: {e}", meta_path.display()))?;
    if !meta_str.contains("\"version\": 2") && !meta_str.contains("\"version\":2") {
        anyhow::bail!(
            "Cache `_meta.json` at {} reports an unsupported version: {}. \
             Expected version 2 (T5 mask-zeroed). Re-run `prepare_anima`.",
            meta_path.display(),
            meta_str.trim()
        );
    }
    log::info!("[cache-meta] {}", meta_str.trim());

    let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);
    let debug_grads_enabled = dbg::enabled("ANIMA_DEBUG_GRADS");
    if debug_grads_enabled {
        log::info!("ANIMA_DEBUG_GRADS=1 — per-step LoRA grad summaries enabled at steps 0/1/2/100/200/...");
    }

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

        let latent = sample.get("latent")
            .ok_or_else(|| anyhow::anyhow!("cached sample missing 'latent'"))?
            .to_dtype(DType::BF16)?;
        let cap_feats = sample.get("text_embedding")
            .ok_or_else(|| anyhow::anyhow!("cached sample missing 'text_embedding'"))?
            .to_dtype(DType::BF16)?;
        let cap_mask = sample.get("text_mask").cloned();
        let t5_ids = sample.get("t5_input_ids").cloned();
        let t5_mask = sample.get("t5_attn_mask").cloned();
        // Caption dropout: single Bernoulli swaps ALL four caption-dependent
        // fields together (cap_feats, cap_mask, t5_ids, t5_mask). The LLM
        // Adapter inside `AnimaModel::forward` re-encodes T5 ids alongside
        // cap_feats — leaving t5_ids conditional while zeroing cap_feats
        // would still leak the prompt, defeating CFG training.
        // Default-off (prob == 0.0 OR null_text == None) draws no rng.
        let (cap_feats, cap_mask, t5_ids, t5_mask) =
            if let Some((ref nt, ref nm, ref nt5_ids, ref nt5_mask)) = null_text {
            use rand::Rng;
            if rng.r#gen::<f32>() < effective_caption_dropout_prob {
                (
                    nt.clone(),
                    nm.clone(),
                    nt5_ids.clone(),
                    nt5_mask.clone(),
                )
            } else {
                (cap_feats, cap_mask, t5_ids, t5_mask)
            }
        } else {
            (cap_feats, cap_mask, t5_ids, t5_mask)
        };

        // Timestep sampling.
        let raw_t = if let Some(ref tcfg) = unified_timestep_cfg {
            // Unified dispatch: sample u in [0,1] then scale to [0, NUM_TRAIN_TIMESTEPS).
            tcfg.sample_one(&mut rng) * NUM_TRAIN_TIMESTEPS as f32
        } else {
            match args.timestep_sampling.as_str() {
                "sigmoid" => sample_timestep_sigmoid(&mut rng, args.sigmoid_scale),
                "uniform" => sample_timestep_uniform(&mut rng),
                "shift" => {
                    let raw = sample_timestep_sigmoid(&mut rng, args.sigmoid_scale);
                    let s = raw / NUM_TRAIN_TIMESTEPS as f32;
                    let shifted = apply_shift(s, args.discrete_flow_shift);
                    shifted * NUM_TRAIN_TIMESTEPS as f32
                }
                other => anyhow::bail!("unknown --timestep-sampling: {other}"),
            }
        };
        // Default-off: Strategy::None → returns raw_t unchanged.
        let t_continuous = timestep_bias::apply_bias(
            raw_t,
            NUM_TRAIN_TIMESTEPS as f32,
            &timestep_bias_cfg,
        );
        let sigma_continuous = (t_continuous / NUM_TRAIN_TIMESTEPS as f32).clamp(0.0, 1.0);
        // Apply discrete_flow_shift to the sigma used for noising (matches
        // kohya `flux_train_utils.get_noisy_model_input_and_timesteps` shift path).
        let sigma = if args.timestep_sampling != "shift" {
            apply_shift(sigma_continuous, args.discrete_flow_shift)
        } else {
            sigma_continuous
        };

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
        // clean noise distribution; input perturbation feeds model input only.
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
        let target = clean_noise.sub(&latent)?;
        // Anima's DiT receives timestep in [0, 1] (kohya divides by 1000 in
        // anima_train_network.py:279 BEFORE feeding into model_pred).
        let timestep = Tensor::from_vec(
            vec![sigma],
            Shape::from_dims(&[1]),
            device.clone(),
        )?;

        if step == 0 {
            log::info!("step 0 | latent={:?} cap={:?} sigma={:.4} (t={:.2})",
                latent.shape().dims(), cap_feats.shape().dims(), sigma, t_continuous);
        }

        // Build context vector for TrainableModel::forward.
        let mut context: Vec<Tensor> = vec![cap_feats];
        if let Some(m) = cap_mask { context.push(m); }
        if let Some(ids) = t5_ids { context.push(ids); }
        if let Some(m) = t5_mask { context.push(m); }

        let pred = <AnimaModel as TrainableModel>::forward(&mut model, &noisy, &timestep, &context, None)?;

        if pred.shape().dims() != target.shape().dims() {
            anyhow::bail!(
                "predicted shape {:?} != target {:?} — model.forward output mismatch",
                pred.shape().dims(), target.shape().dims()
            );
        }

        // MSE in F32 with optional per-sample sigma weighting.
        // Phase 1: combined loss + per-step weighting layered ON TOP of the
        // existing weighting_scheme weight so both signals compose cleanly.
        let weight = loss_weight(&args.weighting_scheme, sigma);
        let pred_f32 = pred.to_dtype(DType::F32)?;
        let target_f32 = target.to_dtype(DType::F32)?;
        let raw_loss = feat_loss_weight::combined_loss(
            &pred_f32,
            &target_f32,
            config.mse_strength as f32,
            config.mae_strength as f32,
            args.huber_strength,
        )?;
        let weighted = if (weight - 1.0).abs() > 1e-6 {
            raw_loss.mul_scalar(weight)?
        } else {
            raw_loss
        };
        let loss = feat_loss_weight::apply_loss_weight(
            &weighted,
            sigma,
            config.loss_weight_fn,
            args.min_snr_gamma,
            true,
        )?;
        let loss_val = loss.to_vec()?[0];
        total_loss += loss_val;

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

        if debug_grads_enabled && (step < 3 || (step + 1) % 100 == 0) {
            // Phase 2b: per-class LoRA grad summary is wired only for the
            // legacy plain-LoRA path (it expects `&[LoRALinear]`). LyCORIS
            // adapters report through `params`/`grads` already; per-class
            // breakdown for non-LoRA algos is a Phase 2c task.
            if model.bundle.lyc_adapters.is_none() {
                dbg::print_lora_grad_summary(step, &model.bundle.adapters, &ANIMA_LORA_CLASSES, &grads);
            } else if step == 0 {
                log::info!(
                    "ANIMA_DEBUG_GRADS: per-class summary disabled for LyCORIS algo `{}` (Phase 2c)",
                    model.bundle.algo.as_str(),
                );
            }
        }

        // OT default: clip_grad_norm = 1.0.
        const CLIP_GRAD_NORM: f32 = 1.0;
        // Fusion Sprint Phase 5: device-resident global L2 norm. Replaces a
        // per-tensor `.to_vec()?[0]` loop (N D2H syncs/step) with one D2H.
        let grad_refs: Vec<&flame_core::Tensor> = params
            .iter()
            .filter_map(|p| grads.get(p.id()))
            .collect();
        let total_norm = flame_core::ops::grad_norm::global_l2_norm(&grad_refs)?
            .item()? as f32;
        let scale = if total_norm > CLIP_GRAD_NORM { CLIP_GRAD_NORM / total_norm } else { 1.0 };
        for param in &params {
            if let Some(g) = grads.get(param.id()) {
                let g_scaled = if scale < 1.0 { g.mul_scalar(scale)? } else { g.clone() };
                param.set_grad(g_scaled)?;
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

        let _ = total_loss;
        eridiffusion_core::training::progress::log_step(
            "anima-lora",
            step, args.steps, cache_files.len(), 1,
            loss_val, total_norm, cur_lr, t_start, board.as_ref(),
        );

        // Periodic save (in-trainer sample rendering not yet wired).
        let step_num = step + 1;
        let do_periodic_save =
            args.save_every > 0 && step_num % args.save_every == 0 && step_num < args.steps;
        // EMA swap: when --ema --ema-validation-swap, save observes
        // EMA-averaged weights. Backup is restored at the end of this block.
        let ema_backup = if do_periodic_save && args.ema_validation_swap {
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
        if do_periodic_save {
            let mid_ckpt = args.output_dir.join(format!("anima_lora_step{step_num}.safetensors"));
            if save_mode_full {
                if let Optimizer::AdamW(ref adam) = opt {
                    let header = CkptHeader::from_adamw(
                        "train_anima", step_num as u64, adam,
                        args.rank, args.lora_alpha as f32, SEED, String::new(),
                    );
                    let named = model.named_parameters();
                    if let Err(e) = checkpoint::save_full(&mid_ckpt, &named, adam, &header) {
                        log::warn!("[mid-save step {step_num}] full save failed: {e}");
                    }
                } else {
                    log::warn!(
                        "[mid-save step {step_num}] full-state save not yet implemented for {:?}; saving weights only",
                        opt.kind()
                    );
                    if let Err(e) = model.save_weights(&mid_ckpt.to_string_lossy()) {
                        log::warn!("[mid-save step {step_num}] weights-only save failed: {e}");
                    }
                }
            } else if let Err(e) = model.save_weights(&mid_ckpt.to_string_lossy()) {
                log::warn!("[mid-save step {step_num}] save_weights failed: {e}");
            } else {
                log::info!("[mid-save step {step_num}] {}", mid_ckpt.display());
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

    let ckpt = args.output_dir.join(format!("anima_lora_{}steps.safetensors", args.steps));
    if save_mode_full {
        if let Optimizer::AdamW(ref adam) = opt {
            let header = CkptHeader::from_adamw(
                "train_anima", args.steps as u64, adam,
                args.rank, args.lora_alpha as f32, SEED, String::new(),
            );
            let named = model.named_parameters();
            if let Err(e) = checkpoint::save_full(&ckpt, &named, adam, &header) {
                log::warn!("save_full failed: {e}");
            } else {
                log::info!("Saved checkpoint to {}", ckpt.display());
            }
        } else {
            log::warn!(
                "[final] full-state save not yet implemented for {:?}; saving weights only",
                opt.kind()
            );
            if let Err(e) = model.save_weights(&ckpt.to_string_lossy()) {
                log::warn!("weights-only save failed: {e}");
            } else {
                log::info!("Saved weights-only checkpoint to {}", ckpt.display());
            }
        }
    } else if let Err(e) = model.save_weights(&ckpt.to_string_lossy()) {
        log::warn!("save_weights returned error: {e}");
    } else {
        log::info!("Saved checkpoint to {}", ckpt.display());
    }

    Ok(())
}
