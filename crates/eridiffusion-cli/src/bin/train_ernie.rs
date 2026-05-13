//! train_ernie — Ernie LoRA training binary, mirroring EriDiffusion Python flow.
//!
//! Reference: /home/alex/upstream Python/modules/modelSetup/BaseErnieSetup.py predict() + calculate_loss().
//! Pipeline per step:
//!   1. Load cached `latent` ([B,128,h,w], scale_latents already applied at cache time)
//!      and `text_embedding` ([B,T,3072]).
//!   2. Sample integer-valued timestep ∈ [0, num_train_timesteps) per LOGIT_NORMAL
//!      distribution with shift=1 (OT preset default).
//!   3. sigma = (floor(t)+1) / num_train_timesteps; noisy = noise·sigma + clean·(1-sigma).
//!   4. Forward → [B,128,h,w]; target = noise - clean (rectified flow).
//!   5. Loss = mean MSE in F32 (loss_weight_fn=CONSTANT, mse_strength=1.0 default).
use clap::Parser;
use std::path::PathBuf;
use flame_core::{autograd::AutogradContext, DType, Shape, Tensor};
use eridiffusion_core::config::{LrScheduler, TrainConfig, TrainingMethod};
use eridiffusion_core::lycoris::{LoraInitType, LycorisAlgo, LycorisBundleConfig};
use eridiffusion_core::training::ema::ParameterEma;
use eridiffusion_core::training::features::ema_advanced::EmaConfig;
use eridiffusion_core::training::features::{loss_weight, lr_schedule, noise_modifiers, timestep_bias, validation::ValidationLoop};
use eridiffusion_core::training::training_features::{Optimizer, OptimizerKind};
use eridiffusion_core::training::training_features::timestep_dist::{TimestepConfig, TimestepDistribution};
use std::str::FromStr as _;
use eridiffusion_core::debug as dbg;
use eridiffusion_core::encoders::{mistral3b::Mistral3bEncoder, vae::KleinVaeDecoder};
use eridiffusion_core::models::{ErnieModel, TrainableModel};
use eridiffusion_core::sampler::ernie_sampler;
use eridiffusion_core::training::board::BoardWriter;
use eridiffusion_core::training::checkpoint::{self, CkptHeader};

/// Class names for the 7 LoRA modules per ERNIE layer (must mirror
/// `ErnieModel::load`'s adapter creation order: Q, K, V, out, gate, up, down).
const ERNIE_LORA_CLASSES: [&str; 7] = [
    "self_attn.to_q", "self_attn.to_k", "self_attn.to_v", "self_attn.to_out",
    "mlp.gate_proj", "mlp.up_proj", "mlp.linear_fc2",
];

const NUM_TRAIN_TIMESTEPS: usize = 1000;
const LOGIT_NORMAL_BIAS: f32 = 0.0;       // TrainConfig.noising_bias default
const LOGIT_NORMAL_SCALE: f32 = 1.0;      // noising_weight + 1.0 = 0.0 + 1.0
const TIMESTEP_SHIFT: f32 = 1.0;          // TrainConfig.timestep_shift default
const SEED: u64 = 42;                     // memory: feedback_default_seed_42 — fixed across step

#[derive(Parser)]
struct Args {
    #[arg(long)] config: PathBuf,
    #[arg(long)] cache_dir: PathBuf,
    #[arg(long, default_value = "100")] steps: usize,
    #[arg(long, default_value = "16")] rank: usize,
    #[arg(long, default_value = "1.0")] lora_alpha: f64,
    #[arg(long, default_value = "3e-4")] lr: f32,
    /// Resume LoRA weights only (no optimizer state).
    #[arg(long, conflicts_with = "resume_full")] resume_lora: Option<PathBuf>,
    /// Full resume: LoRA weights + AdamW (m, v, t) + step counter.
    #[arg(long, conflicts_with = "resume_lora")] resume_full: Option<PathBuf>,
    /// Periodic + final save mode. Default `full` (LoRA + AdamW + step) for
    /// resumable runs. `weights` writes legacy weights-only files.
    #[arg(long, default_value = "full")] save_mode: String,
    #[arg(long, default_value = "output")] output_dir: PathBuf,

    // ── Periodic save + sample (every N steps) ─────────────────────────
    /// Save a LoRA checkpoint AND render a sample image every N steps.
    /// `0` disables.
    #[arg(long, default_value = "0")] sample_every: usize,
    #[arg(long, default_value = "")] sample_prompt: String,
    #[arg(long, default_value = "")] sample_neg_prompt: String,
    /// ERNIE Klein-VAE safetensors (single full file; see prepare_ernie's --vae-ckpt).
    #[arg(long)] sample_vae: Option<PathBuf>,
    /// Mistral-3B text encoder checkpoint (matches prepare_ernie --text-ckpt).
    #[arg(long)] sample_text_ckpt: Option<PathBuf>,
    /// Tokenizer.json path for the Mistral-3B encoder.
    #[arg(long)] sample_tokenizer: Option<PathBuf>,
    #[arg(long, default_value = "1024")] sample_size: usize,
    #[arg(long, default_value = "20")] sample_steps: usize,
    #[arg(long, default_value = "4.0")] sample_cfg: f32,
    #[arg(long, default_value = "42")] sample_seed: u64,

    // ── Phase 0 multi-feature rollout (default-off; Phase 1+ will consume) ──
    #[arg(long)] min_snr_gamma: Option<f32>,
    #[arg(long, default_value_t = 0.0)] caption_dropout_probability: f32,
    /// Path to a single cache file produced by `prepare_ernie` from an empty-
    /// caption sample. When `--caption-dropout-probability > 0`, the trainer
    /// loads `text_embedding` + `text_real_len` from this file and swaps them
    /// in with probability `p` per step. If unset and dropout > 0, the feature
    /// is disabled with a warning (preserves prior behaviour).
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
    /// Timestep distribution. `logit_normal` (default — ERNIE preset),
    /// `uniform`, `sigmoid`, `heavy_tail`, `cos_map`, `inverted_parabola`.
    #[arg(long, default_value = "logit_normal")] timestep_distribution: String,
    /// Distribution-specific weight knob (default 0.0 — ERNIE preset).
    #[arg(long, default_value_t = 0.0)] noising_weight: f32,
    /// Distribution-specific bias knob (default 0.0 — ERNIE preset).
    #[arg(long, default_value_t = 0.0)] noising_bias: f32,
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

    // ── LyCORIS algo selection (Phase 2b) ──
    //
    // `--algo lora` (default) keeps the legacy LoRALinear path — byte-identical
    // training to pre-Phase-2b. Other values select LyCORIS algos via
    // `LycorisBundleConfig`.
    /// LyCORIS algo: `lora` (default, legacy path) | `locon` | `loha` | `lokr`
    /// | `full` | `oft`. ERNIE is linear-only (no Conv) so `use_tucker` is
    /// a no-op. `full` and `oft` build successfully but their `forward_delta`
    /// is incompatible with ernie's `base + delta_on_input` call pattern;
    /// Phase 2c will wire merge-into-base.
    #[arg(long, default_value = "lora")] algo: String,
    /// LoKr Kronecker split factor (ignored for non-LoKr).
    #[arg(long, default_value_t = 16)] lokr_factor: i32,
    /// OFT/BOFT block size (ignored for non-OFT/BOFT).
    #[arg(long, default_value_t = 32)] oft_block_size: usize,
    /// OFT Cayley-Neumann series term count.
    #[arg(long, default_value_t = 5)] oft_neumann_terms: usize,
    /// Tucker decomposition for non-1×1 conv kernels (ernie is linear-only).
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
    /// SimpleTuner-parity perturbed-normal LoKr init. Phase 2b ernie:
    /// no-op stub (ERNIE base weights live in BlockOffloader pinned RAM,
    /// not resident at swap time — same situation as qwenimage). A warning
    /// is logged when scale > 0; Phase 2c will plumb the resident weight
    /// map into this path.
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

/// LOGIT_NORMAL timestep sample matching OT _get_timestep_discrete.
/// Superseded by the unified `TimestepConfig` dispatch — kept for reference.
#[allow(dead_code)]
fn sample_timestep_logit_normal(rng: &mut rand::rngs::StdRng) -> f32 {
    use rand_distr::{Distribution, Normal};
    let normal = Normal::new(LOGIT_NORMAL_BIAS, LOGIT_NORMAL_SCALE).unwrap();
    let z = normal.sample(rng);
    let logit_normal = 1.0 / (1.0 + (-z).exp());
    let t = logit_normal * NUM_TRAIN_TIMESTEPS as f32;
    // shift transform: ts * shift / ((shift-1)*ts + N). With shift=1 → identity.
    if (TIMESTEP_SHIFT - 1.0).abs() < 1e-6 {
        t
    } else {
        NUM_TRAIN_TIMESTEPS as f32 * TIMESTEP_SHIFT * t
            / ((TIMESTEP_SHIFT - 1.0) * t + NUM_TRAIN_TIMESTEPS as f32)
    }
}

/// Build the unified `TimestepConfig` from CLI args.
fn build_timestep_config(
    distribution: &str,
    weight: f32,
    bias: f32,
) -> anyhow::Result<TimestepConfig> {
    let dist = TimestepDistribution::from_str(distribution)
        .map_err(|e| anyhow::anyhow!("--timestep-distribution: {e}"))?;
    Ok(TimestepConfig {
        distribution: dist,
        noising_weight: weight,
        noising_bias: bias,
        min_strength: 0.0,
        max_strength: 1.0,
    })
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
            "[masked-loss] --masked-loss-weight={:.3} requested but ERNIE's prepare_ernie cache schema has no `latent_mask` field; flag is a no-op for this trainer.",
            args.masked_loss_weight
        );
    }
    config.ema_inv_gamma = args.ema_inv_gamma;
    config.ema_power = args.ema_power;
    config.ema_update_after_step = args.ema_update_after_step;
    config.ema_min_decay = args.ema_min_decay;
    config.tread_route_pattern = args.tread_route_pattern.clone();

    let model_base = std::path::Path::new(&config.base_model_name);
    let shards: Vec<PathBuf> = (1..=2).map(|i|
        model_base.join("transformer").join(format!("diffusion_pytorch_model-0000{i}-of-00002.safetensors"))
    ).collect();

    log::info!("Loading Ernie transformer (rank={} alpha={})...", args.rank, args.lora_alpha);
    let mut model = ErnieModel::load(&shards, &config, device.clone())?;

    // Phase 2b: parse the LyCORIS algo selector. `lora` (default) keeps the
    // legacy LoRALinear bundle (byte-identical pre-2b path); anything else
    // swaps in a LyCORIS-aware adapter set in place. See
    // train_chroma.rs:284 for the canonical pattern.
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
            "[ernie] LyCORIS algo='{}' rank={} alpha={} factor={} block_size={} dora={}",
            algo.as_str(),
            lyc_config.rank,
            lyc_config.alpha,
            lyc_config.factor,
            lyc_config.block_size,
            lyc_config.dora,
        );
        if matches!(algo, LycorisAlgo::Full | LycorisAlgo::Oft) {
            log::warn!(
                "[ernie] algo='{}' selected — bundle construction will succeed, but \
                 forward_delta will error inside ernie's `base + delta_on_input` call \
                 pattern. Phase 2c will wire merge-into-base for these algos.",
                algo.as_str()
            );
        }
        model.swap_lycoris_bundle(&lyc_config)
            .map_err(|e| anyhow::anyhow!("LyCORIS bundle swap: {e}"))?;
        if matches!(algo, LycorisAlgo::LoKr) && args.init_lokr_norm > 0.0 {
            // Phase 2c — perturbed-normal LoKr init. Walks lycoris_adapters
            // and looks up `layers.{N}.{slot_suffix}.weight` in
            // `model.weights`. When the model is loaded with `--offload`
            // the per-layer weights are absent from the resident map and
            // the apply method logs per-key warnings and skips them.
            let skipped = model
                .apply_init_perturbed_normal(args.init_lokr_norm)
                .map_err(|e| anyhow::anyhow!("init_lokr_norm: {e}"))?;
            if skipped > 0 {
                log::warn!(
                    "[ernie] init_lokr_norm: {} slot(s) skipped (see warnings above; \
                     usually means model was loaded with --offload and base weights \
                     are streamed rather than resident).",
                    skipped
                );
            }
        }
        // Suppress unused-warnings for dropout flags until LycorisLinear
        // exposes a per-call dropout knob (Phase 2c).
        let _ = (args.lora_dropout, args.rank_dropout, args.module_dropout, args.rank_dropout_scale);
    } else {
        log::info!("[ernie] algo='lora' (legacy LoRALinear path, byte-identical)");
    }

    let params = model.parameters();
    log::info!("Loaded {} trainable LoRA tensors", params.len());
    if params.is_empty() {
        anyhow::bail!("No trainable parameters — TrainingMethod::Lora produced empty param list");
    }

    let opt_kind = OptimizerKind::parse(&args.optimizer)
        .map_err(|e| anyhow::anyhow!("--optimizer: {e}"))?;
    log::info!("[ERNIE] optimizer={}", opt_kind.as_str());
    // Phase 1: caption_dropout. ERNIE has no inline encoder, so the user
    // supplies a `--null-text-cache` produced by `prepare_ernie` on a single
    // empty-caption sample. We load `text_embedding` + `text_real_len` once
    // and swap them in per-step with the configured probability. Without
    // `--null-text-cache`, the feature is disabled with a warning.
    let mut effective_caption_dropout_prob = args.caption_dropout_probability;
    let null_text: Option<(Tensor, Option<usize>)> = if effective_caption_dropout_prob > 0.0 {
        match args.null_text_cache.as_ref() {
            Some(p) => match flame_core::serialization::load_file(p, &device) {
                Ok(s) => {
                    let nt = s.get("text_embedding")
                        .ok_or_else(|| anyhow::anyhow!("--null-text-cache missing 'text_embedding'"))?
                        .to_dtype(DType::BF16)?;
                    let nrl: Option<usize> = if let Some(rl_t) = s.get("text_real_len") {
                        let rl = rl_t.to_dtype(DType::F32)?.to_vec()?[0] as usize;
                        Some(rl)
                    } else {
                        None
                    };
                    log::info!(
                        "[caption-dropout] WIRED — prob={:.3} (null_text_embedding={:?}, null_text_real_len={:?})",
                        effective_caption_dropout_prob,
                        nt.shape().dims(),
                        nrl
                    );
                    Some((nt, nrl))
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

    // Unified OneTrainer timestep distribution dispatch.
    let timestep_cfg = build_timestep_config(
        &args.timestep_distribution,
        args.noising_weight,
        args.noising_bias,
    )?;

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

    // Cached samples produced by `prepare_ernie`. Order is shuffled by seed to match
    // OT's deterministic-batch scheme without depending on filesystem order.
    let mut cache_files: Vec<PathBuf> = std::fs::read_dir(&args.cache_dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map_or(false, |e| e == "safetensors"))
        .collect();
    cache_files.sort();
    if cache_files.is_empty() {
        anyhow::bail!("No cached samples in {:?}", args.cache_dir);
    }
    log::info!("Found {} cached samples", cache_files.len());

    // Phase 2: validation harness — held-out cache + cadence. None at default
    // (validation_every_steps == 0) → no harness, no branch, byte-identical.
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

    // Single seeded RNG drives both timestep + per-step noise (memory: feedback_default_seed_42).
    let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);
    let debug_grads_enabled = dbg::enabled("ERNIE_DEBUG_GRADS");
    if debug_grads_enabled {
        log::info!("ERNIE_DEBUG_GRADS=1 — per-step LoRA grad summaries enabled at steps 0/1/2/100/200/...");
    }

    // ── Periodic-sample setup ────────────────────────────────────────────
    // Pre-encode cond/uncond prompts ONCE then drop the text encoder from VRAM.
    // VAE is loaded lazily per sample (small, cheap).
    let periodic = args.sample_every > 0;
    let (sample_cap, sample_uncond, sample_vae_path) = if periodic {
        let te_path = args.sample_text_ckpt.as_ref()
            .ok_or_else(|| anyhow::anyhow!("--sample-every > 0 requires --sample-text-ckpt"))?;
        let tok_path = args.sample_tokenizer.as_ref()
            .ok_or_else(|| anyhow::anyhow!("--sample-every > 0 requires --sample-tokenizer"))?;
        let vae_path = args.sample_vae.as_ref()
            .ok_or_else(|| anyhow::anyhow!("--sample-every > 0 requires --sample-vae"))?
            .clone();
        const ERNIE_MAX_LEN: usize = 512;
        const ERNIE_PAD_ID: i32 = 11;
        log::info!("[sample-setup] encoding prompt + uncond once with Mistral-3B...");
        let tok = tokenizers::Tokenizer::from_file(tok_path)
            .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
        let encode = |text: &str| -> anyhow::Result<Vec<i32>> {
            let e = tok.encode(text, true).map_err(|e| anyhow::anyhow!("{e}"))?;
            let mut ids: Vec<i32> = e.get_ids().iter().map(|&x| x as i32).collect();
            if ids.len() > ERNIE_MAX_LEN { ids.truncate(ERNIE_MAX_LEN); }
            Ok(ids)
        };
        let (cond_ids, unc_ids) = (encode(&args.sample_prompt)?, encode(&args.sample_neg_prompt)?);
        let te = Mistral3bEncoder::load(te_path.to_str().unwrap(), &device)?;
        let cap = te.encode_with_pad(&cond_ids, ERNIE_MAX_LEN, ERNIE_PAD_ID)?;
        let unc = te.encode_with_pad(&unc_ids, ERNIE_MAX_LEN, ERNIE_PAD_ID)?;
        let cap_len = cond_ids.len().max(1);
        let unc_len = unc_ids.len().max(1);
        let cap_trim = cap.narrow(1, 0, cap_len)?.contiguous()?;
        let unc_trim = unc.narrow(1, 0, unc_len)?.contiguous()?;
        log::info!("[sample-setup] cap={:?} uncond={:?}; dropping text encoder",
            cap_trim.shape().dims(), unc_trim.shape().dims());
        drop(te);
        flame_core::cuda_alloc_pool::clear_pool_cache();
        flame_core::trim_cuda_mempool(0);
        log::info!("[sample-setup] periodic sample enabled (every {} steps).", args.sample_every);
        (Some(cap_trim), Some(unc_trim), Some(vae_path))
    } else {
        (None, None, None)
    };

    // Step-0 baseline sample (LoRA-init = base model output)
    if periodic {
        let cap = sample_cap.as_ref().unwrap();
        let unc = sample_uncond.as_ref().unwrap();
        let vae_path = sample_vae_path.as_ref().unwrap();
        let out_path = args.output_dir.join("sample_step0_base.png");
        log::info!("[sample step=0] BASELINE → {}", out_path.display());
        if let Err(e) = ernie_inline_sample(
            &mut model, cap, unc, vae_path, &out_path,
            args.sample_size, args.sample_steps, args.sample_cfg, args.sample_seed, &device,
        ) {
            log::warn!("[sample step=0] failed: {e}");
        }
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

        // `latent` is post-VAE post-patchify post-scale: [B, 128, h, w] in BF16.
        let latent = sample
            .get("latent")
            .ok_or_else(|| anyhow::anyhow!("cached sample missing 'latent'"))?
            .to_dtype(DType::BF16)?;
        let txt_full = sample
            .get("text_embedding")
            .ok_or_else(|| anyhow::anyhow!("cached sample missing 'text_embedding'"))?
            .to_dtype(DType::BF16)?;
        // Read real_len from sample (None → full pad length).
        let sample_rl: Option<usize> = if let Some(rl_t) = sample.get("text_real_len") {
            let rl = rl_t.to_dtype(DType::F32)?.to_vec()?[0] as usize;
            Some(rl)
        } else {
            None
        };
        // Caption dropout: single Bernoulli per step swaps both the text
        // embedding AND the real-length together (correlated, matching CFG
        // training convention). Default-off (prob == 0.0 OR null_text == None)
        // draws no rng.
        let (txt_full, real_len_opt) = if let Some((ref nt, nrl)) = null_text {
            use rand::Rng;
            if rng.r#gen::<f32>() < effective_caption_dropout_prob {
                (nt.clone(), nrl)
            } else {
                (txt_full, sample_rl)
            }
        } else {
            (txt_full, sample_rl)
        };
        // Trim padded text positions before feeding the DiT — matches upstream Python
        // ErnieModel.py:153-154 `text_encoder_output[:, :text_lengths.max(), :]`.
        // With batch_size=1 cache, real_len IS the trim length. If the cache was
        // produced by an older prepare_ernie that didn't write text_real_len,
        // fall back to the full padded length (legacy 77-pad behaviour).
        let txt = if let Some(rl) = real_len_opt {
            let tdims = txt_full.shape().dims().to_vec();
            let max_len = tdims[1];
            let real = rl.min(max_len).max(1);
            txt_full.narrow(1, 0, real)?.contiguous()?
        } else {
            txt_full
        };

        // Flow-matching noise schedule (OT _add_noise_discrete with discrete sigmas):
        //   sigma_idx ∈ [0, 999], sigma = (sigma_idx + 1) / 1000.
        // Continuous timestep in [0, 1000) is what the transformer's sin/cos sees.
        let raw_t = timestep_cfg.sample_one(&mut rng) * NUM_TRAIN_TIMESTEPS as f32;
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
        let timestep = Tensor::from_vec(
            vec![t_continuous],
            Shape::from_dims(&[1]),
            device.clone(),
        )?;

        if step == 0 {
            let l_dims = latent.shape().dims().to_vec();
            let t_dims = txt.shape().dims().to_vec();
            log::info!("step 0 | latent={:?} text={:?} sigma={:.4} (idx={})",
                l_dims, t_dims, sigma, sigma_idx);
        }

        let pred = model.forward(&noisy, &txt, &timestep)?;

        // Predicted flow shape now matches target shape: [B, 128, h, w].
        if pred.shape().dims() != target.shape().dims() {
            anyhow::bail!(
                "predicted flow shape {:?} != target {:?} — model.forward output mismatch",
                pred.shape().dims(), target.shape().dims()
            );
        }

        // OT loss: F.mse_loss(pred.float(), target.float(), reduction='none').mean(spatial).mean(batch)
        // — mse_strength=1.0, loss_weight_fn=CONSTANT, loss_weight=1.0 default.
        // Phase 1: combined loss + per-step weighting. Default-off invariant.
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

        // === Debug: per-LoRA-module-class gradient summary ===
        // Set ERNIE_DEBUG_GRADS=1 to enable. Catches the convergence-killer
        // class of bug: certain module classes silently getting near-zero grads.
        if debug_grads_enabled && (step < 3 || (step + 1) % 100 == 0) {
            dbg::print_lora_grad_summary(step, &model.lora_adapters, &ERNIE_LORA_CLASSES, &grads);
        }

        // === OT_DEBUG_STATS-format per-step line ===
        // Same fields, same key names as our patched upstream Python's
        // `[OT_DEBUG step=N] t=T loss(pre-scale)=L | pred[mean=… std=… max|·|=…] target[…]`.
        // Side-by-side `diff` against an upstream Python run on the same dataset
        // immediately surfaces forward / loss-magnitude divergence.
        if debug_grads_enabled || std::env::var("OT_DEBUG_STATS").map_or(false, |v| !matches!(v.as_str(), "0"|""|"false"|"FALSE")) {
            let p_st = dbg::stats(&pred);
            let t_st = dbg::stats(&target);
            eprintln!(
                "[OT_DEBUG step={:5}] t={:.2} loss(pre-scale)={:.4} | pred[mean={:+.3e} std={:.3e} max|·|={:.3e}] target[mean={:+.3e} std={:.3e} max|·|={:.3e}]",
                step, t_continuous, loss_val,
                p_st.mean, p_st.std, p_st.abs_max,
                t_st.mean, t_st.std, t_st.abs_max,
            );
        }

        // OT default: clip_grad_norm = 1.0. Without this, large early-step
        // gradients destabilize training and the LoRA never converges on
        // identity (verified vs OT preset).
        const CLIP_GRAD_NORM: f32 = 1.0;
        // Fusion Sprint Phase 5: device-resident global L2 norm — one D2H per step.
        let grad_refs: Vec<&flame_core::Tensor> = params
            .iter()
            .filter_map(|p| grads.get(p.id()))
            .collect();
        let total_norm = flame_core::ops::grad_norm::global_l2_norm(&grad_refs)?
            .item()? as f32;
        // Match OT_DEBUG_STATS line in upstream Python so a `grep grad_norm_pre_clip` diffs cleanly.
        if debug_grads_enabled || std::env::var("OT_DEBUG_STATS").map_or(false, |v| !matches!(v.as_str(), "0"|""|"false"|"FALSE")) {
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
            "ERNIE-lora",
            step, args.steps, cache_files.len(), 1,
            loss_val, total_norm, cur_lr, t_start, board.as_ref(),
        );

        // Phase 2: validation eval pass (no_grad) every `validation_every_steps`.
        // step+1 because `step` here is 0-based; ValidationLoop::should_run
        // expects the 1-based completed-step number.
        //
        // ERNIE divergence from Klein: we honour `text_real_len` if present in
        // the eval cache, mirroring the training-step trim path above (ErnieModel
        // forward expects `text_encoder_output[:, :real_len, :]`). Older eval
        // caches without that field fall back to the full padded length, which
        // is the same legacy behaviour as the training path.
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
                    let v_lat = match sample.get("latent") {
                        Some(t) => t.to_dtype(DType::BF16)?,
                        None => {
                            log::warn!("[validation] {} missing latent", vfile.display());
                            continue;
                        }
                    };
                    let v_txt_full = match sample.get("text_embedding") {
                        Some(t) => t.to_dtype(DType::BF16)?,
                        None => {
                            log::warn!("[validation] {} missing text_embedding", vfile.display());
                            continue;
                        }
                    };
                    // Mirror training-step text_real_len trim. Required so the
                    // val forward sees the same shape distribution as training.
                    let v_txt = if let Some(rl_t) = sample.get("text_real_len") {
                        let rl = rl_t.to_dtype(DType::F32)?.to_vec()?[0] as usize;
                        let tdims = v_txt_full.shape().dims().to_vec();
                        let max_len = tdims[1];
                        let real = rl.min(max_len).max(1);
                        v_txt_full.narrow(1, 0, real)?.contiguous()?
                    } else {
                        v_txt_full
                    };
                    // Sample timestep + noise identically to training. Validation
                    // uses its OWN run-side RNG so it does not perturb the
                    // training-side seeded sequence (byte invariance).
                    let mut vrng = rand::rngs::StdRng::seed_from_u64(SEED ^ (step as u64 + 1));
                    let t_continuous = timestep_cfg.sample_one(&mut vrng) * NUM_TRAIN_TIMESTEPS as f32;
                    let sigma_idx = (t_continuous.floor() as usize).min(NUM_TRAIN_TIMESTEPS - 1);
                    let sigma = (sigma_idx + 1) as f32 / NUM_TRAIN_TIMESTEPS as f32;
                    let v_noise = Tensor::randn(v_lat.shape().clone(), 0.0, 1.0, device.clone())?
                        .to_dtype(DType::BF16)?;
                    let v_noisy = v_noise.mul_scalar(sigma)?
                        .add(&v_lat.mul_scalar(1.0 - sigma)?)?;
                    let v_target = v_noise.sub(&v_lat)?;
                    let v_timestep = Tensor::from_vec(
                        vec![t_continuous],
                        Shape::from_dims(&[1]),
                        device.clone(),
                    )?;
                    let v_pred = match model.forward(&v_noisy, &v_txt, &v_timestep) {
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

        // ── Periodic save + inline sample (every N steps) ───────────────
        let step_num = step + 1;
        // EMA swap: when --ema --ema-validation-swap, save and sample see
        // EMA-averaged weights. Backup is restored at the end of this block.
        let ema_backup = if periodic && step_num % args.sample_every == 0
            && step_num < args.steps
            && args.ema_validation_swap
        {
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
        if periodic && step_num % args.sample_every == 0 && step_num < args.steps {
            let mid_ckpt = args.output_dir.join(format!("ernie_lora_step{step_num}.safetensors"));
            if save_mode_full {
                if let Optimizer::AdamW(ref adam) = opt {
                    let header = CkptHeader::from_adamw(
                        "train_ernie", step_num as u64, adam,
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
            let cap = sample_cap.as_ref().unwrap();
            let unc = sample_uncond.as_ref().unwrap();
            let vae_path = sample_vae_path.as_ref().unwrap();
            let sample_out = args.output_dir.join(format!("sample_step{step_num}.png"));
            log::info!("[sample step={step_num}] → {}", sample_out.display());
            if let Err(e) = ernie_inline_sample(
                &mut model, cap, unc, vae_path, &sample_out,
                args.sample_size, args.sample_steps, args.sample_cfg, args.sample_seed, &device,
            ) {
                log::warn!("[sample step={step_num}] failed: {e}");
            }
        }
        if let (Some(backup), Some(ref e)) = (ema_backup, &ema) {
            let _g = AutogradContext::no_grad();
            e.restore_swapped(&params, backup)
                .map_err(|err| anyhow::anyhow!("EMA restore_swapped (mid) failed: {err}"))?;
        }
    }

    // Final EMA swap before final save+sample. No restore — process exits.
    if args.ema_validation_swap {
        if let Some(ref e) = ema {
            let _g = AutogradContext::no_grad();
            let _ = e.swap_with_live(&params)
                .map_err(|err| anyhow::anyhow!("EMA swap_with_live (final) failed: {err}"))?;
            log::info!("[ema] swapped EMA shadow into live params for final save + sample");
        }
    }

    let trained = args.steps - start_step;
    let avg_loss = if trained > 0 { total_loss / trained as f32 } else { 0.0 };
    log::info!("Training complete: {trained} new steps (total={}), avg loss={:.4}", args.steps, avg_loss);
    if let Some(b) = &board { b.set_status("completed"); }

    let ckpt = args.output_dir.join(format!("ernie_lora_{}steps.safetensors", args.steps));
    if save_mode_full {
        if let Optimizer::AdamW(ref adam) = opt {
            let header = CkptHeader::from_adamw(
                "train_ernie", args.steps as u64, adam,
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
        log::warn!("save_weights returned error (currently a stub): {e}");
    } else {
        log::info!("Saved checkpoint to {}", ckpt.display());
    }

    // ── Final sample ───────────────────────────────────────────────────
    if periodic {
        let cap = sample_cap.as_ref().unwrap();
        let unc = sample_uncond.as_ref().unwrap();
        let vae_path = sample_vae_path.as_ref().unwrap();
        let sample_out = args.output_dir.join(format!("sample_step{}.png", args.steps));
        log::info!("[sample step={} FINAL] → {}", args.steps, sample_out.display());
        if let Err(e) = ernie_inline_sample(
            &mut model, cap, unc, vae_path, &sample_out,
            args.sample_size, args.sample_steps, args.sample_cfg, args.sample_seed, &device,
        ) {
            log::warn!("[sample final] failed: {e}");
        }
    }
    Ok(())
}

/// Inline sampler — uses live in-training model state and pre-encoded prompts.
/// Loads + drops VAE per call to bound VRAM.
fn ernie_inline_sample(
    model: &mut ErnieModel,
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
    let _no_grad = AutogradContext::no_grad();
    let h_lat = size / 16;
    let w_lat = size / 16;
    let sigmas = ernie_sampler::schedule(steps);

    use rand::SeedableRng;
    let _rng = rand::rngs::StdRng::seed_from_u64(seed);
    let latent_shape = Shape::from_dims(&[1, 128, h_lat, w_lat]);
    let mut latent = Tensor::randn(latent_shape, 0.0, 1.0, device.clone())?
        .to_dtype(DType::BF16)?;

    for step in 0..steps {
        let sigma = sigmas[step];
        let sigma_next = sigmas[step + 1];
        let t = ernie_sampler::sigma_to_timestep(sigma);
        let t_tensor = Tensor::from_vec(vec![t], Shape::from_dims(&[1]), device.clone())?;
        let pred_cond = model.forward(&latent, cond, &t_tensor)?;
        let pred_uncond = model.forward(&latent, uncond, &t_tensor)?;
        let pred = pred_uncond.add(&pred_cond.sub(&pred_uncond)?.mul_scalar(cfg)?)?;
        latent = ernie_sampler::euler_step(&latent, &pred, sigma, sigma_next)?;
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
