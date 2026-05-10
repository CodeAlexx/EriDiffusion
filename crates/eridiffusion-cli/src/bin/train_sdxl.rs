//! train_sdxl — SDXL LoRA training, mirroring EriDiffusion Python flow.
//!
//! Reference: OT preset `/home/alex/upstream Python/training_presets/#sdxl 1.0 LoRA.json`:
//!   - learning_rate     = 0.0003
//!   - batch_size        = 4   (per-step here is 1; grad-accum not yet wired)
//!   - resolution        = 1024
//!   - layer_filter_preset = "attn-mlp" → here covers attention only; FF/conv LoRA TBD
//!   - unet/te dtype     = FLOAT_16 / BFLOAT_16  (we do BF16 throughout — quant forbidden)
//!   - vae dtype         = FLOAT_32  (we keep VAE in BF16 at runtime; preset value is for
//!                                    fp4/fp8 quant pipelines we don't run)
//!
//! Pipeline per step (sd-scripts `sdxl_train.py:597-737`):
//!   1. Load cached `latent` [1,4,h,w], `text_embedding` [1,77,2048], `pooled` [1,2816].
//!   2. Sample integer timestep ∈ [0, 1000) — preset doesn't override → uniform.
//!   3. ε ~ N(0, I);  noisy = sqrt(ᾱ_t)·latent + sqrt(1-ᾱ_t)·ε
//!   4. target = ε  (epsilon prediction; preset doesn't set v-pred)
//!   5. UNet forward → pred
//!   6. Loss = mean MSE(pred, target) in F32 (mse_strength = 1.0, no min-SNR by default)
//!   7. clip_grad_norm = 1.0; AdamW step (β=(0.9, 0.999), ε=1e-8, wd=0.01).
//!
//! Cached sample format (produced by prepare_sdxl):
//!   latent          [1, 4, H/8, W/8]  BF16
//!   text_embedding  [1, 77, 2048]      BF16   (concat CLIP-L + CLIP-G hiddens)
//!   pooled          [1, 2816]          BF16   (concat CLIP-G pool + size_ids embed)

use clap::Parser;
use flame_core::{adam::AdamW, autograd::AutogradContext, DType, Shape, Tensor};
use eridiffusion_core::config::{LrScheduler, TrainConfig, TrainingMethod};
use eridiffusion_core::lycoris::{LycorisAlgo, LycorisBundleConfig};
use eridiffusion_core::models::{sdxl::SDXLModel, TrainableModel};
use eridiffusion_core::sampler::sdxl_sampler::sin_embed_256;
use eridiffusion_core::training::board::BoardWriter;
use eridiffusion_core::training::checkpoint::{self, CkptHeader};
use eridiffusion_core::training::ema::ParameterEma;
use eridiffusion_core::training::features::ema_advanced::EmaConfig;
use eridiffusion_core::training::features::{loss_weight, lr_schedule, noise_modifiers, timestep_bias, validation::ValidationLoop};
use eridiffusion_core::training::training_features::OptimizerKind;
use std::path::PathBuf;

const NUM_TRAIN_TIMESTEPS: usize = 1000;
const BETA_START: f64 = 0.00085;
const BETA_END: f64 = 0.012;
const SEED: u64 = 42;
const CLIP_GRAD_NORM: f32 = 1.0;

#[derive(Parser)]
struct Args {
    /// Optional OT-format JSON config; otherwise TrainConfig::default().
    #[arg(long)] config: Option<PathBuf>,
    #[arg(long)] cache_dir: PathBuf,
    /// SDXL UNet checkpoint (single safetensors file or directory of shards).
    #[arg(long)] unet: PathBuf,
    #[arg(long, default_value = "100")] steps: usize,
    #[arg(long, default_value = "16")] rank: usize,
    #[arg(long, default_value = "1.0")] lora_alpha: f64,
    /// Preset learning rate (3e-4).
    #[arg(long, default_value = "3e-4")] lr: f32,
    /// Save a LoRA checkpoint every N steps (0 = end-only).
    #[arg(long, default_value = "0")] save_every: usize,
    /// Resume LoRA weights only (no optimizer state).
    #[arg(long, conflicts_with = "resume_full")] resume_lora: Option<PathBuf>,
    /// Full resume: LoRA weights + AdamW (m, v, t) + step counter.
    #[arg(long, conflicts_with = "resume_lora")] resume_full: Option<PathBuf>,
    /// Periodic + final save mode. Default `full` (LoRA + AdamW + step) for
    /// resumable runs. `weights` writes legacy weights-only files.
    #[arg(long, default_value = "full")] save_mode: String,
    #[arg(long, default_value = "output")] output_dir: PathBuf,

    // ── Phase 0 multi-feature rollout (default-off; Phase 1+ will consume) ──
    #[arg(long)] min_snr_gamma: Option<f32>,
    #[arg(long, default_value_t = 0.0)] caption_dropout_probability: f32,
    /// Path to a single cache file produced by `prepare_sdxl` from an empty-
    /// caption sample. When `--caption-dropout-probability > 0`, the trainer
    /// loads `text_embedding` + `pooled` from this file and swaps them in
    /// (correlated, before pooled is concatenated with size embeds) with
    /// probability `p` per step. If unset and dropout > 0, the feature is
    /// disabled with a warning.
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
    /// Zero-terminal SNR rescale of `ᾱ` (Lin et al. 2024). Default off → byte-
    /// identical. Pair with `force_v_prediction=true` (config) — terminal
    /// ᾱ=0 makes ε-prediction degenerate at the last step.
    #[arg(long, default_value_t = false)] zero_terminal_snr: bool,

    // ── LyCORIS algo selection (Phase 2b) ──
    //
    // `--algo lora` (default) keeps the legacy `LoRALinear` path — byte-
    // identical training to pre-Phase-2b. Other values select LyCORIS algos
    // via `SDXLLoraBundle::new_with_config`. `lora_alpha` and `rank` are
    // shared with the legacy CLI flags above.
    //
    // SDXL note: `enumerate_lora_targets()` emits Linear-only targets (attn
    // q/k/v/o, ff.net.0.proj, ff.net.2, proj_in/proj_out). Conv layers
    // (conv_in, ResBlock convs, downsample/upsample) are NOT covered, so
    // `--use_tucker` has no effect on the current target set — it is
    // forwarded to `build_lycoris_linear` for forward-compat with a future
    // conv-target expansion.
    /// LyCORIS algo: `lora` (default, legacy path) | `locon` | `loha` | `lokr`
    /// | `full` | `oft`. `full` and `oft` build successfully but their
    /// `forward_delta` will error inside SDXL's `base + delta_on_input` call
    /// pattern. Phase 2c will wire a `merge_into_base` path.
    #[arg(long, default_value = "lora")] algo: String,
    /// LoKr Kronecker split factor (ignored for non-LoKr).
    #[arg(long, default_value_t = 16)] lokr_factor: i32,
    /// OFT block size (ignored for non-OFT).
    #[arg(long, default_value_t = 32)] oft_block_size: usize,
    /// OFT Cayley-Neumann series term count (ignored for non-OFT).
    #[arg(long, default_value_t = 5)] oft_neumann_terms: usize,
    /// LoCon / LoHa / LoKr conv variant — Tucker decomposition for non-1×1
    /// kernels. SDXL's current target set is Linear-only so this is
    /// forwarded but inert. Reserved for a future conv-LoRA target expansion.
    #[arg(long, default_value_t = false)] use_tucker: bool,
    /// LoKr only: factorize both W1 *and* W2 (default false: only W2).
    #[arg(long, default_value_t = false)] decompose_both: bool,
    /// Enable DoRA (weight-decomposed LoRA). Applies to LoCon/LoHa/LoKr
    /// (Full inherits, OFT errors).
    ///
    /// Phase 2b limitation: SDXL's bundle ctor doesn't have access to the
    /// loaded base weights at construction time so DoRA's magnitude is
    /// initialized from `||I||_2 = 1` rather than `||W_orig||_2`. The
    /// trainer should still converge but will spend the first few hundred
    /// steps adjusting the magnitude.
    #[arg(long, default_value_t = false)] dora: bool,
    /// DoRA magnitude axis. Default `true` matches lycoris-upstream
    /// (norm over input dims, magnitude shape `[out, 1]`).
    #[arg(long, default_value_t = true)] dora_wd_on_out: bool,
    #[arg(long, default_value_t = 1e-6)] dora_eps: f32,
    /// SimpleTuner-parity: perturbed-normal LoKr init.  Scale `>0` triggers
    /// `lokr_w1=1, lokr_w2 ~ N(μ_W, σ_W)·scale`.  No-op when algo != lokr or
    /// value is 0.0. Phase 2b: SDXL's resident `weights` map IS available at
    /// bundle-construction time (no streaming for SDXL UNet) so this can be
    /// wired; current impl logs a TODO and no-ops until the per-prefix
    /// base-weight lookup helper lands.
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

/// Pre-compute `ᾱ_t` table — scaled-linear DDPM schedule (β_start=0.00085,
/// β_end=0.012). Same values as the sampler; duplicated here to keep the
/// trainer self-contained (no autograd path, no shared state with sampler).
fn compute_alpha_bar() -> Vec<f32> {
    let sqrt_start = BETA_START.sqrt();
    let sqrt_end = BETA_END.sqrt();
    let mut ab = Vec::with_capacity(NUM_TRAIN_TIMESTEPS);
    let mut cum = 1.0f32;
    for i in 0..NUM_TRAIN_TIMESTEPS {
        let t = i as f64 / (NUM_TRAIN_TIMESTEPS as f64 - 1.0);
        let sqrt_beta = sqrt_start + t * (sqrt_end - sqrt_start);
        cum *= 1.0 - (sqrt_beta * sqrt_beta) as f32;
        ab.push(cum);
    }
    ab
}

/// Lin et al. 2024 zero-terminal SNR rescale of `sqrt(ᾱ)`. Linearly shifts
/// + scales so that the terminal value is exactly zero while preserving
/// `sqrt_ab[0]`. Idempotent if already zero-terminal. Returns the rescaled
/// `ᾱ` table.
fn rescale_zero_terminal_snr(ab: &[f32]) -> Vec<f32> {
    let n = ab.len();
    let mut sa: Vec<f64> = ab.iter().map(|x| (*x as f64).sqrt()).collect();
    let sa0 = sa[0];
    let sa_t = sa[n - 1];
    if sa_t <= 0.0 || sa0 <= sa_t { return ab.to_vec(); }
    let scale = sa0 / (sa0 - sa_t);
    for v in &mut sa {
        *v = (*v - sa_t) * scale;
    }
    sa.iter().map(|x| (x * x) as f32).collect()
}

fn main() -> anyhow::Result<()> {
    use rand::{Rng, SeedableRng};
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
    // SDXL audit HIGH-3: warn if user set a noise-offset Bernoulli gate.
    // OT has no such gate (offset noise applies every step when weight > 0).
    // The CLI flag is currently ignored to preserve OT parity; this warning
    // makes the divergence loud rather than silent.
    if (args.noise_offset_probability - 1.0).abs() > 1e-6 {
        log::warn!(
            "--noise-offset-probability={:.3} is IGNORED (OT-parity = 1.0). \
            OneTrainer applies offset noise every step when offset_noise_weight > 0; \
            no Bernoulli gate. Flag retained for forward compat only.",
            args.noise_offset_probability
        );
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

    // Phase 2b: parse the LyCORIS algo selector. `lora` (default) keeps the
    // legacy `LoRALinear` bundle constructed inside `SDXLModel::load`.
    // Anything else swaps the bundle in-place after model construction so we
    // don't have to re-plumb the per-trainer constructor signatures.
    //
    // NOTE: `LycorisAlgo::parse("lora")` aliases to `LycorisAlgo::LoCon`
    // (since LoCon-Linear is the canonical LoRA decomposition). For SDXL we
    // need to distinguish LEGACY plain `LoRALinear` (byte-identical) from the
    // new `LycorisAdapter::LoCon` path, so re-map `"lora"` → `None` here
    // explicitly. Users who want the new LoCon path pass `--algo locon`.
    let algo_str = args.algo.trim().to_ascii_lowercase();
    let algo = if algo_str == "lora" || algo_str == "none" || algo_str.is_empty() {
        LycorisAlgo::None
    } else {
        LycorisAlgo::parse(&args.algo).map_err(|e| anyhow::anyhow!("--algo: {e}"))?
    };
    // SDXL trainer rule: BF16/F32 throughout (no FP8 / no AdamW8bit). The
    // default `LycorisBundleConfig::default()` storage (F32) matches that.
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

    let shards = collect_shards(&args.unet)?;
    log::info!("[SDXL] loading UNet from {} shard(s) (rank={}, alpha={})",
        shards.len(), args.rank, args.lora_alpha);
    let mut model = SDXLModel::load(&shards, &config, device.clone())?;

    // If a LyCORIS algo other than the legacy plain LoRA was requested, swap
    // the bundle. Plain `--algo lora` (or `lora`/`none`) keeps the legacy
    // bundle as-is so this branch is byte-equivalent to the pre-Phase-2b
    // pipeline.
    if algo != LycorisAlgo::None && config.is_lora() {
        log::info!(
            "[SDXL] LyCORIS algo='{}' rank={} alpha={} factor={} block_size={} dora={}",
            algo.as_str(),
            lyc_config.rank,
            lyc_config.alpha,
            lyc_config.factor,
            lyc_config.block_size,
            lyc_config.dora,
        );
        if matches!(algo, LycorisAlgo::Full | LycorisAlgo::Oft) {
            log::warn!(
                "[SDXL] algo='{}' selected — bundle construction will succeed, but \
                 forward_delta will error inside SDXL's `base + delta_on_input` call \
                 pattern. Phase 2c will wire merge-into-base for these algos.",
                algo.as_str()
            );
        }
        if args.lora_dropout > 0.0 || args.rank_dropout > 0.0
            || args.module_dropout > 0.0 || args.rank_dropout_scale
        {
            log::warn!(
                "[SDXL] dropout flags (lora_dropout={}, rank_dropout={}, module_dropout={}, \
                 rank_dropout_scale={}) plumbed but not yet wired through the LycorisLinear \
                 forward path; defaults stay byte-identical.",
                args.lora_dropout, args.rank_dropout, args.module_dropout, args.rank_dropout_scale
            );
        }
        model.swap_to_lycoris_bundle(&lyc_config, device.clone(), SEED)
            .map_err(|e| anyhow::anyhow!("LyCORIS bundle construction: {e}"))?;
        if matches!(algo, LycorisAlgo::LoKr) && args.init_lokr_norm > 0.0 {
            // SDXL's `weights` map is resident — wired end-to-end.  Walks
            // `lycoris_adapters` and dispatches `init_perturbed_normal_lokr`
            // per-adapter via the AdapterModule trait.
            let skipped = model
                .apply_init_perturbed_normal(args.init_lokr_norm)
                .map_err(|e| anyhow::anyhow!("init_lokr_norm: {e}"))?;
            log::info!(
                "[SDXL] --init_lokr_norm={} applied (skipped={skipped} non-LoKr/factored)",
                args.init_lokr_norm
            );
        }
    } else if algo == LycorisAlgo::None {
        log::info!("[SDXL] algo='lora' (legacy LoRALinear path, byte-identical)");
    }

    let params = model.parameters();
    log::info!("trainable LoRA tensors: {}", params.len());
    if params.is_empty() {
        anyhow::bail!("no trainable parameters — TrainingMethod::Lora produced empty list");
    }

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

    // SDXL audit HIGH-4: caption-dropout uses the OT-parity ZERO-MULTIPLY
    // path (TE hidden × 0, pooled × 0). The legacy `--null-text-cache` swap
    // is no longer required; we still load it here when provided so the
    // file is validated at startup, and so a future
    // `--caption-dropout-mode null_cache` flag can pick it up without a
    // re-launch.
    let effective_caption_dropout_prob = args.caption_dropout_probability;
    let null_text: Option<(Tensor, Tensor)> = if effective_caption_dropout_prob > 0.0 {
        log::info!(
            "[caption-dropout] WIRED — prob={:.3} mode=zero-multiply (OT-parity, \
            `StableDiffusionXLModel.py:273-284`). TE1+TE2 share one Bernoulli mask \
            (MEDIUM divergence vs OT's independent draws — see source).",
            effective_caption_dropout_prob
        );
        match args.null_text_cache.as_ref() {
            Some(p) => match flame_core::serialization::load_file(p, &device) {
                Ok(s) => {
                    let nt = s.get("text_embedding")
                        .ok_or_else(|| anyhow::anyhow!("--null-text-cache missing 'text_embedding'"))?
                        .to_dtype(DType::BF16)?;
                    let np = s.get("pooled")
                        .ok_or_else(|| anyhow::anyhow!("--null-text-cache missing 'pooled'"))?
                        .to_dtype(DType::BF16)?;
                    log::info!(
                        "[caption-dropout] --null-text-cache loaded ({:?} / {:?}) but \
                        UNUSED in zero-multiply mode; retained for forward-compat with \
                        a future --caption-dropout-mode flag.",
                        nt.shape().dims(),
                        np.shape().dims()
                    );
                    Some((nt, np))
                }
                Err(e) => {
                    log::warn!("[caption-dropout] failed to load --null-text-cache {}: {e} — \
                        proceeding with zero-multiply (no cache needed for OT parity)",
                        p.display());
                    None
                }
            },
            None => None,
        }
    } else {
        if args.null_text_cache.is_some() {
            log::warn!(
                "[caption-dropout] --null-text-cache provided but \
                --caption-dropout-probability == 0.0; cache will not be used. \
                Set --caption-dropout-probability > 0 to enable dropout."
            );
        }
        None
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

    // Cache discovery
    let mut cache_files: Vec<PathBuf> = std::fs::read_dir(&args.cache_dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map_or(false, |e| e == "safetensors"))
        .collect();
    cache_files.sort();
    if cache_files.is_empty() {
        anyhow::bail!("no cached samples in {:?}", args.cache_dir);
    }
    log::info!("Found {} cached samples", cache_files.len());

    let alpha_bar = if args.zero_terminal_snr {
        let rescaled = rescale_zero_terminal_snr(&compute_alpha_bar());
        log::info!(
            "[zero-terminal-snr] WIRED — ᾱ[0]={:.6} ᾱ[T-1]={:.6e} (rescaled to terminal=0)",
            rescaled[0], rescaled[NUM_TRAIN_TIMESTEPS - 1]
        );
        if !config.force_v_prediction {
            log::warn!("[zero-terminal-snr] running with ε-prediction — Lin et al. recommend pairing with v-prediction");
        }
        rescaled
    } else {
        compute_alpha_bar()
    };
    let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);

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
            .ok_or_else(|| anyhow::anyhow!("missing 'latent' in {:?}", cache_files[cache_idx]))?
            .to_dtype(DType::BF16)?;
        let text_embedding = sample.get("text_embedding")
            .ok_or_else(|| anyhow::anyhow!("missing 'text_embedding'"))?
            .to_dtype(DType::BF16)?;
        let pooled_clip_g = sample.get("pooled")
            .ok_or_else(|| anyhow::anyhow!("missing 'pooled'"))?
            .to_dtype(DType::BF16)?;
        // Caption dropout — SDXL audit HIGH-4 (OT parity).
        //
        // OT (`StableDiffusionXLModel.py:273-284`) applies a per-example
        // ZERO-MULTIPLY to the TE hidden states and the pooled CLIP-G
        // output, NOT a swap to the empty-prompt encoding. With batch=1
        // hardcoded in this trainer the per-example mask collapses to a
        // single scalar 0.0 / 1.0 per step.
        //
        // Divergence note (MEDIUM): OT draws INDEPENDENT Bernoullis for
        // TE1 and TE2 (`text_encoder.dropout_probability` and
        // `text_encoder_2.dropout_probability` are separate config fields).
        // EDv2 caches `text_embedding` as the concat of TE1+TE2 hidden
        // states, so we use a single shared mask. If independent dropout
        // is needed, the cache must split TE1/TE2 first.
        //
        // The legacy `--null-text-cache` swap path (kept as a fallback) is
        // available when `--caption-dropout-mode null_cache` is requested
        // by future CLI work; for now the zero-multiply is unconditional
        // when `--caption-dropout-probability > 0`.
        let (text_embedding, pooled_clip_g) = if effective_caption_dropout_prob > 0.0 {
            use rand::Rng;
            if rng.r#gen::<f32>() < effective_caption_dropout_prob {
                // Zero-multiply both TE hidden and pooled CLIP-G. Matches
                // OT line 280-281: `text_encoder_*_output * 0`.
                (text_embedding.mul_scalar(0.0)?,
                 pooled_clip_g.mul_scalar(0.0)?)
            } else {
                (text_embedding, pooled_clip_g)
            }
        } else {
            (text_embedding, pooled_clip_g)
        };
        // `null_text` is loaded but unused under the OT-parity zero-multiply
        // path. Bind to `_` to keep the load alive (validates user's cache
        // file at startup) without flagging unused-variable.
        let _ = &null_text;
        // SDXL audit H2: per-sample `add_time_ids` is stored raw in the
        // cache; we rebuild the sinusoidal embedding here so bucketed
        // datasets see the right per-image conditioning. Pre-baking at
        // prepare time would freeze every sample to one resolution.
        let time_ids_t = sample.get("time_ids")
            .ok_or_else(|| anyhow::anyhow!("missing 'time_ids' (re-run prepare_sdxl with the H2 fix)"))?
            .to_dtype(DType::F32)?;
        let time_ids_v = time_ids_t.to_vec()?;
        if time_ids_v.len() != 6 {
            anyhow::bail!("expected time_ids length 6, got {}", time_ids_v.len());
        }
        let mut size_emb = Vec::with_capacity(6 * 256);
        for &v in &time_ids_v { size_emb.extend_from_slice(&sin_embed_256(v)); }
        let size_t = Tensor::from_vec(
            size_emb,
            Shape::from_dims(&[1, 1536]),
            device.clone(),
        )?.to_dtype(DType::BF16)?;
        // ADM input `y` = concat(CLIP-G pool [1280], size_emb [1536]) → [1, 2816]
        let pooled = Tensor::cat(&[&pooled_clip_g, &size_t], 1)?.to_dtype(DType::BF16)?;

        // Uniform integer timestep in [0, 1000)
        let raw_t = rng.gen_range(0..NUM_TRAIN_TIMESTEPS) as f32;
        // Default-off: Strategy::None → returns raw_t unchanged.
        let t_continuous = timestep_bias::apply_bias(
            raw_t,
            NUM_TRAIN_TIMESTEPS as f32,
            &timestep_bias_cfg,
        );
        let t_idx = (t_continuous.floor() as usize).min(NUM_TRAIN_TIMESTEPS - 1);
        let ab = alpha_bar[t_idx];
        let sqrt_ab = ab.sqrt();
        let sqrt_1m_ab = (1.0 - ab).sqrt();

        // ε ~ N(0, I) at latent shape
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
        // clean noise distribution; input perturbation feeds model input only,
        // target keeps the unperturbed noise.
        //
        // SDXL audit HIGH-3: OT applies offset noise on EVERY step when
        // `offset_noise_weight > 0` — there is NO Bernoulli gate
        // (`ModelSetupNoiseMixin.py:92-108`). To match OT, force the
        // probability to 1.0 here. The `--noise-offset-probability` CLI flag
        // is preserved as a knob; if the user explicitly chose < 1.0 we
        // warn once at startup that this diverges from OT behaviour.
        let _ = args.noise_offset_probability; // documented divergence; OT-parity uses 1.0
        let clean_noise = noise_modifiers::maybe_apply_offset_noise(
            &noise,
            config.offset_noise_weight as f32,
            1.0,
            &mut rng,
        )?;
        let perturbed_noise = noise_modifiers::maybe_apply_input_perturbation(
            &clean_noise,
            args.gamma_input_perturbation,
            &mut rng,
        )?;
        let noisy = latent.mul_scalar(sqrt_ab)?.add(&perturbed_noise.mul_scalar(sqrt_1m_ab)?)?;
        // Phase 1: force_v_prediction. ε-pred default → target = noise.
        // v-pred: target = sqrt(ᾱ_t)·noise - sqrt(1-ᾱ_t)·latent.
        let target = if config.force_v_prediction {
            clean_noise.mul_scalar(sqrt_ab)?
                .sub(&latent.mul_scalar(sqrt_1m_ab)?)?
        } else {
            clean_noise.clone()
        };

        let timestep = Tensor::from_vec(
            vec![t_idx as f32], Shape::from_dims(&[1]), device.clone(),
        )?;

        if step == 0 {
            log::info!("step 0 | latent={:?} txt={:?} pooled={:?} t_idx={} ᾱ={:.4}",
                latent.shape().dims(), text_embedding.shape().dims(),
                pooled.shape().dims(), t_idx, ab);
        }

        let pred = <SDXLModel as TrainableModel>::forward(
            &mut model, &noisy, &timestep,
            std::slice::from_ref(&text_embedding), Some(&pooled),
        )?;

        if pred.shape().dims() != target.shape().dims() {
            anyhow::bail!("pred shape {:?} != target {:?}",
                pred.shape().dims(), target.shape().dims());
        }

        // F32 MSE loss (sd-scripts default mse_strength=1.0, no min-SNR by default)
        // Phase 1: combined loss + per-step weighting. Default-off invariant.
        // SDXL is ε-prediction (force_v_prediction picks v-pred SNR weighting).
        let pred_f32 = pred.to_dtype(DType::F32)?;
        let target_f32 = target.to_dtype(DType::F32)?;
        let raw_loss = loss_weight::combined_loss(
            &pred_f32,
            &target_f32,
            config.mse_strength as f32,
            config.mae_strength as f32,
            args.huber_strength,
        )?;
        // SDXL SNR = ᾱ / (1 - ᾱ) (DDPM-style; not the flow-matching form).
        let snr_ddpm = ab.max(1e-8) / (1.0 - ab).max(1e-8);
        let loss = loss_weight::apply_loss_weight_from_snr(
            &raw_loss,
            snr_ddpm,
            config.loss_weight_fn,
            args.min_snr_gamma,
            config.force_v_prediction,
        )?;
        let loss_val = loss.to_vec()?[0];
        total_loss += loss_val;
        if !loss_val.is_finite() {
            anyhow::bail!("step {step}: non-finite loss {loss_val}");
        }

        let grads = loss.backward()?;

        // clip_grad_norm = 1.0 (OT default)
        // Fusion Sprint Phase 5: device-resident global L2 norm — one D2H per step.
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
            model.post_optimizer_step();
            if let Some(ref mut e) = ema {
                e.update_with_schedule(&params, &ema_cfg, (step + 1) as u64)
                    .map_err(|err| anyhow::anyhow!("EMA update failed at step {}: {err}", step + 1))?;
            }
        }
        AutogradContext::clear();

        let _ = total_loss;
        eridiffusion_core::training::progress::log_step(
            step, args.steps, cache_files.len(), 1,
            loss_val, total_norm, cur_lr, t_start, board.as_ref(),
        );

        // Phase 2: validation eval pass (no_grad) every `validation_every_steps`.
        // SDXL is DDPM ε-prediction (NOT flow-matching). Mirrors training step's
        // schedule under no_grad with a SIDE-RNG seeded from SEED ^ (step+1) so
        // it never perturbs the training-side seeded sequence (byte invariance).
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
                        Some(t) => match t.to_dtype(DType::BF16) {
                            Ok(x) => x,
                            Err(e) => { log::warn!("[validation] latent dtype: {e}"); continue; }
                        },
                        None => {
                            log::warn!("[validation] {} missing latent", vfile.display());
                            continue;
                        }
                    };
                    let v_txt = match sample.get("text_embedding") {
                        Some(t) => match t.to_dtype(DType::BF16) {
                            Ok(x) => x,
                            Err(e) => { log::warn!("[validation] text_embedding dtype: {e}"); continue; }
                        },
                        None => {
                            log::warn!("[validation] {} missing text_embedding", vfile.display());
                            continue;
                        }
                    };
                    let v_pooled_clip_g = match sample.get("pooled") {
                        Some(t) => match t.to_dtype(DType::BF16) {
                            Ok(x) => x,
                            Err(e) => { log::warn!("[validation] pooled dtype: {e}"); continue; }
                        },
                        None => {
                            log::warn!("[validation] {} missing pooled", vfile.display());
                            continue;
                        }
                    };
                    // SDXL audit H2: rebuild sinusoidal `add_time_ids` embedding
                    // exactly as the training step does (bucketed-resolution-aware).
                    let v_time_ids_t = match sample.get("time_ids") {
                        Some(t) => match t.to_dtype(DType::F32) {
                            Ok(x) => x,
                            Err(e) => { log::warn!("[validation] time_ids dtype: {e}"); continue; }
                        },
                        None => {
                            log::warn!("[validation] {} missing time_ids", vfile.display());
                            continue;
                        }
                    };
                    let v_time_ids_v = match v_time_ids_t.to_vec() {
                        Ok(v) => v,
                        Err(e) => { log::warn!("[validation] time_ids to_vec: {e}"); continue; }
                    };
                    if v_time_ids_v.len() != 6 {
                        log::warn!("[validation] {} time_ids len {} != 6", vfile.display(), v_time_ids_v.len());
                        continue;
                    }
                    let mut v_size_emb = Vec::with_capacity(6 * 256);
                    for &v in &v_time_ids_v { v_size_emb.extend_from_slice(&sin_embed_256(v)); }
                    let v_size_t = match Tensor::from_vec(
                        v_size_emb,
                        Shape::from_dims(&[1, 1536]),
                        device.clone(),
                    ).and_then(|t| t.to_dtype(DType::BF16)) {
                        Ok(x) => x,
                        Err(e) => { log::warn!("[validation] size_t build: {e}"); continue; }
                    };
                    let v_pooled = match Tensor::cat(&[&v_pooled_clip_g, &v_size_t], 1)
                        .and_then(|t| t.to_dtype(DType::BF16))
                    {
                        Ok(x) => x,
                        Err(e) => { log::warn!("[validation] pooled cat: {e}"); continue; }
                    };

                    // Side-RNG: never touches training rng. Mirrors training
                    // step's uniform integer timestep + timestep_bias dispatch.
                    let mut vrng = rand::rngs::StdRng::seed_from_u64(SEED ^ (step as u64 + 1));
                    let raw_t = vrng.gen_range(0..NUM_TRAIN_TIMESTEPS) as f32;
                    let t_continuous = timestep_bias::apply_bias(
                        raw_t,
                        NUM_TRAIN_TIMESTEPS as f32,
                        &timestep_bias_cfg,
                    );
                    let t_idx = (t_continuous.floor() as usize).min(NUM_TRAIN_TIMESTEPS - 1);
                    let ab = alpha_bar[t_idx];
                    let sqrt_ab = ab.sqrt();
                    let sqrt_1m_ab = (1.0 - ab).sqrt();

                    let v_noise = match Tensor::randn(v_lat.shape().clone(), 0.0, 1.0, device.clone())
                        .and_then(|t| t.to_dtype(DType::BF16))
                    {
                        Ok(x) => x,
                        Err(e) => { log::warn!("[validation] noise: {e}"); continue; }
                    };
                    let v_noisy = match v_lat.mul_scalar(sqrt_ab)
                        .and_then(|a| v_noise.mul_scalar(sqrt_1m_ab).and_then(|b| a.add(&b)))
                    {
                        Ok(x) => x,
                        Err(e) => { log::warn!("[validation] noisy: {e}"); continue; }
                    };
                    // SDXL ε-prediction default; force_v_prediction → v-target.
                    let v_target = if config.force_v_prediction {
                        match v_noise.mul_scalar(sqrt_ab)
                            .and_then(|a| v_lat.mul_scalar(sqrt_1m_ab).and_then(|b| a.sub(&b)))
                        {
                            Ok(x) => x,
                            Err(e) => { log::warn!("[validation] v-target: {e}"); continue; }
                        }
                    } else {
                        v_noise.clone()
                    };

                    let v_timestep = match Tensor::from_vec(
                        vec![t_idx as f32], Shape::from_dims(&[1]), device.clone(),
                    ) {
                        Ok(x) => x,
                        Err(e) => { log::warn!("[validation] timestep: {e}"); continue; }
                    };

                    let v_pred = match <SDXLModel as TrainableModel>::forward(
                        &mut model, &v_noisy, &v_timestep,
                        std::slice::from_ref(&v_txt), Some(&v_pooled),
                    ) {
                        Ok(p) => p,
                        Err(e) => {
                            log::warn!("[validation] forward failed: {e}");
                            continue;
                        }
                    };
                    let v_loss = match v_pred.to_dtype(DType::F32)
                        .and_then(|p| v_target.to_dtype(DType::F32).and_then(|t| p.sub(&t)))
                        .and_then(|d| d.square())
                        .and_then(|s| s.mean())
                    {
                        Ok(l) => l,
                        Err(e) => { log::warn!("[validation] loss: {e}"); continue; }
                    };
                    let v_loss_val = match v_loss.to_vec() {
                        Ok(v) => v[0],
                        Err(e) => { log::warn!("[validation] loss to_vec: {e}"); continue; }
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

        let step_num = step + 1;
        let save_now = args.save_every > 0 && step_num % args.save_every == 0 && step_num < args.steps;
        // EMA swap: when --ema --ema-validation-swap, save sees EMA-averaged
        // weights. Backup is restored at the end of this block.
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
            let mid_ckpt = args.output_dir.join(format!("sdxl_lora_step{step_num}.safetensors"));
            if save_mode_full {
                let header = CkptHeader::from_adamw(
                    "train_sdxl", step_num as u64, &opt,
                    args.rank, args.lora_alpha as f32, SEED, String::new(),
                );
                let named = model.named_parameters();
                if let Err(e) = checkpoint::save_full(&mid_ckpt, &named, &opt, &header) {
                    log::warn!("[save step {step_num}] full save failed: {e}");
                }
            } else if let Err(e) = model.save_weights(&mid_ckpt.to_string_lossy()) {
                log::warn!("[save step {step_num}] failed: {e}");
            } else {
                log::info!("[save step {step_num}] {}", mid_ckpt.display());
            }
            // ComfyUI companion: Kohya-format LoRA, weights only, no
            // optimizer state. Failures are non-fatal (training continues).
            let comfy_path = args
                .output_dir
                .join(format!("sdxl_lora_step{step_num}_comfyui.safetensors"));
            if let Err(e) = save_kohya_companion(
                &model, args.lora_alpha as f32, &device, &comfy_path,
            ) {
                log::warn!("[save step {step_num}] kohya companion failed: {e}");
            } else {
                log::info!("[save step {step_num}] kohya: {}", comfy_path.display());
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

    let ckpt = args.output_dir.join(format!("sdxl_lora_{}steps.safetensors", args.steps));
    if save_mode_full {
        let header = CkptHeader::from_adamw(
            "train_sdxl", args.steps as u64, &opt,
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

    // ComfyUI companion at end of training.
    let comfy_final = args
        .output_dir
        .join(format!("sdxl_lora_{}steps_comfyui.safetensors", args.steps));
    if let Err(e) = save_kohya_companion(&model, args.lora_alpha as f32, &device, &comfy_final) {
        log::warn!("kohya companion (final) failed: {e}");
    } else {
        log::info!("Saved ComfyUI/Kohya LoRA to {}", comfy_final.display());
    }
    Ok(())
}

/// Build a Kohya-format LoRA state from `model.named_parameters()` and
/// write it to `path`. The Kohya/ComfyUI loader expects:
///   `lora_unet_<underscored_path>.lora_down.weight`
///   `lora_unet_<underscored_path>.lora_up.weight`
///   `lora_unet_<underscored_path>.alpha`  (F32 scalar = `lora_alpha`)
///
/// Implemented as a port of SimpleTuner PR #2704 — see
/// `eridiffusion_core::training::checkpoint::convert_sdxl_unet_to_kohya`.
fn save_kohya_companion(
    model: &SDXLModel,
    lora_alpha: f32,
    device: &std::sync::Arc<flame_core::CudaDevice>,
    path: &std::path::Path,
) -> anyhow::Result<()> {
    use std::collections::HashMap;
    let named = model.named_parameters();
    let mut state: HashMap<String, flame_core::Tensor> = HashMap::with_capacity(named.len());
    for (name, param) in &named {
        state.insert(name.clone(), param.tensor()?);
    }
    let kohya = checkpoint::convert_sdxl_unet_to_kohya(&state, lora_alpha, device)?;
    flame_core::serialization::save_file(&kohya, path)
        .map_err(|e| anyhow::anyhow!("save_file: {e}"))?;
    Ok(())
}
