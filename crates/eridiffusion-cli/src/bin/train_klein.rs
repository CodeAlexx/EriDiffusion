//! train_klein — Klein 4B/9B LoRA training, mirroring upstream Python BaseFlux2Setup.
//!
//! Pipeline per step (matches OT preset `klein9b_lora_boxjana.json` defaults):
//!   1. Load cached `latent` ([1, 128, h, w] BF16, KleinVaeEncoder.encode output)
//!      and `text_embedding` ([1, 512, joint_dim] BF16).
//!   2. Sample timestep ∈ [0, num_train_timesteps) per LOGIT_NORMAL distribution
//!      with `timestep_shift=1.0` (4B+9B preset default).
//!   3. sigma = (floor(t)+1) / 1000;  noisy = noise·sigma + clean·(1-sigma).
//!   4. Forward → [1, 128, h, w]; target = noise - clean (rectified flow).
//!   5. Loss = mean MSE in F32.  clip_grad_norm = 1.0 (preset default; matches ERNIE).
//!
//! Single seed=42 (memory: feedback_default_seed_42).
//! AdamW(lr=3e-5 by default, beta=0.9/0.999, weight_decay=0.01) — matches Klein 9B preset.

use clap::Parser;
use std::path::PathBuf;
use flame_core::{autograd::AutogradContext, DType, Shape, Tensor};
use eridiffusion_core::config::{TrainConfig, TrainingMethod};
use eridiffusion_core::debug as dbg;
use eridiffusion_core::encoders::qwen3::Qwen3Encoder;
use eridiffusion_core::models::{klein::KleinModel, TrainableModel};
use eridiffusion_core::sampler::klein_sampler;
use eridiffusion_core::training::board::BoardWriter;
use eridiffusion_core::training::checkpoint::{self, CkptHeader};
use eridiffusion_core::training::features::{
    caption_dropout, disk_check, ema_advanced::EmaConfig, loss_weight, masked_loss,
    multi_backend::MultiBackend, noise_modifiers, sample_library::SampleLibrary, timestep_bias,
    tread, validation::ValidationLoop,
};
use eridiffusion_core::training::ema::ParameterEma;
use eridiffusion_core::training::features::health::GpuHealthMonitor;
use eridiffusion_core::training::features::lr_schedule;
use eridiffusion_core::training::features::webhook::WebhookClient;
use eridiffusion_core::training::training_features::{Optimizer, OptimizerKind};
use eridiffusion_core::training::training_features::timestep_dist::{TimestepConfig, TimestepDistribution};
use std::str::FromStr as _;
use std::sync::{Mutex, OnceLock};

// Process-wide cache for the multi-tensor scale metadata buffer. Used only
// when `FLAME_MT_SCALE=1` enables the multi-tensor clip-scale path. See
// EriDiffusion-v2/HANDOFF_2026-05-12_PHASE2_SCALE_FOLLOWUP.md.
static MT_SCALE_CACHE: OnceLock<Mutex<flame_core::ops::multi_tensor::MultiTensorMetaCache>> =
    OnceLock::new();

const NUM_TRAIN_TIMESTEPS: usize = 1000;
const LOGIT_NORMAL_BIAS: f32 = 0.0;
const LOGIT_NORMAL_SCALE: f32 = 1.0;
const TIMESTEP_SHIFT: f32 = 1.0;        // klein preset default
/// Default training seed. Used when `--seed` is not specified. Matches the
/// historical hard-coded constant so default-off byte invariance against
/// pre-flag runs is preserved.
const DEFAULT_SEED: u64 = 42;
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
    /// Pyramid / multi-resolution noise: number of additional resolution
    /// levels to mix into the per-step training noise. `0` (default) is a
    /// no-op — byte-identical to no-multires. Each level `k ∈ 1..=N` adds
    /// `discount^k * bilinear_up(randn(H/2^k, W/2^k))` on top of the base
    /// randn. Kohya / SimpleTuner / OneTrainer all expose this; reasonable
    /// values are `4..10`.
    #[arg(long, default_value_t = 0)] multires_noise_iterations: usize,
    /// Per-level discount factor for `--multires-noise-iterations`. Standard
    /// values: 0.3 (default — OneTrainer convention) or 0.5 (Kohya).
    /// Smaller = subtler. No effect when iterations = 0.
    #[arg(long, default_value_t = 0.3)] multires_noise_discount: f32,
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
    /// Master switch for EMA shadow. When `true` an F32 shadow is built from
    /// the trainable LoRA params at startup, and updated after every
    /// `opt.step` via the diffusers-style power-decay schedule
    /// (see `--ema-inv-gamma`, `--ema-power`, `--ema-min-decay`,
    /// `--ema-update-after-step`, `--ema-max-decay`). Training loss is
    /// byte-identical to `--ema=false` because the shadow is parallel — only
    /// `--ema-validation-swap` makes it visible at sample / checkpoint time.
    /// Adds ~rank·param_count·4 bytes of GPU memory; on Klein 9B at rank=16
    /// that's ~200 MB. Shadow is NOT yet persisted across `--resume-full`
    /// (re-initialises from live params on resume).
    #[arg(long, default_value_t = false)] ema: bool,
    #[arg(long, default_value_t = 1.0)] ema_inv_gamma: f32,
    #[arg(long, default_value_t = 0.6667)] ema_power: f32,
    #[arg(long, default_value_t = 0)] ema_update_after_step: u64,
    #[arg(long, default_value_t = 0.0)] ema_min_decay: f32,
    /// Asymptote of the decay schedule. `update_with_schedule` clamps the
    /// per-step computed decay to `[ema_min_decay, ema_max_decay]`. Standard
    /// values: 0.999 (fast averaging), 0.9999 (default — diffusers EMAModel),
    /// 0.99995 (very slow).
    #[arg(long, default_value_t = 0.9999)] ema_max_decay: f32,
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
    /// Stochastic rounding on the F32 → BF16 store at the end of each fused
    /// AdamW step. Default off → byte-identical to prior commits. When on,
    /// long-horizon BF16-param training accumulates small grads correctly
    /// instead of stalling when the per-step update is below ½·ulp(BF16).
    /// Per-element rounding entropy is derived from the optimizer's step
    /// counter mixed with `(tensor_idx, elem_idx)` — reproducible across
    /// reruns with the same seed and step count.
    #[arg(long, default_value_t = false)] adamw_stochastic_round: bool,

    /// Master seed for the training-side RNG (timestep + caption-dropout +
    /// noise-modifier rng) and for things like the periodic-sample seed
    /// derivative. Default `42` matches the previous hard-coded constant
    /// — runs without `--seed` are byte-identical to pre-flag commits.
    /// To repro a non-default run end-to-end, also pass the same value to
    /// `prepare_klein` (its `--crop-style random` rng seed lives there).
    #[arg(long, default_value_t = DEFAULT_SEED)] seed: u64,

    /// Multi-distribution timestep bias strategy. Reshapes the per-step
    /// timestep distribution after the base sampler. `none` (default) is
    /// byte-identical to no biasing. `later` pulls samples toward the
    /// high-noise end (×`--timestep-bias-multiplier`); `earlier` pulls
    /// toward 0. `range` clamps the entire distribution into
    /// `[--timestep-bias-range-min, --timestep-bias-range-max]` (fractions
    /// of NUM_TRAIN_TIMESTEPS) by linear remap.
    #[arg(long, default_value = "none")] timestep_bias_strategy: String,
    /// Strength for `--timestep-bias-strategy later|earlier`. `0.0` = no
    /// bias, `1.0` = fully collapsed to the target end. Clamped at apply
    /// time. Ignored for `none` and `range`.
    #[arg(long, default_value_t = 0.0)] timestep_bias_multiplier: f32,
    /// Lower bound for `--timestep-bias-strategy range`, fraction of
    /// NUM_TRAIN_TIMESTEPS in `[0, 1]`. Ignored otherwise.
    #[arg(long, default_value_t = 0.0)] timestep_bias_range_min: f32,
    /// Upper bound for `--timestep-bias-strategy range`, fraction in
    /// `[0, 1]`. Ignored otherwise.
    #[arg(long, default_value_t = 1.0)] timestep_bias_range_max: f32,

    /// Timestep distribution. `logit_normal` (default — klein 4B+9B preset),
    /// `uniform`, `sigmoid`, `heavy_tail`, `cos_map`, `inverted_parabola`.
    #[arg(long, default_value = "logit_normal")] timestep_distribution: String,
    /// Distribution-specific weight knob. `logit_normal` uses `scale = weight + 1`
    /// (default 0.0 → scale=1.0 matching the existing klein default).
    #[arg(long, default_value_t = 0.0)] noising_weight: f32,
    /// Distribution-specific bias knob (default 0.0 — klein default).
    #[arg(long, default_value_t = 0.0)] noising_bias: f32,

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

    // ── LyCORIS bundle (Phase 2b — wired through KleinModel) ─────────────
    // Default `--algo lora` → legacy plain-LoRA path (byte-identical to
    // all pre-LyCORIS commits — Klein 5-step regression smoke gates it).
    // Non-`lora` values build a `LycorisBundle`-style adapter store inside
    // `KleinModel::new_inner` that the forward path threads through the
    // dyn-AdapterModule call sites.
    /// LyCORIS algo: lora|locon|loha|lokr|full|oft. Default `lora` →
    /// legacy `LoRALinear` path. `lora` and `none` are aliases for the
    /// legacy path.
    #[arg(long, default_value = "lora")] pub algo: String,
    /// LoKr Kronecker split factor.
    #[arg(long, default_value_t = 16)] pub lokr_factor: i32,
    /// OFT block size — must divide the per-target `min(in, out)` evenly.
    #[arg(long, default_value_t = 32)] pub oft_block_size: usize,
    /// OFT Cayley-Neumann series term count (for the matrix exponential
    /// expansion of the rotation).
    #[arg(long, default_value_t = 5)] pub oft_neumann_terms: usize,
    /// LoCon / LoHa / LoKr conv variant Tucker decomposition. Klein's
    /// LoRA targets are linear-only so this is a no-op for LoRA/LoCon
    /// here, but exposing the flag for parity with the upstream LyCORIS
    /// CLI surface.
    #[arg(long, default_value_t = false)] pub use_tucker: bool,
    /// LoKr-only: factorize both `W1` *and* `W2`. Default factorizes
    /// only `W2` (matches lycoris-upstream default).
    #[arg(long, default_value_t = false)] pub decompose_both: bool,
    /// Enable DoRA (weight-decomposed LoRA). Applies to LoCon / LoHa /
    /// LoKr only — `--algo full` is rejected at construction time and
    /// `--algo oft` is rejected because OFT is multiplicative.
    #[arg(long, default_value_t = false)] pub dora: bool,
    /// DoRA magnitude axis. `true` (default, lycoris-upstream) = norm
    /// over input dims, magnitude `[out, 1]`. `false` (OneTrainer) =
    /// norm over output dim, magnitude `[1, in]`.
    #[arg(long, default_value_t = true)] pub dora_wd_on_out: bool,
    /// SimpleTuner-style perturbed-normal LoKr init magnitude. `0.0`
    /// (default) keeps the canonical zero-W2 init. With factored LoKr
    /// (rank < max(out_k, in_n) / 2), zero-W2_B dead-leafs gradients
    /// under ScheduleFree warmup; a small `1e-3..1e-2` perturbation
    /// breaks the dead-leaf in a base-weight-statistical envelope. No-
    /// op unless `--algo lokr`.
    #[arg(long, default_value_t = 0.0)] pub init_lokr_norm: f32,
    /// SimpleTuner / ai-toolkit `network.conv` — per-LyCORIS rank for
    /// CONV-layer targets (separate from linear `--rank`). `0` (default)
    /// = fall back to linear rank. Inert when no conv targets are wired
    /// in the model bundle (current state on all EDv2 trainers).
    #[arg(long, default_value_t = 0)] conv_rank: usize,
    /// SimpleTuner / ai-toolkit `network.conv_alpha` — alpha for CONV
    /// targets. `0.0` (default) = fall back to linear `--lora-alpha`.
    #[arg(long, default_value_t = 0.0)] conv_alpha: f32,

    // ── Phase 5b: autograd v2 bridge opt-in ────────────────────────────────
    /// Route the backward pass through `AutogradContext::backward_v2`
    /// (`MatchInsertedDtype` policy → BF16 grads end-to-end). Default OFF
    /// preserves v3 byte-equivalence. See train_zimage.rs:269 for full doc.
    #[arg(long, default_value_t = false)] use_autograd_v2: bool,

    // ── Gap 2 (2026-05-13): activation offload opt-in ──────────────────────
    /// Install the global activation-offload pool. When set, klein.rs's
    /// `checkpoint_offload` saves block sub-tape activations into pinned RAM
    /// instead of recomputing at backward. For Klein 9B at 512²/batch=1 the
    /// per-block MLP intermediate is ~100 MB and the pool can't fit them all
    /// (system gracefully falls back to recompute) — the win is at higher
    /// resolution / batch where the model would otherwise OOM. Default OFF.
    #[arg(long, default_value_t = false)] activation_offload: bool,
}

/// LOGIT_NORMAL timestep sample. Returns continuous t in [0, 1000).
///
/// Superseded by the unified `TimestepConfig` dispatch — kept for reference
/// and to make the Box-Muller-vs-Ziggurat divergence visible in diff. The
/// klein training loop now uses `timestep_cfg.sample_one(&mut rng)` then
/// scales by `NUM_TRAIN_TIMESTEPS` and applies the (no-op for default) shift.
#[allow(dead_code)]
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
    // Disable the flame-core CUDA alloc pool before any flame_core call.
    // The pool is initialized via OnceLock on first read of FLAME_ALLOC_POOL,
    // so this MUST be set before env_logger::init() (which itself doesn't
    // touch the pool, but any subsequent flame_core call would lock it in).
    //
    // Why this matters for Klein 9B + --offload: empirical bisect (2026-05-13)
    // showed Klein 9B + --offload hits `CUDA_ERROR_INVALID_VALUE` at step 2's
    // `load_file` when the pool is enabled. Step 0 + 1 succeed; pool state
    // gets corrupted somewhere between backward replay (which exercises the
    // BlockOffloader) and step 2's first cudaMalloc. Disabling the pool
    // (every drop → direct cudaFree) makes step 2+ run cleanly. Same root
    // cause as the `feedback_prepare_bins_pool_off` memory and the
    // HANDOFF_2026-05-08_KLEIN9B_TRAINING.md "FLAME_ALLOC_POOL=0 MANDATORY"
    // note — we now enforce it in-trainer so users don't have to remember.
    //
    // Klein 4B without --offload still works either way (no offloader →
    // no pool corruption), so this only changes behavior for the case
    // that was broken without it.
    if std::env::var_os("FLAME_ALLOC_POOL").is_none() {
        // SAFETY: single-threaded at this point (before main's first action).
        unsafe { std::env::set_var("FLAME_ALLOC_POOL", "0"); }
    }
    env_logger::init();
    let args = Args::parse();
    std::fs::create_dir_all(&args.output_dir)?;

    flame_core::config::set_default_dtype(DType::BF16);
    let device = flame_core::global_cuda_device();

    // ── LyCORIS bundle setup (Phase 2b — wired through KleinModel) ───────
    // Build a `LycorisBundleConfig` from the dedicated `--algo` flag set.
    // `--algo lora|none` resolves to `LycorisAlgo::None` → legacy
    // `LoRALinear` path; everything else builds an adapter store inside
    // `KleinModel::new_inner` and the forward path threads it through.
    let lycoris_cfg = {
        // Default-construct via `from_cli(default)` to inherit `storage =
        // StorageDtype::F32` (which is private to the lycoris-rs crate
        // path). Then override the fields the trainer cares about.
        let mut cfg = eridiffusion_core::lycoris::LycorisBundleConfig::from_cli(
            &eridiffusion_core::lycoris::LycorisCliArgs::default(),
        )?;
        cfg.algo = eridiffusion_core::lycoris::LycorisAlgo::parse(&args.algo)?;
        cfg.rank = args.rank;
        cfg.alpha = args.lora_alpha as f32;
        cfg.factor = args.lokr_factor;
        cfg.conv_rank = args.conv_rank;
        cfg.conv_alpha = args.conv_alpha;
        cfg.block_size = args.oft_block_size;
        cfg.neumann_terms = args.oft_neumann_terms;
        cfg.use_tucker = args.use_tucker;
        cfg.decompose_both = args.decompose_both;
        cfg.use_scalar = false;
        cfg.dora = args.dora;
        cfg.dora_wd_on_out = args.dora_wd_on_out;
        cfg.dora_eps = 1e-6;
        cfg
    };
    // `LycorisAlgo::parse("lora")` returns `LoCon`; the spec wants
    // `--algo lora` to take the LEGACY plain-LoRA path. Treat the plain
    // string `"lora"` (case-insensitive) and `"none"` as the legacy
    // sentinel by overriding the parsed algo back to None.
    let algo_lower = args.algo.trim().to_ascii_lowercase();
    let force_legacy = matches!(algo_lower.as_str(), "lora" | "none" | "off" | "");
    let lycoris_cfg = if force_legacy {
        eridiffusion_core::lycoris::LycorisBundleConfig {
            algo: eridiffusion_core::lycoris::LycorisAlgo::None,
            ..lycoris_cfg
        }
    } else {
        lycoris_cfg
    };
    let use_lycoris =
        lycoris_cfg.algo != eridiffusion_core::lycoris::LycorisAlgo::None;
    if use_lycoris {
        log::info!(
            "[lycoris] algo={} rank={} alpha={} dora={} dora_wd_on_out={} factor={} block_size={} neumann={} tucker={} decompose_both={}",
            lycoris_cfg.algo.as_str(),
            lycoris_cfg.rank,
            lycoris_cfg.alpha,
            lycoris_cfg.dora,
            lycoris_cfg.dora_wd_on_out,
            lycoris_cfg.factor,
            lycoris_cfg.block_size,
            lycoris_cfg.neumann_terms,
            lycoris_cfg.use_tucker,
            lycoris_cfg.decompose_both,
        );
    }

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

    // ── Periodic sample setup (must run BEFORE Klein DiT load) ───────────
    // Klein 9B DiT is ~18 GB; Qwen3 8B is ~16 GB; loading both at once on
    // 24 GB OOMs. Encode the sample prompt FIRST, drop Qwen3, then load DiT.
    // (Klein 4B + Qwen3 4B fit together, so this never bit before the 9B run.)
    let periodic = args.sample_every > 0;
    let (sample_cap, sample_uncond, sample_vae_path) = if periodic {
        let qwen3_path = args.sample_qwen3.as_ref()
            .ok_or_else(|| anyhow::anyhow!("--sample-every > 0 requires --sample-qwen3"))?;
        let tok_path = args.sample_tokenizer.as_ref()
            .ok_or_else(|| anyhow::anyhow!("--sample-every > 0 requires --sample-tokenizer"))?;
        let vae_path = args.sample_vae.as_ref()
            .ok_or_else(|| anyhow::anyhow!("--sample-every > 0 requires --sample-vae"))?
            .clone();
        log::info!("[sample-setup] loading Qwen3 + tokenizer to encode prompt once (before DiT load)...");
        let qwen_w = klein_load_qwen3(qwen3_path, &device)?;
        let qcfg = Qwen3Encoder::config_from_weights(&qwen_w)?;
        let qwen = Qwen3Encoder::new(qwen_w, qcfg, device.clone());
        let tok = tokenizers::Tokenizer::from_file(tok_path)
            .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
        let cap = klein_encode_prompt(&qwen, &tok, &args.sample_prompt)?;
        let unc = klein_encode_prompt(&qwen, &tok, &args.sample_neg_prompt)?;
        log::info!("[sample-setup] cap={:?} uncond={:?}", cap.shape().dims(), unc.shape().dims());
        drop(qwen);
        flame_core::cuda_alloc_pool::clear_pool_cache();
        flame_core::trim_cuda_mempool(0);
        log::info!("[sample-setup] Qwen3 dropped; VAE will load lazily per sample. Periodic sample enabled (every {} steps).", args.sample_every);
        (Some(cap), Some(unc), Some(vae_path))
    } else {
        (None, None, None)
    };

    let shards = collect_klein_shards(&args.transformer)?;
    log::info!("Loading Klein transformer from {} shard(s) (rank={} alpha={})",
        shards.len(), args.rank, args.lora_alpha);
    // Phase 2b: when `--algo` is non-lora, pass the LyCORIS config through
    // so the model builds an `AdapterStore` instead of `Vec<LoRALinear>`.
    // For the legacy path we pass `None` to `load_with_lycoris`, which
    // forwards to the byte-identical `KleinModel::load` codepath.
    let mut model = KleinModel::load_with_lycoris(
        &shards,
        &config,
        device.clone(),
        if use_lycoris { Some(lycoris_cfg.clone()) } else { None },
    )?;
    // Phase 2c — perturbed-normal LoKr init. MUST happen BEFORE
    // `enable_offload` because the apply method reads from `self.weights`,
    // which `enable_offload` strips of `double_blocks.*` / `single_blocks.*`.
    if use_lycoris
        && lycoris_cfg.algo == eridiffusion_core::lycoris::LycorisAlgo::LoKr
        && args.init_lokr_norm > 0.0
    {
        let skipped = model
            .apply_init_perturbed_normal(args.init_lokr_norm)
            .map_err(|e| anyhow::anyhow!("init_lokr_norm: {e}"))?;
        if skipped > 0 {
            log::warn!(
                "[klein] init_lokr_norm: {} slot(s) skipped (see warnings above)",
                skipped
            );
        }
    }
    if args.offload {
        model.enable_offload(shards.clone())?;
        log::info!("  block-offload enabled — per-block streaming from {} shard(s)", shards.len());
    }

    // 2026-05-13 Gap 2: install the global activation-offload pool when block
    // offload is on. `checkpoint_offload` in `klein.rs` consults this pool to
    // route saved sub-tape tensors into pinned RAM instead of recomputing the
    // block at backward time. Falls back to plain recompute when the pool
    // isn't installed (autograd.rs:2021-2024) — safe on both paths.
    //
    // Sizing — empirical Klein 9B observation 2026-05-13: the biggest saved
    // activation inside a Klein block is the MLP intermediate at shape
    // `[1, seq, inner*6]` (SwiGLU gate+up concat × ratio). 9B: 1008*24576*2 ≈
    // 47 MB BF16. 4B: scales with inner_dim=3072. First wiring (4096*2048*2 =
    // 16 MB slot) was too small — every save hit "exceeds slot capacity" and
    // fell back to recompute, losing the benefit. Slot now sized for the
    // worst case: `max_seq * inner_dim * 6 * 2`.
    //
    // FP8 compression halves pinned bytes per slot (and roughly doubles
    // effective slot count); per-slot GPU staging is uncompressed BF16 size.
    // slots_per_block=2 keeps GPU staging within budget on 24 GB cards.
    if args.activation_offload {
        use eridiffusion_core::training::offload::{setup_activation_offload, OffloadConfig};
        use flame_core::activation_offload::OffloadCompression;
        let num_blocks = model.kconfig.num_double + model.kconfig.num_single;
        let inner_dim = model.kconfig.inner_dim;
        // Conservative max_seq: 1568 (44×24 image + 512 text) → 2048 padding.
        // Klein 9B per-block MLP intermediate is `[1, seq, inner*9]` BF16 ≈
        // 112 MB raw at seq=1520/inner=4096; observed via warning logs
        // 2026-05-13. We size for inner*9 to fit; FP8 halves pinned cost.
        let max_seq = 2048usize;
        let mlp_factor = 9usize;
        let max_activation_bytes = max_seq * inner_dim * mlp_factor * 2;
        let slots_per_block = 2usize;
        let total_slots = num_blocks * slots_per_block + 8;
        let cfg = OffloadConfig {
            num_blocks,
            max_activation_bytes,
            compression: OffloadCompression::FP8,
            extra_slots: total_slots - num_blocks,
        };
        match setup_activation_offload(&device, &cfg) {
            Ok((slots, bytes)) => log::info!(
                "[activation_offload] {slots} slots, {:.2} GB pinned (FP8), slot={:.1} MB raw",
                bytes as f64 / 1e9,
                max_activation_bytes as f64 / 1e6,
            ),
            Err(e) => log::warn!(
                "[activation_offload] setup failed ({e}); klein.rs:checkpoint_offload \
                 will fall back to recompute (slower but no crash)"
            ),
        }
    }

    let params = model.parameters();
    log::info!("Loaded {} trainable LoRA tensors", params.len());
    if params.is_empty() {
        anyhow::bail!("No trainable parameters — TrainingMethod::Lora produced empty param list");
    }

    // Phase 2b: bundle population now happens INSIDE
    // `KleinModel::load_with_lycoris` above. The model's `lyc_adapters`
    // field carries the per-target adapter list; `forward_inner` branches
    // on `self.lyc_adapters.is_some()` to dispatch through the
    // dyn-AdapterModule path. When `use_lycoris == false` (default), the
    // model takes the legacy `LoRALinear` path with byte-identical
    // training to the pre-LyCORIS commits.
    if use_lycoris {
        let n_adapt = model.lyc_adapters.as_ref().map(|v| v.len()).unwrap_or(0);
        log::info!(
            "[lycoris] adapter bundle live: {} per-target adapters, {} optimizer params",
            n_adapt,
            params.len(),
        );
    }

    // EMA shadow (Phase 3 advanced). Built from current live params (post
    // resume_lora / pre-step-0). Updated after each opt.step via
    // `update_with_schedule`. Optional swap into live params at sample /
    // checkpoint time when --ema-validation-swap is set.
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

    // Phase B (2026-05-10): unified Optimizer enum dispatches all kinds.
    let opt_kind = OptimizerKind::parse(&args.optimizer)
        .map_err(|e| anyhow::anyhow!("--optimizer: {e}"))?;
    log::info!("[Klein] optimizer={}", opt_kind.as_str());
    let mut opt = Optimizer::new(opt_kind, args.lr, 0.9, 0.999, 1e-8, 0.01);
    if let Optimizer::AdamW(ref mut adam) = opt {
        adam.set_stochastic_round(args.adamw_stochastic_round);
    } else if args.adamw_stochastic_round {
        log::warn!(
            "--adamw-stochastic-round only applies to AdamW; ignored for {:?}",
            opt.kind()
        );
    }
    if args.adamw_stochastic_round {
        log::info!(
            "[adamw] stochastic-round enabled — F32→BF16 stores will use lower-16-bit hash-driven rounding (loss curves will diverge from round-to-nearest baseline by tiny per-step noise)"
        );
    }

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

    let mut rng = rand::rngs::StdRng::seed_from_u64(args.seed);

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
        // Phase 3: optional per-pixel masks. Only allocated when masked-loss
        // is active (`masked_loss_weight > 0.0`); otherwise stays empty and
        // the loss path is byte-identical to the prior commit.
        let mut masks: Vec<flame_core::Tensor> = if config.masked_loss_weight > 0.0 {
            Vec::with_capacity(bs)
        } else {
            Vec::new()
        };
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
            if config.masked_loss_weight > 0.0 {
                let m = masked_loss::load_mask(&sample, l.shape(), device.clone())?;
                masks.push(m);
            }
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

        // Phase 1: caption dropout — with prob `p`, swap the conditional
        // caption embedding for the cached unconditional one. When `p == 0.0`
        // (default), this is a noop and consumes no rng.
        let txt = if effective_caption_dropout_prob > 0.0 {
            if let Some(unc) = sample_uncond.as_ref() {
                // Tile uncond to match batch size if needed.
                let uncond_b = if unc.shape().dims()[0] == bs {
                    unc.clone()
                } else {
                    let mut tgt = unc.shape().dims().to_vec();
                    tgt[0] = bs;
                    unc.broadcast_to(&Shape::from_dims(&tgt))?
                };
                caption_dropout::maybe_drop_caption(
                    &txt, &uncond_b, effective_caption_dropout_prob, &mut rng,
                )?
            } else {
                txt
            }
        } else {
            txt
        };

        // Per-batch-element timesteps. upstream Python samples shape [B] (line
        // 99 BaseFlux2Setup.py: `batch_size=batch['latent_image'].shape[0]`).
        let mut t_per_b: Vec<f32> = Vec::with_capacity(bs);
        let mut sigma_per_b: Vec<f32> = Vec::with_capacity(bs);
        let mut t_model_per_b: Vec<f32> = Vec::with_capacity(bs);
        for _ in 0..bs {
            // Sample u in [0,1] via unified dispatcher → scale to [0, NUM_TRAIN_TIMESTEPS).
            // With klein's `TIMESTEP_SHIFT=1.0` the legacy post-shift was a no-op.
            let raw_t = timestep_cfg.sample_one(&mut rng) * NUM_TRAIN_TIMESTEPS as f32;
            // Default-off: Strategy::None → returns raw_t unchanged.
            let t_continuous = timestep_bias::apply_bias(
                raw_t,
                NUM_TRAIN_TIMESTEPS as f32,
                &timestep_bias_cfg,
            );
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
        // Pyramid / multi-resolution noise (additive). Default-off when
        // `multires_noise_iterations == 0`: returns noise.clone() with no rng
        // consumption and no extra alloc → byte-identical to baseline.
        let noise = noise_modifiers::maybe_apply_multires_noise(
            &noise,
            args.multires_noise_iterations,
            args.multires_noise_discount,
            &mut rng,
        )?;
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
        let target = clean_noise.sub(&latent)?;
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
            log::info!("step 0 | batch={} latent={:?} text={:?} sigma[0]={:.4} (idx={})",
                bs, latent.shape().dims(), txt.shape().dims(), sigma, sigma_idx);
        }

        // Build a TreadStep this step iff TREAD is configured AND keep_ratio<1.
        // The single-block stream concatenates [txt, img] → T_total tokens.
        // Klein latent is `[B, in_ch, H_lat, W_lat]` → n_img = H_lat*W_lat.
        let tread_step = if let Some(ref ranges) = tread_ranges {
            let dims = noisy.shape().dims();
            let (h_lat, w_lat) = (dims[2], dims[3]);
            let n_img = h_lat * w_lat;
            let txt_len = txt.shape().dims()[1];
            let t_total = txt_len + n_img;
            // Use the FIRST range; multi-range routing is a Phase 5 follow-up.
            let (lo, hi) = ranges[0];
            Some(tread::TreadStep::new(
                t_total,
                args.tread_keep_ratio,
                (lo, hi),
                &mut rng,
            ))
        } else {
            None
        };

        let pred = model.forward_train(&noisy, &txt, &timestep, tread_step.as_ref())?;
        if pred.shape().dims() != target.shape().dims() {
            anyhow::bail!(
                "predicted velocity shape {:?} != target {:?}",
                pred.shape().dims(), target.shape().dims());
        }

        // F32 mean MSE — matches OT default (loss_weight_fn=CONSTANT, mse_strength=1.0).
        // Phase 1: combined MSE+MAE+Huber loss + per-step loss weighting.
        // Default-off invariance: when mse=1.0, mae=0.0, huber=0.0 AND
        // loss_weight_fn=Constant AND min_snr_gamma=None, this collapses to
        // exactly the previous (pred-target).square().mean() formula.
        let pred_f32 = pred.to_dtype(DType::F32)?;
        let target_f32 = target.to_dtype(DType::F32)?;
        // Phase 3: when masked-loss is active, take the manual diff path so we
        // can multiply the per-element diff by a per-pixel mask BEFORE squaring
        // and reducing. When masked_loss_weight == 0.0 (default) we route
        // through `combined_loss` exactly like Phase 1+2 → byte invariance.
        let raw_loss = if config.masked_loss_weight > 0.0 && !masks.is_empty() {
            let mask_t = if bs == 1 {
                masks.into_iter().next().unwrap()
            } else {
                Tensor::cat(&masks.iter().collect::<Vec<_>>(), 0)?
            };
            // Caller is responsible for square + mean after `apply_loss_mask`.
            // Combined MSE/MAE/Huber strengths are NOT applied on this path —
            // masked-loss currently only supports the MSE-equivalent reduction.
            // mae/huber under masked-loss is a Phase-future enhancement.
            let diff = pred_f32.sub(&target_f32)?;
            let masked_diff =
                masked_loss::apply_loss_mask(&diff, &mask_t, config.masked_loss_weight)?;
            masked_diff.square()?.mean()?
        } else {
            loss_weight::combined_loss(
                &pred_f32,
                &target_f32,
                config.mse_strength as f32,
                config.mae_strength as f32,
                args.huber_strength,
            )?
        };
        // Klein is flow-matching → v-prediction-style SNR weighting.
        let loss = loss_weight::apply_loss_weight(
            &raw_loss,
            sigma,
            config.loss_weight_fn,
            args.min_snr_gamma,
            true,
        )?;
        let loss_val = loss.to_vec()?[0];
        total_loss += loss_val;

        // === OT_DEBUG_STATS-format per-step line (mirrors train_ernie + upstream Python patch) ===
        let dbg_on = dbg::enabled("OT_DEBUG_STATS");
        if dbg_on {
            let p_st = dbg::stats(&pred);
            let t_st = dbg::stats(&target);
            eprintln!(
                "[OT_DEBUG step={:5}] t={:.2} loss(pre-scale)={:.4} | pred[mean={:+.3e} std={:.3e} max|·|={:.3e}] target[mean={:+.3e} std={:.3e} max|·|={:.3e}]",
                step, t_continuous, loss_val,
                p_st.mean, p_st.std, p_st.abs_max,
                t_st.mean, t_st.std, t_st.abs_max,
            );
        }

        // FORWARD-ONLY BENCH MODE: skip backward + optimizer when
        // FLAME_FORWARD_ONLY_BENCH=1. Used only for isolating forward
        // vs backward s/step.
        let forward_only_bench = std::env::var("FLAME_FORWARD_ONLY_BENCH").is_ok();
        if forward_only_bench {
            AutogradContext::clear();
            eridiffusion_core::training::progress::log_step(
                "Klein-fwd-only",
                step, args.steps, dataset_len, args.batch_size.max(1),
                loss_val, 0.0, 0.0, t_start, board.as_ref(),
            );
            continue;
        }

        // Phase 5b: Route (ii) bridge. `--use-autograd-v2` flips the
        // backward entry to construct a `MatchInsertedDtype` GradientMap.
        let mut grads = if args.use_autograd_v2 {
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

        // FLAME_MT_SCALE=1 collapses the per-parameter mul_scalar loop into a
        // single multi-tensor kernel launch when the clip path fires. Default
        // off: klein's grad_norm sits at 0.004–0.17 in production configs and
        // never trips the clip path, so the multi-tensor path adds no value.
        // See EriDiffusion-v2/HANDOFF_2026-05-12_PHASE2_SCALE_FOLLOWUP.md.
        // Phase 5b: under `--use-autograd-v2`, grads are BF16 in the map;
        // the FLAME_MT_SCALE fast path asserts F32 and would bail. Fall
        // back to the per-param mul_scalar loop in that case.
        let mt_scale_enabled = std::env::var("FLAME_MT_SCALE")
            .ok()
            .as_deref()
            .map(|v| matches!(v, "1" | "true" | "TRUE"))
            .unwrap_or(false)
            && !args.use_autograd_v2;

        if mt_scale_enabled && scale < 1.0 {
            let mut ptrs: Vec<u64> = Vec::with_capacity(params.len());
            let mut sizes: Vec<u64> = Vec::with_capacity(params.len());
            let mut device_opt: Option<std::sync::Arc<flame_core::CudaDevice>> = None;
            for param in &params {
                if let Some(g) = grads.get_mut(param.id()) {
                    if g.dtype() != flame_core::DType::F32 {
                        anyhow::bail!(
                            "FLAME_MT_SCALE expects F32 grads (GradientMap policy is F32), got {:?}",
                            g.dtype()
                        );
                    }
                    if device_opt.is_none() {
                        device_opt = Some(g.device().clone());
                    }
                    ptrs.push(g.as_mut_device_ptr_f32("mt_scale:g")?);
                    sizes.push(g.shape().elem_count() as u64);
                }
            }
            let n = ptrs.len();
            if n > 0 {
                let mut packed: Vec<u64> = Vec::with_capacity(2 * n);
                packed.extend(ptrs);
                packed.extend(sizes);
                let device = device_opt.expect("at least one grad present");
                let cache_cell = MT_SCALE_CACHE
                    .get_or_init(|| Mutex::new(flame_core::ops::multi_tensor::MultiTensorMetaCache::new()));
                let mut cache = cache_cell
                    .lock()
                    .map_err(|_| anyhow::anyhow!("MT_SCALE_CACHE mutex poisoned"))?;
                flame_core::ops::multi_tensor::multi_tensor_scale_inplace_packed(
                    &mut cache,
                    &device,
                    n,
                    &packed,
                    scale,
                    /* is_bf16 = */ false,
                )?;
            }
            for param in &params {
                if let Some(g) = grads.get(param.id()) {
                    param.set_grad(g.clone())?;
                }
            }
        } else {
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
            if let Some(ref mut e) = ema {
                // 1-based step → matches the schedule's `update_after_step`
                // semantics (step==update_after_step returns 0 / "skip").
                e.update_with_schedule(&params, &ema_cfg, (step + 1) as u64)
                    .map_err(|err| anyhow::anyhow!("EMA update failed at step {}: {err}", step + 1))?;
            }
        }
        AutogradContext::clear();

        let _ = total_loss;
        eridiffusion_core::training::progress::log_step(
            "Klein-lora",
            step, args.steps, dataset_len, args.batch_size.max(1),
            loss_val, total_norm, cur_lr, t_start, board.as_ref(),
        );

        // Phase 2: validation eval pass (no_grad) every `validation_every_steps`.
        // step+1 because `step` here is 0-based; ValidationLoop::should_run
        // expects the 1-based completed-step number.
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
                    let v_txt = match sample.get("text_embedding") {
                        Some(t) => t.to_dtype(DType::BF16)?,
                        None => {
                            log::warn!("[validation] {} missing text_embedding", vfile.display());
                            continue;
                        }
                    };
                    // Sample timestep + noise identically to training. Validation
                    // uses its OWN run-side RNG so it does not perturb the
                    // training-side seeded sequence (byte invariance).
                    let mut vrng = rand::rngs::StdRng::seed_from_u64(args.seed ^ (step as u64 + 1));
                    let t_continuous = timestep_cfg.sample_one(&mut vrng) * NUM_TRAIN_TIMESTEPS as f32;
                    let sigma_idx = (t_continuous.floor() as usize).min(NUM_TRAIN_TIMESTEPS - 1);
                    let sigma = (sigma_idx + 1) as f32 / NUM_TRAIN_TIMESTEPS as f32;
                    let v_noise = Tensor::randn(v_lat.shape().clone(), 0.0, 1.0, device.clone())?
                        .to_dtype(DType::BF16)?;
                    let v_noisy = v_noise.mul_scalar(sigma)?
                        .add(&v_lat.mul_scalar(1.0 - sigma)?)?;
                    let v_target = v_noise.sub(&v_lat)?;
                    let v_t_model = sigma_idx as f32 / NUM_TRAIN_TIMESTEPS as f32;
                    let v_timestep = Tensor::from_vec(
                        vec![v_t_model],
                        Shape::from_dims(&[v_lat.shape().dims()[0]]),
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
        if periodic && step_num % args.sample_every == 0 && step_num < args.steps {
            // EMA swap: when `--ema --ema-validation-swap`, save and sample
            // see EMA-averaged weights. `backup` returned only in that case;
            // restored at the end of this block. Updates resume against the
            // ORIGINAL live tensors after restore.
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
                    if let Optimizer::AdamW(ref adam) = opt {
                        let header = CkptHeader::from_adamw(
                            "train_klein", step_num as u64, adam,
                            args.rank, args.lora_alpha as f32, args.seed, String::new(),
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
            // Restore live params before the next training step so the
            // optimizer's accumulated moments stay consistent with the
            // tensors they were taken against.
            if let (Some(backup), Some(ref e)) = (ema_backup, &ema) {
                let _g = AutogradContext::no_grad();
                e.restore_swapped(&params, backup)
                    .map_err(|err| anyhow::anyhow!("EMA restore_swapped (mid) failed: {err}"))?;
            }
        }
    }

    let trained = args.steps - start_step;
    let avg_loss = if trained > 0 { total_loss / trained as f32 } else { 0.0 };
    let wall_time = t_start.elapsed().as_secs_f64();
    log::info!("Training complete: {trained} new steps (total={}), avg loss={:.4}", args.steps, avg_loss);
    if let Some(b) = &board { b.set_status("completed"); }

    // Final EMA swap (covers both final save and final sample below). No
    // restore at the very end — process exits, no further training. Skipped
    // when --ema-validation-swap is off or no EMA was constructed.
    if args.ema_validation_swap {
        if let Some(ref e) = ema {
            let _g = AutogradContext::no_grad();
            // Discard the backup: training is over, restore is unnecessary.
            let _ = e.swap_with_live(&params)
                .map_err(|err| anyhow::anyhow!("EMA swap_with_live (final) failed: {err}"))?;
            log::info!("[ema] swapped EMA shadow into live params for final save + sample");
        }
    }

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
            if let Optimizer::AdamW(ref adam) = opt {
                let header = CkptHeader::from_adamw(
                    "train_klein", args.steps as u64, adam,
                    args.rank, args.lora_alpha as f32, args.seed, String::new(),
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
