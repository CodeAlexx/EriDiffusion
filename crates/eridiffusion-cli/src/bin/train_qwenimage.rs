//! train_qwenimage — Qwen-Image-2512 LoRA training binary.
//!
//! Patterned after `train_ernie`. Loads cached `(latent, text_embedding)`
//! samples from `prepare_qwenimage` and runs flow-matching LoRA training.
//!
//! Reference: `flame-diffusion/qwenimage-trainer/src/{main,pipeline}.rs`,
//! `musubi-tuner/qwen_image_train_network.py`.
//!
//! Schedule: logit-normal then qwen_shift remap
//!   `t = sigmoid(z); t_shifted = t * shift / (1 + (shift - 1) * t)`
//!   shift comes from `shift_for_resolution([w, h])` (linear-mu over image
//!   seq-len anchors 256→0.5, 8192→0.9, exp).
//!
//! Loss: MSE in F32 between pred and target = noise - latent.

use clap::Parser;
use flame_core::{autograd::AutogradContext, DType, Shape, Tensor};
use flame_core::adam::AdamW;
use flame_core::gradient_clip::GradientClipper;
use eridiffusion_core::encoders::qwen25vl::Qwen25VLEncoder;
use eridiffusion_core::models::{qwenimage as qwen_model, QwenImageTrainingModel};
use eridiffusion_core::sampler::qwenimage_sampler;
use eridiffusion_core::training::board::BoardWriter;
use eridiffusion_core::training::checkpoint::{self, CkptHeader};
use eridiffusion_core::config::LrScheduler;
use eridiffusion_core::training::features::{
    ema_advanced::EmaConfig, loss_weight, lr_schedule, noise_modifiers, timestep_bias,
};
use eridiffusion_core::training::ema::ParameterEma;
use eridiffusion_core::training::training_features::OptimizerKind;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::path::PathBuf;

const SEED_DEFAULT: u64 = 42;
const QWEN_PAD_ID: i32 = 151643;
/// Qwen-Image canonical prompt template (`pipeline_qwenimage.py::
/// PROMPT_TEMPLATE_ENCODE`). The DiT was trained against text embeddings
/// produced by this template; raw captions = OOD conditioning.
const PROMPT_PREFIX: &str =
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, \
     texture, quantity, text, spatial relationships of the objects and background:\
     <|im_end|>\n<|im_start|>user\n";
const PROMPT_SUFFIX: &str = "<|im_end|>\n<|im_start|>assistant\n";
/// Drop the system-prompt prefix (matches Python
/// `PROMPT_TEMPLATE_ENCODE_START_IDX`).
const DROP_IDX: usize = 34;

#[derive(Parser)]
struct Args {
    /// Qwen-Image-2512 transformer directory (the `transformer/` subdir, with
    /// `diffusion_pytorch_model-0000{N}-of-00009.safetensors` shards).
    #[arg(long)] model: PathBuf,
    /// Cache dir produced by `prepare_qwenimage`.
    #[arg(long)] cache_dir: PathBuf,
    #[arg(long, default_value = "3000")] steps: usize,
    #[arg(long, default_value = "16")] rank: usize,
    #[arg(long, default_value = "16.0")] lora_alpha: f32,
    /// OneTrainer "qwen LoRA 24GB" preset default = 3e-4.
    #[arg(long, default_value = "3e-4")] lr: f32,
    /// Resolution at which the cache was prepared (used for qwen_shift).
    #[arg(long, default_value = "512")] resolution: usize,
    #[arg(long, default_value = "200")] warmup_steps: usize,
    /// Optional fixed shift (overrides resolution-based qwen_shift).
    #[arg(long)] qwen_shift: Option<f32>,
    /// Resume LoRA weights only (no optimizer state).
    #[arg(long, conflicts_with = "resume_full")] resume_lora: Option<PathBuf>,
    /// Full resume: LoRA + AdamW + step.
    #[arg(long, conflicts_with = "resume_lora")] resume_full: Option<PathBuf>,
    /// Save mode: `full` (LoRA + AdamW + step) or `weights` (legacy).
    #[arg(long, default_value = "full")] save_mode: String,
    #[arg(long, default_value = "0")] save_every: usize,
    #[arg(long, default_value = "output")] output_dir: PathBuf,
    #[arg(long, default_value_t = SEED_DEFAULT)] seed: u64,

    // ── Periodic-sample (mirrors train_ernie pattern) ─────────────────────
    /// Render samples every N steps. `0` disables. When > 0: renders a
    /// step-0 baseline (LoRA = identity, base-model output for sanity
    /// check), then every N steps, then a final sample after training.
    /// Two prompts are rendered each time so we see both pose/style
    /// variations as the LoRA imprints. Per-sample cost: ~30s + denoise.
    #[arg(long, default_value = "0")] sample_every: usize,
    /// First prompt — caption-style with the LoRA trigger word.
    #[arg(long)] sample_prompt_1: Option<String>,
    /// Second prompt — different scene/outfit but same trigger word style.
    #[arg(long)] sample_prompt_2: Option<String>,
    /// Negative prompt (shared across both prompts). Empty disables uncond.
    #[arg(long, default_value = "")] sample_neg_prompt: String,
    /// `qwen_image_vae.safetensors` (wan21 internal-key format) for the
    /// in-process VAE decode. Required if --sample-every > 0.
    #[arg(long)] sample_vae: Option<PathBuf>,
    /// Qwen2.5-VL text encoder dir (or single combined safetensors).
    /// Required if --sample-every > 0.
    #[arg(long)] sample_text_encoder: Option<PathBuf>,
    /// `tokenizer.json` for Qwen2.5-VL. Required if --sample-every > 0.
    #[arg(long)] sample_tokenizer: Option<PathBuf>,
    #[arg(long, default_value = "1024")] sample_size: usize,
    /// Inference-flame's qwenimage_gen defaults to 50 steps; lower (20) shows
    /// visible texture artifacts at 1024² even with norm-rescaled CFG.
    #[arg(long, default_value = "50")] sample_steps: usize,
    #[arg(long, default_value = "4.0")] sample_cfg: f32,
    #[arg(long, default_value = "42")] sample_seed: u64,
    #[arg(long, default_value_t = 512)] sample_max_text_len: usize,

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
    /// Master switch for EMA shadow. When `true` an F32 shadow is built from
    /// current live params (post resume_lora / pre-step-0) and updated after
    /// each opt.step via `update_with_schedule` per the EmaConfig schedule
    /// (see `--ema-inv-gamma`, `--ema-power`, `--ema-min-decay`,
    /// `--ema-update-after-step`, `--ema-max-decay`). Training loss is
    /// byte-identical to `--ema=false` because the shadow is parallel — only
    /// `--ema-validation-swap` makes it visible at sample / checkpoint time.
    #[arg(long, default_value_t = false)] ema: bool,
    #[arg(long, default_value_t = 1.0)] ema_inv_gamma: f32,
    #[arg(long, default_value_t = 0.6667)] ema_power: f32,
    #[arg(long, default_value_t = 0)] ema_update_after_step: u64,
    #[arg(long, default_value_t = 0.0)] ema_min_decay: f32,
    /// Phase 3: EMA decay clamp upper bound. The schedule clamps the
    /// per-step computed decay to `[ema_min_decay, ema_max_decay]`. Standard
    /// values: 0.999 (fast averaging), 0.9999 (default — diffusers EMAModel),
    /// 0.99999 (very slow averaging).
    #[arg(long, default_value_t = 0.9999)] ema_max_decay: f32,
    /// Phase 3: swap EMA shadow weights into live params at sample/checkpoint
    /// time. Default false. No effect when EMA is not constructed.
    #[arg(long, default_value_t = false)] ema_validation_swap: bool,
    #[arg(long)] tread_route_pattern: Option<String>,

    // ── Multi-resolution noise (default-off; byte-invariant when 0) ──────
    #[arg(long, default_value_t = 0)] multires_noise_iterations: usize,
    #[arg(long, default_value_t = 0.3)] multires_noise_discount: f32,

    // ── Timestep biasing (default `none` is byte-identity) ───────────────
    #[arg(long, default_value = "none")] timestep_bias_strategy: String,
    #[arg(long, default_value_t = 0.0)] timestep_bias_multiplier: f32,
    #[arg(long, default_value_t = 0.0)] timestep_bias_range_min: f32,
    #[arg(long, default_value_t = 1.0)] timestep_bias_range_max: f32,
    /// Phase 1: optimizer family CLI surface (Phase 5 wires full dispatch).
    #[arg(long, default_value = "adamw")] optimizer: String,

    // ── Phase 6 multi-feature rollout (plumb-only; multi-backend wired in Klein) ──
    #[arg(long, num_args = 0..)] multi_backend_repeats: Vec<u32>,
    #[arg(long, default_value_t = false)] caption_tag_shuffle: bool,
    #[arg(long, default_value_t = false)] cache_clear_each_epoch: bool,
    #[arg(long, default_value_t = false)] cache_invalidate: bool,
    /// Phase 5: LR scheduler family. Default `constant` is byte-equivalent to
    /// the legacy linear-warmup-then-flat path qwenimage has used since launch.
    /// Accepted: constant, linear, cosine, cosine_with_restarts, polynomial, rex.
    #[arg(long, default_value = "constant")] lr_scheduler: String,
    /// Phase 5: cosine-with-restarts cycle count. Ignored for other schedulers.
    #[arg(long, default_value_t = 1.0)] lr_cycles: f32,
}

/// Self-adjusting shift based on image sequence length.
/// musubi-tuner `qwen_image_utils.py:956-959` — anchors 256→0.5, 8192→0.9, exp.
fn shift_for_resolution(resolution: [usize; 2]) -> f32 {
    const VAE_SCALE: usize = 8;
    const PATCH_SIZE: usize = 2;
    const BASE_SEQ_LEN: f32 = 256.0;
    const MAX_SEQ_LEN: f32 = 8192.0;
    const BASE_SHIFT: f32 = 0.5;
    const MAX_SHIFT: f32 = 0.9;
    let [w, h] = resolution;
    let seq_len = ((h / VAE_SCALE / PATCH_SIZE) * (w / VAE_SCALE / PATCH_SIZE)) as f32;
    let m = (MAX_SHIFT - BASE_SHIFT) / (MAX_SEQ_LEN - BASE_SEQ_LEN);
    let b = BASE_SHIFT - m * BASE_SEQ_LEN;
    let mu = seq_len * m + b;
    mu.exp()
}

/// Format seconds as `HH:MM:SS` (< 24h) or `HHH:MM:SS` (≥ 24h).
fn format_elapsed(secs: f32) -> String {
    let total = secs.max(0.0) as u64;
    let h = total / 3600;
    let m = (total % 3600) / 60;
    let s = total % 60;
    if h >= 24 {
        format!("{:03}:{:02}:{:02}", h, m, s)
    } else {
        format!("{:02}:{:02}:{:02}", h, m, s)
    }
}

/// Sample logit-normal then apply qwen_shift remap. Matches musubi
/// `t = sigmoid(z); t = t*shift / (1 + (shift-1)*t)` byte-for-byte.
fn sample_timestep_logit_normal_qwenshift(rng: &mut StdRng, shift: f32) -> f32 {
    use rand_distr::{Distribution, Normal};
    let normal = Normal::new(0.0f32, 1.0f32).unwrap();
    let z = normal.sample(rng);
    let t = 1.0 / (1.0 + (-z).exp());
    shift * t / (1.0 + (shift - 1.0) * t)
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
    flame_core::config::set_default_dtype(DType::BF16);
    let device = flame_core::global_cuda_device();

    let shift = args.qwen_shift.unwrap_or_else(|| {
        shift_for_resolution([args.resolution, args.resolution])
    });
    log::info!(
        "[train_qwenimage] model={}, cache={}, steps={}, rank={}, alpha={}, lr={}, res={}², shift={:.3}",
        args.model.display(), args.cache_dir.display(),
        args.steps, args.rank, args.lora_alpha, args.lr, args.resolution, shift,
    );

    // ── Sample-setup MUST run before DiT load ────────────────────────────
    // Qwen2.5-VL is ~14 GB BF16 on GPU. After the DiT (60-block BlockOffloader
    // + activation pool) loads, GPU has ~5 GB free which can't fit the TE.
    // So pre-encode prompts NOW (TE only resident on GPU), drop the TE, then
    // load DiT into the freed memory. Mirrors train_ernie with order swapped.
    let periodic_sample = args.sample_every > 0;
    let (sample_cond_1, sample_cond_2, sample_uncond, sample_vae_path) = if periodic_sample {
        let te_path = args.sample_text_encoder.as_ref()
            .ok_or_else(|| anyhow::anyhow!("--sample-every > 0 requires --sample-text-encoder"))?;
        let tok_path = args.sample_tokenizer.as_ref()
            .ok_or_else(|| anyhow::anyhow!("--sample-every > 0 requires --sample-tokenizer"))?;
        let vae_path = args.sample_vae.as_ref()
            .ok_or_else(|| anyhow::anyhow!("--sample-every > 0 requires --sample-vae"))?
            .clone();
        let p1 = args.sample_prompt_1.as_ref()
            .ok_or_else(|| anyhow::anyhow!("--sample-every > 0 requires --sample-prompt-1"))?
            .clone();
        let p2 = args.sample_prompt_2.as_ref()
            .ok_or_else(|| anyhow::anyhow!("--sample-every > 0 requires --sample-prompt-2"))?
            .clone();

        log::info!("[sample-setup] loading Qwen2.5-VL + tokenizer for prompt pre-encode...");
        let tok = tokenizers::Tokenizer::from_file(tok_path)
            .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
        let te_weights = load_te_weights(te_path, &device)?;
        let te_cfg = Qwen25VLEncoder::config_from_weights(&te_weights)?;
        let te = Qwen25VLEncoder::new(te_weights, te_cfg, device.clone());

        let encode_one = |text: &str| -> anyhow::Result<Tensor> {
            // Match prepare_qwenimage's PROMPT_TEMPLATE_ENCODE wrap +
            // DROP_IDX slice + trailing-pad trim. The DiT was trained
            // against variable-length text embeddings (per-prompt content
            // length); padding to a fixed `sample_max_text_len` would
            // pollute joint attention with junk pad-token hiddens. Mirrors
            // inference-flame's `qwenimage_encode::encode_and_trim`.
            let wrapped = format!("{PROMPT_PREFIX}{text}{PROMPT_SUFFIX}");
            let enc = tok.encode(wrapped, false)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let raw_ids: Vec<i32> = enc.get_ids().iter().map(|&i| i as i32).collect();
            let work_len = args.sample_max_text_len + DROP_IDX;
            let mut ids: Vec<i32> = raw_ids.iter().take(work_len).copied().collect();
            let real_len_pre_pad = ids.len();
            ids.resize(work_len, QWEN_PAD_ID);
            let real_len = real_len_pre_pad.min(work_len);
            if real_len <= DROP_IDX {
                anyhow::bail!(
                    "sample prompt tokenized to only {real_len} ids; expected > {DROP_IDX} after PROMPT_TEMPLATE_ENCODE wrap"
                );
            }
            let kept_len = real_len - DROP_IDX;
            let full_hidden = te.encode(&ids)?.to_dtype(DType::BF16)?;
            full_hidden.narrow(1, DROP_IDX, kept_len)
                .map_err(|e| anyhow::anyhow!("narrow: {e}"))
        };
        let cond_1 = encode_one(&p1)?;
        let cond_2 = encode_one(&p2)?;
        let uncond = if args.sample_cfg > 1.0 {
            Some(encode_one(&args.sample_neg_prompt)?)
        } else {
            None
        };
        log::info!("[sample-setup] dropping text encoder; cond_1={:?} cond_2={:?}{}",
            cond_1.shape().dims(), cond_2.shape().dims(),
            uncond.as_ref().map(|u| format!(", uncond={:?}", u.shape().dims())).unwrap_or_default(),
        );
        drop(te);
        flame_core::cuda_alloc_pool::clear_pool_cache();
        flame_core::trim_cuda_mempool(0);
        log::info!("[sample-setup] periodic sample enabled (every {} steps; 2 prompts).", args.sample_every);
        (Some(cond_1), Some(cond_2), uncond, Some(vae_path))
    } else {
        (None, None, None, None)
    };

    log::info!("Loading Qwen-Image transformer...");
    let mut model = QwenImageTrainingModel::load(
        &args.model, args.rank, args.lora_alpha, /*full_finetune*/ false,
        device.clone(), args.seed,
    )?;

    // Activation offload pool — sized for 24 GB GPU + 62 GB system after
    // accounting for BlockOffloader's ~38 GB pinned weights. Headroom for
    // the activation pool is ~22 GB pinned.
    //
    // Per-slot size (BF16): max_seq * MLP_HIDDEN * 2
    //   max_seq = pack_h * pack_w + 512 = 1024 + 512 = 1536 at 512²
    //   slot = 1536 * 12288 * 2 = ~38 MB
    //
    // Budget: 8 slots/block × 60 blocks + 32 buffer = 512 slots total.
    //   No compression: 512 × 38 MB ≈ 19 GB pinned.
    //   FP8 compression: ~9.5 GB pinned (effectively 1024-slot equivalent).
    //
    // Saved tensors that don't fit auto-fall-back to recompute via the
    // graceful path (autograd.rs:1795) — slower but no crash. The chroma
    // pattern's slots_per_block=130 was 16× over budget and OOM'd 62 GB
    // system RAM hard last attempt.
    {
        use eridiffusion_core::training::offload::{setup_activation_offload, OffloadConfig};
        use flame_core::activation_offload::OffloadCompression;
        let pack_h = args.resolution / 8 / 2;
        let pack_w = args.resolution / 8 / 2;
        let max_seq = pack_h * pack_w + 512;
        let max_activation_bytes = max_seq * qwen_model::MLP_HIDDEN * 2;
        // FP8 mode allocates per-slot **GPU** staging buffers for the
        // quant kernel (activation_offload.rs:367). Each slot costs:
        //   pinned: max_bytes/2 (FP8) ≈ 19 MB
        //   GPU staging: max_bytes (BF16) ≈ 38 MB
        //
        // Closure-capture leak fix in qwenimage.rs freed the previous
        // ~38 GB of weight-Arc GPU pressure, so we can fit a bigger pool
        // now. Bumped from 4→8 slots/block for ~2× more offload coverage:
        //   512 slots × 38 MB GPU staging ≈ 19 GB GPU
        //   512 slots × 19 MB pinned       ≈ 10 GB pinned
        // BlockOffloader 2 blocks GPU = 1.3 GB, leaves ~3-4 GB GPU for
        // activations / LoRA state. Tight but workable on 24 GB.
        // If this OOMs, drop back to 4. If it doesn't OOM and pool still
        // fills (warns about exhaustion), try 12.
        let slots_per_block = 8;
        let total_slots = qwen_model::NUM_LAYERS * slots_per_block + 32;
        let cfg = OffloadConfig {
            num_blocks: qwen_model::NUM_LAYERS,
            max_activation_bytes,
            // FP8 compression halves pinned bytes per slot. Saved tensor
            // values are 8-bit quant on push, dequant on pop. Roughly 2x
            // effective slot count vs None at the same pinned budget.
            compression: OffloadCompression::FP8,
            extra_slots: total_slots - qwen_model::NUM_LAYERS,
        };
        match setup_activation_offload(&device, &cfg) {
            Ok((slots, bytes)) => log::info!(
                "[activation_offload] {slots} slots, {:.1} GB pinned (FP8), slot={:.1}MB raw",
                bytes as f64 / 1e9, max_activation_bytes as f64 / 1e6
            ),
            Err(e) => log::warn!(
                "[activation_offload] setup failed ({e}); falling back to recompute checkpoint (slower)"
            ),
        }
    }
    let params = model.parameters();
    log::info!("Loaded {} trainable LoRA tensors", params.len());
    if params.is_empty() {
        anyhow::bail!("No trainable parameters — model produced empty param list");
    }

    // Cache enumeration.
    std::fs::create_dir_all(&args.output_dir)?;
    let mut cache_files: Vec<PathBuf> = std::fs::read_dir(&args.cache_dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("safetensors"))
        .collect();
    cache_files.sort();
    if cache_files.is_empty() {
        anyhow::bail!("No cached samples in {}", args.cache_dir.display());
    }
    log::info!("Found {} cached samples", cache_files.len());

    match OptimizerKind::parse(&args.optimizer) {
        Ok(OptimizerKind::AdamW) => {}
        Ok(other) => log::warn!(
            "non-AdamW optimizer selected: {} — Phase 1 falls back to AdamW (full dispatch in Phase 5)",
            other.as_str()
        ),
        Err(e) => log::warn!("--optimizer parse: {} — falling back to AdamW", e),
    }
    if args.caption_dropout_probability > 0.0 {
        log::warn!(
            "caption_dropout_probability={:.3} requested but Qwen-Image trainer has no inline encoder — feature disabled",
            args.caption_dropout_probability
        );
    }
    let mut optimizer = AdamW::new(args.lr, 0.9, 0.999, 1e-8, 0.01);

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

    // Timestep bias config — defaults are byte-identical (Strategy::None).
    // qwenimage operates in continuous-sigma space [0, 1] (logit-normal then
    // qwen_shift remap), so we pass `total = 1.0` to apply_bias rather than
    // the `NUM_TRAIN_TIMESTEPS = 1000` convention used by Klein/Z-Image.
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

    let mut start_step = 0usize;

    // Resume.
    if let Some(path) = &args.resume_lora {
        log::info!("Resuming LoRA weights from {}", path.display());
        let loaded = checkpoint::load_full(path, &device)?;
        let named = qwen_named_parameters(&model);
        checkpoint::apply_lora_weights(&loaded, &named)?;
    } else if let Some(path) = &args.resume_full {
        log::info!("Full-resume from {}", path.display());
        let loaded = checkpoint::load_full(path, &device)?;
        let named = qwen_named_parameters(&model);
        checkpoint::apply_lora_weights(&loaded, &named)?;
        checkpoint::apply_to_optimizer(&loaded, &mut optimizer, &named, args.rank, args.lora_alpha)?;
        start_step = loaded.header.step as usize;
        if start_step >= args.steps {
            log::warn!("Resumed step {} >= --steps {}; nothing to do.", start_step, args.steps);
            return Ok(());
        }
        log::info!("Continuing from step {}/{}", start_step, args.steps);
    }

    // No step-0 / no in-loop sampling — per "do it once, like inference".
    // The single sample lands AFTER training completes, with the activation
    // offload pool torn down so the 1024² VAE decode has full GPU.

    let clipper = GradientClipper::clip_by_norm(1.0);
    let mut rng = StdRng::seed_from_u64(args.seed);
    let mut total_loss = 0.0f32;
    let board = BoardWriter::open(
        &args.output_dir,
        BoardWriter::new_session_id(),
        if start_step > 0 { Some(start_step as u64) } else { None },
    ).map_err(|e| log::warn!("board.db open failed: {e}")).ok();
    if let Some(b) = &board {
        log::info!("SerenityBoard: writing scalars to {}", b.db_path.display());
    }
    let t_start = std::time::Instant::now();

    log::info!("Training {} steps from step={}", args.steps, start_step);
    let sched: LrScheduler = args.lr_scheduler.parse().unwrap_or_else(|e: String| {
        log::warn!("[lr_scheduler] {e} — falling back to Constant");
        LrScheduler::Constant
    });
    for step in start_step..args.steps {
        // Phase 5: dispatch via LrScheduler enum. Default `Constant` is
        // byte-equivalent to the legacy hand-rolled linear-warmup-then-flat path.
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

        // Load one cached sample.
        let idx = step % cache_files.len();
        let tensors = flame_core::serialization::load_file(&cache_files[idx], &device)?;
        let latent = tensors.get("latent")
            .ok_or_else(|| anyhow::anyhow!("cache missing 'latent'"))?
            .to_dtype(DType::BF16)?;
        let txt_embed = tensors.get("text_embedding")
            .ok_or_else(|| anyhow::anyhow!("cache missing 'text_embedding'"))?
            .to_dtype(DType::BF16)?;

        let lat_dims = latent.shape().dims().to_vec();
        let (b, _c, latent_h, latent_w) = (lat_dims[0], lat_dims[1], lat_dims[2], lat_dims[3]);
        let _ = b;

        // Sample timestep with qwen_shift.
        let raw_t = sample_timestep_logit_normal_qwenshift(&mut rng, shift);
        // Default-off: Strategy::None → returns raw_t unchanged. qwenimage
        // sigma is already in [0, 1], so we pass total=1.0.
        let t_continuous = timestep_bias::apply_bias(
            raw_t,
            1.0,
            &timestep_bias_cfg,
        );
        let sigma = t_continuous;
        let timestep = Tensor::from_vec(vec![sigma], Shape::from_dims(&[1]), device.clone())?
            .to_dtype(DType::BF16)?;

        // Flow-matching: x_t = (1 - sigma) * latent + sigma * noise; target = noise - latent.
        let noise = Tensor::randn(latent.shape().clone(), 0.0, 1.0, device.clone())?
            .to_dtype(DType::BF16)?;
        // Multi-resolution noise (default-off). When iterations==0 returns
        // noise.clone() with no rng draw — byte-identical to baseline.
        let noise = noise_modifiers::maybe_apply_multires_noise(
            &noise,
            args.multires_noise_iterations,
            args.multires_noise_discount,
            &mut rng,
        )?;
        // Phase 1: noise modifiers (default-off). Qwen-Image trainer doesn't
        // load TrainConfig JSON — `offset_noise_weight` is hardcoded 0.0.
        // Offset noise is part of the clean noise distribution; input
        // perturbation feeds model input only.
        let clean_noise = noise_modifiers::maybe_apply_offset_noise(
            &noise,
            0.0,
            args.noise_offset_probability,
            &mut rng,
        )?;
        let perturbed_noise = noise_modifiers::maybe_apply_input_perturbation(
            &clean_noise,
            args.gamma_input_perturbation,
            &mut rng,
        )?;
        let xt = latent.mul_scalar(1.0 - sigma)?.add(&perturbed_noise.mul_scalar(sigma)?)?;
        let target = clean_noise.sub(&latent)?;

        // Pack [B, 16, H, W] → [B, H/2 * W/2, 64] for forward.
        let xt_packed = qwen_model::pack_latents(&xt)?;
        let target_packed = qwen_model::pack_latents(&target)?;

        // Forward.
        AutogradContext::clear();
        let pred = model.forward(&xt_packed, &timestep, &txt_embed, latent_h, latent_w)?;

        // MSE loss in F32.
        // Phase 1: combined loss + per-step weighting. Default-off invariant.
        let pred_f32 = pred.to_dtype(DType::F32)?;
        let target_f32 = target_packed.to_dtype(DType::F32)?;
        let raw_loss = loss_weight::combined_loss(
            &pred_f32,
            &target_f32,
            1.0,
            0.0,
            args.huber_strength,
        )?;
        let loss = loss_weight::apply_loss_weight(
            &raw_loss,
            sigma,
            eridiffusion_core::config::LossWeight::Constant,
            args.min_snr_gamma,
            true,
        )?;

        let loss_val: f32 = loss.to_vec()?.first().copied().unwrap_or(f32::NAN);
        if !loss_val.is_finite() {
            anyhow::bail!("step {}: non-finite loss {}", step + 1, loss_val);
        }
        total_loss += loss_val;

        // Backward.
        let grads = loss.backward()?;
        for param in &params {
            if let Some(g) = grads.get(param.id()) {
                let g = if g.dtype() == DType::F32 { g.clone() } else { g.to_dtype(DType::F32)? };
                param.set_grad(g)?;
            }
        }

        // Clip + step.
        let grad_norm = {
            let mut grad_tensors: Vec<Tensor> = Vec::new();
            let mut owners: Vec<usize> = Vec::new();
            for (i, param) in params.iter().enumerate() {
                if let Some(g) = param.grad() {
                    grad_tensors.push(g);
                    owners.push(i);
                }
            }
            let mut grad_refs: Vec<&mut Tensor> = grad_tensors.iter_mut().collect();
            let norm = clipper.clip_grads(&mut grad_refs)?;
            for (owner, grad) in owners.into_iter().zip(grad_tensors.into_iter()) {
                params[owner].set_grad(grad)?;
            }
            norm
        };

        {
            let _guard = AutogradContext::no_grad();
            optimizer.step(&params)?;
            optimizer.zero_grad(&params);
            model.refresh_lora_cache();
            if let Some(ref mut e) = ema {
                e.update_with_schedule(&params, &ema_cfg, (step + 1) as u64)
                    .map_err(|err| anyhow::anyhow!("EMA update failed at step {}: {err}", step + 1))?;
            }
        }
        AutogradContext::clear();
        flame_core::cuda_alloc_pool::clear_pool_cache();

        let step_num = step + 1;
        let _ = total_loss;
        eridiffusion_core::training::progress::log_step(
            step, args.steps, cache_files.len(), 1,
            loss_val, grad_norm, current_lr, t_start, board.as_ref(),
        );

        if args.save_every > 0 && step_num % args.save_every == 0 && step_num < args.steps {
            // EMA swap: when `--ema --ema-validation-swap`, the periodic save
            // captures EMA-averaged weights. Restored after save so optimizer
            // moments stay consistent with the live tensors they were taken
            // against.
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
            let path = args.output_dir.join(format!("qwenimage_lora_step{}.safetensors", step_num));
            save_ckpt(&path, &model, &optimizer, args.rank, args.lora_alpha, args.seed, &args.save_mode, step_num)?;
            if let (Some(backup), Some(ref e)) = (ema_backup, &ema) {
                let _g = AutogradContext::no_grad();
                e.restore_swapped(&params, backup)
                    .map_err(|err| anyhow::anyhow!("EMA restore_swapped (mid) failed: {err}"))?;
            }
        }

        // No in-loop sampling — see top comment.
    }

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

    let final_path = args.output_dir.join(format!("qwenimage_lora_{}steps.safetensors", args.steps));
    save_ckpt(&final_path, &model, &optimizer, args.rank, args.lora_alpha, args.seed, &args.save_mode, args.steps)?;

    // Final sample after training — single pass, both prompts, 1024².
    //
    // Tear down the activation offload pool first: it holds ~19 GB of GPU
    // staging buffers that the VAE decoder needs for the 1024² mid-block
    // attention. With the pool gone, the 1024² decode fits on 24 GB GPU.
    // (Once training is complete the pool isn't needed anymore.)
    if periodic_sample {
        log::info!("[sample FINAL] tearing down activation offload pool to free GPU for 1024² decode...");
        flame_core::autograd::clear_activation_offload_pool();
        flame_core::cuda_alloc_pool::clear_pool_cache();
        flame_core::trim_cuda_mempool(0);

        let cond_1 = sample_cond_1.as_ref().unwrap();
        let cond_2 = sample_cond_2.as_ref().unwrap();
        let uncond = sample_uncond.as_ref();
        let vae_path = sample_vae_path.as_ref().unwrap();
        for (idx, cond) in [cond_1, cond_2].iter().enumerate() {
            let out_path = args.output_dir.join(format!("sample_step{}_final_p{}.png", args.steps, idx + 1));
            log::info!("[sample FINAL step={} p{}] → {}", args.steps, idx + 1, out_path.display());
            if let Err(e) = qwenimage_inline_sample(
                &mut model, cond, uncond, vae_path, &out_path,
                args.sample_size, args.sample_steps, args.sample_cfg, args.sample_seed, &device,
            ) {
                log::warn!("[sample FINAL p{}] failed: {e}", idx + 1);
            }
            flame_core::cuda_alloc_pool::clear_pool_cache();
            flame_core::trim_cuda_mempool(0);
        }
    }
    let trained = args.steps - start_step;
    log::info!(
        "Training complete: {} new steps (total {}). avg loss={:.4}. Saved to {}",
        trained, args.steps,
        total_loss / trained.max(1) as f32,
        final_path.display(),
    );
    if let Some(b) = &board { b.set_status("completed"); }
    Ok(())
}

/// Build the canonical `(name, Parameter)` pairs for `checkpoint::save_full`
/// and `apply_to_optimizer`. The names match `QwenImageLoraBundle::save`
/// byte-for-byte so resumed AdamW state lines up with live params.
///
/// Iteration order: deterministic (sorted by `(block_idx, target_suffix)`).
/// Required because `HashMap` iteration is random across runs and
/// save→reload must produce a stable key→tensor mapping.
fn qwen_named_parameters(model: &QwenImageTrainingModel)
    -> Vec<(String, flame_core::parameter::Parameter)>
{
    use eridiffusion_core::models::qwenimage::QwenImageLoraBundle;
    let mut entries: Vec<((usize, &str), &eridiffusion_core::lora::LoRALinear)> = model
        .bundle
        .adapters
        .iter()
        .map(|(&(idx, target), lora)| ((idx, QwenImageLoraBundle::target_suffix(target)), lora))
        .collect();
    entries.sort_by(|a, b| a.0.cmp(&b.0));

    let mut out: Vec<(String, flame_core::parameter::Parameter)> =
        Vec::with_capacity(entries.len() * 2);
    for ((block_idx, suffix), lora) in entries {
        let prefix = format!("transformer_blocks.{block_idx}.{suffix}");
        // LoRALinear::parameters() returns [lora_a, lora_b]; the safetensors
        // save scheme is `{prefix}.lora_A.weight` then `{prefix}.lora_B.weight`.
        out.push((format!("{prefix}.lora_A.weight"), lora.lora_a().clone()));
        out.push((format!("{prefix}.lora_B.weight"), lora.lora_b().clone()));
    }
    out
}

/// In-process sample call. Pre-encoded prompts skip the (huge) text-encoder
/// load. Disables FLAME_CHECKPOINT inside the sampler scope (no autograd
/// during inference) and restores it on exit so training continues with
/// the same recompute behavior.
fn qwenimage_inline_sample(
    model: &mut QwenImageTrainingModel,
    cond: &Tensor,
    uncond: Option<&Tensor>,
    vae_path: &std::path::Path,
    out_path: &std::path::Path,
    size: usize,
    steps: usize,
    cfg: f32,
    seed: u64,
    device: &std::sync::Arc<flame_core::CudaDevice>,
) -> anyhow::Result<()> {
    qwenimage_sampler::sample_image(
        model,
        cond,
        uncond,
        size, size,
        steps,
        cfg,
        seed,
        vae_path,
        out_path,
        device,
    ).map_err(|e| anyhow::anyhow!("qwenimage sample: {e}"))?;
    Ok(())
}

/// Load Qwen2.5-VL text encoder weights from a directory of shards or a
/// single combined safetensors file.
fn load_te_weights(
    path: &std::path::Path,
    device: &std::sync::Arc<flame_core::CudaDevice>,
) -> anyhow::Result<std::collections::HashMap<String, Tensor>> {
    if path.is_file() {
        return flame_core::serialization::load_file(path, device)
            .map_err(|e| anyhow::anyhow!("text-encoder load: {e}"));
    }
    let mut all = std::collections::HashMap::new();
    for entry in std::fs::read_dir(path)
        .map_err(|e| anyhow::anyhow!("read_dir {}: {e}", path.display()))?
    {
        let p = entry.map_err(|e| anyhow::anyhow!("entry: {e}"))?.path();
        if p.extension().and_then(|e| e.to_str()) == Some("safetensors") {
            let part = flame_core::serialization::load_file(&p, device)
                .map_err(|e| anyhow::anyhow!("text-encoder shard {}: {e}", p.display()))?;
            all.extend(part);
        }
    }
    Ok(all)
}

fn save_ckpt(
    path: &std::path::Path,
    model: &QwenImageTrainingModel,
    optimizer: &AdamW,
    rank: usize,
    alpha: f32,
    seed: u64,
    mode: &str,
    step: usize,
) -> anyhow::Result<()> {
    if mode == "weights" {
        model.save_weights(path)?;
        log::info!("[save] {} (weights only)", path.display());
        return Ok(());
    }
    let header = CkptHeader::from_adamw(
        "train_qwenimage",
        step as u64,
        optimizer,
        rank,
        alpha,
        seed,
        String::new(),
    );
    let named = qwen_named_parameters(model);
    if named.is_empty() {
        // Until qwenimage gets a named_parameters() helper, fall back to
        // weights-only for the periodic save (full-resume from this file is
        // disabled by the named.is_empty() guard above).
        log::warn!("[save] qwenimage missing named_parameters; falling back to weights-only");
        model.save_weights(path)?;
        let _ = header;
        return Ok(());
    }
    checkpoint::save_full(path, &named, optimizer, &header)
        .map_err(|e| anyhow::anyhow!("save_full: {e}"))?;
    Ok(())
}
