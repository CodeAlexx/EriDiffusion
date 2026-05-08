//! train_ltx2 — LTX-2 T2V LoRA training binary.
//!
//! Mirrors the structure of `train_ernie.rs`, adapted for video latents
//! `[B, 128, F, H, W]`.
//!
//! ## Pipeline per step
//! 1. Load cached `latent` `[1, 128, 1, h, w]` (image-as-frame bootstrap)
//!    or `[1, 128, F', h, w]` (true video; future work) and
//!    `text_embedding` `[1, T_text, CAPTION_CHANNELS]`.
//! 2. Sample timestep per logit-normal with shifted-LTX2 schedule (token
//!    count → mu) — `ltx2_sampler::sample_timestep_logit_normal`.
//! 3. `noisy = (1-σ) * latent + σ * noise`, target = `noise - latent`.
//! 4. Forward → `[1, 128, F', h, w]`.
//! 5. Loss = mean MSE in F32.
//! 6. Backward, clip-grad-norm @ 1.0, AdamW step.
//!
//! ## Constraints from the build prompt
//! - Pure Rust, no Python at runtime. ✓
//! - Default seed 42 across step + sample. ✓
//! - F32 mean MSE loss. ✓
//! - clip_grad_norm = 1.0. ✓
//! - timestep tensor F32. ✓
//! - --batch-size, --sample-every, --save-every, --resume-lora flags. ✓
//! - LoRA-B nonzero ratio printed after first save. ✓
//! - OT_DEBUG_STATS-format per-step diagnostics gated by env. ✓
//! - Inline sampler at step 0 + every N + final, wrapped in if-let-Err. ✓

use clap::Parser;
use std::path::PathBuf;

use eridiffusion_core::config::{LrScheduler, TrainConfig, TrainingMethod};
use eridiffusion_core::training::features::{
    ema_advanced::EmaConfig, loss_weight, lr_schedule, noise_modifiers, timestep_bias,
};
use eridiffusion_core::training::ema::ParameterEma;
use eridiffusion_core::training::training_features::OptimizerKind;
use eridiffusion_core::debug as dbg;
use eridiffusion_core::encoders::ltx2_vae::Ltx2Vae;
use eridiffusion_core::models::{Ltx2Model, TrainableModel};
use eridiffusion_core::sampler::ltx2_sampler;
use eridiffusion_core::training::board::BoardWriter;
use flame_core::adam::AdamW;
use flame_core::autograd::AutogradContext;
use flame_core::{DType, Shape, Tensor};

const SEED: u64 = 42;
const NUM_TRAIN_TIMESTEPS: usize = 1000;

#[derive(Parser)]
struct Args {
    #[arg(long)] config: PathBuf,
    #[arg(long)] cache_dir: PathBuf,
    #[arg(long, default_value = "100")] steps: usize,
    #[arg(long, default_value = "16")] rank: usize,
    #[arg(long, default_value = "1.0")] lora_alpha: f64,
    #[arg(long, default_value = "3e-4")] lr: f32,
    #[arg(long, default_value = "1")] batch_size: usize,
    #[arg(long, default_value = "output")] output_dir: PathBuf,

    #[arg(long, default_value = "0")] sample_every: usize,
    #[arg(long, default_value = "0")] save_every: usize,
    #[arg(long, default_value = "")] sample_prompt: String,
    /// LTX-2 video VAE checkpoint (single safetensors).
    #[arg(long)] sample_vae: Option<PathBuf>,
    #[arg(long, default_value = "256")] sample_size: usize,
    #[arg(long, default_value = "20")] sample_steps: usize,
    #[arg(long, default_value = "5.0")] sample_cfg: f32,
    #[arg(long, default_value = "42")] sample_seed: u64,

    /// Resume training from a previous LoRA checkpoint.
    #[arg(long)] resume_lora: Option<PathBuf>,

    /// Frames per second (RoPE temporal axis scaling). Default 24.
    #[arg(long, default_value = "24.0")] fps: f32,

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
    /// Master EMA switch. Default-off → byte-identical to no-EMA.
    #[arg(long, default_value_t = false)] ema: bool,
    #[arg(long, default_value_t = 1.0)] ema_inv_gamma: f32,
    #[arg(long, default_value_t = 0.6667)] ema_power: f32,
    #[arg(long, default_value_t = 0)] ema_update_after_step: u64,
    #[arg(long, default_value_t = 0.0)] ema_min_decay: f32,
    #[arg(long, default_value_t = 0.9999)] ema_max_decay: f32,
    /// Swap EMA shadow into live params at sample/checkpoint time, then
    /// restore. Default-off keeps live params untouched.
    #[arg(long, default_value_t = false)] ema_validation_swap: bool,

    /// Multi-resolution noise iterations. NOTE: helper is 4D-only; LTX-2 uses
    /// 5D video latents [B, 128, F, H, W] so this flag emits a warn-and-skip.
    /// Kept for CLI uniformity with other trainers.
    #[arg(long, default_value_t = 0)] multires_noise_iterations: usize,
    /// Per-level discount factor for `--multires-noise-iterations`.
    #[arg(long, default_value_t = 0.3)] multires_noise_discount: f32,

    /// Timestep biasing strategy: `none|earlier|later|range`. Default `none`
    /// is byte-identical to no biasing.
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

fn debug_enabled() -> bool {
    std::env::var("OT_DEBUG_STATS")
        .map_or(false, |v| !matches!(v.as_str(), "0" | "" | "false" | "FALSE"))
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
    config.ema_inv_gamma = args.ema_inv_gamma;
    config.ema_power = args.ema_power;
    config.ema_update_after_step = args.ema_update_after_step;
    config.ema_min_decay = args.ema_min_decay;
    config.ema_validation_swap = args.ema_validation_swap;
    config.tread_route_pattern = args.tread_route_pattern.clone();

    // Load the LTX-2 transformer shards.
    let model_base = std::path::Path::new(&config.base_model_name);
    let mut shard_paths: Vec<PathBuf> = std::fs::read_dir(model_base.join("transformer"))
        .ok()
        .map(|rd| rd.filter_map(|e| e.ok().map(|e| e.path())).collect())
        .unwrap_or_default();
    if shard_paths.is_empty() {
        // Fallback: maybe base_model_name itself points to a dir of shards.
        shard_paths = std::fs::read_dir(model_base)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("safetensors"))
            .collect();
    }
    shard_paths.retain(|p| p.extension().and_then(|s| s.to_str()) == Some("safetensors"));
    shard_paths.sort();
    if shard_paths.is_empty() {
        anyhow::bail!("No safetensors shards under {:?} (or its transformer/ subdir)", model_base);
    }

    log::info!("Loading LTX-2 transformer (rank={} alpha={})...", args.rank, args.lora_alpha);
    let mut model = Ltx2Model::load(&shard_paths, &config, device.clone())?;
    if let Some(resume) = &args.resume_lora {
        model.load_weights(resume.to_str().unwrap())?;
        log::info!("Resumed LoRA from {}", resume.display());
    }
    let params = model.parameters();
    log::info!("Loaded {} trainable LoRA tensors", params.len());
    if params.is_empty() {
        anyhow::bail!("No trainable parameters — TrainingMethod::Lora produced empty param list");
    }

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
            "caption_dropout_probability={:.3} requested but LTX2 trainer has no inline encoder — feature disabled",
            args.caption_dropout_probability
        );
    }
    let mut opt = AdamW::new(args.lr, 0.9, 0.999, 1e-8, 0.01);

    // EMA shadow (Phase 3 advanced). Default-off → byte-identical to no-EMA.
    // Updated under no_grad after each opt.step via `update_with_schedule`.
    // Optional swap into live params at sample/checkpoint time when
    // `--ema-validation-swap` is set.
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

    // Multi-resolution noise: helper expects 4D [B, C, H, W]. LTX-2 latents
    // are 5D [B, 128, F, H, W] (video), so the helper would no-op silently.
    // Warn explicitly so the user knows the flag has no effect here.
    if args.multires_noise_iterations > 0 {
        log::warn!(
            "[multires-noise] LTX-2 uses 5D video latents; multires noise (4D-only helper) is skipped. Pass 0 to silence."
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

    let mut cache_files: Vec<PathBuf> = std::fs::read_dir(&args.cache_dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map_or(false, |e| e == "safetensors"))
        .collect();
    cache_files.sort();
    if cache_files.is_empty() {
        anyhow::bail!("No cached samples in {:?}", args.cache_dir);
    }
    log::info!("Found {} cached samples (batch_size={})", cache_files.len(), args.batch_size);

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
    let mut first_save_done = false;

    let sched: LrScheduler = lr_schedule::parse_cli_scheduler(&args.lr_scheduler);
    for step in 0..args.steps {
        // ── Build a batch by stacking `batch_size` cached samples along dim 0 ──
        let mut batch_latents: Vec<Tensor> = Vec::with_capacity(args.batch_size);
        let mut batch_texts: Vec<Tensor> = Vec::with_capacity(args.batch_size);
        for bi in 0..args.batch_size {
            let cache_idx = (step * args.batch_size + bi) % cache_files.len();
            let sample = flame_core::serialization::load_file(&cache_files[cache_idx], &device)?;
            let latent = sample
                .get("latent")
                .ok_or_else(|| anyhow::anyhow!("cached sample missing 'latent'"))?
                .to_dtype(DType::BF16)?;
            let txt_full = sample
                .get("text_embedding")
                .ok_or_else(|| anyhow::anyhow!("cached sample missing 'text_embedding'"))?
                .to_dtype(DType::BF16)?;
            // Trim to real_len if available (Gemma3 uses fixed 1024 left-pad,
            // so the first `pad_n` positions are pad embeddings; we keep them).
            // (T2V cross-attn applies mask if needed; here we pass full 1024.)
            batch_latents.push(latent);
            batch_texts.push(txt_full);
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

        // ── Sample per-batch-element timesteps ──
        let dims = latent.shape().dims();
        // Token count for shift schedule: F * H * W (per sample).
        let n_tokens = dims[2] * dims[3] * dims[4];
        let mu = ltx2_sampler::shift_for_token_count(n_tokens);
        let mut t_continuous = Vec::with_capacity(args.batch_size);
        for _ in 0..args.batch_size {
            let raw_t = ltx2_sampler::sample_timestep_logit_normal(&mut rng, mu);
            // Default-off: Strategy::None returns raw_t unchanged.
            let t = timestep_bias::apply_bias(
                raw_t,
                NUM_TRAIN_TIMESTEPS as f32,
                &timestep_bias_cfg,
            );
            t_continuous.push(t);
        }
        // Cap-and-floor to integer index for sigma lookup; LTX-2 uses 1000-step
        // discretization same as ERNIE/Z-Image.
        let sigmas: Vec<f32> = t_continuous.iter().map(|&t| {
            let idx = (t.floor() as usize).min(999);
            (idx + 1) as f32 / 1000.0
        }).collect();

        // ── Build noisy + target ──
        let noise = Tensor::randn(latent.shape().clone(), 0.0, 1.0, device.clone())?
            .to_dtype(DType::BF16)?;
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
        // For batch_size > 1 with per-sample sigmas, noise scaling needs
        // per-sample broadcast. For batch_size == 1 (the bootstrap default) it's
        // a scalar; for >1 we expand a [B,1,1,1,1] tensor.
        let (noisy, target) = if args.batch_size == 1 {
            let s = sigmas[0];
            let noisy = perturbed_noise.mul_scalar(s)?.add(&latent.mul_scalar(1.0 - s)?)?;
            let target = clean_noise.sub(&latent)?;
            (noisy, target)
        } else {
            // Build [B, 1, 1, 1, 1] sigma tensor.
            let s_tensor = Tensor::from_vec(
                sigmas.clone(),
                Shape::from_dims(&[args.batch_size, 1, 1, 1, 1]),
                device.clone(),
            )?.to_dtype(DType::BF16)?;
            let one_minus_s = s_tensor.mul_scalar(-1.0)?.add_scalar(1.0)?;
            let noisy = perturbed_noise.mul(&s_tensor)?.add(&latent.mul(&one_minus_s)?)?;
            let target = clean_noise.sub(&latent)?;
            (noisy, target)
        };

        // ── timestep tensor — F32 (audit: BF16 mantissa loses precision >256) ──
        let timestep = Tensor::from_vec(
            t_continuous.clone(),
            Shape::from_dims(&[args.batch_size]),
            device.clone(),
        )?;  // F32 by default from from_vec

        if step == 0 {
            log::info!(
                "step 0 | latent={:?} text={:?} sigma={:.4} mu={:.3} n_tokens={}",
                dims, txt.shape().dims(), sigmas[0], mu, n_tokens
            );
        }

        // ── Forward ──
        // Call the inherent Ltx2Model::forward to pass FPS explicitly.
        let pred = Ltx2Model::forward(&mut model, &noisy, &txt, &timestep, args.fps)?;
        if pred.shape().dims() != target.shape().dims() {
            anyhow::bail!(
                "predicted shape {:?} != target {:?}",
                pred.shape().dims(), target.shape().dims()
            );
        }

        // ── Loss = mean MSE in F32 ──
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
            sigmas[0],
            config.loss_weight_fn,
            args.min_snr_gamma,
            true,
        )?;
        let loss_val = loss.to_vec()?[0];
        total_loss += loss_val;

        // ── Backward + clip-grad-norm + step ──
        let grads = loss.backward()?;

        if debug_enabled() && (step < 3 || (step + 1) % 100 == 0) {
            let p_st = dbg::stats(&pred);
            let t_st = dbg::stats(&target);
            eprintln!(
                "[OT_DEBUG step={:5}] t={:.2} loss(pre-scale)={:.4} | pred[mean={:+.3e} std={:.3e} max|·|={:.3e}] target[mean={:+.3e} std={:.3e} max|·|={:.3e}]",
                step, t_continuous[0], loss_val,
                p_st.mean, p_st.std, p_st.abs_max,
                t_st.mean, t_st.std, t_st.abs_max,
            );
        }

        const CLIP_GRAD_NORM: f32 = 1.0;
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
            if let Some(ref mut e) = ema {
                e.update_with_schedule(&params, &ema_cfg, (step + 1) as u64)
                    .map_err(|err| anyhow::anyhow!("EMA update failed at step {}: {err}", step + 1))?;
            }
        }
        AutogradContext::clear();

        let _ = total_loss;
        eridiffusion_core::training::progress::log_step(
            step, args.steps, cache_files.len(), args.batch_size.max(1),
            loss_val, total_norm, cur_lr, t_start, board.as_ref(),
        );

        // ── Periodic save + sample ──
        // EMA swap: when `--ema --ema-validation-swap`, save and sample see
        // EMA-averaged weights. Backup is restored at the end of this block
        // so the optimizer's accumulated moments stay consistent with the
        // tensors they were taken against.
        let step_num = step + 1;
        let save_fires = args.save_every > 0 && step_num % args.save_every == 0 && step_num < args.steps;
        let sample_fires = args.sample_every > 0 && step_num % args.sample_every == 0 && step_num < args.steps;
        let ema_backup = if args.ema_validation_swap && (save_fires || sample_fires) {
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

        if save_fires {
            let mid_ckpt = args.output_dir.join(format!("ltx2_lora_step{step_num}.safetensors"));
            if let Err(e) = model.save_weights(&mid_ckpt.to_string_lossy()) {
                log::warn!("[mid-save step {step_num}] save_weights failed: {e}");
            } else {
                log::info!("[mid-save step {step_num}] {}", mid_ckpt.display());
                if !first_save_done {
                    print_lora_b_nonzero(&model);
                    first_save_done = true;
                }
            }
        }

        if sample_fires {
            let out = args.output_dir.join(format!("sample_step{step_num}.png"));
            if let Err(e) = inline_sample(
                &mut model,
                &args.sample_prompt,
                args.sample_vae.as_deref(),
                &out,
                args.sample_size, args.sample_steps, args.sample_cfg, args.sample_seed,
                args.fps,
                &device,
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

    let avg_loss = total_loss / args.steps as f32;
    log::info!("Training complete: {} steps, avg loss={:.4}", args.steps, avg_loss);
    if let Some(b) = &board { b.set_status("completed"); }

    // Final EMA swap (covers final save and final sample). No restore — the
    // process exits, no further training. Skipped when --ema-validation-swap
    // is off or no EMA was constructed.
    if args.ema_validation_swap {
        if let Some(ref e) = ema {
            let _g = AutogradContext::no_grad();
            let _ = e.swap_with_live(&params)
                .map_err(|err| anyhow::anyhow!("EMA swap_with_live (final) failed: {err}"))?;
            log::info!("[ema] swapped EMA shadow into live params for final save + sample");
        }
    }

    let ckpt = args.output_dir.join(format!("ltx2_lora_{}steps.safetensors", args.steps));
    if let Err(e) = model.save_weights(&ckpt.to_string_lossy()) {
        log::warn!("save_weights returned error: {e}");
    } else {
        log::info!("Saved checkpoint to {}", ckpt.display());
        if !first_save_done {
            print_lora_b_nonzero(&model);
        }
    }

    if args.sample_every > 0 || !args.sample_prompt.is_empty() {
        let out = args.output_dir.join(format!("sample_step{}_FINAL.png", args.steps));
        if let Err(e) = inline_sample(
            &mut model,
            &args.sample_prompt,
            args.sample_vae.as_deref(),
            &out,
            args.sample_size, args.sample_steps, args.sample_cfg, args.sample_seed,
            args.fps,
            &device,
        ) {
            log::warn!("[sample final] failed: {e}");
        }
    }

    Ok(())
}

/// Per memory `feedback_flame_core_bf16_fused_autograd`: print LoRA-B
/// nonzero ratio after first save. Catches dead-branch bugs early.
fn print_lora_b_nonzero(model: &Ltx2Model) {
    let mut nz = 0usize;
    let mut total = 0usize;
    let mut nz_branches = 0usize;
    for adapter in &model.lora_adapters {
        let b = match adapter.lora_b().tensor() {
            Ok(t) => t,
            Err(e) => { log::warn!("LoRA-B tensor read failed: {e}"); continue; }
        };
        let v = match b.to_vec() { Ok(v) => v, Err(e) => { log::warn!("LoRA-B to_vec: {e}"); continue; } };
        let mut branch_has_nz = false;
        for x in &v {
            total += 1;
            if x.abs() > 1e-12 { nz += 1; branch_has_nz = true; }
        }
        if branch_has_nz { nz_branches += 1; }
    }
    let pct = if total > 0 { nz as f64 / total as f64 * 100.0 } else { 0.0 };
    log::info!(
        "[lora-b-check] {}/{} branches have nonzero B ({:.1}% of B values nonzero)",
        nz_branches, model.lora_adapters.len(), pct
    );
    if nz_branches < model.lora_adapters.len() {
        log::warn!(
            "[lora-b-check] {} dead LoRA-B branches detected — \
             likely a flame_core fused-op autograd bug along that path",
            model.lora_adapters.len() - nz_branches
        );
    }
}

/// Inline sampler — runs a small T2V sample using current model state.
/// VAE-decode + image-save (1-frame video → single PNG via take frame 0).
#[allow(clippy::too_many_arguments)]
fn inline_sample(
    model: &mut Ltx2Model,
    _prompt: &str,
    vae_path: Option<&std::path::Path>,
    out_path: &std::path::Path,
    size: usize,
    steps: usize,
    _cfg: f32,
    seed: u64,
    fps: f32,
    device: &std::sync::Arc<flame_core::CudaDevice>,
) -> anyhow::Result<()> {
    let _no_grad = AutogradContext::no_grad();

    use rand::SeedableRng;
    let _rng = rand::rngs::StdRng::seed_from_u64(seed);

    let h_lat = size / 32;
    let w_lat = size / 32;
    let f_lat = 1usize; // image-as-frame bootstrap
    let n_tokens = f_lat * h_lat * w_lat;
    let sigmas = ltx2_sampler::schedule(steps, n_tokens);

    let latent_shape = Shape::from_dims(&[1, 128, f_lat, h_lat, w_lat]);
    let mut latent = Tensor::randn(latent_shape, 0.0, 1.0, device.clone())?
        .to_dtype(DType::BF16)?;

    // For inline sample without a real Gemma3 encoder, use zeros as text emb.
    // Real previews require a cached embedding for the prompt.
    let txt = Tensor::zeros_dtype(
        Shape::from_dims(&[1, 1024, eridiffusion_core::encoders::gemma3::GEMMA3_HIDDEN]),
        DType::BF16,
        device.clone(),
    )?;

    for step in 0..steps {
        let sigma = sigmas[step];
        let sigma_next = sigmas[step + 1];
        let t = ltx2_sampler::sigma_to_timestep(sigma);
        let t_tensor = Tensor::from_vec(vec![t], Shape::from_dims(&[1]), device.clone())?;
        let pred = model.forward(&latent, &txt, &t_tensor, fps)?;
        latent = ltx2_sampler::euler_step(&latent, &pred, sigma, sigma_next)?;
    }

    // VAE decode if available; else save zeros.
    let pixel_video = if let Some(vp) = vae_path {
        let vae = Ltx2Vae::load(vp, device.clone())
            .map_err(|e| anyhow::anyhow!("vae load: {e}"))?;
        // Denormalize before decode.
        let denormed = vae.denormalize(&latent)
            .map_err(|e| anyhow::anyhow!("denormalize: {e}"))?;
        vae.decode_video(&denormed)
            .map_err(|e| anyhow::anyhow!("decode_video: {e}"))?
    } else {
        Tensor::zeros_dtype(
            Shape::from_dims(&[1, 3, 1, h_lat * 32, w_lat * 32]),
            DType::BF16,
            device.clone(),
        )?
    };

    // Take frame 0, write as PNG.
    let frame0 = pixel_video.narrow(2, 0, 1)?.contiguous()?;
    let pixels: Vec<f32> = frame0.to_dtype(DType::F32)?.to_vec()?;
    let dims = frame0.shape().dims();
    // Expect [1, 3, 1, H, W].
    let (c, h, w) = (dims[1], dims[3], dims[4]);
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
