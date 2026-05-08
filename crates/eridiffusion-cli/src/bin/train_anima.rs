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
//! Compiles + runs through scaffolding + pre/post-step bookkeeping. The forward
//! pass goes via `AnimaModel::forward` which currently returns NotImplemented
//! (see crates/eridiffusion-core/src/models/anima.rs). The training loop will
//! error out at the first step; the structure is in place for when the model
//! port lands.

use clap::Parser;
use std::path::PathBuf;
use flame_core::{autograd::AutogradContext, DType, Shape, Tensor};
use flame_core::adam::AdamW;
use eridiffusion_core::config::{LrScheduler, TrainConfig, TrainingMethod};
use eridiffusion_core::debug as dbg;
use eridiffusion_core::models::{anima as anima_mod, AnimaModel, TrainableModel};
use eridiffusion_core::training::board::BoardWriter;
use eridiffusion_core::training::checkpoint::{self, CkptHeader};
use eridiffusion_core::training::ema::ParameterEma;
use eridiffusion_core::training::features::ema_advanced::EmaConfig;
use eridiffusion_core::training::features::{
    loss_weight as feat_loss_weight, lr_schedule, noise_modifiers, timestep_bias,
};
use eridiffusion_core::training::training_features::OptimizerKind;

/// Slot class names for the 10 LoRA modules per Anima block. Used by debug
/// gradient summaries. MUST match `anima::LORA_SLOT_KEYS` order.
const ANIMA_LORA_CLASSES: [&str; anima_mod::LORA_SLOTS_PER_BLOCK] = [
    "self_attn.q", "self_attn.k", "self_attn.v", "self_attn.out",
    "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.out",
    "mlp.layer1", "mlp.layer2",
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
    #[arg(long, default_value = "3e-4")] lr: f32,

    /// Timestep sampling: "sigmoid" (kohya default), "uniform", "shift".
    #[arg(long, default_value = "sigmoid")] timestep_sampling: String,
    /// Sigmoid scale used by `sigmoid` sampling (kohya default 1.0).
    #[arg(long, default_value = "1.0")] sigmoid_scale: f32,
    /// Rectified-flow timestep shift (kohya `--discrete_flow_shift`, default 1.0).
    #[arg(long, default_value = "1.0")] discrete_flow_shift: f32,
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

    // ── Periodic save (sample-image rendering deferred until model forward lands) ──
    #[arg(long, default_value = "0")] save_every: usize,

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
    #[arg(long)] tread_route_pattern: Option<String>,
    /// Phase 1: optimizer family CLI surface (Phase 5 wires full dispatch).
    #[arg(long, default_value = "adamw")] optimizer: String,

    // ── Phase 6 multi-feature rollout (plumb-only; multi-backend wired in Klein) ──
    #[arg(long, num_args = 0..)] multi_backend_repeats: Vec<u32>,
    #[arg(long, default_value_t = false)] caption_tag_shuffle: bool,
    #[arg(long, default_value_t = false)] cache_clear_each_epoch: bool,
    #[arg(long, default_value_t = false)] cache_invalidate: bool,
    /// Phase 5: LR scheduler family. Default `constant` + `warmup_steps=0` is
    /// byte-equivalent to the prior fixed-LR behaviour.
    #[arg(long, default_value = "constant")] lr_scheduler: String,
    /// Phase 5: linear LR warmup steps. Default 0 keeps prior behaviour.
    #[arg(long, default_value_t = 0)] warmup_steps: usize,
    /// Phase 5: cosine-with-restarts cycle count.
    #[arg(long, default_value_t = 1.0)] lr_cycles: f32,
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
    config.tread_route_pattern = args.tread_route_pattern.clone();

    log::info!("Loading Anima DiT (rank={} alpha={}) from {}...",
        args.rank, args.lora_alpha, args.dit_path.display());
    let mut model = AnimaModel::load(&args.dit_path, &config, device.clone())?;
    let params = model.parameters();
    log::info!("Loaded {} trainable LoRA tensors ({} adapters across {} blocks)",
        params.len(), model.bundle.adapters.len(), anima_mod::NUM_BLOCKS);
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
            "caption_dropout_probability={:.3} requested but Anima trainer has no inline encoder — feature disabled",
            args.caption_dropout_probability
        );
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

    let mut cache_files: Vec<PathBuf> = std::fs::read_dir(&args.cache_dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map_or(false, |e| e == "safetensors"))
        .collect();
    cache_files.sort();
    if cache_files.is_empty() {
        anyhow::bail!("No cached samples in {:?}", args.cache_dir);
    }
    log::info!("Found {} cached samples", cache_files.len());

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

        // Timestep sampling.
        let raw_t = match args.timestep_sampling.as_str() {
            "sigmoid" => sample_timestep_sigmoid(&mut rng, args.sigmoid_scale),
            "uniform" => sample_timestep_uniform(&mut rng),
            "shift" => {
                let raw = sample_timestep_sigmoid(&mut rng, args.sigmoid_scale);
                let s = raw / NUM_TRAIN_TIMESTEPS as f32;
                let shifted = apply_shift(s, args.discrete_flow_shift);
                shifted * NUM_TRAIN_TIMESTEPS as f32
            }
            other => anyhow::bail!("unknown --timestep-sampling: {other}"),
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

        let grads = loss.backward()?;

        if debug_grads_enabled && (step < 3 || (step + 1) % 100 == 0) {
            dbg::print_lora_grad_summary(step, &model.bundle.adapters, &ANIMA_LORA_CLASSES, &grads);
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
            step, args.steps, cache_files.len(), 1,
            loss_val, total_norm, cur_lr, t_start, board.as_ref(),
        );

        // Periodic save (sample-rendering deferred — model.forward stub).
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
                let header = CkptHeader::from_adamw(
                    "train_anima", step_num as u64, &opt,
                    args.rank, args.lora_alpha as f32, SEED, String::new(),
                );
                let named = model.named_parameters();
                if let Err(e) = checkpoint::save_full(&mid_ckpt, &named, &opt, &header) {
                    log::warn!("[mid-save step {step_num}] full save failed: {e}");
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
        let header = CkptHeader::from_adamw(
            "train_anima", args.steps as u64, &opt,
            args.rank, args.lora_alpha as f32, SEED, String::new(),
        );
        let named = model.named_parameters();
        if let Err(e) = checkpoint::save_full(&ckpt, &named, &opt, &header) {
            log::warn!("save_full failed: {e}");
        } else {
            log::info!("Saved checkpoint to {}", ckpt.display());
        }
    } else if let Err(e) = model.save_weights(&ckpt.to_string_lossy()) {
        log::warn!("save_weights returned error: {e}");
    } else {
        log::info!("Saved checkpoint to {}", ckpt.display());
    }

    Ok(())
}
