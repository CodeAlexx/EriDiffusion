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
use flame_core::adam::AdamW;
use flame_core::gradient_clip::GradientClipper;
use eridiffusion_core::models::AceStepLoRAModel;
use eridiffusion_core::training::board::BoardWriter;
use eridiffusion_core::training::checkpoint::{self, CkptHeader};
use eridiffusion_core::config::LrScheduler;
use eridiffusion_core::training::features::{
    ema_advanced::EmaConfig, loss_weight, lr_schedule, noise_modifiers, timestep_bias,
};
use eridiffusion_core::training::ema::ParameterEma;
use eridiffusion_core::training::schedule;
use eridiffusion_core::training::training_features::OptimizerKind;
use rand::{rngs::StdRng, Rng, SeedableRng};
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
    #[arg(long, default_value = "-0.4")] timestep_mu: f32,
    /// Logit-normal timestep sigma (default 1.0).
    #[arg(long, default_value = "1.0")] timestep_sigma: f32,
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
    flame_core::config::set_default_dtype(DType::BF16);
    let device = flame_core::global_cuda_device();

    log::info!("Loading ACE-Step DiT (rank={} alpha={}) from {}...",
        args.rank, args.lora_alpha, args.model.display());
    let model = AceStepLoRAModel::from_safetensors(
        &args.model, args.rank, args.lora_alpha, device.clone(),
    )?;
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
            "caption_dropout_probability={:.3} requested but ACE-Step trainer has no inline encoder — feature disabled",
            args.caption_dropout_probability
        );
    }
    let mut optimizer = AdamW::new(args.lr, 0.9, 0.999, 1e-8, 0.01);
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
        checkpoint::apply_to_optimizer(&loaded, &mut optimizer, &named, args.rank, args.lora_alpha)?;
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
            let raw_t = schedule::sample_timestep_logit_normal(&mut rng, args.timestep_mu, args.timestep_sigma);
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
            step, args.steps, cache_files.len(), 1,
            loss_val, grad_norm, current_lr, t_start, board.as_ref(),
        );

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
    optimizer: &AdamW,
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
    let header = CkptHeader::from_adamw(
        "train_acestep",
        step as u64,
        optimizer,
        rank,
        alpha,
        seed,
        String::new(),
    );
    let named = model.named_parameters();
    checkpoint::save_full(path, &named, optimizer, &header)
        .map_err(|e| anyhow::anyhow!("save_full: {e}"))?;
    Ok(())
}
