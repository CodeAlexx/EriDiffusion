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
use eridiffusion_core::models::{sdxl::SDXLModel, TrainableModel};
use eridiffusion_core::sampler::sdxl_sampler::sin_embed_256;
use eridiffusion_core::training::board::BoardWriter;
use eridiffusion_core::training::checkpoint::{self, CkptHeader};
use eridiffusion_core::training::ema::ParameterEma;
use eridiffusion_core::training::features::ema_advanced::EmaConfig;
use eridiffusion_core::training::features::{loss_weight, lr_schedule, noise_modifiers, timestep_bias};
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

    let shards = collect_shards(&args.unet)?;
    log::info!("[SDXL] loading UNet from {} shard(s) (rank={}, alpha={})",
        shards.len(), args.rank, args.lora_alpha);
    let mut model = SDXLModel::load(&shards, &config, device.clone())?;
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

    if args.caption_dropout_probability > 0.0 {
        log::warn!(
            "caption_dropout_probability={:.3} requested but SDXL trainer has no inline encoder — feature disabled",
            args.caption_dropout_probability
        );
    }

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

    let alpha_bar = compute_alpha_bar();
    let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);

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
