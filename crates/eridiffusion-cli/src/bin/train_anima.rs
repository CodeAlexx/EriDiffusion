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
use eridiffusion_core::config::{TrainConfig, TrainingMethod};
use eridiffusion_core::debug as dbg;
use eridiffusion_core::models::{anima as anima_mod, AnimaModel, TrainableModel};
use eridiffusion_core::training::checkpoint::{self, CkptHeader};

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
    std::fs::create_dir_all(&args.output_dir)?;

    flame_core::config::set_default_dtype(DType::BF16);
    let device = flame_core::global_cuda_device();

    let mut config = TrainConfig::from_json_path(&args.config.to_string_lossy())?;
    config.training_method = TrainingMethod::Lora;
    config.lora_rank = args.rank as u64;
    config.lora_alpha = args.lora_alpha;
    config.learning_rate = args.lr as f64;

    log::info!("Loading Anima DiT (rank={} alpha={}) from {}...",
        args.rank, args.lora_alpha, args.dit_path.display());
    let mut model = AnimaModel::load(&args.dit_path, &config, device.clone())?;
    let params = model.parameters();
    log::info!("Loaded {} trainable LoRA tensors ({} adapters across {} blocks)",
        params.len(), model.bundle.adapters.len(), anima_mod::NUM_BLOCKS);
    if params.is_empty() {
        anyhow::bail!("No trainable parameters — TrainingMethod::Lora produced empty param list");
    }

    let mut opt = AdamW::new(args.lr, 0.9, 0.999, 1e-8, 0.01);

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

    let t_start = std::time::Instant::now();
    let mut total_loss = 0f32;

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
        let t_continuous = match args.timestep_sampling.as_str() {
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
        let noisy = noise.mul_scalar(sigma)?
            .add(&latent.mul_scalar(1.0 - sigma)?)?;
        let target = noise.sub(&latent)?;
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
        let weight = loss_weight(&args.weighting_scheme, sigma);
        let diff = pred.to_dtype(DType::F32)?.sub(&target.to_dtype(DType::F32)?)?;
        let raw_loss = diff.square()?.mean()?;
        let loss = if (weight - 1.0).abs() > 1e-6 {
            raw_loss.mul_scalar(weight)?
        } else {
            raw_loss
        };
        let loss_val = loss.to_vec()?[0];
        total_loss += loss_val;

        let grads = loss.backward()?;

        if debug_grads_enabled && (step < 3 || (step + 1) % 100 == 0) {
            dbg::print_lora_grad_summary(step, &model.bundle.adapters, &ANIMA_LORA_CLASSES, &grads);
        }

        // OT default: clip_grad_norm = 1.0.
        const CLIP_GRAD_NORM: f32 = 1.0;
        let mut total_norm_sq = 0f32;
        for param in &params {
            if let Some(g) = grads.get(param.id()) {
                let g_f32 = g.to_dtype(DType::F32)?;
                let sq = g_f32.square()?.mean()?;
                let n = g_f32.shape().dims().iter().product::<usize>() as f32;
                total_norm_sq += sq.to_vec()?[0] * n;
            }
        }
        let total_norm = total_norm_sq.sqrt();
        let scale = if total_norm > CLIP_GRAD_NORM { CLIP_GRAD_NORM / total_norm } else { 1.0 };
        for param in &params {
            if let Some(g) = grads.get(param.id()) {
                let g_scaled = if scale < 1.0 { g.mul_scalar(scale)? } else { g.clone() };
                param.set_grad(g_scaled)?;
            }
        }

        {
            let _g = AutogradContext::no_grad();
            opt.step(&params)?;
            opt.zero_grad(&params);
        }
        AutogradContext::clear();

        if step == 0 || (step + 1) % 10 == 0 {
            let avg = total_loss / (step + 1) as f32;
            let elapsed = t_start.elapsed().as_secs_f32();
            let sps = (step + 1) as f32 / elapsed.max(0.001);
            log::info!("step {}/{} | loss={:.4} avg={:.4} | grad_norm={:.4} | {:.2} step/s",
                step + 1, args.steps, loss_val, avg, total_norm, sps);
        }

        // Periodic save (sample-rendering deferred — model.forward stub).
        let step_num = step + 1;
        if args.save_every > 0 && step_num % args.save_every == 0 && step_num < args.steps {
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
    }

    let trained = args.steps - start_step;
    let avg_loss = if trained > 0 { total_loss / trained as f32 } else { 0.0 };
    log::info!("Training complete: {trained} new steps (total={}), avg loss={:.4}", args.steps, avg_loss);

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
