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
use eridiffusion_core::training::checkpoint::{self, CkptHeader};
use eridiffusion_core::training::schedule;
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

    let mut optimizer = AdamW::new(args.lr, 0.9, 0.999, 1e-8, 0.01);
    let mut start_step = 0usize;

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
    let t_start = std::time::Instant::now();

    log::info!("Training {} steps from step={}, lr={} warmup={} cfg_ratio={}",
        args.steps, start_step, args.lr, args.warmup_steps, args.cfg_ratio);

    for step in start_step..args.steps {
        // Warmup-only LR schedule (matches original main.rs behaviour).
        let current_lr = schedule::constant_with_warmup(args.lr, step, args.warmup_steps);
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
        let x0 = target_latents;
        let t_val = {
            let z = schedule::sample_timestep_logit_normal(&mut rng, args.timestep_mu, args.timestep_sigma);
            // schedule helper already returns sigmoid(z*sigma+mu)-equivalent in (0,1).
            z
        };
        let t_tensor = Tensor::from_vec(vec![t_val], Shape::from_dims(&[1]), device.clone())?
            .to_dtype(DType::BF16)?;

        // x_t = t * x1 + (1 - t) * x0
        let xt = x1.mul_scalar(t_val)?.add(&x0.mul_scalar(1.0 - t_val)?)?;

        // Forward + flow-matching loss.
        AutogradContext::clear();
        let pred = model.forward(
            &xt,
            &t_tensor,
            &t_tensor, // r = t for ACE-Step training
            &encoder_hs,
            &context_latents,
        )?;
        let flow = x1.sub(&x0)?;
        let diff = pred.sub(&flow)?;
        let loss = diff.mul(&diff)?.mean()?;

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
        }
        AutogradContext::clear();

        let avg = loss_sum / loss_count as f32;
        if (step + 1) == 1
            || (step + 1) % 10 == 0
            || (step + 1) == args.steps
        {
            let dt = t_start.elapsed().as_secs_f32().max(1e-3);
            log::info!(
                "step {}/{} | loss={:.4} avg={:.4} | grad_norm={:.4} | lr={:.2e} | {:.2} step/s",
                step + 1, args.steps, loss_val, avg, grad_norm, current_lr,
                (loss_count as f32) / dt,
            );
        }

        // Periodic save.
        if args.save_every > 0 && (step + 1) % args.save_every == 0 && (step + 1) < args.steps {
            let path = args.output_dir.join(format!("acestep_lora_step{}.safetensors", step + 1));
            save_ckpt(&path, &model, &optimizer, args.rank, args.lora_alpha, args.seed, &args.save_mode, step + 1)?;
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
