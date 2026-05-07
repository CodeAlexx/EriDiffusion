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
use flame_core::adam::AdamW;
use eridiffusion_core::config::{TrainConfig, TrainingMethod};
use eridiffusion_core::debug as dbg;
use eridiffusion_core::encoders::{mistral3b::Mistral3bEncoder, vae::KleinVaeDecoder};
use eridiffusion_core::models::{ErnieModel, TrainableModel};
use eridiffusion_core::sampler::ernie_sampler;
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
}

/// LOGIT_NORMAL timestep sample matching OT _get_timestep_discrete.
/// Returns continuous timestep in [0, num_train_timesteps), passed to the model.
/// Caller floors it to look up sigma.
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

    let model_base = std::path::Path::new(&config.base_model_name);
    let shards: Vec<PathBuf> = (1..=2).map(|i|
        model_base.join("transformer").join(format!("diffusion_pytorch_model-0000{i}-of-00002.safetensors"))
    ).collect();

    log::info!("Loading Ernie transformer (rank={} alpha={})...", args.rank, args.lora_alpha);
    let mut model = ErnieModel::load(&shards, &config, device.clone())?;
    let params = model.parameters();
    log::info!("Loaded {} trainable LoRA tensors", params.len());
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

    let t_start = std::time::Instant::now();
    let mut total_loss = 0f32;

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
        // Trim padded text positions before feeding the DiT — matches upstream Python
        // ErnieModel.py:153-154 `text_encoder_output[:, :text_lengths.max(), :]`.
        // With batch_size=1 cache, real_len IS the trim length. If the cache was
        // produced by an older prepare_ernie that didn't write text_real_len,
        // fall back to the full padded length (legacy 77-pad behaviour).
        let txt = if let Some(rl_t) = sample.get("text_real_len") {
            let rl = rl_t.to_dtype(DType::F32)?.to_vec()?[0] as usize;
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
        let t_continuous = sample_timestep_logit_normal(&mut rng);
        let sigma_idx = (t_continuous.floor() as usize).min(NUM_TRAIN_TIMESTEPS - 1);
        let sigma = (sigma_idx + 1) as f32 / NUM_TRAIN_TIMESTEPS as f32;

        let noise = Tensor::randn(latent.shape().clone(), 0.0, 1.0, device.clone())?
            .to_dtype(DType::BF16)?;
        let noisy = noise.mul_scalar(sigma)?
            .add(&latent.mul_scalar(1.0 - sigma)?)?;
        let target = noise.sub(&latent)?;
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
        let diff = pred.to_dtype(DType::F32)?.sub(&target.to_dtype(DType::F32)?)?;
        let loss = diff.square()?.mean()?;
        let loss_val = loss.to_vec()?[0];
        total_loss += loss_val;

        let grads = loss.backward()?;

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
            log::info!("step {}/{} | loss={:.4} avg={:.4} | {:.2} step/s",
                step + 1, args.steps, loss_val, avg, sps);
        }

        // ── Periodic save + inline sample (every N steps) ───────────────
        let step_num = step + 1;
        if periodic && step_num % args.sample_every == 0 && step_num < args.steps {
            let mid_ckpt = args.output_dir.join(format!("ernie_lora_step{step_num}.safetensors"));
            if save_mode_full {
                let header = CkptHeader::from_adamw(
                    "train_ernie", step_num as u64, &opt,
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
    }

    let trained = args.steps - start_step;
    let avg_loss = if trained > 0 { total_loss / trained as f32 } else { 0.0 };
    log::info!("Training complete: {trained} new steps (total={}), avg loss={:.4}", args.steps, avg_loss);

    let ckpt = args.output_dir.join(format!("ernie_lora_{}steps.safetensors", args.steps));
    if save_mode_full {
        let header = CkptHeader::from_adamw(
            "train_ernie", args.steps as u64, &opt,
            args.rank, args.lora_alpha as f32, SEED, String::new(),
        );
        let named = model.named_parameters();
        if let Err(e) = checkpoint::save_full(&ckpt, &named, &opt, &header) {
            log::warn!("save_full failed: {e}");
        } else {
            log::info!("Saved checkpoint to {}", ckpt.display());
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
