//! train_zimage — Z-Image LoRA training binary, mirroring EriDiffusion's released preset.
//!
//! Reference: ALL six released `/home/alex/upstream Python/training_presets/#z-image *.json`
//! (verified 2026-05-04 — LoRA 8GB/16GB, Finetune 16GB/24GB, DeTurbo LoRA 8GB/16GB):
//!   - timestep_distribution: LOGIT_NORMAL  (NOT sigmoid — earlier note was wrong;
//!     SIGMOID appears only in `configs/eri2_zimage_base_2500.json` which is a
//!     personal experiment, not a released preset)
//!   - noising_weight: 0.0  (TrainConfig default since presets don't override)
//!   - noising_bias: 0.0
//!   - timestep_shift: 1.0
//!   - dynamic_timestep_shifting: false
//!   - learning_rate: 0.0003
//!   - resolution: 512
//!   - training_method: LORA
//!
//! Pipeline per step:
//!   1. Load cached `{latent, text_embedding, text_mask}` (prepared by prepare_zimage).
//!   2. Sample LOGIT_NORMAL timestep ∈ [0, num_train_timesteps).
//!   3. sigma = (idx+1)/1000; noisy = noise·sigma + clean·(1-sigma).
//!   4. Forward → predicted velocity; target = noise - clean (rectified flow).
//!   5. Loss = mean MSE in F32, with clip_grad_norm=1.0.

use clap::Parser;
use flame_core::{autograd::AutogradContext, DType, Shape, Tensor};
use flame_core::adam::AdamW;
use rand::distributions::Distribution as _;
use std::path::PathBuf;

use eridiffusion_core::encoders::qwen3::Qwen3Encoder;
use eridiffusion_core::models::zimage::ZImageModel;
use eridiffusion_core::sampler::zimage_sampler;
use eridiffusion_core::training::checkpoint::{self, CkptHeader};

const ZIMAGE_TEMPLATE_PRE: &str = "<|im_start|>user\n";
const ZIMAGE_TEMPLATE_POST: &str = "<|im_end|>\n<|im_start|>assistant\n";
const ZIMAGE_PAD_TOKEN_ID: i32 = 151643;
const ZIMAGE_TXT_PAD_LEN: usize = 512;

const NUM_TRAIN_TIMESTEPS: usize = 1000;
const LOGIT_NORMAL_BIAS: f32 = 0.0;   // OT TrainConfig default
const LOGIT_NORMAL_SCALE: f32 = 1.0;  // noising_weight + 1.0 = 0.0 + 1.0
const TIMESTEP_SHIFT: f32 = 1.0;
const SEED: u64 = 42;
// Z-Image VAE scale/shift — must be applied at train time (and inverted at
// sample time before VAE decode). Pretrained Z-Image DiT was trained on
// scaled latents per OT BaseZImageSetup.predict() and musubi's zimage train.
// Caching raw `posterior.mode()` is correct (matches musubi's encode pattern);
// the (latent-shift)*scale transformation belongs at predict time.
const ZIMAGE_VAE_SHIFT: f32 = 0.1159;
const ZIMAGE_VAE_SCALE: f32 = 0.3611;

#[derive(Parser)]
struct Args {
    /// Single-file Z-Image transformer safetensors.
    #[arg(long)] model: PathBuf,
    #[arg(long)] cache_dir: PathBuf,
    #[arg(long, default_value = "500")] steps: usize,
    #[arg(long, default_value = "16")] rank: usize,
    /// OT TrainConfig default. Earlier `16.0` came from a misread of the
    /// `alpha=rank` convention — OT presets do NOT override `lora_alpha`,
    /// so it stays at its TrainConfig default of 1.0 (effective scale =
    /// alpha/rank = 0.0625). Using alpha=16 made the LoRA branch contribute
    /// 16× more than OT trains/loads at, miscalibrating gradient magnitudes
    /// and over-driving the LoRA delta during inference.
    #[arg(long, default_value = "1.0")] lora_alpha: f32,
    #[arg(long, default_value = "3e-4")] lr: f32,
    /// Per-step batch size — N cached samples stacked along dim 0. OT
    /// Python preset uses batch=2; ED-v2 default 1 keeps single-image flow.
    #[arg(long, default_value = "1")] batch_size: usize,
    /// Save a LoRA checkpoint every N steps WITHOUT rendering an image
    /// (independent from `--sample-every`). 0 disables. Useful for protecting
    /// long runs against crashes.
    #[arg(long, default_value = "0")] save_every: usize,
    /// Resume from a saved LoRA checkpoint — overwrites the freshly-init
    /// zeros after model load. Use to continue training (e.g. phase-2 at
    /// 1024² resuming from phase-1 at 512²). Optimizer state is NOT
    /// resumed; AdamW restarts fresh, which is fine for fine-tune.
    #[arg(long, conflicts_with = "resume_full")] resume_lora: Option<PathBuf>,
    /// Full resume: LoRA weights + AdamW (m, v, t) + step counter. Refuses
    /// rank/alpha mismatch. `--steps N` is the TARGET total step (loop
    /// continues from `step` in the ckpt up to N). Use over `--resume-lora`
    /// for continuous training across stops/restarts.
    #[arg(long, conflicts_with = "resume_lora")] resume_full: Option<PathBuf>,
    /// Periodic + final save mode. Default is `full` (LoRA + AdamW state +
    /// step) so resume is true. `weights` writes legacy weights-only files.
    #[arg(long, default_value = "full")] save_mode: String,
    #[arg(long, default_value = "output")] output_dir: PathBuf,

    // ── Periodic save + sample (every N steps) ─────────────────────────
    #[arg(long, default_value = "0")] sample_every: usize,
    #[arg(long, default_value = "")] sample_prompt: String,
    #[arg(long, default_value = "")] sample_neg_prompt: String,
    /// LDM Z-Image VAE safetensors (e.g. qwen_image_vae.safetensors).
    #[arg(long)] sample_vae: Option<PathBuf>,
    /// Qwen3 4B weights for prompt encoding.
    #[arg(long)] sample_qwen3: Option<PathBuf>,
    /// Tokenizer.json from Z-Image base.
    #[arg(long)] sample_tokenizer: Option<PathBuf>,
    #[arg(long, default_value = "1024")] sample_size: usize,
    #[arg(long, default_value = "20")] sample_steps: usize,
    #[arg(long, default_value = "4.0")] sample_cfg: f32,
    #[arg(long, default_value = "3.0")] sample_shift: f32,
    #[arg(long, default_value = "42")] sample_seed: u64,
}

/// LOGIT_NORMAL timestep sample matching OT _get_timestep_discrete.
/// Returns continuous timestep in [0, num_train_timesteps), passed to the model.
/// Caller floors it to look up sigma. Same math as `train_ernie.rs`.
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

fn main() -> anyhow::Result<()> {
    use rand::SeedableRng;
    env_logger::init();
    let args = Args::parse();
    std::fs::create_dir_all(&args.output_dir)?;

    flame_core::config::set_default_dtype(DType::BF16);
    let device = flame_core::global_cuda_device();

    log::info!("Loading Z-Image transformer (rank={} alpha={})...", args.rank, args.lora_alpha);
    let mut model = ZImageModel::load(
        &args.model,
        args.rank,
        args.lora_alpha,
        device.clone(),
        SEED,
    )?;
    if let Some(resume_path) = args.resume_lora.as_ref() {
        log::info!("Resuming LoRA weights only (no optimizer state) from {}", resume_path.display());
        model.bundle.load(resume_path, &device)?;
        model.refresh_lora_cache();
    }
    let params = model.parameters();
    log::info!("Loaded {} trainable LoRA tensors", params.len());
    if params.is_empty() {
        anyhow::bail!("No trainable parameters — ZImageModel produced empty param list");
    }

    let mut opt = AdamW::new(args.lr, 0.9, 0.999, 1e-8, 0.01);

    // ── Full resume: weights + AdamW state + step counter ────────────────
    let mut start_step: usize = 0;
    if let Some(resume_path) = args.resume_full.as_ref() {
        log::info!("Full-resume from {}", resume_path.display());
        let loaded = checkpoint::load_full(resume_path, &device)?;
        let named = model.bundle.named_parameters();
        checkpoint::apply_lora_weights(&loaded, &named)?;
        checkpoint::apply_to_optimizer(&loaded, &mut opt, &named, args.rank, args.lora_alpha)?;
        model.refresh_lora_cache();
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

    // ── Periodic-sample setup ────────────────────────────────────────────
    // Pre-encode cond/uncond prompts ONCE then drop Qwen3 from VRAM.
    let periodic = args.sample_every > 0;
    let (sample_cap, sample_cap_mask, sample_uncond, sample_uncond_mask, sample_vae_path) = if periodic {
        let qwen3_path = args.sample_qwen3.as_ref()
            .ok_or_else(|| anyhow::anyhow!("--sample-every > 0 requires --sample-qwen3"))?;
        let tok_path = args.sample_tokenizer.as_ref()
            .ok_or_else(|| anyhow::anyhow!("--sample-every > 0 requires --sample-tokenizer"))?;
        let vae_path = args.sample_vae.as_ref()
            .ok_or_else(|| anyhow::anyhow!("--sample-every > 0 requires --sample-vae"))?
            .clone();
        log::info!("[sample-setup] loading Qwen3 to encode sample prompt once...");
        let qwen_w = if qwen3_path.is_file() {
            flame_core::serialization::load_file(qwen3_path, &device)?
        } else {
            let mut all = std::collections::HashMap::new();
            for entry in std::fs::read_dir(qwen3_path)? {
                let p = entry?.path();
                if p.extension().and_then(|e| e.to_str()) == Some("safetensors") {
                    let part = flame_core::serialization::load_file(&p, &device)?;
                    all.extend(part);
                }
            }
            all
        };
        let mut qcfg = Qwen3Encoder::config_from_weights(&qwen_w)?;
        qcfg.extract_layers = vec![34]; // Z-Image canonical (matches prepare_zimage)
        let qwen = Qwen3Encoder::new(qwen_w, qcfg, device.clone());
        let tok = tokenizers::Tokenizer::from_file(tok_path)
            .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
        let encode = |prompt: &str| -> anyhow::Result<(Tensor, Tensor)> {
            let wrapped = format!("{ZIMAGE_TEMPLATE_PRE}{}{ZIMAGE_TEMPLATE_POST}", prompt.trim());
            let enc = tok.encode(wrapped.as_str(), false)
                .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
            let mut ids: Vec<i32> = enc.get_ids().iter().map(|&i| i as i32).collect();
            let real_len = ids.len().min(ZIMAGE_TXT_PAD_LEN);
            ids.resize(ZIMAGE_TXT_PAD_LEN, ZIMAGE_PAD_TOKEN_ID);
            let hidden = qwen.encode(&ids)?.to_dtype(DType::BF16)?;
            let mut mask_data = vec![0f32; ZIMAGE_TXT_PAD_LEN];
            for slot in mask_data.iter_mut().take(real_len) { *slot = 1.0; }
            let mask = Tensor::from_vec(mask_data, Shape::from_dims(&[1, ZIMAGE_TXT_PAD_LEN]), device.clone())?
                .to_dtype(DType::BF16)?;
            Ok((hidden, mask))
        };
        let (cap, cap_mask) = encode(&args.sample_prompt)?;
        let (unc, unc_mask) = encode(&args.sample_neg_prompt)?;
        log::info!("[sample-setup] cap={:?} uncond={:?}; dropping Qwen3", cap.shape().dims(), unc.shape().dims());
        drop(qwen);
        flame_core::cuda_alloc_pool::clear_pool_cache();
        flame_core::trim_cuda_mempool(0);
        log::info!("[sample-setup] periodic sample enabled (every {} steps).", args.sample_every);
        (Some(cap), Some(cap_mask), Some(unc), Some(unc_mask), Some(vae_path))
    } else {
        (None, None, None, None, None)
    };

    // Step-0 baseline (LoRA-init = base output)
    if periodic {
        let out_path = args.output_dir.join("sample_step0_base.png");
        log::info!("[sample step=0] BASELINE → {}", out_path.display());
        if let Err(e) = zimage_sampler::sample_image(
            &mut model,
            sample_cap.as_ref().unwrap(),
            sample_cap_mask.as_ref(),
            sample_uncond.as_ref(),
            sample_uncond_mask.as_ref(),
            args.sample_size, args.sample_size,
            args.sample_steps,
            args.sample_cfg,
            args.sample_shift,
            args.sample_seed,
            sample_vae_path.as_ref().unwrap(),
            &out_path,
            &device,
        ) {
            log::warn!("[sample step=0] failed: {e}");
        }
    }

    let t_start = std::time::Instant::now();
    let mut total_loss = 0f32;
    let total_steps = args.steps;
    if start_step >= total_steps {
        anyhow::bail!("start_step {} >= --steps {}", start_step, total_steps);
    }

    for step in start_step..total_steps {
        // Stack `batch_size` cached samples (matches upstream Python klein9b/zimage preset = batch=2).
        let bs = args.batch_size.max(1);
        let mut latents_raw = Vec::with_capacity(bs);
        let mut caps = Vec::with_capacity(bs);
        let mut masks = Vec::with_capacity(bs);
        for b in 0..bs {
            let cache_idx = (step * bs + b) % cache_files.len();
            let s = flame_core::serialization::load_file(&cache_files[cache_idx], &device)?;
            latents_raw.push(s.get("latent")
                .ok_or_else(|| anyhow::anyhow!("cache {cache_idx} missing 'latent'"))?
                .to_dtype(DType::BF16)?);
            caps.push(s.get("text_embedding")
                .ok_or_else(|| anyhow::anyhow!("cache {cache_idx} missing 'text_embedding'"))?
                .to_dtype(DType::BF16)?);
            if let Some(m) = s.get("text_mask") {
                masks.push(Some(m.to_dtype(DType::BF16)?));
            } else {
                masks.push(None);
            }
        }
        let raw_latent = if bs == 1 { latents_raw.into_iter().next().unwrap() }
            else { Tensor::cat(&latents_raw.iter().collect::<Vec<_>>(), 0)? };
        let latent = raw_latent.add_scalar(-ZIMAGE_VAE_SHIFT)?.mul_scalar(ZIMAGE_VAE_SCALE)?;
        let cap_feats = if bs == 1 { caps.into_iter().next().unwrap() }
            else { Tensor::cat(&caps.iter().collect::<Vec<_>>(), 0)? };
        let cap_mask = if masks.iter().all(|m| m.is_some()) {
            let ms: Vec<Tensor> = masks.into_iter().map(|m| m.unwrap()).collect();
            Some(if bs == 1 { ms.into_iter().next().unwrap() }
                 else { Tensor::cat(&ms.iter().collect::<Vec<_>>(), 0)? })
        } else { None };

        // LOGIT_NORMAL continuous timestep ∈ [0, NUM_TRAIN_TIMESTEPS), then floor → sigma idx.
        let t_continuous = sample_timestep_logit_normal(&mut rng);
        let sigma_idx = (t_continuous.floor() as usize).min(NUM_TRAIN_TIMESTEPS - 1);
        let sigma = (sigma_idx + 1) as f32 / NUM_TRAIN_TIMESTEPS as f32;
        // Z-Image's training-time `t` to the model is `1 - sigma` (per zimage_sampler).
        let t_value = 1.0 - sigma;

        let noise = Tensor::randn(latent.shape().clone(), 0.0, 1.0, device.clone())?
            .to_dtype(DType::BF16)?;
        let noisy = noise.mul_scalar(sigma)?
            .add(&latent.mul_scalar(1.0 - sigma)?)?;
        // Rectified-flow target: pred ≈ -velocity where velocity = noise - clean.
        // Caller of model.forward in the sampler does `pred * -1`, so the trained
        // pred is `clean - noise`. Train against that.
        let target = latent.sub(&noise)?;
        let timestep = Tensor::from_vec(
            vec![t_value], Shape::from_dims(&[1]), device.clone(),
        )?.to_dtype(DType::BF16)?;

        if step == 0 {
            log::info!("step 0 | latent={:?} cap={:?} sigma={:.4} (idx={})",
                latent.shape().dims(), cap_feats.shape().dims(), sigma, sigma_idx);
        }

        let pred = model.forward(&noisy, &timestep, &cap_feats, cap_mask.as_ref())?;

        if pred.shape().dims() != target.shape().dims() {
            anyhow::bail!("pred {:?} != target {:?}", pred.shape().dims(), target.shape().dims());
        }

        let diff = pred.to_dtype(DType::F32)?.sub(&target.to_dtype(DType::F32)?)?;
        let loss = diff.square()?.mean()?;
        let loss_val = loss.to_vec()?[0];
        total_loss += loss_val;

        let grads = loss.backward()?;
        // OT default: clip_grad_norm = 1.0. Mirrors train_ernie.rs.
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
        {
            let _g = AutogradContext::no_grad();
            opt.step(&params)?;
            opt.zero_grad(&params);
        }
        AutogradContext::clear();
        model.refresh_lora_cache();

        if step == 0 || (step + 1) % 10 == 0 {
            let avg = total_loss / (step + 1) as f32;
            let elapsed = t_start.elapsed().as_secs_f32();
            let sps = (step + 1) as f32 / elapsed.max(0.001);
            log::info!("step {}/{} | loss={:.4} avg={:.4} | {:.2} step/s",
                step + 1, args.steps, loss_val, avg, sps);
        }

        // ── Periodic save (independent of sample) ──────────────────────
        let step_num = step + 1;
        let save_now = (periodic && step_num % args.sample_every == 0
            || args.save_every > 0 && step_num % args.save_every == 0)
            && step_num < args.steps;
        if save_now {
            let mid_ckpt = args.output_dir.join(format!("zimage_lora_step{step_num}.safetensors"));
            if save_mode_full {
                let header = CkptHeader::from_adamw(
                    "train_zimage",
                    step_num as u64,
                    &opt,
                    args.rank,
                    args.lora_alpha,
                    SEED,
                    String::new(),
                );
                let named = model.bundle.named_parameters();
                if let Err(e) = checkpoint::save_full(&mid_ckpt, &named, &opt, &header) {
                    log::warn!("[mid-save step {step_num}] full save failed: {e}");
                }
            } else {
                if let Err(e) = model.bundle.save(&mid_ckpt) {
                    log::warn!("[mid-save step {step_num}] weights save failed: {e}");
                } else {
                    log::info!("[mid-save step {step_num}] {}", mid_ckpt.display());
                }
            }
        }
        // ── Periodic inline sample ─────────────────────────────────────
        if periodic && step_num % args.sample_every == 0 && step_num < args.steps {
            let sample_out = args.output_dir.join(format!("sample_step{step_num}.png"));
            log::info!("[sample step={step_num}] → {}", sample_out.display());
            if let Err(e) = zimage_sampler::sample_image(
                &mut model,
                sample_cap.as_ref().unwrap(),
                sample_cap_mask.as_ref(),
                sample_uncond.as_ref(),
                sample_uncond_mask.as_ref(),
                args.sample_size, args.sample_size,
                args.sample_steps,
                args.sample_cfg,
                args.sample_shift,
                args.sample_seed,
                sample_vae_path.as_ref().unwrap(),
                &sample_out,
                &device,
            ) {
                log::warn!("[sample step={step_num}] failed: {e}");
            }
        }
    }

    let trained = total_steps - start_step;
    let avg_loss = if trained > 0 { total_loss / trained as f32 } else { 0.0 };
    log::info!("Training complete: {trained} new steps (total step={total_steps}), avg loss={:.4}", avg_loss);

    let ckpt = args.output_dir.join(format!("zimage_lora_{}steps.safetensors", args.steps));
    if save_mode_full {
        let header = CkptHeader::from_adamw(
            "train_zimage",
            total_steps as u64,
            &opt,
            args.rank,
            args.lora_alpha,
            SEED,
            String::new(),
        );
        let named = model.bundle.named_parameters();
        checkpoint::save_full(&ckpt, &named, &opt, &header)?;
    } else {
        model.save_weights(&ckpt)?;
    }
    log::info!("Saved checkpoint to {}", ckpt.display());

    // Final sample
    if periodic {
        let sample_out = args.output_dir.join(format!("sample_step{}.png", args.steps));
        log::info!("[sample step={} FINAL] → {}", args.steps, sample_out.display());
        if let Err(e) = zimage_sampler::sample_image(
            &mut model,
            sample_cap.as_ref().unwrap(),
            sample_cap_mask.as_ref(),
            sample_uncond.as_ref(),
            sample_uncond_mask.as_ref(),
            args.sample_size, args.sample_size,
            args.sample_steps,
            args.sample_cfg,
            args.sample_shift,
            args.sample_seed,
            sample_vae_path.as_ref().unwrap(),
            &sample_out,
            &device,
        ) {
            log::warn!("[sample final] failed: {e}");
        }
    }
    Ok(())
}
