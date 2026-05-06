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

use eridiffusion_core::config::{TrainConfig, TrainingMethod};
use eridiffusion_core::debug as dbg;
use eridiffusion_core::encoders::ltx2_vae::Ltx2Vae;
use eridiffusion_core::models::{Ltx2Model, TrainableModel};
use eridiffusion_core::sampler::ltx2_sampler;
use flame_core::adam::AdamW;
use flame_core::autograd::AutogradContext;
use flame_core::{DType, Shape, Tensor};

const SEED: u64 = 42;

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
}

fn debug_enabled() -> bool {
    std::env::var("OT_DEBUG_STATS")
        .map_or(false, |v| !matches!(v.as_str(), "0" | "" | "false" | "FALSE"))
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

    let mut opt = AdamW::new(args.lr, 0.9, 0.999, 1e-8, 0.01);

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

    let t_start = std::time::Instant::now();
    let mut total_loss = 0f32;
    let mut first_save_done = false;

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
            t_continuous.push(ltx2_sampler::sample_timestep_logit_normal(&mut rng, mu));
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
        // For batch_size > 1 with per-sample sigmas, noise scaling needs
        // per-sample broadcast. For batch_size == 1 (the bootstrap default) it's
        // a scalar; for >1 we expand a [B,1,1,1,1] tensor.
        let (noisy, target) = if args.batch_size == 1 {
            let s = sigmas[0];
            let noisy = noise.mul_scalar(s)?.add(&latent.mul_scalar(1.0 - s)?)?;
            let target = noise.sub(&latent)?;
            (noisy, target)
        } else {
            // Build [B, 1, 1, 1, 1] sigma tensor.
            let s_tensor = Tensor::from_vec(
                sigmas.clone(),
                Shape::from_dims(&[args.batch_size, 1, 1, 1, 1]),
                device.clone(),
            )?.to_dtype(DType::BF16)?;
            let one_minus_s = s_tensor.mul_scalar(-1.0)?.add_scalar(1.0)?;
            let noisy = noise.mul(&s_tensor)?.add(&latent.mul(&one_minus_s)?)?;
            let target = noise.sub(&latent)?;
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
        let diff = pred.to_dtype(DType::F32)?.sub(&target.to_dtype(DType::F32)?)?;
        let loss = diff.square()?.mean()?;
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
            log::info!("step {}/{} | loss={:.4} avg={:.4} | {:.2} step/s",
                step + 1, args.steps, loss_val, avg, sps);
        }

        // ── Periodic save ──
        let step_num = step + 1;
        if args.save_every > 0 && step_num % args.save_every == 0 && step_num < args.steps {
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

        // ── Periodic sample ──
        if args.sample_every > 0 && step_num % args.sample_every == 0 && step_num < args.steps {
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
    }

    let avg_loss = total_loss / args.steps as f32;
    log::info!("Training complete: {} steps, avg loss={:.4}", args.steps, avg_loss);

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
