//! train_klein — Klein 4B/9B LoRA training, mirroring upstream Python BaseFlux2Setup.
//!
//! Pipeline per step (matches OT preset `klein9b_lora_boxjana.json` defaults):
//!   1. Load cached `latent` ([1, 128, h, w] BF16, KleinVaeEncoder.encode output)
//!      and `text_embedding` ([1, 512, joint_dim] BF16).
//!   2. Sample timestep ∈ [0, num_train_timesteps) per LOGIT_NORMAL distribution
//!      with `timestep_shift=1.0` (4B+9B preset default).
//!   3. sigma = (floor(t)+1) / 1000;  noisy = noise·sigma + clean·(1-sigma).
//!   4. Forward → [1, 128, h, w]; target = noise - clean (rectified flow).
//!   5. Loss = mean MSE in F32.  clip_grad_norm = 1.0 (preset default; matches ERNIE).
//!
//! Single seed=42 (memory: feedback_default_seed_42).
//! AdamW(lr=3e-5 by default, beta=0.9/0.999, weight_decay=0.01) — matches Klein 9B preset.

use clap::Parser;
use std::path::PathBuf;
use flame_core::{autograd::AutogradContext, DType, Shape, Tensor};
use flame_core::adam::AdamW;
use eridiffusion_core::config::{TrainConfig, TrainingMethod};
use eridiffusion_core::debug as dbg;
use eridiffusion_core::encoders::qwen3::Qwen3Encoder;
use eridiffusion_core::models::{klein::KleinModel, TrainableModel};
use eridiffusion_core::sampler::klein_sampler;
use eridiffusion_core::training::checkpoint::{self, CkptHeader};

const NUM_TRAIN_TIMESTEPS: usize = 1000;
const LOGIT_NORMAL_BIAS: f32 = 0.0;
const LOGIT_NORMAL_SCALE: f32 = 1.0;
const TIMESTEP_SHIFT: f32 = 1.0;        // klein preset default
const SEED: u64 = 42;
const CLIP_GRAD_NORM: f32 = 1.0;        // klein preset default — essential for convergence

#[derive(Parser)]
struct Args {
    #[arg(long)] config: PathBuf,
    #[arg(long)] cache_dir: PathBuf,
    /// Klein transformer safetensors path. Either a directory of shards or a
    /// single-file checkpoint (e.g. `flux-2-klein-base-4b.safetensors`).
    #[arg(long)] transformer: PathBuf,
    #[arg(long, default_value = "100")] steps: usize,
    #[arg(long, default_value = "16")] rank: usize,
    #[arg(long, default_value = "16.0")] lora_alpha: f64,
    /// Klein 9B preset = 3e-5; 4B can usually take a touch higher.
    #[arg(long, default_value = "3e-5")] lr: f32,
    /// Per-step batch size — N cached samples are loaded and stacked along
    /// dim 0 each step. upstream Python's klein9b preset uses batch=2; ED-v2
    /// previously silently used batch=1 by ignoring the config field.
    #[arg(long, default_value = "1")] batch_size: usize,
    /// Resume from a saved LoRA checkpoint — overwrites freshly-init zeros
    /// after model load. Use to continue training. Optimizer state NOT resumed.
    #[arg(long, conflicts_with = "resume_full")] resume_lora: Option<PathBuf>,
    /// Full resume: LoRA weights + AdamW (m, v, t) + step counter. Refuses
    /// rank/alpha mismatch. `--steps N` is the TARGET total step.
    #[arg(long, conflicts_with = "resume_lora")] resume_full: Option<PathBuf>,
    /// Periodic + final save mode. Default `full` (LoRA + AdamW + step) for
    /// resumable runs. `weights` writes legacy weights-only files.
    #[arg(long, default_value = "full")] save_mode: String,
    #[arg(long, default_value = "output")] output_dir: PathBuf,

    // ── Periodic save + sample (every N steps) ──────────────────────────
    /// Save a LoRA checkpoint AND render a sample image every N steps.
    /// `0` disables. Default 500 — matches user's iteration cadence.
    #[arg(long, default_value = "500")] sample_every: usize,
    /// Prompt for the periodic sample. Required if `--sample-every > 0`.
    #[arg(long, default_value = "")] sample_prompt: String,
    /// Negative / unconditional prompt for CFG.
    #[arg(long, default_value = "")] sample_neg_prompt: String,
    /// Klein VAE safetensors. Required if `--sample-every > 0`.
    #[arg(long)] sample_vae: Option<PathBuf>,
    /// Qwen3 weights (single file or sharded directory). Required if `--sample-every > 0`.
    #[arg(long)] sample_qwen3: Option<PathBuf>,
    /// Qwen3 tokenizer.json. Required if `--sample-every > 0`.
    #[arg(long)] sample_tokenizer: Option<PathBuf>,
    /// Sample resolution. Default 1024² — gives the actual visual quality the
    /// model is targeted for. Klein 4B fits 1024² inference comfortably on
    /// 24 GB even with training state still resident (model ~8 GB + VAE 0.5 GB
    /// + sample intermediates 4-6 GB ≈ 14 GB peak; train intermediates are
    /// dropped under no_grad scope during the sample call).
    #[arg(long, default_value = "1024")] sample_size: usize,
    /// Denoise steps for periodic sample. Klein is guidance-distilled-ish
    /// so default is short.
    #[arg(long, default_value = "20")] sample_steps: usize,
    /// CFG scale for periodic sample. 1.0 = single forward (no CFG).
    #[arg(long, default_value = "4.0")] sample_cfg: f32,
    /// Fixed seed for periodic sample (so visual progression is comparable across steps).
    #[arg(long, default_value = "42")] sample_seed: u64,
}

/// LOGIT_NORMAL timestep sample. Returns continuous t in [0, 1000).
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

fn collect_klein_shards(path: &std::path::Path) -> anyhow::Result<Vec<PathBuf>> {
    if path.is_file() {
        return Ok(vec![path.to_path_buf()]);
    }
    let mut shards: Vec<PathBuf> = std::fs::read_dir(path)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("safetensors"))
        .collect();
    shards.sort();
    if shards.is_empty() {
        anyhow::bail!("no klein safetensors at {:?}", path);
    }
    Ok(shards)
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

    let shards = collect_klein_shards(&args.transformer)?;
    log::info!("Loading Klein transformer from {} shard(s) (rank={} alpha={})",
        shards.len(), args.rank, args.lora_alpha);
    let mut model = KleinModel::load(&shards, &config, device.clone())?;
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

    // ── Full resume: weights + AdamW state + step counter ────────────────
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

    // ── Periodic sample setup ────────────────────────────────────────────
    // If --sample-every > 0, encode the sample prompt with Qwen3 ONCE up-front
    // (and the unconditional prompt for CFG), then DROP Qwen3 from VRAM. We
    // reuse the cached embeddings every N steps. This avoids resident-Qwen3
    // memory cost during training while still giving inline visibility.
    let periodic = args.sample_every > 0;
    let (sample_cap, sample_uncond, sample_vae_path) = if periodic {
        let qwen3_path = args.sample_qwen3.as_ref()
            .ok_or_else(|| anyhow::anyhow!("--sample-every > 0 requires --sample-qwen3"))?;
        let tok_path = args.sample_tokenizer.as_ref()
            .ok_or_else(|| anyhow::anyhow!("--sample-every > 0 requires --sample-tokenizer"))?;
        let vae_path = args.sample_vae.as_ref()
            .ok_or_else(|| anyhow::anyhow!("--sample-every > 0 requires --sample-vae"))?
            .clone();
        log::info!("[sample-setup] loading Qwen3 + tokenizer to encode prompt once...");
        let qwen_w = klein_load_qwen3(qwen3_path, &device)?;
        let mut qcfg = Qwen3Encoder::config_from_weights(&qwen_w)?;
        // Klein-canonical: hidden_states[9, 18, 27] (multi-layer extract).
        qcfg.extract_layers = vec![9, 18, 27];
        let qwen = Qwen3Encoder::new(qwen_w, qcfg, device.clone());
        let tok = tokenizers::Tokenizer::from_file(tok_path)
            .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
        let cap = klein_encode_prompt(&qwen, &tok, &args.sample_prompt)?;
        let unc = klein_encode_prompt(&qwen, &tok, &args.sample_neg_prompt)?;
        log::info!("[sample-setup] cap={:?} uncond={:?}", cap.shape().dims(), unc.shape().dims());
        drop(qwen);
        flame_core::cuda_alloc_pool::clear_pool_cache();
        flame_core::trim_cuda_mempool(0);
        log::info!("[sample-setup] Qwen3 dropped; VAE will load lazily per sample. Periodic sample enabled (every {} steps).", args.sample_every);
        (Some(cap), Some(unc), Some(vae_path))
    } else {
        (None, None, None)
    };

    // ── Step-0 baseline sample (LoRA-init = base model output) ───────────
    if periodic {
        let cap = sample_cap.as_ref().unwrap();
        let unc = sample_uncond.as_ref().unwrap();
        let vae_path = sample_vae_path.as_ref().unwrap();
        let out_path = args.output_dir.join("sample_step0_base.png");
        log::info!("[sample step=0] BASELINE (LoRA at init) → {}", out_path.display());
        if let Err(e) = klein_inline_sample(
            &mut model, cap, unc, vae_path, &out_path,
            args.sample_size, args.sample_steps, args.sample_cfg, args.sample_seed,
            &device,
        ) {
            log::warn!("[sample step=0] failed: {e}");
        }
    }

    let t_start = std::time::Instant::now();
    let mut total_loss = 0f32;

    for step in start_step..args.steps {
        // Stack `batch_size` cache files. upstream Python's klein9b preset uses
        // batch=2; the previous ED-v2 impl silently loaded one sample per
        // step regardless of config, breaking apples-to-apples comparison.
        // Per-element timesteps + per-element noise — matches upstream Python
        // `ModelSetupNoiseMixin._get_timestep_discrete(batch_size=...)`.
        let bs = args.batch_size.max(1);
        let mut latents = Vec::with_capacity(bs);
        let mut txts = Vec::with_capacity(bs);
        for b in 0..bs {
            let cache_idx = (step * bs + b) % cache_files.len();
            let sample = flame_core::serialization::load_file(&cache_files[cache_idx], &device)?;
            let l = sample.get("latent")
                .ok_or_else(|| anyhow::anyhow!("cached sample {cache_idx} missing 'latent'"))?
                .to_dtype(DType::BF16)?;
            let t = sample.get("text_embedding")
                .ok_or_else(|| anyhow::anyhow!("cached sample {cache_idx} missing 'text_embedding'"))?
                .to_dtype(DType::BF16)?;
            latents.push(l);
            txts.push(t);
        }
        let latent = if bs == 1 {
            latents.into_iter().next().unwrap()
        } else {
            Tensor::cat(&latents.iter().collect::<Vec<_>>(), 0)?
        };
        let txt = if bs == 1 {
            txts.into_iter().next().unwrap()
        } else {
            Tensor::cat(&txts.iter().collect::<Vec<_>>(), 0)?
        };

        // Per-batch-element timesteps. upstream Python samples shape [B] (line
        // 99 BaseFlux2Setup.py: `batch_size=batch['latent_image'].shape[0]`).
        let mut t_per_b: Vec<f32> = Vec::with_capacity(bs);
        let mut sigma_per_b: Vec<f32> = Vec::with_capacity(bs);
        let mut t_model_per_b: Vec<f32> = Vec::with_capacity(bs);
        for _ in 0..bs {
            let t_continuous = sample_timestep_logit_normal(&mut rng);
            let sigma_idx = (t_continuous.floor() as usize).min(NUM_TRAIN_TIMESTEPS - 1);
            let sigma = (sigma_idx + 1) as f32 / NUM_TRAIN_TIMESTEPS as f32;
            t_per_b.push(t_continuous);
            sigma_per_b.push(sigma);
            t_model_per_b.push(sigma_idx as f32 / NUM_TRAIN_TIMESTEPS as f32);
        }
        // For the noise/blend math we broadcast sigma over [B, C, H, W]
        // by multiplying each batch element separately and stacking.
        let noise = Tensor::randn(latent.shape().clone(), 0.0, 1.0, device.clone())?
            .to_dtype(DType::BF16)?;
        let noisy = if bs == 1 {
            noise.mul_scalar(sigma_per_b[0])?
                .add(&latent.mul_scalar(1.0 - sigma_per_b[0])?)?
        } else {
            // Per-element scaling. Slice batch dim, scale each, re-stack.
            let mut pieces = Vec::with_capacity(bs);
            for b in 0..bs {
                let n_b = noise.narrow(0, b, 1)?;
                let l_b = latent.narrow(0, b, 1)?;
                let s = sigma_per_b[b];
                pieces.push(n_b.mul_scalar(s)?.add(&l_b.mul_scalar(1.0 - s)?)?);
            }
            Tensor::cat(&pieces.iter().collect::<Vec<_>>(), 0)?
        };
        let target = noise.sub(&latent)?;
        // timestep tensor shape [B] — model.forward broadcasts over batch.
        let timestep = Tensor::from_vec(
            t_model_per_b.clone(),
            Shape::from_dims(&[bs]),
            device.clone(),
        )?;
        let t_continuous = t_per_b[0]; // for OT_DEBUG line; per-step single number
        let sigma = sigma_per_b[0];
        let sigma_idx = (t_per_b[0].floor() as usize).min(NUM_TRAIN_TIMESTEPS - 1);

        if step == 0 {
            log::info!("step 0 | batch={} latent={:?} text={:?} sigma[0]={:.4} (idx={})",
                bs, latent.shape().dims(), txt.shape().dims(), sigma, sigma_idx);
        }

        let pred = model.forward(&noisy, &txt, &timestep)?;
        if pred.shape().dims() != target.shape().dims() {
            anyhow::bail!(
                "predicted velocity shape {:?} != target {:?}",
                pred.shape().dims(), target.shape().dims());
        }

        // F32 mean MSE — matches OT default (loss_weight_fn=CONSTANT, mse_strength=1.0).
        let diff = pred.to_dtype(DType::F32)?.sub(&target.to_dtype(DType::F32)?)?;
        let loss = diff.square()?.mean()?;
        let loss_val = loss.to_vec()?[0];
        total_loss += loss_val;

        // === OT_DEBUG_STATS-format per-step line (mirrors train_ernie + upstream Python patch) ===
        let dbg_on = dbg::enabled("OT_DEBUG_STATS");
        if dbg_on {
            let p_st = dbg::stats(&pred);
            let t_st = dbg::stats(&target);
            eprintln!(
                "[OT_DEBUG step={:5}] t={:.2} loss(pre-scale)={:.4} | pred[mean={:+.3e} std={:.3e} max|·|={:.3e}] target[mean={:+.3e} std={:.3e} max|·|={:.3e}]",
                step, t_continuous, loss_val,
                p_st.mean, p_st.std, p_st.abs_max,
                t_st.mean, t_st.std, t_st.abs_max,
            );
        }

        let grads = loss.backward()?;

        // clip_grad_norm = 1.0 (klein preset default; ERNIE memory: convergence killer if omitted).
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
        if dbg_on {
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
            let mid_ckpt = args.output_dir.join(format!("klein_lora_step{step_num}.safetensors"));
            if save_mode_full {
                let header = CkptHeader::from_adamw(
                    "train_klein", step_num as u64, &opt,
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
            if let Err(e) = klein_inline_sample(
                &mut model, cap, unc, vae_path, &sample_out,
                args.sample_size, args.sample_steps, args.sample_cfg, args.sample_seed,
                &device,
            ) {
                log::warn!("[sample step={step_num}] failed: {e}");
            }
        }
    }

    let trained = args.steps - start_step;
    let avg_loss = if trained > 0 { total_loss / trained as f32 } else { 0.0 };
    log::info!("Training complete: {trained} new steps (total={}), avg loss={:.4}", args.steps, avg_loss);

    let ckpt = args.output_dir.join(format!("klein_lora_{}steps.safetensors", args.steps));
    if save_mode_full {
        let header = CkptHeader::from_adamw(
            "train_klein", args.steps as u64, &opt,
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

    // ── Final sample at the end of training ──────────────────────────────
    if periodic {
        let cap = sample_cap.as_ref().unwrap();
        let unc = sample_uncond.as_ref().unwrap();
        let vae_path = sample_vae_path.as_ref().unwrap();
        let sample_out = args.output_dir.join(format!("sample_step{}.png", args.steps));
        log::info!("[sample step={} FINAL] → {}", args.steps, sample_out.display());
        if let Err(e) = klein_inline_sample(
            &mut model, cap, unc, vae_path, &sample_out,
            args.sample_size, args.sample_steps, args.sample_cfg, args.sample_seed,
            &device,
        ) {
            log::warn!("[sample final] failed: {e}");
        }
    }
    Ok(())
}

// ── Periodic-sample helpers ──────────────────────────────────────────────

/// Klein chat template — must match `prepare_klein` and `sample_klein`.
const KLEIN_TEMPLATE_PRE: &str = "<|im_start|>user\n";
const KLEIN_TEMPLATE_POST: &str = "<|im_end|>\n<|im_start|>assistant\n";
const KLEIN_PAD_TOKEN_ID: i32 = 151643;
const KLEIN_TXT_PAD_LEN: usize = 512;

fn klein_load_qwen3(
    path: &std::path::Path,
    device: &std::sync::Arc<flame_core::CudaDevice>,
) -> anyhow::Result<std::collections::HashMap<String, Tensor>> {
    if path.is_file() {
        return flame_core::serialization::load_file(path, device)
            .map_err(|e| anyhow::anyhow!("load_file: {e}"));
    }
    let mut all = std::collections::HashMap::new();
    for entry in std::fs::read_dir(path)? {
        let p = entry?.path();
        if p.extension().and_then(|e| e.to_str()) == Some("safetensors") {
            let part = flame_core::serialization::load_file(&p, device)
                .map_err(|e| anyhow::anyhow!("load_file {}: {e}", p.display()))?;
            all.extend(part);
        }
    }
    Ok(all)
}

/// Render one sample using the live in-training model state and pre-encoded
/// prompt embeddings. Loads + drops the VAE per call to bound VRAM.
fn klein_inline_sample(
    model: &mut KleinModel,
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
    use eridiffusion_core::encoders::vae::KleinVaeDecoder;
    let _no_grad = AutogradContext::no_grad();
    let h_lat = size / 16;
    let w_lat = size / 16;
    let n_img = h_lat * w_lat;
    let timesteps = klein_sampler::get_schedule(steps, n_img);

    use rand::SeedableRng;
    let _rng = rand::rngs::StdRng::seed_from_u64(seed);
    let latent_shape = Shape::from_dims(&[1, 128, h_lat, w_lat]);
    let mut latent = Tensor::randn(latent_shape, 0.0, 1.0, device.clone())?
        .to_dtype(DType::BF16)?;

    for step in 0..steps {
        let sigma = timesteps[step];
        let sigma_next = timesteps[step + 1];
        let t = klein_sampler::sigma_to_timestep(sigma);
        let t_tensor = Tensor::from_vec(vec![t], Shape::from_dims(&[1]), device.clone())?;
        let pred_cond = model.forward(&latent, cond, &t_tensor)?;
        let pred_uncond = model.forward(&latent, uncond, &t_tensor)?;
        let pred = pred_uncond.add(&pred_cond.sub(&pred_uncond)?.mul_scalar(cfg)?)?;
        latent = klein_sampler::euler_step(&latent, &pred, sigma, sigma_next)?;
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

fn klein_encode_prompt(
    qwen: &Qwen3Encoder,
    tok: &tokenizers::Tokenizer,
    prompt: &str,
) -> anyhow::Result<Tensor> {
    let wrapped = format!("{KLEIN_TEMPLATE_PRE}{}{KLEIN_TEMPLATE_POST}", prompt.trim());
    let enc = tok.encode(wrapped.as_str(), false)
        .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
    let mut ids: Vec<i32> = enc.get_ids().iter().map(|&i| i as i32).collect();
    ids.resize(KLEIN_TXT_PAD_LEN, KLEIN_PAD_TOKEN_ID);
    let hidden = qwen.encode(&ids)?;
    Ok(hidden.to_dtype(DType::BF16)?)
}
