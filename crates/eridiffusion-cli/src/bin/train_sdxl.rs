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
use eridiffusion_core::config::{TrainConfig, TrainingMethod};
use eridiffusion_core::models::{sdxl::SDXLModel, TrainableModel};
use eridiffusion_core::sampler::sdxl_sampler::sin_embed_256;
use eridiffusion_core::training::checkpoint::{self, CkptHeader};
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

    let shards = collect_shards(&args.unet)?;
    log::info!("[SDXL] loading UNet from {} shard(s) (rank={}, alpha={})",
        shards.len(), args.rank, args.lora_alpha);
    let mut model = SDXLModel::load(&shards, &config, device.clone())?;
    let params = model.parameters();
    log::info!("trainable LoRA tensors: {}", params.len());
    if params.is_empty() {
        anyhow::bail!("no trainable parameters — TrainingMethod::Lora produced empty list");
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

    let t_start = std::time::Instant::now();
    let mut total_loss = 0f32;

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
        let t_idx = rng.gen_range(0..NUM_TRAIN_TIMESTEPS);
        let ab = alpha_bar[t_idx];
        let sqrt_ab = ab.sqrt();
        let sqrt_1m_ab = (1.0 - ab).sqrt();

        // ε ~ N(0, I) at latent shape
        let noise = Tensor::randn(latent.shape().clone(), 0.0, 1.0, device.clone())?
            .to_dtype(DType::BF16)?;
        let noisy = latent.mul_scalar(sqrt_ab)?.add(&noise.mul_scalar(sqrt_1m_ab)?)?;
        let target = noise.clone();

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
        let diff = pred.to_dtype(DType::F32)?.sub(&target.to_dtype(DType::F32)?)?;
        let loss = diff.square()?.mean()?;
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

        {
            let _g = AutogradContext::no_grad();
            opt.step(&params)?;
            opt.zero_grad(&params);
            model.post_optimizer_step();
        }
        AutogradContext::clear();

        if step == 0 || (step + 1) % 10 == 0 {
            let avg = total_loss / (step + 1) as f32;
            let elapsed = t_start.elapsed().as_secs_f32();
            let sps = (step + 1) as f32 / elapsed.max(0.001);
            log::info!("step {}/{} | loss={:.4} avg={:.4} grad_norm={:.4} | {:.2} step/s",
                step + 1, args.steps, loss_val, avg, total_norm, sps);
        }

        let step_num = step + 1;
        if args.save_every > 0 && step_num % args.save_every == 0 && step_num < args.steps {
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
        }
    }

    let trained = args.steps - start_step;
    let avg_loss = if trained > 0 { total_loss / trained as f32 } else { 0.0 };
    log::info!("Training complete: {trained} new steps (total={}), avg loss={:.4}", args.steps, avg_loss);

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
    Ok(())
}
