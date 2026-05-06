//! train_qwenimage — Qwen-Image-2512 LoRA training binary.
//!
//! Patterned after `train_ernie`. Loads cached `(latent, text_embedding)`
//! samples from `prepare_qwenimage` and runs flow-matching LoRA training.
//!
//! Reference: `flame-diffusion/qwenimage-trainer/src/{main,pipeline}.rs`,
//! `musubi-tuner/qwen_image_train_network.py`.
//!
//! Schedule: logit-normal then qwen_shift remap
//!   `t = sigmoid(z); t_shifted = t * shift / (1 + (shift - 1) * t)`
//!   shift comes from `shift_for_resolution([w, h])` (linear-mu over image
//!   seq-len anchors 256→0.5, 8192→0.9, exp).
//!
//! Loss: MSE in F32 between pred and target = noise - latent.

use clap::Parser;
use flame_core::{autograd::AutogradContext, DType, Shape, Tensor};
use flame_core::adam::AdamW;
use flame_core::gradient_clip::GradientClipper;
use eridiffusion_core::encoders::qwen25vl::Qwen25VLEncoder;
use eridiffusion_core::models::{qwenimage as qwen_model, QwenImageTrainingModel};
use eridiffusion_core::sampler::qwenimage_sampler;
use eridiffusion_core::training::checkpoint::{self, CkptHeader};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::path::PathBuf;

const SEED_DEFAULT: u64 = 42;
const QWEN_PAD_ID: i32 = 151643;

#[derive(Parser)]
struct Args {
    /// Qwen-Image-2512 transformer directory (the `transformer/` subdir, with
    /// `diffusion_pytorch_model-0000{N}-of-00009.safetensors` shards).
    #[arg(long)] model: PathBuf,
    /// Cache dir produced by `prepare_qwenimage`.
    #[arg(long)] cache_dir: PathBuf,
    #[arg(long, default_value = "3000")] steps: usize,
    #[arg(long, default_value = "16")] rank: usize,
    #[arg(long, default_value = "16.0")] lora_alpha: f32,
    /// OneTrainer "qwen LoRA 24GB" preset default = 3e-4.
    #[arg(long, default_value = "3e-4")] lr: f32,
    /// Resolution at which the cache was prepared (used for qwen_shift).
    #[arg(long, default_value = "512")] resolution: usize,
    #[arg(long, default_value = "200")] warmup_steps: usize,
    /// Optional fixed shift (overrides resolution-based qwen_shift).
    #[arg(long)] qwen_shift: Option<f32>,
    /// Resume LoRA weights only (no optimizer state).
    #[arg(long, conflicts_with = "resume_full")] resume_lora: Option<PathBuf>,
    /// Full resume: LoRA + AdamW + step.
    #[arg(long, conflicts_with = "resume_lora")] resume_full: Option<PathBuf>,
    /// Save mode: `full` (LoRA + AdamW + step) or `weights` (legacy).
    #[arg(long, default_value = "full")] save_mode: String,
    #[arg(long, default_value = "0")] save_every: usize,
    #[arg(long, default_value = "output")] output_dir: PathBuf,
    #[arg(long, default_value_t = SEED_DEFAULT)] seed: u64,

    // ── Periodic-sample (mirrors train_ernie pattern) ─────────────────────
    /// Render a sample every N steps. `0` disables. ALWAYS renders a
    /// step-0 baseline (LoRA = identity, so this captures base-model output)
    /// when > 0, plus every N steps thereafter, plus a final sample at the
    /// end of training. Per-sample cost: ~30s + denoise time.
    #[arg(long, default_value = "0")] sample_every: usize,
    #[arg(long, default_value = "")] sample_prompt: String,
    #[arg(long, default_value = "")] sample_neg_prompt: String,
    /// `qwen_image_vae.safetensors` (wan21 internal-key format) for the
    /// in-process VAE decode. Required if --sample-every > 0.
    #[arg(long)] sample_vae: Option<PathBuf>,
    /// Qwen2.5-VL text encoder dir (or single combined safetensors).
    /// Required if --sample-every > 0.
    #[arg(long)] sample_text_encoder: Option<PathBuf>,
    /// `tokenizer.json` for Qwen2.5-VL. Required if --sample-every > 0.
    #[arg(long)] sample_tokenizer: Option<PathBuf>,
    #[arg(long, default_value = "512")] sample_size: usize,
    #[arg(long, default_value = "20")] sample_steps: usize,
    #[arg(long, default_value = "4.0")] sample_cfg: f32,
    #[arg(long, default_value = "42")] sample_seed: u64,
    #[arg(long, default_value_t = 512)] sample_max_text_len: usize,
}

/// Self-adjusting shift based on image sequence length.
/// musubi-tuner `qwen_image_utils.py:956-959` — anchors 256→0.5, 8192→0.9, exp.
fn shift_for_resolution(resolution: [usize; 2]) -> f32 {
    const VAE_SCALE: usize = 8;
    const PATCH_SIZE: usize = 2;
    const BASE_SEQ_LEN: f32 = 256.0;
    const MAX_SEQ_LEN: f32 = 8192.0;
    const BASE_SHIFT: f32 = 0.5;
    const MAX_SHIFT: f32 = 0.9;
    let [w, h] = resolution;
    let seq_len = ((h / VAE_SCALE / PATCH_SIZE) * (w / VAE_SCALE / PATCH_SIZE)) as f32;
    let m = (MAX_SHIFT - BASE_SHIFT) / (MAX_SEQ_LEN - BASE_SEQ_LEN);
    let b = BASE_SHIFT - m * BASE_SEQ_LEN;
    let mu = seq_len * m + b;
    mu.exp()
}

/// Format seconds as `HH:MM:SS` (< 24h) or `HHH:MM:SS` (≥ 24h).
fn format_elapsed(secs: f32) -> String {
    let total = secs.max(0.0) as u64;
    let h = total / 3600;
    let m = (total % 3600) / 60;
    let s = total % 60;
    if h >= 24 {
        format!("{:03}:{:02}:{:02}", h, m, s)
    } else {
        format!("{:02}:{:02}:{:02}", h, m, s)
    }
}

/// Sample logit-normal then apply qwen_shift remap. Matches musubi
/// `t = sigmoid(z); t = t*shift / (1 + (shift-1)*t)` byte-for-byte.
fn sample_timestep_logit_normal_qwenshift(rng: &mut StdRng, shift: f32) -> f32 {
    use rand_distr::{Distribution, Normal};
    let normal = Normal::new(0.0f32, 1.0f32).unwrap();
    let z = normal.sample(rng);
    let t = 1.0 / (1.0 + (-z).exp());
    shift * t / (1.0 + (shift - 1.0) * t)
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let args = Args::parse();
    flame_core::config::set_default_dtype(DType::BF16);
    let device = flame_core::global_cuda_device();

    let shift = args.qwen_shift.unwrap_or_else(|| {
        shift_for_resolution([args.resolution, args.resolution])
    });
    log::info!(
        "[train_qwenimage] model={}, cache={}, steps={}, rank={}, alpha={}, lr={}, res={}², shift={:.3}",
        args.model.display(), args.cache_dir.display(),
        args.steps, args.rank, args.lora_alpha, args.lr, args.resolution, shift,
    );

    // ── Sample-setup MUST run before DiT load ────────────────────────────
    // Qwen2.5-VL is ~14 GB BF16 on GPU. After the DiT (60-block BlockOffloader
    // + activation pool) loads, GPU has ~5 GB free which can't fit the TE.
    // So pre-encode prompts NOW (TE only resident on GPU), drop the TE, then
    // load DiT into the freed memory. Mirrors train_ernie with order swapped.
    let periodic_sample = args.sample_every > 0;
    let (sample_cond, sample_uncond, sample_vae_path) = if periodic_sample {
        let te_path = args.sample_text_encoder.as_ref()
            .ok_or_else(|| anyhow::anyhow!("--sample-every > 0 requires --sample-text-encoder"))?;
        let tok_path = args.sample_tokenizer.as_ref()
            .ok_or_else(|| anyhow::anyhow!("--sample-every > 0 requires --sample-tokenizer"))?;
        let vae_path = args.sample_vae.as_ref()
            .ok_or_else(|| anyhow::anyhow!("--sample-every > 0 requires --sample-vae"))?
            .clone();

        log::info!("[sample-setup] loading Qwen2.5-VL + tokenizer for prompt pre-encode...");
        let tok = tokenizers::Tokenizer::from_file(tok_path)
            .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
        let te_weights = load_te_weights(te_path, &device)?;
        let te_cfg = Qwen25VLEncoder::config_from_weights(&te_weights)?;
        let te = Qwen25VLEncoder::new(te_weights, te_cfg, device.clone());

        let encode_one = |text: &str| -> anyhow::Result<Tensor> {
            let enc = tok.encode(text, true)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let mut ids: Vec<i32> = enc.get_ids().iter().map(|&i| i as i32).collect();
            if ids.len() > args.sample_max_text_len { ids.truncate(args.sample_max_text_len); }
            ids.resize(args.sample_max_text_len, QWEN_PAD_ID);
            Ok(te.encode(&ids)?.to_dtype(DType::BF16)?)
        };
        let cond = encode_one(&args.sample_prompt)?;
        let uncond = if args.sample_cfg > 1.0 {
            Some(encode_one(&args.sample_neg_prompt)?)
        } else {
            None
        };
        log::info!("[sample-setup] dropping text encoder; cond={:?}{}",
            cond.shape().dims(),
            uncond.as_ref().map(|u| format!(", uncond={:?}", u.shape().dims())).unwrap_or_default(),
        );
        drop(te);
        flame_core::cuda_alloc_pool::clear_pool_cache();
        flame_core::trim_cuda_mempool(0);
        log::info!("[sample-setup] periodic sample enabled (every {} steps).", args.sample_every);
        (Some(cond), uncond, Some(vae_path))
    } else {
        (None, None, None)
    };

    log::info!("Loading Qwen-Image transformer...");
    let mut model = QwenImageTrainingModel::load(
        &args.model, args.rank, args.lora_alpha, /*full_finetune*/ false,
        device.clone(), args.seed,
    )?;

    // Activation offload pool — sized for 24 GB GPU + 62 GB system after
    // accounting for BlockOffloader's ~38 GB pinned weights. Headroom for
    // the activation pool is ~22 GB pinned.
    //
    // Per-slot size (BF16): max_seq * MLP_HIDDEN * 2
    //   max_seq = pack_h * pack_w + 512 = 1024 + 512 = 1536 at 512²
    //   slot = 1536 * 12288 * 2 = ~38 MB
    //
    // Budget: 8 slots/block × 60 blocks + 32 buffer = 512 slots total.
    //   No compression: 512 × 38 MB ≈ 19 GB pinned.
    //   FP8 compression: ~9.5 GB pinned (effectively 1024-slot equivalent).
    //
    // Saved tensors that don't fit auto-fall-back to recompute via the
    // graceful path (autograd.rs:1795) — slower but no crash. The chroma
    // pattern's slots_per_block=130 was 16× over budget and OOM'd 62 GB
    // system RAM hard last attempt.
    {
        use eridiffusion_core::training::offload::{setup_activation_offload, OffloadConfig};
        use flame_core::activation_offload::OffloadCompression;
        let pack_h = args.resolution / 8 / 2;
        let pack_w = args.resolution / 8 / 2;
        let max_seq = pack_h * pack_w + 512;
        let max_activation_bytes = max_seq * qwen_model::MLP_HIDDEN * 2;
        // FP8 mode allocates per-slot **GPU** staging buffers for the
        // quant kernel (activation_offload.rs:367). Each slot costs:
        //   pinned: max_bytes/2 (FP8) ≈ 19 MB
        //   GPU staging: max_bytes (BF16) ≈ 38 MB
        //
        // Closure-capture leak fix in qwenimage.rs freed the previous
        // ~38 GB of weight-Arc GPU pressure, so we can fit a bigger pool
        // now. Bumped from 4→8 slots/block for ~2× more offload coverage:
        //   512 slots × 38 MB GPU staging ≈ 19 GB GPU
        //   512 slots × 19 MB pinned       ≈ 10 GB pinned
        // BlockOffloader 2 blocks GPU = 1.3 GB, leaves ~3-4 GB GPU for
        // activations / LoRA state. Tight but workable on 24 GB.
        // If this OOMs, drop back to 4. If it doesn't OOM and pool still
        // fills (warns about exhaustion), try 12.
        let slots_per_block = 8;
        let total_slots = qwen_model::NUM_LAYERS * slots_per_block + 32;
        let cfg = OffloadConfig {
            num_blocks: qwen_model::NUM_LAYERS,
            max_activation_bytes,
            // FP8 compression halves pinned bytes per slot. Saved tensor
            // values are 8-bit quant on push, dequant on pop. Roughly 2x
            // effective slot count vs None at the same pinned budget.
            compression: OffloadCompression::FP8,
            extra_slots: total_slots - qwen_model::NUM_LAYERS,
        };
        match setup_activation_offload(&device, &cfg) {
            Ok((slots, bytes)) => log::info!(
                "[activation_offload] {slots} slots, {:.1} GB pinned (FP8), slot={:.1}MB raw",
                bytes as f64 / 1e9, max_activation_bytes as f64 / 1e6
            ),
            Err(e) => log::warn!(
                "[activation_offload] setup failed ({e}); falling back to recompute checkpoint (slower)"
            ),
        }
    }
    let params = model.parameters();
    log::info!("Loaded {} trainable LoRA tensors", params.len());
    if params.is_empty() {
        anyhow::bail!("No trainable parameters — model produced empty param list");
    }

    // Cache enumeration.
    std::fs::create_dir_all(&args.output_dir)?;
    let mut cache_files: Vec<PathBuf> = std::fs::read_dir(&args.cache_dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("safetensors"))
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
        let named = qwen_named_parameters(&model);
        checkpoint::apply_lora_weights(&loaded, &named)?;
    } else if let Some(path) = &args.resume_full {
        log::info!("Full-resume from {}", path.display());
        let loaded = checkpoint::load_full(path, &device)?;
        let named = qwen_named_parameters(&model);
        checkpoint::apply_lora_weights(&loaded, &named)?;
        checkpoint::apply_to_optimizer(&loaded, &mut optimizer, &named, args.rank, args.lora_alpha)?;
        start_step = loaded.header.step as usize;
        if start_step >= args.steps {
            log::warn!("Resumed step {} >= --steps {}; nothing to do.", start_step, args.steps);
            return Ok(());
        }
        log::info!("Continuing from step {}/{}", start_step, args.steps);
    }

    // Step-0 baseline sample (LoRA-init = base model output). Only meaningful
    // when starting fresh (start_step == 0). The sample-setup ran already
    // (before DiT load) — `sample_cond`/`sample_uncond`/`sample_vae_path`
    // are bound from there.
    if periodic_sample && start_step == 0 {
        let cond = sample_cond.as_ref().unwrap();
        let uncond = sample_uncond.as_ref();
        let vae_path = sample_vae_path.as_ref().unwrap();
        let out_path = args.output_dir.join("sample_step0_base.png");
        log::info!("[sample step=0] BASELINE → {}", out_path.display());
        if let Err(e) = qwenimage_inline_sample(
            &mut model, cond, uncond, vae_path, &out_path,
            args.sample_size, args.sample_steps, args.sample_cfg, args.sample_seed, &device,
        ) {
            log::warn!("[sample step=0] failed: {e}");
        }
    }

    let clipper = GradientClipper::clip_by_norm(1.0);
    let mut rng = StdRng::seed_from_u64(args.seed);
    let mut total_loss = 0.0f32;
    let t_start = std::time::Instant::now();

    log::info!("Training {} steps from step={}", args.steps, start_step);
    for step in start_step..args.steps {
        // LR warmup.
        let current_lr = if step < args.warmup_steps {
            args.lr * (step as f32 + 1.0) / args.warmup_steps as f32
        } else {
            args.lr
        };
        optimizer.set_lr(current_lr);

        // Load one cached sample.
        let idx = step % cache_files.len();
        let tensors = flame_core::serialization::load_file(&cache_files[idx], &device)?;
        let latent = tensors.get("latent")
            .ok_or_else(|| anyhow::anyhow!("cache missing 'latent'"))?
            .to_dtype(DType::BF16)?;
        let txt_embed = tensors.get("text_embedding")
            .ok_or_else(|| anyhow::anyhow!("cache missing 'text_embedding'"))?
            .to_dtype(DType::BF16)?;

        let lat_dims = latent.shape().dims().to_vec();
        let (b, _c, latent_h, latent_w) = (lat_dims[0], lat_dims[1], lat_dims[2], lat_dims[3]);
        let _ = b;

        // Sample timestep with qwen_shift.
        let sigma = sample_timestep_logit_normal_qwenshift(&mut rng, shift);
        let timestep = Tensor::from_vec(vec![sigma], Shape::from_dims(&[1]), device.clone())?
            .to_dtype(DType::BF16)?;

        // Flow-matching: x_t = (1 - sigma) * latent + sigma * noise; target = noise - latent.
        let noise = Tensor::randn(latent.shape().clone(), 0.0, 1.0, device.clone())?
            .to_dtype(DType::BF16)?;
        let xt = latent.mul_scalar(1.0 - sigma)?.add(&noise.mul_scalar(sigma)?)?;
        let target = noise.sub(&latent)?;

        // Pack [B, 16, H, W] → [B, H/2 * W/2, 64] for forward.
        let xt_packed = qwen_model::pack_latents(&xt)?;
        let target_packed = qwen_model::pack_latents(&target)?;

        // Forward.
        AutogradContext::clear();
        let pred = model.forward(&xt_packed, &timestep, &txt_embed, latent_h, latent_w)?;

        // MSE loss in F32.
        let pred_f32 = pred.to_dtype(DType::F32)?;
        let target_f32 = target_packed.to_dtype(DType::F32)?;
        let diff = pred_f32.sub(&target_f32)?;
        let sq = diff.mul(&diff)?;
        let loss = sq.mean()?;

        let loss_val: f32 = loss.to_vec()?.first().copied().unwrap_or(f32::NAN);
        if !loss_val.is_finite() {
            anyhow::bail!("step {}: non-finite loss {}", step + 1, loss_val);
        }
        total_loss += loss_val;

        // Backward.
        let grads = loss.backward()?;
        for param in &params {
            if let Some(g) = grads.get(param.id()) {
                let g = if g.dtype() == DType::F32 { g.clone() } else { g.to_dtype(DType::F32)? };
                param.set_grad(g)?;
            }
        }

        // Clip + step.
        let grad_norm = {
            let mut grad_tensors: Vec<Tensor> = Vec::new();
            let mut owners: Vec<usize> = Vec::new();
            for (i, param) in params.iter().enumerate() {
                if let Some(g) = param.grad() {
                    grad_tensors.push(g);
                    owners.push(i);
                }
            }
            let mut grad_refs: Vec<&mut Tensor> = grad_tensors.iter_mut().collect();
            let norm = clipper.clip_grads(&mut grad_refs)?;
            for (owner, grad) in owners.into_iter().zip(grad_tensors.into_iter()) {
                params[owner].set_grad(grad)?;
            }
            norm
        };

        {
            let _guard = AutogradContext::no_grad();
            optimizer.step(&params)?;
            optimizer.zero_grad(&params);
            model.refresh_lora_cache();
        }
        AutogradContext::clear();
        flame_core::cuda_alloc_pool::clear_pool_cache();

        // Per-step log (matches the readable zimage_lora_train format the user
        // approved, with elapsed in HH:MM:SS / HHH:MM:SS for >24h runs).
        let step_num = step + 1;
        let avg = total_loss / (step_num - start_step) as f32;
        let elapsed = t_start.elapsed().as_secs_f32();
        let done = (step_num - start_step) as f32;
        let s_per_step = elapsed / done.max(1.0);
        let remaining = (args.steps - step_num) as f32 * s_per_step;
        log::info!(
            "step {}/{} | loss={:.4} avg={:.4} | grad_norm={:.4} | lr={:.2e} | {:.2}s/step | elapsed {} | ETA {}",
            step_num, args.steps, loss_val, avg, grad_norm, current_lr,
            s_per_step,
            format_elapsed(elapsed),
            format_elapsed(remaining),
        );

        if args.save_every > 0 && step_num % args.save_every == 0 && step_num < args.steps {
            let path = args.output_dir.join(format!("qwenimage_lora_step{}.safetensors", step_num));
            save_ckpt(&path, &model, &optimizer, args.rank, args.lora_alpha, args.seed, &args.save_mode, step_num)?;
        }

        // Periodic inline sample. After the sample returns, drop the
        // sampler's transient GPU buffers (VAE decoder + Euler-loop scratch)
        // before resuming training, otherwise the next forward step OOMs at
        // mid-block prefetch (verified at the 15-step smoke 2026-05-06).
        if periodic_sample && step_num % args.sample_every == 0 && step_num < args.steps {
            let cond = sample_cond.as_ref().unwrap();
            let uncond = sample_uncond.as_ref();
            let vae_path = sample_vae_path.as_ref().unwrap();
            let out_path = args.output_dir.join(format!("sample_step{}.png", step_num));
            log::info!("[sample step={}] → {}", step_num, out_path.display());
            if let Err(e) = qwenimage_inline_sample(
                &mut model, cond, uncond, vae_path, &out_path,
                args.sample_size, args.sample_steps, args.sample_cfg, args.sample_seed, &device,
            ) {
                log::warn!("[sample step={}] failed: {e}", step_num);
            }
            // Force release of GPU memory the sampler held (VAE weights,
            // Euler-loop scratch). flame_core's mempool keeps a release
            // threshold of MAX by default; trim back to 0 here so training
            // gets a fresh GPU budget.
            flame_core::cuda_alloc_pool::clear_pool_cache();
            flame_core::trim_cuda_mempool(0);
            AutogradContext::clear();
        }
    }

    let final_path = args.output_dir.join(format!("qwenimage_lora_{}steps.safetensors", args.steps));
    save_ckpt(&final_path, &model, &optimizer, args.rank, args.lora_alpha, args.seed, &args.save_mode, args.steps)?;

    // Final sample after training completes.
    if periodic_sample {
        let cond = sample_cond.as_ref().unwrap();
        let uncond = sample_uncond.as_ref();
        let vae_path = sample_vae_path.as_ref().unwrap();
        let out_path = args.output_dir.join(format!("sample_step{}_final.png", args.steps));
        log::info!("[sample FINAL step={}] → {}", args.steps, out_path.display());
        if let Err(e) = qwenimage_inline_sample(
            &mut model, cond, uncond, vae_path, &out_path,
            args.sample_size, args.sample_steps, args.sample_cfg, args.sample_seed, &device,
        ) {
            log::warn!("[sample FINAL] failed: {e}");
        }
    }
    let trained = args.steps - start_step;
    log::info!(
        "Training complete: {} new steps (total {}). avg loss={:.4}. Saved to {}",
        trained, args.steps,
        total_loss / trained.max(1) as f32,
        final_path.display(),
    );
    Ok(())
}

/// Build the canonical `(name, Parameter)` pairs for `checkpoint::save_full`
/// and `apply_to_optimizer`. The names match `QwenImageLoraBundle::save`
/// byte-for-byte so resumed AdamW state lines up with live params.
///
/// Iteration order: deterministic (sorted by `(block_idx, target_suffix)`).
/// Required because `HashMap` iteration is random across runs and
/// save→reload must produce a stable key→tensor mapping.
fn qwen_named_parameters(model: &QwenImageTrainingModel)
    -> Vec<(String, flame_core::parameter::Parameter)>
{
    use eridiffusion_core::models::qwenimage::QwenImageLoraBundle;
    let mut entries: Vec<((usize, &str), &eridiffusion_core::lora::LoRALinear)> = model
        .bundle
        .adapters
        .iter()
        .map(|(&(idx, target), lora)| ((idx, QwenImageLoraBundle::target_suffix(target)), lora))
        .collect();
    entries.sort_by(|a, b| a.0.cmp(&b.0));

    let mut out: Vec<(String, flame_core::parameter::Parameter)> =
        Vec::with_capacity(entries.len() * 2);
    for ((block_idx, suffix), lora) in entries {
        let prefix = format!("transformer_blocks.{block_idx}.{suffix}");
        // LoRALinear::parameters() returns [lora_a, lora_b]; the safetensors
        // save scheme is `{prefix}.lora_A.weight` then `{prefix}.lora_B.weight`.
        out.push((format!("{prefix}.lora_A.weight"), lora.lora_a().clone()));
        out.push((format!("{prefix}.lora_B.weight"), lora.lora_b().clone()));
    }
    out
}

/// In-process sample call. Pre-encoded prompts skip the (huge) text-encoder
/// load. Disables FLAME_CHECKPOINT inside the sampler scope (no autograd
/// during inference) and restores it on exit so training continues with
/// the same recompute behavior.
fn qwenimage_inline_sample(
    model: &mut QwenImageTrainingModel,
    cond: &Tensor,
    uncond: Option<&Tensor>,
    vae_path: &std::path::Path,
    out_path: &std::path::Path,
    size: usize,
    steps: usize,
    cfg: f32,
    seed: u64,
    device: &std::sync::Arc<flame_core::CudaDevice>,
) -> anyhow::Result<()> {
    qwenimage_sampler::sample_image(
        model,
        cond,
        uncond,
        size, size,
        steps,
        cfg,
        seed,
        vae_path,
        out_path,
        device,
    ).map_err(|e| anyhow::anyhow!("qwenimage sample: {e}"))?;
    Ok(())
}

/// Load Qwen2.5-VL text encoder weights from a directory of shards or a
/// single combined safetensors file.
fn load_te_weights(
    path: &std::path::Path,
    device: &std::sync::Arc<flame_core::CudaDevice>,
) -> anyhow::Result<std::collections::HashMap<String, Tensor>> {
    if path.is_file() {
        return flame_core::serialization::load_file(path, device)
            .map_err(|e| anyhow::anyhow!("text-encoder load: {e}"));
    }
    let mut all = std::collections::HashMap::new();
    for entry in std::fs::read_dir(path)
        .map_err(|e| anyhow::anyhow!("read_dir {}: {e}", path.display()))?
    {
        let p = entry.map_err(|e| anyhow::anyhow!("entry: {e}"))?.path();
        if p.extension().and_then(|e| e.to_str()) == Some("safetensors") {
            let part = flame_core::serialization::load_file(&p, device)
                .map_err(|e| anyhow::anyhow!("text-encoder shard {}: {e}", p.display()))?;
            all.extend(part);
        }
    }
    Ok(all)
}

fn save_ckpt(
    path: &std::path::Path,
    model: &QwenImageTrainingModel,
    optimizer: &AdamW,
    rank: usize,
    alpha: f32,
    seed: u64,
    mode: &str,
    step: usize,
) -> anyhow::Result<()> {
    if mode == "weights" {
        model.save_weights(path)?;
        log::info!("[save] {} (weights only)", path.display());
        return Ok(());
    }
    let header = CkptHeader::from_adamw(
        "train_qwenimage",
        step as u64,
        optimizer,
        rank,
        alpha,
        seed,
        String::new(),
    );
    let named = qwen_named_parameters(model);
    if named.is_empty() {
        // Until qwenimage gets a named_parameters() helper, fall back to
        // weights-only for the periodic save (full-resume from this file is
        // disabled by the named.is_empty() guard above).
        log::warn!("[save] qwenimage missing named_parameters; falling back to weights-only");
        model.save_weights(path)?;
        let _ = header;
        return Ok(());
    }
    checkpoint::save_full(path, &named, optimizer, &header)
        .map_err(|e| anyhow::anyhow!("save_full: {e}"))?;
    Ok(())
}
