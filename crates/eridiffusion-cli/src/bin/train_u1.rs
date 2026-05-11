//! train_u1 — SenseNova-U1-8B-MoT LoRA / mvp-finetune training binary.
//!
//! Mirrors `train_u1/scripts/train_bf16_offload.py` (config-driven). Two modes:
//!
//! * **Smoke mode** (no `--data-dir`): synthetic single-sample batch, used to
//!   validate the autograd chain. Run with `FLAME_ASSERT_GRAD_FLOW=1`.
//! * **Real-data mode** (`--data-dir <folder>`): scan paired `<id>.{jpg|png|webp}`
//!   + `<id>.txt` files. Each step samples one (image, caption) pair, resizes
//!   to `--image-hw`, normalizes to `[-1, 1]`, tokenizes the caption through the
//!   official chat template, and runs one FM training step:
//!     `x0 = patchify(image, p=32, channel_first=false)`
//!     `eps ~ N(0,1)`
//!     `t  ~ U(t_eps, 1]`
//!     `z_t = t*x0 + (1-t)*eps`     (computed inside forward_t2i_step)
//!     `noisy = unpatchify(z_t, p=32)`
//!     forward_t2i_step → MSE(x_pred, x0) → backward → AdamW step
//!
//! Defaults mirror Python's `TrainConfig` in `train_u1/config.py`:
//!     lr=5e-5, betas=(0.9, 0.95), seed=42, lora.preset="default" (r=64
//!     attn+mlp+fm_head), grad_accum=1, checkpoint_every=500.

use clap::Parser;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, Context};
use flame_core::{CudaDevice, DType, Shape, Tensor};
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::{AddedToken, Tokenizer};

use eridiffusion_core::models::sensenova_u1::{self, SenseNovaU1};
use eridiffusion_core::models::sensenova_u1_lora as u1lora;
use eridiffusion_core::training::training_features::{Optimizer, OptimizerKind};

const SEED_DEFAULT: u64 = 42;

const SYSTEM_MESSAGE_FOR_GEN: &str = concat!(
    "You are an image generation and editing assistant that accurately understands and executes ",
    "user intent.\n\nYou support two modes:\n\n",
    "1. Think Mode:\nIf the task requires reasoning, you MUST start with a <think></think> block. ",
    "Put all reasoning inside the block using plain text. DO NOT include any image tags. ",
    "Keep it reasonable and directly useful for producing the final image.\n\n",
    "2. Non-Think Mode:\nIf no reasoning is needed, directly produce the final image.\n\n",
    "Task Types:\n\nA. Text-to-Image Generation:\n",
    "- Generate a high-quality image based on the user's description.\n",
    "- Ensure visual clarity, semantic consistency, and completeness.\n",
    "- DO NOT introduce elements that contradict or override the user's intent.\n\n",
    "B. Image Editing:\n",
    "- Use the provided image(s) as input or reference for modification or transformation.\n",
    "- The result can be an edited image or a new image based on the reference(s).\n",
    "- Preserve all unspecified attributes unless explicitly changed.\n\n",
    "General Rules:\n",
    "- For any visible text in the image, follow the language specified for the rendered text in ",
    "the user's description, not the language of the prompt. If no language is specified, use the ",
    "user's input language."
);

#[derive(Parser)]
#[command(about = "SenseNova-U1-8B-MoT LoRA / mvp training binary.")]
struct Args {
    /// Directory containing the 8-shard `model.safetensors.index.json` +
    /// `model-{i}-of-{N}.safetensors` files + `vocab.json` + `merges.txt`
    /// + `added_tokens.json`.
    #[arg(long)]
    model_path: PathBuf,

    /// Folder mode dataset directory (image+caption pairs). When omitted,
    /// runs in smoke mode with a single synthetic sample.
    #[arg(long)]
    data_dir: Option<PathBuf>,

    /// Number of training steps. Python's default for full runs is 6000;
    /// 2000 is a reasonable LoRA fit.
    #[arg(long, default_value = "3")]
    steps: usize,

    /// Adam learning rate. Python's default = 5e-5 across all scenarios.
    #[arg(long, default_value = "5e-5")]
    lr: f32,

    #[arg(long, default_value_t = SEED_DEFAULT)]
    seed: u64,

    /// Square image side. Must be divisible by `patch_size * merge_size = 32`.
    /// Default keeps activations tight on 24 GB; raise carefully.
    #[arg(long, default_value = "512")]
    image_hw: usize,

    /// Smoke-mode synthetic text-prefix length (ignored when --data-dir set).
    #[arg(long, default_value = "24")]
    text_len: usize,

    /// Optional output path for the final mvp F32 master state (safetensors).
    #[arg(long)]
    save_to: Option<PathBuf>,

    /// LoRA preset name: `default` (r64 attn+mlp+fm_head), `attn_only`,
    /// `attn_mlp`, `official_r128`. Mutually exclusive with `--lora-spec`.
    #[arg(long, conflicts_with = "lora_spec")]
    lora_preset: Option<String>,

    /// LoRA spec string, e.g. `attn=r64a64;mlp=r64a64;fm_head=r128a128`.
    #[arg(long)]
    lora_spec: Option<String>,

    /// Output path for the LoRA PEFT-format save.
    #[arg(long)]
    lora_save_to: Option<PathBuf>,

    /// Resume LoRA training from a PEFT-format checkpoint. Loads the
    /// down/up/alpha tensors from disk and OVERWRITES the freshly-built
    /// adapters from spec, preserving Parameter TensorIds so AdamW state
    /// (built on first opt.step) is still keyed correctly. Optimizer m/v
    /// state itself is NOT restored — momentum re-warms over ~10-20 steps.
    /// Step counter restarts from 0 in logs (cosmetic; the LoRA weights
    /// continue from where they left off).
    #[arg(long)]
    resume_lora: Option<PathBuf>,

    /// Save a checkpoint every N steps. 0 disables.
    #[arg(long, default_value = "0")]
    checkpoint_every: usize,

    /// Gradient accumulation steps. Default 1 = no accumulation.
    #[arg(long, default_value = "1")]
    grad_accum: usize,

    /// Shuffle dataset order (real-data mode only).
    #[arg(long, default_value = "true")]
    shuffle: bool,

    /// Optimizer kind. Adafactor avoids the m+v state of AdamW (~600 MB at
    /// default preset) — useful when training at 2048² on 24 GB.
    #[arg(long, default_value = "adamw")]
    optimizer: String,

    /// SerenityBoard SQLite output directory. When set, training emits
    /// loss/grad_norm/lr/steps_per_sec scalars to `<board-dir>/board.db`
    /// alongside the stdout progress lines (universal display).
    #[arg(long)]
    board_dir: Option<PathBuf>,
}

// ---------------------------------------------------------------------------
// Tokenizer helpers (shared with sample_u1.rs — could be extracted into a
// shared module later, but keeping the bin self-contained for now).
// ---------------------------------------------------------------------------

fn build_tokenizer(weights_dir: &Path) -> anyhow::Result<Tokenizer> {
    let vocab = weights_dir.join("vocab.json");
    let merges = weights_dir.join("merges.txt");
    let added = weights_dir.join("added_tokens.json");

    let bpe = BPE::from_file(
        vocab.to_str().context("vocab path not utf-8")?,
        merges.to_str().context("merges path not utf-8")?,
    )
    .build()
    .map_err(|e| anyhow!("BPE::build failed: {e}"))?;

    let mut tok = Tokenizer::new(bpe);
    tok.with_pre_tokenizer(Some(ByteLevel::default().add_prefix_space(false)));
    tok.with_decoder(Some(ByteLevel::default()));

    let raw = std::fs::read_to_string(&added)
        .with_context(|| format!("read {}", added.display()))?;
    let map: serde_json::Map<String, serde_json::Value> =
        serde_json::from_str(&raw).context("added_tokens.json")?;
    let mut entries: Vec<(String, u64)> = map
        .into_iter()
        .filter_map(|(k, v)| v.as_u64().map(|id| (k, id)))
        .collect();
    entries.sort_by_key(|(_, id)| *id);

    let added_tokens: Vec<AddedToken> = entries
        .into_iter()
        .map(|(content, _)| AddedToken::from(content, true))
        .collect();
    tok.add_special_tokens(&added_tokens);
    Ok(tok)
}

fn build_t2i_query(system: &str, user: &str, append: &str) -> String {
    let mut q = String::new();
    if !system.is_empty() {
        q.push_str("<|im_start|>system\n");
        q.push_str(system);
        q.push_str("<|im_end|>\n");
    }
    q.push_str("<|im_start|>user\n");
    q.push_str(user);
    q.push_str("<|im_end|>\n");
    q.push_str("<|im_start|>assistant\n");
    q.push_str(append);
    q
}

fn encode_query(tok: &Tokenizer, query: &str) -> anyhow::Result<Vec<i32>> {
    let enc = tok.encode(query, false).map_err(|e| anyhow!("tokenize: {e}"))?;
    Ok(enc.get_ids().iter().map(|&id| id as i32).collect())
}

// ---------------------------------------------------------------------------
// Folder dataset: scan <id>.{jpg|png|webp} + <id>.txt pairs
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct SamplePair {
    image_path: PathBuf,
    caption_path: PathBuf,
    sample_id: String,
}

fn scan_dataset(dir: &Path) -> anyhow::Result<Vec<SamplePair>> {
    let exts: &[&str] = &["jpg", "jpeg", "png", "webp"];
    let mut out: Vec<SamplePair> = Vec::new();
    for entry in std::fs::read_dir(dir)
        .with_context(|| format!("read_dir {}", dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("").to_lowercase();
        if !exts.iter().any(|e| *e == ext.as_str()) {
            continue;
        }
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        if stem.is_empty() {
            continue;
        }
        let cap = path.with_extension("txt");
        if !cap.exists() {
            continue;
        }
        out.push(SamplePair {
            image_path: path.clone(),
            caption_path: cap,
            sample_id: stem.to_string(),
        });
    }
    out.sort_by(|a, b| a.sample_id.cmp(&b.sample_id));
    Ok(out)
}

/// Load image, resize to `target_hw x target_hw`, normalize to `[-1, 1]`
/// (x0 space: `(pixel/255 - 0.5)/0.5`), return BF16 tensor `[1, 3, H, W]`.
fn load_image_x0(
    path: &Path,
    target_hw: usize,
    device: &Arc<CudaDevice>,
) -> anyhow::Result<Tensor> {
    let img = image::open(path)
        .with_context(|| format!("open image {}", path.display()))?
        .to_rgb8();
    let resized = image::imageops::resize(
        &img,
        target_hw as u32,
        target_hw as u32,
        image::imageops::FilterType::Lanczos3,
    );
    let h = resized.height() as usize;
    let w = resized.width() as usize;
    let mut chw = vec![0.0_f32; 3 * h * w];
    for y in 0..h {
        for x in 0..w {
            let px = resized.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                let v = px[c] as f32 / 255.0;
                let v_norm = (v - 0.5) / 0.5; // -> [-1, 1]
                chw[c * h * w + y * w + x] = v_norm;
            }
        }
    }
    let t = Tensor::from_vec(chw, Shape::from_dims(&[1, 3, h, w]), device.clone())?;
    Ok(t.to_dtype(DType::BF16)?)
}

// ---------------------------------------------------------------------------
// Misc helpers
// ---------------------------------------------------------------------------

fn gaussian_bf16(
    seed: u64,
    shape: &[usize],
    device: &Arc<CudaDevice>,
) -> anyhow::Result<Tensor> {
    use rand::{Rng, SeedableRng};
    let numel: usize = shape.iter().product();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut data = Vec::with_capacity(numel);
    for _ in 0..numel {
        let u1: f32 = rng.gen_range(f32::EPSILON..1.0);
        let u2: f32 = rng.gen();
        data.push((-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos());
    }
    let t = Tensor::from_vec(data, Shape::from_dims(shape), device.clone())?;
    Ok(t.to_dtype(DType::BF16)?)
}

fn save_lora_checkpoint(
    model: &SenseNovaU1,
    device: &Arc<CudaDevice>,
    path: &Path,
) -> anyhow::Result<()> {
    u1lora::save_adapters(model.lora_adapters(), path, device)
        .map_err(|e| anyhow!("save_adapters {:?}: {e}", path))?;
    Ok(())
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let args = Args::parse();

    let device = flame_core::CudaDevice::new(0)
        .map_err(|e| anyhow!("CudaDevice::new(0): {e}"))?;
    log::info!("[train_u1] using device 0");

    // ---- LoRA specs ------------------------------------------------------
    let lora_specs: Option<Vec<u1lora::LoraSpec>> = match (&args.lora_preset, &args.lora_spec) {
        (Some(_), Some(_)) => unreachable!("clap conflicts_with prevents this"),
        (Some(p), None) => Some(u1lora::resolve_preset(p)?),
        (None, Some(s)) => Some(u1lora::parse_lora_spec_str(s)?),
        (None, None) => None,
    };
    let use_lora = lora_specs.is_some();

    // ---- Load model ------------------------------------------------------
    log::info!("[train_u1] loading model from {}", args.model_path.display());
    let mut model = if use_lora {
        let specs = lora_specs.as_ref().unwrap();
        log::info!("[train_u1] LoRA specs: {} target(s)", specs.len());
        for s in specs {
            log::info!(
                "  - target={:<24} r={:<3} alpha={:<5} enabled={}",
                s.target, s.r, s.alpha, s.enabled,
            );
        }
        SenseNovaU1::load_for_training_lora(&args.model_path, &device, specs, args.seed)?
    } else {
        SenseNovaU1::load_for_training_mvp(&args.model_path, &device)?
    };

    // ---- Resume from prior LoRA checkpoint --------------------------------
    // Must happen BEFORE `model.parameters()` so the optimizer keys onto the
    // newly-attached Parameters' TensorIds (not the discarded fresh-init
    // ones). Loaded adapters fully replace per-key entries in the model's
    // LoRA HashMap.
    if let Some(resume_path) = args.resume_lora.as_ref() {
        if !use_lora {
            anyhow::bail!(
                "--resume-lora requires --lora-preset or --lora-spec (resume \
                 needs the same LoRA target shape as the saved checkpoint)"
            );
        }
        log::info!("[train_u1] resuming LoRA from {}", resume_path.display());
        let loaded = u1lora::load_adapters(resume_path, device.clone())?;
        let expected = model.lora_adapters().len();
        if loaded.len() != expected {
            log::warn!(
                "[train_u1] checkpoint has {} adapters, current spec expects {} — \
                 keys present in both will be loaded; mismatches kept fresh-init",
                loaded.len(), expected,
            );
        }
        log::info!("[train_u1] attaching {} loaded LoRA adapters", loaded.len());
        model.attach_lora_adapters(loaded);
    }

    let params = model.parameters();
    log::info!("[train_u1] {} trainable Parameters", params.len());
    if params.is_empty() {
        anyhow::bail!("no trainable parameters — loader failed silently");
    }

    let opt_kind = match args.optimizer.to_lowercase().as_str() {
        "adamw" => OptimizerKind::AdamW,
        // Python's U1 trainer uses bnb.optim.PagedAdamW8bit — 8-bit moment
        // state, ~4× smaller than F32 m+v. The project-wide no-quantization
        // rule is Z-Image-only; Wan22 + U1 are documented exceptions.
        "adamw8bit" | "adamw_8bit" => OptimizerKind::AdamW8bit,
        "adafactor" => OptimizerKind::Adafactor,
        "lion" => OptimizerKind::Lion,
        "prodigy" => OptimizerKind::Prodigy,
        "stable_adamw" | "stableadamw" => OptimizerKind::StableAdamW,
        other => anyhow::bail!(
            "unknown --optimizer {other:?}; valid: adamw | adamw8bit | adafactor | lion | prodigy | stable_adamw"
        ),
    };
    let mut opt = Optimizer::new(opt_kind, args.lr, 0.9, 0.95, 1e-8, 0.0);
    log::info!(
        "[train_u1] optimizer={:?}(lr={})  grad_accum={}",
        opt_kind, args.lr, args.grad_accum,
    );

    // ---- Geometry --------------------------------------------------------
    let (p, merge, fm_dim, t_eps, bos_id) = {
        let cfg = model.config();
        (cfg.patch_size, cfg.merge_size(), cfg.fm_head_out_dim(),
         cfg.t_eps, cfg.bos_token_id)
    };
    let token_p = p * merge;
    if args.image_hw % token_p != 0 {
        anyhow::bail!(
            "--image-hw {} must be divisible by patch_size*merge_size = {token_p}",
            args.image_hw,
        );
    }
    let h_img = args.image_hw;
    let w_img = args.image_hw;
    let grid_h = h_img / p;
    let grid_w = w_img / p;
    let token_h = grid_h / merge;
    let token_w = grid_w / merge;
    let n_image = token_h * token_w;
    log::info!(
        "[train_u1] geometry: HxW={}x{}  grid={}x{}  tokens={}x{}={}  fm_dim={}",
        h_img, w_img, grid_h, grid_w, token_h, token_w, n_image, fm_dim,
    );

    // ---- Decide mode ----------------------------------------------------
    let (samples, tokenizer): (Vec<SamplePair>, Option<Tokenizer>) =
        if let Some(data_dir) = args.data_dir.as_ref() {
            let samples = scan_dataset(data_dir)?;
            if samples.is_empty() {
                anyhow::bail!("no <id>.{{jpg|png|webp}} + <id>.txt pairs in {}", data_dir.display());
            }
            log::info!(
                "[train_u1] dataset {}: {} samples",
                data_dir.display(), samples.len(),
            );
            let tok = build_tokenizer(&args.model_path)?;
            (samples, Some(tok))
        } else {
            (Vec::new(), None)
        };

    // ---- Smoke-mode constants (only used when real-data mode is off) ----
    let smoke_noisy = if samples.is_empty() {
        Some(gaussian_bf16(args.seed, &[1, 3, h_img, w_img], &device)?)
    } else { None };
    let smoke_x0 = if samples.is_empty() {
        Some(gaussian_bf16(args.seed.wrapping_add(1), &[1, n_image, fm_dim], &device)?)
    } else { None };
    let smoke_input_ids: Vec<i32> = if samples.is_empty() {
        let bos = bos_id as i32;
        let mut v = Vec::with_capacity(args.text_len);
        v.push(bos);
        for _ in 1..args.text_len { v.push(100i32); }
        v
    } else { Vec::new() };

    // ---- Training loop --------------------------------------------------
    log::info!("[train_u1] starting {} steps  mode={}",
        args.steps,
        if samples.is_empty() { "SMOKE (synthetic)" } else { "DATASET" });
    let mut losses: Vec<f32> = Vec::with_capacity(args.steps);
    let n_samples = samples.len().max(1);
    let mut accum_count: usize = 0;
    let t_run = std::time::Instant::now();

    for step in 0..args.steps {
        let t_step = std::time::Instant::now();

        // Pick sample (real-data mode) or use the synthetic one.
        let idx: usize = if samples.is_empty() {
            0
        } else if args.shuffle {
            use rand::{Rng, SeedableRng};
            let mut rng = rand::rngs::StdRng::seed_from_u64(args.seed.wrapping_add(step as u64));
            rng.gen_range(0..n_samples)
        } else {
            step % n_samples
        };

        // Build (noisy_pixel_values, x0_patch, eps, input_ids) for this step.
        let (noisy_pixel_values_step, x0_patch_step, eps_step, input_ids_step): (Tensor, Tensor, Tensor, Vec<i32>) =
            if samples.is_empty() {
                (
                    smoke_noisy.as_ref().unwrap().clone(),
                    smoke_x0.as_ref().unwrap().clone(),
                    // Resample eps each step from the same seed for repro
                    gaussian_bf16(args.seed.wrapping_add(2_000_000 + step as u64), &[1, n_image, fm_dim], &device)?,
                    smoke_input_ids.clone(),
                )
            } else {
                let s = &samples[idx];
                let img = load_image_x0(&s.image_path, h_img, &device)?;
                let x0 = sensenova_u1::patchify(&img, token_p, false)?;
                let eps = gaussian_bf16(args.seed.wrapping_add(2_000_000 + step as u64), &[1, n_image, fm_dim], &device)?;
                let caption = std::fs::read_to_string(&s.caption_path)
                    .with_context(|| format!("read {}", s.caption_path.display()))?;
                let query = build_t2i_query(
                    SYSTEM_MESSAGE_FOR_GEN,
                    caption.trim(),
                    "<think>\n\n</think>\n\n<img>",
                );
                let ids = encode_query(tokenizer.as_ref().unwrap(), &query)?;
                (img, x0, eps, ids)
            };

        // Compute noisy = unpatchify(z_t) so vision tower sees the noisy-pixel input.
        // z_t = t * x0 + (1-t) * eps. We do this on CPU-side scalar t for clarity
        // (one tensor mul + add per step is cheap relative to the 42-layer forward).
        let t_val: f32 = {
            use rand::{Rng, SeedableRng};
            let mut rng = rand::rngs::StdRng::seed_from_u64(args.seed.wrapping_add(3_000_000 + step as u64));
            rng.gen_range(t_eps..=1.0_f32)
        };
        let t_tensor = Tensor::from_vec(vec![t_val], Shape::from_dims(&[1]), device.clone())?
            .to_dtype(DType::BF16)?;

        // In real-data mode, we want the noisy pixel image (not the clean one)
        // for extract_feature_gen. Build it by unpatchify(z_t) at p=32.
        let noisy_pixel_for_gen: Tensor = if samples.is_empty() {
            noisy_pixel_values_step.clone()
        } else {
            let z_t = sensenova_u1::linear_z_t(&x0_patch_step, &eps_step, &t_tensor)?;
            sensenova_u1::unpatchify(&z_t, token_p, h_img, w_img)?
        };

        let out = model.forward_t2i_step(
            &noisy_pixel_for_gen,
            &x0_patch_step,
            &eps_step,
            &t_tensor,
            grid_h,
            grid_w,
            &input_ids_step,
            None,
            None,
        )?;

        // Loss: MSE(x_pred, x0) in F32.
        let pred_f32 = out.x_pred.to_dtype(DType::F32)?;
        let target_f32 = x0_patch_step.to_dtype(DType::F32)?;
        let loss = flame_core::loss::mse_loss(&pred_f32, &target_f32)?;
        let loss_val = loss.to_vec()?[0];
        losses.push(loss_val);

        // Scale loss for grad accumulation.
        let loss_scaled = if args.grad_accum > 1 {
            loss.mul_scalar(1.0 / args.grad_accum as f32)?
        } else { loss };

        let grads = loss_scaled.backward()?;

        // Grad-flow assertion at step 1. Only meaningful for AdamW: other
        // optimizers (Adafactor, AdamW8bit, Lion, Prodigy, StableAdamW) call
        // `Parameter::set_data()` in their `.step()`, which replaces the
        // stored tensor with a fresh `TensorId` while leaving `Parameter.id`
        // untouched. At step 1 those optimizers' params have already been
        // through one .step(), so the diagnostic queries `grads.get(stale_id)`
        // and reports every param "missing" even though training is healthy.
        // Loss-vs-time is the real signal in that case.
        let grad_flow_meaningful = matches!(opt_kind, OptimizerKind::AdamW);
        if step == 1 && grad_flow_meaningful {
            let named = model.named_parameters();
            let named_refs: Vec<(&str, &flame_core::parameter::Parameter)> =
                named.iter().map(|(n, p)| (n.as_str(), p)).collect();
            let report = flame_core::diagnostics::assert_grad_flow(&grads, &named_refs)?;
            if report.is_clean() {
                log::info!("[train_u1] step 1 grad-flow clean ({} params)", report.ok_count);
            } else {
                log::warn!("[train_u1] grad-flow {}", report.summary());
            }
        } else if step == 1 {
            log::info!(
                "[train_u1] step 1 grad-flow check skipped (optimizer={:?} \
                 uses Parameter::set_data which breaks the id-based diagnostic; \
                 training is healthy if loss decreases)",
                opt_kind,
            );
        }

        // Accumulate grads into Parameter.grad. For grad_accum > 1, sum
        // across micro-steps; otherwise just set.
        //
        // NOTE: Only correct for AdamW. Other optimizers in flame-core call
        // `Parameter::set_data` in `.step()` which replaces the inner Tensor
        // with one that has a fresh TensorId — but `Parameter.id` field is
        // NOT updated. After step 0 with those optimizers, every `param.id()`
        // is stale relative to the next backward's GradientMap keys, so the
        // `grads.get(...)` below returns None for every param → no set_grad
        // → no update. Loss looks like it varies but it's purely t-variance.
        // Until flame-core's Parameter::set_data is patched to refresh
        // self.id (or those optimizers switched to with_data_mut), AdamW is
        // the only safe choice for this trainer.
        for param in &params {
            if let Some(g) = grads.get(param.id()) {
                if args.grad_accum > 1 && accum_count > 0 {
                    // Add to existing
                    let existing = param.grad();
                    let new_g = match existing {
                        Some(prev) => prev.add(g)?,
                        None => g.clone(),
                    };
                    param.set_grad(new_g)?;
                } else {
                    param.set_grad(g.clone())?;
                }
            }
        }
        accum_count += 1;

        // Global L2 grad norm for the progress line (cheap — sums over the
        // GradientMap entries that match our params).
        let grad_norm: f32 = {
            let mut sq_sum: f64 = 0.0;
            for param in &params {
                if let Some(g) = grads.get(param.id()) {
                    let abs2 = g.to_dtype(DType::F32)?.square()?.sum_all()?.to_vec()?[0];
                    sq_sum += abs2 as f64;
                }
            }
            (sq_sum.sqrt()) as f32
        };

        let do_step = accum_count >= args.grad_accum;
        if do_step {
            opt.step(&params)?;
            opt.zero_grad(&params);
            accum_count = 0;
        }

        let _ = t_step; // step-local timing folded into log_step's sec/step

        // Universal progress line (used across EDv2 trainers — see
        // crates/eridiffusion-core/src/training/progress.rs). Emits
        // `[<tag>] step N/T | epoch | loss | grad_norm | s/step | elapsed | ETA`.
        let tag = if use_lora { "SenseNova-U1-lora" } else { "SenseNova-U1-mvp" };
        eridiffusion_core::training::progress::log_step(
            tag,
            step,
            args.steps,
            n_samples,
            1, // batch_size = 1
            loss_val,
            grad_norm,
            args.lr,
            t_run,
            None, // BoardWriter — wired in follow-up
        );

        // Periodic checkpoint
        if args.checkpoint_every > 0
            && (step + 1) % args.checkpoint_every == 0
            && do_step
            && use_lora
        {
            let base = args.lora_save_to.as_ref()
                .or(args.save_to.as_ref())
                .map(|p| p.clone())
                .unwrap_or_else(|| PathBuf::from("/tmp/u1_lora.safetensors"));
            let ckpt = base.with_file_name(format!(
                "{}.step{:06}.safetensors",
                base.file_stem().and_then(|s| s.to_str()).unwrap_or("u1_lora"),
                step + 1,
            ));
            save_lora_checkpoint(&model, &device, &ckpt)?;
            log::info!("[train_u1] checkpoint → {}", ckpt.display());
        }
    }

    let run_secs = t_run.elapsed().as_secs_f32();
    if losses.len() >= 5 {
        let first = losses[..5].iter().sum::<f32>() / 5.0;
        let last = losses[losses.len() - 5..].iter().sum::<f32>() / 5.0;
        log::info!(
            "[train_u1] DONE in {:.1}s  mean(loss[:5])={:.6}  mean(loss[-5:])={:.6}  ratio={:.3}",
            run_secs, first, last, last / first.max(1e-12),
        );
    } else {
        log::info!("[train_u1] DONE in {:.1}s ({} steps)", run_secs, losses.len());
    }

    // ---- Save trainable state ------------------------------------------
    if use_lora {
        let lora_path = args
            .lora_save_to
            .clone()
            .or_else(|| {
                args.save_to.as_ref().map(|p| {
                    let mut out = p.clone();
                    let stem = out.file_stem().map(|s| s.to_os_string()).unwrap_or_default();
                    let mut new_name = stem;
                    new_name.push(".lora.safetensors");
                    out.set_file_name(new_name);
                    out
                })
            });
        if let Some(path) = lora_path.as_ref() {
            save_lora_checkpoint(&model, &device, path)?;
            log::info!(
                "[train_u1] saved {} LoRA adapters (PEFT format) → {}",
                model.lora_adapters().len(), path.display(),
            );
        }
    } else if let Some(path) = args.save_to.as_ref() {
        let named = model.named_parameters();
        let mut tensors: std::collections::HashMap<String, Tensor> =
            std::collections::HashMap::with_capacity(named.len());
        for (k, p) in &named {
            tensors.insert(k.clone(), p.tensor()?);
        }
        flame_core::serialization::save_file(&tensors, path)
            .map_err(|e| anyhow!("save_file {:?}: {e}", path))?;
        log::info!("[train_u1] saved {} tensors → {}", tensors.len(), path.display());
    }

    Ok(())
}
