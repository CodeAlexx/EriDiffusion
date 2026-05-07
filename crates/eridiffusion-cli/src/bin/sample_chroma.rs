//! sample_chroma — text → Chroma image generation.
//!
//! Pipeline:
//!   1. Tokenize prompt with T5-XXL tokenizer, encode via T5-XXL.
//!   2. Load Chroma transformer (BlockOffloader or resident, auto-selected).
//!   3. Build noise [1, 16, H_lat, W_lat] and FLUX-style Euler CFG schedule.
//!   4. CFG denoise with `ChromaTrainingModel::forward`.
//!   5. LDM VAE decode → RGB → PNG.
//!
//! Chroma is NOT distilled — uses real CFG, so each step runs 2 forwards.
//! Default paths match the Chroma1-HD HuggingFace snapshot on this machine.

use clap::Parser;
use std::path::PathBuf;
use flame_core::{autograd::AutogradContext, DType, Shape, Tensor};
use eridiffusion_core::encoders::t5_xxl::T5Encoder;
use eridiffusion_core::models::ChromaTrainingModel;
use eridiffusion_core::sampler::flux_sampler;

const DEFAULT_DIT_DIR: &str =
    "/home/alex/.cache/huggingface/hub/models--lodestones--Chroma1-HD/snapshots/0e0c60ece1e82b17cb7f77342d765ba5024c40c0/transformer";

const DEFAULT_VAE: &str =
    "/home/alex/.cache/huggingface/hub/models--lodestones--Chroma1-HD/snapshots/0e0c60ece1e82b17cb7f77342d765ba5024c40c0/vae/diffusion_pytorch_model.safetensors";

const DEFAULT_T5_PATH: &str =
    "/home/alex/.serenity/models/text_encoders/t5xxl_fp16.safetensors";

const DEFAULT_T5_TOKENIZER: &str =
    "/home/alex/.serenity/models/text_encoders/t5xxl_fp16.tokenizer.json";

/// Chroma uses fixed T5 padding to 256 tokens (not 512) — shorter is fine
/// because the training model's forward always uses the full token sequence
/// length as passed in. Chroma trains against fixed-length padding.
// Match inference-flame's chroma_gen / chroma_encode (both use 512). T5
// attends to pad tokens, so different total seq lengths produce different
// real-token outputs even for short prompts. Pre-fix at 256, the encoded
// cond differed from chroma1-HD's expected distribution and produced
// content-dependent glitches (some prompts denoised cleanly, others
// collapsed into striped/speckle output).
const T5_SEQ_LEN: usize = 512;

/// FLUX VAE constants — Chroma uses the same VAE as FLUX.
const AE_IN_CHANNELS: usize = 16;
const AE_SCALE_FACTOR: f32 = 0.3611;
const AE_SHIFT_FACTOR: f32 = 0.1159;

#[derive(Parser, Debug)]
#[command(about = "Chroma image generation")]
struct Args {
    /// Single prompt. Mutually exclusive with `--prompts-file`.
    #[arg(long)] prompt: Option<String>,
    /// Newline-separated prompts file for batch sampling. Blank lines and
    /// `#`-prefixed comments are skipped. Requires `--output-dir`. The
    /// text encoder is loaded once, every prompt is encoded, the TE is
    /// dropped, then the DiT loads once and serves all prompts; the VAE
    /// loads once at the end and decodes every collected latent.
    #[arg(long)] prompts_file: Option<PathBuf>,
    #[arg(long, default_value = "")] negative: String,
    /// Single-prompt output path. Used when `--prompt` is given.
    #[arg(long, default_value = "output/chroma_sample.png")] output: PathBuf,
    /// Multi-prompt output directory. Required with `--prompts-file`.
    /// Files are written as `sample_001.png`, `sample_002.png`, ...
    #[arg(long)] output_dir: Option<PathBuf>,
    /// Chroma transformer: directory of shards OR single safetensors.
    #[arg(long, default_value = DEFAULT_DIT_DIR)] transformer: PathBuf,
    #[arg(long, default_value = DEFAULT_VAE)] vae_path: PathBuf,
    #[arg(long, default_value = DEFAULT_T5_PATH)] t5_path: PathBuf,
    #[arg(long, default_value = DEFAULT_T5_TOKENIZER)] tokenizer_path: PathBuf,
    /// Output image height (must be divisible by 16).
    #[arg(long, default_value = "1024")] height: usize,
    /// Output image width (must be divisible by 16).
    #[arg(long, default_value = "1024")] width: usize,
    #[arg(long, default_value = "20")] steps: usize,
    #[arg(long, default_value = "4.0")] cfg: f32,
    #[arg(long, default_value = "42")] seed: u64,
    /// Use BlockOffloader (recommended for 24 GB cards — keeps ~3 GB free).
    #[arg(long)] offload: bool,
}

fn tokenize_t5(
    tokenizer_path: &std::path::Path,
    prompt: &str,
    seq_len: usize,
) -> anyhow::Result<Vec<i32>> {
    let tok = tokenizers::Tokenizer::from_file(tokenizer_path)
        .map_err(|e| anyhow::anyhow!("T5 tokenizer load: {e}"))?;
    let enc = tok
        .encode(prompt, true)
        .map_err(|e| anyhow::anyhow!("T5 tokenize: {e}"))?;
    let mut ids: Vec<i32> = enc.get_ids().iter().map(|&i| i as i32).collect();
    // Pad with 0 (T5 pad token) or truncate to seq_len.
    ids.resize(seq_len, 0);
    Ok(ids)
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let args = Args::parse();

    // Disable autograd globally — inference only.
    let _no_grad = AutogradContext::no_grad();
    std::env::set_var("FLAME_ALLOC_POOL", "0");
    flame_core::config::set_default_dtype(DType::BF16);
    let device = flame_core::global_cuda_device();

    let t_total = std::time::Instant::now();

    // ------------------------------------------------------------------
    // Stage 1: T5-XXL encode (load + encode + drop before DiT loads)
    // ------------------------------------------------------------------
    log::info!("[1/4] Loading T5-XXL...");
    let t5_path_str = args
        .t5_path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("t5_path not valid UTF-8"))?;

    let mut t5 = T5Encoder::load(t5_path_str, &device)
        .map_err(|e| anyhow::anyhow!("T5 load: {e}"))?;

    // Resolve prompt list. Single-prompt mode keeps the legacy
    // `--prompt` / `--output` contract; multi-prompt mode reads
    // `--prompts-file` and writes to `--output-dir/sample_NNN.png`.
    let prompts: Vec<String> = match (&args.prompt, &args.prompts_file) {
        (Some(p), None) => vec![p.clone()],
        (None, Some(path)) => {
            let content = std::fs::read_to_string(path)
                .map_err(|e| anyhow::anyhow!("read --prompts-file {}: {e}", path.display()))?;
            content.lines()
                .map(|l| l.trim())
                .filter(|l| !l.is_empty() && !l.starts_with('#'))
                .map(|l| l.to_string())
                .collect()
        }
        (Some(_), Some(_)) => anyhow::bail!("--prompt and --prompts-file are mutually exclusive"),
        (None, None) => anyhow::bail!("provide --prompt or --prompts-file"),
    };
    if prompts.is_empty() {
        anyhow::bail!("no prompts found in --prompts-file");
    }
    let multi_mode = args.prompts_file.is_some();
    if multi_mode && args.output_dir.is_none() {
        anyhow::bail!("--prompts-file requires --output-dir");
    }
    if let Some(dir) = &args.output_dir {
        std::fs::create_dir_all(dir)
            .map_err(|e| anyhow::anyhow!("create --output-dir {}: {e}", dir.display()))?;
    }

    log::info!("[1/4] Encoding {} prompt(s) + uncond (seq_len={})...", prompts.len(), T5_SEQ_LEN);
    let conds: Vec<Tensor> = prompts.iter().enumerate()
        .map(|(i, p)| {
            let tokens = tokenize_t5(&args.tokenizer_path, p, T5_SEQ_LEN)?;
            let c = t5.encode(&tokens)
                .map_err(|e| anyhow::anyhow!("T5 cond encode {}: {e}", i + 1))?;
            let c = c.to_dtype(DType::BF16).map_err(|e| anyhow::anyhow!("cond dtype: {e}"))?;
            log::info!("  prompt {}/{}: cond shape={:?}", i + 1, prompts.len(), c.shape().dims());
            Ok(c)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    let uncond_tokens = tokenize_t5(&args.tokenizer_path, &args.negative, T5_SEQ_LEN)?;
    let uncond = t5
        .encode(&uncond_tokens)
        .map_err(|e| anyhow::anyhow!("T5 uncond encode: {e}"))?;
    let uncond = uncond.to_dtype(DType::BF16).map_err(|e| anyhow::anyhow!("uncond dtype: {e}"))?;

    log::info!("[1/4] uncond={:?}", uncond.shape().dims());
    drop(t5); // free ~10 GB before loading DiT

    // ------------------------------------------------------------------
    // Stage 2: Load Chroma transformer
    // ------------------------------------------------------------------
    log::info!("[2/4] Loading Chroma transformer (offload={})...", args.offload);
    let model = if args.offload {
        ChromaTrainingModel::load_swapped(
            &args.transformer,
            "lora", // mode=lora means no FFT params, just frozen weights
            1,      // lora_rank (not used in inference)
            1.0,    // lora_alpha (not used)
            device.clone(),
            args.seed,
        )
        .map_err(|e| anyhow::anyhow!("Chroma load_swapped: {e}"))?
    } else {
        ChromaTrainingModel::load(
            &args.transformer,
            "lora",
            1,
            1.0,
            device.clone(),
            args.seed,
        )
        .map_err(|e| anyhow::anyhow!("Chroma load: {e}"))?
    };

    // ------------------------------------------------------------------
    // Stage 3: Denoise
    // ------------------------------------------------------------------
    if args.height % 16 != 0 || args.width % 16 != 0 {
        anyhow::bail!("height and width must be divisible by 16 (got {}x{})", args.height, args.width);
    }

    // Latent geometry: VAE 8x + patchify 2x = 16x total.
    let latent_h = args.height / 8;   // VAE-level (before patchify)
    let latent_w = args.width / 8;

    log::info!(
        "[3/4] Denoising {} prompt(s) at {}x{} → latent {}x{}, {} steps, cfg={}...",
        conds.len(), args.width, args.height, latent_w, latent_h, args.steps, args.cfg
    );

    let timesteps = flux_sampler::schedule(args.steps, args.width, args.height);
    log::info!(
        "  schedule: t[0]={:.4} t[-2]={:.4} t[-1]={:.4}",
        timesteps[0],
        timesteps[args.steps.saturating_sub(1)],
        timesteps[args.steps]
    );

    let pad_width = std::cmp::max(3, conds.len().to_string().len());
    let t_denoise_total = std::time::Instant::now();
    // Collect each prompt's final latent so we can drop the DiT once and
    // decode every latent through a single VAE-load pass.
    let mut latents: Vec<Tensor> = Vec::with_capacity(conds.len());

    for (idx, cond) in conds.iter().enumerate() {
        log::info!("  [{}/{}] denoising...", idx + 1, conds.len());

        // Box-Muller noise, matches chroma_sampler::sample_image seeding.
        // Same seed across prompts keeps noise pattern fixed; if you want
        // variety, vary the seed or add a per-prompt offset.
        let numel = AE_IN_CHANNELS * latent_h * latent_w;
        let noise_data: Vec<f32> = {
            use rand::{rngs::StdRng, Rng, SeedableRng};
            let mut rng = StdRng::seed_from_u64(args.seed);
            let mut v = Vec::with_capacity(numel);
            while v.len() < numel {
                let u1: f32 = rng.gen::<f32>().max(1e-10);
                let u2: f32 = rng.gen::<f32>();
                let mag = (-2.0 * u1.ln()).sqrt();
                let theta = 2.0 * std::f32::consts::PI * u2;
                v.push(mag * theta.cos());
                if v.len() < numel {
                    v.push(mag * theta.sin());
                }
            }
            v
        };

        let mut x = Tensor::from_f32_to_bf16(
            noise_data,
            Shape::from_dims(&[1, AE_IN_CHANNELS, latent_h, latent_w]),
            device.clone(),
        )
        .map_err(|e| anyhow::anyhow!("noise tensor: {e}"))?;

        let t_step = std::time::Instant::now();
        for step in 0..args.steps {
            let t_curr = timesteps[step];
            let t_next = timesteps[step + 1];
            let dt = t_next - t_curr;

            let t_vec = Tensor::from_f32_to_bf16(
                vec![t_curr],
                Shape::from_dims(&[1]),
                device.clone(),
            )
            .map_err(|e| anyhow::anyhow!("t_vec: {e}"))?;

            // Cond forward
            let pred_cond = model
                .forward(&x, cond, &t_vec)
                .map_err(|e| anyhow::anyhow!("forward cond prompt {} step {step}: {e}", idx + 1))?;
            // Uncond forward
            let pred_uncond = model
                .forward(&x, &uncond, &t_vec)
                .map_err(|e| anyhow::anyhow!("forward uncond prompt {} step {step}: {e}", idx + 1))?;

            // CFG: pred = uncond + cfg_scale * (cond - uncond)
            let diff = pred_cond
                .sub(&pred_uncond)
                .map_err(|e| anyhow::anyhow!("cfg diff: {e}"))?;
            let scaled = diff
                .mul_scalar(args.cfg)
                .map_err(|e| anyhow::anyhow!("cfg scale: {e}"))?;
            let pred = pred_uncond
                .add(&scaled)
                .map_err(|e| anyhow::anyhow!("cfg add: {e}"))?;

            // Euler step: x_next = x + dt * pred
            x = x
                .add(&pred.mul_scalar(dt).map_err(|e| anyhow::anyhow!("dt mul: {e}"))?)
                .map_err(|e| anyhow::anyhow!("euler step: {e}"))?;

            if (step + 1) % 5 == 0 || step == 0 || step + 1 == args.steps {
                log::info!(
                    "  prompt {}/{} step {}/{}, t={:.4} ({:.1}s elapsed)",
                    idx + 1,
                    conds.len(),
                    step + 1,
                    args.steps,
                    t_curr,
                    t_step.elapsed().as_secs_f32()
                );
            }
        }
        latents.push(x);
    }
    let denoise_secs = t_denoise_total.elapsed().as_secs_f32();
    log::info!("[3/4] Denoising {} prompt(s) done in {:.1}s", latents.len(), denoise_secs);

    // ------------------------------------------------------------------
    // Stage 4: VAE decode + save PNG (single VAE load for all prompts)
    // ------------------------------------------------------------------
    log::info!("[4/4] VAE decode for {} prompt(s)...", latents.len());
    drop(model); // free DiT before loading VAE
    let vae_path_str = args
        .vae_path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("vae_path not valid UTF-8"))?;

    let vae = eridiffusion_core::encoders::flux_vae_decoder::LdmVAEDecoder::from_safetensors(
        vae_path_str,
        AE_IN_CHANNELS,
        AE_SCALE_FACTOR,
        AE_SHIFT_FACTOR,
        &device,
    )
    .map_err(|e| anyhow::anyhow!("VAE load: {e}"))?;

    let mut saved_paths: Vec<PathBuf> = Vec::with_capacity(latents.len());
    for (idx, latent) in latents.iter().enumerate() {
        let rgb = vae
            .decode(latent)
            .map_err(|e| anyhow::anyhow!("VAE decode prompt {}: {e}", idx + 1))?;
        let out_path = if multi_mode {
            let dir = args.output_dir.as_ref().unwrap();
            dir.join(format!("sample_{:0>width$}.png", idx + 1, width = pad_width))
        } else {
            args.output.clone()
        };
        save_rgb_png(&rgb, &out_path).map_err(|e| anyhow::anyhow!("save PNG: {e}"))?;
        log::info!("  [{}/{}] saved {}", idx + 1, latents.len(), out_path.display());
        saved_paths.push(out_path);
    }
    drop(vae);

    let total_secs = t_total.elapsed().as_secs_f32();
    log::info!(
        "Done. {} sample(s) saved | denoise={:.1}s | total={:.1}s",
        saved_paths.len(),
        denoise_secs,
        total_secs
    );
    for p in &saved_paths {
        println!("Saved: {}", p.display());
    }
    println!(
        "Timing: denoise={:.1}s  total={:.1}s",
        denoise_secs, total_secs
    );
    Ok(())
}

fn save_rgb_png(rgb: &Tensor, path: &std::path::Path) -> flame_core::Result<()> {
    let rgb_f32 = rgb.to_dtype(DType::F32)?;
    let data = rgb_f32.to_vec()?;
    let dims = rgb_f32.shape().dims().to_vec();
    if dims.len() != 4 || dims[1] != 3 {
        return Err(flame_core::Error::InvalidInput(format!(
            "expected [B,3,H,W], got {dims:?}"
        )));
    }
    let (out_h, out_w) = (dims[2], dims[3]);
    let mut pixels = vec![0u8; out_h * out_w * 3];
    for y in 0..out_h {
        for x_col in 0..out_w {
            for c in 0..3 {
                let idx = c * out_h * out_w + y * out_w + x_col;
                let val = (127.5 * (data[idx].clamp(-1.0, 1.0) + 1.0)) as u8;
                pixels[(y * out_w + x_col) * 3 + c] = val;
            }
        }
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| flame_core::Error::Io(format!("create dir {}: {e}", parent.display())))?;
    }
    image::RgbImage::from_raw(out_w as u32, out_h as u32, pixels)
        .ok_or_else(|| flame_core::Error::InvalidInput("RgbImage::from_raw failed".into()))?
        .save(path)
        .map_err(|e| flame_core::Error::Io(format!("save png {}: {e}", path.display())))?;
    Ok(())
}
