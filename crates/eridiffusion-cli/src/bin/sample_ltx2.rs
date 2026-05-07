//! sample_ltx2 — text → LTX-2 video generation (image-as-frame for now).
//!
//! T2V sampling pipeline:
//!   1. Encode prompt with Gemma-3 (currently STUB → zeros).
//!   2. Init latent noise `[1, 128, F', H/32, W/32]`.
//!   3. Run rectified-flow Euler with the LTX-2 shifted schedule for
//!      `--steps` iterations; CFG = `--guidance`.
//!   4. Denormalize latent (per-channel mean/std) and run VAE decode
//!      → `[1, 3, F, H, W]` pixel video.
//!   5. Save first frame as PNG (or future: encode as MP4 for true T2V).
//!
//! Mirrors `sample_ernie.rs` but loops over frames at decode/output stage.

use clap::Parser;
use std::path::PathBuf;

use eridiffusion_core::config::{TrainConfig, TrainingMethod};
use eridiffusion_core::encoders::gemma3::Gemma3Encoder;
use eridiffusion_core::encoders::ltx2_vae::Ltx2Vae;
use eridiffusion_core::models::{Ltx2Model, TrainableModel};
use eridiffusion_core::sampler::ltx2_sampler;
use flame_core::{DType, Shape, Tensor};

#[derive(Parser)]
struct Args {
    /// Single prompt. Mutually exclusive with `--prompts-file`.
    #[arg(long)] prompt: Option<String>,
    /// Newline-separated prompts file for batch sampling. Blank lines and
    /// `#`-prefixed comments are skipped. Requires `--output-dir`. Encoder
    /// loads once for all prompts; DiT and VAE each load once total.
    #[arg(long)] prompts_file: Option<PathBuf>,
    #[arg(long, default_value = "")] negative_prompt: String,
    #[arg(long, default_value = "ltx2_sample.png")] output: PathBuf,
    /// Multi-prompt output directory. Required with `--prompts-file`.
    /// Files are written as `sample_001.png`, `sample_002.png`, ...
    #[arg(long)] output_dir: Option<PathBuf>,
    /// Directory of safetensors shards for the LTX-2 transformer.
    #[arg(long)] transformer_dir: PathBuf,
    /// LTX-2 video VAE checkpoint.
    #[arg(long)] vae_path: PathBuf,
    /// Gemma-3 text encoder directory.
    #[arg(long)] text_ckpt_dir: PathBuf,
    /// Tokenizer.json path for Gemma-3.
    #[arg(long)] tokenizer_path: PathBuf,
    #[arg(long, default_value = "256")] size: usize,
    /// Number of latent frames. Bootstrap default = 1 (image-as-frame).
    /// Real video must satisfy `(num_frames - 1) % 8 == 0` (1, 9, 17, 25, ...).
    #[arg(long, default_value = "1")] frames: usize,
    #[arg(long, default_value = "20")] steps: usize,
    #[arg(long, default_value = "5.0")] guidance: f32,
    #[arg(long, default_value = "42")] seed: u64,
    #[arg(long, default_value = "24.0")] fps: f32,
    /// Optional trained LoRA to overlay.
    #[arg(long)] lora_path: Option<PathBuf>,
    #[arg(long, default_value = "16")] lora_rank: usize,
    #[arg(long, default_value = "1.0")] lora_alpha: f64,
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let args = Args::parse();
    let device = flame_core::global_cuda_device();
    let _no_grad = flame_core::autograd::AutogradContext::no_grad();
    flame_core::config::set_default_dtype(DType::BF16);

    if args.frames > 1 && (args.frames - 1) % 8 != 0 {
        anyhow::bail!(
            "--frames {} must satisfy (frames - 1) %% 8 == 0 (1, 9, 17, 25, ...)",
            args.frames
        );
    }
    if args.size % 32 != 0 {
        anyhow::bail!("--size {} must be divisible by 32 (LTX-2 spatial compression)", args.size);
    }

    let h_lat = args.size / 32;
    let w_lat = args.size / 32;
    let f_lat = if args.frames == 1 { 1 } else { 1 + (args.frames - 1) / 8 };
    log::info!(
        "Size {}x{} (frames={}) → latent {}x{} (f_lat={})",
        args.size, args.size, args.frames, h_lat, w_lat, f_lat
    );

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

    // 1. Encode prompts.
    log::info!("[1/4] Text encoding {} prompt(s) + uncond...", prompts.len());
    let tokenizer = tokenizers::Tokenizer::from_file(&args.tokenizer_path)
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    let te = Gemma3Encoder::load(&args.text_ckpt_dir, device.clone())
        .map_err(|e| anyhow::anyhow!("Gemma3Encoder::load: {e}"))?;
    let encode = |text: &str| -> anyhow::Result<Tensor> {
        let e = tokenizer.encode(text, true).map_err(|e| anyhow::anyhow!("{e}"))?;
        let mut ids: Vec<i32> = e.get_ids().iter().map(|&x| x as i32).collect();
        if ids.len() > eridiffusion_core::encoders::gemma3::GEMMA3_PROMPT_LEN {
            ids.truncate(eridiffusion_core::encoders::gemma3::GEMMA3_PROMPT_LEN);
        }
        let pad_n = eridiffusion_core::encoders::gemma3::GEMMA3_PROMPT_LEN - ids.len();
        let mut padded = vec![0i32; pad_n];
        padded.extend_from_slice(&ids);
        te.encode(&padded).map_err(|e| anyhow::anyhow!("encode: {e}"))
    };
    let conds: Vec<Tensor> = prompts.iter().enumerate()
        .map(|(i, p)| {
            let c = encode(p)?;
            log::info!("  prompt {}/{}: cond shape={:?}", i + 1, prompts.len(), c.shape().dims());
            Ok(c)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let uncond = encode(&args.negative_prompt)?;
    drop(te);

    // 2. Load model.
    log::info!("[2/4] Loading LTX-2 transformer...");
    let mut shard_paths: Vec<PathBuf> = std::fs::read_dir(&args.transformer_dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("safetensors"))
        .collect();
    shard_paths.sort();
    if shard_paths.is_empty() {
        anyhow::bail!("no safetensors shards in {:?}", args.transformer_dir);
    }
    let mut tc = TrainConfig::default();
    tc.training_method = TrainingMethod::Lora;
    tc.lora_rank = args.lora_rank as u64;
    tc.lora_alpha = args.lora_alpha;
    let mut model = Ltx2Model::load(&shard_paths, &tc, device.clone())?;
    model.num_frames = f_lat;
    if let Some(lp) = &args.lora_path {
        model.load_weights(lp.to_str().unwrap())?;
        log::info!("  Applied LoRA from {:?} (rank={}, alpha={})",
            lp, args.lora_rank, args.lora_alpha);
    }

    // 3. Denoise — once per prompt, all latents collected.
    log::info!("[3/4] Denoising {} prompt(s) × {} steps...", conds.len(), args.steps);
    let n_tokens = f_lat * h_lat * w_lat;
    let sigmas = ltx2_sampler::schedule(args.steps, n_tokens);
    let pad_width = std::cmp::max(3, conds.len().to_string().len());
    let mut latents: Vec<Tensor> = Vec::with_capacity(conds.len());

    for (idx, cond) in conds.iter().enumerate() {
        log::info!("  [{}/{}] denoising prompt...", idx + 1, conds.len());
        use rand::SeedableRng;
        let _rng = rand::rngs::StdRng::seed_from_u64(args.seed);
        let mut latent = Tensor::randn(
            Shape::from_dims(&[1, 128, f_lat, h_lat, w_lat]),
            0.0, 1.0, device.clone(),
        )?
        .to_dtype(DType::BF16)?;

        for step in 0..args.steps {
            let sigma = sigmas[step];
            let sigma_next = sigmas[step + 1];
            let t = ltx2_sampler::sigma_to_timestep(sigma);
            let t_tensor = Tensor::from_vec(vec![t], Shape::from_dims(&[1]), device.clone())?;

            let pred_cond = model.forward(&latent, cond, &t_tensor, args.fps)?;
            let pred_uncond = model.forward(&latent, &uncond, &t_tensor, args.fps)?;
            let pred = pred_uncond.add(&pred_cond.sub(&pred_uncond)?.mul_scalar(args.guidance)?)?;

            latent = ltx2_sampler::euler_step(&latent, &pred, sigma, sigma_next)?;

            if step % 5 == 0 || step == args.steps - 1 {
                log::info!("    prompt {}/{} step {}/{} sigma={:.4}",
                    idx + 1, conds.len(), step + 1, args.steps, sigma);
            }
        }
        latents.push(latent);
    }

    // 4. VAE decode (single VAE load for all latents).
    log::info!("[4/4] VAE decoding {} latent(s)...", latents.len());
    let vae = Ltx2Vae::load(&args.vae_path, device.clone())
        .map_err(|e| anyhow::anyhow!("vae load: {e}"))?;

    for (idx, latent) in latents.iter().enumerate() {
        let denormed = vae.denormalize(latent)
            .map_err(|e| anyhow::anyhow!("denormalize prompt {}: {e}", idx + 1))?;
        let pixel_video = vae.decode_video(&denormed)
            .map_err(|e| anyhow::anyhow!("decode_video prompt {}: {e}", idx + 1))?;

        // 5. Save first frame as PNG. TODO: full video MP4 encode for F > 1.
        let dims = pixel_video.shape().dims();
        let (_b, c, f, h, w) = (dims[0], dims[1], dims[2], dims[3], dims[4]);
        if idx == 0 {
            log::info!("Decoded video: B={} C={} F={} H={} W={}", _b, c, f, h, w);
        }
        if f > 1 && idx == 0 {
            log::warn!(
                "Multi-frame video output: only saving first frame as PNG per latent. \
                 Implement video-MP4 export for F > 1."
            );
        }
        let frame0 = pixel_video.narrow(2, 0, 1)?.contiguous()?;
        let pixels: Vec<f32> = frame0.to_dtype(DType::F32)?.to_vec()?;
        let mut buf = vec![0u8; h * w * 3];
        for y in 0..h {
            for x in 0..w {
                for ch in 0..c.min(3) {
                    let id = ch * h * w + y * w + x;
                    let v = pixels.get(id).copied().unwrap_or(0.0);
                    buf[(y * w + x) * 3 + ch] = ((v.clamp(-1.0, 1.0) + 1.0) * 127.5) as u8;
                }
            }
        }
        let out_path = if multi_mode {
            let dir = args.output_dir.as_ref().unwrap();
            dir.join(format!("sample_{:0>width$}.png", idx + 1, width = pad_width))
        } else {
            args.output.clone()
        };
        image::save_buffer(&out_path, &buf, w as u32, h as u32, image::ColorType::Rgb8)?;
        log::info!("  [{}/{}] saved {:?}", idx + 1, latents.len(), out_path);
    }
    log::info!("Done — {} sample(s) saved", latents.len());
    Ok(())
}
