//! sample_ernie — text → Ernie image generation. Tests the sampling pipeline.
//! Supports `--lora-path` to overlay a trained LoRA on top of the base transformer.
use clap::Parser;
use std::path::PathBuf;
use flame_core::{DType, Shape, Tensor};
use eridiffusion_core::config::{TrainConfig, TrainingMethod};
use eridiffusion_core::encoders::{vae::KleinVaeDecoder, mistral3b::Mistral3bEncoder};
use eridiffusion_core::models::{ErnieModel, TrainableModel};
use eridiffusion_core::sampler::ernie_sampler;

#[derive(Parser)]
struct Args {
    /// Single prompt. Mutually exclusive with `--prompts-file`.
    #[arg(long)] prompt: Option<String>,
    /// Newline-separated prompts file for batch sampling. Blank lines and
    /// `#`-prefixed comments are skipped. Requires `--output-dir`. Encoder
    /// loads once for all prompts; DiT and VAE each load once total.
    #[arg(long)] prompts_file: Option<PathBuf>,
    #[arg(long, default_value = "output.png")] output: PathBuf,
    /// Multi-prompt output directory. Required with `--prompts-file`.
    /// Files are written as `sample_001.png`, `sample_002.png`, ...
    #[arg(long)] output_dir: Option<PathBuf>,
    #[arg(long)] transformer_dir: PathBuf,
    #[arg(long)] vae_path: PathBuf,
    #[arg(long)] text_ckpt: PathBuf,
    #[arg(long)] tokenizer_path: PathBuf,
    #[arg(long, default_value = "1024")] size: usize,
    #[arg(long, default_value = "20")] steps: usize,
    #[arg(long, default_value = "4.0")] guidance: f32,
    #[arg(long, default_value = "42")] seed: u64,
    /// Optional safetensors of a trained LoRA (matches `train_ernie` save format).
    #[arg(long)] lora_path: Option<PathBuf>,
    /// Rank used when the LoRA was trained. Must match or load fails.
    #[arg(long, default_value = "16")] lora_rank: usize,
    /// Alpha used when the LoRA was trained.
    #[arg(long, default_value = "1.0")] lora_alpha: f64,
    /// Per-layer block offloading (LoRA-only): drop transformer layer weights from VRAM
    /// and stream them from disk per-layer. Frees ~10 GB; required for 1024² on 24 GB.
    #[arg(long)] offload: bool,
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let args = Args::parse();
    let device = flame_core::global_cuda_device();
    let _no_grad = flame_core::autograd::AutogradContext::no_grad();
    flame_core::config::set_default_dtype(DType::BF16);

    let hp = args.size / 16; // 8x VAE spatial compression, 2x patchify
    let wp = args.size / 16;
    log::info!("Size {}x{} → latent {}x{}", args.size, args.size, hp, wp);

    // 1. Text encode
    // Match upstream Python (model/ErnieModel.py:5,128-135): PROMPT_MAX_LENGTH=512,
    // pad_token_id=11 per text_encoder/config.json. **Same params at train time.**
    const ERNIE_MAX_LEN: usize = 512;
    const ERNIE_PAD_ID: i32 = 11;
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

    log::info!("[1/3] Text encoding {} prompt(s) (max_len={ERNIE_MAX_LEN}, pad_id={ERNIE_PAD_ID})...", prompts.len());
    let tokenizer = tokenizers::Tokenizer::from_file(&args.tokenizer_path)
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    let encode = |text: &str| -> anyhow::Result<(Vec<i32>, usize)> {
        let e = tokenizer.encode(text, true).map_err(|e| anyhow::anyhow!("{e}"))?;
        let mut ids: Vec<i32> = e.get_ids().iter().map(|&x| x as i32).collect();
        if ids.len() > ERNIE_MAX_LEN { ids.truncate(ERNIE_MAX_LEN); }
        let real_len = ids.len();
        Ok((ids, real_len))
    };
    let prompt_ids: Vec<(Vec<i32>, usize)> = prompts.iter()
        .map(|p| encode(p))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let (uncond_ids, uncond_len) = encode("")?;
    let te = Mistral3bEncoder::load(args.text_ckpt.to_str().unwrap(), &device)?;
    // Encode every prompt with the live TE before dropping.
    let cond_pairs: Vec<(Tensor, usize)> = prompt_ids.iter().enumerate()
        .map(|(i, (ids, len))| {
            let emb = te.encode_with_pad(ids, ERNIE_MAX_LEN, ERNIE_PAD_ID)?;
            log::info!("  prompt {}/{}: cond shape={:?} (real_len={})",
                i + 1, prompts.len(), emb.shape().dims(), len);
            Ok((emb, *len))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let uncond = te.encode_with_pad(&uncond_ids, ERNIE_MAX_LEN, ERNIE_PAD_ID)?;
    drop(te);
    log::info!("  uncond={:?} (real_len={uncond_len})", uncond.shape().dims());

    // 2. Load model. Base-only takes the manual no-parameters path (avoids the F32
    // full-weight parameter copy that ErnieModel::load does for FineTune mode and that
    // OOMs on a 24 GB card). LoRA path goes through ErnieModel::load with TrainingMethod::Lora
    // so adapters get allocated, then load_weights overwrites them with the trained values.
    log::info!("[2/3] Loading DiT...");
    let model: ErnieModel = if let Some(lp) = &args.lora_path {
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
        let mut m = ErnieModel::load(&shard_paths, &tc, device.clone())?;
        m.load_weights(lp.to_str().unwrap())?;
        log::info!("  Applied LoRA from {:?} (rank={}, alpha={})",
            lp, args.lora_rank, args.lora_alpha);
        if args.offload {
            m.enable_offload(shard_paths.clone())?;
            log::info!("  Block offload enabled — per-layer streaming from {} shards", shard_paths.len());
        }
        m
    } else {
        let mut shard_paths: Vec<PathBuf> = std::fs::read_dir(&args.transformer_dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("safetensors"))
            .collect();
        shard_paths.sort();
        if shard_paths.is_empty() {
            anyhow::bail!("no safetensors shards in {:?}", args.transformer_dir);
        }
        // Load shared (non-layer) weights only if offloading; otherwise load all.
        let all_weights = if args.offload {
            // Load only non-layer weights — layer weights will stream per-block.
            let shared_prefixes = [
                "x_embedder.", "text_proj.", "time_embedding.", "time_proj.",
                "adaLN_modulation.", "final_norm.", "final_linear.", "pos_embed.",
            ];
            let mut wt = std::collections::HashMap::new();
            for p in &shard_paths {
                let part = flame_core::serialization::load_file(p, &device)?;
                for (k, v) in part {
                    if shared_prefixes.iter().any(|px| k.starts_with(px)) {
                        wt.insert(k, v.to_dtype(flame_core::DType::BF16).unwrap_or(v));
                    }
                }
            }
            wt
        } else {
            let mut wt = std::collections::HashMap::new();
            for p in &shard_paths {
                let part = flame_core::serialization::load_file(p, &device)?;
                wt.extend(part);
            }
            wt
        };
        let mut m = ErnieModel {
            config: TrainConfig::default(),
            device: device.clone(),
            weights: all_weights,
            lora_adapters: Vec::new(),
            parameters: Vec::new(),
            is_lora: false,
            offloader: None,
        };
        if args.offload {
            m.enable_offload(shard_paths.clone())?;
            log::info!("  Block offload enabled (base model) — per-layer streaming from {} shards", shard_paths.len());
        }
        m
    };
    let mut config = model;

    // 3. Denoise — collect a final latent per prompt while DiT is resident.
    log::info!("[3/4] Denoising {} prompt(s) × {} steps...", prompts.len(), args.steps);
    let sigmas = ernie_sampler::schedule(args.steps);
    let pad_width = std::cmp::max(3, prompts.len().to_string().len());

    // Trim uncond once (same across all prompts).
    let trim_uncond = uncond.narrow(1, 0, uncond_len.min(ERNIE_MAX_LEN).max(1))?.contiguous()?;

    let mut latents: Vec<Tensor> = Vec::with_capacity(cond_pairs.len());
    for (idx, (embeds, len)) in cond_pairs.iter().enumerate() {
        log::info!("  [{}/{}] denoising prompt...", idx + 1, cond_pairs.len());
        let trim_cond = embeds.narrow(1, 0, len.min(&ERNIE_MAX_LEN).max(&1).clone())?.contiguous()?;

        // Same noise for every prompt (fixed seed). If varied noise is
        // wanted, advance seed by `idx`.
        let mut latent = {
            use rand::SeedableRng;
            let _rng = rand::rngs::StdRng::seed_from_u64(args.seed);
            Tensor::randn(Shape::from_dims(&[1, 128, hp, wp]), 0.0, 1.0, device.clone())?
                .to_dtype(DType::BF16)?
        };

        for step in 0..args.steps {
            let sigma = sigmas[step];
            let sigma_next = sigmas[step + 1];
            let t = ernie_sampler::sigma_to_timestep(sigma);
            let t_tensor = Tensor::from_vec(vec![t], Shape::from_dims(&[1]), device.clone())?;

            // Sequential CFG: pred = uncond + guidance * (cond - uncond)
            let pred_cond = config.forward(&latent, &trim_cond, &t_tensor)?;
            let pred_uncond = config.forward(&latent, &trim_uncond, &t_tensor)?;
            let pred = pred_uncond.add(&pred_cond.sub(&pred_uncond)?.mul_scalar(args.guidance)?)?;

            latent = ernie_sampler::euler_step(&latent, &pred, sigma, sigma_next)?;

            if step % 10 == 0 || step == args.steps - 1 {
                log::info!("    prompt {}/{} step {}/{}, sigma={:.4}",
                    idx + 1, cond_pairs.len(), step + 1, args.steps, sigma);
            }
        }
        latents.push(latent);
    }

    // 4. VAE decode + save (single VAE load for all latents).
    log::info!("[4/4] VAE decoding {} latent(s)...", latents.len());
    drop(config); // free DiT memory before VAE
    let vae_weights = flame_core::serialization::load_file(&args.vae_path, &device)?;
    let dev = flame_core::Device::from(device.clone());
    let decoder = KleinVaeDecoder::load(&vae_weights, &dev)?;

    for (idx, latent) in latents.iter().enumerate() {
        let img = decoder.decode(latent)?;
        let pixels: Vec<f32> = img.to_dtype(DType::F32)?.to_vec()?;
        let dims = img.shape().dims();
        let (c, h, w) = if dims.len() == 4 { (dims[1], dims[2], dims[3]) } else { (3, dims[0], dims[1]) };
        let mut buf = vec![0u8; c * h * w];
        for y in 0..h {
            for x in 0..w {
                for ch in 0..c {
                    let id = if dims.len() == 4 { ch * h * w + y * w + x } else { y * w * c + x * c + ch };
                    let v = pixels.get(id).copied().unwrap_or(0.0);
                    buf[y * w * c + x * c + ch] = ((v.clamp(-1.0, 1.0) + 1.0) * 127.5) as u8;
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
