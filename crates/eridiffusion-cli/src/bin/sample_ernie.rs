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
    #[arg(long)] prompt: String,
    #[arg(long, default_value = "output.png")] output: PathBuf,
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
    log::info!("[1/3] Text encoding (max_len={ERNIE_MAX_LEN}, pad_id={ERNIE_PAD_ID})...");
    let tokenizer = tokenizers::Tokenizer::from_file(&args.tokenizer_path)
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    let encode = |text: &str| -> anyhow::Result<(Vec<i32>, usize)> {
        let e = tokenizer.encode(text, true).map_err(|e| anyhow::anyhow!("{e}"))?;
        let mut ids: Vec<i32> = e.get_ids().iter().map(|&x| x as i32).collect();
        if ids.len() > ERNIE_MAX_LEN { ids.truncate(ERNIE_MAX_LEN); }
        let real_len = ids.len();
        Ok((ids, real_len))
    };
    let (ids, len) = encode(&args.prompt)?;
    let (uncond_ids, uncond_len) = encode("")?;
    let te = Mistral3bEncoder::load(args.text_ckpt.to_str().unwrap(), &device)?;
    let embeds = te.encode_with_pad(&ids, ERNIE_MAX_LEN, ERNIE_PAD_ID)?;
    let uncond = te.encode_with_pad(&uncond_ids, ERNIE_MAX_LEN, ERNIE_PAD_ID)?;
    drop(te);
    log::info!("  Text encoded: cond={:?} (real_len={len}) uncond={:?} (real_len={uncond_len})",
        embeds.shape().dims(), uncond.shape().dims());

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
            offload_shards: None,
        };
        if args.offload {
            m.enable_offload(shard_paths.clone())?;
            log::info!("  Block offload enabled (base model) — per-layer streaming from {} shards", shard_paths.len());
        }
        m
    };
    let mut config = model;

    // 3. Denoise
    log::info!("[3/3] Denoising {} steps...", args.steps);
    let sigmas = ernie_sampler::schedule(args.steps);
    let mut latent = {
        use rand::SeedableRng;
        let rng = rand::rngs::StdRng::seed_from_u64(args.seed);
        Tensor::randn(Shape::from_dims(&[1, 128, hp, wp]), 0.0, 1.0, device.clone())?
            .to_dtype(DType::BF16)?
    };

    // Trim padded text positions to real-token-count, matching upstream Python
    // ErnieModel.py:153-154 and the trainer side. After this, the DiT only
    // ever attends to real text tokens — n_txt for image-token RoPE matches
    // training (which uses the same trim).
    let cond_l = vec![len.min(ERNIE_MAX_LEN).max(1)];
    let uncond_l = vec![uncond_len.min(ERNIE_MAX_LEN).max(1)];
    let trim_cond = embeds.narrow(1, 0, cond_l[0])?.contiguous()?;
    let trim_uncond = uncond.narrow(1, 0, uncond_l[0])?.contiguous()?;

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
            log::info!("  step {}/{}, sigma={:.4}", step + 1, args.steps, sigma);
        }
    }

    // 4. VAE decode
    log::info!("[4/4] VAE decoding...");
    let vae_weights = flame_core::serialization::load_file(&args.vae_path, &device)?;
    let dev = flame_core::Device::from(device.clone());
    let decoder = KleinVaeDecoder::load(&vae_weights, &dev)?;
    let img = decoder.decode(&latent)?;

    // 5. Save image
    let pixels: Vec<f32> = img.to_dtype(DType::F32)?.to_vec()?;
    let dims = img.shape().dims();
    let (c, h, w) = if dims.len() == 4 { (dims[1], dims[2], dims[3]) } else { (3, dims[0], dims[1]) };
    let mut buf = vec![0u8; c * h * w];
    for y in 0..h {
        for x in 0..w {
            for ch in 0..c {
                let idx = if dims.len() == 4 { ch * h * w + y * w + x } else { y * w * c + x * c + ch };
                let v = pixels.get(idx).copied().unwrap_or(0.0);
                buf[y * w * c + x * c + ch] = ((v.clamp(-1.0, 1.0) + 1.0) * 127.5) as u8;
            }
        }
    }
    image::save_buffer(&args.output, &buf, w as u32, h as u32, image::ColorType::Rgb8)?;
    log::info!("Saved to {:?}", args.output);
    Ok(())
}
