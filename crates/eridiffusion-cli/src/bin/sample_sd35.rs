//! sample_sd35 — text → SD 3.5 image generation, optional `--lora-path`.
//!
//! Pipeline (matches `inference-flame/sd3_lora_infer`):
//!   1. CLIP-L + CLIP-G + T5-XXL → combined `(context, pooled)`.
//!   2. Init noise ε ~ N(0, I) at `[1, 16, H/8, W/8]`.
//!   3. Build flow-matching schedule with shift=3.0 (SD3 reference default).
//!   4. Euler integration with CFG: `pred = uncond + cfg*(cond - uncond)`.
//!   5. SD3 16-channel VAE decode → save PNG.

use clap::Parser;
use flame_core::{DType, Shape, Tensor};
use eridiffusion_core::config::{TrainConfig, TrainingMethod};
use eridiffusion_core::encoders::{
    clip_g::ClipGEncoder,
    clip_l::{ClipConfig, ClipEncoder},
    flux_vae_decoder::LdmVAEDecoder,
    t5_xxl::T5Encoder,
};
use eridiffusion_core::models::{sd35::SD35Model, TrainableModel};
use std::collections::HashMap;
use std::path::PathBuf;

const CLIP_MAX_LEN: usize = 77;
const CLIP_L_PAD_ID: i32 = 49407;
const CLIP_G_PAD_ID: i32 = 0;

const VAE_LATENT_CHANNELS: usize = 16;
const VAE_SCALE: f32 = 1.5305;
const VAE_SHIFT: f32 = 0.0609;

#[derive(Parser)]
struct Args {
    /// Single prompt. Mutually exclusive with `--prompts-file`.
    #[arg(long)] prompt: Option<String>,
    /// Newline-separated prompts file for batch sampling. Blank lines and
    /// `#`-prefixed comments are skipped. Requires `--output-dir`. CLIPs +
    /// T5 load once for all prompts; DiT and VAE each load once total.
    #[arg(long)] prompts_file: Option<PathBuf>,
    #[arg(long, default_value = "")] negative_prompt: String,
    #[arg(long, default_value = "output.png")] output: PathBuf,
    /// Multi-prompt output directory. Required with `--prompts-file`.
    /// Files are written as `sample_001.png`, `sample_002.png`, ...
    #[arg(long)] output_dir: Option<PathBuf>,

    /// SD 3.5 transformer (combined ckpt or DiT-only). Same file works for VAE.
    #[arg(long)] transformer: PathBuf,
    /// Optional separate VAE ckpt; defaults to --transformer (combined).
    #[arg(long)] vae_ckpt: Option<PathBuf>,
    #[arg(long)] clip_l_ckpt: PathBuf,
    #[arg(long)] clip_g_ckpt: PathBuf,
    #[arg(long)] t5_ckpt: PathBuf,
    #[arg(long)] clip_l_tokenizer: PathBuf,
    #[arg(long)] clip_g_tokenizer: PathBuf,
    #[arg(long)] t5_tokenizer: PathBuf,

    #[arg(long, default_value = "1024")] size: usize,
    #[arg(long, default_value = "28")] steps: usize,
    /// CFG scale. Set to 1.0 to disable CFG (single forward).
    #[arg(long, default_value = "4.5")] cfg_scale: f32,
    /// Discrete-flow schedule shift. SD3 reference default 3.0.
    #[arg(long, default_value = "3.0")] shift: f32,
    #[arg(long, default_value = "256")] t5_max_len: usize,
    #[arg(long, default_value = "42")] seed: u64,

    /// Optional trained LoRA safetensors (PEFT-format keys produced by train_sd35).
    #[arg(long)] lora_path: Option<PathBuf>,
    #[arg(long, default_value = "16")] lora_rank: usize,
    #[arg(long, default_value = "1.0")] lora_alpha: f64,
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

fn load_one_or_dir(
    path: &std::path::Path, device: &std::sync::Arc<flame_core::CudaDevice>,
) -> flame_core::Result<HashMap<String, Tensor>> {
    if path.is_file() { return flame_core::serialization::load_file(path, device); }
    let mut all = HashMap::new();
    let mut entries: Vec<PathBuf> = std::fs::read_dir(path)
        .map_err(|e| flame_core::Error::Io(format!("read_dir: {e}")))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("safetensors"))
        .collect();
    entries.sort();
    for p in entries {
        all.extend(flame_core::serialization::load_file(&p, device)?);
    }
    Ok(all)
}

fn tokenize_clip(tok: &tokenizers::Tokenizer, text: &str, pad_id: i32) -> anyhow::Result<Vec<i32>> {
    let enc = tok.encode(text, true).map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
    let mut ids: Vec<i32> = enc.get_ids().iter().map(|&x| x as i32).collect();
    if ids.len() > CLIP_MAX_LEN { ids.truncate(CLIP_MAX_LEN); }
    while ids.len() < CLIP_MAX_LEN { ids.push(pad_id); }
    Ok(ids)
}

fn tokenize_t5(tok: &tokenizers::Tokenizer, text: &str, max_len: usize) -> anyhow::Result<Vec<i32>> {
    let enc = tok.encode(text, true).map_err(|e| anyhow::anyhow!("t5 tokenize: {e}"))?;
    let mut ids: Vec<i32> = enc.get_ids().iter().map(|&x| x as i32).collect();
    if ids.len() > max_len { ids.truncate(max_len); }
    while ids.len() < max_len { ids.push(0); }
    Ok(ids)
}

fn encode_sd3_prompt(
    text: &str,
    clip_l: &ClipEncoder, clip_g: &ClipGEncoder, t5: &mut T5Encoder,
    tok_l: &tokenizers::Tokenizer, tok_g: &tokenizers::Tokenizer, tok_t5: &tokenizers::Tokenizer,
    t5_max_len: usize,
    device: &std::sync::Arc<flame_core::CudaDevice>,
) -> anyhow::Result<(Tensor, Tensor)> {
    let ids_l = tokenize_clip(tok_l, text, CLIP_L_PAD_ID)?;
    let (clip_l_h, clip_l_pool) = clip_l.encode_sd3(&ids_l)?;
    let ids_g = tokenize_clip(tok_g, text, CLIP_G_PAD_ID)?;
    let (clip_g_h, clip_g_pool) = clip_g.encode_sdxl(&ids_g)?;
    let ids_t5 = tokenize_t5(tok_t5, text, t5_max_len)?;
    let t5_h = t5.encode(&ids_t5)?;

    let clip_lg = Tensor::cat(&[&clip_l_h, &clip_g_h], 2)?;
    let pad_zeros = Tensor::zeros_dtype(
        Shape::from_dims(&[clip_lg.dims()[0], clip_lg.dims()[1], 4096 - 2048]),
        DType::BF16, device.clone(),
    )?;
    let clip_lg_padded = Tensor::cat(&[&clip_lg.to_dtype(DType::BF16)?, &pad_zeros], 2)?;
    let context = Tensor::cat(&[&clip_lg_padded, &t5_h.to_dtype(DType::BF16)?], 1)?
        .to_dtype(DType::BF16)?;
    let pooled = Tensor::cat(&[&clip_l_pool, &clip_g_pool], 1)?.to_dtype(DType::BF16)?;
    Ok((context, pooled))
}

fn build_schedule(num_steps: usize, shift: f32) -> Vec<f32> {
    let mut t: Vec<f32> = (0..=num_steps)
        .map(|i| 1.0 - i as f32 / num_steps as f32)
        .collect();
    if (shift - 1.0).abs() > f32::EPSILON {
        for v in t.iter_mut() {
            if *v > 0.0 && *v < 1.0 {
                *v = shift * *v / (1.0 + (shift - 1.0) * *v);
            }
        }
    }
    t
}

fn save_png(rgb: &Tensor, path: &std::path::Path) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() { std::fs::create_dir_all(parent)?; }
    let rgb_f32 = rgb.to_dtype(DType::F32)?;
    let data = rgb_f32.to_vec()?;
    let dims = rgb_f32.dims().to_vec();
    let (h, w) = (dims[2], dims[3]);
    let mut pixels = vec![0u8; h * w * 3];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let idx = c * h * w + y * w + x;
                let v = data[idx].clamp(-1.0, 1.0);
                let u = ((v + 1.0) * 127.5).round().clamp(0.0, 255.0) as u8;
                pixels[(y * w + x) * 3 + c] = u;
            }
        }
    }
    image::RgbImage::from_raw(w as u32, h as u32, pixels)
        .ok_or_else(|| anyhow::anyhow!("RgbImage::from_raw failed"))?
        .save(path)?;
    Ok(())
}

fn main() -> anyhow::Result<()> {
    use rand::{Rng, SeedableRng};
    env_logger::init();
    let args = Args::parse();
    let device = flame_core::global_cuda_device();
    let _no_grad = flame_core::autograd::AutogradContext::no_grad();
    flame_core::config::set_default_dtype(DType::BF16);

    let h_lat = args.size / 8;
    let w_lat = args.size / 8;
    log::info!("Sampling SD 3.5: {}x{} → latent {}x{} (cfg={}, steps={}, shift={})",
        args.size, args.size, h_lat, w_lat, args.cfg_scale, args.steps, args.shift);

    // 1. Load text encoders + tokenizers
    log::info!("[1/5] Loading text encoders...");
    // CLIP-L from HF ships as F32; layer_norm_bf16 is BF16-strict.
    // Cast at load (same fix as sample_flux/sample_sdxl).
    let clip_l_w = load_one_or_dir(&args.clip_l_ckpt, &device)?;
    let clip_l_w: std::collections::HashMap<String, flame_core::Tensor> = clip_l_w
        .into_iter()
        .map(|(k, t)| Ok::<_, anyhow::Error>((k, t.to_dtype(DType::BF16)?)))
        .collect::<anyhow::Result<_>>()?;
    let clip_l = ClipEncoder::new(clip_l_w, ClipConfig::default(), device.clone());
    let clip_g_w = load_one_or_dir(&args.clip_g_ckpt, &device)?;
    let clip_g_w: std::collections::HashMap<String, flame_core::Tensor> = clip_g_w
        .into_iter()
        .map(|(k, t)| Ok::<_, anyhow::Error>((k, t.to_dtype(DType::BF16)?)))
        .collect::<anyhow::Result<_>>()?;
    let clip_g = ClipGEncoder::new(clip_g_w, device.clone());
    let mut t5 = T5Encoder::load(
        args.t5_ckpt.to_str().ok_or_else(|| anyhow::anyhow!("t5 path utf8"))?,
        &device,
    )?;
    let tok_l = tokenizers::Tokenizer::from_file(&args.clip_l_tokenizer)
        .map_err(|e| anyhow::anyhow!("clip_l tokenizer: {e}"))?;
    let tok_g = tokenizers::Tokenizer::from_file(&args.clip_g_tokenizer)
        .map_err(|e| anyhow::anyhow!("clip_g tokenizer: {e}"))?;
    let tok_t5 = tokenizers::Tokenizer::from_file(&args.t5_tokenizer)
        .map_err(|e| anyhow::anyhow!("t5 tokenizer: {e}"))?;

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

    // 2. Encode prompts (cond × N + uncond if CFG enabled). Encoders all
    // load once and stay resident until every prompt is encoded.
    // T5-XXL leaves heavy intermediate activations in the alloc pool;
    // clear between encodes so they're recycled (prior loop OOM'd at
    // ~3 prompts × T5-XXL on a 24 GB card).
    log::info!("[2/5] Encoding {} prompt(s) + uncond...", prompts.len());
    let mut conds: Vec<(Tensor, Tensor)> = Vec::with_capacity(prompts.len());
    for (i, p) in prompts.iter().enumerate() {
        let pair = encode_sd3_prompt(
            p, &clip_l, &clip_g, &mut t5,
            &tok_l, &tok_g, &tok_t5, args.t5_max_len, &device,
        )?;
        log::info!("  prompt {}/{}: ctx={:?} pool={:?}",
            i + 1, prompts.len(), pair.0.shape().dims(), pair.1.shape().dims());
        conds.push(pair);
        flame_core::cuda_alloc_pool::clear_pool_cache();
        flame_core::trim_cuda_mempool(0);
    }
    let do_cfg = args.cfg_scale > 1.0;
    let uncond = if do_cfg {
        Some(encode_sd3_prompt(
            &args.negative_prompt, &clip_l, &clip_g, &mut t5,
            &tok_l, &tok_g, &tok_t5, args.t5_max_len, &device,
        )?)
    } else { None };
    drop(clip_l); drop(clip_g); drop(t5);
    flame_core::cuda_alloc_pool::clear_pool_cache();
    flame_core::trim_cuda_mempool(0);

    // 3. Load DiT (+ optional LoRA)
    log::info!("[3/5] Loading SD 3.5 transformer...");
    let shards = collect_shards(&args.transformer)?;
    let mut tc = TrainConfig::default();
    let lora_mode = args.lora_path.is_some();
    if lora_mode {
        tc.training_method = TrainingMethod::Lora;
        tc.lora_rank = args.lora_rank as u64;
        tc.lora_alpha = args.lora_alpha;
    } else {
        // Inference without LoRA: same trick as `sample_sdxl.rs` — load with
        // training_method=Lora at rank 1 so `SD35Model::load` doesn't try to
        // promote every weight to F32. Zero-init B means the rank-1 adapters
        // contribute nothing to the forward pass.
        tc.training_method = TrainingMethod::Lora;
        tc.lora_rank = 1;
        tc.lora_alpha = 1.0;
    }
    let mut model = SD35Model::load(&shards, &tc, device.clone())?;
    if let Some(lp) = &args.lora_path {
        model.load_weights(lp.to_str().unwrap())
            .map_err(|e| anyhow::anyhow!("load_weights {}: {e}", lp.display()))?;
        log::info!("  Applied LoRA from {:?} (rank={}, alpha={})",
            lp, args.lora_rank, args.lora_alpha);
    }

    // 4. Init noise + denoise — once per prompt, all latents collected
    //    while the DiT is resident.
    let timesteps = build_schedule(args.steps, args.shift);
    log::info!("[4/5] Denoising {} prompt(s) × {} steps...", conds.len(), args.steps);
    let pad_width = std::cmp::max(3, conds.len().to_string().len());
    let mut latents: Vec<Tensor> = Vec::with_capacity(conds.len());

    for (idx, (ctx_cond, pool_cond)) in conds.iter().enumerate() {
        log::info!("  [{}/{}] denoising prompt...", idx + 1, conds.len());
        let numel = VAE_LATENT_CHANNELS * h_lat * w_lat;
        let mut rng = rand::rngs::StdRng::seed_from_u64(args.seed);
        let mut data = Vec::with_capacity(numel);
        while data.len() < numel {
            let u1 = rng.gen::<f32>().max(1e-10);
            let u2 = rng.gen::<f32>();
            let mag = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f32::consts::PI * u2;
            data.push(mag * theta.cos());
            if data.len() < numel { data.push(mag * theta.sin()); }
        }
        let mut latent = Tensor::from_vec(
            data, Shape::from_dims(&[1, VAE_LATENT_CHANNELS, h_lat, w_lat]), device.clone(),
        )?.to_dtype(DType::BF16)?;

        for i in 0..args.steps {
            let t_curr = timesteps[i];
            let t_next = timesteps[i + 1];
            let t_vec = Tensor::from_vec(
                vec![t_curr * 1000.0], Shape::from_dims(&[1]), device.clone(),
            )?.to_dtype(DType::BF16)?;
            let pred_cond = <SD35Model as TrainableModel>::forward(
                &mut model, &latent, &t_vec,
                std::slice::from_ref(ctx_cond), Some(pool_cond),
            )?;
            let pred = if let Some((ref ctx_u, ref pool_u)) = uncond {
                let pred_uncond = <SD35Model as TrainableModel>::forward(
                    &mut model, &latent, &t_vec,
                    std::slice::from_ref(ctx_u), Some(pool_u),
                )?;
                let diff = pred_cond.sub(&pred_uncond)?;
                pred_uncond.add(&diff.mul_scalar(args.cfg_scale)?)?
            } else { pred_cond };
            let dt = t_next - t_curr;
            latent = latent.add(&pred.mul_scalar(dt)?)?;
            if i % 5 == 0 || i == args.steps - 1 {
                log::info!("    prompt {}/{} step {}/{} t={:.4}",
                    idx + 1, conds.len(), i + 1, args.steps, t_curr);
            }
        }
        latents.push(latent);
    }

    // 5. VAE decode (single VAE load for all latents)
    log::info!("[5/5] VAE decoding {} latent(s)...", latents.len());
    drop(model);
    let vae_path = args.vae_ckpt.as_ref().unwrap_or(&args.transformer);
    let vae = LdmVAEDecoder::from_safetensors(
        vae_path.to_str().ok_or_else(|| anyhow::anyhow!("vae path utf8"))?,
        VAE_LATENT_CHANNELS, VAE_SCALE, VAE_SHIFT, &device,
    )?;
    for (idx, latent) in latents.iter().enumerate() {
        let rgb = vae.decode(latent)?;
        let out_path = if multi_mode {
            let dir = args.output_dir.as_ref().unwrap();
            dir.join(format!("sample_{:0>width$}.png", idx + 1, width = pad_width))
        } else {
            args.output.clone()
        };
        save_png(&rgb, &out_path)?;
        log::info!("  [{}/{}] saved {:?}", idx + 1, latents.len(), out_path);
    }
    log::info!("Done — {} sample(s) saved", latents.len());
    Ok(())
}
