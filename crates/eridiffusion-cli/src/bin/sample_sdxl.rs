//! sample_sdxl — text → SDXL image generation. Tests the sampling pipeline.
//! Supports `--lora-path` to overlay a trained LoRA on top of the base UNet.
//!
//! Pipeline:
//!   1. Tokenize prompt for both CLIP-L and CLIP-G.
//!   2. CLIP-L → penultimate hidden [768] + dummy pool (unused).
//!      CLIP-G → penultimate hidden [1280] + projected pool [1280].
//!   3. Concat hiddens along last dim → context [1, 77, 2048].
//!      Concat CLIP-G pool [1280] + size_emb [1536] → y [1, 2816].
//!   4. Init noise ε ~ N(0, I) at [1, 4, H/8, W/8].
//!   5. DDIM denoising with epsilon prediction (preset default), 20 steps,
//!      CFG-scale 7.5 (sd-scripts SDXL default).
//!   6. SDXL VAE decode → save PNG.

use clap::Parser;
use flame_core::{DType, Shape, Tensor};
use eridiffusion_core::config::{TrainConfig, TrainingMethod};
use eridiffusion_core::encoders::{
    clip_g::ClipGEncoder,
    clip_l::{ClipConfig, ClipEncoder},
    sdxl_vae::SdxlVaeDecoder,
};
use eridiffusion_core::models::{sdxl::SDXLModel, TrainableModel};
use eridiffusion_core::sampler::sdxl_sampler::{
    build_time_ids, compute_alpha_bar, ddim_step, euler_a_step, sin_embed_256, timesteps,
    Prediction, SchedulerKind,
};
use std::collections::HashMap;
use std::path::PathBuf;

const CLIP_MAX_LEN: usize = 77;
// SDXL audit H1: split per-encoder pad ids. CLIP-L pads with EOS, CLIP-G
// pads with id 0 (HF tokenizer_2 `"pad_token": "!"`). Sharing 49407 corrupts
// CLIP-G hidden states at every pad position.
const CLIP_L_PAD_ID: i32 = 49407;
const CLIP_G_PAD_ID: i32 = 0;

#[derive(Parser)]
struct Args {
    /// Single prompt. Mutually exclusive with `--prompts-file`.
    #[arg(long)] prompt: Option<String>,
    /// Newline-separated prompts file for batch sampling. Blank lines and
    /// `#`-prefixed comments are skipped. Requires `--output-dir`. CLIPs
    /// load once for all prompts; UNet and VAE each load once total.
    #[arg(long)] prompts_file: Option<PathBuf>,
    /// Negative prompt for CFG. Empty string disables CFG (uses cond pred only).
    #[arg(long, default_value = "")] negative_prompt: String,
    #[arg(long, default_value = "output.png")] output: PathBuf,
    /// Multi-prompt output directory. Required with `--prompts-file`.
    /// Files are written as `sample_001.png`, `sample_002.png`, ...
    #[arg(long)] output_dir: Option<PathBuf>,
    /// SDXL UNet checkpoint (single safetensors or shard dir).
    #[arg(long)] unet: PathBuf,
    /// SDXL VAE.
    #[arg(long)] vae_ckpt: PathBuf,
    /// CLIP-L weights.
    #[arg(long)] clip_l_ckpt: PathBuf,
    /// CLIP-G weights.
    #[arg(long)] clip_g_ckpt: PathBuf,
    #[arg(long)] clip_l_tokenizer: PathBuf,
    #[arg(long)] clip_g_tokenizer: PathBuf,

    #[arg(long, default_value = "1024")] size: usize,
    /// Inference steps. SDXL audit H4: OT preset default is 30 (Euler-A).
    #[arg(long, default_value = "30")] steps: usize,
    #[arg(long, default_value = "7.5")] cfg_scale: f32,
    /// SDXL audit H4: CFG-rescale (Lin et al. 2023) default 0.7 per OT
    /// `__sample_base(cfg_rescale=0.7)`. Set to 0.0 for raw classifier-free
    /// guidance without rescaling.
    #[arg(long, default_value = "0.7")] cfg_rescale: f32,
    #[arg(long, default_value = "42")] seed: u64,

    /// Sampler scheduler. SDXL audit H4: OT preset default is `euler_a`.
    /// `ddim` retained as a legacy / determinism opt-in.
    #[arg(long, default_value = "euler_a")] scheduler: String,

    /// Optional safetensors of a trained LoRA (`train_sdxl` save format).
    #[arg(long)] lora_path: Option<PathBuf>,
    #[arg(long, default_value = "16")] lora_rank: usize,
    #[arg(long, default_value = "1.0")] lora_alpha: f64,

    /// Use v-prediction in the sampler (cosmos-style SDXL fine-tunes).
    #[arg(long)] v_prediction: bool,
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

fn tokenize(tok: &tokenizers::Tokenizer, text: &str, pad_id: i32) -> anyhow::Result<Vec<i32>> {
    let enc = tok.encode(text, true).map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
    let mut ids: Vec<i32> = enc.get_ids().iter().map(|&x| x as i32).collect();
    if ids.len() > CLIP_MAX_LEN { ids.truncate(CLIP_MAX_LEN); }
    while ids.len() < CLIP_MAX_LEN { ids.push(pad_id); }
    Ok(ids)
}

/// Encode one prompt into (context [1,77,2048], y [1,2816]).
fn encode_prompt(
    text: &str,
    tok_l: &tokenizers::Tokenizer, tok_g: &tokenizers::Tokenizer,
    clip_l: &ClipEncoder, clip_g: &ClipGEncoder,
    size_emb: &[f32], device: &std::sync::Arc<flame_core::CudaDevice>,
) -> anyhow::Result<(Tensor, Tensor)> {
    // SDXL audit H1: per-encoder pad ids (CLIP-L uses EOS, CLIP-G uses 0).
    let ids_l = tokenize(tok_l, text, CLIP_L_PAD_ID)?;
    let ids_g = tokenize(tok_g, text, CLIP_G_PAD_ID)?;
    let (clip_l_h, _) = clip_l.encode_sd3(&ids_l)?;
    let (clip_g_h, clip_g_p) = clip_g.encode_sdxl(&ids_g)?;
    let context = Tensor::cat(&[&clip_l_h, &clip_g_h], 2)?.to_dtype(DType::BF16)?;
    let size_t = Tensor::from_vec(size_emb.to_vec(), Shape::from_dims(&[1, 1536]), device.clone())?
        .to_dtype(DType::BF16)?;
    let y = Tensor::cat(&[&clip_g_p, &size_t], 1)?.to_dtype(DType::BF16)?;
    Ok((context, y))
}

fn save_png(rgb: &Tensor, path: &std::path::Path) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() { std::fs::create_dir_all(parent)?; }
    let rgb_f32 = rgb.to_dtype(DType::F32)?;
    let data = rgb_f32.to_vec()?;
    let dims = rgb_f32.shape().dims().to_vec();
    let (h, w) = (dims[2], dims[3]);
    let mut pixels = vec![0u8; h * w * 3];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let idx = c * h * w + y * w + x;
                let v = ((data[idx] + 1.0) * 127.5).clamp(0.0, 255.0) as u8;
                pixels[(y * w + x) * 3 + c] = v;
            }
        }
    }
    image::RgbImage::from_raw(w as u32, h as u32, pixels)
        .ok_or_else(|| anyhow::anyhow!("failed to create image"))?
        .save(path)?;
    Ok(())
}

fn main() -> anyhow::Result<()> {
    use rand::SeedableRng;
    env_logger::init();
    let args = Args::parse();
    let device = flame_core::global_cuda_device();
    let _no_grad = flame_core::autograd::AutogradContext::no_grad();
    flame_core::config::set_default_dtype(DType::BF16);

    let h_lat = args.size / 8;
    let w_lat = args.size / 8;
    let res = args.size as u32;
    let scheduler = match args.scheduler.as_str() {
        "euler_a" | "euler-a" | "eulera" => SchedulerKind::EulerA,
        "ddim" => SchedulerKind::Ddim,
        other => anyhow::bail!("unknown scheduler '{}': use 'euler_a' or 'ddim'", other),
    };
    log::info!("Sampling {}x{} → latent {}x{} (sched={:?}, cfg={}, rescale={}, steps={}, v_pred={})",
        args.size, args.size, h_lat, w_lat, scheduler, args.cfg_scale, args.cfg_rescale,
        args.steps, args.v_prediction);

    // 1. Load encoders
    log::info!("[1/4] Loading text encoders...");
    // CLIP-L safetensors from HF ship as F32; the encoder calls
    // `layer_norm_bf16` directly which is BF16-strict. Cast at load so
    // every weight (token_embedding, position_embedding, all linear
    // weights, layer_norms) is BF16. Same fix sample_flux.rs applies.
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
    let tok_l = tokenizers::Tokenizer::from_file(&args.clip_l_tokenizer)
        .map_err(|e| anyhow::anyhow!("clip_l tokenizer: {e}"))?;
    let tok_g = tokenizers::Tokenizer::from_file(&args.clip_g_tokenizer)
        .map_err(|e| anyhow::anyhow!("clip_g tokenizer: {e}"))?;

    // 2. Pre-compute size embeddings
    let time_ids = build_time_ids(res, res, 0, 0, res, res);
    let mut size_emb = Vec::with_capacity(6 * 256);
    for v in time_ids.iter() { size_emb.extend_from_slice(&sin_embed_256(*v)); }

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

    // 3. Encode cond × N / uncond — encoders all load once and stay
    //    resident until every prompt is encoded.
    log::info!("[2/4] Encoding {} prompt(s) + uncond...", prompts.len());
    let conds: Vec<(Tensor, Tensor)> = prompts.iter().enumerate()
        .map(|(i, p)| {
            let pair = encode_prompt(p, &tok_l, &tok_g, &clip_l, &clip_g, &size_emb, &device)?;
            log::info!("  prompt {}/{}: ctx={:?} y={:?}",
                i + 1, prompts.len(), pair.0.shape().dims(), pair.1.shape().dims());
            Ok(pair)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let do_cfg = args.cfg_scale > 1.0;
    let uncond_pair = if do_cfg {
        Some(encode_prompt(&args.negative_prompt, &tok_l, &tok_g, &clip_l, &clip_g, &size_emb, &device)?)
    } else { None };
    drop(clip_l); drop(clip_g); // free TE VRAM before loading UNet

    // 4. Load UNet (+ optional LoRA)
    log::info!("[3/4] Loading SDXL UNet...");
    let shards = collect_shards(&args.unet)?;
    let mut tc = TrainConfig::default();
    let lora_mode = args.lora_path.is_some();
    if lora_mode {
        tc.training_method = TrainingMethod::Lora;
        tc.lora_rank = args.lora_rank as u64;
        tc.lora_alpha = args.lora_alpha;
    } else {
        // Allocate as LoRA-disabled base; SDXLModel::load with non-LoRA mode
        // does a full F32 parameter copy (~5 GB extra) which we don't need
        // for inference. The current ED-v2 SDXLModel doesn't have a
        // base-only construction path yet, so we reuse LoRA mode with a
        // tiny rank to avoid the F32 promotion cost — adapters at rank 1
        // are zero-init (B=0) and contribute exactly nothing to the
        // forward pass. NOTE: minor footprint cost (~few MB), correctness
        // is preserved (zero adapters = identity).
        tc.training_method = TrainingMethod::Lora;
        tc.lora_rank = 1;
        tc.lora_alpha = 1.0;
    }
    let mut model = SDXLModel::load(&shards, &tc, device.clone())?;
    if let Some(lp) = &args.lora_path {
        model.load_weights(lp.to_str().unwrap())?;
        log::info!("  Applied LoRA from {:?} (rank={}, alpha={})",
            lp, args.lora_rank, args.lora_alpha);
    }

    // 5. Denoise (Euler-A default per SDXL audit H4; DDIM is opt-in)
    let alpha_bar = compute_alpha_bar();
    let ts = timesteps(args.steps);
    let pred_kind = if args.v_prediction { Prediction::V } else { Prediction::Epsilon };

    use rand::Rng;
    // Single RNG drives both initial noise and Euler-A's per-step ancestral
    // noise. Seeded once so the full sample is reproducible.
    let mut rng = rand::rngs::StdRng::seed_from_u64(args.seed);

    fn sample_normal(rng: &mut rand::rngs::StdRng, n: usize) -> Vec<f32> {
        let mut data = Vec::with_capacity(n);
        while data.len() < n {
            let u1 = rng.gen::<f32>().max(1e-6);
            let u2 = rng.gen::<f32>();
            let mag = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f32::consts::PI * u2;
            data.push(mag * theta.cos());
            if data.len() < n { data.push(mag * theta.sin()); }
        }
        data.truncate(n);
        data
    }

    log::info!("[4/4] Denoising {} prompt(s) × {} steps (sched={:?}, cfg={}, rescale={})...",
        conds.len(), args.steps, scheduler, args.cfg_scale, args.cfg_rescale);
    let pad_width = std::cmp::max(3, conds.len().to_string().len());
    let mut latents: Vec<Tensor> = Vec::with_capacity(conds.len());

    for (idx, (ctx_cond, y_cond)) in conds.iter().enumerate() {
        log::info!("  [{}/{}] denoising prompt...", idx + 1, conds.len());
        // Reset RNG for each prompt so seed semantics match per-invocation.
        rng = rand::rngs::StdRng::seed_from_u64(args.seed);

        let n_lat = 1 * 4 * h_lat * w_lat;
        let mut latent = Tensor::from_vec(
            sample_normal(&mut rng, n_lat),
            Shape::from_dims(&[1, 4, h_lat, w_lat]),
            device.clone(),
        )?.to_dtype(DType::BF16)?;

        // Euler-A: scale x0 by σ_init = sqrt((1-ᾱ_max)/ᾱ_max). Matches diffusers
        // `EulerAncestralDiscreteScheduler.init_noise_sigma`.
        if matches!(scheduler, SchedulerKind::EulerA) {
            let t0 = ts[0];
            let ab0 = alpha_bar[t0];
            let sigma_init = ((1.0 - ab0) / ab0).sqrt();
            latent = latent.mul_scalar(sigma_init)?;
        }

        for (i, &t) in ts.iter().enumerate() {
            let t_tensor = Tensor::from_vec(vec![t as f32], Shape::from_dims(&[1]), device.clone())?;

            // For Euler-A we hand the model the σ-scaled latent rescaled to
            // unit-variance noisy form (model expects noisy = sqrt(ᾱ)·x0 + sqrt(1-ᾱ)·ε).
            let ab_t = alpha_bar[t];
            let model_input = match scheduler {
                SchedulerKind::Ddim => latent.clone(),
                SchedulerKind::EulerA => {
                    let scale = (1.0 / (1.0 + (1.0 - ab_t) / ab_t)).sqrt(); // 1 / sqrt(1+σ²) = sqrt(ᾱ)
                    latent.mul_scalar(scale)?
                }
            };

            let pred_cond = <SDXLModel as TrainableModel>::forward(
                &mut model, &model_input, &t_tensor,
                std::slice::from_ref(ctx_cond), Some(y_cond),
            )?;
            let pred = if let Some((ref ctx_u, ref y_u)) = uncond_pair {
                let pred_uncond = <SDXLModel as TrainableModel>::forward(
                    &mut model, &model_input, &t_tensor,
                    std::slice::from_ref(ctx_u), Some(y_u),
                )?;
                // CFG: pred = uncond + cfg_scale * (cond - uncond)
                let pred_cfg = pred_uncond.add(
                    &pred_cond.sub(&pred_uncond)?.mul_scalar(args.cfg_scale)?)?;
                // CFG-rescale (Lin et al. 2023 §3.4): rescale = std(cond)/std(cfg).
                // pred = mix(pred_cfg, pred_cfg * rescale, cfg_rescale).
                // Default 0.7 per OT __sample_base. Skip for cfg_rescale ≤ 0.
                if args.cfg_rescale > 0.0 {
                    let cond_f32 = pred_cond.to_dtype(DType::F32)?;
                    let cfg_f32 = pred_cfg.to_dtype(DType::F32)?;
                    let std_cond = cond_f32.square()?.mean()?.to_vec()?[0].sqrt();
                    let std_cfg = cfg_f32.square()?.mean()?.to_vec()?[0].sqrt().max(1e-8);
                    let rescale = std_cond / std_cfg;
                    let mix = args.cfg_rescale * rescale + (1.0 - args.cfg_rescale);
                    pred_cfg.mul_scalar(mix)?
                } else {
                    pred_cfg
                }
            } else { pred_cond };

            let next_t = ts.get(i + 1).copied();
            let ab_prev = match next_t {
                Some(t_n) => alpha_bar[t_n],
                None => 1.0, // synthetic clean step at t=-1
            };

            latent = match scheduler {
                SchedulerKind::Ddim => {
                    ddim_step(&latent, &pred, ab_t, ab_prev, pred_kind)?
                }
                SchedulerKind::EulerA => {
                    let n = 1 * 4 * h_lat * w_lat;
                    let noise = Tensor::from_vec(
                        sample_normal(&mut rng, n),
                        Shape::from_dims(&[1, 4, h_lat, w_lat]),
                        device.clone(),
                    )?.to_dtype(DType::BF16)?;
                    euler_a_step(&latent, &pred, ab_t, ab_prev, &noise, pred_kind)?
                }
            };

            if i % 5 == 0 || i == ts.len() - 1 {
                log::info!("    prompt {}/{} step {}/{} t={} ᾱ={:.4}",
                    idx + 1, conds.len(), i + 1, ts.len(), t, ab_t);
            }
        }
        latents.push(latent);
    }

    // For Euler-A the loop output is x0 (final ab_prev=1.0 returns clean
    // latent directly). Pass through unchanged to VAE decode.

    // 6. VAE decode (single VAE load for all latents)
    log::info!("VAE decoding {} latent(s)...", latents.len());
    drop(model);
    let vae = SdxlVaeDecoder::from_safetensors(args.vae_ckpt.to_str().unwrap(), &device)?;
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
