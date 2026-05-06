//! sample_flux — text → FLUX.1 (Dev/Schnell) image generation. Optional `--lora-path`.
//!
//! Pipeline mirrors `sample_ernie`/`sample_klein`:
//!   1. Tokenize prompt with T5-XXL + CLIP-L tokenizers, encode separately.
//!   2. Load Flux transformer (auto-detect Dev vs Schnell from `guidance_in.in_layer.weight`).
//!   3. Build noise [1, N_img, 64] (already in packed Flux DiT input space).
//!   4. Euler denoise per `flux_sampler::schedule(num_steps, w, h)` (sequence-length-shifted mu).
//!   5. unpack_latents → un-scale with `flux_vae::SHIFT/SCALE` → LdmVAEDecoder → RGB → PNG.
//!
//! Variant: `--variant dev|schnell`. Dev uses guidance_value=3.5 (default Flux config),
//! Schnell uses 1.0 and skips guidance injection (model has no guidance_in).

use clap::{Parser, ValueEnum};
use flame_core::{DType, Shape, Tensor};
use eridiffusion_core::config::{TrainConfig, TrainingMethod};
use eridiffusion_core::encoders::{
    clip_l::{ClipConfig, ClipEncoder},
    flux_vae::{FluxVaeDecoder, LATENT_CHANNELS, SCALE, SHIFT},
    t5_xxl::T5Encoder,
};
use eridiffusion_core::models::{flux::FluxModel, TrainableModel};
use eridiffusion_core::sampler::flux_sampler;
use std::collections::HashMap;
use std::path::PathBuf;

const T5_MAX_LEN: usize = 512;

#[derive(Copy, Clone, ValueEnum, Debug)]
enum Variant { Dev, Schnell }

#[derive(Parser)]
struct Args {
    #[arg(long)] prompt: String,
    #[arg(long, default_value = "")] negative: String,
    #[arg(long, default_value = "output.png")] output: PathBuf,
    /// Flux transformer (single .safetensors or directory).
    #[arg(long)] transformer: PathBuf,
    #[arg(long)] vae_path: PathBuf,
    #[arg(long)] t5_ckpt: PathBuf,
    #[arg(long)] clip_ckpt: PathBuf,
    #[arg(long)] t5_tokenizer: PathBuf,
    #[arg(long)] clip_tokenizer: PathBuf,
    #[arg(long, value_enum, default_value_t = Variant::Dev)] variant: Variant,
    #[arg(long, default_value = "1024")] size: usize,
    #[arg(long, default_value = "20")] steps: usize,
    /// External classifier-free guidance. **Disabled by default** — FLUX.1 Dev/Schnell
    /// are guidance-distilled (single forward, guidance fed via model input).
    /// Audit fix FLUX_VERIFY §H3 / §H8 / SKEPTIC §H8: pre-fix the sampler ran
    /// 2 forwards/step and combined `pred_uncond + cfg*(pred_cond - pred_uncond)`
    /// — over-amplifies the prediction (uncond branch was never trained as a
    /// separate distribution). Only honoured when `> 1.0`.
    #[arg(long, default_value = "1.0")] cfg: f32,
    /// Internal Flux Dev guidance value (passed to the DiT via `guidance_in`
    /// MLP). 3.5 is the BFL inference default; Schnell ignores this.
    #[arg(long, default_value = "3.5")] flux_guidance: f32,
    #[arg(long, default_value = "42")] seed: u64,
    #[arg(long)] lora_path: Option<PathBuf>,
    #[arg(long, default_value = "16")] lora_rank: usize,
    /// Convention: alpha = rank (effective scale 1.0). FLUX_VERIFY §H12.
    #[arg(long, default_value = "16.0")] lora_alpha: f64,
    #[arg(long)] offload: bool,
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

fn load_clip_weights(path: &std::path::Path, device: &std::sync::Arc<flame_core::CudaDevice>)
    -> flame_core::Result<HashMap<String, Tensor>>
{
    if path.is_file() {
        return flame_core::serialization::load_file(path, device);
    }
    let mut all = HashMap::new();
    for entry in std::fs::read_dir(path)
        .map_err(|e| flame_core::Error::Io(format!("read_dir: {e}")))? {
        let p = entry.map_err(|e| flame_core::Error::Io(format!("entry: {e}")))?.path();
        if p.extension().and_then(|s| s.to_str()) == Some("safetensors") {
            let part = flame_core::serialization::load_file(&p, device)?;
            all.extend(part);
        }
    }
    Ok(all)
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let args = Args::parse();
    let device = flame_core::global_cuda_device();
    let _no_grad = flame_core::autograd::AutogradContext::no_grad();
    flame_core::config::set_default_dtype(DType::BF16);

    // Latent grid: 8× VAE → 2× patch → /16. Packed N = (size/16)².
    let h_tok = args.size / 16;
    let w_tok = args.size / 16;
    let n_img = h_tok * w_tok;
    log::info!("size={}² → packed n_img={} ({}x{})", args.size, n_img, h_tok, w_tok);

    // ── 1. Encode text ──
    log::info!("[1/4] T5 + CLIP encode...");
    let t5_tok = tokenizers::Tokenizer::from_file(&args.t5_tokenizer)
        .map_err(|e| anyhow::anyhow!("T5 tokenizer: {e}"))?;
    let clip_tok = tokenizers::Tokenizer::from_file(&args.clip_tokenizer)
        .map_err(|e| anyhow::anyhow!("CLIP tokenizer: {e}"))?;

    let mut t5 = T5Encoder::load(args.t5_ckpt.to_str().unwrap(), &device)?;
    let clip_weights = load_clip_weights(&args.clip_ckpt, &device)?;
    let clip = ClipEncoder::new(clip_weights, ClipConfig::default(), device.clone());

    let mut encode_t5 = |text: &str| -> anyhow::Result<Tensor> {
        let e = t5_tok.encode(text, true).map_err(|e| anyhow::anyhow!("{e}"))?;
        let mut ids: Vec<i32> = e.get_ids().iter().map(|&x| x as i32).collect();
        if ids.len() > T5_MAX_LEN { ids.truncate(T5_MAX_LEN); }
        Ok(t5.encode(&ids)?)
    };
    let encode_clip = |text: &str| -> anyhow::Result<Tensor> {
        let e = clip_tok.encode(text, true).map_err(|e| anyhow::anyhow!("{e}"))?;
        let ids: Vec<i32> = e.get_ids().iter().map(|&x| x as i32).collect();
        let (_h, pool) = clip.encode(&ids)?;
        Ok(pool)
    };

    let cond_t5 = encode_t5(&args.prompt)?;
    let cond_clip = encode_clip(&args.prompt)?;
    // CFG (uncond) embeds only computed when explicitly enabled (cfg > 1.0)
    // — FLUX is guidance-distilled and CFG is structurally absent. H8.
    let cfg_enabled = args.cfg > 1.0 + f32::EPSILON;
    let (uncond_t5, uncond_clip) = if cfg_enabled {
        (Some(encode_t5(&args.negative)?), Some(encode_clip(&args.negative)?))
    } else {
        (None, None)
    };
    drop(t5);

    log::info!("  cond t5={:?} clip={:?}", cond_t5.shape().dims(), cond_clip.shape().dims());

    // ── 2. Load DiT ──
    log::info!("[2/4] Loading FLUX transformer...");
    let shards = collect_shards(&args.transformer)?;
    let mut tc = TrainConfig::default();
    if args.lora_path.is_some() {
        tc.training_method = TrainingMethod::Lora;
        tc.lora_rank = args.lora_rank as u64;
        tc.lora_alpha = args.lora_alpha;
    } else {
        // Use LoRA mode anyway so we don't allocate F32 FFT params (mirrors sample_ernie).
        tc.training_method = TrainingMethod::Lora;
        tc.lora_rank = 4;
        tc.lora_alpha = 1.0;
    }
    let mut model = FluxModel::load(&shards[0], &tc, device.clone())?;
    match args.variant {
        Variant::Dev => {
            model.guidance_value = args.flux_guidance;
        }
        Variant::Schnell => {
            model.has_guidance = false;
            model.guidance_value = 1.0;
        }
    }
    if let Some(lp) = &args.lora_path {
        model.load_weights(lp.to_str().unwrap())?;
        log::info!("  Applied LoRA from {:?} (rank={}, alpha={})", lp, args.lora_rank, args.lora_alpha);
    }
    if args.offload {
        model.enable_offload(shards.clone())?;
        log::info!("  block-offload enabled");
    }

    // ── 3. Denoise ──
    log::info!("[3/4] Denoising {} steps...", args.steps);
    let sigmas = flux_sampler::schedule(args.steps, args.size, args.size);

    let img_ids = flux_sampler::build_img_ids(h_tok, w_tok, device.clone())?
        .to_dtype(DType::BF16)?;
    let txt_ids = flux_sampler::build_txt_ids(T5_MAX_LEN, device.clone())?
        .to_dtype(DType::BF16)?;

    // Audit fix FLUX_VERIFY §M3 / SKEPTIC §H11: previously the StdRng was
    // built and immediately discarded (`let _ = ...`). Wire `args.seed` into
    // flame_core's global RNG so `Tensor::randn` is deterministic.
    flame_core::rng::set_seed(args.seed)
        .map_err(|e| anyhow::anyhow!("flame_core set_seed: {e}"))?;
    let mut latent = Tensor::randn(Shape::from_dims(&[1, n_img, 64]), 0.0, 1.0, device.clone())?
        .to_dtype(DType::BF16)?;

    for step in 0..args.steps {
        let s = sigmas[step];
        let s_next = sigmas[step + 1];
        // Audit fix FLUX_VERIFY §H1 / SKEPTIC §H1: pass sigma directly as the
        // model timestep (already in `[0, 1]`). `flux.rs::timestep_embedding`
        // multiplies by 1000 exactly once. Pre-fix multiplied here AND inside
        // the embedder → `s * 1_000_000` in the sinusoid arg.
        let t_tensor = Tensor::from_vec(vec![s], Shape::from_dims(&[1]), device.clone())?;

        // Audit fix FLUX_VERIFY §H3 / §H8 / SKEPTIC §H8: single forward.
        // FLUX is guidance-distilled — the DiT was never trained as a separate
        // unconditional distribution. CFG still honoured when `args.cfg > 1.0`
        // (default 1.0 = disabled), but using it on Dev/Schnell over-amplifies
        // the prediction without improving sample quality.
        let ctx_cond = vec![cond_t5.clone(), img_ids.clone(), txt_ids.clone()];
        let pred_cond = <FluxModel as TrainableModel>::forward(
            &mut model, &latent, &t_tensor, &ctx_cond, Some(&cond_clip),
        )?;
        let pred = if cfg_enabled {
            let ut5 = uncond_t5.as_ref().unwrap();
            let uclip = uncond_clip.as_ref().unwrap();
            let ctx_uncond = vec![ut5.clone(), img_ids.clone(), txt_ids.clone()];
            let pred_uncond = <FluxModel as TrainableModel>::forward(
                &mut model, &latent, &t_tensor, &ctx_uncond, Some(uclip),
            )?;
            pred_uncond.add(&pred_cond.sub(&pred_uncond)?.mul_scalar(args.cfg)?)?
        } else {
            pred_cond
        };

        latent = flux_sampler::euler_step(&latent, &pred, s, s_next)?;
        if step % 5 == 0 || step == args.steps - 1 {
            log::info!("  step {}/{} sigma={:.4}", step + 1, args.steps, s);
        }
    }

    // ── 4. Unpack + un-scale + decode ──
    log::info!("[4/4] VAE decode...");
    let unpacked = flux_sampler::unpack_latents(&latent, h_tok, w_tok)?;
    // Audit fix FLUX_VERIFY §H2 / SKEPTIC §H2: BFL decode is `raw = scaled /
    // SCALE + SHIFT` (`autoencoder.py:308-315` and `FluxSampler.py:159`).
    // Pre-fix the sampler did `scaled * SCALE + SHIFT` (the *encode*
    // direction); combined with prepare_flux's inverted encode, it round-
    // tripped visually but kept the DiT in the wrong latent magnitude band.
    let latent_for_vae = unpacked.mul_scalar(1.0 / SCALE)?.add_scalar(SHIFT)?;

    let dec = FluxVaeDecoder::from_safetensors(
        args.vae_path.to_str().unwrap(),
        LATENT_CHANNELS,
        /*scaling_factor*/ 1.0, // already un-scaled above
        /*shift_factor*/ 0.0,
        &device,
    )?;
    let img = dec.decode(&latent_for_vae)?;

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
