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
    /// Single prompt. Mutually exclusive with `--prompts-file`.
    #[arg(long)] prompt: Option<String>,
    /// Newline-separated prompts file for batch sampling. Blank lines and
    /// `#`-prefixed comments are skipped. Requires `--output-dir`. T5 +
    /// CLIP load once for all prompts; DiT and VAE each load once total.
    #[arg(long)] prompts_file: Option<PathBuf>,
    #[arg(long, default_value = "")] negative: String,
    #[arg(long, default_value = "output.png")] output: PathBuf,
    /// Multi-prompt output directory. Required with `--prompts-file`.
    /// Files are written as `sample_001.png`, `sample_002.png`, ...
    #[arg(long)] output_dir: Option<PathBuf>,
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
    // CLIP-L safetensors ship as F32 from HF; the encoder calls
    // `layer_norm_bf16` directly which is BF16-strict. Cast at load time so
    // every weight (token_embedding, position_embedding, all linear weights,
    // layer_norms) is BF16. Otherwise embed lookup returns F32 and the very
    // first encoder LayerNorm errors with "expected BF16, got logical F32".
    let clip_weights: HashMap<String, Tensor> = clip_weights
        .into_iter()
        .map(|(k, t)| {
            let t = t.to_dtype(DType::BF16)?;
            Ok::<_, anyhow::Error>((k, t))
        })
        .collect::<anyhow::Result<_>>()?;
    let clip = ClipEncoder::new(clip_weights, ClipConfig::default(), device.clone());

    let mut encode_t5 = |text: &str| -> anyhow::Result<Tensor> {
        let e = t5_tok.encode(text, true).map_err(|e| anyhow::anyhow!("{e}"))?;
        let mut ids: Vec<i32> = e.get_ids().iter().map(|&x| x as i32).collect();
        // Truncate to T5_MAX_LEN, then pad with 0 to exactly T5_MAX_LEN.
        // `build_txt_ids` always produces T5_MAX_LEN RoPE positions; the T5
        // embedding must match that length or the RoPE table is misaligned.
        // Mirrors inference-flame's `tokenize_t5` (flux1_infer.rs pad-to-512).
        if ids.len() > T5_MAX_LEN { ids.truncate(T5_MAX_LEN); }
        while ids.len() < T5_MAX_LEN { ids.push(0); }
        Ok(t5.encode(&ids)?)
    };
    let encode_clip = |text: &str| -> anyhow::Result<Tensor> {
        let e = clip_tok.encode(text, true).map_err(|e| anyhow::anyhow!("{e}"))?;
        let ids: Vec<i32> = e.get_ids().iter().map(|&x| x as i32).collect();
        let (_h, pool) = clip.encode(&ids)?;
        Ok(pool)
    };

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

    // T5 / CLIP encoders return F32 by default; flux's DiT layer_norm is
    // strict-BF16. Cast at the boundary before passing into the model.
    // Encode all prompts while T5 + CLIP are resident, then drop them.
    let conds: Vec<(Tensor, Tensor)> = prompts.iter().enumerate()
        .map(|(i, p)| {
            let t5 = encode_t5(p)?.to_dtype(DType::BF16)?;
            let cl = encode_clip(p)?.to_dtype(DType::BF16)?;
            log::info!("  prompt {}/{}: t5={:?} clip={:?}",
                i + 1, prompts.len(), t5.shape().dims(), cl.shape().dims());
            Ok((t5, cl))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    // CFG (uncond) embeds only computed when explicitly enabled (cfg > 1.0)
    // — FLUX is guidance-distilled and CFG is structurally absent. H8.
    let cfg_enabled = args.cfg > 1.0 + f32::EPSILON;
    let (uncond_t5, uncond_clip) = if cfg_enabled {
        (
            Some(encode_t5(&args.negative)?.to_dtype(DType::BF16)?),
            Some(encode_clip(&args.negative)?.to_dtype(DType::BF16)?),
        )
    } else {
        (None, None)
    };
    drop(t5);
    drop(clip);

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
        // Honor user flags so the speckle-bug bisect can disable adapter contribution
        // by passing --lora-alpha 0.0 (delta = α/r * lora_b @ lora_a → zero when α=0).
        tc.training_method = TrainingMethod::Lora;
        tc.lora_rank = args.lora_rank as u64;
        tc.lora_alpha = args.lora_alpha;
    }
    let mut model = if args.offload {
        FluxModel::load_offload(&shards[0], &tc, device.clone())?
    } else {
        FluxModel::load(&shards[0], &tc, device.clone())?
    };
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

        // ── BlockOffloader byte-fidelity probe (FLUX speckle bisect 2026-05-07) ──
        // Compare the same weight tensor as delivered three ways:
        //   1. ensure_block(0) — first call
        //   2. ensure_block(0) — second call after evict (slot-cycle through)
        //   3. direct safetensors load via serialization::load_file_filtered
        // If 1, 2, 3 all match → BlockOffloader is correct, bug is elsewhere.
        // If any differ → BlockOffloader is delivering wrong bytes, also breaks
        // training, must be fixed first.
        if let Some(off_arc) = model.offloader.as_ref() {
            let target_key = "double_blocks.0.img_attn.qkv.weight";
            let stats = |t: &flame_core::Tensor, label: &str| -> anyhow::Result<()> {
                let v = t.to_dtype(flame_core::DType::F32)?.to_vec()?;
                let n = v.len() as f64;
                let mean: f64 = v.iter().map(|x| *x as f64).sum::<f64>() / n;
                let sq: f64 = v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / n;
                let var = sq - mean * mean;
                let mn = v.iter().cloned().fold(f32::INFINITY, f32::min);
                let mx = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let first10: Vec<f32> = v.iter().take(10).copied().collect();
                log::info!("[probe {label}] shape={:?} mean={:.6} std={:.6} min={:.4} max={:.4} first10={:?}",
                    t.shape().dims(), mean, var.sqrt(), mn, mx, first10);
                Ok(())
            };
            // (1)
            {
                let mut g = off_arc.lock().unwrap();
                let arc1 = g.ensure_block(0)
                    .map_err(|e| anyhow::anyhow!("probe ensure_block(0)#1: {e}"))?;
                let t1 = arc1.get(target_key)
                    .ok_or_else(|| anyhow::anyhow!("probe: missing {target_key}"))?;
                stats(t1, "1: ensure_block(0)")?;
                g.evict_block();
            }
            // (2)
            {
                let mut g = off_arc.lock().unwrap();
                let arc2 = g.ensure_block(0)
                    .map_err(|e| anyhow::anyhow!("probe ensure_block(0)#2: {e}"))?;
                let t2 = arc2.get(target_key)
                    .ok_or_else(|| anyhow::anyhow!("probe: missing {target_key} on second call"))?;
                stats(t2, "2: ensure_block(0) again")?;
                g.evict_block();
            }
            // (3) direct load — find the shard containing this key
            let prefix = "double_blocks.0.";
            for shard in &shards {
                let part = flame_core::serialization::load_file_filtered(
                    shard, &device, |k| k.starts_with(prefix),
                ).map_err(|e| anyhow::anyhow!("probe direct load: {e}"))?;
                if let Some(t3) = part.get(target_key) {
                    let t3_bf16 = t3.to_dtype(flame_core::DType::BF16)?;
                    stats(&t3_bf16, "3: serialization::load_file_filtered")?;
                    break;
                }
            }
        }
    }

    // ── 3. Denoise — once per prompt, all latents collected ──
    log::info!("[3/4] Denoising {} prompt(s) × {} steps...", conds.len(), args.steps);
    let sigmas = flux_sampler::schedule(args.steps, args.size, args.size);

    let img_ids = flux_sampler::build_img_ids(h_tok, w_tok, device.clone())?
        .to_dtype(DType::BF16)?;
    let txt_ids = flux_sampler::build_txt_ids(T5_MAX_LEN, device.clone())?
        .to_dtype(DType::BF16)?;

    let pad_width = std::cmp::max(3, conds.len().to_string().len());
    let mut latents: Vec<Tensor> = Vec::with_capacity(conds.len());

    for (idx, (cond_t5, cond_clip)) in conds.iter().enumerate() {
        log::info!("  [{}/{}] denoising prompt...", idx + 1, conds.len());
        // Per-prompt seed offset → each prompt gets a different noise
        // initial. Same `--seed` is deterministic across runs but yields
        // a diverse batch (idx 0 uses seed, idx 1 uses seed+1, ...).
        flame_core::rng::set_seed(args.seed.wrapping_add(idx as u64))
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
                &mut model, &latent, &t_tensor, &ctx_cond, Some(cond_clip),
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
                log::info!("    prompt {}/{} step {}/{} sigma={:.4}",
                    idx + 1, conds.len(), step + 1, args.steps, s);
            }
        }
        latents.push(latent);
    }

    // Drop the model and offloader before VAE decode: 57 Flux blocks in
    // staging RAM hold 1.3 GB of pinned memory and the offloader's two GPU
    // slots hold another ~1.3 GB. Freeing them first avoids an OOM during
    // the cuDNN conv2d workspace allocation in the VAE decoder.
    drop(model);
    flame_core::cuda_alloc_pool::clear_pool_cache();
    flame_core::trim_cuda_mempool(0);

    // ── 4. Unpack + un-scale + decode (single VAE load for all latents) ──
    log::info!("[4/4] VAE decode {} latent(s)...", latents.len());
    let dec = FluxVaeDecoder::from_safetensors(
        args.vae_path.to_str().unwrap(),
        LATENT_CHANNELS,
        /*scaling_factor*/ 1.0, // already un-scaled below per-latent
        /*shift_factor*/ 0.0,
        &device,
    )?;

    for (idx, latent) in latents.iter().enumerate() {
        // ── speckle-bug probe: log stats at every stage ──
        let lat_f32 = latent.to_dtype(DType::F32)?.to_vec()?;
        let n = lat_f32.len() as f32;
        let mean: f32 = lat_f32.iter().sum::<f32>() / n;
        let var: f32 = lat_f32.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
        let mn = lat_f32.iter().cloned().fold(f32::INFINITY, f32::min);
        let mx = lat_f32.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        log::info!("  packed latent stats: mean={:.4} std={:.4} min={:.4} max={:.4}",
            mean, var.sqrt(), mn, mx);

        let unpacked = flux_sampler::unpack_latents(latent, h_tok, w_tok)?;
        // Audit fix FLUX_VERIFY §H2 / SKEPTIC §H2: BFL decode is `raw = scaled /
        // SCALE + SHIFT` (`autoencoder.py:308-315` and `FluxSampler.py:159`).
        let latent_for_vae = unpacked.mul_scalar(1.0 / SCALE)?.add_scalar(SHIFT)?;

        let lat2_f32 = latent_for_vae.to_dtype(DType::F32)?.to_vec()?;
        let mean2: f32 = lat2_f32.iter().sum::<f32>() / lat2_f32.len() as f32;
        let var2: f32 = lat2_f32.iter().map(|v| (v - mean2).powi(2)).sum::<f32>() / lat2_f32.len() as f32;
        let mn2 = lat2_f32.iter().cloned().fold(f32::INFINITY, f32::min);
        let mx2 = lat2_f32.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        log::info!("  unpacked+unscale stats: mean={:.4} std={:.4} min={:.4} max={:.4}",
            mean2, var2.sqrt(), mn2, mx2);

        let img = dec.decode(&latent_for_vae)?;

        let img_f32_full = img.to_dtype(DType::F32)?.to_vec()?;
        let m3: f32 = img_f32_full.iter().sum::<f32>() / img_f32_full.len() as f32;
        let v3: f32 = img_f32_full.iter().map(|v| (v - m3).powi(2)).sum::<f32>() / img_f32_full.len() as f32;
        let mn3 = img_f32_full.iter().cloned().fold(f32::INFINITY, f32::min);
        let mx3 = img_f32_full.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        log::info!("  decoded RGB stats: mean={:.4} std={:.4} min={:.4} max={:.4}",
            m3, v3.sqrt(), mn3, mx3);

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
