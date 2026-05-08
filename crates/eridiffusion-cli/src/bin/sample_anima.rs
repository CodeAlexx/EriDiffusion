//! sample_anima — text → Anima image generation, optionally with a trained LoRA.
//!
//! Reference: kohya `library/anima_train_utils.do_sample` + `_sample_image_inference`.
//! Pipeline:
//!   1. Tokenize prompt with Qwen3 + T5 tokenizers, encode prompt with Qwen3-0.6B
//!      (last_hidden_state, mask zero-padded positions per
//!      `strategy_anima.AnimaTextEncodingStrategy.encode_tokens`).
//!   2. Build sigma schedule with `flow_shift` (kohya default 3.0 for inference).
//!   3. Init latent ~ N(0, I), shape [1, 16, H/8, W/8] (image case T=1).
//!   4. For each step: pred = forward(x, sigma, cond, mask, t5_ids, t5_mask);
//!      apply CFG; x = x + (sigma_next - sigma) * pred.
//!   5. VAE decode (decoder side reuses the Qwen-image VAE — TODO when an
//!      `LdmVAEDecoder` for the qwen-image VAE lands; currently uses the
//!      `ZImageVAEDecoder` since both share the WAN-VAE family layout).
//!
//! ## STATUS
//! Compiles + scaffolds the inference pipeline. The `model.forward` call goes
//! to `AnimaModel::forward` which currently returns NotImplemented. End-to-end
//! sampling will work once the model port lands.

use clap::Parser;
use std::path::PathBuf;
use flame_core::{DType, Shape, Tensor};
use eridiffusion_core::config::{TrainConfig, TrainingMethod};
use eridiffusion_core::encoders::{
    qwen3::Qwen3Encoder,
    wan21_decoder::Wan21VaeDecoder,
};
use eridiffusion_core::models::{AnimaModel, TrainableModel, anima as anima_mod};
use eridiffusion_core::sampler::anima_sampler;
use std::collections::HashMap;

const QWEN3_PAD_ID: i32 = 151643;
const T5_PAD_ID: i32 = 0;
const QWEN3_MAX_LEN: usize = 512;
const T5_MAX_LEN: usize = 512;

#[derive(Parser)]
struct Args {
    /// Single prompt. Mutually exclusive with `--prompts-file`.
    #[arg(long)] prompt: Option<String>,
    /// Newline-separated prompts file for batch sampling. Blank lines and
    /// `#`-prefixed comments are skipped. Requires `--output-dir`. Encoders
    /// load once for all prompts; DiT and VAE each load once total.
    #[arg(long)] prompts_file: Option<PathBuf>,
    #[arg(long, default_value = "")] negative_prompt: String,
    #[arg(long, default_value = "output.png")] output: PathBuf,
    /// Multi-prompt output directory. Required with `--prompts-file`.
    /// Files are written as `sample_001.png`, `sample_002.png`, ...
    #[arg(long)] output_dir: Option<PathBuf>,
    /// Single safetensors (e.g. `anima-preview.safetensors`).
    #[arg(long)] dit_path: PathBuf,
    #[arg(long)] vae_path: PathBuf,
    #[arg(long)] qwen3: PathBuf,
    #[arg(long)] tokenizer_path: PathBuf,
    #[arg(long)] t5_tokenizer_path: PathBuf,
    #[arg(long, default_value = "1024")] size: usize,
    #[arg(long, default_value = "20")] steps: usize,
    #[arg(long, default_value = "5.0")] guidance: f32,
    /// kohya `do_sample` default flow_shift = 3.0.
    #[arg(long, default_value = "3.0")] flow_shift: f32,
    #[arg(long, default_value = "42")] seed: u64,
    /// Optional safetensors of a trained LoRA (ai-toolkit/PEFT format).
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

    let h_lat = args.size / 8;
    let w_lat = args.size / 8;
    log::info!("Size {}x{} → latent {}x{}", args.size, args.size, h_lat, w_lat);

    // ── 1. Text encode ────────────────────────────────────────────────────
    log::info!("[1/4] Loading tokenizers + Qwen3-0.6B encoder...");
    let qwen_tok = tokenizers::Tokenizer::from_file(&args.tokenizer_path)
        .map_err(|e| anyhow::anyhow!("qwen3 tokenizer: {e}"))?;
    let t5_tok = tokenizers::Tokenizer::from_file(&args.t5_tokenizer_path)
        .map_err(|e| anyhow::anyhow!("t5 tokenizer: {e}"))?;

    let qwen_weights = load_qwen3_weights(&args.qwen3, &device)?;
    let mut qcfg = Qwen3Encoder::config_from_weights(&qwen_weights)?;
    qcfg.extract_layers = vec![qcfg.num_layers - 1];
    let qwen3 = Qwen3Encoder::new(qwen_weights, qcfg, device.clone());

    let encode_pair = |text: &str| -> anyhow::Result<(Tensor, Tensor, Tensor, Tensor)> {
        let qwen_enc = qwen_tok.encode(text, true).map_err(|e| anyhow::anyhow!("{e}"))?;
        let mut qids: Vec<i32> = qwen_enc.get_ids().iter().map(|&i| i as i32).collect();
        let qvalid = qids.len().min(QWEN3_MAX_LEN);
        if qids.len() > QWEN3_MAX_LEN { qids.truncate(QWEN3_MAX_LEN); }
        qids.resize(QWEN3_MAX_LEN, QWEN3_PAD_ID);
        let q_hidden = qwen3.encode(&qids)?; // [1, QWEN3_MAX_LEN, 1024]
        let mut qmask = vec![0.0f32; QWEN3_MAX_LEN];
        for slot in qmask.iter_mut().take(qvalid) { *slot = 1.0; }
        let qmask_t = Tensor::from_vec(qmask, Shape::from_dims(&[1, QWEN3_MAX_LEN]), device.clone())?;

        let t5_enc = t5_tok.encode(text, true).map_err(|e| anyhow::anyhow!("{e}"))?;
        let mut t5ids: Vec<i32> = t5_enc.get_ids().iter().map(|&i| i as i32).collect();
        let t5valid = t5ids.len().min(T5_MAX_LEN);
        if t5ids.len() > T5_MAX_LEN { t5ids.truncate(T5_MAX_LEN); }
        t5ids.resize(T5_MAX_LEN, T5_PAD_ID);
        let t5_f32: Vec<f32> = t5ids.iter().map(|&i| i as f32).collect();
        let t5_t = Tensor::from_vec(t5_f32, Shape::from_dims(&[1, T5_MAX_LEN]), device.clone())?;
        let mut t5mask = vec![0.0f32; T5_MAX_LEN];
        for slot in t5mask.iter_mut().take(t5valid) { *slot = 1.0; }
        let t5mask_t = Tensor::from_vec(t5mask, Shape::from_dims(&[1, T5_MAX_LEN]), device.clone())?;

        Ok((q_hidden.to_dtype(DType::BF16)?, qmask_t, t5_t, t5mask_t))
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

    log::info!("[2/4] Encoding {} prompt(s) + uncond...", prompts.len());
    let cond_quads: Vec<(Tensor, Tensor, Tensor, Tensor)> = prompts.iter().enumerate()
        .map(|(i, p)| {
            let q = encode_pair(p)?;
            log::info!("  prompt {}/{}: cap={:?}", i + 1, prompts.len(), q.0.shape().dims());
            Ok(q)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let (cap_uncond, mask_uncond, t5_uncond, t5mask_uncond) = encode_pair(&args.negative_prompt)?;
    drop(qwen3);
    flame_core::cuda_alloc_pool::clear_pool_cache();
    flame_core::trim_cuda_mempool(0);

    // ── 2. Load DiT (with optional LoRA overlay) ─────────────────────────
    log::info!("[3/4] Loading Anima DiT...");
    let mut tc = TrainConfig::default();
    tc.training_method = TrainingMethod::Lora;
    tc.lora_rank = args.lora_rank as u64;
    tc.lora_alpha = args.lora_alpha;
    let mut model = AnimaModel::load(&args.dit_path, &tc, device.clone())?;
    if let Some(lp) = &args.lora_path {
        model.load_weights(lp.to_str().unwrap())?;
        log::info!("  Applied LoRA from {} (rank={}, alpha={})",
            lp.display(), args.lora_rank, args.lora_alpha);
    }

    // ── 3. Denoise ─ once per prompt, all latents collected ─
    log::info!("[4/4] Denoising {} prompt(s) × {} steps (flow_shift={:.2}, cfg={:.2})...",
        cond_quads.len(), args.steps, args.flow_shift, args.guidance);
    let sigmas = anima_sampler::schedule(args.steps, args.flow_shift);
    let pad_width = std::cmp::max(3, cond_quads.len().to_string().len());
    let mut latents: Vec<Tensor> = Vec::with_capacity(cond_quads.len());

    for (idx, (cap_cond, mask_cond, t5_cond, t5mask_cond)) in cond_quads.iter().enumerate() {
        log::info!("  [{}/{}] denoising prompt...", idx + 1, cond_quads.len());
        let mut latent = {
            use rand::SeedableRng;
            // Per-prompt seed offset → diverse compositions across the batch.
            let _rng = rand::rngs::StdRng::seed_from_u64(args.seed.wrapping_add(idx as u64));
            Tensor::randn(
                Shape::from_dims(&[1, anima_mod::IN_CHANNELS, h_lat, w_lat]),
                0.0, 1.0, device.clone(),
            )?.to_dtype(DType::BF16)?
        };

        for step in 0..args.steps {
            let sigma = sigmas[step];
            let sigma_next = sigmas[step + 1];
            let t = anima_sampler::sigma_to_timestep(sigma);
            let t_tensor = Tensor::from_vec(vec![t], Shape::from_dims(&[1]), device.clone())?;

            let cond_ctx = vec![
                cap_cond.clone(), mask_cond.clone(), t5_cond.clone(), t5mask_cond.clone(),
            ];
            let unc_ctx = vec![
                cap_uncond.clone(), mask_uncond.clone(), t5_uncond.clone(), t5mask_uncond.clone(),
            ];
            let pred_cond = <AnimaModel as TrainableModel>::forward(&mut model, &latent, &t_tensor, &cond_ctx, None)?;
            let pred = if args.guidance > 1.0 {
                let pred_uncond = <AnimaModel as TrainableModel>::forward(&mut model, &latent, &t_tensor, &unc_ctx, None)?;
                pred_uncond.add(&pred_cond.sub(&pred_uncond)?.mul_scalar(args.guidance)?)?
            } else {
                pred_cond
            };

            latent = anima_sampler::euler_step(&latent, &pred, sigma, sigma_next)?;
            if step % 5 == 0 || step == args.steps - 1 {
                log::info!("    prompt {}/{} step {}/{} sigma={:.4}",
                    idx + 1, cond_quads.len(), step + 1, args.steps, sigma);
            }
        }
        latents.push(latent);
    }

    // ── 4. VAE decode (single VAE load for all latents) ─
    // Latent is per-channel normalized (Anima trainer side). Wan21VaeDecoder
    // unnormalizes (z * STD + MEAN) internally then runs the decoder, output
    // clamped to [-1, 1].
    log::info!("[VAE] decoding {} latent(s)...", latents.len());
    let vae = Wan21VaeDecoder::from_safetensors(&args.vae_path.to_string_lossy(), &device)?;

    for (idx, latent) in latents.iter().enumerate() {
        let img = vae.decode_image_normalized(latent)?;
        // CHW → HWC PNG.
        let pixels: Vec<f32> = img.to_dtype(DType::F32)?.to_vec()?;
        let dims = img.shape().dims();
        let (c, h, w) = if dims.len() == 4 { (dims[1], dims[2], dims[3]) } else { (3, dims[0], dims[1]) };
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
        log::info!("  [{}/{}] saved {}", idx + 1, latents.len(), out_path.display());
    }
    log::info!("Done — {} sample(s) saved", latents.len());
    Ok(())
}

fn load_qwen3_weights(path: &std::path::Path, device: &std::sync::Arc<flame_core::CudaDevice>)
    -> flame_core::Result<HashMap<String, Tensor>>
{
    if path.is_file() {
        return flame_core::serialization::load_file(path, device);
    }
    let mut all = HashMap::new();
    for entry in std::fs::read_dir(path).map_err(|e| flame_core::Error::Io(format!("read_dir {}: {e}", path.display())))? {
        let p = entry.map_err(|e| flame_core::Error::Io(format!("entry: {e}")))?.path();
        if p.extension().and_then(|e| e.to_str()) == Some("safetensors") {
            let part = flame_core::serialization::load_file(&p, device)?;
            all.extend(part);
        }
    }
    Ok(all)
}
