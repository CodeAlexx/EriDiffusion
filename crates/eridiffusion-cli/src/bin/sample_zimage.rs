//! sample_zimage — text → Z-Image generation. Optional LoRA via `--lora-path`.
//!
//! Mirrors sample_ernie's CLI shape but uses Qwen3 / Z-Image model / Z-Image VAE.

use clap::Parser;
use flame_core::DType;
use std::path::PathBuf;
use eridiffusion_core::encoders::qwen3::Qwen3Encoder;
use eridiffusion_core::models::zimage::ZImageModel;
use eridiffusion_core::sampler::zimage_sampler;

// See prepare_zimage.rs for the full justification of dropping the
// `<think>\n\n</think>\n\n` block. Train and sample MUST use identical templates;
// the prior asymmetry-class bug in ERNIE was lethal.
const ZIMAGE_TEMPLATE_PRE: &str = "<|im_start|>user\n";
const ZIMAGE_TEMPLATE_POST: &str = "<|im_end|>\n<|im_start|>assistant\n";
const PAD_TOKEN_ID: i32 = 151643;
const TXT_PAD_LEN: usize = 512;

#[derive(Parser)]
struct Args {
    /// Single prompt. Mutually exclusive with `--prompts-file`.
    #[arg(long)] prompt: Option<String>,
    /// Newline-separated prompts file for batch sampling. Blank lines and
    /// `#`-prefixed comments are skipped. Requires `--output-dir`. Encoder
    /// loads once for all prompts; DiT loads once and serves all denoises.
    #[arg(long)] prompts_file: Option<PathBuf>,
    #[arg(long, default_value = "output.png")] output: PathBuf,
    /// Multi-prompt output directory. Required with `--prompts-file`.
    /// Files are written as `sample_001.png`, `sample_002.png`, ...
    #[arg(long)] output_dir: Option<PathBuf>,
    /// Single-file Z-Image transformer safetensors (e.g. z_image_base_bf16.safetensors).
    #[arg(long)] model: PathBuf,
    #[arg(long)] vae_path: PathBuf,
    /// Path to Qwen3 weights (single file or directory of shards).
    #[arg(long)] qwen3: PathBuf,
    #[arg(long)] tokenizer_path: PathBuf,
    #[arg(long, default_value = "512")] size: usize,
    #[arg(long, default_value = "20")] steps: usize,
    #[arg(long, default_value = "4.0")] cfg: f32,
    #[arg(long, default_value = "3.0")] shift: f32,
    #[arg(long, default_value = "42")] seed: u64,
    /// Optional safetensors of a trained LoRA (matches train_zimage save format).
    #[arg(long)] lora_path: Option<PathBuf>,
    #[arg(long, default_value = "16")] lora_rank: usize,
    /// Match OT default = 1.0. See train_zimage.rs for justification.
    #[arg(long, default_value = "1.0")] lora_alpha: f32,
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let args = Args::parse();
    let _no_grad = flame_core::autograd::AutogradContext::no_grad();
    flame_core::config::set_default_dtype(DType::BF16);
    let device = flame_core::global_cuda_device();

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

    log::info!("[1/4] Loading Qwen3 + tokenizer...");
    let qwen_weights = load_qwen3_weights(&args.qwen3, &device)?;
    let mut qcfg = Qwen3Encoder::config_from_weights(&qwen_weights)?;
    // Qwen3-4B layer 34 = hidden_states[-2]. See prepare_zimage.rs.
    qcfg.extract_layers = vec![34];
    let qwen3 = Qwen3Encoder::new(qwen_weights, qcfg, device.clone());
    let tokenizer = tokenizers::Tokenizer::from_file(&args.tokenizer_path)
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;

    log::info!("[2/4] Encoding {} prompt(s) + uncond...", prompts.len());
    let cond_pairs: Vec<(flame_core::Tensor, flame_core::Tensor)> = prompts.iter().enumerate()
        .map(|(i, p)| {
            let pair = encode_prompt(&qwen3, &tokenizer, p, &device)?;
            log::info!("  prompt {}/{}: cond shape={:?}", i + 1, prompts.len(), pair.0.shape().dims());
            Ok(pair)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let (cap_uncond, cap_mask_uncond) = encode_prompt(&qwen3, &tokenizer, "", &device)?;
    drop(qwen3);
    log::info!("  uncond={:?}", cap_uncond.shape().dims());

    log::info!("[3/4] Loading Z-Image transformer + LoRA...");
    let mut model = ZImageModel::load(
        &args.model,
        args.lora_rank,
        args.lora_alpha,
        device.clone(),
        args.seed,
    )?;
    if let Some(lp) = &args.lora_path {
        model.bundle.load(lp, &device)?;
        log::info!("  applied LoRA from {:?} (rank={}, alpha={})",
            lp, args.lora_rank, args.lora_alpha);
    }

    log::info!("[4/4] Sampling {} prompt(s) at {}² ({} steps, cfg={}, shift={})...",
        cond_pairs.len(), args.size, args.steps, args.cfg, args.shift);

    // Split path: denoise every prompt while the transformer is resident,
    // then unload it and VAE-decode each collected latent. At 1024² the
    // Z-Image transformer (~11.5 GB) and the VAE conv workspace cannot
    // both fit in 24 GB simultaneously, so dropping the model before
    // decode is required.
    let _no_grad = flame_core::autograd::AutogradContext::no_grad();
    let _ckpt = eridiffusion_core::sampler::zimage_sampler::CheckpointGuard::disable();
    let pad_width = std::cmp::max(3, cond_pairs.len().to_string().len());
    let mut latents: Vec<flame_core::Tensor> = Vec::with_capacity(cond_pairs.len());

    for (idx, (cap_feats, cap_mask)) in cond_pairs.iter().enumerate() {
        log::info!("  [{}/{}] denoising prompt...", idx + 1, cond_pairs.len());
        let latent = zimage_sampler::denoise_latent(
            &mut model,
            cap_feats, Some(cap_mask),
            Some(&cap_uncond), Some(&cap_mask_uncond),
            args.size, args.size,
            args.steps,
            args.cfg,
            args.shift,
            args.seed,
            &device,
        )?;
        latents.push(latent);
    }

    // Drop transformer weights before VAE decode to free VRAM.
    drop(model);
    flame_core::cuda_alloc_pool::clear_pool_cache();
    flame_core::trim_cuda_mempool(0);

    for (idx, latent) in latents.iter().enumerate() {
        let out_path = if multi_mode {
            let dir = args.output_dir.as_ref().unwrap();
            dir.join(format!("sample_{:0>width$}.png", idx + 1, width = pad_width))
        } else {
            args.output.clone()
        };
        zimage_sampler::decode_latent_to_png(latent, &args.vae_path, &out_path, &device)?;
        log::info!("  [{}/{}] saved {:?}", idx + 1, latents.len(), out_path);
    }
    log::info!("Done — {} sample(s) saved", latents.len());
    Ok(())
}

/// Tokenize + Qwen3-encode the prompt with the OT chat template, returning
/// (hidden_state, mask). Mask is 1 at valid positions, 0 at PAD positions —
/// model.forward uses it to substitute the trained `cap_pad_token` for
/// PAD-position outputs (matches trainer behavior).
fn encode_prompt(
    qwen: &Qwen3Encoder,
    tok: &tokenizers::Tokenizer,
    prompt: &str,
    device: &std::sync::Arc<flame_core::CudaDevice>,
) -> anyhow::Result<(flame_core::Tensor, flame_core::Tensor)> {
    let wrapped = format!("{ZIMAGE_TEMPLATE_PRE}{}{ZIMAGE_TEMPLATE_POST}", prompt.trim());
    let enc = tok.encode(wrapped.as_str(), false)
        .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
    let mut ids: Vec<i32> = enc.get_ids().iter().map(|&i| i as i32).collect();
    let valid_len = ids.len().min(TXT_PAD_LEN);
    ids.resize(TXT_PAD_LEN, PAD_TOKEN_ID);
    let hidden = qwen.encode(&ids)?.to_dtype(DType::BF16)?;
    let mut mask_data = vec![0.0f32; TXT_PAD_LEN];
    for slot in mask_data.iter_mut().take(valid_len) { *slot = 1.0; }
    let mask = flame_core::Tensor::from_vec(
        mask_data,
        flame_core::Shape::from_dims(&[1, TXT_PAD_LEN]),
        device.clone(),
    )?.to_dtype(DType::BF16)?;
    Ok((hidden, mask))
}

fn load_qwen3_weights(path: &std::path::Path, device: &std::sync::Arc<flame_core::CudaDevice>)
    -> flame_core::Result<std::collections::HashMap<String, flame_core::Tensor>>
{
    if path.is_file() {
        return flame_core::serialization::load_file(path, device);
    }
    let mut all = std::collections::HashMap::new();
    for entry in std::fs::read_dir(path).map_err(|e| flame_core::Error::Io(format!("read_dir: {e}")))? {
        let p = entry.map_err(|e| flame_core::Error::Io(format!("entry: {e}")))?.path();
        if p.extension().and_then(|e| e.to_str()) == Some("safetensors") {
            let part = flame_core::serialization::load_file(&p, device)?;
            all.extend(part);
        }
    }
    Ok(all)
}
