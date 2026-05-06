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
    #[arg(long)] prompt: String,
    #[arg(long, default_value = "output.png")] output: PathBuf,
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

    log::info!("[1/4] Loading Qwen3 + tokenizer...");
    let qwen_weights = load_qwen3_weights(&args.qwen3, &device)?;
    let mut qcfg = Qwen3Encoder::config_from_weights(&qwen_weights)?;
    // Qwen3-4B layer 34 = hidden_states[-2]. See prepare_zimage.rs.
    qcfg.extract_layers = vec![34];
    let qwen3 = Qwen3Encoder::new(qwen_weights, qcfg, device.clone());
    let tokenizer = tokenizers::Tokenizer::from_file(&args.tokenizer_path)
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;

    log::info!("[2/4] Encoding prompt...");
    let (cap_feats, cap_mask) = encode_prompt(&qwen3, &tokenizer, &args.prompt, &device)?;
    let (cap_uncond, cap_mask_uncond) = encode_prompt(&qwen3, &tokenizer, "", &device)?;
    drop(qwen3);
    log::info!("  cond={:?} uncond={:?}", cap_feats.shape().dims(), cap_uncond.shape().dims());

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

    log::info!("[4/4] Sampling at {}² ({} steps, cfg={}, shift={})...",
        args.size, args.steps, args.cfg, args.shift);
    zimage_sampler::sample_image(
        &mut model,
        &cap_feats, Some(&cap_mask),
        Some(&cap_uncond), Some(&cap_mask_uncond),
        args.size, args.size,
        args.steps,
        args.cfg,
        args.shift,
        args.seed,
        &args.vae_path,
        &args.output,
        &device,
    )?;

    log::info!("Saved {:?}", args.output);
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
