//! sample_qwenimage — text → Qwen-Image-2512 generation, optionally with a
//! trained LoRA. Patterned after `sample_anima.rs`. Uses the existing
//! `qwenimage_sampler::sample_image` for the denoise + VAE-decode pipeline.

use clap::Parser;
use flame_core::{DType, Tensor};
use std::path::PathBuf;

use eridiffusion_core::encoders::qwen25vl::Qwen25VLEncoder;
use eridiffusion_core::models::QwenImageTrainingModel;
use eridiffusion_core::sampler::qwenimage_sampler;

const QWEN_PAD_ID: i32 = 151643;
const TXT_PAD_LEN_DEFAULT: usize = 512;
/// Qwen-Image PROMPT_TEMPLATE_ENCODE — must match prepare_qwenimage and
/// train_qwenimage's sample-setup exactly, otherwise the DiT sees
/// out-of-distribution conditioning.
const PROMPT_PREFIX: &str =
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, \
     texture, quantity, text, spatial relationships of the objects and background:\
     <|im_end|>\n<|im_start|>user\n";
const PROMPT_SUFFIX: &str = "<|im_end|>\n<|im_start|>assistant\n";
const DROP_IDX: usize = 34;

#[derive(Parser)]
struct Args {
    /// Single prompt. Mutually exclusive with `--prompts-file`.
    #[arg(long)] prompt: Option<String>,
    /// Newline-separated prompts file for batch sampling. Blank lines and
    /// `#`-prefixed comments are skipped. Requires `--output-dir`. The
    /// text encoder is loaded once, every prompt is encoded, the TE is
    /// dropped, then the DiT loads once and serves all prompts — instead
    /// of paying TE+DiT load on every standalone invocation.
    #[arg(long)] prompts_file: Option<PathBuf>,
    #[arg(long, default_value = "")] negative_prompt: String,
    /// Single-prompt output path. Used when `--prompt` is given.
    #[arg(long, default_value = "output.png")] output: PathBuf,
    /// Multi-prompt output directory. Required with `--prompts-file`.
    /// Files are written as `sample_001.png`, `sample_002.png`, ...
    #[arg(long)] output_dir: Option<PathBuf>,
    /// Qwen-Image-2512 transformer dir (the `transformer/` subdir of the HF
    /// release, with 9 sharded `diffusion_pytorch_model-...safetensors`).
    #[arg(long)] model: PathBuf,
    /// `qwen_image_vae.safetensors` (wan21 internal-key format).
    #[arg(long)] vae_path: PathBuf,
    /// Directory of Qwen2.5-VL text encoder safetensors shards
    /// (`text_encoder/` subdir of `qwen-image-2512`), or a single combined file.
    #[arg(long)] text_encoder: PathBuf,
    /// `tokenizer.json` for Qwen2.5-VL (Qwen-Image-2512's tokenizer subdir).
    #[arg(long)] tokenizer_path: PathBuf,
    #[arg(long, default_value = "512")] size: usize,
    #[arg(long, default_value = "50")] steps: usize,
    /// CFG scale. Set to 1.0 to disable classifier-free guidance.
    #[arg(long, default_value = "4.0")] cfg: f32,
    #[arg(long, default_value = "42")] seed: u64,
    #[arg(long, default_value_t = TXT_PAD_LEN_DEFAULT)] max_text_len: usize,
    /// Optional trained LoRA safetensors. Accepts `weights`-mode (bare
    /// LoRA tensors) and `full`-mode (LoRA + AdamW + step) checkpoints
    /// from `train_qwenimage` — the loader matches LoRA prefixes and
    /// ignores the `__opt__/` optimizer-state entries.
    #[arg(long)] lora_path: Option<PathBuf>,
    #[arg(long, default_value = "16")] lora_rank: usize,
    /// Match the alpha used at training time. Mismatch = silent scale drift.
    #[arg(long, default_value = "16.0")] lora_alpha: f32,
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let args = Args::parse();
    let device = flame_core::global_cuda_device();
    let _no_grad = flame_core::autograd::AutogradContext::no_grad();
    flame_core::config::set_default_dtype(DType::BF16);

    if args.size % 16 != 0 {
        anyhow::bail!("size must be divisible by 16, got {}", args.size);
    }

    log::info!("[1/4] Loading Qwen2.5-VL-7B text encoder + tokenizer...");
    let tokenizer = tokenizers::Tokenizer::from_file(&args.tokenizer_path)
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    let te_weights = load_te_weights(&args.text_encoder, &device)?;
    let te_cfg = Qwen25VLEncoder::config_from_weights(&te_weights)?;
    log::info!(
        "  config: hidden={} layers={} heads={} kv_heads={} head_dim={}",
        te_cfg.hidden_size, te_cfg.num_layers, te_cfg.num_heads,
        te_cfg.num_kv_heads, te_cfg.head_dim,
    );
    let te = Qwen25VLEncoder::new(te_weights, te_cfg, device.clone());

    let encode = |text: &str| -> anyhow::Result<Tensor> {
        // Wrap in PROMPT_TEMPLATE_ENCODE, then drop the leading system
        // prompt and TRIM trailing pad tokens — matches inference-flame's
        // `qwenimage_encode::encode_and_trim` (qwenimage_encode.rs:92-115).
        // The diffusers QwenImage DiT was trained with variable-length
        // text embeddings; padding the embedding to a fixed `max_text_len`
        // pollutes joint attention with junk pad-token hidden states and
        // produces noise-only output. Trim back to the actual content
        // length (real tokens − DROP_IDX) before returning.
        let wrapped = format!("{PROMPT_PREFIX}{text}{PROMPT_SUFFIX}");
        let enc = tokenizer.encode(wrapped, false)
            .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
        let raw_ids: Vec<i32> = enc.get_ids().iter().map(|&i| i as i32).collect();
        let work_len = args.max_text_len + DROP_IDX;
        let mut ids: Vec<i32> = raw_ids.iter().take(work_len).copied().collect();
        let real_len_pre_pad = ids.len();
        ids.resize(work_len, QWEN_PAD_ID);
        let real_len = real_len_pre_pad.min(work_len);
        if real_len <= DROP_IDX {
            anyhow::bail!(
                "prompt tokenized to only {real_len} ids; expected > {DROP_IDX} after PROMPT_TEMPLATE_ENCODE wrap"
            );
        }
        let kept_len = real_len - DROP_IDX;
        let full_hidden = te.encode(&ids)?.to_dtype(DType::BF16)?;
        full_hidden.narrow(1, DROP_IDX, kept_len)
            .map_err(|e| anyhow::anyhow!("narrow: {e}"))
    };

    // Resolve prompt list. Single-prompt mode keeps the legacy
    // `--prompt` / `--output` contract; multi-prompt mode reads
    // `--prompts-file` and writes to `--output-dir`.
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
    let conds: Vec<Tensor> = prompts.iter().enumerate()
        .map(|(i, p)| {
            let c = encode(p)?;
            log::info!("  prompt {}/{}: cond shape={:?}", i + 1, prompts.len(), c.shape().dims());
            Ok(c)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let uncond = if args.cfg > 1.0 {
        Some(encode(&args.negative_prompt)?)
    } else {
        None
    };
    drop(te);
    flame_core::cuda_alloc_pool::clear_pool_cache();
    flame_core::trim_cuda_mempool(0);

    log::info!("[3/4] Loading Qwen-Image transformer...");
    let mut model = QwenImageTrainingModel::load(
        &args.model, args.lora_rank, args.lora_alpha,
        /*full_finetune*/ false, device.clone(), args.seed,
    )?;
    if let Some(lp) = &args.lora_path {
        model.bundle.load(lp, &device)?;
        log::info!(
            "  applied LoRA from {} (rank={}, alpha={})",
            lp.display(), args.lora_rank, args.lora_alpha,
        );
    }

    log::info!(
        "[4/4] Sampling {} prompt(s) at {}² ({} steps, cfg={})",
        conds.len(), args.size, args.steps, args.cfg,
    );
    // Width of the zero-padded sample index. Three digits is enough for
    // any practical batch; widen automatically if a future caller passes
    // 1000+ prompts.
    let pad_width = std::cmp::max(3, conds.len().to_string().len());
    for (i, cond) in conds.iter().enumerate() {
        let out_path = if multi_mode {
            let dir = args.output_dir.as_ref().unwrap();
            dir.join(format!("sample_{:0>width$}.png", i + 1, width = pad_width))
        } else {
            args.output.clone()
        };
        log::info!("  [{}/{}] → {}", i + 1, conds.len(), out_path.display());
        qwenimage_sampler::sample_image(
            &mut model,
            cond,
            uncond.as_ref(),
            args.size, args.size,
            args.steps,
            args.cfg,
            args.seed,
            &args.vae_path,
            &out_path,
            &device,
        )?;
    }
    log::info!("Saved {} sample(s)", conds.len());
    Ok(())
}

fn load_te_weights(
    path: &std::path::Path,
    device: &std::sync::Arc<flame_core::CudaDevice>,
) -> flame_core::Result<std::collections::HashMap<String, Tensor>> {
    if path.is_file() {
        return flame_core::serialization::load_file(path, device);
    }
    let mut all = std::collections::HashMap::new();
    for entry in std::fs::read_dir(path)
        .map_err(|e| flame_core::Error::Io(format!("read_dir {}: {e}", path.display())))?
    {
        let p = entry
            .map_err(|e| flame_core::Error::Io(format!("entry: {e}")))?
            .path();
        if p.extension().and_then(|e| e.to_str()) == Some("safetensors") {
            let part = flame_core::serialization::load_file(&p, device)?;
            all.extend(part);
        }
    }
    Ok(all)
}
