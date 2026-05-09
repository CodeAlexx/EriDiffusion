//! prepare_wan22 — build cached video latents + UMT5 text embeddings
//! for Wan 2.2 LoRA training.
//!
//! ## Status (2026-05-09 — Phase 2)
//!
//! Image-input path is real: Wan 2.2 VAE encoder + UMT5-XXL text encoder
//! both run end-to-end and emit non-zero cache entries.
//!
//! Video-input path is NOT yet wired (no ffmpeg/libavcodec dep).
//! `.mp4`/`.webm`/`.mov` files are silently skipped — log a warn so the
//! user can wire ffmpeg out-of-process if they need video before the
//! decoder lands.
//!
//! ## Cache schema
//! Each `<hash>.safetensors` contains:
//!   - `latent`         BF16 `[1, C, T', H', W']` normalized VAE latent
//!     - 5B: C=48, H'=H/16, W'=W/16
//!     - 14B: C=16, H'=H/8, W'=W/8
//!   - `text_embedding` BF16 `[1, 512, 4096]` UMT5 hidden states (right-padded)
//!   - `text_mask`      F32  `[1, 512]` 1 for real tokens, 0 for pad
//!   - `text_real_len`  F32  `[1]` count of real (non-pad) tokens
//!
//! ## CLI
//!   --input-dir <DIR>             images + matching .txt caption files
//!   --output-dir <DIR>            cache destination
//!   --vae-ckpt <FILE>             Wan 2.2 VAE safetensors (5B variant)
//!                                 OR Wan 2.1 VAE safetensors (14B variant)
//!   --text-ckpt <FILE>            UMT5-XXL safetensors
//!   --tokenizer-path <FILE>       UMT5 tokenizer.json
//!   --variant ti2v_5b|t2v_14b|i2v_14b   default ti2v_5b
//!   --size <PX>                   square pixel size (default 256)
//!   --num-latent-frames <N>       default 1 (single-frame image-as-video)

use clap::Parser;
use std::collections::HashMap;
use std::path::PathBuf;

use eridiffusion_core::encoders::umt5::Umt5Encoder;
use eridiffusion_core::encoders::wan21_encoder::Wan21VaeEncoder;
use eridiffusion_core::encoders::wan22_vae::Wan22VaeEncoder;
use flame_core::{serialization::save_file, DType, Shape, Tensor};

const TEXT_MAX_LEN: usize = 512;

/// Variant-erased Wan VAE — internally either Wan2.2 (TI2V-5B) or Wan2.1
/// (T2V/I2V-A14B). Both implementations produce normalized latents
/// `[B, C, T', H', W']` and accept video `[B, 3, T, H, W]` BF16 in [-1, 1].
enum WanVae {
    V22(Wan22VaeEncoder),
    V21(Wan21VaeEncoder),
}

impl WanVae {
    fn encode_video(&self, video: &Tensor) -> flame_core::Result<Tensor> {
        match self {
            WanVae::V22(e) => e.encode(video),
            WanVae::V21(e) => {
                // Wan21VaeEncoder exposes encode_image_raw for [B,3,H,W];
                // for video [B,3,T,H,W] we need encode_video_raw which is
                // currently private. Single-frame fallback: squeeze T,
                // run image path, unsqueeze T.
                let dims = video.shape().dims().to_vec();
                if dims.len() != 5 {
                    return Err(flame_core::Error::InvalidInput(format!(
                        "WanVae::encode_video expects [B,3,T,H,W], got {dims:?}"
                    )));
                }
                if dims[2] != 1 {
                    return Err(flame_core::Error::InvalidInput(
                        "Wan2.1 VAE multi-frame video path not wired in EDv2 yet \
                         (Wan21VaeEncoder::encode_video_raw is private). Use --num-latent-frames 1 \
                         or switch to TI2V-5B (Wan2.2 VAE) for multi-frame.".into()
                    ));
                }
                let img = video.squeeze(Some(2))?; // [B, 3, H, W]
                // Wan22VaeEncoder::encode returns NORMALIZED latents
                // (mean/std baked in). Match that contract here so cache
                // semantics are uniform across variants — trainer doesn't
                // need to know which VAE produced the cache.
                let z = e.encode_image_normalized(&img)?; // [B, 16, H/8, W/8]
                z.unsqueeze(2) // [B, 16, 1, H/8, W/8]
            }
        }
    }
}

#[derive(Parser)]
struct Args {
    #[arg(long)] input_dir: PathBuf,
    #[arg(long)] output_dir: PathBuf,
    /// Wan 2.2 VAE for `--variant ti2v_5b`, Wan 2.1 VAE for the 14B variants.
    #[arg(long)] vae_ckpt: PathBuf,
    /// UMT5-XXL safetensors (HF `T5EncoderModel` key layout).
    #[arg(long)] text_ckpt: PathBuf,
    /// UMT5 SentencePiece tokenizer.json.
    #[arg(long)] tokenizer_path: PathBuf,

    /// Variant: `ti2v_5b` (Wan2.2 VAE 16× / 48 ch) or `t2v_14b`/`i2v_14b`
    /// (Wan2.1 VAE 8× / 16 ch).
    #[arg(long, default_value = "ti2v_5b")] variant: String,

    /// Square spatial size in pixels.
    #[arg(long, default_value = "256")] size: usize,
    /// Number of latent frames. For single images, leave at 1. Source frame
    /// count must satisfy `(F_src - 1) % 4 == 0`; we apply
    /// `F_lat = 1 + (F_src - 1) / 4`.
    #[arg(long, default_value = "1")] num_latent_frames: usize,

    #[arg(long, default_value_t = 0)] max_samples: usize,
    #[arg(long, default_value_t = true)] skip_existing: bool,
}

fn main() -> anyhow::Result<()> {
    if std::env::var_os("FLAME_ALLOC_POOL").is_none() {
        // SAFETY: single-threaded at this point.
        unsafe { std::env::set_var("FLAME_ALLOC_POOL", "0"); }
    }
    env_logger::init();
    let args = Args::parse();
    std::fs::create_dir_all(&args.output_dir)?;

    flame_core::config::set_default_dtype(DType::BF16);
    let _no_grad = flame_core::autograd::AutogradContext::no_grad();
    let device = flame_core::global_cuda_device();

    // Variant-aware VAE stride + channel count.
    let (vae_stride, _expected_channels): (usize, usize) = match args.variant.as_str() {
        "ti2v_5b" | "ti2v-5b" | "5b" => (16, 48),
        "t2v_14b" | "t2v-14b" | "14b" | "t2v" | "i2v_14b" | "i2v-14b" | "i2v" => (8, 16),
        other => anyhow::bail!("unknown --variant '{other}'"),
    };
    let pixel_step = 2 * vae_stride;
    if args.size % pixel_step != 0 {
        anyhow::bail!(
            "--size {} must be a multiple of {} for variant {} (VAE stride {} × patch 2)",
            args.size, pixel_step, args.variant, vae_stride
        );
    }

    // 1) Load VAE.
    log::info!("[1/3] Loading Wan VAE ({})...", args.variant);
    let vae: WanVae = match args.variant.as_str() {
        "ti2v_5b" | "ti2v-5b" | "5b" => {
            WanVae::V22(Wan22VaeEncoder::load(&args.vae_ckpt, &device)?)
        }
        _ => {
            // Wan21VaeEncoder uses `from_safetensors(path: &str, device)` —
            // legacy entry name, predates the wan22 sibling.
            WanVae::V21(Wan21VaeEncoder::from_safetensors(
                args.vae_ckpt.to_str().ok_or_else(|| anyhow::anyhow!("--vae-ckpt path is not utf-8"))?,
                &device,
            )?)
        }
    };

    // 2) Load UMT5.
    log::info!("[2/3] Loading UMT5-XXL...");
    let mut text_enc = Umt5Encoder::load(&args.text_ckpt, &device)?;
    let tok = tokenizers::Tokenizer::from_file(&args.tokenizer_path)
        .map_err(|e| anyhow::anyhow!("UMT5 tokenizer: {e}"))?;

    // 3) Walk input dir using shared video::prep helper.
    log::info!("[3/3] Encoding samples at {}²...", args.size);
    let mut pairs = eridiffusion_core::video::prep::walk_dataset(&args.input_dir)?;
    log::info!("Found {} image/video samples", pairs.len());
    if args.max_samples > 0 && pairs.len() > args.max_samples {
        pairs.truncate(args.max_samples);
    }

    // Frame count for video clips. For images this is ignored (always F=1).
    // Snap to Wan VAE temporal constraint `(F-1) % 4 == 0` so the latent
    // frame count is integral.
    let pixel_frames = eridiffusion_core::video::shape::pixel_frames(args.num_latent_frames, 4);
    let snapped_pixel_frames =
        eridiffusion_core::video::prep::snap_to_wan_frame_count(pixel_frames);
    if snapped_pixel_frames != pixel_frames {
        log::warn!(
            "Wan VAE temporal constraint: requested {pixel_frames} pixel frames \
             snapped to {snapped_pixel_frames} (must satisfy (F-1) % 4 == 0)"
        );
    }

    // Cache schema sentinel.
    let meta = format!(
        r#"{{"version": 1, "format": "wan22-v1", "variant": "{}", "fields": ["latent", "text_embedding", "text_mask", "text_real_len"]}}"#,
        args.variant
    );
    let _ = std::fs::write(args.output_dir.join("_meta.json"), meta);

    let mut cached = 0usize;
    for pair in pairs.iter() {
        let media = &pair.media;
        let stem = media.file_stem().unwrap().to_string_lossy().into_owned();
        let hash = format!("{:x}", md5::compute(media.to_string_lossy().as_bytes()));
        let out_path = args.output_dir.join(format!("{hash}.safetensors"));
        if args.skip_existing && out_path.exists() {
            continue;
        }

        // ── Decode image OR video → [1, 3, F, H, W] BF16 in [-1, 1] ──
        let clip = eridiffusion_core::video::decode::decode_to_tensor(
            media,
            args.size,
            snapped_pixel_frames,
            None,
            device.clone(),
        )?;
        let video = clip.video;

        let latent = vae.encode_video(&video)?; // [1, C, T', H/stride, W/stride]

        // ── Caption → UMT5 ──
        let caption = pair.read_caption();
        let enc = tok
            .encode(caption, true)
            .map_err(|e| anyhow::anyhow!("UMT5 tokenize: {e}"))?;
        let mut ids: Vec<i32> = enc.get_ids().iter().map(|&x| x as i32).collect();
        if ids.len() > TEXT_MAX_LEN {
            ids.truncate(TEXT_MAX_LEN);
        }
        let real_len = ids.len();
        let text_embed = text_enc.encode(&ids)?; // [1, 512, 4096] (encoder pads internally)

        // text_mask + real_len
        let mut mask_data = vec![0f32; TEXT_MAX_LEN];
        for v in mask_data.iter_mut().take(real_len) {
            *v = 1.0;
        }
        let text_mask = Tensor::from_vec(
            mask_data,
            Shape::from_dims(&[1, TEXT_MAX_LEN]),
            device.clone(),
        )?;
        let text_real_len_t = Tensor::from_vec(
            vec![real_len as f32],
            Shape::from_dims(&[1]),
            device.clone(),
        )?;

        // Save
        let mut tensors = HashMap::new();
        tensors.insert("latent".to_string(), latent.to_dtype(DType::BF16)?);
        tensors.insert("text_embedding".to_string(), text_embed.to_dtype(DType::BF16)?);
        tensors.insert("text_mask".to_string(), text_mask);
        tensors.insert("text_real_len".to_string(), text_real_len_t);
        save_file(&tensors, &out_path)?;
        cached += 1;
        if cached % 5 == 0 || cached == pairs.len() {
            log::info!("Cached {}/{} ({stem})", cached, pairs.len());
        }
    }

    log::info!("Done. {cached} samples cached to {:?}", args.output_dir);
    Ok(())
}
