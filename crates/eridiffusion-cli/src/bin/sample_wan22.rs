//! sample_wan22 — Wan 2.2 video DiT inference (Euler flow-matching).
//!
//! ## Status
//!
//! Like the trainer, the Wan 2.2 transformer forward is not yet ported
//! into eridiffusion-core. This binary parses the full CLI surface,
//! loads both experts, picks the right one per timestep via
//! `wan22_sampler::expert_for_timestep`, and walks the Euler schedule —
//! but every step hits the deferred forward and bails. Use it to smoke
//! the dispatch wiring; real samples need the forward port.

use clap::Parser;
use std::path::PathBuf;

use eridiffusion_core::models::wan22::{Wan22Config, Wan22Model, Wan22Variant};
use eridiffusion_core::sampler::wan22_sampler::{
    self as wan22, Expert, DEFAULT_NOISE_BOUNDARY_T2V, DEFAULT_SHIFT_TI2V_5B,
};
use flame_core::autograd::AutogradContext;
use flame_core::{DType, Shape, Tensor};

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "t2v_14b")] variant: String,
    /// Single-expert checkpoint for 5B; low-noise checkpoint for 14B.
    #[arg(long)] low_noise: PathBuf,
    /// High-noise checkpoint (14B only).
    #[arg(long)] high_noise: Option<PathBuf>,
    #[arg(long, default_value_t = DEFAULT_NOISE_BOUNDARY_T2V)] noise_boundary: f32,
    #[arg(long, default_value = "bf16")] weight_dtype: String,

    /// Optional LoRA pair to merge at sample time.
    #[arg(long)] low_lora: Option<PathBuf>,
    #[arg(long)] high_lora: Option<PathBuf>,

    #[arg(long, default_value = "16")] rank: usize,
    #[arg(long, default_value = "16.0")] lora_alpha: f32,

    #[arg(long, default_value = "")] prompt: String,
    /// Pre-encoded UMT5 text embedding cache (`.safetensors` with
    /// `text_embedding`). Required until the encoder is ported.
    #[arg(long)] prompt_embed: PathBuf,

    #[arg(long, default_value = "256")] size: usize,
    #[arg(long, default_value = "1")] num_frames: usize,
    #[arg(long, default_value = "20")] steps: usize,
    #[arg(long, default_value = "5.0")] cfg: f32,
    #[arg(long, default_value_t = DEFAULT_SHIFT_TI2V_5B)] shift: f32,
    #[arg(long, default_value = "42")] seed: u64,
    #[arg(long, default_value = "output/wan22_sample.safetensors")] out: PathBuf,
}

fn parse_weight_dtype(s: &str) -> anyhow::Result<DType> {
    match s.to_ascii_lowercase().as_str() {
        "bf16" => Ok(DType::BF16),
        "fp16" => Ok(DType::F16),
        "fp8" | "fp8_scaled" | "fp8_e4m3" => Ok(DType::BF16),
        other => anyhow::bail!("unknown --weight-dtype: {other}"),
    }
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let args = Args::parse();
    flame_core::config::set_default_dtype(DType::BF16);
    let device = flame_core::global_cuda_device();

    let variant = Wan22Variant::parse(&args.variant)
        .map_err(|e| anyhow::anyhow!("--variant: {e}"))?;
    let cfg = Wan22Config::for_variant(variant);
    let weight_dtype = parse_weight_dtype(&args.weight_dtype)?;
    let dual = variant.is_dual_expert();

    let mut low = Wan22Model::load(
        &args.low_noise, cfg.clone(),
        args.rank, args.lora_alpha,
        weight_dtype, device.clone(), 42, "low",
    )?;
    let mut high = if dual {
        let p = args.high_noise.as_ref().ok_or_else(|| anyhow::anyhow!(
            "variant {} is dual-expert; --high-noise required", variant.as_str()
        ))?;
        Some(Wan22Model::load(
            p, cfg.clone(),
            args.rank, args.lora_alpha,
            weight_dtype, device.clone(), 42, "high",
        )?)
    } else { None };

    if let Some(p) = &args.low_lora {
        log::info!("[wan22:low] LoRA <- {}", p.display());
        let _ = (p, &mut low); // hydrate path: re-uses train_wan22's rehydrate_bundle when consolidated.
        log::warn!("[wan22:low] LoRA hydration in sampler is a stub; merge externally for now.");
    }
    if let Some(p) = &args.high_lora {
        log::info!("[wan22:high] LoRA <- {}", p.display());
        let _ = (p, &mut high);
        log::warn!("[wan22:high] LoRA hydration in sampler is a stub.");
    }

    // Load text embedding from cache.
    let txt_map = flame_core::serialization::load_file(&args.prompt_embed, &device)?;
    let txt = txt_map.get("text_embedding")
        .ok_or_else(|| anyhow::anyhow!("--prompt-embed missing 'text_embedding'"))?
        .to_dtype(DType::BF16)?;

    // Build initial noise.
    let h_lat = args.size / 32;
    let w_lat = args.size / 32;
    let f_lat = args.num_frames.max(1);
    let mut latent = Tensor::randn(
        Shape::from_dims(&[1, cfg.in_channels, f_lat, h_lat, w_lat]),
        0.0, 1.0, device.clone(),
    )?.to_dtype(DType::BF16)?;

    // Schedule.
    let sigmas = wan22::schedule(args.steps, args.shift);
    log::info!(
        "[wan22] steps={} shift={} cfg_scale={} variant={} boundary={}",
        args.steps, args.shift, args.cfg, variant.as_str(), args.noise_boundary
    );

    let _no_grad = AutogradContext::no_grad();
    for step in 0..args.steps {
        let sigma = sigmas[step];
        let sigma_next = sigmas[step + 1];
        let t = wan22::sigma_to_timestep(sigma);
        let t_tensor = Tensor::from_vec(
            vec![t], Shape::from_dims(&[1]), device.clone(),
        )?;
        // Continuous-t for dispatch: sigma is already shift-applied, so
        // use the raw t_continuous = sigma here (the boundary is in the
        // shift-applied space too — the trainer samples and applies shift,
        // then dispatches; here we walk the shifted sigmas).
        let chosen = if dual {
            wan22::expert_for_timestep(sigma, args.noise_boundary)
        } else {
            Expert::Low
        };
        let pred = match chosen {
            Expert::High => match high.as_mut() {
                Some(hm) => hm.forward(&latent, &t_tensor, &txt),
                None => Err(eridiffusion_core::EriDiffusionError::Model(
                    "high-noise expert not loaded".into()
                )),
            },
            Expert::Low => low.forward(&latent, &t_tensor, &txt),
        }
        .map_err(|e| anyhow::anyhow!("step {step} expert={:?}: {e}", chosen))?;
        latent = wan22::euler_step(&latent, &pred, sigma, sigma_next)?;
    }

    // Save the latent (decoding requires Wan22 VAE, not yet ported).
    if let Some(parent) = args.out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut out = std::collections::HashMap::new();
    out.insert("latent".to_string(), latent.to_dtype(DType::BF16)?);
    flame_core::serialization::save_file(&out, &args.out)?;
    log::info!("Saved latent {}", args.out.display());
    Ok(())
}
