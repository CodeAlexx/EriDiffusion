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
    /// `text_embedding` and optional `text_mask`).
    #[arg(long)] prompt_embed: PathBuf,
    /// Pre-encoded UMT5 NEGATIVE text embedding cache. When set AND
    /// --cfg > 1.0, real CFG kicks in (2 forwards per step). When unset,
    /// falls back to single-forward (cond only) — matches diffusers's
    /// "no negative prompt" behavior.
    #[arg(long)] negative_embed: Option<PathBuf>,

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

    // 2026-05-09 (audit C4/M5): wired. Both experts load via the shared
    // `Wan22LoraBundle::rehydrate_from_path` (same key format the trainer
    // saves with) — fail-fast on miss instead of warn-and-noop.
    if let Some(p) = &args.low_lora {
        log::info!("[wan22:low] LoRA <- {}", p.display());
        let (hits, total) = low.lora.rehydrate_from_path(p, device.clone())
            .map_err(|e| anyhow::anyhow!("--low-lora load: {e}"))?;
        if hits == 0 {
            anyhow::bail!("--low-lora {} has no matching adapters (key prefix mismatch)", p.display());
        }
        log::info!("[wan22:low] hydrated {}/{} adapters", hits, total);
    }
    if let Some(p) = &args.high_lora {
        let high_ref = high.as_mut().ok_or_else(|| anyhow::anyhow!(
            "--high-lora given but variant {} has no high-noise expert", variant.as_str()
        ))?;
        log::info!("[wan22:high] LoRA <- {}", p.display());
        let (hits, total) = high_ref.lora.rehydrate_from_path(p, device.clone())
            .map_err(|e| anyhow::anyhow!("--high-lora load: {e}"))?;
        if hits == 0 {
            anyhow::bail!("--high-lora {} has no matching adapters (key prefix mismatch)", p.display());
        }
        log::info!("[wan22:high] hydrated {}/{} adapters", hits, total);
    }

    // Load conditional text embedding from cache.
    let txt_map = flame_core::serialization::load_file(&args.prompt_embed, &device)?;
    let txt = txt_map.get("text_embedding")
        .ok_or_else(|| anyhow::anyhow!("--prompt-embed missing 'text_embedding'"))?
        .to_dtype(DType::BF16)?;
    let txt_mask = txt_map.get("text_mask").and_then(|t| t.to_dtype(DType::F32).ok());

    // Optional unconditional embedding for CFG.
    let (uncond, uncond_mask): (Option<Tensor>, Option<Tensor>) =
        match args.negative_embed.as_ref() {
            Some(p) => {
                let m = flame_core::serialization::load_file(p, &device)?;
                let u = m.get("text_embedding")
                    .ok_or_else(|| anyhow::anyhow!("--negative-embed missing 'text_embedding'"))?
                    .to_dtype(DType::BF16)?;
                let um = m.get("text_mask").and_then(|t| t.to_dtype(DType::F32).ok());
                (Some(u), um)
            }
            None => (None, None),
        };
    let do_cfg = args.cfg > 1.0 && uncond.is_some();

    // Build initial noise.
    // 2026-05-09 (audit C3): variant-aware VAE stride. 5B uses Wan2.2 VAE
    // (16× spatial); 14B uses Wan2.1 VAE (8×).
    let vae_stride: usize = match variant {
        eridiffusion_core::models::Wan22Variant::Ti2v5b => 16,
        _ => 8,
    };
    if args.size % (2 * vae_stride) != 0 {
        anyhow::bail!(
            "--size {} must be a multiple of {} for variant {} (VAE stride {} × patch 2)",
            args.size, 2 * vae_stride, variant.as_str(), vae_stride
        );
    }
    let h_lat = args.size / vae_stride;
    let w_lat = args.size / vae_stride;
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
        // 2026-05-09 (audit C1): Wan22Model::forward wants [C, F, H, W];
        // sampler latent is [1, C, F, H, W]. Squeeze before forward and
        // re-add the leading dim afterwards so euler_step shapes match.
        let latent_4d = latent.squeeze(Some(0))?;
        // 2026-05-09 (audit C4): real CFG when --negative-embed + --cfg > 1.
        // Two forwards per step: cond + uncond, then guidance interpolation
        // `pred = uncond + s * (cond - uncond)`.
        let mut run_forward = |which: Expert, ctx: &Tensor, mask: Option<&Tensor>|
            -> anyhow::Result<Tensor>
        {
            match which {
                Expert::High => match high.as_mut() {
                    Some(hm) => hm.forward(&latent_4d, &t_tensor, ctx, mask)
                        .map_err(|e| anyhow::anyhow!(e)),
                    None => Err(anyhow::anyhow!("high-noise expert not loaded")),
                },
                Expert::Low => low.forward(&latent_4d, &t_tensor, ctx, mask)
                    .map_err(|e| anyhow::anyhow!(e)),
            }
        };
        let pred_4d = if do_cfg {
            let v_cond = run_forward(chosen, &txt, txt_mask.as_ref())?;
            let v_uncond = run_forward(
                chosen,
                uncond.as_ref().unwrap(),
                uncond_mask.as_ref(),
            )?;
            // F32 for guidance numerical safety, then back to BF16.
            let vc_f32 = v_cond.to_dtype(DType::F32)?;
            let vu_f32 = v_uncond.to_dtype(DType::F32)?;
            let diff = vc_f32.sub(&vu_f32)?;
            let scaled = diff.mul_scalar(args.cfg)?;
            let v_guided = vu_f32.add(&scaled)?;
            v_guided.to_dtype(DType::BF16)?
        } else {
            run_forward(chosen, &txt, txt_mask.as_ref())?
        };
        let pred = pred_4d.unsqueeze(0)?;
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
