//! Z-Image (NextDiT) image generation — pure Rust, no Python.
//!
//! Usage: cargo run --release --bin zimage_generate
//!
//! Pipeline:
//! 1. Load cached text embeddings (safetensors)
//! 2. Load Z-Image NextDiT model weights (safetensors, ~12GB)
//! 3. Create noise, run Euler denoise loop with CFG
//! 4. Save denoised latents as safetensors
//! 5. VAE decode via separate script (different VAE format from Klein)

use eridiffusion::models::zimage_model::ZImageTransformer;
use eridiffusion::inference::zimage_sampling::{build_sigma_schedule, euler_denoise};
use flame_core::{global_cuda_device, DType, Shape, Tensor};
use std::time::Instant;

const MODEL_PATH: &str = "/home/alex/.serenity/models/checkpoints/z_image_de_turbo_v1_bf16.safetensors";
const EMBEDDINGS_PATH: &str = "/home/alex/EriDiffusion/eridiffusion/eridiffusion/cached_zimage_embeddings.safetensors";
const VAE_PATH: &str = "/home/alex/.serenity/models/vaes/ae.safetensors";
const OUTPUT_LATENTS: &str = "/home/alex/serenity/output/zimage_denoised_latents.safetensors";

const NUM_STEPS: usize = 8;
const SHIFT: f32 = 1.8;
const CFG_SCALE: f32 = 0.0;
const SEED: u64 = 42;
const WIDTH: usize = 512;
const HEIGHT: usize = 512;

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let t_total = Instant::now();

    println!("============================================================");
    println!("Z-Image (NextDiT) — Pure Rust Inference");
    println!("============================================================");

    let device = global_cuda_device();

    // ------------------------------------------------------------------
    // Stage 1: Load cached embeddings
    // ------------------------------------------------------------------
    println!("\n--- Stage 1: Load cached embeddings ---");
    let t0 = Instant::now();
    let cached = flame_core::serialization::load_file(
        std::path::Path::new(EMBEDDINGS_PATH), &device
    )?;
    let pos_hidden = cached.get("pos_hidden")
        .ok_or_else(|| anyhow::anyhow!("Missing pos_hidden"))?;
    let neg_hidden = cached.get("neg_hidden")
        .ok_or_else(|| anyhow::anyhow!("Missing neg_hidden"))?;
    println!("  pos_hidden: {:?} {:?}", pos_hidden.dims(), pos_hidden.dtype());
    println!("  neg_hidden: {:?} {:?}", neg_hidden.dims(), neg_hidden.dtype());
    println!("  Loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // ------------------------------------------------------------------
    // Stage 2: Load Z-Image NextDiT model
    // ------------------------------------------------------------------
    println!("\n--- Stage 2: Load Z-Image model ---");
    let t0 = Instant::now();
    let model = ZImageTransformer::from_safetensors(MODEL_PATH)?;
    println!("  Config: {:?}", model.config());
    println!("  Loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // ------------------------------------------------------------------
    // Stage 3: Create noise + sigma schedule
    // ------------------------------------------------------------------
    println!("\n--- Stage 3: Prepare noise + sigmas ---");

    let latent_h = HEIGHT / 8;  // Z-Image VAE uses /8 (not /16 like Flux)
    let latent_w = WIDTH / 8;
    let in_channels = 16;

    // Seeded noise: [1, 16, latent_h, latent_w]
    let numel = 1 * in_channels * latent_h * latent_w;
    let noise_data: Vec<f32> = {
        use rand::prelude::*;
        let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);
        let mut v = Vec::with_capacity(numel);
        for _ in 0..numel / 2 {
            let u1: f32 = rng.gen::<f32>().max(1e-10);
            let u2: f32 = rng.gen::<f32>();
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f32::consts::PI * u2;
            v.push(r * theta.cos());
            v.push(r * theta.sin());
        }
        if numel % 2 == 1 {
            let u1: f32 = rng.gen::<f32>().max(1e-10);
            let u2: f32 = rng.gen::<f32>();
            v.push((-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos());
        }
        v
    };

    let noise = Tensor::from_f32_to_bf16(
        noise_data,
        Shape::from_dims(&[1, in_channels, latent_h, latent_w]),
        device.clone(),
    )?;
    println!("  Noise: {:?} (spatial)", noise.dims());

    let sigmas = build_sigma_schedule(NUM_STEPS, SHIFT);
    println!("  Sigmas: {} values, max={:.4}, min={:.4}",
             sigmas.len(), sigmas[0], sigmas[sigmas.len()-2]);

    // ------------------------------------------------------------------
    // Stage 4: Denoise
    // ------------------------------------------------------------------
    println!("\n--- Stage 4: Denoise ({} steps, Euler velocity, CFG={}) ---", NUM_STEPS, CFG_SCALE);
    let t0 = Instant::now();

    let denoised = euler_denoise(
        |x, sigma| {
            // Create sigma tensor [sigma] as BF16
            let sigma_t = Tensor::from_f32_to_bf16(
                vec![sigma],
                Shape::from_dims(&[1]),
                device.clone(),
            )?;

            // Conditional forward pass — model returns velocity directly
            // (negation is baked into the model, musubi negates post-hoc)
            let model_output = model.forward(x, &sigma_t, pos_hidden)?;

            // CFG: turbo uses 0.0 (no guidance). For base model cfg>1:
            // noise_pred = model_out + cfg * (model_out - uncond_out)
            if CFG_SCALE > 0.0 {
                let uncond = model.forward(x, &sigma_t, neg_hidden)?;
                let diff = model_output.sub(&uncond)?;
                Ok(model_output.add(&diff.mul_scalar(CFG_SCALE)?)?)
            } else {
                Ok(model_output)
            }
        },
        noise,
        &sigmas,
    )?;

    let dt = t0.elapsed().as_secs_f32();
    println!("  Denoised in {:.1}s ({:.2}s/step)", dt, dt / NUM_STEPS as f32);
    println!("  Output: {:?}", denoised.dims());

    // ------------------------------------------------------------------
    // Stage 5: VAE Decode
    // ------------------------------------------------------------------
    println!("\n--- Stage 5: VAE Decode ---");
    let t0 = Instant::now();

    // Free DiT weights to make room for VAE
    drop(model);
    println!("  DiT weights freed");

    // Load VAE with Z-Image scaling (scale=0.3611, shift=0.1159)
    use eridiffusion::models::flux_vae::AutoencoderKL;
    let vae = AutoencoderKL::from_safetensors(VAE_PATH, 0.3611, 0.1159)?;
    println!("  VAE loaded");

    let rgb = vae.decode(&denoised)?;
    println!("  Decoded: {:?} in {:.1}s", rgb.dims(), t0.elapsed().as_secs_f32());

    // ------------------------------------------------------------------
    // Stage 6: Save PNG
    // ------------------------------------------------------------------
    println!("\n--- Stage 6: Save PNG ---");
    let rgb_f32 = rgb.to_dtype(DType::F32)?;
    let data = rgb_f32.to_vec()?;
    let (_, _, out_h, out_w) = {
        let d = rgb_f32.dims();
        (d[0], d[1], d[2], d[3])
    };

    // CHW → HWC, [0,1] → [0,255]
    let mut pixels = vec![0u8; out_h * out_w * 3];
    for y in 0..out_h {
        for x_pos in 0..out_w {
            for c in 0..3 {
                let idx = c * out_h * out_w + y * out_w + x_pos;
                let v = data[idx].clamp(0.0, 1.0);
                pixels[(y * out_w + x_pos) * 3 + c] = (v * 255.0) as u8;
            }
        }
    }

    let output_path = OUTPUT_LATENTS.replace("_latents.safetensors", ".png");
    let img = image::RgbImage::from_raw(out_w as u32, out_h as u32, pixels)
        .ok_or_else(|| anyhow::anyhow!("Failed to create image"))?;
    img.save(&output_path)?;

    let dt_total = t_total.elapsed().as_secs_f32();
    println!("\n============================================================");
    println!("IMAGE SAVED: {}", output_path);
    println!("Total time: {:.1}s", dt_total);
    println!("============================================================");

    Ok(())
}
