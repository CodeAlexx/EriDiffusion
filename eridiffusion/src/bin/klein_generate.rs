//! Klein 4B image generation — pure Rust, no Python.
//!
//! Usage: cargo run --release --bin klein_generate
//!
//! Pipeline:
//! 1. Load cached text embeddings (safetensors)
//! 2. Load Klein 4B model weights (safetensors)
//! 3. Create noise, run Euler denoise loop
//! 4. VAE decode → RGB → PNG

use eridiffusion::models::klein_model::KleinTransformer;
use eridiffusion::models::klein_vae::{KleinVaeDecoder, unpatchify_latents};
use eridiffusion::inference::klein_sampling::{build_sigma_schedule, euler_denoise};
use flame_core::{global_cuda_device, DType, Shape, Tensor};
use std::time::Instant;

const MODEL_PATH: &str = "/home/alex/.serenity/models/checkpoints/flux-2-klein-base-4b.safetensors";
const EMBEDDINGS_PATH: &str = "/home/alex/EriDiffusion/flame-core/inference-test/cached_embeddings.safetensors";
const VAE_PATH: &str = "/home/alex/.serenity/models/vaes/flux2-vae.safetensors";
const OUTPUT_PATH: &str = "/home/alex/serenity/output/klein4b_rust.png";

const NUM_STEPS: usize = 35;
const SHIFT: f32 = 2.02;
const CFG_SCALE: f32 = 3.5;
const SEED: u64 = 42;
const WIDTH: usize = 512;
const HEIGHT: usize = 512;

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let t_total = Instant::now();

    println!("============================================================");
    println!("Klein 4B — Pure Rust Inference");
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
    let img_ids = cached.get("img_ids")
        .ok_or_else(|| anyhow::anyhow!("Missing img_ids"))?;
    let txt_ids = cached.get("txt_ids")
        .ok_or_else(|| anyhow::anyhow!("Missing txt_ids"))?;
    println!("  pos_hidden: {:?} {:?}", pos_hidden.dims(), pos_hidden.dtype());
    println!("  Loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // ------------------------------------------------------------------
    // Stage 2: Load Klein 4B model
    // ------------------------------------------------------------------
    println!("\n--- Stage 2: Load Klein 4B model ---");
    let t0 = Instant::now();
    let model = KleinTransformer::from_safetensors(MODEL_PATH)?;
    println!("  Config: {:?}", model.config());
    println!("  Loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // ------------------------------------------------------------------
    // Stage 3: Create noise + sigma schedule
    // ------------------------------------------------------------------
    println!("\n--- Stage 3: Prepare noise + sigmas ---");
    let latent_h = HEIGHT / 16;
    let latent_w = WIDTH / 16;

    // Seeded noise
    // Simple seeded normal noise using Box-Muller
    let numel = 1 * 128 * latent_h * latent_w;
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

    // Upload as BF16: [1, 128, latent_h, latent_w]
    let noise_spatial = Tensor::from_f32_to_bf16(
        noise_data,
        Shape::from_dims(&[1, 128, latent_h, latent_w]),
        device.clone(),
    )?;

    // Pack: [1, 128, 32, 32] → [1, 32, 32, 128] → [1, 1024, 128]
    let noise = noise_spatial
        .permute(&[0, 2, 3, 1])?
        .reshape(&[1, latent_h * latent_w, 128])?;
    println!("  Noise: {:?} (packed)", noise.dims());

    let sigmas = build_sigma_schedule(NUM_STEPS, SHIFT);
    println!("  Sigmas: {} values, max={:.4}, min={:.4}",
             sigmas.len(), sigmas[0], sigmas[sigmas.len()-2]);

    // ------------------------------------------------------------------
    // Stage 4: Denoise
    // ------------------------------------------------------------------
    println!("\n--- Stage 4: Denoise ({} steps, Euler, CFG={}) ---", NUM_STEPS, CFG_SCALE);
    let t0 = Instant::now();

    let denoised = euler_denoise(
        |x, sigma| {
            // Create sigma tensor
            let sigma_t = Tensor::from_f32_to_bf16(
                vec![sigma],
                Shape::from_dims(&[1]),
                device.clone(),
            )?;

            // Conditional (positive)
            let cond = model.forward(x, pos_hidden, &sigma_t, img_ids, txt_ids)?;
            // Unconditional (negative)
            let uncond = model.forward(x, neg_hidden, &sigma_t, img_ids, txt_ids)?;

            // CFG: uncond + scale * (cond - uncond)
            let diff = cond.sub(&uncond)?;
            let guided = uncond.add(&diff.mul_scalar(CFG_SCALE)?)?;

            // Flow matching: denoised = x - guided * sigma
            x.sub(&guided.mul_scalar(sigma)?)
        },
        noise,
        &sigmas,
    )?;

    let dt = t0.elapsed().as_secs_f32();
    println!("  Denoised in {:.1}s ({:.2}s/step)", dt, dt / NUM_STEPS as f32);
    println!("  Output: {:?}", denoised.dims());

    // ------------------------------------------------------------------
    // Stage 5: Unpack + VAE decode
    // ------------------------------------------------------------------
    println!("\n--- Stage 5: VAE Decode ---");
    let t0 = Instant::now();

    // Free DiT weights to make room for VAE
    drop(model);
    println!("  DiT weights freed");

    // Unpack: [1, 1024, 128] → [1, 32, 32, 128] → [1, 128, 32, 32]
    let latents = denoised
        .reshape(&[1, latent_h, latent_w, 128])?
        .permute(&[0, 3, 1, 2])?;

    // Unpatchify: [1, 128, 32, 32] → [1, 32, 64, 64]
    let latents_32ch = unpatchify_latents(&latents)?;
    println!("  Unpatchified: {:?}", latents_32ch.dims());

    // Load VAE
    println!("  Loading VAE weights...");
    let vae_weights = flame_core::serialization::load_file(
        std::path::Path::new(VAE_PATH), &device
    )?;
    println!("  VAE weights loaded: {} keys", vae_weights.len());
    let vae_device = flame_core::device::Device::from_arc(device.clone());
    println!("  Building VAE decoder...");
    let vae = KleinVaeDecoder::load(&vae_weights, &vae_device)?;
    println!("  VAE decoder built");

    // Decode
    println!("  Decoding...");
    let rgb = vae.decode(&latents_32ch)?;
    println!("  Decoded: {:?}", rgb.dims());
    println!("  VAE in {:.1}s", t0.elapsed().as_secs_f32());

    // ------------------------------------------------------------------
    // Stage 6: Save PNG
    // ------------------------------------------------------------------
    println!("\n--- Stage 6: Save PNG ---");

    // rgb: [1, 3, H, W] in [-1, 1] → clamp → [0, 255] uint8
    let rgb_f32 = rgb.to_dtype(DType::F32)?;
    let data = rgb_f32.to_vec()?; // flat f32
    let (_, _, out_h, out_w) = {
        let d = rgb_f32.dims();
        (d[0], d[1], d[2], d[3])
    };

    // Convert to RGB u8
    let mut pixels = vec![0u8; out_h * out_w * 3];
    for y in 0..out_h {
        for x in 0..out_w {
            for c in 0..3 {
                let idx = c * out_h * out_w + y * out_w + x; // NCHW layout
                let val = (data[idx].clamp(-1.0, 1.0) + 1.0) / 2.0 * 255.0;
                pixels[(y * out_w + x) * 3 + c] = val as u8;
            }
        }
    }

    let img = image::RgbImage::from_raw(out_w as u32, out_h as u32, pixels)
        .ok_or_else(|| anyhow::anyhow!("Failed to create image"))?;
    img.save(OUTPUT_PATH)?;

    let dt_total = t_total.elapsed().as_secs_f32();
    println!("\n============================================================");
    println!("IMAGE SAVED: {}", OUTPUT_PATH);
    println!("Total time: {:.1}s", dt_total);
    println!("============================================================");

    Ok(())
}
