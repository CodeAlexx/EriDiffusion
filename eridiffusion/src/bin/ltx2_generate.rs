//! LTX-2.3 video generation — pure Rust, no Python.
//!
//! Pipeline:
//! 1. Load cached Gemma text embeddings (safetensors)
//! 2. Load LTX-2 transformer (video-only keys, ~16.5B params)
//! 3. Create noise [B, 128, F_lat, H_lat, W_lat]
//! 4. Run Euler denoise loop (8 steps distilled)
//! 5. Save denoised latents as safetensors
//! 6. Decode with Python VAE (separate step)

use eridiffusion::models::ltx2_model::{LTX2Config, LTX2StreamingModel};
use eridiffusion::inference::ltx2_sampling::{euler_denoise_ltx2, LTX2_DISTILLED_SIGMAS};
use flame_core::{global_cuda_device, DType, Shape, Tensor};
use std::time::Instant;

const MODEL_PATH: &str =
    "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-dev.safetensors";
const EMBEDDINGS_PATH: &str =
    "/home/alex/EriDiffusion/eridiffusion/eridiffusion/cached_ltx2_embeddings.safetensors";
const OUTPUT_PATH: &str =
    "/home/alex/serenity/output/ltx2_denoised_latents.safetensors";

const NUM_FRAMES: usize = 17;  // 8n+1 where n=2 (minimum)
const WIDTH: usize = 512;
const HEIGHT: usize = 512;
const SEED: u64 = 42;
const FRAME_RATE: f32 = 25.0;

// Latent space dimensions (VAE compression: T/8, H/32, W/32)
const LATENT_CHANNELS: usize = 128;

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let t_total = Instant::now();

    println!("============================================================");
    println!("LTX-2.3 Video Generation — Pure Rust Inference");
    println!("============================================================");
    println!("  {}×{}, {} frames @ {} fps", WIDTH, HEIGHT, NUM_FRAMES, FRAME_RATE);

    let device = global_cuda_device();

    // Compute latent dimensions
    let latent_f = ((NUM_FRAMES - 1) / 8) + 1;  // temporal: (17-1)/8+1 = 3
    let latent_h = HEIGHT / 32;                   // spatial: 512/32 = 16
    let latent_w = WIDTH / 32;                    // spatial: 512/32 = 16
    let num_tokens = latent_f * latent_h * latent_w;
    println!("  Latent: [{}, {}, {}, {}] = {} tokens",
             LATENT_CHANNELS, latent_f, latent_h, latent_w, num_tokens);

    // ------------------------------------------------------------------
    // Stage 1: Load cached embeddings
    // ------------------------------------------------------------------
    println!("\n--- Stage 1: Load cached embeddings ---");
    let t0 = Instant::now();
    let cached = flame_core::serialization::load_file(
        std::path::Path::new(EMBEDDINGS_PATH),
        &device,
    )?;
    let text_hidden = cached.get("text_hidden")
        .ok_or_else(|| anyhow::anyhow!("Missing text_hidden in embeddings"))?;
    println!("  text_hidden: {:?} {:?}", text_hidden.dims(), text_hidden.dtype());
    println!("  Loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // ------------------------------------------------------------------
    // Stage 2: Load LTX-2 transformer (video-only)
    // ------------------------------------------------------------------
    println!("\n--- Stage 2: Load LTX-2 transformer (video-only) ---");
    let t0 = Instant::now();

    let config = LTX2Config::default();
    println!("  Config: {}×{} dim, {} heads, {} layers",
             config.num_attention_heads, config.attention_head_dim,
             config.num_attention_heads, config.num_layers);

    // Load only global params to GPU (~400MB). Blocks stream from disk.
    let model = LTX2StreamingModel::load_globals(MODEL_PATH, &config)?;
    println!("  Global params loaded in {:.1}s (blocks stream on demand)", t0.elapsed().as_secs_f32());

    // ------------------------------------------------------------------
    // Stage 3: Create noise + sigma schedule
    // ------------------------------------------------------------------
    println!("\n--- Stage 3: Prepare noise + sigmas ---");

    let numel = 1 * LATENT_CHANNELS * latent_f * latent_h * latent_w;
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
        Shape::from_dims(&[1, LATENT_CHANNELS, latent_f, latent_h, latent_w]),
        device.clone(),
    )?;
    println!("  Noise: {:?}", noise.dims());

    let sigmas = LTX2_DISTILLED_SIGMAS.to_vec();
    println!("  Sigmas: {} steps (distilled)", sigmas.len() - 1);

    // ------------------------------------------------------------------
    // Stage 4: Denoise
    // ------------------------------------------------------------------
    println!("\n--- Stage 4: Denoise ({} steps, Euler, distilled) ---", sigmas.len() - 1);
    let t0 = Instant::now();

    let denoised = euler_denoise_ltx2(
        |x, sigma| {
            let sigma_t = Tensor::from_f32_to_bf16(
                vec![sigma],
                Shape::from_dims(&[1]),
                device.clone(),
            )?;
            model.forward_video_only(x, &sigma_t, text_hidden, FRAME_RATE)
        },
        noise,
        &sigmas,
    )?;

    let dt = t0.elapsed().as_secs_f32();
    println!("  Denoised in {:.1}s ({:.2}s/step)", dt, dt / (sigmas.len() - 1) as f32);
    println!("  Output: {:?}", denoised.dims());

    // ------------------------------------------------------------------
    // Stage 5: Save latents
    // ------------------------------------------------------------------
    println!("\n--- Stage 5: Save latents ---");
    let mut save_map = std::collections::HashMap::new();
    save_map.insert("latents".to_string(), denoised);
    flame_core::serialization::save_tensors(
        &save_map,
        std::path::Path::new(OUTPUT_PATH),
        flame_core::serialization::SerializationFormat::SafeTensors,
    )?;
    println!("  Saved to {}", OUTPUT_PATH);

    let dt_total = t_total.elapsed().as_secs_f32();
    println!("\n============================================================");
    println!("LATENTS SAVED: {}", OUTPUT_PATH);
    println!("Total time: {:.1}s", dt_total);
    println!("============================================================");

    Ok(())
}
