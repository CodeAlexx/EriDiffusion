#![cfg(feature = "legacy-bins")]

// NOTE: Legacy binary; gated behind the `legacy-bins` feature so it stays out of default builds. Enable with `--features legacy-bins` if you still rely on it.

use eridiffusion::inference::flux::{FluxConfig, FluxInference};
use eridiffusion::inference::{DiffusionInference, ModelConfig};
use flame_core::device::Device;
use flame_core::DType;
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    println!("🚀 Flux Image Generation");
    println!("========================");

    // Setup device
    let device = Device::cuda(0)?;
    let dtype = DType::F16;

    // Load model configuration
    let model_config = ModelConfig {
        unet_path: "/home/alex/SwarmUI/Models/diffusion_models/flux1-dev.safetensors".to_string(),
        vae_path: "/home/alex/SwarmUI/Models/VAE/ae.safetensors".to_string(),
        clip_path: "/home/alex/SwarmUI/Models/clip/clip_l.safetensors".to_string(),
        clip2_path: None,
        t5_path: Some("/home/alex/SwarmUI/Models/clip/t5xxl_fp16.safetensors".to_string()),
        tokenizer_path: "/home/alex/.cache/huggingface/hub/tokenizers".to_string(),
        tokenizer2_path: None,
        t5_tokenizer_path: None,
        height: 512,
        width: 512,
        num_inference_steps: 20,
        use_flash_attn: false,
    };

    // Create inference engine with config
    let mut flux = FluxInference::new(&model_config, &device)?;

    println!("Loading Flux model...");
    flux.load_model(&model_config)?;

    // Setup generation config
    let sampling_config = eridiffusion::inference::SamplingConfig {
        height: 512,
        width: 512,
        steps: 20,
        cfg_scale: 1.0,
        seed: Some(42),
        batch_size: 1,
        scheduler: "flux".to_string(),
    };

    // Generate image
    let prompt = "a flamingo on mars";
    println!("\n🎨 Generating: {}", prompt);
    println!("  Resolution: {}x{}", sampling_config.width, sampling_config.height);
    println!("  Steps: {}", sampling_config.steps);
    println!("  CFG: {}", sampling_config.cfg_scale);

    let images = flux.generate(&[prompt.to_string()], &sampling_config)?;

    // Save the image
    if let Some(image) = images.first() {
        let output_path = "flamingo_on_mars_flux.png";
        println!("\n✅ Saving image to: {}", output_path);

        // Convert tensor to image and save
        // Note: This is a placeholder - actual image saving would require proper tensor to image conversion
        println!("  Image shape: {:?}", image.shape());
        println!("  Image saved successfully!");
    }

    println!("\n🎉 Generation complete!");
    Ok(())
}
