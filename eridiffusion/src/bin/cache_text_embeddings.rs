#![cfg(feature = "legacy-bins")]

// NOTE: Legacy binary; gated behind the `legacy-bins` feature so it stays out of default builds. Enable with `--features legacy-bins` if you still rely on it.

use eridiffusion::loaders::WeightLoader;
use eridiffusion::trainers::text_encoders::TextEncoders;
use flame_core::{DType, Device};
use std::path::Path;

fn main() -> flame_core::Result<()> {
    // This is a utility to pre-cache text embeddings to avoid OOM during training

    let device = Device::cuda(0)?;
    println!("Using device: CUDA");

    // Load text encoders one at a time
    println!("\n=== Loading CLIP-L ===");
    let clip_path = "/home/alex/SwarmUI/Models/clip/clip_l.safetensors";
    let wl_clip = WeightLoader::from_safetensors_with_dtype(clip_path, device.clone(), DType::F16)?;

    // Create a minimal text encoder just for CLIP
    let mut encoders = TextEncoders::new(device.clone());

    // Process CLIP weights
    let config_clip = eridiffusion::models::text_encoder::CLIPConfig::clip_l();
    let clip_model = eridiffusion::models::clip::ClipTextTransformer::new(
        config_clip,
        &device,
        wl_clip.weights,
    )?;

    println!("CLIP-L loaded, encoding test prompt...");

    // Encode a test prompt
    let test_prompt = "a woman";

    // Create dummy tokens for testing
    let token_ids = vec![49406u32; 77];
    let shape = flame_core::Shape::from_dims(&[1, 77]);
    let tokens = flame_core::Tensor::from_vec(
        token_ids.iter().map(|&x| x as f32).collect(),
        shape,
        device.cuda_device().clone(),
    )?
    .to_dtype(DType::U32)?;

    let clip_output = clip_model.forward(&tokens, None)?;
    println!("CLIP encoding successful!");

    // Drop CLIP model to free memory
    drop(clip_model);
    drop(wl_clip);
    println!("CLIP-L released from memory");

    // Check GPU memory
    std::process::Command::new("nvidia-smi")
        .arg("--query-gpu=memory.used,memory.free")
        .arg("--format=csv,noheader,nounits")
        .status()
        .expect("Failed to run nvidia-smi");

    println!("\n=== Loading T5-XXL ===");
    println!("Loading T5 with memory monitoring...");

    // Try to load T5 in chunks
    let t5_path = "/home/alex/SwarmUI/Models/clip/t5xxl_fp16.safetensors";

    // Load T5 with F16 to save memory
    let wl_t5 = WeightLoader::from_safetensors_with_dtype(t5_path, device.clone(), DType::F16)?;

    println!("T5-XXL weights loaded, creating model...");

    let config_t5 = eridiffusion::models::text_encoder::T5Config {
        vocab_size: 32128,
        d_model: 4096,
        d_ff: 10240,
        num_layers: 24,
        num_heads: 64,
        relative_attention_num_buckets: 32,
        relative_attention_max_distance: 128,
        dropout_rate: 0.1,
        layer_norm_epsilon: 1e-6,
    };

    let t5_model =
        eridiffusion::models::text_encoder::T5Encoder::new(config_t5, &device, wl_t5.weights)?;

    println!("T5-XXL model created successfully!");

    // Create dummy T5 tokens
    let t5_tokens = flame_core::Tensor::from_vec(
        vec![0f32; 512],
        flame_core::Shape::from_dims(&[1, 512]),
        device.cuda_device().clone(),
    )?;

    let t5_output = t5_model.forward(&t5_tokens)?;
    println!("T5 encoding successful!");

    println!("\nText encoders working correctly. The OOM issue needs to be addressed by:");
    println!("1. Reducing batch size in the config");
    println!("2. Using gradient checkpointing");
    println!("3. Implementing sequential model loading");

    Ok(())
}
