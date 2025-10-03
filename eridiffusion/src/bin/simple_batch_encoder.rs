#![cfg(feature = "legacy-bins")]

// NOTE: Legacy binary; gated behind the `legacy-bins` feature so it stays out of default builds. Enable with `--features legacy-bins` if you still rely on it.

use flame_core::device::Device;
use std::path::PathBuf;

fn main() -> flame_core::Result<()> {
    println!("=== Simple Batch Text Embedding Encoder ===");

    // Initialize device
    let device = Device::cuda(0)?;

    // Check initial memory
    println!("\nInitial GPU Memory:");
    std::process::Command::new("nvidia-smi")
        .arg("--query-gpu=memory.used,memory.free,memory.total")
        .arg("--format=csv,noheader,nounits")
        .status()
        .expect("Failed to run nvidia-smi");

    // Test captions
    let test_captions = vec![
        "a beautiful woman with long hair",
        "a woman in a white dress",
        "portrait of a woman smiling",
    ];

    println!("\n=== Phase 1: Loading and encoding with CLIP only ===");
    {
        let clip_path = PathBuf::from("/home/alex/SwarmUI/Models/clip/clip_l.safetensors");

        println!("Loading CLIP-L from: {}", clip_path.display());

        // Use the simplified approach from the training pipeline
        let clip_config = eridiffusion::models::text_encoder::CLIPConfig::clip_l();
        let wl = eridiffusion::loaders::WeightLoader::from_safetensors_with_dtype(
            &clip_path,
            device.clone(),
            flame_core::DType::F16,
        )?;

        let clip_model =
            eridiffusion::models::clip::ClipTextTransformer::new(clip_config, &device, wl.weights)?;

        println!("✅ CLIP-L loaded successfully");

        // Check memory
        println!("\nGPU Memory after loading CLIP:");
        std::process::Command::new("nvidia-smi")
            .arg("--query-gpu=memory.used,memory.free,memory.total")
            .arg("--format=csv,noheader,nounits")
            .status()
            .expect("Failed to run nvidia-smi");

        // Process test captions with simple tokenization
        for (idx, caption) in test_captions.iter().enumerate() {
            println!("\n[CLIP {}/{}] Processing: \"{}\"", idx + 1, test_captions.len(), caption);

            // Simple tokenization - create dummy tokens for testing
            let token_ids = vec![49406u32; 77]; // Start token repeated
            let shape = flame_core::Shape::from_dims(&[1, 77]);
            let tokens = flame_core::Tensor::from_vec(
                token_ids.iter().map(|&x| x as f32).collect(),
                shape,
                device.cuda_device().clone(),
            )?
            .to_dtype(flame_core::DType::U32)?;

            let output = clip_model.forward(&tokens, None)?;
            println!("  ✅ CLIP encoding shape: {:?}", output.last_hidden_state.shape());
        }

        // Explicitly drop CLIP model
        drop(clip_model);
        // wl is already consumed by clip_model, no need to drop
        println!("\n✅ CLIP model freed from memory");
    }

    // Check memory after freeing CLIP
    println!("\nGPU Memory after freeing CLIP:");
    std::process::Command::new("nvidia-smi")
        .arg("--query-gpu=memory.used,memory.free,memory.total")
        .arg("--format=csv,noheader,nounits")
        .status()
        .expect("Failed to run nvidia-smi");

    println!("\n=== Phase 2: Loading and encoding with T5 only ===");
    {
        let t5_path = PathBuf::from("/home/alex/SwarmUI/Models/clip/t5xxl_fp16.safetensors");

        println!("Loading T5-XXL from: {}", t5_path.display());

        // Use streaming loader for T5
        let wl = eridiffusion::loaders::WeightLoader::from_safetensors_streaming(
            &t5_path,
            device.clone(),
            flame_core::DType::F16,
        )?;

        let t5_config = eridiffusion::models::text_encoder::T5Config {
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
            eridiffusion::models::text_encoder::T5Encoder::new(t5_config, &device, wl.weights)?;

        println!("✅ T5-XXL loaded successfully");

        // Check memory
        println!("\nGPU Memory after loading T5:");
        std::process::Command::new("nvidia-smi")
            .arg("--query-gpu=memory.used,memory.free,memory.total")
            .arg("--format=csv,noheader,nounits")
            .status()
            .expect("Failed to run nvidia-smi");

        // Process test captions
        for (idx, caption) in test_captions.iter().enumerate() {
            println!("\n[T5 {}/{}] Processing: \"{}\"", idx + 1, test_captions.len(), caption);

            // Simple tokenization for T5 - use shorter sequence
            let token_ids: Vec<f32> = vec![0.0; 256]; // Dummy tokens
            let shape = flame_core::Shape::from_dims(&[1, 256]);
            let tokens =
                flame_core::Tensor::from_vec(token_ids, shape, device.cuda_device().clone())?;

            let output = t5_model.forward(&tokens)?;
            println!("  ✅ T5 encoding shape: {:?}", output.last_hidden_state.shape());
        }

        drop(t5_model);
        // wl is already consumed by t5_model, no need to drop
        println!("\n✅ T5 model freed from memory");
    }

    // Final memory check
    println!("\nFinal GPU Memory:");
    std::process::Command::new("nvidia-smi")
        .arg("--query-gpu=memory.used,memory.free,memory.total")
        .arg("--format=csv,noheader,nounits")
        .status()
        .expect("Failed to run nvidia-smi");

    println!("\n✅ Sequential loading test complete!");
    println!("   This demonstrates that we can load CLIP and T5 separately");
    println!("   without running out of memory.");

    Ok(())
}
