//! Test the memory-efficient Flux loader

use anyhow::Result;
use candle_core::{Device, DType};
use eridiffusion::trainers::flux_efficient_loader::{FluxEfficientLoader, create_flux_for_24gb_training};
use eridiffusion::models::flux_custom::lora::LoRAConfig;
use std::path::Path;

fn main() -> Result<()> {
    println!("Testing Memory-Efficient Flux Loader");
    println!("====================================\n");
    
    // Setup
    let device = Device::cuda_if_available(0)?;
    let model_path = Path::new("/home/alex/SwarmUI/Models/diffusion_models/flux1-dev.safetensors");
    
    // Check if model exists
    if !model_path.exists() {
        eprintln!("Model not found at: {:?}", model_path);
        eprintln!("Please set the correct path to your Flux model");
        return Err(anyhow::anyhow!("Model file not found"));
    }
    
    // LoRA configuration
    let lora_config = LoRAConfig {
        rank: 16,
        alpha: 16.0,
        dropout: Some(0.0),
        target_modules: vec!["attn".to_string(), "mlp".to_string()],
        module_filters: vec![],
        init_scale: 0.01,
    };
    
    println!("Model path: {:?}", model_path);
    println!("Device: {:?}", device);
    println!("LoRA rank: {}", lora_config.rank);
    println!("LoRA alpha: {}", lora_config.alpha);
    
    // Test 1: Create loader
    println!("\n--- Test 1: Creating Efficient Loader ---");
    let loader = FluxEfficientLoader::new(model_path, device.clone())?;
    println!("✅ Loader created successfully");
    
    // Test 2: Load model with LoRA
    println!("\n--- Test 2: Loading Model for Training ---");
    let start = std::time::Instant::now();
    
    let model = create_flux_for_24gb_training(
        model_path,
        &lora_config,
        device.clone(),
    )?;
    
    let elapsed = start.elapsed();
    println!("✅ Model loaded in {:.2} seconds", elapsed.as_secs_f32());
    
    // Test 3: Get trainable parameters
    println!("\n--- Test 3: Checking Trainable Parameters ---");
    let trainable_params = model.get_trainable_params();
    println!("Number of trainable parameters: {}", trainable_params.len());
    
    // Calculate memory usage
    let param_count: usize = trainable_params.iter()
        .map(|p| p.as_tensor().elem_count())
        .sum();
    
    let memory_mb = (param_count * 2) as f32 / (1024.0 * 1024.0); // FP16 = 2 bytes per param
    println!("LoRA parameter memory: {:.1} MB", memory_mb);
    
    // Test 4: Forward pass with dummy data
    println!("\n--- Test 4: Testing Forward Pass ---");
    use candle_core::Tensor;
    
    let batch_size = 1;
    let seq_len = 1024; // 32x32 patches for 1024x1024 image
    let hidden_size = 3072;
    
    // Create dummy inputs
    let img = Tensor::randn(0f32, 1.0, (batch_size, seq_len, 64), &device)?;
    let img_ids = Tensor::zeros((batch_size, 32, 32, 3), DType::F32, &device)?;
    let txt = Tensor::randn(0f32, 1.0, (batch_size, 256, 4096), &device)?;
    let txt_ids = Tensor::zeros((batch_size, 256, 3), DType::F32, &device)?;
    let timesteps = Tensor::new(&[500.0f32], &device)?;
    let y = Tensor::randn(0f32, 1.0, (batch_size, 768), &device)?;
    let guidance = Some(Tensor::new(&[3.5f32], &device)?);
    
    println!("Running forward pass...");
    let output = model.forward(&img, &img_ids, &txt, &txt_ids, &timesteps, &y, guidance.as_ref())?;
    println!("✅ Forward pass successful!");
    println!("Output shape: {:?}", output.shape());
    
    // Print memory usage
    println!("\n--- Memory Usage Summary ---");
    println!("Base model precision: FP16 (~11GB)");
    println!("LoRA parameters: {:.1} MB", memory_mb);
    println!("Estimated total during training: ~12-14GB");
    println!("Available for gradients/activations: ~10-12GB");
    
    println!("\n✅ All tests passed!");
    
    Ok(())
}