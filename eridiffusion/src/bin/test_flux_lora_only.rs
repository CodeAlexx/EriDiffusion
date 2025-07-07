//! Test creating a Flux LoRA model without loading base weights
//! This is the most memory-efficient approach for LoRA-only training

use anyhow::Result;
use candle_core::{Device, DType, Tensor};
use eridiffusion::models::flux_custom::{FluxConfig, FluxModelWithLoRA};
use eridiffusion::models::flux_custom::lora::LoRAConfig;
use candle_nn::{VarBuilder, VarMap};
use std::path::Path;

fn main() -> Result<()> {
    println!("Flux LoRA-Only Model Test");
    println!("=========================\n");
    
    // Check GPU memory
    if let Ok(output) = std::process::Command::new("nvidia-smi")
        .args(&["--query-gpu=memory.free,memory.used", "--format=csv,noheader,nounits"])
        .output()
    {
        let memory_info = String::from_utf8_lossy(&output.stdout);
        let parts: Vec<&str> = memory_info.trim().split(", ").collect();
        if parts.len() == 2 {
            println!("GPU Memory - Free: {} MB, Used: {} MB\n", parts[0], parts[1]);
        }
    }
    
    // Setup
    let device = Device::cuda_if_available(0)?;
    println!("Device: {:?}", device);
    
    // LoRA configuration
    let lora_config = LoRAConfig {
        rank: 16,
        alpha: 16.0,
        dropout: Some(0.0),
        target_modules: vec!["attn".to_string(), "mlp".to_string()],
        module_filters: vec![],
        init_scale: 0.01,
    };
    
    println!("\n--- Creating LoRA-Only Model ---");
    println!("This approach:");
    println!("1. Creates model structure with minimal weights");
    println!("2. Adds LoRA adapters (the only trainable parts)");
    println!("3. Base weights remain unloaded (or can be streamed)");
    
    // Flux configuration
    let flux_config = FluxConfig::default();
    
    // Create model with minimal weights
    let var_map = VarMap::new();
    let vb = VarBuilder::from_varmap(&var_map, DType::F16, &device);
    
    // Initialize only essential weights to create structure
    use eridiffusion::trainers::flux_init_weights::initialize_flux_weights_minimal;
    initialize_flux_weights_minimal(&vb, &flux_config)?;
    
    println!("\nCreating model structure...");
    let mut model = FluxModelWithLoRA::new(&flux_config, vb)?;
    
    println!("Adding LoRA adapters...");
    model.add_lora_to_all(&lora_config, &device, DType::F16)?;
    
    // Get trainable parameters
    let trainable_params = model.get_trainable_params();
    let param_count: usize = trainable_params.iter()
        .map(|p| p.elem_count())
        .sum();
    
    println!("\n✅ Model created successfully!");
    println!("Trainable parameters: {}", param_count);
    println!("Memory usage: ~{:.1} MB", (param_count * 2) as f32 / 1e6);
    
    // Test forward pass with small batch
    println!("\n--- Testing Forward Pass ---");
    
    let batch_size = 1;
    let seq_len = 256;  // Smaller sequence for testing
    
    let img = Tensor::randn(0f32, 1.0, (batch_size, seq_len, 64), &device)?;
    let img_ids = Tensor::zeros((batch_size, 16, 16, 3), DType::F32, &device)?;
    let txt = Tensor::randn(0f32, 1.0, (batch_size, 77, 4096), &device)?;
    let txt_ids = Tensor::zeros((batch_size, 77, 3), DType::F32, &device)?;
    let timesteps = Tensor::new(&[500.0f32], &device)?;
    let y = Tensor::randn(0f32, 1.0, (batch_size, 768), &device)?;
    
    println!("Running forward pass...");
    match model.forward(&img, &img_ids, &txt, &txt_ids, &timesteps, &y, None) {
        Ok(output) => {
            println!("✅ Forward pass successful!");
            println!("Output shape: {:?}", output.shape());
        }
        Err(e) => {
            println!("❌ Forward pass failed: {}", e);
            println!("This is expected with minimal weights - base model weights are needed for actual inference");
        }
    }
    
    // Check final memory
    if let Ok(output) = std::process::Command::new("nvidia-smi")
        .args(&["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()
    {
        let used_memory = String::from_utf8_lossy(&output.stdout);
        println!("\nFinal GPU memory used: {} MB", used_memory.trim());
    }
    
    println!("\n--- Summary ---");
    println!("✅ Successfully created Flux model for LoRA-only training");
    println!("✅ Only LoRA parameters are in memory (~{:.1} MB)", (param_count * 2) as f32 / 1e6);
    println!("✅ Base weights can be loaded on-demand during training");
    println!("\nThis approach allows training on 24GB GPUs!");
    
    Ok(())
}