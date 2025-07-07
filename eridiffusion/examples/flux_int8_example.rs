use anyhow::Result;
use candle_core::{Device, DType};
use eridiffusion::trainers::flux_int8_loader::load_flux_int8;

/// Example showing how to use INT8 quantization with Flux models
/// This reduces memory usage from ~22GB to ~11GB
fn main() -> Result<()> {
    // Initialize device
    let device = Device::cuda_if_available(0)?;
    println!("Using device: {:?}", device);
    
    // Example model path
    let model_path = "/home/alex/SwarmUI/Models/diffusion_models/flux1-dev.safetensors";
    
    println!("\n=== Loading Flux with INT8 Quantization ===");
    println!("This will reduce memory usage by ~50%");
    
    // Load and quantize the model
    let int8_model = load_flux_int8(model_path, device)?;
    
    // Show memory statistics
    let (original_bytes, quantized_bytes) = int8_model.memory_stats()?;
    println!("\nMemory Comparison:");
    println!("  Original (FP16/BF16): {:.2} GB", original_bytes as f64 / 1e9);
    println!("  Quantized (INT8): {:.2} GB", quantized_bytes as f64 / 1e9);
    println!("  Savings: {:.2} GB ({:.1}% reduction)", 
        (original_bytes - quantized_bytes) as f64 / 1e9,
        (1.0 - quantized_bytes as f64 / original_bytes as f64) * 100.0
    );
    
    // Example: Using the quantized model in training/inference
    println!("\n=== Using Quantized Weights ===");
    
    // Get a specific weight (automatically dequantized)
    if int8_model.has_weight("time_in.in_layer.weight") {
        let weight = int8_model.get_weight("time_in.in_layer.weight")?;
        println!("Loaded weight 'time_in.in_layer.weight':");
        println!("  Shape: {:?}", weight.dims());
        println!("  Dtype: {:?} (dequantized from INT8)", weight.dtype());
    }
    
    // Example: Get multiple weights for a layer
    let layer_weights = int8_model.get_weights(&[
        "double_blocks.0.img_attn.qkv.weight",
        "double_blocks.0.img_attn.qkv.bias",
    ])?;
    
    println!("\nLoaded {} weights for attention layer", layer_weights.len());
    for (name, tensor) in &layer_weights {
        println!("  {}: shape {:?}", name, tensor.dims());
    }
    
    // Integration with LoRA training
    println!("\n=== Integration with LoRA Training ===");
    println!("When using INT8 quantization with LoRA:");
    println!("  - Base model weights are stored as INT8 (~11GB)");
    println!("  - LoRA adapters remain in FP16/BF16 (~100MB for rank 16)");
    println!("  - Weights are dequantized on-the-fly during forward pass");
    println!("  - Total memory usage: ~11.1GB instead of ~22.1GB");
    
    println!("\n=== Performance Considerations ===");
    println!("INT8 quantization trades memory for compute:");
    println!("  - 50% memory reduction");
    println!("  - ~10-20% slower due to dequantization");
    println!("  - Minimal quality loss (< 0.1% in most cases)");
    println!("  - Ideal for 24GB GPUs training large models");
    
    Ok(())
}