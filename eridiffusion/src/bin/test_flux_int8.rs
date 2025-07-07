use anyhow::Result;
use candle_core::{Device, DType};
use eridiffusion::trainers::flux_int8_loader::load_flux_int8;

fn main() -> Result<()> {
    // Initialize CUDA device
    let device = Device::cuda_if_available(0)?;
    println!("Using device: {:?}", device);
    
    // Example: Load and quantize Flux model
    let model_path = "/home/alex/SwarmUI/Models/diffusion_models/flux1-dev.safetensors";
    
    println!("Loading Flux model and quantizing to INT8...");
    let int8_model = load_flux_int8(model_path, device)?;
    
    // Show memory statistics
    let (original_bytes, quantized_bytes) = int8_model.memory_stats()?;
    println!("\nMemory usage comparison:");
    println!("  Original FP16/BF16: {:.2} GB", original_bytes as f64 / 1e9);
    println!("  Quantized INT8: {:.2} GB", quantized_bytes as f64 / 1e9);
    println!("  Reduction: {:.1}%", (1.0 - quantized_bytes as f64 / original_bytes as f64) * 100.0);
    
    // Example: Get a specific weight
    if int8_model.has_weight("time_in.in_layer.weight") {
        let weight = int8_model.get_weight("time_in.in_layer.weight")?;
        println!("\nExample weight 'time_in.in_layer.weight':");
        println!("  Shape: {:?}", weight.dims());
        println!("  Dtype: {:?}", weight.dtype());
    }
    
    // List some weights
    let weight_names = int8_model.weight_names();
    println!("\nTotal weights: {}", weight_names.len());
    println!("First 10 weights:");
    for (i, name) in weight_names.iter().take(10).enumerate() {
        println!("  {}: {}", i + 1, name);
    }
    
    // Example: Get multiple weights for a layer
    let layer_weights = int8_model.get_weights(&[
        "time_in.in_layer.weight",
        "time_in.in_layer.bias",
        "time_in.out_layer.weight",
        "time_in.out_layer.bias",
    ])?;
    
    println!("\nLoaded {} weights for time_in layer", layer_weights.len());
    
    Ok(())
}