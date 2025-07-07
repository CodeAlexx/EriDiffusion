//! Test Flux model loading with different strategies

use anyhow::Result;
use candle_core::{Device, DType};
use eridiffusion::trainers::cached_device::get_single_device;
use eridiffusion::memory::MemoryManager;
use std::path::PathBuf;

fn main() -> Result<()> {
    println!("=== Testing Flux Model Loading ===\n");
    
    // Get cached device
    let device = get_single_device()?;
    println!("Using device: {:?}", device);
    
    // Check memory before
    MemoryManager::log_memory_usage("Before any loading")?;
    
    // Model path
    let model_path = PathBuf::from("/home/alex/SwarmUI/Models/diffusion_models/flux1-dev.safetensors");
    
    // Test 1: Check file exists
    if !model_path.exists() {
        panic!("Model file not found: {}", model_path.display());
    }
    println!("✓ Model file exists");
    
    // Test 2: Try loading metadata only
    println!("\nTest 2: Loading metadata...");
    use safetensors::SafeTensors;
    use std::fs::File;
    use memmap2::Mmap;
    
    let file = File::open(&model_path)?;
    let file_size = file.metadata()?.len();
    println!("File size: {:.2} GB", file_size as f64 / 1e9);
    
    let mmap = unsafe { Mmap::map(&file)? };
    let tensors = SafeTensors::deserialize(&mmap)?;
    println!("✓ Found {} tensors in file", tensors.names().len());
    
    // Test 3: Check a few tensor names
    println!("\nTest 3: Checking tensor names...");
    let names: Vec<_> = tensors.names().into_iter().take(5).collect();
    for name in &names {
        println!("  - {}", name);
    }
    
    // Test 4: Try the adaptive loader
    println!("\nTest 4: Testing adaptive loader...");
    use eridiffusion::models::flux_adaptive_loader::FluxAdaptiveLoader;
    
    println!("Creating adaptive loader...");
    let loader = FluxAdaptiveLoader::from_file(
        &model_path,
        device.clone(),
        DType::F16,
        3072,  // hidden_size for Flux-dev
    )?;
    
    println!("Loader created!");
    
    // Check memory after
    MemoryManager::log_memory_usage("After creating loader")?;
    
    println!("\n✓ All tests passed!");
    Ok(())
}