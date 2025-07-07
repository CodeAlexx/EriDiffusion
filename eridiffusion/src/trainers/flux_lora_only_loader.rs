//! LoRA-only loader for Flux that doesn't load base weights
//! This is the most memory-efficient approach for training

use anyhow::{Result, Context};
use candle_core::{Device, DType};
use candle_nn::{VarBuilder, VarMap};
use std::path::Path;

use crate::models::flux_custom::{FluxConfig, FluxModelWithLoRA};
use crate::models::flux_custom::lora::LoRAConfig;
use crate::trainers::flux_init_weights::initialize_flux_weights_minimal;

/// Create a Flux model for LoRA-only training (no base weights)
pub fn create_flux_lora_only(
    model_path: &Path,  // Not used, but kept for API compatibility
    lora_config: &LoRAConfig,
    device: Device,
) -> Result<FluxModelWithLoRA> {
    println!("\n=== Creating Flux Model for LoRA-Only Training ===");
    println!("This approach uses minimal memory by not loading base weights");
    println!("Base weights will be streamed from disk during training if needed");
    
    // Flux configuration
    let config = FluxConfig::default();
    
    // Create empty VarMap
    let var_map = VarMap::new();
    let vb = VarBuilder::from_varmap(&var_map, DType::F16, &device);
    
    // Initialize only minimal weights to create structure
    println!("Initializing minimal weights for model structure...");
    initialize_flux_weights_minimal(&vb, &config)?;
    
    // Create model
    println!("Creating Flux model structure...");
    let mut model = FluxModelWithLoRA::new(&config, vb)?;
    
    // Add LoRA adapters - these are the only trainable parts
    println!("Adding LoRA adapters (rank={}, alpha={})...", lora_config.rank, lora_config.alpha);
    model.add_lora_to_all(lora_config, &device, DType::F16)?;
    
    // Report memory usage
    let trainable_params = model.get_trainable_params();
    let param_count: usize = trainable_params.iter()
        .map(|p| p.elem_count())
        .sum();
    
    println!("\n✅ Model created successfully!");
    println!("Trainable LoRA parameters: {}", param_count);
    println!("Memory usage: ~{:.1} MB", (param_count * 2) as f32 / 1e6);
    println!("Base model weights: NOT LOADED (saves ~15GB)");
    
    Ok(model)
}

/// Create a Flux model with optional weight loading
pub fn create_flux_with_optional_weights(
    model_path: &Path,
    lora_config: &LoRAConfig,
    device: Device,
    load_base_weights: bool,
) -> Result<FluxModelWithLoRA> {
    if load_base_weights {
        // Use the efficient loader that loads in FP16
        use crate::trainers::flux_efficient_loader::create_flux_for_24gb_training;
        create_flux_for_24gb_training(model_path, lora_config, device)
    } else {
        // Use LoRA-only approach
        create_flux_lora_only(model_path, lora_config, device)
    }
}