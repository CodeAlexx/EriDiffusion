//! Initialize Flux model with random weights for LoRA training
//! 
//! This creates a model with properly initialized weights that can be used
//! for training without needing to load the full model from disk.

use anyhow::{Result, Context};
use candle_core::{Device, DType};
use candle_nn::{VarBuilder, VarMap, Init};

use crate::models::flux_custom::{FluxConfig, FluxModelWithLoRA};

/// Initialize minimal weights for Flux model (just enough to create structure)
pub fn initialize_flux_weights_minimal(vb: &VarBuilder, config: &FluxConfig) -> Result<()> {
    let init = Init::Const(0.0);
    
    // Time and vector embeddings (minimal)
    vb.get_with_hints(&[config.hidden_size, 256], "time_in.0.weight", init)?;
    vb.get_with_hints(&[config.hidden_size], "time_in.0.bias", init)?;
    vb.get_with_hints(&[config.hidden_size, config.hidden_size], "time_in.2.weight", init)?;
    vb.get_with_hints(&[config.hidden_size], "time_in.2.bias", init)?;
    
    vb.get_with_hints(&[config.hidden_size, config.vec_in_dim], "vector_in.0.weight", init)?;
    vb.get_with_hints(&[config.hidden_size], "vector_in.0.bias", init)?;
    vb.get_with_hints(&[config.hidden_size, config.hidden_size], "vector_in.2.weight", init)?;
    vb.get_with_hints(&[config.hidden_size], "vector_in.2.bias", init)?;
    
    // Input/output projections
    vb.get_with_hints(&[config.hidden_size, config.in_channels], "img_in.weight", init)?;
    vb.get_with_hints(&[config.hidden_size, config.context_in_dim], "txt_in.weight", init)?;
    vb.get_with_hints(&[config.in_channels, config.hidden_size], "final_layer.weight", init)?;
    
    Ok(())
}

/// Create a Flux model with random weights for LoRA training
pub fn create_flux_with_random_weights(
    config: &FluxConfig,
    device: Device,
    dtype: DType,
) -> Result<FluxModelWithLoRA> {
    println!("\n=== Creating Flux Model with Random Weights ===");
    println!("This allows LoRA training without loading base weights");
    
    // Create VarMap with proper initialization
    let var_map = VarMap::new();
    let vb = VarBuilder::from_varmap(&var_map, dtype, &device);
    
    // We need to pre-initialize some key weights that the model expects
    // This is a workaround for the empty VarMap issue
    
    // Initialize input projections with proper dimensions
    let init = Init::Randn { mean: 0.0, stdev: 0.02 };
    
    // Image input projection: in_channels -> hidden_size
    vb.get_with_hints(&[config.hidden_size, config.in_channels], "img_in.weight", init)?;
    
    // Text input projection: context_in_dim -> hidden_size
    vb.get_with_hints(&[config.hidden_size, config.context_in_dim], "txt_in.weight", init)?;
    
    // Time embedding MLP
    vb.get_with_hints(&[config.hidden_size, 256], "time_in.weight", init)?;
    vb.get_with_hints(&[config.hidden_size], "time_in.bias", Init::Const(0.0))?;
    
    // Vector input MLP
    vb.get_with_hints(&[config.hidden_size, config.vec_in_dim], "vector_in.weight", init)?;
    vb.get_with_hints(&[config.hidden_size], "vector_in.bias", Init::Const(0.0))?;
    
    // Guidance embedding (if used)
    if config.guidance_embed {
        vb.get_with_hints(&[config.hidden_size, 256], "guidance_in.weight", init)?;
        vb.get_with_hints(&[config.hidden_size], "guidance_in.bias", Init::Const(0.0))?;
    }
    
    // Final layer projection
    vb.get_with_hints(&[config.in_channels, config.hidden_size], "final_layer.weight", init)?;
    
    // Initialize key layer norms
    let ln_init = Init::Const(1.0);
    let ln_bias_init = Init::Const(0.0);
    
    // Pre-initialize some transformer block weights to avoid issues
    for i in 0..config.depth {
        let prefix = format!("double_blocks.{}", i);
        
        // Layer norms
        vb.get_with_hints(&[config.hidden_size], &format!("{}.img_norm1.weight", prefix), ln_init)?;
        vb.get_with_hints(&[config.hidden_size], &format!("{}.img_norm1.bias", prefix), ln_bias_init)?;
        vb.get_with_hints(&[config.hidden_size], &format!("{}.txt_norm1.weight", prefix), ln_init)?;
        vb.get_with_hints(&[config.hidden_size], &format!("{}.txt_norm1.bias", prefix), ln_bias_init)?;
        
        // Attention projections (combined QKV)
        let qkv_dim = 3 * config.hidden_size;
        
        vb.get_with_hints(&[qkv_dim, config.hidden_size], &format!("{}.img_attn.qkv.weight", prefix), init)?;
        if config.qkv_bias {
            vb.get_with_hints(&[qkv_dim], &format!("{}.img_attn.qkv.bias", prefix), Init::Const(0.0))?;
        }
        vb.get_with_hints(&[qkv_dim, config.hidden_size], &format!("{}.txt_attn.qkv.weight", prefix), init)?;
        if config.qkv_bias {
            vb.get_with_hints(&[qkv_dim], &format!("{}.txt_attn.qkv.bias", prefix), Init::Const(0.0))?;
        }
        
        // Output projections
        vb.get_with_hints(&[config.hidden_size, config.hidden_size], &format!("{}.img_attn.proj.weight", prefix), init)?;
        vb.get_with_hints(&[config.hidden_size], &format!("{}.img_attn.proj.bias", prefix), Init::Const(0.0))?;
        vb.get_with_hints(&[config.hidden_size, config.hidden_size], &format!("{}.txt_attn.proj.weight", prefix), init)?;
        vb.get_with_hints(&[config.hidden_size], &format!("{}.txt_attn.proj.bias", prefix), Init::Const(0.0))?;
    }
    
    // Create the model
    println!("Creating model structure with initialized weights...");
    let model = FluxModelWithLoRA::new(config, vb)?;
    
    println!("✅ Model created with random weights!");
    println!("Base weights are random - only LoRA weights will be trained");
    
    Ok(model)
}

/// Create a minimal set of weights for testing
pub fn create_minimal_weights(
    config: &FluxConfig,
    device: Device,
    dtype: DType,
) -> Result<FluxModelWithLoRA> {
    println!("\n=== Creating Minimal Flux Model ===");
    
    // Create VarMap
    let var_map = VarMap::new();
    let vb = VarBuilder::from_varmap(&var_map, dtype, &device);
    
    // Use very small initialization to reduce memory usage
    let init = Init::Const(0.001);
    
    // Only initialize the absolute minimum required weights
    vb.get_with_hints(&[config.hidden_size, config.in_channels], "img_in.weight", init)?;
    vb.get_with_hints(&[config.hidden_size, config.context_in_dim], "txt_in.weight", init)?;
    vb.get_with_hints(&[config.in_channels, config.hidden_size], "final_layer.weight", init)?;
    
    // Create model
    let model = FluxModelWithLoRA::new(config, vb)?;
    
    println!("✅ Minimal model created!");
    
    Ok(model)
}