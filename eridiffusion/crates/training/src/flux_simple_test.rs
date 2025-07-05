//! Simple Flux forward pass test
//! Tests basic functionality without full model loading

use eridiffusion_core::{Device, Result, Error, VarExt};
use candle_core::{DType, Device as CandleDevice, Module, Tensor, D};
use candle_nn::{VarBuilder, VarMap};
use candle_transformers::models::flux;
use std::time::Instant;

pub fn run_simple_flux_test() -> Result<()> {
    println!("🧪 Running Simple Flux Forward Pass Test");
    println!("========================================\n");
    
    // Create device
    let device = CandleDevice::Cpu;
    println!("Device: CPU");
    
    // Create Flux config
    let flux_config = flux::model::Config::dev();
    println!("✓ Created Flux-Dev configuration");
    
    // Initialize VarMap with random weights
    let var_map = VarMap::new();
    let vb = VarBuilder::from_varmap(&var_map, DType::F32, &device);
    println!("✓ Initialized VarMap");
    
    // Create model
    let start = Instant::now();
    let model = flux::model::Flux::new(&flux_config, vb)?;
    println!("✓ Model created in {:.2}s", start.elapsed().as_secs_f32());
    
    // Count parameters
    let total_params = count_parameters(&var_map)?;
    println!("✓ Total parameters: {:.2}M", total_params as f64 / 1e6);
    
    // Test forward pass with small tensors
    println!("\n📐 Testing Forward Pass");
    
    let batch_size = 1;
    let latent_size = 32; // Small size for testing
    
    // Create dummy inputs
    let latents = Tensor::randn(
        0.0, 
        1.0, 
        &[batch_size, 16, latent_size, latent_size], // Flux uses 16 latent channels
        &device
    )?;
    
    // Create text embeddings (T5-XXL dimensions)
    let text_embeddings = Tensor::randn(
        0.0,
        1.0,
        &[batch_size, 64, 4096], // Reduced sequence length for testing
        &device
    )?;
    
    // Create pooled embeddings (CLIP dimensions)
    let pooled_embeddings = Tensor::randn(
        0.0,
        1.0,
        &[batch_size, 768], // CLIP-L pooled: 768 dim
        &device
    )?;
    
    // Create timesteps
    let timesteps = Tensor::new(&[500i64], &device)?;
    
    // Test forward pass
    let start = Instant::now();
    let state = flux::sampling::State::new(&text_embeddings, &pooled_embeddings, &latents)?;
    
    // For a single denoising step
    let output = flux::sampling::denoise(
        &model,
        &state.img,
        &state.img_ids,
        &state.txt,
        &state.txt_ids,
        &state.vec,
        &[timesteps.to_scalar::<f64>()?],
        3.5, // guidance scale
    )?;
    
    let forward_time = start.elapsed();
    
    println!("✓ Input latents shape: {:?}", latents.shape());
    println!("✓ Text embeddings shape: {:?}", text_embeddings.shape());
    println!("✓ Pooled embeddings shape: {:?}", pooled_embeddings.shape());
    println!("✓ Output shape: {:?}", output.shape());
    println!("✓ Forward pass time: {:.3}s", forward_time.as_secs_f32());
    
    // Test gradient computation
    println!("\n🔄 Testing Gradient Flow");
    
    // Create a simple loss
    let target = Tensor::randn(0.0, 1.0, output.shape(), &device)?;
    let loss = output.sub(&target)?.sqr()?.mean_all()?;
    
    println!("✓ Loss computed: {:.6}", loss.to_scalar::<f32>()?);
    
    // Backward pass
    let start = Instant::now();
    loss.backward()?;
    let backward_time = start.elapsed();
    
    println!("✓ Backward pass completed in {:.3}s", backward_time.as_secs_f32());
    
    // Check gradients
    let mut has_gradients = false;
    let mut grad_count = 0;
    
    let all_vars = var_map.all_vars();
    for (idx, var) in all_vars.iter().enumerate() {
        let name = format!("var_{}", idx);
        if let Ok(_grad) = var.grad() {
            has_gradients = true;
            grad_count += 1;
            if grad_count <= 5 {
                println!("✓ Gradient found for: {}", name);
            }
        }
    }
    
    if has_gradients {
        println!("✓ Total variables with gradients: {}", grad_count);
    } else {
        return Err(Error::Training("No gradients found!".to_string()));
    }
    
    println!("\n✅ Simple Flux test passed!");
    Ok(())
}

/// Count total parameters in VarMap
fn count_parameters(var_map: &VarMap) -> Result<usize> {
    let mut total = 0;
    let all_vars = var_map.all_vars();
    for var in all_vars.iter() {
        total += var.as_tensor().elem_count();
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_simple_flux() -> Result<()> {
        run_simple_flux_test()
    }
}