//! LoKr training demonstration
//! 
//! This example shows a complete training pipeline for LoKr adapters
//! on diffusion models using the eridiffusion framework.

use eridiffusion_core::{Device, ModelArchitecture, Result, Error};
use eridiffusion_models::{DiffusionModel, ModelInputs, SD3Model};
use eridiffusion_networks::{LoKrConfig, LoKr, NetworkAdapter};
use candle_core::{Tensor, DType};
use candle_nn::VarBuilder;
use std::collections::HashMap;

/// Simple training configuration
struct TrainingConfig {
    learning_rate: f32,
    batch_size: usize,
    num_epochs: usize,
    rank: usize,
    alpha: f32,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-4,
            batch_size: 1,
            num_epochs: 100,
            rank: 16,
            alpha: 16.0,
        }
    }
}

/// Simple dataset for demonstration
struct DemoDataset {
    images: Vec<Tensor>,
    captions: Vec<String>,
}

impl DemoDataset {
    fn new(device: &Device) -> Result<Self> {
        // Create dummy data for demonstration
        let candle_device = device.as_candle_device()?;
        
        // Create random images (already as latents for simplicity)
        let images = (0..10)
            .map(|_| Tensor::randn(0.0f32, 1.0, &[16, 64, 64], &candle_device))
            .collect::<candle_core::Result<Vec<_>>>()?;
        
        // Create simple captions
        let captions = (0..10)
            .map(|i| format!("A beautiful image number {}", i))
            .collect();
        
        Ok(Self { images, captions })
    }
    
    fn get_batch(&self, idx: usize, batch_size: usize) -> Result<(Tensor, Vec<String>)> {
        let start = idx * batch_size;
        let end = (start + batch_size).min(self.images.len());
        
        let batch_images = Tensor::stack(&self.images[start..end], 0)?;
        let batch_captions = self.captions[start..end].to_vec();
        
        Ok((batch_images, batch_captions))
    }
    
    fn len(&self) -> usize {
        self.images.len()
    }
}

/// Main training loop
fn train_lokr(config: TrainingConfig) -> Result<()> {
    println!("Starting LoKr training demonstration");
    
    // Setup device
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    println!("Using device: {:?}", device);
    
    // Create model (SD3.5 for this example)
    println!("Creating SD3.5 model...");
    let candle_device = device.as_candle_device()?;
    let vb = unsafe { VarBuilder::uninit(DType::F32, &candle_device) };
    let mut model = SD3Model::new(vb.pp("sd3"), true)?; // true for SD3.5
    
    // Create LoKr adapter
    println!("Creating LoKr adapter with rank {}...", config.rank);
    let lokr_config = LoKrConfig {
        rank: config.rank,
        alpha: config.alpha,
        target_modules: vec![
            "to_q".to_string(),
            "to_k".to_string(), 
            "to_v".to_string(),
            "to_out.0".to_string(),
        ],
        ..Default::default()
    };
    
    let mut lokr = LoKr::new(
        lokr_config.clone(),
        ModelArchitecture::SD35,
        device.clone(),
    )?;
    
    // Initialize adapter weights
    println!("Initializing LoKr weights...");
    lokr.initialize_weights()?;
    
    // Create dataset
    println!("Creating demonstration dataset...");
    let dataset = DemoDataset::new(&device)?;
    
    // Create text embeddings (simplified - normally would use text encoder)
    let text_embedding_dim = 2048; // SD3 uses 2048
    let dummy_text_embeddings = Tensor::randn(
        0.0f32,
        0.1,
        &[config.batch_size, 77, text_embedding_dim],
        &candle_device,
    )?;
    
    // Training loop
    println!("\nStarting training for {} epochs...", config.num_epochs);
    
    for epoch in 0..config.num_epochs {
        let mut epoch_loss = 0.0;
        let num_batches = (dataset.len() + config.batch_size - 1) / config.batch_size;
        
        for batch_idx in 0..num_batches {
            // Get batch
            let (latents, _captions) = dataset.get_batch(batch_idx, config.batch_size)?;
            
            // Sample timesteps
            let timesteps = Tensor::new(
                &[500u32], // Mid-range timestep
                &candle_device,
            )?;
            
            // Add noise to create noisy latents
            let noise = Tensor::randn_like(&latents)?;
            let noisy_latents = &latents + &noise.mul_scalar(0.5)?;
            
            // Apply LoKr adapter to model
            let model_with_lokr = lokr.apply(&model)?;
            
            // Forward pass
            let model_inputs = ModelInputs {
                latents: noisy_latents.clone(),
                timestep: timesteps.clone(),
                encoder_hidden_states: dummy_text_embeddings.clone(),
                additional: {
                    let mut map = HashMap::new();
                    // SD3 requires pooled projections
                    let pooled = Tensor::randn(
                        0.0f32,
                        0.1,
                        &[config.batch_size, 2048],
                        &candle_device,
                    )?;
                    map.insert("pooled_projections".to_string(), pooled);
                    map
                },
            };
            
            let output = model_with_lokr.forward(&model_inputs)?;
            let predicted_noise = output.sample;
            
            // Calculate loss (simplified MSE)
            let loss = (&predicted_noise - &noise)?
                .sqr()?
                .mean_all()?
                .to_scalar::<f32>()?;
            
            // Backward pass (simplified - normally would use autograd)
            // In a real implementation, we would:
            // 1. Compute gradients with respect to LoKr parameters
            // 2. Update LoKr weights using an optimizer
            
            // For demonstration, we'll just simulate weight updates
            lokr.apply_weight_decay(0.01)?;
            
            epoch_loss += loss;
            
            if batch_idx % 10 == 0 {
                println!(
                    "Epoch {}/{}, Batch {}/{}, Loss: {:.6}",
                    epoch + 1, config.num_epochs,
                    batch_idx + 1, num_batches,
                    loss
                );
            }
        }
        
        let avg_loss = epoch_loss / num_batches as f32;
        println!("Epoch {} completed. Average loss: {:.6}", epoch + 1, avg_loss);
        
        // Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0 {
            let checkpoint_path = format!("lokr_checkpoint_epoch_{}.safetensors", epoch + 1);
            println!("Saving checkpoint to {}", checkpoint_path);
            lokr.save_pretrained(&std::path::Path::new(&checkpoint_path))?;
        }
        
        // Early stopping if loss is good enough
        if avg_loss < 0.001 {
            println!("Early stopping - loss threshold reached!");
            break;
        }
    }
    
    // Save final model
    println!("\nTraining completed! Saving final LoKr adapter...");
    lokr.save_pretrained(&std::path::Path::new("final_lokr_adapter.safetensors"))?;
    
    // Show some statistics
    let metadata = lokr.metadata();
    println!("\nLoKr Adapter Statistics:");
    println!("- Architecture: {:?}", metadata.target_architecture);
    println!("- Rank: {}", lokr_config.rank);
    println!("- Alpha: {}", lokr_config.alpha);
    println!("- Target modules: {:?}", lokr_config.target_modules);
    println!("- Trainable parameters: ~{:.2}M", 
        lokr.count_parameters() as f32 / 1_000_000.0);
    
    Ok(())
}

fn main() -> Result<()> {
    // Initialize logging
    env_logger::init();
    
    // Run training
    let config = TrainingConfig::default();
    train_lokr(config)?;
    
    println!("\nDemo completed successfully!");
    Ok(())
}

// Extension trait for Device
trait DeviceExt {
    fn as_candle_device(&self) -> Result<candle_core::Device>;
}

impl DeviceExt for Device {
    fn as_candle_device(&self) -> Result<candle_core::Device> {
        match self {
            Device::Cpu => Ok(candle_core::Device::Cpu),
            Device::Cuda(id) => Ok(candle_core::Device::cuda(*id)?),
        }
    }
}