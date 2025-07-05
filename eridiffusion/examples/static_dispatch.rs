//! Example demonstrating static dispatch for performance

use eridiffusion_core::{
    static_dispatch::{ModelDispatch, NetworkDispatch},
    ModelArchitecture, Device, Error, Result,
};
use eridiffusion_models::model::{ModelInputs, DiffusionModel};
use eridiffusion_networks::{LoRAConfig, NetworkAdapter};
use candle_core::Tensor;
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<()> {
    let device = Device::cuda_if_available()?;
    
    // Example 1: Using static dispatch for models
    println!("=== Static Dispatch Model Example ===");
    
    // Create models using static dispatch
    let sd15_model = ModelDispatch::from_architecture(ModelArchitecture::SD15, device.clone())?;
    let sdxl_model = ModelDispatch::from_architecture(ModelArchitecture::SDXL, device.clone())?;
    let flux_model = ModelDispatch::from_architecture(
        ModelArchitecture::Flux(eridiffusion_core::FluxVariant::Dev),
        device.clone()
    )?;
    
    println!("Created models with static dispatch:");
    println!("- SD1.5: {} parameters", sd15_model.num_parameters());
    println!("- SDXL: {} parameters", sdxl_model.num_parameters());
    println!("- Flux: {} parameters", flux_model.num_parameters());
    
    // Example 2: Performance comparison
    println!("\n=== Performance Comparison ===");
    
    // Create dummy inputs
    let batch_size = 1;
    let latent_channels = 4;
    let height = 64;
    let width = 64;
    
    let inputs = ModelInputs {
        latents: Tensor::randn(0f32, 1f32, &[batch_size, latent_channels, height, width], &device)?,
        timestep: Tensor::new(&[500i64], &device)?,
        encoder_hidden_states: Tensor::randn(0f32, 1f32, &[batch_size, 77, 768], &device)?,
        ..Default::default()
    };
    
    // Benchmark static dispatch
    let start = Instant::now();
    for _ in 0..10 {
        let _ = sd15_model.forward(&inputs)?;
    }
    let static_time = start.elapsed();
    
    // Compare with dynamic dispatch (using trait object)
    let dynamic_model: Box<dyn DiffusionModel + Send + Sync> = Box::new(
        eridiffusion_models::SD15Model::new(device.clone())?
    );
    
    let start = Instant::now();
    for _ in 0..10 {
        let _ = dynamic_model.forward(&inputs)?;
    }
    let dynamic_time = start.elapsed();
    
    println!("Static dispatch: {:?}", static_time);
    println!("Dynamic dispatch: {:?}", dynamic_time);
    println!("Speedup: {:.2}x", dynamic_time.as_secs_f64() / static_time.as_secs_f64());
    
    // Example 3: Using static dispatch for network adapters
    println!("\n=== Static Dispatch Network Adapter Example ===");
    
    let lora_config = LoRAConfig {
        rank: 32,
        alpha: 32.0,
        dropout: 0.0,
        target_modules: vec!["attn".to_string()],
    };
    
    let lora_adapter = eridiffusion_networks::LoRAAdapter::new(lora_config, &device)?;
    let network = NetworkDispatch::LoRA(lora_adapter);
    
    println!("Created LoRA adapter with static dispatch");
    println!("Type: {}", network.adapter_type());
    println!("Parameters: {}", network.num_parameters());
    
    // Example 4: Switching between models efficiently
    println!("\n=== Model Switching Example ===");
    
    let architectures = vec![
        ModelArchitecture::SD15,
        ModelArchitecture::SD2,
        ModelArchitecture::SDXL,
    ];
    
    for arch in architectures {
        let model = ModelDispatch::from_architecture(arch.clone(), device.clone())?;
        
        // Process with model
        match &model {
            ModelDispatch::SD15(_) => println!("Processing with SD1.5..."),
            ModelDispatch::SD2(_) => println!("Processing with SD2..."),
            ModelDispatch::SDXL(_) => println!("Processing with SDXL..."),
            _ => println!("Processing with other model..."),
        }
        
        // Model-specific optimizations can be applied
        let output = model.forward(&inputs)?;
        println!("Output shape: {:?}", output.sample.shape());
    }
    
    // Example 5: Fallback to dynamic dispatch for plugins
    println!("\n=== Dynamic Fallback Example ===");
    
    // Simulate a plugin model
    struct PluginModel {
        device: Device,
    }
    
    impl DiffusionModel for PluginModel {
        fn forward(&self, inputs: &ModelInputs) -> Result<eridiffusion_models::model::ModelOutput> {
            Ok(eridiffusion_models::model::ModelOutput {
                sample: inputs.latents.clone(),
                ..Default::default()
            })
        }
        
        fn architecture(&self) -> ModelArchitecture {
            ModelArchitecture::SD15 // Placeholder
        }
        
        fn device(&self) -> &Device {
            &self.device
        }
        
        // ... other trait methods ...
        fn to_device(&mut self, _device: &Device) -> Result<()> { Ok(()) }
        async fn load_pretrained(&mut self, _path: &std::path::Path) -> Result<()> { Ok(()) }
        async fn save_pretrained(&self, _path: &std::path::Path) -> Result<()> { Ok(()) }
        fn state_dict(&self) -> Result<std::collections::HashMap<String, Tensor>> { Ok(Default::default()) }
        fn load_state_dict(&mut self, _state_dict: std::collections::HashMap<String, Tensor>) -> Result<()> { Ok(()) }
        fn trainable_parameters(&self) -> Vec<&Tensor> { vec![] }
        fn num_parameters(&self) -> usize { 0 }
        fn memory_usage(&self) -> usize { 0 }
    }
    
    let plugin = PluginModel { device: device.clone() };
    let dynamic_dispatch = ModelDispatch::Dynamic(Box::new(plugin));
    
    println!("Created plugin model with dynamic fallback");
    let _ = dynamic_dispatch.forward(&inputs)?;
    println!("Plugin model processed successfully");
    
    Ok(())
}