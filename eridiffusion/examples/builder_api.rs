//! Example demonstrating the new builder API patterns

use eridiffusion_core::{
    builders::{ModelConfigBuilder, TrainingConfigBuilder, InferenceConfigBuilder},
    ModelArchitecture, Device, Precision,
};
use eridiffusion_models::ModelFactory;
use eridiffusion_training::Trainer;
use eridiffusion_inference::InferencePipeline;
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    // Example 1: Building model configuration with fluent API
    println!("=== Model Configuration Builder Example ===");
    
    let model_config = ModelConfigBuilder::new()
        .architecture(ModelArchitecture::SDXL)
        .device(Device::cuda_if_available()?)
        .precision(Precision::Float16)
        .model_path("stabilityai/stable-diffusion-xl-base-1.0")
        .compile(true)
        .use_flash_attention(true)
        .custom("vae_scaling_factor", 0.13025)?
        .build()?;
    
    println!("Model config: {:?}", model_config);
    
    // Example 2: Building training configuration
    println!("\n=== Training Configuration Builder Example ===");
    
    let training_config = TrainingConfigBuilder::new()
        .model(model_config.clone())
        .learning_rate(1e-5)
        .batch_size(4)
        .epochs(100)
        .gradient_accumulation(4)
        .optimizer("adamw")
        .adam_params(0.9, 0.999, 1e-8)
        .weight_decay(0.01)
        .scheduler("cosine_with_restarts")
        .warmup_steps(500)
        .output_dir("./training_output")
        .checkpointing(1000)
        .logging(50)
        .evaluation(500)
        .seed(42)
        .mixed_precision(true)
        .gradient_checkpointing(true)
        .build()?;
    
    println!("Training config: learning_rate={}, batch_size={}, epochs={}", 
        training_config.learning_rate,
        training_config.batch_size,
        training_config.num_epochs
    );
    
    // Example 3: Building inference configuration
    println!("\n=== Inference Configuration Builder Example ===");
    
    let inference_config = InferenceConfigBuilder::new()
        .model(model_config)
        .scheduler("ddim")
        .steps(25)
        .guidance_scale(7.5)
        .eta(0.0)
        .seed(12345)
        .batch_size(1)
        .compile(false)
        .build()?;
    
    println!("Inference config: scheduler={}, steps={}, guidance_scale={}", 
        inference_config.scheduler,
        inference_config.num_inference_steps,
        inference_config.guidance_scale
    );
    
    // Example 4: Using configurations in practice
    println!("\n=== Using Configurations ===");
    
    // Create model with builder config
    let model = ModelFactory::from_config(&training_config.model_config).await?;
    println!("Created model: {} parameters", model.num_parameters());
    
    // Create trainer with builder config
    let trainer = Trainer::from_config(training_config)?;
    println!("Created trainer with optimizer: {}", trainer.config().optimizer_type);
    
    // Create inference pipeline with builder config
    let pipeline = InferencePipeline::from_config(inference_config)?;
    println!("Created inference pipeline");
    
    // Example 5: Chaining multiple configurations
    println!("\n=== Chaining Configurations ===");
    
    // Quick inference setup for different models
    for arch in [ModelArchitecture::SD15, ModelArchitecture::SDXL, ModelArchitecture::SD3] {
        let config = InferenceConfigBuilder::new()
            .model(
                ModelConfigBuilder::new()
                    .architecture(arch)
                    .model_path(format!("model_{:?}", arch))
                    .build()?
            )
            .steps(50)
            .build()?;
        
        println!("Config for {:?}: {} steps", arch, config.num_inference_steps);
    }
    
    Ok(())
}