//! Example demonstrating model-specific training pipelines

use eridiffusion_core::{Device, ModelArchitecture, FluxVariant, Result};
use eridiffusion_models::ModelFactory;
use eridiffusion_networks::{LoRAConfig, NetworkFactory};
use eridiffusion_training::{
    TrainingConfig, PipelineConfig, PipelineFactory,
    Trainer, TrainerConfig, OptimizerConfig,
};
use eridiffusion_data::{DatasetConfig, DataLoader};
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize device
    let device = Device::cuda_if_available()?;
    println!("Using device: {:?}", device);
    
    // Example 1: Train SD1.5 with LoRA
    train_sd15_lora(device.clone()).await?;
    
    // Example 2: Train SDXL with different settings
    train_sdxl(device.clone()).await?;
    
    // Example 3: Train SD3 with flow matching
    train_sd3_flow(device.clone()).await?;
    
    // Example 4: Train Flux
    train_flux(device.clone()).await?;
    
    Ok(())
}

/// Train SD1.5 with LoRA
async fn train_sd15_lora(device: Device) -> Result<()> {
    println!("\n=== Training SD1.5 with LoRA ===");
    
    // Create model
    let model = ModelFactory::create(ModelArchitecture::SD15, device.clone()).await?;
    
    // Create LoRA adapter
    let lora_config = LoRAConfig {
        rank: 32,
        alpha: 32.0,
        dropout: 0.0,
        target_modules: vec!["to_q".to_string(), "to_v".to_string()],
    };
    let network = NetworkFactory::create_lora(lora_config, &model)?;
    
    // Create pipeline configuration
    let pipeline_config = PipelineConfig {
        training_config: TrainingConfig {
            learning_rate: 1e-4,
            batch_size: 4,
            num_epochs: 10,
            gradient_accumulation_steps: 2,
            ..Default::default()
        },
        device: device.clone(),
        dtype: candle_core::DType::F16,
        use_ema: false,
        gradient_checkpointing: true,
        loss_type: "mse".to_string(),
        snr_gamma: Some(5.0),
        v_parameterization: false,
        flow_matching: false,
        noise_offset: 0.1,
        ..Default::default()
    };
    
    // Create model-specific pipeline
    let pipeline = PipelineFactory::create(ModelArchitecture::SD15, pipeline_config)?;
    
    // Setup data loader
    let dataset_config = DatasetConfig {
        data_dir: PathBuf::from("./data/sd15_dataset"),
        caption_column: "text".to_string(),
        resolution: 512,
        center_crop: true,
        random_flip: true,
        cache_latents: true,
        ..Default::default()
    };
    let dataloader = DataLoader::new(dataset_config)?;
    
    // Create trainer
    let trainer_config = TrainerConfig {
        model,
        network: Some(network),
        pipeline: Box::new(pipeline),
        dataloader,
        output_dir: PathBuf::from("./output/sd15_lora"),
        checkpoint_steps: 500,
        logging_steps: 50,
        ..Default::default()
    };
    
    let mut trainer = Trainer::new(trainer_config)?;
    
    // Train
    println!("Starting SD1.5 LoRA training...");
    // trainer.train().await?;
    println!("SD1.5 training setup complete!");
    
    Ok(())
}

/// Train SDXL with dual text encoders
async fn train_sdxl(device: Device) -> Result<()> {
    println!("\n=== Training SDXL ===");
    
    // Create model
    let model = ModelFactory::create(ModelArchitecture::SDXL, device.clone()).await?;
    
    // SDXL-specific pipeline configuration
    let pipeline_config = PipelineConfig {
        training_config: TrainingConfig {
            learning_rate: 5e-5,
            batch_size: 1,
            num_epochs: 20,
            gradient_accumulation_steps: 4,
            ..Default::default()
        },
        device: device.clone(),
        dtype: candle_core::DType::F16,
        use_ema: true,
        ema_decay: 0.9999,
        gradient_checkpointing: true,
        loss_type: "mse".to_string(),
        snr_gamma: Some(5.0),
        v_parameterization: false,
        flow_matching: false,
        noise_offset: 0.05,
        ..Default::default()
    };
    
    // Create SDXL pipeline
    let pipeline = PipelineFactory::create(ModelArchitecture::SDXL, pipeline_config)?;
    
    println!("SDXL pipeline created with dual text encoder support");
    println!("- Resolution: 1024x1024");
    println!("- Additional conditioning: time_ids, aesthetic scores");
    
    Ok(())
}

/// Train SD3 with flow matching
async fn train_sd3_flow(device: Device) -> Result<()> {
    println!("\n=== Training SD3 with Flow Matching ===");
    
    // Create model
    let model = ModelFactory::create(ModelArchitecture::SD3, device.clone()).await?;
    
    // SD3-specific pipeline configuration
    let pipeline_config = PipelineConfig {
        training_config: TrainingConfig {
            learning_rate: 1e-4,
            batch_size: 1,
            num_epochs: 10,
            gradient_accumulation_steps: 8,
            ..Default::default()
        },
        device: device.clone(),
        dtype: candle_core::DType::BF16,
        gradient_checkpointing: true,
        loss_type: "mse".to_string(),
        flow_matching: true, // SD3 uses flow matching
        mean_flow_loss: true, // Enable mean flow loss
        noise_offset: 0.0, // No noise offset for flow matching
        ..Default::default()
    };
    
    // Create SD3 pipeline
    let pipeline = PipelineFactory::create(ModelArchitecture::SD3, pipeline_config)?;
    
    println!("SD3 pipeline created with flow matching");
    println!("- MMDiT architecture");
    println!("- 16-channel latents");
    println!("- Triple text encoders (CLIP-L, CLIP-G, T5-XXL)");
    println!("- Flow matching loss");
    
    Ok(())
}

/// Train Flux model
async fn train_flux(device: Device) -> Result<()> {
    println!("\n=== Training Flux ===");
    
    // Create Flux model
    let model = ModelFactory::create(
        ModelArchitecture::Flux(FluxVariant::Dev),
        device.clone()
    ).await?;
    
    // Flux-specific pipeline configuration
    let pipeline_config = PipelineConfig {
        training_config: TrainingConfig {
            learning_rate: 1e-5,
            batch_size: 1,
            num_epochs: 5,
            gradient_accumulation_steps: 16,
            ..Default::default()
        },
        device: device.clone(),
        dtype: candle_core::DType::BF16,
        gradient_checkpointing: true,
        loss_type: "mse".to_string(),
        flow_matching: true,
        turbo_training: true, // Flux supports turbo training
        ..Default::default()
    };
    
    // Create Flux pipeline
    let pipeline = PipelineFactory::create(
        ModelArchitecture::Flux(FluxVariant::Dev),
        pipeline_config
    )?;
    
    println!("Flux pipeline created");
    println!("- Pure transformer architecture");
    println!("- Packed latent format");
    println!("- T5-XXL text encoder");
    println!("- Guidance conditioning");
    
    Ok(())
}

/// Example showing pipeline-specific features
#[allow(dead_code)]
async fn demonstrate_pipeline_features() -> Result<()> {
    let device = Device::cuda_if_available()?;
    
    // Show different noise scheduling
    println!("\n=== Noise Scheduling by Model ===");
    println!("SD1.5/SD2: Linear beta schedule");
    println!("SDXL: Scaled linear beta schedule");
    println!("SD3: Flow matching (linear interpolation)");
    println!("Flux: Shifted sigmoid flow");
    println!("PixArt: Cosine beta schedule");
    println!("AuraFlow: Continuous normalizing flow");
    
    // Show different text encoding
    println!("\n=== Text Encoding by Model ===");
    println!("SD1.5: CLIP ViT-L/14 (768 dim)");
    println!("SDXL: CLIP ViT-L + CLIP ViT-G (2048 dim combined)");
    println!("SD3: CLIP-L + CLIP-G + T5-XXL (6144 dim combined)");
    println!("Flux: T5-XXL only (4096 dim)");
    println!("PixArt: T5 (4096 dim, up to 300 tokens)");
    println!("AuraFlow: T5 + CLIP (separate streams)");
    
    // Show different loss functions
    println!("\n=== Loss Functions by Model ===");
    println!("SD1.5/SDXL: Noise prediction (epsilon)");
    println!("SD3/Flux: Velocity prediction (flow matching)");
    println!("PixArt: V-parameterization");
    println!("AuraFlow: Flow vector field");
    
    Ok(())
}