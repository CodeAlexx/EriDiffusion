//! Flux LoRA training binary

use eridiffusion_core::{Device, Result};
use eridiffusion_models::{FluxModel, FluxVariant};
use eridiffusion_networks::{LoRANetwork, LoRAConfig};
use eridiffusion_training::{
    TrainingConfig, Trainer, LossType, LossConfig, create_loss,
    optimizers::{AdamWOptimizer, OptimizerConfig},
    schedulers::{CosineAnnealingLR, SchedulerConfig},
};
use eridiffusion_data::{Dataset, DataLoader, DataLoaderConfig};
use clap::Parser;
use std::path::PathBuf;
use tracing::{info, warn};

#[derive(Parser, Debug)]
#[command(author, version, about = "Train Flux with LoRA", long_about = None)]
struct Args {
    /// Path to the Flux model directory
    #[arg(long)]
    model_path: PathBuf,
    
    /// Path to the training data directory
    #[arg(long)]
    data_path: PathBuf,
    
    /// Output directory for checkpoints
    #[arg(long, default_value = "./output")]
    output_dir: PathBuf,
    
    /// Flux variant (schnell or dev)
    #[arg(long, default_value = "schnell")]
    variant: String,
    
    /// LoRA rank
    #[arg(long, default_value_t = 32)]
    lora_rank: usize,
    
    /// LoRA alpha
    #[arg(long, default_value_t = 32.0)]
    lora_alpha: f32,
    
    /// Learning rate
    #[arg(long, default_value_t = 1e-4)]
    learning_rate: f64,
    
    /// Batch size
    #[arg(long, default_value_t = 1)]
    batch_size: usize,
    
    /// Number of epochs
    #[arg(long, default_value_t = 100)]
    num_epochs: usize,
    
    /// Gradient accumulation steps
    #[arg(long, default_value_t = 1)]
    gradient_accumulation_steps: usize,
    
    /// Mixed precision training
    #[arg(long)]
    mixed_precision: bool,
    
    /// Gradient checkpointing
    #[arg(long)]
    gradient_checkpointing: bool,
    
    /// Device (cpu, cuda, mps)
    #[arg(long, default_value = "cuda")]
    device: String,
    
    /// Save every N steps
    #[arg(long, default_value_t = 1000)]
    save_steps: usize,
    
    /// Validation every N steps
    #[arg(long, default_value_t = 100)]
    validation_steps: usize,
    
    /// Resume from checkpoint
    #[arg(long)]
    resume_from: Option<PathBuf>,
    
    /// Text encoder learning rate multiplier
    #[arg(long, default_value_t = 0.1)]
    text_encoder_lr_multiplier: f32,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    let args = Args::parse();
    
    info!("Starting Flux LoRA training");
    
    // Parse device
    let device = match args.device.as_str() {
        "cpu" => Device::Cpu,
        "cuda" => Device::Cuda(0),
        "mps" => Device::Metal(0),
        _ => {
            warn!("Unknown device {}, defaulting to CPU", args.device);
            Device::Cpu
        }
    };
    
    // Parse variant
    let variant = match args.variant.as_str() {
        "schnell" => FluxVariant::Schnell,
        "dev" => FluxVariant::Dev,
        _ => {
            warn!("Unknown variant {}, defaulting to Schnell", args.variant);
            FluxVariant::Schnell
        }
    };
    
    // Create output directory
    std::fs::create_dir_all(&args.output_dir)?;
    
    // Load Flux model
    info!("Loading Flux {:?} model from {:?}", variant, args.model_path);
    let mut model = FluxModel::new(variant, &device)?;
    model.load_pretrained(&args.model_path).await?;
    
    // Create LoRA network
    info!("Creating LoRA network with rank {}", args.lora_rank);
    let lora_config = LoRAConfig {
        rank: args.lora_rank,
        alpha: args.lora_alpha,
        dropout: 0.0,
        target_modules: vec![
            // Target attention layers in double blocks
            "double_blocks.*.img_attn.qkv".to_string(),
            "double_blocks.*.img_attn.proj".to_string(),
            "double_blocks.*.txt_attn.qkv".to_string(),
            "double_blocks.*.txt_attn.proj".to_string(),
            // Target attention layers in single blocks
            "single_blocks.*.linear1".to_string(),
            "single_blocks.*.linear2".to_string(),
        ],
    };
    
    let lora_network = LoRANetwork::new(lora_config, &device)?;
    
    // Apply LoRA to model
    lora_network.apply_to_model(&mut model)?;
    
    // Create dataset
    info!("Loading dataset from {:?}", args.data_path);
    let dataset = Dataset::from_directory(&args.data_path)?;
    let dataloader_config = DataLoaderConfig {
        batch_size: args.batch_size,
        shuffle: true,
        num_workers: 4,
        pin_memory: true,
        drop_last: true,
        prefetch_factor: Some(2),
    };
    let dataloader = DataLoader::new(dataset, dataloader_config)?;
    
    // Create optimizer
    let optimizer_config = OptimizerConfig {
        learning_rate: args.learning_rate,
        weight_decay: Some(0.01),
        betas: Some((0.9, 0.999)),
        eps: Some(1e-8),
    };
    let optimizer = AdamWOptimizer::new(optimizer_config);
    
    // Create scheduler
    let scheduler_config = SchedulerConfig {
        num_warmup_steps: Some(500),
        num_training_steps: None,
        num_cycles: Some(1),
        last_epoch: -1,
    };
    let scheduler = CosineAnnealingLR::new(scheduler_config);
    
    // Create loss function
    let loss_config = LossConfig::default();
    let loss_fn = create_loss(LossType::FlowMatching, loss_config)?;
    
    // Create training config
    let training_config = TrainingConfig {
        output_dir: args.output_dir.clone(),
        num_train_epochs: args.num_epochs,
        gradient_accumulation_steps: args.gradient_accumulation_steps,
        mixed_precision: args.mixed_precision,
        gradient_checkpointing: args.gradient_checkpointing,
        save_steps: args.save_steps,
        validation_steps: args.validation_steps,
        logging_steps: 10,
        max_grad_norm: Some(1.0),
        dataloader_num_workers: 4,
        seed: Some(42),
        resume_from_checkpoint: args.resume_from,
        push_to_hub: false,
        hub_model_id: None,
        hub_token: None,
    };
    
    // Create trainer
    let mut trainer = Trainer::new(
        model,
        training_config,
        dataloader,
        optimizer,
        scheduler,
        loss_fn,
        device,
    )?;
    
    // Add LoRA network to trainer for saving
    trainer.add_network("lora", lora_network)?;
    
    // Start training
    info!("Starting training for {} epochs", args.num_epochs);
    trainer.train().await?;
    
    info!("Training completed successfully!");
    
    Ok(())
}