//! SD 3.5 LoRA training binary

use eridiffusion_core::{Device, Result};
use eridiffusion_models::{SD3Model, TextEncoder};
use eridiffusion_networks::{LoRANetwork, LoRAConfig};
use eridiffusion_training::{
    TrainingConfig, Trainer, LossType, LossConfig, create_loss,
    optimizers::{AdamWOptimizer, OptimizerConfig},
    schedulers::{CosineAnnealingLR, SchedulerConfig},
    SD35Trainer, SD35TrainingConfig, SD35ModelVariant,
};
use eridiffusion_data::{Dataset, DataLoader, DataLoaderConfig};
use clap::Parser;
use std::path::PathBuf;
use tracing::{info, warn};
use candle_core;
use candle_nn;

#[derive(Parser, Debug)]
#[command(author, version, about = "Train SD 3.5 with LoRA", long_about = None)]
struct Args {
    /// Path to the SD 3.5 model directory or file
    #[arg(long)]
    model_path: PathBuf,
    
    /// Path to CLIP-L model
    #[arg(long)]
    clip_l_path: PathBuf,
    
    /// Path to CLIP-G model
    #[arg(long)]
    clip_g_path: PathBuf,
    
    /// Path to T5-XXL model
    #[arg(long)]
    t5_path: PathBuf,
    
    /// Path to VAE model
    #[arg(long)]
    vae_path: PathBuf,
    
    /// Path to the training data directory
    #[arg(long)]
    data_path: PathBuf,
    
    /// Output directory for checkpoints
    #[arg(long, default_value = "./output")]
    output_dir: PathBuf,
    
    /// Model variant (medium, large, large-turbo)
    #[arg(long, default_value = "medium")]
    variant: String,
    
    /// LoRA rank
    #[arg(long, default_value_t = 32)]
    lora_rank: usize,
    
    /// LoRA alpha
    #[arg(long, default_value_t = 32.0)]
    lora_alpha: f32,
    
    /// LoRA dropout
    #[arg(long, default_value_t = 0.0)]
    lora_dropout: f32,
    
    /// Include MLP layers in LoRA
    #[arg(long)]
    lora_include_mlp: bool,
    
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
    #[arg(long, default_value_t = 4)]
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
    
    /// Max gradient norm
    #[arg(long, default_value_t = 1.0)]
    max_grad_norm: f32,
    
    /// Resume from checkpoint
    #[arg(long)]
    resume_from: Option<PathBuf>,
    
    /// Train text encoders
    #[arg(long)]
    train_text_encoders: bool,
    
    /// Text encoder learning rate multiplier
    #[arg(long, default_value_t = 0.1)]
    text_encoder_lr_multiplier: f32,
    
    /// CFG scale for training
    #[arg(long, default_value_t = 7.0)]
    cfg_scale: f32,
    
    /// VAE scaling factor
    #[arg(long, default_value_t = 1.5305)]
    vae_scaling_factor: f32,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    let args = Args::parse();
    
    info!("Starting SD 3.5 LoRA training");
    
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
        "medium" => SD35ModelVariant::Medium,
        "large" => SD35ModelVariant::Large,
        "large-turbo" => SD35ModelVariant::LargeTurbo,
        _ => {
            warn!("Unknown variant {}, defaulting to Medium", args.variant);
            SD35ModelVariant::Medium
        }
    };
    
    // Create output directory
    std::fs::create_dir_all(&args.output_dir)?;
    
    // Load SD 3.5 model
    info!("Loading SD 3.5 {:?} model from {:?}", variant, args.model_path);
    let is_sd35 = true;
    // Convert Device to candle Device
    let candle_device = match &device {
        Device::Cpu => candle_core::Device::Cpu,
        Device::Cuda(id) => candle_core::Device::new_cuda(*id)?,
        Device::Metal(_) => return Err(anyhow::anyhow!("Metal device not supported for SD 3.5").into()),
    };
    
    let mut model = SD3Model::new(
        candle_nn::VarBuilder::from_safetensors(
            vec![args.model_path.to_str().unwrap()],
            candle_core::DType::F32,
            &candle_device,
        )?,
        is_sd35,
    )?;
    
    // Load text encoders
    info!("Loading text encoders");
    let clip_l = TextEncoder::load_clip(&args.clip_l_path, &device).await?;
    let clip_g = TextEncoder::load_clip(&args.clip_g_path, &device).await?;
    let t5 = TextEncoder::load_t5(&args.t5_path, &device).await?;
    
    // Create LoRA network
    info!("Creating LoRA network with rank {}", args.lora_rank);
    let target_modules = SD35Trainer::get_target_modules(args.lora_include_mlp);
    let lora_config = LoRAConfig {
        rank: args.lora_rank,
        alpha: args.lora_alpha,
        dropout: args.lora_dropout,
        target_modules,
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
        max_grad_norm: Some(args.max_grad_norm),
        dataloader_num_workers: 4,
        seed: Some(42),
        resume_from_checkpoint: args.resume_from,
        push_to_hub: false,
        hub_model_id: None,
        hub_token: None,
    };
    
    // Create SD 3.5 specific config
    let sd35_config = SD35TrainingConfig {
        model_variant: variant,
        gradient_checkpointing: args.gradient_checkpointing,
        train_text_encoders: args.train_text_encoders,
        text_encoder_lr_multiplier: args.text_encoder_lr_multiplier,
        vae_scaling_factor: args.vae_scaling_factor,
        num_inference_steps: 28,
        mixed_precision: args.mixed_precision,
        cfg_scale: args.cfg_scale,
        lora_include_mlp: args.lora_include_mlp,
    };
    
    // Create trainer with SD 3.5 specific settings
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
    
    // Add text encoders if training them
    if args.train_text_encoders {
        trainer.add_model("clip_l", clip_l)?;
        trainer.add_model("clip_g", clip_g)?;
        trainer.add_model("t5", t5)?;
    }
    
    // Store SD 3.5 config
    trainer.set_config("sd35_config", sd35_config)?;
    
    // Start training
    info!("Starting training for {} epochs", args.num_epochs);
    info!("Model variant: {:?}", variant);
    info!("LoRA rank: {}, alpha: {}", args.lora_rank, args.lora_alpha);
    info!("Target modules: {:?}", lora_config.target_modules);
    
    trainer.train().await?;
    
    info!("Training completed successfully!");
    
    // Save final checkpoint
    let final_checkpoint_path = args.output_dir.join("final_checkpoint");
    trainer.save_checkpoint(&final_checkpoint_path)?;
    info!("Saved final checkpoint to {:?}", final_checkpoint_path);
    
    Ok(())
}