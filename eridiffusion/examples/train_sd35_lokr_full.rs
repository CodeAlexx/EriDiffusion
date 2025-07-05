//! Full SD 3.5 LoKr training with real models
//! This example shows how to train LoKr adapters on SD 3.5 using the Candle backend

use eridiffusion_core::{Device, Error, Result};
use eridiffusion_data::{DataLoader, ImageDataset, DatasetConfig, BucketSampler};
use eridiffusion_models::{DiffusionModel, mmdit_with_lokr::{MMDiTWithLoKr, get_sd35_target_modules}};
use eridiffusion_networks::NetworkType;
use eridiffusion_training::{
    Trainer, TrainerConfig, TrainingConfig,
    pipelines::{SD3Pipeline, PipelineConfig},
    optimizer::{AdamWOptimizer, OptimizerConfig, OptimizerType},
};
use candle_core::Tensor;
use candle_nn::VarMap;
use clap::Parser;
use std::path::PathBuf;
use tracing::{info, error};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to SD 3.5 model (safetensors format)
    #[arg(long, default_value = "/home/alex/SwarmUI/Models/diffusion_models/sd3.5_large.safetensors")]
    model_path: PathBuf,
    
    /// Path to dataset directory
    #[arg(long, default_value = "/home/alex/diffusers-rs/datasets/40_woman")]
    dataset_path: PathBuf,
    
    /// Path to configuration file
    #[arg(long, default_value = "/home/alex/diffusers-rs/config/eri1024.yaml")]
    config_path: PathBuf,
    
    /// LoKr rank
    #[arg(long, default_value_t = 64)]
    rank: usize,
    
    /// LoKr alpha
    #[arg(long, default_value_t = 64.0)]
    alpha: f32,
    
    /// Training steps
    #[arg(long, default_value_t = 4000)]
    steps: usize,
    
    /// Batch size
    #[arg(long, default_value_t = 4)]
    batch_size: usize,
    
    /// Learning rate
    #[arg(long, default_value_t = 5e-5)]
    learning_rate: f64,
    
    /// Output directory for checkpoints
    #[arg(long, default_value = "./outputs")]
    output_dir: PathBuf,
    
    /// Sample every N steps
    #[arg(long, default_value_t = 500)]
    sample_every: usize,
    
    /// Save checkpoint every N steps
    #[arg(long, default_value_t = 500)]
    save_every: usize,
}

fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    let args = Args::parse();
    info!("Starting SD 3.5 LoKr training");
    info!("Configuration: {:?}", args);
    
    // Setup device
    let device = Device::cuda_if_available(0)?;
    info!("Using device: {:?}", device);
    
    // Load model with LoKr adapters
    info!("Loading SD 3.5 model from: {:?}", args.model_path);
    let target_modules = get_sd35_target_modules(38); // SD 3.5 Large has 38 layers
    let mut model = MMDiTWithLoKr::new(
        &args.model_path,
        args.rank,
        args.alpha,
        target_modules,
        &device,
    )?;
    info!("Model loaded with {} LoKr adapters", model.get_lokr_parameters().len());
    
    // Setup dataset
    info!("Loading dataset from: {:?}", args.dataset_path);
    let dataset_config = DatasetConfig {
        root_dir: args.dataset_path.clone(),
        caption_ext: "txt".to_string(),
        resolution: 1024,
        center_crop: false,
        random_flip: true,
        cache_latents: true,
        cache_dir: Some("./cache".into()),
    };
    
    let dataset = ImageDataset::new(dataset_config)?;
    info!("Dataset loaded with {} images", dataset.len());
    
    // Create bucket sampler for aspect ratio bucketing
    let bucket_sampler = BucketSampler::new(
        vec![(1024, 1024), (768, 1024), (1024, 768)], // Common resolutions
        args.batch_size,
        true, // shuffle
    );
    
    // Create data loader
    let data_loader = DataLoader::new(
        dataset,
        bucket_sampler,
        args.batch_size,
        true, // drop_last
        4, // num_workers
    )?;
    
    // Setup training pipeline
    let pipeline_config = PipelineConfig {
        model_type: "sd3".to_string(),
        loss_type: "mse".to_string(),
        noise_scheduler: "flow_matching".to_string(),
        learning_rate: args.learning_rate as f32,
        adam_beta1: 0.9,
        adam_beta2: 0.999,
        adam_weight_decay: 0.01,
        adam_epsilon: 1e-8,
        noise_offset: 0.0,
        input_perturbation: 0.0,
        min_snr_gamma: None,
        mean_flow_loss: true,
        enable_xformers: false,
        gradient_checkpointing: true,
        mixed_precision: "bf16".to_string(),
        training_config: Default::default(),
    };
    
    let pipeline = SD3Pipeline::new(pipeline_config)?;
    
    // Setup optimizer
    let lokr_params = model.get_lokr_parameters();
    let varmap = VarMap::new();
    for (i, param) in lokr_params.iter().enumerate() {
        varmap.set_var(format!("lokr_param_{}", i), param.clone())?;
    }
    
    let optimizer_config = OptimizerConfig {
        optimizer_type: OptimizerType::AdamW,
        lr: args.learning_rate,
        weight_decay: 0.01,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8,
        momentum: 0.0,
        use_8bit: false,
    };
    
    let param_refs: Vec<&Tensor> = lokr_params.iter().collect();
    let optimizer = AdamWOptimizer::new(optimizer_config, &param_refs)?;
    
    // Setup trainer
    let trainer_config = TrainerConfig {
        output_dir: args.output_dir.clone(),
        num_train_steps: args.steps,
        checkpointing_steps: args.save_every,
        validation_steps: args.sample_every,
        logging_steps: 100,
        gradient_accumulation_steps: 1,
        mixed_precision: "bf16".to_string(),
        max_grad_norm: 1.0,
        resume_from_checkpoint: None,
        seed: 42,
        use_wandb: false,
        wandb_project: None,
        compile_model: false,
        dataloader_num_workers: 4,
        enable_progress_bar: true,
    };
    
    let mut trainer = Trainer::new(
        trainer_config,
        model,
        optimizer,
        data_loader,
        pipeline,
    )?;
    
    // Training loop
    info!("Starting training for {} steps", args.steps);
    let start_time = std::time::Instant::now();
    
    trainer.train()?;
    
    let elapsed = start_time.elapsed();
    info!("Training completed in {:.2} minutes", elapsed.as_secs_f32() / 60.0);
    
    // Save final checkpoint
    let final_path = args.output_dir.join("sd35_lokr_final.safetensors");
    trainer.save_checkpoint(&final_path)?;
    info!("Final checkpoint saved to: {:?}", final_path);
    
    Ok(())
}

/// Generate sample images during training
fn generate_samples(
    model: &MMDiTWithLoKr,
    step: usize,
    output_dir: &PathBuf,
    device: &Device,
) -> Result<()> {
    info!("Generating samples at step {}", step);
    
    let prompts = vec![
        "eri, a woman with curly hair, high resolution, 4k",
        "eri woman standing, professional photo",
        "portrait of eri, detailed face, beautiful lighting",
    ];
    
    // Mock text embeddings (in real impl would use actual encoders)
    let batch_size = prompts.len();
    let encoder_hidden_states = Tensor::randn(0.0f32, 0.1f32, &[batch_size, 77, 6144], device)?;
    let pooled_projections = Tensor::randn(0.0f32, 0.1f32, &[batch_size, 2048], device)?;
    
    // Generate latents using flow matching
    let num_steps = 50;
    let mut latents = Tensor::randn(0.0f32, 1.0f32, &[batch_size, 16, 128, 128], device)?;
    
    for i in 0..num_steps {
        let t = 1.0 - (i as f32 / (num_steps - 1) as f32);
        let timestep = Tensor::new(&[t], device)?;
        
        // Model forward pass
        let velocity = model.forward(
            &latents.view(),
            &timestep.view(),
            &encoder_hidden_states.view(),
            Some(&pooled_projections.view()),
        )?;
        
        // Update latents
        let dt = 1.0 / num_steps as f32;
        latents = (latents - velocity * dt)?;
    }
    
    // Save latents info (in real impl would decode to images)
    for (i, prompt) in prompts.iter().enumerate() {
        let sample_path = output_dir.join(format!("sample_step_{}_prompt_{}.txt", step, i));
        let info = format!(
            "Step: {}\nPrompt: {}\nLatent shape: {:?}\nLatent mean: {:.6}\nLatent std: {:.6}",
            step,
            prompt,
            latents.shape(),
            latents.mean_all()?.to_scalar::<f32>()?,
            latents.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?,
        );
        std::fs::write(&sample_path, info)?;
    }
    
    info!("Samples saved to: {:?}", output_dir);
    Ok(())
}