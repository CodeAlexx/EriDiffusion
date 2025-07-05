//! Example script for training SD3.5 LoKr on the 40_woman dataset

use eridiffusion_core::{Result, Error, Device, ModelArchitecture};
use eridiffusion_data::{WomanDataset, WomanDatasetConfig, DataLoader, BucketSampler, LatentCache};
use eridiffusion_models::{DiffusionModel, VAE, TextEncoder};
use eridiffusion_networks::{LoKrConfig, NetworkAdapterFactory};
use eridiffusion_training::{
    TrainerConfig, DiffusionTrainer, PipelineFactory, PipelineConfig,
    OptimizerConfig, create_optimizer, create_lr_scheduler,
};
use std::path::PathBuf;
use std::sync::Arc;
use clap::Parser;
use tracing::{info, error};

/// SD3.5 LoKr training example
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to 40_woman dataset
    #[arg(long, default_value = "datasets/40_woman")]
    dataset_path: PathBuf,
    
    /// Path to SD3.5 model weights
    #[arg(long)]
    model_path: PathBuf,
    
    /// Path to VAE weights
    #[arg(long)]
    vae_path: PathBuf,
    
    /// Path to CLIP-L weights
    #[arg(long)]
    clip_l_path: PathBuf,
    
    /// Path to CLIP-G weights
    #[arg(long)]
    clip_g_path: PathBuf,
    
    /// Path to T5-XXL weights
    #[arg(long)]
    t5_path: PathBuf,
    
    /// Output directory for checkpoints
    #[arg(long, default_value = "outputs/sd35_lokr")]
    output_dir: PathBuf,
    
    /// Learning rate
    #[arg(long, default_value = "1e-4")]
    learning_rate: f32,
    
    /// Batch size
    #[arg(long, default_value = "1")]
    batch_size: usize,
    
    /// Number of epochs
    #[arg(long, default_value = "100")]
    num_epochs: usize,
    
    /// LoKr rank
    #[arg(long, default_value = "16")]
    rank: usize,
    
    /// LoKr alpha
    #[arg(long, default_value = "16.0")]
    alpha: f32,
    
    /// Resume from checkpoint
    #[arg(long)]
    resume: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    let args = Args::parse();
    
    info!("Starting SD3.5 LoKr training");
    info!("Dataset: {}", args.dataset_path.display());
    info!("Model: {}", args.model_path.display());
    
    // Setup device
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    info!("Using device: {:?}", device);
    
    // Load dataset
    info!("Loading 40_woman dataset...");
    let dataset_config = WomanDatasetConfig {
        root_dir: args.dataset_path,
        resolution: 1024,
        repeats: 20,
        cache_latents: true,
    };
    let dataset = WomanDataset::new(dataset_config)?;
    info!("Dataset loaded: {} base images, {} total samples", 
        dataset.base_len(), dataset.len());
    
    // Create VAE for latent caching
    info!("Loading VAE...");
    let vae = load_vae(&args.vae_path, &device).await?;
    let vae = Arc::new(vae);
    
    // Create latent cache
    let cache_dir = PathBuf::from(".cache/latents");
    let latent_cache = Arc::new(LatentCache::new(cache_dir, vae.clone())?);
    
    // Create text encoders
    info!("Loading text encoders...");
    let clip_l = Arc::new(load_clip_l(&args.clip_l_path, &device).await?);
    let clip_g = Arc::new(load_clip_g(&args.clip_g_path, &device).await?);
    let t5_xxl = Arc::new(load_t5_xxl(&args.t5_path, &device).await?);
    
    // Create dataloader with aspect ratio bucketing
    let buckets = vec![
        (1024, 1024),  // Square
        (1152, 896),   // 4:3
        (896, 1152),   // 3:4
        (1216, 832),   // 3:2
        (832, 1216),   // 2:3
    ];
    let sampler = BucketSampler::new(buckets, args.batch_size, true);
    let mut dataloader = DataLoader::new(
        dataset,
        sampler,
        args.batch_size,
        true,  // drop_last
        4,     // num_workers
    )?;
    dataloader = dataloader
        .with_latent_cache(latent_cache)
        .with_text_encoder(clip_l.clone());
    
    // Initialize dataloader
    dataloader.initialize().await?;
    
    // Load SD3.5 model
    info!("Loading SD3.5 model...");
    let model = load_sd35_model(&args.model_path, &device).await?;
    
    // Create LoKr adapter
    info!("Creating LoKr adapter...");
    let lokr_config = LoKrConfig {
        rank: args.rank,
        alpha: args.alpha,
        dropout: 0.0,
        target_modules: vec![
            "to_q".to_string(),
            "to_k".to_string(),
            "to_v".to_string(),
            "to_out.0".to_string(),
        ],
        decompose_factor: 4,
        use_tucker: true,
    };
    let network_adapter = NetworkAdapterFactory::create_lokr(
        lokr_config,
        model.as_ref(),
    )?;
    
    // Create training pipeline
    let pipeline_config = PipelineConfig {
        model_family: "sd3".to_string(),
        prediction_type: "flow_matching".to_string(),
        loss_type: "mse".to_string(),
        snr_gamma: None,
        noise_offset: 0.0,
        input_perturbation: 0.0,
        min_snr_gamma: None,
        use_ema: false,
        prior_loss_weight: 0.0,
        gradient_checkpointing: false,
        mixed_precision: "fp16".to_string(),
        enable_xformers: true,
        mean_flow_loss: true,
        training_config: Default::default(),
    };
    let pipeline = PipelineFactory::create(ModelArchitecture::SD35, pipeline_config)?
        .with_vae(vae.clone())
        .with_text_encoders(clip_l, clip_g, t5_xxl);
    
    // Create optimizer
    let optimizer_config = OptimizerConfig {
        optimizer_type: "adamw".to_string(),
        learning_rate: args.learning_rate as f64,
        weight_decay: 0.01,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8,
    };
    let mut optimizer = create_optimizer(optimizer_config)?;
    
    // Create trainer config
    let trainer_config = TrainerConfig {
        output_dir: args.output_dir,
        num_epochs: args.num_epochs,
        batch_size: args.batch_size,
        learning_rate: args.learning_rate,
        logging_steps: 10,
        validation_steps: 100,
        checkpointing_steps: 500,
        save_total_limit: Some(5),
        mixed_precision: "fp16".to_string(),
        gradient_accumulation_steps: 1,
        max_grad_norm: 1.0,
    };
    
    // Create trainer
    let mut trainer = DiffusionTrainer::new(
        trainer_config,
        model,
        vae,
        Box::new(DummyTextEncoder), // Text encoding is handled by pipeline
        network_adapter,
        pipeline,
    )?;
    
    // Resume from checkpoint if requested
    if let Some(checkpoint_path) = args.resume {
        trainer.resume_from_checkpoint(checkpoint_path).await?;
    }
    
    // Training loop
    info!("Starting training...");
    for epoch in 0..args.num_epochs {
        info!("Epoch {}/{}", epoch + 1, args.num_epochs);
        
        let epoch_loss = trainer.train_epoch(&mut dataloader, &mut optimizer).await?;
        
        info!("Epoch {} completed, average loss: {:.4}", epoch + 1, epoch_loss);
    }
    
    info!("Training completed!");
    Ok(())
}

/// Load VAE model
async fn load_vae(path: &PathBuf, device: &Device) -> Result<Box<dyn VAE>> {
    // This would load the actual VAE weights
    // For now, return a placeholder
    use eridiffusion_models::candle_sd3_vae::CandleSD3VAE;
    Ok(Box::new(CandleSD3VAE::new(path, device)?))
}

/// Load CLIP-L encoder
async fn load_clip_l(path: &PathBuf, device: &Device) -> Result<Box<dyn TextEncoder>> {
    use eridiffusion_models::candle_clip_encoder::CandleCLIPEncoder;
    Ok(Box::new(CandleCLIPEncoder::new(path, device, "clip-l")?))
}

/// Load CLIP-G encoder
async fn load_clip_g(path: &PathBuf, device: &Device) -> Result<Box<dyn TextEncoder>> {
    use eridiffusion_models::candle_clip_encoder::CandleCLIPEncoder;
    Ok(Box::new(CandleCLIPEncoder::new(path, device, "clip-g")?))
}

/// Load T5-XXL encoder
async fn load_t5_xxl(path: &PathBuf, device: &Device) -> Result<Box<dyn TextEncoder>> {
    use eridiffusion_models::candle_t5_encoder::CandleT5Encoder;
    Ok(Box::new(CandleT5Encoder::new(path, device)?))
}

/// Load SD3.5 model
async fn load_sd35_model(path: &PathBuf, device: &Device) -> Result<Box<dyn DiffusionModel>> {
    use eridiffusion_models::candle_sd3_adapter::CandleSD3Adapter;
    Ok(Box::new(CandleSD3Adapter::new(path, device)?))
}

/// Dummy text encoder for trainer (actual encoding is handled by pipeline)
struct DummyTextEncoder;

impl TextEncoder for DummyTextEncoder {
    fn encode(&self, _prompts: &[String]) -> Result<(Tensor, Option<Tensor>)> {
        Err(Error::NotImplemented("Text encoding handled by pipeline".into()))
    }
    
    fn device(&self) -> &Device {
        &Device::Cpu
    }
}