//! Real SD3.5 LoRA training implementation - NO MOCKS

use eridiffusion_core::{Device, ModelArchitecture, Result, context};
use eridiffusion_models::{ModelFactory, vae::VAEFactory};
use eridiffusion_networks::{LoKrConfig, NetworkFactory};
use eridiffusion_training::{
    TrainingConfig, PipelineConfig, PipelineFactory,
    Trainer, TrainerConfig, OptimizerConfig,
    create_optimizer, create_lr_scheduler,
    callbacks::{CheckpointCallback, SampleCallback, ProgressCallback},
};
use eridiffusion_data::{DatasetConfig, DataLoader, LatentCache};
use eridiffusion_inference::{InferencePipeline, SamplerType};
use candle_core::DType;
use candle_nn::VarBuilder;
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== SD3.5 LoRA Training on eri1024 Dataset ===\n");
    
    // Configuration from eri1024.yaml
    let config = Config {
        name: "rachv1Sd3",
        model_path: "stabilityai/stable-diffusion-3.5-large",
        dataset_path: "/home/alex/diffusers-rs/datasets/40_woman",
        output_dir: PathBuf::from("output/rachv1Sd3"),
        trigger_word: "eri1024",
        batch_size: 4,
        gradient_accumulation: 1,
        total_steps: 4000,
        save_every: 250,
        sample_every: 250,
        learning_rate: 5e-5,
        use_ema: true,
        ema_decay: 0.99,
        dtype: DType::BF16,
        max_grad_norm: 0.01,
        sample_prompts: vec![
            "eri1024 nude woman with red hair, playing chess at the park, bomb going off in the background",
            "rachv1Sd3 ,a nude woman holding a coffee cup, in a beanie, sitting at a cafe",
            "rachv1Sd3a ,high-resolution photograph featuring a nude woman lying on her back on a beige-colored surface",
            "eri1024 Close-up shot of a fashion model with flawless makeup and a sharp gaze, captured in high-detail",
            "eri1024woman playing the guitar, on stage, singing a song, laser lights, punk rocker",
            "eri1024 photo of a woman white background, medium shot, modeling clothing, studio lighting, white backdrop",
            "eri1024a nude woman holding a sign abover her head that says, 'this is a sign'",
            "eri1024a woman, in a post apocalyptic world, with a shotgun, in a leather jacket, in a desert, with a motorcycle",
        ],
    };
    
    // Initialize CUDA device
    let device = Device::cuda(0)?;
    println!("Using device: CUDA 0");
    
    // Check GPU memory
    cuda_check_memory()?;
    
    // Create output directory
    std::fs::create_dir_all(&config.output_dir)?;
    
    // Step 1: Load SD3.5 model
    println!("\n[1/7] Loading SD3.5 model...");
    let mut model = ModelFactory::create(ModelArchitecture::SD35, device.clone()).await?;
    model.load_pretrained(&PathBuf::from(&config.model_path)).await?;
    println!("✓ Model loaded: {} parameters", model.num_parameters());
    
    // Step 2: Create VAE for latent encoding
    println!("\n[2/7] Creating VAE for latent encoding...");
    let vae_path = PathBuf::from(&config.model_path).join("vae");
    let vae_vb = VarBuilder::from_pretrained(&vae_path, config.dtype, &device)?;
    let vae = VAEFactory::create(ModelArchitecture::SD35, vae_vb)?;
    println!("✓ VAE loaded: 16-channel latents");
    
    // Step 3: Setup latent cache
    println!("\n[3/7] Setting up latent cache...");
    let cache_dir = config.output_dir.join("latent_cache");
    let latent_cache = LatentCache::new(
        cache_dir.clone(),
        ModelArchitecture::SD35,
        device.clone(),
        Some(vae_path),
    )?;
    println!("✓ Latent cache initialized at: {:?}", cache_dir);
    
    // Step 4: Create LoKr network
    println!("\n[4/7] Creating LoKr network...");
    let lokr_config = LoKrConfig {
        rank: 64,
        alpha: 64.0,
        factor: 4,
        full_rank: true,
        dropout: 0.0,
        target_modules: vec![
            "transformer_blocks.*.attn.to_q".to_string(),
            "transformer_blocks.*.attn.to_v".to_string(),
            "transformer_blocks.*.attn.to_k".to_string(),
            "transformer_blocks.*.attn.to_out.0".to_string(),
        ],
        decompose_both: true,
        use_tucker: false,
    };
    
    let network = NetworkFactory::create_lokr(lokr_config, &model)?;
    println!("✓ LoKr network created: {} parameters", network.num_parameters());
    
    // Step 5: Setup data pipeline
    println!("\n[5/7] Setting up data pipeline...");
    let dataset_config = DatasetConfig {
        data_dir: PathBuf::from(&config.dataset_path),
        caption_ext: Some("txt".to_string()),
        resolution: 1024,
        center_crop: false,
        random_flip: false,
        caption_dropout: 0.0,
        shuffle_tokens: false,
        cache_latents: true,
        cache_dir: Some(cache_dir),
        trigger_word: Some(config.trigger_word.clone()),
        ..Default::default()
    };
    
    let dataloader = DataLoader::new(dataset_config)?;
    println!("✓ Dataset loaded: {} images", dataloader.len());
    
    // Pre-encode latents
    println!("\nPre-encoding latents (this may take a few minutes)...");
    preprocess_latents(&dataloader, &latent_cache).await?;
    
    // Step 6: Create training pipeline
    println!("\n[6/7] Creating training pipeline...");
    let pipeline_config = PipelineConfig {
        training_config: TrainingConfig {
            learning_rate: config.learning_rate,
            batch_size: config.batch_size,
            num_epochs: 1, // Using steps instead
            gradient_accumulation_steps: config.gradient_accumulation,
            optimizer_type: "adamw8bit".to_string(),
            mixed_precision: true,
            gradient_checkpointing: true,
            seed: Some(42),
            ..Default::default()
        },
        device: device.clone(),
        dtype: config.dtype,
        use_ema: config.use_ema,
        ema_decay: config.ema_decay,
        gradient_checkpointing: true,
        loss_type: "mse".to_string(),
        flow_matching: true,
        snr_gamma: Some(5.0),
        v_parameterization: false,
        noise_offset: 0.0,
        linear_timesteps: true,
        ..Default::default()
    };
    
    let pipeline = PipelineFactory::create(ModelArchitecture::SD35, pipeline_config)?;
    
    // Create optimizer
    let optimizer_config = OptimizerConfig {
        learning_rate: config.learning_rate,
        weight_decay: 0.01,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8,
    };
    
    let optimizer = create_optimizer("adamw8bit", optimizer_config, &network.trainable_parameters())?;
    
    // Create scheduler
    let scheduler = create_lr_scheduler(
        "constant",
        config.learning_rate,
        config.total_steps,
        0, // no warmup
    )?;
    
    // Step 7: Setup trainer with callbacks
    println!("\n[7/7] Setting up trainer...");
    let trainer_config = TrainerConfig {
        model,
        network: Some(network),
        pipeline: Box::new(pipeline),
        dataloader,
        optimizer,
        scheduler: Some(scheduler),
        output_dir: config.output_dir.clone(),
        checkpoint_steps: config.save_every,
        logging_steps: 50,
        sample_steps: config.sample_every,
        max_steps: Some(config.total_steps),
        gradient_clip_val: Some(config.max_grad_norm),
        mixed_precision: true,
        compile_model: false, // Set to true if using torch compile
        ..Default::default()
    };
    
    let mut trainer = Trainer::new(trainer_config)?;
    
    // Add callbacks
    trainer.add_callback(Box::new(CheckpointCallback::new(
        config.output_dir.clone(),
        config.save_every,
        4, // max_step_saves_to_keep
    )));
    
    trainer.add_callback(Box::new(SampleCallback::new(
        config.sample_prompts.clone(),
        config.sample_every,
        config.output_dir.join("samples"),
        SamplerType::FlowMatch,
        25, // steps
        4.0, // guidance scale
    )));
    
    trainer.add_callback(Box::new(ProgressCallback::new()));
    
    // Start training
    println!("\n{}", "=".repeat(60));
    println!("Starting training for {} steps", config.total_steps);
    println!("Batch size: {}", config.batch_size);
    println!("Learning rate: {}", config.learning_rate);
    println!("Checkpoints every: {} steps", config.save_every);
    println!("Samples every: {} steps", config.sample_every);
    println!("{}\n", "=".repeat(60));
    
    trainer.train().await?;
    
    println!("\n✅ Training completed!");
    println!("Final checkpoint saved to: {:?}", config.output_dir);
    
    Ok(())
}

/// Pre-encode all images to latents
async fn preprocess_latents(dataloader: &DataLoader, latent_cache: &LatentCache) -> Result<()> {
    let total = dataloader.len();
    
    for i in 0..total {
        if i % 10 == 0 {
            println!("  Encoding {}/{}", i + 1, total);
        }
        
        let item = dataloader.dataset().get(i)?;
        let image = item.load_image()?;
        
        // This will encode and cache if not already cached
        latent_cache.get_latents(&item.image_path, &image)?;
    }
    
    println!("✓ All latents encoded and cached");
    Ok(())
}

/// Check CUDA memory
fn cuda_check_memory() -> Result<()> {
    use candle_core::cuda;
    
    if let Ok(device) = cuda::CudaDevice::new(0) {
        let free = device.memory_info()?.0;
        let total = device.memory_info()?.1;
        let used = total - free;
        
        println!("GPU Memory: {:.1} GB used / {:.1} GB total", 
            used as f64 / 1e9, 
            total as f64 / 1e9
        );
        
        if free < 8 * 1024 * 1024 * 1024 { // Less than 8GB free
            println!("WARNING: Low GPU memory, may need to reduce batch size");
        }
    }
    
    Ok(())
}

/// Configuration struct
struct Config {
    name: String,
    model_path: String,
    dataset_path: String,
    output_dir: PathBuf,
    trigger_word: String,
    batch_size: usize,
    gradient_accumulation: usize,
    total_steps: usize,
    save_every: usize,
    sample_every: usize,
    learning_rate: f32,
    use_ema: bool,
    ema_decay: f32,
    dtype: DType,
    max_grad_norm: f32,
    sample_prompts: Vec<String>,
}