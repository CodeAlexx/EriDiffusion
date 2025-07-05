//! Train SD3.5 LoKr on the eri1024 dataset

use eridiffusion_core::{Device, ModelArchitecture, Result, context};
use eridiffusion_models::ModelFactory;
use eridiffusion_networks::{LoKrConfig, NetworkFactory};
use eridiffusion_training::{
    TrainingConfig, PipelineConfig, PipelineFactory,
    Trainer, TrainerConfig, OptimizerConfig,
    create_optimizer, create_lr_scheduler,
};
use eridiffusion_data::{DatasetConfig, DataLoader};
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<()> {
    // Parse the YAML config
    let config = TrainingConfigSD3 {
        name: "rachv1Sd3".to_string(),
        trigger_word: "eri1024".to_string(),
        dataset_path: PathBuf::from("/home/alex/diffusers-rs/datasets/40_woman"),
        model_name: "stabilityai/stable-diffusion-3.5-large".to_string(),
        output_dir: PathBuf::from("./output/rachv1Sd3"),
        batch_size: 4,
        gradient_accumulation: 1,
        total_steps: 4000,
        save_every: 250,
        sample_every: 250,
        learning_rate: 5e-5,
        use_ema: true,
        ema_decay: 0.99,
    };
    
    println!("Starting SD3.5 LoKr training on eri1024 dataset");
    println!("Configuration:");
    println!("  Model: {}", config.model_name);
    println!("  Dataset: {:?}", config.dataset_path);
    println!("  Trigger word: {}", config.trigger_word);
    println!("  Batch size: {}", config.batch_size);
    println!("  Total steps: {}", config.total_steps);
    
    // Initialize device
    let device = Device::cuda(0)?;
    println!("Using device: {:?}", device);
    
    // Create model
    println!("\nLoading SD3.5 model...");
    let mut model = ModelFactory::create(ModelArchitecture::SD35, device.clone()).await
        .context("Failed to create SD3.5 model")?;
    
    // Load pretrained weights
    model.load_pretrained(&PathBuf::from(&config.model_name)).await
        .context("Failed to load pretrained weights")?;
    
    // Create LoKr adapter (lokr_full_rank with factor 4)
    println!("\nCreating LoKr network...");
    let lokr_config = LoKrConfig {
        rank: 64,           // linear: 64 from config
        alpha: 64.0,        // linear_alpha: 64
        factor: 4,          // lokr_factor: 4
        full_rank: true,    // lokr_full_rank: true
        dropout: 0.0,
        target_modules: vec![
            "to_q".to_string(),
            "to_v".to_string(),
            "to_k".to_string(),
            "to_out.0".to_string(),
        ],
        decompose_both: true,
        use_tucker: false,
    };
    
    let network = NetworkFactory::create_lokr(lokr_config, &model)?;
    println!("LoKr network created with {} parameters", network.num_parameters());
    
    // Create pipeline configuration for SD3
    let pipeline_config = PipelineConfig {
        training_config: TrainingConfig {
            learning_rate: config.learning_rate,
            batch_size: config.batch_size,
            num_epochs: 1, // We use steps instead
            gradient_accumulation_steps: config.gradient_accumulation,
            optimizer_type: "adamw8bit".to_string(),
            mixed_precision: true,
            gradient_checkpointing: true,
            seed: Some(42),
            ..Default::default()
        },
        device: device.clone(),
        dtype: candle_core::DType::BF16, // bf16 as specified
        use_ema: config.use_ema,
        ema_decay: config.ema_decay,
        gradient_checkpointing: true,
        loss_type: "mse".to_string(),
        flow_matching: true, // SD3 uses flow matching
        snr_gamma: Some(5.0),
        v_parameterization: false,
        noise_offset: 0.0,
        linear_timesteps: true, // from config
        ..Default::default()
    };
    
    // Create model-specific pipeline
    let pipeline = PipelineFactory::create(ModelArchitecture::SD35, pipeline_config)?;
    
    // Setup data loader
    println!("\nSetting up data loader...");
    let dataset_config = DatasetConfig {
        data_dir: config.dataset_path.clone(),
        caption_column: "txt".to_string(),
        caption_ext: Some("txt".to_string()),
        resolution: 1024,
        center_crop: false,
        random_flip: false,
        caption_dropout: 0.0, // 0% dropout as specified
        shuffle_tokens: false,
        cache_latents: true,
        trigger_word: Some(config.trigger_word.clone()),
        ..Default::default()
    };
    
    let dataloader = DataLoader::new(dataset_config)?;
    println!("Dataset loaded with {} images", dataloader.len());
    
    // Create optimizer
    let optimizer_config = OptimizerConfig {
        learning_rate: config.learning_rate,
        weight_decay: 0.01,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8,
    };
    
    let optimizer = create_optimizer("adamw8bit", optimizer_config, &network.trainable_parameters())?;
    
    // Create learning rate scheduler
    let scheduler = create_lr_scheduler(
        "constant",
        config.learning_rate,
        config.total_steps,
        0, // no warmup specified
    )?;
    
    // Create trainer configuration
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
        gradient_clip_val: Some(0.01), // max_grad_norm from config
        ..Default::default()
    };
    
    // Add sample prompts
    let sample_prompts = vec![
        "eri1024 nude woman with red hair, playing chess at the park, bomb going off in the background",
        "eri1024 a nude woman holding a coffee cup, in a beanie, sitting at a cafe",
        "eri1024 woman playing the guitar, on stage, singing a song, laser lights, punk rocker",
        "eri1024 photo of a woman white background, medium shot, modeling clothing, studio lighting",
    ];
    
    // Create trainer
    let mut trainer = Trainer::new(trainer_config)?;
    trainer.set_sample_prompts(sample_prompts);
    
    // Training callbacks
    trainer.add_callback(Box::new(LoggingCallback::new()));
    trainer.add_callback(Box::new(SampleCallback::new(config.sample_every)));
    
    // Start training
    println!("\nStarting training...");
    println!("Total steps: {}", config.total_steps);
    println!("Saving every: {} steps", config.save_every);
    println!("Sampling every: {} steps", config.sample_every);
    
    let start_time = std::time::Instant::now();
    
    trainer.train().await?;
    
    let elapsed = start_time.elapsed();
    println!("\nTraining completed in {:?}", elapsed);
    println!("Final checkpoint saved to: {:?}", config.output_dir);
    
    Ok(())
}

/// Configuration structure matching the YAML
struct TrainingConfigSD3 {
    name: String,
    trigger_word: String,
    dataset_path: PathBuf,
    model_name: String,
    output_dir: PathBuf,
    batch_size: usize,
    gradient_accumulation: usize,
    total_steps: usize,
    save_every: usize,
    sample_every: usize,
    learning_rate: f32,
    use_ema: bool,
    ema_decay: f32,
}

/// Custom logging callback
struct LoggingCallback {
    start_time: std::time::Instant,
}

impl LoggingCallback {
    fn new() -> Self {
        Self {
            start_time: std::time::Instant::now(),
        }
    }
}

impl eridiffusion_training::Callback for LoggingCallback {
    fn on_step_end(&mut self, step: usize, loss: f32, _lr: f32) -> Result<()> {
        if step % 10 == 0 {
            let elapsed = self.start_time.elapsed().as_secs();
            let steps_per_sec = step as f64 / elapsed as f64;
            println!(
                "Step {}: loss={:.4}, steps/sec={:.2}",
                step, loss, steps_per_sec
            );
        }
        Ok(())
    }
}

/// Custom sampling callback
struct SampleCallback {
    sample_every: usize,
}

impl SampleCallback {
    fn new(sample_every: usize) -> Self {
        Self { sample_every }
    }
}

impl eridiffusion_training::Callback for SampleCallback {
    fn on_step_end(&mut self, step: usize, _loss: f32, _lr: f32) -> Result<()> {
        if step % self.sample_every == 0 && step > 0 {
            println!("Generating samples at step {}...", step);
        }
        Ok(())
    }
}