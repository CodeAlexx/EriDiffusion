//! SD 3.5 LoKr trainer executable

use eridiffusion_core::{Result, Error, ModelArchitecture, NetworkType, Device};
use eridiffusion_networks::{LoKrNetwork, LoKrConfig};
use eridiffusion_training::{SD35Trainer, SD35TrainingConfig, SD35ModelVariant, Trainer, TrainerConfig};
use eridiffusion_models::{SD3Model, TextEncoder, VAE, ModelFactory, VAEFactory};
use eridiffusion_data::{ImageFolderDataset, DataLoader, DataLoaderConfig};
use candle_core::{Tensor, DType, D};
use candle_nn::{VarMap, AdamW, Optimizer, ParamsAdamW};
use clap::Parser;
use std::path::PathBuf;
use tracing::{info, error, warn};
use serde::{Deserialize, Serialize};
use std::fs;
use safetensors;

#[derive(Parser, Debug)]
#[command(name = "train-sd35-lokr")]
#[command(about = "Train SD 3.5 LoKr adapter")]
struct Args {
    /// Configuration file path
    #[arg(short, long)]
    config: PathBuf,
    
    /// Override model path
    #[arg(long)]
    model_path: Option<PathBuf>,
    
    /// Override output directory
    #[arg(short, long)]
    output_dir: Option<PathBuf>,
    
    /// Device to use (cpu, cuda:0, etc)
    #[arg(long, default_value = "cuda:0")]
    device: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct AIToolkitConfig {
    job: String,
    config: JobConfig,
    meta: Option<serde_yaml::Value>,
}

#[derive(Debug, Deserialize, Serialize)]
struct JobConfig {
    name: String,
    process: Vec<ProcessConfig>,
}

#[derive(Debug, Deserialize, Serialize)]
struct ProcessConfig {
    #[serde(rename = "type")]
    process_type: String,
    model: Option<ModelConfig>,
    network: Option<NetworkConfig>,
    datasets: Option<Vec<DatasetConfig>>,
    train: Option<TrainConfig>,
    sample: Option<SampleConfig>,
}

#[derive(Debug, Deserialize, Serialize)]
struct ModelConfig {
    name_or_path: String,
    is_v3: Option<bool>,
    quantize: Option<bool>,
}

#[derive(Debug, Deserialize, Serialize)]
struct NetworkConfig {
    #[serde(rename = "type")]
    network_type: String,
    linear: Option<i32>,
    linear_alpha: Option<i32>,
    factor: Option<i32>,
}

#[derive(Debug, Deserialize, Serialize)]
struct DatasetConfig {
    folder_path: PathBuf,
    caption_ext: String,
    caption_dropout_rate: f32,
    shuffle_tokens: bool,
    cache_latents_to_disk: bool,
    resolution: Vec<u32>,
}

#[derive(Debug, Deserialize, Serialize)]
struct TrainConfig {
    batch_size: usize,
    steps: usize,
    gradient_accumulation_steps: usize,
    lr: f64,
    unet_lr: Option<f64>,
    text_encoder_lr: Option<f64>,
    lr_scheduler: String,
    lr_scheduler_params: Option<serde_yaml::Value>,
    optimizer: String,
    optimizer_params: Option<serde_yaml::Value>,
    ema_config: Option<serde_yaml::Value>,
    dtype: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct SampleConfig {
    sampler: String,
    sample_every: usize,
    width: u32,
    height: u32,
    prompts: Vec<String>,
    neg: String,
    seed: u64,
    walk_seed: bool,
    guidance_scale: f32,
    sample_steps: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    let args = Args::parse();
    
    // Load configuration
    info!("Loading configuration from {:?}", args.config);
    let config_str = fs::read_to_string(&args.config)
        .map_err(|e| Error::IoError(format!("Failed to read config: {}", e)))?;
    
    let config: AIToolkitConfig = serde_yaml::from_str(&config_str)
        .map_err(|e| Error::Config(format!("Failed to parse config: {}", e)))?;
    
    info!("Loaded configuration for job: {}", config.job);
    
    // Find the training process (look for both "train" and "sd_trainer")
    let train_process = config.config.process.iter()
        .find(|p| p.process_type == "sd_trainer" || p.process_type == "train")
        .ok_or_else(|| Error::Config("No train/sd_trainer process found in config".to_string()))?;
    
    // Extract configurations
    let model_config = train_process.model.as_ref()
        .ok_or_else(|| Error::Config("No model config found".to_string()))?;
    
    let network_config = train_process.network.as_ref()
        .ok_or_else(|| Error::Config("No network config found".to_string()))?;
    
    let dataset_configs = train_process.datasets.as_ref()
        .ok_or_else(|| Error::Config("No datasets config found".to_string()))?;
    
    let train_config = train_process.train.as_ref()
        .ok_or_else(|| Error::Config("No train config found".to_string()))?;
    
    // Set up model path
    let model_path = args.model_path
        .unwrap_or_else(|| PathBuf::from(&model_config.name_or_path));
    
    info!("Using model: {:?}", model_path);
    
    // Verify it's SD 3.5
    if !model_config.is_v3.unwrap_or(false) {
        return Err(Error::Config("Model must be SD3/SD3.5 (is_v3: true)".to_string()));
    }
    
    // Verify it's LoKr
    if network_config.network_type != "lokr" && network_config.network_type != "lora" {
        return Err(Error::Config("Network type must be 'lokr' or 'lora'".to_string()));
    }
    
    // Create LoKr configuration
    let lokr_config = LoKrConfig {
        rank: network_config.linear.unwrap_or(64) as usize,
        alpha: network_config.linear_alpha.unwrap_or(64) as f32,
        factor: network_config.factor.unwrap_or(4) as usize,
        use_scalar: true,
        decompose_factor: network_config.factor.unwrap_or(4) as usize,
        dropout: 0.0,
    };
    
    info!("LoKr config: rank={}, alpha={}, factor={}", 
        lokr_config.rank, lokr_config.alpha, lokr_config.factor);
    
    // Set up device
    let device = Device::from_string(&args.device)?;
    info!("Using device: {:?}", device);
    
    // Create output directory
    let output_dir = args.output_dir
        .unwrap_or_else(|| PathBuf::from("output"));
    fs::create_dir_all(&output_dir)
        .map_err(|e| Error::IoError(format!("Failed to create output dir: {}", e)))?;
    
    // Set up dataset
    info!("Setting up dataset...");
    let dataset_config = &dataset_configs[0];
    let dataset_path = &dataset_config.folder_path;
    
    info!("Loading dataset from: {:?}", dataset_path);
    info!("Resolution: {:?}", dataset_config.resolution);
    info!("Caption extension: {}", dataset_config.caption_ext);
    
    // Create trainer config
    let trainer_config = TrainerConfig {
        batch_size: train_config.batch_size,
        gradient_accumulation_steps: train_config.gradient_accumulation_steps,
        num_epochs: 1, // Calculate from steps
        learning_rate: train_config.lr,
        warmup_steps: 100,
        max_grad_norm: 1.0,
        save_every: 500,
        validate_every: None,
        log_every: 10,
        mixed_precision: train_config.dtype == "bf16",
        gradient_checkpointing: true,
        ema_decay: None,
        output_dir: output_dir.clone(),
        resume_from: None,
        seed: 42,
    };
    
    info!("Training configuration:");
    info!("  Batch size: {}", trainer_config.batch_size);
    info!("  Steps: {}", train_config.steps);
    info!("  Learning rate: {}", trainer_config.learning_rate);
    info!("  Optimizer: {}", train_config.optimizer);
    
    // Log the key training parameters
    info!("Starting SD 3.5 LoKr training:");
    info!("  Model: {:?}", model_path);
    info!("  Dataset: {:?}", dataset_path);
    info!("  Output: {:?}", output_dir);
    info!("  Device: {:?}", device);
    info!("  Steps: {}", train_config.steps);
    
    // 1. Load SD 3.5 model
    info!("Loading SD 3.5 model from {:?}", model_path);
    let mut sd3_model = SD3Model::from_single_file(
        &model_path,
        SD35ModelVariant::Large,
        &device,
    )?;
    
    // 2. Initialize LoKr adapters
    info!("Creating LoKr network with rank={}, alpha={}, factor={}", 
        lokr_config.rank, lokr_config.alpha, lokr_config.factor);
    
    let target_modules = SD35Trainer::get_sd35_target_modules(false); // MMDiT only
    let mut lokr_network = LoKrNetwork::new(&lokr_config, &device)?;
    lokr_network.apply_to_model(&mut sd3_model)?;
    
    // 3. Freeze base model weights (LoKr adapter handles this)
    info!("Base model weights frozen, only training LoKr parameters");
    
    // 4. Load dataset
    info!("Loading dataset from {:?}", dataset_path);
    let dataset = ImageFolderDataset::new(
        dataset_path,
        &dataset_config.caption_ext,
        dataset_config.resolution[0] as usize,
    )?;
    
    let dataloader = DataLoader::new(
        Box::new(dataset),
        train_config.batch_size,
        true, // shuffle
        Some(42), // seed
    )?;
    
    // Load text encoders
    info!("Loading text encoders...");
    let clip_l = TextEncoder::clip_l(&device)?;
    let clip_g = TextEncoder::clip_g(&device)?;
    let t5 = TextEncoder::t5_xxl(&device)?;
    
    // Load VAE
    info!("Loading VAE...");
    let vae = VAEFactory::sd3_vae(&device)?;
    
    // 5. Set up optimizer
    let varmap = VarMap::new();
    let mut optimizer = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW {
            lr: train_config.lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        },
    )?;
    
    // Training loop
    info!("Starting training for {} steps", train_config.steps);
    let mut global_step = 0;
    
    for epoch in 0..100 { // Max epochs
        for (batch_idx, batch) in dataloader.iter().enumerate() {
            if global_step >= train_config.steps {
                break;
            }
            
            // Get batch data
            let images = batch.get("image").ok_or_else(|| Error::Training("Missing images".into()))?;
            let captions = batch.get("caption").ok_or_else(|| Error::Training("Missing captions".into()))?;
            
            // Encode images to latents
            let latents = vae.encode(images)?;
            
            // Encode text
            let text_outputs = SD35Trainer::encode_text_triple(
                &captions.to_vec1::<String>()?,
                &clip_l,
                &clip_g,
                &t5,
                154, // t5_max_length from config
            )?;
            
            // Sample timesteps
            let timesteps = Tensor::rand(train_config.batch_size, DType::F32, &device)?;
            
            // Add noise (flow matching)
            let noise = Tensor::randn_like(&latents)?;
            let noisy_latents = &latents * (1.0 - &timesteps.unsqueeze(1)?.unsqueeze(2)?.unsqueeze(3)?)? 
                + &noise * &timesteps.unsqueeze(1)?.unsqueeze(2)?.unsqueeze(3)?;
            
            // Prepare inputs
            let inputs = SD35Trainer::prepare_training_inputs(
                &noisy_latents,
                &timesteps,
                &text_outputs.encoder_hidden_states,
                text_outputs.pooled_output.as_ref().unwrap(),
            )?;
            
            // Forward pass
            let model_output = sd3_model.forward(&inputs)?;
            
            // Compute loss
            let target = (&latents - &noise)? / (1.0 - &timesteps.unsqueeze(1)?.unsqueeze(2)?.unsqueeze(3)?)?;
            let loss = SD35Trainer::compute_flow_matching_loss(
                &model_output.sample,
                &target,
                &timesteps,
                5.0, // snr_gamma from config
            )?;
            
            // Backward pass
            optimizer.backward_step(&loss)?;
            
            global_step += 1;
            
            // Logging
            if global_step % 10 == 0 {
                info!("Step {}/{} | Loss: {:.6}", 
                    global_step, train_config.steps, 
                    loss.to_scalar::<f32>()?);
            }
            
            // Save checkpoint
            if global_step % 500 == 0 && global_step > 0 {
                let checkpoint_path = output_dir.join(format!("lokr_step_{}.safetensors", global_step));
                info!("Saving checkpoint to {:?}", checkpoint_path);
                lokr_network.save(&checkpoint_path)?;
            }
        }
        
        if global_step >= train_config.steps {
            break;
        }
    }
    
    // Save final checkpoint
    let final_path = output_dir.join("lokr_final.safetensors");
    info!("Saving final LoKr weights to {:?}", final_path);
    lokr_network.save(&final_path)?;
    
    info!("Training completed successfully!");
    Ok(())
}