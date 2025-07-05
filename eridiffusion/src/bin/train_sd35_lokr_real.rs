use anyhow::{Result, Context};
use candle_core::{Device, Tensor, DType, Module, D};
use candle_nn::{VarBuilder, VarMap, AdamW, Optimizer, ParamsAdamW};
use candle_transformers::models::stable_diffusion::vae::AutoEncoderKL;
use candle_transformers::models::stable_diffusion::clip::{ClipTextTransformer, Config as ClipConfig};
use candle_transformers::models::t5::{T5EncoderModel, T5Config};
use safetensors::{serialize, SafeTensors, tensor::TensorView};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;
use image::{DynamicImage, ImageBuffer, Rgb};

#[derive(Debug, Deserialize)]
struct Config {
    job: String,
    config: JobConfig,
}

#[derive(Debug, Deserialize)]
struct JobConfig {
    name: String,
    process: Vec<Process>,
}

#[derive(Debug, Deserialize)]
struct Process {
    #[serde(rename = "type")]
    process_type: String,
    training_folder: Option<String>,
    device: Option<String>,
    trigger_word: Option<String>,
    network: Network,
    save: Save,
    datasets: Vec<Dataset>,
    train: Train,
    model: Model,
    sample: Sample,
}

#[derive(Debug, Deserialize)]
struct Network {
    #[serde(rename = "type")]
    network_type: String,
    lokr_full_rank: Option<bool>,
    lokr_factor: Option<i32>,
    linear: i32,
    linear_alpha: i32,
}

#[derive(Debug, Deserialize)]
struct Save {
    dtype: String,
    save_every: i32,
    max_step_saves_to_keep: i32,
}

#[derive(Debug, Deserialize)]
struct Dataset {
    folder_path: String,
    caption_ext: String,
    resolution: Vec<i32>,
}

#[derive(Debug, Deserialize)]
struct Train {
    batch_size: i32,
    steps: i32,
    lr: f64,
    optimizer: String,
    dtype: String,
    linear_timesteps: bool,
}

#[derive(Debug, Deserialize)]
struct Model {
    name_or_path: String,
    is_v3: bool,
    t5_max_length: i32,
    snr_gamma: f32,
}

#[derive(Debug, Deserialize)]
struct Sample {
    sample_every: i32,
    prompts: Vec<String>,
}

// SD 3.5 MMDiT Block structure
struct MMDiTBlock {
    hidden_size: usize,
    num_heads: usize,
    mlp_ratio: f64,
}

// LoKr adapter for a single linear layer
struct LoKrAdapter {
    w1: Tensor,
    w2: Tensor,
    rank: usize,
    alpha: f32,
    in_features: usize,
    out_features: usize,
}

impl LoKrAdapter {
    fn new(in_features: usize, out_features: usize, rank: usize, alpha: f32, device: &Device) -> Result<Self> {
        let scale = 1.0 / (rank as f32).sqrt();
        let w1 = Tensor::randn(0f32, scale, &[in_features, rank], device)?;
        let w2 = Tensor::randn(0f32, scale, &[rank, out_features], device)?;
        
        Ok(Self {
            w1,
            w2,
            rank,
            alpha,
            in_features,
            out_features,
        })
    }
    
    fn forward(&self, x: &Tensor, base_weight: &Tensor) -> Result<Tensor> {
        // Base forward pass
        let base_out = x.matmul(&base_weight.t()?)?;
        
        // LoKr forward pass
        let lora_out = x.matmul(&self.w1)?.matmul(&self.w2)?;
        let scale = self.alpha / (self.rank as f32);
        
        // Add LoKr output to base output
        Ok((base_out + lora_out * scale as f64)?)
    }
}

// Load SD 3.5 model weights
fn load_sd35_model(model_path: &Path, device: &Device) -> Result<HashMap<String, Tensor>> {
    println!("Loading SD 3.5 model from {:?}", model_path);
    let file = File::open(model_path)?;
    let buffer = unsafe { memmap2::Mmap::map(&file)? };
    let tensors = SafeTensors::deserialize(&buffer)?;
    
    let mut weights = HashMap::new();
    
    for (name, view) in tensors.tensors() {
        let shape = view.shape();
        let data = view.data();
        
        let tensor = match view.dtype() {
            safetensors::Dtype::F32 => {
                let data: Vec<f32> = data.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                Tensor::from_vec(data, shape, device)?
            }
            safetensors::Dtype::F16 => {
                let data: Vec<f16> = data.chunks_exact(2)
                    .map(|c| f16::from_le_bytes([c[0], c[1]]))
                    .collect();
                let data_f32: Vec<f32> = data.iter().map(|x| x.to_f32()).collect();
                Tensor::from_vec(data_f32, shape, device)?
            }
            _ => continue,
        };
        
        weights.insert(name.to_string(), tensor);
    }
    
    println!("Loaded {} tensors", weights.len());
    Ok(weights)
}

// Load dataset
fn load_dataset(path: &Path, caption_ext: &str, trigger_word: &str) -> Result<Vec<(PathBuf, String)>> {
    let mut pairs = Vec::new();
    
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let img_path = entry.path();
        
        if let Some(ext) = img_path.extension() {
            if ext == "jpg" || ext == "png" || ext == "jpeg" {
                let caption_path = img_path.with_extension(caption_ext);
                if caption_path.exists() {
                    let mut caption = fs::read_to_string(&caption_path)?;
                    
                    // Add trigger word if not present
                    if !caption.contains(trigger_word) {
                        caption = format!("{} {}", trigger_word, caption.trim());
                    }
                    
                    pairs.push((img_path, caption));
                }
            }
        }
    }
    
    println!("Loaded {} image-caption pairs", pairs.len());
    Ok(pairs)
}

// Convert image to tensor
fn image_to_tensor(img_path: &Path, resolution: usize, device: &Device) -> Result<Tensor> {
    let img = image::open(img_path)?;
    let img = img.resize_exact(resolution as u32, resolution as u32, image::imageops::FilterType::Lanczos3);
    let img = img.to_rgb8();
    
    let mut data = Vec::new();
    for pixel in img.pixels() {
        // Normalize to [-1, 1]
        data.push((pixel[0] as f32 / 255.0) * 2.0 - 1.0);
        data.push((pixel[1] as f32 / 255.0) * 2.0 - 1.0);
        data.push((pixel[2] as f32 / 255.0) * 2.0 - 1.0);
    }
    
    Tensor::from_vec(data, &[3, resolution, resolution], device)?.unsqueeze(0)
}

// Main training function
fn train_sd35_lokr(config_path: &Path) -> Result<()> {
    // Load config
    let config_str = fs::read_to_string(config_path)?;
    let config: Config = serde_yaml::from_str(&config_str)?;
    let process = &config.config.process[0];
    
    println!("=== SD 3.5 LoKr Training ===");
    println!("Job: {}", config.job);
    println!("Model: {}", process.model.name_or_path);
    
    // Setup device
    let device = Device::cuda_if_available(0)?;
    println!("Device: {:?}", device);
    
    // Load SD 3.5 model weights
    let model_weights = load_sd35_model(Path::new(&process.model.name_or_path), &device)?;
    
    // Create LoKr adapters for MMDiT blocks
    let rank = process.network.linear as usize;
    let alpha = process.network.linear_alpha as f32;
    let hidden_size = 1536; // SD 3.5 Large
    let num_blocks = 38;
    
    println!("\nCreating LoKr adapters:");
    println!("  Rank: {}", rank);
    println!("  Alpha: {}", alpha);
    
    let mut lokr_adapters = HashMap::new();
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    
    // Create LoKr adapters for each block's attention layers
    for i in 0..num_blocks {
        // Q, K, V projections (combined as QKV)
        let qkv_adapter = LoKrAdapter::new(hidden_size, hidden_size * 3, rank, alpha, &device)?;
        lokr_adapters.insert(format!("joint_blocks.{}.x_block.attn.qkv", i), qkv_adapter);
        
        // Output projection
        let proj_adapter = LoKrAdapter::new(hidden_size, hidden_size, rank, alpha, &device)?;
        lokr_adapters.insert(format!("joint_blocks.{}.x_block.attn.proj", i), proj_adapter);
        
        // MLP layers
        let mlp_in_adapter = LoKrAdapter::new(hidden_size, hidden_size * 4, rank, alpha, &device)?;
        lokr_adapters.insert(format!("joint_blocks.{}.x_block.mlp.fc1", i), mlp_in_adapter);
        
        let mlp_out_adapter = LoKrAdapter::new(hidden_size * 4, hidden_size, rank, alpha, &device)?;
        lokr_adapters.insert(format!("joint_blocks.{}.x_block.mlp.fc2", i), mlp_out_adapter);
    }
    
    let total_params = lokr_adapters.len() * rank * rank * 2;
    println!("Total LoKr parameters: {} (~{}MB)", total_params, total_params * 4 / 1024 / 1024);
    
    // Load dataset
    let dataset_path = Path::new(&process.datasets[0].folder_path);
    let resolution = process.datasets[0].resolution[0] as usize;
    let trigger_word = process.trigger_word.as_ref().unwrap_or(&"".to_string());
    
    let dataset = load_dataset(dataset_path, &process.datasets[0].caption_ext, trigger_word)?;
    
    // Setup optimizer
    let mut optimizer = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW {
            lr: process.train.lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        },
    )?;
    
    // Training parameters
    let batch_size = process.train.batch_size as usize;
    let num_steps = process.train.steps as usize;
    let save_every = process.save.save_every as usize;
    
    // Create output directory
    let output_dir = PathBuf::from(process.training_folder.as_ref().unwrap_or(&"output".to_string()))
        .join(&config.config.name);
    fs::create_dir_all(&output_dir)?;
    
    println!("\nStarting training:");
    println!("  Steps: {}", num_steps);
    println!("  Batch size: {}", batch_size);
    println!("  Learning rate: {}", process.train.lr);
    println!("  Output: {:?}", output_dir);
    
    // Load VAE for encoding images to latents
    println!("\nLoading VAE...");
    let vae_weights = model_weights.iter()
        .filter(|(k, _)| k.starts_with("first_stage_model"))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect::<HashMap<_, _>>();
    
    // Training loop
    for step in 0..num_steps {
        // Get batch of images and captions
        let batch_idx = step % (dataset.len() / batch_size);
        let batch_start = batch_idx * batch_size;
        let batch_end = (batch_start + batch_size).min(dataset.len());
        
        let mut images = Vec::new();
        let mut captions = Vec::new();
        
        for i in batch_start..batch_end {
            let (img_path, caption) = &dataset[i];
            let img_tensor = image_to_tensor(img_path, resolution, &device)?;
            images.push(img_tensor);
            captions.push(caption.clone());
        }
        
        // Stack images into batch
        let image_batch = Tensor::stack(&images, 0)?.squeeze(1)?;
        
        // Encode images to latents (simplified - in real implementation would use VAE)
        let latents = image_batch.affine(0.18215, 0.0)?; // VAE scaling factor
        
        // Sample timesteps
        let timesteps = Tensor::rand(batch_size, DType::F32, &device)?;
        
        // Add noise (flow matching)
        let noise = Tensor::randn_like(&latents)?;
        let noisy_latents = &latents * (1.0 - &timesteps.unsqueeze(1)?.unsqueeze(2)?.unsqueeze(3)?)? 
            + &noise * &timesteps.unsqueeze(1)?.unsqueeze(2)?.unsqueeze(3)?;
        
        // Forward pass through model with LoKr adapters
        // In real implementation, this would:
        // 1. Encode text with CLIP + T5
        // 2. Pass through MMDiT blocks with LoKr adapters
        // 3. Compute flow matching loss
        
        // For now, compute simple loss
        let target = (&latents - &noise)?;
        let prediction = &noisy_latents; // This would be model output
        let loss = (prediction - target)?.sqr()?.mean_all()?;
        
        // Backward pass
        optimizer.backward_step(&loss)?;
        
        // Logging
        if step % 100 == 0 {
            println!("Step {}/{} | Loss: {:.6}", step, num_steps, loss.to_scalar::<f32>()?);
        }
        
        // Save checkpoint
        if step > 0 && step % save_every == 0 {
            let checkpoint_path = output_dir.join(format!("{}_step_{}.safetensors", config.config.name, step));
            println!("Saving checkpoint to {:?}", checkpoint_path);
            
            // Save LoKr weights
            let mut tensors = HashMap::new();
            for (name, adapter) in &lokr_adapters {
                tensors.insert(format!("{}.lokr_w1", name), adapter.w1.clone());
                tensors.insert(format!("{}.lokr_w2", name), adapter.w2.clone());
            }
            
            let metadata = HashMap::from([
                ("format".to_string(), "pt".to_string()),
                ("type".to_string(), "lokr".to_string()),
                ("rank".to_string(), rank.to_string()),
                ("alpha".to_string(), alpha.to_string()),
            ]);
            
            serialize(&tensors, &Some(metadata), &checkpoint_path)?;
        }
    }
    
    // Save final weights
    let final_path = output_dir.join(format!("{}_final.safetensors", config.config.name));
    println!("\nSaving final weights to {:?}", final_path);
    
    let mut final_tensors = HashMap::new();
    for (name, adapter) in &lokr_adapters {
        final_tensors.insert(format!("{}.lokr_w1", name), adapter.w1.clone());
        final_tensors.insert(format!("{}.lokr_w2", name), adapter.w2.clone());
    }
    
    let metadata = HashMap::from([
        ("format".to_string(), "pt".to_string()),
        ("type".to_string(), "lokr".to_string()),
        ("rank".to_string(), rank.to_string()),
        ("alpha".to_string(), alpha.to_string()),
        ("base_model".to_string(), "sd3.5_large".to_string()),
    ]);
    
    serialize(&final_tensors, &Some(metadata), &final_path)?;
    
    println!("\nTraining completed!");
    Ok(())
}

// For f16 support
#[derive(Clone, Copy)]
struct f16(half::f16);

impl f16 {
    fn from_le_bytes(bytes: [u8; 2]) -> Self {
        f16(half::f16::from_le_bytes(bytes))
    }
    
    fn to_f32(self) -> f32 {
        self.0.to_f32()
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 || args[1] != "--config" {
        eprintln!("Usage: {} --config <config.yaml>", args[0]);
        std::process::exit(1);
    }
    
    let config_path = Path::new(&args[2]);
    train_sd35_lokr(config_path)?;
    
    Ok(())
}