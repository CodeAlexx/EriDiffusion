//! Preprocess dataset for Flux training
//! This MUST be run before training to fit in 24GB VRAM

use eridiffusion_core::{Device, Result};
use eridiffusion_training::flux_preprocessor::{
    FluxPreprocessor, FluxPreprocessorConfig, print_memory_savings
};
use eridiffusion_data::ImageFolderDataset;
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about = "Preprocess dataset for Flux training", long_about = None)]
struct Args {
    /// Dataset directory containing images and captions
    #[arg(long)]
    dataset_dir: PathBuf,
    
    /// Output cache directory for preprocessed data
    #[arg(long, default_value = "flux_cache")]
    cache_dir: PathBuf,
    
    /// VAE model path
    #[arg(long)]
    vae_path: PathBuf,
    
    /// T5-XXL model path
    #[arg(long)]
    t5_path: PathBuf,
    
    /// CLIP-L model path
    #[arg(long)]
    clip_path: PathBuf,
    
    /// T5 tokenizer path
    #[arg(long)]
    t5_tokenizer: PathBuf,
    
    /// CLIP tokenizer path
    #[arg(long)]
    clip_tokenizer: PathBuf,
    
    /// Batch size for encoding
    #[arg(long, default_value = "4")]
    batch_size: usize,
    
    /// Device ID (GPU)
    #[arg(long, default_value = "0")]
    device_id: usize,
    
    /// Overwrite existing cache
    #[arg(long)]
    overwrite: bool,
    
    /// Caption file extension
    #[arg(long, default_value = "txt")]
    caption_ext: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();
    
    println!("🔥 Flux Dataset Preprocessor");
    println!("════════════════════════════════════════");
    println!();
    println!("This tool preprocesses your dataset for memory-efficient Flux training.");
    println!("It encodes images → latents and text → embeddings, saving them to disk.");
    println!("This allows Flux training to fit in 24GB VRAM!");
    println!();
    
    // Show memory savings
    print_memory_savings(1000); // Example for 1000 images
    
    println!("\n📂 Configuration:");
    println!("─────────────────────────────────");
    println!("Dataset: {}", args.dataset_dir.display());
    println!("Cache: {}", args.cache_dir.display());
    println!("Device: CUDA:{}", args.device_id);
    println!("Batch size: {}", args.batch_size);
    
    // Create dataset
    println!("\n📸 Loading dataset...");
    let dataset = ImageFolderDataset::new(
        &args.dataset_dir,
        Some(&args.caption_ext),
        None, // No transforms needed - VAE will handle it
        100,  // Cache size
    ).await?;
    
    println!("✓ Found {} images", dataset.len());
    
    // Setup device
    let device = Device::Cuda(args.device_id);
    
    // Create preprocessor config
    let config = FluxPreprocessorConfig {
        cache_dir: args.cache_dir,
        device: device.clone(),
        batch_size: args.batch_size,
        overwrite: args.overwrite,
    };
    
    // Load models
    println!("\n🧠 Loading encoders (this is the only time they're all in memory)...");
    
    // Load VAE
    println!("Loading VAE from {}...", args.vae_path.display());
    use eridiffusion_training::flux_model_loader::FluxVAE;
    let vae_vb = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(
            &[&args.vae_path],
            candle_core::DType::F32,
            &device.to_candle()?,
        )?
    };
    let vae_config = candle_transformers::models::flux::autoencoder::Config::dev();
    let vae_model = candle_transformers::models::flux::autoencoder::AutoEncoder::new(&vae_config, vae_vb)?;
    let vae = Box::new(FluxVAE::new(vae_model, device.clone()));
    
    // Load T5
    println!("Loading T5-XXL from {}...", args.t5_path.display());
    use eridiffusion_training::flux_model_loader::T5TextEncoder;
    use tokenizers::Tokenizer;
    let t5_vb = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(
            &[&args.t5_path],
            candle_core::DType::F32,
            &device.to_candle()?,
        )?
    };
    let t5_config = candle_transformers::models::t5::Config::default();
    let t5_model = candle_transformers::models::t5::T5EncoderModel::load(t5_vb, &t5_config)?;
    let t5_tokenizer = Tokenizer::from_file(&args.t5_tokenizer)?;
    let t5_encoder = Box::new(T5TextEncoder::new(t5_model, t5_tokenizer, device.clone()));
    
    // Load CLIP
    println!("Loading CLIP-L from {}...", args.clip_path.display());
    use eridiffusion_training::flux_model_loader::CLIPTextEncoder;
    let clip_vb = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(
            &[&args.clip_path],
            candle_core::DType::F32,
            &device.to_candle()?,
        )?
    };
    let clip_config = candle_transformers::models::clip::text_model::ClipTextConfig::default();
    let clip_model = candle_transformers::models::clip::text_model::ClipTextTransformer::new(&clip_config, clip_vb)?;
    let clip_tokenizer = Tokenizer::from_file(&args.clip_tokenizer)?;
    let clip_encoder = Box::new(CLIPTextEncoder::new(clip_model, clip_tokenizer, device.clone(), clip_config));
    
    println!("✓ All encoders loaded");
    
    // Create preprocessor
    let mut preprocessor = FluxPreprocessor::new(config)?
        .with_vae(vae)
        .with_t5_encoder(t5_encoder)
        .with_clip_encoder(clip_encoder);
    
    // Process dataset
    println!("\n⚡ Starting preprocessing...");
    println!("This will:");
    println!("1. Encode each image through VAE → latents");
    println!("2. Encode each caption through T5 → embeddings"); 
    println!("3. Encode each caption through CLIP → pooled embeddings");
    println!("4. Save everything to disk");
    println!("5. FREE all encoders from memory");
    println!();
    
    let items = preprocessor.preprocess_dataset(&*dataset).await?;
    
    println!("\n✅ Preprocessing complete!");
    println!("Preprocessed {} items", items.len());
    println!("Cache size: {:.2} GB", 
        std::fs::read_dir(&args.cache_dir)?
            .filter_map(|e| e.ok())
            .filter_map(|e| e.metadata().ok())
            .map(|m| m.len() as f64)
            .sum::<f64>() / 1e9
    );
    
    println!("\n🎯 Next steps:");
    println!("1. Run the Flux trainer with --cache-dir {}", args.cache_dir.display());
    println!("2. The trainer will load ONLY the Flux model (~12GB)");
    println!("3. Training will fit comfortably in 24GB VRAM!");
    
    Ok(())
}