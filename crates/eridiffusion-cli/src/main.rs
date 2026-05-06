use clap::{Parser, ValueEnum};
use eridiffusion_core::config::{ModelType, TrainConfig, TrainingMethod};
use eridiffusion_core::data::CachedDataset;
use eridiffusion_core::training::GenericTrainer;
use anyhow::Context;

#[derive(Parser)]
#[command(name = "eridiffusion", version, about = "EriDiffusion Rust — AI model training")]
struct Cli {
    /// Path to JSON config file (same format as Python EriDiffusion)
    #[arg(short, long)]
    config: String,

    /// Output directory
    #[arg(short, long, default_value = "output")]
    output_dir: String,

    /// Override max training steps
    #[arg(long)]
    max_steps: Option<usize>,

    /// Log level
    #[arg(long, default_value = "info")]
    log_level: String,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or(&cli.log_level)
    ).init();

    log::info!("EriDiffusion Rust v{}", env!("CARGO_PKG_VERSION"));

    // Load config
    let config = TrainConfig::from_json_path(&cli.config)
        .with_context(|| format!("loading config {}", cli.config))?;

    log::info!("model_type={:?} training_method={:?} lora_rank={}",
        config.model_type, config.training_method, config.lora_rank);

    // Init Flame
    flame_core::init();

    // Load dataset
    let cache_path = std::path::PathBuf::from(&config.cache_dir);
    let dataset = CachedDataset::load(&cache_path)
        .with_context(|| "loading dataset")?;

    if dataset.is_empty() {
        log::warn!("Dataset is empty. Did you run the prepare_dataset tool?");
        log::info!("Usage: eridiffusion-prepare --config {}", cli.config);
        return Ok(());
    }

    let output_dir = std::path::PathBuf::from(&cli.output_dir);
    std::fs::create_dir_all(&output_dir)?;

    log::info!("Dataset: {} samples. Output: {}", dataset.len(), output_dir.display());
    log::info!("Ready for training. Model implementations coming next.");
    log::info!("Current status: config + dataset loading ✅ | model forward 🔨");
    log::info!("");

    Ok(())
}
