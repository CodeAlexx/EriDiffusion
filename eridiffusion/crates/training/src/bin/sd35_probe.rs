//! SD-3.5 streaming validation harness. Loads a strict shard and walks the first
//! few blocks to ensure the runtime + loader wiring succeeds on the target GPU.

use std::path::PathBuf;

use anyhow::{anyhow, Result};
use clap::Parser;

use eridiffusion_training::sd35::train::{self, Config as Sd35Config};

#[derive(Debug, Parser)]
#[command(name = "sd35_probe", about = "SD-3.5 streaming validation", version)]
struct Args {
    /// Path to sd3.5 shard (.safetensors or strict mmap).
    #[arg(long = "weights", value_name = "PATH")]
    weights: PathBuf,

    /// CUDA device ordinal.
    #[arg(long = "device", default_value_t = 0)]
    device: usize,

    /// Number of blocks to probe (0 = all).
    #[arg(long = "probe-blocks", default_value_t = 3)]
    probe_blocks: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if !args.weights.exists() {
        return Err(anyhow!("weights not found at {}", args.weights.display()));
    }
    let cfg = Sd35Config {
        shard_path: args.weights.to_string_lossy().into_owned(),
        device_ordinal: args.device,
        probe_blocks: args.probe_blocks,
    };
    train::train_loop(&cfg).map_err(|e| anyhow!(e.to_string()))
}
