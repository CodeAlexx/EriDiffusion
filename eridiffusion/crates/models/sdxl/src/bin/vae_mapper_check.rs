use anyhow::{Result, bail};
use clap::Parser;

#[derive(Parser, Debug)]
struct Args {
    /// Path to VAE weights (.safetensors)
    #[arg(long)]
    weights: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let ld = eridiffusion_common_weights::SafeLoader::open(&args.weights)?;
    let keys = ld.list_keys()?;
    let mut total = 0usize; let mut conv2 = 0usize; let mut conv4 = 0usize;
    for k in &keys {
        if !(k.starts_with("encoder.") || k.starts_with("decoder.")) { continue; }
        total += 1;
        let shp = ld.shape_of(k)?;
        match shp.len() { 2 => conv2 += 1, 4 => conv4 += 1, _ => bail!("{} has invalid rank {}", k, shp.len()) }
    }
    println!("OK: VAE tensors under encoder./decoder.: total={}, linear(2D)={}, conv(4D)={}", total, conv2, conv4);
    if total == 0 { bail!("no VAE tensors under encoder./decoder."); }
    Ok(())
}

