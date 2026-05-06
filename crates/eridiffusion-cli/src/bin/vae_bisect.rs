//! VAE encoder bisect: load upstream Python's pixel tensor (saved as safetensors) and
//! run KleinVaeEncoder, dumping every intermediate to `dump_dir/`.
use clap::Parser;
use flame_core::{DType, autograd::AutogradContext};
use eridiffusion_core::encoders::vae::KleinVaeEncoder;
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    /// Input safetensors with key "pixels" of shape [1, 3, H, W] in [-1, 1].
    #[arg(long)] pixels: PathBuf,
    /// Klein VAE safetensors (full file).
    #[arg(long)] vae: PathBuf,
    /// Output directory for intermediate dumps.
    #[arg(long)] dump_dir: PathBuf,
}

fn main() -> anyhow::Result<()> {
    if std::env::var_os("FLAME_ALLOC_POOL").is_none() {
        unsafe { std::env::set_var("FLAME_ALLOC_POOL", "0"); }
    }
    env_logger::init();
    let args = Args::parse();
    flame_core::config::set_default_dtype(DType::BF16);
    let _no_grad = AutogradContext::no_grad();
    let device = flame_core::global_cuda_device();

    log::info!("Loading pixels from {}...", args.pixels.display());
    let pix_map = flame_core::serialization::load_file(&args.pixels, &device)?;
    let pixels = pix_map.get("pixels").ok_or_else(|| anyhow::anyhow!("missing 'pixels'"))?
        .to_dtype(DType::BF16)?;
    log::info!("  shape={:?} dtype={:?}", pixels.shape().dims(), pixels.dtype());

    log::info!("Loading VAE from {}...", args.vae.display());
    let vae_w = flame_core::serialization::load_file(&args.vae, &device)?;
    let dev = flame_core::Device::from(device.clone());
    let vae = KleinVaeEncoder::load(&vae_w, &dev)?;
    drop(vae_w);

    log::info!("Encoding with intermediate dumps → {}", args.dump_dir.display());
    let final_out = vae.encode_with_dump(&pixels, &args.dump_dir)?;
    log::info!("FINAL latent shape={:?} dtype={:?}", final_out.shape().dims(), final_out.dtype());
    Ok(())
}
