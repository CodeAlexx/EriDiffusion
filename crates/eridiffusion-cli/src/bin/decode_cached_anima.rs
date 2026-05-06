//! Decoder roundtrip smoke. Reads a Wan-VAE-normalized latent from a
//! prepare_anima cache file, decodes it with Wan21VaeDecoder, writes PNG.
use clap::Parser;
use flame_core::DType;
use std::path::PathBuf;
use eridiffusion_core::encoders::wan21_decoder::Wan21VaeDecoder;

#[derive(Parser)]
struct Args {
    #[arg(long)] cache: PathBuf,
    #[arg(long)] vae_path: PathBuf,
    #[arg(long, default_value = "decoded.png")] output: PathBuf,
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let args = Args::parse();
    let device = flame_core::global_cuda_device();
    let _no_grad = flame_core::autograd::AutogradContext::no_grad();
    flame_core::config::set_default_dtype(DType::BF16);

    let tensors = flame_core::serialization::load_file(&args.cache, &device)?;
    let latent = tensors.get("latent")
        .ok_or_else(|| anyhow::anyhow!("'latent' not in cache"))?
        .clone()
        .to_dtype(DType::BF16)?;
    log::info!("latent shape={:?} dtype={:?}", latent.shape().dims(), latent.dtype());

    let vae = Wan21VaeDecoder::from_safetensors(&args.vae_path.to_string_lossy(), &device)?;
    let img = vae.decode_image_normalized(&latent)?;
    log::info!("decoded shape={:?}", img.shape().dims());

    let pixels: Vec<f32> = img.to_dtype(DType::F32)?.to_vec()?;
    let dims = img.shape().dims();
    let (c, h, w) = (dims[1], dims[2], dims[3]);
    let mut buf = vec![0u8; h * w * 3];
    for y in 0..h {
        for x in 0..w {
            for ch in 0..c.min(3) {
                let idx = ch * h * w + y * w + x;
                let v = pixels.get(idx).copied().unwrap_or(0.0);
                buf[(y * w + x) * 3 + ch] = ((v.clamp(-1.0, 1.0) + 1.0) * 127.5) as u8;
            }
        }
    }
    image::save_buffer(&args.output, &buf, w as u32, h as u32, image::ColorType::Rgb8)?;
    log::info!("Saved {:?}", args.output);
    Ok(())
}
