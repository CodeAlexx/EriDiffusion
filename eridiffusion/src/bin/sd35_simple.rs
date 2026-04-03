use clap::Parser;
use eridiffusion::inference::sd35_simple::{
    resolve_sd35_simple_paths, save_tensor_image, Sd35SimpleModel, encode_prompts,
};
use flame_core::device::Device;
use flame_core::{Result, DType};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "sd35_simple", about = "Minimal SD3.5 inference runner")]
struct Args {
    /// Prompt text
    #[arg(long)]
    prompt: String,

    /// Negative prompt
    #[arg(long, default_value = "")]
    negative: String,

    /// Output PNG path
    #[arg(long, default_value = "output/sd35_simple.png")]
    output: PathBuf,

    /// Guidance scale
    #[arg(long, default_value_t = 5.0)]
    cfg: f32,

    /// Number of inference steps
    #[arg(long, default_value_t = 20)]
    steps: usize,

    /// Image width
    #[arg(long, default_value_t = 512)]
    width: usize,

    /// Image height
    #[arg(long, default_value_t = 512)]
    height: usize,

    /// Model variant (medium or large)
    #[arg(long, default_value = "medium")]
    variant: String,

    /// Optional seed (informational only for now)
    #[arg(long)]
    seed: Option<u64>,
}



fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();
    std::fs::create_dir_all(
        args.output
            .parent()
            .unwrap_or_else(|| std::path::Path::new(".")),
    )
    .ok();

    let variant = args.variant.to_lowercase();
    let paths = resolve_sd35_simple_paths(&variant)?;
    let device = Device::cuda(0)?;

    println!(
        "SD35 simple runner\n  prompt: {}\n  steps: {}\n  cfg: {}\n  size: {}x{}\n  variant: {}",
        args.prompt, args.steps, args.cfg, args.width, args.height, variant
    );

    let (cond, uncond, cond_pooled, uncond_pooled) = encode_prompts(&paths, &args.prompt, &args.negative, device.clone())?;

    // Save text encoder outputs for parity check
    {
        let mut tensors = std::collections::HashMap::new();
        tensors.insert("cond".to_string(), cond.clone());
        tensors.insert("uncond".to_string(), uncond.clone());
        tensors.insert("cond_pooled".to_string(), cond_pooled.clone());
        tensors.insert("uncond_pooled".to_string(), uncond_pooled.clone());

        let mut data_buffers = Vec::new();
        let mut keys = Vec::new();
        let mut shapes = Vec::new();
        let mut dtypes = Vec::new();
        for (k, v) in &tensors {
            keys.push(k.clone());
            shapes.push(v.shape().dims().to_vec());
            dtypes.push(match v.dtype() {
                DType::F32 => safetensors::tensor::Dtype::F32,
                DType::BF16 => safetensors::tensor::Dtype::BF16,
                _ => safetensors::tensor::Dtype::F32,
            });
            data_buffers.push(v.to_bytes().unwrap());
        }
        let tensor_views: Vec<(String, safetensors::tensor::TensorView)> = keys.iter().enumerate().map(|(i, k)| {
            (k.clone(), safetensors::tensor::TensorView::new(dtypes[i], shapes[i].clone(), &data_buffers[i]).unwrap())
        }).collect();
        safetensors::serialize_to_file(tensor_views, &None, std::path::Path::new("debug_text_encoders.safetensors")).ok();
        println!("Saved debug_text_encoders.safetensors");
    }

    let mut model = Sd35SimpleModel::load(&paths, device)?;
    let _guard = flame_core::autograd::AutogradContext::no_grad();
    let image = model.sample(
        cond,
        uncond,
        cond_pooled,
        uncond_pooled,
        args.steps,
        args.cfg,
        args.width,
        args.height,
        args.seed,
    )?;
    save_tensor_image(&image, &args.output)?;
    println!("Image written to {}", args.output.display());

    // Save raw tensor for parity check
    let output_path = args.output.with_extension("safetensors");
    let data = image.to_vec_f32()?;
    let shape = image.shape().dims();
    let shape_usize: Vec<usize> = shape.iter().map(|&x| x).collect();
    
    use safetensors::tensor::{TensorView, Dtype};
    let tensor = TensorView::new(Dtype::F32, shape_usize, &data_as_u8(&data)).map_err(|e| flame_core::Error::Io(e.to_string()))?;
    let tensors = [("image", tensor)];
    safetensors::serialize_to_file(tensors, &None, &output_path).map_err(|e| flame_core::Error::Io(e.to_string()))?;
    println!("Raw tensor written to {}", output_path.display());

    Ok(())
}

fn data_as_u8(data: &[f32]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const u8,
            data.len() * std::mem::size_of::<f32>(),
        )
    }
}
