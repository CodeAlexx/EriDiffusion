use candle::{Device, Tensor, DType, IndexOp};
use candle_nn::VarBuilder;
use candle_transformers::models::flux;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Setup device
    let device = Device::cuda_if_available(0)?;
    
    // Local model path
    let model_path = "/home/alex/SwarmUI/Models/diffusion_models/flux1-dev.safetensors";
    
    println!("Loading Flux model from local file...");
    let vb = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(&[model_path], DType::BF16, &device)?
    };
    
    // Create Flux config for Dev variant
    let config = flux::Config::dev();
    
    // Load the model
    let flux_model = flux::Flux::new(&config, vb)?;
    
    // Generate noise for "lady at the beach"
    let height = 1024;
    let width = 1024;
    let img = flux::sampling::get_noise(1, height, width, &device)?;
    
    // For now, create dummy inputs to test the model loads
    let img_ids = Tensor::zeros((1, (height/16) * (width/16), 3), DType::F32, &device)?;
    let txt = Tensor::randn(0f32, 1f32, (1, 512, 4096), &device)?; // T5 embeddings
    let txt_ids = Tensor::zeros((1, 512, 3), DType::F32, &device)?;
    let timesteps = Tensor::new(&[1.0f32], &device)?;
    let y = Tensor::randn(0f32, 1f32, (1, 768), &device)?; // CLIP embeddings
    let guidance = Tensor::new(&[3.5f32], &device)?;
    
    println!("Model loaded successfully!");
    println!("Ready to generate 'lady at the beach'");
    
    // Create a state for sampling
    let state = flux::sampling::State::new(&txt, &y, &img)?;
    
    // Get timesteps
    let timesteps = flux::sampling::get_schedule(28, Some(((height/16) * (width/16), 0.5, 1.15)));
    
    println!("Would generate with {} timesteps", timesteps.len());
    println!("To complete generation, need T5 and CLIP text encoders");
    
    Ok(())
}