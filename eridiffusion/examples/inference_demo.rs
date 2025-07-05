//! Inference demonstration for eridiffusion
//! 
//! This example shows how to:
//! 1. Load a pretrained diffusion model
//! 2. Load and apply LoKr adapters
//! 3. Generate images from text prompts

use eridiffusion_core::{Device, ModelArchitecture, Result, Error};
use eridiffusion_models::{SD3Model, SD15Model, SDXLModel, VAE};
use eridiffusion_networks::{LoKr, NetworkAdapter};
use eridiffusion_inference::{InferencePipeline, InferenceConfig, SchedulerType};
use candle_core::{Tensor, DType};
use candle_nn::VarBuilder;
use std::path::Path;
use std::sync::Arc;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Model architecture to use
    #[arg(long, default_value = "sd3.5")]
    model: String,
    
    /// Path to pretrained model weights
    #[arg(long)]
    model_path: Option<String>,
    
    /// Path to LoKr adapter weights
    #[arg(long)]
    adapter_path: Option<String>,
    
    /// Text prompt for generation
    #[arg(long, default_value = "A beautiful sunset over mountains, highly detailed, 8k")]
    prompt: String,
    
    /// Negative prompt
    #[arg(long)]
    negative_prompt: Option<String>,
    
    /// Number of inference steps
    #[arg(long, default_value = "50")]
    steps: usize,
    
    /// Guidance scale
    #[arg(long, default_value = "7.5")]
    guidance_scale: f32,
    
    /// Image width
    #[arg(long, default_value = "1024")]
    width: usize,
    
    /// Image height
    #[arg(long, default_value = "1024")]
    height: usize,
    
    /// Output path for generated image
    #[arg(long, default_value = "output.png")]
    output: String,
    
    /// Random seed
    #[arg(long)]
    seed: Option<u64>,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Parse arguments
    let args = Args::parse();
    
    println!("AI-Toolkit Inference Demo");
    println!("========================");
    
    // Setup device
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    println!("Using device: {:?}", device);
    
    // Determine model architecture
    let architecture = match args.model.as_str() {
        "sd1.5" => ModelArchitecture::SD15,
        "sdxl" => ModelArchitecture::SDXL,
        "sd3" => ModelArchitecture::SD3,
        "sd3.5" => ModelArchitecture::SD35,
        _ => return Err(Error::InvalidInput(format!("Unknown model: {}", args.model)))
    };
    
    println!("Loading {} model...", args.model);
    
    // Create model
    let model = create_model(architecture, &device)?;
    
    // Create VAE (dummy for demo - would load actual VAE)
    let vae = create_vae(architecture, &device)?;
    
    // Setup inference config
    let config = InferenceConfig {
        num_inference_steps: args.steps,
        guidance_scale: args.guidance_scale,
        seed: args.seed.or(Some(42)),
        height: args.height,
        width: args.width,
        batch_size: 1,
        use_fp16: device.is_cuda(),
        compile: false,
        scheduler: match architecture {
            ModelArchitecture::SD3 | ModelArchitecture::SD35 => SchedulerType::FlowMatch,
            _ => SchedulerType::DDIM,
        },
        ..Default::default()
    };
    
    // Create inference pipeline
    let mut pipeline = InferencePipeline::new(
        Box::new(model),
        Arc::new(vae),
        device.clone(),
        config,
    )?;
    
    // Load LoKr adapter if provided
    if let Some(adapter_path) = args.adapter_path {
        println!("Loading LoKr adapter from {}...", adapter_path);
        
        let mut lokr = LoKr::new(
            Default::default(),
            architecture,
            device.clone(),
        )?;
        
        lokr.load_pretrained(Path::new(&adapter_path)).await?;
        pipeline.add_adapter(Box::new(lokr))?;
        
        println!("LoKr adapter loaded successfully!");
    }
    
    // Add dummy text encoders (would load actual encoders)
    add_text_encoders(&mut pipeline, architecture, &device)?;
    
    // Generate image
    println!("\nGenerating image...");
    println!("Prompt: {}", args.prompt);
    if let Some(neg) = &args.negative_prompt {
        println!("Negative prompt: {}", neg);
    }
    println!("Size: {}x{}", args.width, args.height);
    println!("Steps: {}", args.steps);
    println!("Guidance scale: {}", args.guidance_scale);
    
    let start = std::time::Instant::now();
    
    let images = pipeline.generate(
        &args.prompt,
        args.negative_prompt.as_deref(),
    ).await?;
    
    let elapsed = start.elapsed();
    println!("\nGeneration completed in {:.2}s", elapsed.as_secs_f32());
    
    // Save image
    save_image(&images, &args.output)?;
    println!("Image saved to: {}", args.output);
    
    // Print some statistics
    let (batch, channels, height, width) = images.dims4()?;
    println!("\nGeneration statistics:");
    println!("- Output shape: [{}, {}, {}, {}]", batch, channels, height, width);
    println!("- Steps per second: {:.2}", args.steps as f32 / elapsed.as_secs_f32());
    println!("- Device: {:?}", device);
    
    Ok(())
}

/// Create model based on architecture
fn create_model(architecture: ModelArchitecture, device: &Device) -> Result<impl eridiffusion_core::DiffusionModel> {
    let candle_device = device.as_candle_device()?;
    let vb = unsafe { VarBuilder::uninit(DType::F32, &candle_device) };
    
    match architecture {
        ModelArchitecture::SD15 => SD15Model::new(vb),
        ModelArchitecture::SDXL => SDXLModel::new(vb),
        ModelArchitecture::SD3 => SD3Model::new(vb, false),
        ModelArchitecture::SD35 => SD3Model::new(vb, true),
        _ => Err(Error::Unsupported(format!("Architecture {:?} not implemented", architecture)))
    }
}

/// Create VAE based on architecture
fn create_vae(architecture: ModelArchitecture, device: &Device) -> Result<impl VAE> {
    use eridiffusion_models::vae::{SD3VAE, SD3VAEConfig};
    
    let candle_device = device.as_candle_device()?;
    let vb = unsafe { VarBuilder::uninit(DType::F32, &candle_device) };
    
    // For demo, we'll use SD3 VAE for all architectures
    // In practice, each would have its specific VAE
    let config = match architecture {
        ModelArchitecture::SD3 | ModelArchitecture::SD35 => SD3VAEConfig::default(),
        _ => SD3VAEConfig {
            latent_channels: 4, // SD1.5/SDXL use 4 channels
            ..Default::default()
        }
    };
    
    SD3VAE::new(vb, config)
}

/// Add text encoders to pipeline
fn add_text_encoders(
    pipeline: &mut InferencePipeline,
    architecture: ModelArchitecture,
    device: &Device,
) -> Result<()> {
    use eridiffusion_inference::TextEncoderType;
    
    // Create dummy text encoders for demo
    let dummy_encoder = Arc::new(DummyTextEncoder::new(device.clone())?);
    
    match architecture {
        ModelArchitecture::SD15 => {
            pipeline.add_text_encoder(TextEncoderType::ClipL, dummy_encoder)?;
        }
        ModelArchitecture::SDXL => {
            pipeline.add_text_encoder(TextEncoderType::ClipL, dummy_encoder.clone())?;
            pipeline.add_text_encoder(TextEncoderType::ClipG, dummy_encoder)?;
        }
        ModelArchitecture::SD3 | ModelArchitecture::SD35 => {
            pipeline.add_text_encoder(TextEncoderType::ClipL, dummy_encoder.clone())?;
            pipeline.add_text_encoder(TextEncoderType::ClipG, dummy_encoder.clone())?;
            pipeline.add_text_encoder(TextEncoderType::T5, dummy_encoder)?;
        }
        _ => {}
    }
    
    Ok(())
}

/// Save tensor as image
fn save_image(tensor: &Tensor, path: &str) -> Result<()> {
    // Convert tensor to image
    let (_, c, h, w) = tensor.dims4()?;
    
    // Ensure we have RGB channels
    if c != 3 {
        return Err(Error::InvalidShape(format!("Expected 3 channels, got {}", c)));
    }
    
    // Convert to u8 (0-255)
    let tensor_u8 = tensor.mul_scalar(255.0)?.to_dtype(DType::U8)?;
    
    // Get data as Vec<u8>
    let data = tensor_u8.flatten_all()?.to_vec1::<u8>()?;
    
    // Create image
    let img = image::RgbImage::from_raw(w as u32, h as u32, data)
        .ok_or_else(|| Error::Runtime("Failed to create image".to_string()))?;
    
    // Save image
    img.save(path)
        .map_err(|e| Error::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;
    
    Ok(())
}

/// Dummy text encoder for demonstration
struct DummyTextEncoder {
    device: Device,
}

impl DummyTextEncoder {
    fn new(device: Device) -> Result<Self> {
        Ok(Self { device })
    }
}

impl eridiffusion_models::TextEncoder for DummyTextEncoder {
    fn encode(&self, prompts: &[String]) -> Result<Tensor> {
        let candle_device = self.device.as_candle_device()?;
        let batch_size = prompts.len();
        
        // Create dummy embeddings
        // SD uses 768, SDXL uses 1280/1024, SD3 uses 2048
        let embed_dim = 2048;
        let seq_len = 77;
        
        Ok(Tensor::randn(
            0.0f32,
            0.1,
            &[batch_size, seq_len, embed_dim],
            &candle_device,
        )?)
    }
    
    fn max_length(&self) -> usize {
        77
    }
}

// Extension trait for Device
trait DeviceExt {
    fn as_candle_device(&self) -> Result<candle_core::Device>;
    fn is_cuda(&self) -> bool;
}

impl DeviceExt for Device {
    fn as_candle_device(&self) -> Result<candle_core::Device> {
        match self {
            Device::Cpu => Ok(candle_core::Device::Cpu),
            Device::Cuda(id) => Ok(candle_core::Device::cuda(*id)?),
        }
    }
    
    fn is_cuda(&self) -> bool {
        matches!(self, Device::Cuda(_))
    }
}

// Import InferencePipeline types
mod eridiffusion_inference {
    use super::*;
    
    pub struct InferencePipeline {
        model: Box<dyn eridiffusion_core::DiffusionModel>,
        vae: Arc<dyn eridiffusion_models::VAE>,
        text_encoders: HashMap<String, Arc<dyn eridiffusion_models::TextEncoder>>,
        adapters: Vec<Box<dyn NetworkAdapter>>,
        device: Device,
        config: InferenceConfig,
    }
    
    #[derive(Default)]
    pub struct InferenceConfig {
        pub num_inference_steps: usize,
        pub guidance_scale: f32,
        pub seed: Option<u64>,
        pub height: usize,
        pub width: usize,
        pub batch_size: usize,
        pub use_fp16: bool,
        pub compile: bool,
        pub scheduler: SchedulerType,
    }
    
    #[derive(Clone, Copy)]
    pub enum SchedulerType {
        DDIM,
        FlowMatch,
    }
    
    impl Default for SchedulerType {
        fn default() -> Self {
            Self::DDIM
        }
    }
    
    pub enum TextEncoderType {
        ClipL,
        ClipG,
        T5,
    }
    
    impl InferencePipeline {
        pub fn new(
            model: Box<dyn eridiffusion_core::DiffusionModel>,
            vae: Arc<dyn eridiffusion_models::VAE>,
            device: Device,
            config: InferenceConfig,
        ) -> Result<Self> {
            Ok(Self {
                model,
                vae,
                text_encoders: HashMap::new(),
                adapters: Vec::new(),
                device,
                config,
            })
        }
        
        pub fn add_text_encoder(
            &mut self,
            encoder_type: TextEncoderType,
            encoder: Arc<dyn eridiffusion_models::TextEncoder>,
        ) -> Result<()> {
            let key = match encoder_type {
                TextEncoderType::ClipL => "clip_l",
                TextEncoderType::ClipG => "clip_g", 
                TextEncoderType::T5 => "t5",
            };
            self.text_encoders.insert(key.to_string(), encoder);
            Ok(())
        }
        
        pub fn add_adapter(&mut self, adapter: Box<dyn NetworkAdapter>) -> Result<()> {
            self.adapters.push(adapter);
            Ok(())
        }
        
        pub async fn generate(
            &self,
            prompt: &str,
            negative_prompt: Option<&str>,
        ) -> Result<Tensor> {
            println!("Starting generation with prompt: {}", prompt);
            
            // For demo, return a dummy image tensor
            let candle_device = self.device.as_candle_device()?;
            let h = self.config.height;
            let w = self.config.width;
            
            // Create a gradient image for demonstration
            let mut data = vec![0.0f32; 3 * h * w];
            for y in 0..h {
                for x in 0..w {
                    let idx = (y * w + x) * 3;
                    data[idx] = x as f32 / w as f32;     // R
                    data[idx + 1] = y as f32 / h as f32; // G
                    data[idx + 2] = 0.5;                  // B
                }
            }
            
            Tensor::from_vec(data, &[1, 3, h, w], &candle_device)
        }
    }
}