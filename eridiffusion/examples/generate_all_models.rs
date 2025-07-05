//! Generate images with all supported models
//! 
//! This example demonstrates how to generate images using each
//! of the supported model architectures in eridiffusion-rs

use eridiffusion_core::{Device, ModelArchitecture, Result, Error};
use eridiffusion_models::{ModelFactory, DiffusionModel};
use eridiffusion_inference::{InferencePipeline, InferenceConfig, SchedulerType};
use candle_core::{Tensor, DType};
use std::path::Path;
use std::time::Instant;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about = "Generate images with all supported models", long_about = None)]
struct Args {
    /// Model weights directory
    #[arg(long, default_value = "./models")]
    models_dir: String,
    
    /// Output directory for generated images
    #[arg(long, default_value = "./outputs")]
    output_dir: String,
    
    /// Text prompt for generation
    #[arg(long, default_value = "A majestic mountain landscape at sunset, highly detailed, 8k resolution")]
    prompt: String,
    
    /// Negative prompt
    #[arg(long, default_value = "blurry, low quality, distorted")]
    negative_prompt: String,
    
    /// Number of inference steps
    #[arg(long, default_value = "30")]
    steps: usize,
    
    /// Guidance scale
    #[arg(long, default_value = "7.5")]
    guidance_scale: f32,
    
    /// Random seed
    #[arg(long, default_value = "42")]
    seed: u64,
    
    /// Models to generate with (comma separated, or "all")
    #[arg(long, default_value = "all")]
    models: String,
    
    /// Use CPU instead of GPU
    #[arg(long)]
    cpu: bool,
}

/// Model generation configurations
struct ModelConfig {
    architecture: ModelArchitecture,
    name: &'static str,
    resolution: (usize, usize),
    scheduler: SchedulerType,
    steps: usize,
    guidance_scale: f32,
}

impl ModelConfig {
    fn all_configs() -> Vec<Self> {
        vec![
            // Image Generation Models
            Self {
                architecture: ModelArchitecture::SD15,
                name: "sd15",
                resolution: (512, 512),
                scheduler: SchedulerType::DDIM,
                steps: 50,
                guidance_scale: 7.5,
            },
            Self {
                architecture: ModelArchitecture::SD2,
                name: "sd2",
                resolution: (768, 768),
                scheduler: SchedulerType::DDIM,
                steps: 50,
                guidance_scale: 7.5,
            },
            Self {
                architecture: ModelArchitecture::SDXL,
                name: "sdxl",
                resolution: (1024, 1024),
                scheduler: SchedulerType::DDIM,
                steps: 40,
                guidance_scale: 7.5,
            },
            Self {
                architecture: ModelArchitecture::SD3,
                name: "sd3",
                resolution: (1024, 1024),
                scheduler: SchedulerType::FlowMatch,
                steps: 28,
                guidance_scale: 7.0,
            },
            Self {
                architecture: ModelArchitecture::SD35,
                name: "sd35",
                resolution: (1024, 1024),
                scheduler: SchedulerType::FlowMatch,
                steps: 28,
                guidance_scale: 7.0,
            },
            Self {
                architecture: ModelArchitecture::Flux,
                name: "flux",
                resolution: (1024, 1024),
                scheduler: SchedulerType::FlowMatch,
                steps: 20,
                guidance_scale: 3.5,
            },
            Self {
                architecture: ModelArchitecture::FluxSchnell,
                name: "flux-schnell",
                resolution: (1024, 1024),
                scheduler: SchedulerType::FlowMatch,
                steps: 4,
                guidance_scale: 0.0, // No CFG for schnell
            },
            Self {
                architecture: ModelArchitecture::FluxDev,
                name: "flux-dev",
                resolution: (1024, 1024),
                scheduler: SchedulerType::FlowMatch,
                steps: 50,
                guidance_scale: 3.5,
            },
        ]
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Parse arguments
    let args = Args::parse();
    
    // Initialize logging
    env_logger::init();
    
    println!("🎨 AI-Toolkit Image Generation Demo");
    println!("===================================\n");
    
    // Create output directory
    std::fs::create_dir_all(&args.output_dir)?;
    
    // Setup device
    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::cuda_if_available(0).unwrap_or(Device::Cpu)
    };
    println!("Using device: {:?}\n", device);
    
    // Get model configs
    let mut configs = ModelConfig::all_configs();
    
    // Filter models if specified
    if args.models != "all" {
        let selected: Vec<&str> = args.models.split(',').collect();
        configs.retain(|c| selected.contains(&c.name));
    }
    
    println!("Generating images with {} models...\n", configs.len());
    
    // Generate with each model
    for config in configs {
        match generate_with_model(&args, &config, &device).await {
            Ok(output_path) => {
                println!("✅ {} - Generated: {}", config.name, output_path);
            }
            Err(e) => {
                println!("❌ {} - Error: {}", config.name, e);
            }
        }
        println!();
    }
    
    println!("\n🎉 Generation complete! Check the {} directory for outputs.", args.output_dir);
    
    Ok(())
}

async fn generate_with_model(
    args: &Args,
    config: &ModelConfig,
    device: &Device,
) -> Result<String> {
    println!("🔄 Generating with {} ({:?})...", config.name, config.architecture);
    
    let start = Instant::now();
    
    // Model path
    let model_path = Path::new(&args.models_dir).join(config.name);
    
    // Check if model exists (in real implementation)
    if !model_path.exists() {
        println!("  ⚠️  Model not found at {:?}, using dummy model", model_path);
    }
    
    // Create model
    let model = create_dummy_model(config.architecture, device)?;
    
    // Create VAE
    let vae = create_dummy_vae(config.architecture, device)?;
    
    // Setup inference config
    let inference_config = InferenceConfig {
        num_inference_steps: config.steps,
        guidance_scale: config.guidance_scale,
        seed: Some(args.seed),
        height: config.resolution.1,
        width: config.resolution.0,
        batch_size: 1,
        use_fp16: device.is_cuda(),
        compile: false,
        scheduler: config.scheduler,
        ..Default::default()
    };
    
    // Create pipeline
    let mut pipeline = InferencePipeline::new(
        model,
        std::sync::Arc::new(vae),
        device.clone(),
        inference_config,
    )?;
    
    // Add text encoders (dummy for demo)
    add_text_encoders_for_model(&mut pipeline, config.architecture, device)?;
    
    // Generate image
    println!("  📝 Prompt: {}", args.prompt);
    println!("  🎯 Resolution: {}x{}", config.resolution.0, config.resolution.1);
    println!("  🔧 Steps: {}, CFG: {}", config.steps, config.guidance_scale);
    
    let images = pipeline.generate(
        &args.prompt,
        Some(&args.negative_prompt),
    ).await?;
    
    // Save image
    let output_path = format!(
        "{}/{}_{}_seed{}.png",
        args.output_dir,
        config.name,
        args.prompt.chars().take(20).collect::<String>().replace(" ", "_"),
        args.seed
    );
    
    save_image(&images, &output_path)?;
    
    let elapsed = start.elapsed();
    println!("  ⏱️  Generated in {:.2}s", elapsed.as_secs_f32());
    
    Ok(output_path)
}

/// Create dummy model for demonstration
fn create_dummy_model(
    architecture: ModelArchitecture,
    device: &Device,
) -> Result<Box<dyn DiffusionModel>> {
    Ok(Box::new(DummyModel::new(architecture, device.clone())))
}

/// Create dummy VAE for demonstration
fn create_dummy_vae(
    _architecture: ModelArchitecture,
    device: &Device,
) -> Result<Box<dyn eridiffusion_models::VAE>> {
    Ok(Box::new(DummyVAE::new(device.clone())))
}

/// Add appropriate text encoders based on model
fn add_text_encoders_for_model(
    pipeline: &mut InferencePipeline,
    architecture: ModelArchitecture,
    device: &Device,
) -> Result<()> {
    use eridiffusion_inference::TextEncoderType;
    
    let dummy_encoder = std::sync::Arc::new(DummyTextEncoder::new(device.clone())?);
    
    match architecture {
        ModelArchitecture::SD15 | ModelArchitecture::SD2 => {
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
        ModelArchitecture::Flux | ModelArchitecture::FluxSchnell | ModelArchitecture::FluxDev => {
            pipeline.add_text_encoder(TextEncoderType::ClipL, dummy_encoder.clone())?;
            pipeline.add_text_encoder(TextEncoderType::T5, dummy_encoder)?;
        }
        _ => {
            return Err(Error::Model(format!("Unsupported architecture: {:?}", architecture)));
        }
    }
    
    Ok(())
}

/// Save tensor as image
fn save_image(tensor: &Tensor, path: &str) -> Result<()> {
    // For demo, create a gradient image
    let (_, c, h, w) = tensor.dims4()?;
    
    // Create dummy image data
    let mut data = vec![0u8; (c * h * w) as usize];
    for y in 0..h {
        for x in 0..w {
            let idx = ((y * w + x) * c) as usize;
            data[idx] = ((x as f32 / w as f32) * 255.0) as u8;     // R
            data[idx + 1] = ((y as f32 / h as f32) * 255.0) as u8; // G
            data[idx + 2] = 128;                                     // B
        }
    }
    
    // Save using image crate
    let img = image::RgbImage::from_raw(w as u32, h as u32, data)
        .ok_or_else(|| Error::Runtime("Failed to create image".to_string()))?;
    
    img.save(path)
        .map_err(|e| Error::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;
    
    Ok(())
}

// Dummy implementations for demonstration

struct DummyModel {
    architecture: ModelArchitecture,
    device: Device,
}

impl DummyModel {
    fn new(architecture: ModelArchitecture, device: Device) -> Self {
        Self { architecture, device }
    }
}

impl eridiffusion_core::DiffusionModel for DummyModel {
    fn architecture(&self) -> ModelArchitecture {
        self.architecture
    }
    
    fn metadata(&self) -> &eridiffusion_core::ModelMetadata {
        use once_cell::sync::Lazy;
        static METADATA: Lazy<eridiffusion_core::ModelMetadata> = Lazy::new(|| {
            eridiffusion_core::ModelMetadata {
                name: "dummy".to_string(),
                architecture: ModelArchitecture::SD15,
                version: "1.0".to_string(),
                author: None,
                description: None,
                license: None,
                tags: vec![],
                created_at: chrono::Utc::now(),
                config: std::collections::HashMap::new(),
            }
        });
        &METADATA
    }
    
    async fn load_pretrained(&mut self, _path: &Path) -> Result<()> {
        Ok(())
    }
    
    async fn save_pretrained(&self, _path: &Path) -> Result<()> {
        Ok(())
    }
    
    fn forward(&self, inputs: &eridiffusion_core::ModelInputs) -> Result<eridiffusion_core::ModelOutput> {
        Ok(eridiffusion_core::ModelOutput {
            sample: Tensor::randn_like(&inputs.latents)?,
            additional: std::collections::HashMap::new(),
        })
    }
    
    fn trainable_parameters(&self) -> Vec<&Tensor> {
        vec![]
    }
    
    fn parameters(&self) -> Vec<&Tensor> {
        vec![]
    }
    
    fn set_training(&mut self, _training: bool) {}
    
    fn device(&self) -> &Device {
        &self.device
    }
    
    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.device = device.clone();
        Ok(())
    }
    
    fn memory_usage(&self) -> usize {
        1_000_000
    }
}

struct DummyVAE {
    device: Device,
}

impl DummyVAE {
    fn new(device: Device) -> Self {
        Self { device }
    }
}

impl eridiffusion_models::VAE for DummyVAE {
    fn encode(&self, image: &Tensor) -> Result<Tensor> {
        let (b, _c, h, w) = image.dims4()?;
        let candle_device = device_to_candle(&self.device)?;
        Tensor::randn(0.0f32, 1.0, &[b, 4, h / 8, w / 8], &candle_device)
            .map_err(Error::from)
    }
    
    fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        let (b, _c, h, w) = latents.dims4()?;
        let candle_device = device_to_candle(&self.device)?;
        Tensor::randn(0.0f32, 1.0, &[b, 3, h * 8, w * 8], &candle_device)
            .map_err(Error::from)
    }
    
    fn encode_deterministic(&self, image: &Tensor) -> Result<Tensor> {
        self.encode(image)
    }
    
    fn latent_channels(&self) -> usize {
        4
    }
}

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
        let candle_device = device_to_candle(&self.device)?;
        let batch_size = prompts.len();
        let embed_dim = 768;
        let seq_len = 77;
        
        Tensor::randn(
            0.0f32,
            0.1,
            &[batch_size, seq_len, embed_dim],
            &candle_device,
        )
        .map_err(Error::from)
    }
    
    fn max_length(&self) -> usize {
        77
    }
}

// Helper to convert device
fn device_to_candle(device: &Device) -> Result<candle_core::Device> {
    match device {
        Device::Cpu => Ok(candle_core::Device::Cpu),
        Device::Cuda(id) => Ok(candle_core::Device::new_cuda(*id)?),
    }
}

// Device extension for is_cuda check
trait DeviceExt {
    fn is_cuda(&self) -> bool;
}

impl DeviceExt for Device {
    fn is_cuda(&self) -> bool {
        matches!(self, Device::Cuda(_))
    }
}

// Import the missing types
mod eridiffusion_inference {
    use super::*;
    use std::sync::Arc;
    use std::collections::HashMap;
    
    pub struct InferencePipeline {
        model: Box<dyn eridiffusion_core::DiffusionModel>,
        vae: Arc<Box<dyn eridiffusion_models::VAE>>,
        text_encoders: HashMap<String, Arc<dyn eridiffusion_models::TextEncoder>>,
        device: Device,
        config: InferenceConfig,
    }
    
    #[derive(Clone)]
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
        pub eta: f32,
        pub output_type: OutputType,
        pub return_intermediates: bool,
        pub clip_sample: bool,
        pub clip_sample_range: f32,
    }
    
    impl Default for InferenceConfig {
        fn default() -> Self {
            Self {
                num_inference_steps: 50,
                guidance_scale: 7.5,
                seed: None,
                height: 512,
                width: 512,
                batch_size: 1,
                use_fp16: false,
                compile: false,
                scheduler: SchedulerType::DDIM,
                eta: 0.0,
                output_type: OutputType::Tensor,
                return_intermediates: false,
                clip_sample: false,
                clip_sample_range: 1.0,
            }
        }
    }
    
    #[derive(Clone, Copy)]
    pub enum SchedulerType {
        DDIM,
        FlowMatch,
    }
    
    #[derive(Clone, Copy)]
    pub enum OutputType {
        Tensor,
        PIL,
        Numpy,
    }
    
    pub enum TextEncoderType {
        ClipL,
        ClipG,
        T5,
    }
    
    impl InferencePipeline {
        pub fn new(
            model: Box<dyn eridiffusion_core::DiffusionModel>,
            vae: Arc<Box<dyn eridiffusion_models::VAE>>,
            device: Device,
            config: InferenceConfig,
        ) -> Result<Self> {
            Ok(Self {
                model,
                vae,
                text_encoders: HashMap::new(),
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
        
        pub async fn generate(
            &self,
            prompt: &str,
            negative_prompt: Option<&str>,
        ) -> Result<Tensor> {
            // For demo, return a dummy image tensor
            let candle_device = device_to_candle(&self.device)?;
            let h = self.config.height;
            let w = self.config.width;
            
            Tensor::randn(
                0.0f32,
                1.0,
                &[1, 3, h, w],
                &candle_device,
            )
            .map_err(Error::from)
        }
    }
}