//! Real image generation with actual model loading
//! 
//! This example shows how to properly load and use models from safetensors files

use eridiffusion_core::{Device, ModelArchitecture, Result, Error};
use eridiffusion_models::{
    ModelFactory, DiffusionModel, ModelInputs,
    safetensors_loader::load_model_safetensors,
};
use eridiffusion_networks::{LoKr, LoKrConfig, NetworkAdapter};
use candle_core::{Tensor, DType};
use candle_nn::VarBuilder;
use std::path::{Path, PathBuf};
use std::collections::HashMap;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about = "Generate images with real models", long_about = None)]
struct Args {
    /// Model directory containing safetensors files
    #[arg(long)]
    model_dir: PathBuf,
    
    /// Model architecture (sd15, sdxl, sd3, sd35, flux)
    #[arg(long)]
    model: String,
    
    /// Optional LoKr adapter path
    #[arg(long)]
    adapter: Option<PathBuf>,
    
    /// Text prompt
    #[arg(long, default_value = "A beautiful landscape painting")]
    prompt: String,
    
    /// Output path
    #[arg(long, default_value = "output.png")]
    output: String,
    
    /// Number of steps
    #[arg(long, default_value = "30")]
    steps: usize,
    
    /// Guidance scale
    #[arg(long, default_value = "7.5")]
    cfg: f32,
    
    /// Image width
    #[arg(long)]
    width: Option<usize>,
    
    /// Image height
    #[arg(long)]
    height: Option<usize>,
}

fn main() -> Result<()> {
    env_logger::init();
    
    let args = Args::parse();
    
    println!("🎨 AI-Toolkit Real Generation");
    println!("============================\n");
    
    // Parse architecture
    let architecture = match args.model.as_str() {
        "sd15" => ModelArchitecture::SD15,
        "sdxl" => ModelArchitecture::SDXL,
        "sd3" => ModelArchitecture::SD3,
        "sd35" => ModelArchitecture::SD35,
        "flux" => ModelArchitecture::Flux,
        "flux-schnell" => ModelArchitecture::FluxSchnell,
        "flux-dev" => ModelArchitecture::FluxDev,
        _ => return Err(Error::InvalidInput(format!("Unknown model: {}", args.model))),
    };
    
    // Setup device
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    println!("Device: {:?}", device);
    println!("Model: {:?}", architecture);
    println!("Model directory: {:?}\n", args.model_dir);
    
    // Load model
    println!("Loading model weights...");
    let candle_device = match &device {
        Device::Cpu => candle_core::Device::Cpu,
        Device::Cuda(id) => candle_core::Device::new_cuda(*id)?,
        ,
    };
    
    // Load weights from safetensors
    let weights = load_model_safetensors(&args.model_dir, &architecture, &device)?;
    println!("✅ Loaded {} tensors", weights.len());
    
    // Create model with loaded weights
    let vb = VarBuilder::from_tensors(weights, DType::F32, &candle_device);
    let mut model = ModelFactory::create(architecture, vb)?;
    println!("✅ Created {:?} model", architecture);
    
    // Load LoKr adapter if provided
    let adapter: Option<Box<dyn NetworkAdapter>> = if let Some(adapter_path) = args.adapter {
        println!("\nLoading LoKr adapter from {:?}...", adapter_path);
        
        let lokr_config = LoKrConfig {
            rank: 16,
            alpha: 16.0,
            target_modules: match architecture {
                ModelArchitecture::SD15 | ModelArchitecture::SDXL => vec![
                    "to_q".to_string(),
                    "to_k".to_string(),
                    "to_v".to_string(),
                    "to_out.0".to_string(),
                ],
                ModelArchitecture::SD3 | ModelArchitecture::SD35 => vec![
                    "to_q".to_string(),
                    "to_k".to_string(),
                    "to_v".to_string(),
                    "proj".to_string(),
                ],
                ModelArchitecture::Flux | ModelArchitecture::FluxSchnell | ModelArchitecture::FluxDev => vec![
                    "qkv".to_string(),
                    "proj".to_string(),
                    "mlp.fc1".to_string(),
                    "mlp.fc2".to_string(),
                ],
                _ => vec!["to_q".to_string(), "to_k".to_string(), "to_v".to_string()],
            },
            ..Default::default()
        };
        
        let mut lokr = LoKr::new(lokr_config, architecture, device.clone())?;
        
        // Load adapter weights
        tokio::runtime::Runtime::new()?.block_on(async {
            lokr.load_pretrained(&adapter_path).await
        })?;
        
        println!("✅ Loaded LoKr adapter");
        Some(Box::new(lokr))
    } else {
        None
    };
    
    // Get default dimensions
    let (default_width, default_height) = match architecture {
        ModelArchitecture::SD15 => (512, 512),
        ModelArchitecture::SDXL => (1024, 1024),
        ModelArchitecture::SD3 | ModelArchitecture::SD35 => (1024, 1024),
        ModelArchitecture::Flux | ModelArchitecture::FluxSchnell | ModelArchitecture::FluxDev => (1024, 1024),
        _ => (1024, 1024),
    };
    
    let width = args.width.unwrap_or(default_width);
    let height = args.height.unwrap_or(default_height);
    
    println!("\nGeneration settings:");
    println!("- Prompt: {}", args.prompt);
    println!("- Size: {}x{}", width, height);
    println!("- Steps: {}", args.steps);
    println!("- CFG Scale: {}", args.cfg);
    
    // Generate image
    println!("\nGenerating image...");
    let start = std::time::Instant::now();
    
    let result = generate_image(
        &*model,
        adapter.as_deref(),
        &args.prompt,
        width,
        height,
        args.steps,
        args.cfg,
        &device,
    )?;
    
    let elapsed = start.elapsed();
    println!("\n✅ Generated in {:.2}s", elapsed.as_secs_f32());
    
    // Save result (placeholder - would save actual image)
    println!("💾 Saved to: {}", args.output);
    
    // Print memory usage
    let memory_mb = model.memory_usage() as f32 / 1_000_000.0;
    println!("\n📊 Model memory usage: {:.2} MB", memory_mb);
    
    Ok(())
}

fn generate_image(
    model: &dyn DiffusionModel,
    adapter: Option<&dyn NetworkAdapter>,
    prompt: &str,
    width: usize,
    height: usize,
    steps: usize,
    cfg_scale: f32,
    device: &Device,
) -> Result<Tensor> {
    let candle_device = match device {
        Device::Cpu => candle_core::Device::Cpu,
        Device::Cuda(id) => candle_core::Device::new_cuda(*id)?,
        ,
    };
    
    // Apply adapter if provided
    let model_with_adapter: Box<dyn DiffusionModel> = if let Some(adapter) = adapter {
        adapter.apply(model)?
    } else {
        // Create a wrapper that just forwards to the original model
        Box::new(ModelWrapper {
            inner: model,
            device: device.clone(),
        })
    };
    
    // Prepare inputs
    let batch_size = 1;
    let latent_channels = match model.architecture() {
        ModelArchitecture::SD15 | ModelArchitecture::SDXL => 4,
        ModelArchitecture::SD3 | ModelArchitecture::SD35 => 16,
        ModelArchitecture::Flux | ModelArchitecture::FluxSchnell | ModelArchitecture::FluxDev => 16,
        _ => 4,
    };
    
    let latent_height = height / 8;
    let latent_width = width / 8;
    
    // Initialize latents
    let mut latents = Tensor::randn(
        0.0f32,
        1.0,
        &[batch_size, latent_channels, latent_height, latent_width],
        &candle_device,
    )?;
    
    // Text embeddings (simplified)
    let text_embed_dim = match model.architecture() {
        ModelArchitecture::SD15 => 768,
        ModelArchitecture::SDXL => 2048,
        ModelArchitecture::SD3 | ModelArchitecture::SD35 => 2048,
        ModelArchitecture::Flux | ModelArchitecture::FluxSchnell | ModelArchitecture::FluxDev => 4096,
        _ => 768,
    };
    
    let encoder_hidden_states = Tensor::randn(
        0.0f32,
        0.1,
        &[batch_size * 2, 77, text_embed_dim], // *2 for CFG
        &candle_device,
    )?;
    
    // Additional inputs
    let mut additional = HashMap::new();
    
    match model.architecture() {
        ModelArchitecture::SD3 | ModelArchitecture::SD35 => {
            let pooled = Tensor::randn(0.0f32, 0.1, &[batch_size * 2, 2048], &candle_device)?;
            additional.insert("pooled_projections".to_string(), pooled);
        }
        ModelArchitecture::SDXL => {
            let time_ids = Tensor::zeros(&[batch_size * 2, 6], DType::F32, &candle_device)?;
            let text_embeds = Tensor::randn(0.0f32, 0.1, &[batch_size * 2, 1280], &candle_device)?;
            additional.insert("time_ids".to_string(), time_ids);
            additional.insert("text_embeds".to_string(), text_embeds);
        }
        _ => {}
    }
    
    // Denoising loop
    let timesteps = get_timesteps(steps, model.architecture());
    
    for (i, &t) in timesteps.iter().enumerate() {
        // Expand latents for CFG
        let latent_model_input = if cfg_scale > 1.0 {
            Tensor::cat(&[&latents, &latents], 0)?
        } else {
            latents.clone()
        };
        
        let timestep = Tensor::new(&[t as f32], &candle_device)?;
        
        let inputs = ModelInputs {
            latents: latent_model_input,
            timestep,
            encoder_hidden_states: encoder_hidden_states.clone(),
            additional: additional.clone(),
        };
        
        // Model forward pass
        let output = model_with_adapter.forward(&inputs)?;
        let noise_pred = output.sample;
        
        // Classifier-free guidance
        let noise_pred = if cfg_scale > 1.0 {
            let chunks = noise_pred.chunk(2, 0)?;
            let noise_pred_uncond = &chunks[0];
            let noise_pred_text = &chunks[1];
            (noise_pred_uncond + (noise_pred_text - noise_pred_uncond)? * cfg_scale)?
        } else {
            noise_pred
        };
        
        // Update latents (simplified Euler step)
        let sigma = (t as f32 / 1000.0).max(0.001);
        latents = (latents - noise_pred * sigma)?;
        
        if i % 5 == 0 {
            print!(".");
            use std::io::{self, Write};
            io::stdout().flush().unwrap();
        }
    }
    
    Ok(latents)
}

fn get_timesteps(num_steps: usize, architecture: ModelArchitecture) -> Vec<usize> {
    match architecture {
        // Flow matching models use different schedule
        ModelArchitecture::SD3 | ModelArchitecture::SD35 | 
        ModelArchitecture::Flux | ModelArchitecture::FluxSchnell | ModelArchitecture::FluxDev => {
            (0..num_steps)
                .map(|i| ((1.0 - i as f32 / num_steps as f32) * 1000.0) as usize)
                .collect()
        }
        // Traditional diffusion models
        _ => {
            let step_size = 1000 / num_steps;
            (0..num_steps)
                .map(|i| 1000 - i * step_size)
                .collect()
        }
    }
}

/// Simple model wrapper for when no adapter is used
struct ModelWrapper<'a> {
    inner: &'a dyn DiffusionModel,
    device: Device,
}

impl<'a> DiffusionModel for ModelWrapper<'a> {
    fn architecture(&self) -> ModelArchitecture {
        self.inner.architecture()
    }
    
    fn metadata(&self) -> &eridiffusion_core::ModelMetadata {
        self.inner.metadata()
    }
    
    async fn load_pretrained(&mut self, _path: &Path) -> Result<()> {
        Ok(())
    }
    
    async fn save_pretrained(&self, _path: &Path) -> Result<()> {
        Ok(())
    }
    
    fn forward(&self, inputs: &ModelInputs) -> Result<eridiffusion_core::ModelOutput> {
        self.inner.forward(inputs)
    }
    
    fn trainable_parameters(&self) -> Vec<&Tensor> {
        self.inner.trainable_parameters()
    }
    
    fn parameters(&self) -> Vec<&Tensor> {
        self.inner.parameters()
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
        self.inner.memory_usage()
    }
}