//! Quick image generation example
//! 
//! Generate images with SD1.5, SDXL, SD3.5, and Flux

use eridiffusion_core::{Device, ModelArchitecture, Result, Error};
use eridiffusion_models::{ModelFactory, DiffusionModel, ModelInputs, ModelOutput};
use candle_core::{Tensor, DType};
use candle_nn::VarBuilder;
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

/// Generate configuration for a model
#[derive(Debug)]
struct GenerationConfig {
    model: ModelArchitecture,
    prompt: String,
    negative_prompt: String,
    width: usize,
    height: usize,
    steps: usize,
    cfg_scale: f32,
    seed: u64,
}

impl GenerationConfig {
    fn new(model: ModelArchitecture, prompt: &str) -> Self {
        let (width, height, steps, cfg_scale) = match model {
            ModelArchitecture::SD15 => (512, 512, 50, 7.5),
            ModelArchitecture::SDXL => (1024, 1024, 40, 7.5),
            ModelArchitecture::SD3 | ModelArchitecture::SD35 => (1024, 1024, 28, 7.0),
            ModelArchitecture::Flux => (1024, 1024, 20, 3.5),
            _ => (1024, 1024, 30, 7.0),
        };
        
        Self {
            model,
            prompt: prompt.to_string(),
            negative_prompt: "blurry, low quality".to_string(),
            width,
            height,
            steps,
            cfg_scale,
            seed: 42,
        }
    }
}

fn main() -> Result<()> {
    println!("🎨 AI-Toolkit Quick Generation Demo");
    println!("===================================\n");
    
    // Setup device
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    println!("Using device: {:?}\n", device);
    
    // Define test prompts
    let prompts = vec![
        "A serene Japanese garden with cherry blossoms",
        "A futuristic cityscape at night with neon lights",
        "A majestic dragon flying over mountain peaks",
        "An astronaut riding a horse on Mars",
    ];
    
    // Models to test
    let models = vec![
        ModelArchitecture::SD15,
        ModelArchitecture::SDXL,
        ModelArchitecture::SD35,
        ModelArchitecture::Flux,
    ];
    
    // Generate with each model
    for model in models {
        println!("\n📸 Generating with {:?}", model);
        println!("=" .repeat(50));
        
        for (i, prompt) in prompts.iter().enumerate() {
            let config = GenerationConfig::new(model, prompt);
            
            match generate_image(&config, &device) {
                Ok(duration) => {
                    println!("✅ Image {}: Generated in {:.2}s", i + 1, duration);
                    println!("   Prompt: {}", prompt);
                    println!("   Size: {}x{}, Steps: {}", config.width, config.height, config.steps);
                }
                Err(e) => {
                    println!("❌ Image {}: Failed - {}", i + 1, e);
                }
            }
        }
    }
    
    println!("\n✨ Demo complete!");
    Ok(())
}

fn generate_image(config: &GenerationConfig, device: &Device) -> Result<f32> {
    let start = Instant::now();
    
    // Create model
    let candle_device = match device {
        Device::Cpu => candle_core::Device::Cpu,
        Device::Cuda(id) => candle_core::Device::new_cuda(*id)?,
        ,
    };
    
    // Create dummy variable builder
    let vb = unsafe { VarBuilder::uninit(DType::F32, &candle_device) };
    
    // Create model based on architecture
    let model: Box<dyn DiffusionModel> = match config.model {
        ModelArchitecture::SD15 => {
            Box::new(DemoModel::new(config.model, device.clone()))
        }
        ModelArchitecture::SDXL => {
            Box::new(DemoModel::new(config.model, device.clone()))
        }
        ModelArchitecture::SD3 | ModelArchitecture::SD35 => {
            Box::new(DemoModel::new(config.model, device.clone()))
        }
        ModelArchitecture::Flux => {
            Box::new(DemoModel::new(config.model, device.clone()))
        }
        _ => return Err(Error::Unsupported(format!("Model {:?} not implemented", config.model))),
    };
    
    // Prepare inputs
    let batch_size = 1;
    let latent_channels = match config.model {
        ModelArchitecture::SD15 | ModelArchitecture::SDXL => 4,
        ModelArchitecture::SD3 | ModelArchitecture::SD35 | ModelArchitecture::Flux => 16,
        _ => 4,
    };
    
    let latent_height = config.height / 8;
    let latent_width = config.width / 8;
    
    // Initialize random latents
    let latents = Tensor::randn(
        0.0f32,
        1.0,
        &[batch_size, latent_channels, latent_height, latent_width],
        &candle_device,
    )?;
    
    // Create timesteps
    let timesteps: Vec<f32> = (0..config.steps)
        .map(|i| 1000.0 * (1.0 - i as f32 / config.steps as f32))
        .collect();
    
    // Dummy text embeddings
    let text_embed_dim = match config.model {
        ModelArchitecture::SD15 => 768,
        ModelArchitecture::SDXL => 2048,
        ModelArchitecture::SD3 | ModelArchitecture::SD35 => 2048,
        ModelArchitecture::Flux => 4096,
        _ => 768,
    };
    
    let encoder_hidden_states = Tensor::randn(
        0.0f32,
        0.1,
        &[batch_size, 77, text_embed_dim],
        &candle_device,
    )?;
    
    // Additional inputs for specific models
    let mut additional = HashMap::new();
    
    match config.model {
        ModelArchitecture::SD3 | ModelArchitecture::SD35 => {
            // SD3/3.5 need pooled projections
            let pooled = Tensor::randn(0.0f32, 0.1, &[batch_size, 2048], &candle_device)?;
            additional.insert("pooled_projections".to_string(), pooled);
        }
        ModelArchitecture::SDXL => {
            // SDXL needs time ids and text embeds
            let time_ids = Tensor::zeros(&[batch_size, 6], DType::F32, &candle_device)?;
            let text_embeds = Tensor::randn(0.0f32, 0.1, &[batch_size, 1280], &candle_device)?;
            additional.insert("time_ids".to_string(), time_ids);
            additional.insert("text_embeds".to_string(), text_embeds);
        }
        _ => {}
    }
    
    // Simulate denoising loop
    let mut current_latents = latents.clone();
    
    for (step, &t) in timesteps.iter().enumerate() {
        let timestep = Tensor::new(&[t], &candle_device)?;
        
        let inputs = ModelInputs {
            latents: current_latents.clone(),
            timestep,
            encoder_hidden_states: encoder_hidden_states.clone(),
            additional: additional.clone(),
        };
        
        // Get model prediction
        let output = model.forward(&inputs)?;
        
        // Simple Euler step (for demo)
        let noise_pred = output.sample;
        let sigma = (t / 1000.0).max(0.001);
        current_latents = (current_latents - noise_pred * sigma)?;
        
        if step % 10 == 0 {
            print!(".");
            use std::io::{self, Write};
            io::stdout().flush().unwrap();
        }
    }
    
    println!();
    
    // In real implementation, decode latents to image here
    // For demo, we just return the generation time
    
    Ok(start.elapsed().as_secs_f32())
}

/// Demo model for testing
struct DemoModel {
    architecture: ModelArchitecture,
    device: Device,
}

impl DemoModel {
    fn new(architecture: ModelArchitecture, device: Device) -> Self {
        Self { architecture, device }
    }
}

impl eridiffusion_core::DiffusionModel for DemoModel {
    fn architecture(&self) -> ModelArchitecture {
        self.architecture
    }
    
    fn metadata(&self) -> &eridiffusion_core::ModelMetadata {
        use once_cell::sync::Lazy;
        static METADATA: Lazy<eridiffusion_core::ModelMetadata> = Lazy::new(|| {
            eridiffusion_core::ModelMetadata {
                name: "demo".to_string(),
                architecture: ModelArchitecture::SD15,
                version: "1.0".to_string(),
                author: None,
                description: None,
                license: None,
                tags: vec![],
                created_at: chrono::Utc::now(),
                config: HashMap::new(),
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
    
    fn forward(&self, inputs: &ModelInputs) -> Result<ModelOutput> {
        // Return noise prediction
        let noise = Tensor::randn_like(&inputs.latents)?;
        Ok(ModelOutput {
            sample: noise,
            additional: HashMap::new(),
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
        match self.architecture {
            ModelArchitecture::SD15 => 1_000_000_000,      // ~1GB
            ModelArchitecture::SDXL => 6_000_000_000,      // ~6GB
            ModelArchitecture::SD3 => 8_000_000_000,       // ~8GB
            ModelArchitecture::SD35 => 8_000_000_000,      // ~8GB
            ModelArchitecture::Flux => 12_000_000_000,     // ~12GB
            _ => 1_000_000_000,
        }
    }
}