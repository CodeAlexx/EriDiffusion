//! Standalone image generation demo
//! This example uses only the successfully built crates (core, models, networks)

use eridiffusion_core::{Device, ModelArchitecture, Result, Error, ModelInputs, ModelOutput, DiffusionModel, ModelMetadata};
use eridiffusion_networks::{LoKr, LoKrConfig, NetworkAdapter};
use candle_core::{Tensor, DType};
use candle_nn::VarBuilder;
use std::collections::HashMap;
use std::time::Instant;

fn main() -> Result<()> {
    println!("🎨 AI-Toolkit Standalone Generation Demo");
    println!("=======================================\n");
    
    // Setup device
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    let candle_device = match &device {
        Device::Cpu => candle_core::Device::Cpu,
        Device::Cuda(id) => match candle_core::Device::new_cuda(*id) {
            Ok(d) => d,
            Err(_) => candle_core::Device::Cpu,
        },
            Ok(d) => d,
            Err(_) => candle_core::Device::Cpu,
        },
    };
    
    println!("Device: {:?}", device);
    
    // Test all supported models
    let models = vec![
        ("SD 1.5", ModelArchitecture::SD15, 512, 512),
        ("SDXL", ModelArchitecture::SDXL, 1024, 1024),
        ("SD 3", ModelArchitecture::SD3, 1024, 1024),
        ("SD 3.5", ModelArchitecture::SD35, 1024, 1024),
        ("Flux", ModelArchitecture::Flux, 1024, 1024),
        ("Flux-Schnell", ModelArchitecture::FluxSchnell, 1024, 1024),
        ("Flux-Dev", ModelArchitecture::FluxDev, 1024, 1024),
        ("PixArt-α", ModelArchitecture::PixArt, 1024, 1024),
        ("PixArt-Σ", ModelArchitecture::PixArtSigma, 1024, 1024),
        ("AuraFlow", ModelArchitecture::AuraFlow, 1024, 1024),
        ("HiDream", ModelArchitecture::HiDream, 768, 768),
        ("KonText", ModelArchitecture::KonText, 1024, 1024),
        ("OmniGen 2", ModelArchitecture::OmniGen2, 1024, 1024),
        ("Flex 1", ModelArchitecture::Flex1, 768, 768),
        ("Flex 2", ModelArchitecture::Flex2, 1024, 1024),
        ("Chroma", ModelArchitecture::Chroma, 768, 768),
        ("Lumina", ModelArchitecture::Lumina, 1024, 1024),
        ("Wan 2.1", ModelArchitecture::Wan21, 1024, 576),
        ("LTX", ModelArchitecture::LTX, 768, 512),
        ("Hunyuan Video", ModelArchitecture::HunyuanVideo, 1280, 720),
    ];
    
    println!("\n📊 Testing {} model architectures:\n", models.len());
    
    for (name, arch, width, height) in models {
        print!("Testing {:<15} ", format!("{}:", name));
        
        match test_model_generation(arch, width, height, &device, &candle_device) {
            Ok(duration) => {
                println!("✅ Generated {}x{} in {:.3}s", width, height, duration);
            }
            Err(e) => {
                println!("❌ Error: {}", e);
            }
        }
    }
    
    // Test LoKr adapter
    println!("\n🔌 Testing LoKr Adapter Integration:");
    test_lokr_adapter(&device)?;
    
    println!("\n✨ Demo complete!");
    Ok(())
}

fn test_model_generation(
    architecture: ModelArchitecture,
    width: usize,
    height: usize,
    device: &Device,
    candle_device: &candle_core::Device,
) -> Result<f32> {
    let start = Instant::now();
    
    // Create dummy model
    let model = DummyModel::new(architecture, device.clone());
    
    // Prepare inputs based on architecture
    let batch_size = 1;
    let latent_channels = match architecture {
        ModelArchitecture::SD15 | ModelArchitecture::SDXL => 4,
        ModelArchitecture::SD3 | ModelArchitecture::SD35 => 16,
        ModelArchitecture::Flux | ModelArchitecture::FluxSchnell | ModelArchitecture::FluxDev => 16,
        ModelArchitecture::Wan21 | ModelArchitecture::LTX | ModelArchitecture::HunyuanVideo => 16,
        _ => 4,
    };
    
    // For video models, add temporal dimension
    let is_video = matches!(
        architecture,
        ModelArchitecture::Wan21 | ModelArchitecture::LTX | ModelArchitecture::HunyuanVideo
    );
    
    let latents = if is_video {
        // Video: [B, C, T, H, W]
        let frames = 8;
        Tensor::randn(
            0.0f32,
            1.0,
            &[batch_size, latent_channels, frames, height / 8, width / 8],
            candle_device,
        )?
    } else {
        // Image: [B, C, H, W]
        Tensor::randn(
            0.0f32,
            1.0,
            &[batch_size, latent_channels, height / 8, width / 8],
            candle_device,
        )?
    };
    
    // Text embeddings dimension
    let text_dim = match architecture {
        ModelArchitecture::SD15 => 768,
        ModelArchitecture::SDXL => 2048,
        ModelArchitecture::SD3 | ModelArchitecture::SD35 => 2048,
        ModelArchitecture::Flux | ModelArchitecture::FluxSchnell | ModelArchitecture::FluxDev => 4096,
        ModelArchitecture::KonText => 4096,
        ModelArchitecture::HunyuanVideo => 4096,
        _ => 1024,
    };
    
    let timestep = Tensor::new(&[500.0f32], candle_device)?;
    let encoder_hidden_states = Tensor::randn(
        0.0f32,
        0.1,
        &[batch_size, 77, text_dim],
        candle_device,
    )?;
    
    // Additional inputs for specific models
    let mut additional = HashMap::new();
    
    match architecture {
        ModelArchitecture::SD3 | ModelArchitecture::SD35 => {
            let pooled = Tensor::randn(0.0f32, 0.1, &[batch_size, 2048], candle_device)?;
            additional.insert("pooled_projections".to_string(), pooled);
        }
        ModelArchitecture::SDXL => {
            let time_ids = Tensor::zeros(&[batch_size, 6], DType::F32, candle_device)?;
            let text_embeds = Tensor::randn(0.0f32, 0.1, &[batch_size, 1280], candle_device)?;
            additional.insert("time_ids".to_string(), time_ids);
            additional.insert("text_embeds".to_string(), text_embeds);
        }
        _ => {}
    }
    
    let inputs = ModelInputs {
        latents,
        timestep,
        encoder_hidden_states,
        additional,
    };
    
    // Forward pass
    let _output = model.forward(&inputs)?;
    
    Ok(start.elapsed().as_secs_f32())
}

fn test_lokr_adapter(device: &Device) -> Result<()> {
    println!("\nCreating LoKr adapter for SD3.5...");
    
    let config = LoKrConfig {
        rank: 16,
        alpha: 16.0,
        target_modules: vec![
            "to_q".to_string(),
            "to_k".to_string(),
            "to_v".to_string(),
            "proj".to_string(),
        ],
        use_scalar: true,
        decompose_factor: 2,
        decompose_method: eridiffusion_networks::DecomposeMethod::Tucker,
        ..Default::default()
    };
    
    let mut lokr = LoKr::new(config, ModelArchitecture::SD35, device.clone())?;
    lokr.initialize_weights()?;
    
    let param_count = lokr.count_parameters();
    println!("✅ Created LoKr with {} parameters ({:.2}M)", 
        param_count, param_count as f32 / 1_000_000.0);
    
    // Test applying to model
    let dummy_model = DummyModel::new(ModelArchitecture::SD35, device.clone());
    let _adapted_model = lokr.apply(&dummy_model)?;
    println!("✅ Successfully applied LoKr to model");
    
    Ok(())
}

// Minimal dummy model implementation
struct DummyModel {
    architecture: ModelArchitecture,
    device: Device,
}

impl DummyModel {
    fn new(architecture: ModelArchitecture, device: Device) -> Self {
        Self { architecture, device }
    }
}

impl DiffusionModel for DummyModel {
    fn architecture(&self) -> ModelArchitecture {
        self.architecture
    }
    
    fn metadata(&self) -> &ModelMetadata {
        use once_cell::sync::Lazy;
        static METADATA: Lazy<ModelMetadata> = Lazy::new(|| ModelMetadata {
            name: "dummy".to_string(),
            architecture: ModelArchitecture::SD15,
            version: "1.0".to_string(),
            author: None,
            description: None,
            license: None,
            tags: vec![],
            created_at: chrono::Utc::now(),
            config: HashMap::new(),
        });
        &METADATA
    }
    
    async fn load_pretrained(&mut self, _path: &std::path::Path) -> Result<()> {
        Ok(())
    }
    
    async fn save_pretrained(&self, _path: &std::path::Path) -> Result<()> {
        Ok(())
    }
    
    fn forward(&self, inputs: &ModelInputs) -> Result<ModelOutput> {
        let noise = match self.architecture {
            // Video models need to match input shape
            ModelArchitecture::Wan21 | ModelArchitecture::LTX | ModelArchitecture::HunyuanVideo => {
                Tensor::randn_like(&inputs.latents)?
            }
            // Image models
            _ => Tensor::randn_like(&inputs.latents)?
        };
        
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
            ModelArchitecture::SD15 => 1_000_000_000,
            ModelArchitecture::SDXL => 6_000_000_000,
            ModelArchitecture::SD3 | ModelArchitecture::SD35 => 8_000_000_000,
            ModelArchitecture::Flux => 12_000_000_000,
            ModelArchitecture::FluxSchnell => 3_000_000_000,
            ModelArchitecture::HunyuanVideo => 16_000_000_000,
            _ => 2_000_000_000,
        }
    }
}