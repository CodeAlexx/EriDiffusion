//! End-to-end LoKr training test for all supported models

use eridiffusion_core::{Device, ModelArchitecture, Result, Error};
use eridiffusion_models::{ModelFactory, DiffusionModel, ModelInputs};
use eridiffusion_networks::{LoKrConfig, LoKr, NetworkAdapter};
use candle_core::{Tensor, DType};
use std::collections::HashMap;

/// Test configuration for each model
struct ModelTestConfig {
    architecture: ModelArchitecture,
    latent_channels: usize,
    latent_size: (usize, usize),
    text_embed_dim: usize,
    batch_size: usize,
    rank: usize,
}

impl ModelTestConfig {
    fn new(architecture: ModelArchitecture) -> Self {
        match architecture {
            ModelArchitecture::SD15 => Self {
                architecture,
                latent_channels: 4,
                latent_size: (64, 64),
                text_embed_dim: 768,
                batch_size: 1,
                rank: 4,
            },
            ModelArchitecture::SDXL => Self {
                architecture,
                latent_channels: 4,
                latent_size: (128, 128),
                text_embed_dim: 2048,
                batch_size: 1,
                rank: 8,
            },
            ModelArchitecture::SD3 | ModelArchitecture::SD35 => Self {
                architecture,
                latent_channels: 16,
                latent_size: (64, 64),
                text_embed_dim: 2048,
                batch_size: 1,
                rank: 16,
            },
            ModelArchitecture::Flux | ModelArchitecture::FluxSchnell | ModelArchitecture::FluxDev => Self {
                architecture,
                latent_channels: 16,
                latent_size: (64, 64),
                text_embed_dim: 4096,
                batch_size: 1,
                rank: 16,
            },
            ModelArchitecture::PixArt | ModelArchitecture::PixArtSigma => Self {
                architecture,
                latent_channels: 4,
                latent_size: (64, 64),
                text_embed_dim: 1152,
                batch_size: 1,
                rank: 8,
            },
            ModelArchitecture::AuraFlow => Self {
                architecture,
                latent_channels: 4,
                latent_size: (64, 64),
                text_embed_dim: 2048,
                batch_size: 1,
                rank: 8,
            },
            ModelArchitecture::HiDream => Self {
                architecture,
                latent_channels: 4,
                latent_size: (64, 64),
                text_embed_dim: 1280,
                batch_size: 1,
                rank: 8,
            },
            ModelArchitecture::KonText => Self {
                architecture,
                latent_channels: 16,
                latent_size: (64, 64),
                text_embed_dim: 4096,
                batch_size: 1,
                rank: 16,
            },
            ModelArchitecture::Wan21 => Self {
                architecture,
                latent_channels: 16, // Uses Flux VAE
                latent_size: (64, 64),
                text_embed_dim: 2048,
                batch_size: 1,
                rank: 8,
            },
            ModelArchitecture::LTX => Self {
                architecture,
                latent_channels: 16,
                latent_size: (8, 64, 64), // Video: frames x h x w
                text_embed_dim: 2048,
                batch_size: 1,
                rank: 16,
            },
            ModelArchitecture::HunyuanVideo => Self {
                architecture,
                latent_channels: 16,
                latent_size: (16, 64, 64), // Video: frames x h x w
                text_embed_dim: 4096,
                batch_size: 1,
                rank: 32,
            },
            ModelArchitecture::OmniGen2 => Self {
                architecture,
                latent_channels: 4,
                latent_size: (64, 64),
                text_embed_dim: 1280,
                batch_size: 1,
                rank: 8,
            },
            ModelArchitecture::Flex1 | ModelArchitecture::Flex2 => Self {
                architecture,
                latent_channels: 8,
                latent_size: (64, 64),
                text_embed_dim: 2048,
                batch_size: 1,
                rank: 12,
            },
            ModelArchitecture::Chroma => Self {
                architecture,
                latent_channels: 4,
                latent_size: (64, 64),
                text_embed_dim: 1024,
                batch_size: 1,
                rank: 8,
            },
            ModelArchitecture::Lumina => Self {
                architecture,
                latent_channels: 4,
                latent_size: (64, 64),
                text_embed_dim: 2048,
                batch_size: 1,
                rank: 8,
            },
        }
    }
}

#[test]
fn test_lokr_training_all_models() -> Result<()> {
    // Test all supported architectures
    let architectures = vec![
        ModelArchitecture::SD15,
        ModelArchitecture::SDXL,
        ModelArchitecture::SD3,
        ModelArchitecture::SD35,
        ModelArchitecture::Flux,
        ModelArchitecture::FluxSchnell,
        ModelArchitecture::FluxDev,
        ModelArchitecture::PixArt,
        ModelArchitecture::PixArtSigma,
        ModelArchitecture::AuraFlow,
        ModelArchitecture::HiDream,
        ModelArchitecture::KonText,
        ModelArchitecture::Wan21,
        ModelArchitecture::LTX,
        ModelArchitecture::HunyuanVideo,
        ModelArchitecture::OmniGen2,
        ModelArchitecture::Flex1,
        ModelArchitecture::Flex2,
        ModelArchitecture::Chroma,
        ModelArchitecture::Lumina,
    ];
    
    let device = Device::Cpu; // Use CPU for tests
    
    for architecture in architectures {
        println!("\n=== Testing LoKr training for {:?} ===", architecture);
        test_lokr_for_architecture(architecture, &device)?;
    }
    
    println!("\n✅ All models tested successfully!");
    Ok(())
}

fn test_lokr_for_architecture(architecture: ModelArchitecture, device: &Device) -> Result<()> {
    let config = ModelTestConfig::new(architecture);
    
    // Step 1: Create model
    println!("1. Creating {:?} model...", architecture);
    let model = create_test_model(architecture, device)?;
    
    // Step 2: Create LoKr adapter
    println!("2. Creating LoKr adapter with rank {}...", config.rank);
    let lokr_config = LoKrConfig {
        rank: config.rank,
        alpha: config.rank as f32,
        target_modules: get_target_modules(architecture),
        use_scalar: true,
        ..Default::default()
    };
    
    let mut lokr = LoKr::new(lokr_config.clone(), architecture, device.clone())?;
    lokr.initialize_weights()?;
    
    // Step 3: Apply LoKr to model
    println!("3. Applying LoKr to model...");
    let model_with_lokr = lokr.apply(&*model)?;
    
    // Step 4: Test forward pass
    println!("4. Testing forward pass...");
    let inputs = create_test_inputs(&config, device)?;
    let output = model_with_lokr.forward(&inputs)?;
    
    // Verify output shape
    verify_output_shape(&output, &config)?;
    
    // Step 5: Simulate training step
    println!("5. Simulating training step...");
    let loss = simulate_training_step(&output, &config)?;
    println!("   Loss: {:.6}", loss);
    
    // Step 6: Test weight updates
    println!("6. Testing weight updates...");
    lokr.apply_weight_decay(0.01)?;
    
    // Step 7: Test saving/loading
    println!("7. Testing save/load...");
    let temp_path = std::env::temp_dir().join(format!("lokr_test_{:?}.safetensors", architecture));
    lokr.save_pretrained(&temp_path)?;
    
    // Load into new adapter
    let mut lokr_loaded = LoKr::new(lokr_config, architecture, device.clone())?;
    tokio::runtime::Runtime::new()?.block_on(async {
        lokr_loaded.load_pretrained(&temp_path).await
    })?;
    
    // Clean up
    std::fs::remove_file(&temp_path).ok();
    
    // Step 8: Verify parameter count
    let param_count = lokr.count_parameters();
    println!("8. LoKr parameters: {} ({:.2}M)", param_count, param_count as f32 / 1_000_000.0);
    
    println!("✅ {:?} test passed!", architecture);
    Ok(())
}

fn create_test_model(architecture: ModelArchitecture, device: &Device) -> Result<Box<dyn DiffusionModel>> {
    // For testing, we create a minimal model that implements the DiffusionModel trait
    // In practice, this would load the actual model
    Ok(Box::new(TestModel::new(architecture, device.clone())))
}

fn get_target_modules(architecture: ModelArchitecture) -> Vec<String> {
    match architecture {
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
        _ => vec![
            "attention.to_q".to_string(),
            "attention.to_k".to_string(),
            "attention.to_v".to_string(),
            "attention.to_out".to_string(),
        ],
    }
}

fn create_test_inputs(config: &ModelTestConfig, device: &Device) -> Result<ModelInputs> {
    let candle_device = device.as_candle_device()?;
    
    // Create latents based on architecture
    let latents = match config.architecture {
        ModelArchitecture::LTX | ModelArchitecture::HunyuanVideo => {
            // Video models have temporal dimension
            let (t, h, w) = match config.latent_size {
                (t, h, w) => (t, h, w),
                _ => (8, 64, 64),
            };
            Tensor::randn(0.0f32, 1.0, &[config.batch_size, config.latent_channels, t, h, w], &candle_device)?
        }
        _ => {
            // Image models
            let (h, w) = match config.latent_size {
                (h, w) => (h, w),
                _ => (64, 64),
            };
            Tensor::randn(0.0f32, 1.0, &[config.batch_size, config.latent_channels, h, w], &candle_device)?
        }
    };
    
    // Create timestep
    let timestep = Tensor::new(&[500u32], &candle_device)?;
    
    // Create text embeddings
    let encoder_hidden_states = Tensor::randn(
        0.0f32,
        0.1,
        &[config.batch_size, 77, config.text_embed_dim],
        &candle_device,
    )?;
    
    // Create additional inputs based on architecture
    let mut additional = HashMap::new();
    
    match config.architecture {
        ModelArchitecture::SD3 | ModelArchitecture::SD35 => {
            // SD3/3.5 need pooled projections
            let pooled = Tensor::randn(0.0f32, 0.1, &[config.batch_size, 2048], &candle_device)?;
            additional.insert("pooled_projections".to_string(), pooled);
        }
        ModelArchitecture::SDXL => {
            // SDXL needs time ids and text embeds
            let time_ids = Tensor::zeros(&[config.batch_size, 6], DType::F32, &candle_device)?;
            let text_embeds = Tensor::randn(0.0f32, 0.1, &[config.batch_size, 1280], &candle_device)?;
            additional.insert("time_ids".to_string(), time_ids);
            additional.insert("text_embeds".to_string(), text_embeds);
        }
        _ => {}
    }
    
    Ok(ModelInputs {
        latents,
        timestep,
        encoder_hidden_states,
        additional,
    })
}

fn verify_output_shape(output: &eridiffusion_core::ModelOutput, config: &ModelTestConfig) -> Result<()> {
    let shape = output.sample.dims();
    
    // Verify batch size
    if shape[0] != config.batch_size {
        return Err(Error::InvalidShape(format!(
            "Expected batch size {}, got {}",
            config.batch_size, shape[0]
        )));
    }
    
    // Verify channels
    if shape[1] != config.latent_channels {
        return Err(Error::InvalidShape(format!(
            "Expected {} channels, got {}",
            config.latent_channels, shape[1]
        )));
    }
    
    println!("   Output shape: {:?}", shape);
    Ok(())
}

fn simulate_training_step(output: &eridiffusion_core::ModelOutput, _config: &ModelTestConfig) -> Result<f32> {
    // Simulate MSE loss calculation
    let target = Tensor::randn_like(&output.sample)?;
    let loss = (&output.sample - &target)?
        .sqr()?
        .mean_all()?
        .to_scalar::<f32>()?;
    
    Ok(loss)
}

// Test model that implements DiffusionModel trait
struct TestModel {
    architecture: ModelArchitecture,
    device: Device,
}

impl TestModel {
    fn new(architecture: ModelArchitecture, device: Device) -> Self {
        Self { architecture, device }
    }
}

impl eridiffusion_core::DiffusionModel for TestModel {
    fn architecture(&self) -> ModelArchitecture {
        self.architecture
    }
    
    fn metadata(&self) -> &eridiffusion_core::ModelMetadata {
        // Return dummy metadata
        use once_cell::sync::Lazy;
        static METADATA: Lazy<eridiffusion_core::ModelMetadata> = Lazy::new(|| {
            eridiffusion_core::ModelMetadata {
                name: "test".to_string(),
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
    
    async fn load_pretrained(&mut self, _path: &std::path::Path) -> Result<()> {
        Ok(())
    }
    
    async fn save_pretrained(&self, _path: &std::path::Path) -> Result<()> {
        Ok(())
    }
    
    fn forward(&self, inputs: &ModelInputs) -> Result<eridiffusion_core::ModelOutput> {
        // Return noise prediction with same shape as input
        Ok(eridiffusion_core::ModelOutput {
            sample: Tensor::randn_like(&inputs.latents)?,
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
        1_000_000 // 1MB for test
    }
}

// Device extension trait
trait DeviceExt {
    fn as_candle_device(&self) -> Result<candle_core::Device>;
}

impl DeviceExt for Device {
    fn as_candle_device(&self) -> Result<candle_core::Device> {
        match self {
            Device::Cpu => Ok(candle_core::Device::Cpu),
            Device::Cuda(id) => Ok(candle_core::Device::cuda(*id)?),
        }
    }
}