//! Integration tests for model-specific training pipelines

use eridiffusion_core::{Device, ModelArchitecture, FluxVariant, Result};
use eridiffusion_training::{
    pipelines::{PipelineFactory, PipelineConfig, TrainingPipeline, PreparedBatch, PromptEmbeds},
    TrainingConfig,
};
use eridiffusion_data::DataLoaderBatch;
use candle_core::{Tensor, DType};

#[test]
fn test_pipeline_factory() -> Result<()> {
    let config = PipelineConfig::default();
    
    // Test creating pipelines for all architectures
    let architectures = vec![
        ModelArchitecture::SD15,
        ModelArchitecture::SDXL,
        ModelArchitecture::SD3,
        ModelArchitecture::SD35,
        ModelArchitecture::Flux(FluxVariant::Dev),
        ModelArchitecture::PixArt,
        ModelArchitecture::AuraFlow,
    ];
    
    for arch in architectures {
        let pipeline = PipelineFactory::create(arch.clone(), config.clone())?;
        assert_eq!(pipeline.architecture(), arch);
    }
    
    Ok(())
}

#[test]
fn test_sd15_pipeline() -> Result<()> {
    let device = Device::Cpu;
    let config = PipelineConfig {
        device: device.clone(),
        dtype: DType::F32,
        ..Default::default()
    };
    
    let pipeline = PipelineFactory::create(ModelArchitecture::SD15, config)?;
    
    // Test batch preparation
    let batch = create_test_batch(&device, 4, 512, 512)?;
    let prepared = pipeline.prepare_batch(&batch)?;
    assert_eq!(prepared.latents.dims(), &[4, 4, 64, 64]); // VAE downscale 8x
    
    // Test prompt encoding
    let prompts = vec!["a cat".to_string(); 4];
    let mock_model = MockModel::new(&device);
    let embeds = pipeline.encode_prompts(&prompts, &mock_model)?;
    assert_eq!(embeds.text_embeds.dims(), &[4, 77, 768]);
    
    // Test noise addition
    let noise = Tensor::randn_like(&prepared.latents)?;
    let timesteps = Tensor::new(&[500i64, 500, 500, 500], &device)?;
    let noisy = pipeline.add_noise(&prepared.latents, &noise, &timesteps)?;
    assert_eq!(noisy.dims(), prepared.latents.dims());
    
    Ok(())
}

#[test]
fn test_sdxl_pipeline() -> Result<()> {
    let device = Device::Cpu;
    let config = PipelineConfig {
        device: device.clone(),
        dtype: DType::F32,
        ..Default::default()
    };
    
    let pipeline = PipelineFactory::create(ModelArchitecture::SDXL, config)?;
    
    // Test batch preparation with SDXL metadata
    let batch = create_test_batch(&device, 2, 1024, 1024)?;
    let prepared = pipeline.prepare_batch(&batch)?;
    
    // SDXL should have original sizes and crop coords
    assert!(prepared.original_sizes.is_some());
    assert!(prepared.crop_coords.is_some());
    assert!(prepared.aesthetic_scores.is_some());
    
    // Test dual encoder prompts
    let prompts = vec!["a beautiful landscape".to_string(); 2];
    let mock_model = MockModel::new(&device);
    let embeds = pipeline.encode_prompts(&prompts, &mock_model)?;
    
    // SDXL concatenates CLIP-L and CLIP-G
    assert_eq!(embeds.text_embeds.dims(), &[2, 77, 2048]);
    assert!(embeds.pooled_embeds.is_some());
    assert!(embeds.text_embeds_2.is_some());
    
    Ok(())
}

#[test]
fn test_sd3_flow_matching() -> Result<()> {
    let device = Device::Cpu;
    let config = PipelineConfig {
        device: device.clone(),
        dtype: DType::F32,
        flow_matching: true,
        ..Default::default()
    };
    
    let pipeline = PipelineFactory::create(ModelArchitecture::SD3, config)?;
    
    // Test flow matching noise
    let batch = create_test_batch(&device, 2, 1024, 1024)?;
    let prepared = pipeline.prepare_batch(&batch)?;
    let noise = Tensor::randn_like(&prepared.latents)?;
    let timesteps = Tensor::new(&[250i64, 750], &device)?;
    
    let noisy = pipeline.add_noise(&prepared.latents, &noise, &timesteps)?;
    
    // Flow matching should interpolate between data and noise
    assert_eq!(noisy.dims(), prepared.latents.dims());
    
    // Test triple encoder
    let prompts = vec!["test".to_string(); 2];
    let mock_model = MockModel::new(&device);
    let embeds = pipeline.encode_prompts(&prompts, &mock_model)?;
    
    // SD3 concatenates CLIP-L, CLIP-G, and T5
    assert_eq!(embeds.text_embeds.dims(), &[2, 77, 6144]);
    assert!(embeds.text_embeds_3.is_some()); // T5 embeds
    
    Ok(())
}

#[test]
fn test_flux_pipeline() -> Result<()> {
    let device = Device::Cpu;
    let config = PipelineConfig {
        device: device.clone(),
        dtype: DType::F32,
        flow_matching: true,
        ..Default::default()
    };
    
    let pipeline = PipelineFactory::create(
        ModelArchitecture::Flux(FluxVariant::Dev),
        config
    )?;
    
    // Test Flux-specific features
    let batch = create_test_batch(&device, 1, 1024, 1024)?;
    let prepared = pipeline.prepare_batch(&batch)?;
    
    // Test T5-only encoding
    let prompts = vec!["flux test prompt".to_string()];
    let mock_model = MockModel::new(&device);
    let embeds = pipeline.encode_prompts(&prompts, &mock_model)?;
    
    // Flux uses longer T5 sequences
    assert_eq!(embeds.text_embeds.dims(), &[1, 256, 4096]);
    assert!(embeds.pooled_embeds.is_some());
    
    Ok(())
}

#[test]
fn test_loss_computation() -> Result<()> {
    let device = Device::Cpu;
    
    // Test MSE loss
    let config = PipelineConfig {
        device: device.clone(),
        dtype: DType::F32,
        loss_type: "mse".to_string(),
        ..Default::default()
    };
    
    let pipeline = PipelineFactory::create(ModelArchitecture::SD15, config)?;
    let batch = create_test_batch(&device, 2, 512, 512)?;
    let prepared = pipeline.prepare_batch(&batch)?;
    
    let noise = Tensor::randn_like(&prepared.latents)?;
    let timesteps = Tensor::new(&[500i64, 500], &device)?;
    let noisy = pipeline.add_noise(&prepared.latents, &noise, &timesteps)?;
    
    let prompts = vec!["test".to_string(); 2];
    let mock_model = MockModel::new(&device);
    let embeds = pipeline.encode_prompts(&prompts, &mock_model)?;
    
    let loss = pipeline.compute_loss(
        &mock_model,
        &noisy,
        &noise,
        &timesteps,
        &embeds,
        &prepared,
    )?;
    
    assert_eq!(loss.dims(), &[]); // Scalar loss
    
    Ok(())
}

#[test]
fn test_snr_weighting() -> Result<()> {
    let device = Device::Cpu;
    let config = PipelineConfig {
        device: device.clone(),
        dtype: DType::F32,
        snr_gamma: Some(5.0),
        ..Default::default()
    };
    
    let pipeline = PipelineFactory::create(ModelArchitecture::SD15, config)?;
    
    let loss = Tensor::new(1.0f32, &device)?;
    let timesteps = Tensor::new(&[100i64, 500, 900], &device)?;
    
    let weighted = pipeline.apply_snr_weight(&loss, &timesteps)?;
    
    // Loss should be weighted differently for different timesteps
    assert_eq!(weighted.dims(), loss.dims());
    
    Ok(())
}

// Helper functions

fn create_test_batch(
    device: &Device,
    batch_size: usize,
    height: usize,
    width: usize,
) -> Result<DataLoaderBatch> {
    Ok(DataLoaderBatch {
        images: Tensor::randn(0f32, 1f32, &[batch_size, 3, height, width], device)?,
        captions: vec!["test caption".to_string(); batch_size],
        masks: None,
        loss_weights: vec![1.0; batch_size],
    })
}

// Mock model for testing
struct MockModel {
    device: Device,
}

impl MockModel {
    fn new(device: &Device) -> Self {
        Self {
            device: device.clone(),
        }
    }
}

impl eridiffusion_models::DiffusionModel for MockModel {
    fn forward(&self, inputs: &eridiffusion_models::ModelInputs) -> Result<eridiffusion_models::ModelOutput> {
        Ok(eridiffusion_models::ModelOutput {
            sample: inputs.latents.clone(),
            ..Default::default()
        })
    }
    
    fn architecture(&self) -> ModelArchitecture {
        ModelArchitecture::SD15
    }
    
    fn device(&self) -> &Device {
        &self.device
    }
    
    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.device = device.clone();
        Ok(())
    }
    
    async fn load_pretrained(&mut self, _path: &std::path::Path) -> Result<()> {
        Ok(())
    }
    
    async fn save_pretrained(&self, _path: &std::path::Path) -> Result<()> {
        Ok(())
    }
    
    fn state_dict(&self) -> Result<std::collections::HashMap<String, Tensor>> {
        Ok(std::collections::HashMap::new())
    }
    
    fn load_state_dict(&mut self, _state_dict: std::collections::HashMap<String, Tensor>) -> Result<()> {
        Ok(())
    }
    
    fn trainable_parameters(&self) -> Vec<&Tensor> {
        vec![]
    }
    
    fn num_parameters(&self) -> usize {
        0
    }
    
    fn memory_usage(&self) -> usize {
        0
    }
}