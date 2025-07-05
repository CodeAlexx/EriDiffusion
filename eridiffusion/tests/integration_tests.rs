//! Integration tests for AI-Toolkit

use eridiffusion_core::{Device, ModelArchitecture, initialize as init_core};
use eridiffusion_models::{ModelLoader, initialize as init_models};
use eridiffusion_training::{Trainer, TrainerConfig, initialize as init_training};
use eridiffusion_networks::{LoRAConfig, LoRAAdapter};
use eridiffusion_data::{ImageFolderDataset, DataLoader, DataLoaderConfig};
use std::path::PathBuf;
use std::sync::Arc;
use tokio;

#[tokio::test]
async fn test_model_loading_and_detection() {
    // Initialize modules
    init_core().unwrap();
    init_models().unwrap();
    
    let loader = ModelLoader::new();
    
    // Test architecture detection
    // Note: In real tests, we'd use a test model file
    let test_config = eridiffusion_models::loader::ModelLoadConfig {
        device: Device::Cpu,
        precision: eridiffusion_core::dtype::Precision::Float32,
        compile: false,
        use_flash_attention: false,
    };
    
    // This would work with a real model path
    // let model = loader.load("test_models/sd15", test_config).await.unwrap();
    // assert_eq!(model.architecture, ModelArchitecture::SD15);
}

#[tokio::test]
async fn test_lora_adapter_creation() {
    init_core().unwrap();
    
    let config = LoRAConfig {
        rank: 16,
        alpha: 32.0,
        target_modules: vec!["to_q".to_string(), "to_v".to_string()],
        dropout: 0.1,
        use_dora: false,
        use_rslora: false,
        rank_pattern: vec![],
        alpha_pattern: vec![],
    };
    
    let adapter = LoRAAdapter::new(
        config,
        ModelArchitecture::SD15,
        Device::Cpu,
    ).unwrap();
    
    assert_eq!(adapter.adapter_type(), "LoRA");
}

#[test]
fn test_data_pipeline() {
    use std::fs;
    use std::io::Write;
    
    // Create temporary test directory
    let temp_dir = tempfile::tempdir().unwrap();
    let dataset_path = temp_dir.path().join("images");
    fs::create_dir(&dataset_path).unwrap();
    
    // Create dummy image and caption files
    let img_path = dataset_path.join("test.jpg");
    fs::write(&img_path, b"dummy image data").unwrap();
    
    let caption_path = dataset_path.join("test.txt");
    let mut file = fs::File::create(&caption_path).unwrap();
    writeln!(file, "a test image").unwrap();
    
    // Create dataset
    let dataset = ImageFolderDataset::new(dataset_path, "txt".to_string());
    
    // Create dataloader
    let config = DataLoaderConfig {
        batch_size: 1,
        num_workers: 1,
        shuffle: false,
        pin_memory: false,
        drop_last: false,
        prefetch_factor: None,
    };
    
    let dataloader = DataLoader::new(Arc::new(dataset), config).unwrap();
    
    // Test that we can create it without errors
    assert_eq!(dataloader.batch_size(), 1);
}

#[tokio::test]
async fn test_training_loop_initialization() {
    init_core().unwrap();
    init_training().unwrap();
    
    // Create dummy model
    let model = DummyModel::new(Device::Cpu);
    
    // Create trainer config
    let config = TrainerConfig {
        device: Device::Cpu,
        precision: eridiffusion_core::dtype::Precision::Float32,
        gradient_accumulation_steps: 1,
        gradient_checkpointing: false,
        max_grad_norm: 1.0,
        seed: 42,
        num_epochs: 1,
        save_steps: 100,
        eval_steps: 50,
        logging_steps: 10,
        output_dir: PathBuf::from("./test_output"),
        resume_from_checkpoint: None,
        gradient_clip_norm: Some(1.0),
        checkpoint_steps: 100,
    };
    
    let trainer = Trainer::new(config, Box::new(model), None);
    assert!(trainer.is_ok());
}

// Dummy model for testing
struct DummyModel {
    device: Device,
}

impl DummyModel {
    fn new(device: Device) -> Self {
        Self { device }
    }
}

impl eridiffusion_core::model::DiffusionModel for DummyModel {
    fn forward(&self, _inputs: &eridiffusion_core::model::ModelInputs) -> eridiffusion_core::Result<eridiffusion_core::model::ModelOutput> {
        use candle_core::{Tensor, DType};
        
        Ok(eridiffusion_core::model::ModelOutput {
            sample: Tensor::zeros(&[1, 4, 64, 64], DType::F32, &candle_core::Device::Cpu)?,
            additional: std::collections::HashMap::new(),
        })
    }
    
    fn architecture(&self) -> ModelArchitecture {
        ModelArchitecture::SD15
    }
    
    fn device(&self) -> &Device {
        &self.device
    }
    
    fn to_device(&mut self, device: &Device) -> eridiffusion_core::Result<()> {
        self.device = device.clone();
        Ok(())
    }
    
    async fn load_pretrained(&mut self, _path: &std::path::Path) -> eridiffusion_core::Result<()> {
        Ok(())
    }
    
    async fn save_pretrained(&self, _path: &std::path::Path) -> eridiffusion_core::Result<()> {
        Ok(())
    }
    
    fn state_dict(&self) -> eridiffusion_core::Result<std::collections::HashMap<String, candle_core::Tensor>> {
        Ok(std::collections::HashMap::new())
    }
    
    fn load_state_dict(&mut self, _state_dict: std::collections::HashMap<String, candle_core::Tensor>) -> eridiffusion_core::Result<()> {
        Ok(())
    }
    
    fn trainable_parameters(&self) -> Vec<&candle_core::Tensor> {
        vec![]
    }
    
    fn num_parameters(&self) -> usize {
        0
    }
    
    fn memory_usage(&self) -> usize {
        0
    }
}

#[test]
fn test_error_propagation() {
    use eridiffusion_core::{Error, ErrorContext};
    
    // Test error context
    let base_error = Error::Model("test error".to_string());
    let with_context: Result<(), Error> = Err(base_error).context("additional context");
    
    match with_context {
        Err(Error::Runtime(msg)) => {
            assert!(msg.contains("additional context"));
            assert!(msg.contains("test error"));
        }
        _ => panic!("Expected Runtime error with context"),
    }
}

#[test]
fn test_validation_integration() {
    use eridiffusion_core::validation::{TensorValidator, ConfigValidator};
    use candle_core::{Tensor, Device, DType};
    
    let device = Device::Cpu;
    
    // Test tensor validation pipeline
    let tensor = Tensor::randn(0f32, 1f32, &[2, 3, 4], &device).unwrap();
    
    // Validate multiple properties
    assert!(TensorValidator::validate_not_empty(&tensor, "test").is_ok());
    assert!(TensorValidator::validate_shape(&tensor, Some(3), None, "test").is_ok());
    assert!(TensorValidator::validate_finite(&tensor, "test").is_ok());
    
    // Test config validation pipeline
    assert!(ConfigValidator::validate_range(0.5f32, Some(0.0), Some(1.0), "learning_rate").is_ok());
    assert!(ConfigValidator::validate_not_empty_string("model_name", "name").is_ok());
}

#[tokio::test]
async fn test_inference_pipeline() {
    use eridiffusion_inference::{InferencePipeline, InferenceConfig};
    
    init_core().unwrap();
    
    let config = InferenceConfig {
        model_path: "test_model".to_string(),
        device: Device::Cpu,
        precision: eridiffusion_core::dtype::Precision::Float32,
        compile_model: false,
        use_flash_attention: false,
        scheduler: eridiffusion_inference::pipeline::SchedulerType::DDIM,
        guidance_scale: 7.5,
        num_inference_steps: 25,
        eta: 0.0,
        generator_seed: Some(42),
    };
    
    // This would work with a real model
    // let pipeline = InferencePipeline::new(config).await.unwrap();
    // let output = pipeline.generate("test prompt", None, None).await.unwrap();
}

#[test]
fn test_memory_pool() {
    use eridiffusion_core::memory::{MemoryPool, MemoryPoolConfig};
    use candle_core::DType;
    
    let config = MemoryPoolConfig {
        initial_size: 1024 * 1024, // 1MB
        max_size: 10 * 1024 * 1024, // 10MB
        enable_defrag: true,
        reuse_threshold: 0.8,
    };
    
    let pool = MemoryPool::new(config).unwrap();
    
    // Test allocation
    let allocation = pool.allocate(1024, DType::F32).unwrap();
    assert!(allocation.size >= 1024);
    
    // Test deallocation
    pool.deallocate(allocation).unwrap();
    
    // Test stats
    let stats = pool.get_stats();
    assert_eq!(stats.deallocations, 1);
}