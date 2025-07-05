use eridiffusion_core::*;

#[test]
fn test_device_management() {
    // Initialize the system
    initialize().unwrap();
    
    // Get device manager
    let device_manager = device::device_manager();
    
    // List devices
    let devices = device_manager.list_devices();
    assert!(!devices.is_empty());
    
    // At least CPU should be available
    let cpu_device = devices.iter().find(|d| matches!(d.device, Device::Cpu));
    assert!(cpu_device.is_some());
}

#[test]
fn test_model_architecture() {
    // Test architecture detection
    let architectures = ModelArchitecture::all();
    assert!(!architectures.is_empty());
    
    // Test properties
    assert!(ModelArchitecture::SD35.uses_flow_matching());
    assert_eq!(ModelArchitecture::SD35.latent_channels(), 16);
    assert_eq!(ModelArchitecture::SDXL.latent_channels(), 4);
}

#[test]
fn test_plugin_registry() {
    // Initialize plugin system
    plugin::initialize_plugin_system().unwrap();
    
    // Get registry
    let registry = plugin::PluginRegistry::global();
    
    // Should have search paths
    assert!(registry.list_plugins().is_empty()); // No plugins loaded yet
}

#[test]
fn test_network_module_matcher() {
    use network::ModuleMatcher;
    
    let patterns = vec![
        "*.to_q".to_string(),
        "*.to_k".to_string(),
        "transformer.blocks.*.attn.*".to_string(),
    ];
    
    let matcher = ModuleMatcher::new(patterns);
    
    assert!(matcher.matches("model.layer1.to_q"));
    assert!(matcher.matches("encoder.block.to_k"));
    assert!(matcher.matches("transformer.blocks.0.attn.qkv"));
    assert!(!matcher.matches("model.layer1.to_v")); // Not in patterns
}

#[test]
fn test_tensor_view() {
    use candle_core::{Tensor, Device as CandleDevice};
    use tensor::TensorView;
    
    // Create a test tensor
    let device = CandleDevice::Cpu;
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor = Tensor::from_vec(data, &[2, 3], &device).unwrap();
    
    // Create a view
    let view = TensorView::new(tensor);
    
    // Test slicing
    let sliced = view.slice(&[0..1, 1..3]).unwrap();
    assert_eq!(sliced.shape().dims(), &[1, 2]);
    
    // Test reshape
    let reshaped = view.reshape(&[3, 2]).unwrap();
    assert_eq!(reshaped.shape().dims(), &[3, 2]);
}

#[tokio::test]
async fn test_config_system() {
    use config::{ConfigLoader, UniversalConfig};
    use std::io::Write;
    
    // Create a test config file
    let config_content = r#"
job:
  name: "test_training"
  job_type: "training"
  priority: "normal"
  tags: ["test", "sdxl"]

model:
  architecture: "SDXL"
  pretrained_model_name_or_path: "stabilityai/stable-diffusion-xl-base-1.0"
  use_safetensors: true

network:
  network_type: "LoRA"
  rank: 16
  alpha: 16.0
  dropout: 0.0
  target_modules: ["*.to_q", "*.to_k", "*.to_v"]
  use_bias: false
  init_strategy: "normal"
"#;
    
    // Write to temp file
    let temp_dir = tempfile::tempdir().unwrap();
    let config_path = temp_dir.path().join("test_config.yaml");
    let mut file = std::fs::File::create(&config_path).unwrap();
    file.write_all(config_content.as_bytes()).unwrap();
    
    // Load config
    let mut loader = ConfigLoader::new();
    loader.add_search_path(temp_dir.path().to_path_buf());
    
    let config: UniversalConfig = loader.load("test_config").unwrap();
    
    // Verify
    assert_eq!(config.job.name, "test_training");
    assert_eq!(config.model.architecture, ModelArchitecture::SDXL);
    assert!(config.network.is_some());
    
    let network = config.network.unwrap();
    assert_eq!(network.network_type, NetworkType::LoRA);
    assert_eq!(network.rank, 16);
}