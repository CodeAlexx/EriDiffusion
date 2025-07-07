//! Fixed Flux loader that handles memory properly

use anyhow::Result;
use candle_core::{Device, DType};
use eridiffusion::models::flux_custom::{FluxConfig, FluxModelWithLoRA};
use eridiffusion::models::flux_custom::lora::LoRAConfig;
use std::path::Path;

fn main() -> Result<()> {
    println!("Fixed Flux LoRA Loader Test");
    println!("===========================\n");
    
    // Setup
    let device = Device::cuda_if_available(0)?;
    let model_path = Path::new("/home/alex/SwarmUI/Models/diffusion_models/flux.1-lite-8B.safetensors");
    
    // LoRA configuration
    let lora_config = LoRAConfig {
        rank: 16,
        alpha: 16.0,
        dropout: Some(0.0),
        target_modules: vec!["attn".to_string(), "mlp".to_string()],
        module_filters: vec![],
        init_scale: 0.01,
    };
    
    println!("Model path: {:?}", model_path);
    println!("Device: {:?}", device);
    
    // Method 1: Try using the quantized loader for memory efficiency
    println!("\n--- Method 1: Using Quantized Loader ---");
    
    use eridiffusion::trainers::flux_quantized_loader::QuantizedFluxLoader;
    
    let quantized_loader = QuantizedFluxLoader::new_with_dtype(
        device.clone(),
        DType::BF16,  // Use BF16 for training
    )?;
    
    println!("Loading model with INT8 quantization...");
    println!("This will reduce 15.21GB model to ~8GB");
    
    let flux_config = FluxConfig::default();
    
    match quantized_loader.load_quantized_model(model_path, &flux_config) {
        Ok(quantized_weights) => {
            println!("✅ Successfully loaded with quantization!");
            
            // Create model from quantized weights
            println!("\nCreating LoRA model from quantized weights...");
            let mut model = quantized_loader.create_model_with_lora(
                &flux_config,
                &lora_config,
                &quantized_weights,
            )?;
            
            println!("✅ LoRA adapters added!");
            
            // Test forward pass
            use candle_core::Tensor;
            let batch_size = 1;
            let seq_len = 1024;
            
            let img = Tensor::randn(0f32, 1.0, (batch_size, seq_len, 64), &device)?;
            let img_ids = Tensor::zeros((batch_size, 32, 32, 3), DType::F32, &device)?;
            let txt = Tensor::randn(0f32, 1.0, (batch_size, 256, 4096), &device)?;
            let txt_ids = Tensor::zeros((batch_size, 256, 3), DType::F32, &device)?;
            let timesteps = Tensor::new(&[500.0f32], &device)?;
            let y = Tensor::randn(0f32, 1.0, (batch_size, 768), &device)?;
            
            println!("\nRunning forward pass...");
            let output = model.forward(&img, &img_ids, &txt, &txt_ids, &timesteps, &y, None)?;
            println!("✅ Forward pass successful! Output shape: {:?}", output.shape());
        }
        Err(e) => {
            println!("❌ Quantized loading failed: {}", e);
            println!("\n--- Method 2: Manual Memory-Efficient Loading ---");
            
            // Try manual approach with CPU staging
            println!("Loading weights to CPU first, then moving to GPU in chunks...");
            
            use candle_nn::{VarBuilder, VarMap};
            
            // Create VarMap on CPU first
            let cpu_var_map = VarMap::new();
            let cpu_vb = VarBuilder::from_varmap(&cpu_var_map, DType::F16, &Device::Cpu);
            
            // Load a minimal set of weights just to test
            println!("Creating model with minimal weights...");
            
            // Create model with zeros (minimal memory)
            let var_map = VarMap::new();
            let vb = VarBuilder::from_varmap(&var_map, DType::F16, &device);
            
            // Initialize with zeros to avoid loading from file
            use eridiffusion::trainers::flux_init_weights::initialize_flux_weights_minimal;
            initialize_flux_weights_minimal(&vb, &flux_config)?;
            
            let mut model = FluxModelWithLoRA::new(&flux_config, vb)?;
            
            // Add LoRA (these are the only trainable parts)
            model.add_lora_to_all(&lora_config, &device, DType::F16)?;
            
            println!("✅ Model created with LoRA adapters!");
            println!("Note: Base weights are zeros - only for testing structure");
        }
    }
    
    // Check final memory usage
    if let Ok(output) = std::process::Command::new("nvidia-smi")
        .args(&["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()
    {
        let used_memory = String::from_utf8_lossy(&output.stdout);
        println!("\nFinal GPU memory used: {} MB", used_memory.trim());
    }
    
    Ok(())
}