use candle_core::{Device, DType};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "/home/alex/SwarmUI/Models/diffusion_models/flux1-dev.safetensors";
    
    println!("Inspecting Flux model tensors...");
    
    // Use metadata function to get tensor names without loading
    let metadata = candle_core::safetensors::load_metadata(path)?;
    let mut tensor_names: Vec<_> = metadata.tensors.keys().cloned().collect();
    tensor_names.sort();
    
    println!("\nTotal tensors: {}", tensor_names.len());
    
    println!("\nMLP tensors (first 30):");
    for name in tensor_names.iter().filter(|n| n.contains("mlp")).take(30) {
        println!("  {}", name);
    }
    
    println!("\nDouble block 0 tensors:");
    for name in tensor_names.iter().filter(|n| n.starts_with("double_blocks.0.")) {
        println!("  {}", name);
    }
    
    // Check specific tensor existence
    let check_tensor = "double_blocks.0.img_mlp.w1.weight";
    println!("\nChecking for tensor: {}", check_tensor);
    println!("Exists: {}", tensor_names.contains(&check_tensor.to_string()));
    
    Ok(())
}