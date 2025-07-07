use anyhow::Result;
use candle_core::{Device, DType, Tensor};
use candle_nn::{VarBuilder, VarMap};

fn main() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let dtype = DType::BF16;
    
    println!("Testing VarMap behavior...");
    
    // Create a simple VarMap
    let mut var_map = VarMap::new();
    
    // Try adding tensors with different patterns
    let tensor1 = Tensor::zeros((100, 100), dtype, &device)?;
    let tensor2 = Tensor::zeros((200, 200), dtype, &device)?;
    
    println!("Adding first tensor...");
    var_map.set_one("layer.weight", tensor1.clone())?;
    println!("Success!");
    
    println!("Adding hierarchical tensor...");
    var_map.set_one("blocks.0.layer.weight", tensor2.clone())?;
    println!("Success!");
    
    // Try adding with intermediate paths
    println!("Adding with sub-paths...");
    var_map.set_one("blocks.0.weight", tensor1.clone())?;
    var_map.set_one("weight", tensor1.clone())?;
    println!("Success!");
    
    // Create VarBuilder and test access
    let vb = VarBuilder::from_varmap(&var_map, dtype, &device);
    
    println!("\nTesting VarBuilder access:");
    
    // Direct access
    match vb.get((100, 100), "layer.weight") {
        Ok(_) => println!("✓ Direct access: layer.weight"),
        Err(e) => println!("✗ Direct access failed: {}", e),
    }
    
    // Hierarchical access
    match vb.pp("blocks").pp("0").get((200, 200), "layer.weight") {
        Ok(_) => println!("✓ Hierarchical access: blocks.0.layer.weight"),
        Err(e) => println!("✗ Hierarchical access failed: {}", e),
    }
    
    // Test what happens when we create a VarBuilder with pp
    let vb_blocks = vb.pp("blocks.0");
    match vb_blocks.get((200, 200), "layer.weight") {
        Ok(_) => println!("✓ Pre-prefixed access: layer.weight via blocks.0"),
        Err(e) => println!("✗ Pre-prefixed access failed: {}", e),
    }
    
    Ok(())
}