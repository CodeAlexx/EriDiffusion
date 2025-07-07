// Debug version to figure out what's going wrong
-- text to describe -- 

I've created a debug version that will help us figure out exactly what's going wrong. Run this to see:

What's actually in the checkpoint
What names your model is looking for
How VarBuilder behaves with different access patterns
A brute-force loader that just makes it work
To use it:

rust
// Debug everything
debug_everything(Path::new("path/to/flux_dev.safetensors"))?;

// Or just use the brute force loader that should work
let vb = brute_force_flux_loader(
    Path::new("path/to/flux_dev.safetensors"),
    device,
    dtype,
)?;
The brute force loader just:

Loads all tensors
Applies all the name transformations
Splits QKV weights
Puts everything in the VarMap with the exact names your model expects
This should tell us exactly what's failing and give you a working loader. And yeah, writing our own engine would avoid all this framework impedance mismatch bullshit!
== end text

use candle_core::{Device, DType, Result, Tensor};
use candle_nn::{VarBuilder, VarMap};
use std::collections::HashMap;
use std::path::Path;

pub fn debug_flux_loading(
    checkpoint_path: &Path,
    device: Device,
    dtype: DType,
) -> Result<()> {
    println!("\n=== FLUX LOADER DEBUG ===\n");
    
    // Step 1: Load checkpoint
    println!("1. Loading checkpoint...");
    let checkpoint = safetensors::load(checkpoint_path, &device)?;
    println!("   Loaded {} tensors", checkpoint.len());
    
    // Step 2: Show what's in the checkpoint
    println!("\n2. Checkpoint contents (first 30 tensors):");
    let mut keys: Vec<_> = checkpoint.keys().collect();
    keys.sort();
    for (i, key) in keys.iter().enumerate() {
        if i < 30 {
            if let Some(tensor) = checkpoint.get(*key) {
                println!("   {} -> {:?}", key, tensor.shape().dims());
            }
        }
    }
    
    // Step 3: Show what tensors we're looking for
    println!("\n3. Expected tensor names (what model wants):");
    let expected = vec![
        "time_in.mlp.0.weight",
        "time_in.mlp.2.weight",
        "vector_in.mlp.0.weight",
        "double_blocks.0.img_attn.to_q.weight",
        "double_blocks.0.img_mlp.fc1.weight",
        "single_blocks.0.attn.to_q.weight",
        "single_blocks.0.mlp.fc1.weight",
    ];
    for name in &expected {
        println!("   {}", name);
    }
    
    // Step 4: Create VarMap and adapt
    println!("\n4. Adapting tensors...");
    let var_map = VarMap::new();
    
    // Adapt time embeddings
    if let Some(t) = checkpoint.get("time_in.in_layer.weight") {
        var_map.set_one("time_in.mlp.0.weight", t.to_device(&device)?.to_dtype(dtype)?)?;
        println!("   ✓ Adapted time_in.in_layer.weight -> time_in.mlp.0.weight");
    } else {
        println!("   ✗ Missing time_in.in_layer.weight");
    }
    
    // Test double block 0
    if let Some(t) = checkpoint.get("double_blocks.0.img_mlp.0.weight") {
        var_map.set_one("double_blocks.0.img_mlp.fc1.weight", t.to_device(&device)?.to_dtype(dtype)?)?;
        println!("   ✓ Adapted double_blocks.0.img_mlp.0.weight -> double_blocks.0.img_mlp.fc1.weight");
    } else {
        println!("   ✗ Missing double_blocks.0.img_mlp.0.weight");
    }
    
    // Step 5: Test VarBuilder access
    println!("\n5. Testing VarBuilder access:");
    let vb = VarBuilder::from_varmap(&var_map, dtype, &device);
    
    // Direct access
    match vb.get("time_in.mlp.0.weight") {
        Ok(t) => println!("   ✓ Direct access: time_in.mlp.0.weight -> {:?}", t.shape()),
        Err(e) => println!("   ✗ Direct access failed: {}", e),
    }
    
    // Prefixed access
    let vb_time = vb.pp("time_in").pp("mlp");
    match vb_time.get("0.weight") {
        Ok(t) => println!("   ✓ Prefixed access: time_in.mlp.0.weight -> {:?}", t.shape()),
        Err(e) => println!("   ✗ Prefixed access failed: {}", e),
    }
    
    // Double block access
    let vb_block = vb.pp("double_blocks.0");
    let vb_mlp = vb_block.pp("img_mlp");
    match vb_mlp.get("fc1.weight") {
        Ok(t) => println!("   ✓ Block access: double_blocks.0.img_mlp.fc1.weight -> {:?}", t.shape()),
        Err(e) => println!("   ✗ Block access failed: {}", e),
    }
    
    // Step 6: Show what's actually in VarMap
    println!("\n6. VarMap contents:");
    // Note: VarMap doesn't expose its contents directly, so we test known keys
    let test_keys = vec![
        "time_in.mlp.0.weight",
        "double_blocks.0.img_mlp.fc1.weight",
    ];
    for key in test_keys {
        match vb.get(key) {
            Ok(_) => println!("   ✓ {}", key),
            Err(_) => println!("   ✗ {}", key),
        }
    }
    
    println!("\n=== END DEBUG ===\n");
    
    Ok(())
}

/// Quick fix: Just dump everything with the right names
pub fn brute_force_flux_loader(
    checkpoint_path: &Path,
    device: Device,
    dtype: DType,
) -> Result<VarBuilder> {
    println!("Brute force loading Flux checkpoint...");
    
    let checkpoint = safetensors::load(checkpoint_path, &device)?;
    let var_map = VarMap::new();
    
    // Just iterate through everything and adapt names
    for (name, tensor) in &checkpoint {
        let converted = tensor.to_device(&device)?.to_dtype(dtype)?;
        
        // Adapt the name based on patterns
        let new_name = if name.contains(".in_layer.") {
            name.replace(".in_layer.", ".mlp.0.")
        } else if name.contains(".out_layer.") {
            name.replace(".out_layer.", ".mlp.2.")
        } else if name.contains("double_blocks") && name.contains(".0.weight") && !name.contains("norm") {
            name.replace(".0.weight", ".fc1.weight")
        } else if name.contains("double_blocks") && name.contains(".0.bias") && !name.contains("norm") {
            name.replace(".0.bias", ".fc1.bias")
        } else if name.contains("double_blocks") && name.contains(".2.weight") {
            name.replace(".2.weight", ".fc2.weight")
        } else if name.contains("double_blocks") && name.contains(".2.bias") {
            name.replace(".2.bias", ".fc2.bias")
        } else if name.contains("single_blocks") && name.contains(".linear1.") {
            name.replace(".linear1.", ".mlp.fc1.")
        } else if name.contains("single_blocks") && name.contains(".linear2.") {
            name.replace(".linear2.", ".mlp.fc2.")
        } else if name.contains(".proj.") && name.contains("attn") {
            name.replace(".proj.", ".to_out.0.")
        } else if name == "final_layer.linear.weight" {
            "final_layer.weight".to_string()
        } else if name == "final_layer.linear.bias" {
            "final_layer.bias".to_string()
        } else {
            name.clone()
        };
        
        // Handle QKV splitting
        if name.contains(".qkv.weight") {
            let prefix = name.trim_end_matches(".qkv.weight");
            let (total_dim, _) = tensor.dims2()?;
            let head_dim = total_dim / 3;
            
            let q = tensor.narrow(0, 0, head_dim)?;
            let k = tensor.narrow(0, head_dim, head_dim)?;
            let v = tensor.narrow(0, head_dim * 2, head_dim)?;
            
            var_map.set_one(&format!("{}.to_q.weight", prefix), q.to_device(&device)?.to_dtype(dtype)?)?;
            var_map.set_one(&format!("{}.to_k.weight", prefix), k.to_device(&device)?.to_dtype(dtype)?)?;
            var_map.set_one(&format!("{}.to_v.weight", prefix), v.to_device(&device)?.to_dtype(dtype)?)?;
        } else if name.contains(".qkv.bias") {
            let prefix = name.trim_end_matches(".qkv.bias");
            let total_dim = tensor.dims1()?;
            let head_dim = total_dim / 3;
            
            let q = tensor.narrow(0, 0, head_dim)?;
            let k = tensor.narrow(0, head_dim, head_dim)?;
            let v = tensor.narrow(0, head_dim * 2, head_dim)?;
            
            var_map.set_one(&format!("{}.to_q.bias", prefix), q.to_device(&device)?.to_dtype(dtype)?)?;
            var_map.set_one(&format!("{}.to_k.bias", prefix), k.to_device(&device)?.to_dtype(dtype)?)?;
            var_map.set_one(&format!("{}.to_v.bias", prefix), v.to_device(&device)?.to_dtype(dtype)?)?;
        } else {
            var_map.set_one(&new_name, converted)?;
        }
    }
    
    println!("Adapted all tensors");
    
    Ok(VarBuilder::from_varmap(&var_map, dtype, &device))
}

/// Test what's going wrong
pub fn test_varbuilder_behavior() -> Result<()> {
    let device = Device::Cpu;
    let dtype = DType::F32;
    
    println!("\n=== Testing VarBuilder behavior ===\n");
    
    // Create a test VarMap
    let var_map = VarMap::new();
    
    // Add some test tensors
    let t1 = Tensor::ones(&[10, 10], dtype, &device)?;
    var_map.set_one("double_blocks.0.img_mlp.fc1.weight", t1.clone())?;
    
    // Create VarBuilder
    let vb = VarBuilder::from_varmap(&var_map, dtype, &device);
    
    // Test different access patterns
    println!("Testing access patterns:");
    
    // Pattern 1: Direct full path
    match vb.get("double_blocks.0.img_mlp.fc1.weight") {
        Ok(_) => println!("✓ Direct: vb.get('double_blocks.0.img_mlp.fc1.weight')"),
        Err(e) => println!("✗ Direct: {}", e),
    }
    
    // Pattern 2: With pp()
    let vb2 = vb.pp("double_blocks").pp("0").pp("img_mlp");
    match vb2.get("fc1.weight") {
        Ok(_) => println!("✓ PP: vb.pp('double_blocks').pp('0').pp('img_mlp').get('fc1.weight')"),
        Err(e) => println!("✗ PP: {}", e),
    }
    
    // Pattern 3: With formatted pp()
    let vb3 = vb.pp("double_blocks.0").pp("img_mlp");
    match vb3.get("fc1.weight") {
        Ok(_) => println!("✓ PP2: vb.pp('double_blocks.0').pp('img_mlp').get('fc1.weight')"),
        Err(e) => println!("✗ PP2: {}", e),
    }
    
    // Pattern 4: What Candle expects
    let vb4 = vb.pp("double_blocks.0.img_mlp");
    match vb4.get("fc1.weight") {
        Ok(_) => println!("✓ PP3: vb.pp('double_blocks.0.img_mlp').get('fc1.weight')"),
        Err(e) => println!("✗ PP3: {}", e),
    }
    
    println!("\n=== End test ===\n");
    
    Ok(())
}

// Run all debug functions
pub fn debug_everything(checkpoint_path: &Path) -> Result<()> {
    let device = Device::cuda_if_available(0)?;
    let dtype = DType::BF16;
    
    // Test VarBuilder behavior first
    test_varbuilder_behavior()?;
    
    // Debug the actual loading
    debug_flux_loading(checkpoint_path, device, dtype)?;
    
    // Try brute force approach
    let vb = brute_force_flux_loader(checkpoint_path, device, dtype)?;
    
    // Test if it worked
    println!("\nTesting brute force loader:");
    match vb.get("time_in.mlp.0.weight") {
        Ok(t) => println!("✓ Can access time_in.mlp.0.weight: {:?}", t.shape()),
        Err(e) => println!("✗ Failed: {}", e),
    }
    
    Ok(())
}
