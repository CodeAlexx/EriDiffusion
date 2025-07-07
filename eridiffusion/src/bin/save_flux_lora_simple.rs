//! Simple example of Flux LoRA tensor naming for AI-Toolkit compatibility

use anyhow::Result;

fn main() -> Result<()> {
    println!("Flux LoRA Tensor Naming Convention for AI-Toolkit:\n");
    
    // Show the expected tensor naming pattern
    let rank = 32;
    let block_types = ["double_blocks", "single_blocks"];
    
    println!("Expected tensor names for Flux LoRA (rank {}):", rank);
    println!("==========================================\n");
    
    // Double blocks
    println!("Double Blocks (19 blocks total):");
    for block_idx in 0..2 {  // Just show first 2 as example
        println!("\n  Block {}:", block_idx);
        
        // Image attention
        for target in ["to_q", "to_k", "to_v"] {
            println!("    transformer.double_blocks.{}.img_attn.{}.lora_A", block_idx, target);
            println!("    transformer.double_blocks.{}.img_attn.{}.lora_B", block_idx, target);
        }
        
        // Text attention
        for target in ["to_q", "to_k", "to_v"] {
            println!("    transformer.double_blocks.{}.txt_attn.{}.lora_A", block_idx, target);
            println!("    transformer.double_blocks.{}.txt_attn.{}.lora_B", block_idx, target);
        }
        
        // MLPs
        println!("    transformer.double_blocks.{}.img_mlp.0.lora_A", block_idx);
        println!("    transformer.double_blocks.{}.img_mlp.0.lora_B", block_idx);
        println!("    transformer.double_blocks.{}.img_mlp.2.lora_A", block_idx);
        println!("    transformer.double_blocks.{}.img_mlp.2.lora_B", block_idx);
        println!("    transformer.double_blocks.{}.txt_mlp.0.lora_A", block_idx);
        println!("    transformer.double_blocks.{}.txt_mlp.0.lora_B", block_idx);
        println!("    transformer.double_blocks.{}.txt_mlp.2.lora_A", block_idx);
        println!("    transformer.double_blocks.{}.txt_mlp.2.lora_B", block_idx);
    }
    
    println!("\n  ... (17 more double blocks)");
    
    // Single blocks
    println!("\n\nSingle Blocks (38 blocks total):");
    for block_idx in 0..2 {  // Just show first 2 as example
        println!("\n  Block {}:", block_idx);
        
        // Self attention
        for target in ["to_q", "to_k", "to_v"] {
            println!("    transformer.single_blocks.{}.attn.{}.lora_A", block_idx, target);
            println!("    transformer.single_blocks.{}.attn.{}.lora_B", block_idx, target);
        }
        
        // MLP
        println!("    transformer.single_blocks.{}.mlp.0.lora_A", block_idx);
        println!("    transformer.single_blocks.{}.mlp.0.lora_B", block_idx);
        println!("    transformer.single_blocks.{}.mlp.2.lora_A", block_idx);
        println!("    transformer.single_blocks.{}.mlp.2.lora_B", block_idx);
    }
    
    println!("\n  ... (36 more single blocks)");
    
    println!("\n\nKey differences from our current implementation:");
    println!("1. Use 'transformer.' prefix (not 'model.diffusion_model.')");
    println!("2. Use 'lora_A' and 'lora_B' (not 'lora_up' and 'lora_down')");
    println!("3. MLP layers use indices 0 and 2 (not 'fc1' and 'fc2')");
    println!("4. Attention uses 'to_q', 'to_k', 'to_v' (not 'qkv')");
    
    println!("\n\nShape information:");
    println!("- lora_A shapes: [original_dim, {}]", rank);
    println!("- lora_B shapes: [{}, original_dim]", rank);
    println!("- Hidden size: 3072");
    println!("- MLP hidden size: 12288 (4x hidden)");
    
    println!("\n\nTo save with safetensors crate:");
    println!("1. Create tensors with correct names");
    println!("2. Implement View trait for tensor wrapper");
    println!("3. Use safetensors::serialize_to_file()");
    println!("4. Include metadata: rank, alpha, format='pt'");
    
    Ok(())
}