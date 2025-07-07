//! Convert between AI-Toolkit LoRA format (to_q/to_k/to_v) and Candle format (qkv)

use anyhow::Result;

fn main() -> Result<()> {
    println!("=== Flux LoRA Format Conversion ===\n");
    
    println!("The Issue:");
    println!("- Base Flux model uses: qkv (combined)");
    println!("- AI-Toolkit LoRA uses: to_q, to_k, to_v (separate)");
    
    println!("\nConversion Strategy:");
    
    println!("\n1. Loading AI-Toolkit LoRA into Candle model:");
    println!("   When you see:");
    println!("     transformer.double_blocks.0.img_attn.to_q.lora_A");
    println!("     transformer.double_blocks.0.img_attn.to_k.lora_A");
    println!("     transformer.double_blocks.0.img_attn.to_v.lora_A");
    println!("   ");
    println!("   These need to be applied to the qkv layer:");
    println!("     - lora_A weights can be the same (they share input dim)");
    println!("     - lora_B weights need to be concatenated along output dim");
    
    println!("\n2. Example conversion code:");
    println!("   ```rust");
    println!("   // For lora_A (input projection) - just use one of them");
    println!("   let qkv_lora_a = to_q_lora_a; // They're identical for Q,K,V");
    println!("   ");
    println!("   // For lora_B (output projection) - concatenate");
    println!("   let qkv_lora_b = Tensor::cat(&[");
    println!("       &to_q_lora_b,");
    println!("       &to_k_lora_b,");
    println!("       &to_v_lora_b");
    println!("   ], 0)?; // Concatenate along dim 0");
    println!("   ```");
    
    println!("\n3. Alternative: Modify attention to use separate layers");
    println!("   Instead of:");
    println!("     qkv: Linear // [D, 3*D]");
    println!("   ");
    println!("   Use:");
    println!("     to_q: Linear // [D, D]");
    println!("     to_k: Linear // [D, D]");
    println!("     to_v: Linear // [D, D]");
    
    println!("\n4. At forward pass:");
    println!("   ```rust");
    println!("   // Original (with qkv):");
    println!("   let qkv = xs.apply(&self.qkv)?;");
    println!("   let q = qkv.i((.., .., 0))?;");
    println!("   ");
    println!("   // Modified (with separate):");
    println!("   let q = xs.apply(&self.to_q)?;");
    println!("   let k = xs.apply(&self.to_k)?;");
    println!("   let v = xs.apply(&self.to_v)?;");
    println!("   ```");
    
    Ok(())
}