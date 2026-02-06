use eridiffusion::models::mmdit_blocks::{
    SelfAttention, QkNormKind, ArenaScratch, QKNorm, JointTransformerBlock, DismantledBlock,
    AdaLayerNorm, MLP,
};
use flame_core::{Device, Tensor, Shape, DType, Result};
use safetensors::SafeTensors;
use memmap2::Mmap;
use std::path::Path;
use std::sync::Arc;
use eridiffusion::ops::Linear;

fn load_tensor(tensors: &SafeTensors, name: &str, device: &Device) -> Result<Tensor> {
    let view = tensors.tensor(name).map_err(|e| flame_core::Error::InvalidOperation(e.to_string()))?;
    let shape = Shape::from_dims(view.shape());
    let data = view.data();
    
    // Assume F32
    let values: Vec<f32> = data.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
        
    let tensor = Tensor::from_vec(values, shape, device.cuda_device_arc())?;
    tensor.to_dtype(DType::BF16)
}

fn main() -> Result<()> {
    env_logger::init();
    let device = Device::cuda(0)?;
    
    // Load test data
    let file = std::fs::File::open("attention_test.safetensors").unwrap();
    let mmap = unsafe { Mmap::map(&file).unwrap() };
    let tensors = SafeTensors::deserialize(&mmap).unwrap();
    
    let hidden_states = load_tensor(&tensors, "hidden_states", &device)?;
    let encoder_hidden_states = load_tensor(&tensors, "encoder_hidden_states", &device)?;
    let temb = load_tensor(&tensors, "temb", &device)?;
    
    let block_output_hidden_ref = load_tensor(&tensors, "block_output_hidden", &device)?;
    let block_output_context_ref = load_tensor(&tensors, "block_output_context", &device)?;
    
    println!("Loaded test data.");
    println!("hidden_states: {:?}", hidden_states.shape().dims());
    println!("encoder_hidden_states: {:?}", encoder_hidden_states.shape().dims());
    println!("temb: {:?}", temb.shape().dims());
    
    // Load weights
    let mmdit_path = "/home/alex/SwarmUI/Models/diffusion_models/sd35-medium-mmdit.safetensors";
    let mmdit_file = std::fs::File::open(mmdit_path).unwrap();
    let mmdit_mmap = unsafe { Mmap::map(&mmdit_file).unwrap() };
    let mmdit_tensors = SafeTensors::deserialize(&mmdit_mmap).unwrap();
    
    // Construct JointTransformerBlock (Block 0)
    let hidden_size = 1536;
    let num_heads = 24;
    let mlp_ratio = 4.0;
    let qk_norm = QkNormKind::Rms;
    let cond_dim = 1536; // temb dim
    
    // We need to manually construct DismantledBlocks because new() is private?
    // No, I can use JointTransformerBlock::new() if I make it public?
    // Or just construct fields manually since I made structs public.
    
    // But constructing manually is tedious.
    // I'll assume JointTransformerBlock::new is NOT public (it wasn't in my edits).
    // I'll make JointTransformerBlock::new public?
    // Or just construct it.
    
    // Let's make JointTransformerBlock::new public in a separate step if needed.
    // For now, let's try to construct it.
    
    // Wait, DismantledBlock::new is private.
    // I need to construct DismantledBlock manually.
    
    let x_block = build_dismantled_block(hidden_size, num_heads, mlp_ratio, qk_norm, cond_dim, false, true, &device)?;
    let context_block = build_dismantled_block(hidden_size, num_heads, mlp_ratio, qk_norm, cond_dim, false, false, &device)?; // Context block has self_attn=false?
    // Wait, context block has self_attn=true usually?
    // In SD3, context block is just a transformer block.
    // Let's check SD3 structure.
    // Both have attention.
    
    let mut block = JointTransformerBlock {
        x_block,
        context_block,
    };
    
    // Load weights
    let prefix = "model.diffusion_model.joint_blocks.0";
    load_dismantled_block(&mut block.x_block, &mmdit_tensors, &format!("{}.x_block", prefix), &device)?;
    load_dismantled_block(&mut block.context_block, &mmdit_tensors, &format!("{}.context_block", prefix), &device)?;
    
    println!("Weights loaded.");
    
    // Run Forward
    let (x_out, context_out) = block.forward(&hidden_states, &encoder_hidden_states, &temb)?;
    
    // Compare
    compare("block_output_hidden", &x_out, &block_output_hidden_ref)?;
    compare("block_output_context", &context_out, &block_output_context_ref)?;
    
    Ok(())
}

use eridiffusion::ops::LayerNorm;

fn build_dismantled_block(
    hidden: usize,
    num_heads: usize,
    mlp_ratio: f32,
    qk_norm: QkNormKind,
    cond_dim: usize,
    pre_only: bool,
    self_attn: bool,
    device: &Device,
) -> Result<DismantledBlock> {
    let norm1 = LayerNorm::new(vec![hidden], 1e-6, device.cuda_device().clone())?;
    
    let attn = SelfAttention::new(hidden, num_heads, true, qk_norm, pre_only, device)?;
    
    let norm2 = if !pre_only {
        Some(LayerNorm::new(vec![hidden], 1e-6, device.cuda_device().clone())?)
    } else {
        None
    };
    
    let mlp = if !pre_only {
        Some(MLP::new(hidden, mlp_ratio, device)?)
    } else {
        None
    };
    
    // Modulation slots
    let modulation_slots = if self_attn {
        9
    } else if pre_only {
        2
    } else {
        6
    };
    
    let modulation = Linear::new_zeroed(cond_dim, hidden * modulation_slots, true, &device.cuda_device())?;
    
    let attn2 = if self_attn {
        Some(SelfAttention::new(hidden, num_heads, true, qk_norm, false, device)?)
    } else {
        None
    };
    
    Ok(DismantledBlock {
        norm1,
        attn,
        norm2,
        mlp,
        modulation,
        attn2,
        pre_only,
        self_attn,
        hidden,
        num_heads,
    })
}

fn load_dismantled_block(block: &mut DismantledBlock, tensors: &SafeTensors, prefix: &str, device: &Device) -> Result<()> {
    // norm1 (AdaLayerNormZeroX part 1)
    // No weights for norm1 (LayerNorm)
    
    // Load Modulation Linear weights
    let mod_w = load_weight(tensors, &format!("{}.adaLN_modulation.1.weight", prefix), device)?;
    let mod_b = load_weight(tensors, &format!("{}.adaLN_modulation.1.bias", prefix), device)?;
    
    println!("Loaded modulation weight for {}: {:?}", prefix, mod_w.shape().dims());
    
    block.modulation.copy_weight_from(&mod_w)?;
    block.modulation.copy_bias_from(&mod_b)?;
    
    // Load Attn
    load_attn_weights(&mut block.attn, tensors, &format!("{}.attn", prefix), device)?;
    
    // Load Attn2 (if exists)
    if let Some(attn2) = &mut block.attn2 {
        load_attn_weights(attn2, tensors, &format!("{}.attn2", prefix), device)?;
    }
    
    // Load Norm2 (if exists)
    // No weights for norm2 (LayerNorm)
    
    // Load MLP
    if let Some(mlp) = &mut block.mlp {
        // prefix.ff.net.0.proj.weight (fc1)
        // prefix.ff.net.2.weight (fc2)
        // Python `FeedForward` uses `GEGLU` which has `proj` (dim -> dim*4 * 2) and `linear` (dim*4 -> dim)?
        // No, `SD3` uses `GELU`.
        // `FeedForward` in `diffusers` for SD3:
        // `net`:
        // 0: GELU(approx, tanh)
        // 1: Linear(dim, dim*4) ? No.
        
        // Let's check `MLP` structure in Rust.
        // `fc1`: Linear(hidden, hidden_dim).
        // `fc2`: Linear(hidden_dim, hidden).
        
        // In Python `FeedForward`:
        // `net` is `nn.ModuleList`.
        // `net[0]` is `GELU`.
        // `net[1]` is `Linear`.
        // `net[2]` is `Linear`.
        // Wait, usually it's `Linear -> GELU -> Linear`.
        
        // I need to check `FeedForward` source in `diffusers`.
        // I'll assume standard keys:
        // `ff.net.0.proj.weight` (fc1)
        // `ff.net.2.weight` (fc2)
        
        let fc1_w = load_weight(tensors, &format!("{}.mlp.fc1.weight", prefix), device)?;
        let fc1_b = load_weight(tensors, &format!("{}.mlp.fc1.bias", prefix), device)?;
        let fc2_w = load_weight(tensors, &format!("{}.mlp.fc2.weight", prefix), device)?;
        let fc2_b = load_weight(tensors, &format!("{}.mlp.fc2.bias", prefix), device)?;
        
        let (fc1, fc2) = mlp.fc_layers_mut();
        fc1.copy_weight_from(&fc1_w)?;
        fc1.copy_bias_from(&fc1_b)?;
        fc2.copy_weight_from(&fc2_w)?;
        fc2.copy_bias_from(&fc2_b)?;
    }
    
    Ok(())
}

fn load_adaln(adaln: &mut AdaLayerNorm, tensors: &SafeTensors, prefix: &str, device: &Device) -> Result<()> {
    // Not used anymore
    Ok(())
}

// ... helper functions ...

fn load_attn_weights(attn: &mut SelfAttention, tensors: &SafeTensors, prefix: &str, device: &Device) -> Result<()> {
    // qkv
    let qkv_w = load_weight(tensors, &format!("{}.qkv.weight", prefix), device)?;
    let qkv_b = load_weight(tensors, &format!("{}.qkv.bias", prefix), device)?;
    attn.copy_qkv_weight_from(&qkv_w)?;
    attn.copy_qkv_bias_from(&qkv_b)?;
    
    // proj
    let proj_w = load_weight(tensors, &format!("{}.proj.weight", prefix), device)?;
    let proj_b = load_weight(tensors, &format!("{}.proj.bias", prefix), device)?;
    if let Some(proj) = attn.proj_mut() {
        proj.copy_weight_from(&proj_w)?;
        proj.copy_bias_from(&proj_b)?;
    }
    
    // qk_norm
    let ln_q_w = load_weight(tensors, &format!("{}.ln_q.weight", prefix), device)?;
    let ln_k_w = load_weight(tensors, &format!("{}.ln_k.weight", prefix), device)?;
    
    let (rms_q, rms_k) = attn.qk_norm_mut().rms_norms_mut();
    if let Some(rms) = rms_q {
        rms.copy_weight_from(&ln_q_w)?;
    }
    if let Some(rms) = rms_k {
        rms.copy_weight_from(&ln_k_w)?;
    }
    
    Ok(())
}

fn load_weight(tensors: &SafeTensors, name: &str, device: &Device) -> Result<Tensor> {
    let view = tensors.tensor(name).map_err(|e| flame_core::Error::InvalidOperation(format!("Missing {}: {}", name, e)))?;
    let shape = Shape::from_dims(view.shape());
    
    // Weights are BF16 in safetensors file?
    // sd35-medium-mmdit.safetensors is likely BF16 or F16.
    // We need to handle dtype.
    
    match view.dtype() {
        safetensors::Dtype::BF16 => {
             let data = view.data();
             let values: Vec<f32> = data.chunks_exact(2)
                .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect();
             let t = Tensor::from_vec(values, shape, device.cuda_device_arc())?;
             t.to_dtype(DType::BF16)
        }
        safetensors::Dtype::F16 => {
             let data = view.data();
             let values: Vec<f32> = data.chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect();
             let t = Tensor::from_vec(values, shape, device.cuda_device_arc())?;
             t.to_dtype(DType::BF16)
        }
        safetensors::Dtype::F32 => {
             let data = view.data();
             let values: Vec<f32> = data.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
             let t = Tensor::from_vec(values, shape, device.cuda_device_arc())?;
             t.to_dtype(DType::BF16)
        }
        _ => panic!("Unsupported dtype"),
    }
}

fn compare(name: &str, a: &Tensor, b: &Tensor) -> Result<()> {
    let a_f32 = a.to_vec_f32()?;
    let b_f32 = b.to_vec_f32()?;
    
    let mut max_diff = 0.0;
    let mut sum_diff = 0.0;
    
    for (v1, v2) in a_f32.iter().zip(b_f32.iter()) {
        let diff = (v1 - v2).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        sum_diff += diff;
    }
    
    let mean_diff = sum_diff / a_f32.len() as f32;
    println!("{}: Max diff: {:.6}, Mean diff: {:.6}", name, max_diff, mean_diff);
    
    if max_diff > 0.1 {
        println!("!! SIGNIFICANT DIFFERENCE !!");
    }
    
    Ok(())
}
