//! SDXL UNet forward pass with integrated LoRA
//! This implements the actual forward computation without using the standard UNet model

use anyhow::Result;
use candle_core::{Device, DType, Tensor, Module, D};
use std::collections::HashMap;
use super::sdxl_lora_trainer_fixed::LoRACollection;

/// Timestep embedding
pub fn timestep_embedding(timesteps: &Tensor, dim: usize, max_period: f32) -> Result<Tensor> {
    let half_dim = dim / 2;
    let freqs = (0..half_dim)
        .map(|i| (-(i as f32) * (max_period.ln() / half_dim as f32)).exp())
        .collect::<Vec<_>>();
    
    let device = timesteps.device();
    let freqs_tensor = Tensor::from_vec(freqs, &[half_dim], device)?;
    
    // Ensure timesteps is 1D by flattening
    let timesteps = timesteps.flatten_all()?;
    
    // Convert timesteps to f32 to match freqs
    let timesteps = timesteps.to_dtype(DType::F32)?;
    
    // For broadcasting, we need timesteps to be [batch_size, 1] and freqs to be [1, half_dim]
    // Then the result will be [batch_size, half_dim]
    let timesteps_expanded = timesteps.unsqueeze(1)?; // [batch_size, 1]
    let freqs_expanded = freqs_tensor.unsqueeze(0)?; // [1, half_dim]
    
    // Use broadcast_mul for proper broadcasting
    let args = timesteps_expanded.broadcast_mul(&freqs_expanded)?;
    
    let cos = args.cos()?;
    let sin = args.sin()?;
    
    Ok(Tensor::cat(&[cos, sin], 1)?)
}

/// ResNet block forward
fn resnet_block(
    x: &Tensor,
    time_emb: &Tensor,
    weights: &HashMap<String, Tensor>,
    prefix: &str,
) -> Result<Tensor> {
    let (batch_size, channels, height, width) = x.dims4()?;
    
    // First norm + conv
    let norm1_weight_key = format!("{}.norm1.weight", prefix);
    let norm1_bias_key = format!("{}.norm1.bias", prefix);
    
    // Check if key exists, if not try alternative naming
    let norm1_weight = weights.get(&norm1_weight_key)
        .ok_or_else(|| anyhow::anyhow!("Missing key: {}", norm1_weight_key))?;
    let norm1_bias = weights.get(&norm1_bias_key)
        .ok_or_else(|| anyhow::anyhow!("Missing key: {}", norm1_bias_key))?;
    
    let h = group_norm_32(x, 32, norm1_weight, norm1_bias)?;
    let h = h.silu()?;
    let h = conv2d(&h, &weights[&format!("{}.conv1.weight", prefix)], 
                   &weights[&format!("{}.conv1.bias", prefix)], 1, 1)?;
    
    // Time embedding projection
    let time_proj = time_emb
        .silu()?
        .matmul(&weights[&format!("{}.time_emb_proj.weight", prefix)].t()?)?
        .broadcast_add(&weights[&format!("{}.time_emb_proj.bias", prefix)])?
        .unsqueeze(2)?
        .unsqueeze(3)?;
    
    let h = (h + time_proj)?;
    
    // Second norm + conv
    let h = group_norm_32(&h, 32, &weights[&format!("{}.norm2.weight", prefix)], 
                         &weights[&format!("{}.norm2.bias", prefix)])?;
    let h = h.silu()?;
    let h = conv2d(&h, &weights[&format!("{}.conv2.weight", prefix)], 
                   &weights[&format!("{}.conv2.bias", prefix)], 1, 1)?;
    
    // Skip connection
    let skip = if x.dims()[1] != h.dims()[1] {
        // Channel change - use conv_shortcut
        conv2d(x, &weights[&format!("{}.conv_shortcut.weight", prefix)], 
               &weights[&format!("{}.conv_shortcut.bias", prefix)], 1, 1)?
    } else {
        x.clone()
    };
    
    Ok((h + skip)?)
}

/// Attention block with LoRA integration
fn attention_block(
    hidden_states: &Tensor,
    encoder_hidden_states: Option<&Tensor>,
    weights: &HashMap<String, Tensor>,
    lora: &LoRACollection,
    prefix: &str,
    heads: usize,
    dim_head: usize,
) -> Result<Tensor> {
    let (batch_size, sequence_length, channels) = hidden_states.dims3()?;
    let context = encoder_hidden_states.unwrap_or(hidden_states);
    
    // Debug context shape
    println!("DEBUG attention_block {}:", prefix);
    println!("  hidden_states shape: {:?}", hidden_states.dims());
    println!("  context shape: {:?}", context.dims());
    
    // Layer norm
    let norm_hidden = layer_norm(hidden_states, &weights[&format!("{}.norm.weight", prefix)], 
                                &weights[&format!("{}.norm.bias", prefix)])?;
    
    // Queries, keys, values with LoRA
    let q = lora.apply(
        &format!("{}.to_q", prefix),
        &norm_hidden,
        &weights[&format!("{}.to_q.weight", prefix)],
        weights.get(&format!("{}.to_q.bias", prefix)),
    )?;
    
    let k = lora.apply(
        &format!("{}.to_k", prefix),
        context,
        &weights[&format!("{}.to_k.weight", prefix)],
        weights.get(&format!("{}.to_k.bias", prefix)),
    )?;
    
    let v = lora.apply(
        &format!("{}.to_v", prefix),
        context,
        &weights[&format!("{}.to_v.weight", prefix)],
        weights.get(&format!("{}.to_v.bias", prefix)),
    )?;
    
    // Reshape for attention - handle both 2D and 3D context
    let context_seq_len = if context.dims().len() == 3 {
        context.dims()[1]
    } else if context.dims().len() == 2 {
        context.dims()[0]
    } else {
        return Err(anyhow::anyhow!("Unexpected context dimensions: {:?}", context.dims()));
    };
    
    let q = q.reshape((batch_size, sequence_length, heads, dim_head))?
            .transpose(1, 2)?;
    let k = k.reshape((batch_size, context_seq_len, heads, dim_head))?
            .transpose(1, 2)?;
    let v = v.reshape((batch_size, context_seq_len, heads, dim_head))?
            .transpose(1, 2)?;
    
    // Scaled dot-product attention
    let scale = (dim_head as f64).sqrt();
    let scores = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? / scale)?;
    let attn_weights = candle_nn::ops::softmax_last_dim(&scores)?;
    let attn_output = attn_weights.matmul(&v)?;
    
    // Reshape back
    let attn_output = attn_output
        .transpose(1, 2)?
        .reshape((batch_size, sequence_length, channels))?;
    
    // Output projection with LoRA
    let output = lora.apply(
        &format!("{}.to_out.0", prefix),
        &attn_output,
        &weights[&format!("{}.to_out.0.weight", prefix)],
        weights.get(&format!("{}.to_out.0.bias", prefix)),
    )?;
    
    // Add residual
    Ok((hidden_states + output)?)
}

/// SDXL UNet forward pass with LoRA
pub fn forward_sdxl_with_lora(
    sample: &Tensor,
    timestep: &Tensor,
    encoder_hidden_states: &Tensor,
    weights: &HashMap<String, Tensor>,
    lora: &LoRACollection,
) -> Result<Tensor> {
    let device = sample.device();
    let dtype = sample.dtype();
    
    // Time embedding
    let t_emb = {
        // SDXL uses 320 dimensional timestep embedding (not 320*4)
        let t = timestep_embedding(timestep, 320, 10000.0)?;
        // Convert to the same dtype as the weights
        let t = t.to_dtype(dtype)?;
        // Check which key format we have
        let (linear1_weight_key, linear1_bias_key, linear2_weight_key, linear2_bias_key) = 
            if weights.contains_key("time_embedding.linear_1.weight") {
                // Diffusers format
                ("time_embedding.linear_1.weight", "time_embedding.linear_1.bias",
                 "time_embedding.linear_2.weight", "time_embedding.linear_2.bias")
            } else {
                // Original SD format
                ("time_embed.0.weight", "time_embed.0.bias",
                 "time_embed.2.weight", "time_embed.2.bias")
            };
        
        let t = t.matmul(&weights[linear1_weight_key].t()?)?
                 .broadcast_add(&weights[linear1_bias_key])?
                 .silu()?;
        t.matmul(&weights[linear2_weight_key].t()?)?
         .broadcast_add(&weights[linear2_bias_key])?
    };
    
    // Initial convolution - check which key format we have
    let (conv_in_weight_key, conv_in_bias_key) = 
        if weights.contains_key("conv_in.weight") {
            // Diffusers format
            ("conv_in.weight", "conv_in.bias")
        } else {
            // Original SD format
            ("input_blocks.0.0.weight", "input_blocks.0.0.bias")
        };
    
    let mut h = conv2d(sample, &weights[conv_in_weight_key], &weights[conv_in_bias_key], 1, 1)?;
    
    // Down blocks
    let mut down_block_res_samples = Vec::new();
    let channel_mult = vec![1, 2, 4];
    
    for (i, &ch_mult) in channel_mult.iter().enumerate() {
        for j in 0..2 { // num_res_blocks = 2
            // ResNet block
            h = resnet_block(&h, &t_emb, weights, &format!("down_blocks.{}.resnets.{}", i, j))?;
            
            // Transformer block (attention)
            if weights.contains_key(&format!("down_blocks.{}.attentions.{}.norm.weight", i, j)) {
                let (b, c, h_size, w_size) = h.dims4()?;
                let h_flat = h.transpose(1, 2)?.transpose(2, 3)?
                            .reshape((b, h_size * w_size, c))?;
                
                // Self attention
                let h_flat = attention_block(
                    &h_flat,
                    None,
                    weights,
                    lora,
                    &format!("down_blocks.{}.attentions.{}.transformer_blocks.0.attn1", i, j),
                    8,
                    c / 8,
                )?;
                
                // Cross attention
                let h_flat = attention_block(
                    &h_flat,
                    Some(encoder_hidden_states),
                    weights,
                    lora,
                    &format!("down_blocks.{}.attentions.{}.transformer_blocks.0.attn2", i, j),
                    8,
                    c / 8,
                )?;
                
                // Feed forward
                let h_flat = feed_forward(&h_flat, weights, 
                    &format!("down_blocks.{}.attentions.{}.transformer_blocks.0.ff", i, j))?;
                
                h = h_flat.reshape((b, h_size, w_size, c))?
                         .transpose(2, 3)?.transpose(1, 2)?;
            }
            
            down_block_res_samples.push(h.clone());
        }
        
        // Downsample
        if i < channel_mult.len() - 1 {
            let ds = conv2d(&h, &weights[&format!("down_blocks.{}.downsamplers.0.conv.weight", i)],
                           &weights[&format!("down_blocks.{}.downsamplers.0.conv.bias", i)], 2, 1)?;
            down_block_res_samples.push(ds.clone());
            h = ds;
        }
    }
    
    // Middle block
    h = resnet_block(&h, &t_emb, weights, "mid_block.resnets.0")?;
    
    // Middle block attention
    let (b, c, h_size, w_size) = h.dims4()?;
    let h_flat = h.transpose(1, 2)?.transpose(2, 3)?
                 .reshape((b, h_size * w_size, c))?;
    
    let h_flat = attention_block(
        &h_flat,
        None,
        weights,
        lora,
        "mid_block.attentions.0.transformer_blocks.0.attn1",
        8,
        c / 8,
    )?;
    
    let h_flat = attention_block(
        &h_flat,
        Some(encoder_hidden_states),
        weights,
        lora,
        "mid_block.attentions.0.transformer_blocks.0.attn2",
        8,
        c / 8,
    )?;
    
    h = h_flat.reshape((b, h_size, w_size, c))?
             .transpose(2, 3)?.transpose(1, 2)?;
    
    h = resnet_block(&h, &t_emb, weights, "mid_block.resnets.1")?;
    
    // Up blocks - process in reverse order with skip connections
    for (i, &ch_mult) in channel_mult.iter().enumerate().rev() {
        for j in 0..(if i == 0 { 3 } else { 2 }) + 1 { // Extra layer in first up block
            // Pop skip connection from down blocks
            let skip = down_block_res_samples.pop()
                .ok_or_else(|| anyhow::anyhow!("Missing skip connection"))?;
            
            // Concatenate skip connection
            h = Tensor::cat(&[h, skip], 1)?;
            
            // ResNet block
            h = resnet_block(&h, &t_emb, weights, &format!("up_blocks.{}.resnets.{}", 3 - i, j))?;
            
            // Transformer block (attention) if not the last layer
            if j < (if i == 0 { 3 } else { 2 }) && weights.contains_key(&format!("up_blocks.{}.attentions.{}.norm.weight", 3 - i, j)) {
                let (b, c, h_size, w_size) = h.dims4()?;
                let h_flat = h.transpose(1, 2)?.transpose(2, 3)?
                            .reshape((b, h_size * w_size, c))?;
                
                // Self attention
                let h_flat = attention_block(
                    &h_flat,
                    None,
                    weights,
                    lora,
                    &format!("up_blocks.{}.attentions.{}.transformer_blocks.0.attn1", 3 - i, j),
                    8,
                    c / 8,
                )?;
                
                // Cross attention
                let h_flat = attention_block(
                    &h_flat,
                    Some(encoder_hidden_states),
                    weights,
                    lora,
                    &format!("up_blocks.{}.attentions.{}.transformer_blocks.0.attn2", 3 - i, j),
                    8,
                    c / 8,
                )?;
                
                // Feed forward
                let h_flat = feed_forward(&h_flat, weights, 
                    &format!("up_blocks.{}.attentions.{}.transformer_blocks.0.ff", 3 - i, j))?;
                
                h = h_flat.reshape((b, h_size, w_size, c))?
                         .transpose(2, 3)?.transpose(1, 2)?;
            }
        }
        
        // Upsample
        if i > 0 {
            if let Some(w) = weights.get(&format!("up_blocks.{}.upsamplers.0.conv.weight", 3 - i)) {
                // Nearest neighbor upsampling followed by conv
                h = h.upsample_nearest2d(h.dims()[2] * 2, h.dims()[3] * 2)?;
                h = h.conv2d(w, 1, 1, 1, 1)?;
                if let Some(b) = weights.get(&format!("up_blocks.{}.upsamplers.0.conv.bias", 3 - i)) {
                    h = h.broadcast_add(b)?;
                }
            }
        }
    }
    
    // Final output layers
    h = group_norm_32(&h, 32, &weights["conv_norm_out.weight"], &weights["conv_norm_out.bias"])?;
    h = h.silu()?;
    h = conv2d(&h, &weights["conv_out.weight"], &weights["conv_out.bias"], 1, 1)?;
    
    Ok(h)
}

// Helper functions

fn conv2d(x: &Tensor, weight: &Tensor, bias: &Tensor, stride: usize, padding: usize) -> Result<Tensor> {
    let out = x.conv2d(weight, stride, padding, 1, 1)?;
    // Reshape bias for proper broadcasting [C] -> [1, C, 1, 1]
    let bias = bias.reshape((1, bias.dims()[0], 1, 1))?;
    Ok(out.broadcast_add(&bias)?)
}

fn group_norm_32(x: &Tensor, groups: usize, scale: &Tensor, bias: &Tensor) -> Result<Tensor> {
    let (b, c, h, w) = x.dims4()?;
    let x = x.reshape((b, groups, c / groups, h, w))?;
    
    // Compute mean and variance for each group
    let mean = x.mean_keepdim(2)?.mean_keepdim(3)?.mean_keepdim(4)?;
    let x_centered = (x - mean)?;
    let var = x_centered.sqr()?.mean_keepdim(2)?.mean_keepdim(3)?.mean_keepdim(4)?;
    
    // Normalize
    let x_norm = (x_centered / (var + 1e-5)?.sqrt()?)?;
    
    // Reshape back and apply scale/bias
    let x_norm = x_norm.reshape((b, c, h, w))?;
    
    // Apply affine transformation
    let scale = scale.reshape((1, c, 1, 1))?;
    let bias = bias.reshape((1, c, 1, 1))?;
    
    Ok(x_norm.broadcast_mul(&scale)?.broadcast_add(&bias)?)
}

fn layer_norm(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    Ok(candle_nn::ops::layer_norm(x, weight, bias, 1e-5)?)
}

fn feed_forward(x: &Tensor, weights: &HashMap<String, Tensor>, prefix: &str) -> Result<Tensor> {
    let h = x.matmul(&weights[&format!("{}.net.0.proj.weight", prefix)].t()?)?
            .broadcast_add(&weights[&format!("{}.net.0.proj.bias", prefix)])?;
    
    // GEGLU activation
    let chunks = h.chunk(2, D::Minus1)?;
    let h1 = &chunks[0];
    let h2 = &chunks[1];
    let h = (h1.gelu()? * h2)?;
    
    let h = h.matmul(&weights[&format!("{}.net.2.weight", prefix)].t()?)?
            .broadcast_add(&weights[&format!("{}.net.2.bias", prefix)])?;
    
    Ok((x + h)?)
}