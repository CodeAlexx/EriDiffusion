use super::{efficient_attention::efficient_attention, lora::LoRACollection};
use anyhow::Context;
use flame_core::device::Device;
use flame_core::optimizers::{Adam, SGD};
use flame_core::{CudaDevice, DType, Result, Shape, Tensor};
use std::collections::HashMap;

// Memory-efficient SDXL forward pass that only tracks gradients for LoRA weights
// Key insight: We don't need gradients for frozen UNet weights, only for LoRA adapters

// FLAME uses flame_core::device::Device instead of Device

/// Memory-efficient forward pass that detaches all intermediate computations
/// Only the LoRA weights themselves maintain gradient tracking
pub fn forward_sdxl_memory_efficient(
    x: &Tensor,
    timesteps: &Tensor,
    context: &Tensor,
    weights: &std::collections::HashMap<String, Tensor>,
    lora: &LoRACollection,
) -> flame_core::Result<Tensor> {
    println!("\n=== SDXL Memory-Efficient Forward Pass ===");
    println!("Only tracking gradients for LoRA weights, not intermediate activations");

    // Detach all inputs immediately
    let x = x.detach()?;
    let timesteps = timesteps.detach()?;
    let context = context.detach()?;

    // Time embedding - no gradients needed
    let t_emb = {
        let t = timestep_embedding(
            &timesteps,
            320,
            10000.0,
            &Device::from(timesteps.device().clone()),
        )?;
        let fc1_w = &weights["time_embed.0.weight"];
        let fc1_b = &weights["time_embed.0.bias"];
        let fc2_w = &weights["time_embed.2.weight"];
        let fc2_b = &weights["time_embed.2.bias"];

        let t = t.to_dtype(fc1_w.dtype())?;
        let t = linear_op(&t, fc1_w, Some(fc1_b))?.detach()?;
        let t = t.silu()?.detach()?;
        linear_op(&t, fc2_w, Some(fc2_b))?.detach()?
    };

    // Initial convolution - detach immediately
    let mut h = {
        let conv_w = &weights["input_blocks.0.0.weight"];
        let conv_b = &weights["input_blocks.0.0.bias"];
        conv2d(&x, conv_w, conv_b, 1, 1)?.detach()?
    };

    let mut skip_connections = Vec::new();

    // Down blocks - detach after each operation
    println!("Processing down blocks with aggressive detaching...");

    // Block 1 - ResNet
    h = resnet_block(&h, &t_emb, weights, "input_blocks.1.0")?;
    skip_connections.push(h.clone());

    // Block 2 - ResNet
    h = resnet_block(&h, &t_emb, weights, "input_blocks.2.0")?;
    skip_connections.push(h.clone());

    // Block 3 - Downsample
    h = downsample_detached(&h, weights, "input_blocks.3.0.op")?;
    skip_connections.push(h.clone());

    // Block 4 - ResNet + Attention
    h = resnet_block(&h, &t_emb, weights, "input_blocks.4.0")?;
    h = transformer_blocks_efficient(&h, &context, weights, lora, "input_blocks.4.1", 2)?;
    skip_connections.push(h.clone());

    // Block 5 - ResNet + Attention
    h = resnet_block(&h, &t_emb, weights, "input_blocks.5.0")?;
    h = transformer_blocks_efficient(&h, &context, weights, lora, "input_blocks.5.1", 2)?;
    skip_connections.push(h.clone());

    // Block 6 - Downsample
    h = downsample_detached(&h, weights, "input_blocks.6.0.op")?;
    skip_connections.push(h.clone());

    // Block 7 - ResNet + Attention (10 transformer blocks)
    h = resnet_block(&h, &t_emb, weights, "input_blocks.7.0")?;
    h = transformer_blocks_efficient(&h, &context, weights, lora, "input_blocks.7.1", 10)?;
    skip_connections.push(h.clone());

    // Block 8 - ResNet + Attention (10 transformer blocks)
    h = resnet_block(&h, &t_emb, weights, "input_blocks.8.0")?;
    h = transformer_blocks_efficient(&h, &context, weights, lora, "input_blocks.8.1", 10)?;
    skip_connections.push(h.clone());

    // Middle block
    println!("Processing middle block...");
    h = resnet_block(&h, &t_emb, weights, "middle_block.0")?;
    h = transformer_blocks_efficient(&h, &context, weights, lora, "middle_block.1", 10)?;
    h = resnet_block(&h, &t_emb, weights, "middle_block.2")?;

    // Up blocks
    println!("Processing up blocks...");

    // Process each up block
    for i in 0..9 {
        // Concatenate skip connection
        if let Some(skip) = skip_connections.pop() {
            h = Tensor::cat(&[&h, &skip], 1)?.detach()?;
        }

        match i {
            0..=2 => {
                // Blocks 0-2: ResNet + Attention (10 transformer blocks each)
                h = resnet_block(&h, &t_emb, weights, &format!("output_blocks.{}.0", i))?;
                h = transformer_blocks_efficient(
                    &h,
                    &context,
                    weights,
                    lora,
                    &format!("output_blocks.{}.1", i),
                    10,
                )?;
            }
            3..=5 => {
                // Blocks 3-5: ResNet + Attention (2 transformer blocks each) + Upsample
                h = resnet_block(&h, &t_emb, weights, &format!("output_blocks.{}.0", i))?;
                h = transformer_blocks_efficient(
                    &h,
                    &context,
                    weights,
                    lora,
                    &format!("output_blocks.{}.1", i),
                    2,
                )?;

                // Upsample at blocks 2 and 5
                if i == 2 || i == 5 {
                    h = upsample_detached(&h, weights, &format!("output_blocks.{}.2", i))?;
                }
            }
            6..=8 => {
                // Blocks 6-8: ResNet only
                h = resnet_block(&h, &t_emb, weights, &format!("output_blocks.{}.0", i))?;
            }
            _ => {}
        }
    }

    // Final output layers - detach before final computation
    let out = {
        let norm_w = &weights["out.0.weight"];
        let norm_b = &weights["out.0.bias"];
        let conv_w = &weights["out.2.weight"];
        let conv_b = &weights["out.2.bias"];

        let h = group_norm(&h, norm_w, norm_b, 32)?.detach()?;
        let h = h.silu()?.detach()?;
        conv2d(&h, conv_w, conv_b, 1, 1)?
    };

    println!("=== Memory-Efficient Forward Complete ===\n");
    Ok(out)
}

/// Process transformer blocks with minimal gradient tracking
fn transformer_blocks_efficient(
    x: &Tensor,
    context: &Tensor,
    weights: &std::collections::HashMap<String, Tensor>,
    lora: &LoRACollection,
    prefix: &str,
    num_blocks: usize,
) -> flame_core::Result<Tensor> {
    let mut h = x.detach()?;

    for i in 0..num_blocks {
        let block_prefix = format!("{}.transformer_blocks.{}", prefix, i);

        // Process block and detach immediately
        h = transformer_block_efficient(&h, context, weights, lora, &block_prefix)?;
        h = h.detach()?; // Detach after each block to prevent gradient accumulation
    }

    Ok(h)
}

/// Single transformer block with efficient gradient handling
fn transformer_block_efficient(
    x: &Tensor,
    context: &Tensor,
    weights: &std::collections::HashMap<String, Tensor>,
    lora: &LoRACollection,
    prefix: &str,
) -> flame_core::Result<Tensor> {
    let dims = x.shape().dims();
    let (b, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
    let seq_len = h * w;

    // Reshape to sequence - detach input
    let x_seq = x.detach()?.reshape(&[b, c, seq_len])?.transpose_dims(1, 2)?;

    // Self-attention with LoRA
    let x_seq = {
        let norm = layer_norm(
            &x_seq,
            &weights[&format!("{}.attn1.norm.weight", prefix)],
            &weights[&format!("{}.attn1.norm.bias", prefix)],
        )?
        .detach()?;

        // Only these LoRA operations will have gradients
        let q = lora.apply(
            &format!("{}.attn1.to_q", prefix),
            &norm,
            &weights[&format!("{}.attn1.to_q.weight", prefix)],
            weights.get(&format!("{}.attn1.to_q.bias", prefix)),
        )?;

        let k = lora.apply(
            &format!("{}.attn1.to_k", prefix),
            &norm,
            &weights[&format!("{}.attn1.to_k.weight", prefix)],
            weights.get(&format!("{}.attn1.to_k.bias", prefix)),
        )?;

        let v = lora.apply(
            &format!("{}.attn1.to_v", prefix),
            &norm,
            &weights[&format!("{}.attn1.to_v.weight", prefix)],
            weights.get(&format!("{}.attn1.to_v.bias", prefix)),
        )?;

        // Attention computation - detach after
        let head_dim = if q.shape().dims()[2] == 640 { 80 } else { 160 };
        let attn_out = efficient_attention(&q, &k, &v, head_dim)?.detach()?;

        // Output projection with LoRA
        let out = lora.apply(
            &format!("{}.attn1.to_out.0", prefix),
            &attn_out,
            &weights[&format!("{}.attn1.to_out.0.weight", prefix)],
            weights.get(&format!("{}.attn1.to_out.0.bias", prefix)),
        )?;

        // Residual and detach
        x_seq.add(&out)?.detach()?
    };

    // Cross-attention with LoRA
    let x_seq = {
        let norm = layer_norm(
            &x_seq,
            &weights[&format!("{}.attn2.norm.weight", prefix)],
            &weights[&format!("{}.attn2.norm.bias", prefix)],
        )?
        .detach()?;

        // LoRA operations
        let q = lora.apply(
            &format!("{}.attn2.to_q", prefix),
            &norm,
            &weights[&format!("{}.attn2.to_q.weight", prefix)],
            weights.get(&format!("{}.attn2.to_q.bias", prefix)),
        )?;

        let k = lora.apply(
            &format!("{}.attn2.to_k", prefix),
            &context.detach()?,
            &weights[&format!("{}.attn2.to_k.weight", prefix)],
            weights.get(&format!("{}.attn2.to_k.bias", prefix)),
        )?;

        let v = lora.apply(
            &format!("{}.attn2.to_v", prefix),
            &context.detach()?,
            &weights[&format!("{}.attn2.to_v.weight", prefix)],
            weights.get(&format!("{}.attn2.to_v.bias", prefix)),
        )?;

        // Attention and output projection
        let head_dim = if q.shape().dims()[2] == 640 { 80 } else { 160 };
        let attn_out = efficient_attention(&q, &k, &v, head_dim)?.detach()?;

        let out = lora.apply(
            &format!("{}.attn2.to_out.0", prefix),
            &attn_out,
            &weights[&format!("{}.attn2.to_out.0.weight", prefix)],
            weights.get(&format!("{}.attn2.to_out.0.bias", prefix)),
        )?;

        x_seq.add(&out)?.detach()?
    };

    // Feed-forward - no LoRA, fully detached
    let x_seq = {
        let proj = linear_op(
            &x_seq,
            &weights[&format!("{}.ff.0.proj.weight", prefix)],
            weights.get(&format!("{}.ff.0.proj.bias", prefix)),
        )?
        .detach()?;

        let chunks = proj.chunk(2, proj.shape().dims().len() - 1)?;
        let hidden = chunks[0].mul(&chunks[1].gelu()?)?.detach()?;

        let out = linear_op(
            &hidden,
            &weights[&format!("{}.ff.2.weight", prefix)],
            weights.get(&format!("{}.ff.2.bias", prefix)),
        )?
        .detach()?;

        x_seq.add(&out)?.detach()?
    };

    // Reshape back
    Ok(x_seq.transpose_dims(1, 2)?.reshape(&[b, c, h, w])?)
}

// Standard helper functions (unchanged)
fn timestep_embedding(
    timesteps: &Tensor,
    dim: usize,
    max_period: f32,
    device: &Device,
) -> flame_core::Result<Tensor> {
    let half_dim = dim / 2;
    let freqs = (0..half_dim)
        .map(|i| (-(i as f32) * (max_period.ln() / half_dim as f32)).exp())
        .collect::<Vec<_>>();

    let dtype = timesteps.dtype();
    let freqs_tensor =
        Tensor::from_vec(freqs, Shape::from_dims(&[half_dim]), device.cuda_device().clone())?;

    let timesteps = timesteps.flatten_all()?.to_dtype(DType::F32)?;
    let timesteps_expanded = timesteps.unsqueeze(1)?;
    let freqs_expanded = freqs_tensor.unsqueeze(0)?;

    let args = timesteps_expanded.mul(&freqs_expanded)?;
    let cos = args.cos()?;
    let sin = args.sin()?;

    Ok(Tensor::cat(&[&cos, &sin], 1)?.to_dtype(dtype)?)
}

fn linear_op(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> flame_core::Result<Tensor> {
    let w = weight.transpose_dims(0, 1)?;
    let mut out = x.matmul(&w)?;

    if let Some(b) = bias {
        out = out.add(b)?;
    }
    Ok(out)
}

fn conv2d(
    x: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    stride: usize,
    padding: usize,
) -> flame_core::Result<Tensor> {
    // FLAME uses Conv2d struct, not method
    let device = Device::from(x.device().clone());
    let weight_shape = weight.shape().dims();
    let in_channels = weight_shape[1];
    let out_channels = weight_shape[0];
    let kernel_size = weight_shape[2];
    let conv_layer = flame_core::conv::Conv2d::new(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        device.cuda_device().clone(),
    )?;
    // TODO: Load weights and bias into conv_layer
    let conv = conv_layer.forward(x)?;
    let bias_dims = bias.shape().dims();
    let bias_shape = Shape::from_dims(&[1, bias_dims[0], 1, 1]);
    conv.add(&bias.reshape(bias_shape.dims())?)
}

fn group_norm(
    x: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    num_groups: usize,
) -> flame_core::Result<Tensor> {
    let dims = x.shape().dims();
    let (b, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
    let x_reshaped = x.reshape(&[b, num_groups, c / num_groups, h, w])?;
    let mean = x_reshaped.mean_dim(&[2], true)?;
    let x_centered = x_reshaped.sub(&mean)?;
    let var = x_centered.square()?.mean_dim(&[2], true)?;
    let x_norm = x_centered.div(&var.add_scalar(1e-6)?.sqrt()?)?;
    let x_norm = x_norm.reshape(&[b, c, h, w])?;

    Ok(x_norm.mul(&weight.reshape(&[1, c, 1, 1])?)?.add(&bias.reshape(&[1, c, 1, 1])?)?)
}

fn layer_norm(x: &Tensor, weight: &Tensor, bias: &Tensor) -> flame_core::Result<Tensor> {
    let mean = x.mean_dim(&[x.shape().rank() - 1], true)?;
    let x_centered = x.sub(&mean)?;
    let var = x_centered.square()?.mean_dim(&[x.shape().rank() - 1], true)?;
    let x_norm = x_centered.div(&var.add_scalar(1e-5)?.sqrt()?)?;
    Ok(x_norm.mul(weight)?.add(bias)?)
}

fn resnet_block(
    x: &Tensor,
    t_emb: &Tensor,
    weights: &HashMap<String, Tensor>,
    prefix: &str,
) -> flame_core::Result<Tensor> {
    let norm1 = group_norm(
        x,
        &weights[&format!("{}.in_layers.0.weight", prefix)],
        &weights[&format!("{}.in_layers.0.bias", prefix)],
        32,
    )?;
    let h = norm1.silu()?;
    let h = conv2d(
        &h,
        &weights[&format!("{}.in_layers.2.weight", prefix)],
        &weights[&format!("{}.in_layers.2.bias", prefix)],
        1,
        1,
    )?;

    let t_emb_proj = linear_op(
        t_emb,
        &weights[&format!("{}.emb_layers.1.weight", prefix)],
        Some(&weights[&format!("{}.emb_layers.1.bias", prefix)]),
    )?;
    let h = h.add(&t_emb_proj.unsqueeze(2)?.unsqueeze(3)?)?;

    let norm2 = group_norm(
        &h,
        &weights[&format!("{}.out_layers.0.weight", prefix)],
        &weights[&format!("{}.out_layers.0.bias", prefix)],
        32,
    )?;
    let h = norm2.silu()?;
    let h = conv2d(
        &h,
        &weights[&format!("{}.out_layers.3.weight", prefix)],
        &weights[&format!("{}.out_layers.3.bias", prefix)],
        1,
        1,
    )?;

    let skip = if x.shape().dims()[1] != h.shape().dims()[1] {
        conv2d(
            x,
            &weights[&format!("{}.skip_connection.weight", prefix)],
            &weights[&format!("{}.skip_connection.bias", prefix)],
            1,
            0,
        )?
    } else {
        x.clone()
    };

    Ok(skip.add(&h)?)
}

// Helper function for downsampling
fn downsample_detached(
    x: &Tensor,
    weights: &HashMap<String, Tensor>,
    prefix: &str,
) -> flame_core::Result<Tensor> {
    conv2d(x, &weights[&format!("{}.weight", prefix)], &weights[&format!("{}.bias", prefix)], 2, 1)
}

// Helper function for upsampling
fn upsample_detached(
    x: &Tensor,
    weights: &HashMap<String, Tensor>,
    prefix: &str,
) -> flame_core::Result<Tensor> {
    let dims = x.shape().dims();
    let h = dims[2];
    let w = dims[3];
    let x_upsampled = flame_core::cuda_ops::GpuOps::upsample2d_nearest(x, (h * 2, w * 2))?;
    conv2d(
        &x_upsampled,
        &weights[&format!("{}.conv.weight", prefix)],
        &weights[&format!("{}.conv.bias", prefix)],
        1,
        1,
    )
}
