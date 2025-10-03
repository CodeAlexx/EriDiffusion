use super::{lora::LoRACollection, sdxl_transformer_block_flash::transformer_block};
use flame_core::device::Device;
use flame_core::optimizers::{Adam, SGD};
use flame_core::{CudaDevice, DType, Result, Shape, Tensor};

// SDXL UNet forward pass for SD-format checkpoints with Flash Attention support
// This handles the actual SDXL checkpoint format which uses SD naming conventions

// FLAME uses flame_core::device::Device instead of Device

// FLAME uses flame_core::device::Device instead of Device

/// Time embedding with SD format
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
        Tensor::from_vec(freqs, Shape::from_dims(&[half_dim]), device.cuda_device().clone())?
            .to_dtype(dtype)?;

    // Use the dtype of the input timesteps to maintain consistency
    let timesteps = timesteps.flatten_all()?;
    let timesteps_expanded = timesteps.unsqueeze(1)?;
    let freqs_expanded = freqs_tensor.unsqueeze(0)?;

    let args = timesteps_expanded.mul(&freqs_expanded)?;
    let cos = args.cos()?;
    let sin = args.sin()?;

    Tensor::cat(&[&cos, &sin], 1)
}

/// ResNet block with SD naming convention
fn resnet_block(
    x: &Tensor,
    time_emb: &Tensor,
    weights: &std::collections::HashMap<String, Tensor>,
    prefix: &str,
) -> flame_core::Result<Tensor> {
    let _device = x.device();
    let _dtype = x.dtype();

    // GroupNorm with SD naming
    let norm1_weight_key = format!("{}.in_layers.0.weight", prefix);
    let norm1_bias_key = format!("{}.in_layers.0.bias", prefix);

    let h = if let (Some(norm_weight), Some(norm_bias)) =
        (weights.get(&norm1_weight_key), weights.get(&norm1_bias_key))
    {
        group_norm_32(x, 32, norm_weight, norm_bias)?
    } else {
        // Skip normalization if weights not found
        x.clone()
    };

    // SiLU activation
    let h = h.silu()?;

    // First conv
    let conv1_weight = weights.get(&format!("{}.in_layers.2.weight", prefix)).ok_or_else(|| {
        flame_core::Error::InvalidOperation("Missing conv1 weight".into())
    })?;
    let conv1_bias = weights.get(&format!("{}.in_layers.2.bias", prefix)).ok_or_else(|| {
        flame_core::Error::InvalidOperation("Missing conv1 bias".into())
    })?;

    let h = conv2d(&h, conv1_weight, conv1_bias, 1, 1)?;
    println!("    Conv1 output shape: {:?}", h.shape());

    // Time embedding projection
    if let (Some(time_weight), Some(time_bias)) = (
        weights.get(&format!("{}.emb_layers.1.weight", prefix)),
        weights.get(&format!("{}.emb_layers.1.bias", prefix)),
    ) {
        let time_proj = time_emb
            .silu()?
            .matmul(&time_weight.transpose_dims(0, 1)?)?
            .add(time_bias)?
            .unsqueeze(2)?
            .unsqueeze(3)?;

        // Broadcast time projection to match h dimensions
        let h = h.add(&time_proj)?;

        // Second norm + conv
        let norm2_weight =
            weights.get(&format!("{}.out_layers.0.weight", prefix)).ok_or_else(|| {
                flame_core::Error::InvalidOperation("Missing norm2 weight".into())
            })?;
        let norm2_bias =
            weights.get(&format!("{}.out_layers.0.bias", prefix)).ok_or_else(|| {
                flame_core::Error::InvalidOperation("Missing norm2 bias".into())
            })?;

        let h = group_norm_32(&h, 32, norm2_weight, norm2_bias)?;
        let h = h.silu()?;

        let conv2_weight =
            weights.get(&format!("{}.out_layers.3.weight", prefix)).ok_or_else(|| {
                flame_core::Error::InvalidOperation("Missing conv2 weight".into())
            })?;
        let conv2_bias =
            weights.get(&format!("{}.out_layers.3.bias", prefix)).ok_or_else(|| {
                flame_core::Error::InvalidOperation("Missing conv2 bias".into())
            })?;

        let h = conv2d(&h, conv2_weight, conv2_bias, 1, 1)?;

        // Skip connection
        let skip = if x.shape().dims()[1] != h.shape().dims()[1] {
            println!(
                "    Channel change detected: {} -> {}",
                x.shape().dims()[1],
                h.shape().dims()[1]
            );
            // Use skip_connection if channels changed
            let skip_weight =
                weights.get(&format!("{}.skip_connection.weight", prefix)).ok_or_else(|| {
                    flame_core::Error::InvalidOperation(
                        "Missing skip connection weight".to_string(),
                    )
                })?;
            let skip_bias =
                weights.get(&format!("{}.skip_connection.bias", prefix)).ok_or_else(|| {
                    flame_core::Error::InvalidOperation(
                        "Missing skip connection bias".to_string(),
                    )
                })?;
            conv2d(x, skip_weight, skip_bias, 1, 1)?
        } else {
            x.clone()
        };

        h.add(&skip)
    } else {
        // No time embedding, just return h
        Ok(h)
    }
}

/// 2D convolution
fn conv2d(
    x: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    stride: usize,
    padding: usize,
) -> flame_core::Result<Tensor> {
    let x = x.conv2d(weight, Some(bias), stride, padding)?;
    Ok(x)
}

/// GroupNorm
fn group_norm_32(
    x: &Tensor,
    num_groups: usize,
    weight: &Tensor,
    bias: &Tensor,
) -> flame_core::Result<Tensor> {
    let eps = 1e-5;
    let dims = x.shape().dims();
    let (b, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);

    // Reshape for group norm
    let x_reshaped = x.reshape(&[b, num_groups, c / num_groups, h * w])?;

    // Calculate mean and variance
    let mean = x_reshaped.mean_dim(&[2], true)?.mean_dim(&[3], true)?;
    let var = x_reshaped.sub(&mean)?.square()?.mean_dim(&[2], true)?.mean_dim(&[3], true)?;

    // Normalize
    let x_norm = x_reshaped.sub(&mean)?.div(&var.add_scalar(eps as f32)?.sqrt()?)?;
    let x_norm = x_norm.reshape(&[b, c, h, w])?;

    // Apply scale and shift
    let weight = weight.reshape(&[1, c, 1, 1])?;
    let bias = bias.reshape(&[1, c, 1, 1])?;

    Ok(x_norm.mul(&weight)?.add(&bias)?)
}

/// Forward pass for SDXL with SD checkpoint format and Flash Attention
pub fn forward_sdxl_sd_format_flash(
    latents: &Tensor,
    timesteps: &Tensor,
    encoder_hidden_states: &Tensor,
    weights: &crate::loaders::WeightLoader,
    lora: &LoRACollection,
    use_flash_attention: bool,
) -> flame_core::Result<Tensor> {
    println!("\n=== SDXL SD Format Forward Pass (Flash Attention: {}) ===", use_flash_attention);

    // Time embeddings
    let device = latents.device();
    let t_emb = timestep_embedding(timesteps, 320, 10000.0, &Device::from(device.clone()))?;
    let t_emb_dim = 1280; // SDXL time embedding dimension

    // Time embedding MLP (SD format naming)
    let time_embedding = if let (Ok(fc1_weight), Ok(fc1_bias)) =
        (weights.get("time_embed.0.weight"), weights.get("time_embed.0.bias"))
    {
        // Convert time embedding to match weight dtype
        let t_emb = t_emb.to_dtype(fc1_weight.dtype())?;
        let t_emb = t_emb.matmul(&fc1_weight.transpose_dims(0, 1)?)?.add(fc1_bias)?;
        let t_emb = t_emb.silu()?;
        if let (Ok(fc2_weight), Ok(fc2_bias)) =
            (weights.get("time_embed.2.weight"), weights.get("time_embed.2.bias"))
        {
            t_emb.matmul(&fc2_weight.transpose_dims(0, 1)?)?.add(fc2_bias)?
        } else {
            println!("WARNING: time_embed weights not found, using raw time embedding");
            t_emb
        }
    } else {
        t_emb
    };

    let time_emb = time_embedding.reshape(&[timesteps.shape().dims()[0], t_emb_dim])?;
    println!("Time embedding shape: {:?}", time_emb.shape());

    // Initial convolution
    let conv_in_weight = weights.get("conv_in.weight")?;
    let conv_in_bias = weights.get("conv_in.bias")?;

    let mut h = conv2d(latents, conv_in_weight, conv_in_bias, 1, 1)?;
    println!("Initial conv output shape: {:?}", h.shape());

    // Down blocks
    let mut down_block_res_samples = Vec::new();
    let channels = [320, 640, 1280];

    for (i, &out_channels) in channels.iter().enumerate() {
        println!("\n--- Processing down block {} ---", i);

        // ResNet blocks
        for j in 0..2 {
            let resnet_prefix = format!("down_blocks.{}.resnets.{}", i, j);
            h = resnet_block(&h, &time_emb, &weights.weights, &resnet_prefix)?;
            down_block_res_samples.push(h.clone());
            println!("  ResNet {} output shape: {:?}", j, h.shape());
        }

        // Transformer blocks (if present)
        if i < 2 {
            // Only first two down blocks have transformers in SDXL
            let transformer_prefix = format!("down_blocks.{}.attentions.{}", i, 0);
            let num_transformer_blocks = if i == 0 { 2 } else { 10 };
            h = transformer_block(
                &h,
                Some(encoder_hidden_states),
                &weights.weights,
                lora,
                &transformer_prefix,
                num_transformer_blocks,
                use_flash_attention,
            )?;
            down_block_res_samples.push(h.clone());
        }

        // Downsampling (except for last block)
        if i < 2 {
            let downsample_prefix = format!("down_blocks.{}.downsamplers.0", i);
            let downsample_weight = weights.get(&format!("{}.conv.weight", downsample_prefix))?;
            let downsample_bias = weights.get(&format!("{}.conv.bias", downsample_prefix))?;
            h = conv2d(&h, downsample_weight, downsample_bias, 2, 1)?;
            println!("  Downsample output shape: {:?}", h.shape());
        }
    }

    // Middle block
    println!("\n--- Processing middle block ---");
    for j in 0..2 {
        let resnet_prefix = format!("mid_block.resnets.{}", j);
        h = resnet_block(&h, &time_emb, &weights.weights, &resnet_prefix)?;
    }

    h = transformer_block(
        &h,
        Some(encoder_hidden_states),
        &weights.weights,
        lora,
        "mid_block.attentions.0",
        10, // Middle block has 10 transformer blocks
        use_flash_attention,
    )?;

    // Up blocks
    for (i, &out_channels) in channels.iter().enumerate().rev() {
        println!("\n--- Processing up block {} ---", 2 - i);

        // ResNet blocks with skip connections
        for j in 0..3 {
            // Pop skip connection
            if !down_block_res_samples.is_empty() {
                let skip = down_block_res_samples.pop().unwrap();
                h = Tensor::cat(&[&h, &skip], 1)?;
                println!("  After concat with skip: {:?}", h.shape());
            }

            let resnet_prefix = format!("up_blocks.{}.resnets.{}", 2 - i, j);
            h = resnet_block(&h, &time_emb, &weights.weights, &resnet_prefix)?;
        }

        // Transformer blocks (if present)
        if (2 - i) > 0 {
            // Last two up blocks have transformers in SDXL
            let transformer_prefix = format!("up_blocks.{}.attentions.{}", 2 - i, 0);
            let num_transformer_blocks = if (2 - i) == 2 { 10 } else { 2 };
            h = transformer_block(
                &h,
                Some(encoder_hidden_states),
                &weights.weights,
                lora,
                &transformer_prefix,
                num_transformer_blocks,
                use_flash_attention,
            )?;
        }

        // Upsampling (except for last block)
        if i > 0 {
            let upsample_prefix = format!("up_blocks.{}.upsamplers.0", 2 - i);
            let upsample_weight = weights.get(&format!("{}.conv.weight", upsample_prefix))?;
            let upsample_bias = weights.get(&format!("{}.conv.bias", upsample_prefix))?;

            h = h.upsample_nearest2d(h.shape().dims()[2] * 2, h.shape().dims()[3] * 2)?;
            h = conv2d(&h, upsample_weight, upsample_bias, 1, 1)?;
            println!("  Upsample output shape: {:?}", h.shape());
        }
    }

    // Final layers
    let norm_weight = weights.get("conv_norm_out.weight")?;
    let norm_bias = weights.get("conv_norm_out.bias")?;

    h = group_norm_32(&h, 32, norm_weight, norm_bias)?;
    h = h.silu()?;

    let conv_out_weight = weights.get("conv_out.weight")?;
    let conv_out_bias = weights.get("conv_out.bias")?;

    let output = conv2d(&h, conv_out_weight, conv_out_bias, 1, 1)?;
    println!("\nFinal output shape: {:?}", output.shape());

    Ok(output)
}
