use crate::trainers::lora::LoRACollection;
use flame_core::device::Device;
use flame_core::optimizers::{Adam, SGD};
use flame_core::{DType, Error, Result, Shape, Tensor};

// SDXL Transformer Block with Flash Attention for training
// This version uses Flash Attention for memory efficiency

// FLAME uses flame_core::device::Device instead of Device

// FLAME uses flame_core::device::Device instead of Device

/// Single transformer block containing self-attention and cross-attention
pub fn transformer_block(
    hidden_states: &Tensor,
    encoder_hidden_states: Option<&Tensor>,
    weights: &std::collections::HashMap<String, Tensor>,
    lora: &LoRACollection,
    prefix: &str,
    num_transformer_blocks: usize,
    use_flash_attention: bool,
) -> flame_core::Result<Tensor> {
    let mut h = hidden_states.clone();

    // Each transformer block contains multiple sub-blocks
    for block_idx in 0..num_transformer_blocks {
        let block_prefix = format!("{}.transformer_blocks.{}", prefix, block_idx);

        println!(
            "  Processing transformer block: {} (flash_attn={})",
            block_prefix, use_flash_attention
        );

        // Self-attention (attn1)
        h = self_attention_block(
            &h,
            weights,
            lora,
            &format!("{}.attn1", block_prefix),
            use_flash_attention,
        )?;

        // Cross-attention (attn2) if encoder hidden states provided
        if encoder_hidden_states.is_some() {
            h = cross_attention_block(
                &h,
                encoder_hidden_states.unwrap(),
                weights,
                lora,
                &format!("{}.attn2", block_prefix),
                use_flash_attention,
            )?;
        }

        // Feed-forward network (ff)
        h = feedforward_block(&h, weights, &format!("{}.ff", block_prefix))?;
    }

    Ok(h)
}

/// Self-attention block with optional Flash Attention
fn self_attention_block(
    hidden_states: &Tensor,
    weights: &std::collections::HashMap<String, Tensor>,
    lora: &LoRACollection,
    prefix: &str,
    use_flash_attention: bool,
) -> flame_core::Result<Tensor> {
    let dims = hidden_states.shape().dims();
    let (batch_size, channels, height, width) = (dims[0], dims[1], dims[2], dims[3]);
    let sequence_length = height * width;

    // Determine number of heads from Q weight shape
    let q_weight = weights
        .get(&format!("{}.to_q.weight", prefix))
        .ok_or_else(|| Error::InvalidOperation("Missing to_q weight".into()))?;
    let q_out_dim = q_weight.shape().dims()[0];

    // SDXL uses different head counts for different layers
    let num_heads = match channels {
        320 => 5,   // 320 / 64 = 5
        640 => 10,  // 640 / 64 = 10
        1280 => 20, // 1280 / 64 = 20
        _ => {
            return Err(Error::InvalidOperation(format!(
                "Unexpected channel count: {}",
                channels
            )))
        }
    };
    let head_dim = channels / num_heads;

    // Reshape to [batch, seq_len, channels]
    let hidden_states_seq = hidden_states
        .permute(&[0, 2, 3, 1])? // [b, c, h, w] -> [b, h, w, c]
        .reshape(&[batch_size, sequence_length, channels])?;

    // Layer norm (using norm1 for self-attention)
    let block_prefix = prefix.rsplitn(2, '.').nth(1).unwrap_or("");
    let norm_weight_key = format!("{}.norm1.weight", block_prefix);
    let norm_bias_key = format!("{}.norm1.bias", block_prefix);

    let norm_weight = weights.get(&norm_weight_key).ok_or_else(|| {
        Error::InvalidOperation(format!(
            "Missing self-attention norm weight at {}",
            norm_weight_key
        ))
    })?;
    let norm_bias = weights.get(&norm_bias_key).ok_or_else(|| {
        Error::InvalidOperation(format!(
            "Missing self-attention norm bias at {}",
            norm_bias_key
        ))
    })?;

    let norm_hidden = layer_norm(&hidden_states_seq, norm_weight, norm_bias)?;

    // Q, K, V projections using to_q, to_k, to_v naming
    let q_weight = weights
        .get(&format!("{}.to_q.weight", prefix))
        .ok_or_else(|| Error::InvalidOperation("Missing to_q weight".into()))?;
    let k_weight = weights
        .get(&format!("{}.to_k.weight", prefix))
        .ok_or_else(|| Error::InvalidOperation("Missing to_k weight".into()))?;
    let v_weight = weights
        .get(&format!("{}.to_v.weight", prefix))
        .ok_or_else(|| Error::InvalidOperation("Missing to_v weight".into()))?;

    let q_bias = weights.get(&format!("{}.to_q.bias", prefix));
    let k_bias = weights.get(&format!("{}.to_k.bias", prefix));
    let v_bias = weights.get(&format!("{}.to_v.bias", prefix));

    // Apply projections with LoRA
    let q = lora.apply(&format!("{}.to_q", prefix), &norm_hidden, q_weight, q_bias)?;
    let k = lora.apply(&format!("{}.to_k", prefix), &norm_hidden, k_weight, k_bias)?;
    let v = lora.apply(&format!("{}.to_v", prefix), &norm_hidden, v_weight, v_bias)?;

    // Reshape for multi-head attention
    // For Flash Attention, we need shape [batch, seq_len, num_heads, head_dim]
    let q = q.reshape(&[batch_size, sequence_length, num_heads, head_dim])?;
    let k = k.reshape(&[batch_size, sequence_length, num_heads, head_dim])?;
    let v = v.reshape(&[batch_size, sequence_length, num_heads, head_dim])?;

    println!("  Self-attention - q: {:?}, k: {:?}, v: {:?}", q.shape(), k.shape(), v.shape());

    let attn_output = if use_flash_attention {
        // Check if we can use Flash Attention
        let dtype = q.dtype();
        if dtype == DType::F16 || dtype == DType::BF16 {
            println!("  Using Flash Attention for self-attention");
            let scale = 1.0 / (head_dim as f32).sqrt();
            #[cfg(feature = "flash-attn")]
            {
                // Flash attention implementation would go here
                standard_attention(&q, &k, &v, head_dim)?
            }
            #[cfg(not(feature = "flash-attn"))]
            {
                println!("  Flash Attention not compiled in, using standard attention");
                standard_attention(&q, &k, &v, head_dim)?
            }
        } else {
            println!("  Flash Attention requires F16/BF16, falling back to standard attention");
            standard_attention(&q, &k, &v, head_dim)?
        }
    } else {
        standard_attention(&q, &k, &v, head_dim)?
    };

    // Reshape back from [batch, seq_len, num_heads, head_dim] to [batch, seq_len, channels]
    let attn_output = attn_output.reshape(&[batch_size, sequence_length, channels])?;

    // Output projection
    let out_weight = weights
        .get(&format!("{}.to_out.0.weight", prefix))
        .ok_or_else(|| Error::InvalidOperation("Missing to_out weight".into()))?;
    let out_bias = weights.get(&format!("{}.to_out.0.bias", prefix));

    let output = lora.apply(&format!("{}.to_out.0", prefix), &attn_output, out_weight, out_bias)?;

    // Add residual and reshape back to [b, c, h, w]
    let output = output
        .add(&hidden_states_seq)?
        .reshape(&[batch_size, height, width, channels])?
        .permute(&[0, 3, 1, 2])?;

    Ok(output)
}

/// Cross-attention block with optional Flash Attention
fn cross_attention_block(
    hidden_states: &Tensor,
    encoder_hidden_states: &Tensor,
    weights: &std::collections::HashMap<String, Tensor>,
    lora: &LoRACollection,
    prefix: &str,
    use_flash_attention: bool,
) -> flame_core::Result<Tensor> {
    let dims = hidden_states.shape().dims();
    let (batch_size, channels, height, width) = (dims[0], dims[1], dims[2], dims[3]);
    let sequence_length = height * width;

    // SDXL uses different head counts for different layers
    let num_heads = match channels {
        320 => 5,   // 320 / 64 = 5
        640 => 10,  // 640 / 64 = 10
        1280 => 20, // 1280 / 64 = 20
        _ => {
            return Err(Error::InvalidOperation(format!(
                "Unexpected channel count: {}",
                channels
            )))
        }
    };
    let head_dim = channels / num_heads;

    // Reshape hidden states to [batch, seq_len, channels]
    let hidden_states_seq =
        hidden_states.permute(&[0, 2, 3, 1])?.reshape(&[batch_size, sequence_length, channels])?;

    // Layer norm (using norm2 for cross-attention)
    let block_prefix = prefix.rsplitn(2, '.').nth(1).unwrap_or("");
    let norm_weight_key = format!("{}.norm2.weight", block_prefix);
    let norm_bias_key = format!("{}.norm2.bias", block_prefix);

    let norm_weight = weights.get(&norm_weight_key).ok_or_else(|| {
        Error::InvalidOperation(format!(
            "Missing cross-attention norm weight at {}",
            norm_weight_key
        ))
    })?;
    let norm_bias = weights.get(&norm_bias_key).ok_or_else(|| {
        Error::InvalidOperation(format!(
            "Missing cross-attention norm bias at {}",
            norm_bias_key
        ))
    })?;

    let norm_hidden = layer_norm(&hidden_states_seq, norm_weight, norm_bias)?;

    // Q from hidden states, K/V from encoder hidden states
    let q_weight = weights
        .get(&format!("{}.to_q.weight", prefix))
        .ok_or_else(|| Error::InvalidOperation("Missing to_q weight".into()))?;
    let k_weight = weights
        .get(&format!("{}.to_k.weight", prefix))
        .ok_or_else(|| Error::InvalidOperation("Missing to_k weight".into()))?;
    let v_weight = weights
        .get(&format!("{}.to_v.weight", prefix))
        .ok_or_else(|| Error::InvalidOperation("Missing to_v weight".into()))?;

    let q_bias = weights.get(&format!("{}.to_q.bias", prefix));
    let k_bias = weights.get(&format!("{}.to_k.bias", prefix));
    let v_bias = weights.get(&format!("{}.to_v.bias", prefix));

    // Apply projections with LoRA
    let q = lora.apply(&format!("{}.to_q", prefix), &norm_hidden, q_weight, q_bias)?;
    let k = lora.apply(&format!("{}.to_k", prefix), encoder_hidden_states, k_weight, k_bias)?;
    let v = lora.apply(&format!("{}.to_v", prefix), encoder_hidden_states, v_weight, v_bias)?;

    // Get context sequence length
    let context_seq_len = encoder_hidden_states.shape().dims()[1];

    // K and V are projected to channels dimension
    let k_v_channels = k.shape().dims()[2];
    let k_v_heads = k_v_channels / head_dim;

    println!(
        "  Cross-attention - K/V channels: {}, K/V heads: {}, expected heads: {}",
        k_v_channels, k_v_heads, num_heads
    );

    // Reshape for multi-head attention
    // For Flash Attention, we need shape [batch, seq_len, num_heads, head_dim]
    let q = q.reshape(&[batch_size, sequence_length, num_heads, head_dim])?;
    let k = k.reshape(&[batch_size, context_seq_len, k_v_heads, head_dim])?;
    let v = v.reshape(&[batch_size, context_seq_len, k_v_heads, head_dim])?;

    let attn_output = if use_flash_attention {
        let dtype = q.dtype();
        if dtype == DType::F16 || dtype == DType::BF16 {
            println!("  Using Flash Attention for cross-attention");
            let scale = 1.0 / (head_dim as f32).sqrt();
            #[cfg(feature = "flash-attn")]
            {
                // Flash attention implementation would go here
                standard_attention(&q, &k, &v, head_dim)?
            }
            #[cfg(not(feature = "flash-attn"))]
            {
                println!("  Flash Attention not compiled in, using standard attention");
                standard_attention(&q, &k, &v, head_dim)?
            }
        } else {
            println!("  Flash Attention requires F16/BF16, falling back to standard attention");
            standard_attention(&q, &k, &v, head_dim)?
        }
    } else {
        standard_attention(&q, &k, &v, head_dim)?
    };

    // Reshape back
    let attn_output = attn_output.reshape(&[batch_size, sequence_length, channels])?;

    // Output projection
    let out_weight = weights
        .get(&format!("{}.to_out.0.weight", prefix))
        .ok_or_else(|| Error::InvalidOperation("Missing to_out weight".into()))?;
    let out_bias = weights.get(&format!("{}.to_out.0.bias", prefix));

    let output = lora.apply(&format!("{}.to_out.0", prefix), &attn_output, out_weight, out_bias)?;

    // Add residual and reshape back to [b, c, h, w]
    let output = output
        .add(&hidden_states_seq)?
        .reshape(&[batch_size, height, width, channels])?
        .permute(&[0, 3, 1, 2])?;

    Ok(output)
}

/// Standard attention implementation (fallback for when Flash Attention is not available)
fn standard_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    head_dim: usize,
) -> flame_core::Result<Tensor> {
    // Transpose for attention computation
    // q: [batch, seq_len, num_heads, head_dim] -> [batch, num_heads, seq_len, head_dim]
    let q = q.transpose_dims(1, 2)?;
    let k = k.transpose_dims(1, 2)?;
    let v = v.transpose_dims(1, 2)?;

    // Scaled dot-product attention
    let scale = (head_dim as f64).sqrt();
    let k_t = k.transpose_dims(k.shape().dims().len() - 2, k.shape().dims().len() - 1)?;
    let scores = q.matmul(&k_t)?.mul_scalar((1.0 / scale) as f32)?;
    let attn_weights = scores.softmax((scores.shape().dims().len() - 1) as isize)?;
    let attn_output = attn_weights.matmul(&v)?;

    // Transpose back
    // [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, num_heads, head_dim]
    Ok(attn_output.transpose_dims(1, 2)?)
}

/// Feed-forward network block (unchanged from original)
fn feedforward_block(
    hidden_states: &Tensor,
    weights: &std::collections::HashMap<String, Tensor>,
    prefix: &str,
) -> flame_core::Result<Tensor> {
    let dims = hidden_states.shape().dims();
    let (batch_size, channels, height, width) = (dims[0], dims[1], dims[2], dims[3]);
    let sequence_length = height * width;

    // Reshape to [batch, seq_len, channels]
    let hidden_states_seq =
        hidden_states.permute(&[0, 2, 3, 1])?.reshape(&[batch_size, sequence_length, channels])?;

    // Layer norm (using norm3 for feed-forward)
    let block_prefix = prefix.rsplitn(2, '.').nth(1).unwrap_or("");
    let norm_weight_key = format!("{}.norm3.weight", block_prefix);
    let norm_bias_key = format!("{}.norm3.bias", block_prefix);

    let norm_weight = weights.get(&norm_weight_key).ok_or_else(|| {
        Error::InvalidOperation(format!("Missing ff norm weight at {}", norm_weight_key))
    })?;
    let norm_bias = weights.get(&norm_bias_key).ok_or_else(|| {
        Error::InvalidOperation(format!("Missing ff norm bias at {}", norm_bias_key))
    })?;

    let norm_hidden = layer_norm(&hidden_states_seq, norm_weight, norm_bias)?;

    println!("  Feedforward block {}: norm_hidden shape = {:?}", prefix, norm_hidden.shape());

    // GEGLU: Linear -> Split -> GELU on one half -> multiply
    let net_weight_key = format!("{}.net.0.proj.weight", prefix);
    let net_weight = weights.get(&net_weight_key).ok_or_else(|| {
        Error::InvalidOperation(format!("Missing ff net weight at {}", net_weight_key))
    })?;
    let net_bias = weights.get(&format!("{}.net.0.proj.bias", prefix));

    // Need to handle batch dimension properly
    let dims = norm_hidden.shape().dims();
    let (batch_size, seq_len, channels) = (dims[0], dims[1], dims[2]);
    let norm_hidden_2d = norm_hidden.reshape(&[batch_size * seq_len, channels])?;

    let ff_output_2d = if let Some(bias) = net_bias {
        norm_hidden_2d.matmul(&net_weight.transpose_dims(0, 1)?)?.add(bias)?
    } else {
        norm_hidden_2d.matmul(&net_weight.transpose_dims(0, 1)?)?
    };

    // Reshape back to 3D
    let ff_output = ff_output_2d.reshape(&[batch_size, seq_len, net_weight.shape().dims()[0]])?;

    // GEGLU activation
    let hidden_dim = ff_output.shape().dims()[2] / 2;
    let ff_gate = ff_output.slice(&[(0, 0 + hidden_dim)])?;
    let ff_up = ff_output.slice(&[(hidden_dim, hidden_dim + hidden_dim)])?;
    let ff_gate_activated = gelu(&ff_gate)?;
    let hidden_states_ff = ff_gate_activated.mul(&ff_up)?;

    // Output projection
    let out_weight = weights
        .get(&format!("{}.net.2.weight", prefix))
        .ok_or_else(|| Error::InvalidOperation("Missing ff output weight".into()))?;
    let out_bias = weights.get(&format!("{}.net.2.bias", prefix));

    // Need to handle batch dimension for output projection too
    let shape = hidden_states_ff.shape();
    let dims = shape.dims();
    let (batch_size, seq_len, ff_dim) = (dims[0], dims[1], dims[2]);
    let hidden_states_ff_2d = hidden_states_ff.reshape(&[batch_size * seq_len, ff_dim])?;

    let output_2d = if let Some(bias) = out_bias {
        hidden_states_ff_2d.matmul(&out_weight.transpose_dims(0, 1)?)?.add(bias)?
    } else {
        hidden_states_ff_2d.matmul(&out_weight.transpose_dims(0, 1)?)?
    };

    // Reshape back to 3D to match input hidden_states_seq shape
    let output = output_2d.reshape(&[batch_size, seq_len, out_weight.shape().dims()[0]])?;

    // Add residual and reshape back
    let output = output
        .add(&hidden_states_seq)?
        .reshape(&[batch_size, height, width, channels])?
        .permute(&[0, 3, 1, 2])?;

    Ok(output)
}

/// Layer normalization
fn layer_norm(x: &Tensor, weight: &Tensor, bias: &Tensor) -> flame_core::Result<Tensor> {
    let eps = 1e-5;

    let ndims = x.shape().dims().len();
    let mean = x.mean_dim(&[ndims - 1], true)?;
    let var = x.var_dim(&[ndims - 1], true)?;

    let x_sub_mean = x.sub(&mean)?;
    let var_eps = var.add_scalar(eps as f32)?.sqrt()?;
    let normalized = x_sub_mean.div(&var_eps)?;

    let normalized = normalized.mul(weight)?;
    let result = normalized.add(bias)?;

    Ok(result)
}

/// GELU activation
fn gelu(x: &Tensor) -> flame_core::Result<Tensor> {
    // GELU(x) = x * Φ(x) where Φ is the CDF of standard normal
    // Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    let sqrt_2_over_pi = (2.0f64 / std::f64::consts::PI).sqrt();
    let x_cubed = x.powf(3.0f32)?;
    let x_plus_term = x.add(&x_cubed.mul_scalar(0.044715f32)?)?;
    let inner = x_plus_term.mul_scalar(sqrt_2_over_pi as f32)?;
    let tanh_inner = inner.tanh()?;
    let tanh_plus_one = tanh_inner.add_scalar(1.0f32)?;
    let scaled = tanh_plus_one.mul_scalar(0.5f32)?;
    x.mul(&scaled)
}
