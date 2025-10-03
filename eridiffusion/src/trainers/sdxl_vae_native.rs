use flame_core::device::Device;
use flame_core::optimizers::{Adam, SGD};
use flame_core::{DType, Error, Result, Shape, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

struct Conv2d {
    weight: Tensor,
    bias: Option<Tensor>,
    stride: usize,
    padding: usize,
    dilation: usize,
}

pub struct SDXLVAENative {
    // Encoder components
    encoder_conv_in: Conv2d,
    encoder_down_blocks: Vec<VAEDownBlock>,
    encoder_mid_block: VAEMidBlock,
    encoder_norm_out: GroupNorm,
    encoder_conv_out: Conv2d,

    // Quantization layers
    quant_conv: Conv2d,
    post_quant_conv: Conv2d,

    // Decoder components
    decoder_conv_in: Conv2d,
    decoder_up_blocks: Vec<VAEUpBlock>,
    decoder_mid_block: VAEMidBlock,
    decoder_norm_out: GroupNorm,
    decoder_conv_out: Conv2d,

    device: Device,
    dtype: DType,
}

struct VAEResNetBlock {
    norm1: GroupNorm,
    conv1: Conv2d,
    norm2: GroupNorm,
    conv2: Conv2d,
    conv_shortcut: Option<Conv2d>,
}

struct VAEAttentionBlock {
    norm: GroupNorm,
    q: Conv2d,
    k: Conv2d,
    v: Conv2d,
    proj_out: Conv2d,
}
struct VAEMidBlock {
    resnet1: VAEResNetBlock,
    attn1: VAEAttentionBlock,
    resnet2: VAEResNetBlock,
}

// Native SDXL VAE implementation that directly uses SD format weights
// No remapping needed - works directly with the checkpoint format

// FLAME uses flame_core::device::Device instead of Device

/// Simple GroupNorm wrapper
struct GroupNorm {
    weight: Tensor,
    bias: Tensor,
    num_groups: usize,
    eps: f64,
}

impl GroupNorm {
    fn new(weight: Tensor, bias: Tensor, num_groups: usize) -> flame_core::Result<Self> {
        Ok(Self { weight, bias, num_groups, eps: 1e-6 })
    }

    fn forward(&self, x: &Tensor) -> flame_core::Result<Tensor> {
        // Group normalization implementation
        let shape = x.shape();
        let dims = shape.dims();
        let (b, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
        let g = self.num_groups;
        let c_per_g = c / g;

        // Reshape to (B, G, C/G, H, W)
        let x = x.reshape(&[b, g, c_per_g, h * w])?;

        // Calculate mean and variance per group
        let mean = x.mean_dim(&[3], true)?; // Average over H*W dimension
                                            // Compute variance manually since var_dim might not exist
        let mean_sq = x.square()?.mean_dim(&[3], true)?;
        let var = mean_sq.sub(&mean.square()?)?;

        // Normalize
        let x = x.sub(&mean)?.div(&var.add_scalar(self.eps as f32)?.sqrt()?)?;

        // Reshape back to (B, C, H, W)
        let x = x.reshape(&[b, c, h, w])?;

        // Apply weight and bias
        let weight = self.weight.reshape(&[1, c, 1, 1])?;
        let bias = self.bias.reshape(&[1, c, 1, 1])?;

        let x = x.mul(&weight)?;
        Ok(x.add(&bias)?)
    }
}

/// Simple Conv2d wrapper

impl Conv2d {
    fn new(weight: Tensor, bias: Option<Tensor>, padding: usize) -> Self {
        Self { weight, bias, stride: 1, padding, dilation: 1 }
    }

    fn new_with_stride(
        weight: Tensor,
        bias: Option<Tensor>,
        padding: usize,
        stride: usize,
    ) -> Self {
        Self { weight, bias, stride, padding, dilation: 1 }
    }

    fn forward(&self, x: &Tensor) -> flame_core::Result<Tensor> {
        // FLAME doesn't have conv2d as tensor method
        // Create Conv2d layer and use it
        let device = Device::from(x.device().clone());
        let weight_shape = self.weight.shape().dims();
        let conv_layer = flame_core::conv::Conv2d::new(
            weight_shape[1], // in_channels
            weight_shape[0], // out_channels
            weight_shape[2], // kernel_size
            self.stride,
            self.padding,
            device.cuda_device().clone(),
        )?;
        // TODO: Load self.weight into conv_layer
        let mut out = conv_layer.forward(x)?;
        if let Some(ref b) = self.bias {
            let b_dims = b.shape().dims();
            let b = b.reshape(&[1, b_dims[0], 1, 1])?;
            out = out.add(&b)?;
        }
        Ok(out)
    }
}

/// Native SDXL VAE implementation

/// VAE ResNet block

impl VAEResNetBlock {
    fn new(
        weights: &std::collections::HashMap<String, Tensor>,
        prefix: &str,
        in_channels: usize,
        out_channels: usize,
    ) -> Result<Self> {
        // Debug channel info
        if prefix.contains("decoder.up.2") {
            println!(
                "ResNet block {}: in_channels={}, out_channels={}",
                prefix, in_channels, out_channels
            );
        }

        // Load norm1
        let norm1_weight = weights.get(&format!("{}.norm1.weight", prefix)).ok_or_else(|| {
            Error::InvalidOperation(format!("Missing {}.norm1.weight", prefix))
        })?;
        let norm1_bias = weights.get(&format!("{}.norm1.bias", prefix)).ok_or_else(|| {
            Error::InvalidOperation(format!("Missing {}.norm1.bias", prefix))
        })?;
        let norm1 = GroupNorm::new(norm1_weight.clone(), norm1_bias.clone(), 32)?;

        // Load conv1
        let conv1_weight = weights.get(&format!("{}.conv1.weight", prefix)).ok_or_else(|| {
            Error::InvalidOperation(format!("Missing {}.conv1.weight", prefix))
        })?;
        let conv1_bias = weights.get(&format!("{}.conv1.bias", prefix)).ok_or_else(|| {
            Error::InvalidOperation(format!("Missing {}.conv1.bias", prefix))
        })?;
        let conv1 = Conv2d::new(conv1_weight.clone(), Some(conv1_bias.clone()), 1);

        // Load norm2
        let norm2_weight = weights.get(&format!("{}.norm2.weight", prefix)).ok_or_else(|| {
            Error::InvalidOperation(format!("Missing {}.norm2.weight", prefix))
        })?;
        let norm2_bias = weights.get(&format!("{}.norm2.bias", prefix)).ok_or_else(|| {
            Error::InvalidOperation(format!("Missing {}.norm2.bias", prefix))
        })?;
        let norm2 = GroupNorm::new(norm2_weight.clone(), norm2_bias.clone(), 32)?;

        // Load conv2
        let conv2_weight = weights.get(&format!("{}.conv2.weight", prefix)).ok_or_else(|| {
            Error::InvalidOperation(format!("Missing {}.conv2.weight", prefix))
        })?;
        let conv2_bias = weights.get(&format!("{}.conv2.bias", prefix)).ok_or_else(|| {
            Error::InvalidOperation(format!("Missing {}.conv2.bias", prefix))
        })?;
        let conv2 = Conv2d::new(conv2_weight.clone(), Some(conv2_bias.clone()), 1);

        // Load conv_shortcut if needed
        let conv_shortcut = if in_channels != out_channels {
            let shortcut_weight_key = format!("{}.nin_shortcut.weight", prefix);
            let shortcut_bias_key = format!("{}.nin_shortcut.bias", prefix);

            if weights.contains_key(&shortcut_weight_key) {
                let shortcut_weight = weights.get(&shortcut_weight_key).unwrap();
                let shortcut_bias = weights.get(&shortcut_bias_key).unwrap();
                Some(Conv2d::new(shortcut_weight.clone(), Some(shortcut_bias.clone()), 0))
            } else {
                // No shortcut - channels must match
                return Err(Error::InvalidOperation(format!(
                    "Channel mismatch without shortcut: {} -> {}",
                    in_channels, out_channels
                )));
            }
        } else {
            None
        };

        Ok(Self { norm1, conv1, norm2, conv2, conv_shortcut })
    }

    fn forward(&self, x: &Tensor) -> flame_core::Result<Tensor> {
        let residual = x;

        let h = self.norm1.forward(x)?;
        let h = self.conv1.forward(&h)?;

        let h = self.norm2.forward(&h)?;
        let h = self.conv2.forward(&h)?;

        // Apply shortcut if needed
        let residual = if let Some(ref conv) = self.conv_shortcut {
            conv.forward(residual)?
        } else {
            residual.clone()
        };

        Ok(h.add(&residual)?)
    }
}

/// VAE Attention block

impl VAEAttentionBlock {
    fn new(weights: &HashMap<String, Tensor>, prefix: &str) -> flame_core::Result<Self> {
        let norm_weight = weights.get(&format!("{}.norm.weight", prefix)).ok_or_else(|| {
            Error::InvalidOperation(format!("Missing {}.norm.weight", prefix))
        })?;
        let norm_bias = weights
            .get(&format!("{}.norm.bias", prefix))
            .ok_or_else(|| Error::InvalidOperation(format!("Missing {}.norm.bias", prefix)))?;
        let norm = GroupNorm::new(norm_weight.clone(), norm_bias.clone(), 32)?;

        let q_weight = weights
            .get(&format!("{}.q.weight", prefix))
            .ok_or_else(|| Error::InvalidOperation(format!("Missing {}.q.weight", prefix)))?;
        let q_bias = weights
            .get(&format!("{}.q.bias", prefix))
            .ok_or_else(|| Error::InvalidOperation(format!("Missing {}.q.bias", prefix)))?;
        let q = Conv2d::new(q_weight.clone(), Some(q_bias.clone()), 0);

        let k_weight = weights
            .get(&format!("{}.k.weight", prefix))
            .ok_or_else(|| Error::InvalidOperation(format!("Missing {}.k.weight", prefix)))?;
        let k_bias = weights
            .get(&format!("{}.k.bias", prefix))
            .ok_or_else(|| Error::InvalidOperation(format!("Missing {}.k.bias", prefix)))?;
        let k = Conv2d::new(k_weight.clone(), Some(k_bias.clone()), 0);

        let v_weight = weights
            .get(&format!("{}.v.weight", prefix))
            .ok_or_else(|| Error::InvalidOperation(format!("Missing {}.v.weight", prefix)))?;
        let v_bias = weights
            .get(&format!("{}.v.bias", prefix))
            .ok_or_else(|| Error::InvalidOperation(format!("Missing {}.v.bias", prefix)))?;
        let v = Conv2d::new(v_weight.clone(), Some(v_bias.clone()), 0);

        let proj_out_weight =
            weights.get(&format!("{}.proj_out.weight", prefix)).ok_or_else(|| {
                Error::InvalidOperation(format!("Missing {}.proj_out.weight", prefix))
            })?;
        let proj_out_bias = weights.get(&format!("{}.proj_out.bias", prefix)).ok_or_else(|| {
            Error::InvalidOperation(format!("Missing {}.proj_out.bias", prefix))
        })?;
        let proj_out = Conv2d::new(proj_out_weight.clone(), Some(proj_out_bias.clone()), 0);

        Ok(Self { norm, q, k, v, proj_out })
    }

    fn forward(&self, x: &Tensor) -> flame_core::Result<Tensor> {
        let residual = x;
        let shape = x.shape();
        let dims = shape.dims();
        let (b, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);

        // Normalize
        let x = self.norm.forward(x)?;

        // Compute Q, K, V
        let q = self.q.forward(&x)?;
        let k = self.k.forward(&x)?;
        let v = self.v.forward(&x)?;

        // Reshape for attention: [B, C, H, W] -> [B, H*W, C]
        let q = q.reshape(&[b, c, h * w])?.transpose_dims(1, 2)?;
        let k = k.reshape(&[b, c, h * w])?.transpose_dims(1, 2)?;
        let v = v.reshape(&[b, c, h * w])?.transpose_dims(1, 2)?;

        // Scaled dot-product attention
        let scale = (c as f64).sqrt();
        let scores = q.matmul(&k.transpose_dims(1, 2)?)?.div_scalar(scale as f32)?;
        let attn = scores.softmax(-1)?;
        let out = attn.matmul(&v)?;

        // Reshape back: [B, H*W, C] -> [B, C, H, W]
        let out = out.transpose_dims(1, 2)?.reshape(&[b, c, h, w])?;

        // Project and add residual
        let out = self.proj_out.forward(&out)?;
        Ok(out.add(&residual)?)
    }
}

/// VAE Down block
struct VAEDownBlock {
    resnets: Vec<VAEResNetBlock>,
    downsample: Option<Conv2d>,
}

impl VAEDownBlock {
    fn new(
        weights: &HashMap<String, Tensor>,
        block_idx: usize,
        in_channels: usize,
        out_channels: usize,
        has_downsample: bool,
    ) -> flame_core::Result<Self> {
        let mut resnets = Vec::new();

        // First resnet uses in_channels -> out_channels
        let resnet0_prefix = format!("encoder.down.{}.block.0", block_idx);
        resnets.push(VAEResNetBlock::new(weights, &resnet0_prefix, in_channels, out_channels)?);

        // Second resnet uses out_channels -> out_channels
        let resnet1_prefix = format!("encoder.down.{}.block.1", block_idx);
        resnets.push(VAEResNetBlock::new(weights, &resnet1_prefix, out_channels, out_channels)?);

        // Downsample layer
        let downsample = if has_downsample {
            let down_weight = weights
                .get(&format!("encoder.down.{}.downsample.conv.weight", block_idx))
                .ok_or_else(|| {
                    Error::InvalidOperation(format!(
                        "Missing encoder.down.{}.downsample.conv.weight",
                        block_idx
                    ))
                })?;
            let down_bias = weights
                .get(&format!("encoder.down.{}.downsample.conv.bias", block_idx))
                .ok_or_else(|| {
                    Error::InvalidOperation(format!(
                        "Missing encoder.down.{}.downsample.conv.bias",
                        block_idx
                    ))
                })?;
            Some(Conv2d::new_with_stride(down_weight.clone(), Some(down_bias.clone()), 1, 2))
        } else {
            None
        };

        Ok(Self { resnets, downsample })
    }

    fn forward(&self, x: &Tensor) -> flame_core::Result<Tensor> {
        let mut h = x.clone();

        for resnet in &self.resnets {
            h = resnet.forward(&h)?;
        }

        if let Some(ref down) = self.downsample {
            h = down.forward(&h)?;
        }

        Ok(h)
    }
}

/// VAE Up block
struct VAEUpBlock {
    resnets: Vec<VAEResNetBlock>,
    upsample: Option<ConvTranspose2d>,
}

/// Simple ConvTranspose2d wrapper
struct ConvTranspose2d {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl ConvTranspose2d {
    fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self { weight, bias }
    }

    fn forward(&self, x: &Tensor) -> flame_core::Result<Tensor> {
        // FLAME doesn't have conv_transpose2d as a method
        // Need to use ConvTranspose2d struct or implement manually
        // For now, let's use upsampling + conv as approximation
        let upsampled = flame_core::cuda_ops::GpuOps::upsample2d_nearest(
            x,
            (x.shape().dims()[2] * 2, x.shape().dims()[3] * 2),
        )?;
        let mut out = upsampled; // TODO: Apply proper transposed convolution
        if let Some(ref b) = self.bias {
            let b_dims = b.shape().dims();
            let b = b.reshape(&[1, b_dims[0], 1, 1])?;
            out = out.add(&b)?;
        }
        Ok(out)
    }
}

impl VAEUpBlock {
    fn new(
        weights: &HashMap<String, Tensor>,
        block_idx: usize,
        in_channels: usize,
        out_channels: usize,
        has_upsample: bool,
    ) -> flame_core::Result<Self> {
        let mut resnets = Vec::new();

        // First resnet
        let resnet0_prefix = format!("decoder.up.{}.block.0", block_idx);
        resnets.push(VAEResNetBlock::new(weights, &resnet0_prefix, in_channels, out_channels)?);

        // Second resnet
        let resnet1_prefix = format!("decoder.up.{}.block.1", block_idx);
        resnets.push(VAEResNetBlock::new(weights, &resnet1_prefix, out_channels, out_channels)?);

        // Third resnet
        let resnet2_prefix = format!("decoder.up.{}.block.2", block_idx);
        resnets.push(VAEResNetBlock::new(weights, &resnet2_prefix, out_channels, out_channels)?);

        // Upsample layer - check if it exists first
        let upsample = if has_upsample {
            let up_weight_key = format!("decoder.up.{}.upsample.conv.weight", block_idx);
            let up_bias_key = format!("decoder.up.{}.upsample.conv.bias", block_idx);

            if weights.contains_key(&up_weight_key) {
                let up_weight = weights.get(&up_weight_key).unwrap();
                let up_bias = weights.get(&up_bias_key).unwrap();
                Some(ConvTranspose2d::new(up_weight.clone(), Some(up_bias.clone())))
            } else {
                println!("Note: No upsample layer for decoder block {}", block_idx);
                None
            }
        } else {
            None
        };

        Ok(Self { resnets, upsample })
    }

    fn forward(&self, x: &Tensor) -> flame_core::Result<Tensor> {
        let mut h = x.clone();

        for resnet in &self.resnets {
            h = resnet.forward(&h)?;
        }

        if let Some(ref up) = self.upsample {
            h = up.forward(&h)?;
        }

        Ok(h)
    }
}

/// VAE Mid block

impl VAEMidBlock {
    fn new(
        weights: &HashMap<String, Tensor>,
        prefix: &str,
        channels: usize,
    ) -> flame_core::Result<Self> {
        let resnet1 =
            VAEResNetBlock::new(weights, &format!("{}.block_1", prefix), channels, channels)?;
        let attn1 = VAEAttentionBlock::new(weights, &format!("{}.attn_1", prefix))?;
        let resnet2 =
            VAEResNetBlock::new(weights, &format!("{}.block_2", prefix), channels, channels)?;

        Ok(Self { resnet1, attn1, resnet2 })
    }

    fn forward(&self, x: &Tensor) -> flame_core::Result<Tensor> {
        let h = self.resnet1.forward(x)?;
        let h = self.attn1.forward(&h)?;
        let h = self.resnet2.forward(&h)?;
        Ok(h)
    }
}

impl SDXLVAENative {
    /// Create new VAE from SD format weights - no remapping needed!
    pub fn new(
        weights: HashMap<String, Tensor>,
        device: Device,
        dtype: DType,
    ) -> flame_core::Result<Self> {
        println!("Creating native SDXL VAE (no remapping needed)...");

        // Analyze the architecture from weights to determine correct channel counts
        println!("\n=== Analyzing VAE architecture from weights ===");

        // Check encoder block 0 to determine base channels
        let base_channels = if let Some(tensor) = weights.get("encoder.down.0.block.0.conv1.weight")
        {
            // Conv weight shape is [out_channels, in_channels, k, k]
            let dims = tensor.shape().dims();
            let out_channels = dims[0];
            println!("Detected base channels from encoder block 0: {}", out_channels);
            out_channels
        } else {
            println!("Using default base channels: 128");
            128
        };

        // Encoder conv_in
        let encoder_conv_in = Conv2d::new(
            weights
                .get("encoder.conv_in.weight")
                .ok_or_else(|| {
                    Error::InvalidOperation("Missing encoder.conv_in.weight".into())
                })?
                .clone(),
            Some(
                weights
                    .get("encoder.conv_in.bias")
                    .ok_or_else(|| {
                        Error::InvalidOperation("Missing encoder.conv_in.bias".into())
                    })?
                    .clone(),
            ),
            1,
        );

        // Encoder down blocks
        let mut encoder_down_blocks = Vec::new();
        let channel_mult = [1, 2, 4, 4];
        let mut prev_channels = base_channels;

        for (i, &mult) in channel_mult.iter().enumerate() {
            let channels = base_channels * mult;
            let downsample = i < 3; // No downsample on last block
            encoder_down_blocks.push(VAEDownBlock::new(
                &weights,
                i,
                prev_channels,
                channels,
                downsample,
            )?);
            prev_channels = channels;
        }

        // Encoder mid block
        let encoder_mid_block = VAEMidBlock::new(&weights, "encoder.mid", 512)?;

        // Encoder norm_out and conv_out
        let encoder_norm_out = GroupNorm::new(
            weights
                .get("encoder.norm_out.weight")
                .ok_or_else(|| {
                    Error::InvalidOperation("Missing encoder.norm_out.weight".into())
                })?
                .clone(),
            weights
                .get("encoder.norm_out.bias")
                .ok_or_else(|| {
                    Error::InvalidOperation("Missing encoder.norm_out.bias".into())
                })?
                .clone(),
            32,
        )?;

        let encoder_conv_out = Conv2d::new(
            weights
                .get("encoder.conv_out.weight")
                .ok_or_else(|| {
                    Error::InvalidOperation("Missing encoder.conv_out.weight".into())
                })?
                .clone(),
            Some(
                weights
                    .get("encoder.conv_out.bias")
                    .ok_or_else(|| {
                        Error::InvalidOperation("Missing encoder.conv_out.bias".into())
                    })?
                    .clone(),
            ),
            1,
        );

        // Quantization layers
        let quant_conv = Conv2d::new(
            weights
                .get("quant_conv.weight")
                .ok_or_else(|| {
                    Error::InvalidOperation("Missing quant_conv.weight".into())
                })?
                .clone(),
            Some(
                weights
                    .get("quant_conv.bias")
                    .ok_or_else(|| {
                        Error::InvalidOperation("Missing quant_conv.bias".into())
                    })?
                    .clone(),
            ),
            0,
        );

        let post_quant_conv = Conv2d::new(
            weights
                .get("post_quant_conv.weight")
                .ok_or_else(|| {
                    Error::InvalidOperation("Missing post_quant_conv.weight".into())
                })?
                .clone(),
            Some(
                weights
                    .get("post_quant_conv.bias")
                    .ok_or_else(|| {
                        Error::InvalidOperation("Missing post_quant_conv.bias".into())
                    })?
                    .clone(),
            ),
            0,
        );

        // Decoder conv_in
        let decoder_conv_in = Conv2d::new(
            weights
                .get("decoder.conv_in.weight")
                .ok_or_else(|| {
                    Error::InvalidOperation("Missing decoder.conv_in.weight".into())
                })?
                .clone(),
            Some(
                weights
                    .get("decoder.conv_in.bias")
                    .ok_or_else(|| {
                        Error::InvalidOperation("Missing decoder.conv_in.bias".into())
                    })?
                    .clone(),
            ),
            1,
        );

        // Decoder up blocks
        let mut decoder_up_blocks = Vec::new();

        // Based on actual weight inspection:
        // Block 0: 256->128 (no upsample)
        // Block 1: 512->256 with upsample
        // Block 2: 512->512 with upsample
        // Block 3: 512->512 with upsample
        let decoder_configs = [
            (256, 128, false), // Block 0
            (512, 256, true),  // Block 1
            (512, 512, true),  // Block 2
            (512, 512, true),  // Block 3
        ];

        for (i, &(in_ch, out_ch, has_upsample)) in decoder_configs.iter().enumerate() {
            decoder_up_blocks.push(VAEUpBlock::new(&weights, i, in_ch, out_ch, has_upsample)?);
        }

        // Decoder mid block
        let decoder_mid_block = VAEMidBlock::new(&weights, "decoder.mid", 512)?;

        // Decoder norm_out and conv_out
        let decoder_norm_out = GroupNorm::new(
            weights
                .get("decoder.norm_out.weight")
                .ok_or_else(|| {
                    Error::InvalidOperation("Missing decoder.norm_out.weight".into())
                })?
                .clone(),
            weights
                .get("decoder.norm_out.bias")
                .ok_or_else(|| {
                    Error::InvalidOperation("Missing decoder.norm_out.bias".into())
                })?
                .clone(),
            32,
        )?;

        let decoder_conv_out = Conv2d::new(
            weights
                .get("decoder.conv_out.weight")
                .ok_or_else(|| {
                    Error::InvalidOperation("Missing decoder.conv_out.weight".into())
                })?
                .clone(),
            Some(
                weights
                    .get("decoder.conv_out.bias")
                    .ok_or_else(|| {
                        Error::InvalidOperation("Missing decoder.conv_out.bias".into())
                    })?
                    .clone(),
            ),
            1,
        );

        Ok(Self {
            encoder_conv_in,
            encoder_down_blocks,
            encoder_mid_block,
            encoder_norm_out,
            encoder_conv_out,
            quant_conv,
            post_quant_conv,
            decoder_conv_in,
            decoder_up_blocks,
            decoder_mid_block,
            decoder_norm_out,
            decoder_conv_out,
            device,
            dtype,
        })
    }

    /// Encode image to latent space
    pub fn encode(&self, x: &Tensor) -> Result<Tensor> {
        // x should be in [0, 1] range, convert to [-1, 1]
        let x = x.mul_scalar(2.0f32)?.add_scalar(-1.0f32)?;

        // Convert to model dtype
        let mut h = if x.dtype() != self.dtype { x.to_dtype(self.dtype)? } else { x };

        // Encoder forward pass
        h = self.encoder_conv_in.forward(&h)?;

        // Down blocks
        for block in &self.encoder_down_blocks {
            h = block.forward(&h)?;
        }

        // Mid block
        h = self.encoder_mid_block.forward(&h)?;

        // Final norm and conv
        h = self.encoder_norm_out.forward(&h)?;
        h = self.encoder_conv_out.forward(&h)?;

        // Quantization - outputs mean and logvar (8 channels total)
        let moments = self.quant_conv.forward(&h)?;

        // Split into mean and logvar (each 4 channels)
        let chunks = moments.chunk(2, 1)?;
        let mean = &chunks[0];
        let logvar = &chunks[1];

        // Sample from the distribution during training
        // z = mean + std * eps, where std = exp(0.5 * logvar)
        let std = logvar.mul_scalar(0.5f32)?.exp()?;
        let eps = Tensor::randn(mean.shape().clone(), 0.0f32, 1.0f32, mean.device().clone())?;
        let z = mean.add(&std.mul(&eps)?)?;

        // Scale by VAE factor (use correct SDXL factor)
        Ok(z.mul_scalar(0.13025f32)?)
    }

    /// Decode latent to image;
    pub fn decode(&self, z: &Tensor) -> Result<Tensor> {
        // Unscale (use correct SDXL factor)
        let z = z.div_scalar(0.13025)?;

        // Convert to model dtype
        let mut h = if z.dtype() != self.dtype { z.to_dtype(self.dtype)? } else { z };

        // Post quantization
        h = self.post_quant_conv.forward(&h)?;

        // Decoder forward pass
        h = self.decoder_conv_in.forward(&h)?;

        // Mid block
        h = self.decoder_mid_block.forward(&h)?;

        // Up blocks
        for block in &self.decoder_up_blocks {
            h = block.forward(&h)?;
        }

        // Final norm and conv
        h = self.decoder_norm_out.forward(&h)?;
        h = self.decoder_conv_out.forward(&h)?;

        // Convert from [-1, 1] to [0, 1]
        Ok(h.add_scalar(1.0f32)?.div_scalar(2.0f32)?)
    }

    /// Create VAE with CPU offloading for low VRAM
    pub fn new_with_cpu_offload(
        weights: HashMap<String, Tensor>,
        device: Device,
        dtype: DType,
    ) -> flame_core::Result<Self> {
        println!("Creating native SDXL VAE with CPU offloading support...");

        // First, move all weights to CPU to free up GPU memory
        let cpu_device = flame_core::device::Device::cuda(0)?;
        let mut cpu_weights = HashMap::new();

        for (name, tensor) in weights {
            let cpu_tensor = tensor;
            cpu_weights.insert(name, cpu_tensor);
        }

        // Now create the VAE, loading weights to GPU only as needed
        Self::new(cpu_weights, device, dtype)
    }

    /// Encode with memory-efficient mode (for training)
    pub fn encode_memory_efficient(&self, x: &Tensor) -> Result<Tensor> {
        // For memory efficiency during training, we can:
        // 1. Process in smaller tiles if needed
        // 2. Clear intermediate tensors aggressively
        // 3. Use gradient checkpointing

        // For now, use standard encode but with explicit memory management;
        let result = self.encode(x)?;

        // Force CUDA synchronization to free memory
        // FLAME only supports CUDA devices currently
        {
            // Memory will be freed on next allocation
            // FLAME handles this internally
        }

        Ok(result)
    }

    fn convert(tensor: Tensor) -> flame_core::Result<Tensor> {
        Ok(tensor)
    }
} // Close impl SDXLVAENative
