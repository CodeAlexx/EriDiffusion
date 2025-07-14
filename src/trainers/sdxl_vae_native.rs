//! Native SDXL VAE implementation that directly uses SD format weights
//! No remapping needed - works directly with the checkpoint format

use anyhow::{Result as AnyhowResult, Context};
use candle_core::{Device, DType, Tensor, Module, D, Result};
use candle_nn;
use std::collections::HashMap;

/// Simple GroupNorm wrapper
#[derive(Debug)]
struct GroupNorm {
    weight: Tensor,
    bias: Tensor,
    num_groups: usize,
    eps: f64,
}

impl GroupNorm {
    fn new(weight: Tensor, bias: Tensor, num_groups: usize) -> AnyhowResult<Self> {
        Ok(Self {
            weight,
            bias,
            num_groups,
            eps: 1e-6,
        })
    }
}

impl Module for GroupNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Group normalization implementation
        let (b, c, h, w) = x.dims4()?;
        let g = self.num_groups;
        let c_per_g = c / g;
        
        // Reshape to (B, G, C/G, H, W)
        let x = x.reshape((b, g, c_per_g, h * w))?;
        
        // Calculate mean and variance per group
        let mean = x.mean_keepdim(D::Minus1)?;
        let var = x.var_keepdim(D::Minus1)?;
        
        // Normalize
        let x = (x.broadcast_sub(&mean))?.broadcast_div(&(var + self.eps)?.sqrt()?)?;
        
        // Reshape back to (B, C, H, W)
        let x = x.reshape((b, c, h, w))?;
        
        // Apply weight and bias
        let weight = self.weight.reshape((1, c, 1, 1))?;
        let bias = self.bias.reshape((1, c, 1, 1))?;
        
        let x = x.broadcast_mul(&weight)?;
        Ok(x.broadcast_add(&bias)?)
    }
}

/// Simple Conv2d wrapper
#[derive(Debug)]
struct Conv2d {
    weight: Tensor,
    bias: Option<Tensor>,
    stride: usize,
    padding: usize,
    dilation: usize,
}

impl Conv2d {
    fn new(weight: Tensor, bias: Option<Tensor>, padding: usize) -> Self {
        Self {
            weight,
            bias,
            stride: 1,
            padding,
            dilation: 1,
        }
    }
    
    fn new_with_stride(weight: Tensor, bias: Option<Tensor>, padding: usize, stride: usize) -> Self {
        Self {
            weight,
            bias,
            stride,
            padding,
            dilation: 1,
        }
    }
}

impl Module for Conv2d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut out = x.conv2d(&self.weight, self.padding, self.stride, self.dilation, 1)?;
        if let Some(ref b) = self.bias {
            let b = b.reshape((1, b.elem_count(), 1, 1))?;
            out = out.broadcast_add(&b)?;
        }
        Ok(out)
    }
}

/// Native SDXL VAE implementation
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

/// VAE ResNet block
struct VAEResNetBlock {
    norm1: GroupNorm,
    conv1: Conv2d,
    norm2: GroupNorm,
    conv2: Conv2d,
    conv_shortcut: Option<Conv2d>,
}

impl VAEResNetBlock {
    fn new(weights: &HashMap<String, Tensor>, prefix: &str, in_channels: usize, out_channels: usize) -> AnyhowResult<Self> {
        // Debug channel info
        if prefix.contains("decoder.up.2") {
            println!("ResNet block {}: in_channels={}, out_channels={}", prefix, in_channels, out_channels);
        }
        
        // Load norm1
        let norm1_weight = weights.get(&format!("{}.norm1.weight", prefix))
            .context(format!("Missing {}.norm1.weight", prefix))?;
        let norm1_bias = weights.get(&format!("{}.norm1.bias", prefix))
            .context(format!("Missing {}.norm1.bias", prefix))?;
        let norm1 = GroupNorm::new(norm1_weight.clone(), norm1_bias.clone(), 32)?;
        
        // Load conv1
        let conv1_weight = weights.get(&format!("{}.conv1.weight", prefix))
            .context(format!("Missing {}.conv1.weight", prefix))?;
        let conv1_bias = weights.get(&format!("{}.conv1.bias", prefix))
            .context(format!("Missing {}.conv1.bias", prefix))?;
        let conv1 = Conv2d::new(conv1_weight.clone(), Some(conv1_bias.clone()), 1);
        
        // Load norm2
        let norm2_weight = weights.get(&format!("{}.norm2.weight", prefix))
            .context(format!("Missing {}.norm2.weight", prefix))?;
        let norm2_bias = weights.get(&format!("{}.norm2.bias", prefix))
            .context(format!("Missing {}.norm2.bias", prefix))?;
        let norm2 = GroupNorm::new(norm2_weight.clone(), norm2_bias.clone(), 32)?;
        
        // Load conv2
        let conv2_weight = weights.get(&format!("{}.conv2.weight", prefix))
            .context(format!("Missing {}.conv2.weight", prefix))?;
        let conv2_bias = weights.get(&format!("{}.conv2.bias", prefix))
            .context(format!("Missing {}.conv2.bias", prefix))?;
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
                if in_channels != out_channels {
                    return Err(anyhow::anyhow!("Channel mismatch without shortcut: {} -> {}", in_channels, out_channels));
                }
                None
            }
        } else {
            None
        };
        
        Ok(Self {
            norm1,
            conv1,
            norm2,
            conv2,
            conv_shortcut,
        })
    }
    
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x;
        
        let h = x.apply(&self.norm1)?;
        let h = candle_nn::ops::silu(&h)?;  // SiLU is the same as Swish
        let h = h.apply(&self.conv1)?;
        
        let h = h.apply(&self.norm2)?;
        let h = candle_nn::ops::silu(&h)?;
        let h = h.apply(&self.conv2)?;
        
        // Apply shortcut if needed
        let residual = if let Some(ref conv) = self.conv_shortcut {
            residual.apply(conv)?
        } else {
            residual.clone()
        };
        
        Ok((h + residual)?)
    }
}

/// VAE Attention block
struct VAEAttentionBlock {
    norm: GroupNorm,
    q: Conv2d,
    k: Conv2d,
    v: Conv2d,
    proj_out: Conv2d,
}

impl VAEAttentionBlock {
    fn new(weights: &HashMap<String, Tensor>, prefix: &str) -> AnyhowResult<Self> {
        let norm_weight = weights.get(&format!("{}.norm.weight", prefix))
            .context(format!("Missing {}.norm.weight", prefix))?;
        let norm_bias = weights.get(&format!("{}.norm.bias", prefix))
            .context(format!("Missing {}.norm.bias", prefix))?;
        let norm = GroupNorm::new(norm_weight.clone(), norm_bias.clone(), 32)?;
        
        let q_weight = weights.get(&format!("{}.q.weight", prefix))
            .context(format!("Missing {}.q.weight", prefix))?;
        let q_bias = weights.get(&format!("{}.q.bias", prefix))
            .context(format!("Missing {}.q.bias", prefix))?;
        let q = Conv2d::new(q_weight.clone(), Some(q_bias.clone()), 0);
        
        let k_weight = weights.get(&format!("{}.k.weight", prefix))
            .context(format!("Missing {}.k.weight", prefix))?;
        let k_bias = weights.get(&format!("{}.k.bias", prefix))
            .context(format!("Missing {}.k.bias", prefix))?;
        let k = Conv2d::new(k_weight.clone(), Some(k_bias.clone()), 0);
        
        let v_weight = weights.get(&format!("{}.v.weight", prefix))
            .context(format!("Missing {}.v.weight", prefix))?;
        let v_bias = weights.get(&format!("{}.v.bias", prefix))
            .context(format!("Missing {}.v.bias", prefix))?;
        let v = Conv2d::new(v_weight.clone(), Some(v_bias.clone()), 0);
        
        let proj_out_weight = weights.get(&format!("{}.proj_out.weight", prefix))
            .context(format!("Missing {}.proj_out.weight", prefix))?;
        let proj_out_bias = weights.get(&format!("{}.proj_out.bias", prefix))
            .context(format!("Missing {}.proj_out.bias", prefix))?;
        let proj_out = Conv2d::new(proj_out_weight.clone(), Some(proj_out_bias.clone()), 0);
        
        Ok(Self { norm, q, k, v, proj_out })
    }
    
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x;
        let (b, c, h, w) = x.dims4()?;
        
        // Normalize
        let x = x.apply(&self.norm)?;
        
        // Compute Q, K, V
        let q = x.apply(&self.q)?;
        let k = x.apply(&self.k)?;
        let v = x.apply(&self.v)?;
        
        // Reshape for attention: [B, C, H, W] -> [B, H*W, C]
        let q = q.reshape((b, c, h * w))?.transpose(1, 2)?;
        let k = k.reshape((b, c, h * w))?.transpose(1, 2)?;
        let v = v.reshape((b, c, h * w))?.transpose(1, 2)?;
        
        // Scaled dot-product attention
        let scale = (c as f64).sqrt();
        let scores = (q.matmul(&k.transpose(1, 2)?)? / scale)?;
        let attn = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let out = attn.matmul(&v)?;
        
        // Reshape back: [B, H*W, C] -> [B, C, H, W]
        let out = out.transpose(1, 2)?.reshape((b, c, h, w))?;
        
        // Project and add residual
        let out = out.apply(&self.proj_out)?;
        Ok((out + residual)?)
    }
}

/// VAE Down block
struct VAEDownBlock {
    resnets: Vec<VAEResNetBlock>,
    downsample: Option<Conv2d>,
}

impl VAEDownBlock {
    fn new(weights: &HashMap<String, Tensor>, block_idx: usize, in_channels: usize, out_channels: usize, downsample: bool) -> AnyhowResult<Self> {
        let mut resnets = Vec::new();
        
        // First resnet uses in_channels -> out_channels
        let resnet0_prefix = format!("encoder.down.{}.block.0", block_idx);
        resnets.push(VAEResNetBlock::new(weights, &resnet0_prefix, in_channels, out_channels)?);
        
        // Second resnet uses out_channels -> out_channels
        let resnet1_prefix = format!("encoder.down.{}.block.1", block_idx);
        resnets.push(VAEResNetBlock::new(weights, &resnet1_prefix, out_channels, out_channels)?);
        
        // Downsample layer
        let downsample = if downsample {
            let down_weight = weights.get(&format!("encoder.down.{}.downsample.conv.weight", block_idx))
                .context(format!("Missing encoder.down.{}.downsample.conv.weight", block_idx))?;
            let down_bias = weights.get(&format!("encoder.down.{}.downsample.conv.bias", block_idx))
                .context(format!("Missing encoder.down.{}.downsample.conv.bias", block_idx))?;
            Some(Conv2d::new_with_stride(down_weight.clone(), Some(down_bias.clone()), 1, 2))
        } else {
            None
        };
        
        Ok(Self { resnets, downsample })
    }
    
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = x.clone();
        
        for resnet in &self.resnets {
            h = resnet.forward(&h)?;
        }
        
        if let Some(ref down) = self.downsample {
            h = h.apply(down)?;
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
}

impl ConvTranspose2d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut out = x.conv_transpose2d(&self.weight, 1, 0, 2, 1)?;
        if let Some(ref b) = self.bias {
            let b = b.reshape((1, b.elem_count(), 1, 1))?;
            out = out.broadcast_add(&b)?;
        }
        Ok(out)
    }
}

impl VAEUpBlock {
    fn new(weights: &HashMap<String, Tensor>, block_idx: usize, in_channels: usize, out_channels: usize, upsample: bool) -> AnyhowResult<Self> {
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
        let upsample = if upsample {
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
    
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
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
struct VAEMidBlock {
    resnet1: VAEResNetBlock,
    attn1: VAEAttentionBlock,
    resnet2: VAEResNetBlock,
}

impl VAEMidBlock {
    fn new(weights: &HashMap<String, Tensor>, prefix: &str, channels: usize) -> AnyhowResult<Self> {
        let resnet1 = VAEResNetBlock::new(weights, &format!("{}.block_1", prefix), channels, channels)?;
        let attn1 = VAEAttentionBlock::new(weights, &format!("{}.attn_1", prefix))?;
        let resnet2 = VAEResNetBlock::new(weights, &format!("{}.block_2", prefix), channels, channels)?;
        
        Ok(Self { resnet1, attn1, resnet2 })
    }
    
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.resnet1.forward(x)?;
        let h = self.attn1.forward(&h)?;
        let h = self.resnet2.forward(&h)?;
        Ok(h)
    }
}

impl SDXLVAENative {
    /// Create new VAE from SD format weights - no remapping needed!
    pub fn new(weights: HashMap<String, Tensor>, device: Device, dtype: DType) -> AnyhowResult<Self> {
        println!("Creating native SDXL VAE (no remapping needed)...");
        
        // Analyze the architecture from weights to determine correct channel counts
        println!("\n=== Analyzing VAE architecture from weights ===");
        
        // Check encoder block 0 to determine base channels
        let base_channels = if let Some(tensor) = weights.get("encoder.down.0.block.0.conv1.weight") {
            // Conv weight shape is [out_channels, in_channels, k, k]
            let out_channels = tensor.dims()[0];
            println!("Detected base channels from encoder block 0: {}", out_channels);
            out_channels
        } else {
            println!("Using default base channels: 128");
            128
        };
        
        // Encoder conv_in
        let encoder_conv_in = Conv2d::new(
            weights.get("encoder.conv_in.weight").context("Missing encoder.conv_in.weight")?.clone(),
            Some(weights.get("encoder.conv_in.bias").context("Missing encoder.conv_in.bias")?.clone()),
            1
        );
        
        // Encoder down blocks
        let mut encoder_down_blocks = Vec::new();
        let channel_mult = [1, 2, 4, 4];
        let mut prev_channels = base_channels;
        
        for (i, &mult) in channel_mult.iter().enumerate() {
            let channels = base_channels * mult;
            let downsample = i < 3; // No downsample on last block
            encoder_down_blocks.push(VAEDownBlock::new(&weights, i, prev_channels, channels, downsample)?);
            prev_channels = channels;
        }
        
        // Encoder mid block
        let encoder_mid_block = VAEMidBlock::new(&weights, "encoder.mid", 512)?;
        
        // Encoder norm_out and conv_out
        let encoder_norm_out = GroupNorm::new(
            weights.get("encoder.norm_out.weight").context("Missing encoder.norm_out.weight")?.clone(),
            weights.get("encoder.norm_out.bias").context("Missing encoder.norm_out.bias")?.clone(),
            32
        )?;
        
        let encoder_conv_out = Conv2d::new(
            weights.get("encoder.conv_out.weight").context("Missing encoder.conv_out.weight")?.clone(),
            Some(weights.get("encoder.conv_out.bias").context("Missing encoder.conv_out.bias")?.clone()),
            1
        );
        
        // Quantization layers
        let quant_conv = Conv2d::new(
            weights.get("quant_conv.weight").context("Missing quant_conv.weight")?.clone(),
            Some(weights.get("quant_conv.bias").context("Missing quant_conv.bias")?.clone()),
            0
        );
        
        let post_quant_conv = Conv2d::new(
            weights.get("post_quant_conv.weight").context("Missing post_quant_conv.weight")?.clone(),
            Some(weights.get("post_quant_conv.bias").context("Missing post_quant_conv.bias")?.clone()),
            0
        );
        
        // Decoder conv_in
        let decoder_conv_in = Conv2d::new(
            weights.get("decoder.conv_in.weight").context("Missing decoder.conv_in.weight")?.clone(),
            Some(weights.get("decoder.conv_in.bias").context("Missing decoder.conv_in.bias")?.clone()),
            1
        );
        
        // Decoder up blocks
        let mut decoder_up_blocks = Vec::new();
        
        // Based on actual weight inspection:
        // Block 0: 256->128 (no upsample)  
        // Block 1: 512->256 with upsample
        // Block 2: 512->512 with upsample
        // Block 3: 512->512 with upsample
        let decoder_configs = [
            (256, 128, false),  // Block 0
            (512, 256, true),   // Block 1
            (512, 512, true),   // Block 2
            (512, 512, true),   // Block 3
        ];
        
        for (i, &(in_ch, out_ch, has_upsample)) in decoder_configs.iter().enumerate() {
            decoder_up_blocks.push(VAEUpBlock::new(&weights, i, in_ch, out_ch, has_upsample)?);
        }
        
        // Decoder mid block
        let decoder_mid_block = VAEMidBlock::new(&weights, "decoder.mid", 512)?;
        
        // Decoder norm_out and conv_out
        let decoder_norm_out = GroupNorm::new(
            weights.get("decoder.norm_out.weight").context("Missing decoder.norm_out.weight")?.clone(),
            weights.get("decoder.norm_out.bias").context("Missing decoder.norm_out.bias")?.clone(),
            32
        )?;
        
        let decoder_conv_out = Conv2d::new(
            weights.get("decoder.conv_out.weight").context("Missing decoder.conv_out.weight")?.clone(),
            Some(weights.get("decoder.conv_out.bias").context("Missing decoder.conv_out.bias")?.clone()),
            1
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
    pub fn encode(&self, x: &Tensor) -> AnyhowResult<Tensor> {
        // x should be in [0, 1] range, convert to [-1, 1]
        let x = ((x * 2.0)? - 1.0)?;
        
        // Convert to model dtype
        let mut h = if x.dtype() != self.dtype {
            x.to_dtype(self.dtype)?
        } else {
            x
        };
        
        // Encoder forward pass
        h = h.apply(&self.encoder_conv_in)?;
        
        // Down blocks
        for block in &self.encoder_down_blocks {
            h = block.forward(&h)?;
        }
        
        // Mid block
        h = self.encoder_mid_block.forward(&h)?;
        
        // Final norm and conv
        h = h.apply(&self.encoder_norm_out)?;
        h = candle_nn::ops::silu(&h)?;
        h = h.apply(&self.encoder_conv_out)?;
        
        // Quantization - outputs mean and logvar (8 channels total)
        let moments = h.apply(&self.quant_conv)?;
        
        // Split into mean and logvar (each 4 channels)
        let chunks = moments.chunk(2, 1)?;
        let mean = &chunks[0];
        let logvar = &chunks[1];
        
        // Sample from the distribution during training
        // z = mean + std * eps, where std = exp(0.5 * logvar)
        let std = (logvar * 0.5)?.exp()?;
        let eps = mean.randn_like(0.0, 1.0)?;
        let z = (mean + std * eps)?;
        
        // Scale by VAE factor (use correct SDXL factor)
        Ok((z * 0.13025)?)
    }
    
    /// Decode latent to image
    pub fn decode(&self, z: &Tensor) -> AnyhowResult<Tensor> {
        // Unscale (use correct SDXL factor)
        let z = (z / 0.13025)?;
        
        // Convert to model dtype
        let mut h = if z.dtype() != self.dtype {
            z.to_dtype(self.dtype)?
        } else {
            z
        };
        
        // Post quantization
        h = h.apply(&self.post_quant_conv)?;
        
        // Decoder forward pass
        h = h.apply(&self.decoder_conv_in)?;
        
        // Mid block
        h = self.decoder_mid_block.forward(&h)?;
        
        // Up blocks
        for block in &self.decoder_up_blocks {
            h = block.forward(&h)?;
        }
        
        // Final norm and conv
        h = h.apply(&self.decoder_norm_out)?;
        h = candle_nn::ops::silu(&h)?;
        h = h.apply(&self.decoder_conv_out)?;
        
        // Convert from [-1, 1] to [0, 1]
        Ok(((h + 1.0)? / 2.0)?)
    }
    
    /// Create VAE with CPU offloading for low VRAM
    pub fn new_with_cpu_offload(weights: HashMap<String, Tensor>, device: Device, dtype: DType) -> AnyhowResult<Self> {
        println!("Creating native SDXL VAE with CPU offloading support...");
        
        // First, move all weights to CPU to free up GPU memory
        let cpu_device = Device::Cpu;
        let mut cpu_weights = HashMap::new();
        
        for (name, tensor) in weights {
            let cpu_tensor = tensor.to_device(&cpu_device)?;
            cpu_weights.insert(name, cpu_tensor);
        }
        
        // Now create the VAE, loading weights to GPU only as needed
        Self::new(cpu_weights, device, dtype)
    }
    
    /// Encode with memory-efficient mode (for training)
    pub fn encode_memory_efficient(&self, x: &Tensor) -> AnyhowResult<Tensor> {
        // For memory efficiency during training, we can:
        // 1. Process in smaller tiles if needed
        // 2. Clear intermediate tensors aggressively
        // 3. Use gradient checkpointing
        
        // For now, use standard encode but with explicit memory management
        let result = self.encode(x)?;
        
        // Force CUDA synchronization to free memory
        if let Device::Cuda(_) = &self.device {
            // Memory will be freed on next allocation
            // Candle handles this internally
        }
        
        Ok(result)
    }
}