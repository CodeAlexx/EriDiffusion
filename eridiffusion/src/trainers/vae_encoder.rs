use flame_core::device::Device;
use flame_core::optimizers::{Adam, SGD};
use flame_core::{DType, Result, Shape, Tensor};
use std::collections::HashMap;

// VAE encoder for SDXL - NO VARBUILDER
// Direct weight implementation for encoding images to latents

// FLAME uses flame_core::device::Device instead of Device

pub struct SDXLVAEEncoder {
    weights: std::collections::HashMap<String, Tensor>,
    device: Device,
}

impl SDXLVAEEncoder {
    pub fn new(
        weights: std::collections::HashMap<String, Tensor>,
        device: Device,
    ) -> flame_core::Result<Self> {
        Ok(Self { weights, device })
    }

    /// Encode image to latent space
    pub fn encode(&self, image: &Tensor) -> flame_core::Result<Tensor> {
        // SDXL VAE expects input normalized to [-1, 1]
        // Input should already be in [0, 1] range
        let two = Tensor::full(image.shape().clone(), 2.0f32, image.device().clone())?;
        let one = Tensor::full(image.shape().clone(), 1.0f32, image.device().clone())?;
        let x = image.mul(&two)?.sub(&one)?;

        // Ensure correct dtype to match weights
        let x = if x.dtype() != DType::F16 { x.to_dtype(DType::F16)? } else { x.clone() };

        // Initial conv
        let mut h = conv2d(
            &x,
            &self.weights["encoder.conv_in.weight"],
            &self.weights["encoder.conv_in.bias"],
            1,
            1,
        )?;

        // Down blocks
        let down_block_configs = vec![
            // block_idx, num_layers, downsample
            (0, 2, false),
            (1, 2, true),
            (2, 2, true),
            (3, 2, true),
        ];

        for (block_idx, num_layers, should_downsample) in down_block_configs {
            for layer_idx in 0..num_layers {
                h = self.resnet_block(&h, block_idx, layer_idx)?;
            }

            if should_downsample && block_idx < 3 {
                h = downsample(&h, &self.weights, block_idx as usize)?;
            }
        }

        // Middle blocks
        h = self.resnet_block(&h, -1, 0)?; // mid.block_1
        h = self.attention_block(&h)?;
        h = self.resnet_block(&h, -1, 1)?; // mid.block_2

        // Final layers
        h = group_norm(
            &h,
            32,
            &self.weights["encoder.norm_out.weight"],
            &self.weights["encoder.norm_out.bias"],
        )?;
        h = h.silu()?;
        h = conv2d(
            &h,
            &self.weights["encoder.conv_out.weight"],
            &self.weights["encoder.conv_out.bias"],
            1,
            1,
        )?;

        // Split into mean and logvar
        let chunks = h.chunk(2, 1)?;
        let mean = &chunks[0];
        let logvar = &chunks[1];

        // Sample from distribution
        let half = Tensor::full(logvar.shape().clone(), 0.5f32, logvar.device().clone())?;
        let std = (logvar.mul(&half))?.exp()?;
        let eps = Tensor::randn(mean.shape().clone(), 0.0f32, 1.0f32, mean.device().clone())?;
        let z = mean.add(&std.mul(&eps)?)?;

        // Scale by latent scaling factor
        let scale = Tensor::full(z.shape().clone(), 0.18215f32, z.device().clone())?;
        Ok(z.mul(&scale)?)
    }

    /// Decode latent to image
    pub fn decode(&self, latent: &Tensor) -> flame_core::Result<Tensor> {
        // Scale back
        let scale = Tensor::full(latent.shape().clone(), 0.18215f32, latent.device().clone())?;
        let z = latent.div(&scale)?;

        // Initial conv
        let mut h = conv2d(
            &z,
            &self.weights["decoder.conv_in.weight"],
            &self.weights["decoder.conv_in.bias"],
            1,
            1,
        )?;

        // Middle blocks
        h = self.decoder_resnet_block(&h, -1, 0)?;
        h = self.decoder_attention_block(&h)?;
        h = self.decoder_resnet_block(&h, -1, 1)?;

        // Up blocks
        let up_block_configs = vec![
            // block_idx, num_layers, upsample
            (0, 3, true),
            (1, 3, true),
            (2, 3, true),
            (3, 3, false),
        ];

        for (block_idx, num_layers, should_upsample) in up_block_configs {
            for layer_idx in 0..num_layers {
                h = self.decoder_resnet_block(&h, block_idx, layer_idx)?;
            }

            if should_upsample && block_idx < 3 {
                h = upsample(&h, &self.weights, block_idx as usize)?;
            }
        }

        // Final layers
        h = group_norm(
            &h,
            32,
            &self.weights["decoder.norm_out.weight"],
            &self.weights["decoder.norm_out.bias"],
        )?;
        h = h.silu()?;
        h = conv2d(
            &h,
            &self.weights["decoder.conv_out.weight"],
            &self.weights["decoder.conv_out.bias"],
            1,
            1,
        )?;

        // Denormalize from [-1, 1] to [0, 1]
        let one = Tensor::full(h.shape().clone(), 1.0f32, h.device().clone())?;
        let two = Tensor::full(h.shape().clone(), 2.0f32, h.device().clone())?;
        Ok(h.add(&one)?.div(&two)?)
    }

    fn resnet_block(
        &self,
        x: &Tensor,
        block_idx: i32,
        layer_idx: usize,
    ) -> flame_core::Result<Tensor> {
        let prefix = if block_idx >= 0 {
            format!("encoder.down_blocks.{}.resnets.{}", block_idx, layer_idx)
        } else {
            format!("encoder.mid_block.resnets.{}", layer_idx)
        };

        // First conv block
        let h = group_norm(
            x,
            32,
            &self.weights[&format!("{}.norm1.weight", prefix)],
            &self.weights[&format!("{}.norm1.bias", prefix)],
        )?;
        let h = h.silu()?;
        let h = conv2d(
            &h,
            &self.weights[&format!("{}.conv1.weight", prefix)],
            &self.weights[&format!("{}.conv1.bias", prefix)],
            1,
            1,
        )?;

        // Second conv block
        let h = group_norm(
            &h,
            32,
            &self.weights[&format!("{}.norm2.weight", prefix)],
            &self.weights[&format!("{}.norm2.bias", prefix)],
        )?;
        let h = h.silu()?;
        let h = conv2d(
            &h,
            &self.weights[&format!("{}.conv2.weight", prefix)],
            &self.weights[&format!("{}.conv2.bias", prefix)],
            1,
            1,
        )?;

        // Skip connection
        let skip = if x.shape().dims()[1] != h.shape().dims()[1] {
            conv2d(
                x,
                &self.weights[&format!("{}.conv_shortcut.weight", prefix)],
                &self.weights[&format!("{}.conv_shortcut.bias", prefix)],
                1,
                1,
            )?
        } else {
            x.clone()
        };

        Ok(h.add(&skip)?)
    }

    fn decoder_resnet_block(
        &self,
        x: &Tensor,
        block_idx: i32,
        layer_idx: usize,
    ) -> flame_core::Result<Tensor> {
        let prefix = if block_idx >= 0 {
            format!("decoder.up_blocks.{}.resnets.{}", block_idx, layer_idx)
        } else {
            format!("decoder.mid_block.resnets.{}", layer_idx)
        };

        self.resnet_block_base(x, &prefix)
    }

    fn resnet_block_base(&self, x: &Tensor, prefix: &str) -> flame_core::Result<Tensor> {
        // Similar to encoder resnet but with decoder prefix
        let h = group_norm(
            x,
            32,
            &self.weights[&format!("{}.norm1.weight", prefix)],
            &self.weights[&format!("{}.norm1.bias", prefix)],
        )?;
        let h = h.silu()?;
        let h = conv2d(
            &h,
            &self.weights[&format!("{}.conv1.weight", prefix)],
            &self.weights[&format!("{}.conv1.bias", prefix)],
            1,
            1,
        )?;

        let h = group_norm(
            &h,
            32,
            &self.weights[&format!("{}.norm2.weight", prefix)],
            &self.weights[&format!("{}.norm2.bias", prefix)],
        )?;
        let h = h.silu()?;
        let h = conv2d(
            &h,
            &self.weights[&format!("{}.conv2.weight", prefix)],
            &self.weights[&format!("{}.conv2.bias", prefix)],
            1,
            1,
        )?;

        let skip = if x.shape().dims()[1] != h.shape().dims()[1] {
            conv2d(
                x,
                &self.weights[&format!("{}.conv_shortcut.weight", prefix)],
                &self.weights[&format!("{}.conv_shortcut.bias", prefix)],
                1,
                1,
            )?
        } else {
            x.clone()
        };

        Ok(h.add(&skip)?)
    }

    fn attention_block(&self, x: &Tensor) -> flame_core::Result<Tensor> {
        let prefix = "encoder.mid_block.attentions.0";
        self.attention_impl(x, prefix)
    }

    fn decoder_attention_block(&self, x: &Tensor) -> flame_core::Result<Tensor> {
        let prefix = "decoder.mid_block.attentions.0";
        self.attention_impl(x, prefix)
    }

    fn attention_impl(&self, x: &Tensor, prefix: &str) -> flame_core::Result<Tensor> {
        let shape = x.shape().dims();
        let (b, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);

        // Reshape for attention
        let x_flat = x.transpose_dims(1, 2)?.transpose_dims(2, 3)?.reshape(&[b, h * w, c])?;

        // Group norm
        let norm = group_norm_1d(
            &x_flat,
            32,
            &self.weights[&format!("{}.group_norm.weight", prefix)],
            &self.weights[&format!("{}.group_norm.bias", prefix)],
        )?;

        // Q, K, V projections
        let q = linear_op(
            &norm,
            &self.weights[&format!("{}.to_q.weight", prefix)],
            self.weights.get(&format!("{}.to_q.bias", prefix)),
        )?;
        let k = linear_op(
            &norm,
            &self.weights[&format!("{}.to_k.weight", prefix)],
            self.weights.get(&format!("{}.to_k.bias", prefix)),
        )?;
        let v = linear_op(
            &norm,
            &self.weights[&format!("{}.to_v.weight", prefix)],
            self.weights.get(&format!("{}.to_v.bias", prefix)),
        )?;

        // Attention
        let num_heads = 1; // VAE uses single head attention
        let head_dim = c / num_heads;

        let q = q.reshape(&[b, h * w, num_heads, head_dim])?.transpose_dims(1, 2)?;
        let k = k.reshape(&[b, h * w, num_heads, head_dim])?.transpose_dims(1, 2)?;
        let v = v.reshape(&[b, h * w, num_heads, head_dim])?.transpose_dims(1, 2)?;

        let scale = Tensor::full(q.shape().clone(), (head_dim as f32).sqrt(), q.device().clone())?;
        let scores = q.matmul(&k.transpose_dims(2, 3)?)?.div(&scale)?;
        let attn = scores.softmax(-1)?;
        let out = attn.matmul(&v)?;

        let out = out.transpose_dims(1, 2)?.reshape(&[b, h * w, c])?;

        // Output projection
        let out = linear_op(
            &out,
            &self.weights[&format!("{}.to_out.0.weight", prefix)],
            self.weights.get(&format!("{}.to_out.0.bias", prefix)),
        )?;

        // Reshape back
        let out = out.reshape(&[b, h, w, c])?.transpose_dims(2, 3)?.transpose_dims(1, 2)?;

        Ok(x.add(&out)?)
    }

    fn downsample(&self, x: &Tensor, block_idx: usize) -> flame_core::Result<Tensor> {
        downsample(x, &self.weights, block_idx)
    }

    fn upsample(&self, x: &Tensor, block_idx: usize) -> flame_core::Result<Tensor> {
        upsample(x, &self.weights, block_idx)
    }
}

// Helper functions
fn conv2d(
    x: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    stride: usize,
    padding: usize,
) -> flame_core::Result<Tensor> {
    // Ensure matching dtypes
    let x = if x.dtype() != weight.dtype() { x.to_dtype(weight.dtype())? } else { x.clone() };

    let out = x.conv2d(weight, None, stride, padding)?;
    // Handle bias shape - may need to be unsqueezed for broadcasting
    let bias = if bias.shape().rank() == 1 {
        bias.unsqueeze(0)?.unsqueeze(2)?.unsqueeze(3)?
    } else {
        bias.clone()
    };
    out.add(&bias)
}

fn downsample(
    x: &Tensor,
    weights: &HashMap<String, Tensor>,
    block_idx: usize,
) -> flame_core::Result<Tensor> {
    let prefix = format!("encoder.down_blocks.{}.downsamplers.0", block_idx);
    conv2d(
        x,
        &weights[&format!("{}.conv.weight", prefix)],
        &weights[&format!("{}.conv.bias", prefix)],
        2,
        1,
    )
}

fn upsample(
    x: &Tensor,
    weights: &HashMap<String, Tensor>,
    block_idx: usize,
) -> flame_core::Result<Tensor> {
    let prefix = format!("decoder.up_blocks.{}.upsamplers.0", block_idx);
    let h = x.shape().dims()[2];
    let w = x.shape().dims()[3];
    let x = flame_core::cuda_ops::GpuOps::upsample2d_nearest(x, (h * 2, w * 2))?;
    conv2d(
        &x,
        &weights[&format!("{}.conv.weight", prefix)],
        &weights[&format!("{}.conv.bias", prefix)],
        1,
        1,
    )
}

fn group_norm(
    x: &Tensor,
    groups: usize,
    scale: &Tensor,
    bias: &Tensor,
) -> flame_core::Result<Tensor> {
    let shape = x.shape();
    let dims = shape.dims();
    let b = dims[0];
    let c = dims[1];
    let h = dims[2];
    let w = dims[3];

    let x = x.reshape(&[b, groups, c / groups, h, w])?;

    let mean = x.mean_dim(&[2], true)?;
    let var = x.var_dim(&[2], true)?;
    let epsilon = Tensor::full(var.shape().clone(), 1e-5f32, var.device().clone())?;
    let x = x.sub(&mean)?.div(&var.add(&epsilon)?.sqrt()?)?;

    let x = x.reshape(&[b, c, h, w])?;
    let scale = scale.reshape(&[1, c, 1, 1])?;
    let bias = bias.reshape(&[1, c, 1, 1])?;
    x.mul(&scale)?.add(&bias)
}

fn group_norm_1d(
    x: &Tensor,
    groups: usize,
    scale: &Tensor,
    bias: &Tensor,
) -> flame_core::Result<Tensor> {
    let shape = x.shape();
    let dims = shape.dims();
    let b = dims[0];
    let seq = dims[1];
    let c = dims[2];

    let x = x.reshape(&[b, seq, groups, c / groups])?;

    let mean = x.mean_dim(&[3], true)?;
    let var = x.var_dim(&[3], true)?;
    let epsilon = Tensor::full(var.shape().clone(), 1e-5f32, var.device().clone())?;
    let x = x.sub(&mean)?.div(&var.add(&epsilon)?.sqrt()?)?;

    let x = x.reshape(&[b, seq, c])?;
    let scale = scale.reshape(&[1, 1, c])?;
    let bias = bias.reshape(&[1, 1, c])?;
    x.mul(&scale)?.add(&bias)
}

fn linear_op(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> flame_core::Result<Tensor> {
    let out = x.matmul(&weight.transpose_dims(0, 1)?)?;
    if let Some(b) = bias {
        Ok(out.add(b)?)
    } else {
        Ok(out)
    }
}
