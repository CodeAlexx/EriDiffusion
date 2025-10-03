//! Fast VAE implementation without attention blocks for high-resolution encoding
use crate::models::tensor_utils::to_dtype_aligned;
use crate::models::vae::{ResnetBlock, VAEConfig};
use flame_core::conv::Conv2d;
use flame_core::device::Device;
use flame_core::group_norm::GroupNorm;
use flame_core::{DType, Result, Shape, Tensor};
use std::collections::HashMap;

/// Fast version of DiagonalGaussianDistribution for internal use
pub struct FastDistribution {
    mean: Tensor,
    logvar: Tensor,
}

impl FastDistribution {
    pub fn sample(&self) -> Result<Tensor> {
        let std = self.logvar.mul_scalar(0.5)?.exp()?;
        let eps = Tensor::randn(self.mean.shape().clone(), 0.0, 1.0, self.mean.device().clone())?;
        self.mean.add(&std.mul(&eps)?)
    }
}

/// Fast encoder block without attention
pub struct FastEncoderBlock {
    resnets: Vec<ResnetBlock>,
    downsample: Option<Conv2d>,
}

impl FastEncoderBlock {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        num_layers: usize,
        norm_groups: usize,
        add_downsample: bool,
        device: Device,
    ) -> Result<Self> {
        let mut resnets = Vec::new();

        for i in 0..num_layers {
            let in_ch = if i == 0 { in_channels } else { out_channels };
            resnets.push(ResnetBlock::new(in_ch, out_channels, norm_groups, device.clone())?);
        }

        let downsample = if add_downsample {
            Some(Conv2d::new(out_channels, out_channels, 3, 2, 1, device.cuda_device().clone())?)
        } else {
            None
        };

        Ok(Self { resnets, downsample })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = x.clone();

        for resnet in &self.resnets {
            h = resnet.forward(&h)?;
            // No attention here!
        }

        if let Some(downsample) = &self.downsample {
            h = downsample.forward(&h)?;
        }

        Ok(h)
    }
}

/// Fast decoder block without attention
pub struct FastDecoderBlock {
    resnets: Vec<ResnetBlock>,
    upsample: Option<Conv2d>,
}

impl FastDecoderBlock {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        num_layers: usize,
        norm_groups: usize,
        add_upsample: bool,
        device: Device,
    ) -> Result<Self> {
        let mut resnets = Vec::new();

        for i in 0..num_layers {
            let out_ch = if i == num_layers - 1 { out_channels } else { in_channels };
            resnets.push(ResnetBlock::new(in_channels, out_ch, norm_groups, device.clone())?);
        }

        let upsample = if add_upsample {
            // Simple upsample using nearest neighbor + conv
            Some(Conv2d::new(out_channels, out_channels, 3, 1, 1, device.cuda_device().clone())?)
        } else {
            None
        };

        Ok(Self { resnets, upsample })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = x.clone();

        for resnet in &self.resnets {
            h = resnet.forward(&h)?;
            // No attention here!
        }

        if let Some(upsample) = &self.upsample {
            // Upsample using nearest neighbor interpolation
            let dims = h.shape().dims();
            let (b, c, h_size, w_size) = (dims[0], dims[1], dims[2], dims[3]);
            h = h
                .reshape(&[b, c, h_size, 1, w_size, 1])?
                .broadcast_to(&Shape::from_dims(&[b, c, h_size, 2, w_size, 2]))?
                .reshape(&[b, c, h_size * 2, w_size * 2])?;
            h = upsample.forward(&h)?;
        }

        Ok(h)
    }
}

/// Fast VAE without attention blocks - optimized for high resolution
pub struct FastVAE {
    encoder: FastEncoder,
    decoder: FastDecoder,
    quant_conv: Option<Conv2d>,
    post_quant_conv: Option<Conv2d>,
    config: VAEConfig,
}

pub struct FastEncoder {
    conv_in: Conv2d,
    down_blocks: Vec<FastEncoderBlock>,
    mid_block: FastEncoderBlock,
    norm_out: GroupNorm,
    conv_out: Conv2d,
}

pub struct FastDecoder {
    conv_in: Conv2d,
    up_blocks: Vec<FastDecoderBlock>,
    mid_block: FastDecoderBlock,
    norm_out: GroupNorm,
    conv_out: Conv2d,
}

impl FastVAE {
    pub fn new(
        config: VAEConfig,
        device: Device,
        weights: HashMap<String, Tensor>,
    ) -> Result<Self> {
        // Create encoder
        let encoder = FastEncoder::new(&config, device.clone())?;

        // Create decoder
        let decoder = FastDecoder::new(&config, device.clone())?;

        // Optional quant convolutions
        let quant_conv = if config.use_quant_conv {
            Some(Conv2d::new(
                2 * config.latent_channels,
                2 * config.latent_channels,
                1,
                1,
                0,
                device.cuda_device().clone(),
            )?)
        } else {
            None
        };

        let post_quant_conv = if config.use_post_quant_conv {
            Some(Conv2d::new(
                config.latent_channels,
                config.latent_channels,
                1,
                1,
                0,
                device.cuda_device().clone(),
            )?)
        } else {
            None
        };

        let mut vae = Self { encoder, decoder, quant_conv, post_quant_conv, config };

        // Load weights
        vae.load_weights(weights)?;

        Ok(vae)
    }

    pub fn encode(&self, x: &Tensor) -> Result<FastDistribution> {
        let h = self.encoder.forward(x)?;

        let moments = if let Some(qc) = &self.quant_conv { qc.forward(&h)? } else { h };

        let chunks = moments.chunk(2, 1)?;
        let mean = chunks[0].clone();
        let logvar = chunks[1].clone();
        Ok(FastDistribution { mean, logvar })
    }

    pub fn decode(&self, z: &Tensor) -> Result<Tensor> {
        let h = if let Some(pqc) = &self.post_quant_conv { pqc.forward(z)? } else { z.clone() };

        self.decoder.forward(&h)
    }

    fn load_weights(&mut self, weights: HashMap<String, Tensor>) -> Result<()> {
        // Load encoder weights
        self.encoder.load_weights(&weights)?;

        // Load decoder weights
        self.decoder.load_weights(&weights)?;

        // Load quant conv weights if present
        if let Some(qc) = &mut self.quant_conv {
            if let Some(w) = weights.get("quant_conv.weight") {
                qc.weight = to_dtype_aligned(w, DType::BF16)?;
            }
            if let Some(b) = weights.get("quant_conv.bias") {
                qc.bias = Some(to_dtype_aligned(b, DType::BF16)?);
            }
        }

        if let Some(pqc) = &mut self.post_quant_conv {
            if let Some(w) = weights.get("post_quant_conv.weight") {
                pqc.weight = to_dtype_aligned(w, DType::BF16)?;
            }
            if let Some(b) = weights.get("post_quant_conv.bias") {
                pqc.bias = Some(to_dtype_aligned(b, DType::BF16)?);
            }
        }

        Ok(())
    }
}

impl FastEncoder {
    fn new(config: &VAEConfig, device: Device) -> Result<Self> {
        let conv_in =
            Conv2d::new(3, config.block_out_channels[0], 3, 1, 1, device.cuda_device().clone())?;

        // Create down blocks
        let mut down_blocks = Vec::new();
        let mut ch = config.block_out_channels[0];

        for (i, &out_ch) in config.block_out_channels.iter().enumerate() {
            let is_last = i == config.block_out_channels.len() - 1;

            down_blocks.push(FastEncoderBlock::new(
                ch,
                out_ch,
                config.layers_per_block,
                config.norm_num_groups,
                !is_last,
                device.clone(),
            )?);

            ch = out_ch;
        }

        // Mid block without attention
        let mid_block = FastEncoderBlock::new(
            ch,
            ch,
            config.layers_per_block,
            config.norm_num_groups,
            false,
            device.clone(),
        )?;

        let norm_out =
            GroupNorm::new(config.norm_num_groups, ch, 1e-6, true, device.cuda_device().clone())?;
        let conv_out =
            Conv2d::new(ch, 2 * config.latent_channels, 3, 1, 1, device.cuda_device().clone())?;

        Ok(Self { conv_in, down_blocks, mid_block, norm_out, conv_out })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = self.conv_in.forward(x)?;

        // Down blocks
        for block in &self.down_blocks {
            h = block.forward(&h)?;
        }

        // Mid block
        h = self.mid_block.forward(&h)?;

        // Output
        h = self.norm_out.forward(&h)?;
        h = h.silu()?;
        self.conv_out.forward(&h)
    }

    fn load_weights(&mut self, weights: &HashMap<String, Tensor>) -> Result<()> {
        // Load conv_in
        if let Some(w) = weights.get("encoder.conv_in.weight") {
            self.conv_in.weight = to_dtype_aligned(w, DType::BF16)?;
        }
        if let Some(b) = weights.get("encoder.conv_in.bias") {
            self.conv_in.bias = Some(to_dtype_aligned(b, DType::BF16)?);
        }

        // Load down blocks
        for (i, block) in self.down_blocks.iter_mut().enumerate() {
            let prefix = format!("encoder.down.{}", i);
            for (j, resnet) in block.resnets.iter_mut().enumerate() {
                resnet.load_weights(weights, &format!("{}.block.{}", prefix, j))?;
            }

            if let Some(ds) = &mut block.downsample {
                if let Some(w) = weights.get(&format!("{}.downsample.conv.weight", prefix)) {
                    ds.weight = to_dtype_aligned(w, DType::BF16)?;
                }
                if let Some(b) = weights.get(&format!("{}.downsample.conv.bias", prefix)) {
                    ds.bias = Some(to_dtype_aligned(b, DType::BF16)?);
                }
            }
        }

        // Load mid block
        for (i, resnet) in self.mid_block.resnets.iter_mut().enumerate() {
            resnet.load_weights(weights, &format!("encoder.mid.block_{}", i + 1))?;
        }

        // Load norm_out and conv_out
        if let Some(w) = weights.get("encoder.norm_out.weight") {
            self.norm_out.weight = Some(to_dtype_aligned(w, DType::BF16)?);
        }
        if let Some(b) = weights.get("encoder.norm_out.bias") {
            self.norm_out.bias = Some(to_dtype_aligned(b, DType::BF16)?);
        }

        if let Some(w) = weights.get("encoder.conv_out.weight") {
            self.conv_out.weight = to_dtype_aligned(w, DType::BF16)?;
        }
        if let Some(b) = weights.get("encoder.conv_out.bias") {
            self.conv_out.bias = Some(to_dtype_aligned(b, DType::BF16)?);
        }

        Ok(())
    }
}

impl FastDecoder {
    fn new(config: &VAEConfig, device: Device) -> Result<Self> {
        let conv_in = Conv2d::new(
            config.latent_channels,
            config.block_out_channels.last().unwrap().clone(),
            3,
            1,
            1,
            device.cuda_device().clone(),
        )?;

        // Mid block without attention
        let mid_block = FastDecoderBlock::new(
            config.block_out_channels.last().unwrap().clone(),
            config.block_out_channels.last().unwrap().clone(),
            config.layers_per_block,
            config.norm_num_groups,
            false,
            device.clone(),
        )?;

        // Up blocks
        let mut up_blocks = Vec::new();
        let reversed_channels: Vec<_> = config.block_out_channels.iter().rev().cloned().collect();

        for i in 0..reversed_channels.len() - 1 {
            let in_ch = reversed_channels[i];
            let out_ch = reversed_channels[i + 1];
            let is_last = i == reversed_channels.len() - 2;

            up_blocks.push(FastDecoderBlock::new(
                in_ch,
                out_ch,
                config.layers_per_block + 1,
                config.norm_num_groups,
                !is_last,
                device.clone(),
            )?);
        }

        let norm_out = GroupNorm::new(
            config.norm_num_groups,
            config.block_out_channels[0],
            1e-6,
            true,
            device.cuda_device().clone(),
        )?;
        let conv_out =
            Conv2d::new(config.block_out_channels[0], 3, 3, 1, 1, device.cuda_device().clone())?;

        Ok(Self { conv_in, up_blocks, mid_block, norm_out, conv_out })
    }

    fn forward(&self, z: &Tensor) -> Result<Tensor> {
        let mut h = self.conv_in.forward(z)?;

        // Mid block
        h = self.mid_block.forward(&h)?;

        // Up blocks
        for block in &self.up_blocks {
            h = block.forward(&h)?;
        }

        // Output
        h = self.norm_out.forward(&h)?;
        h = h.silu()?;
        self.conv_out.forward(&h)
    }

    fn load_weights(&mut self, weights: &HashMap<String, Tensor>) -> Result<()> {
        // Load conv_in
        if let Some(w) = weights.get("decoder.conv_in.weight") {
            self.conv_in.weight = to_dtype_aligned(w, DType::BF16)?;
        }
        if let Some(b) = weights.get("decoder.conv_in.bias") {
            self.conv_in.bias = Some(to_dtype_aligned(b, DType::BF16)?);
        }

        // Load mid block
        for (i, resnet) in self.mid_block.resnets.iter_mut().enumerate() {
            resnet.load_weights(weights, &format!("decoder.mid.block_{}", i + 1))?;
        }

        // Load up blocks
        for (i, block) in self.up_blocks.iter_mut().enumerate() {
            let prefix = format!("decoder.up.{}", i);
            for (j, resnet) in block.resnets.iter_mut().enumerate() {
                resnet.load_weights(weights, &format!("{}.block.{}", prefix, j))?;
            }

            if let Some(us) = &mut block.upsample {
                if let Some(w) = weights.get(&format!("{}.upsample.conv.weight", prefix)) {
                    us.weight = to_dtype_aligned(w, DType::BF16)?;
                }
                if let Some(b) = weights.get(&format!("{}.upsample.conv.bias", prefix)) {
                    us.bias = Some(to_dtype_aligned(b, DType::BF16)?);
                }
            }
        }

        // Load norm_out and conv_out
        if let Some(w) = weights.get("decoder.norm_out.weight") {
            self.norm_out.weight = Some(to_dtype_aligned(w, DType::BF16)?);
        }
        if let Some(b) = weights.get("decoder.norm_out.bias") {
            self.norm_out.bias = Some(to_dtype_aligned(b, DType::BF16)?);
        }

        if let Some(w) = weights.get("decoder.conv_out.weight") {
            self.conv_out.weight = to_dtype_aligned(w, DType::BF16)?;
        }
        if let Some(b) = weights.get("decoder.conv_out.bias") {
            self.conv_out.bias = Some(to_dtype_aligned(b, DType::BF16)?);
        }

        Ok(())
    }
}
