use crate::ops::{Conv2d, GroupNorm, LayerNorm, Linear};
use anyhow::anyhow;
use flame_core::device::Device;
use flame_core::Result;
use flame_core::{DType, Error, Shape, Tensor};
use std::{collections::HashMap, sync::Arc};

pub struct ResnetBlock {
    norm1: GroupNorm,
    conv1: Conv2d,
    norm2: GroupNorm,
    conv2: Conv2d,
    conv_shortcut: Option<Conv2d>,
}
pub struct AttentionBlock {
    norm: GroupNorm,
    q: Conv2d,
    k: Conv2d,
    v: Conv2d,
    proj_out: Conv2d,
    scale: f32,
}
pub struct EncoderBlock {
    resnets: Vec<ResnetBlock>,
    attentions: Vec<Option<AttentionBlock>>,
    downsample: Option<Conv2d>,
}
pub struct DecoderBlock {
    resnets: Vec<ResnetBlock>,
    attentions: Vec<Option<AttentionBlock>>,
    downsample: Option<Conv2d>,
    upsample: Option<Upsample>,
}
struct Upsample {
    conv: Conv2d,
}
pub struct AutoEncoderKL {
    encoder: Encoder,
    decoder: Decoder,
    quant_conv: Option<Conv2d>,
    post_quant_conv: Option<Conv2d>,
    config: VAEConfig,
    device: flame_core::device::Device,
}
struct Encoder {
    conv_in: Conv2d,
    down_blocks: Vec<EncoderBlock>,
    mid_block: MidBlock,
    norm_out: GroupNorm,
    conv_out: Conv2d,
}
struct Decoder {
    conv_in: Conv2d,
    up_blocks: Vec<DecoderBlock>,
    mid_block: MidBlock,
    norm_out: GroupNorm,
    conv_out: Conv2d,
}
struct MidBlock {
    resnet1: ResnetBlock,
    attn: AttentionBlock,
    resnet2: ResnetBlock,
}
pub struct DiagonalGaussianDistribution {
    mean: Tensor,
    logvar: Tensor,
}

// FLAME uses flame_core::device::Device instead of Device
/// VAE configuration
#[derive(Clone, Debug)]
pub struct VAEConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub block_out_channels: Vec<usize>,
    pub layers_per_block: usize,
    pub latent_channels: usize,
    pub norm_num_groups: usize,
    pub use_quant_conv: bool,
    pub use_post_quant_conv: bool,
    pub scaling_factor: f32,
}

impl Default for VAEConfig {
    fn default() -> Self {
        Self::sdxl()
    }
}

impl VAEConfig {
    /// SDXL VAE config
    pub fn sdxl() -> Self {
        Self {
            in_channels: 3,
            out_channels: 3,
            block_out_channels: vec![128, 256, 512, 512],
            layers_per_block: 2,
            latent_channels: 4,
            norm_num_groups: 32,
            use_quant_conv: true,
            use_post_quant_conv: true,
            scaling_factor: 0.18215,
        }
    }

    /// SD3 VAE config (16-channel)
    pub fn sd3() -> Self {
        Self {
            in_channels: 3,
            out_channels: 3,
            block_out_channels: vec![128, 256, 512, 512],
            layers_per_block: 2,
            latent_channels: 16,
            norm_num_groups: 32,
            use_quant_conv: false,
            use_post_quant_conv: false,
            scaling_factor: 1.5305,
        }
    }
}

/// ResNet block for VAE

impl ResnetBlock {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        norm_groups: usize,
        device: &flame_core::device::Device,
    ) -> Result<Self> {
        println!("ResnetBlock::new in_channels={} out_channels={}", in_channels, out_channels);
        let norm1 = GroupNorm::new(norm_groups, in_channels, 1e-6, true, DType::BF16, device.cuda_device_arc())?;
        let conv1 = Conv2d::new(in_channels, out_channels, 3, 1, 1, device.cuda_device_arc())?;
        let norm2 =
            GroupNorm::new(norm_groups, out_channels, 1e-6, true, DType::BF16, device.cuda_device_arc())?;
        let conv2 = Conv2d::new(out_channels, out_channels, 3, 1, 1, device.cuda_device_arc())?;

        let conv_shortcut = if in_channels != out_channels {
            Some(Conv2d::new(in_channels, out_channels, 1, 1, 0, device.cuda_device_arc())?)
        } else {
            None
        };

        Ok(Self { norm1, conv1, norm2, conv2, conv_shortcut })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        println!("ResnetBlock::forward x.shape={:?}", x.shape());
        let residual: Tensor = x.clone();
        let mut h = self.norm1.forward_nchw(x)?;
        h = h.silu()?;
        h = self.conv1.forward(&h)?;
        println!("ResnetBlock::forward after conv1: {:?}", h.shape());
        h = self.norm2.forward_nchw(&h)?;
        h = h.silu()?;
        h = self.conv2.forward(&h)?;
        println!("ResnetBlock::forward after conv2: {:?}", h.shape());

        let skip_connection =
            if let Some(conv) = &self.conv_shortcut { conv.forward(&residual)? } else { residual };

        h.add(&skip_connection)
    }
}

/// Attention block for VAE

impl AttentionBlock {
    pub fn new(
        channels: usize,
        norm_groups: usize,
        device: &flame_core::device::Device,
    ) -> Result<Self> {
        let norm = GroupNorm::new(norm_groups, channels, 1e-6, true, DType::BF16, device.cuda_device_arc())?;
        let q = Conv2d::new(channels, channels, 1, 1, 0, device.cuda_device_arc())?;
        let k = Conv2d::new(channels, channels, 1, 1, 0, device.cuda_device_arc())?;
        let v = Conv2d::new(channels, channels, 1, 1, 0, device.cuda_device_arc())?;
        let proj_out = Conv2d::new(channels, channels, 1, 1, 0, device.cuda_device_arc())?;
        let scale = (channels as f32).powf(-0.5);

        Ok(Self { norm, q, k, v, proj_out, scale })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual: Tensor = x.clone();
        let x = self.norm.forward_nchw(x)?;

        let shape = x.shape().dims();
        let (b, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);

        // Compute Q, K, V
        let q = self.q.forward(&x)?.reshape(&[b, c, h * w])?.transpose_dims(1, 2)?;
        let k = self.k.forward(&x)?.reshape(&[b, c, h * w])?;
        let v = self.v.forward(&x)?.reshape(&[b, c, h * w])?.transpose_dims(1, 2)?;

        // Attention scores
        let scores = q.bmm(&k)?;
        let scores = scores.mul_scalar(self.scale as f32)?;
        let attn = scores.softmax(-1)?;

        // Apply attention
        let out = attn.bmm(&v)?.transpose_dims(1, 2)?.reshape(&[b, c, h, w])?;

        let out = self.proj_out.forward(&out)?;

        out.add(&residual)
    }
}

/// Encoder block

impl EncoderBlock {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        num_layers: usize,
        norm_groups: usize,
        add_downsample: bool,
        has_attention: bool,
        device: &flame_core::device::Device,
    ) -> Result<Self> {
        let mut resnets = Vec::new();
        let mut attentions = Vec::new();

        for i in 0..num_layers {
            let in_ch = if i == 0 { in_channels } else { out_channels };
            resnets.push(ResnetBlock::new(in_ch, out_channels, norm_groups, device)?);

            if has_attention {
                attentions.push(Some(AttentionBlock::new(out_channels, norm_groups, device)?));
            } else {
                attentions.push(None);
            }
        }

        let downsample = if add_downsample {
            Some(Conv2d::new(out_channels, out_channels, 3, 2, 1, device.cuda_device_arc())?)
        } else {
            None
        };

        Ok(Self { resnets, attentions, downsample })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h: Tensor = x.clone();

        for (resnet, attn) in self.resnets.iter().zip(self.attentions.iter()) {
            h = resnet.forward(&h)?;
            if let Some(attention) = attn {
                h = attention.forward(&h)?;
            }
        }

        if let Some(downsample) = &self.downsample {
            h = downsample.forward(&h)?;
        }

        Ok(h)
    }
}

/// Decoder block

impl DecoderBlock {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        num_layers: usize,
        norm_groups: usize,
        add_upsample: bool,
        has_attention: bool,
        device: &flame_core::device::Device,
    ) -> Result<Self> {
        let mut resnets = Vec::new();
        let mut attentions = Vec::new();

        for i in 0..num_layers {
            let out_ch = if i == num_layers - 1 { out_channels } else { in_channels };
            resnets.push(ResnetBlock::new(in_channels, out_ch, norm_groups, device)?);

            if has_attention {
                attentions.push(Some(AttentionBlock::new(out_ch, norm_groups, device)?));
            } else {
                attentions.push(None);
            }
        }

        let upsample = if add_upsample {
            Some(Upsample::new(out_channels, out_channels, &device)?)
        } else {
            None
        };

        Ok(Self { resnets, attentions, downsample: None, upsample })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h: Tensor = x.clone();

        for (resnet, attn) in self.resnets.iter().zip(self.attentions.iter()) {
            h = resnet.forward(&h)?;
            if let Some(attention) = attn {
                h = attention.forward(&h)?;
            }
        }

        if let Some(upsample) = &self.upsample {
            h = upsample.forward(&h)?;
        }

        Ok(h)
    }
}

/// Upsampling layer

impl Upsample {
    fn new(
        in_channels: usize,
        out_channels: usize,
        device: &flame_core::device::Device,
    ) -> Result<Self> {
        let conv = Conv2d::new(in_channels, out_channels, 3, 1, 1, device.cuda_device_arc())?;
        Ok(Self { conv })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Nearest neighbor upsample followed by conv
        let shape = x.shape().dims();
        let upsampled =
            flame_core::cuda_ops::GpuOps::upsample2d_nearest(x, (shape[2] * 2, shape[3] * 2))?;
        self.conv.forward(&upsampled)
    }
}

/// FLAME AutoEncoderKL

impl AutoEncoderKL {
    pub fn new(
        config: VAEConfig,
        device: &flame_core::device::Device,
        weights: std::collections::HashMap<String, Tensor>,
    ) -> Result<Self> {
        let encoder = Encoder::new(&config, &device, &weights)?;
        let decoder = Decoder::new(&config, &device, &weights)?;

        let quant_conv = if config.use_quant_conv {
            let weight = weights.get("quant_conv.weight").ok_or_else(|| {
                flame_core::Error::InvalidOperation("quant_conv.weight not found".into())
            })?;
            let bias = weights.get("quant_conv.bias");
            let weight_shape = weight.shape().dims();
            Some(Conv2d::new(
                weight_shape[1],
                weight_shape[0],
                weight_shape[2],
                1,
                0,
                device.cuda_device_arc(),
            )?)
        } else {
            None
        };

        let post_quant_conv = if config.use_post_quant_conv {
            let weight = weights.get("post_quant_conv.weight").ok_or_else(|| {
                flame_core::Error::InvalidOperation(
                    "post_quant_conv.weight not found".to_string(),
                )
            })?;
            let bias = weights.get("post_quant_conv.bias");
            let weight_shape = weight.shape().dims();
            Some(Conv2d::new(
                weight_shape[1],
                weight_shape[0],
                weight_shape[2],
                1,
                0,
                device.cuda_device_arc(),
            )?)
        } else {
            None
        };

        Ok(Self { encoder, decoder, quant_conv, post_quant_conv, config, device: device.clone() })
    }

    /// Encode images to latents
    pub fn encode(&self, x: &Tensor) -> Result<DiagonalGaussianDistribution> {
        let h = self.encoder.forward(x)?;

        let moments = if let Some(qc) = &self.quant_conv { qc.forward(&h)? } else { h };

        DiagonalGaussianDistribution::new(moments)
    }

    /// Decode latents to images
    pub fn decode(&self, z: &Tensor) -> Result<Tensor> {
        // Scale latents
        let z = z.mul_scalar(1.0 / self.config.scaling_factor as f32)?;

        let h = if let Some(pqc) = &self.post_quant_conv { pqc.forward(&z)? } else { z };

        self.decoder.forward(&h)
    }

    /// Encode and get latent sample
    pub fn encode_to_latent(&self, x: &Tensor) -> Result<Tensor> {
        let dist = self.encode(x)?;
        let latent = dist.sample()?;
        // Apply scaling factor
        latent.mul_scalar(self.config.scaling_factor as f32)
    }

    /// Encode and get latent mode (no sampling)
    pub fn encode_to_latent_mode(&self, x: &Tensor) -> Result<Tensor> {
        let dist = self.encode(x)?;
        let latent = dist.mode()?;
        // Apply scaling factor
        latent.mul_scalar(self.config.scaling_factor as f32)
    }
}

/// Encoder

impl Encoder {
    fn new(
        config: &VAEConfig,
        device: &flame_core::device::Device,
        weights: &std::collections::HashMap<String, Tensor>,
    ) -> Result<Self> {
        let weight = weights.get("encoder.conv_in.weight").ok_or_else(|| {
            flame_core::Error::InvalidOperation("Missing encoder.conv_in.weight".into())
        })?;
        let weight_shape = weight.shape().dims();
        let conv_in = Conv2d::new(
            weight_shape[1],
            weight_shape[0],
            weight_shape[2],
            1,
            1,
            device.cuda_device_arc(),
        )?;

        // Create down blocks
        let mut down_blocks = Vec::new();
        let mut ch = config.block_out_channels[0];

        for (i, &out_ch) in config.block_out_channels.iter().enumerate() {
            let has_attention = i >= 2; // Only last blocks have attention
            let is_last = i == config.block_out_channels.len() - 1;

            down_blocks.push(EncoderBlock::new(
                ch,
                out_ch,
                config.layers_per_block,
                config.norm_num_groups,
                !is_last, // add downsample except for last
                has_attention,
                &device,
            )?);

            ch = out_ch;
        }

        // Mid block
        let mid_block = MidBlock::new(ch, config.norm_num_groups, &device)?;

        // Output layers
        let norm_out =
            GroupNorm::new(config.norm_num_groups, ch, 1e-6, true, DType::BF16, device.cuda_device_arc())?;

        let weight = weights.get("encoder.conv_out.weight").ok_or_else(|| {
            flame_core::Error::InvalidOperation("Missing encoder.conv_out.weight".into())
        })?;
        let weight_shape = weight.shape().dims();
        let conv_out = Conv2d::new(
            weight_shape[1],
            weight_shape[0],
            weight_shape[2],
            1,
            1,
            device.cuda_device_arc(),
        )?;

        Ok(Self { conv_in, down_blocks, mid_block, norm_out, conv_out })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = self.conv_in.forward(x)?;

        // Down blocks
        for block in &self.down_blocks {
            h = block.forward(&h)?;
        }

        // Middle block
        h = self.mid_block.forward(&h)?;

        // Output
        h = self.norm_out.forward_nchw(&h)?;
        h = h.silu()?;
        self.conv_out.forward(&h)
    }
}

/// Decoder

impl Decoder {
    fn new(
        config: &VAEConfig,
        device: &flame_core::device::Device,
        weights: &std::collections::HashMap<String, Tensor>,
    ) -> Result<Self> {
        let weight = weights.get("decoder.conv_in.weight").ok_or_else(|| {
            flame_core::Error::InvalidOperation("Missing decoder.conv_in.weight".into())
        })?;
        let weight_shape = weight.shape().dims();
        println!("Decoder::new conv_in weight_shape: {:?}", weight_shape);
        let conv_in = Conv2d::new(
            weight_shape[1],
            weight_shape[0],
            weight_shape[2],
            1,
            1,
            device.cuda_device_arc(),
        )?;
        println!("Decoder::new conv_in config: in={} out={}", conv_in.config.in_channels, conv_in.config.out_channels);

        // Create up blocks (reversed order)
        let mut up_blocks = Vec::new();
        let block_out_channels: Vec<_> = config.block_out_channels.iter().rev().cloned().collect();
        println!("Decoder::new block_out_channels (reversed): {:?}", block_out_channels);
        let mut ch = block_out_channels[0];

        // Mid block
        let mid_block = MidBlock::new(ch, config.norm_num_groups, &device)?;

        for (i, &out_ch) in block_out_channels.iter().enumerate() {
            let has_attention = i < 2; // First blocks have attention (reversed)
            let is_last = i == block_out_channels.len() - 1;

            up_blocks.push(DecoderBlock::new(
                ch,
                out_ch,
                config.layers_per_block,
                config.norm_num_groups,
                !is_last, // add upsample except for last
                has_attention,
                &device,
            )?);

            ch = out_ch;
        }

        // Output layers
        let norm_out =
            GroupNorm::new(config.norm_num_groups, ch, 1e-6, true, DType::BF16, device.cuda_device_arc())?;

        let weight = weights.get("decoder.conv_out.weight").ok_or_else(|| {
            flame_core::Error::InvalidOperation("Missing decoder.conv_out.weight".into())
        })?;
        let weight_shape = weight.shape().dims();
        let conv_out = Conv2d::new(
            weight_shape[1],
            weight_shape[0],
            weight_shape[2],
            1,
            0,
            device.cuda_device_arc(),
        )?;

        Ok(Self { conv_in, up_blocks, mid_block, norm_out, conv_out })
    }

    fn forward(&self, z: &Tensor) -> Result<Tensor> {
        let mut h = self.conv_in.forward(z)?;
        println!("Decoder::forward after conv_in: {:?}", h.shape());

        // Middle block
        h = self.mid_block.forward(&h)?;

        // Up blocks
        for block in &self.up_blocks {
            h = block.forward(&h)?;
        }

        // Output
        h = self.norm_out.forward_nchw(&h)?;
        h = h.silu()?;
        self.conv_out.forward(&h)
    }
}

/// Middle block

impl MidBlock {
    fn new(
        channels: usize,
        norm_groups: usize,
        device: &flame_core::device::Device,
    ) -> Result<Self> {
        println!("MidBlock::new channels={}", channels);
        Ok(Self {
            resnet1: ResnetBlock::new(channels, channels, norm_groups, device)?,
            attn: AttentionBlock::new(channels, norm_groups, device)?,
            resnet2: ResnetBlock::new(channels, channels, norm_groups, device)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        println!("MidBlock::forward x.shape={:?}", x.shape());
        let h = self.resnet1.forward(x)?;
        let h = self.attn.forward(&h)?;
        self.resnet2.forward(&h)
    }
}

/// Diagonal Gaussian distribution for VAE

impl DiagonalGaussianDistribution {
    fn new(parameters: Tensor) -> Result<Self> {
        let chunks = parameters.chunk(2, 1)?;
        if chunks.len() != 2 {
            return Err(flame_core::Error::InvalidOperation(
                "Expected 2 chunks for mean and logvar".to_string(),
            ));
        }

        Ok(Self { mean: chunks[0].clone(), logvar: chunks[1].clone() })
    }

    pub fn sample(&self) -> Result<Tensor> {
        let std = self.logvar.mul_scalar(0.5 as f32)?.exp()?;
        let eps = Tensor::randn(self.mean.shape().clone(), 0.0, 1.0, self.mean.device().clone())?;
        self.mean.add(&std.mul(&eps)?)
    }

    pub fn mode(&self) -> Result<Tensor> {
        Ok(self.mean.clone())
    }

    pub fn kl(&self) -> Result<Tensor> {
        // KL divergence: -0.5 * sum(1 + log(var) - mu^2 - var)
        let var = self.logvar.exp()?;
        let kl = self
            .mean
            .mul(&self.mean)?
            .add(&var)?
            .sub(&self.logvar)?
            .add_scalar(-1.0)?
            .mul_scalar(-0.5 as f32)?;
        kl.sum()
    }
}

// GroupNorm is imported from flame_core::group_norm::GroupNorm
