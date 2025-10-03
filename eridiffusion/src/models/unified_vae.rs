//! VAE (Variational Autoencoder) implementation for diffusion models
//! Supports both 4-channel (SD1.x/SDXL) and 16-channel (SD3.5/Flux) VAEs

use crate::ops::{Conv2d, GroupNorm, Linear};
use flame_core::device::Device;
use flame_core::upsampling::ConvTranspose2d;
use flame_core::{CudaDevice, DType, Error, Result, Shape, Tensor};
use std::sync::Arc;

/// VAE Configuration
#[derive(Clone)]
pub struct VAEConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub latent_channels: usize,
    pub block_out_channels: Vec<usize>,
    pub layers_per_block: usize,
    pub norm_num_groups: usize,
    pub scaling_factor: f32,
    pub force_upcast: bool,
}

impl VAEConfig {
    /// Config for SD 1.x/2.x VAE (4-channel)
    pub fn sd_vae() -> Self {
        Self {
            in_channels: 3,
            out_channels: 3,
            latent_channels: 4,
            block_out_channels: vec![128, 256, 512, 512],
            layers_per_block: 2,
            norm_num_groups: 32,
            scaling_factor: 0.18215,
            force_upcast: false,
        }
    }

    /// Config for SDXL VAE (4-channel, larger)
    pub fn sdxl_vae() -> Self {
        Self {
            in_channels: 3,
            out_channels: 3,
            latent_channels: 4,
            block_out_channels: vec![128, 256, 512, 512],
            layers_per_block: 2,
            norm_num_groups: 32,
            scaling_factor: 0.13025,
            force_upcast: true,
        }
    }

    /// Config for SD3.5/Flux VAE (16-channel)
    pub fn sd3_vae() -> Self {
        Self {
            in_channels: 3,
            out_channels: 3,
            latent_channels: 16,
            block_out_channels: vec![128, 256, 512, 512],
            layers_per_block: 2,
            norm_num_groups: 32,
            scaling_factor: 1.5305,
            force_upcast: false,
        }
    }
}

/// VAE ResNet Block
pub struct ResnetBlock2D {
    pub norm1: GroupNorm,
    pub conv1: Conv2d,
    pub norm2: GroupNorm,
    pub conv2: Conv2d,
    pub conv_shortcut: Option<Conv2d>,
    pub dropout_prob: f32,
}

impl ResnetBlock2D {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        dropout: f32,
        norm_groups: usize,
        device: Arc<CudaDevice>,
    ) -> Result<Self> {
        let norm1 = GroupNorm::new(norm_groups, in_channels, 1e-6, true, device.clone())?;
        let conv1 = Conv2d::new(in_channels, out_channels, 3, 1, 1, device.clone())?;

        let norm2 = GroupNorm::new(norm_groups, out_channels, 1e-6, true, device.clone())?;
        let conv2 = Conv2d::new(out_channels, out_channels, 3, 1, 1, device.clone())?;

        let conv_shortcut = if in_channels != out_channels {
            Some(Conv2d::new(in_channels, out_channels, 1, 1, 0, device.clone())?)
        } else {
            None
        };

        Ok(Self { norm1, conv1, norm2, conv2, conv_shortcut, dropout_prob: dropout })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x.clone();

        let h = self.norm1.forward(x)?;
        let h = h.silu()?;
        let h = self.conv1.forward(&h)?;

        let h = self.norm2.forward(&h)?;
        let h = h.silu()?;
        let h = self.conv2.forward(&h)?;

        let residual =
            if let Some(conv) = &self.conv_shortcut { conv.forward(&residual)? } else { residual };

        h.add(&residual)
    }
}

/// VAE Attention Block
pub struct AttentionBlock {
    pub group_norm: GroupNorm,
    pub query: Conv2d,
    pub key: Conv2d,
    pub value: Conv2d,
    pub proj_attn: Conv2d,
    pub num_heads: usize,
}

impl AttentionBlock {
    pub fn new(
        channels: usize,
        num_heads: usize,
        norm_groups: usize,
        device: Arc<CudaDevice>,
    ) -> Result<Self> {
        Ok(Self {
            group_norm: GroupNorm::new(norm_groups, channels, 1e-6, true, device.clone())?,
            query: Conv2d::new(channels, channels, 1, 1, 0, device.clone())?,
            key: Conv2d::new(channels, channels, 1, 1, 0, device.clone())?,
            value: Conv2d::new(channels, channels, 1, 1, 0, device.clone())?,
            proj_attn: Conv2d::new(channels, channels, 1, 1, 0, device.clone())?,
            num_heads,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x.clone();
        let dims = x.shape().dims();
        if dims.len() != 4 {
            return Err(flame_core::Error::InvalidOperation("Expected 4D tensor".into()));
        }
        let (b, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);

        let x = self.group_norm.forward(x)?;

        let q = self.query.forward(&x)?;
        let k = self.key.forward(&x)?;
        let v = self.value.forward(&x)?;

        // Reshape for multi-head attention
        let head_dim = c / self.num_heads;
        let q = q.reshape(&[b, self.num_heads, head_dim, h * w])?;
        let k = k.reshape(&[b, self.num_heads, head_dim, h * w])?;
        let v = v.reshape(&[b, self.num_heads, head_dim, h * w])?;

        // Compute attention
        let scale = (head_dim as f32).sqrt();
        let scores = q.transpose_dims(2, 3)?.matmul(&k)?.div_scalar(scale)?;
        let attn = scores.softmax(3)?;
        let out = attn.matmul(&v.transpose_dims(2, 3)?)?;

        // Reshape back
        let out = out.transpose_dims(2, 3)?.reshape(&[b, c, h, w])?;
        let out = self.proj_attn.forward(&out)?;

        out.add(&residual)
    }
}

/// VAE Encoder
pub struct Encoder {
    pub conv_in: Conv2d,
    pub down_blocks: Vec<(Vec<ResnetBlock2D>, Option<Conv2d>)>,
    pub mid_block: (ResnetBlock2D, AttentionBlock, ResnetBlock2D),
    pub norm_out: GroupNorm,
    pub conv_out: Conv2d,
}

impl Encoder {
    pub fn new(config: &VAEConfig, device: Arc<CudaDevice>) -> Result<Self> {
        let conv_in =
            Conv2d::new(config.in_channels, config.block_out_channels[0], 3, 1, 1, device.clone())?;

        let mut down_blocks = Vec::new();
        let mut ch = config.block_out_channels[0];

        for (i, &out_ch) in config.block_out_channels.iter().enumerate() {
            let mut resnets = Vec::new();

            for j in 0..config.layers_per_block {
                let in_ch = if j == 0 { ch } else { out_ch };
                resnets.push(ResnetBlock2D::new(
                    in_ch,
                    out_ch,
                    0.0,
                    config.norm_num_groups,
                    device.clone(),
                )?);
            }

            let downsample = if i < config.block_out_channels.len() - 1 {
                Some(Conv2d::new(out_ch, out_ch, 3, 2, 1, device.clone())?)
            } else {
                None
            };

            down_blocks.push((resnets, downsample));
            ch = out_ch;
        }

        // Mid block
        let mid_res1 = ResnetBlock2D::new(ch, ch, 0.0, config.norm_num_groups, device.clone())?;
        let mid_attn = AttentionBlock::new(ch, 8, config.norm_num_groups, device.clone())?;
        let mid_res2 = ResnetBlock2D::new(ch, ch, 0.0, config.norm_num_groups, device.clone())?;

        let norm_out = GroupNorm::new(config.norm_num_groups, ch, 1e-6, true, device.clone())?;
        let conv_out = Conv2d::new(ch, 2 * config.latent_channels, 3, 1, 1, device.clone())?;

        Ok(Self {
            conv_in,
            down_blocks,
            mid_block: (mid_res1, mid_attn, mid_res2),
            norm_out,
            conv_out,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = self.conv_in.forward(x)?;

        // Down blocks
        for (resnets, downsample) in &self.down_blocks {
            for resnet in resnets {
                h = resnet.forward(&h)?;
            }
            if let Some(down) = downsample {
                h = down.forward(&h)?;
            }
        }

        // Mid block
        h = self.mid_block.0.forward(&h)?;
        h = self.mid_block.1.forward(&h)?;
        h = self.mid_block.2.forward(&h)?;

        // Output
        h = self.norm_out.forward(&h)?;
        h = h.silu()?;
        self.conv_out.forward(&h)
    }
}

/// VAE Decoder
pub struct Decoder {
    pub conv_in: Conv2d,
    pub mid_block: (ResnetBlock2D, AttentionBlock, ResnetBlock2D),
    pub up_blocks: Vec<(Vec<ResnetBlock2D>, Option<Conv2d>)>,
    pub norm_out: GroupNorm,
    pub conv_out: Conv2d,
}

impl Decoder {
    pub fn new(config: &VAEConfig, device: Arc<CudaDevice>) -> Result<Self> {
        let ch = config.block_out_channels[config.block_out_channels.len() - 1];
        let conv_in = Conv2d::new(config.latent_channels, ch, 3, 1, 1, device.clone())?;

        // Mid block
        let mid_res1 = ResnetBlock2D::new(ch, ch, 0.0, config.norm_num_groups, device.clone())?;
        let mid_attn = AttentionBlock::new(ch, 8, config.norm_num_groups, device.clone())?;
        let mid_res2 = ResnetBlock2D::new(ch, ch, 0.0, config.norm_num_groups, device.clone())?;

        // Up blocks
        let mut up_blocks = Vec::new();
        let reversed_channels: Vec<_> = config.block_out_channels.iter().rev().cloned().collect();

        for (i, &out_ch) in reversed_channels.iter().enumerate() {
            let in_ch = if i == 0 { ch } else { reversed_channels[i - 1] };
            let mut resnets = Vec::new();

            for _ in 0..config.layers_per_block + 1 {
                resnets.push(ResnetBlock2D::new(
                    in_ch,
                    out_ch,
                    0.0,
                    config.norm_num_groups,
                    device.clone(),
                )?);
            }

            let upsample = if i < reversed_channels.len() - 1 {
                // TODO: Replace with proper ConvTranspose2d once we understand the API
                None // Temporarily disable upsampling until ConvTranspose2d is properly integrated
            } else {
                None
            };

            up_blocks.push((resnets, upsample));
        }

        let final_ch = config.block_out_channels[0];
        let norm_out =
            GroupNorm::new(config.norm_num_groups, final_ch, 1e-6, true, device.clone())?;
        let conv_out = Conv2d::new(final_ch, config.out_channels, 3, 1, 1, device.clone())?;

        Ok(Self {
            conv_in,
            mid_block: (mid_res1, mid_attn, mid_res2),
            up_blocks,
            norm_out,
            conv_out,
        })
    }

    pub fn forward(&self, z: &Tensor) -> Result<Tensor> {
        let mut h = self.conv_in.forward(z)?;

        // Mid block
        h = self.mid_block.0.forward(&h)?;
        h = self.mid_block.1.forward(&h)?;
        h = self.mid_block.2.forward(&h)?;

        // Up blocks
        for (resnets, upsample) in &self.up_blocks {
            for resnet in resnets {
                h = resnet.forward(&h)?;
            }
            if let Some(up) = upsample {
                h = up.forward(&h)?;
            }
        }

        // Output
        h = self.norm_out.forward(&h)?;
        h = h.silu()?;
        self.conv_out.forward(&h)
    }
}

/// Complete VAE model
pub struct VAE {
    pub config: VAEConfig,
    pub encoder: Encoder,
    pub decoder: Decoder,
    pub quant_conv: Conv2d,
    pub post_quant_conv: Conv2d,
}

impl VAE {
    pub fn new(config: VAEConfig, device: Arc<CudaDevice>) -> Result<Self> {
        let encoder = Encoder::new(&config, device.clone())?;
        let decoder = Decoder::new(&config, device.clone())?;

        let quant_conv = Conv2d::new(
            2 * config.latent_channels,
            2 * config.latent_channels,
            1,
            1,
            0,
            device.clone(),
        )?;
        let post_quant_conv =
            Conv2d::new(config.latent_channels, config.latent_channels, 1, 1, 0, device.clone())?;

        Ok(Self { config, encoder, decoder, quant_conv, post_quant_conv })
    }

    pub fn encode(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let h = self.encoder.forward(x)?;
        let moments = self.quant_conv.forward(&h)?;

        let chunks = moments.chunk(2, 1)?;
        let (mean, logvar) = (chunks[0].clone(), chunks[1].clone());
        Ok((mean, logvar))
    }

    pub fn decode(&self, z: &Tensor) -> Result<Tensor> {
        let z = self.post_quant_conv.forward(z)?;
        let x = self.decoder.forward(&z)?;

        // Apply scaling
        x.div_scalar(self.config.scaling_factor)
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (mean, _logvar) = self.encode(x)?;
        self.decode(&mean)
    }
}
