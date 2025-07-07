// flux_vae_loader.rs
// Working code to load Flux VAE from ae.safetensors

use candle_core::{Device, DType, Tensor, D};
use candle_nn::{VarBuilder, Module};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct AutoencoderKLConfig {
    pub block_out_channels: Vec<usize>,
    pub layers_per_block: usize,
    pub latent_channels: usize,
    pub norm_num_groups: usize,
    pub scaling_factor: f64,
}

impl Default for AutoencoderKLConfig {
    fn default() -> Self {
        Self {
            block_out_channels: vec![128, 256, 512, 512],
            layers_per_block: 2,
            latent_channels: 16,  // Flux uses 16
            norm_num_groups: 32,
            scaling_factor: 0.3611,
        }
    }
}

pub struct ResnetBlock2D {
    norm1: candle_nn::GroupNorm,
    conv1: candle_nn::Conv2d,
    norm2: candle_nn::GroupNorm,
    conv2: candle_nn::Conv2d,
    conv_shortcut: Option<candle_nn::Conv2d>,
}

impl ResnetBlock2D {
    fn new(
        in_channels: usize,
        out_channels: usize,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        let norm1 = candle_nn::group_norm(32, in_channels, 1e-6, vb.pp("norm1"))?;
        let conv1 = candle_nn::conv2d(in_channels, out_channels, 3, Default::default(), vb.pp("conv1"))?;
        let norm2 = candle_nn::group_norm(32, out_channels, 1e-6, vb.pp("norm2"))?;
        let conv2 = candle_nn::conv2d(out_channels, out_channels, 3, Default::default(), vb.pp("conv2"))?;
        
        let conv_shortcut = if in_channels != out_channels {
            Some(candle_nn::conv2d(in_channels, out_channels, 1, Default::default(), vb.pp("conv_shortcut"))?)
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
}

impl Module for ResnetBlock2D {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let residual = x;
        let x = self.norm1.forward(x)?;
        let x = x.silu()?;
        let x = self.conv1.forward(&x)?;
        let x = self.norm2.forward(&x)?;
        let x = x.silu()?;
        let x = self.conv2.forward(&x)?;
        
        let residual = if let Some(conv) = &self.conv_shortcut {
            conv.forward(residual)?
        } else {
            residual.clone()
        };
        
        x + residual
    }
}

pub struct Decoder {
    conv_in: candle_nn::Conv2d,
    mid_block_1: ResnetBlock2D,
    mid_block_2: ResnetBlock2D,
    up_blocks: Vec<Vec<ResnetBlock2D>>,
    up_samples: Vec<candle_nn::ConvTranspose2d>,
    conv_norm_out: candle_nn::GroupNorm,
    conv_out: candle_nn::Conv2d,
}

impl Decoder {
    fn new(config: &AutoencoderKLConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let latent_channels = config.latent_channels;
        let out_channels = 3;  // RGB
        let block_out_channels = &config.block_out_channels;
        
        let conv_in = candle_nn::conv2d(
            latent_channels,
            block_out_channels[block_out_channels.len() - 1],
            3,
            Default::default(),
            vb.pp("conv_in"),
        )?;
        
        // Mid blocks
        let mid_channels = block_out_channels[block_out_channels.len() - 1];
        let mid_block_1 = ResnetBlock2D::new(mid_channels, mid_channels, vb.pp("mid_block.resnets.0"))?;
        let mid_block_2 = ResnetBlock2D::new(mid_channels, mid_channels, vb.pp("mid_block.resnets.1"))?;
        
        // Up blocks
        let mut up_blocks = Vec::new();
        let mut up_samples = Vec::new();
        
        for (i, (&in_channels, &out_channels)) in block_out_channels.iter()
            .rev()
            .zip(block_out_channels.iter().rev().skip(1))
            .enumerate()
        {
            let mut block_resnets = Vec::new();
            for j in 0..config.layers_per_block {
                let res_block = ResnetBlock2D::new(
                    if j == 0 { in_channels } else { out_channels },
                    out_channels,
                    vb.pp(&format!("up_blocks.{}.resnets.{}", i, j)),
                )?;
                block_resnets.push(res_block);
            }
            up_blocks.push(block_resnets);
            
            if i < block_out_channels.len() - 1 {
                let upsample = candle_nn::conv_transpose2d(
                    out_channels,
                    out_channels,
                    2,
                    candle_nn::ConvTranspose2dConfig {
                        stride: 2,
                        ..Default::default()
                    },
                    vb.pp(&format!("up_blocks.{}.upsamplers.0", i)),
                )?;
                up_samples.push(upsample);
            }
        }
        
        let conv_norm_out = candle_nn::group_norm(32, block_out_channels[0], 1e-6, vb.pp("conv_norm_out"))?;
        let conv_out = candle_nn::conv2d(block_out_channels[0], out_channels, 3, Default::default(), vb.pp("conv_out"))?;
        
        Ok(Self {
            conv_in,
            mid_block_1,
            mid_block_2,
            up_blocks,
            up_samples,
            conv_norm_out,
            conv_out,
        })
    }
    
    fn forward(&self, z: &Tensor) -> candle_core::Result<Tensor> {
        let mut h = self.conv_in.forward(z)?;
        
        // Mid blocks
        h = self.mid_block_1.forward(&h)?;
        h = self.mid_block_2.forward(&h)?;
        
        // Up blocks
        for (block_resnets, upsample) in self.up_blocks.iter().zip(self.up_samples.iter()) {
            for resnet in block_resnets {
                h = resnet.forward(&h)?;
            }
            h = upsample.forward(&h)?;
        }
        
        // Final blocks if any
        if self.up_blocks.len() > self.up_samples.len() {
            for resnet in self.up_blocks.last().unwrap() {
                h = resnet.forward(&h)?;
            }
        }
        
        h = self.conv_norm_out.forward(&h)?;
        h = h.silu()?;
        h = self.conv_out.forward(&h)?;
        
        Ok(h)
    }
}

pub struct AutoencoderKL {
    decoder: Decoder,
    config: AutoencoderKLConfig,
}

impl AutoencoderKL {
    pub fn new(config: &AutoencoderKLConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let decoder = Decoder::new(config, vb.pp("decoder"))?;
        
        Ok(Self {
            decoder,
            config: config.clone(),
        })
    }
    
    pub fn decode(&self, z: &Tensor) -> candle_core::Result<Tensor> {
        // Scale latents
        let z = (z / self.config.scaling_factor)?;
        
        // Decode
        let decoded = self.decoder.forward(&z)?;
        
        // Convert from [-1, 1] to [0, 1]
        decoded.affine(0.5, 0.5)
    }
}

/// Load Flux VAE from ae.safetensors
pub fn load_flux_vae(
    vae_path: &Path,
    device: &Device,
) -> candle_core::Result<AutoencoderKL> {
    println!("Loading Flux VAE from: {:?}", vae_path);
    
    let vb = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(&[vae_path], DType::F32, device)?
    };
    
    let config = AutoencoderKLConfig::default();
    AutoencoderKL::new(&config, vb)
}

// Example usage
pub fn decode_latents(
    vae: &AutoencoderKL,
    latents: &Tensor,
) -> candle_core::Result<Tensor> {
    // Latents should be [batch, 16, height/8, width/8]
    let images = vae.decode(latents)?;
    
    // Output is [batch, 3, height, width] in range [0, 1]
    Ok(images)
}
