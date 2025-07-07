//! Complete Flux VAE implementation with proper channel handling

use candle_core::{Device, DType, Module, Result, Tensor, D};
use candle_nn::{Conv2d, Conv2dConfig, VarBuilder, GroupNorm, Activation};

/// Flux VAE configuration
#[derive(Debug, Clone)]
pub struct FluxVAEConfig {
   pub in_channels: usize,
   pub out_channels: usize,
   pub latent_channels: usize,
   pub block_out_channels: Vec<usize>,
   pub layers_per_block: usize,
   pub norm_num_groups: usize,
   pub scaling_factor: f32,
}

impl Default for FluxVAEConfig {
   fn default() -> Self {
       Self {
           in_channels: 3,
           out_channels: 3,
           latent_channels: 16,
           block_out_channels: vec![128, 256, 512, 512],
           layers_per_block: 2,
           norm_num_groups: 32,
           scaling_factor: 0.3611,
       }
   }
}

/// ResNet block for VAE
pub struct ResnetBlock {
   norm1: GroupNorm,
   conv1: Conv2d,
   norm2: GroupNorm,
   conv2: Conv2d,
   conv_shortcut: Option<Conv2d>,
}

impl ResnetBlock {
   pub fn new(
       in_channels: usize,
       out_channels: usize,
       vb: VarBuilder,
   ) -> Result<Self> {
       let norm1 = GroupNorm::new(32, in_channels, 1e-6, vb.pp("norm1"))?;
       let conv1 = Conv2d::new(
           in_channels,
           out_channels,
           3,
           Conv2dConfig {
               padding: 1,
               ..Default::default()
           },
           vb.pp("conv1"),
       )?;
       
       let norm2 = GroupNorm::new(32, out_channels, 1e-6, vb.pp("norm2"))?;
       let conv2 = Conv2d::new(
           out_channels,
           out_channels,
           3,
           Conv2dConfig {
               padding: 1,
               ..Default::default()
           },
           vb.pp("conv2"),
       )?;
       
       let conv_shortcut = if in_channels != out_channels {
           Some(Conv2d::new(
               in_channels,
               out_channels,
               1,
               Default::default(),
               vb.pp("conv_shortcut"),
           )?)
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

impl Module for ResnetBlock {
   fn forward(&self, x: &Tensor) -> Result<Tensor> {
       let hidden = self.norm1.forward(x)?;
       let hidden = hidden.silu()?;
       let hidden = self.conv1.forward(&hidden)?;
       
       let hidden = self.norm2.forward(&hidden)?;
       let hidden = hidden.silu()?;
       let hidden = self.conv2.forward(&hidden)?;
       
       let shortcut = if let Some(ref conv) = self.conv_shortcut {
           conv.forward(x)?
       } else {
           x.clone()
       };
       
       hidden + shortcut
   }
}

/// Attention block for VAE
pub struct AttentionBlock {
   group_norm: GroupNorm,
   query: Conv2d,
   key: Conv2d,
   value: Conv2d,
   proj_attn: Conv2d,
   num_heads: usize,
}

impl AttentionBlock {
   pub fn new(channels: usize, num_heads: usize, vb: VarBuilder) -> Result<Self> {
       let group_norm = GroupNorm::new(32, channels, 1e-6, vb.pp("group_norm"))?;
       let query = Conv2d::new(channels, channels, 1, Default::default(), vb.pp("query"))?;
       let key = Conv2d::new(channels, channels, 1, Default::default(), vb.pp("key"))?;
       let value = Conv2d::new(channels, channels, 1, Default::default(), vb.pp("value"))?;
       let proj_attn = Conv2d::new(channels, channels, 1, Default::default(), vb.pp("proj_attn"))?;
       
       Ok(Self {
           group_norm,
           query,
           key,
           value,
           proj_attn,
           num_heads,
       })
   }
}

impl Module for AttentionBlock {
   fn forward(&self, x: &Tensor) -> Result<Tensor> {
       let residual = x;
       let (b, c, h, w) = x.dims4()?;
       
       let x = self.group_norm.forward(x)?;
       
       let q = self.query.forward(&x)?;
       let k = self.key.forward(&x)?;
       let v = self.value.forward(&x)?;
       
       // Reshape for attention
       let head_dim = c / self.num_heads;
       let q = q.reshape((b, self.num_heads, head_dim, h * w))?
           .transpose(2, 3)?;
       let k = k.reshape((b, self.num_heads, head_dim, h * w))?
           .transpose(2, 3)?;
       let v = v.reshape((b, self.num_heads, head_dim, h * w))?
           .transpose(2, 3)?;
       
       // Attention
       let scale = 1.0 / (head_dim as f64).sqrt();
       let attn = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * scale)?;
       let attn = candle_nn::ops::softmax(&attn, D::Minus1)?;
       let attn = attn.matmul(&v)?;
       
       // Reshape back
       let attn = attn.transpose(2, 3)?
           .reshape((b, c, h, w))?;
       
       let attn = self.proj_attn.forward(&attn)?;
       
       attn + residual
   }
}

/// Mid block with attention
pub struct MidBlock {
   resnet1: ResnetBlock,
   attn: AttentionBlock,
   resnet2: ResnetBlock,
}

impl MidBlock {
   pub fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
       let resnet1 = ResnetBlock::new(channels, channels, vb.pp("resnets.0"))?;
       let attn = AttentionBlock::new(channels, 1, vb.pp("attentions.0"))?;
       let resnet2 = ResnetBlock::new(channels, channels, vb.pp("resnets.1"))?;
       
       Ok(Self { resnet1, attn, resnet2 })
   }
}

impl Module for MidBlock {
   fn forward(&self, x: &Tensor) -> Result<Tensor> {
       let x = self.resnet1.forward(x)?;
       let x = self.attn.forward(&x)?;
       self.resnet2.forward(&x)
   }
}

/// Encoder block
pub struct EncoderBlock {
   resnets: Vec<ResnetBlock>,
   downsampler: Option<Conv2d>,
}

impl EncoderBlock {
   pub fn new(
       in_channels: usize,
       out_channels: usize,
       num_layers: usize,
       add_downsample: bool,
       vb: VarBuilder,
   ) -> Result<Self> {
       let mut resnets = Vec::new();
       
       // First resnet handles channel change
       resnets.push(ResnetBlock::new(
           in_channels,
           out_channels,
           vb.pp("resnets.0"),
       )?);
       
       // Rest maintain same channels
       for i in 1..num_layers {
           resnets.push(ResnetBlock::new(
               out_channels,
               out_channels,
               vb.pp(format!("resnets.{}", i)),
           )?);
       }
       
       let downsampler = if add_downsample {
           Some(Conv2d::new(
               out_channels,
               out_channels,
               3,
               Conv2dConfig {
                   stride: 2,
                   padding: 1,
                   ..Default::default()
               },
               vb.pp("downsamplers.0"),
           )?)
       } else {
           None
       };
       
       Ok(Self { resnets, downsampler })
   }
}

impl Module for EncoderBlock {
   fn forward(&self, x: &Tensor) -> Result<Tensor> {
       let mut hidden = x.clone();
       
       for resnet in &self.resnets {
           hidden = resnet.forward(&hidden)?;
       }
       
       if let Some(ref downsampler) = self.downsampler {
           hidden = downsampler.forward(&hidden)?;
       }
       
       Ok(hidden)
   }
}

/// Decoder block
pub struct DecoderBlock {
   resnets: Vec<ResnetBlock>,
   upsampler: Option<Conv2d>,
}

impl DecoderBlock {
   pub fn new(
       in_channels: usize,
       out_channels: usize,
       num_layers: usize,
       add_upsample: bool,
       vb: VarBuilder,
   ) -> Result<Self> {
       let mut resnets = Vec::new();
       
       // All resnets go from in_channels to out_channels
       for i in 0..num_layers {
           let resnet_in = if i == 0 { in_channels } else { out_channels };
           resnets.push(ResnetBlock::new(
               resnet_in,
               out_channels,
               vb.pp(format!("resnets.{}", i)),
           )?);
       }
       
       let upsampler = if add_upsample {
           Some(Conv2d::new(
               out_channels,
               out_channels,
               3,
               Conv2dConfig {
                   padding: 1,
                   ..Default::default()
               },
               vb.pp("upsamplers.0"),
           )?)
       } else {
           None
       };
       
       Ok(Self { resnets, upsampler })
   }
}

impl Module for DecoderBlock {
   fn forward(&self, x: &Tensor) -> Result<Tensor> {
       let mut hidden = x.clone();
       
       for resnet in &self.resnets {
           hidden = resnet.forward(&hidden)?;
       }
       
       if let Some(ref upsampler) = self.upsampler {
           let (b, c, h, w) = hidden.dims4()?;
           hidden = hidden.upsample_nearest2d(h * 2, w * 2)?;
           hidden = upsampler.forward(&hidden)?;
       }
       
       Ok(hidden)
   }
}

/// VAE Encoder
pub struct FluxVAEEncoder {
   conv_in: Conv2d,
   down_blocks: Vec<EncoderBlock>,
   mid_block: MidBlock,
   conv_norm_out: GroupNorm,
   conv_out: Conv2d,
}

impl FluxVAEEncoder {
   pub fn new(config: &FluxVAEConfig, vb: VarBuilder) -> Result<Self> {
       let conv_in = Conv2d::new(
           config.in_channels,
           config.block_out_channels[0],
           3,
           Conv2dConfig {
               padding: 1,
               ..Default::default()
           },
           vb.pp("conv_in"),
       )?;
       
       let mut down_blocks = Vec::new();
       let mut current_channels = config.block_out_channels[0];
       
       for (i, &out_channels) in config.block_out_channels.iter().enumerate() {
           let is_final = i == config.block_out_channels.len() - 1;
           
           down_blocks.push(EncoderBlock::new(
               current_channels,
               out_channels,
               config.layers_per_block,
               !is_final,
               vb.pp(format!("down_blocks.{}", i)),
           )?);
           
           current_channels = out_channels;
       }
       
       let mid_block = MidBlock::new(current_channels, vb.pp("mid_block"))?;
       
       let conv_norm_out = GroupNorm::new(
           config.norm_num_groups,
           current_channels,
           1e-6,
           vb.pp("conv_norm_out"),
       )?;
       
       let conv_out = Conv2d::new(
           current_channels,
           2 * config.latent_channels,
           3,
           Conv2dConfig {
               padding: 1,
               ..Default::default()
           },
           vb.pp("conv_out"),
       )?;
       
       Ok(Self {
           conv_in,
           down_blocks,
           mid_block,
           conv_norm_out,
           conv_out,
       })
   }
}

impl Module for FluxVAEEncoder {
   fn forward(&self, x: &Tensor) -> Result<Tensor> {
       let mut hidden = self.conv_in.forward(x)?;
       
       for block in &self.down_blocks {
           hidden = block.forward(&hidden)?;
       }
       
       hidden = self.mid_block.forward(&hidden)?;
       hidden = self.conv_norm_out.forward(&hidden)?;
       hidden = hidden.silu()?;
       hidden = self.conv_out.forward(&hidden)?;
       
       Ok(hidden)
   }
}

/// VAE Decoder
pub struct FluxVAEDecoder {
   conv_in: Conv2d,
   mid_block: MidBlock,
   up_blocks: Vec<DecoderBlock>,
   conv_norm_out: GroupNorm,
   conv_out: Conv2d,
}

impl FluxVAEDecoder {
   pub fn new(config: &FluxVAEConfig, vb: VarBuilder) -> Result<Self> {
       let block_out_channels: Vec<usize> = config.block_out_channels.iter().rev().copied().collect();
       let last_channels = *block_out_channels.last().unwrap();
       
       let conv_in = Conv2d::new(
           config.latent_channels,
           block_out_channels[0],
           3,
           Conv2dConfig {
               padding: 1,
               ..Default::default()
           },
           vb.pp("conv_in"),
       )?;
       
       let mid_block = MidBlock::new(block_out_channels[0], vb.pp("mid_block"))?;
       
       let mut up_blocks = Vec::new();
       let mut current_channels = block_out_channels[0];
       
       for (i, &out_channels) in block_out_channels.iter().enumerate() {
           let is_final = i == block_out_channels.len() - 1;
           
           up_blocks.push(DecoderBlock::new(
               current_channels,
               out_channels,
               config.layers_per_block + 1,
               !is_final,
               vb.pp(format!("up_blocks.{}", i)),
           )?);
           
           current_channels = out_channels;
       }
       
       let conv_norm_out = GroupNorm::new(
           config.norm_num_groups,
           last_channels,
           1e-6,
           vb.pp("conv_norm_out"),
       )?;
       
       let conv_out = Conv2d::new(
           last_channels,
           config.out_channels,
           3,
           Conv2dConfig {
               padding: 1,
               ..Default::default()
           },
           vb.pp("conv_out"),
       )?;
       
       Ok(Self {
           conv_in,
           mid_block,
           up_blocks,
           conv_norm_out,
           conv_out,
       })
   }
}

impl Module for FluxVAEDecoder {
   fn forward(&self, x: &Tensor) -> Result<Tensor> {
       let mut hidden = self.conv_in.forward(x)?;
       hidden = self.mid_block.forward(&hidden)?;
       
       for block in &self.up_blocks {
           hidden = block.forward(&hidden)?;
       }
       
       hidden = self.conv_norm_out.forward(&hidden)?;
       hidden = hidden.silu()?;
       hidden = self.conv_out.forward(&hidden)?;
       
       Ok(hidden)
   }
}

/// Complete Flux VAE
pub struct FluxVAE {
   encoder: FluxVAEEncoder,
   decoder: FluxVAEDecoder,
   quant_conv: Conv2d,
   post_quant_conv: Conv2d,
   config: FluxVAEConfig,
}

impl FluxVAE {
   pub fn new(config: FluxVAEConfig, vb: VarBuilder) -> Result<Self> {
       let encoder = FluxVAEEncoder::new(&config, vb.pp("encoder"))?;
       let decoder = FluxVAEDecoder::new(&config, vb.pp("decoder"))?;
       
       let quant_conv = Conv2d::new(
           2 * config.latent_channels,
           2 * config.latent_channels,
           1,
           Default::default(),
           vb.pp("quant_conv"),
       )?;
       
       let post_quant_conv = Conv2d::new(
           config.latent_channels,
           config.latent_channels,
           1,
           Default::default(),
           vb.pp("post_quant_conv"),
       )?;
       
       Ok(Self {
           encoder,
           decoder,
           quant_conv,
           post_quant_conv,
           config,
       })
   }
   
   pub fn encode(&self, x: &Tensor) -> Result<Tensor> {
       let h = self.encoder.forward(x)?;
       let moments = self.quant_conv.forward(&h)?;
       
       // Split into mean and logvar
       let chunks = moments.chunk(2, 1)?;
       let (mean, logvar) = (&chunks[0], &chunks[1]);
       
       // Clamp logvar for stability
       let logvar = logvar.clamp(-30.0, 20.0)?;
       let std = (logvar * 0.5)?.exp()?;
       
       // Sample using reparameterization trick
       let eps = Tensor::randn(0., 1., mean.shape(), mean.device())?;
       let z = mean + (eps * std)?;
       
       // Scale by the Flux scaling factor
       z * self.config.scaling_factor
   }
   
   pub fn decode(&self, z: &Tensor) -> Result<Tensor> {
       // Unscale
       let z = (z / self.config.scaling_factor)?;
       let z = self.post_quant_conv.forward(&z)?;
       self.decoder.forward(&z)
   }
}

/// Image preprocessing
pub fn image_to_tensor(
   img_path: &str,
   height: usize,
   width: usize,
   device: &Device,
) -> Result<Tensor> {
   use image::io::Reader as ImageReader;
   
   let img = ImageReader::open(img_path)?
       .decode()
       .map_err(|e| candle_core::Error::Msg(format!("Failed to decode image: {}", e)))?;
   
   let img = img.resize_exact(
       width as u32,
       height as u32,
       image::imageops::FilterType::Lanczos3,
   );
   
   let img = img.to_rgb8();
   let (width, height) = img.dimensions();
   
   // Convert to tensor and normalize to [-1, 1]
   let data: Vec<f32> = img
       .pixels()
       .flat_map(|p| {
           vec![
               (p[0] as f32 / 255.0) * 2.0 - 1.0,
               (p[1] as f32 / 255.0) * 2.0 - 1.0,
               (p[2] as f32 / 255.0) * 2.0 - 1.0,
           ]
       })
       .collect();
   
   let tensor = Tensor::from_vec(
       data,
       (3, height as usize, width as usize),
       device,
   )?;
   
   tensor.unsqueeze(0)
}

/// Batch image preprocessing
pub fn batch_images_to_tensor(
   img_paths: &[String],
   height: usize,
   width: usize,
   device: &Device,
) -> Result<Tensor> {
   let mut tensors = Vec::new();
   
   for path in img_paths {
       let tensor = image_to_tensor(path, height, width, device)?;
       tensors.push(tensor);
   }
   
   Tensor::cat(&tensors, 0)
}
