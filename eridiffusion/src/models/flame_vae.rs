use crate::models::attention::AttentionBlock;
use crate::models::resnet::{Downsample2D, ResnetBlock2D, Upsample2D};
use crate::ops::Conv2d;
use crate::ops::GroupNorm;
use crate::ops::LayerNorm;
use crate::ops::Linear;
use flame_core::device::Device;
use flame_core::Parameter;
use flame_core::Result;
/// VAE implementation using direct FLAME
use flame_core::{DType, Tensor};
// Module trait is in tensor module
use crate::flame_training::FLAMEModel;
use crate::loaders::WeightLoader;
use crate::models::flame_migration_helpers::{collect_with_prefix, CollectParameters};
use std::collections::HashMap;

/// VAE Encoder
pub struct VAEEncoder {
    conv_in: Conv2d,
    down_blocks: Vec<VAEEncoderBlock>,
    mid_block: VAEMidBlock,
    conv_norm_out: GroupNorm,
    conv_out: Conv2d,
    quant_conv: Option<Conv2d>,
}

impl VAEEncoder {
    pub fn load(weights: &WeightLoader, prefix: &str) -> Result<Self> {
        let prefixed_weights = weights.pp(prefix);

        // Load conv_in
        let conv_in = Conv2d::new(3, 128, 3, 1, 1, weights.device().cuda_device().clone())?;

        // Load down blocks
        let mut down_blocks = Vec::new();
        for i in 0..4 {
            let block = VAEEncoderBlock::load(weights, &format!("down.{}", i))?;
            down_blocks.push(block);
        }

        // Load mid block
        let mid_block = VAEMidBlock::load(weights, "mid")?;

        // Load output layers
        let conv_norm_out =
            GroupNorm::new(32, 512, 1e-6, true, prefixed_weights.device().cuda_device().clone())?;
        let conv_out =
            Conv2d::new(512, 8, 3, 1, 1, prefixed_weights.device().cuda_device().clone())?;

        // Quantization conv if present
        let quant_conv = if prefixed_weights.tensor("quant_conv.weight", &[8, 8, 1, 1]).is_ok() {
            Some(Conv2d::new(8, 8, 1, 1, 0, prefixed_weights.device().cuda_device().clone())?)
        } else {
            None
        };

        Ok(Self { conv_in, down_blocks, mid_block, conv_norm_out, conv_out, quant_conv })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = self.conv_in.forward(x)?;

        // Down blocks
        for block in &self.down_blocks {
            h = block.forward(&h)?;
        }

        // Mid block
        h = self.mid_block.forward(&h)?;

        // Output
        h = self.conv_norm_out.forward(&h)?;
        h = h.silu()?;
        h = self.conv_out.forward(&h)?;

        // Quantization
        if let Some(quant) = &self.quant_conv {
            h = quant.forward(&h)?;
        }

        Ok(h)
    }
}

/// VAE Decoder
pub struct VAEDecoder {
    conv_in: Conv2d,
    mid_block: VAEMidBlock,
    up_blocks: Vec<VAEDecoderBlock>,
    conv_norm_out: GroupNorm,
    conv_out: Conv2d,
    post_quant_conv: Option<Conv2d>,
}

impl VAEDecoder {
    pub fn load(weights: &WeightLoader, prefix: &str) -> Result<Self> {
        let prefixed_weights = weights.pp(prefix);

        // Post-quantization conv if present
        let post_quant_conv =
            if prefixed_weights.tensor("post_quant_conv.weight", &[512, 4, 1, 1]).is_ok() {
                Some(Conv2d::new(4, 512, 1, 1, 0, prefixed_weights.device().cuda_device().clone())?)
            } else {
                None
            };

        // Load conv_in
        let conv_in =
            Conv2d::new(4, 512, 3, 1, 1, prefixed_weights.device().cuda_device().clone())?;

        // Load mid block
        let mid_block = VAEMidBlock::load(weights, "mid_block")?;

        // Load up blocks
        let mut up_blocks = Vec::new();
        for i in 0..4 {
            let block = VAEDecoderBlock::load(weights, &format!("up.{}", i))?;
            up_blocks.push(block);
        }

        // Load output layers
        let conv_norm_out =
            GroupNorm::new(32, 128, 1e-6, true, prefixed_weights.device().cuda_device().clone())?;
        let conv_out =
            Conv2d::new(128, 3, 3, 1, 1, prefixed_weights.device().cuda_device().clone())?;

        Ok(Self { conv_in, mid_block, up_blocks, conv_norm_out, conv_out, post_quant_conv })
    }

    pub fn forward(&self, z: &Tensor) -> Result<Tensor> {
        let mut h = z.clone();

        // Post-quantization
        if let Some(post_quant) = &self.post_quant_conv {
            h = post_quant.forward(&h)?;
        }

        h = self.conv_in.forward(&h)?;

        // Mid block
        h = self.mid_block.forward(&h)?;

        // Up blocks
        for block in &self.up_blocks {
            h = block.forward(&h)?;
        }

        // Output
        let h_norm = self.conv_norm_out.forward(&h)?;
        let h_activated = h_norm.silu()?;
        let output = self.conv_out.forward(&h_activated)?;

        Ok(output)
    }
}

/// Complete VAE
pub struct VAE {
    encoder: VAEEncoder,
    decoder: VAEDecoder,
    scaling_factor: f32,
}

impl VAE {
    pub fn load(weights: &WeightLoader) -> Result<Self> {
        let encoder = VAEEncoder::load(weights, "encoder")?;
        let decoder = VAEDecoder::load(weights, "decoder")?;

        Ok(Self { encoder, decoder, scaling_factor: 0.18215 })
    }

    pub fn encode(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.encoder.forward(x)?;

        // Split into mean and logvar
        let chunks = h.chunk(2, 1)?;
        let mean = &chunks[0];
        let logvar = &chunks[1];

        // Sample from distribution
        let std = logvar.mul_scalar(0.5)?.exp()?;
        let eps = Tensor::randn(mean.shape().clone(), 0.0, 1.0, mean.device().clone())?;
        let sample = mean.add(&std.mul(&eps)?)?;

        // Scale
        Ok(sample.mul_scalar(self.scaling_factor)?)
    }

    pub fn decode(&self, z: &Tensor) -> Result<Tensor> {
        // Unscale
        let z = z.div_scalar(self.scaling_factor)?;
        self.decoder.forward(&z)
    }
}

impl crate::flame_training::FLAMEModel for VAE {
    fn parameters(&self) -> Vec<&Parameter> {
        let mut params = Vec::new();
        // Collect parameters from encoder and decoder
        // This would be implemented by traversing all layers
        params
    }

    fn named_parameters(&self) -> std::collections::HashMap<String, &Parameter> {
        let mut params = HashMap::new();
        // Collect named parameters
        params
    }
}

/// VAE blocks
struct VAEEncoderBlock {
    resnets: Vec<ResnetBlock2D>,
    downsamplers: Option<Downsample2D>,
}

impl VAEEncoderBlock {
    fn load(weights: &WeightLoader, prefix: &str) -> Result<Self> {
        let prefixed_weights = weights.pp(prefix);
        let mut resnets = Vec::new();

        // Load all resnet blocks dynamically based on what's in the weights
        let mut i = 0;
        while prefixed_weights.get(&format!("resnets.{}.norm1.weight", i)).is_ok() {
            // Determine in/out channels from weight shapes
            let norm1_weight = prefixed_weights.get(&format!("resnets.{}.norm1.weight", i))?;
            let in_channels = norm1_weight.shape().dims()[0];

            let conv1_weight = prefixed_weights.get(&format!("resnets.{}.conv1.weight", i))?;
            let out_channels = conv1_weight.shape().dims()[0];

            let resnet =
                ResnetBlock2D::load(weights, &format!("{}.resnets.{}", prefix, i), in_channels)?;
            resnets.push(resnet);
            i += 1;
        }

        // Downsampler if exists
        let downsamplers = if prefixed_weights.get("downsamplers.0.conv.weight").is_ok() {
            Some(Downsample2D::load(weights, &format!("{}.downsamplers.0", prefix))?)
        } else {
            None
        };

        Ok(Self { resnets, downsamplers })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = x.clone();

        for resnet in &self.resnets {
            h = resnet.forward(&h, None)?;
        }

        if let Some(downsampler) = &self.downsamplers {
            h = downsampler.forward(&h)?;
        }

        Ok(h)
    }
}

struct VAEDecoderBlock {
    resnets: Vec<ResnetBlock2D>,
    upsamplers: Option<Upsample2D>,
}

impl VAEDecoderBlock {
    fn load(weights: &WeightLoader, prefix: &str) -> Result<Self> {
        let prefixed_weights = weights.pp(prefix);
        let mut resnets = Vec::new();

        // Load all resnet blocks dynamically
        let mut i = 0;
        while prefixed_weights.get(&format!("resnets.{}.norm1.weight", i)).is_ok() {
            // Determine channels from weight shapes
            let norm1_weight = prefixed_weights.get(&format!("resnets.{}.norm1.weight", i))?;
            let in_channels = norm1_weight.shape().dims()[0];

            let resnet =
                ResnetBlock2D::load(weights, &format!("{}.resnets.{}", prefix, i), in_channels)?;
            resnets.push(resnet);
            i += 1;
        }

        // Upsampler if exists
        let upsamplers = if prefixed_weights.get("upsamplers.0.conv.weight").is_ok() {
            Some(Upsample2D::load(weights, &format!("{}.upsamplers.0", prefix))?)
        } else {
            None
        };

        Ok(Self { resnets, upsamplers })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = x.clone();

        for resnet in &self.resnets {
            h = resnet.forward(&h, None)?;
        }

        if let Some(upsampler) = &self.upsamplers {
            h = upsampler.forward(&h)?;
        }

        Ok(h)
    }
}

struct VAEMidBlock {
    resnets: Vec<ResnetBlock2D>,
    attentions: Option<Vec<AttentionBlock>>,
}

impl VAEMidBlock {
    fn load(weights: &WeightLoader, prefix: &str) -> Result<Self> {
        let prefixed_weights = weights.pp(prefix);
        let mut resnets = Vec::new();

        // Load all resnet blocks dynamically
        let mut i = 0;
        while prefixed_weights.get(&format!("resnets.{}.norm1.weight", i)).is_ok() {
            // Determine channels from weight shapes
            let norm1_weight = prefixed_weights.get(&format!("resnets.{}.norm1.weight", i))?;
            let channels = norm1_weight.shape().dims()[0];

            let resnet =
                ResnetBlock2D::load(weights, &format!("{}.resnets.{}", prefix, i), channels)?;
            resnets.push(resnet);
            i += 1;
        }

        // Load all attention blocks if they exist
        let mut attentions = Vec::new();
        let mut i = 0;
        while prefixed_weights.get(&format!("attentions.{}.norm.weight", i)).is_ok() {
            let attention = AttentionBlock::load(weights, &format!("{}.attentions.{}", prefix, i))?;
            attentions.push(attention);
            i += 1;
        }

        let attentions = if attentions.is_empty() { None } else { Some(attentions) };

        Ok(Self { resnets, attentions })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = x.clone();

        h = self.resnets[0].forward(&h, None)?;

        if let Some(attentions) = &self.attentions {
            h = attentions[0].forward(&h, None)?;
        }

        h = self.resnets[1].forward(&h, None)?;

        Ok(h)
    }
}
