// Parameter is just Tensor in FLAME
use crate::models::attention::AttentionBlock;
use crate::models::resnet::{Downsample2D, ResNetTensorExt, ResnetBlock2D, Upsample2D};
use flame_core::Result;
// Init trait is in tensor module
use crate::ops::Conv2d;
use crate::ops::GroupNorm;
use crate::ops::LayerNorm;
use crate::ops::Linear;
use flame_core::device::Device;
/// UNet implementation using direct FLAME
use flame_core::{DType, Parameter, Tensor};
// Module trait is in tensor module
use crate::flame_training::FLAMEModel;
use crate::loaders::WeightLoader;
use std::collections::HashMap;

/// UNet2D Condition Model for SDXL
pub struct UNet2DConditionModel {
    conv_in: Conv2d,
    time_embedding: TimeEmbedding,
    down_blocks: Vec<DownBlock2D>,
    mid_block: MidBlock2D,
    up_blocks: Vec<UpBlock2D>,
    conv_norm_out: GroupNorm,
    conv_out: Conv2d,
}

impl UNet2DConditionModel {
    pub fn load(weights: &WeightLoader) -> Result<Self> {
        // Load conv_in
        let conv_in = Conv2d::new(4, 320, 3, 1, 1, weights.device().cuda_device().clone())?;

        // Load time embedding
        let time_embedding = TimeEmbedding::load(weights, "time_embedding")?;

        // Load down blocks
        let mut down_blocks = Vec::new();
        let channels = [320, 640, 1280];
        for (i, &ch) in channels.iter().enumerate() {
            let block = DownBlock2D::load(weights, &format!("down_blocks.{}", i), ch)?;
            down_blocks.push(block);
        }

        // Load mid block
        let mid_block = MidBlock2D::load(weights, "mid_block")?;

        // Load up blocks
        let mut up_blocks = Vec::new();
        for (i, &ch) in channels.iter().rev().enumerate() {
            let block = UpBlock2D::load(weights, &format!("up_blocks.{}", i), ch)?;
            up_blocks.push(block);
        }

        // Load output layers
        let conv_norm_out =
            GroupNorm::new(32, 320, 1e-5, true, weights.device().cuda_device().clone())?;
        let conv_out = Conv2d::new(320, 4, 3, 1, 1, weights.device().cuda_device().clone())?;

        Ok(Self {
            conv_in,
            time_embedding,
            down_blocks,
            mid_block,
            up_blocks,
            conv_norm_out,
            conv_out,
        })
    }

    pub fn forward(
        &self,
        sample: &Tensor,
        timestep: &Tensor,
        encoder_hidden_states: &Tensor,
        additional_residuals: Option<Vec<Tensor>>,
    ) -> Result<Tensor> {
        // Time embedding
        let t_emb = self.time_embedding.forward(timestep)?;

        // Initial conv
        let mut sample = self.conv_in.forward(sample)?;

        // Down blocks
        let mut down_block_residuals = Vec::new();
        for (i, down_block) in self.down_blocks.iter().enumerate() {
            let (sample_new, residuals) =
                down_block.forward(&sample, &t_emb, encoder_hidden_states)?;
            sample = sample_new;
            down_block_residuals.extend(residuals);
        }

        // Mid block
        sample = self.mid_block.forward(&sample, &t_emb, encoder_hidden_states)?;

        // Up blocks
        for up_block in &self.up_blocks {
            sample =
                up_block.forward(&sample, &down_block_residuals, &t_emb, encoder_hidden_states)?;
        }

        // Output
        sample = self.conv_norm_out.forward(&sample)?;
        sample = sample.silu()?;
        sample = self.conv_out.forward(&sample)?;

        Ok(sample)
    }
}

impl crate::flame_training::FLAMEModel for UNet2DConditionModel {
    fn parameters(&self) -> Vec<&Parameter> {
        let mut params = Vec::new();
        // Collect all parameters from blocks
        params
    }

    fn named_parameters(&self) -> std::collections::HashMap<String, &Parameter> {
        let mut params = HashMap::new();
        // Collect named parameters
        params
    }
}

/// Time embedding
struct TimeEmbedding {
    linear_1: Linear,
    linear_2: Linear,
}

impl TimeEmbedding {
    fn load(weights: &WeightLoader, prefix: &str) -> Result<Self> {
        let linear_1 = Linear::new(320, 1280, true, &weights.device().cuda_device())?;
        let linear_2 = Linear::new(1280, 1280, true, &weights.device().cuda_device())?;

        Ok(Self { linear_1, linear_2 })
    }

    fn forward(&self, timesteps: &Tensor) -> Result<Tensor> {
        // Create sinusoidal embeddings
        let half_dim = 160;
        let emb =
            flame_core::Tensor::arange(0.0, half_dim as f32, 1.0, timesteps.device().clone())?;
        let scale_val = -f32::ln(10000.0) / (half_dim as f32 - 1.0);
        let scale = emb.mul_scalar(0.0)?.add_scalar(scale_val)?; // Create filled tensor
        let emb = emb.mul(&scale)?.exp()?;

        let emb = timesteps.unsqueeze(timesteps.shape().dims().len())?.mul(&emb.unsqueeze(0)?)?;
        let emb_sin = emb.sin()?;
        let emb_cos = emb.cos()?;
        let emb = Tensor::cat(&[&emb_sin, &emb_cos], emb_sin.shape().dims().len() - 1)?;

        // Project through MLP
        let emb = self.linear_1.forward(&emb)?;
        let emb = emb.silu()?;
        let emb = self.linear_2.forward(&emb)?;

        Ok(emb)
    }
}

// Additional blocks would be implemented...
struct DownBlock2D {
    resnets: Vec<ResnetBlock2D>,
    attentions: Vec<AttentionBlock>,
    downsamplers: Option<Downsample2D>,
}

struct MidBlock2D {
    resnets: Vec<ResnetBlock2D>,
    attentions: Vec<AttentionBlock>,
}

struct UpBlock2D {
    resnets: Vec<ResnetBlock2D>,
    attentions: Vec<AttentionBlock>,
    upsamplers: Option<Upsample2D>,
}

impl DownBlock2D {
    fn load(weights: &WeightLoader, prefix: &str, channels: usize) -> Result<Self> {
        let prefixed_weights = weights.pp(prefix);
        let mut resnets = Vec::new();
        let mut attentions = Vec::new();

        // Load resnet blocks (typically 2 per down block)
        for i in 0..2 {
            let resnet = ResnetBlock2D::load(weights, &format!("resnets.{}", i), channels)?;
            resnets.push(resnet);

            // Load attention blocks if they exist
            if prefixed_weights.get(&format!("attentions.{}.norm.weight", i)).is_ok() {
                let attention = AttentionBlock::load(weights, &format!("attentions.{}", i))?;
                attentions.push(attention);
            }
        }

        // Load downsampler if exists
        let downsamplers = if prefixed_weights.get("downsamplers.0.conv.weight").is_ok() {
            Some(Downsample2D::load(weights, "downsamplers.0")?)
        } else {
            None
        };

        Ok(Self { resnets, attentions, downsamplers })
    }

    fn forward(
        &self,
        sample: &Tensor,
        temb: &Tensor,
        encoder_hidden_states: &Tensor,
    ) -> Result<(Tensor, Vec<Tensor>)> {
        let mut h = sample.clone();
        let mut output_states = Vec::new();

        for (i, resnet) in self.resnets.iter().enumerate() {
            h = resnet.forward(&h, Some(temb))?;

            if i < self.attentions.len() {
                h = self.attentions[i].forward(&h, Some(encoder_hidden_states))?;
            }

            output_states.push(h.clone());
        }

        if let Some(downsampler) = &self.downsamplers {
            h = downsampler.forward(&h)?;
            output_states.push(h.clone());
        }

        Ok((h, output_states))
    }
}

impl MidBlock2D {
    fn load(weights: &WeightLoader, prefix: &str) -> Result<Self> {
        let prefixed_weights = weights.pp(prefix);
        let mut resnets = Vec::new();
        let mut attentions = Vec::new();

        // Mid block typically has 1 resnet, 1 attention, 1 resnet
        for i in 0..2 {
            let resnet = ResnetBlock2D::load(weights, &format!("resnets.{}", i), 1280)?;
            resnets.push(resnet);
        }

        let attention = AttentionBlock::load(weights, "attentions.0")?;
        attentions.push(attention);

        Ok(Self { resnets, attentions })
    }

    fn forward(
        &self,
        sample: &Tensor,
        temb: &Tensor,
        encoder_hidden_states: &Tensor,
    ) -> Result<Tensor> {
        let mut h = sample.clone();

        h = self.resnets[0].forward(&h, Some(temb))?;
        h = self.attentions[0].forward(&h, Some(encoder_hidden_states))?;
        h = self.resnets[1].forward(&h, Some(temb))?;

        Ok(h)
    }
}

impl UpBlock2D {
    fn load(weights: &WeightLoader, prefix: &str, channels: usize) -> Result<Self> {
        let prefixed_weights = weights.pp(prefix);
        let mut resnets = Vec::new();
        let mut attentions = Vec::new();

        // Load resnet blocks (typically 3 per up block)
        for i in 0..3 {
            let resnet = ResnetBlock2D::load(weights, &format!("resnets.{}", i), channels)?;
            resnets.push(resnet);

            // Load attention blocks if they exist
            if prefixed_weights.get(&format!("attentions.{}.norm.weight", i)).is_ok() {
                let attention = AttentionBlock::load(weights, &format!("attentions.{}", i))?;
                attentions.push(attention);
            }
        }

        // Load upsampler if exists
        let upsamplers = if prefixed_weights.get("upsamplers.0.conv.weight").is_ok() {
            Some(Upsample2D::load(weights, "upsamplers.0")?)
        } else {
            None
        };

        Ok(Self { resnets, attentions, upsamplers })
    }

    fn forward(
        &self,
        sample: &Tensor,
        res_hidden_states: &[Tensor],
        temb: &Tensor,
        encoder_hidden_states: &Tensor,
    ) -> Result<Tensor> {
        let mut h = sample.clone();

        for (i, resnet) in self.resnets.iter().enumerate() {
            // Concatenate with residual from encoder
            if i < res_hidden_states.len() {
                h = Tensor::cat(&[&h, &res_hidden_states[i]], 1)?;
            }

            h = resnet.forward(&h, Some(temb))?;

            if i < self.attentions.len() {
                h = self.attentions[i].forward(&h, Some(encoder_hidden_states))?;
            }
        }

        if let Some(upsampler) = &self.upsamplers {
            h = upsampler.forward(&h)?;
        }

        Ok(h)
    }
}
