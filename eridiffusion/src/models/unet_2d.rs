use crate::loaders::WeightLoader;
use crate::models::attention::SpatialTransformer;
use crate::models::resnet::{Downsample2D, ResNetTensorExt, ResnetBlock2D, Upsample2D};
use crate::ops::Conv2d;
use crate::ops::Linear;
use flame_core::device::Device;
use flame_core::{DType, Error, Result, Shape, Tensor};
use std::{collections::HashMap, sync::Arc};

pub struct UNetMidBlock2D {
    resnets: Vec<ResnetBlock2D>,
    attentions: Vec<Option<SpatialTransformer>>,
}

// SDXL UNet implementation in FLAME

// FLAME uses flame_core::device::Device instead of Device

/// Time embedding block

// Extension trait for Tensor to add missing methods

pub struct TimestepEmbedding {
    linear_1: Linear,
    linear_2: Linear,
}

impl TimestepEmbedding {
    pub fn new(
        channel: usize,
        time_embed_dim: usize,
        device: &flame_core::device::Device,
    ) -> Result<Self> {
        let linear_1 = Linear::new(channel, time_embed_dim, true, &device.cuda_device())?;
        let linear_2 = Linear::new(time_embed_dim, time_embed_dim, true, &device.cuda_device())?;

        Ok(Self { linear_1, linear_2 })
    }

    pub fn forward(&self, sample: &Tensor) -> Result<Tensor> {
        let sample = self.linear_1.forward(sample)?;
        let sample = sample.silu()?;
        self.linear_2.forward(&sample)
    }
}

/// Timestep projection
pub struct Timesteps {
    num_channels: usize,
    flip_sin_to_cos: bool,
    downscale_freq_shift: f32,
}

impl Timesteps {
    pub fn new(num_channels: usize) -> Self {
        Self { num_channels, flip_sin_to_cos: true, downscale_freq_shift: 0.0 }
    }

    pub fn forward(&self, timesteps: &Tensor) -> Result<Tensor> {
        let half_dim = self.num_channels / 2;
        let exponent =
            (0..half_dim).map(|i| -(i as f32 * 2.0 / self.num_channels as f32)).collect::<Vec<_>>();

        let exponent = Tensor::from_slice(
            &exponent,
            Shape::from_dims(&[half_dim]),
            timesteps.device().clone(),
        )?;
        let ln_10000_val = 10000f32.ln();
        let ln_10000 = exponent.mul_scalar(0.0)?.add_scalar(ln_10000_val)?; // Create tensor filled with ln(10000)
        let exponent = exponent.mul(&ln_10000)?;
        let emb = exponent.exp()?;

        // Create sinusoidal embeddings
        let emb = timesteps.unsqueeze(1)?.mul(&emb.unsqueeze(0)?)?;

        let sin_emb = emb.sin()?;
        let cos_emb = emb.cos()?;

        let emb = if self.flip_sin_to_cos {
            Tensor::cat(&[&cos_emb, &sin_emb], 1)?
        } else {
            Tensor::cat(&[&sin_emb, &cos_emb], 1)?
        };

        Ok(emb)
    }
}

/// Down block that combines ResNet and Transformer blocks
pub struct DownBlock2D {
    resnets: Vec<ResnetBlock2D>,
    attentions: Vec<Option<SpatialTransformer>>,
    downsamplers: Option<Vec<Downsample2D>>,
}

impl DownBlock2D {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        temb_channels: usize,
        num_layers: usize,
        add_downsample: bool,
        attention_head_dim: Option<usize>,
        device: &flame_core::device::Device,
    ) -> Result<Self> {
        let mut resnets = Vec::new();
        let mut attentions = Vec::new();

        for i in 0..num_layers {
            let in_ch = if i == 0 { in_channels } else { out_channels };

            resnets.push(ResnetBlock2D::new(
                in_ch,
                out_channels,
                Some(temb_channels),
                32, // groups
                device.clone(),
            )?);

            if let Some(head_dim) = attention_head_dim {
                attentions.push(Some(SpatialTransformer::new(
                    out_channels,
                    out_channels / head_dim,
                    head_dim,
                    1,          // depth
                    Some(2048), // context_dim for SDXL
                    device,
                )?));
            } else {
                attentions.push(None);
            }
        }

        let downsamplers = if add_downsample {
            Some(vec![Downsample2D::new(out_channels, device.clone())?])
        } else {
            None
        };

        Ok(Self { resnets, attentions, downsamplers })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        temb: &Tensor,
        encoder_hidden_states: Option<&Tensor>,
    ) -> Result<(Tensor, Vec<Tensor>)> {
        let mut output_states = Vec::new();
        let mut hidden_states = hidden_states.clone();

        for (resnet, attention) in self.resnets.iter().zip(&self.attentions) {
            hidden_states = resnet.forward(&hidden_states, Some(temb))?;

            if let Some(attn) = attention {
                hidden_states = attn.forward(&hidden_states, encoder_hidden_states)?;
            }

            output_states.push(hidden_states.clone());
        }

        if let Some(downsamplers) = &self.downsamplers {
            for downsampler in downsamplers {
                hidden_states = downsampler.forward(&hidden_states)?;
            }
            output_states.push(hidden_states.clone());
        }

        Ok((hidden_states, output_states))
    }
}

/// Up block that combines ResNet and Transformer blocks
pub struct UpBlock2D {
    resnets: Vec<ResnetBlock2D>,
    attentions: Vec<Option<SpatialTransformer>>,
    upsamplers: Option<Vec<Upsample2D>>,
}

impl UpBlock2D {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        prev_output_channel: usize,
        temb_channels: usize,
        num_layers: usize,
        add_upsample: bool,
        attention_head_dim: Option<usize>,
        device: &flame_core::device::Device,
    ) -> Result<Self> {
        let mut resnets = Vec::new();
        let mut attentions = Vec::new();

        for i in 0..num_layers {
            let res_skip_channels = if i == num_layers - 1 { in_channels } else { out_channels };

            let resnet_in_channels = if i == 0 {
                prev_output_channel + res_skip_channels
            } else {
                out_channels + res_skip_channels
            };

            resnets.push(ResnetBlock2D::new(
                resnet_in_channels,
                out_channels,
                Some(temb_channels),
                32, // groups
                device.clone(),
            )?);

            if let Some(head_dim) = attention_head_dim {
                attentions.push(Some(SpatialTransformer::new(
                    out_channels,
                    out_channels / head_dim,
                    head_dim,
                    1,          // depth
                    Some(2048), // context_dim for SDXL
                    &device,
                )?));
            } else {
                attentions.push(None);
            }
        }

        let upsamplers = if add_upsample {
            Some(vec![Upsample2D::new(out_channels, device.clone())?])
        } else {
            None
        };

        Ok(Self { resnets, attentions, upsamplers })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        res_hidden_states_tuple: &[Tensor],
        temb: &Tensor,
        encoder_hidden_states: Option<&Tensor>,
    ) -> Result<Tensor> {
        let mut hidden_states = hidden_states.clone();

        for (idx, (resnet, attention)) in self.resnets.iter().zip(&self.attentions).enumerate() {
            let res_hidden_states =
                &res_hidden_states_tuple[res_hidden_states_tuple.len() - idx - 1];
            hidden_states = Tensor::cat(&[&hidden_states, &res_hidden_states], 1)?;

            hidden_states = resnet.forward(&hidden_states, Some(temb))?;

            if let Some(attn) = attention {
                hidden_states = attn.forward(&hidden_states, encoder_hidden_states)?;
            }
        }

        if let Some(upsamplers) = &self.upsamplers {
            for upsampler in upsamplers {
                hidden_states = upsampler.forward(&hidden_states)?;
            }
        }

        Ok(hidden_states)
    }
}

/// UNet middle block

impl UNetMidBlock2D {
    pub fn new(
        in_channels: usize,
        temb_channels: usize,
        attention_head_dim: Option<usize>,
        device: &flame_core::device::Device,
    ) -> Result<Self> {
        let mut resnets = vec![ResnetBlock2D::new(
            in_channels,
            in_channels,
            Some(temb_channels),
            32,
            device.clone(),
        )?];

        let attentions = if let Some(head_dim) = attention_head_dim {
            vec![Some(SpatialTransformer::new(
                in_channels,
                in_channels / head_dim,
                head_dim,
                1,
                Some(2048),
                &device,
            )?)]
        } else {
            vec![None]
        };

        resnets.push(ResnetBlock2D::new(
            in_channels,
            in_channels,
            Some(temb_channels),
            32,
            device.clone(),
        )?);

        Ok(Self { resnets, attentions })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        temb: &Tensor,
        encoder_hidden_states: Option<&Tensor>,
    ) -> Result<Tensor> {
        let mut hidden_states = self.resnets[0].forward(hidden_states, Some(temb))?;

        if let Some(attn) = &self.attentions[0] {
            hidden_states = attn.forward(&hidden_states, encoder_hidden_states)?;
        }

        hidden_states = self.resnets[1].forward(&hidden_states, Some(temb))?;

        Ok(hidden_states)
    }
}

/// SDXL UNet model
pub struct UNet2DConditionModel {
    // Input
    conv_in: Conv2d,

    // Time
    time_proj: Timesteps,
    time_embedding: TimestepEmbedding,

    // Additional embeddings
    add_embedding: Linear,

    // Down blocks
    down_blocks: Vec<DownBlock2D>,

    // Mid block
    mid_block: UNetMidBlock2D,

    // Up blocks
    up_blocks: Vec<UpBlock2D>,

    // Output
    conv_norm_out: flame_core::GroupNorm,
    conv_out: Conv2d,

    // Config
    sample_size: usize,
    in_channels: usize,
    out_channels: usize,
}

impl UNet2DConditionModel {
    pub fn new_sdxl(device: flame_core::device::Device) -> Result<Self> {
        // SDXL configuration
        let sample_size = 128;
        let in_channels = 4;
        let out_channels = 4;
        let block_out_channels = vec![320, 640, 1280];
        let layers_per_block = vec![2, 2, 2];
        let attention_head_dim = vec![Some(40), Some(80), None];
        let time_embed_dim = 1280;

        // Conv in
        let conv_in =
            Conv2d::new(in_channels, block_out_channels[0], 3, 1, 1, device.cuda_device().clone())?;

        // Time embeddings
        let time_proj = Timesteps::new(block_out_channels[0]);
        let time_embedding =
            TimestepEmbedding::new(block_out_channels[0], time_embed_dim, &device)?;

        // Additional embedding for SDXL
        let add_embedding = Linear::new(2816, time_embed_dim, true, &device.cuda_device())?;

        // Down blocks
        let mut down_blocks = Vec::new();
        let mut current_channels = block_out_channels[0];

        for (i, (&out_ch, (&num_layers, &attn_dim))) in block_out_channels
            .iter()
            .zip(layers_per_block.iter().zip(&attention_head_dim))
            .enumerate()
        {
            let is_final = i == block_out_channels.len() - 1;

            down_blocks.push(DownBlock2D::new(
                current_channels,
                out_ch,
                time_embed_dim,
                num_layers,
                !is_final, // add_downsample
                attn_dim,
                &device,
            )?);

            current_channels = out_ch;
        }

        // Mid block
        let mid_block = UNetMidBlock2D::new(
            current_channels,
            time_embed_dim,
            Some(40), // attention_head_dim
            &device,
        )?;

        // Up blocks
        let mut up_blocks = Vec::new();
        let reversed_block_out_channels: Vec<_> =
            block_out_channels.iter().rev().cloned().collect();

        for (i, (&out_ch, (&num_layers, &attn_dim))) in reversed_block_out_channels
            .iter()
            .zip(layers_per_block.iter().rev().zip(attention_head_dim.iter().rev()))
            .enumerate()
        {
            let prev_output_channel =
                if i == 0 { current_channels } else { reversed_block_out_channels[i - 1] };

            let is_final = i == reversed_block_out_channels.len() - 1;

            up_blocks.push(UpBlock2D::new(
                out_ch,
                reversed_block_out_channels.get(i + 1).cloned().unwrap_or(block_out_channels[0]),
                prev_output_channel,
                time_embed_dim,
                num_layers,
                !is_final, // add_upsample
                attn_dim,
                &device,
            )?);
        }

        // Output
        let conv_norm_out = flame_core::GroupNorm::new(
            32,
            block_out_channels[0],
            1e-5,
            true,
            device.cuda_device().clone(),
        )?;
        let conv_out = Conv2d::new(
            block_out_channels[0],
            out_channels,
            3,
            1,
            1,
            device.cuda_device().clone(),
        )?;

        Ok(Self {
            conv_in,
            time_proj,
            time_embedding,
            add_embedding,
            down_blocks,
            mid_block,
            up_blocks,
            conv_norm_out,
            conv_out,
            sample_size,
            in_channels,
            out_channels,
        })
    }

    pub fn forward(
        &self,
        sample: &Tensor,
        timestep: &Tensor,
        encoder_hidden_states: &Tensor,
        added_cond_kwargs: Option<&std::collections::HashMap<String, Tensor>>,
    ) -> Result<Tensor> {
        // 1. Time embedding
        let t_emb = self.time_proj.forward(timestep)?;
        let emb = self.time_embedding.forward(&t_emb)?;

        // 2. Add additional embeddings (for SDXL)
        let emb = if let Some(kwargs) = added_cond_kwargs {
            if let Some(text_embeds) = kwargs.get("text_embeds") {
                let aug_emb = self.add_embedding.forward(text_embeds)?;
                emb.add(&aug_emb)?
            } else {
                emb
            }
        } else {
            emb
        };

        // 3. Conv in
        let mut hidden_states = self.conv_in.forward(sample)?;

        // 4. Down blocks
        let mut down_block_res_samples = Vec::new();
        for down_block in &self.down_blocks {
            let (h, res_samples) =
                down_block.forward(&hidden_states, &emb, Some(encoder_hidden_states))?;
            hidden_states = h;
            down_block_res_samples.extend(res_samples);
        }

        // 5. Mid block
        hidden_states =
            self.mid_block.forward(&hidden_states, &emb, Some(encoder_hidden_states))?;

        // 6. Up blocks
        for up_block in &self.up_blocks {
            hidden_states = up_block.forward(
                &hidden_states,
                &down_block_res_samples,
                &emb,
                Some(encoder_hidden_states),
            )?;
        }

        // 7. Output
        hidden_states = self.conv_norm_out.forward(&hidden_states)?;
        hidden_states = hidden_states.silu()?;
        self.conv_out.forward(&hidden_states)
    }

    pub fn load_weights(
        &mut self,
        weights: &std::collections::HashMap<String, Tensor>,
    ) -> Result<()> {
        // Load weights into the model layers
        println!("Loading {} weight tensors into UNet", weights.len());

        // This is a complex mapping process that would require knowing the exact
        // structure of the saved weights. For now, we rely on the weight loading
        // happening during construction via WeightLoader.
        //
        // In a full implementation, this would map keys like:
        // "model.diffusion_model.input_blocks.0.0.weight" -> self.down_blocks[0].resnets[0].conv1.weight
        // etc.

        Err(flame_core::Error::InvalidOperation(
            "Direct weight loading not yet implemented. Use WeightLoader during construction."
                .to_string(),
        ))
    }
}

// Helper for concatenating tensors
pub fn concat_tensors(tensors: &[Tensor], dim: usize) -> Result<Tensor> {
    if tensors.is_empty() {
        return Err(flame_core::Error::InvalidOperation(
            "Cannot concatenate empty tensor list".to_string(),
        ));
    }

    // Validate all tensors have same shape except for concat dimension
    let first_shape = tensors[0].shape().dims();
    let mut total_size = 0;

    for tensor in tensors {
        let shape = tensor.shape().dims();
        if shape.len() != first_shape.len() {
            return Err(flame_core::Error::InvalidOperation(
                "All tensors must have same number of dimensions".to_string(),
            ));
        }

        for (i, (&s1, &s2)) in first_shape.iter().zip(shape.iter()).enumerate() {
            if i != dim && s1 != s2 {
                return Err(flame_core::Error::InvalidOperation(
                    "All dimensions except concat dimension must match".to_string(),
                ));
            }
        }
        total_size += shape[dim];
    }

    // Create output shape
    let mut output_shape = first_shape.to_vec();
    output_shape[dim] = total_size;

    // Now actually concatenates tensors
    let tensor_refs: Vec<&Tensor> = tensors.iter().collect();
    Tensor::cat(&tensor_refs, dim)
}
