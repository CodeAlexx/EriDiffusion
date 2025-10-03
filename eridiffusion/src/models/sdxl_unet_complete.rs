use crate::models::attention::SpatialTransformer;
use crate::models::resnet::{Downsample2D, ResnetBlock2D, Upsample2D};
use crate::ops::Conv2d;
use crate::ops::GroupNorm;
use crate::ops::Linear;
use anyhow::anyhow;
use flame_core::device::Device;
use flame_core::Result;
use flame_core::{DType, Shape, Tensor};
use std::collections::HashMap;

/// SDXL UNet configuration
#[derive(Clone, Debug)]
pub struct UNet2DConditionModelConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub center_input_sample: bool,
    pub flip_sin_to_cos: bool,
    pub freq_shift: usize,
    pub down_block_types: Vec<String>,
    pub mid_block_type: String,
    pub up_block_types: Vec<String>,
    pub block_out_channels: Vec<usize>,
    pub layers_per_block: usize,
    pub downsample_padding: usize,
    pub mid_block_scale_factor: f32,
    pub act_fn: String,
    pub norm_num_groups: Option<usize>,
    pub norm_eps: f32,
    pub cross_attention_dim: usize,
    pub transformer_layers_per_block: Vec<usize>,
    pub attention_head_dim: Vec<usize>,
    pub use_linear_projection: bool,
    pub addition_embed_type: String,
    pub addition_time_embed_dim: usize,
    pub projection_class_embeddings_input_dim: usize,
    pub time_embedding_type: String,
    pub time_embedding_dim: Option<usize>,
    pub time_embedding_act_fn: Option<String>,
    pub timestep_post_act: Option<String>,
    pub time_cond_proj_dim: Option<usize>,
    pub conv_in_kernel: usize,
    pub conv_out_kernel: usize,
}

impl Default for UNet2DConditionModelConfig {
    fn default() -> Self {
        Self::sdxl()
    }
}

impl UNet2DConditionModelConfig {
    pub fn sdxl() -> Self {
        Self {
            in_channels: 4,
            out_channels: 4,
            center_input_sample: false,
            flip_sin_to_cos: true,
            freq_shift: 0,
            down_block_types: vec![
                "CrossAttnDownBlock2D".to_string(),
                "CrossAttnDownBlock2D".to_string(),
                "DownBlock2D".to_string(),
            ],
            mid_block_type: "UNetMidBlock2DCrossAttn".to_string(),
            up_block_types: vec![
                "UpBlock2D".to_string(),
                "CrossAttnUpBlock2D".to_string(),
                "CrossAttnUpBlock2D".to_string(),
            ],
            block_out_channels: vec![320, 640, 1280],
            layers_per_block: 2,
            downsample_padding: 1,
            mid_block_scale_factor: 1.0,
            act_fn: "silu".to_string(),
            norm_num_groups: Some(32),
            norm_eps: 1e-5,
            cross_attention_dim: 2048,
            transformer_layers_per_block: vec![1, 2, 10],
            attention_head_dim: vec![5, 10, 20],
            use_linear_projection: true,
            addition_embed_type: "text_time".to_string(),
            addition_time_embed_dim: 256,
            projection_class_embeddings_input_dim: 2816,
            time_embedding_type: "fourier".to_string(),
            time_embedding_dim: Some(256),
            time_embedding_act_fn: None,
            timestep_post_act: None,
            time_cond_proj_dim: None,
            conv_in_kernel: 3,
            conv_out_kernel: 1,
        }
    }
}

/// Time embedding for SDXL
pub struct TimestepEmbedding {
    linear_1: Linear,
    linear_2: Linear,
    act: bool,
}

impl TimestepEmbedding {
    pub fn new(channel: usize, time_embed_dim: usize, act: bool, device: &Device) -> Result<Self> {
        Ok(Self {
            linear_1: Linear::new(channel, time_embed_dim, true, &device.cuda_device())?,
            linear_2: Linear::new(time_embed_dim, time_embed_dim, true, &device.cuda_device())?,
            act,
        })
    }

    pub fn forward(&self, sample: &Tensor) -> Result<Tensor> {
        let mut sample = self.linear_1.forward(sample)?;
        if self.act {
            sample = sample.silu()?;
        }
        self.linear_2.forward(&sample)
    }
}

/// Timestep sinusoidal embeddings
pub struct Timesteps {
    num_channels: usize,
    flip_sin_to_cos: bool,
    downscale_freq_shift: f32,
}

impl Timesteps {
    pub fn new(num_channels: usize, flip_sin_to_cos: bool, downscale_freq_shift: f32) -> Self {
        Self { num_channels, flip_sin_to_cos, downscale_freq_shift }
    }

    pub fn forward(&self, timesteps: &Tensor, device: &Device) -> Result<Tensor> {
        let half_dim = self.num_channels / 2;
        let exponent =
            (0..half_dim).map(|i| i as f32 * 2.0 / self.num_channels as f32).collect::<Vec<_>>();

        let device = Device::from(timesteps.device().clone());
        let exponent = Tensor::from_vec(
            exponent,
            Shape::from_dims(&[half_dim]),
            device.cuda_device().clone(),
        )?;
        let exponent = exponent.mul_scalar(-10000f32.ln())?;
        let emb = exponent.exp()?;

        let emb = emb.mul_scalar(1.0 / (self.downscale_freq_shift + 1.0))?;

        // Broadcast multiply
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

/// CrossAttnDownBlock2D
pub struct CrossAttnDownBlock2D {
    resnets: Vec<ResnetBlock2D>,
    attentions: Vec<SpatialTransformer>,
    downsamplers: Option<Vec<Downsample2D>>,
}

impl CrossAttnDownBlock2D {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        temb_channels: usize,
        num_layers: usize,
        transformer_layers_per_block: usize,
        resnet_eps: f32,
        resnet_groups: usize,
        cross_attention_dim: usize,
        num_attention_heads: usize,
        use_linear_projection: bool,
        add_downsample: bool,
        device: &Device,
    ) -> Result<Self> {
        let mut resnets = Vec::new();
        let mut attentions = Vec::new();

        for i in 0..num_layers {
            let in_ch = if i == 0 { in_channels } else { out_channels };

            resnets.push(ResnetBlock2D::new(
                in_ch,
                out_channels,
                Some(temb_channels),
                resnet_groups,
                device.clone(),
            )?);

            attentions.push(SpatialTransformer::new(
                out_channels,
                num_attention_heads,
                out_channels / num_attention_heads,
                transformer_layers_per_block,
                Some(cross_attention_dim),
                &device,
            )?);
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
        temb: Option<&Tensor>,
        encoder_hidden_states: Option<&Tensor>,
    ) -> Result<(Tensor, Vec<Tensor>)> {
        let mut output_states = Vec::new();
        let mut hidden_states = hidden_states.clone();

        for (resnet, attn) in self.resnets.iter().zip(&self.attentions) {
            hidden_states = resnet.forward(&hidden_states, temb)?;
            hidden_states = attn.forward(&hidden_states, encoder_hidden_states)?;
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

/// CrossAttnUpBlock2D
pub struct CrossAttnUpBlock2D {
    resnets: Vec<ResnetBlock2D>,
    attentions: Vec<SpatialTransformer>,
    upsamplers: Option<Vec<Upsample2D>>,
}

impl CrossAttnUpBlock2D {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        prev_output_channel: usize,
        temb_channels: usize,
        num_layers: usize,
        transformer_layers_per_block: usize,
        resnet_eps: f32,
        resnet_groups: usize,
        cross_attention_dim: usize,
        num_attention_heads: usize,
        use_linear_projection: bool,
        add_upsample: bool,
        device: &Device,
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
                resnet_groups,
                device.clone(),
            )?);

            attentions.push(SpatialTransformer::new(
                out_channels,
                num_attention_heads,
                out_channels / num_attention_heads,
                transformer_layers_per_block,
                Some(cross_attention_dim),
                &device,
            )?);
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
        res_hidden_states_list: &[Tensor],
        temb: Option<&Tensor>,
        encoder_hidden_states: Option<&Tensor>,
    ) -> Result<Tensor> {
        let mut hidden_states = hidden_states.clone();

        for (i, (resnet, attn)) in self.resnets.iter().zip(&self.attentions).enumerate() {
            let res_hidden_states = &res_hidden_states_list[i];
            hidden_states = Tensor::cat(&[&hidden_states, res_hidden_states], 1)?;

            hidden_states = resnet.forward(&hidden_states, temb)?;
            hidden_states = attn.forward(&hidden_states, encoder_hidden_states)?;
        }

        if let Some(upsamplers) = &self.upsamplers {
            for upsampler in upsamplers {
                hidden_states = upsampler.forward(&hidden_states)?;
            }
        }

        Ok(hidden_states)
    }
}

/// UNetMidBlock2DCrossAttn
pub struct UNetMidBlock2DCrossAttn {
    resnets: Vec<ResnetBlock2D>,
    attentions: Vec<SpatialTransformer>,
}

impl UNetMidBlock2DCrossAttn {
    pub fn new(
        in_channels: usize,
        temb_channels: usize,
        num_layers: usize,
        transformer_layers_per_block: usize,
        resnet_eps: f32,
        resnet_groups: usize,
        cross_attention_dim: usize,
        num_attention_heads: usize,
        use_linear_projection: bool,
        device: &Device,
    ) -> Result<Self> {
        let mut resnets = Vec::new();
        let mut attentions = Vec::new();

        for _ in 0..num_layers {
            resnets.push(ResnetBlock2D::new(
                in_channels,
                in_channels,
                Some(temb_channels),
                resnet_groups,
                device.clone(),
            )?);

            attentions.push(SpatialTransformer::new(
                in_channels,
                num_attention_heads,
                in_channels / num_attention_heads,
                transformer_layers_per_block,
                Some(cross_attention_dim),
                &device,
            )?);
        }

        Ok(Self { resnets, attentions })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        temb: Option<&Tensor>,
        encoder_hidden_states: Option<&Tensor>,
    ) -> Result<Tensor> {
        let mut hidden_states = hidden_states.clone();

        for (resnet, attn) in self.resnets.iter().zip(&self.attentions) {
            hidden_states = resnet.forward(&hidden_states, temb)?;
            hidden_states = attn.forward(&hidden_states, encoder_hidden_states)?;
        }

        Ok(hidden_states)
    }
}

/// DownBlock2D
struct DownBlock2D {
    resnets: Vec<ResnetBlock2D>,
    downsamplers: Option<Vec<Downsample2D>>,
}

impl DownBlock2D {
    fn new(
        in_channels: usize,
        out_channels: usize,
        temb_channels: usize,
        num_layers: usize,
        resnet_eps: f32,
        resnet_groups: usize,
        add_downsample: bool,
        device: &Device,
    ) -> Result<Self> {
        let mut resnets = Vec::new();

        for i in 0..num_layers {
            let in_ch = if i == 0 { in_channels } else { out_channels };
            resnets.push(ResnetBlock2D::new(
                in_ch,
                out_channels,
                Some(temb_channels),
                resnet_groups,
                device.clone(),
            )?);
        }

        let downsamplers = if add_downsample {
            Some(vec![Downsample2D::new(out_channels, device.clone())?])
        } else {
            None
        };

        Ok(Self { resnets, downsamplers })
    }
}

/// UpBlock2D
struct UpBlock2D {
    resnets: Vec<ResnetBlock2D>,
    upsamplers: Option<Vec<Upsample2D>>,
}

impl UpBlock2D {
    fn new(
        in_channels: usize,
        out_channels: usize,
        prev_output_channel: usize,
        temb_channels: usize,
        num_layers: usize,
        resnet_eps: f32,
        resnet_groups: usize,
        add_upsample: bool,
        device: &Device,
    ) -> Result<Self> {
        let mut resnets = Vec::new();

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
                resnet_groups,
                device.clone(),
            )?);
        }

        let upsamplers = if add_upsample {
            Some(vec![Upsample2D::new(out_channels, device.clone())?])
        } else {
            None
        };

        Ok(Self { resnets, upsamplers })
    }
}

/// Additional conditioning for SDXL
pub struct AddedCondKwargs {
    pub text_embeds: Tensor,
    pub text_time_embeds: Tensor,
    pub time_ids: Tensor,
}

// Trait definitions for blocks
trait DownBlock: Send + Sync {
    fn forward(
        &self,
        hidden_states: &Tensor,
        temb: Option<&Tensor>,
        encoder_hidden_states: Option<&Tensor>,
    ) -> Result<(Tensor, Vec<Tensor>)>;
}

trait UpBlock: Send + Sync {
    fn forward(
        &self,
        hidden_states: &Tensor,
        res_hidden_states_list: &[Tensor],
        temb: Option<&Tensor>,
        encoder_hidden_states: Option<&Tensor>,
    ) -> Result<Tensor>;
}

// Implement DownBlock trait for DownBlock2D
impl DownBlock for DownBlock2D {
    fn forward(
        &self,
        hidden_states: &Tensor,
        temb: Option<&Tensor>,
        _encoder_hidden_states: Option<&Tensor>,
    ) -> Result<(Tensor, Vec<Tensor>)> {
        let mut output_states = Vec::new();
        let mut hidden_states = hidden_states.clone();

        for resnet in &self.resnets {
            hidden_states = resnet.forward(&hidden_states, temb)?;
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

// Implement DownBlock trait for CrossAttnDownBlock2D
impl DownBlock for CrossAttnDownBlock2D {
    fn forward(
        &self,
        hidden_states: &Tensor,
        temb: Option<&Tensor>,
        encoder_hidden_states: Option<&Tensor>,
    ) -> Result<(Tensor, Vec<Tensor>)> {
        self.forward(hidden_states, temb, encoder_hidden_states)
    }
}

// Implement UpBlock trait for UpBlock2D
impl UpBlock for UpBlock2D {
    fn forward(
        &self,
        hidden_states: &Tensor,
        res_hidden_states_list: &[Tensor],
        temb: Option<&Tensor>,
        _encoder_hidden_states: Option<&Tensor>,
    ) -> Result<Tensor> {
        let mut hidden_states = hidden_states.clone();

        for (i, resnet) in self.resnets.iter().enumerate() {
            let res_hidden_states = &res_hidden_states_list[i];
            hidden_states = Tensor::cat(&[&hidden_states, res_hidden_states], 1)?;
            hidden_states = resnet.forward(&hidden_states, temb)?;
        }

        if let Some(upsamplers) = &self.upsamplers {
            for upsampler in upsamplers {
                hidden_states = upsampler.forward(&hidden_states)?;
            }
        }

        Ok(hidden_states)
    }
}

// Implement UpBlock trait for CrossAttnUpBlock2D
impl UpBlock for CrossAttnUpBlock2D {
    fn forward(
        &self,
        hidden_states: &Tensor,
        res_hidden_states_list: &[Tensor],
        temb: Option<&Tensor>,
        encoder_hidden_states: Option<&Tensor>,
    ) -> Result<Tensor> {
        self.forward(hidden_states, res_hidden_states_list, temb, encoder_hidden_states)
    }
}

/// Main SDXL UNet model
pub struct UNet2DConditionModel {
    config: UNet2DConditionModelConfig,
    conv_in: Conv2d,
    time_proj: Timesteps,
    time_embedding: TimestepEmbedding,
    add_time_proj: Timesteps,
    add_embedding: TimestepEmbedding,
    down_blocks: Vec<Box<dyn DownBlock>>,
    mid_block: UNetMidBlock2DCrossAttn,
    up_blocks: Vec<Box<dyn UpBlock>>,
    conv_norm_out: GroupNorm,
    conv_out: Conv2d,
    device: Device,
}

impl UNet2DConditionModel {
    pub fn new(
        config: UNet2DConditionModelConfig,
        device: &Device,
        weights: HashMap<String, Tensor>,
    ) -> Result<Self> {
        // Input convolution
        let conv_in_weight = weights.get("conv_in.weight").ok_or_else(|| {
            flame_core::Error::InvalidOperation("Missing conv_in.weight".into())
        })?;
        let weight_shape = conv_in_weight.shape();
        let out_channels = weight_shape.dims()[0];
        let in_channels = weight_shape.dims()[1];
        let conv_in = Conv2d::new(
            in_channels,
            out_channels,
            config.conv_in_kernel,
            1,
            1,
            device.cuda_device().clone(),
        )?;
        // TODO: Load the conv weights after creating the Conv2d layer

        // Time embeddings
        let time_embed_dim = config.block_out_channels[0] * 4;
        let time_proj = Timesteps::new(
            config.block_out_channels[0],
            config.flip_sin_to_cos,
            config.freq_shift as f32,
        );
        let time_embedding =
            TimestepEmbedding::new(config.block_out_channels[0], time_embed_dim, true, device)?;

        // Additional embeddings for SDXL
        let add_time_proj = Timesteps::new(
            config.addition_time_embed_dim,
            config.flip_sin_to_cos,
            config.freq_shift as f32,
        );
        let add_embedding = TimestepEmbedding::new(
            config.projection_class_embeddings_input_dim,
            time_embed_dim,
            true,
            device,
        )?;

        // Create down blocks
        let mut down_blocks: Vec<Box<dyn DownBlock>> = Vec::new();
        let mut output_channel = config.block_out_channels[0];

        for (i, down_block_type) in config.down_block_types.iter().enumerate() {
            let input_channel = output_channel;
            output_channel = config.block_out_channels[i];
            let is_final_block = i == config.down_block_types.len() - 1;

            match down_block_type.as_str() {
                "CrossAttnDownBlock2D" => {
                    down_blocks.push(Box::new(CrossAttnDownBlock2D::new(
                        input_channel,
                        output_channel,
                        time_embed_dim,
                        config.layers_per_block,
                        config.transformer_layers_per_block[i],
                        config.norm_eps,
                        config.norm_num_groups.unwrap_or(32),
                        config.cross_attention_dim,
                        config.attention_head_dim[i],
                        config.use_linear_projection,
                        !is_final_block,
                        device,
                    )?));
                }
                "DownBlock2D" => {
                    down_blocks.push(Box::new(DownBlock2D::new(
                        input_channel,
                        output_channel,
                        time_embed_dim,
                        config.layers_per_block,
                        config.norm_eps,
                        config.norm_num_groups.unwrap_or(32),
                        !is_final_block,
                        device,
                    )?));
                }
                _ => {
                    return Err(flame_core::Error::InvalidOperation(format!(
                        "Unknown down block type: {}",
                        down_block_type
                    )))
                }
            }
        }

        // Create mid block
        let mid_block = UNetMidBlock2DCrossAttn::new(
            config.block_out_channels.last().copied().unwrap(),
            time_embed_dim,
            1,
            config.transformer_layers_per_block.last().copied().unwrap_or(1),
            config.norm_eps,
            config.norm_num_groups.unwrap_or(32),
            config.cross_attention_dim,
            config.attention_head_dim.last().copied().unwrap(),
            config.use_linear_projection,
            device,
        )?;

        // Create up blocks
        let mut up_blocks: Vec<Box<dyn UpBlock>> = Vec::new();
        let reversed_block_out_channels: Vec<_> =
            config.block_out_channels.iter().rev().copied().collect();

        for (i, up_block_type) in config.up_block_types.iter().enumerate() {
            let is_final_block = i == config.up_block_types.len() - 1;
            let prev_output_channel = output_channel;
            output_channel = reversed_block_out_channels[i];
            let input_channel =
                if i == 0 { prev_output_channel } else { reversed_block_out_channels[i - 1] };

            match up_block_type.as_str() {
                "CrossAttnUpBlock2D" => {
                    up_blocks.push(Box::new(CrossAttnUpBlock2D::new(
                        input_channel,
                        output_channel,
                        prev_output_channel,
                        time_embed_dim,
                        config.layers_per_block + 1,
                        config.transformer_layers_per_block[config.up_block_types.len() - 1 - i],
                        config.norm_eps,
                        config.norm_num_groups.unwrap_or(32),
                        config.cross_attention_dim,
                        config.attention_head_dim[config.up_block_types.len() - 1 - i],
                        config.use_linear_projection,
                        !is_final_block,
                        device,
                    )?));
                }
                "UpBlock2D" => {
                    up_blocks.push(Box::new(UpBlock2D::new(
                        input_channel,
                        output_channel,
                        prev_output_channel,
                        time_embed_dim,
                        config.layers_per_block + 1,
                        config.norm_eps,
                        config.norm_num_groups.unwrap_or(32),
                        !is_final_block,
                        device,
                    )?));
                }
                _ => {
                    return Err(flame_core::Error::InvalidOperation(format!(
                        "Unknown up block type: {}",
                        up_block_type
                    )))
                }
            }
        }

        // Output layers
        let conv_norm_out = GroupNorm::new(
            config.norm_num_groups.unwrap_or(32),
            config.block_out_channels[0],
            config.norm_eps,
            true,
            device.cuda_device().clone(),
        )?;

        let conv_out_weight = weights.get("conv_out.weight").ok_or_else(|| {
            flame_core::Error::InvalidOperation("Missing conv_out.weight".into())
        })?;
        let weight_shape = conv_out_weight.shape();
        let out_channels = weight_shape.dims()[0];
        let in_channels = weight_shape.dims()[1];
        let conv_out = Conv2d::new(
            in_channels,
            out_channels,
            config.conv_out_kernel,
            1,
            1,
            device.cuda_device().clone(),
        )?;
        // TODO: Load the conv weights after creating the Conv2d layer

        Ok(Self {
            config,
            conv_in,
            time_proj,
            time_embedding,
            add_time_proj,
            add_embedding,
            down_blocks,
            mid_block,
            up_blocks,
            conv_norm_out,
            conv_out,
            device: device.clone(),
        })
    }

    pub fn forward(
        &self,
        sample: &Tensor,
        timestep: &Tensor,
        encoder_hidden_states: &Tensor,
        added_cond_kwargs: Option<&AddedCondKwargs>,
    ) -> Result<Tensor> {
        // 1. Time embeddings
        let timesteps = timestep.reshape(&[timestep.shape().dims()[0]])?;
        let t_emb = self.time_proj.forward(&timesteps, &self.device)?;
        let mut emb = self.time_embedding.forward(&t_emb)?;

        // SDXL: Add additional time embeddings
        if let Some(added_cond) = added_cond_kwargs {
            let add_embeds = self.add_embedding.forward(&added_cond.text_time_embeds)?;
            emb = emb.add(&add_embeds)?;
        }

        // 2. Pre-process
        let mut sample = self.conv_in.forward(sample)?;

        // 3. Down
        let mut down_block_res_samples = Vec::new();
        for down_block in &self.down_blocks {
            let (hidden_states, res_samples) =
                down_block.forward(&sample, Some(&emb), Some(encoder_hidden_states))?;
            sample = hidden_states;
            down_block_res_samples.extend(res_samples);
        }

        // 4. Mid
        sample = self.mid_block.forward(&sample, Some(&emb), Some(encoder_hidden_states))?;

        // 5. Up
        for up_block in &self.up_blocks {
            let res_samples = down_block_res_samples
                .split_off(down_block_res_samples.len() - (self.config.layers_per_block + 1));
            sample =
                up_block.forward(&sample, &res_samples, Some(&emb), Some(encoder_hidden_states))?;
        }

        // 6. Post-process
        sample = self.conv_norm_out.forward(&sample)?;
        sample = sample.silu()?;
        sample = self.conv_out.forward(&sample)?;

        Ok(sample)
    }
}
