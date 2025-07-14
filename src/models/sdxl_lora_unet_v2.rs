//! SDXL UNet with true LoRA integration
//! This implementation properly injects LoRA into attention layers

use candle_core::{Device, DType, Module, Result, Tensor, D};
use candle_nn as nn;
use candle_transformers::models::stable_diffusion::{
    embeddings::{TimestepEmbedding, Timesteps},
    unet_2d_blocks::{
        UNetMidBlock2D, UNetMidBlock2DConfig,
        DownBlock2D, DownBlock2DConfig,
        UpBlock2D, UpBlock2DConfig,
    },
};
use crate::models::with_tracing::{conv2d, Conv2d};
use super::{
    lora_unet_blocks::{
        CrossAttnDownBlock2DWithLoRA, CrossAttnDownBlock2DConfig,
        CrossAttnUpBlock2DWithLoRA, CrossAttnUpBlock2DConfig,
        UNetMidBlock2DCrossAttnWithLoRA, UNetMidBlock2DCrossAttnConfig,
    },
    lora_attention::LoRAAttentionConfig,
    lora_with_gradients::LoRALinearWithGradients,
};
use std::collections::HashMap;

/// SDXL UNet configuration
#[derive(Debug, Clone)]
pub struct SDXLUNetConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub center_input_sample: bool,
    pub time_embedding_type: String,
    pub flip_sin_to_cos: bool,
    pub freq_shift: f64,
    pub down_block_types: Vec<String>,
    pub mid_block_type: String,
    pub up_block_types: Vec<String>,
    pub only_cross_attention: Vec<bool>,
    pub block_out_channels: Vec<usize>,
    pub layers_per_block: usize,
    pub downsample_padding: usize,
    pub mid_block_scale_factor: f64,
    pub act_fn: String,
    pub norm_num_groups: Option<usize>,
    pub norm_eps: f64,
    pub cross_attention_dim: usize,
    pub transformer_layers_per_block: Vec<usize>,
    pub attention_head_dim: Vec<usize>,
    pub num_class_embeds: Option<usize>,
    pub use_linear_projection: bool,
    pub class_embed_type: Option<String>,
    pub addition_embed_type: Option<String>,
    pub addition_time_embed_dim: Option<usize>,
    pub num_attention_heads: Vec<usize>,
    pub projection_class_embeddings_input_dim: Option<usize>,
    pub attention_type: String,
}

impl Default for SDXLUNetConfig {
    fn default() -> Self {
        Self {
            in_channels: 4,
            out_channels: 4,
            center_input_sample: false,
            time_embedding_type: "positional".to_string(),
            flip_sin_to_cos: true,
            freq_shift: 0.0,
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
            only_cross_attention: vec![false, false, false],
            block_out_channels: vec![320, 640, 1280, 1280],
            layers_per_block: 2,
            downsample_padding: 1,
            mid_block_scale_factor: 1.0,
            act_fn: "silu".to_string(),
            norm_num_groups: Some(32),
            norm_eps: 1e-5,
            cross_attention_dim: 2048,
            transformer_layers_per_block: vec![1, 2, 10],
            attention_head_dim: vec![5, 10, 20, 20],
            num_class_embeds: None,
            use_linear_projection: true,
            class_embed_type: None,
            addition_embed_type: Some("text_time".to_string()),
            addition_time_embed_dim: Some(256),
            num_attention_heads: vec![],
            projection_class_embeddings_input_dim: Some(2816),
            attention_type: "default".to_string(),
        }
    }
}

/// SDXL UNet with true LoRA injection
pub struct SDXLUNetWithLoRA {
    // Initial convolution
    conv_in: Conv2d,
    
    // Time embeddings
    time_proj: Timesteps,
    time_embedding: TimestepEmbedding,
    add_time_proj: Option<Timesteps>,
    add_embedding: Option<TimestepEmbedding>,
    
    // Down blocks (mix of regular and LoRA-enabled)
    down_blocks: Vec<DownBlockEnum>,
    
    // Mid block with LoRA
    mid_block: MidBlockEnum,
    
    // Up blocks (mix of regular and LoRA-enabled)
    up_blocks: Vec<UpBlockEnum>,
    
    // Output layers
    conv_norm_out: nn::GroupNorm,
    conv_out: Conv2d,
    
    // Configuration
    config: SDXLUNetConfig,
    device: Device,
    dtype: DType,
}

/// Enum to hold different down block types
enum DownBlockEnum {
    Regular(DownBlock2D),
    CrossAttnWithLoRA(CrossAttnDownBlock2DWithLoRA),
}

/// Enum to hold different mid block types
enum MidBlockEnum {
    Regular(UNetMidBlock2D),
    CrossAttnWithLoRA(UNetMidBlock2DCrossAttnWithLoRA),
}

/// Enum to hold different up block types
enum UpBlockEnum {
    Regular(UpBlock2D),
    CrossAttnWithLoRA(CrossAttnUpBlock2DWithLoRA),
}

impl SDXLUNetWithLoRA {
    pub fn new(
        vs: nn::VarBuilder,
        config: SDXLUNetConfig,
        use_flash_attn: bool,
        lora_config: Option<&LoRAAttentionConfig>,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        let norm_num_groups = config.norm_num_groups.unwrap_or(32);
        let time_embed_dim = config.block_out_channels[0] * 4;
        
        // Initial convolution
        let conv_in = conv2d(
            config.in_channels,
            config.block_out_channels[0],
            3,
            nn::Conv2dConfig { padding: 1, ..Default::default() },
            vs.pp("conv_in"),
        )?;
        
        // Time embeddings
        let time_proj = Timesteps::new(
            config.block_out_channels[0],
            config.flip_sin_to_cos,
            config.freq_shift,
        );
        
        let time_embedding = TimestepEmbedding::new(
            vs.pp("time_embedding"),
            config.block_out_channels[0],
            time_embed_dim,
        )?;
        
        // Additional embeddings for SDXL
        let (add_time_proj, add_embedding) = if config.addition_embed_type.is_some() {
            let add_embed_dim = config.addition_time_embed_dim.unwrap_or(256);
            let add_time_proj = Timesteps::new(
                add_embed_dim,
                config.flip_sin_to_cos,
                config.freq_shift,
            );
            let add_embedding = TimestepEmbedding::new(
                vs.pp("add_embedding"),
                config.projection_class_embeddings_input_dim.unwrap_or(2816),
                time_embed_dim,
            )?;
            (Some(add_time_proj), Some(add_embedding))
        } else {
            (None, None)
        };
        
        // Create down blocks
        let mut down_blocks = Vec::new();
        let mut output_channel = config.block_out_channels[0];
        
        for (i, down_block_type) in config.down_block_types.iter().enumerate() {
            let input_channel = output_channel;
            output_channel = config.block_out_channels[i];
            let is_final_block = i == config.down_block_types.len() - 1;
            
            match down_block_type.as_str() {
                "CrossAttnDownBlock2D" => {
                    let config = CrossAttnDownBlock2DConfig {
                        downblock: DownBlock2DConfig {
                            num_layers: config.layers_per_block,
                            resnet_eps: config.norm_eps,
                            resnet_groups: norm_num_groups,
                            add_downsample: !is_final_block,
                            downsample_padding: config.downsample_padding,
                            ..Default::default()
                        },
                        attn_num_head_channels: config.attention_head_dim[i],
                        cross_attention_dim: config.cross_attention_dim,
                        use_linear_projection: config.use_linear_projection,
                        transformer_layers_per_block: config.transformer_layers_per_block[i],
                        ..Default::default()
                    };
                    
                    let block = CrossAttnDownBlock2DWithLoRA::new(
                        vs.pp(format!("down_blocks.{}", i)),
                        input_channel,
                        output_channel,
                        Some(time_embed_dim),
                        use_flash_attn,
                        config,
                        lora_config,
                        device.clone(),
                        dtype,
                    )?;
                    down_blocks.push(DownBlockEnum::CrossAttnWithLoRA(block));
                }
                "DownBlock2D" => {
                    let config = DownBlock2DConfig {
                        num_layers: config.layers_per_block,
                        resnet_eps: config.norm_eps,
                        resnet_groups: norm_num_groups,
                        add_downsample: !is_final_block,
                        downsample_padding: config.downsample_padding,
                        ..Default::default()
                    };
                    
                    let block = DownBlock2D::new(
                        vs.pp(format!("down_blocks.{}", i)),
                        input_channel,
                        output_channel,
                        Some(time_embed_dim),
                        config,
                    )?;
                    down_blocks.push(DownBlockEnum::Regular(block));
                }
                _ => return Err(candle_core::Error::Msg(
                    format!("Unknown down block type: {}", down_block_type)
                )),
            }
        }
        
        // Create mid block
        let mid_block_channel = config.block_out_channels.last().unwrap();
        let mid_block = match config.mid_block_type.as_str() {
            "UNetMidBlock2DCrossAttn" => {
                let config = UNetMidBlock2DCrossAttnConfig {
                    resnet_eps: config.norm_eps,
                    resnet_groups: Some(norm_num_groups),
                    attn_num_head_channels: config.attention_head_dim.last().copied().unwrap_or(20),
                    output_scale_factor: config.mid_block_scale_factor,
                    cross_attn_dim: config.cross_attention_dim,
                    use_linear_projection: config.use_linear_projection,
                    transformer_layers_per_block: config.transformer_layers_per_block.last().copied().unwrap_or(1),
                    ..Default::default()
                };
                
                let block = UNetMidBlock2DCrossAttnWithLoRA::new(
                    vs.pp("mid_block"),
                    *mid_block_channel,
                    Some(time_embed_dim),
                    use_flash_attn,
                    config,
                    lora_config,
                    device.clone(),
                    dtype,
                )?;
                MidBlockEnum::CrossAttnWithLoRA(block)
            }
            _ => {
                let config = UNetMidBlock2DConfig {
                    resnet_eps: config.norm_eps,
                    resnet_groups: Some(norm_num_groups),
                    attn_num_head_channels: Some(config.attention_head_dim.last().copied().unwrap_or(20)),
                    output_scale_factor: config.mid_block_scale_factor,
                    ..Default::default()
                };
                
                let block = UNetMidBlock2D::new(
                    vs.pp("mid_block"),
                    *mid_block_channel,
                    Some(time_embed_dim),
                    config,
                )?;
                MidBlockEnum::Regular(block)
            }
        };
        
        // Create up blocks
        let mut up_blocks = Vec::new();
        let mut reversed_block_out_channels = config.block_out_channels.clone();
        reversed_block_out_channels.reverse();
        
        for (i, up_block_type) in config.up_block_types.iter().enumerate() {
            let prev_output_channel = output_channel;
            output_channel = reversed_block_out_channels[i];
            let input_channel = reversed_block_out_channels.get(i + 1).copied().unwrap_or(prev_output_channel);
            let is_final_block = i == config.up_block_types.len() - 1;
            
            match up_block_type.as_str() {
                "CrossAttnUpBlock2D" => {
                    let config = CrossAttnUpBlock2DConfig {
                        num_layers: config.layers_per_block + 1,
                        resnet_eps: config.norm_eps,
                        resnet_groups: norm_num_groups,
                        add_upsample: !is_final_block,
                        attn_num_head_channels: config.attention_head_dim.get(config.up_block_types.len() - 1 - i).copied().unwrap_or(config.attention_head_dim[0]),
                        cross_attention_dim: config.cross_attention_dim,
                        use_linear_projection: config.use_linear_projection,
                        transformer_layers_per_block: 1,
                        ..Default::default()
                    };
                    
                    let block = CrossAttnUpBlock2DWithLoRA::new(
                        vs.pp(format!("up_blocks.{}", i)),
                        input_channel,
                        prev_output_channel,
                        output_channel,
                        Some(time_embed_dim),
                        use_flash_attn,
                        config,
                        lora_config,
                        device.clone(),
                        dtype,
                    )?;
                    up_blocks.push(UpBlockEnum::CrossAttnWithLoRA(block));
                }
                "UpBlock2D" => {
                    let config = UpBlock2DConfig {
                        num_layers: config.layers_per_block + 1,
                        resnet_eps: config.norm_eps,
                        resnet_groups: norm_num_groups,
                        add_upsample: !is_final_block,
                        ..Default::default()
                    };
                    
                    let block = UpBlock2D::new(
                        vs.pp(format!("up_blocks.{}", i)),
                        input_channel,
                        prev_output_channel,
                        output_channel,
                        Some(time_embed_dim),
                        config,
                    )?;
                    up_blocks.push(UpBlockEnum::Regular(block));
                }
                _ => return Err(candle_core::Error::Msg(
                    format!("Unknown up block type: {}", up_block_type)
                )),
            }
        }
        
        // Output layers
        let conv_norm_out = nn::group_norm(
            norm_num_groups,
            config.block_out_channels[0],
            config.norm_eps,
            vs.pp("conv_norm_out"),
        )?;
        
        let conv_out = conv2d(
            config.block_out_channels[0],
            config.out_channels,
            3,
            nn::Conv2dConfig { padding: 1, ..Default::default() },
            vs.pp("conv_out"),
        )?;
        
        Ok(Self {
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
            config,
            device,
            dtype,
        })
    }
    
    pub fn forward(
        &self,
        sample: &Tensor,
        timestep: &Tensor,
        encoder_hidden_states: &Tensor,
        added_cond_kwargs: Option<&HashMap<String, Tensor>>,
    ) -> Result<Tensor> {
        // Prepare time embeddings
        let t_emb = self.time_proj.forward(timestep)?;
        let emb = self.time_embedding.forward(&t_emb)?;
        
        // Add additional embeddings if present
        let emb = if let (Some(add_time_proj), Some(add_embedding)) = (&self.add_time_proj, &self.add_embedding) {
            if let Some(kwargs) = added_cond_kwargs {
                if let (Some(text_embeds), Some(time_ids)) = (
                    kwargs.get("text_embeds"),
                    kwargs.get("time_ids"),
                ) {
                    let aug_emb = add_time_proj.forward(&time_ids.flatten(0, 1)?)?;
                    let aug_emb = Tensor::cat(&[text_embeds, &aug_emb], D::Minus1)?;
                    let aug_emb = add_embedding.forward(&aug_emb)?;
                    (emb + aug_emb)?
                } else {
                    emb
                }
            } else {
                emb
            }
        } else {
            emb
        };
        
        // Initial convolution
        let mut hidden_states = self.conv_in.forward(sample)?;
        
        // Down blocks
        let mut down_block_res_samples = Vec::new();
        for block in &self.down_blocks {
            match block {
                DownBlockEnum::Regular(block) => {
                    let (hidden_states_new, res_samples) = block.forward(&hidden_states, Some(&emb))?;
                    hidden_states = hidden_states_new;
                    down_block_res_samples.extend(res_samples);
                }
                DownBlockEnum::CrossAttnWithLoRA(block) => {
                    let (hidden_states_new, res_samples) = block.forward(
                        &hidden_states,
                        Some(&emb),
                        Some(encoder_hidden_states),
                    )?;
                    hidden_states = hidden_states_new;
                    down_block_res_samples.extend(res_samples);
                }
            }
        }
        
        // Mid block
        hidden_states = match &self.mid_block {
            MidBlockEnum::Regular(block) => block.forward(&hidden_states, Some(&emb))?,
            MidBlockEnum::CrossAttnWithLoRA(block) => block.forward(
                &hidden_states,
                Some(&emb),
                Some(encoder_hidden_states),
            )?,
        };
        
        // Up blocks
        for (i, block) in self.up_blocks.iter().enumerate() {
            let res_samples = down_block_res_samples.split_off(
                down_block_res_samples.len() - self.config.layers_per_block - 1
            );
            
            hidden_states = match block {
                UpBlockEnum::Regular(block) => block.forward(
                    &hidden_states,
                    &res_samples,
                    Some(&emb),
                    None,
                )?,
                UpBlockEnum::CrossAttnWithLoRA(block) => block.forward(
                    &hidden_states,
                    &res_samples,
                    Some(&emb),
                    None,
                    Some(encoder_hidden_states),
                )?,
            };
        }
        
        // Output layers
        let hidden_states = self.conv_norm_out.forward(&hidden_states)?;
        let hidden_states = candle_nn::ops::silu(&hidden_states)?;
        self.conv_out.forward(&hidden_states)
    }
    
    /// Get all LoRA layers from the UNet
    pub fn get_all_lora_layers(&self) -> Vec<(String, &LoRALinear)> {
        let mut all_layers = Vec::new();
        
        // Collect from down blocks
        for (i, block) in self.down_blocks.iter().enumerate() {
            if let DownBlockEnum::CrossAttnWithLoRA(block) = block {
                for (name, layer) in block.get_lora_layers() {
                    all_layers.push((format!("down_blocks.{}.{}", i, name), layer));
                }
            }
        }
        
        // Collect from mid block
        if let MidBlockEnum::CrossAttnWithLoRA(block) = &self.mid_block {
            for (name, layer) in block.get_lora_layers() {
                all_layers.push((format!("mid_block.{}", name), layer));
            }
        }
        
        // Collect from up blocks
        for (i, block) in self.up_blocks.iter().enumerate() {
            if let UpBlockEnum::CrossAttnWithLoRA(block) = block {
                for (name, layer) in block.get_lora_layers() {
                    all_layers.push((format!("up_blocks.{}.{}", i, name), layer));
                }
            }
        }
        
        all_layers
    }
}