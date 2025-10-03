use crate::loaders::WeightLoader;
use crate::models::resnet::{Downsample2D, ResnetBlock2D, Upsample2D};
use flame_core::{DType, Shape, Tensor};
use flame_core::device::Device;
use flame_core::optimizers::{Adam, SGD};
use std::collections::HashMap;
use flame_core::{Result};

pub struct CrossAttnDownBlock2DWithLoRA {
    resnets: Vec<ResnetBlock2D>,
attentions: Vec<SpatialTransformerWithLoRA>,
downsampler: Option<Downsample2D>,
config: CrossAttnDownBlock2DConfig,
}
pub struct CrossAttnUpBlock2DConfig {
    pub num_layers: usize,
pub resnet_eps: f64,
pub resnet_groups: usize,
pub output_scale_factor: f64,
pub add_upsample: bool,
pub attn_num_head_channels: usize,
pub cross_attention_dim: usize,
pub sliced_attention_size: Option<usize>,
pub use_linear_projection: bool,
pub transformer_layers_per_block: usize,
}
pub struct CrossAttnUpBlock2DWithLoRA {
    resnets: Vec<ResnetBlock2D>,
attentions: Vec<SpatialTransformerWithLoRA>,
upsampler: Option<Upsample2D>,
config: CrossAttnUpBlock2DConfig,
}
pub struct UNetMidBlock2DCrossAttnConfig {
    pub num_layers: usize,
pub resnet_eps: f64,
pub resnet_groups: Option<usize>,
pub attn_num_head_channels: usize,
pub output_scale_factor: f64,
pub cross_attn_dim: usize,
pub sliced_attention_size: Option<usize>,
pub use_linear_projection: bool,
pub transformer_layers_per_block: usize,
}
pub struct UNetMidBlock2DCrossAttnWithLoRA {
    resnet: ResnetBlock2D,
attn_resnets: Vec<(SpatialTransformerWithLoRA, ResnetBlock2D)>,
config: UNetMidBlock2DCrossAttnConfig,
}

/// LoRA-enabled UNet blocks for SDXL
// This module provides UNet building blocks with integrated LoRA support

// WeightLoader implementation is in crate::loaders::WeightLoader)
}

pub fn from_safetensors_multi(paths: &[&str], device: Device) -> Result<Self> {
let mut weights = HashMap::new();
for path in paths {
let path_weights = crate::loaders::WeightLoader::from_safetensors(path, device.clone())?;
weights.extend(path_weights);
}
Ok(Self { weights, device })

pub fn tensor(&self, key: &str, shape: &[usize]) -> Result<Tensor> {
let weight = self.get(key)?;
if weight.shape() != shape {
return Err(flame_core::Error::InvalidOperation(format!("Shape mismatch for {}: expected {:?}, got {:?}",
key, shape, weight.shape())));
}
Ok(weight.clone())
}

pub fn get_prefix(&self, prefix: &str) -> std::collections::HashMap<String, &Tensor> {
self.weights.iter()
.filter(|(k, _)| k.starts_with(prefix))
.map(|(k, v)| (k.clone(), v))
.collect()
}

pub fn pp(&self, prefix: &str) -> PrefixedWeightLoader {
PrefixedWeightLoader {
loader: self.clone(),
prefix: self.prefix.to_string(),
}
}

#[derive(Clone)]
pub struct PrefixedWeightLoader {
    loader: WeightLoader,
prefix: String,
}

impl PrefixedWeightLoader {
pub fn get(&self, key: &str) -> Result<&Tensor> {
let full_key = format!("{}.{}", self.prefix, key);
self.loader.get(&full_key)
}

pub fn tensor(&self, key: &str, shape: &[usize]) -> Result<Tensor> {
let full_key = format!("{}.{}", self.prefix, key);
self.loader.tensor(&full_key, shape)
}

pub fn pp(&self, prefix: &str) -> PrefixedWeightLoader {
PrefixedWeightLoader {
loader: self.loader.clone(),
            prefix: format!("{}.{}", self.prefix, prefix),
}
}

resnet::{ResnetBlock2D, ResnetBlock2DConfig};
lora_transformer::{SpatialTransformerWithLoRA, SpatialTransformerConfig},
lora_attention::LoRAAttentionConfig,;

/// Configuration for DownBlock2D
#[derive(Debug, Clone, Copy)]
pub struct DownBlock2DConfig {
    pub num_layers: usize,
pub resnet_eps: f64,
pub resnet_groups: usize,
pub output_scale_factor: f64,
pub add_downsample: bool,
pub downsample_padding: usize,
}

impl Default for DownBlock2DConfig {
fn default() -> Self {
Self {
num_layers: 1,
resnet_eps: 1e-6,
resnet_groups: 32,
output_scale_factor: 1.,
add_downsample: true,

/// Configuration for CrossAttnDownBlock2D with LoRA
#[derive(Debug, Clone, Copy)]
pub struct CrossAttnDownBlock2DConfig {
    pub downblock: DownBlock2DConfig,
pub attn_num_head_channels: usize,
pub cross_attention_dim: usize,
pub sliced_attention_size: Option<usize>,
pub use_linear_projection: bool,
pub transformer_layers_per_block: usize,
}

impl Default for CrossAttnDownBlock2DConfig {
fn default() -> Self {
Self {
downblock: Default::default(),
attn_num_head_channels: 1,
cross_attention_dim: 1280,
sliced_attention_size: None,
use_linear_projection: false,

/// CrossAttnDownBlock2D with LoRA support

impl CrossAttnDownBlock2DWithLoRA {
    fn new(
vs: nn::WeightLoader,
in_channels: usize,
out_channels: usize,
temb_channels: Option<usize>,
use_flash_attn: bool,
config: CrossAttnDownBlock2DConfig,
lora_config: Option<&LoRAAttentionConfig>,
device: Device,
dtype: DType,
) -> Result<Tensor> {
// Create ResNet blocks
let vs_resnets = vs.pp("resnets");
let resnet_cfg = ResnetBlock2DConfig {
out_channels: Some(out_channels),
eps: config.downblock.resnet_eps,
groups: config.downblock.resnet_groups,
output_scale_factor: config.downblock.output_scale_factor,
temb_channels,
..Default::default()
};

let resnets = (0..config.downblock.num_layers)
.map(|i| {
let in_channels = if i == 0 { in_channels } else { out_channels };
ResnetBlock2D::new(vs_resnets.pp(i.to_string()), in_channels, resnet_cfg)
})
.collect::<Result<Tensor>>()?;

// Create attention blocks with LoRA
let n_heads = config.attn_num_head_channels;
let transformer_cfg = SpatialTransformerConfig {
depth: config.transformer_layers_per_block,
context_dim: Some(config.cross_attention_dim),
num_groups: config.downblock.resnet_groups,
sliced_attention_size: config.sliced_attention_size,;

let vs_attn = vs.pp("attentions");
let attentions = (0..config.downblock.num_layers)
.map(|i| {
SpatialTransformerWithLoRA::new(
vs_attn.pp(i.to_string()),
out_channels,
n_heads,
out_channels / n_heads,
use_flash_attn,
transformer_cfg,
lora_config,
device,
dtype,
)
})
.collect::<Result<Tensor>>()?;

// Create downsampler if needed
let downsampler = if config.downblock.add_downsample {
Some(Downsample2D::new(
vs.pp("downsamplers").pp("0"),
out_channels,
true,
out_channels,
config.downblock.downsample_padding,
)?)
} else {
None
};

Ok(Self {
resnets,
attentions,
downsampler,
config})
}

pub fn forward(
&self,
xs: &Tensor,
temb: Option<&Tensor>,
encoder_hidden_states: Option<&Tensor>,
) -> Result<Tensor>)> {
let mut output_states = vec![];
let mut xs = xs.clone();

// Apply ResNet and attention blocks
for (resnet, attn) in self.resnets.iter().zip(self.attentions.iter()) {
xs = resnet.forward(&xs, temb)?;
xs = attn.forward(&xs, encoder_hidden_states)?;
output_states.push(xs.clone());
}

// Apply downsampler if present
let xs = match &self.downsampler {
Some(downsampler) => {
let xs = downsampler.forward(&xs)?;
output_states.push(xs.clone());
xs
}
None => xs};

Ok((xs, output_states))
}

/// Get all LoRA layers from attention blocks
pub fn get_lora_layers(&self) -> Vec<(&str, &super::sdxl_lora_layer::LoRALinear)> {
let mut all_layers = Vec::new();

for (block_idx, attn) in self.attentions.iter().enumerate() {
let block_prefix = format!("attentions.{}", block_idx);
for (layer_name, layer) in attn.get_lora_layers() {
all_layers.push((layer_name, layer));
}

all_layers
}

/// Configuration for CrossAttnUpBlock2D

impl Default for CrossAttnUpBlock2DConfig {
fn default() -> Self {
Self {
num_layers: 1,
resnet_eps: 1e-6,
resnet_groups: 32,
output_scale_factor: 1.,
add_upsample: true,
attn_num_head_channels: 1,
cross_attention_dim: 1280,
sliced_attention_size: None,
use_linear_projection: false,

/// CrossAttnUpBlock2D with LoRA support

impl CrossAttnUpBlock2DWithLoRA {
    fn new(
vs: nn::WeightLoader,
in_channels: usize,
prev_output_channels: usize,
out_channels: usize,
temb_channels: Option<usize>,
use_flash_attn: bool,
config: CrossAttnUpBlock2DConfig,
lora_config: Option<&LoRAAttentionConfig>,
device: Device,
dtype: DType,
) -> Result<Tensor> {
// Create ResNet blocks
let vs_resnets = vs.pp("resnets");
let resnet_cfg = ResnetBlock2DConfig {
out_channels: Some(out_channels),
temb_channels,
eps: config.resnet_eps,
groups: config.resnet_groups,
output_scale_factor: config.output_scale_factor,
..Default::default()
};

let resnets = (0..config.num_layers)
.map(|i| {
let res_skip_channels = if i == config.num_layers - 1 {
in_channels
} else {
out_channels
};
let resnet_in_channels = if i == 0 {
prev_output_channels
} else {
out_channels
};
let in_channels = resnet_in_channels + res_skip_channels;
ResnetBlock2D::new(vs_resnets.pp(i.to_string()), in_channels, resnet_cfg)
})
.collect::<Result<Tensor>>()?;

// Create attention blocks with LoRA
let n_heads = config.attn_num_head_channels;
let transformer_cfg = SpatialTransformerConfig {
depth: config.transformer_layers_per_block,
context_dim: Some(config.cross_attention_dim),
num_groups: config.resnet_groups,
sliced_attention_size: config.sliced_attention_size,;

let vs_attn = vs.pp("attentions");
let attentions = (0..config.num_layers)
.map(|i| {
SpatialTransformerWithLoRA::new(
vs_attn.pp(i.to_string()),
out_channels,
n_heads,
out_channels / n_heads,
use_flash_attn,
transformer_cfg,
lora_config,
device,
dtype,
)
})
.collect::<Result<Tensor>>()?;

// Create upsampler if needed
let upsampler = if config.add_upsample {
Some(Upsample2D::new(
vs.pp("upsamplers").pp("0"),
out_channels,
out_channels,
)?)
} else {
None
};

Ok(Self {
resnets,
attentions,
upsampler,
config})
}

pub fn forward(
&self,
xs: &Tensor,
res_xs: &[Tensor],
temb: Option<&Tensor>,
upsample_size: Option<(usize, usize)>,
encoder_hidden_states: Option<&Tensor>,
) -> Result<Tensor> {
let mut xs = xs.clone();

for (index, (resnet, attn)) in self.resnets.iter().zip(self.attentions.iter()).enumerate() {
// Concatenate with residual from encoder
xs = Tensor::cat(&[&xs, &res_xs[res_xs.len() - index - 1]], 1)?;
xs = xs;
xs = resnet.forward(&xs, temb)?;
xs = attn.forward(&xs, encoder_hidden_states)?;
}

// Apply upsampler if present
match &self.upsampler {
Some(upsampler) => upsampler.forward(&xs, upsample_size),
None => Ok(xs)}

/// Get all LoRA layers from attention blocks
pub fn get_lora_layers(&self) -> Vec<(&str, &super::sdxl_lora_layer::LoRALinear)> {
let mut all_layers = Vec::new();

for (block_idx, attn) in self.attentions.iter().enumerate() {
let block_prefix = format!("attentions.{}", block_idx);
for (layer_name, layer) in attn.get_lora_layers() {
all_layers.push((layer_name, layer));
}

all_layers
}

/// Configuration for UNetMidBlock2DCrossAttn

impl Default for UNetMidBlock2DCrossAttnConfig {
fn default() -> Self {
Self {
num_layers: 1,
resnet_eps: 1e-6,
resnet_groups: Some(32),
attn_num_head_channels: 1,
output_scale_factor: 1.,
cross_attn_dim: 1280,
sliced_attention_size: None,
use_linear_projection: false,

/// UNetMidBlock2DCrossAttn with LoRA support

impl UNetMidBlock2DCrossAttnWithLoRA {
    fn new(
vs: nn::WeightLoader,
in_channels: usize,
temb_channels: Option<usize>,
use_flash_attn: bool,
config: UNetMidBlock2DCrossAttnConfig,
lora_config: Option<&LoRAAttentionConfig>,
device: Device,
dtype: DType,
) -> Result<Tensor> {
let vs_resnets = vs.pp("resnets");
let vs_attns = vs.pp("attentions");

let resnet_groups = config
.resnet_groups
.unwrap_or_else(|| usize::min(in_channels / 4, 32));

let resnet_cfg = ResnetBlock2DConfig {
eps: config.resnet_eps,
groups: resnet_groups,
output_scale_factor: config.output_scale_factor,
temb_channels,
..Default::default()
};

let resnet = ResnetBlock2D::new(vs_resnets.pp("0"), in_channels, resnet_cfg)?;

let n_heads = config.attn_num_head_channels;
let transformer_cfg = SpatialTransformerConfig {
depth: config.transformer_layers_per_block,
num_groups: resnet_groups,
context_dim: Some(config.cross_attn_dim),
sliced_attention_size: config.sliced_attention_size,;

let mut attn_resnets = vec![];
for index in 0..config.num_layers {
let attn = SpatialTransformerWithLoRA::new(
vs_attns.pp(index.to_string()),
in_channels,
n_heads,
in_channels / n_heads,
use_flash_attn,
transformer_cfg,
lora_config,
device,
dtype,
)?;
let resnet = ResnetBlock2D::new(
vs_resnets.pp((index + 1).to_string()),
in_channels,
resnet_cfg,
)?;
attn_resnets.push((attn, resnet));
}

Ok(Self {
resnet,
attn_resnets,
config})
}

pub fn forward(
&self,
xs: &Tensor,
temb: Option<&Tensor>,
encoder_hidden_states: Option<&Tensor>,
) -> Result<Tensor> {
let mut xs = self.resnet.forward(xs, temb)?;

for (attn, resnet) in self.attn_resnets.iter() {
xs = attn.forward(&xs, encoder_hidden_states)?;
xs = resnet.forward(&xs, temb)?;
}

Ok(xs)
}

/// Get all LoRA layers from attention blocks
pub fn get_lora_layers(&self) -> Vec<(&str, &super::sdxl_lora_layer::LoRALinear)> {
let mut all_layers = Vec::new();

for (block_idx, (attn, _)) in self.attn_resnets.iter().enumerate() {
let block_prefix = format!("attentions.{}", block_idx);
for (layer_name, layer) in attn.get_lora_layers() {
all_layers.push((layer_name, layer));
}

all_layers
}

// Helper to fix import issues
impl Downsample2D {
fn new(
vs: nn::WeightLoader,
in_channels: usize,
use_conv: bool,
out_channels: usize,
padding: usize,
) -> Result<Tensor> {
let conv = if use_conv {
let config = nn::Conv2dConfig {
stride: 2,
padding,
..Default::default()
};
let conv = conv2d(in_channels, out_channels, 3, config, vs.pp("conv"))?;
Some(conv)
} else {
None
};
Ok(Self {
conv,
padding})
}

#[derive(Debug)]
struct Downsample2D {
    conv: Option<Conv2d>,
padding: usize,
}

impl Downsample2D {
fn forward(&self, xs: &Tensor) -> Result<Tensor> {
match &self.conv {
None => xs.avg_pool2d(2),
Some(conv) => {
if self.padding == 0 {
let xs = xs
.pad_with_zeros(-1, 0, 1)?
.pad_with_zeros(-2, 0, 1)?;
conv.forward(&xs)
} else {
conv.forward(xs)
}
}

impl Upsample2D {
fn new(vs: nn::WeightLoader, in_channels: usize, out_channels: usize) -> Result<Tensor> {
let config = nn::Conv2dConfig {
padding: 1,
..Default::default()
};
let conv = conv2d(in_channels, out_channels, 3, config, vs.pp("conv"))?;
Ok(Self { conv })
}

#[derive(Debug)]
struct Upsample2D {
    conv: Conv2d,
}

impl Upsample2D {
fn forward(&self, xs: &Tensor, size: Option<(usize, usize)>) -> Result<Tensor> {
let xs = match size {
None => {
let (_bsize, _channels, h, w) = xs(.shape()[0], .shape()[1], .shape()[2], .shape()[3])?;
xs.upsample_nearest2d(2 * h, 2 * w)?
}
Some((h, w)) => xs.upsample_nearest2d(h, w)?};
self.conv.forward(&xs)
}

}
}
}
}
}
}
}
}
}
}
}
