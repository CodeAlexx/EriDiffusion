use anyhow::anyhow;
use flame_core::{DType, Error, Result, Result as FlameResult, Shape, Tensor, CudaDevice};
use flame_core::device::Device;
use std::sync::Arc;
use std::collections::HashMap;
use crate::ops::{Conv2d, Linear, LayerNorm, RMSNorm};
use super::mmdit_blocks::{MMDiTConfig, MMDiT, AdaLayerNorm as MMDiTAdaLayerNorm, JointTransformerBlock, RoPE2D as MMDiTRoPE2D};

pub struct SD35MMDiT {
    config: SD35Config,

// Patch embedding
patch_embed: PatchEmbed,

// Position embedding
pos_embed: RoPE2D,

// Time embedding
time_proj: TimestepEmbedding,
vector_in: MlpEmbedder,

// Transformer blocks
joint_blocks: Vec<JointTransformerBlock>,

// Output
norm_out: AdaLayerNorm,
proj_out: Linear,

device: flame_core::device::Device,
}
struct PatchEmbed {
    proj: Conv2d,
}
struct TimestepEmbedding {
    linear_1: Linear,
linear_2: Linear,
}
struct MlpEmbedder {
    in_layer: Linear,
out_layer: Linear,
}
// JointTransformerBlock is imported from mmdit_blocks
struct MLP {
    fc1: Linear,
fc2: Linear,
}

// FLAME uses flame_core::device::Device instead of Device

/// 2D Rotary Position Embedding wrapper

/// Adaptive Layer Normalization for SD3.5

/// Joint attention for SD3.5 that processes image and text together
pub struct JointAttention {
    num_heads: usize,
head_dim: usize,
qkv_img: Linear,
qkv_txt: Linear,
proj: Linear,
scale: f32,
}

impl JointAttention {
pub fn new(hidden_size: usize, num_heads: usize, device: &Device) -> Result<Self> {
let head_dim = hidden_size / num_heads;
Ok(Self {
num_heads,
head_dim,
qkv_img: Linear::new(hidden_size, hidden_size * 3, true, &device.cuda_device())?,
qkv_txt: Linear::new(hidden_size, hidden_size * 3, true, &device.cuda_device())?,
proj: Linear::new(hidden_size, hidden_size, true, &device.cuda_device())?,
scale: (head_dim as f32).powf(-0.5),
})
}

pub fn forward(&self, x_img: &Tensor, x_txt: &Tensor) -> Result<(Tensor, Tensor)> {
let b = x_img.shape().dims()[0];
let h = self.num_heads;
let d = self.head_dim;

// Get Q, K, V for both image and text
let qkv_img = self.qkv_img.forward(x_img)?.reshape(&[b, -1, 3, h, d])?.permute(&[2, 0, 3, 1, 4])?;
let qkv_txt = self.qkv_txt.forward(x_txt)?.reshape(&[b, -1, 3, h, d])?.permute(&[2, 0, 3, 1, 4])?;

let q_img = qkv_img.slice(&[(0, 1)])?.squeeze(Some(0))?;
let k_img = qkv_img.slice(&[(1, 2)])?.squeeze(Some(0))?;
let v_img = qkv_img.slice(&[(2, 3)])?.squeeze(Some(0))?;

let q_txt = qkv_txt.slice(&[(0, 1)])?.squeeze(Some(0))?;
let k_txt = qkv_txt.slice(&[(1, 2)])?.squeeze(Some(0))?;
let v_txt = qkv_txt.slice(&[(2, 3)])?.squeeze(Some(0))?;

// Concatenate keys and values
let k = Tensor::cat(&[k_img, k_txt], 2)?;
let v = Tensor::cat(&[v_img, v_txt], 2)?;

// Compute attention for image
let attn_img = (q_img.matmul(&k.transpose_dims(2, 3)?)? * self.scale)?.softmax(-1)?;
let out_img = attn_img.matmul(&v)?;

// Compute attention for text
let attn_txt = (q_txt.matmul(&k.transpose_dims(2, 3)?)? * self.scale)?.softmax(-1)?;
let out_txt = attn_txt.matmul(&v)?;

// Reshape and project
let out_img = out_img.permute(&[0, 2, 1, 3])?.reshape(&[b, -1, h * d])?;
let out_txt = out_txt.permute(&[0, 2, 1, 3])?.reshape(&[b, -1, h * d])?;

Ok((self.proj.forward(&out_img)?, self.proj.forward(&out_txt)?))
}
}

pub struct AdaLayerNorm {
    linear: Linear,
norm: LayerNorm,
}

impl AdaLayerNorm {
pub fn new(in_features: usize, out_features: usize, eps: f32, device: &Device) -> Result<Self> {
Ok(Self {
linear: Linear::new(in_features, out_features * 2, true, &device.cuda_device())?,
norm: LayerNorm::new(vec![out_features], eps, device.cuda_device().clone())?,
})
}

pub fn forward(&self, x: &Tensor, emb: &Tensor) -> Result<(Tensor, Tensor)> {
let emb = self.linear.forward(emb)?;
let chunks = emb.chunk(2, 1)?;
let shift = &chunks[0];
let scale = &chunks[1];

let x = self.norm.forward(x)?;
let result = x.mul(&(scale.unsqueeze(2)?.unsqueeze(3)?.add_scalar(1.0)?))?.add(&shift.unsqueeze(2)?.unsqueeze(3)?)?;
Ok((result, emb))
}
}

pub struct RoPE2D {
    dim: usize,
max_seq_len: usize,
}

impl RoPE2D {
pub fn new(dim: usize, max_seq_len: usize, _device: &flame_core::device::Device) -> Result<Self> {
Ok(Self { dim, max_seq_len })
}

pub fn forward(&self, batch_size: usize, height: usize, width: usize) -> Result<Tensor> {
    // Generate actual position embeddings based on grid
    let device = Device::cuda(0)?;
    let seq_len = height * width;
    let embed_dim = self.dim;
    
    // Create proper sinusoidal position embeddings instead of zeros
    let mut positions = Vec::new();
    for y in 0..height {
        for x in 0..width {
            // 2D positional encoding
            let pos_y = y as f32 / height as f32;
            let pos_x = x as f32 / width as f32;
            
            // Create sinusoidal embeddings for each dimension
            for i in 0..(embed_dim / 4) {
                let div_term = 10000.0_f32.powf(2.0 * i as f32 / embed_dim as f32);
                positions.push((pos_y * div_term).sin());
                positions.push((pos_y * div_term).cos());
                positions.push((pos_x * div_term).sin());
                positions.push((pos_x * div_term).cos());
            }
        }
    }
    
    // Handle dimension padding if needed
    while positions.len() < batch_size * seq_len * embed_dim {
        positions.push(0.0);
    }
    positions.truncate(batch_size * seq_len * embed_dim);
    
    Tensor::from_vec(
        positions,
        Shape::from_dims(&[batch_size, seq_len, embed_dim]),
        device.cuda_device().clone(),
)?
}
}

/// SD3.5 model configuration
#[derive(Clone, Debug)]
pub struct SD35Config {
    pub model_type: String,
pub in_channels: usize,
pub out_channels: usize,
pub patch_size: usize,
pub hidden_size: usize,
pub num_layers: usize,
pub num_heads: usize,
pub mlp_ratio: f32,
pub pos_embed_max_size: usize,
pub dual_attention_layers: Vec<usize>,
pub qkv_bias: bool,
pub qk_norm: bool,
}

impl SD35Config {
pub fn sd35_large() -> Self {
Self {
model_type: "sd3.5-large".to_string(),
in_channels: 16,  // SD3 uses 16-channel VAE
out_channels: 16,
patch_size: 2,
hidden_size: 1536,
num_layers: 38,
num_heads: 24,
mlp_ratio: 4.0,
pos_embed_max_size: 192,
dual_attention_layers: vec![],  // All layers use joint attention
qkv_bias: false,
qk_norm: false,
}
}

pub fn sd35_medium() -> Self {
Self {
model_type: "sd3.5-medium".to_string(),
in_channels: 16,
out_channels: 16,
patch_size: 2,
hidden_size: 1024,
num_layers: 24,
num_heads: 16,
mlp_ratio: 4.0,
pos_embed_max_size: 192,
dual_attention_layers: vec![],
qkv_bias: false,
qk_norm: false,
}
}
}

/// SD3.5 MMDiT model

impl SD35MMDiT {
pub fn new(
config: SD35Config,
device: flame_core::device::Device,
weights: std::collections::HashMap<String, Tensor>,
) -> Result<Self> {
// Patch embedding
let patch_embed = PatchEmbed::new(
config.in_channels,
config.hidden_size,
config.patch_size,
&device,
&weights,
)?;

// Position embedding (RoPE)
let pos_embed = RoPE2D::new(
config.hidden_size / config.num_heads,
config.pos_embed_max_size,
&device,
)?;

// Time embedding
let time_proj = TimestepEmbedding::new(
256,  // Initial time embed dim
config.hidden_size,
&device,
)?;

// Vector embedder for pooled text embeddings
let vector_in = MlpEmbedder::new(
768 + 1280,  // CLIP-L + CLIP-G pooled dimensions
config.hidden_size,
&device,
)?;

// Create transformer blocks
let mut joint_blocks = Vec::new();
for i in 0..config.num_layers {
joint_blocks.push(JointTransformerBlock::new(
config.hidden_size,
config.num_heads,
config.mlp_ratio,
config.qkv_bias,
config.qk_norm,
i,
&device,
&weights,
)?);
}

// Output layers
let norm_out = AdaLayerNorm::new(
config.hidden_size,
config.hidden_size,
1e-6,
&device,
)?;

// Create proj_out linear layer
let proj_out_weight = weights.get("proj_out.weight")
    .ok_or_else(|| flame_core::Error::InvalidOperation("proj_out.weight not found".into()))?;
let weight_shape = proj_out_weight.shape();
let out_features = weight_shape.dims()[0];
let in_features = weight_shape.dims()[1];
let proj_out = Linear::new(in_features, out_features, weights.get("proj_out.bias").is_some(), &device)?;
// TODO: Load the weights after creating the Linear layer

Ok(Self {
config,
patch_embed,
pos_embed,
time_proj,
vector_in,
joint_blocks,
norm_out,
proj_out,
device})
}

pub fn forward(
&self,
x: &Tensor,              // Latent input [B, C, H, W]
timestep: &Tensor,       // Timestep [B]
context: &Tensor,        // Text embeddings [B, seq_len, hidden_size]
y: &Tensor,              // Pooled text embeddings [B, pooled_dim]
) -> Result<Tensor> {
let x_dims = x.shape().dims();
let batch_size = x_dims[0];
let orig_height = x_dims[2];
let orig_width = x_dims[3];

// 1. Patchify input
let x = self.patch_embed.forward(x)?;  // [B, num_patches, hidden_size]
let x_patched_dims = x.shape().dims();
let num_patches = x_patched_dims[1];

// 2. Create position embeddings
let pos_embed = self.pos_embed.forward(batch_size, orig_height / self.config.patch_size, orig_width / self.config.patch_size)?;

// 3. Time conditioning
let t = self.time_proj.forward(timestep)?;
let c_vector = self.vector_in.forward(y)?;
let c = t.add(&c_vector)?;  // Combined conditioning

// 4. Prepare for joint attention
// Concatenate image patches and text tokens
let x_and_context = Tensor::cat(&[&x, context], 1)?;

// 5. Apply transformer blocks
let mut hidden_states = x_and_context;
for block in &self.joint_blocks {
hidden_states = block.forward(&hidden_states, &c, &pos_embed)?;
}

// 6. Split output back to image part
let x_out = hidden_states.slice(&[(0, 0 + num_patches)])?;

// 7. Final norm and projection
let (x_out, _) = self.norm_out.forward(&x_out, &c)?;
let x_out = self.proj_out.forward(&x_out)?;

// 8. Unpatchify
let x_out = self.unpatchify(&x_out, orig_height / self.config.patch_size, orig_width / self.config.patch_size)?;

Ok(x_out)
}

fn unpatchify(&self, x: &Tensor, h: usize, w: usize) -> Result<Tensor> {
let x_dims = x.shape().dims();
let batch = x_dims[0];
let p = self.config.patch_size;
let c = self.config.out_channels;

// x: [B, h*w, p*p*c] -> [B, h, w, p, p, c]
let x = x.reshape(&[batch, h, w, p, p, c])?;

// Rearrange to [B, h, p, w, p, c]
let x = x.transpose_dims(2, 3)?;

// Reshape to [B, c, h*p, w*p]
let x = x.reshape(&[batch, c, h * p, w * p])?;

Ok(x)
}
}



/// Patch embedding layer
impl PatchEmbed {
    fn new(
        in_channels: usize,
        embed_dim: usize,
        patch_size: usize,
        device: &flame_core::device::Device,
        weights: &std::collections::HashMap<String, Tensor>,
    ) -> Result<Self> {
// Get weight shape to determine dimensions
let weight = weights.get("patch_embed.proj.weight")
    .ok_or_else(|| flame_core::Error::InvalidOperation("Missing patch_embed.proj.weight".into()))?;
let weight_shape = weight.shape();
let out_channels = weight_shape.dims()[0];
let has_bias = weights.get("patch_embed.proj.bias").is_some();

let proj = Conv2d::new(
    in_channels,
    out_channels,
    patch_size,
    patch_size,
    0,
    1,
    has_bias,
    device.cuda_device().clone(),
)?;
// TODO: Load weights after creating the Conv2d layer

Ok(Self { proj })
}

fn forward(&self, x: &Tensor) -> Result<Tensor> {
// Conv2d patchification
let x = self.proj.forward(x)?;

// Flatten patches: [B, C, H, W] -> [B, C, H*W] -> [B, H*W, C]
let shape = x.shape().dims();
let b = shape[0];
let c = shape[1];
let h = shape[2];
let w = shape[3];

x.reshape(&[b, c, h * w])?
.transpose_dims(1, 2)
    }
}

/// Time/timestep embedding
impl TimestepEmbedding {
    fn new(time_embed_dim: usize, out_dim: usize, device: &flame_core::device::Device) -> Result<Self> {
        Ok(Self {
            linear_1: Linear::new(time_embed_dim, out_dim, true, &device.cuda_device())?,
            linear_2: Linear::new(out_dim, out_dim, true, &device.cuda_device())?
        })
    }

    fn forward(&self, timestep: &Tensor) -> Result<Tensor> {
        // Convert timestep to sinusoidal embeddings
        let timesteps = get_timestep_embedding(timestep, 256)?;

        // MLP projection
        let emb = self.linear_1.forward(&timesteps)?;
        let emb = emb.silu()?;
        self.linear_2.forward(&emb)
    }
}

/// MLP embedder for vector inputs
impl MlpEmbedder {
    fn new(in_dim: usize, hidden_dim: usize, device: &flame_core::device::Device) -> Result<Self> {
        Ok(Self {
            in_layer: Linear::new(in_dim, hidden_dim, true, &device.cuda_device())?,
            out_layer: Linear::new(hidden_dim, hidden_dim, true, &device.cuda_device())?
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.in_layer.forward(x)?;
        let x = x.silu()?;
        self.out_layer.forward(&x)
    }
}

/// Joint transformer block implementation
impl JointTransformerBlock {
    fn new(
        hidden_size: usize,
        num_heads: usize,
        mlp_ratio: f32,
        qkv_bias: bool,
        qk_norm: bool,
        block_idx: usize,
        device: &flame_core::device::Device,
        weights: &std::collections::HashMap<String, Tensor>,
    ) -> Result<Self> {
let prefix = format!("joint_blocks.{}", block_idx);

Ok(Self {
norm1_context: RMSNorm::new(vec![hidden_size], 1e-6, true, device.cuda_device().clone())?,

norm1: AdaLayerNorm::new(hidden_size, hidden_size, 1e-6, device)?,


attn: JointAttention::new(
hidden_size,
num_heads,
device,
)?,

norm2_context: RMSNorm::new(vec![hidden_size], 1e-6, true, device.cuda_device().clone())?,

norm2: AdaLayerNorm::new(hidden_size, hidden_size, 1e-6, device)?,


mlp: MLP::new(
hidden_size,
(hidden_size as f32 * mlp_ratio) as usize,
device,
)?,

mlp_context: MLP::new(
hidden_size,
(hidden_size as f32 * mlp_ratio) as usize,
device,
)?})
}

fn forward(
&self,
x_and_context: &Tensor,
c: &Tensor,
pos_embed: &Tensor,
) -> Result<Tensor> {
// Split into image and text parts
let seq_len = x_and_context.shape().dims()[1];
let img_seq_len = pos_embed.shape().dims()[1];
let txt_seq_len = seq_len - img_seq_len;

let x = x_and_context.slice(&[(0, 0 + img_seq_len)])?;
let context = x_and_context.slice(&[(img_seq_len, img_seq_len + txt_seq_len)])?;

// Norm + attention
let (x_norm, _) = self.norm1.forward(&x, c)?;
let c_norm = self.norm1_context.forward(&context)?;

// Joint attention
let (x_out, c_out) = self.attn.forward(&x_norm, &c_norm)?;

// First residual
let x = x.add(&x_out)?;
let context = context.add(&c_out)?;

// Norm + MLP
let (x_norm, _) = self.norm2.forward(&x, c)?;
let c_norm = self.norm2_context.forward(&context)?;

let x_out = self.mlp.forward(&x_norm)?;
let c_out = self.mlp_context.forward(&c_norm)?;

// Second residual
let x = x.add(&x_out)?;
let context = context.add(&c_out)?;

// Concatenate back
Tensor::cat(&[&x, &context], 1)
    }
}

/// MLP block
impl MLP {
    fn new(in_features: usize, hidden_features: usize, device: &flame_core::device::Device) -> Result<Self> {
Ok(Self {
fc1: Linear::new(in_features, hidden_features, true, &device.cuda_device())?,
fc2: Linear::new(hidden_features, in_features, true, &device.cuda_device())?})
}

fn forward(&self, x: &Tensor) -> Result<Tensor> {
let x = self.fc1.forward(x)?;
let x = x.gelu()?;
self.fc2.forward(&x)
    }
}

/// Get sinusoidal timestep embeddings
fn get_timestep_embedding(timesteps: &Tensor, embedding_dim: usize) -> Result<Tensor> {
let device = Device::from(timesteps.device().clone());
let half_dim = embedding_dim / 2;

let emb = -(0..half_dim)
.map(|i| i as f32 * 2.0 / embedding_dim as f32)
.collect::<Vec<_>>();

let emb = Tensor::from_vec(emb, Shape::from_dims(&[half_dim]), device.cuda_device().clone())?;
let emb = emb.mul_scalar(10000f32.ln())?.exp()?;

let emb = timesteps.unsqueeze(1)?.mul(&emb.unsqueeze(0)?)?;
let sin = emb.sin()?;
let cos = emb.cos()?;

Tensor::cat(&[&sin, &cos], 1)
}

// FLAME uses flame_core::device::Device instead of Device

// FLAMEModel trait implementation removed - trait doesn't exist in FLAME
