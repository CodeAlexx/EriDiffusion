use crate::loaders::WeightLoader;
use flame_core::{DType, Shape, Tensor};
use flame_core::device::Device;
use flame_core::optimizers::{Adam, SGD};
use flame_core::{Result};

use std::collections::HashMap;
pub struct AttentionWithLoRA {
    num_heads: usize,
head_dim: usize,

// Base projections (frozen)

// LoRA modules (trainable with Parameter)
to_q_lora: Option<LoRALinearWithGradients>,
to_k_lora: Option<LoRALinearWithGradients>,
to_v_lora: Option<LoRALinearWithGradients>,
to_out_lora: Option<LoRALinearWithGradients>,

scale: f64,
}
pub struct CrossAttentionWithLoRA {
    num_heads: usize,
head_dim: usize,

to_q_lora: Option<LoRALinearWithGradients>,
to_k_lora: Option<LoRALinearWithGradients>,
to_v_lora: Option<LoRALinearWithGradients>,
to_out_lora: Option<LoRALinearWithGradients>,

scale: f64,
}

// LoRA implementation with proper gradient tracking using Parameter
// This implementation ensures LoRA weights are trainable and gradients flow correctly

// FLAME uses flame_core::device::Device instead of Device

// FLAME uses flame_core::device::Device instead of Device

/// LoRA Linear layer with gradient tracking

pub enum Init { Uniform { lo: f32, up: f32 }, Const(f32), Normal { mean: f32, std: f32 } }

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

#[derive(Clone)]
pub struct LoRALinearWithGradients {
    // Base layer weights (frozen - regular Tensor)
base_weight: Tensor,
base_bias: Option<Tensor>,

// LoRA parameters (trainable - using Parameter)
lora_a: Parameter,
lora_b: Parameter,

scale: f32,
}

impl LoRALinearWithGradients {
/// Create a new LoRA layer with gradient tracking
pub fn new(
wl: &WeightLoader,
in_features: usize,
out_features: usize,
rank: usize,
alpha: f32,
dropout_rate: f32,
) -> Result<Tensor> {
// Load base weights as regular tensors (frozen)
let base_weight = weights.tensor((out_features, in_features), "weight")?;
let base_bias = weights.tensor(out_features, "bias").ok();

// Create LoRA A matrix with Parameter for gradient tracking
let bound = (1.0 / (in_features as f64)).sqrt();
let init_a = Init::Uniform { lo: -bound, up: bound };
let lora_a_tensor = weights.get_with_hints((rank, in_features), "weight", init_a)?;
let lora_a = lora_a_tensor.clone();

// Create LoRA B matrix with Parameter (zero initialized)
let lora_b_tensor = Tensor::zeros_dtype(Shape::from_dims(&[out_features, rank]), wl.dtype(), wl.device())?;
let lora_b = lora_b_tensor.clone();

let scale = alpha / rank as f32;

let dropout = if dropout_rate > 0.0 {
Some(Dropout::new(dropout_rate))
} else {
None
};

Ok(Self {
base_weight,
base_bias,
lora_a,
lora_b,
scale,
dropout}),
}

/// Create without base layer (for adding LoRA to existing layers)
pub fn new_without_base(
device: &Device,
dtype: DType,
in_features: usize,
out_features: usize,
rank: usize,
alpha: f32,
dropout_rate: f32,
) -> Result<Tensor> {
// Create zero base weights
let base_weight = Tensor::zeros_dtype(Shape::from_dims(&[out_features, in_features]), dtype, device)?;

// Initialize LoRA A with kaiming uniform
let bound = (1.0 / (in_features as f32)).sqrt();
let lora_a_init = Tensor::randn(Shape::from_dims(&[rank, in_features]), 0.0, bound, device.cuda_device())?
.to_dtype(dtype)?;
let lora_a = lora_a_init.clone();

// Initialize LoRA B with zeros
let lora_b_init = Tensor::zeros_dtype(Shape::from_dims(&[out_features, rank]), dtype, device)?;
let lora_b = lora_b_init.clone();

let scale = alpha / rank as f32;

let dropout = if dropout_rate > 0.0 {
Some(Dropout::new(dropout_rate))
} else {
None
};

Ok(Self {
base_weight,
base_bias: None,
lora_a,
lora_b,
scale,
dropout})
}

/// Get trainable parameters (returns Parameter references)
pub fn trainable_vars(&self) -> Vec<&Parameter> {
vec![&self.lora_a, &self.lora_b]
}

/// Forward pass with LoRA computation in the graph
pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
// Base linear transformation: x @ W^T + b
let mut output = x.matmul(&self.base_weight.transpose_dims(0, 1)?)?;
if let Some(bias) = &self.base_bias {
output = output.add(bias)?;
}

// Apply dropout to input if training
let lora_input = if let Some(dropout) = &self.dropout {
dropout.forward(x, true)?
} else {
x.clone()
};

// LoRA computation: (x @ A^T) @ B^T * scale
// Using as_tensor() to get Tensor view of Parameter for computation
let h = lora_input.matmul(&self.lora_a.transpose_dims(0, 1)?)?;
let lora_out = h.matmul(&self.lora_b.transpose_dims(0, 1)?)?;
let scaled_lora = lora_out.mul_scalar(self.scale as f64)??;

// Add LoRA to base output
output.add(&scaled_lora)
}

/// Forward pass for LoRA only (to be added to external base output)
pub fn forward_lora_only(&self, x: &Tensor) -> Result<Tensor> {
// Apply dropout to input if training
let lora_input = if let Some(dropout) = &self.dropout {
dropout.forward(x, true)?
} else {
x.clone()
};

// LoRA computation: (x @ A^T) @ B^T * scale
let h = lora_input.matmul(&self.lora_a.transpose_dims(0, 1)?)?;
let lora_out = h.matmul(&self.lora_b.transpose_dims(0, 1)?)?;
lora_out.mul_scalar(self.scale as f64)?
}

/// Attention layer with built-in LoRA using gradient tracking

impl AttentionWithLoRA {
pub fn new(
wl: &WeightLoader,
embed_dim: usize,
num_heads: usize,
use_lora: bool,
lora_rank: usize,
lora_alpha: f32,
lora_dropout: f32,
target_modules: &[String],
) -> Result<Tensor> {
let head_dim = embed_dim / num_heads;
let scale = 1.0 / (head_dim as f64).sqrt();

// Load base layers

// Create LoRA layers if enabled
let (to_q_lora, to_k_lora, to_v_lora, to_out_lora) = if use_lora {
let device = Device::from(wl.device().clone());
let dtype = wl.dtype();

let q_lora = if target_modules.contains(&"to_q".to_string()) {
Some(LoRALinearWithGradients::new_without_base(
device, dtype, embed_dim, embed_dim, lora_rank, lora_alpha, lora_dropout
)?)
} else {
None
};

let k_lora = if target_modules.contains(&"to_k".to_string()) {
Some(LoRALinearWithGradients::new_without_base(
device, dtype, embed_dim, embed_dim, lora_rank, lora_alpha, lora_dropout
)?)
} else {
None
};

let v_lora = if target_modules.contains(&"to_v".to_string()) {
Some(LoRALinearWithGradients::new_without_base(
device, dtype, embed_dim, embed_dim, lora_rank, lora_alpha, lora_dropout
)?)
} else {
None
};

let out_lora = if target_modules.iter().any(|m| m == "to_out" || m == "to_out.0") {
Some(LoRALinearWithGradients::new_without_base(
device, dtype, embed_dim, embed_dim, lora_rank, lora_alpha, lora_dropout
)?)
} else {
None
};

(q_lora, k_lora, v_lora, out_lora)
} else {
(None, None, None, None)
};

Ok(Self {
num_heads,
head_dim,
to_q_base,
to_k_base,
to_v_base,
to_out_base,
to_q_lora,
to_k_lora,
to_v_lora,
to_out_lora,
scale})
}

fn apply_projection(
&self,
lora: &Option<LoRALinearWithGradients>,
x: &Tensor,
) -> Result<Tensor> {
let base_out = base.forward(x)?;

if let Some(lora_module) = lora {
// Add LoRA to base output
let lora_out = lora_module.forward_lora_only(x)?;
base_out.add(&lora_out)
} else {
Ok(base_out)
}

pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
let (b, n, c) = hidden_states(.shape()[0], .shape()[1], .shape()[2])?;

// Apply projections with LoRA
let q = self.apply_projection(&self.to_q_base, &self.to_q_lora, hidden_states)?;
let k = self.apply_projection(&self.to_k_base, &self.to_k_lora, hidden_states)?;
let v = self.apply_projection(&self.to_v_base, &self.to_v_lora, hidden_states)?;

// Reshape for multi-head attention
let q = q.reshape((b, n, self.num_heads, self.head_dim))?.transpose_dims(1, 2)?;
let k = k.reshape((b, n, self.num_heads, self.head_dim))?.transpose_dims(1, 2)?;
let v = v.reshape((b, n, self.num_heads, self.head_dim))?.transpose_dims(1, 2)?;

// Scaled dot-product attention
let scores = q.matmul(&k.transpose_dims(2, 3)?)?;
let scores = (scores * self.scale)?;

// Apply attention to values
let out = attn.matmul(&v)?;
let out = out.transpose_dims(1, 2)?.reshape((b, n, c))?;

// Output projection with LoRA
self.apply_projection(&self.to_out_base, &self.to_out_lora, &out)
}

/// Get all trainable LoRA parameters
pub fn trainable_vars(&self) -> Vec<&Parameter> {
let mut vars = vec![];

if let Some(lora) = &self.to_q_lora {
vars.extend(lora.trainable_vars());
}
if let Some(lora) = &self.to_k_lora {
vars.extend(lora.trainable_vars());
}
if let Some(lora) = &self.to_v_lora {
vars.extend(lora.trainable_vars());
}
if let Some(lora) = &self.to_out_lora {
vars.extend(lora.trainable_vars());
}

vars
}

/// Cross-attention with LoRA and gradient tracking

impl CrossAttentionWithLoRA {
pub fn new(
wl: &WeightLoader,
query_dim: usize,
context_dim: usize,
num_heads: usize,
use_lora: bool,
lora_rank: usize,
lora_alpha: f32,
lora_dropout: f32,
target_modules: &[String],
) -> Result<Tensor> {
let head_dim = query_dim / num_heads;
let scale = 1.0 / (head_dim as f64).sqrt();

// Base layers

// LoRA layers
let (to_q_lora, to_k_lora, to_v_lora, to_out_lora) = if use_lora {
let device = Device::from(wl.device().clone());
let dtype = wl.dtype();

(
if target_modules.contains(&"to_q".to_string()) {
Some(LoRALinearWithGradients::new_without_base(
device, dtype, query_dim, query_dim, lora_rank, lora_alpha, lora_dropout
)?)
} else { None },

if target_modules.contains(&"to_k".to_string()) {
Some(LoRALinearWithGradients::new_without_base(
device, dtype, context_dim, query_dim, lora_rank, lora_alpha, lora_dropout
)?)
} else { None },

if target_modules.contains(&"to_v".to_string()) {
Some(LoRALinearWithGradients::new_without_base(
device, dtype, context_dim, query_dim, lora_rank, lora_alpha, lora_dropout
)?)
} else { None },

if target_modules.iter().any(|m| m == "to_out" || m == "to_out.0") {
Some(LoRALinearWithGradients::new_without_base(
device, dtype, query_dim, query_dim, lora_rank, lora_alpha, lora_dropout
)?)
} else { None },
)
} else {
(None, None, None, None)
};

Ok(Self {
num_heads,
head_dim,
to_q_base,
to_k_base,
to_v_base,
to_out_base,
to_q_lora,
to_k_lora,
to_v_lora,
to_out_lora,
scale})
}

fn apply_projection(
&self,
lora: &Option<LoRALinearWithGradients>,
x: &Tensor,
) -> Result<Tensor> {
let base_out = base.forward(x)?;

if let Some(lora_module) = lora {
// Add LoRA to base output
let lora_out = lora_module.forward_lora_only(x)?;
base_out.add(&lora_out)
} else {
Ok(base_out)
}

pub fn forward(&self, hidden_states: &Tensor, encoder_hidden_states: &Tensor) -> Result<Tensor> {
let (b, n, c) = hidden_states(.shape()[0], .shape()[1], .shape()[2])?;
let (_, n_enc, _) = encoder_hidden_states(.shape()[0], .shape()[1], .shape()[2])?;

// Projections with LoRA
let q = self.apply_projection(&self.to_q_base, &self.to_q_lora, hidden_states)?;
let k = self.apply_projection(&self.to_k_base, &self.to_k_lora, encoder_hidden_states)?;
let v = self.apply_projection(&self.to_v_base, &self.to_v_lora, encoder_hidden_states)?;

// Reshape for attention
let q = q.reshape((b, n, self.num_heads, self.head_dim))?.transpose_dims(1, 2)?;
let k = k.reshape((b, n_enc, self.num_heads, self.head_dim))?.transpose_dims(1, 2)?;
let v = v.reshape((b, n_enc, self.num_heads, self.head_dim))?.transpose_dims(1, 2)?;

// Attention
let scores = q.matmul(&k.transpose_dims(2, 3)?)?;
let scores = (scores * self.scale)?;

let out = attn.matmul(&v)?;
let out = out.transpose_dims(1, 2)?.reshape((b, n, c))?;

// Output projection
self.apply_projection(&self.to_out_base, &self.to_out_lora, &out)
}

pub fn trainable_vars(&self) -> Vec<&Parameter> {
let mut vars = vec![];

if let Some(lora) = &self.to_q_lora {
vars.extend(lora.trainable_vars());
}
if let Some(lora) = &self.to_k_lora {
vars.extend(lora.trainable_vars());
}
if let Some(lora) = &self.to_v_lora {
vars.extend(lora.trainable_vars());
}
if let Some(lora) = &self.to_out_lora {
vars.extend(lora.trainable_vars());
}

vars
}

#[cfg(test)]
}
}
}
}
