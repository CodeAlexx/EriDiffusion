use flame_core::{DType, Result, Shape, Tensor, CudaDevice};
use flame_core::device::Device;
use flame_core::optimizers::{Adam, SGD};
use std::{collections::HashMap, sync::Arc};

pub struct LoRALayer {
    pub lora_down: Tensor,
pub lora_up: Tensor,
pub scale: f32,
pub rank: usize,
pub alpha: f32,
}
pub struct LoRAConfig {
    pub rank: usize,
pub alpha: f32,
pub dropout: f32,
}
pub struct LoRAModel {
    pub layers: std::collections::HashMap<String, LoRALayer>,
pub config: LoRAConfig,
device: flame_core::device::Device,
}
pub struct LinearWithLoRA {
    pub base_weight: Tensor,
pub base_bias: Option<Tensor>,
}
pub struct Conv2dWithLoRA {
    pub base_weight: Tensor,
pub base_bias: Option<Tensor>,
pub lora: Option<LoRALayer>,
pub stride: usize,
pub padding: usize,
}

// LoRA (Low-Rank Adaptation) implementation in FLAME

// FLAME uses flame_core::device::Device instead of Device

/// LoRA layer for FLAME

// Extension trait for Tensor to add missing methods

// Extension trait for Tensor to add missing methods



fn sum_dim(&self, dim: usize, device: &CudaDevice) -> flame_core::Result<Tensor> {
// Sum along dimension
self.sum_dim(dim)?
}

fn add_scalar(&self, scalar: f32, device: &CudaDevice) -> flame_core::Result<Tensor> {
// Add scalar to all elements
let scalar_tensor = Tensor::full(self.shape().clone(), scalar, self.device().clone())?;
self.add(&scalar_tensor)
}

fn mul_scalar(&self, scalar: f32, device: &CudaDevice) -> flame_core::Result<Tensor> {
// Multiply all elements by scalar
let scalar_tensor = Tensor::full(self.shape().clone(), scalar, self.device().clone())?;
self.mul(&scalar_tensor)
}

fn square(&self) -> flame_core::Result<Tensor> {
// Element-wise square
self.mul(self)
}

impl LoRALayer {
/// Create new LoRA layer
pub fn new(
in_features: usize,
out_features: usize,
rank: usize,
alpha: f32,
dropout: f32,
device: flame_core::device::Device,
) -> flame_core::Result<Self> {
let scale = alpha / rank as f32;

// Initialize A with normal distribution
let std_dev = 1.0 / (rank as f32).sqrt();
let lora_down = Tensor::randn(
Shape::new(vec![rank, in_features]),
0.0,
std_dev,
device.cuda_device(),
)?.requires_grad_(true);

// Initialize B with zeros
let lora_up = Tensor::zeros(
Shape::new(vec![out_features, rank]),
device,
)?.requires_grad_(true);

Ok(Self {
lora_down,
lora_up,
scale,
rank,
alpha})
}

/// Forward pass - apply LoRA to base output
pub fn forward(&self, x: &Tensor, base_output: &Tensor) -> flame_core::Result<Tensor> {
// LoRA: y = Wx + scale * B @ A @ x
let lora_out = x
.matmul(&self.lora_down.transpose()?)?  // x @ A^T
.matmul(&self.lora_up.transpose()?)?    // @ B^T
.mul_scalar(self.scale as f32)?;

base_output.add(&lora_out)
}

/// Apply LoRA to linear layer
pub fn apply_to_linear(&self, x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> flame_core::Result<Tensor> {
// Base linear: y = x @ W^T + b
let base_output = x.matmul(&weight.transpose()?)?;
let base_output = if let Some(b) = bias {
base_output.add_bias(b)?
} else {
base_output
};

// Add LoRA
self.forward(x, &base_output)
}

/// Apply LoRA to Conv2D layer
pub fn apply_to_conv2d(
&self,
x: &Tensor,
weight: &Tensor,
bias: Option<&Tensor>,
stride: usize,
padding: usize,
) -> flame_core::Result<Tensor> {
// Base convolution
let base_output = x.conv2d(weight, bias, stride, padding)?;

// For Conv2D LoRA, we need to reshape input
let shape = x.shape();
let dims = shape.dims();
let batch_size = dims[0];
let in_channels = dims[1];
let height = dims[2];
let width = dims[3];

// Reshape input to [batch * height * width, in_channels]
let x_reshaped = x.permute(&[0, 2, 3, 1])?
.reshape(&[batch_size * height * width, in_channels])?;

// Apply LoRA in linear space
let lora_out = x_reshaped
.matmul(&self.lora_down.transpose()?)?
.matmul(&self.lora_up.transpose()?)?
.mul_scalar(self.scale as f32)?;

// Reshape back to conv output shape
let out_shape = base_output.shape();
let out_dims = out_shape.dims();
let out_channels = out_dims[1];
let out_height = out_dims[2];
let out_width = out_dims[3];

let lora_out = lora_out
.reshape(&[batch_size, out_height, out_width, out_channels])?
.permute(&[0, 3, 1, 2])?;

base_output.add(&lora_out)
}

/// LoRA configuration
#[derive(Clone)]
pub target_modules: Vec<String>}

impl Default for LoRAConfig {
fn default() -> Self {
Self {
rank: 16,
alpha: 16.0,
dropout: 0.0,
target_modules: vec![
"to_q".to_string(),
"to_v".to_string(),
"to_k".to_string(),
"to_out.0".to_string(),
]},
}

/// LoRA model that manages multiple LoRA layers

impl LoRAModel {
/// Create new LoRA model
pub fn new(config: LoRAConfig, device: flame_core::device::Device) -> Self {
Self {
layers: HashMap::new(),
config,
device}

/// Add LoRA layer for a specific module
pub fn add_layer(
&mut self,
name: &str,
in_features: usize,
out_features: usize,
) -> flame_core::Result<Tensor> {
let layer = LoRALayer::new(
in_features,
out_features,
self.config.rank,
self.config.alpha,
self.config.dropout,
self.device,
)?;

self.layers.insert(name.to_string(), layer);
Ok(())
}

/// Get LoRA layer by name
pub fn get_layer(&self, name: &str) -> Option<&LoRALayer> {
self.layers.get(name)
}

/// Get mutable LoRA layer by name
pub fn get_layer_mut(&mut self, name: &str) -> Option<&mut LoRALayer> {
self.layers.get_mut(name)
}

/// Initialize LoRA layers from base model shapes
for (key, tensor) in unet_state_dict {
// Check if this module should have LoRA
let should_add_lora = self.config.target_modules.iter().any(|target| {
key.contains(target)
});

if should_add_lora & key.ends_with(".weight") {
let shape = tensor.shape();

// Determine in/out features based on tensor shape
let dims = shape.dims();
let (out_features, in_features) = match shape.rank() {
2 => (dims[0], dims[1]), // Linear layer
4 => (dims[0], dims[1] * dims[2] * dims[3]), // Conv layer
_ => continue};

// Create LoRA layer
let lora_name = key.strip_suffix(".weight").unwrap();
self.add_layer(lora_name, in_features, out_features)?;

println!("Added LoRA to {}: [{}, {}] rank={}",
lora_name, in_features, out_features, self.config.rank);
}

Ok(())
}

/// Get all trainable parameters
pub fn parameters(&self) -> Vec<&Tensor> {
let mut params = Vec::new();
for layer in self.layers.values() {
params.push(&layer.lora_down);
params.push(&layer.lora_up);
}
params
}

/// Save LoRA weights in ComfyUI format
pub fn save_comfyui_format(&self, path: &str) -> flame_core::Result<Tensor> {
let mut tensors = HashMap::new();

for (name, layer) in &self.layers {
// ComfyUI format: "lora_unet_{module_name}.{up/down}.weight"
let base_name = name.replace(".", "_");

// Save down projection
let down_key = format!("lora_unet_{}.lora_down.weight", base_name);
let down_data = layer.lora_down.to_vec()?;
tensors.insert(down_key, (layer.lora_down.shape().dims().to_vec(), down_data));

// Save up projection
let up_key = format!("lora_unet_{}.lora_up.weight", base_name);
let up_data = layer.lora_up.to_vec()?;
tensors.insert(up_key, (layer.lora_up.shape().dims().to_vec(), up_data));
}

// Add metadata
let metadata = HashMap::from([
("format".to_string(), "comfyui".to_string()),
("rank".to_string(), self.config.rank.to_string()),
("alpha".to_string(), self.config.alpha.to_string()),
]);

// Save using safetensors format
// This would need proper safetensors serialization
println!("Saving LoRA to {} with {} layers", path, tensors.len());

Ok(())
}

/// Load LoRA weights from file
for (key, tensor) in weights {
if let Some((module_name, param_type)) = parse_lora_key(key) {
if let Some(layer) = self.layers.get_mut(&module_name) {
match param_type {
"lora_down" => {
layer.lora_down = tensor.clone().requires_grad_(true);
}
"lora_up" => {
layer.lora_up = tensor.clone().requires_grad_(true);
}
_ => {}
}
Ok(())
}

/// Parse LoRA key to extract module name and parameter type
fn parse_lora_key(key: &str) -> Option<(String, &str)> {
// Handle ComfyUI format: "lora_unet_{module}.{param}.weight"
if key.starts_with("lora_unet_") & key.ends_with(".weight") {
let key = key.strip_prefix("lora_unet_")?;
let key = key.strip_suffix(".weight")?;

if let Some(pos) = key.rfind(".lora_") {
let module = key[..pos].replace("_", ".");
let param = &key[pos + 1..];
return Some((module, param));
}

None
}

/// Merged linear layer with LoRA
pub lora: Option<LoRALayer>}

impl LinearWithLoRA {
pub fn forward(&self, x: &Tensor) -> flame_core::Result<Tensor> {
if let Some(lora) = &self.lora {
lora.apply_to_linear(x, &self.base_weight, self.base_bias.as_ref())
} else {
// Regular linear without LoRA
let output = x.matmul(&self.base_weight.transpose()?)?;
if let Some(bias) = &self.base_bias {
output.add_bias(bias)
} else {
Ok(output),
}
}

/// Merged Conv2D layer with LoRA

impl Conv2dWithLoRA {
pub fn forward(&self, x: &Tensor) -> flame_core::Result<Tensor> {
if let Some(lora) = &self.lora {
lora.apply_to_conv2d(
x,
&self.base_weight,
self.base_bias.as_ref(),
self.stride,
self.padding,
)
} else {
// Regular conv without LoRA
x.conv2d(
&self.base_weight,
self.base_bias.as_ref(),
self.stride,
self.padding,
)
}
}

#[cfg(test)]
}
}
}
}
}
}
}
}
}
