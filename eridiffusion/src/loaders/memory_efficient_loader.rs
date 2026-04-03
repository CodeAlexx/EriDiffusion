use crate::loaders::WeightLoader;
use anyhow::Context;
use flame_core::{DType, Error, Result, Shape, Tensor, CudaDevice};
use flame_core::device::Device;
use flame_core::optimizers::{Adam, SGD};
use std::{Mutex, collections::HashMap, path::Path};
use std::sync::Arc;
use super::{Architecture, FluxAdapter};
use super::unified_loader::WeightAdapter;

pub struct PrefixedWeightLoader {
    loader: WeightLoader,
prefix: String,
}
pub struct LazyHashMap<String, Tensor> {
/// CPU tensors waiting to be moved to GPU
/// GPU tensors that have been accessed
gpu_vars: HashMap<String, Tensor>,
/// Target device
device: Device,
/// Target dtype
pub struct MemoryEfficientFluxLoader {
    device: Device,
dtype: DType,
adapter: FluxAdapter,
}

// Memory-efficient model loading for large models
//
// This loader keeps weights on CPU and only moves them to GPU when needed

// FLAME uses flame_core::device::Device instead of Device

// FLAME uses flame_core::device::Device instead of Device

/// A HashMap<String, Tensor> that keeps tensors on CPU until accessed

// WeightLoader implementation is in crate::loaders::WeightLoader)
}

pub fn from_safetensors_multi(paths: &[&str], device: Device) -> flame_core::Result<Self> {
let mut weights = HashMap::new();
for path in paths {
let path_weights = crate::loaders::WeightLoader::from_safetensors(path, device.clone())?;
weights.extend(path_weights);
}
Ok(Self { weights, device })

pub fn tensor(&self, key: &str, shape: &[usize]) -> flame_core::Result<Tensor> {
let weight = self.get(key)?;
if weight.shape() != shape {
return Err(anyhow::anyhow!("Shape mismatch for {}: expected {:?}, got {:?}",
key, shape, weight.shape());
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


impl PrefixedWeightLoader {
pub fn get(&self, key: &str) -> flame_core::Result<&Tensor> {
let full_key = format!("{}.{}", self.prefix, key);
self.loader.get(&full_key)
}

pub fn tensor(&self, key: &str, shape: &[usize]) -> flame_core::Result<Tensor> {
let full_key = format!("{}.{}", self.prefix, key);
self.loader.tensor(&full_key, shape)
}

pub fn pp(&self, prefix: &str, device: &CudaDevice) -> PrefixedWeightLoader {
PrefixedWeightLoader {
loader: self.loader.clone(),
            prefix: format!("{}.{}", self.prefix, prefix),
}
}

impl LazyHashMap<String, Tensor> {
pub fn new(device: Device, dtype: DType) -> Self {
Self {
cpu_tensors: Arc::new(Mutex::new(HashMap::new())),
gpu_vars: HashMap::new(),
device,
dtype}

/// Add a tensor (kept on CPU)
pub fn insert(&self, name: String, tensor: Tensor) -> flame_core::Result<Tensor> {
let mut cpu_tensors = self.cpu_tensors.lock().unwrap();
cpu_tensors.insert(name, tensor);
Ok(())
}

/// Get or create a Parameter, moving to GPU on demand
pub fn get_or_create_var(&self, name: &str) -> flame_core::Result<Tensor> {
// Check if already on GPU
if let Some(var) = self.gpu_vars.data().lock().unwrap().get(name) {
return Ok(var.clone());
}

// Move from CPU to GPU
let tensor = {
let mut cpu_tensors = self.cpu_tensors.lock().unwrap();
cpu_tensors.remove(name)
.map_err(|e| flame_core::Error::InvalidOperation(format!("Tensor '{}' not found", name)))?
};

println!("Moving tensor '{}' to GPU ({})", name, tensor.shape().dims().iter().product::<usize>());

// Move to target device and dtype
let tensor = if tensor.device().location() != self.device.location() {
tensor
} else {
tensor
};

let tensor = if tensor.dtype() != self.dtype {
tensor.to_dtype(self.dtype)?
} else {
tensor
};

// Create Parameter and store
let var = tensor.clone();
self.gpu_vars.data().lock().unwrap().insert(name.to_string(), var.clone());

Ok(var)
}

/// Create a WeightLoader that loads tensors on demand
pub fn create_var_builder(&self, device: &CudaDevice) -> WeightLoader {
WeightLoader::from_varmap(&self.gpu_vars, self.dtype, &self.device)
}

/// Memory-efficient loader for Flux models

impl MemoryEfficientFluxLoader {
pub fn new(device: Device, dtype: DType, hidden_size: usize) -> Self {
Self {
device,
dtype,

/// Load model with memory-efficient strategy
pub fn load(&self, path: &Path) -> flame_core::Result<Tensor> {
println!("Loading model with memory-efficient strategy...");

// Load to CPU
.context("Failed to load safetensors file")?;

println!("Loaded {} tensors to CPU", tensors.len());

// Detect architecture
let source_arch = Architecture::detect(&tensors);
println!("Detected architecture: {:?}", source_arch);

// Create lazy var map
let lazy_map = LazyHashMap<String, Tensor>::new(self.device, self.dtype);

// Adapt and store tensors (keeping on CPU)
for (name, tensor) in tensors {
let adapted_tensors = self.adapter.adapt_tensor(&name, tensor)?;
for (new_name, new_tensor) in adapted_tensors {
lazy_map.insert(new_name, new_tensor)?;
}

println!("Prepared {} tensors for lazy loading",
lazy_map.cpu_tensors.lock().unwrap().len());

Ok(lazy_map)
}

/// Create a memory-efficient Flux model
pub fn create_memory_efficient_flux_model(
checkpoint_path: &Path,
device: Device,
dtype: DType,
) -> flame_core::Result<Tensor> {

println!("Creating memory-efficient Flux model...");

// Load weights with memory-efficient loader
let loader = MemoryEfficientFluxLoader::new(device, dtype, 3072);
let lazy_map = loader.load(checkpoint_path)?;

// Create model configuration
let config = FluxCustomConfig::default();

// Create model with empty weights first
let wl = lazy_map.create_var_builder();

// Build model structure (doesn't load weights yet)
println!("Building model structure...");
let mut model = FluxModelWithLoRA::new(&config, wl)?;

// Configure LoRA
let lora_config = LoRAConfig {
rank: 16,
alpha: 16.0,
dropout: Some(0.0),
target_modules: vec![
"attn".to_string(),
"mlp".to_string(),
],
module_filters: vec![],;

println!("Adding LoRA layers...");
model.add_lora_to_all(&lora_config, &device, dtype)?;

println!("Model created successfully with lazy weight loading!");
println!("Weights will be moved to GPU on demand during forward passes");

Ok(())
}
}
}
}
}
}
}
}
