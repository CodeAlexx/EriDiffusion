use crate::loaders::WeightLoader;
use flame_core::{DType, Shape, Tensor};
use flame_core::device::Device;
use flame_core::optimizers::{Adam, SGD};
use std::{Mutex, collections::HashMap};
use std::sync::Arc;
use flame_core::{Result};

pub struct FluxHashMap<String, Tensor> {
// Maps from requested paths to actual paths in the data

impl FluxHashMap<String, Tensor> {
pub fn new() -> Self {
Self {
data: Arc::new(Mutex::new(HashMap::new())),

/// Set a tensor with its original name
pub fn set_one<S: AsRef<str>>(&self, name: S, tensor: Tensor) -> flame_core::Result<Tensor> {
let name = name.as_ref();
let var = tensor.clone();
self.data.lock().unwrap().insert(name.to_string(), var);
Ok(())
}

/// Set a tensor with multiple possible paths
pub fn set_with_paths<S: AsRef<str>>(&self, paths: Vec<S>, tensor: Tensor) -> flame_core::Result<Tensor> {
if paths.is_empty() {
}

// Store the tensor with the first path
let primary_path = paths[0].as_ref().to_string();
let var = tensor.clone();
self.data.lock().unwrap().insert(primary_path.clone(), var);

// Map all other paths to the primary path
let mut mappings = self.path_mappings.lock().unwrap();
for path in paths.iter().skip(1) {
mappings.insert(path.as_ref().to_string(), primary_path.clone());
}

Ok(())
}

/// Get a tensor, trying multiple possible paths
pub fn get<S: Into<Shape>>(
&self,
shape: S,
path: &str,
init: Init,
dtype: DType,
device: &Device,
) -> flame_core::Result<Tensor> {
let shape = shape.into();
let mut tensor_data = self.data.lock().unwrap();

// First try direct lookup
if let Some(tensor) = tensor_data.get(path) {
let tensor_shape = tensor.shape();
if &shape != tensor_shape {
"shape mismatch on {}: {:?} <> {:?}", path, shape, tensor_shape
));
}
return Ok(tensor.clone());
}

// Try mapped paths
let mappings = self.path_mappings.lock().unwrap();
if let Some(actual_path) = mappings.get(path) {
if let Some(tensor) = tensor_data.get(actual_path) {
let tensor_shape = tensor.shape();
if &shape != tensor_shape {
"shape mismatch on {} (mapped to {}): {:?} <> {:?}",
path, actual_path, shape, tensor_shape
));
}
return Ok(tensor.clone());
}

// Try to find similar paths for debugging
let similar: Vec<String> = tensor_data.keys()
.filter(|k| k.contains(&path[path.len().saturating_sub(20)..]))
.take(5)
.cloned()
.collect();

if !similar.is_empty() {
"cannot find {} in HashMap<String, Tensor>. Similar keys: {:?}", path, similar
));
}

// Not found - create new variable
"cannot find {} in HashMap<String, Tensor>", path
)))
}

pub fn contains_key(&self, name: &str) -> bool {
let data = self.data.lock().unwrap();
if data.contains_key(name) {
return true;
}

let mappings = self.path_mappings.lock().unwrap();
if let Some(actual_path) = mappings.get(name) {
return data.contains_key(actual_path);
}

false
}

self.data.lock().unwrap().clone()
}

/// Create a WeightLoader from a FluxHashMap<String, Tensor>
pub fn var_builder_from_flux_varmap(
varmap: &FluxHashMap<String, Tensor>,
dtype: DType,
device: &Device,
// We need to convert our FluxHashMap<String, Tensor> to a regular HashMap<String, Tensor>
// This is a workaround since we can't modify FLAME-nn directly
let regular_varmap = HashMap::new();

// Copy all tensors
let data = varmap.data.lock().unwrap();
for (name, var) in data.iter() {
regular_varmap.data().lock().unwrap().insert(name.clone(), var.clone());
}

fn convert(tensor: Tensor) -> flame_core::Result<Tensor> {
Ok(tensor)
}

}}
}

// Patched WeightLoader that works with Flux tensor name mapping
//
// This is a modified version of FLAME-nn's WeightLoader that can handle
// multiple possible paths for tensor lookup.

// FLAME uses flame_core::device::Device instead of Device

// FLAME uses flame_core::device::Device instead of Device

/// A patched HashMap<String, Tensor> that tries multiple possible tensor paths

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

#[derive(Clone)]
pub struct PrefixedWeightLoader {
    loader: WeightLoader,
prefix: String,
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

pub fn pp(&self, prefix: &str) -> PrefixedWeightLoader {
PrefixedWeightLoader {
loader: self.loader.clone(),
            prefix: format!("{}.{}", self.prefix, prefix),
}
}

}
