use anyhow::Context;
use flame_core::device::Device;
use flame_core::optimizers::{Adam, SGD};
use flame_core::{DType, Error, Result, Shape, Tensor};
use std::{collections::HashMap, path::Path};

// FLAME uses flame_core::device::Device instead of Device

/// INT8 quantization for Flux models to reduce memory usage from ~22GB to ~11GB
/// Similar to bitsandbytes but in pure Rust
pub struct FluxInt8Loader {
    /// Quantized weights stored as INT8
    quantized_weights: std::collections::HashMap<String, Tensor>,
    /// Scale factors for each quantized tensor
    scale_factors: std::collections::HashMap<String, Tensor>,
    /// Device to load tensors on (for final output)
    device: Device,
    /// Target device for final tensors
    target_device: Device,
}

impl FluxInt8Loader {
    /// Create a new INT8 loader
    pub fn new(device: Device) -> flame_core::Result<Self> {
        Ok(Self {
            quantized_weights: HashMap::new(),
            scale_factors: HashMap::new(),
            device: flame_core::device::Device::cuda(0)?, // Always quantize on CPU first
            target_device: device,                        // Store the target device
        })
    }

    /// Load and quantize a Flux model from safetensors files
    pub fn load_and_quantize<P: AsRef<Path>>(&mut self, model_path: P) -> flame_core::Result<()> {
        let model_path = model_path.as_ref();

        println!("Loading model to CPU for quantization...");
        // Load the safetensors file to CPU first to avoid OOM
        let tensors = crate::loaders::WeightLoader::from_safetensors(
            model_path,
            flame_core::device::Device::cuda(0)?,
        )
        .map_err(|e| {
            flame_core::Error::InvalidOperation(format!(
                "Failed to load model from {:?}",
                model_path
            ))
        })?;

        println!("Loaded {} tensors, starting quantization...", tensors.weights.len());

        // Process tensors in batches to manage memory
        let mut processed = 0;
        for (name, tensor) in tensors.weights.into_iter() {
            let name: String = name;
            // Skip non-weight tensors (biases, norms, etc.)
            if self.should_quantize(&name) {
                let (quantized, scale) = self.quantize_tensor(&tensor)?;
                // Keep on CPU for now, will move to GPU later
                self.quantized_weights.insert(name.clone(), quantized);
                self.scale_factors.insert(name.clone(), scale);
            } else {
                // Keep small tensors on CPU too
                self.quantized_weights.insert(name, tensor);
            }

            processed += 1;
            if processed % 50 == 0 {
                println!("Quantized {}/{} tensors", processed, 780); // Flux has ~780 tensors
            }
        }

        println!("Quantization complete.");
        Ok(())
    }

    /// Quantize a single tensor to INT8 with scale factor
    fn quantize_tensor(&self, tensor: &Tensor) -> flame_core::Result<(Tensor, Tensor)> {
        // Convert to f32 for quantization if needed
        let tensor_f32 = if tensor.dtype() == DType::F32 {
            tensor.clone()
        } else {
            tensor.to_dtype(DType::F32)?
        };

        // Get absolute max value for scaling
        let abs_max = tensor_f32.abs()?.max_all()?;

        // Compute scale factor (127 is max INT8 value)
        let scale_value = abs_max / 127.0;

        // Avoid division by zero
        let scale = if scale_value < 1e-10 { 1e-10 } else { scale_value };

        // Create scale tensor
        let scale_tensor = Tensor::full(Shape::from_dims(&[1]), scale, tensor.device().clone())?;

        // Quantize: round(tensor / scale) and clamp to U8 range (0-255)
        // We'll add 128 to shift from [-128, 127] to [0, 255]
        let scaled = tensor_f32.div(&scale_tensor)?;
        let shifted = scaled.add_scalar(128.0)?;
        let quantized = shifted.round()?.clamp(0.0, 255.0)?.to_dtype(DType::U8)?;

        Ok((quantized, scale_tensor))
    }

    /// Dequantize a tensor back to FP16/BF16
    pub fn dequantize(&self, name: &str, dtype: DType) -> flame_core::Result<Tensor> {
        let quantized = self.quantized_weights.get(name).ok_or_else(|| {
            flame_core::Error::InvalidOperation(format!("Weight {} not found", name))
        })?;

        // Check if this tensor was quantized
        if let Some(scale) = self.scale_factors.get(name) {
            // Dequantize: (tensor.saturating_sub(128)) * scale
            let dequantized = quantized.to_dtype(DType::F32)?;
            let shifted = dequantized.sub_scalar(128.0)?;
            let result = shifted.mul(scale)?;

            // Convert to target dtype
            result.to_dtype(dtype).map_err(|e| flame_core::Error::from(e))
        } else {
            // Not quantized, return as-is
            quantized.to_dtype(dtype).map_err(|e| flame_core::Error::from(e))
        }
    }

    /// Get a dequantized weight by name
    pub fn get_weight(&self, name: &str, dtype: DType) -> flame_core::Result<Tensor> {
        self.dequantize(name, dtype)
    }

    /// Check if a tensor should be quantized based on its name and size
    fn should_quantize(&self, name: &str) -> bool {
        // Quantize linear/conv weights but not biases, norms, embeddings
        let quantizable_patterns = [
            ".weight",
            "to_q.weight",
            "to_k.weight",
            "to_v.weight",
            "to_out.weight",
            "ff.net.0.weight",
            "ff.net.2.weight",
            "proj_in.weight",
            "proj_out.weight",
        ];

        let skip_patterns = [
            "bias",
            "norm",
            "ln",
            "embedding",
            "pe", // positional encoding
        ];

        // Check if it matches quantizable patterns
        let should_quantize = quantizable_patterns.iter().any(|pattern| name.contains(pattern));

        // But skip if it matches skip patterns
        let should_skip = skip_patterns.iter().any(|pattern| name.contains(pattern));

        should_quantize && !should_skip
    }

    /// Get all weight names
    pub fn weight_names(&self) -> Vec<String> {
        self.quantized_weights.keys().cloned().collect()
    }

    pub fn dequantize_all(&self, dtype: DType) -> flame_core::Result<HashMap<String, Tensor>> {
        let mut out = HashMap::with_capacity(self.quantized_weights.len());
        for key in self.quantized_weights.keys() {
            let weight = self.get_weight(key, dtype)?;
            out.insert(key.clone(), weight);
        }
        Ok(out)
    }

    /// Get memory usage statistics
    pub fn memory_stats(&self) -> flame_core::Result<(usize, usize)> {
        let mut original_size = 0;
        let mut quantized_size = 0;

        for (name, tensor) in &self.quantized_weights {
            let element_count = tensor.shape().dims().iter().product::<usize>();

            if self.scale_factors.contains_key(name) {
                // This was quantized
                quantized_size += element_count; // INT8 = 1 byte per element
                quantized_size += std::mem::size_of::<f32>(); // Scale factor

                // Original would be FP16/BF16 = 2 bytes per element
                original_size += element_count * 2;
            } else {
                // Not quantized, count actual size
                let dtype_size = match tensor.dtype() {
                    DType::F32 => 4,
                    DType::F16 | DType::BF16 => 2,
                    DType::U8 => 1,
                    _ => 4, // Default to F32 size
                };
                let size = element_count * dtype_size;
                original_size += size;
                quantized_size += size;
            }
        }

        Ok((original_size, quantized_size))
    }

    /// Create a quantized Flux model wrapper that dequantizes on-the-fly
    pub fn create_model_wrapper(self) -> FluxInt8Model {
        FluxInt8Model {
            loader: self,
            dtype: DType::BF16, // Default to BF16 for Flux
        }
    }
}

/// Wrapper for INT8 quantized Flux model that dequantizes weights on-the-fly
pub struct FluxInt8Model {
    loader: FluxInt8Loader,
    dtype: DType,
}

impl FluxInt8Model {
    /// Set the dtype for dequantization
    pub fn with_dtype(mut self, dtype: DType) -> Self {
        self.dtype = dtype;
        self
    }

    /// Get a dequantized weight by name
    pub fn get_weight(&self, name: &str) -> flame_core::Result<Tensor> {
        self.loader.get_weight(name, self.dtype)
    }

    /// Get multiple weights as a HashMap
    pub fn get_weights(&self, names: &[&str]) -> flame_core::Result<HashMap<String, Tensor>> {
        let mut weights = HashMap::new();
        for name in names {
            weights.insert(name.to_string(), self.get_weight(name)?);
        }
        Ok(weights)
    }

    /// Get all weight names
    pub fn weight_names(&self) -> Vec<String> {
        self.loader.quantized_weights.keys().cloned().collect()
    }

    /// Check if a weight exists
    pub fn has_weight(&self, name: &str) -> bool {
        self.loader.quantized_weights.contains_key(name)
    }

    /// Get memory statistics
    pub fn memory_stats(&self, device: &Device) -> flame_core::Result<(usize, usize)> {
        self.loader.memory_stats()
    }

    pub fn dequantize_all(&self) -> flame_core::Result<HashMap<String, Tensor>> {
        self.loader.dequantize_all(self.dtype)
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }
}

/// Helper function to load and quantize a Flux model
pub fn load_flux_int8<P: AsRef<Path>>(
    model_path: P,
    device: Device,
) -> flame_core::Result<FluxInt8Model> {
    let mut loader = FluxInt8Loader::new(device.clone())?;
    loader.load_and_quantize(model_path)?;

    // Print memory statistics
    let wrapper = loader.create_model_wrapper();
    let (original, quantized) = wrapper.memory_stats(&device)?;
    println!("Flux INT8 Quantization:");
    println!("  Original size: {:.2} GB", original as f64 / 1e9);
    println!("  Quantized size: {:.2} GB", quantized as f64 / 1e9);
    println!("  Compression ratio: {:.2}x", original as f64 / quantized as f64);

    Ok(wrapper)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization() {
        // Add tests here
    }
}
