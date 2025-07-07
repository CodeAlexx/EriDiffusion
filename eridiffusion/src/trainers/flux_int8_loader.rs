use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use std::collections::HashMap;
use std::path::Path;

/// INT8 quantization for Flux models to reduce memory usage from ~22GB to ~11GB
/// Similar to bitsandbytes but in pure Rust
pub struct FluxInt8Loader {
    /// Quantized weights stored as INT8
    quantized_weights: HashMap<String, Tensor>,
    /// Scale factors for each quantized tensor
    scale_factors: HashMap<String, Tensor>,
    /// Device to load tensors on (for final output)
    device: Device,
    /// Target device for final tensors
    target_device: Device,
}

impl FluxInt8Loader {
    /// Create a new INT8 loader
    pub fn new(device: Device) -> Self {
        Self {
            quantized_weights: HashMap::new(),
            scale_factors: HashMap::new(),
            device: Device::Cpu, // Always quantize on CPU first
            target_device: device, // Store the target device
        }
    }

    /// Load and quantize a Flux model from safetensors files
    pub fn load_and_quantize<P: AsRef<Path>>(
        &mut self,
        model_path: P,
    ) -> Result<()> {
        let model_path = model_path.as_ref();
        
        println!("Loading model to CPU for quantization...");
        // Load the safetensors file to CPU first to avoid OOM
        let tensors = candle_core::safetensors::load(model_path, &Device::Cpu)
            .with_context(|| format!("Failed to load model from {:?}", model_path))?;
        
        println!("Loaded {} tensors, starting quantization...", tensors.len());

        // Process tensors in batches to manage memory
        let mut processed = 0;
        for (name, tensor) in tensors {
            // Skip non-weight tensors (biases, norms, etc.)
            if self.should_quantize(&name) {
                let (quantized, scale) = self.quantize_tensor(&tensor)?;
                // Keep on CPU for now, will move to GPU later
                self.quantized_weights.insert(name.clone(), quantized);
                self.scale_factors.insert(name, scale);
            } else {
                // Keep small tensors on CPU too
                self.quantized_weights.insert(name, tensor);
            }
            
            processed += 1;
            if processed % 50 == 0 {
                println!("Quantized {}/{} tensors", processed, 780); // Flux has ~780 tensors
            }
        }

        Ok(())
    }

    /// Quantize a single tensor to INT8 with scale factor
    fn quantize_tensor(&self, tensor: &Tensor) -> Result<(Tensor, Tensor)> {
        // Convert to f32 for quantization if needed
        let tensor_f32 = if tensor.dtype() == DType::F32 {
            tensor.clone()
        } else {
            tensor.to_dtype(DType::F32)?
        };

        // Get absolute max value for scaling
        let abs_max = tensor_f32.abs()?.max_all()?;
        
        // Compute scale factor (127 is max INT8 value)
        let scale = (abs_max / 127.0)?;
        
        // Avoid division by zero by using broadcast_maximum
        let min_scale = Tensor::new(1e-10f32, tensor.device())?;
        let scale = scale.broadcast_maximum(&min_scale)?;
        
        // Quantize: round(tensor / scale) and clamp to U8 range (0-255)
        // We'll add 128 to shift from [-128, 127] to [0, 255]
        let scaled = tensor_f32.broadcast_div(&scale)?;
        let shifted = (scaled + 128.0)?;
        let quantized = shifted
            .round()?
            .clamp(0.0, 255.0)?
            .to_dtype(DType::U8)?;
        
        Ok((quantized, scale))
    }

    /// Dequantize a tensor back to FP16/BF16
    pub fn dequantize(&self, name: &str, dtype: DType) -> Result<Tensor> {
        let quantized = self.quantized_weights.get(name)
            .with_context(|| format!("Weight {} not found", name))?;
        
        // Check if this tensor was quantized
        if let Some(scale) = self.scale_factors.get(name) {
            // Dequantize: (tensor - 128) * scale
            let dequantized = quantized.to_dtype(DType::F32)?;
            let shifted = (dequantized - 128.0)?;
            let result = shifted.broadcast_mul(scale)?;
            
            // Convert to target dtype
            result.to_dtype(dtype).map_err(Into::into)
        } else {
            // Not quantized, return as-is
            quantized.to_dtype(dtype).map_err(Into::into)
        }
    }

    /// Get a dequantized weight by name
    pub fn get_weight(&self, name: &str, dtype: DType) -> Result<Tensor> {
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
            "pe",  // positional encoding
        ];
        
        // Check if it matches quantizable patterns
        let should_quantize = quantizable_patterns.iter()
            .any(|pattern| name.contains(pattern));
        
        // But skip if it matches skip patterns
        let should_skip = skip_patterns.iter()
            .any(|pattern| name.contains(pattern));
        
        should_quantize && !should_skip
    }

    /// Get memory usage statistics
    pub fn memory_stats(&self) -> Result<(usize, usize)> {
        let mut original_size = 0;
        let mut quantized_size = 0;
        
        for (name, tensor) in &self.quantized_weights {
            let element_count = tensor.dims().iter().product::<usize>();
            
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
    pub fn get_weight(&self, name: &str) -> Result<Tensor> {
        self.loader.get_weight(name, self.dtype)
    }

    /// Get multiple weights as a HashMap
    pub fn get_weights(&self, names: &[&str]) -> Result<HashMap<String, Tensor>> {
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
    pub fn memory_stats(&self) -> Result<(usize, usize)> {
        self.loader.memory_stats()
    }
}

/// Helper function to load and quantize a Flux model
pub fn load_flux_int8<P: AsRef<Path>>(
    model_path: P,
    device: Device,
) -> Result<FluxInt8Model> {
    let mut loader = FluxInt8Loader::new(device);
    loader.load_and_quantize(model_path)?;
    
    // Print memory statistics
    let (original, quantized) = loader.memory_stats()?;
    println!("Flux INT8 Quantization:");
    println!("  Original size: {:.2} GB", original as f64 / 1e9);
    println!("  Quantized size: {:.2} GB", quantized as f64 / 1e9);
    println!("  Compression ratio: {:.2}x", original as f64 / quantized as f64);
    
    Ok(loader.create_model_wrapper())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization() -> Result<()> {
        let device = Device::Cpu;
        let loader = FluxInt8Loader::new(device.clone());
        
        // Create a test tensor
        let original = Tensor::randn(0.0f32, 1.0f32, &[256, 256], &device)?;
        
        // Quantize it
        let (quantized, scale) = loader.quantize_tensor(&original)?;
        
        // Check dimensions
        assert_eq!(quantized.dims(), original.dims());
        assert_eq!(quantized.dtype(), DType::U8);
        
        // Dequantize
        let quantized_f32 = quantized.to_dtype(DType::F32)?;
        let shifted = (quantized_f32 - 128.0)?;
        let dequantized = shifted.broadcast_mul(&scale)?;
        
        // Check that values are close (some loss is expected)
        let diff = (original - dequantized)?.abs()?.mean_all()?;
        let diff_val = diff.to_scalar::<f32>()?;
        
        // Should be small but not zero due to quantization
        assert!(diff_val < 0.1);
        assert!(diff_val > 0.0);
        
        Ok(())
    }

    #[test]
    fn test_should_quantize() {
        let device = Device::Cpu;
        let loader = FluxInt8Loader::new(device);
        
        // Should quantize
        assert!(loader.should_quantize("model.to_q.weight"));
        assert!(loader.should_quantize("transformer.ff.net.0.weight"));
        
        // Should not quantize
        assert!(!loader.should_quantize("model.norm.weight"));
        assert!(!loader.should_quantize("model.to_q.bias"));
        assert!(!loader.should_quantize("text_embedding.weight"));
    }
}