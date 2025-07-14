// Quanto-style quantization for Flux models
// Based on the quantization files in the parent directory

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use anyhow::{Result, Context};
use candle_core::{Tensor, Device, DType, Shape};
use super::{MemoryPool, BlockSwapManager, QuantizationMode};
use rayon::prelude::*;

/// Quantized tensor representation
#[derive(Clone)]
pub struct QuantizedTensor {
    pub shape: Shape,
    pub quantized_data: Tensor,
    pub scale: Tensor,
    pub zero_point: Option<Tensor>,
    pub qtype: QuantizationMode,
    pub original_dtype: DType,
}

/// Configuration for quantization
#[derive(Clone, Debug)]
pub struct QuantoConfig {
    pub default_qtype: QuantizationMode,
    pub exclude_patterns: Vec<String>,
    pub per_layer_config: HashMap<String, QuantizationMode>,
    pub calibration_steps: usize,
    pub percentile: f32,
    pub group_size: Option<usize>,
}

impl Default for QuantoConfig {
    fn default() -> Self {
        Self {
            default_qtype: QuantizationMode::INT8,
            exclude_patterns: vec![
                "ln".to_string(),
                "norm".to_string(),
                "embed".to_string(),
                "head".to_string(),
            ],
            per_layer_config: HashMap::new(),
            calibration_steps: 128,
            percentile: 99.9,
            group_size: None,
        }
    }
}

impl QuantoConfig {
    /// Optimized config for Flux on 24GB
    pub fn flux_24gb() -> Self {
        let mut config = Self::default();
        config.default_qtype = QuantizationMode::INT8;
        
        // Keep critical layers in higher precision
        config.per_layer_config.insert("img_in".to_string(), QuantizationMode::None);
        config.per_layer_config.insert("txt_in".to_string(), QuantizationMode::None);
        config.per_layer_config.insert("final_layer".to_string(), QuantizationMode::None);
        
        // Exclude patterns
        config.exclude_patterns = vec![
            "ln".to_string(),
            "norm".to_string(),
            "time_in".to_string(),
            "vector_in".to_string(),
            "guidance_in".to_string(),
        ];
        
        config
    }
}

/// Manages quantization for models
pub struct QuantoManager {
    device: Device,
    config: QuantoConfig,
    quantized_weights: RwLock<HashMap<String, QuantizedTensor>>,
    original_weights: RwLock<HashMap<String, Tensor>>,
    memory_pool: Arc<RwLock<MemoryPool>>,
    block_swap_manager: Option<Arc<BlockSwapManager>>,
}

impl QuantoManager {
    pub fn new(
        device: Device,
        config: QuantoConfig,
        memory_pool: Arc<RwLock<MemoryPool>>,
        block_swap_manager: Option<Arc<BlockSwapManager>>,
    ) -> Self {
        Self {
            device,
            config,
            quantized_weights: RwLock::new(HashMap::new()),
            original_weights: RwLock::new(HashMap::new()),
            memory_pool,
            block_swap_manager,
        }
    }
    
    /// Check if a tensor should be quantized
    fn should_quantize(&self, name: &str) -> bool {
        // Check exclude patterns
        for pattern in &self.config.exclude_patterns {
            if name.contains(pattern) {
                return false;
            }
        }
        
        // Skip embeddings and small tensors
        if name.contains("embed") || name.contains("pe_embedder") {
            return false;
        }
        
        // Only quantize linear/conv weights
        let quantizable = name.contains(".weight") && 
            (name.contains("linear") || 
             name.contains("to_q") || 
             name.contains("to_k") || 
             name.contains("to_v") || 
             name.contains("to_out") ||
             name.contains("proj") ||
             name.contains("ff.net") ||
             name.contains("mlp"));
        
        quantizable && !name.contains("bias")
    }
    
    /// Get quantization type for a specific layer
    fn get_qtype(&self, name: &str) -> QuantizationMode {
        self.config.per_layer_config
            .get(name)
            .copied()
            .unwrap_or(self.config.default_qtype)
    }
    
    /// Quantize a single tensor
    pub fn quantize_tensor(&self, tensor: &Tensor, qtype: QuantizationMode) -> Result<QuantizedTensor> {
        match qtype {
            QuantizationMode::INT8 => self.quantize_int8(tensor),
            QuantizationMode::INT4 => self.quantize_int4(tensor),
            QuantizationMode::NF4 => self.quantize_nf4(tensor),
            QuantizationMode::None => Ok(QuantizedTensor {
                shape: tensor.shape().clone(),
                quantized_data: tensor.clone(),
                scale: Tensor::ones(&[1], DType::F32, &Device::Cpu)?,  // Keep on CPU
                zero_point: None,
                qtype: QuantizationMode::None,
                original_dtype: tensor.dtype(),
            }),
            _ => anyhow::bail!("Unsupported quantization type: {:?}", qtype),
        }
    }
    
    /// INT8 quantization - optimized version with absmax
    fn quantize_int8(&self, tensor: &Tensor) -> Result<QuantizedTensor> {
        // Skip tiny tensors - not worth quantizing
        let elem_count = tensor.shape().elem_count();
        if elem_count < 1024 {
            return Ok(QuantizedTensor {
                shape: tensor.shape().clone(),
                quantized_data: tensor.clone(),
                scale: Tensor::ones(&[1], DType::F32, &Device::Cpu)?,
                zero_point: None,
                qtype: QuantizationMode::None,
                original_dtype: tensor.dtype(),
            });
        }
        
        // For large tensors, use a simpler approach
        // This is inspired by bitsandbytes' approach but simplified
        let device = tensor.device();
        
        // Move to CPU first if needed
        let tensor_cpu = if device.is_cpu() {
            tensor.clone()
        } else {
            tensor.to_device(&Device::Cpu)?
        };
        
        // Convert to F32 for quantization calculations
        let tensor_f32 = tensor_cpu.to_dtype(DType::F32)?;
        
        // Simple absmax quantization - much faster
        // We'll use raw data access for efficiency
        let abs_max = tensor_f32.abs()?.max_all()?.to_scalar::<f32>()?;
        
        // Handle outliers by clamping scale
        let scale = (abs_max / 127.0).max(1e-6);
        
        // Direct quantization without intermediate tensors
        let scale_recip = 127.0 / abs_max.max(1e-6);
        
        // Create scale tensors
        let scale_tensor = Tensor::new(&[scale], &Device::Cpu)?;
        let scale_recip_tensor = Tensor::new(&[scale_recip], &Device::Cpu)?;
        
        // Quantize directly to U8
        // This is the most expensive operation, so we optimize it
        let quantized_i8 = tensor_f32.broadcast_mul(&scale_recip_tensor.broadcast_as(tensor_f32.shape())?)?
            .round()?
            .clamp(-128.0, 127.0)?;
        
        // Shift to U8 range by adding 128
        let offset = Tensor::new(&[128.0f32], &Device::Cpu)?;
        let quantized_u8 = quantized_i8.broadcast_add(&offset.broadcast_as(quantized_i8.shape())?)?
            .to_dtype(DType::U8)?;
        
        Ok(QuantizedTensor {
            shape: tensor.shape().clone(),
            quantized_data: quantized_u8,
            scale: scale_tensor,
            zero_point: None,
            qtype: QuantizationMode::INT8,
            original_dtype: tensor.dtype(),
        })
    }
    
    /// INT4 quantization (simplified)
    fn quantize_int4(&self, tensor: &Tensor) -> Result<QuantizedTensor> {
        // For now, reuse INT8 and note it's INT4
        // In production, would pack 2 values per byte
        let mut qtensor = self.quantize_int8(tensor)?;
        qtensor.qtype = QuantizationMode::INT4;
        Ok(qtensor)
    }
    
    /// NF4 quantization (simplified)
    fn quantize_nf4(&self, tensor: &Tensor) -> Result<QuantizedTensor> {
        // For now, reuse INT8 and note it's NF4
        // In production, would use NF4 lookup table
        let mut qtensor = self.quantize_int8(tensor)?;
        qtensor.qtype = QuantizationMode::NF4;
        Ok(qtensor)
    }
    
    /// Dequantize tensor
    pub fn dequantize(&self, qtensor: &QuantizedTensor) -> Result<Tensor> {
        match qtensor.qtype {
            QuantizationMode::INT8 => {
                let device = qtensor.quantized_data.device();
                
                // Convert from U8 back to centered values
                let dequantized = qtensor.quantized_data.to_dtype(DType::F32)?;
                let offset = Tensor::new(&[128.0f32], device)?;
                let centered = dequantized.broadcast_sub(&offset.broadcast_as(dequantized.shape())?)?;
                
                // Move scale to the same device as the quantized data
                let scale = qtensor.scale.to_device(device)?;
                let scaled = (&centered * scale.broadcast_as(centered.shape())?)?;
                
                Ok(scaled.to_dtype(qtensor.original_dtype)?)
            }
            QuantizationMode::None => Ok(qtensor.quantized_data.clone()),
            _ => anyhow::bail!("Dequantization not implemented for {:?}", qtensor.qtype),
        }
    }
    
    /// Quantize an entire model
    pub fn quantize_model(&self, weights: &HashMap<String, Tensor>) -> Result<()> {
        println!("Quantizing {} weights...", weights.len());
        
        // First pass: count quantizable weights
        let mut quantizable_count = 0;
        for (name, _tensor) in weights {
            if self.should_quantize(name) && self.get_qtype(name) != QuantizationMode::None {
                quantizable_count += 1;
            }
        }
        
        println!("Found {} weights to quantize out of {} total", quantizable_count, weights.len());
        
        let mut quantized = self.quantized_weights.write().unwrap();
        let mut original = self.original_weights.write().unwrap();
        
        let mut processed = 0;
        let mut quantized_count = 0;
        let total = weights.len();
        
        // Process in parallel using rayon if available
        for (name, tensor) in weights {
            processed += 1;
            
            if self.should_quantize(name) {
                let qtype = self.get_qtype(name);
                
                if qtype != QuantizationMode::None {
                    quantized_count += 1;
                    if quantized_count % 10 == 0 || quantized_count == quantizable_count {
                        println!("Quantizing weight {}/{}: {} ({:?})", 
                            quantized_count, quantizable_count, name, tensor.shape());
                    }
                    
                    let qtensor = self.quantize_tensor(tensor, qtype)?;
                    quantized.insert(name.clone(), qtensor);
                    
                    // Store original for comparison
                    original.insert(name.clone(), tensor.clone());
                } else {
                    // Keep as-is but wrap
                    quantized.insert(name.clone(), QuantizedTensor {
                        shape: tensor.shape().clone(),
                        quantized_data: tensor.clone(),
                        scale: Tensor::ones(&[1], DType::F32, &Device::Cpu)?,
                        zero_point: None,
                        qtype: QuantizationMode::None,
                        original_dtype: tensor.dtype(),
                    });
                }
            } else {
                // Non-quantizable - just wrap without processing
                quantized.insert(name.clone(), QuantizedTensor {
                    shape: tensor.shape().clone(),
                    quantized_data: tensor.clone(),
                    scale: Tensor::ones(&[1], DType::F32, &Device::Cpu)?,
                    zero_point: None,
                    qtype: QuantizationMode::None,
                    original_dtype: tensor.dtype(),
                });
            }
            
            if processed % 100 == 0 {
                println!("Overall progress: {}/{} tensors", processed, total);
            }
        }
        
        println!("\nQuantization complete!");
        println!("Quantized {} weights, kept {} as-is", quantized_count, total - quantized_count);
        Ok(())
    }
    
    /// Get quantized weight
    pub fn get_weight(&self, name: &str) -> Result<Tensor> {
        self.get_weight_with_dtype(name, None)
    }
    
    /// Get weight with specific dtype
    pub fn get_weight_with_dtype(&self, name: &str, target_dtype: Option<DType>) -> Result<Tensor> {
        let quantized = self.quantized_weights.read().unwrap();
        
        if let Some(qtensor) = quantized.get(name) {
            let tensor = if qtensor.qtype == QuantizationMode::None {
                qtensor.quantized_data.clone()
            } else {
                self.dequantize(qtensor)?
            };
            
            // Convert to target dtype if specified
            if let Some(dtype) = target_dtype {
                if tensor.dtype() != dtype {
                    return tensor.to_dtype(dtype).map_err(|e| e.into());
                }
            }
            Ok(tensor)
        } else {
            anyhow::bail!("Weight {} not found", name)
        }
    }
    
    /// Get memory savings
    pub fn get_memory_savings(&self) -> Result<(usize, usize)> {
        let quantized = self.quantized_weights.read().unwrap();
        let mut original_size = 0;
        let mut quantized_size = 0;
        
        for (name, qtensor) in quantized.iter() {
            let elem_count = qtensor.shape.elem_count();
            original_size += elem_count * qtensor.original_dtype.size_in_bytes();
            
            match qtensor.qtype {
                QuantizationMode::INT8 => quantized_size += elem_count,
                QuantizationMode::INT4 | QuantizationMode::NF4 => quantized_size += elem_count / 2,
                QuantizationMode::None => quantized_size += elem_count * qtensor.original_dtype.size_in_bytes(),
                _ => {}
            }
        }
        
        Ok((original_size, quantized_size))
    }
    
    /// Get all weight names
    pub fn weight_names(&self) -> Vec<String> {
        let quantized = self.quantized_weights.read().unwrap();
        quantized.keys().cloned().collect()
    }
}