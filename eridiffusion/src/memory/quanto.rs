use super::{BlockSwapManager, MemoryPool, QuantizationMode};
use anyhow::Context;
use flame_core::device::Device;
use flame_core::optimizers::{Adam, SGD};
use flame_core::{CudaDevice, DType, Error, Result, Shape, Tensor};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

pub struct QuantoManager {
    device: Device,
    config: QuantoConfig,
    memory_pool: Arc<RwLock<MemoryPool>>,
    block_swap_manager: Option<Arc<BlockSwapManager>>,
    quantized_weights: RwLock<HashMap<String, QuantizedTensor>>,
    original_weights: RwLock<HashMap<String, Tensor>>,
}

// Quanto-style quantization for Flux models
// Based on the quantization files in the parent directory

// FLAME uses flame_core::device::Device instead of Device

/// Quantized tensor representation
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
    pub per_layer_config: std::collections::HashMap<String, QuantizationMode>,
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
        let quantizable = name.contains(".weight")
            && (name.contains("linear")
                || name.contains("to_q")
                || name.contains("to_k")
                || name.contains("to_v")
                || name.contains("to_out")
                || name.contains("proj")
                || name.contains("ff.net")
                || name.contains("mlp"));

        quantizable & !name.contains("bias")
    }

    /// Get quantization type for a specific layer
    fn get_qtype(&self, name: &str) -> QuantizationMode {
        self.config.per_layer_config.get(name).copied().unwrap_or(self.config.default_qtype)
    }

    /// Quantize a single tensor
    pub fn quantize_tensor(
        &self,
        tensor: &Tensor,
        qtype: QuantizationMode,
        device: &CudaDevice,
    ) -> flame_core::Result<QuantizedTensor> {
        match qtype {
            QuantizationMode::INT8 => self.quantize_int8(tensor),
            QuantizationMode::INT4 => self.quantize_int4(tensor),
            QuantizationMode::NF4 => self.quantize_nf4(tensor),
            QuantizationMode::None => Ok(QuantizedTensor {
                shape: tensor.shape().clone(),
                quantized_data: tensor.clone(),
                scale: Tensor::ones(Shape::from_dims(&[1]), tensor.device().clone())?
                    .to_dtype(tensor.dtype())?, // Keep on CPU
                zero_point: None,
                qtype: QuantizationMode::None,
                original_dtype: tensor.dtype(),
            }),
            _ => {
                return Err(flame_core::Error::InvalidOperation(format!(
                    "Unsupported tensor dtype: {:?}",
                    tensor.dtype()
                )))
            }
        }
    }

    /// INT8 quantization - optimized version with absmax
    fn quantize_int8(&self, tensor: &Tensor) -> flame_core::Result<QuantizedTensor> {
        // Skip tiny tensors - not worth quantizing
        let elem_count = tensor.shape().dims().iter().product::<usize>();
        if elem_count < 1024 {
            return Ok(QuantizedTensor {
                shape: tensor.shape().clone(),
                quantized_data: tensor.clone(),
                scale: Tensor::ones(Shape::from_dims(&[1]), tensor.device().clone())?
                    .to_dtype(tensor.dtype())?,
                zero_point: None,
                qtype: QuantizationMode::None,
                original_dtype: tensor.dtype(),
            });
        }

        // For large tensors, use a simpler approach
        // This is inspired by bitsandbytes' approach but simplified;
        let device = Device::from(tensor.device().clone());

        // Move to CPU first if needed
        let tensor_cpu = if device.is_cpu() { tensor.clone() } else { tensor.clone() };

        // Convert to F32 for quantization calculations
        let tensor_f32 = tensor_cpu.to_dtype(DType::F32)?;

        // Simple absmax quantization - much faster
        // We'll use raw data access for efficiency;
        let abs_max = tensor_f32.abs()?.max_all()?;

        // Handle outliers by clamping scale
        let scale = (abs_max / 127.0).max(1e-6);

        // Direct quantization without intermediate tensors
        let scale_recip = 127.0 / abs_max.max(1e-6);

        // Create scale tensors
        let scale_tensor =
            Tensor::from_vec(vec![scale], Shape::from_dims(&[1]), device.cuda_device().clone())?;
        let scale_recip_tensor = Tensor::from_vec(
            vec![scale_recip],
            Shape::from_dims(&[1]),
            device.cuda_device().clone(),
        )?;

        // Quantize directly to U8
        // This is the most expensive operation, so we optimize it
        let quantized_i8 = tensor_f32
            .mul(&scale_recip_tensor.broadcast_to(tensor_f32.shape())?)?
            .round()?
            .clamp(-128.0, 127.0)?;

        // Shift to U8 range by adding 128
        let offset =
            Tensor::from_vec(vec![128.0f32], Shape::from_dims(&[1]), device.cuda_device().clone())?;
        let quantized_u8 =
            quantized_i8.add(&offset.broadcast_to(quantized_i8.shape())?)?.to_dtype(DType::U8)?;

        Ok(QuantizedTensor {
            shape: tensor.shape().clone(),
            quantized_data: quantized_u8,
            scale: scale_tensor,
            zero_point: None,
            qtype: QuantizationMode::INT8,
            original_dtype: tensor.dtype(),
        })
    }

    /// INT4 quantization
    fn quantize_int4(&self, tensor: &Tensor) -> flame_core::Result<QuantizedTensor> {
        // Skip tiny tensors
        let elem_count = tensor.shape().dims().iter().product::<usize>();
        if elem_count < 1024 {
            return Ok(QuantizedTensor {
                shape: tensor.shape().clone(),
                quantized_data: tensor.clone(),
                scale: Tensor::ones(Shape::from_dims(&[1]), tensor.device().clone())?
                    .to_dtype(tensor.dtype())?,
                zero_point: None,
                qtype: QuantizationMode::None,
                original_dtype: tensor.dtype(),
            });
        }

        let device = Device::from(tensor.device().clone());
        let tensor_f32 = tensor.to_dtype(DType::F32)?;

        // INT4 uses 4 bits: -8 to 7 range
        let abs_max = tensor_f32.abs()?.max_all()?;
        let scale = (abs_max / 7.0).max(1e-6);
        let scale_recip = 7.0 / abs_max.max(1e-6);

        // Create scale tensors
        let scale_tensor =
            Tensor::from_vec(vec![scale], Shape::from_dims(&[1]), device.cuda_device().clone())?;
        let scale_recip_tensor = Tensor::from_vec(
            vec![scale_recip],
            Shape::from_dims(&[1]),
            device.cuda_device().clone(),
        )?;

        // Quantize to INT4 range
        let quantized_i4 = tensor_f32
            .mul(&scale_recip_tensor.broadcast_to(tensor_f32.shape())?)?
            .round()?
            .clamp(-8.0, 7.0)?;

        // Pack 2 INT4 values into each U8 byte
        let shape = quantized_i4.shape().dims();
        let total_elements = shape.iter().product::<usize>();
        let packed_size = (total_elements + 1) / 2; // Round up for odd counts

        // Get the quantized values as a vector
        let quantized_values = quantized_i4.to_vec()?;

        // Pack values: high nibble = first value + 8, low nibble = second value + 8
        let mut packed_data = Vec::with_capacity(packed_size);
        for i in (0..total_elements).step_by(2) {
            let val1 = (quantized_values[i] as i8 + 8) as u8;
            let val2 = if i + 1 < total_elements {
                (quantized_values[i + 1] as i8 + 8) as u8
            } else {
                8 // Neutral value for padding
            };
            let packed = (val1 << 4) | (val2 & 0x0F);
            packed_data.push(packed);
        }

        // Create packed tensor with adjusted shape - convert u8 to f32
        let packed_shape = Shape::from_dims(&[packed_size]);
        let packed_data_f32: Vec<f32> = packed_data.into_iter().map(|x| x as f32).collect();
        let packed_tensor =
            Tensor::from_vec(packed_data_f32, packed_shape, device.cuda_device().clone())?;

        Ok(QuantizedTensor {
            shape: tensor.shape().clone(),
            quantized_data: packed_tensor,
            scale: scale_tensor,
            zero_point: None,
            qtype: QuantizationMode::INT4,
            original_dtype: tensor.dtype(),
        })
    }

    /// NF4 quantization (Normal Float 4-bit)
    fn quantize_nf4(&self, tensor: &Tensor) -> flame_core::Result<QuantizedTensor> {
        // Skip tiny tensors
        let elem_count = tensor.shape().dims().iter().product::<usize>();
        if elem_count < 1024 {
            return Ok(QuantizedTensor {
                shape: tensor.shape().clone(),
                quantized_data: tensor.clone(),
                scale: Tensor::ones(Shape::from_dims(&[1]), tensor.device().clone())?
                    .to_dtype(tensor.dtype())?,
                zero_point: None,
                qtype: QuantizationMode::None,
                original_dtype: tensor.dtype(),
            });
        }

        // NF4 lookup table (16 values optimized for normal distribution)
        const NF4_QUANT_TABLE: [f32; 16] = [
            -1.0,
            -0.6961928009986877,
            -0.5250730514526367,
            -0.39491748809814453,
            -0.28444138169288635,
            -0.18477343022823334,
            -0.09105003625154495,
            0.0,
            0.07958029955625534,
            0.16093020141124725,
            0.24611230194568634,
            0.33791524171829224,
            0.44070982933044434,
            0.5626170039176941,
            0.7229568362236023,
            1.0,
        ];

        let device = Device::from(tensor.device().clone());
        let tensor_f32 = tensor.to_dtype(DType::F32)?;

        // Compute scale based on absolute max
        let abs_max = tensor_f32.abs()?.max_all()?;
        let scale = abs_max.max(1e-6);
        let scale_recip = 1.0 / scale;

        // Create scale tensors
        let scale_tensor =
            Tensor::from_vec(vec![scale], Shape::from_dims(&[1]), device.cuda_device().clone())?;
        let scale_recip_tensor = Tensor::from_vec(
            vec![scale_recip],
            Shape::from_dims(&[1]),
            device.cuda_device().clone(),
        )?;

        // Normalize tensor to [-1, 1] range
        let normalized = tensor_f32.mul(&scale_recip_tensor.broadcast_to(tensor_f32.shape())?)?;
        let normalized_values = normalized.to_vec()?;

        // Quantize each value to nearest NF4 table entry
        let mut quantized_indices = Vec::with_capacity(elem_count);
        for &val in &normalized_values {
            // Find nearest value in NF4 table
            let mut best_idx = 0;
            let mut best_dist = (val - NF4_QUANT_TABLE[0]).abs();

            for (idx, &table_val) in NF4_QUANT_TABLE.iter().enumerate().skip(1) {
                let dist = (val - table_val).abs();
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = idx;
                }
            }

            quantized_indices.push(best_idx as u8);
        }

        // Pack 2 NF4 indices into each U8 byte
        let packed_size = (elem_count + 1) / 2;
        let mut packed_data = Vec::with_capacity(packed_size);

        for i in (0..elem_count).step_by(2) {
            let idx1 = quantized_indices[i];
            let idx2 = if i + 1 < elem_count {
                quantized_indices[i + 1]
            } else {
                7 // Neutral value (0.0) for padding
            };
            let packed = (idx1 << 4) | (idx2 & 0x0F);
            packed_data.push(packed);
        }

        // Create packed tensor - convert u8 to f32
        let packed_shape = Shape::from_dims(&[packed_size]);
        let packed_data_f32: Vec<f32> = packed_data.into_iter().map(|x| x as f32).collect();
        let packed_tensor =
            Tensor::from_vec(packed_data_f32, packed_shape, device.cuda_device().clone())?;

        // Store the NF4 table as a constant tensor for dequantization
        let nf4_table_tensor = Tensor::from_vec(
            NF4_QUANT_TABLE.to_vec(),
            Shape::from_dims(&[16]),
            device.cuda_device().clone(),
        )?
        .to_dtype(DType::F32)?;

        Ok(QuantizedTensor {
            shape: tensor.shape().clone(),
            quantized_data: packed_tensor,
            scale: scale_tensor,
            zero_point: Some(nf4_table_tensor), // Store lookup table in zero_point field
            qtype: QuantizationMode::NF4,
            original_dtype: tensor.dtype(),
        })
    }

    /// Dequantize tensor
    pub fn dequantize(
        &self,
        qtensor: &QuantizedTensor,
        device: &CudaDevice,
    ) -> flame_core::Result<Tensor> {
        match qtensor.qtype {
            QuantizationMode::INT8 => {
                let device = qtensor.quantized_data.device();

                // Convert from U8 back to centered values
                let dequantized = qtensor.quantized_data.to_dtype(DType::F32)?;
                let offset =
                    Tensor::from_vec(vec![128.0f32], Shape::from_dims(&[1]), device.clone())?;
                let centered = dequantized.sub(&offset.broadcast_to(dequantized.shape())?)?;

                // Apply scale
                let scale = &qtensor.scale;
                let scaled = centered.mul(&scale.broadcast_to(centered.shape())?)?;

                Ok(scaled.to_dtype(qtensor.original_dtype)?)
            }
            QuantizationMode::INT4 => {
                // Unpack INT4 values from packed U8 bytes
                let packed_data = qtensor.quantized_data.to_vec()?;
                let total_elements = qtensor.shape.dims().iter().product::<usize>();
                let mut unpacked_values = Vec::with_capacity(total_elements);

                for (i, &packed) in packed_data.iter().enumerate() {
                    // Cast f32 to u8 for bit operations
                    let packed_byte = packed as u8;

                    // Extract high nibble (first value)
                    let val1 = ((packed_byte >> 4) as i8 - 8) as f32;
                    unpacked_values.push(val1);

                    // Extract low nibble (second value) if within bounds
                    if unpacked_values.len() < total_elements {
                        let val2 = ((packed_byte & 0x0F) as i8 - 8) as f32;
                        unpacked_values.push(val2);
                    }
                }

                // Truncate to exact size
                unpacked_values.truncate(total_elements);

                // Create tensor from unpacked values
                let unpacked = Tensor::from_vec(
                    unpacked_values,
                    qtensor.shape.clone(),
                    qtensor.quantized_data.device().clone(),
                )?
                .to_dtype(DType::F32)?;

                // Apply scale
                let scale = &qtensor.scale;
                let scaled = unpacked.mul(&scale.broadcast_to(unpacked.shape())?)?;

                Ok(scaled.to_dtype(qtensor.original_dtype)?)
            }
            QuantizationMode::NF4 => {
                // Get NF4 lookup table from zero_point field
                let nf4_table = qtensor.zero_point.as_ref().ok_or_else(|| {
                    flame_core::Error::InvalidOperation("NF4 table missing".into())
                })?;
                let nf4_values = nf4_table.to_vec()?;

                // Unpack NF4 indices from packed U8 bytes
                let packed_data = qtensor.quantized_data.to_vec()?;
                let total_elements = qtensor.shape.dims().iter().product::<usize>();
                let mut dequantized_values = Vec::with_capacity(total_elements);

                for &packed in &packed_data {
                    // Cast f32 to u8 for bit operations
                    let packed_byte = packed as u8;

                    // Extract high nibble (first index)
                    let idx1 = (packed_byte >> 4) as usize;
                    if idx1 < 16 {
                        dequantized_values.push(nf4_values[idx1]);
                    }

                    // Extract low nibble (second index) if within bounds
                    if dequantized_values.len() < total_elements {
                        let idx2 = (packed_byte & 0x0F) as usize;
                        if idx2 < 16 {
                            dequantized_values.push(nf4_values[idx2]);
                        }
                    }
                }

                // Truncate to exact size
                dequantized_values.truncate(total_elements);

                // Create tensor from dequantized values
                let dequantized = Tensor::from_vec(
                    dequantized_values,
                    qtensor.shape.clone(),
                    qtensor.quantized_data.device().clone(),
                )?
                .to_dtype(DType::F32)?;

                // Apply scale
                let scale = &qtensor.scale;
                let scaled = dequantized.mul(&scale.broadcast_to(dequantized.shape())?)?;

                Ok(scaled.to_dtype(qtensor.original_dtype)?)
            }
            QuantizationMode::None => Ok(qtensor.quantized_data.clone()),
            _ => Err(flame_core::Error::InvalidOperation(format!(
                "Unsupported quantization mode: {:?}",
                qtensor.qtype
            ))),
        }
    }

    /// Quantize an entire model
    pub fn quantize_model(&self, weights: &HashMap<String, Tensor>) -> flame_core::Result<()> {
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
                        println!(
                            "Quantizing weight {}/{}: {} ({:?})",
                            quantized_count,
                            quantizable_count,
                            name,
                            tensor.shape()
                        );
                    }

                    let qtensor = self.quantize_tensor(tensor, qtype, self.device.cuda_device())?;
                    quantized.insert(name.clone(), qtensor);

                    // Store original for comparison
                    original.insert(name.clone(), tensor.clone());
                } else {
                    // Keep as-is but wrap
                    quantized.insert(
                        name.clone(),
                        QuantizedTensor {
                            shape: tensor.shape().clone(),
                            quantized_data: tensor.clone(),
                            scale: Tensor::ones(Shape::from_dims(&[1]), tensor.device().clone())?
                                .to_dtype(tensor.dtype())?,
                            zero_point: None,
                            qtype: QuantizationMode::None,
                            original_dtype: tensor.dtype(),
                        },
                    );
                }

                if processed % 100 == 0 {
                    println!("Overall progress: {}/{} tensors", processed, total);
                }
            }
        }

        println!("\nQuantization complete!");
        println!("Quantized {} weights, kept {} as-is", quantized_count, total - quantized_count);
        Ok(())
    }

    /// Get quantized weight
    pub fn get_weight(&self, name: &str) -> flame_core::Result<Tensor> {
        self.get_weight_with_dtype(name, None)
    }

    /// Get weight with specific dtype
    pub fn get_weight_with_dtype(
        &self,
        name: &str,
        target_dtype: Option<DType>,
    ) -> flame_core::Result<Tensor> {
        let quantized = self.quantized_weights.read().unwrap();

        if let Some(qtensor) = quantized.get(name) {
            let tensor = if qtensor.qtype == QuantizationMode::None {
                qtensor.quantized_data.clone()
            } else {
                self.dequantize(qtensor, self.device.cuda_device())?
            };

            // Convert to target dtype if specified
            if let Some(dtype) = target_dtype {
                if tensor.dtype() != dtype {
                    return tensor.to_dtype(dtype).map_err(|e| e.into());
                }
            }
            Ok(tensor)
        } else {
            return Err(flame_core::Error::InvalidOperation(format!(
                "Weight not found: {}",
                name
            )));
        }
    }

    /// Get memory savings
    pub fn get_memory_savings(&self) -> flame_core::Result<(usize, usize)> {
        let quantized = self.quantized_weights.read().unwrap();
        let mut original_size = 0;
        let mut quantized_size = 0;

        for (name, qtensor) in quantized.iter() {
            let elem_count = qtensor.shape.dims().iter().product::<usize>();
            original_size += elem_count * qtensor.original_dtype.size_in_bytes();

            match qtensor.qtype {
                QuantizationMode::INT8 => quantized_size += elem_count,
                QuantizationMode::INT4 | QuantizationMode::NF4 => quantized_size += elem_count / 2,
                QuantizationMode::None => {
                    quantized_size += elem_count * qtensor.original_dtype.size_in_bytes()
                }
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
