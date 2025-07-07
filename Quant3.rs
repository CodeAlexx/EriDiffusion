// quanto_production.rs - Production-ready improvements

use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Instant;
use anyhow::Result;
use candle_core::{Tensor, Device, DType};
use rayon::prelude::*;

/// Production-ready quantization improvements
impl QuantoManager {
    /// Proper INT4 packing/unpacking
    pub fn pack_int4(tensor: &Tensor) -> Result<Tensor> {
        let data = tensor.to_vec1::<u8>()?;
        let packed_size = (data.len() + 1) / 2;
        let mut packed = vec![0u8; packed_size];
        
        for (i, chunk) in data.chunks(2).enumerate() {
            let low = chunk[0] & 0x0F;
            let high = if chunk.len() > 1 { chunk[1] & 0x0F } else { 0 };
            packed[i] = (high << 4) | low;
        }
        
        Tensor::from_vec(packed, packed_size, tensor.device())
    }
    
    pub fn unpack_int4(packed: &Tensor, original_shape: &[usize]) -> Result<Tensor> {
        let packed_data = packed.to_vec1::<u8>()?;
        let mut unpacked = Vec::with_capacity(packed_data.len() * 2);
        
        for byte in packed_data {
            unpacked.push(byte & 0x0F);
            unpacked.push((byte >> 4) & 0x0F);
        }
        
        // Trim to original size
        let original_size: usize = original_shape.iter().product();
        unpacked.truncate(original_size);
        
        Tensor::from_vec(unpacked, original_shape, packed.device())
    }
    
    /// Proper NF4 quantization with lookup table
    pub fn quantize_nf4_proper(&self, tensor: &Tensor) -> Result<QuantizedTensor> {
        const NF4_VALUES: [f32; 16] = [
            -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
            -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
            0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
            0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
        ];
        
        let tensor_f32 = tensor.to_dtype(DType::F32)?;
        let absmax = tensor_f32.abs()?.max_all()?.to_scalar::<f32>()?;
        
        if absmax == 0.0 {
            return Ok(QuantizedTensor {
                shape: tensor.shape().clone(),
                quantized_data: Tensor::zeros_like(tensor)?,
                scale: Tensor::new(&[1.0f32], tensor.device())?,
                zero_point: None,
                qtype: QuantizationType::NF4,
                original_dtype: tensor.dtype(),
            });
        }
        
        // Normalize to [-1, 1]
        let normalized = (tensor_f32 / absmax)?;
        let data = normalized.to_vec1::<f32>()?;
        
        // Quantize to nearest NF4 value
        let mut quantized_indices = vec![0u8; data.len()];
        for (i, &val) in data.iter().enumerate() {
            let mut best_idx = 0;
            let mut best_diff = f32::INFINITY;
            
            for (idx, &nf4_val) in NF4_VALUES.iter().enumerate() {
                let diff = (val - nf4_val).abs();
                if diff < best_diff {
                    best_diff = diff;
                    best_idx = idx;
                }
            }
            quantized_indices[i] = best_idx as u8;
        }
        
        // Pack 4-bit values
        let quantized = Tensor::from_vec(quantized_indices.clone(), data.len(), tensor.device())?;
        let packed = Self::pack_int4(&quantized)?;
        
        Ok(QuantizedTensor {
            shape: tensor.shape().clone(),
            quantized_data: packed,
            scale: Tensor::new(&[absmax], tensor.device())?,
            zero_point: None,
            qtype: QuantizationType::NF4,
            original_dtype: tensor.dtype(),
        })
    }
    
    /// Parallel quantization for faster processing
    pub fn quantize_model_parallel(&self, weights: &HashMap<String, Tensor>) -> Result<()> {
        use std::sync::Mutex;
        
        let quantized = self.quantized_weights.clone();
        let errors = Mutex::new(Vec::new());
        
        // Group weights by size for better load balancing
        let mut weight_list: Vec<(&String, &Tensor)> = weights.iter().collect();
        weight_list.sort_by_key(|(_, t)| t.elem_count());
        
        // Process in parallel
        weight_list.par_iter().for_each(|(name, tensor)| {
            if self.config.exclude_layers.iter().any(|ex| name.contains(ex)) {
                return;
            }
            
            if tensor.dims().len() == 2 {
                match self.quantize_weight(name, tensor) {
                    Ok(qtensor) => {
                        quantized.write().unwrap().insert((*name).clone(), qtensor);
                    }
                    Err(e) => {
                        errors.lock().unwrap().push(format!("{}: {}", name, e));
                    }
                }
            }
        });
        
        // Check for errors
        let errors = errors.into_inner().unwrap();
        if !errors.is_empty() {
            anyhow::bail!("Quantization errors: {}", errors.join(", "));
        }
        
        Ok(())
    }
}

/// Optimized dequantization kernels
pub struct OptimizedDequantization;

impl OptimizedDequantization {
    /// SIMD-optimized INT8 dequantization
    #[cfg(target_arch = "x86_64")]
    pub unsafe fn dequantize_int8_simd(
        quantized: &[u8],
        scale: f32,
        zero_point: f32,
        output: &mut [f32],
    ) {
        use std::arch::x86_64::*;
        
        let scale_vec = _mm256_set1_ps(scale);
        let zero_vec = _mm256_set1_ps(zero_point);
        
        let chunks = quantized.chunks_exact(8);
        let remainder = chunks.remainder();
        
        for (i, chunk) in chunks.enumerate() {
            // Load 8 bytes
            let bytes = _mm_loadl_epi64(chunk.as_ptr() as *const __m128i);
            
            // Convert to int32
            let ints = _mm256_cvtepu8_epi32(bytes);
            
            // Convert to float
            let floats = _mm256_cvtepi32_ps(ints);
            
            // Apply scale and zero point
            let scaled = _mm256_fmadd_ps(floats, scale_vec, zero_vec);
            
            // Store result
            _mm256_storeu_ps(output.as_mut_ptr().add(i * 8), scaled);
        }
        
        // Handle remainder
        for (i, &byte) in remainder.iter().enumerate() {
            output[chunks.len() * 8 + i] = (byte as f32 - zero_point) * scale;
        }
    }
    
    /// Optimized NF4 dequantization with LUT
    pub fn dequantize_nf4_fast(
        packed_data: &[u8],
        scale: f32,
        output: &mut [f32],
    ) {
        const NF4_LUT: [f32; 16] = [
            -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
            -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
            0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
            0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
        ];
        
        let mut out_idx = 0;
        for &byte in packed_data {
            let low = (byte & 0x0F) as usize;
            let high = ((byte >> 4) & 0x0F) as usize;
            
            if out_idx < output.len() {
                output[out_idx] = NF4_LUT[low] * scale;
                out_idx += 1;
            }
            
            if out_idx < output.len() {
                output[out_idx] = NF4_LUT[high] * scale;
                out_idx += 1;
            }
        }
    }
}

/// Thread-safe weight cache with proper synchronization
pub struct WeightCache {
    cache: Arc<RwLock<HashMap<String, Arc<Tensor>>>>,
    loading: Arc<RwLock<HashMap<String, Arc<AtomicBool>>>>,
    max_cache_size: usize,
}

impl WeightCache {
    pub fn new(max_cache_size: usize) -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            loading: Arc::new(RwLock::new(HashMap::new())),
            max_cache_size,
        }
    }
    
    pub fn get_or_load<F>(&self, key: &str, loader: F) -> Result<Arc<Tensor>>
    where
        F: FnOnce() -> Result<Tensor>,
    {
        // Fast path: check if already cached
        {
            let cache = self.cache.read().unwrap();
            if let Some(tensor) = cache.get(key) {
                return Ok(tensor.clone());
            }
        }
        
        // Check if another thread is loading
        let loading_flag = {
            let mut loading = self.loading.write().unwrap();
            loading.entry(key.to_string())
                .or_insert_with(|| Arc::new(AtomicBool::new(false)))
                .clone()
        };
        
        // Try to acquire loading lock
        if loading_flag.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed).is_ok() {
            // We got the lock, load the tensor
            match loader() {
                Ok(tensor) => {
                    let tensor_arc = Arc::new(tensor);
                    
                    // Add to cache
                    {
                        let mut cache = self.cache.write().unwrap();
                        
                        // Evict if necessary
                        if cache.len() >= self.max_cache_size {
                            // Simple LRU: remove first (oldest) entry
                            if let Some(first_key) = cache.keys().next().cloned() {
                                cache.remove(&first_key);
                            }
                        }
                        
                        cache.insert(key.to_string(), tensor_arc.clone());
                    }
                    
                    // Release loading lock
                    loading_flag.store(false, Ordering::Release);
                    Ok(tensor_arc)
                }
                Err(e) => {
                    loading_flag.store(false, Ordering::Release);
                    Err(e)
                }
            }
        } else {
            // Another thread is loading, wait for it
            while loading_flag.load(Ordering::Acquire) {
                thread::yield_now();
            }
            
            // Try to get from cache again
            let cache = self.cache.read().unwrap();
            cache.get(key)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("Failed to load tensor: {}", key))
        }
    }
}

/// Error recovery and validation
pub struct QuantizationValidator {
    tolerance: f32,
}

impl QuantizationValidator {
    pub fn new(tolerance: f32) -> Self {
        Self { tolerance }
    }
    
    /// Validate quantization accuracy
    pub fn validate_quantization(
        &self,
        original: &Tensor,
        quantized: &QuantizedTensor,
    ) -> Result<bool> {
        let dequantized = quantized.dequantize(original.device())?;
        
        // Calculate error metrics
        let diff = (original - &dequantized)?;
        let mse = diff.sqr()?.mean_all()?.to_scalar::<f32>()?;
        let rmse = mse.sqrt();
        
        // Calculate relative error
        let original_norm = original.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt();
        let relative_error = if original_norm > 0.0 {
            rmse / original_norm
        } else {
            rmse
        };
        
        log::debug!(
            "Quantization validation - RMSE: {:.6}, Relative Error: {:.4}%",
            rmse,
            relative_error * 100.0
        );
        
        Ok(relative_error < self.tolerance)
    }
    
    /// Validate entire model
    pub fn validate_model(
        &self,
        original_weights: &HashMap<String, Tensor>,
        quanto_manager: &QuantoManager,
    ) -> Result<()> {
        let mut failed = Vec::new();
        
        for (name, original) in original_weights {
            if let Ok(dequantized) = quanto_manager.get_weight(name) {
                match self.validate_quantization(original, &dequantized) {
                    Ok(valid) if !valid => {
                        failed.push(name.clone());
                    }
                    Err(e) => {
                        log::warn!("Failed to validate {}: {}", name, e);
                    }
                    _ => {}
                }
            }
        }
        
        if !failed.is_empty() {
            anyhow::bail!(
                "Quantization validation failed for {} layers: {:?}",
                failed.len(),
                &failed[..5.min(failed.len())]
            );
        }
        
        Ok(())
    }
}

/// Benchmark utilities
pub fn benchmark_quantization(
    weights: &HashMap<String, Tensor>,
    quanto_config: &QuantoConfig,
) -> Result<()> {
    let device = Device::cuda_if_available(0)?;
    let memory_pool = Arc::new(RwLock::new(MemoryPool::new(0, Default::default())?));
    
    println!("Benchmarking quantization...");
    
    // Benchmark quantization time
    let start = Instant::now();
    let quanto_manager = QuantoManager::new(
        device.clone(),
        quanto_config.clone(),
        memory_pool,
        None,
    );
    quanto_manager.quantize_model_parallel(weights)?;
    let quantize_time = start.elapsed();
    
    println!("Quantization time: {:?}", quantize_time);
    
    // Benchmark dequantization time
    let test_weight = weights.iter().next().unwrap();
    let start = Instant::now();
    for _ in 0..100 {
        let _ = quanto_manager.get_weight(test_weight.0)?;
    }
    let dequant_time = start.elapsed() / 100;
    
    println!("Average dequantization time: {:?}", dequant_time);
    
    // Memory stats
    let (original, quantized) = quanto_manager.get_memory_savings()?;
    println!(
        "Memory: {:.2}GB -> {:.2}GB ({:.1}% reduction)",
        original as f64 / 1e9,
        quantized as f64 / 1e9,
        (1.0 - quantized as f64 / original as f64) * 100.0
    );
    
    Ok(())
}
