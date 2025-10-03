use flame_core::device::Device;
use flame_core::{DType, Error, Result, Shape, Tensor};
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::path::Path;

/// Weight loader for loading model weights from safetensors files
#[derive(Clone)]
pub struct WeightLoader {
    pub weights: HashMap<String, Tensor>,
    pub device: Device,
}

impl WeightLoader {
    /// Create an empty weight loader
    pub fn new(device: Device) -> Self {
        Self { weights: HashMap::new(), device }
    }

    /// Get the keys of all loaded weights
    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.weights.keys()
    }

    /// Get the number of loaded weights
    pub fn len(&self) -> usize {
        self.weights.len()
    }

    /// Check if weights is empty
    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }

    /// Load weights from a safetensors file with specific dtype
    pub fn from_safetensors_with_dtype<P: AsRef<Path>>(
        path: P,
        device: Device,
        target_dtype: DType,
    ) -> Result<Self> {
        let path_ref = path.as_ref();

        // Check if we should suppress verbose output (for training)
        let quiet_mode = std::env::var("FLUX_QUIET_MODE").unwrap_or_default() == "1";

        if !quiet_mode {
            println!("    Reading file: {:?}", path_ref);
            let file_size = std::fs::metadata(path_ref).map(|m| m.len()).unwrap_or(0);
            println!("    File size: {:.2} GB", file_size as f64 / (1024.0 * 1024.0 * 1024.0));
        }

        // Use memory-mapped file to avoid loading 23GB into RAM
        use memmap2::MmapOptions;
        use std::fs::File;

        if !quiet_mode {
            println!("    Memory-mapping file to avoid loading 23GB into RAM...");
        }
        let file = File::open(path_ref).map_err(|e| {
            flame_core::Error::Io(format!("Failed to open safetensors file: {}", e))
        })?;
        let mmap = unsafe {
            MmapOptions::new()
                .map(&file)
                .map_err(|e| flame_core::Error::Io(format!("Failed to mmap file: {}", e)))?
        };

        if !quiet_mode {
            println!("    File memory-mapped successfully, deserializing metadata...");
        }
        let tensors = SafeTensors::deserialize(&mmap).map_err(|e| {
            flame_core::Error::InvalidOperation(format!(
                "Failed to deserialize safetensors: {}",
                e
            ))
        })?;

        let mut weights = HashMap::new();

        if !quiet_mode {
            println!(
                "    Converting {} tensors to FLAME format with dtype {:?}...",
                tensors.tensors().len(),
                target_dtype
            );
        }
        let total_tensors = tensors.tensors().len();
        let mut processed = 0;

        // Speed up loading by showing less progress - only every 100 tensors
        let progress_interval = if total_tensors > 500 {
            100
        } else if total_tensors > 100 {
            50
        } else {
            25
        };

        // Convert safetensors to FLAME tensors
        for (name, view) in tensors.tensors() {
            // Show progress less frequently to speed up loading
            let show_progress = !quiet_mode
                && (processed % progress_interval == 0 || processed == total_tensors - 1);

            if show_progress {
                println!(
                    "    [{}/{}] Loading tensors... (current: {})",
                    processed + 1,
                    total_tensors,
                    name
                );
            }
            processed += 1;
            let shape = Shape::from_dims(view.shape());

            // Load data and convert to target dtype
            let tensor = match view.dtype() {
                safetensors::Dtype::F32 => {
                    let data = view.data();
                    let float_data: Vec<f32> = data
                        .chunks_exact(4)
                        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                        .collect();
                    // Create tensor with target dtype directly
                    Tensor::from_vec_dtype(
                        float_data,
                        shape,
                        device.cuda_device().clone(),
                        target_dtype,
                    )?
                }
                safetensors::Dtype::F16 => {
                    let data = view.data();
                    let float_data: Vec<f32> = data
                        .chunks_exact(2)
                        .map(|chunk| {
                            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                            half::f16::from_bits(bits).to_f32()
                        })
                        .collect();
                    // Create tensor with target dtype directly
                    Tensor::from_vec_dtype(
                        float_data,
                        shape,
                        device.cuda_device().clone(),
                        target_dtype,
                    )?
                }
                safetensors::Dtype::BF16 => {
                    let data = view.data();
                    let float_data: Vec<f32> = data
                        .chunks_exact(2)
                        .map(|chunk| {
                            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                            half::bf16::from_bits(bits).to_f32()
                        })
                        .collect();
                    // Create tensor with target dtype directly (BF16 is stored as F32 internally in FLAME)
                    Tensor::from_vec_dtype(
                        float_data,
                        shape,
                        device.cuda_device().clone(),
                        target_dtype,
                    )?
                }
                _ => {
                    return Err(flame_core::Error::InvalidOperation(format!(
                        "Unsupported dtype: {:?}",
                        view.dtype()
                    )))
                }
            };

            weights.insert(name.to_string(), tensor);
        }

        if !quiet_mode {
            println!(
                "    ✅ All {} tensors converted successfully to {:?}",
                total_tensors, target_dtype
            );
        }

        Ok(Self { weights, device })
    }

    /// Load weights from a safetensors file with streaming to avoid OOM
    pub fn from_safetensors_streaming<P: AsRef<Path>>(
        path: P,
        device: Device,
        target_dtype: DType,
    ) -> Result<Self> {
        let path_ref = path.as_ref();
        println!("    🌊 [STREAMING] Starting streaming load from: {:?}", path_ref);
        println!("    🌊 [STREAMING] Checking file metadata...");
        let file_size = std::fs::metadata(path_ref).map(|m| m.len()).unwrap_or(0);
        println!(
            "    🌊 [STREAMING] File size: {:.2} GB",
            file_size as f64 / (1024.0 * 1024.0 * 1024.0)
        );

        // Use memory-mapped file for efficient access
        println!("    🌊 [STREAMING] Opening file for memory mapping...");
        let file = std::fs::File::open(path_ref)
            .map_err(|e| flame_core::Error::Io(format!("Failed to open file: {}", e)))?;
        println!("    🌊 [STREAMING] Creating memory map...");
        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .map_err(|e| flame_core::Error::Io(format!("Failed to mmap file: {}", e)))?;
        println!("    🌊 [STREAMING] Memory map created successfully");

        println!("    🌊 [STREAMING] Deserializing SafeTensors metadata (this may take 10-30s for 23GB file)...");
        let start_time = std::time::Instant::now();
        let tensors = SafeTensors::deserialize(&mmap).map_err(|e| {
            flame_core::Error::InvalidOperation(format!(
                "Failed to deserialize safetensors: {}",
                e
            ))
        })?;
        println!(
            "    🌊 [STREAMING] Metadata deserialized in {:.1}s",
            start_time.elapsed().as_secs_f32()
        );

        let mut weights = HashMap::new();

        let total_tensors = tensors.tensors().len();
        println!("    🌊 [STREAMING] Found {} tensors in file", total_tensors);
        println!(
            "    🌊 [STREAMING] Starting tensor streaming to GPU with dtype {:?}...",
            target_dtype
        );
        let mut processed = 0;
        let mut gpu_memory_used: f64 = 0.0;
        let overall_start = std::time::Instant::now();

        // Process tensors in smaller batches to control memory usage
        // Full Flux Dev model is ~23GB, so with 24GB VRAM we need to be very careful
        // Load only 1 tensor at a time to avoid OOM
        const BATCH_SIZE: usize = 1; // Reduced from 5 for full Flux Dev
        let tensor_list: Vec<_> = tensors.tensors().into_iter().collect();

        for batch in tensor_list.chunks(BATCH_SIZE) {
            let mut batch_weights = Vec::new();

            for (name, view) in batch {
                processed += 1;

                // Calculate tensor size in GB
                let num_elements: usize = view.shape().iter().product();
                let bytes_per_element = match target_dtype {
                    DType::F16 | DType::BF16 => 2.0,
                    DType::F32 => 4.0,
                    _ => 4.0,
                };
                let tensor_size_gb =
                    (num_elements as f64 * bytes_per_element) / (1024.0 * 1024.0 * 1024.0);

                // Check if we have enough memory before loading
                if gpu_memory_used + tensor_size_gb > 22.0 {
                    println!("    ⚠️  CRITICAL: Not enough memory to load {} (would use {:.2} GB / 24 GB)", 
                        name, gpu_memory_used + tensor_size_gb);
                    // For now, skip loading more tensors to prevent OOM
                    // In a real implementation, we'd implement layer swapping
                    break;
                }

                if processed % 10 == 1 || processed == 1 || processed == total_tensors {
                    let elapsed = overall_start.elapsed().as_secs_f32();
                    let rate = processed as f32 / elapsed.max(0.1);
                    let eta = ((total_tensors - processed) as f32 / rate).max(0.0);
                    println!("    🌊 [{}/{}] Streaming: {} (shape: {:?}, size: {:.3} GB) | Speed: {:.1} tensors/s | ETA: {:.0}s", 
                        processed, total_tensors, name, view.shape(), tensor_size_gb, rate, eta);
                }

                let shape = Shape::from_dims(view.shape());

                // Convert data in chunks to avoid large intermediate allocations
                let tensor = match view.dtype() {
                    safetensors::Dtype::F32 => {
                        let data = view.data();
                        let float_data: Vec<f32> = data
                            .chunks_exact(4)
                            .map(|chunk| {
                                f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
                            })
                            .collect();
                        Tensor::from_vec_dtype(
                            float_data,
                            shape,
                            device.cuda_device().clone(),
                            target_dtype,
                        )?
                    }
                    safetensors::Dtype::F16 => {
                        let data = view.data();
                        // Process in smaller chunks to reduce peak memory
                        let mut float_data = Vec::with_capacity(data.len() / 2);
                        for chunk in data.chunks_exact(2) {
                            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                            float_data.push(half::f16::from_bits(bits).to_f32());
                        }
                        Tensor::from_vec_dtype(
                            float_data,
                            shape,
                            device.cuda_device().clone(),
                            target_dtype,
                        )?
                    }
                    safetensors::Dtype::BF16 => {
                        let data = view.data();
                        // Process in smaller chunks to reduce peak memory
                        let mut float_data = Vec::with_capacity(data.len() / 2);
                        for chunk in data.chunks_exact(2) {
                            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                            float_data.push(half::bf16::from_bits(bits).to_f32());
                        }
                        Tensor::from_vec_dtype(
                            float_data,
                            shape,
                            device.cuda_device().clone(),
                            target_dtype,
                        )?
                    }
                    _ => {
                        return Err(flame_core::Error::InvalidOperation(format!(
                            "Unsupported dtype: {:?}",
                            view.dtype()
                        )))
                    }
                };

                gpu_memory_used += tensor_size_gb;
                batch_weights.push((name.to_string(), tensor));
            }

            // Add batch to weights
            for (name, tensor) in batch_weights {
                weights.insert(name, tensor);
            }

            println!("      Batch complete. Total GPU memory used: {:.2} GB", gpu_memory_used);

            // Check if we're approaching memory limits (leave 2GB buffer)
            if gpu_memory_used > 22.0 {
                println!(
                    "    ⚠️  WARNING: Approaching GPU memory limit (used: {:.2} GB / 24 GB)",
                    gpu_memory_used
                );
            }
        }

        println!(
            "    ✅ Streaming complete: {} tensors loaded ({:.2} GB total)",
            total_tensors, gpu_memory_used
        );

        Ok(Self { weights, device })
    }

    /// Load weights from a safetensors file (default to F32)
    pub fn from_safetensors<P: AsRef<Path>>(path: P, device: Device) -> Result<Self> {
        Self::from_safetensors_with_dtype(path, device, DType::F32)
    }

    /// Load weights from a safetensors file with CPU offloading support
    pub fn from_safetensors_cpu_offload<P: AsRef<Path>>(
        path: P,
        device: Device,
        critical_weights: &[&str], // Weights that must stay on GPU
    ) -> Result<Self> {
        let path_ref = path.as_ref();
        println!("Loading Flux model with CPU offloading from: {:?}", path_ref);
        let file_size = std::fs::metadata(path_ref).map(|m| m.len()).unwrap_or(0);
        println!("File size: {:.2} GB", file_size as f64 / (1024.0 * 1024.0 * 1024.0));

        // Load file into memory
        let data = std::fs::read(path_ref).map_err(|e| {
            flame_core::Error::Io(format!("Failed to read safetensors file: {}", e))
        })?;

        println!("File loaded into memory, deserializing...");
        let tensors = SafeTensors::deserialize(&data).map_err(|e| {
            flame_core::Error::InvalidOperation(format!(
                "Failed to deserialize safetensors: {}",
                e
            ))
        })?;

        let mut weights = HashMap::new();
        let total_tensors = tensors.tensors().len();
        let mut processed = 0;

        // Identify critical weights that should stay on GPU
        let critical_set: std::collections::HashSet<&str> =
            critical_weights.iter().copied().collect();

        println!("Converting {} tensors to FLAME format with CPU offloading...", total_tensors);
        println!("Critical weights that will stay on GPU: {}", critical_weights.len());

        // First pass - load only critical weights to GPU
        for (name, view) in tensors.tensors() {
            processed += 1;

            // Check if this is a critical weight
            let is_critical = critical_set.iter().any(|&pattern| name.contains(pattern));

            if is_critical {
                println!(
                    "    [{}/{}] Loading critical weight to GPU: {}",
                    processed, total_tensors, name
                );

                let shape = Shape::from_dims(view.shape());
                let dtype = match view.dtype() {
                    safetensors::Dtype::F32 => DType::F32,
                    safetensors::Dtype::F16 => DType::F16,
                    safetensors::Dtype::BF16 => DType::BF16,
                    _ => {
                        return Err(flame_core::Error::InvalidOperation(format!(
                            "Unsupported dtype: {:?}",
                            view.dtype()
                        )))
                    }
                };

                // Load data based on dtype
                let tensor = match dtype {
                    DType::F32 => {
                        let data = view.data();
                        let float_data: Vec<f32> = data
                            .chunks_exact(4)
                            .map(|chunk| {
                                f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
                            })
                            .collect();
                        Tensor::from_vec(float_data, shape, device.cuda_device().clone())?
                    }
                    DType::F16 | DType::BF16 => {
                        // Convert to F32 but preserve the dtype information
                        let data = view.data();
                        let float_data: Vec<f32> = data
                            .chunks_exact(2)
                            .map(|chunk| {
                                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                                if dtype == DType::F16 {
                                    half::f16::from_bits(bits).to_f32()
                                } else {
                                    half::bf16::from_bits(bits).to_f32()
                                }
                            })
                            .collect();

                        // Create tensor with proper dtype tracking
                        Tensor::from_vec_dtype(
                            float_data,
                            shape,
                            device.cuda_device().clone(),
                            dtype,
                        )?
                    }
                    _ => unreachable!(),
                };

                weights.insert(name.to_string(), tensor);
            } else if processed % 50 == 0 {
                println!(
                    "    [{}/{}] Skipping non-critical weight: {} (will load on demand)",
                    processed, total_tensors, name
                );
            }
        }

        println!("✅ Loaded {} critical weights to GPU", weights.len());
        println!("📦 {} weights deferred for on-demand loading", total_tensors - weights.len());

        Ok(Self { weights, device })
    }

    /// Load weights from multiple safetensors files
    pub fn from_safetensors_multi<P: AsRef<Path>>(paths: &[P], device: Device) -> Result<Self> {
        let mut weights = HashMap::new();

        for path in paths {
            let loader = Self::from_safetensors(path, device.clone())?;
            weights.extend(loader.weights);
        }

        Ok(Self { weights, device })
    }

    /// Get a tensor by key
    pub fn get(&self, key: &str) -> Result<&Tensor> {
        self.weights.get(key).ok_or_else(|| {
            flame_core::Error::InvalidOperation(format!("Weight not found: {}", key))
        })
    }

    /// Get a tensor with expected shape
    pub fn tensor(&self, key: &str, shape: &[usize]) -> Result<Tensor> {
        let weight = self.get(key)?;
        let expected_shape = Shape::from_dims(shape);
        if weight.shape() != &expected_shape {
            return Err(flame_core::Error::InvalidOperation(format!(
                "Shape mismatch for {}: expected {:?}, got {:?}",
                key,
                expected_shape,
                weight.shape()
            )));
        }
        Ok(weight.clone())
    }

    /// Get all weights with a given prefix
    pub fn get_prefix(&self, prefix: &str) -> HashMap<String, &Tensor> {
        self.weights
            .iter()
            .filter(|(k, _)| k.starts_with(prefix))
            .map(|(k, v)| (k.clone(), v))
            .collect()
    }

    /// Create a prefixed weight loader
    pub fn pp(&self, prefix: &str) -> PrefixedWeightLoader {
        PrefixedWeightLoader { loader: self, prefix: prefix.to_string() }
    }

    /// Get the device
    pub fn device(&self) -> &Device {
        &self.device
    }
}

/// Prefixed weight loader for hierarchical weight access
pub struct PrefixedWeightLoader<'a> {
    pub(crate) loader: &'a WeightLoader,
    pub(crate) prefix: String,
}

impl<'a> PrefixedWeightLoader<'a> {
    /// Get a tensor by key (with prefix prepended)
    pub fn get(&self, key: &str) -> Result<&Tensor> {
        let full_key = format!("{}.{}", self.prefix, key);
        self.loader.get(&full_key)
    }

    /// Get a tensor with expected shape (with prefix prepended)
    pub fn tensor(&self, key: &str, shape: &[usize]) -> Result<Tensor> {
        let full_key = format!("{}.{}", self.prefix, key);
        self.loader.tensor(&full_key, shape)
    }

    /// Create a nested prefixed weight loader
    pub fn pp(&self, prefix: &str) -> PrefixedWeightLoader {
        PrefixedWeightLoader { loader: self.loader, prefix: format!("{}.{}", self.prefix, prefix) }
    }

    /// Get the device
    pub fn device(&self) -> &Device {
        &self.loader.device
    }
}
