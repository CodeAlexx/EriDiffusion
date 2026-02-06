use crate::models::mmdit_blocks::QkNormKind;
use bytemuck::try_cast_slice;
use flame_core::device::Device;
use flame_core::{DType, Error, Result, Shape, Tensor};
use safetensors::{tensor::TensorView, SafeTensors};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// Weight loader for loading model weights from safetensors files
#[derive(Clone)]
pub struct WeightLoader {
    pub weights: HashMap<String, Tensor>,
    pub device: Device,
    raw_mmap: Option<Arc<memmap2::Mmap>>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MMDiTMetadata {
    pub qk_norm: QkNormKind,
    pub x_self_attn_layers: Option<usize>,
    pub hidden_size: Option<usize>,
    pub num_heads: Option<usize>,
    pub depth: Option<usize>,
    pub mlp_ratio: Option<f32>,
}

impl Default for MMDiTMetadata {
    fn default() -> Self {
        Self {
            qk_norm: QkNormKind::Disabled,
            x_self_attn_layers: None,
            hidden_size: None,
            num_heads: None,
            depth: None,
            mlp_ratio: None,
        }
    }
}

impl WeightLoader {
    /// Create an empty weight loader
    pub fn new(device: Device) -> Self {
        Self { weights: HashMap::new(), device, raw_mmap: None }
    }

    /// Construct a loader from an in-memory tensor map (no safetensors backing).
    pub fn from_tensor_map(weights: HashMap<String, Tensor>, device: Device) -> Self {
        Self { weights, device, raw_mmap: None }
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

    /// Returns true when this loader retains a safetensors mmap handle.
    pub fn has_mmap(&self) -> bool {
        self.raw_mmap.is_some()
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
        let mmap = Arc::new(mmap);

        if !quiet_mode {
            println!("    File memory-mapped successfully, deserializing metadata...");
        }
        let tensors = SafeTensors::deserialize(&*mmap).map_err(|e| {
            flame_core::Error::InvalidOperation(format!("Failed to deserialize safetensors: {}", e))
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
            let tensor = tensor_from_view(&device, target_dtype, shape, &view)?;

            weights.insert(name.to_string(), tensor);
        }

        if !quiet_mode {
            println!(
                "    ✅ All {} tensors converted successfully to {:?}",
                total_tensors, target_dtype
            );
        }

        Ok(Self { weights, device, raw_mmap: Some(mmap) })
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
        let mmap = Arc::new(mmap);
        println!("    🌊 [STREAMING] Memory map created successfully");

        println!("    🌊 [STREAMING] Deserializing SafeTensors metadata (this may take 10-30s for 23GB file)...");
        let start_time = std::time::Instant::now();
        let tensors = SafeTensors::deserialize(&*mmap).map_err(|e| {
            flame_core::Error::InvalidOperation(format!("Failed to deserialize safetensors: {}", e))
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
                let tensor = tensor_from_view(&device, target_dtype, shape, view)?;

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

        Ok(Self { weights, device, raw_mmap: Some(mmap) })
    }

    /// Infer MMDiT architectural metadata (QK norm strategy and self-attention layout) from loaded tensor names.
    pub fn infer_mmdit_metadata(&self) -> MMDiTMetadata {
        infer_mmdit_metadata_from_iter(
            self.weights.iter().map(|(name, tensor)| (name.as_str(), Some(tensor.shape().dims()))),
        )
    }

    /// Infer metadata from an arbitrary iterator of tensor keys without requiring a full weight load.
    pub fn infer_mmdit_metadata_from_keys<'a, I>(keys: I) -> MMDiTMetadata
    where
        I: IntoIterator<Item = &'a str>,
    {
        infer_mmdit_metadata_from_iter(keys.into_iter().map(|name| (name, None)))
    }

    /// Execute `func` with a transient `SafeTensors` view backed by the original mmap.
    /// Returns an error if the loader was not constructed from a safetensors mmap.
    pub fn with_safetensors<F, T>(&self, func: F) -> Result<T>
    where
        F: FnOnce(&SafeTensors<'_>) -> Result<T>,
    {
        let mmap = self.raw_mmap.as_ref().ok_or_else(|| {
            Error::InvalidOperation(
                "WeightLoader does not retain a safetensors buffer (not mmap-backed)".into(),
            )
        })?;
        let tensors = SafeTensors::deserialize(&**mmap).map_err(|e| {
            Error::InvalidOperation(format!("Failed to deserialize safetensors: {e}"))
        })?;
        func(&tensors)
    }

    /// Infer metadata directly from a safetensors file without materializing weights.
    pub fn infer_mmdit_metadata_from_file<P: AsRef<Path>>(path: P) -> Result<MMDiTMetadata> {
        let path_ref = path.as_ref();
        let file = std::fs::File::open(path_ref)
            .map_err(|e| Error::Io(format!("Failed to open safetensors file: {}", e)))?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .map_err(|e| Error::Io(format!("Failed to mmap file: {}", e)))?;
        let tensors = SafeTensors::deserialize(&mmap).map_err(|e| {
            Error::InvalidOperation(format!("Failed to deserialize safetensors: {}", e))
        })?;
        Ok(infer_mmdit_metadata_from_iter(
            tensors.tensors().iter().map(|(name, view)| (name.as_str(), Some(view.shape()))),
        ))
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
            flame_core::Error::InvalidOperation(format!("Failed to deserialize safetensors: {}", e))
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

        Ok(Self { weights, device, raw_mmap: None })
    }

    /// Load weights from multiple safetensors files
    pub fn from_safetensors_multi<P: AsRef<Path>>(paths: &[P], device: Device) -> Result<Self> {
        let mut weights = HashMap::new();

        for path in paths {
            let loader = Self::from_safetensors(path, device.clone())?;
            weights.extend(loader.weights);
        }

        Ok(Self { weights, device, raw_mmap: None })
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
            eprintln!(
                "[weight_loader] shape mismatch for {} expected {:?} got {:?}",
                key,
                expected_shape.dims(),
                weight.shape().dims()
            );
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
    pub fn pp(&self, prefix: &str) -> PrefixedWeightLoader<'_> {
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
    pub fn pp(&self, prefix: &str) -> PrefixedWeightLoader<'a> {
        PrefixedWeightLoader { loader: self.loader, prefix: format!("{}.{}", self.prefix, prefix) }
    }

    /// Get the device
    pub fn device(&self) -> &Device {
        &self.loader.device
    }
}

fn tensor_from_view(
    device: &Device,
    target_dtype: DType,
    shape: Shape,
    view: &TensorView<'_>,
) -> Result<Tensor> {
    if target_dtype == DType::BF16 {
        return tensor_from_view_bf16(device, shape, view);
    }
    tensor_from_view_generic(device, target_dtype, shape, view)
}

fn tensor_from_view_generic(
    device: &Device,
    target_dtype: DType,
    shape: Shape,
    view: &TensorView<'_>,
) -> Result<Tensor> {
    match view.dtype() {
        safetensors::Dtype::F32 => {
            let data = view.data();
            let float_data: Vec<f32> = data
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();
            Tensor::from_vec_dtype(float_data, shape, device.cuda_device().clone(), target_dtype)
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
            Tensor::from_vec_dtype(float_data, shape, device.cuda_device().clone(), target_dtype)
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
            Tensor::from_vec_dtype(float_data, shape, device.cuda_device().clone(), target_dtype)
        }
        other => Err(Error::InvalidOperation(format!("Unsupported dtype: {:?}", other))),
    }
}

use rayon::prelude::*;

fn tensor_from_view_bf16(device: &Device, shape: Shape, view: &TensorView<'_>) -> Result<Tensor> {
    match view.dtype() {
        safetensors::Dtype::BF16 => {
            let bytes = try_cast_slice::<u8, u16>(view.data())
                .map_err(|_| Error::InvalidOperation("BF16 tensor data is not aligned".into()))?;
            Tensor::from_bf16_u16_slice(bytes, shape, device.cuda_device_arc())
        }
        safetensors::Dtype::F32 => {
            let src = try_cast_slice::<u8, f32>(view.data())
                .map_err(|_| Error::InvalidOperation("F32 tensor data is not aligned".into()))?;
            let len = shape.elem_count();
            if src.len() != len {
                return Err(Error::ShapeMismatch {
                    expected: shape.clone(),
                    got: Shape::from_dims(&[src.len()]),
                });
            }
            
            // Parallel conversion using rayon
            let bf16_data: Vec<u16> = src.par_iter()
                .map(|&value| half::bf16::from_f32(value).to_bits())
                .collect();
                
            Tensor::from_bf16_u16_slice(&bf16_data, shape, device.cuda_device_arc())
        }
        safetensors::Dtype::F16 => {
            let src = try_cast_slice::<u8, u16>(view.data())
                .map_err(|_| Error::InvalidOperation("F16 tensor data is not aligned".into()))?;
            let len = shape.elem_count();
            if src.len() != len {
                return Err(Error::ShapeMismatch {
                    expected: shape.clone(),
                    got: Shape::from_dims(&[src.len()]),
                });
            }

            // Parallel conversion using rayon
            let bf16_data: Vec<u16> = src.par_iter()
                .map(|&bits| {
                    let value = half::f16::from_bits(bits).to_f32();
                    half::bf16::from_f32(value).to_bits()
                })
                .collect();
                
            Tensor::from_bf16_u16_slice(&bf16_data, shape, device.cuda_device_arc())
        }
        other => {
            Err(Error::InvalidOperation(format!("Unsupported dtype {:?} for BF16 target", other)))
        }
    }
}

fn infer_mmdit_metadata_from_iter<'a, I>(entries: I) -> MMDiTMetadata
where
    I: IntoIterator<Item = (&'a str, Option<&'a [usize]>)>,
{
    let mut meta = MMDiTMetadata::default();
    let mut saw_ln_bias = false;
    let mut saw_ln_weight = false;
    let mut head_dim: Option<usize> = None;
    let mut fc1_out: Option<usize> = None;

    for (key, shape) in entries {
        if let Some(idx) = parse_joint_block_index(key) {
            meta.depth = Some(meta.depth.map_or(idx + 1, |prev| prev.max(idx + 1)));
            if key.contains("attn2") {
                meta.x_self_attn_layers =
                    Some(meta.x_self_attn_layers.map_or(idx, |prev| prev.max(idx)));
            }
        }

        if key.contains("joint_blocks.") && key.contains("attn.ln") {
            if key.contains(".bias") {
                saw_ln_bias = true;
            } else if key.contains(".weight") {
                saw_ln_weight = true;
            }
        }

        if let Some(shape) = shape {
            if meta.hidden_size.is_none()
                && (key.ends_with("attn.qkv.weight") || key.ends_with("attn2.qkv.weight"))
                && shape.len() == 2
            {
                meta.hidden_size = Some(shape[1]);
            }

            if head_dim.is_none()
                && (key.ends_with("attn.ln_q.weight") || key.ends_with("attn.ln_k.weight"))
                && shape.len() == 1
            {
                head_dim = Some(shape[0]);
            }

            if fc1_out.is_none() && key.contains("mlp.fc1.weight") && shape.len() == 2 {
                fc1_out = Some(shape[0]);
                if meta.hidden_size.is_none() {
                    meta.hidden_size = Some(shape[1]);
                }
            }
        }
    }

    meta.qk_norm = if saw_ln_bias {
        QkNormKind::Layer
    } else if saw_ln_weight {
        QkNormKind::Rms
    } else {
        QkNormKind::Disabled
    };

    if meta.num_heads.is_none() {
        if let (Some(hidden), Some(head_dim)) = (meta.hidden_size, head_dim) {
            if head_dim > 0 && hidden % head_dim == 0 {
                meta.num_heads = Some(hidden / head_dim);
            }
        }
    }

    if meta.num_heads.is_none() {
        if let Some(hidden) = meta.hidden_size {
            let inferred = hidden / 64;
            meta.num_heads = Some(std::cmp::max(1usize, inferred));
        }
    }

    if meta.mlp_ratio.is_none() {
        if let (Some(fc1_out), Some(hidden)) = (fc1_out, meta.hidden_size) {
            if hidden > 0 {
                meta.mlp_ratio = Some(fc1_out as f32 / hidden as f32);
            }
        }
    }

    meta
}

fn parse_joint_block_index(key: &str) -> Option<usize> {
    let start = key.find("joint_blocks.")?;
    let rest = &key[start + "joint_blocks.".len()..];
    let mut parts = rest.split('.');
    let idx_str = parts.next()?;
    idx_str.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::bf16;
    use safetensors::{serialize, tensor::TensorView, Dtype as SafeDtype};
    use std::collections::HashMap as StdHashMap;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    #[cfg(feature = "cuda")]
    fn bf16_loader_preserves_device_dtype() -> Result<()> {
        let device = match Device::cuda(0) {
            Ok(dev) => dev,
            Err(err) => {
                eprintln!("skipping bf16 loader test: {err}");
                return Ok(());
            }
        };

        std::env::set_var("STRICT_BF16", "1");

        let values = [0.25f32, -0.5, 1.0, 2.0];
        let bf16_bits: Vec<u16> = values.iter().map(|&v| bf16::from_f32(v).to_bits()).collect();
        let bf16_bytes = bytemuck::cast_slice(&bf16_bits);

        let mut tensors = StdHashMap::new();
        tensors.insert(
            "weight".to_string(),
            TensorView::new(SafeDtype::BF16, vec![2, 2], bf16_bytes)?,
        );

        let data = serialize(tensors, &None)?;
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let temp_path = std::env::temp_dir().join(format!(
            "bf16_loader_preserves_device_dtype_{}_{}.safetensors",
            std::process::id(),
            timestamp
        ));
        std::fs::write(&temp_path, data)?;

        let loader =
            WeightLoader::from_safetensors_with_dtype(&temp_path, device.clone(), DType::BF16)?;
        let tensor = loader.get("weight")?;
        assert_eq!(tensor.dtype(), DType::BF16);

        let _ = std::fs::remove_file(&temp_path);
        Ok(())
    }

    #[test]
    fn infer_metadata_detects_layer_norm() {
        let keys = vec![
            "model.diffusion_model.joint_blocks.0.attn.ln_q.bias",
            "model.diffusion_model.joint_blocks.0.attn.ln_q.weight",
            "model.diffusion_model.joint_blocks.12.attn2.qkv.weight",
        ];
        let meta = infer_mmdit_metadata_from_iter(keys.iter().map(|s| s.as_str()));
        assert_eq!(meta.qk_norm, QkNormKind::Layer);
        assert_eq!(meta.x_self_attn_layers, Some(12));
    }

    #[test]
    fn infer_metadata_detects_rms_norm() {
        let keys = vec![
            "model.diffusion_model.joint_blocks.5.attn.ln_q.weight",
            "model.diffusion_model.joint_blocks.8.attn2.qkv.weight",
            "model.diffusion_model.joint_blocks.3.attn2.qkv.weight",
        ];
        let meta = infer_mmdit_metadata_from_iter(keys.iter().map(|s| s.as_str()));
        assert_eq!(meta.qk_norm, QkNormKind::Rms);
        assert_eq!(meta.x_self_attn_layers, Some(8));
    }

    #[test]
    fn infer_metadata_defaults_when_no_keys() {
        let meta = infer_mmdit_metadata_from_iter(std::iter::empty::<&str>());
        assert_eq!(meta.qk_norm, QkNormKind::Disabled);
        assert_eq!(meta.x_self_attn_layers, None);
    }
}
