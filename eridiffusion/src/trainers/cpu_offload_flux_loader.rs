use flame_core::device::Device;
use flame_core::{DType, Result, Shape, Tensor};
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::path::Path;

/// CPU-offloaded Flux model loader that keeps most weights on CPU
/// and only loads to GPU when needed for forward pass
pub struct CPUOffloadFluxLoader {
    pub device: Device,
    pub cpu_weights: HashMap<String, Vec<f32>>,
    pub gpu_weights: HashMap<String, Tensor>,
    pub weight_shapes: HashMap<String, Vec<usize>>,
}

impl CPUOffloadFluxLoader {
    /// Load Flux model with CPU offloading
    /// Only essential weights stay on GPU, rest stay on CPU
    pub fn load_with_cpu_offload<P: AsRef<Path>>(model_path: P, device: Device) -> Result<Self> {
        let path = model_path.as_ref();
        println!("\n=== CPU-Offloaded Flux Loading ===");
        println!("Model path: {:?}", path);

        // Memory-map the file
        let file = std::fs::File::open(path)
            .map_err(|e| flame_core::Error::Io(format!("Failed to open: {}", e)))?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .map_err(|e| flame_core::Error::Io(format!("Failed to mmap: {}", e)))?;

        let tensors = SafeTensors::deserialize(&mmap).map_err(|e| {
            flame_core::Error::InvalidOperation(format!("Failed to deserialize: {}", e))
        })?;

        let mut cpu_weights = HashMap::new();
        let mut gpu_weights = HashMap::new();
        let mut weight_shapes = HashMap::new();

        let total_tensors = tensors.tensors().len();
        println!("Total tensors in model: {}", total_tensors);

        // Essential patterns that MUST stay on GPU for training
        let gpu_essential = vec![
            "img_in",
            "txt_in", // Input projections
            "time_in",
            "vector_in",   // Embeddings
            "guidance_in", // Guidance
            "final_layer", // Output
        ];

        let mut gpu_count = 0;
        let mut cpu_count = 0;
        let mut gpu_memory_mb = 0.0;

        for (name, view) in tensors.tensors() {
            let shape = view.shape().to_vec();
            let num_elements: usize = shape.iter().product();
            let size_mb = (num_elements * 4) as f32 / (1024.0 * 1024.0);

            weight_shapes.insert(name.to_string(), shape.clone());

            // Check if this should stay on GPU
            let keep_on_gpu = gpu_essential.iter().any(|pattern| name.contains(pattern));

            // Convert to f32 vector
            let float_data = match view.dtype() {
                safetensors::Dtype::F32 => {
                    let data = view.data();
                    data.chunks_exact(4)
                        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                        .collect::<Vec<f32>>()
                }
                safetensors::Dtype::F16 => {
                    let data = view.data();
                    data.chunks_exact(2)
                        .map(|chunk| {
                            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                            half::f16::from_bits(bits).to_f32()
                        })
                        .collect::<Vec<f32>>()
                }
                safetensors::Dtype::BF16 => {
                    let data = view.data();
                    data.chunks_exact(2)
                        .map(|chunk| {
                            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                            half::bf16::from_bits(bits).to_f32()
                        })
                        .collect::<Vec<f32>>()
                }
                _ => {
                    println!("  Skipping unsupported dtype: {:?} for {}", view.dtype(), name);
                    continue;
                }
            };

            if keep_on_gpu {
                // Load to GPU
                println!("  GPU: {} ({:.2} MB)", name, size_mb);
                let tensor = Tensor::from_vec_dtype(
                    float_data,
                    Shape::from_dims(&shape),
                    device.cuda_device_arc(),
                    DType::BF16,
                )?;
                gpu_weights.insert(name.to_string(), tensor);
                gpu_count += 1;
                gpu_memory_mb += size_mb;
            } else {
                // Keep on CPU
                if cpu_count % 100 == 0 {
                    println!(
                        "  CPU: Stored {} weights ({:.0} MB total)...",
                        cpu_count,
                        size_mb * cpu_count as f32
                    );
                }
                cpu_weights.insert(name.to_string(), float_data);
                cpu_count += 1;
            }
        }

        println!("\n=== CPU Offload Complete ===");
        println!("GPU weights: {} ({:.2} MB)", gpu_count, gpu_memory_mb);
        println!("CPU weights: {} (~{:.2} GB)", cpu_count, (cpu_count as f32 * 4.0) / 1024.0);
        println!("Memory saved on GPU: ~{:.1} GB", ((cpu_count as f32 * 4.0) / 1024.0) / 1024.0);

        Ok(Self { device, cpu_weights, gpu_weights, weight_shapes })
    }

    /// Load a weight from CPU to GPU on-demand
    pub fn load_to_gpu(&mut self, name: &str) -> Result<Tensor> {
        // Check if already on GPU
        if let Some(tensor) = self.gpu_weights.get(name) {
            return Ok(tensor.clone());
        }

        // Load from CPU
        if let Some(cpu_data) = self.cpu_weights.get(name) {
            if let Some(shape) = self.weight_shapes.get(name) {
                let tensor = Tensor::from_vec_dtype(
                    cpu_data.clone(),
                    Shape::from_dims(shape),
                    self.device.cuda_device_arc(),
                    DType::BF16,
                )?;

                // Cache it on GPU for this forward pass
                let tensor_clone = tensor.clone();
                self.gpu_weights.insert(name.to_string(), tensor_clone);
                Ok(tensor)
            } else {
                Err(flame_core::Error::InvalidOperation(format!(
                    "Shape not found for weight: {}",
                    name
                )))
            }
        } else {
            Err(flame_core::Error::InvalidOperation(format!("Weight not found: {}", name)))
        }
    }

    /// Clear GPU cache (call after each forward pass to free memory)
    pub fn clear_gpu_cache(&mut self, keep_essential: bool) {
        if keep_essential {
            // Only keep essential weights
            let essential =
                vec!["img_in", "txt_in", "time_in", "vector_in", "guidance_in", "final_layer"];
            self.gpu_weights
                .retain(|name, _| essential.iter().any(|pattern| name.contains(pattern)));
        } else {
            // Clear everything
            self.gpu_weights.clear();
        }
    }

    /// Get total weights count
    pub fn total_weights(&self) -> usize {
        self.cpu_weights.len() + self.gpu_weights.len()
    }
}
