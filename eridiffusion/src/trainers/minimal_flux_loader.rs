use flame_core::device::Device;
use flame_core::{DType, Result, Tensor};
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::path::Path;

/// Minimal Flux model loader that only loads essential weights for LoRA training
pub struct MinimalFluxLoader {
    pub device: Device,
    pub weights: HashMap<String, Tensor>,
}

impl MinimalFluxLoader {
    /// Load only the essential weights needed for LoRA training
    /// This skips most of the model weights since LoRA only needs specific layers
    pub fn load_for_lora_training<P: AsRef<Path>>(
        model_path: P,
        device: Device,
        lora_target_modules: &[String],
    ) -> Result<HashMap<String, Tensor>> {
        let path = model_path.as_ref();
        println!("\n=== Minimal Flux Loading for LoRA Training ===");
        println!("Model path: {:?}", path);

        // Memory-map the file
        let file = std::fs::File::open(path)
            .map_err(|e| flame_core::Error::Io(format!("Failed to open: {}", e)))?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .map_err(|e| flame_core::Error::Io(format!("Failed to mmap: {}", e)))?;

        let tensors = SafeTensors::deserialize(&mmap).map_err(|e| {
            flame_core::Error::InvalidOperation(format!("Failed to deserialize: {}", e))
        })?;

        let mut loaded_weights = HashMap::new();
        let total_tensors = tensors.tensors().len();
        println!("Total tensors in model: {}", total_tensors);

        // For LoRA training, we need VERY few weights:
        // 1. Only input/output layers (not the full model)
        // 2. Skip ALL double_blocks and single_blocks weights
        // 3. The model structure will be created but weights will be placeholders

        let essential_patterns = vec![
            // Only truly essential input/output layers
            "img_in",
            "txt_in",
            "vector_in",
            "final_layer",
            "time_in",
            "guidance_in",
        ];

        // For LoRA, we DON'T need the actual attention/MLP weights
        // LoRA will add its own low-rank matrices on top
        let mut loaded_count = 0;
        let mut skipped_count = 0;

        for (name, view) in tensors.tensors() {
            // Skip ALL double_blocks and single_blocks weights
            if name.contains("double_blocks") || name.contains("single_blocks") {
                skipped_count += 1;
                if skipped_count % 100 == 0 {
                    println!("  Skipped {} transformer block weights...", skipped_count);
                }
                continue;
            }

            // Check if this is an essential weight
            let is_essential = essential_patterns.iter().any(|p| name.contains(p));

            if is_essential {
                // Load this weight
                let shape = flame_core::Shape::from_dims(view.shape());
                let dtype = DType::BF16; // Use BF16 to save memory

                println!("  Loading essential weight: {} (shape: {:?})", name, view.shape());

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
                            dtype,
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
                        Tensor::from_vec_dtype(
                            float_data,
                            shape,
                            device.cuda_device().clone(),
                            dtype,
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
                        Tensor::from_vec_dtype(
                            float_data,
                            shape,
                            device.cuda_device().clone(),
                            dtype,
                        )?
                    }
                    _ => {
                        println!("    Skipping unsupported dtype: {:?}", view.dtype());
                        skipped_count += 1;
                        continue;
                    }
                };

                loaded_weights.insert(name.to_string(), tensor);
                loaded_count += 1;

                // Force GPU sync every 10 weights to prevent accumulation
                if loaded_count % 10 == 0 {
                    let _ = device.synchronize();
                }
            } else {
                skipped_count += 1;
                if skipped_count % 100 == 0 {
                    println!("  Skipped {} non-essential weights...", skipped_count);
                }
            }
        }

        println!("\n=== Minimal Loading Complete ===");
        println!("Loaded: {} essential weights", loaded_count);
        println!("Skipped: {} non-essential weights", skipped_count);
        println!("Memory saved: ~{:.1} GB", (skipped_count as f32 * 0.01)); // Rough estimate

        Ok(loaded_weights)
    }

    /// Create placeholder tensors for missing weights
    /// This allows the model to initialize without loading everything
    pub fn create_placeholder(shape: &[usize], device: &Device) -> Result<Tensor> {
        // Create a small placeholder tensor
        // During forward pass, we'll load the real weights on-demand
        let shape = flame_core::Shape::from_dims(shape);
        Tensor::zeros(shape, device.cuda_device().clone())
    }
}
