use flame_core::device::Device;
use flame_core::Result;
use flame_core::{DType, Shape, Tensor};
use memmap2::MmapOptions;
use safetensors::SafeTensors;
use std::{collections::HashMap, fs::File, path::Path};

// Lazy safetensors loader that memory maps files

pub struct LazySafetensorsLoader {
    file_path: String,
    // name -> (shape, dtype)
    metadata: HashMap<String, (Vec<usize>, DType)>,
}

impl LazySafetensorsLoader {
    pub fn new(path: &Path) -> flame_core::Result<Self> {
        // Memory-map file and collect only metadata (names, shapes, dtypes)
        let file = File::open(path)
            .map_err(|e| flame_core::Error::InvalidOperation(e.to_string()))?;
        let mmap = unsafe {
            MmapOptions::new().map(&file).map_err(|e| {
                flame_core::Error::InvalidOperation(format!("Failed to mmap file: {}", e))
            })?
        };
        let st = SafeTensors::deserialize(&mmap).map_err(|e| {
            flame_core::Error::InvalidOperation(format!(
                "Failed to deserialize safetensors: {}",
                e
            ))
        })?;
        let mut metadata = HashMap::new();

        for name in st.names() {
            if let Ok(tensor_view) = st.tensor(name) {
                let shape = tensor_view.shape().to_vec();
                let dtype = match tensor_view.dtype() {
                    safetensors::Dtype::F32 => DType::F32,
                    safetensors::Dtype::F16 => DType::F16,
                    safetensors::Dtype::BF16 => DType::BF16,
                    _ => DType::F32,
                };
                metadata.insert(name.to_string(), (shape, dtype));
            }
        }

        Ok(Self { file_path: path.to_string_lossy().to_string(), metadata })
    }

    pub fn load_tensor(&self, name: &str, device: &Device) -> flame_core::Result<Tensor> {
        // On-demand: mmap the file, deserialize metadata, fetch only the requested tensor
        let file = File::open(&self.file_path)
            .map_err(|e| flame_core::Error::InvalidOperation(e.to_string()))?;
        let mmap = unsafe {
            MmapOptions::new().map(&file).map_err(|e| {
                flame_core::Error::InvalidOperation(format!("Failed to mmap file: {}", e))
            })?
        };
        let st = SafeTensors::deserialize(&mmap).map_err(|e| {
            flame_core::Error::InvalidOperation(format!(
                "Failed to deserialize safetensors: {}",
                e
            ))
        })?;

        let view = st.tensor(name).map_err(|_| {
            flame_core::Error::InvalidOperation(format!("Tensor {} not found", name))
        })?;

        let shape = Shape::from_dims(view.shape());
        // Convert data into a FLAME tensor preserving original dtype by default
        let tensor = match view.dtype() {
            safetensors::Dtype::F32 => {
                let data = view.data();
                let float_data: Vec<f32> = data
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                Tensor::from_vec_dtype(float_data, shape, device.cuda_device().clone(), DType::F32)?
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
                Tensor::from_vec_dtype(float_data, shape, device.cuda_device().clone(), DType::F16)?
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
                    DType::BF16,
                )?
            }
            _ => {
                // Default to F32 for unknown types
                let data = view.data();
                let float_data: Vec<f32> = data
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                Tensor::from_vec_dtype(float_data, shape, device.cuda_device().clone(), DType::F32)?
            }
        };
        Ok(tensor)
    }
}

pub fn create_lazy_tensor_provider(path: &Path) -> flame_core::Result<LazySafetensorsLoader> {
    LazySafetensorsLoader::new(path)
}
