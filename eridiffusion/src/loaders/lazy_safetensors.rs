use crate::loaders::mmdit_weights::TensorAccess;
use bytemuck::cast_slice;
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
        let file =
            File::open(path).map_err(|e| flame_core::Error::InvalidOperation(e.to_string()))?;
        let mmap = unsafe {
            MmapOptions::new().map(&file).map_err(|e| {
                flame_core::Error::InvalidOperation(format!("Failed to mmap file: {}", e))
            })?
        };
        let st = SafeTensors::deserialize(&mmap).map_err(|e| {
            flame_core::Error::InvalidOperation(format!("Failed to deserialize safetensors: {}", e))
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
            flame_core::Error::InvalidOperation(format!("Failed to deserialize safetensors: {}", e))
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

    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.metadata.keys()
    }

    pub fn has_key(&self, name: &str) -> bool {
        self.metadata.contains_key(name)
    }

    pub fn tensor_meta(&self, name: &str) -> Option<(&[usize], DType)> {
        self.metadata.get(name).map(|(shape, dtype)| (shape.as_slice(), *dtype))
    }

    pub fn load_tensor_bf16(&self, name: &str) -> flame_core::Result<Vec<u16>> {
        let file = File::open(&self.file_path)
            .map_err(|e| flame_core::Error::InvalidOperation(e.to_string()))?;
        let mmap = unsafe {
            MmapOptions::new().map(&file).map_err(|e| {
                flame_core::Error::InvalidOperation(format!("Failed to mmap file: {}", e))
            })?
        };
        let st = SafeTensors::deserialize(&mmap).map_err(|e| {
            flame_core::Error::InvalidOperation(format!("Failed to deserialize safetensors: {}", e))
        })?;

        let view = st.tensor(name).map_err(|_| {
            flame_core::Error::InvalidOperation(format!("Tensor {} not found", name))
        })?;

        let out = match view.dtype() {
            safetensors::Dtype::BF16 => cast_slice(view.data()).to_vec(),
            safetensors::Dtype::F16 => view
                .data()
                .chunks_exact(2)
                .map(|chunk| {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    let value = half::f16::from_bits(bits);
                    half::bf16::from_f32(value.to_f32()).to_bits()
                })
                .collect(),
            safetensors::Dtype::F32 => view
                .data()
                .chunks_exact(4)
                .map(|chunk| {
                    let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    half::bf16::from_f32(value).to_bits()
                })
                .collect(),
            other => {
                return Err(flame_core::Error::InvalidOperation(format!(
                    "Unsupported dtype {:?} for tensor {}",
                    other, name
                )));
            }
        };

        Ok(out)
    }
}

pub fn create_lazy_tensor_provider(path: &Path) -> flame_core::Result<LazySafetensorsLoader> {
    LazySafetensorsLoader::new(path)
}

#[derive(Clone)]
pub struct LazyPrefixedLoader<'a> {
    loader: &'a LazySafetensorsLoader,
    device: Device,
    prefix: String,
}

impl<'a> LazyPrefixedLoader<'a> {
    pub fn new(
        loader: &'a LazySafetensorsLoader,
        prefix: impl Into<String>,
        device: Device,
    ) -> Self {
        Self { loader, device, prefix: prefix.into() }
    }

    pub fn prefixed(&self, prefix: &str) -> Self {
        let new_prefix = if self.prefix.is_empty() {
            prefix.to_string()
        } else {
            format!("{}.{}", self.prefix, prefix)
        };
        Self::new(self.loader, new_prefix, self.device.clone())
    }

    fn full_key(&self, key: &str) -> String {
        if self.prefix.is_empty() {
            key.to_string()
        } else {
            format!("{}.{}", self.prefix, key)
        }
    }
}

impl<'a> TensorAccess for LazyPrefixedLoader<'a> {
    fn get_tensor(&self, key: &str) -> Result<Tensor> {
        let full = self.full_key(key);
        let tensor = self.loader.load_tensor(&full, &self.device)?;
        if tensor.dtype() == DType::BF16 && tensor.storage_dtype() == DType::BF16 {
            Ok(tensor.requires_grad_(false))
        } else {
            tensor.to_dtype(DType::BF16).map(|t| t.requires_grad_(false))
        }
    }

    fn has_tensor(&self, key: &str) -> bool {
        let full = self.full_key(key);
        self.loader.has_key(&full)
    }

    fn full_key(&self, key: &str) -> String {
        self.full_key(key)
    }

    fn with_prefix(&self, prefix: &str) -> Self {
        self.prefixed(prefix)
    }
}
