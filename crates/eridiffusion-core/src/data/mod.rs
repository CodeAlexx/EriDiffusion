pub mod bucket_dataset;

use std::path::PathBuf;
use std::sync::Arc;

use cudarc::driver::CudaDevice;
use flame_core::{Shape, Tensor};
use crate::Result;

/// A single cached latent sample (pre-encoded to safetensors)
pub struct CachedSample {
    latent: Vec<f32>,
    latent_shape: Vec<usize>,
    pub embedding_keys: Vec<String>,
    embedding_data: Vec<Vec<f32>>,
}

impl CachedSample {
    pub fn latent_tensor(&self, device: &Arc<CudaDevice>) -> Result<Tensor> {
        Ok(Tensor::from_vec(
            self.latent.clone(),
            Shape::from_dims(&self.latent_shape),
            device.clone(),
        )?)
    }

    pub fn embedding(&self, key: &str, device: &Arc<CudaDevice>) -> Result<Tensor> {
        for (k, data) in self.embedding_keys.iter().zip(&self.embedding_data) {
            if k == key {
                return Ok(Tensor::from_vec(data.clone(), Shape::from_dims(&[1, data.len()]), device.clone())?);
            }
        }
        Err(crate::EriDiffusionError::Data(format!("embedding key not found: {key}")))
    }
}

/// Dataset of cached latent samples
pub struct CachedDataset {
    samples: Vec<CachedSample>,
}

impl CachedDataset {
    pub fn load(dir: &PathBuf) -> Result<Self> {
        let mut samples = Vec::new();
        if dir.is_dir() {
            // Load .safetensors files from cache directory
            for entry in std::fs::read_dir(dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.extension().map_or(false, |e| e == "safetensors") {
                    if let Ok(sample) = Self::load_cache_file(&path) {
                        samples.push(sample);
                    }
                }
            }
        }
        log::info!("Loaded {} cached samples from {:?}", samples.len(), dir);
        Ok(Self { samples })
    }

    fn load_cache_file(path: &std::path::Path) -> Result<CachedSample> {
        let data = std::fs::read(path)?;
        let tensors = safetensors::SafeTensors::deserialize(&data)
            .map_err(|e| crate::EriDiffusionError::Safetensors(format!("{}", e)))?;

        let mut latent = Vec::new();
        let mut latent_shape = Vec::new();
        let mut embedding_keys = Vec::new();
        let mut embedding_data = Vec::new();

        for name in tensors.names() {
            let view = tensors.tensor(name).unwrap();
            let bytes = view.data();
            let f32_data: Vec<f32> = bytes.chunks(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();

            if name == "latent" {
                latent = f32_data;
                latent_shape = view.shape().to_vec();
            } else {
                embedding_keys.push(name.to_string());
                embedding_data.push(f32_data);
            }
        }

        Ok(CachedSample { latent, latent_shape, embedding_keys, embedding_data })
    }

    pub fn len(&self) -> usize { self.samples.len() }
    pub fn is_empty(&self) -> bool { self.samples.is_empty() }

    pub fn get(&self, idx: usize) -> Option<&CachedSample> { self.samples.get(idx) }
}
