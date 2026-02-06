use crate::loaders::WeightLoader;
use crate::models::text_encoder_complete::{
    CLIPConfig, CLIPTextEncoder as FlameClipTextEncoder, T5Config, T5Encoder,
};
use bytemuck::try_cast_slice;
use flame_core::device::Device;
use flame_core::{DType, Error, Result, Shape, Tensor};
use memmap2::Mmap;
use safetensors::{tensor::TensorView, SafeTensors};
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

/// CPU snapshot for CLIP and T5 weights. Holds a memory-mapped safetensors file and
/// provides helpers to materialize the encoder on a target CUDA device on demand.
pub struct SafetensorSnapshot {
    pub path: PathBuf,
    mmap: Mmap,
    tensors: SafeTensors<'static>,
}

impl SafetensorSnapshot {
    pub fn load(path: &Path) -> Result<Self> {
        let file = File::open(path).map_err(|e| {
            Error::InvalidOperation(format!("Failed to open {}: {}", path.display(), e))
        })?;
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| {
            Error::InvalidOperation(format!("Failed to mmap {}: {}", path.display(), e))
        })?;
        let raw: &'static [u8] = unsafe { std::slice::from_raw_parts(mmap.as_ptr(), mmap.len()) };
        let tensors = SafeTensors::deserialize(raw).map_err(|e| {
            Error::InvalidOperation(format!(
                "Failed to parse safetensors {}: {}",
                path.display(),
                e
            ))
        })?;
        Ok(Self { path: path.to_path_buf(), mmap, tensors })
    }

    pub fn tensors(&self) -> &SafeTensors<'static> {
        &self.tensors
    }
}

pub struct ClipCpuSnapshot {
    snapshot: SafetensorSnapshot,
    config: CLIPConfig,
}

pub struct T5CpuSnapshot {
    snapshot: SafetensorSnapshot,
    config: T5Config,
}

pub struct TextEncodersCpuSnapshot {
    pub clip_l: Option<ClipCpuSnapshot>,
    pub clip_g: Option<ClipCpuSnapshot>,
    pub t5: Option<T5CpuSnapshot>,
}

impl ClipCpuSnapshot {
    pub fn load(path: &Path) -> Result<Self> {
        let snapshot = SafetensorSnapshot::load(path)?;
        let tensors = snapshot.tensors();
        // Infer hidden size from embeddings.
        let hidden_size = tensors
            .tensors()
            .iter()
            .find_map(|(name, view)| {
                if name.ends_with("token_embedding.weight")
                    || name.ends_with("position_embedding.weight")
                {
                    view.shape().get(1).copied()
                } else {
                    None
                }
            })
            .unwrap_or(768);

        let mut config = if hidden_size == 1280 {
            CLIPConfig::clip_g()
        } else if hidden_size == 768 {
            CLIPConfig::clip_l()
        } else {
            let mut cfg = CLIPConfig::clip_l();
            cfg.hidden_size = hidden_size;
            cfg.intermediate_size = hidden_size * 4;
            cfg.num_attention_heads = hidden_size / 64;
            cfg.projection_dim = Some(hidden_size);
            cfg
        };

        if let Ok(view) = tensors.tensor("text_model.embeddings.token_embedding.weight") {
            config.vocab_size = view.shape()[0];
        }
        if let Ok(view) = tensors.tensor("text_model.embeddings.position_embedding.weight") {
            config.max_position_embeddings = view.shape()[0];
        }

        Ok(Self { snapshot, config })
    }

    pub fn instantiate(&self, device: &Device) -> Result<FlameClipTextEncoder> {
        let mut weights = HashMap::new();
        for (name, view) in self.snapshot.tensors().tensors() {
            let tensor =
                tensor_from_view(device, DType::BF16, Shape::from_dims(view.shape()), &view)?;
            weights.insert(name.to_string(), tensor);
        }
        FlameClipTextEncoder::new(self.config.clone(), device.clone(), weights)
            .map_err(|e| Error::InvalidOperation(format!("Failed to instantiate CLIP: {}", e)))
    }

    pub fn config(&self) -> &CLIPConfig {
        &self.config
    }
}

impl T5CpuSnapshot {
    pub fn load(path: &Path) -> Result<Self> {
        let snapshot = SafetensorSnapshot::load(path)?;
        let tensors = snapshot.tensors();
        // T5 config is fixed for SD3.5/Flux (XXL). We'll still infer vocab if available.
        let mut config = T5Config::t5_xxl();
        if let Ok(view) = tensors.tensor("shared.weight") {
            config.vocab_size = view.shape()[0];
            config.d_model = view.shape()[1];
        }
        Ok(Self { snapshot, config })
    }

    pub fn instantiate(&self, device: &Device) -> Result<T5Encoder> {
        let mut weights = HashMap::new();
        let mut total_bytes = 0;
        let mut count = 0;
        for (name, view) in self.snapshot.tensors().tensors() {
            // Only load encoder weights and shared embeddings
            if name == "shared.weight" || name.starts_with("encoder.") {
                let tensor =
                    tensor_from_view(device, DType::BF16, Shape::from_dims(view.shape()), &view)?;
                let size = tensor.shape().elem_count() * 2;
                total_bytes += size;
                count += 1;
                println!("[{}] Loaded {} ({:?}, {:?}) - Total: {:.2} GB", count, name, tensor.shape(), tensor.dtype(), total_bytes as f64 / 1024.0 / 1024.0 / 1024.0);
                weights.insert(name.to_string(), tensor);
            }
        }
        println!("Loaded {} T5 tensors, total size: {:.2} GB", count, total_bytes as f64 / 1024.0 / 1024.0 / 1024.0);
        
        // Pass map directly to from_map to avoid copying
        T5Encoder::from_map(self.config.clone(), device.clone(), weights).map_err(|e| {
            Error::InvalidOperation(format!("Failed to instantiate T5 encoder: {}", e))
        })
    }

    pub fn config(&self) -> &T5Config {
        &self.config
    }
}

impl TextEncodersCpuSnapshot {
    pub fn load(clip_l: Option<&Path>, clip_g: Option<&Path>, t5: Option<&Path>) -> Result<Self> {
        Ok(Self {
            clip_l: match clip_l {
                Some(path) => Some(ClipCpuSnapshot::load(path)?),
                None => None,
            },
            clip_g: match clip_g {
                Some(path) => Some(ClipCpuSnapshot::load(path)?),
                None => None,
            },
            t5: match t5 {
                Some(path) => Some(T5CpuSnapshot::load(path)?),
                None => None,
            },
        })
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
            Tensor::from_bf16_chunks(shape, device.cuda_device_arc(), |offset, chunk| {
                let end = offset + chunk.len();
                if end > src.len() {
                    return Err(Error::InvalidOperation("Chunk exceeds source length".into()));
                }
                let slice = &src[offset..end];
                for (dst, &value) in chunk.iter_mut().zip(slice.iter()) {
                    *dst = half::bf16::from_f32(value).to_bits();
                }
                Ok(())
            })
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
            Tensor::from_bf16_chunks(shape, device.cuda_device_arc(), |offset, chunk| {
                let end = offset + chunk.len();
                if end > src.len() {
                    return Err(Error::InvalidOperation("Chunk exceeds source length".into()));
                }
                let slice = &src[offset..end];
                for (dst, &bits) in chunk.iter_mut().zip(slice.iter()) {
                    let value = half::f16::from_bits(bits).to_f32();
                    *dst = half::bf16::from_f32(value).to_bits();
                }
                Ok(())
            })
        }
        other => {
            Err(Error::InvalidOperation(format!("Unsupported dtype {:?} for BF16 target", other)))
        }
    }
}
