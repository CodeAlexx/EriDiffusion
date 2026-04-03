//! SDXL strict mmap weight provider (Phase-4 style).
//! Mirrors `chroma::weights::MmapWeightProvider` but uses the SDXL key map.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use anyhow::anyhow;
use eridiffusion_common_weights::strict_loader::{tensor_from_bytes, StrictMmapLoader};
use flame_core::{DType, Device as FlameDevice, Tensor};

use super::keymap::SdxlKeyMap;
use crate::streaming::{DeviceWeights, KeyMapOwned, WeightProvider};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum QkvSlice {
    Q,
    K,
    V,
}

fn resolve_qkv_key(key: &str) -> Option<(String, QkvSlice)> {
    const TARGETS: &[(&str, &str, QkvSlice)] = &[
        (".attn1.to_q.weight", ".attn1.to_qkv.weight", QkvSlice::Q),
        (".attn1.to_k.weight", ".attn1.to_qkv.weight", QkvSlice::K),
        (".attn1.to_v.weight", ".attn1.to_qkv.weight", QkvSlice::V),
        (".attn1.to_q.bias", ".attn1.to_qkv.bias", QkvSlice::Q),
        (".attn1.to_k.bias", ".attn1.to_qkv.bias", QkvSlice::K),
        (".attn1.to_v.bias", ".attn1.to_qkv.bias", QkvSlice::V),
        (".attn2.to_q.weight", ".attn2.to_qkv.weight", QkvSlice::Q),
        (".attn2.to_k.weight", ".attn2.to_qkv.weight", QkvSlice::K),
        (".attn2.to_v.weight", ".attn2.to_qkv.weight", QkvSlice::V),
        (".attn2.to_q.bias", ".attn2.to_qkv.bias", QkvSlice::Q),
        (".attn2.to_k.bias", ".attn2.to_qkv.bias", QkvSlice::K),
        (".attn2.to_v.bias", ".attn2.to_qkv.bias", QkvSlice::V),
    ];
    for &(suffix, fused_suffix, slice) in TARGETS {
        if let Some(prefix) = key.strip_suffix(suffix) {
            return Some((format!("{prefix}{fused_suffix}"), slice));
        }
    }
    None
}

fn split_qkv(fused: &Tensor) -> anyhow::Result<(Tensor, Tensor, Tensor)> {
    let dims = fused.shape().dims();
    let (d0, d1) = (dims[0], dims[1]);
    if d0 % 3 == 0 {
        let d = d0 / 3;
        let q = fused.narrow(0, 0 * d, d)?;
        let k = fused.narrow(0, 1 * d, d)?;
        let v = fused.narrow(0, 2 * d, d)?;
        return Ok((q, k, v));
    }
    if d1 % 3 == 0 {
        let d = d1 / 3;
        let q = fused.narrow(1, 0 * d, d)?;
        let k = fused.narrow(1, 1 * d, d)?;
        let v = fused.narrow(1, 2 * d, d)?;
        return Ok((q, k, v));
    }
    Err(anyhow!("QKV fused weight has incompatible shape: {:?}", dims.to_vec()))
}

fn ensure_bf16(t: Tensor) -> anyhow::Result<Tensor> {
    match t.dtype() {
        DType::BF16 => Ok(t),
        DType::F16 | DType::F32 => Ok(t.to_dtype(DType::BF16)?.requires_grad_(false).detach()?),
        other => Err(anyhow!("unsupported tensor dtype {:?}; expected BF16/F16/F32", other)),
    }
}

#[derive(Default)]
struct TensorCache {
    map: Mutex<HashMap<String, Tensor>>,
}

impl TensorCache {
    fn get(&self, key: &str) -> Option<Tensor> {
        self.map.lock().unwrap().get(key).cloned()
    }

    fn insert(&self, key: String, tensor: Tensor) {
        self.map.lock().unwrap().insert(key, tensor);
    }

    fn get_or_try_insert<F>(&self, key: &str, build: F) -> anyhow::Result<Tensor>
    where
        F: FnOnce() -> anyhow::Result<Tensor>,
    {
        {
            let guard = self.map.lock().unwrap();
            if let Some(existing) = guard.get(key) {
                return Ok(existing.clone());
            }
        }

        let tensor = build()?;
        let mut guard = self.map.lock().unwrap();
        let entry = guard.entry(key.to_string()).or_insert_with(|| tensor.clone());
        Ok(entry.clone())
    }
}

pub struct SdxlWeightProvider {
    mmap: Arc<StrictMmapLoader>,
    device: FlameDevice,
    cache: TensorCache,
}

impl SdxlWeightProvider {
    pub fn new(mmap: Arc<StrictMmapLoader>, device: FlameDevice) -> Self {
        Self { mmap, device, cache: TensorCache::default() }
    }

    pub fn device(&self) -> &FlameDevice {
        &self.device
    }

    pub fn tensor_shape(&self, key: &str) -> anyhow::Result<Vec<usize>> {
        let info = self.mmap.info(key).map_err(|e| anyhow!("missing tensor {key}: {e}"))?;
        Ok(info.shape)
    }

    fn load_tensor_raw(&self, key: &str) -> anyhow::Result<Tensor> {
        let info = self.mmap.info(key).map_err(|e| anyhow!("missing tensor {key}: {e}"))?;
        let raw = self.mmap.bytes(key).map_err(|e| anyhow!("read tensor {key}: {e}"))?;
        let tensor = tensor_from_bytes(self.device.clone(), &info, raw)
            .map_err(|e| anyhow!("tensor_from_bytes({key}): {e}"))?
            .requires_grad_(false)
            .detach()?;
        ensure_bf16(tensor)
    }

    fn load_qkv_slice(&self, phys: &str, which: QkvSlice) -> anyhow::Result<Tensor> {
        let cache_key = format!("{}::{}", phys, which.cache_suffix());
        self.cache.get_or_try_insert(&cache_key, || {
            let fused = self.load_tensor_raw(phys)?;
            let (q, k, v) = split_qkv(&fused)?;

            // Populate sibling slices eagerly so future lookups avoid re-splitting.
            if which != QkvSlice::Q {
                self.cache.insert(format!("{}::{}", phys, QkvSlice::Q.cache_suffix()), q.clone());
            }
            if which != QkvSlice::K {
                self.cache.insert(format!("{}::{}", phys, QkvSlice::K.cache_suffix()), k.clone());
            }
            if which != QkvSlice::V {
                self.cache.insert(format!("{}::{}", phys, QkvSlice::V.cache_suffix()), v.clone());
            }

            let result = match which {
                QkvSlice::Q => q,
                QkvSlice::K => k,
                QkvSlice::V => v,
            };
            Ok(result)
        })
    }

    /// Load a single tensor by key and place it on the provider device.
    pub fn load_tensor(&self, key: &str) -> anyhow::Result<Tensor> {
        self.cache.get_or_try_insert(key, || self.load_tensor_raw(key))
    }
}

impl WeightProvider for SdxlWeightProvider {
    fn load_block_to_gpu(&self, i: usize) -> anyhow::Result<DeviceWeights> {
        let mut out = Vec::new();
        for key in SdxlKeyMap::gen_keys_for_block(i) {
            if let Some((phys, which)) = resolve_qkv_key(&key) {
                if self.mmap.info(&phys).is_ok() {
                    out.push(self.load_qkv_slice(&phys, which)?);
                    continue;
                }
            }

            out.push(self.load_tensor(&key)?);
        }
        Ok(DeviceWeights { tensors: out })
    }

    fn load_head_to_gpu(&self) -> anyhow::Result<DeviceWeights> {
        let keys = [
            "model.diffusion_model.label_emb.0.0.weight",
            "model.diffusion_model.label_emb.0.0.bias",
            "model.diffusion_model.label_emb.0.2.weight",
            "model.diffusion_model.label_emb.0.2.bias",
        ];
        let mut tensors = Vec::with_capacity(keys.len());
        for key in keys {
            let tensor = if key.ends_with("label_emb.0.0.weight") {
                self.cache.get_or_try_insert(key, || {
                    let mut t = self.load_tensor_raw(key)?;
                    if let [rows, cols] = t.shape().dims() {
                        if (*rows, *cols) == (1280, 2816) {
                            let orig_dtype = t.dtype();
                            let mut work = if orig_dtype == DType::F32 {
                                t.clone()
                            } else {
                                t.to_dtype(DType::F32)?
                            };
                            work = work.transpose()?;
                            if orig_dtype != DType::F32 {
                                work = work.to_dtype(orig_dtype)?;
                            }
                            t = work;
                        }
                    }
                    Ok(t)
                })?
            } else {
                self.load_tensor(key)?
            };
            tensors.push(tensor);
        }
        Ok(DeviceWeights { tensors })
    }

    fn prefetch_block(&self, _block_id: usize) -> anyhow::Result<()> {
        Ok(())
    }

    fn release_block(&self, _block_id: isize) -> anyhow::Result<()> {
        Ok(())
    }
}

impl QkvSlice {
    fn cache_suffix(&self) -> &'static str {
        match self {
            QkvSlice::Q => "Q",
            QkvSlice::K => "K",
            QkvSlice::V => "V",
        }
    }
}
