//! IO pipeline upgrades: BF16 mmap cache, async pinned staging, and bucketing.

use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
};

use eridiffusion_core::{DType, Device, Error, Result};
use eridiffusion_models::devtensor::tensor_from_vec_on;
use flame_core::{Shape, Tensor};
use tokio::sync::mpsc;

/// Memory-mapped BF16 cache for latents + text embeddings.
/// Expected layout: plain row-major BF16 for each tensor with a JSONL sidecar for checksums and shapes.
pub struct Bf16MmapCache {
    pub root: PathBuf,
    pub manifest: Vec<CacheEntry>,
}

#[derive(Debug, Clone)]
pub struct CacheEntry {
    pub key: String,
    pub path: PathBuf,
    pub shape: Vec<usize>,
    pub checksum: String,
}

impl Bf16MmapCache {
    pub fn open(root: &Path) -> Result<Self> {
        let jsonl = root.join("manifest.jsonl");
        let file = File::open(&jsonl).map_err(|e| Error::Io(e))?;
        let reader = BufReader::new(file);
        let mut manifest = Vec::new();
        for line in reader.lines() {
            let line = line.map_err(|e| Error::Io(e))?;
            if line.trim().is_empty() {
                continue;
            }
            let v: serde_json::Value =
                serde_json::from_str(&line).map_err(|e| Error::Config(e.to_string()))?;
            let key = v["key"].as_str().unwrap_or("").to_string();
            let rel = v["path"].as_str().unwrap_or("");
            let path = root.join(rel);
            let shape = v["shape"]
                .as_array()
                .unwrap_or(&vec![])
                .iter()
                .filter_map(|x| x.as_u64())
                .map(|u| u as usize)
                .collect::<Vec<_>>();
            let checksum = v["checksum"].as_str().unwrap_or("").to_string();
            manifest.push(CacheEntry { key, path, shape, checksum });
        }
        Ok(Self { root: root.to_path_buf(), manifest })
    }

    /// Map BF16 file and present as a FLAME tensor on given device.
    pub fn map_tensor(&self, key: &str, device: &Device) -> Result<Tensor> {
        let rec = self
            .manifest
            .iter()
            .find(|e| e.key == key)
            .ok_or_else(|| Error::DataError(format!("cache missing key {}", key)))?;
        // For now, read file into memory and create tensor (placeholder for true mmap view)
        let data = std::fs::read(&rec.path).map_err(Error::Io)?;
        // BF16 -> F32 conversion (naive): interpret pairs of bytes as BF16, widen to F32
        let n = data.len() / 2;
        let mut f32s = Vec::with_capacity(n);
        for i in 0..n {
            let lo = data[2 * i] as u16;
            let hi = data[2 * i + 1] as u16;
            let bf = (hi << 8) | lo;
            // Simple widening: place BF16 in high bits of F32
            let bits = (bf as u32) << 16;
            let f = f32::from_bits(bits);
            f32s.push(f);
        }
        let tensor = tensor_from_vec_on(f32s, Shape::from_dims(&rec.shape), device, DType::F32)
            .map_err(Error::from)?;
        let tensor = if matches!(device, Device::Cuda(_)) {
            tensor.to_dtype(DType::BF16).map_err(Error::from)?
        } else {
            tensor
        };
        Ok(tensor)
    }
}

/// Bounded MPMC prefetch queue: workers push, trainer pops.
pub struct PrefetchQueue<T> {
    tx: mpsc::Sender<T>,
    rx: mpsc::Receiver<T>,
}

impl<T> PrefetchQueue<T> {
    pub fn bounded(capacity: usize) -> Self {
        let (tx, rx) = mpsc::channel(capacity);
        Self { tx, rx }
    }
    pub fn sender(&self) -> mpsc::Sender<T> {
        self.tx.clone()
    }
    pub fn receiver(&mut self) -> &mut mpsc::Receiver<T> {
        &mut self.rx
    }
}

/// Pinned host staging + async H2D/D2H helpers.
pub mod pinned_staging {
    use core::ffi::c_void;

    use eridiffusion_core::Error;
    use flame_core::{
        device::CudaStreamRawPtrExt, memcpy_async_device_to_host, memcpy_async_host_to_device,
        PinnedAllocFlags, PinnedHostBuffer,
    };

    use super::*;

    pub struct HostStager {
        buffer: PinnedHostBuffer<u8>,
        bytes: usize,
    }

    impl HostStager {
        pub fn ptr(&self) -> *mut c_void {
            self.buffer.as_ptr() as *mut c_void
        }

        pub fn bytes(&self) -> usize {
            self.bytes
        }

        pub fn as_mut_bytes(&mut self) -> &mut [u8] {
            // Safe: length initialised in `alloc`
            self.buffer.as_mut_slice()
        }
    }

    pub fn alloc(bytes: usize) -> Result<HostStager> {
        let mut buffer =
            PinnedHostBuffer::<u8>::with_capacity_elems(bytes, PinnedAllocFlags::WRITE_COMBINED)
                .map_err(|e| Error::Device(format!("pinned alloc failed: {e}")))?;
        unsafe {
            buffer.set_len(bytes);
        }
        Ok(HostStager { buffer, bytes })
    }

    pub fn h2d_async(dst: &mut Tensor, src: &HostStager) -> Result<()> {
        let stream = dst.device().cuda_stream_raw_ptr();
        memcpy_async_host_to_device(
            dst.cuda_ptr_mut() as *mut c_void,
            src.buffer.as_ptr() as *const c_void,
            src.bytes,
            stream,
        )
        .map_err(|e| Error::Device(format!("h2d_async failed: {e}")))
    }
    pub fn d2h_async(src: &Tensor, dst: &mut HostStager) -> Result<()> {
        let stream = src.device().cuda_stream_raw_ptr();
        memcpy_async_device_to_host(
            dst.buffer.as_mut_ptr() as *mut c_void,
            src.cuda_ptr() as *const c_void,
            dst.bytes,
            stream,
        )
        .map_err(|e| Error::Device(format!("d2h_async failed: {e}")))
    }
}

/// Dynamic bucketing based on aspect ratio and token length.
pub fn bucket_key(width: usize, height: usize, token_len: usize) -> (u32, u32) {
    // Quantize aspect ratio and token length into coarse buckets
    let ar = (width as f32) / (height as f32 + 1e-6);
    let ar_bucket = if ar < 0.9 {
        0
    } else if ar < 1.1 {
        1
    } else {
        2
    };
    let tok_bucket = if token_len <= 64 {
        0
    } else if token_len <= 128 {
        1
    } else {
        2
    };
    (ar_bucket, tok_bucket)
}
