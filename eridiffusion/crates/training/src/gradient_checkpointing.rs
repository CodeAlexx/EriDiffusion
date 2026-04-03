use core::ffi::c_void;
use std::collections::HashMap;
use std::sync::Arc;

use eridiffusion_core::{DType, Error};
use flame_core::{
    device::CudaStreamRawPtrExt, memcpy_async_device_to_host, memcpy_async_host_to_device,
    PinnedAllocFlags, PinnedHostBuffer, PinnedPool, Tensor,
};

#[derive(Clone, Copy, Debug)]
pub enum CheckpointPolicy {
    /// Keep activations on GPU; optional small LRU cache still applies
    Off,
    /// Keep activations on GPU with LRU (size via ERID_OFFLOAD_FRACTION)
    Gpu,
    /// Offload to pinned host memory with async copies; keep N-most recent on GPU
    CpuOffloaded,
}

struct HostBlob {
    buffer: PinnedHostBuffer<u8>,
    bytes: usize,
}

impl HostBlob {
    #[allow(dead_code)]
    fn ptr(&self) -> *mut c_void {
        self.buffer.as_ptr() as *mut c_void
    }
}

pub struct Record {
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub device_id: i32,
    host: HostBlob,
}

/// Simple activation checkpoint store (thread-unsafe; wrap in Mutex if needed).
pub struct CheckpointStore {
    policy: CheckpointPolicy,
    map: HashMap<u64, Record>, // CPU pinned store
    // Small GPU cache to reduce thrash: stores most recent tensors
    gpu_cache: Vec<(u64, Tensor)>,
    gpu_keep: usize,
    pinned_pool: Arc<PinnedPool>,
}

impl CheckpointStore {
    pub fn new(policy: CheckpointPolicy) -> Self {
        let gpu_keep = std::env::var("ERID_PREFETCH_WINDOW")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(2);
        let cache_mb = std::env::var("ERID_PINNED_CHECKPOINT_CACHE_MB")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(512);
        let cache_bytes = cache_mb.saturating_mul(1024 * 1024);
        let pool = PinnedPool::new(PinnedAllocFlags::WRITE_COMBINED, cache_bytes);
        Self {
            policy,
            map: HashMap::new(),
            gpu_cache: Vec::new(),
            gpu_keep,
            pinned_pool: Arc::new(pool),
        }
    }

    fn bytes_for(&self, t: &Tensor) -> usize {
        t.shape().elem_count() * t.dtype().size_in_bytes()
    }

    /// Save activation under `key`. On success, a pinned host copy is created.
    pub fn save(&mut self, key: u64, t: &Tensor) -> Result<(), Error> {
        // GPU caching always records most recent tensor
        self.push_gpu_cache(key, t.clone_result()?);

        // CPU offload only if policy enabled and over threshold
        if !matches!(self.policy, CheckpointPolicy::CpuOffloaded) {
            return Ok(());
        }
        let bytes = self.bytes_for(t);
        let threshold = std::env::var("ERID_OFFLOAD_THRESHOLD_MB")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .map(|mb| mb * 1024 * 1024)
            .unwrap_or(64 * 1024 * 1024);
        if bytes < threshold {
            return Ok(());
        }

        let stream = t.device().cuda_stream_raw_ptr();
        let mut pinned = self
            .pinned_pool
            .checkout(bytes)
            .map_err(|e| Error::Device(format!("pinned host alloc failed: {e}")))?;
        unsafe {
            pinned.set_len(bytes);
        }
        memcpy_async_device_to_host(
            pinned.as_mut_ptr() as *mut c_void,
            t.cuda_ptr() as *const c_void,
            bytes,
            stream,
        )
        .map_err(|e| Error::Device(format!("checkpoint D2H async failed: {e}")))?;

        let rec = Record {
            shape: t.shape().dims().to_vec(),
            dtype: t.dtype(),
            device_id: t.device().ordinal() as i32,
            host: HostBlob { buffer: pinned, bytes },
        };
        self.map.insert(key, rec);
        Ok(())
    }

    /// Restore activation into `dst` tensor (shape/dtype must match).
    pub fn restore_into(&self, key: u64, dst: &mut Tensor) -> Result<(), Error> {
        let rec =
            self.map.get(&key).ok_or(Error::InvalidInput("checkpoint: missing key".into()))?;
        if dst.shape().dims() != &rec.shape[..] {
            return Err(Error::InvalidShape("checkpoint: shape mismatch".into()));
        }
        if dst.dtype() != rec.dtype {
            return Err(Error::InvalidInput("checkpoint: dtype mismatch".into()));
        }
        let bytes = rec.host.bytes;
        let stream = dst.device().cuda_stream_raw_ptr();
        memcpy_async_host_to_device(
            dst.cuda_ptr_mut() as *mut c_void,
            rec.host.buffer.as_ptr() as *const c_void,
            bytes,
            stream,
        )
        .map_err(|e| Error::Device(format!("checkpoint H2D async failed: {e}")))?;
        Ok(())
    }

    pub fn remove(&mut self, key: u64) -> Option<Record> {
        self.map.remove(&key)
    }
    pub fn clear(&mut self) {
        self.map.clear();
        self.gpu_cache.clear();
    }

    fn push_gpu_cache(&mut self, key: u64, t: Tensor) {
        // Update if existing
        if let Some(pos) = self.gpu_cache.iter().position(|(k, _)| *k == key) {
            self.gpu_cache.remove(pos);
        }
        self.gpu_cache.push((key, t));
        // Trim
        while self.gpu_cache.len() > self.gpu_keep {
            self.gpu_cache.remove(0);
        }
    }
    pub fn get_gpu_cached(&self, key: u64) -> Option<Tensor> {
        self.gpu_cache
            .iter()
            .rev()
            .find(|(k, _)| *k == key)
            .map(|(_, t)| t.clone_result().ok())
            .flatten()
    }
}

/// Helper: decide if a tensor is large enough to offload under current policy.
pub fn should_offload(policy: CheckpointPolicy, t: &Tensor) -> bool {
    match policy {
        CheckpointPolicy::Off => false,
        CheckpointPolicy::Gpu => false,
        CheckpointPolicy::CpuOffloaded => {
            let thr = std::env::var("ERID_OFFLOAD_THRESHOLD_MB")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .map(|mb| mb * 1024 * 1024)
                .unwrap_or(64 * 1024 * 1024);
            t.shape().elem_count() * t.dtype().size_in_bytes() >= thr
        }
    }
}

// (We’ll add tests and exports next.)

/// Example usage pattern:
pub fn maybe_save(store: &mut CheckpointStore, key: u64, tensor: &Tensor) {
    let _ = store.save(key, tensor); // ignore errors for small tensors / disabled policy
}
pub fn maybe_restore(store: &CheckpointStore, key: u64, dst: &mut Tensor) -> Result<(), Error> {
    store.restore_into(key, dst)
}
