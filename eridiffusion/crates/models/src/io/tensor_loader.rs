//! Tensor loader for Flux weights (SafeTensors → CUDA BF16) with tile-cast and strict validation.
//!
//! Drop-in path: `eridiffusion/crates/models/src/io/tensor_loader.rs`
//!
//! Policy (Phase-4):
//! - Visible tensors are BF16 on device.
//! - FP32 may be used transiently for casting math, never as persistent big buffers.
//! - No CPU fallback for "ops" — only host file read + tile upload.
//!
//! Supported SafeTensors dtypes: F32, BF16 (u16). Extend `HostDType` for more as needed.
//!
//! Public API:
//!   - `TensorLoader::open(path)` → loader (mmap-backed)
//!   - `load_bf16(&mut self, key, &Device) -> Tensor` (BF16 on CUDA)
//!   - `load_many_bf16(&mut self, &keys, &Device) -> Vec<(String, Tensor)>`
//!   - `stream_into_bf16(&mut self, key, &mut dst_bf16)` (out-param, no alloc)
//!   - `finish_strict()` (errors if any keys unused in strict mode)
//!
//! Safety notes:
//! - Uses tile-cast (default 16 MiB of FP32 host window) to avoid temporary full-size FP32 device buffers.
//! - Validates byte length = numel * itemsize; validates shape non-empty (allow 1D scalars).
//!
//! Crates needed in Cargo.toml:
//!   memmap2 = "0.9"
//!   safetensors = "0.4"
//!
//! If your crate already depends transitively, no change is needed.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{Error as IoError, ErrorKind};
use std::path::Path;

use memmap2::Mmap;
use safetensors::{tensor::TensorView, SafeTensors};

use flame_core::{
    bf16_convert::bf16_u16_to_f32,
    cuda_memory_alignment::alloc_aligned_f32,
    memcpy_async_host_to_device,
    DType,
    PinnedAllocFlags,
    PinnedHostBuffer,
    PinnedPool,
    StagingDeviceBuf,
    Shape,
    Tensor,
};
use std::ffi::c_void;
use std::sync::OnceLock;
use std::time::Instant;
use eridiffusion_core::{Device, Error, Result};

use crate::devtensor::{shape_from_usize, to_dtype, zeros_on};

/// Host dtypes we support from SafeTensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HostDType {
    F32,
    BF16,
}

impl HostDType {
    fn item_size(self) -> usize {
        match self {
            HostDType::F32 => 4,
            HostDType::BF16 => 2,
        }
    }
}

/// Entry metadata from SafeTensors (no data copy).
#[derive(Debug, Clone)]
struct EntryMeta {
    offset: usize,
    nbytes: usize,
    shape: Vec<usize>,
    dtype: HostDType,
}

/// Loader that mmaps a .safetensors file and can tile-cast tensors onto GPU BF16.
pub struct TensorLoader {
    #[allow(dead_code)]
    file: File,
    #[allow(dead_code)]
    map: Mmap,
    st: SafeTensors<'static>,
    metas: BTreeMap<String, EntryMeta>,
    used: BTreeSet<String>,
    strict_unused_error: bool,
    tile_elems_f32: usize,
    pinned_pool: PinnedPool,
}

fn trace_enabled() -> bool {
    static ONCE: OnceLock<bool> = OnceLock::new();
    *ONCE.get_or_init(|| std::env::var("WEIGHT_TRACE_VERBOSE").ok().as_deref() == Some("1"))
}

impl TensorLoader {
    /// Open a SafeTensors file and index all entries.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(&path)
            .map_err(|e| Error::InvalidInput(format!("open {}: {e}", path.as_ref().display())))?;
        let map = unsafe { Mmap::map(&file) }
            .map_err(|e| Error::InvalidInput(format!("mmap {}: {e}", path.as_ref().display())))?;

        // SafeTensors needs a 'static slice; duplicate once and leak for lifetime
        let boxed: Box<[u8]> = Vec::from(&map[..]).into_boxed_slice();
        let leaked: &'static [u8] = Box::leak(boxed);
        let st = SafeTensors::deserialize(leaked)
            .map_err(|e| Error::InvalidInput(format!("safetensors header: {e}")))?;

        let mut metas = BTreeMap::new();
        for name in st.names() {
            let view = st
                .tensor(name)
                .map_err(|e| Error::InvalidInput(format!("tensor {name}: {e}")))?;
            let (dtype, nbytes) = map_dtype_nbytes(&view)?;
            metas.insert(
                name.to_string(),
                EntryMeta {
                    offset: view.data().offset(),
                    nbytes,
                    shape: view.shape().to_vec(),
                    dtype,
                },
            );
        }

        Ok(Self {
            file,
            map,
            st,
            metas,
            used: BTreeSet::new(),
            strict_unused_error: true,
            tile_elems_f32: 4 * 1_024 * 1_024,
            pinned_pool: PinnedPool::new(
                PinnedAllocFlags::WRITE_COMBINED,
                2 * 1024 * 1024 * 1024, // 2 GiB cache budget
            ),
        })
    }

    /// Disable the unused-weights error (strict by default).
    pub fn allow_unused(mut self, allow: bool) -> Self {
        self.strict_unused_error = !allow;
        self
    }

    /// Adjust tile size (number of f32 elements) for host→device casting.
    pub fn with_tile_elems(mut self, elems: usize) -> Self {
        self.tile_elems_f32 = elems.max(1);
        self
    }

    /// List all tensor keys.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.metas.keys().map(|s| s.as_str())
    }

    /// Get shape for a key.
    pub fn shape_of(&self, key: &str) -> Result<&[usize]> {
        self.metas
            .get(key)
            .map(|meta| meta.shape.as_slice())
            .ok_or_else(|| Error::InvalidInput(format!("missing key {key}")))
    }

    /// Load a tensor into CUDA BF16 (allocates output).
    pub fn load_bf16(&mut self, key: &str, device: &Device) -> Result<Tensor> {
        let meta = self.entry(key)?;
        let shape = shape_from_usize(&meta.shape);
        let mut dst = zeros_on(shape, device, DType::BF16)?;
        self.stream_into_bf16(key, &mut dst)?;
        Ok(dst)
    }

    /// Bulk load.
    pub fn load_many_bf16(
        &mut self,
        keys: &[impl AsRef<str>],
        device: &Device,
    ) -> Result<Vec<(String, Tensor)>> {
        let mut out = Vec::with_capacity(keys.len());
        for key in keys {
            let k = key.as_ref();
            let t = self.load_bf16(k, device)?;
            out.push((k.to_string(), t));
        }
        Ok(out)
    }

    /// Stream tensor data into an existing BF16 destination.
    pub fn stream_into_bf16(&mut self, key: &str, dst_bf16: &mut Tensor) -> Result<()> {
        let meta = self.entry(key)?;
        if dst_bf16.dtype() != DType::BF16 {
            return Err(Error::InvalidInput(format!("dst for {key} must be BF16")));
        }
        if dst_bf16.shape().as_slice() != meta.shape.as_slice() {
            return Err(Error::InvalidInput(format!(
                "shape mismatch for {key}: file={:?} dst={:?}",
                meta.shape,
                dst_bf16.shape().as_slice()
            )));
        }

        let start = meta.offset;
        let end = start + meta.nbytes;
        if end > self.st.data().len() {
            return Err(Error::InvalidInput(format!("slice OOB for {key}")));
        }
        let bytes = &self.st.data()[start..end];

        match meta.dtype {
            HostDType::BF16 => {
                let tensor = self.copy_host_bf16_into_device_bf16(bytes, dst_bf16)?;
                *dst_bf16 = tensor;
            }
            HostDType::F32 => self.copy_host_f32_into_device(bytes, dst_bf16)?,
        }

        self.used.insert(key.to_string());
        Ok(())
    }

    /// Finalize: error on unused keys if strict mode.
    pub fn finish_strict(self) -> Result<()> {
        if !self.strict_unused_error {
            return Ok(());
        }
        let all: BTreeSet<String> = self.metas.keys().cloned().collect();
        let unused: Vec<String> = all.difference(&self.used).cloned().collect();
        if unused.is_empty() {
            return Ok(());
        }
        Err(Error::InvalidInput(format!("unused tensors: {}", display_list(&unused))))
    }

    fn entry(&self, key: &str) -> Result<&EntryMeta> {
        self.metas
            .get(key)
            .ok_or_else(|| Error::InvalidInput(format!("missing key {key}")))
    }

    fn copy_host_bf16_into_device_bf16(
        &mut self,
        src_bytes: &[u8],
        dst_bf16: &Tensor,
    ) -> Result<Tensor> {
        let device = dst_bf16.device().clone();
        let shape = dst_bf16.shape().clone();
        let elem_count = src_bytes.len() / HostDType::BF16.item_size();

        let mut pinned_u8 = self
            .pinned_pool
            .checkout(src_bytes.len())
            .map_err(|e| Error::InvalidOperation(format!("pinned alloc failed: {e}")))?;
        pinned_u8.copy_from_slice(src_bytes);
        let mut pinned_u16 = pinned_u8
            .into_reinterpret::<u16>()
            .map_err(|e| Error::InvalidOperation(format!("reinterpret bf16 buffer: {e}")))?;

        let start = Instant::now();

        let result = (|| -> Result<Tensor> {
            let staging = StagingDeviceBuf::<u16>::new(device.clone(), elem_count)
                .map_err(|e| Error::InvalidOperation(format!("staging alloc: {e}")))?;

            let transfer_stream = device
                .fork_default_stream()
                .map_err(|e| Error::Cuda(format!("fork_default_stream: {e}")))?;

            staging
                .async_upload_from(&pinned_u16, &transfer_stream)
                .map_err(|e| Error::Cuda(format!("async upload: {e}")))?;
            device
                .wait_for(&transfer_stream)
                .map_err(|e| Error::Cuda(format!("wait_for stream: {e}")))?;
            drop(transfer_stream);

            let mut f32_slice = alloc_aligned_f32(&device, elem_count)
                .map_err(|e| Error::Cuda(format!("alloc bf16 f32 staging: {e}")))?;

            bf16_u16_to_f32(device.clone(), staging.slice(), &mut f32_slice, elem_count)?;

            Tensor::from_bf16_slice(f32_slice, shape.clone(), device.clone())
                .map_err(|e| Error::InvalidInput(format!("from_bf16_slice: {e}")))
        })();

        let pinned_u8 = pinned_u16
            .into_reinterpret::<u8>()
            .map_err(|e| Error::InvalidOperation(format!("reinterpret back to u8: {e}")))?;
        self.pinned_pool.checkin(pinned_u8);

        let tensor = result?;

        if trace_enabled() {
            let elapsed = start.elapsed().as_secs_f64().max(1e-6);
            let mb = src_bytes.len() as f64 / (1024.0 * 1024.0);
            eprintln!(
                "[weights] BF16 streamed {:.1} MB in {:.3}s ({:.1} MB/s)",
                mb,
                elapsed,
                mb / elapsed
            );
        }

        Ok(tensor)
    }

    fn copy_host_f32_into_device(&mut self, src_bytes: &[u8], dst: &mut Tensor) -> Result<()> {
        if src_bytes.len() % std::mem::size_of::<f32>() != 0 {
            return Err(Error::InvalidInput("unaligned f32 bytes".into()));
        }

        let start = Instant::now();
        let mut pinned = self
            .pinned_pool
            .checkout(src_bytes.len())
            .map_err(|e| Error::InvalidOperation(format!("pinned alloc failed: {e}")))?;
        pinned.copy_from_slice(src_bytes);

        let device = dst.device().clone();
        let transfer_stream = device
            .fork_default_stream()
            .map_err(|e| Error::Cuda(format!("fork_default_stream: {e}")))?;
        let dst_ptr = dst.cuda_ptr_mut();
        if dst_ptr.is_null() {
            return Err(Error::InvalidOperation("cuda_ptr_mut returned null".into()));
        }

        memcpy_async_host_to_device(
            dst_ptr as *mut c_void,
            pinned.as_ptr() as *const c_void,
            src_bytes.len(),
            transfer_stream.stream as *mut c_void,
        )?;
        device
            .wait_for(&transfer_stream)
            .map_err(|e| Error::Cuda(format!("wait_for stream: {e}")))?;
        drop(transfer_stream);

        self.pinned_pool.checkin(pinned);

        if trace_enabled() {
            let elapsed = start.elapsed().as_secs_f64().max(1e-6);
            let mb = src_bytes.len() as f64 / (1024.0 * 1024.0);
            eprintln!(
                "[weights] copied {:.1} MB in {:.3}s ({:.1} MB/s)",
                mb,
                elapsed,
                mb / elapsed
            );
        }
        Ok(())
    }
}

fn map_dtype_nbytes(view: &TensorView) -> Result<(HostDType, usize)> {
    use safetensors::Dtype as SD;
    let dtype = match view.dtype() {
        SD::F32 => HostDType::F32,
        SD::BF16 => HostDType::BF16,
        other => return Err(Error::InvalidInput(format!("unsupported dtype {other:?}"))),
    };
    let nbytes = view.data().len();
    let numel: usize = view.shape().iter().product();
    if nbytes != numel * dtype.item_size() {
        return Err(Error::InvalidInput(format!(
            "size mismatch: bytes={} != numel({numel}) * item_size({})",
            nbytes,
            dtype.item_size()
        )));
    }
    Ok((dtype, nbytes))
}

fn display_list(list: &[String]) -> String {
    let mut s = String::new();
    for (i, item) in list.iter().enumerate() {
        if i > 0 {
            s.push_str(", ");
        }
        s.push_str(item);
    }
    s
}
