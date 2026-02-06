//! SDXL block runtime scaffolding (Phase-4).
//! Currently captures per-block tensors from the strict loader; math will be
//! filled in future passes. Forward path returns `anyhow::Error` to highlight
//! unfinished integration.

use anyhow::{anyhow, bail, ensure, Context, Result};
#[cfg(feature = "bf16_u16")]
use flame_core::bf16_ops::gelu_bf16;
use flame_core::debug_device::assert_cuda;
use flame_core::kernels::adaln::{adaln_modulate_bf16_inplace, layernorm_affine_bf16_inplace};
#[cfg(feature = "bf16_u16")]
use flame_core::ops::gemm_bf16::bmm_bf16_fp32acc_out;
#[cfg(feature = "bf16_u16")]
use flame_core::sdpa;
use flame_core::tensor_ext::to_owning_fp32_strong;
use flame_core::{CudaDevice, DType, Shape, Tensor};
use eridiffusion_core::{cuda::cuda_available_memory, Device as CoreDevice};
use std::cell::Cell;
#[cfg(feature = "bf16_u16")]
use std::cell::RefCell;
#[cfg(feature = "bf16_u16")]
use std::collections::HashMap;
use std::fmt::Write as _;
use std::env;
use std::sync::{Arc, OnceLock};
use std::thread_local;
use std::time::Instant;

use cudarc::driver::result;
#[cfg(feature = "bf16_u16")]
use cudarc::driver::{sys, LaunchAsync, LaunchConfig};
#[cfg(feature = "bf16_u16")]
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

const MAX_FP32_ACC_BYTES: usize = 4 * 1024 * 1024 * 1024; // effectively disables chunking for 64--128px latents
const MAX_FP32_LOGITS_BYTES: usize = 4 * 1024 * 1024 * 1024;
const HARD_MAX_Q_TOKENS: usize = 1024;
const HARD_MAX_KV_TOKENS: usize = 2048;
const CHUNK_SMALL_SEQ_GUARD: usize = 2048;
#[cfg(feature = "bf16_u16")]
const FUSED_ATTENTION_MODULE: &str = "sdxl_fused_attn";
#[cfg(feature = "bf16_u16")]
const FUSED_ATTENTION_FUNC: &str = "sdxl_fused_attn_bf16";
#[cfg(feature = "bf16_u16")]
const FUSED_FFN_MODULE: &str = "sdxl_fused_ffn";
#[cfg(feature = "bf16_u16")]
const FUSED_FFN_FUNC: &str = "sdxl_fused_ffn_bf16";
#[cfg(feature = "bf16_u16")]
const FUSED_FFN_TILE: usize = 256;

#[cfg(feature = "bf16_u16")]
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct WorkspaceKey {
    device: usize,
    dtype: DType,
    dims: Vec<usize>,
}
#[cfg(feature = "bf16_u16")]
impl WorkspaceKey {
    fn new(device: &Arc<CudaDevice>, dtype: DType, dims: &[usize]) -> Self {
        Self { device: device.ordinal(), dtype, dims: dims.to_vec() }
    }
}
#[cfg(feature = "bf16_u16")]
#[derive(Default)]
struct WorkspacePool {
    free: HashMap<WorkspaceKey, Vec<Tensor>>,
}
#[cfg(feature = "bf16_u16")]
impl WorkspacePool {
    fn acquire(
        &mut self,
        key: &WorkspaceKey,
        device: &Arc<CudaDevice>,
        dtype: DType,
        dims: &[usize],
    ) -> Result<Tensor> {
        if let Some(buffers) = self.free.get_mut(key) {
            if let Some(tensor) = buffers.pop() {
                if tensor.shape().dims() != dims {
                    return Ok(Tensor::zeros_dtype(Shape::from_dims(dims), dtype, device.clone())?);
                }
                return Ok(tensor);
            }
        }
        Ok(Tensor::zeros_dtype(Shape::from_dims(dims), dtype, device.clone())?)
    }

    fn release(&mut self, key: WorkspaceKey, tensor: Tensor) {
        self.free.entry(key).or_default().push(tensor);
    }
}
#[cfg(feature = "bf16_u16")]
struct WorkspaceHandle {
    key: WorkspaceKey,
    tensor: Option<Tensor>,
}
#[cfg(feature = "bf16_u16")]
impl WorkspaceHandle {
    fn new(device: &Arc<CudaDevice>, dtype: DType, dims: &[usize]) -> Result<Self> {
        let key = WorkspaceKey::new(device, dtype, dims);
        let tensor = WORKSPACE_POOL.with(|pool| {
            let mut pool = pool.borrow_mut();
            pool.acquire(&key, device, dtype, dims)
        })?;
        Ok(Self { key, tensor: Some(tensor) })
    }

    fn tensor(&self) -> &Tensor {
        self.tensor.as_ref().expect("workspace tensor missing")
    }

    fn tensor_mut(&mut self) -> &mut Tensor {
        self.tensor.as_mut().expect("workspace tensor missing")
    }

    fn into_inner(mut self) -> Tensor {
        self.tensor.take().expect("workspace tensor missing")
    }
}
#[cfg(feature = "bf16_u16")]
impl Drop for WorkspaceHandle {
    fn drop(&mut self) {
        if let Some(tensor) = self.tensor.take() {
            WORKSPACE_POOL.with(|pool| {
                pool.borrow_mut().release(self.key.clone(), tensor);
            });
        }
    }
}
#[cfg(feature = "bf16_u16")]
thread_local! {
    static WORKSPACE_POOL: RefCell<WorkspacePool> = RefCell::new(WorkspacePool::default());
}

#[cfg(feature = "bf16_u16")]
fn acquire_workspace_tensor(
    device: &Arc<CudaDevice>,
    dtype: DType,
    dims: &[usize],
) -> Result<WorkspaceHandle> {
    WorkspaceHandle::new(device, dtype, dims)
}

#[cfg(feature = "bf16_u16")]
const FUSED_ATTENTION_KERNEL: &str = r#"
#include <cuda_bf16.h>
#include <cuda_runtime.h>

extern "C" __global__ void sdxl_fused_attn_bf16(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ out,
    int head_dim,
    int seq_q,
    int seq_k,
    float scale
) {
    extern __shared__ float shared[];
    float* q_shared = shared;
    float* out_shared = q_shared + head_dim;
    float* partial = out_shared + head_dim;
    float* ctrl = partial + blockDim.x;
    // ctrl[0] = max, ctrl[1] = sum, ctrl[2] = exp_old, ctrl[3] = exp_val, ctrl[4] = inv_sum

    const int bh = blockIdx.x;
    const int q_idx = blockIdx.y;
    const int tid = threadIdx.x;

    if (tid < head_dim) {
        out_shared[tid] = 0.0f;
    }
    if (tid < blockDim.x) {
        partial[tid] = 0.0f;
    }
    if (tid == 0) {
        ctrl[0] = -1.0e20f;
        ctrl[1] = 0.0f;
    }
    __syncthreads();

    const size_t head_offset = static_cast<size_t>(bh) * seq_q + q_idx;
    const __nv_bfloat16* q_ptr = q + head_offset * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        q_shared[d] = __bfloat162float(q_ptr[d]);
    }
    __syncthreads();

    for (int kv = 0; kv < seq_k; ++kv) {
        const size_t kv_offset = static_cast<size_t>(bh) * seq_k + kv;
        const __nv_bfloat16* k_ptr = k + kv_offset * head_dim;
        float thread_sum = 0.0f;
        for (int d = tid; d < head_dim; d += blockDim.x) {
            thread_sum += q_shared[d] * __bfloat162float(k_ptr[d]);
        }
        partial[tid] = thread_sum;
        __syncthreads();

        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                partial[tid] += partial[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0) {
            const float prev_max = ctrl[0];
            const float prev_sum = ctrl[1];
            const float logits = partial[0] * scale;
            const float new_max = fmaxf(prev_max, logits);
            const float exp_old = (prev_sum == 0.0f) ? 0.0f : __expf(prev_max - new_max);
            const float exp_val = __expf(logits - new_max);
            const float new_sum = prev_sum * exp_old + exp_val;
            ctrl[0] = new_max;
            ctrl[1] = new_sum;
            ctrl[2] = exp_old;
            ctrl[3] = exp_val;
        }
        __syncthreads();

        const float exp_old = ctrl[2];
        const float exp_val = ctrl[3];
        const __nv_bfloat16* v_ptr = v + kv_offset * head_dim;
        for (int d = tid; d < head_dim; d += blockDim.x) {
            float acc = out_shared[d];
            acc = acc * exp_old + exp_val * __bfloat162float(v_ptr[d]);
            out_shared[d] = acc;
        }
        __syncthreads();
    }

    if (tid == 0) {
        const float denom = fmaxf(ctrl[1], 1.0e-6f);
        ctrl[4] = 1.0f / denom;
    }
    __syncthreads();

    const float inv_sum = ctrl[4];
    __nv_bfloat16* out_ptr = out + head_offset * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float val = out_shared[d] * inv_sum;
        out_ptr[d] = __float2bfloat16_rn(val);
    }
}
"#;

#[cfg(feature = "bf16_u16")]
const FUSED_FFN_KERNEL: &str = r#"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_functions.h>

extern "C" __global__ void sdxl_fused_ffn_bf16(
    const __nv_bfloat16* __restrict__ proj,
    const __nv_bfloat16* __restrict__ proj_bias,
    const __nv_bfloat16* __restrict__ out_weight,
    const __nv_bfloat16* __restrict__ out_bias,
    __nv_bfloat16* __restrict__ out,
    int rows,
    int hidden,
    int act_dim
) {
    constexpr int TILE = 256;
    const int row = blockIdx.x;
    const int out_idx = blockIdx.y * blockDim.x + threadIdx.x;

    if (row >= rows) {
        return;
    }

    float acc = 0.0f;
    if (out_idx < hidden) {
        acc = __bfloat162float(out_bias[out_idx]);
    }

    extern __shared__ float shared[];
    float* gated = shared;

    const size_t row_offset = static_cast<size_t>(row) * static_cast<size_t>(act_dim) * 2;
    const __nv_bfloat16* proj_row = proj + row_offset;

    for (int tile = 0; tile < act_dim; tile += TILE) {
        for (int k = threadIdx.x; k < TILE; k += blockDim.x) {
            const int idx = tile + k;
            float value = 0.0f;
            if (idx < act_dim) {
                const float act = __bfloat162float(proj_row[idx]) + __bfloat162float(proj_bias[idx]);
                const float gate = __bfloat162float(proj_row[idx + act_dim])
                    + __bfloat162float(proj_bias[idx + act_dim]);
                const float x = act;
                const float x3 = x * x * x;
                const float gelu = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x3)));
                value = gelu * gate;
            }
            gated[k] = value;
        }
        __syncthreads();

        if (out_idx < hidden) {
            const __nv_bfloat16* weight_row = out_weight
                + static_cast<size_t>(out_idx) * static_cast<size_t>(act_dim)
                + tile;
            for (int k = 0; k < TILE; ++k) {
                const int idx = tile + k;
                if (idx >= act_dim) {
                    break;
                }
                const float w = __bfloat162float(weight_row[k]);
                acc += w * gated[k];
            }
        }
        __syncthreads();
    }

    if (out_idx < hidden) {
        out[static_cast<size_t>(row) * static_cast<size_t>(hidden) + out_idx] =
            __float2bfloat16_rn(acc);
    }
}
"#;

#[inline]
fn trace_verbose() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("SDXL_TRACE_VERBOSE").ok().as_deref() == Some("1"))
}

#[inline]
fn expect_bf16(tag: &str, tensor: &Tensor) -> Result<()> {
    ensure!(tensor.dtype() == DType::BF16, "{tag} expects BF16 tensor, got {:?}", tensor.dtype());
    Ok(())
}

fn timing_enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("SDXL_TIMING").ok().as_deref() == Some("1"))
}

fn chunking_enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("SDXL_ENABLE_CHUNK").ok().as_deref() == Some("1"))
}

fn fused_attention_enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("SDXL_FUSED_ATTENTION").ok().as_deref() == Some("1"))
}

fn fused_ffn_enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("SDXL_FUSED_FFN").ok().as_deref() == Some("1"))
}

fn kernel_debug_enabled() -> bool {
    if let Some(override_value) = TELEMETRY_OVERRIDE.with(|cell| cell.get()) {
        return override_value;
    }
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("SDXL_KERNEL_TELEMETRY").ok().as_deref() == Some("1"))
}

#[cfg(feature = "bf16_u16")]
fn attention_block_dim(head_dim: usize) -> u32 {
    let mut block = head_dim.next_power_of_two();
    if block < 32 {
        block = 32;
    }
    if block > 256 {
        block = 256;
    }
    block as u32
}

#[derive(Clone, Copy, Debug, Default)]
pub struct AttnChunkConfig {
    pub q_chunk: usize,
    pub kv_chunk: usize,
}

#[derive(Clone, Copy, Debug, Default)]
struct AttnRuntimeConfig {
    reuse: bool,
    max_workspace_mb: usize,
    min_free_mb: usize,
}

thread_local! {
    static ATTENTION_CHUNK: Cell<AttnChunkConfig> = Cell::new(AttnChunkConfig::default());
    static TELEMETRY_OVERRIDE: Cell<Option<bool>> = Cell::new(None);
    static ATTENTION_RUNTIME: Cell<Option<AttnRuntimeConfig>> = Cell::new(None);
}

pub(crate) fn with_attn_chunks<R, F: FnOnce() -> R>(chunk: AttnChunkConfig, f: F) -> R {
    ATTENTION_CHUNK.with(|cell| {
        let prev = cell.replace(chunk);
        let result = f();
        cell.set(prev);
        result
    })
}

pub(crate) fn with_kernel_telemetry<R, F: FnOnce() -> R>(enabled: bool, f: F) -> R {
    TELEMETRY_OVERRIDE.with(|cell| {
        let prev = cell.replace(Some(enabled));
        let result = f();
        cell.set(prev);
        result
    })
}

pub(crate) fn current_attn_chunks() -> AttnChunkConfig {
    ATTENTION_CHUNK.with(|cell| cell.get())
}

fn with_attn_runtime<R, F: FnOnce() -> R>(cfg: AttnRuntimeConfig, f: F) -> R {
    ATTENTION_RUNTIME.with(|cell| {
        let prev = cell.replace(Some(cfg));
        let result = f();
        cell.set(prev);
        result
    })
}

fn current_attn_runtime() -> Option<AttnRuntimeConfig> {
    ATTENTION_RUNTIME.with(|cell| cell.get())
}

pub(crate) fn attn_env_from_env() -> Option<AttnEnvOverrides> {
    let chunk_q = env::var("ATTN_Q_CHUNK").ok().and_then(|v| v.parse::<usize>().ok());
    let chunk_kv = env::var("ATTN_KV_CHUNK").ok().and_then(|v| v.parse::<usize>().ok());
    let chunk_shared = env::var("ATTN_CHUNK_SIZE").ok().and_then(|v| v.parse::<usize>().ok());

    let reuse_raw = env::var("ATTN_CHUNK_REUSE").ok();
    let reuse = reuse_raw
        .as_deref()
        .map(|v| matches!(v.trim(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(true);

    let max_ws = env::var("ATTN_MAX_WORKSPACE_MB").ok().and_then(|v| v.parse::<usize>().ok());
    let min_free = env::var("ATTN_MIN_FREE_MB").ok().and_then(|v| v.parse::<usize>().ok());

    let chunk_q_final = chunk_q.or(chunk_shared).unwrap_or(0);
    let chunk_kv_final = chunk_kv.or(chunk_shared).unwrap_or(chunk_q_final);

    let have_env = chunk_q.is_some()
        || chunk_kv.is_some()
        || chunk_shared.is_some()
        || reuse_raw.is_some()
        || max_ws.is_some()
        || min_free.is_some();

    if !have_env {
        return None;
    }

    Some(AttnEnvOverrides {
        chunk: AttnChunkConfig { q_chunk: chunk_q_final, kv_chunk: chunk_kv_final },
        reuse,
        max_workspace_mb: max_ws.unwrap_or(0),
        min_free_mb: min_free.unwrap_or(0),
    })
}

fn effective_chunk_len(seq: usize, requested: usize) -> usize {
    if seq == 0 {
        0
    } else if requested == 0 {
        seq
    } else {
        requested.min(seq).max(1)
    }
}

fn clamp_q_chunk(bh: usize, head_dim: usize, requested: usize) -> usize {
    let bytes_per_token = bh.saturating_mul(head_dim).saturating_mul(4);
    if bytes_per_token == 0 {
        return requested.max(1);
    }
    let max_tokens = (MAX_FP32_ACC_BYTES / bytes_per_token).max(1);
    requested.min(max_tokens).min(HARD_MAX_Q_TOKENS).max(1)
}

fn clamp_kv_chunk(bh: usize, q_len: usize, requested: usize) -> usize {
    let bytes_per_col = bh.saturating_mul(q_len).saturating_mul(4);
    if bytes_per_col == 0 {
        return requested.max(1);
    }
    let max_cols = (MAX_FP32_LOGITS_BYTES / bytes_per_col).max(1);
    requested.min(max_cols).min(HARD_MAX_KV_TOKENS).max(1)
}

fn matmul_f32_batched_rowmajor(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    ensure!(a.dtype() == DType::F32, "matmul expects F32 lhs");
    ensure!(b.dtype() == DType::F32, "matmul expects F32 rhs");

    let a_dims = a.shape().dims().to_vec();
    let b_dims = b.shape().dims().to_vec();
    ensure!(a_dims.len() == 3, "matmul expects lhs [batch, m, k], got {:?}", a_dims);
    ensure!(b_dims.len() == 3, "matmul expects rhs [batch, k, n], got {:?}", b_dims);

    let batch = a_dims[0];
    let m = a_dims[1];
    let k = a_dims[2];
    ensure!(
        b_dims[0] == batch && b_dims[1] == k,
        "matmul shape mismatch lhs {:?}, rhs {:?}",
        a_dims,
        b_dims
    );
    let a_ready = a.reshape(&[batch, m, k])?.clone_result()?;
    let b_ready = b.reshape(&[batch, k, b_dims[2]])?.clone_result()?;

    a_ready.bmm(&b_ready).map_err(Into::into)
}

#[cfg(feature = "bf16_u16")]
fn matmul_bf16_inputs_to_f32(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    ensure!(a.dtype() == DType::BF16, "lhs must be BF16");
    ensure!(b.dtype() == DType::BF16, "rhs must be BF16");

    let a_dims = a.shape().dims().to_vec();
    let b_dims = b.shape().dims().to_vec();
    ensure!(a_dims.len() == 3 && b_dims.len() == 3, "matmul expects 3D tensors");

    let batch = a_dims[0];
    let m = a_dims[1];
    let k = a_dims[2];
    ensure!(b_dims[0] == batch && b_dims[1] == k, "matmul shape mismatch");
    let n = b_dims[2];

    eprintln!(
        "[bf16-inventory] matmul_bf16_inputs_to_f32 widening output to F32 (batch={batch} m={m} n={n} k={k})"
    );

    let device = a.device().clone();
    let mut out_ws = acquire_workspace_tensor(&device, DType::BF16, &[batch, m, n])?;
    {
        let out_tensor = out_ws.tensor_mut();
        bmm_bf16_fp32acc_out(a, b, out_tensor, false, false)?;
    }
    out_ws.tensor().to_dtype(DType::F32).map_err(Into::into)
}

#[cfg(not(feature = "bf16_u16"))]
#[allow(dead_code)]
fn matmul_bf16_inputs_to_f32(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    eprintln!(
        "[bf16-inventory] matmul_bf16_inputs_to_f32 widening inputs to F32 (no bf16_u16 feature)"
    );
    let a32 = to_owning_fp32_strong(a)?;
    let b32 = to_owning_fp32_strong(b)?;
    matmul_f32_batched_rowmajor(&a32, &b32)
}

use super::keymap::SdxlKeyMap;
use crate::streaming::KeyMapOwned;
use crate::tensor_utils::{broadcast_add, broadcast_mul, broadcast_to_as};

const COND_DIM: usize = 1280;
const LN_EPS: f32 = 1e-5;

fn mem_trace_enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("SDXL_MEM_TRACE").ok().as_deref() == Some("1"))
}

pub(crate) fn mem_snap(tag: &str) {
    if !mem_trace_enabled() {
        return;
    }

    if let Ok((free, total)) = result::mem_get_info() {
        let mib = 1024 * 1024;
        let used = total.saturating_sub(free);
        let mut line = String::with_capacity(96);
        let _ = write!(
            &mut line,
            "[mem] {tag}: used={} MiB free={} MiB total={} MiB",
            used / mib,
            free / mib,
            total / mib
        );
        eprintln!("{line}");
    }
}

#[cfg(feature = "bf16_u16")]
fn ensure_fused_attention_kernel(device: &Arc<CudaDevice>) -> Result<()> {
    if device.get_func(FUSED_ATTENTION_MODULE, FUSED_ATTENTION_FUNC).is_some() {
        return Ok(());
    }

    let mut opts = CompileOptions::default();
    if let Ok(extra) = std::env::var("CUDARC_NVRTC_EXTRA_INCLUDE_PATHS") {
        for path in extra.split(':').filter(|p| !p.is_empty()) {
            opts.include_paths.push(path.to_string());
        }
    }
    let mut ensured_default = false;
    if let Ok(include_dir) = std::env::var("CUDA_INCLUDE_DIR") {
        if !opts.include_paths.iter().any(|p| p == &include_dir) {
            opts.include_paths.push(include_dir.clone());
        }
        ensured_default = true;
    } else if let Ok(cuda_home) = std::env::var("CUDA_HOME") {
        let path = format!("{cuda_home}/include");
        if !opts.include_paths.iter().any(|p| p == &path) {
            opts.include_paths.push(path);
        }
        ensured_default = true;
    } else if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
        let path = format!("{cuda_path}/include");
        if !opts.include_paths.iter().any(|p| p == &path) {
            opts.include_paths.push(path);
        }
        ensured_default = true;
    }
    if !ensured_default && !opts.include_paths.iter().any(|p| p == "/usr/local/cuda/include") {
        opts.include_paths.push("/usr/local/cuda/include".into());
    }

    let ptx = compile_ptx_with_opts(FUSED_ATTENTION_KERNEL, opts)
        .map_err(|e| anyhow!("failed to compile fused attention PTX: {e}"))?;

    device
        .load_ptx(ptx, FUSED_ATTENTION_MODULE, &[FUSED_ATTENTION_FUNC])
        .map_err(|e| anyhow!("failed to load fused attention PTX: {e}"))?;
    Ok(())
}

#[cfg(feature = "bf16_u16")]
fn fused_attention_forward(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> Result<Tensor> {
    ensure!(q.dtype() == DType::BF16, "fused attention expects BF16 Q");
    ensure!(k.dtype() == DType::BF16, "fused attention expects BF16 K");
    ensure!(v.dtype() == DType::BF16, "fused attention expects BF16 V");

    let q_dims = q.shape().dims().to_vec();
    let k_dims = k.shape().dims().to_vec();
    let v_dims = v.shape().dims().to_vec();
    ensure!(q_dims.len() == 3, "expected Q [BH,SEQ_Q,D], got {:?}", q_dims);
    ensure!(k_dims.len() == 3, "expected K [BH,SEQ_K,D], got {:?}", k_dims);
    ensure!(v_dims.len() == 3, "expected V [BH,SEQ_K,D], got {:?}", v_dims);
    ensure!(q_dims[0] == k_dims[0] && q_dims[0] == v_dims[0], "batch mismatch");
    ensure!(k_dims[1] == v_dims[1], "seq_k mismatch");
    ensure!(q_dims[2] == k_dims[2] && q_dims[2] == v_dims[2], "head dim mismatch");

    let bh = q_dims[0];
    let seq_q = q_dims[1];
    let seq_k = k_dims[1];
    let head_dim = q_dims[2];

    let device = q.device().clone();
    ensure_fused_attention_kernel(&device)?;

    let q_ptr = q.as_device_ptr_bf16("fused_attention.q").map_err(|e| anyhow!("{e}"))? as usize
        as sys::CUdeviceptr;
    let k_ptr = k.as_device_ptr_bf16("fused_attention.k").map_err(|e| anyhow!("{e}"))? as usize
        as sys::CUdeviceptr;
    let v_ptr = v.as_device_ptr_bf16("fused_attention.v").map_err(|e| anyhow!("{e}"))? as usize
        as sys::CUdeviceptr;

    let mut out_ws = acquire_workspace_tensor(&device, DType::BF16, &[bh, seq_q, head_dim])?;
    let out_ptr = out_ws
        .tensor_mut()
        .as_mut_device_ptr_bf16("fused_attention.out")
        .map_err(|e| anyhow!("{e}"))? as usize as sys::CUdeviceptr;

    let block_dim = attention_block_dim(head_dim);
    let shared_bytes = (head_dim * 2 + block_dim as usize + 5) * std::mem::size_of::<f32>();
    let cfg = LaunchConfig {
        grid_dim: (bh as u32, seq_q as u32, 1),
        block_dim: (block_dim, 1, 1),
        shared_mem_bytes: shared_bytes as u32,
    };

    let func = device
        .get_func(FUSED_ATTENTION_MODULE, FUSED_ATTENTION_FUNC)
        .context("fused attention kernel missing after load")?;

    unsafe {
        func.launch(
            cfg,
            (q_ptr, k_ptr, v_ptr, out_ptr, head_dim as i32, seq_q as i32, seq_k as i32, scale),
        )
        .map_err(|e| anyhow!("launch fused attention kernel failed: {e}"))?;
    }

    Ok(out_ws.into_inner())
}

#[cfg(not(feature = "bf16_u16"))]
fn fused_attention_forward(_q: &Tensor, _k: &Tensor, _v: &Tensor, _scale: f32) -> Result<Tensor> {
    Err(anyhow!("fused BF16 attention requires the bf16_u16 feature flag to be enabled"))
}

#[cfg(feature = "bf16_u16")]
fn fused_ffn_forward(
    proj: &Tensor,
    proj_bias: &Tensor,
    out_weight: &Tensor,
    out_bias: &Tensor,
) -> Result<Tensor> {
    ensure!(proj.dtype() == DType::BF16, "fused FFN expects BF16 proj input");
    ensure!(proj_bias.dtype() == DType::BF16, "fused FFN expects BF16 proj bias");
    ensure!(out_weight.dtype() == DType::BF16, "fused FFN expects BF16 out weight");
    ensure!(out_bias.dtype() == DType::BF16, "fused FFN expects BF16 out bias");

    let dims = proj.shape().dims().to_vec();
    ensure!(dims.len() == 2, "fused FFN expects [rows, 2*act] proj, got {:?}", dims);
    let rows = dims[0] as usize;
    let twice_act = dims[1] as usize;
    ensure!(twice_act % 2 == 0, "fused FFN expects even projection dim, got {}", twice_act);
    let act_dim = twice_act / 2;

    let bias_dims = proj_bias.shape().dims().to_vec();
    ensure!(
        bias_dims.len() == 1 && bias_dims[0] as usize == twice_act,
        "proj bias dim {} != 2*act {}",
        bias_dims.get(0).copied().unwrap_or_default(),
        twice_act
    );

    let out_weight_dims = out_weight.shape().dims().to_vec();
    ensure!(
        out_weight_dims.len() == 2 && out_weight_dims[1] as usize == act_dim,
        "out weight dims {:?} incompatible with act_dim {}",
        out_weight_dims,
        act_dim
    );
    let hidden = out_weight_dims[0] as usize;

    let out_bias_dims = out_bias.shape().dims().to_vec();
    ensure!(
        out_bias_dims.len() == 1 && out_bias_dims[0] as usize == hidden,
        "out bias dim {} != hidden {}",
        out_bias_dims.get(0).copied().unwrap_or_default(),
        hidden
    );

    ensure!(
        proj.device().ordinal() == proj_bias.device().ordinal()
            && proj.device().ordinal() == out_weight.device().ordinal()
            && proj.device().ordinal() == out_bias.device().ordinal(),
        "fused FFN tensors must reside on the same CUDA device"
    );

    let device = proj.device().clone();
    ensure_fused_ffn_kernel(&device)?;

    let proj_ptr = proj.as_device_ptr_bf16("fused_ffn.proj").map_err(|e| anyhow!("{e}"))? as usize
        as sys::CUdeviceptr;
    let proj_bias_ptr = proj_bias
        .as_device_ptr_bf16("fused_ffn.proj_bias")
        .map_err(|e| anyhow!("{e}"))? as usize as sys::CUdeviceptr;
    let out_weight_ptr = out_weight
        .as_device_ptr_bf16("fused_ffn.out_weight")
        .map_err(|e| anyhow!("{e}"))? as usize as sys::CUdeviceptr;
    let out_bias_ptr = out_bias
        .as_device_ptr_bf16("fused_ffn.out_bias")
        .map_err(|e| anyhow!("{e}"))? as usize as sys::CUdeviceptr;

    let mut out_ws = acquire_workspace_tensor(&device, DType::BF16, &[rows, hidden])?;
    let out_ptr =
        out_ws.tensor_mut().as_mut_device_ptr_bf16("fused_ffn.out").map_err(|e| anyhow!("{e}"))?
            as usize as sys::CUdeviceptr;

    let threads = 128u32;
    let grid_x = rows as u32;
    let grid_y = ((hidden + threads as usize - 1) / threads as usize) as u32;
    let shared_bytes = (FUSED_FFN_TILE * std::mem::size_of::<f32>()) as u32;

    let func = device
        .get_func(FUSED_FFN_MODULE, FUSED_FFN_FUNC)
        .context("fused FFN kernel missing after load")?;

    unsafe {
        func.launch(
            LaunchConfig {
                grid_dim: (grid_x, grid_y, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: shared_bytes,
            },
            (
                proj_ptr,
                proj_bias_ptr,
                out_weight_ptr,
                out_bias_ptr,
                out_ptr,
                rows as i32,
                hidden as i32,
                act_dim as i32,
            ),
        )
        .map_err(|e| anyhow!("launch fused FFN kernel failed: {e}"))?;
    }

    Ok(out_ws.into_inner())
}

#[cfg(not(feature = "bf16_u16"))]
#[allow(unused_variables)]
fn fused_ffn_forward(
    proj: &Tensor,
    proj_bias: &Tensor,
    out_weight: &Tensor,
    out_bias: &Tensor,
) -> Result<Tensor> {
    Err(anyhow!("fused BF16 FFN requires the bf16_u16 feature flag to be enabled"))
}

#[cfg(feature = "bf16_u16")]
fn ensure_fused_ffn_kernel(device: &Arc<CudaDevice>) -> Result<()> {
    if device.get_func(FUSED_FFN_MODULE, FUSED_FFN_FUNC).is_some() {
        return Ok(());
    }

    let mut opts = CompileOptions::default();
    if let Ok(extra) = std::env::var("CUDARC_NVRTC_EXTRA_INCLUDE_PATHS") {
        for path in extra.split(':').filter(|p| !p.is_empty()) {
            opts.include_paths.push(path.to_string());
        }
    }
    let mut ensured_default = false;
    if let Ok(include_dir) = std::env::var("CUDA_INCLUDE_DIR") {
        if !opts.include_paths.iter().any(|p| p == &include_dir) {
            opts.include_paths.push(include_dir.clone());
        }
        ensured_default = true;
    } else if let Ok(cuda_home) = std::env::var("CUDA_HOME") {
        let path = format!("{cuda_home}/include");
        if !opts.include_paths.iter().any(|p| p == &path) {
            opts.include_paths.push(path);
        }
        ensured_default = true;
    } else if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
        let path = format!("{cuda_path}/include");
        if !opts.include_paths.iter().any(|p| p == &path) {
            opts.include_paths.push(path);
        }
        ensured_default = true;
    }
    if !ensured_default && !opts.include_paths.iter().any(|p| p == "/usr/local/cuda/include") {
        opts.include_paths.push("/usr/local/cuda/include".into());
    }

    let ptx = compile_ptx_with_opts(FUSED_FFN_KERNEL, opts)
        .map_err(|e| anyhow!("failed to compile fused FFN PTX: {e}"))?;

    device
        .load_ptx(ptx, FUSED_FFN_MODULE, &[FUSED_FFN_FUNC])
        .map_err(|e| anyhow!("failed to load fused FFN PTX: {e}"))?;
    Ok(())
}

#[cfg(not(feature = "bf16_u16"))]
#[allow(unused_variables)]
fn ensure_fused_ffn_kernel(_device: &Arc<CudaDevice>) -> Result<()> {
    Err(anyhow!("fused BF16 FFN requires the bf16_u16 feature flag to be enabled"))
}

fn record_kernel_stats(tag: &str, tensor: &Tensor, start: Option<Instant>) -> Result<()> {
    if !kernel_debug_enabled() {
        return Ok(());
    }

    let tensor_f32 = tensor.to_dtype(DType::F32)?;
    let sum = tensor_f32.sum_all()?.to_scalar::<f32>()?;
    let max_abs = tensor_f32.abs()?.max_all()?;

    if !sum.is_finite() || !max_abs.is_finite() {
        bail!("{tag}: detected non-finite values (sum={sum}, max|x|={max_abs})");
    }

    if let Some(start) = start {
        let elapsed_ms = start.elapsed().as_secs_f64() * 1_000.0;
        eprintln!("[telemetry] {tag} time={:.2}ms max|x|={:.5}", elapsed_ms, max_abs);
    } else {
        eprintln!("[telemetry] {tag} max|x|={:.5}", max_abs);
    }
    Ok(())
}

/// Minimal executable block abstraction for SDXL UNet.
pub trait ExecutableBlock: Send + Sync {
    fn name(&self) -> &str;
    fn hidden_dim(&self) -> usize;
    fn context_dim(&self) -> usize;
    fn forward_with_cond(
        &self,
        sample: &Tensor,
        encoder_hidden_states: &Tensor,
        driver_1280: &Tensor,
        time_proj_1536: Option<&Tensor>,
    ) -> Result<Tensor>;
}

/// Simple BF16 linear (weight only) for attention projections (no bias in SDXL checkpoints).
#[derive(Debug)]
pub struct LinearNoBias {
    pub weight: Tensor,   // [out_dim, in_dim] as stored
    pub weight_t: Tensor, // [in_dim, out_dim] cached transpose
}

/// Linear with bias (used for out projections / FFN).
#[derive(Debug)]
pub struct LinearWithBias {
    pub weight: Tensor,   // [out_dim, in_dim]
    pub weight_t: Tensor, // [in_dim, out_dim]
    pub bias: Tensor,     // [out_dim]
}

/// Norm parameters (scale/bias).
#[derive(Debug)]
pub struct NormParams {
    pub weight: Tensor,
    pub bias: Tensor,
}

#[derive(Debug)]
pub struct AttentionParams {
    pub q: LinearNoBias,
    pub k: LinearNoBias,
    pub v: LinearNoBias,
    pub out: LinearWithBias,
}

#[derive(Debug)]
pub struct FeedForwardParams {
    pub proj: LinearWithBias,
    pub out: LinearWithBias,
}

#[derive(Debug)]
pub struct AdaModParams {
    pub weight: Tensor,
    pub bias: Tensor,
    pub cond_dim: usize,
    pub hidden: usize,
}

impl AdaModParams {
    fn new(device: &flame_core::Device, cond_dim: usize, hidden: usize) -> Result<Self> {
        let cuda = device.cuda_device_arc();
        let weight = Tensor::zeros_dtype(
            Shape::from_dims(&[cond_dim, 3 * hidden]),
            DType::BF16,
            cuda.clone(),
        )?;
        let bias = Tensor::zeros_dtype(Shape::from_dims(&[3 * hidden]), DType::BF16, cuda)?;
        Ok(Self { weight, bias, cond_dim, hidden })
    }
}

fn ensure_linear_2d(t: &Tensor, name: &str) -> Result<()> {
    if t.shape().rank() != 2 {
        return Err(anyhow!("{name} must be 2D, got {:?}", t.shape().dims()));
    }
    Ok(())
}

fn ensure_norm_1d(t: &Tensor, name: &str) -> Result<()> {
    if t.shape().rank() != 1 {
        return Err(anyhow!("{name} must be 1D, got {:?}", t.shape().dims()));
    }
    Ok(())
}

fn ensure_bf16_tensor(tensor: Tensor, name: &str) -> Result<Tensor> {
    if tensor.dtype() == DType::BF16 {
        Ok(tensor)
    } else {
        tensor.to_dtype(DType::BF16).map_err(|e| anyhow!("failed to cast {name} to BF16: {e}"))
    }
}

fn ensure_linear_with_bias_dims(weight: &Tensor, bias: &Tensor, name: &str) -> Result<()> {
    ensure_linear_2d(weight, &format!("{name}.weight"))?;
    ensure_norm_1d(bias, &format!("{name}.bias"))?;
    let out_dim = weight.shape().dims()[0];
    if bias.shape().dims()[0] != out_dim {
        return Err(anyhow!(
            "{name} bias dim {} != weight out dim {}",
            bias.shape().dims()[0],
            out_dim
        ));
    }
    Ok(())
}

fn linear_nb_from(mut weight: Tensor, name: &str) -> Result<LinearNoBias> {
    ensure_linear_2d(&weight, &format!("{name}.weight"))?;
    if weight.dtype() != DType::BF16 {
        weight = weight.to_dtype(DType::BF16)?;
    }
    // Transpose via temporary F32 to keep BF16 storage while avoiding precision loss.
    let weight_f32 = weight.to_dtype(DType::F32)?;
    let weight_t = weight_f32.transpose()?.to_dtype(DType::BF16)?;
    Ok(LinearNoBias { weight, weight_t })
}

fn combine_linear_no_bias(parts: &[&LinearNoBias], tag: &str) -> Result<LinearNoBias> {
    ensure!(!parts.is_empty(), "{tag}: expected at least one linear component");
    let weight_refs: Vec<&Tensor> = parts.iter().map(|p| &p.weight).collect();
    let weight = ensure_bf16_tensor(Tensor::cat(&weight_refs, 0)?, &format!("{tag}.weight"))?;

    let wt_refs: Vec<&Tensor> = parts.iter().map(|p| &p.weight_t).collect();
    let weight_t = ensure_bf16_tensor(Tensor::cat(&wt_refs, 1)?, &format!("{tag}.weight_t"))?;

    Ok(LinearNoBias { weight, weight_t })
}

fn linear_wb_from(weight: Tensor, bias: Tensor, name: &str) -> Result<LinearWithBias> {
    ensure_linear_with_bias_dims(&weight, &bias, name)?;
    let weight =
        if weight.dtype() == DType::BF16 { weight } else { weight.to_dtype(DType::BF16)? };
    let weight_f32 = weight.to_dtype(DType::F32)?;
    let weight_t = weight_f32.transpose()?.to_dtype(DType::BF16)?;
    let bias = if bias.dtype() == DType::BF16 { bias } else { bias.to_dtype(DType::BF16)? };
    Ok(LinearWithBias { weight, weight_t, bias })
}

/// Runtime block with validated tensor layout (math wired in next pass).
pub struct SdxlBlockRuntime {
    name: String,
    pub block_norm: NormParams,
    pub proj_in: LinearWithBias,
    pub norm1: NormParams,
    pub norm2: NormParams,
    pub norm3: NormParams,
    pub attn1: AttentionParams,
    pub attn2: AttentionParams,
    pub attn1_qkv: LinearNoBias,
    pub attn2_kv: LinearNoBias,
    pub ff: FeedForwardParams,
    pub proj_out: LinearWithBias,
    pub mod1: AdaModParams,
    pub mod2: AdaModParams,
    pub mod3: AdaModParams,
    hidden: usize,
    context: usize,
    heads: usize,
    head_dim: usize,
    pub raw_keys: Vec<String>,
    attn_env: Option<AttnEnvOverrides>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct AttnEnvOverrides {
    chunk: AttnChunkConfig,
    reuse: bool,
    max_workspace_mb: usize,
    min_free_mb: usize,
}

impl AttnEnvOverrides {
    fn runtime_config(&self) -> AttnRuntimeConfig {
        AttnRuntimeConfig {
            reuse: self.reuse,
            max_workspace_mb: self.max_workspace_mb,
            min_free_mb: self.min_free_mb,
        }
    }

    pub(crate) fn apply_to(&self, block: &mut SdxlBlockRuntime) {
        let chunk_size = self.chunk.q_chunk.max(self.chunk.kv_chunk);
        block.apply_attn_env(chunk_size, self.reuse, self.max_workspace_mb, self.min_free_mb);
    }
}

impl SdxlBlockRuntime {
    /// Build a block runtime from the strict loader output.
    pub fn from_mmap(block_index: usize, tensors: Vec<Tensor>) -> Result<Self> {
        let keys = SdxlKeyMap::gen_keys_for_block(block_index);
        if keys.len() != tensors.len() {
            return Err(anyhow!(
                "SDXL block {} expected {} tensors, got {}",
                block_index,
                keys.len(),
                tensors.len()
            ));
        }
        let mut iter = tensors.into_iter();

        macro_rules! take {
            () => {
                iter.next().ok_or_else(|| anyhow!("insufficient tensors for SDXL block"))?
            };
        }

        let block_norm = NormParams {
            weight: ensure_bf16_tensor(take!(), "block_norm.weight")?,
            bias: ensure_bf16_tensor(take!(), "block_norm.bias")?,
        };
        let proj_in = linear_wb_from(take!(), take!(), "proj_in")?;
        let norm1 = NormParams {
            weight: ensure_bf16_tensor(take!(), "norm1.weight")?,
            bias: ensure_bf16_tensor(take!(), "norm1.bias")?,
        };
        let attn1 = AttentionParams {
            q: linear_nb_from(take!(), "attn1.to_q")?,
            k: linear_nb_from(take!(), "attn1.to_k")?,
            v: linear_nb_from(take!(), "attn1.to_v")?,
            out: linear_wb_from(take!(), take!(), "attn1.to_out.0")?,
        };
        let norm2 = NormParams {
            weight: ensure_bf16_tensor(take!(), "norm2.weight")?,
            bias: ensure_bf16_tensor(take!(), "norm2.bias")?,
        };
        let attn2 = AttentionParams {
            q: linear_nb_from(take!(), "attn2.to_q")?,
            k: linear_nb_from(take!(), "attn2.to_k")?,
            v: linear_nb_from(take!(), "attn2.to_v")?,
            out: linear_wb_from(take!(), take!(), "attn2.to_out.0")?,
        };
        let attn1_qkv = combine_linear_no_bias(&[&attn1.q, &attn1.k, &attn1.v], "attn1.qkv")?;
        let attn2_kv = combine_linear_no_bias(&[&attn2.k, &attn2.v], "attn2.kv")?;
        let norm3 = NormParams {
            weight: ensure_bf16_tensor(take!(), "norm3.weight")?,
            bias: ensure_bf16_tensor(take!(), "norm3.bias")?,
        };
        let ff = FeedForwardParams {
            proj: linear_wb_from(take!(), take!(), "ff.net.0.proj")?,
            out: linear_wb_from(take!(), take!(), "ff.net.2")?,
        };
        let proj_out = linear_wb_from(take!(), take!(), "proj_out")?;

        if let Some(unexpected) = iter.next() {
            drop(unexpected);
            return Err(anyhow!("internal error: SDXL block tensor count mismatch"));
        }

        // Shape validation (all tensors BF16, correct ranks)
        ensure_norm_1d(&norm1.weight, "norm1.weight")?;
        ensure_norm_1d(&norm1.bias, "norm1.bias")?;
        ensure_norm_1d(&norm2.weight, "norm2.weight")?;
        ensure_norm_1d(&norm2.bias, "norm2.bias")?;
        ensure_norm_1d(&norm3.weight, "norm3.weight")?;
        ensure_norm_1d(&norm3.bias, "norm3.bias")?;

        ensure_linear_2d(&attn1.q.weight, "attn1.to_q.weight")?;
        ensure_linear_2d(&attn1.k.weight, "attn1.to_k.weight")?;
        ensure_linear_2d(&attn1.v.weight, "attn1.to_v.weight")?;
        ensure_linear_with_bias_dims(&attn1.out.weight, &attn1.out.bias, "attn1.to_out.0")?;
        ensure_linear_2d(&attn2.q.weight, "attn2.to_q.weight")?;
        ensure_linear_2d(&attn2.k.weight, "attn2.to_k.weight")?;
        ensure_linear_2d(&attn2.v.weight, "attn2.to_v.weight")?;
        ensure_linear_with_bias_dims(&attn2.out.weight, &attn2.out.bias, "attn2.to_out.0")?;
        ensure_linear_with_bias_dims(&ff.proj.weight, &ff.proj.bias, "ff.net.0.proj")?;
        ensure_linear_with_bias_dims(&ff.out.weight, &ff.out.bias, "ff.net.2")?;

        // Hidden dimension from attn1 input size (weight layout [out,in]).
        let hidden = attn1.q.weight.shape().dims()[1];
        let context = attn2.k.weight.shape().dims()[1];

        let all_bf16 = [
            ("block_norm.weight", &block_norm.weight),
            ("block_norm.bias", &block_norm.bias),
            ("proj_in.weight", &proj_in.weight),
            ("proj_in.bias", &proj_in.bias),
            ("norm1.weight", &norm1.weight),
            ("norm1.bias", &norm1.bias),
            ("norm2.weight", &norm2.weight),
            ("norm2.bias", &norm2.bias),
            ("norm3.weight", &norm3.weight),
            ("norm3.bias", &norm3.bias),
            ("attn1.q.weight", &attn1.q.weight),
            ("attn1.k.weight", &attn1.k.weight),
            ("attn1.v.weight", &attn1.v.weight),
            ("attn1.out.weight", &attn1.out.weight),
            ("attn1.out.bias", &attn1.out.bias),
            ("attn2.q.weight", &attn2.q.weight),
            ("attn2.k.weight", &attn2.k.weight),
            ("attn2.v.weight", &attn2.v.weight),
            ("attn2.out.weight", &attn2.out.weight),
            ("attn2.out.bias", &attn2.out.bias),
            ("attn1_qkv.weight", &attn1_qkv.weight),
            ("attn2_kv.weight", &attn2_kv.weight),
            ("ff.proj.weight", &ff.proj.weight),
            ("ff.proj.bias", &ff.proj.bias),
            ("ff.out.weight", &ff.out.weight),
            ("ff.out.bias", &ff.out.bias),
            ("proj_out.weight", &proj_out.weight),
            ("proj_out.bias", &proj_out.bias),
        ];
        if let Some((name, t)) = all_bf16.iter().find(|(_, t)| t.dtype() != DType::BF16) {
            return Err(anyhow!(
                "SDXL block tensor {name} dtype must be BF16, got {:?}",
                t.dtype()
            ));
        }
        if hidden == 0 || context == 0 {
            return Err(anyhow!("invalid hidden/context dims for SDXL block"));
        }

        let device = flame_core::Device::from(attn1.q.weight.device().clone());
        let mod1 = AdaModParams::new(&device, COND_DIM, hidden)?;
        let mod2 = AdaModParams::new(&device, COND_DIM, hidden)?;
        let mod3 = AdaModParams::new(&device, COND_DIM, hidden)?;

        ensure!(hidden % 64 == 0, "hidden {} not multiple of 64 (expected SDXL head dim)", hidden);
        let heads = hidden / 64;
        let head_dim = 64;

        if trace_verbose() {
            eprintln!(
                "[block_load] idx={} hidden={} context={} attn1_q={:?} norm1_len={} norm2_len={} norm3_len={} ff_proj={:?} ff_out={:?}",
                block_index,
                hidden,
                context,
                attn1.q.weight.shape().dims(),
                block_norm.weight.shape().dims()[0],
                norm2.weight.shape().dims()[0],
                norm3.weight.shape().dims()[0],
                ff.proj.weight.shape().dims(),
                ff.out.weight.shape().dims()
            );
        }

        Ok(Self {
            name: format!("block_{:02}", block_index),
            block_norm,
            proj_in,
            norm1,
            norm2,
            norm3,
            attn1,
            attn2,
            attn1_qkv,
            attn2_kv,
            ff,
            proj_out,
            mod1,
            mod2,
            mod3,
            hidden,
            context,
            heads,
            head_dim,
            raw_keys: keys,
            attn_env: None,
        })
    }

    fn enforce_attn_env(
        &self,
        env: &AttnEnvOverrides,
        sample: &Tensor,
        ctx: &Tensor,
    ) -> Result<()> {
        const MB_BYTES: usize = 1_048_576;
        if !env.reuse {
            bail!(
                "SDXL streaming attention requires ATTN_CHUNK_REUSE=1 (set ATTN_CHUNK_REUSE=1)."
            );
        }

        if env.max_workspace_mb > 0 {
            let dims = sample.shape().dims().to_vec();
            ensure!(
                dims.len() == 4,
                "SDXL block expects NHWC sample tensor, got {:?}",
                dims
            );
            let ctx_dims = ctx.shape().dims().to_vec();
            ensure!(
                ctx_dims.len() == 3,
                "SDXL block expects [B,T,C] context tensor, got {:?}",
                ctx_dims
            );

            let b = dims[0] as usize;
            let h = dims[1] as usize;
            let w = dims[2] as usize;
            let seq_q = h.saturating_mul(w).max(1);
            let seq_k = ctx_dims[1] as usize;
            let bh = b.saturating_mul(self.heads).max(1);
            let q_len = env.chunk.q_chunk.max(1).min(seq_q);
            let kv_len = env.chunk.kv_chunk.max(1).min(seq_k);
            let workspace_bytes = bh
                .saturating_mul(q_len)
                .saturating_mul(kv_len)
                .saturating_mul(4);
            let workspace_mb = (workspace_bytes + MB_BYTES - 1) / MB_BYTES;
            ensure!(
                workspace_mb <= env.max_workspace_mb,
                "SDXL streaming attention chunk (q_len={} kv_len={}) would require ~{} MB of workspace (> {} MB cap). Lower ATTN_CHUNK_SIZE or raise ATTN_MAX_WORKSPACE_MB.",
                q_len,
                kv_len,
                workspace_mb,
                env.max_workspace_mb
            );
        }

        if env.min_free_mb > 0 {
            let core_device = CoreDevice::from_flame_cuda(sample.device().as_ref());
            let free_bytes = cuda_available_memory(&core_device)
                .context("sdxl streaming attention: query available GPU memory")?;
            let free_mb = free_bytes / MB_BYTES;
            ensure!(
                free_mb >= env.min_free_mb,
                "SDXL streaming attention requires at least {} MB of free GPU memory (found {} MB). Reduce ATTN_CHUNK_SIZE or free additional memory.",
                env.min_free_mb,
                free_mb
            );
        }

        Ok(())
    }

    pub fn forward_inference(
        &self,
        sample: &Tensor,
        encoder_hidden_states: &Tensor,
        driver_1280: &Tensor,
        time_proj_1536: Option<&Tensor>,
    ) -> Result<Tensor> {
        let _ = time_proj_1536;
        if let Some(env) = self.attn_env {
            self.enforce_attn_env(&env, sample, encoder_hidden_states)?;
            let runtime_cfg = env.runtime_config();
            with_attn_runtime(runtime_cfg, || {
                with_attn_chunks(env.chunk, || {
                    forward_block_inference(self, sample, encoder_hidden_states, driver_1280)
                })
            })
        } else {
            forward_block_inference(self, sample, encoder_hidden_states, driver_1280)
        }
    }

    pub fn apply_attn_env(
        &mut self,
        chunk_size: usize,
        reuse: bool,
        max_ws_mb: usize,
        min_free_mb: usize,
    ) {
        let chunk = AttnChunkConfig { q_chunk: chunk_size, kv_chunk: chunk_size };
        self.attn_env = Some(AttnEnvOverrides {
            chunk,
            reuse,
            max_workspace_mb: max_ws_mb,
            min_free_mb,
        });
    }
}

impl ExecutableBlock for SdxlBlockRuntime {
    fn name(&self) -> &str {
        &self.name
    }

    fn hidden_dim(&self) -> usize {
        self.hidden
    }

    fn context_dim(&self) -> usize {
        self.context
    }

    fn forward_with_cond(
        &self,
        sample: &Tensor,
        encoder_hidden_states: &Tensor,
        driver_1280: &Tensor,
        time_proj_1536: Option<&Tensor>,
    ) -> Result<Tensor> {
        let _ = time_proj_1536; // reserved for future modulation
        if let Some(env) = self.attn_env {
            self.enforce_attn_env(&env, sample, encoder_hidden_states)?;
            let runtime_cfg = env.runtime_config();
            with_attn_runtime(runtime_cfg, || {
                with_attn_chunks(env.chunk, || {
                    forward_block(self, sample, encoder_hidden_states, driver_1280)
                })
            })
        } else {
            forward_block(self, sample, encoder_hidden_states, driver_1280)
        }
    }
}

impl SdxlKeyMap {
    pub fn owned_attn1_to_q(block: usize) -> String {
        Self::gen_keys_for_block(block)
            .into_iter()
            .find(|k| k.contains("attn1.to_q"))
            .expect("SDXL attn1.to_q key")
    }
}

fn forward_block(
    block: &SdxlBlockRuntime,
    sample: &Tensor,
    ctx: &Tensor,
    cond: &Tensor,
) -> Result<Tensor> {
    let block_timing = timing_enabled();
    let block_start = if block_timing { Some(Instant::now()) } else { None };

    debug_assert_eq!(sample.dtype(), DType::BF16, "forward_block expects BF16 sample");
    let dims_raw = sample.shape().dims();
    if trace_verbose() {
        eprintln!("[forward_block] {} hidden {} sample {:?}", block.name, block.hidden, dims_raw);
    }
    if trace_verbose() {
        eprintln!("[forward_block] sample dims {:?}", dims_raw);
    }
    if mem_trace_enabled() {
        let tag = format!("block {}:entry", block.name);
        mem_snap(&tag);
    }
    ensure!(dims_raw.len() == 4, "sample must be NHWC, got {:?}", dims_raw);
    let (b, h, w, c) =
        (dims_raw[0] as usize, dims_raw[1] as usize, dims_raw[2] as usize, dims_raw[3] as usize);
    ensure!(c == block.hidden, "SDXL block expects hidden dim {}, got {}", block.hidden, c);

    let ctx_dims = ctx.shape().dims();
    ensure!(ctx_dims.len() == 3, "ctx must be [B,S,C] got {:?}", ctx_dims);
    let ctx_b = ctx_dims[0] as usize;
    let ctx_c = ctx_dims[2] as usize;
    ensure!(ctx_c == block.context, "ctx dim {} != expected {}", ctx_c, block.context);
    ensure!(ctx_b == b, "batch mismatch {} vs {}", ctx_b, b);

    let tokens = sample.reshape(&[b, h * w, c])?;
    let ctx =
        if ctx.dtype() == DType::BF16 { ctx.clone_result()? } else { ctx.to_dtype(DType::BF16)? };
    let cond = if cond.dtype() == DType::BF16 {
        cond.clone_result()?
    } else {
        cond.to_dtype(DType::BF16)?
    };

    if trace_verbose() {
        eprintln!(
            "proj_in: weight={:?} bias={:?} input={:?}",
            block.proj_in.weight.shape().dims(),
            block.proj_in.bias.shape().dims(),
            tokens.shape().dims()
        );
    }
    let proj_in_start = Instant::now();
    let proj_in = linear_with_bias(&tokens, &block.proj_in)?; // BF16
    let proj_in_ms =
        if block_timing { proj_in_start.elapsed().as_secs_f64() * 1_000.0 } else { 0.0 };
    expect_bf16("proj_in", &proj_in)?;
    let ln_block_start = Instant::now();
    let mut x = layer_norm_bf16(&proj_in, &block.block_norm, h, w)?; // BF16
    let ln_block_ms =
        if block_timing { ln_block_start.elapsed().as_secs_f64() * 1_000.0 } else { 0.0 };
    expect_bf16("block_norm", &x)?;

    if trace_verbose() {
        eprintln!("[block] before mod1 x dtype {:?}", x.dtype());
    }
    let mod1_start = Instant::now();
    let (ln1, gate1) = adaln_modulate(&x, &block.norm1, &block.mod1, &cond, h, w)?;
    let mod1_ms = if block_timing { mod1_start.elapsed().as_secs_f64() * 1_000.0 } else { 0.0 };
    if mem_trace_enabled() {
        let tag = format!("block {}:adaln1_post", block.name);
        mem_snap(&tag);
    }
    if trace_verbose() {
        eprintln!(
            "[adaln] ln1 {:?} {:?} gate1 {:?} {:?}",
            ln1.dtype(),
            ln1.shape().dims(),
            gate1.dtype(),
            gate1.shape().dims()
        );
    }
    if mem_trace_enabled() {
        let tag = format!("block {}:self_attn_pre", block.name);
        mem_snap(&tag);
    }
    let self_attn_start = Instant::now();
    let self_attn = self_attention(block, &ln1)?;
    let self_attn_ms =
        if block_timing { self_attn_start.elapsed().as_secs_f64() * 1_000.0 } else { 0.0 };
    if mem_trace_enabled() {
        let tag = format!("block {}:self_attn_post", block.name);
        mem_snap(&tag);
    }
    let gated = broadcast_mul(&self_attn, &gate1)?;
    expect_bf16("self_attn_gate", &gated)?;
    x = broadcast_add(&x, &gated)?;
    expect_bf16("residual1", &x)?;

    if trace_verbose() {
        eprintln!("[block] before mod2 x dtype {:?}", x.dtype());
    }
    let mod2_start = Instant::now();
    let (ln2, gate2) = adaln_modulate(&x, &block.norm2, &block.mod2, &cond, h, w)?;
    let mod2_ms = if block_timing { mod2_start.elapsed().as_secs_f64() * 1_000.0 } else { 0.0 };
    if mem_trace_enabled() {
        let tag = format!("block {}:adaln2_post", block.name);
        mem_snap(&tag);
    }
    let cross_start = Instant::now();
    let cross = cross_attention(block, &ln2, &ctx)?;
    let cross_ms = if block_timing { cross_start.elapsed().as_secs_f64() * 1_000.0 } else { 0.0 };
    if mem_trace_enabled() {
        let tag = format!("block {}:cross_attn_post", block.name);
        mem_snap(&tag);
    }
    let cross_gated = broadcast_mul(&cross, &gate2)?;
    expect_bf16("cross_attn_gate", &cross_gated)?;
    x = broadcast_add(&x, &cross_gated)?;
    expect_bf16("residual2", &x)?;

    if trace_verbose() {
        eprintln!("[block] before mod3 x dtype {:?}", x.dtype());
    }
    let mod3_start = Instant::now();
    let (ln3, gate3) = adaln_modulate(&x, &block.norm3, &block.mod3, &cond, h, w)?;
    let mod3_ms = if block_timing { mod3_start.elapsed().as_secs_f64() * 1_000.0 } else { 0.0 };
    if mem_trace_enabled() {
        let tag = format!("block {}:adaln3_post", block.name);
        mem_snap(&tag);
    }
    let ff_start = Instant::now();
    let ff = feed_forward(block, &ln3)?;
    let ff_ms = if block_timing { ff_start.elapsed().as_secs_f64() * 1_000.0 } else { 0.0 };
    if mem_trace_enabled() {
        let tag = format!("block {}:ff_post", block.name);
        mem_snap(&tag);
    }
    let ff_gated = broadcast_mul(&ff, &gate3)?;
    expect_bf16("ff_gate", &ff_gated)?;
    x = broadcast_add(&x, &ff_gated)?;
    expect_bf16("residual3", &x)?;

    let proj_out_start = Instant::now();
    let proj = linear_with_bias(&x, &block.proj_out)?;
    let tokens_out = broadcast_add(&proj_in, &proj)?;
    expect_bf16("proj_out_residual", &tokens_out)?;
    let result = tokens_out.reshape(&[b, h, w, block.hidden])?;
    debug_assert_eq!(result.dtype(), DType::BF16);
    debug_assert_eq!(result.storage_dtype(), DType::BF16);
    if mem_trace_enabled() {
        let tag = format!("block {}:exit", block.name);
        mem_snap(&tag);
    }
    if let Some(start) = block_start {
        let total_ms = start.elapsed().as_secs_f64() * 1_000.0;
        let proj_out_ms =
            if block_timing { proj_out_start.elapsed().as_secs_f64() * 1_000.0 } else { 0.0 };
        eprintln!(
            "[timing] block {} proj_in={:.2}ms ln_block={:.2}ms mod1={:.2}ms self_attn={:.2}ms mod2={:.2}ms cross_attn={:.2}ms mod3={:.2}ms feed_forward={:.2}ms proj_out={:.2}ms total={:.2}ms",
            block.name,
            proj_in_ms,
            ln_block_ms,
            mod1_ms,
            self_attn_ms,
            mod2_ms,
            cross_ms,
            mod3_ms,
            ff_ms,
            proj_out_ms,
            total_ms
        );
    }
    Ok(result)
}

fn forward_block_inference(
    block: &SdxlBlockRuntime,
    sample: &Tensor,
    ctx: &Tensor,
    cond: &Tensor,
) -> Result<Tensor> {
    let block_timing = timing_enabled();
    let block_start = if block_timing { Some(Instant::now()) } else { None };

    debug_assert_eq!(sample.dtype(), DType::BF16, "forward_block expects BF16 sample");
    let dims_raw = sample.shape().dims();
    if trace_verbose() {
        eprintln!(
            "[forward_block_infer] {} hidden {} sample {:?}",
            block.name, block.hidden, dims_raw
        );
    }
    if mem_trace_enabled() {
        let tag = format!("block {}:entry", block.name);
        mem_snap(&tag);
    }
    ensure!(dims_raw.len() == 4, "sample must be NHWC, got {:?}", dims_raw);
    let (b, h, w, c) =
        (dims_raw[0] as usize, dims_raw[1] as usize, dims_raw[2] as usize, dims_raw[3] as usize);
    ensure!(c == block.hidden, "SDXL block expects hidden dim {}, got {}", block.hidden, c);

    let ctx_dims = ctx.shape().dims();
    ensure!(ctx_dims.len() == 3, "ctx must be [B,S,C] got {:?}", ctx_dims);
    let ctx_b = ctx_dims[0] as usize;
    let ctx_c = ctx_dims[2] as usize;
    ensure!(ctx_c == block.context, "ctx dim {} != expected {}", ctx_c, block.context);
    ensure!(ctx_b == b, "batch mismatch {} vs {}", ctx_b, b);

    let tokens = sample.reshape(&[b, h * w, c])?;
    let ctx =
        if ctx.dtype() == DType::BF16 { ctx.clone_result()? } else { ctx.to_dtype(DType::BF16)? };
    let cond = if cond.dtype() == DType::BF16 {
        cond.clone_result()?
    } else {
        cond.to_dtype(DType::BF16)?
    };

    let proj_in_start = Instant::now();
    let proj_in = linear_with_bias(&tokens, &block.proj_in)?;
    let proj_in_ms =
        if block_timing { proj_in_start.elapsed().as_secs_f64() * 1_000.0 } else { 0.0 };
    expect_bf16("proj_in", &proj_in)?;
    let ln_block_start = Instant::now();
    let mut x = layer_norm_bf16(&proj_in, &block.block_norm, h, w)?;
    let ln_block_ms =
        if block_timing { ln_block_start.elapsed().as_secs_f64() * 1_000.0 } else { 0.0 };
    expect_bf16("block_norm", &x)?;

    let mod1_start = Instant::now();
    let (ln1, gate1) = adaln_modulate(&x, &block.norm1, &block.mod1, &cond, h, w)?;
    let mod1_ms = if block_timing { mod1_start.elapsed().as_secs_f64() * 1_000.0 } else { 0.0 };
    if mem_trace_enabled() {
        let tag = format!("block {}:adaln1_post", block.name);
        mem_snap(&tag);
    }
    let self_attn_start = Instant::now();
    let self_attn = self_attention_inference(block, &ln1)?;
    let self_attn_ms =
        if block_timing { self_attn_start.elapsed().as_secs_f64() * 1_000.0 } else { 0.0 };
    if mem_trace_enabled() {
        let tag = format!("block {}:self_attn_post", block.name);
        mem_snap(&tag);
    }
    let gated = broadcast_mul(&self_attn, &gate1)?;
    expect_bf16("self_attn_gate", &gated)?;
    x = broadcast_add(&x, &gated)?;
    expect_bf16("residual1", &x)?;

    let mod2_start = Instant::now();
    let (ln2, gate2) = adaln_modulate(&x, &block.norm2, &block.mod2, &cond, h, w)?;
    let mod2_ms = if block_timing { mod2_start.elapsed().as_secs_f64() * 1_000.0 } else { 0.0 };
    if mem_trace_enabled() {
        let tag = format!("block {}:adaln2_post", block.name);
        mem_snap(&tag);
    }
    let cross_start = Instant::now();
    let cross = cross_attention_inference(block, &ln2, &ctx)?;
    let cross_ms = if block_timing { cross_start.elapsed().as_secs_f64() * 1_000.0 } else { 0.0 };
    if mem_trace_enabled() {
        let tag = format!("block {}:cross_attn_post", block.name);
        mem_snap(&tag);
    }
    let cross_gated = broadcast_mul(&cross, &gate2)?;
    expect_bf16("cross_attn_gate", &cross_gated)?;
    x = broadcast_add(&x, &cross_gated)?;
    expect_bf16("residual2", &x)?;

    let mod3_start = Instant::now();
    let (ln3, gate3) = adaln_modulate(&x, &block.norm3, &block.mod3, &cond, h, w)?;
    let mod3_ms = if block_timing { mod3_start.elapsed().as_secs_f64() * 1_000.0 } else { 0.0 };
    if mem_trace_enabled() {
        let tag = format!("block {}:adaln3_post", block.name);
        mem_snap(&tag);
    }
    let ff_start = Instant::now();
    let ff = feed_forward(block, &ln3)?;
    let ff_ms = if block_timing { ff_start.elapsed().as_secs_f64() * 1_000.0 } else { 0.0 };
    if mem_trace_enabled() {
        let tag = format!("block {}:ff_post", block.name);
        mem_snap(&tag);
    }
    let ff_gated = broadcast_mul(&ff, &gate3)?;
    expect_bf16("ff_gate", &ff_gated)?;
    x = broadcast_add(&x, &ff_gated)?;
    expect_bf16("residual3", &x)?;

    let proj_out_start = Instant::now();
    let proj = linear_with_bias(&x, &block.proj_out)?;
    let tokens_out = broadcast_add(&proj_in, &proj)?;
    expect_bf16("proj_out_residual", &tokens_out)?;
    let result = tokens_out.reshape(&[b, h, w, block.hidden])?;
    debug_assert_eq!(result.dtype(), DType::BF16);
    if mem_trace_enabled() {
        let tag = format!("block {}:exit", block.name);
        mem_snap(&tag);
    }
    if let Some(start) = block_start {
        let total_ms = start.elapsed().as_secs_f64() * 1_000.0;
        let proj_out_ms =
            if block_timing { proj_out_start.elapsed().as_secs_f64() * 1_000.0 } else { 0.0 };
        eprintln!(
            "[timing] block {} proj_in={:.2}ms ln_block={:.2}ms mod1={:.2}ms self_attn={:.2}ms mod2={:.2}ms cross_attn={:.2}ms mod3={:.2}ms feed_forward={:.2}ms proj_out={:.2}ms total={:.2}ms",
            block.name,
            proj_in_ms,
            ln_block_ms,
            mod1_ms,
            self_attn_ms,
            mod2_ms,
            cross_ms,
            mod3_ms,
            ff_ms,
            proj_out_ms,
            total_ms
        );
    }
    Ok(result)
}

fn layer_norm_bf16(x: &Tensor, params: &NormParams, h: usize, w: usize) -> Result<Tensor> {
    if !cfg!(feature = "bf16_kernels") {
        return Err(anyhow!(
            "flame-core built without bf16_kernels; enable the feature for fused layer norm"
        ));
    }
    ensure!(x.dtype() == DType::BF16, "layer_norm_bf16 expects BF16 input, got {:?}", x.dtype());
    let dims = x.shape().dims().to_vec();
    ensure!(dims.len() == 3, "layer_norm_bf16 expects [B, HW, C], got {:?}", dims);
    let b = dims[0] as usize;
    let seq = dims[1] as usize;
    let hidden = dims[2] as usize;
    ensure!(seq == h * w, "layer_norm_bf16 expects seq {} to equal h*w {}", seq, h * w);
    ensure!(
        params.weight.dtype() == DType::BF16,
        "norm weight must be BF16, got {:?}",
        params.weight.dtype()
    );
    ensure!(
        params.bias.dtype() == DType::BF16,
        "norm bias must be BF16, got {:?}",
        params.bias.dtype()
    );

    let mut nhwc = x.reshape(&[b, h, w, hidden])?.clone_result()?;
    layernorm_affine_bf16_inplace(
        &mut nhwc,
        Some(&params.weight),
        Some(&params.bias),
        b as i32,
        h as i32,
        w as i32,
        hidden as i32,
        LN_EPS,
    )
    .map_err(|e| anyhow!("layernorm_affine_bf16_inplace failed: {e}"))?;

    Ok(nhwc.reshape(&[b, seq, hidden])?)
}

fn linear_no_bias(x: &Tensor, lin: &LinearNoBias) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    ensure!(dims.len() >= 2, "linear expects >=2D tensor, got {:?}", dims);
    ensure!(dims.last() == Some(&lin.weight.shape().dims()[1]), "linear in dim mismatch");
    let out_dim = lin.weight.shape().dims()[0];
    let leading: usize = dims[..dims.len() - 1].iter().product();
    let in_dim = *dims.last().unwrap();

    let x_flat = x.reshape(&[leading, in_dim])?;
    let weight_t = &lin.weight_t;

    let y = match (x_flat.dtype(), weight_t.dtype()) {
        (DType::BF16, DType::BF16) => x_flat.matmul_bf16(weight_t)?,
        (DType::F32, DType::F32) => x_flat.matmul(weight_t)?,
        (lhs_dtype, rhs_dtype) => {
            if trace_verbose() {
                eprintln!(
                    "[linear_no_bias] dtype promotion lhs={:?} rhs={:?}",
                    lhs_dtype, rhs_dtype
                );
            }
            let compute = if lhs_dtype == DType::F32 || rhs_dtype == DType::F32 {
                DType::F32
            } else {
                DType::BF16
            };
            let lhs = if x_flat.dtype() == compute {
                x_flat.clone_result()?
            } else {
                x_flat.to_dtype(compute)?
            };
            let rhs = if weight_t.dtype() == compute {
                weight_t.clone_result()?
            } else {
                weight_t.to_dtype(compute)?
            };
            let prod =
                if compute == DType::BF16 { lhs.matmul_bf16(&rhs)? } else { lhs.matmul(&rhs)? };
            if prod.dtype() == x.dtype() {
                prod
            } else {
                prod.to_dtype(x.dtype())?
            }
        }
    };

    let mut new_shape = dims;
    *new_shape.last_mut().unwrap() = out_dim;
    let reshaped = y.reshape(&new_shape)?;
    if reshaped.dtype() == x.dtype() {
        Ok(reshaped)
    } else {
        reshaped.to_dtype(x.dtype()).map_err(Into::into)
    }
}

fn linear_with_bias(x: &Tensor, lin: &LinearWithBias) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    ensure!(dims.len() >= 2, "linear expects >=2D tensor, got {:?}", dims);
    ensure!(dims.last() == Some(&lin.weight.shape().dims()[1]), "linear in dim mismatch");
    let out_dim = lin.weight.shape().dims()[0];
    let leading: usize = dims[..dims.len() - 1].iter().product();
    let in_dim = *dims.last().unwrap();

    let x_flat = x.reshape(&[leading, in_dim])?;
    let weight_t = &lin.weight_t;

    let prod = match (x_flat.dtype(), weight_t.dtype()) {
        (DType::BF16, DType::BF16) => x_flat.matmul_bf16(weight_t)?,
        (DType::F32, DType::F32) => x_flat.matmul(weight_t)?,
        (lhs_dtype, rhs_dtype) => {
            if trace_verbose() {
                eprintln!(
                    "[linear_with_bias] dtype promotion lhs={:?} rhs={:?}",
                    lhs_dtype, rhs_dtype
                );
            }
            let compute = if lhs_dtype == DType::F32 || rhs_dtype == DType::F32 {
                DType::F32
            } else {
                DType::BF16
            };
            let lhs = if x_flat.dtype() == compute {
                x_flat.clone_result()?
            } else {
                x_flat.to_dtype(compute)?
            };
            let rhs = if weight_t.dtype() == compute {
                weight_t.clone_result()?
            } else {
                weight_t.to_dtype(compute)?
            };
            if compute == DType::BF16 {
                lhs.matmul_bf16(&rhs)?
            } else {
                lhs.matmul(&rhs)?
            }
        }
    };

    let bias = if lin.bias.dtype() == prod.dtype() {
        lin.bias.clone_result()?
    } else {
        lin.bias.to_dtype(prod.dtype())?
    };
    let bias = bias.reshape(&[1, out_dim])?;
    let y = broadcast_add(&prod, &bias)?;

    let mut new_shape = dims;
    *new_shape.last_mut().unwrap() = out_dim;
    let reshaped = y.reshape(&new_shape)?;
    if reshaped.dtype() == x.dtype() {
        Ok(reshaped)
    } else {
        reshaped.to_dtype(x.dtype()).map_err(Into::into)
    }
}

fn linear_without_bias(x: &Tensor, lin: &LinearWithBias) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    ensure!(dims.len() >= 2, "linear expects >=2D tensor, got {:?}", dims);
    ensure!(dims.last() == Some(&lin.weight.shape().dims()[1]), "linear in dim mismatch");
    let out_dim = lin.weight.shape().dims()[0];
    let leading: usize = dims[..dims.len() - 1].iter().product();
    let in_dim = *dims.last().unwrap();

    let x_flat = x.reshape(&[leading, in_dim])?;
    let weight_t = &lin.weight_t;

    let prod = match (x_flat.dtype(), weight_t.dtype()) {
        (DType::BF16, DType::BF16) => x_flat.matmul_bf16(weight_t)?,
        (DType::F32, DType::F32) => x_flat.matmul(weight_t)?,
        (lhs_dtype, rhs_dtype) => {
            if trace_verbose() {
                eprintln!(
                    "[linear_without_bias] dtype promotion lhs={:?} rhs={:?}",
                    lhs_dtype, rhs_dtype
                );
            }
            let compute = if lhs_dtype == DType::F32 || rhs_dtype == DType::F32 {
                DType::F32
            } else {
                DType::BF16
            };
            let lhs = if x_flat.dtype() == compute {
                x_flat.clone_result()?
            } else {
                x_flat.to_dtype(compute)?
            };
            let rhs = if weight_t.dtype() == compute {
                weight_t.clone_result()?
            } else {
                weight_t.to_dtype(compute)?
            };
            if compute == DType::BF16 {
                lhs.matmul_bf16(&rhs)?
            } else {
                lhs.matmul(&rhs)?
            }
        }
    };

    let mut new_shape = dims;
    *new_shape.last_mut().unwrap() = out_dim;
    let reshaped = prod.reshape(&new_shape)?;
    if reshaped.dtype() == x.dtype() {
        Ok(reshaped)
    } else {
        reshaped.to_dtype(x.dtype()).map_err(Into::into)
    }
}

fn adaln_modulate(
    x: &Tensor,
    norm: &NormParams,
    params: &AdaModParams,
    cond: &Tensor,
    h: usize,
    w: usize,
) -> Result<(Tensor, Tensor)> {
    let timing = timing_enabled();
    let total_start = if timing { Some(Instant::now()) } else { None };
    if !cfg!(feature = "bf16_kernels") {
        return Err(anyhow!(
            "flame-core built without bf16_kernels; enable the feature for fused AdaLN"
        ));
    }

    ensure!(x.dtype() == DType::BF16, "adaln_modulate expects BF16 input, got {:?}", x.dtype());
    let dims = x.shape().dims().to_vec();
    ensure!(dims.len() == 3, "adaln_modulate expects [B, HW, C], got {:?}", dims);
    let b = dims[0] as usize;
    let seq = dims[1] as usize;
    let hidden = dims[2] as usize;
    ensure!(seq == h * w, "adaln_modulate expects seq {} == h*w {}", seq, h * w);
    ensure!(hidden == params.hidden, "adaln hidden mismatch {} vs {}", hidden, params.hidden);
    ensure!(
        norm.weight.dtype() == DType::BF16,
        "norm weight must be BF16, got {:?}",
        norm.weight.dtype()
    );
    ensure!(
        norm.bias.dtype() == DType::BF16,
        "norm bias must be BF16, got {:?}",
        norm.bias.dtype()
    );

    let mut nhwc = x.reshape(&[b, h, w, hidden])?.clone_result()?;

    ensure!(
        cond.dtype() == DType::BF16,
        "adaln_modulate expects BF16 cond, got {:?}",
        cond.dtype()
    );
    ensure!(
        params.weight.dtype() == DType::BF16,
        "adaln_modulate weight must be BF16, got {:?}",
        params.weight.dtype()
    );
    ensure!(
        params.bias.dtype() == DType::BF16,
        "adaln_modulate bias must be BF16, got {:?}",
        params.bias.dtype()
    );

    let matmul_start = Instant::now();
    let aff = cond.matmul_bf16(&params.weight)?;
    let matmul_ms = if timing { matmul_start.elapsed().as_secs_f64() * 1_000.0 } else { 0.0 };
    let bias = params.bias.reshape(&[1, 3 * hidden])?;
    let aff_bf16 = broadcast_add(&aff, &bias)?;

    let shift = aff_bf16.narrow(1, 0, hidden)?.clone_result()?;
    let scale = aff_bf16.narrow(1, hidden, hidden)?.clone_result()?;
    let gate = aff_bf16.narrow(1, 2 * hidden, hidden)?.clone_result()?;

    let fuse_start = Instant::now();
    adaln_modulate_bf16_inplace(
        &mut nhwc,
        Some(&norm.weight),
        Some(&norm.bias),
        Some(&scale),
        Some(&shift),
        b as i32,
        h as i32,
        w as i32,
        hidden as i32,
        LN_EPS,
    )
    .map_err(|e| anyhow!("adaln_modulate_bf16_inplace failed: {e}"))?;
    let fuse_ms = if timing { fuse_start.elapsed().as_secs_f64() * 1_000.0 } else { 0.0 };

    let y = nhwc.reshape(&[b, seq, hidden])?.clone_result()?;
    expect_bf16("adaln_out", &y)?;

    let gate_base = gate.reshape(&[b, 1, hidden])?;
    let gate_tokens = broadcast_to_as(&gate_base, &[b, seq, hidden], DType::BF16)?;
    expect_bf16("adaln_gate", &gate_tokens)?;

    if let Some(start) = total_start {
        let total_ms = start.elapsed().as_secs_f64() * 1_000.0;
        eprintln!(
            "[timing] adaln_modulate matmul={:.2}ms fused_ln={:.2}ms total={:.2}ms",
            matmul_ms, fuse_ms, total_ms
        );
    }

    Ok((y, gate_tokens))
}

fn split_heads(x: &Tensor, heads: usize, head_dim: usize) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    let b = dims[0];
    let t = dims[1];
    ensure!(dims[2] == heads * head_dim, "head reshape mismatch {:?}", dims);
    Ok(x.reshape(&[b, t, heads, head_dim])?.permute(&[0, 2, 1, 3])?)
}

fn merge_heads(x: &Tensor) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    let b = dims[0];
    let h = dims[1];
    let t = dims[2];
    let dh = dims[3];
    Ok(x.permute(&[0, 2, 1, 3])?.reshape(&[b, t, h * dh])?)
}

fn split_qkv_projection(qkv: &Tensor, hidden: usize) -> Result<(Tensor, Tensor, Tensor)> {
    let dims = qkv.shape().dims().to_vec();
    ensure!(dims.len() == 3, "qkv projection expects [B,SEQ,3*H], got {:?}", dims);
    let last = dims.len() - 1;
    ensure!(
        dims[last] == hidden * 3,
        "qkv projection last dim {} != 3*hidden {}",
        dims[last],
        hidden * 3
    );
    let q = qkv.narrow(last, 0, hidden)?.clone_result()?;
    let k = qkv.narrow(last, hidden, hidden)?.clone_result()?;
    let v = qkv.narrow(last, hidden * 2, hidden)?.clone_result()?;
    Ok((q, k, v))
}

fn split_kv_projection(kv: &Tensor, hidden: usize) -> Result<(Tensor, Tensor)> {
    let dims = kv.shape().dims().to_vec();
    ensure!(dims.len() == 3, "kv projection expects [B,SEQ,2*H], got {:?}", dims);
    let last = dims.len() - 1;
    ensure!(
        dims[last] == hidden * 2,
        "kv projection last dim {} != 2*hidden {}",
        dims[last],
        hidden * 2
    );
    let k = kv.narrow(last, 0, hidden)?.clone_result()?;
    let v = kv.narrow(last, hidden, hidden)?.clone_result()?;
    Ok((k, v))
}

fn self_attention_inference(block: &SdxlBlockRuntime, x: &Tensor) -> Result<Tensor> {
    let qkv = linear_no_bias(x, &block.attn1_qkv)?;
    let (q_tokens, k_tokens, v_tokens) = split_qkv_projection(&qkv, block.hidden)?;

    let q = split_heads(&q_tokens, block.heads, block.head_dim)?.clone_result()?;
    let k = split_heads(&k_tokens, block.heads, block.head_dim)?.clone_result()?;
    let v = split_heads(&v_tokens, block.heads, block.head_dim)?.clone_result()?;
    assert_cuda("self_attn.q_heads", &q)?;
    assert_cuda("self_attn.k_heads", &k)?;
    assert_cuda("self_attn.v_heads", &v)?;

    let dims_heads = q.shape().dims().to_vec();
    let b = dims_heads[0] as usize;
    let heads = dims_heads[1] as usize;
    let seq = dims_heads[2] as usize;
    let seq_k = k.shape().dims()[2];
    let scale = 1.0f32 / (block.head_dim as f32).sqrt();
    let AttnChunkConfig { q_chunk, kv_chunk } = current_attn_chunks();
    let q_chunk_req = effective_chunk_len(seq, q_chunk);
    let kv_chunk_req = effective_chunk_len(seq_k, kv_chunk);
    let chunk_requested = q_chunk_req < seq || kv_chunk_req < seq_k;
    let allow_chunk = chunk_requested && chunking_enabled();
    let q_chunk = if allow_chunk { q_chunk_req } else { seq };
    let kv_chunk = if allow_chunk { kv_chunk_req } else { seq_k };
    let use_full_seq = q_chunk >= seq && kv_chunk >= seq_k;
    let kernel_debug = kernel_debug_enabled();

    #[cfg(feature = "bf16_u16")]
    if fused_attention_enabled() && use_full_seq {
        let bh = b * heads;
        let q_flat =
            q.permute(&[0, 2, 1, 3])?.reshape(&[bh, seq, block.head_dim])?.clone_result()?;
        let k_flat =
            k.permute(&[0, 2, 1, 3])?.reshape(&[bh, seq_k, block.head_dim])?.clone_result()?;
        let v_flat =
            v.permute(&[0, 2, 1, 3])?.reshape(&[bh, seq_k, block.head_dim])?.clone_result()?;

        let fused_start = if kernel_debug { Some(Instant::now()) } else { None };
        match fused_attention_forward(&q_flat, &k_flat, &v_flat, scale) {
            Ok(ctx_heads) => {
                record_kernel_stats("self_attn.fused_ctx", &ctx_heads, fused_start)?;
                let fuse_proj_start = if kernel_debug { Some(Instant::now()) } else { None };
                let ctx = ctx_heads
                    .reshape(&[b, heads, seq, block.head_dim])?
                    .permute(&[0, 2, 1, 3])?
                    .reshape(&[b, seq, block.hidden])?;
                let proj = linear_with_bias(&ctx, &block.attn1.out)?;
                record_kernel_stats("self_attn.fused_proj", &proj, fuse_proj_start)?;
                assert_cuda("self_attn.proj", &proj)?;
                return Ok(proj);
            }
            Err(err) => {
                if seq <= CHUNK_SMALL_SEQ_GUARD && seq_k <= CHUNK_SMALL_SEQ_GUARD {
                    return Err(anyhow!(
                        "fused self-attention failed at seq={} seq_k={} head_dim={} with SDXL_FUSED_ATTENTION=1: {err}",
                        seq,
                        seq_k,
                        block.head_dim
                    ));
                }
                if trace_verbose() {
                    eprintln!("[attn] fused self path fallback: {err}");
                }
            }
        }
    }

    if chunk_requested && !allow_chunk && trace_verbose() {
        eprintln!(
            "[attn_cfg] self chunk request ignored (set SDXL_ENABLE_CHUNK=1 to enable): q_chunk={} kv_chunk={} seq={} seq_k={}",
            q_chunk_req, kv_chunk_req, seq, seq_k
        );
    }

    if mem_trace_enabled() {
        let tag = format!(
            "self_attn:init seq={} seq_k={} q_chunk={} kv_chunk={}",
            seq, seq_k, q_chunk, kv_chunk
        );
        mem_snap(&tag);
    }

    #[allow(unused_mut)]
    let mut ctx_opt: Option<Tensor> = None;

    #[cfg(feature = "bf16_u16")]
    {
        if use_full_seq {
            match sdpa::forward(&q, &k, &v, None) {
                Ok(t) => ctx_opt = Some(t),
                Err(e) => {
                    if trace_verbose() {
                        eprintln!("[attn] self fused sdpa::forward fallback: {e}");
                    }
                }
            }
        }
    }

    let ctx = if let Some(t) = ctx_opt {
        t
    } else if use_full_seq {
        #[cfg(feature = "bf16_u16")]
        {
            sdpa::forward(&q, &k, &v, None).map_err(|e| anyhow!(e.to_string()))?
        }
        #[cfg(not(feature = "bf16_u16"))]
        {
            sdpa_chunked_bf16("self_attn", &q, &k, &v, scale, q_chunk, kv_chunk)?
        }
    } else {
        if trace_verbose() {
            eprintln!(
                "[attn] self fused chunking {} seq={seq} q_chunk={q_chunk} kv_chunk={kv_chunk}",
                block.name
            );
        }
        let mut outputs: Vec<Tensor> = Vec::new();
        let mut start = 0usize;
        while start < seq {
            let len = q_chunk.min(seq - start);
            let q_chunk_tensor = q.narrow(2, start, len)?.clone_result()?;
            let out =
                sdpa_chunked_bf16("self_attn", &q_chunk_tensor, &k, &v, scale, len, kv_chunk)?;
            outputs.push(out);
            start += len;
            if mem_trace_enabled() {
                let tag = format!(
                    "self_attn:fused_chunk block={} q_range=[{}, {})",
                    block.name,
                    start.saturating_sub(len),
                    start
                );
                mem_snap(&tag);
            }
        }
        concat_bf16_chunks(&outputs, "self_attn")?
    };
    expect_bf16("self_attn.ctx_heads", &ctx)?;
    assert_cuda("self_attn.ctx_heads", &ctx)?;
    let ctx = merge_heads(&ctx)?;
    assert_cuda("self_attn.ctx", &ctx)?;

    let proj = linear_with_bias(&ctx, &block.attn1.out)?;
    assert_cuda("self_attn.proj", &proj)?;
    Ok(proj)
}

fn cross_attention_inference(block: &SdxlBlockRuntime, x: &Tensor, ctx: &Tensor) -> Result<Tensor> {
    let q_tokens = linear_no_bias(x, &block.attn2.q)?;
    let kv_tokens = linear_no_bias(ctx, &block.attn2_kv)?;
    let (k_tokens, v_tokens) = split_kv_projection(&kv_tokens, block.hidden)?;

    let q = split_heads(&q_tokens, block.heads, block.head_dim)?.clone_result()?;
    let k = split_heads(&k_tokens, block.heads, block.head_dim)?.clone_result()?;
    let v = split_heads(&v_tokens, block.heads, block.head_dim)?.clone_result()?;
    assert_cuda("cross_attn.q_heads", &q)?;
    assert_cuda("cross_attn.k_heads", &k)?;
    assert_cuda("cross_attn.v_heads", &v)?;

    let dims_heads = q.shape().dims().to_vec();
    let b = dims_heads[0] as usize;
    let heads = dims_heads[1] as usize;
    let seq = dims_heads[2] as usize;
    let seq_k = k.shape().dims()[2];
    let scale = 1.0f32 / (block.head_dim as f32).sqrt();
    let AttnChunkConfig { q_chunk, kv_chunk } = current_attn_chunks();
    let q_chunk_req = effective_chunk_len(seq, q_chunk);
    let kv_chunk_req = effective_chunk_len(seq_k, kv_chunk);
    let chunk_requested = q_chunk_req < seq || kv_chunk_req < seq_k;
    let allow_chunk = chunk_requested && chunking_enabled();
    let q_chunk = if allow_chunk { q_chunk_req } else { seq };
    let kv_chunk = if allow_chunk { kv_chunk_req } else { seq_k };
    let use_full_seq = q_chunk >= seq && kv_chunk >= seq_k;
    let kernel_debug = kernel_debug_enabled();

    #[cfg(feature = "bf16_u16")]
    if fused_attention_enabled() && use_full_seq {
        let bh = b * heads;
        let q_flat =
            q.permute(&[0, 2, 1, 3])?.reshape(&[bh, seq, block.head_dim])?.clone_result()?;
        let k_flat =
            k.permute(&[0, 2, 1, 3])?.reshape(&[bh, seq_k, block.head_dim])?.clone_result()?;
        let v_flat =
            v.permute(&[0, 2, 1, 3])?.reshape(&[bh, seq_k, block.head_dim])?.clone_result()?;

        let fused_start = if kernel_debug { Some(Instant::now()) } else { None };
        match fused_attention_forward(&q_flat, &k_flat, &v_flat, scale) {
            Ok(ctx_heads) => {
                record_kernel_stats("cross_attn.fused_ctx", &ctx_heads, fused_start)?;
                let fuse_proj_start = if kernel_debug { Some(Instant::now()) } else { None };
                let ctx_merge = ctx_heads
                    .reshape(&[b, heads, seq, block.head_dim])?
                    .permute(&[0, 2, 1, 3])?
                    .reshape(&[b, seq, block.hidden])?;
                let proj = linear_with_bias(&ctx_merge, &block.attn2.out)?;
                record_kernel_stats("cross_attn.fused_proj", &proj, fuse_proj_start)?;
                assert_cuda("cross_attn.proj", &proj)?;
                return Ok(proj);
            }
            Err(err) => {
                if seq <= CHUNK_SMALL_SEQ_GUARD && seq_k <= CHUNK_SMALL_SEQ_GUARD {
                    return Err(anyhow!(
                        "fused cross-attention failed at seq={} seq_k={} head_dim={} with SDXL_FUSED_ATTENTION=1: {err}",
                        seq,
                        seq_k,
                        block.head_dim
                    ));
                }
                if trace_verbose() {
                    eprintln!("[attn] fused cross path fallback: {err}");
                }
            }
        }
    }

    if chunk_requested && !allow_chunk && trace_verbose() {
        eprintln!(
            "[attn_cfg] cross chunk request ignored (set SDXL_ENABLE_CHUNK=1 to enable): q_chunk={} kv_chunk={} seq={} seq_k={}",
            q_chunk_req, kv_chunk_req, seq, seq_k
        );
    }

    if mem_trace_enabled() {
        let tag = format!(
            "cross_attn:init seq={} seq_k={} q_chunk={} kv_chunk={}",
            seq, seq_k, q_chunk, kv_chunk
        );
        mem_snap(&tag);
    }

    #[allow(unused_mut)]
    let mut ctx_opt: Option<Tensor> = None;

    #[cfg(feature = "bf16_u16")]
    {
        if use_full_seq {
            match sdpa::forward(&q, &k, &v, None) {
                Ok(t) => ctx_opt = Some(t),
                Err(e) => {
                    if trace_verbose() {
                        eprintln!("[attn] cross fused sdpa::forward fallback: {e}");
                    }
                }
            }
        }
    }

    let ctx_heads = if let Some(t) = ctx_opt {
        t
    } else if use_full_seq {
        #[cfg(feature = "bf16_u16")]
        {
            sdpa::forward(&q, &k, &v, None).map_err(|e| anyhow!(e.to_string()))?
        }
        #[cfg(not(feature = "bf16_u16"))]
        {
            sdpa_chunked_bf16("cross_attn", &q, &k, &v, scale, q_chunk, kv_chunk)?
        }
    } else {
        if fused_attention_enabled()
            && seq <= CHUNK_SMALL_SEQ_GUARD
            && seq_k <= CHUNK_SMALL_SEQ_GUARD
        {
            bail!(
                "chunk fallback entered for cross-attention at seq={} seq_k={} head_dim={} with SDXL_FUSED_ATTENTION=1",
                seq,
                seq_k,
                block.head_dim
            );
        }
        if trace_verbose() {
            eprintln!(
                "[attn] cross chunking {} seq={seq} q_chunk={q_chunk} kv_chunk={kv_chunk}",
                block.name
            );
        }
        let mut outputs: Vec<Tensor> = Vec::new();
        let mut start = 0usize;
        while start < seq {
            let len = q_chunk.min(seq - start);
            let q_chunk_tensor = q.narrow(2, start, len)?.clone_result()?;
            let out =
                sdpa_chunked_bf16("cross_attn", &q_chunk_tensor, &k, &v, scale, len, kv_chunk)?;
            outputs.push(out);
            start += len;
            if mem_trace_enabled() {
                let tag = format!(
                    "cross_attn:fused_chunk block={} q_range=[{}, {})",
                    block.name,
                    start.saturating_sub(len),
                    start
                );
                mem_snap(&tag);
            }
        }
        concat_bf16_chunks(&outputs, "cross_attn")?
    };
    expect_bf16("cross_attn.ctx_heads", &ctx_heads)?;
    let ctx_final = merge_heads(&ctx_heads)?;
    assert_cuda("cross_attn.ctx", &ctx_final)?;

    let proj = linear_with_bias(&ctx_final, &block.attn2.out)?;
    assert_cuda("cross_attn.proj", &proj)?;
    Ok(proj)
}

fn self_attention(block: &SdxlBlockRuntime, x: &Tensor) -> Result<Tensor> {
    let q = linear_no_bias(x, &block.attn1.q)?;
    let k = linear_no_bias(x, &block.attn1.k)?;
    let v = linear_no_bias(x, &block.attn1.v)?;
    assert_cuda("self_attn.q", &q)?;
    assert_cuda("self_attn.k", &k)?;
    assert_cuda("self_attn.v", &v)?;

    let q = split_heads(&q, block.heads, block.head_dim)?.clone_result()?;
    let k = split_heads(&k, block.heads, block.head_dim)?.clone_result()?;
    let v = split_heads(&v, block.heads, block.head_dim)?.clone_result()?;
    assert_cuda("self_attn.q_heads", &q)?;
    assert_cuda("self_attn.k_heads", &k)?;
    assert_cuda("self_attn.v_heads", &v)?;

    let AttnChunkConfig { q_chunk, kv_chunk } = current_attn_chunks();
    let scale = 1.0f32 / (block.head_dim as f32).sqrt();
    let seq = q.shape().dims()[2];
    let seq_k = k.shape().dims()[2];
    let q_chunk_req = effective_chunk_len(seq, q_chunk);
    let kv_chunk_req = effective_chunk_len(seq_k, kv_chunk);
    let chunk_requested = q_chunk_req < seq || kv_chunk_req < seq_k;
    let allow_chunk = chunk_requested && chunking_enabled();
    if chunk_requested && !allow_chunk && trace_verbose() {
        eprintln!(
            "[attn_cfg] self chunk request ignored (set SDXL_ENABLE_CHUNK=1 to enable): q_chunk={} kv_chunk={} seq={} seq_k={}",
            q_chunk_req, kv_chunk_req, seq, seq_k
        );
    }
    let q_chunk = if allow_chunk { q_chunk_req } else { seq };
    let kv_chunk = if allow_chunk { kv_chunk_req } else { seq_k };
    if mem_trace_enabled() {
        let tag = format!(
            "self_attn:init seq={} seq_k={} q_chunk={} kv_chunk={}",
            seq, seq_k, q_chunk, kv_chunk
        );
        mem_snap(&tag);
    }

    let kernel_debug = kernel_debug_enabled();
    let attn_start = if kernel_debug { Some(Instant::now()) } else { None };
    #[allow(unused_mut)]
    let mut ctx_opt: Option<Tensor> = None;
    let use_full_seq = q_chunk >= seq && kv_chunk >= seq_k;

    #[cfg(feature = "bf16_u16")]
    {
        if use_full_seq {
            match sdpa::forward(&q, &k, &v, None) {
                Ok(t) => ctx_opt = Some(t),
                Err(e) => {
                    if trace_verbose() {
                        eprintln!("[attn] self sdpa::forward fallback: {e}");
                    }
                }
            }
        }
    }

    let ctx = if let Some(t) = ctx_opt {
        t
    } else if use_full_seq {
        #[cfg(feature = "bf16_u16")]
        {
            sdpa::forward(&q, &k, &v, None).map_err(|e| anyhow!(e.to_string()))?
        }
        #[cfg(not(feature = "bf16_u16"))]
        {
            sdpa_chunked_bf16("self_attn", &q, &k, &v, scale, q_chunk, kv_chunk)?
        }
    } else {
        if fused_attention_enabled()
            && seq <= CHUNK_SMALL_SEQ_GUARD
            && seq_k <= CHUNK_SMALL_SEQ_GUARD
        {
            bail!(
                "chunk fallback entered for self-attention at seq={} seq_k={} head_dim={} with SDXL_FUSED_ATTENTION=1",
                seq,
                seq_k,
                block.head_dim
            );
        }
        if trace_verbose() {
            eprintln!(
                "[attn] self chunking {} seq={seq} q_chunk={q_chunk} kv_chunk={kv_chunk}",
                block.name
            );
        }
        let mut outputs: Vec<Tensor> = Vec::new();
        let mut start = 0usize;
        while start < seq {
            let len = q_chunk.min(seq - start);
            let q_chunk = q.narrow(2, start, len)?.clone_result()?;
            let out = sdpa_chunked_bf16("self_attn", &q_chunk, &k, &v, scale, len, kv_chunk)?;
            outputs.push(out);
            start += len;
            if mem_trace_enabled() {
                let tag = format!(
                    "self_attn:chunk block={} q_range=[{}, {})",
                    block.name,
                    start.saturating_sub(len),
                    start
                );
                mem_snap(&tag);
            }
        }
        concat_bf16_chunks(&outputs, "self_attn")?
    };
    expect_bf16("self_attn.ctx_heads", &ctx)?;
    assert_cuda("self_attn.ctx_heads", &ctx)?;
    record_kernel_stats("self_attn.core", &ctx, attn_start)?;
    let ctx = merge_heads(&ctx)?;
    assert_cuda("self_attn.ctx", &ctx)?;

    let proj_start = if kernel_debug { Some(Instant::now()) } else { None };
    let proj = linear_with_bias(&ctx, &block.attn1.out)?;
    record_kernel_stats("self_attn.proj", &proj, proj_start)?;
    assert_cuda("self_attn.proj", &proj)?;
    Ok(proj)
}

fn cross_attention(block: &SdxlBlockRuntime, x: &Tensor, ctx: &Tensor) -> Result<Tensor> {
    let q = linear_no_bias(x, &block.attn2.q)?;
    let k = linear_no_bias(ctx, &block.attn2.k)?;
    let v = linear_no_bias(ctx, &block.attn2.v)?;
    assert_cuda("cross_attn.q", &q)?;
    assert_cuda("cross_attn.k", &k)?;
    assert_cuda("cross_attn.v", &v)?;

    let q = split_heads(&q, block.heads, block.head_dim)?.clone_result()?;
    let k = split_heads(&k, block.heads, block.head_dim)?.clone_result()?;
    let v = split_heads(&v, block.heads, block.head_dim)?.clone_result()?;
    assert_cuda("cross_attn.q_heads", &q)?;
    assert_cuda("cross_attn.k_heads", &k)?;
    assert_cuda("cross_attn.v_heads", &v)?;

    let AttnChunkConfig { q_chunk, kv_chunk } = current_attn_chunks();
    let scale = 1.0f32 / (block.head_dim as f32).sqrt();
    let seq = q.shape().dims()[2];
    let seq_k = k.shape().dims()[2];
    let q_chunk_req = effective_chunk_len(seq, q_chunk);
    let kv_chunk_req = effective_chunk_len(seq_k, kv_chunk);
    let chunk_requested = q_chunk_req < seq || kv_chunk_req < seq_k;
    let allow_chunk = chunk_requested && chunking_enabled();
    if chunk_requested && !allow_chunk && trace_verbose() {
        eprintln!(
            "[attn_cfg] cross chunk request ignored (set SDXL_ENABLE_CHUNK=1 to enable): q_chunk={} kv_chunk={} seq={} seq_k={}",
            q_chunk_req, kv_chunk_req, seq, seq_k
        );
    }
    let q_chunk = if allow_chunk { q_chunk_req } else { seq };
    let kv_chunk = if allow_chunk { kv_chunk_req } else { seq_k };
    if mem_trace_enabled() {
        let tag = format!(
            "cross_attn:init seq={} seq_k={} q_chunk={} kv_chunk={}",
            seq, seq_k, q_chunk, kv_chunk
        );
        mem_snap(&tag);
    }

    let kernel_debug = kernel_debug_enabled();
    let attn_start = if kernel_debug { Some(Instant::now()) } else { None };
    #[allow(unused_mut)]
    let mut ctx_opt: Option<Tensor> = None;
    let use_full_seq = q_chunk >= seq && kv_chunk >= seq_k;

    #[cfg(feature = "bf16_u16")]
    {
        if use_full_seq {
            match sdpa::forward(&q, &k, &v, None) {
                Ok(t) => ctx_opt = Some(t),
                Err(e) => {
                    if trace_verbose() {
                        eprintln!("[attn] cross sdpa::forward fallback: {e}");
                    }
                }
            }
        }
    }

    let ctx = if let Some(t) = ctx_opt {
        t
    } else if use_full_seq {
        #[cfg(feature = "bf16_u16")]
        {
            sdpa::forward(&q, &k, &v, None).map_err(|e| anyhow!(e.to_string()))?
        }
        #[cfg(not(feature = "bf16_u16"))]
        {
            sdpa_chunked_bf16("cross_attn", &q, &k, &v, scale, q_chunk, kv_chunk)?
        }
    } else {
        if trace_verbose() {
            eprintln!(
                "[attn] cross chunking {} seq={seq} q_chunk={q_chunk} kv_chunk={kv_chunk}",
                block.name
            );
        }
        let mut outputs: Vec<Tensor> = Vec::new();
        let mut start = 0usize;
        while start < seq {
            let len = q_chunk.min(seq - start);
            let q_chunk = q.narrow(2, start, len)?.clone_result()?;
            let out = sdpa_chunked_bf16("cross_attn", &q_chunk, &k, &v, scale, len, kv_chunk)?;
            outputs.push(out);
            start += len;
            if mem_trace_enabled() {
                let tag = format!(
                    "cross_attn:chunk block={} q_range=[{}, {})",
                    block.name,
                    start.saturating_sub(len),
                    start
                );
                mem_snap(&tag);
            }
        }
        concat_bf16_chunks(&outputs, "cross_attn")?
    };
    expect_bf16("cross_attn.ctx_heads", &ctx)?;
    record_kernel_stats("cross_attn.core", &ctx, attn_start)?;
    let ctx = merge_heads(&ctx)?;
    assert_cuda("cross_attn.ctx", &ctx)?;

    let proj_start = if kernel_debug { Some(Instant::now()) } else { None };
    let proj = linear_with_bias(&ctx, &block.attn2.out)?;
    record_kernel_stats("cross_attn.proj", &proj, proj_start)?;
    assert_cuda("cross_attn.proj", &proj)?;
    Ok(proj)
}

#[cfg(feature = "bf16_u16")]
fn feed_forward_fused(block: &SdxlBlockRuntime, x: &Tensor) -> Result<Tensor> {
    ensure!(x.dtype() == DType::BF16, "fused FFN expects BF16 input, got {:?}", x.dtype());
    let dims = x.shape().dims().to_vec();
    ensure!(dims.len() >= 2, "fused FFN expects >=2D tensor, got {:?}", dims);
    let last = dims.len() - 1;
    let leading: usize = dims[..last].iter().product();
    let hidden = dims[last] as usize;

    let out_weight_dims = block.ff.out.weight.shape().dims();
    ensure!(
        out_weight_dims[0] as usize == hidden,
        "ff.out weight mismatch: {} vs input {}",
        out_weight_dims[0],
        hidden
    );
    let act_dim = out_weight_dims[1] as usize;

    let proj = linear_without_bias(x, &block.ff.proj)?;
    let proj_flat = proj.reshape(&[leading, act_dim * 2])?;
    if trace_verbose() {
        eprintln!(
            "[fused_ffn] rows={} hidden={} act_dim={} proj_flat={:?}",
            leading,
            hidden,
            act_dim,
            proj_flat.shape().dims()
        );
    }
    let flat_out = fused_ffn_forward(
        &proj_flat,
        &block.ff.proj.bias,
        &block.ff.out.weight,
        &block.ff.out.bias,
    )?;

    let mut out_shape = dims.clone();
    out_shape[last] = hidden;
    let reshaped = flat_out.reshape(&out_shape)?;
    if trace_verbose() {
        eprintln!("[fused_ffn] out {:?}", reshaped.shape().dims());
    }
    Ok(reshaped)
}

#[cfg(not(feature = "bf16_u16"))]
fn feed_forward_fused(_block: &SdxlBlockRuntime, _x: &Tensor) -> Result<Tensor> {
    Err(anyhow!("fused BF16 FFN requires the bf16_u16 feature flag to be enabled"))
}

fn feed_forward(block: &SdxlBlockRuntime, x: &Tensor) -> Result<Tensor> {
    let timing = timing_enabled();
    let total_start = if timing { Some(Instant::now()) } else { None };
    let kernel_debug = kernel_debug_enabled();

    if fused_ffn_enabled() {
        let telemetry_start = if kernel_debug { Some(Instant::now()) } else { None };
        let out = feed_forward_fused(block, x)
            .context("fused FFN execution failed while SDXL_FUSED_FFN=1")?;
        record_kernel_stats("feed_forward.fused", &out, telemetry_start)?;

        if let Some(start) = total_start {
            let total_ms = start.elapsed().as_secs_f64() * 1_000.0;
            eprintln!("[timing] feed_forward fused total={:.2}ms", total_ms);
        }
        return Ok(out);
    }

    let telemetry_start = if kernel_debug { Some(Instant::now()) } else { None };
    if trace_verbose() {
        eprintln!(
            "[feed_forward] x {:?} proj_w {:?} out_w {:?}",
            x.shape().dims(),
            block.ff.proj.weight.shape().dims(),
            block.ff.out.weight.shape().dims()
        );
    }
    let proj_start = Instant::now();
    let up = linear_with_bias(x, &block.ff.proj)?;
    let proj_ms = if timing { proj_start.elapsed().as_secs_f64() * 1_000.0 } else { 0.0 };
    if trace_verbose() {
        eprintln!("[feed_forward] up {:?}", up.shape().dims());
    }

    let up_dims = up.shape().dims().to_vec();
    let last = up_dims.len() - 1;
    let act_dim = block.ff.out.weight.shape().dims()[1];
    ensure!(
        up_dims[last] == act_dim * 2,
        "ff.proj output {} != 2 * ff.out input {}",
        up_dims[last],
        act_dim
    );

    let gate = up.narrow(last, act_dim, act_dim)?.clone_result()?;
    let act_slice = up.narrow(last, 0, act_dim)?;
    let gelu_start = Instant::now();
    let activated = if act_slice.dtype() == DType::BF16 {
        #[cfg(feature = "bf16_u16")]
        {
            gelu_bf16(&act_slice)?
        }
        #[cfg(not(feature = "bf16_u16"))]
        {
            act_slice.gelu()?
        }
    } else {
        act_slice.gelu()?
    };
    let gelu_ms = if timing { gelu_start.elapsed().as_secs_f64() * 1_000.0 } else { 0.0 };
    let gated = activated.mul(&gate)?;
    if trace_verbose() {
        eprintln!(
            "[feed_forward] gated {:?} dtype {:?} storage {:?}",
            gated.shape().dims(),
            gated.dtype(),
            gated.storage_dtype()
        );
    }
    let out_start = Instant::now();
    let out = linear_with_bias(&gated, &block.ff.out)?;
    let out_ms = if timing { out_start.elapsed().as_secs_f64() * 1_000.0 } else { 0.0 };
    record_kernel_stats("feed_forward.out", &out, telemetry_start)?;

    if let Some(start) = total_start {
        let total_ms = start.elapsed().as_secs_f64() * 1_000.0;
        eprintln!(
            "[timing] feed_forward proj_in={:.2}ms gelu={:.2}ms proj_out={:.2}ms total={:.2}ms",
            proj_ms, gelu_ms, out_ms, total_ms
        );
    }

    Ok(out)
}

#[cfg(feature = "bf16_u16")]
fn sdpa_full_attn_bf16(
    tag: &str,
    q_flat: &Tensor,
    k_flat: &Tensor,
    v_flat: &Tensor,
    scale: f32,
    b: usize,
    heads: usize,
    seq_q: usize,
    seq_k: usize,
    head_dim: usize,
) -> Result<Tensor> {
    let device = q_flat.device().clone();
    let bh = b * heads;
    let k_t = k_flat.transpose_dims(1, 2)?.clone_result()?;
    let mut logits_ws = acquire_workspace_tensor(&device, DType::BF16, &[bh, seq_q, seq_k])?;
    {
        let logits_buf = logits_ws.tensor_mut();
        bmm_bf16_fp32acc_out(q_flat, &k_t, logits_buf, false, false)?;
    }
    let logits = logits_ws.tensor().to_dtype(DType::F32)?.mul_scalar(scale)?;

    let max_logits = logits.max_dim(2, true)?;
    let logits_shifted = logits.sub(&max_logits)?;
    let exp_logits = logits_shifted.exp()?;
    let denom = exp_logits.sum_dim(2)?.reshape(&[bh, seq_q, 1])?;

    let exp_bf16 = exp_logits.to_dtype(DType::BF16)?;
    let v_bf16 = v_flat.to_dtype(DType::BF16)?;
    let mut out_ws = acquire_workspace_tensor(&device, DType::BF16, &[bh, seq_q, head_dim])?;
    {
        let out_chunk = out_ws.tensor_mut();
        bmm_bf16_fp32acc_out(&exp_bf16, &v_bf16, out_chunk, false, false)?;
    }

    let out = out_ws.tensor().to_dtype(DType::F32)?;
    let normed = out.div(&denom)?;
    let result = normed.to_dtype(DType::BF16)?;
    if trace_verbose() {
        eprintln!("[attn_fallback] {tag} full-seq path bh={} seq_q={} seq_k={}", bh, seq_q, seq_k);
    }
    result.reshape(&[b, heads, seq_q, head_dim]).map_err(Into::into)
}

#[cfg(not(feature = "bf16_u16"))]
fn sdpa_full_attn_bf16(
    tag: &str,
    q_flat: &Tensor,
    k_flat: &Tensor,
    v_flat: &Tensor,
    scale: f32,
    b: usize,
    heads: usize,
    seq_q: usize,
    seq_k: usize,
    head_dim: usize,
) -> Result<Tensor> {
    let bh = b * heads;
    let q32 = to_owning_fp32_strong(q_flat)?;
    let k32 = to_owning_fp32_strong(k_flat)?;
    let v32 = to_owning_fp32_strong(v_flat)?;
    let k_t = k32.transpose_dims(1, 2)?;
    let logits = matmul_f32_batched_rowmajor(&q32, &k_t)?.mul_scalar(scale)?;
    let attn = logits.softmax(-1)?;
    let ctx = attn.bmm(&v32)?;
    if trace_verbose() {
        eprintln!(
            "[attn_fallback] {tag} full-seq f32 path bh={} seq_q={} seq_k={}",
            bh, seq_q, seq_k
        );
    }
    ctx.to_dtype(DType::BF16)?.reshape(&[b, heads, seq_q, head_dim]).map_err(Into::into)
}

fn sdpa_chunked_bf16(
    tag: &str,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    q_chunk: usize,
    kv_chunk: usize,
) -> Result<Tensor> {
    const MB_BYTES: u128 = 1_048_576;
    let dims = q.shape().dims().to_vec();
    ensure!(dims.len() == 4, "sdpa_chunked expects [B,H,SEQ,D], got {:?}", dims);
    let b = dims[0] as usize;
    let heads = dims[1] as usize;
    let seq_q = dims[2] as usize;
    let head_dim = dims[3] as usize;
    let seq_k = k.shape().dims()[2] as usize;

    let bh = b * heads;
    let mut q_chunk = effective_chunk_len(seq_q, q_chunk);
    q_chunk = clamp_q_chunk(bh, head_dim, q_chunk);
    let base_kv_chunk = effective_chunk_len(seq_k, kv_chunk);

    let device = q.device().clone();
    let runtime_cfg = current_attn_runtime();
    if let Some(env) = runtime_cfg {
        if !env.reuse {
            bail!("SDXL streaming attention requires ATTN_CHUNK_REUSE=1 (set ATTN_CHUNK_REUSE=1).");
        }
        if env.min_free_mb > 0 {
            let core_device = CoreDevice::from_flame_cuda(device.as_ref());
            let free_bytes = cuda_available_memory(&core_device)
                .context("sdxl streaming attention: query available GPU memory")?;
            let free_mb = free_bytes / (MB_BYTES as usize);
            ensure!(
                free_mb >= env.min_free_mb,
                "SDXL streaming attention requires at least {} MB of free GPU memory (found {} MB). \
                 Reduce ATTN_CHUNK_SIZE or free additional memory.",
                env.min_free_mb,
                free_mb
            );
        }
    }
    if trace_verbose() {
        eprintln!(
            "[attn_cfg] {tag} bh={} head_dim={} seq_q={} seq_k={} q_chunk={} base_kv_chunk={}",
            bh, head_dim, seq_q, seq_k, q_chunk, base_kv_chunk
        );
    }
    let q_flat = q.reshape(&[bh, seq_q, head_dim])?;
    let k_flat = k.reshape(&[bh, seq_k, head_dim])?;
    let v_flat = v.reshape(&[bh, seq_k, head_dim])?;

    #[cfg(feature = "bf16_u16")]
    {
        if seq_q <= CHUNK_SMALL_SEQ_GUARD
            && (q_chunk < seq_q || base_kv_chunk < seq_k)
            && std::env::var("SDXL_DISABLE_CHUNK_GUARD").ok().as_deref() != Some("1")
        {
            bail!(
                "{tag}: attention fallback requested chunking at seq={} seq_k={} (q_chunk={} kv_chunk={}). \
                 Guarding because small geometry should stay on fused path. Set SDXL_DISABLE_CHUNK_GUARD=1 to override.",
                seq_q, seq_k, q_chunk, base_kv_chunk
            );
        }
    }

    if q_chunk >= seq_q && base_kv_chunk >= seq_k {
        return sdpa_full_attn_bf16(
            tag, &q_flat, &k_flat, &v_flat, scale, b, heads, seq_q, seq_k, head_dim,
        );
    }

    let mut outputs: Vec<Tensor> = Vec::new();
    let mut q_start = 0usize;
    while q_start < seq_q {
        let q_len = q_chunk.min(seq_q - q_start);
        let q_slice = q_flat.narrow(1, q_start, q_len)?.clone_result()?;
        let kv_chunk = clamp_kv_chunk(bh, q_len, base_kv_chunk);
        if let Some(env) = runtime_cfg {
            if env.max_workspace_mb > 0 {
                let workspace_bytes = (bh as u128)
                    .saturating_mul(q_len as u128)
                    .saturating_mul(kv_chunk as u128)
                    .saturating_mul(4u128);
                let workspace_mb =
                    ((workspace_bytes + MB_BYTES - 1) / MB_BYTES) as usize;
                ensure!(
                    workspace_mb <= env.max_workspace_mb,
                    "SDXL streaming attention chunk (q_len={}, kv_len={}) would require ~{} MB of workspace (> {} MB cap). \
                     Lower ATTN_CHUNK_SIZE or raise ATTN_MAX_WORKSPACE_MB.",
                    q_len,
                    kv_chunk,
                    workspace_mb,
                    env.max_workspace_mb
                );
            }
        }
        if trace_verbose() {
            eprintln!(
                "[attn_chunk] {tag} q_start={} q_len={} kv_chunk={} bh={}",
                q_start, q_len, kv_chunk, bh
            );
        }
        let mut m =
            Tensor::zeros_dtype(Shape::from_dims(&[bh, q_len]), DType::F32, device.clone())?
                .add_scalar(-1.0e9)?;
        let mut l =
            Tensor::zeros_dtype(Shape::from_dims(&[bh, q_len]), DType::F32, device.clone())?;
        let mut out_acc = Tensor::zeros_dtype(
            Shape::from_dims(&[bh, q_len, head_dim]),
            DType::F32,
            device.clone(),
        )?;

        let mut kv_start = 0usize;
        while kv_start < seq_k {
            let kv_len = kv_chunk.min(seq_k - kv_start);
            let k_slice = k_flat.narrow(1, kv_start, kv_len)?.clone_result()?;
            let v_slice = v_flat.narrow(1, kv_start, kv_len)?.clone_result()?;

            #[cfg(feature = "bf16_u16")]
            let mut logits = {
                let k_t = k_slice.transpose_dims(1, 2)?.clone_result()?;
                let mut logits_ws =
                    acquire_workspace_tensor(&device, DType::BF16, &[bh, q_len, kv_len])?;
                {
                    let logits_buf = logits_ws.tensor_mut();
                    bmm_bf16_fp32acc_out(&q_slice, &k_t, logits_buf, false, false)?;
                }
                logits_ws.tensor().to_dtype(DType::F32)?
            };

            #[cfg(not(feature = "bf16_u16"))]
            let mut logits = {
                eprintln!("[bf16-inventory] chunked sdpa widening Q/K/V to F32 (fallback path)");
                let q32 = to_owning_fp32_strong(&q_slice)?;
                let k32 = to_owning_fp32_strong(&k_slice)?;
                let k_t = k32.transpose_dims(1, 2)?;
                matmul_f32_batched_rowmajor(&q32, &k_t)?
            };

            logits = logits.mul_scalar(scale)?;

            let chunk_max = logits.max_dim(2, true)?;
            let chunk_max_flat = chunk_max.reshape(&[bh, q_len])?;
            let new_m = Tensor::maximum(&m, &chunk_max_flat)?;

            let m_diff = m.sub(&new_m)?;
            let exp_old = m_diff.exp()?.mul(&l)?;

            let new_m_f32 = new_m.reshape(&[bh, q_len, 1])?;
            let logits_shifted = logits.sub(&new_m_f32)?;
            let exp_chunk = logits_shifted.exp()?;
            let chunk_sum = exp_chunk.sum_dim(2)?.reshape(&[bh, q_len])?;

            let new_l = exp_old.clone().add(&chunk_sum)?;

            let exp_old_factor = exp_old.reshape(&[bh, q_len, 1])?;
            out_acc = out_acc.mul(&exp_old_factor)?;

            #[cfg(feature = "bf16_u16")]
            let attn_chunk = {
                let exp_chunk_bf16 = exp_chunk.to_dtype(DType::BF16)?;
                let v_bf16 = v_slice.to_dtype(DType::BF16)?;
                let mut out_ws =
                    acquire_workspace_tensor(&device, DType::BF16, &[bh, q_len, head_dim])?;
                {
                    let out_chunk = out_ws.tensor_mut();
                    bmm_bf16_fp32acc_out(&exp_chunk_bf16, &v_bf16, out_chunk, false, false)?;
                }
                out_ws.tensor().to_dtype(DType::F32)?
            };

            #[cfg(not(feature = "bf16_u16"))]
            let attn_chunk = {
                let v32 = to_owning_fp32_strong(&v_slice)?;
                matmul_f32_batched_rowmajor(&exp_chunk, &v32)?
            };

            out_acc = out_acc.add(&attn_chunk)?;

            m = new_m;
            l = new_l;
            kv_start += kv_len;
        }

        let norm = l.reshape(&[bh, q_len, 1])?;
        let out_norm = out_acc.div(&norm)?;
        let out_bf16 = out_norm.to_dtype(DType::BF16)?;
        if trace_verbose() {
            eprintln!(
                "[chunk_fallback] {tag} chunk {}..{} dtype {:?}",
                q_start,
                q_start + q_len,
                out_bf16.dtype()
            );
        }
        outputs.push(out_bf16.reshape(&[b, heads, q_len, head_dim])?);

        if let Err(e) = device.synchronize() {
            return Err(anyhow!("{tag}: failed to synchronize chunk: {e}"));
        }

        if trace_verbose() {
            let tag = format!("[chunk_fallback] {tag} q={}..{}", q_start, q_start + q_len);
            mem_snap(&tag);
        }
        q_start += q_len;
    }
    concat_bf16_chunks(&outputs, tag)
}

fn concat_bf16_chunks(chunks: &[Tensor], tag: &str) -> Result<Tensor> {
    ensure!(!chunks.is_empty(), "{tag}: no chunks to concatenate");
    let first_dims = chunks[0].shape().dims().to_vec();
    ensure!(first_dims.len() == 4, "{tag}: expected 4D chunks, got {:?}", first_dims);
    let (b, h, _, d) = (first_dims[0], first_dims[1], first_dims[2], first_dims[3]);
    let device = chunks[0].device().clone();

    let mut total_seq = 0usize;
    for chunk in chunks {
        let dims = chunk.shape().dims();
        ensure!(dims.len() == 4, "{tag}: chunk rank mismatch {:?}", dims);
        ensure!(
            dims[0] == b && dims[1] == h && dims[3] == d,
            "{tag}: chunk dims {:?} incompatible with {:?}",
            dims,
            first_dims
        );
        expect_bf16(tag, chunk)?;
        total_seq += dims[2];
    }

    let result =
        Tensor::zeros_dtype(Shape::from_dims(&[b, h, total_seq, d]), DType::BF16, device.clone())?;

    let bh = b * h;
    let row_stride_dst = total_seq * d;
    let mut offset = 0usize;

    let mut result_flat = result.reshape(&[bh, total_seq, d])?;
    for chunk in chunks {
        let len = chunk.shape().dims()[2];
        let chunk_flat = chunk.reshape(&[bh, len, d])?;
        let row_stride_src = len * d;
        for row in 0..bh {
            let src_start = row * row_stride_src;
            let dst_start = row * row_stride_dst + offset * d;
            result_flat.copy_bf16_region_from(dst_start, &chunk_flat, src_start, row_stride_src)?;
        }
        offset += len;
    }

    drop(result_flat);
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_utils::softmax_stable;
    use anyhow::Result;
    use cudarc::driver::CudaDevice;
    use eridiffusion_core::device::require_cuda_device;
    use flame_core::{Device as FDevice, Shape};
    use std::sync::Arc;

    fn bf16_rand(device: &Arc<CudaDevice>, shape: &[usize]) -> Result<Tensor> {
        let tensor = Tensor::randn(Shape::from_dims(shape), 0.0, 1.0, device.clone())?;
        tensor.to_dtype(DType::BF16).map_err(Into::into)
    }

    #[test]
    #[ignore = "requires CUDA"]
    fn attention_chunk_matches_full() -> Result<()> {
        let device = match FDevice::cuda(0) {
            Ok(dev) => dev,
            Err(err) => {
                eprintln!("[skip] CUDA unavailable: {err}");
                return Ok(());
            }
        };
        require_cuda_device(0);
        let cuda = device.cuda_device_arc();

        let batch = 1usize;
        let heads = 2usize;
        let seq = 64usize;
        let head_dim = 32usize;
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        let shape = [batch * heads, seq, head_dim];
        let q = bf16_rand(&cuda, &shape)?;
        let k = bf16_rand(&cuda, &shape)?;
        let v = bf16_rand(&cuda, &shape)?;

        let full = {
            let q32 = to_owning_fp32_strong(&q)?;
            let k32 = to_owning_fp32_strong(&k)?;
            let v32 = to_owning_fp32_strong(&v)?;
            let k_t = k32.transpose_dims(1, 2)?.clone_result()?;
            let logits = q32.bmm(&k_t)?.mul_scalar(scale)?;
            let attn = softmax_stable(&logits, 2)?;
            attn.bmm(&v32)?
        };

        let chunk = 16usize;
        let chunked = {
            let k32 = to_owning_fp32_strong(&k)?;
            let v32 = to_owning_fp32_strong(&v)?;
            let k_t = k32.transpose_dims(1, 2)?.clone_result()?;
            let mut ctx_chunks: Vec<Tensor> = Vec::new();
            let mut start = 0usize;
            while start < seq {
                let len = chunk.min(seq - start);
                let q_chunk = q.narrow(1, start, len)?.clone_result()?;
                let q_chunk32 = to_owning_fp32_strong(&q_chunk)?;
                let logits = q_chunk32.bmm(&k_t)?.mul_scalar(scale)?;
                let attn = softmax_stable(&logits, 2)?;
                let ctx = attn.bmm(&v32)?;
                ctx_chunks.push(ctx);
                start += len;
            }
            let refs: Vec<&Tensor> = ctx_chunks.iter().collect();
            Tensor::cat(&refs, 1)?
        };

        let full_vec = full.to_dtype(DType::F32)?.to_vec()?;
        let chunk_vec = chunked.to_dtype(DType::F32)?.to_vec()?;
        assert_eq!(full_vec.len(), chunk_vec.len());
        let mut max_abs = 0f32;
        for (a, b) in full_vec.iter().zip(chunk_vec.iter()) {
            max_abs = max_abs.max((a - b).abs());
        }
        assert!(max_abs < 5e-3, "max abs diff {} too high", max_abs);
        Ok(())
    }
}
