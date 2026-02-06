use std::sync::Arc;

use eridiffusion_core::Device;
use flame_core::{bf16_normal, rng, CudaDevice, DType, Error as CoreError, Result as CoreResult, Shape, Tensor};

pub const BF16: DType = DType::BF16;
pub const F16_: DType = DType::F16;
pub const F32_: DType = DType::F32;

pub fn device_from_cuda_index(index: usize) -> CoreResult<Device> {
    // Validate upfront so downstream callers never construct an invalid CUDA handle.
    CudaDevice::new(index)
        .map(|_| Device::Cuda(index))
        .map_err(|e| CoreError::InvalidInput(format!("failed to create CUDA device {index}: {e}")))
}

#[inline]
pub fn device_cpu() -> Device {
    panic!(
        "CPU execution is disabled in this build. Use a CUDA device via device_from_cuda_index."
    )
}

fn cuda_device_arc_for(device: &Device) -> CoreResult<Arc<CudaDevice>> {
    match device {
        Device::Cuda(index) => Ok(CudaDevice::new(*index).map_err(|e| CoreError::InvalidInput(format!("failed to create CUDA device: {e}")))? ),
        Device::Cpu => Err(CoreError::Unsupported("CPU device is not supported (GPU-only build)".into())),
    }
}

fn ensure_supported_dtype(dtype: DType) -> CoreResult<()> {
    match dtype {
        DType::BF16 | DType::F16 | DType::F32 => Ok(()),
        _ => Err(CoreError::Unsupported(format!("unsupported dtype {:?}", dtype))),
    }
}

pub fn tensor_from_vec_on(
    data: Vec<f32>,
    shape: impl Into<Shape>,
    device: &Device,
    dtype: DType,
) -> CoreResult<Tensor> {
    ensure_supported_dtype(dtype)?;
    let cuda = cuda_device_arc_for(device)?;
    Tensor::from_vec_dtype(data, shape.into(), cuda, dtype)
}

pub fn tensor_from_slice_on(
    data: &[f32],
    shape: impl Into<Shape>,
    device: &Device,
    dtype: DType,
) -> CoreResult<Tensor> {
    tensor_from_vec_on(data.to_vec(), shape, device, dtype)
}

pub fn copy_to_device(tensor: &Tensor, device: &Device) -> CoreResult<Tensor> {
    let target = cuda_device_arc_for(device)?;
    if tensor.device().ordinal() != target.ordinal() {
        return Err(CoreError::Unsupported("cross-device tensor moves are not implemented".into()));
    }
    Ok(tensor.clone())
}

#[inline]
pub fn ensure_on_device(tensor: &Tensor, device: &Device) -> CoreResult<Tensor> {
    copy_to_device(tensor, device)
}

pub fn zeros_on(shape: impl Into<Shape>, device: &Device, dtype: DType) -> CoreResult<Tensor> {
    ensure_supported_dtype(dtype)?;
    let cuda = cuda_device_arc_for(device)?;
    Tensor::zeros_dtype(shape.into(), dtype, cuda)
}

pub fn ones_on(shape: impl Into<Shape>, device: &Device, dtype: DType) -> CoreResult<Tensor> {
    ensure_supported_dtype(dtype)?;
    let cuda = cuda_device_arc_for(device)?;
    let shape = shape.into();
    if dtype == DType::BF16 {
        return Tensor::zeros_dtype(shape, dtype, cuda)?.add_scalar(1.0);
    }
    Tensor::ones_dtype(shape, dtype, cuda)
}

pub fn full_on(
    shape: impl Into<Shape>,
    device: &Device,
    dtype: DType,
    value_f32: f32,
) -> CoreResult<Tensor> {
    ensure_supported_dtype(dtype)?;
    let shape = shape.into();
    if dtype == DType::BF16 {
        let cuda = cuda_device_arc_for(device)?;
        return Tensor::zeros_dtype(shape, dtype, cuda)?.add_scalar(value_f32);
    }
    let numel = shape.elem_count() as usize;
    let data = vec![value_f32; numel];
    tensor_from_vec_on(data, shape, device, dtype)
}

pub fn empty_like_on(other: &Tensor, device: &Device, dtype: Option<DType>) -> CoreResult<Tensor> {
    let target = dtype.unwrap_or_else(|| other.dtype());
    ensure_supported_dtype(target)?;
    let shape = other.shape().clone();
    zeros_on(shape, device, target)
}

pub fn zeros_like_on(other: &Tensor, device: &Device, dtype: Option<DType>) -> CoreResult<Tensor> {
    empty_like_on(other, device, dtype)
}

pub fn arange_on(start: f32, end: f32, step: f32, device: &Device, dtype: DType) -> CoreResult<Tensor> {
    ensure_supported_dtype(dtype)?;
    if step == 0.0 {
        return Err(CoreError::InvalidInput("step must be non-zero".into()));
    }
    let mut data = Vec::<f32>::new();
    let mut value = start;
    if step > 0.0 {
        while value < end {
            data.push(value);
            value += step;
        }
    } else {
        while value > end {
            data.push(value);
            value += step;
        }
    }
    let shape = Shape::from_dims(&[data.len()]);
    tensor_from_vec_on(data, shape, device, dtype)
}

pub fn randn_on(
    shape: impl Into<Shape>,
    device: &Device,
    dtype: DType,
    seed: Option<u64>,
) -> CoreResult<Tensor> {
    ensure_supported_dtype(dtype)?;
    let cuda = cuda_device_arc_for(device)?;
    let shape = shape.into();
    if let Some(s) = seed {
        rng::set_seed(s)?;
    }
    if dtype == DType::BF16 {
        let seed = seed.unwrap_or_else(rng::next_u64);
        return bf16_normal::normal_bf16(shape, 0.0, 1.0, seed, cuda);
    }
    let tensor = Tensor::randn(shape, 0.0, 1.0, cuda)?;
    if tensor.dtype() == dtype {
        Ok(tensor)
    } else {
        tensor.to_dtype(dtype)
    }
}

pub fn randn_scaled_on(
    shape: impl Into<Shape>,
    device: &Device,
    dtype: DType,
    mean: f32,
    std: f32,
    seed: Option<u64>,
) -> CoreResult<Tensor> {
    if dtype == DType::BF16 {
        let cuda = cuda_device_arc_for(device)?;
        if let Some(s) = seed {
            rng::set_seed(s)?;
        }
        let seed = seed.unwrap_or_else(rng::next_u64);
        return bf16_normal::normal_bf16(shape.into(), mean, std, seed, cuda);
    }
    let base = randn_on(shape, device, DType::F32, seed)?;
    let scaled = base.mul_scalar(std)?.add_scalar(mean)?;
    if dtype == DType::F32 {
        Ok(scaled)
    } else {
        scaled.to_dtype(dtype)
    }
}

pub fn uniform_on(
    shape: impl Into<Shape>,
    device: &Device,
    low: f32,
    high: f32,
) -> CoreResult<Tensor> {
    let cuda = cuda_device_arc_for(device)?;
    Tensor::uniform(shape.into(), low, high, cuda)
}

pub fn to_dtype(t: &Tensor, dtype: DType) -> CoreResult<Tensor> {
    ensure_supported_dtype(dtype)?;
    if t.dtype() == dtype {
        return Ok(t.clone());
    }
    t.to_dtype(dtype)
}





pub fn to_device_dtype(t: &Tensor, device: &Device, dtype: DType) -> CoreResult<Tensor> {
    ensure_supported_dtype(dtype)?;
    let target = cuda_device_arc_for(device)?;
    if t.device().ordinal() != target.ordinal() {
        return Err(CoreError::Unsupported("cross-device tensor moves are not implemented yet".into()));
    }
    if t.dtype() == dtype {
        Ok(t.clone())
    } else {
        t.to_dtype(dtype)
    }
}

#[inline]
pub fn shape1(n: i64) -> Shape { Shape::from_dims(&[n as usize]) }
#[inline]
pub fn shape2(n: i64, m: i64) -> Shape { Shape::from_dims(&[n as usize, m as usize]) }
#[inline]
pub fn shape3(a: i64, b: i64, c: i64) -> Shape {
    Shape::from_dims(&[a as usize, b as usize, c as usize])
}
#[inline]
pub fn shape4(a: i64, b: i64, c: i64, d: i64) -> Shape {
    Shape::from_dims(&[a as usize, b as usize, c as usize, d as usize])
}
