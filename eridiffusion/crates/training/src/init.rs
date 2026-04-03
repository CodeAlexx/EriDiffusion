use std::sync::Arc;

use eridiffusion_core::{Device, Error};
use flame_core::{rng, CudaDevice, Shape, Tensor};

#[inline]
fn fan_in(dims: &[usize]) -> f32 {
    if dims.len() == 2 {
        dims[0] as f32
    } else {
        (dims.iter().product::<usize>()).max(1) as f32
    }
}

#[inline]
fn kaiming_std(dims: &[usize]) -> f32 {
    (2.0f32 / fan_in(dims)).sqrt()
}

#[inline]
#[allow(dead_code)]
fn glorot_std(dims: &[usize]) -> f32 {
    if dims.len() == 2 {
        let (fin, fout) = (dims[0] as f32, dims[1] as f32);
        (2.0f32 / (fin + fout)).sqrt()
    } else {
        (2.0f32 / fan_in(dims)).sqrt()
    }
}

/// BF16 normal init (feature on) or fallback to legacy path (feature off)
pub fn init_weight_normal(
    shape: &[usize],
    mean: f32,
    std: f32,
    seed: u64,
    dev: Arc<CudaDevice>,
) -> flame_core::Result<Tensor> {
    #[cfg(feature = "bf16_u16")]
    {
        use flame_core::bf16_normal::normal_bf16;
        return normal_bf16(Shape::from_dims(shape), mean, std, seed, dev);
    }
    #[cfg(not(feature = "bf16_u16"))]
    {
        let _ = rng::set_seed(seed);
        let t = Tensor::randn(Shape::from_dims(shape), 0.0, 1.0, dev.clone())?;
        Ok(t.affine(std, mean)?)
    }
}

/// BF16 uniform init (feature on) or fallback
pub fn init_weight_uniform(
    shape: &[usize],
    low: f32,
    high: f32,
    seed: u64,
    dev: Arc<CudaDevice>,
) -> flame_core::Result<Tensor> {
    #[cfg(feature = "bf16_u16")]
    {
        use flame_core::bf16_factories::uniform_bf16;
        return uniform_bf16(Shape::from_dims(shape), low, high, seed, dev);
    }
    #[cfg(not(feature = "bf16_u16"))]
    {
        let _ = rng::set_seed(seed);
        Tensor::uniform(Shape::from_dims(shape), low, high, dev)
    }
}

/// Zeros BF16 / legacy zeros
pub fn init_zeros(shape: &[usize], dev: Arc<CudaDevice>) -> flame_core::Result<Tensor> {
    #[cfg(feature = "bf16_u16")]
    {
        use flame_core::bf16_factories::zeros_bf16;
        return zeros_bf16(Shape::from_dims(shape), dev);
    }
    #[cfg(not(feature = "bf16_u16"))]
    {
        Tensor::zeros(Shape::from_dims(shape), dev)
    }
}

/// Ones BF16 / legacy ones
pub fn init_ones(shape: &[usize], dev: Arc<CudaDevice>) -> flame_core::Result<Tensor> {
    #[cfg(feature = "bf16_u16")]
    {
        use flame_core::bf16_elementwise::add_bf16;
        let z = init_zeros(shape, dev.clone())?;
        // Make a scalar-1 BF16 tensor and add broadcast (keeps BF16 in/out)
        let one = init_weight_uniform(&[1], 1.0, 1.0, 123, dev)?;
        return add_bf16(&z, &one).map_err(flame_core::Error::from);
    }
    #[cfg(not(feature = "bf16_u16"))]
    {
        Tensor::ones(Shape::from_dims(shape), dev)
    }
}

/// Convenience: Kaiming-normal (BF16 when enabled)
pub fn init_kaiming(
    shape: &[usize],
    seed: u64,
    dev: Arc<CudaDevice>,
) -> flame_core::Result<Tensor> {
    init_weight_normal(shape, 0.0, kaiming_std(shape), seed, dev)
}

/// Convenience: Glorot-uniform (BF16 when enabled)
pub fn init_glorot_uniform(
    shape: &[usize],
    seed: u64,
    dev: Arc<CudaDevice>,
) -> flame_core::Result<Tensor> {
    let (fin, fout) = (shape[0] as f32, shape[1] as f32);
    let a = (6.0f32 / (fin + fout)).sqrt();
    init_weight_uniform(shape, -a, a, seed, dev)
}

/// Initialize the global device manager default device from a string like "cuda:0" or "cpu".
/// Returns the parsed Device on success.
pub fn init_global_device(dev_str: &str) -> anyhow::Result<Device> {
    // Parse device string
    let dev = if let Some(s) = dev_str.strip_prefix("cuda:") {
        let ord: usize =
            s.parse().map_err(|_| Error::InvalidInput(format!("bad device: {dev_str}")))?;
        // Validate CUDA ordinal
        let _ = CudaDevice::new(ord)
            .map_err(|e| Error::Device(format!("CUDA device {} unavailable: {}", ord, e)))?;
        Device::Cuda(ord)
    } else if dev_str.eq_ignore_ascii_case("cpu") {
        return Err(Error::Device(
            "CPU execution is disabled in this build. Use a CUDA device (e.g. cuda:0).".into(),
        )
        .into());
    } else {
        return Err(Error::InvalidInput(format!("unknown device: {dev_str}")).into());
    };
    // Set default in global DeviceManager
    let mgr = eridiffusion_core::device::device_manager();
    mgr.set_default_device(dev.clone())?;
    Ok(dev)
}
