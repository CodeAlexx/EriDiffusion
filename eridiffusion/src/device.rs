use flame_core::device::Device as FlameDevice;
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::Mutex;

/// Device type for FLAME compatibility
pub type Device = FlameDevice;

/// Global device cache
static DEVICE_CACHE: Lazy<Mutex<HashMap<usize, Device>>> = Lazy::new(|| Mutex::new(HashMap::new()));

/// Create a new CUDA device (cached)
pub fn cuda_device(ordinal: usize) -> flame_core::Result<Device> {
    let mut cache = DEVICE_CACHE.lock().unwrap();

    if let Some(device) = cache.get(&ordinal) {
        Ok(device.clone())
    } else {
        let device = Device::cuda(ordinal)?;
        cache.insert(ordinal, device.clone());
        Ok(device)
    }
}

/// Get a cached device or create a new one
pub fn get_device(ordinal: usize) -> flame_core::Result<Device> {
    cuda_device(ordinal)
}

/// Create a CPU device (not supported in FLAME, returns error)
pub fn cpu_device() -> flame_core::Result<Device> {
    Err(flame_core::Error::InvalidOperation("CPU device not supported in FLAME".into()))
}

/// Check if device is CUDA
pub fn is_cuda(_device: &Device) -> bool {
    true // FLAME only supports CUDA
}

/// Get device ordinal
pub fn device_ordinal(device: &Device) -> usize {
    device.ordinal()
}

/// Synchronize device
pub fn synchronize(_device: &Device) -> flame_core::Result<()> {
    // FLAME devices handle synchronization internally
    // This is a no-op for now
    Ok(())
}
