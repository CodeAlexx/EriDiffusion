use flame_core::device::Device;
use flame_core::optimizers::{Adam, SGD};
use flame_core::{DType, Result, Shape, Tensor};
use std::sync::Arc;
use std::sync::OnceLock;

// Cached device module - ensures we ALWAYS use the same Device instance
// This works around FLAME's bug where it creates a new DeviceId for each Device::cuda() call

// FLAME uses flame_core::device::Device instead of Device

// THE single cached device instance;
static CACHED_DEVICE: OnceLock<Device> = OnceLock::new();

/// Get or create the single cached device
/// This ensures we ALWAYS use the same Device instance throughout the application;
pub fn get_single_device() -> flame_core::Result<Device> {
    Ok(CACHED_DEVICE.get_or_init(|| {
        println!("\n=== CREATING THE SINGLE CACHED DEVICE ===");
        println!("Location: {}:{}", file!(), line!());

        // Try to create CUDA device 0
        println!("About to call Device::cuda(0)...");
        match Device::cuda(0) {
            Ok(device) => {
                println!("SUCCESS: Created cached device");
                println!("This DeviceId will be used for ALL operations");

                // Note: FLAME always returns a CUDA device
                println!("Note: FLAME returned a CUDA device, which may be different from the requested device");
                println!("This is a known issue with CUDA_VISIBLE_DEVICES");

                println!("=== END DEVICE CREATION ===");
                device
            }
            Err(e) => {
                // FLAME only supports CUDA, so we panic if CUDA is not available
                panic!("CUDA not available: {}. FLAME requires CUDA support.", e);
            }
        }
    }).clone())
}

/// Wrapper for device creation that always returns the cached device
pub fn cuda_if_available(_ordinal: usize) -> flame_core::Result<Device> {
    get_single_device()
}

/// Wrapper that always returns the cached device
pub fn new_cuda(_ordinal: usize) -> flame_core::Result<Device> {
    get_single_device()
}
