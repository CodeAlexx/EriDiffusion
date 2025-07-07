//! Cached device module - ensures we ALWAYS use the same Device instance
//! This works around Candle's bug where it creates a new DeviceId for each Device::new_cuda() call

use anyhow::Result;
use candle_core::Device;
use std::sync::OnceLock;

// THE single cached device instance
static CACHED_DEVICE: OnceLock<Device> = OnceLock::new();

/// Get or create the single cached device
/// This ensures we ALWAYS use the same Device instance throughout the application
pub fn get_single_device() -> Result<Device> {
    Ok(CACHED_DEVICE.get_or_init(|| {
        println!("\n=== CREATING THE SINGLE CACHED DEVICE ===");
        println!("Location: {}:{}", file!(), line!());
        
        // Try to create CUDA device 0
        println!("About to call Device::new_cuda(0)...");
        match Device::new_cuda(0) {
            Ok(device) => {
                println!("SUCCESS: Created cached device: {:?}", device);
                println!("This DeviceId will be used for ALL operations");
                println!("=== END DEVICE CREATION ===");
                device
            }
            Err(e) => {
                println!("CUDA not available: {}", e);
                println!("Falling back to CPU");
                Device::Cpu
            }
        }
    }).clone())
}

/// Wrapper for device creation that always returns the cached device
pub fn cuda_if_available(_ordinal: usize) -> Result<Device> {
    get_single_device()
}

/// Wrapper that always returns the cached device
pub fn new_cuda(_ordinal: usize) -> Result<Device> {
    get_single_device()
}