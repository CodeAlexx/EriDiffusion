//! Device management for multi-GPU support

use crate::Error;
// FLAME doesn't export Device directly, we'll create our own mapping
use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, OnceLock};

/// Device abstraction
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Device {
    Cpu,
    Cuda(usize),
}

impl Device {
    /// Backward-compatible constructor used by older tests/APIs
    pub fn cuda(ordinal: usize) -> anyhow::Result<Self> {
        flame_core::CudaDevice::new(ordinal)
            .map(|_| Device::Cuda(ordinal))
            .map_err(|e| anyhow::Error::from(Error::Device(format!(
                "CUDA device {ordinal} not available: {e}"
            ))))
    }
    /// Get the best available device
    pub fn best_available() -> anyhow::Result<Self> {
        Self::cuda(0)
    }
    
    /// Check if CUDA is available
    pub fn cuda_is_available() -> bool {
        // Check if we can create a CUDA device
        flame_core::CudaDevice::new(0).is_ok()
    }
    
    /// Get CUDA device if available
    pub fn cuda_if_available(ordinal: usize) -> anyhow::Result<Self> {
        Self::cuda(ordinal)
    }
    
    /// Convert to FLAME CUDA device (only for CUDA devices)
    pub fn to_flame_cuda(&self) -> anyhow::Result<std::sync::Arc<flame_core::CudaDevice>> {
        match self {
            Device::Cpu => Err(Error::Device("Cannot convert CPU device to CudaDevice".into()).into()),
            Device::Cuda(ordinal) => flame_core::CudaDevice::new(*ordinal)
                .map_err(|e| Error::Device(format!("Failed to create CUDA device: {}", e)).into()),
        }
    }
    
    /// Create from FLAME CUDA device
    pub fn from_flame_cuda(device: &flame_core::CudaDevice) -> Self {
        // CudaDevice always represents a CUDA device
        Device::Cuda(device.ordinal())
    }
    
    /// Get device ordinal (for GPU devices)
    pub fn ordinal(&self) -> Option<usize> {
        match self { Device::Cpu => None, Device::Cuda(ordinal) => Some(*ordinal) }
    }
    
    /// Check if this is a GPU device
    pub fn is_gpu(&self) -> bool {
        matches!(self, Device::Cuda(_))
    }

    /// Returns the CUDA stream as a raw pointer usable by FFI launchers.
    /// Currently returns the default stream (null) which is valid for CUDA kernels.
    pub fn cuda_stream_raw_ptr(&self) -> *mut core::ffi::c_void {
        core::ptr::null_mut()
    }
}

// FLAME doesn't have a unified Device type, only CudaDevice
// So we can't implement From trait for a general device

/// Device information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceInfo {
    pub device: Device,
    pub name: String,
    pub total_memory: usize,
    pub available_memory: usize,
    pub compute_capability: Option<(u32, u32)>,
}

/// Device manager for multi-GPU systems
pub struct DeviceManager {
    devices: DashMap<Device, DeviceInfo>,
    allocations: DashMap<Device, Vec<AllocationInfo>>,
    default_device: RwLock<Device>,
}

/// Ensure a CUDA context exists on the selected GPU. Panics with a clear
/// diagnostic if the device is unavailable so downstream code never silently
/// falls back to CPU execution.
pub fn require_cuda_device(ordinal: usize) {
    if let Err(e) = Device::cuda(ordinal) {
        panic!(
            "CUDA GPU required. Set CUDA_HOME=/usr/local/cuda and export LD_LIBRARY_PATH=$CUDA_HOME/lib64. Detail: {e}"
        );
    }
}

/// Memory allocation information
#[derive(Debug, Clone)]
struct AllocationInfo {
    pub size: usize,
    pub _purpose: String,
    pub _allocated_at: std::time::Instant,
}

impl DeviceManager {
    /// Create a new device manager
    pub fn new() -> anyhow::Result<Self> {
        let default = Device::cuda(0)?;
        let manager = Self {
            devices: DashMap::new(),
            allocations: DashMap::new(),
            default_device: RwLock::new(default.clone()),
        };

        manager.discover_devices()?;
        Ok(manager)
    }
    
    /// Discover available devices
    fn discover_devices(&self) -> anyhow::Result<()> {
        use crate::cuda::{cuda_device_properties, cuda_init};

        match cuda_init() {
            Ok(()) => {}
            Err(e) => {
                return Err(Error::Device(format!(
                    "CUDA initialization failed: {e}. Set CUDA_HOME=/usr/local/cuda and ensure the NVIDIA driver is loaded."
                ))
                .into())
            }
        }

        let mut discovered = 0usize;
        for ordinal in 0..16 {
            match flame_core::CudaDevice::new(ordinal) {
                Ok(_) => {
                    discovered = ordinal + 1;
                    match cuda_device_properties(ordinal) {
                        Ok(props) => {
                            let info = DeviceInfo {
                                device: Device::Cuda(ordinal),
                                name: props.name,
                                total_memory: props.total_memory,
                                available_memory: props.available_memory,
                                compute_capability: Some((
                                    props.compute_capability.0 as u32,
                                    props.compute_capability.1 as u32,
                                )),
                            };
                            self.devices.insert(Device::Cuda(ordinal), info);
                        }
                        Err(e) => {
                            tracing::warn!(
                                "Failed to query properties for CUDA device {}: {}",
                                ordinal,
                                e
                            );
                        }
                    }
                }
                Err(_) => break,
            }
        }

        if discovered == 0 {
            return Err(Error::Device(
                "CUDA GPU required. Set CUDA_HOME=/usr/local/cuda and export LD_LIBRARY_PATH=$CUDA_HOME/lib64."
                    .into(),
            )
            .into());
        }

        *self.default_device.write() = Device::Cuda(0);
        Ok(())
    }
    
    /// Get all available devices
    pub fn list_devices(&self) -> Vec<DeviceInfo> {
        self.devices.iter()
            .map(|entry| entry.value().clone())
            .collect()
    }
    
    /// Get device info
    pub fn get_device_info(&self, device: &Device) -> Option<DeviceInfo> {
        self.devices.get(device).map(|entry| entry.clone())
    }
    
    /// Get the default device
    pub fn default_device(&self) -> Device {
        self.default_device.read().clone()
    }
    
    /// Set the default device
    pub fn set_default_device(&self, device: Device) -> anyhow::Result<()> {
        if !self.devices.contains_key(&device) {
            return Err(Error::Device(format!("Device not available: {:?}", device)).into());
        }
        
        *self.default_device.write() = device;
        Ok(())
    }
    
    /// Allocate memory on a device
    pub fn allocate(&self, device: &Device, size: usize, purpose: String) -> anyhow::Result<()> {
        // Check if device exists
        if !self.devices.contains_key(device) {
            return Err(Error::Device(format!("Device not available: {:?}", device)).into());
        }
        
        // Record allocation
        self.allocations
            .entry(device.clone())
            .or_default()
            .push(AllocationInfo {
                size,
                _purpose: purpose,
                _allocated_at: std::time::Instant::now(),
            });
        
        Ok(())
    }
    
    /// Get memory usage for a device
    pub fn memory_usage(&self, device: &Device) -> usize {
        self.allocations
            .get(device)
            .map(|allocations| allocations.iter().map(|a| a.size).sum())
            .unwrap_or(0)
    }
    
    /// Find the best device for an allocation
    pub fn find_best_device(&self, required_memory: usize) -> anyhow::Result<Device> {
        // Try GPU devices first
        let mut candidates: Vec<(Device, usize)> = self.devices
            .iter()
            .filter(|entry| entry.key().is_gpu())
            .map(|entry| {
                let device = entry.key().clone();
                let info = entry.value();
                let used = self.memory_usage(&device);
                let available = info.total_memory.saturating_sub(used);
                (device, available)
            })
            .filter(|(_, available)| *available >= required_memory)
            .collect();
        
        // Sort by available memory (descending)
        candidates.sort_by_key(|(_, available)| std::cmp::Reverse(*available));
        
        if let Some((device, _)) = candidates.first() {
            Ok(device.clone())
        } else {
            // Fall back to CPU
            Ok(Device::Cpu)
        }
    }
}

/// Global device manager instance
static DEVICE_MANAGER: once_cell::sync::OnceCell<Arc<DeviceManager>> = once_cell::sync::OnceCell::new();

/// Initialize device management
pub fn initialize_devices() -> anyhow::Result<()> {
    let manager = Arc::new(DeviceManager::new()?);
    DEVICE_MANAGER.set(manager)
        .map_err(|_| Error::Device("Device manager already initialized".into()))?;
    Ok(())
}

/// Get the global device manager
pub fn device_manager() -> Arc<DeviceManager> {
    DEVICE_MANAGER.get()
        .expect("Device manager not initialized")
        .clone()
}

/// Convenience: get the shared CUDA device as Arc<CudaDevice> using the global DeviceManager.
/// Falls back to CPU error if CUDA is not available.
static SHARED_CUDA: OnceLock<Arc<flame_core::CudaDevice>> = OnceLock::new();

pub fn shared_cuda_device() -> anyhow::Result<Arc<flame_core::CudaDevice>> {
    let _init = || -> anyhow::Result<Arc<flame_core::CudaDevice>> {
        if DEVICE_MANAGER.get().is_none() {
            initialize_devices()?;
        }
        let mgr = device_manager();
        let dev = mgr.default_device();
        match dev {
            Device::Cuda(ord) => {
                let cuda = flame_core::CudaDevice::new(ord)
                    .map_err(|e| Error::Device(format!("Failed to create CUDA device: {}", e)))?;
                Ok(cuda)
            }
            Device::Cpu => Err(Error::Device("CUDA device not available".into()).into()),
        }
    };
    let arc = SHARED_CUDA.get_or_init(|| {
        // Fallback init without error propagation
        let _ = initialize_devices();
        let mgr = device_manager();
        match mgr.default_device() {
            Device::Cuda(ord) => flame_core::CudaDevice::new(ord).unwrap(),
            Device::Cpu => panic!("CUDA device not available"),
        }
    });
    Ok(arc.clone())
}
