//! Device management for multi-GPU support

use crate::{Result, Error};
use candle_core::Device as CandleDevice;
use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Device abstraction
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Device {
    Cpu,
    Cuda(usize),
}

impl Device {
    /// Get the best available device
    pub fn best_available() -> Result<Self> {
        if candle_core::utils::cuda_is_available() {
            Ok(Device::Cuda(0))
        } else {
            Ok(Device::Cpu)
        }
    }
    
    /// Check if CUDA is available
    pub fn cuda_is_available() -> bool {
        candle_core::utils::cuda_is_available()
    }
    
    /// Get CUDA device if available
    pub fn cuda_if_available(ordinal: usize) -> Result<Self> {
        if Self::cuda_is_available() {
            Ok(Device::Cuda(ordinal))
        } else {
            Ok(Device::Cpu)
        }
    }
    
    /// Convert to Candle device
    pub fn to_candle(&self) -> Result<CandleDevice> {
        match self {
            Device::Cpu => Ok(CandleDevice::Cpu),
            Device::Cuda(ordinal) => CandleDevice::new_cuda(*ordinal)
                .map_err(|e| Error::Device(format!("Failed to create CUDA device: {}", e))),
        }
    }
    
    /// Create from Candle device
    pub fn from_candle(device: &CandleDevice) -> Self {
        match device {
            CandleDevice::Cpu => Device::Cpu,
            CandleDevice::Cuda(_) => Device::Cuda(0), // Default to device 0 - ordinal not accessible
            CandleDevice::Metal(_) => Device::Cpu, // Fallback to CPU for Metal
        }
    }
    
    /// Get device ordinal (for GPU devices)
    pub fn ordinal(&self) -> Option<usize> {
        match self {
            Device::Cpu => None,
            Device::Cuda(ordinal) => Some(*ordinal),
        }
    }
    
    /// Check if this is a GPU device
    pub fn is_gpu(&self) -> bool {
        !matches!(self, Device::Cpu)
    }
}

impl From<CandleDevice> for Device {
    fn from(device: CandleDevice) -> Self {
        match device {
            CandleDevice::Cpu => Device::Cpu,
            CandleDevice::Cuda(_) => Device::Cuda(0), // Default to device 0 - ordinal not accessible
            CandleDevice::Metal(_) => Device::Cpu, // Fallback to CPU for Metal
        }
    }
}

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

/// Memory allocation information
#[derive(Debug, Clone)]
struct AllocationInfo {
    pub size: usize,
    pub purpose: String,
    pub allocated_at: std::time::Instant,
}

impl DeviceManager {
    /// Create a new device manager
    pub fn new() -> Result<Self> {
        let manager = Self {
            devices: DashMap::new(),
            allocations: DashMap::new(),
            default_device: RwLock::new(Device::Cpu),
        };
        
        manager.discover_devices()?;
        Ok(manager)
    }
    
    /// Discover available devices
    fn discover_devices(&self) -> Result<()> {
        // Always add CPU
        self.devices.insert(
            Device::Cpu,
            DeviceInfo {
                device: Device::Cpu,
                name: "CPU".to_string(),
                total_memory: sys_info::mem_info()
                    .map(|info| info.total as usize * 1024)
                    .unwrap_or(0),
                available_memory: sys_info::mem_info()
                    .map(|info| info.avail as usize * 1024)
                    .unwrap_or(0),
                compute_capability: None,
            },
        );
        
        // Check for CUDA devices
        if Device::cuda_is_available() {
            use crate::cuda::{cuda_init, cuda_device_properties};
            
            // Initialize CUDA with error handling
            match cuda_init() {
                Ok(()) => {
                    // Get device count safely
                    // In candle 0.9, we need to handle device enumeration differently
                    let device_count = 1; // For now, assume at least 1 CUDA device if available
                    
                    for ordinal in 0..device_count {
                        match cuda_device_properties(ordinal) {
                            Ok(props) => {
                                let device = Device::Cuda(ordinal);
                                let info = DeviceInfo {
                                    device: device.clone(),
                                    name: props.name,
                                    total_memory: props.total_memory,
                                    available_memory: props.available_memory,
                                    compute_capability: Some((props.compute_capability.0 as u32, props.compute_capability.1 as u32)),
                                };
                                self.devices.insert(device, info);
                            }
                            Err(e) => {
                                tracing::warn!("Failed to get properties for CUDA device {}: {}", ordinal, e);
                            }
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!("CUDA initialization failed: {}. Falling back to CPU only.", e);
                }
            }
            
            // Set default to first CUDA device if any were found
            if !self.devices.is_empty() && self.devices.iter().any(|d| matches!(d.key(), Device::Cuda(_))) {
                *self.default_device.write() = Device::Cuda(0);
            }
        }
        
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
    pub fn set_default_device(&self, device: Device) -> Result<()> {
        if !self.devices.contains_key(&device) {
            return Err(Error::Device(format!("Device not available: {:?}", device)));
        }
        
        *self.default_device.write() = device;
        Ok(())
    }
    
    /// Allocate memory on a device
    pub fn allocate(&self, device: &Device, size: usize, purpose: String) -> Result<()> {
        // Check if device exists
        if !self.devices.contains_key(device) {
            return Err(Error::Device(format!("Device not available: {:?}", device)));
        }
        
        // Record allocation
        self.allocations
            .entry(device.clone())
            .or_default()
            .push(AllocationInfo {
                size,
                purpose,
                allocated_at: std::time::Instant::now(),
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
    pub fn find_best_device(&self, required_memory: usize) -> Result<Device> {
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
pub fn initialize_devices() -> Result<()> {
    let manager = Arc::new(DeviceManager::new()?);
    DEVICE_MANAGER.set(manager)
        .map_err(|_| Error::Device("Device manager already initialized".to_string()))?;
    Ok(())
}

/// Get the global device manager
pub fn device_manager() -> Arc<DeviceManager> {
    DEVICE_MANAGER.get()
        .expect("Device manager not initialized")
        .clone()
}