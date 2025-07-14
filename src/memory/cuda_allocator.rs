//! CUDA memory allocator wrapper

use std::ptr;
use std::ffi::c_void;

/// Error type for memory operations
#[derive(thiserror::Error, Debug)]
pub enum MemoryError {
    #[error("CUDA error: {0}")]
    CudaError(String),
    #[error("Invalid device ID: {0}")]
    InvalidDevice(i32),
    #[error("Allocation failed: {0} bytes")]
    AllocationFailed(usize),
    #[error("Memory not found")]
    MemoryNotFound,
}

pub type Result<T> = std::result::Result<T, MemoryError>;

/// CUDA memory allocator wrapper
pub struct CudaAllocator {
    device_id: i32,
    device: candle_core::Device,
    allocations: std::sync::Mutex<std::collections::HashMap<*mut c_void, (usize, std::alloc::Layout)>>,
}

impl CudaAllocator {
    pub fn new(device_id: i32) -> Result<Self> {
        // Create Candle device to verify CUDA is available
        let device = match candle_core::Device::new_cuda(device_id as usize) {
            Ok(dev) => dev,
            Err(e) => {
                // Fall back to CPU if CUDA not available
                log::warn!("CUDA device {} not available: {}. Using CPU.", device_id, e);
                candle_core::Device::Cpu
            }
        };
        
        Ok(Self { 
            device_id,
            device,
            allocations: std::sync::Mutex::new(std::collections::HashMap::new()),
        })
    }

    pub fn allocate(&self, size: usize) -> Result<*mut c_void> {
        // Now actually tries to allocate GPU memory through Candle
        match &self.device {
            candle_core::Device::Cuda(cuda_dev) => {
                // Create a tensor to allocate GPU memory
                let num_elements = size / std::mem::size_of::<f32>();
                match Tensor::zeros((num_elements,), DType::F32, &self.device) {
                    Ok(tensor) => {
                        // Get raw pointer from tensor storage
                        let storage = tensor.storage();
                        match storage.as_any().downcast_ref::<candle_core::CudaStorage>() {
                            Some(cuda_storage) => {
                                let ptr = cuda_storage.as_cuda_slice::<f32>()?.as_ptr() as *mut c_void;
                                
                                // Track allocation
                                let layout = std::alloc::Layout::from_size_align(size, 256)
                                    .map_err(|_| MemoryError::AllocationFailed(size))?;
                                {
                                    let mut allocations = self.allocations.lock().unwrap();
                                    allocations.insert(ptr, (size, layout));
                                }
                                Ok(ptr)
                            }
                            None => Err(MemoryError::AllocationFailed(size))
                        }
                    }
                    Err(e) => Err(MemoryError::CudaError(format!("CUDA allocation failed: {}", e)))
                }
            }
            candle_core::Device::Cpu => {
                // Fallback to aligned CPU allocation
                let layout = std::alloc::Layout::from_size_align(size, 256)
                    .map_err(|_| MemoryError::AllocationFailed(size))?;
                
                let ptr = unsafe {
                    let ptr = std::alloc::alloc(layout) as *mut c_void;
                    if ptr.is_null() {
                        return Err(MemoryError::AllocationFailed(size));
                    }
                    ptr
                };
                
                // Track allocation
                {
                    let mut allocations = self.allocations.lock().unwrap();
                    allocations.insert(ptr, (size, layout));
                }
                
                Ok(ptr)
            }
            _ => Err(MemoryError::CudaError("Unsupported device type".to_string()))
        }
    }

    pub fn deallocate(&self, ptr: *mut c_void) -> Result<()> {
        if ptr.is_null() {
            return Ok(());
        }
        
        // Find and remove allocation
        let (size, layout) = {
            let mut allocations = self.allocations.lock().unwrap();
            allocations.remove(&ptr)
                .ok_or(MemoryError::MemoryNotFound)?
        };
        
        // Deallocate
        unsafe {
            std::alloc::dealloc(ptr as *mut u8, layout);
        }
        
        Ok(())
    }

    pub fn get_memory_info(&self) -> Result<(usize, usize)> {
        // Now actually tries to get GPU memory info
        match &self.device {
            candle_core::Device::Cuda(cuda_device) => {
                // Try to create a large tensor to probe available memory
                let mut available = 0usize;
                let mut total = 24 * 1024 * 1024 * 1024; // Default to 24GB
                
                // Binary search for available memory
                let mut low = 0usize;
                let mut high = total;
                
                while low < high {
                    let mid = (low + high + 1) / 2;
                    let test_size = mid / std::mem::size_of::<f32>();
                    
                    // Try to allocate
                    match Tensor::zeros((test_size,), DType::F32, &self.device) {
                        Ok(_) => {
                            available = mid;
                            low = mid;
                        }
                        Err(_) => {
                            high = mid - 1;
                        }
                    }
                    
                    // Don't search too precisely
                    if high - low < 100 * 1024 * 1024 { // 100MB precision
                        break;
                    }
                }
                
                Ok((available, total))
            }
            candle_core::Device::Cpu => {
                // For CPU, try to get actual system memory
                #[cfg(target_os = "linux")]
                {
                    use std::fs;
                    if let Ok(meminfo) = fs::read_to_string("/proc/meminfo") {
                        let mut total_kb = 0u64;
                        let mut available_kb = 0u64;
                        
                        for line in meminfo.lines() {
                            if line.starts_with("MemTotal:") {
                                if let Some(val) = line.split_whitespace().nth(1) {
                                    total_kb = val.parse().unwrap_or(0);
                                }
                            } else if line.starts_with("MemAvailable:") {
                                if let Some(val) = line.split_whitespace().nth(1) {
                                    available_kb = val.parse().unwrap_or(0);
                                }
                            }
                        }
                        
                        if total_kb > 0 {
                            return Ok((available_kb as usize * 1024, total_kb as usize * 1024));
                        }
                    }
                }
                
                // Fallback
                let total: usize = 64 * 1024 * 1024 * 1024;
                let used: usize = {
                    let allocations = self.allocations.lock().unwrap();
                    allocations.values().map(|(size, _)| *size).sum::<usize>()
                };
                
                let free = total.saturating_sub(used);
                Ok((free, total))
            }
            _ => Err(MemoryError::CudaError("Unsupported device type".to_string()))
        }
    }
}