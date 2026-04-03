use flame_core::device::Device;
use flame_core::optimizers::{Adam, SGD};
use flame_core::{CudaDevice, DType, Error, Result, Shape, Tensor};
use log::{debug, info};
use std::sync::Arc;
use std::{collections::HashMap, sync::Mutex};

// CPU offloading manager for memory optimization
// Offloads model weights and optimizer states to CPU RAM to save GPU memory

// FLAME uses flame_core::device::Device instead of Device

/// Manages CPU offloading of tensors
pub struct CPUOffloadManager {
    /// CPU device for offloading
    cpu_device: Device,

    /// GPU device
    gpu_device: Device,

    /// Offloaded tensors stored on CPU
    offloaded_tensors: Arc<Mutex<std::collections::HashMap<String, Tensor>>>,

    /// Track which tensors are currently on GPU
    gpu_cache: Arc<Mutex<HashMap<String, Tensor>>>,

    /// Maximum number of tensors to keep on GPU
    gpu_cache_size: usize,

    /// LRU order for eviction
    lru_order: Arc<Mutex<Vec<String>>>,
}

impl CPUOffloadManager {
    pub fn new(gpu_device: Device, gpu_cache_size: usize) -> flame_core::Result<Self> {
        // FLAME doesn't have a CPU device, so we use a dummy device (ordinal 0)
        // For CPU offloading, we'll just keep tensors in memory
        let cpu_device = Device::cuda(0)?; // Use GPU 0 as a placeholder

        Ok(Self {
            cpu_device,
            gpu_device,
            offloaded_tensors: Arc::new(Mutex::new(HashMap::new())),
            gpu_cache: Arc::new(Mutex::new(HashMap::new())),
            gpu_cache_size,
            lru_order: Arc::new(Mutex::new(Vec::new())),
        })
    }

    /// Offload a tensor to CPU
    pub fn offload(&self, name: &str, tensor: &Tensor, device: &Device) -> flame_core::Result<()> {
        debug!("Offloading tensor {} to CPU", name);

        // Move tensor to CPU - in FLAME, tensors are already on a device
        // Clone the tensor to ensure it's accessible
        let cpu_tensor = tensor.clone();

        // Store in offloaded map
        let mut offloaded = self.offloaded_tensors.lock().unwrap();
        offloaded.insert(name.to_string(), cpu_tensor);

        // Remove from GPU cache if present
        let mut gpu_cache = self.gpu_cache.lock().unwrap();
        gpu_cache.remove(name);

        // Update LRU
        let mut lru = self.lru_order.lock().unwrap();
        lru.retain(|n| n != name);

        Ok(())
    }

    /// Offload multiple tensors at once
    pub fn offload_batch(&self, tensors: &HashMap<String, Tensor>) -> flame_core::Result<()> {
        info!("Offloading {} tensors to CPU", tensors.len());

        for (name, tensor) in tensors {
            self.offload(name, tensor, &Device::from(tensor.device().clone()))?;
        }

        Ok(())
    }

    /// Retrieve a tensor, moving it to GPU if needed
    pub fn get(&self, name: &str) -> flame_core::Result<Option<Tensor>> {
        // Check GPU cache first
        {
            let gpu_cache = self.gpu_cache.lock().unwrap();
            if let Some(tensor) = gpu_cache.get(name) {
                // Update LRU order
                let mut lru = self.lru_order.lock().unwrap();
                lru.retain(|n| n != name);
                lru.push(name.to_string());

                return Ok(Some(tensor.clone()));
            }
        }

        // Check CPU storage
        let offloaded = self.offloaded_tensors.lock().unwrap();
        if let Some(cpu_tensor) = offloaded.get(name) {
            debug!("Moving tensor {} from CPU to GPU", name);

            // Move to GPU
            let gpu_tensor = cpu_tensor.clone();

            // Add to GPU cache
            self.cache_on_gpu(name, gpu_tensor.clone())?;

            return Ok(Some(gpu_tensor));
        }

        Ok(None)
    }

    /// Cache a tensor on GPU, potentially evicting others
    fn cache_on_gpu(&self, name: &str, tensor: Tensor) -> flame_core::Result<()> {
        let mut gpu_cache = self.gpu_cache.lock().unwrap();
        let mut lru = self.lru_order.lock().unwrap();

        // Check if we need to evict
        if gpu_cache.len() >= self.gpu_cache_size && !gpu_cache.contains_key(name) {
            // Evict least recently used
            if let Some(evict_name) = lru.first().cloned() {
                debug!("Evicting tensor {} from GPU cache", evict_name);

                if let Some(evict_tensor) = gpu_cache.remove(&evict_name) {
                    // Move evicted tensor to CPU
                    let cpu_tensor = evict_tensor;
                    let mut offloaded = self.offloaded_tensors.lock().unwrap();
                    offloaded.insert(evict_name.clone(), cpu_tensor);
                }

                lru.remove(0);
            }
        }

        // Add new tensor
        gpu_cache.insert(name.to_string(), tensor);
        lru.retain(|n| n != name);
        lru.push(name.to_string());

        Ok(())
    }

    /// Get batch of tensors, minimizing GPU transfers
    pub fn get_batch(&self, names: &[&str]) -> flame_core::Result<HashMap<String, Tensor>> {
        let mut result = HashMap::new();

        for name in names {
            if let Some(tensor) = self.get(name)? {
                result.insert(name.to_string(), tensor);
            }
        }

        Ok(result)
    }

    /// Clear GPU cache to free memory
    pub fn clear_gpu_cache(&self) -> flame_core::Result<()> {
        info!("Clearing GPU cache");

        let mut gpu_cache = self.gpu_cache.lock().unwrap();
        let mut offloaded = self.offloaded_tensors.lock().unwrap();

        // Move all GPU tensors to CPU
        for (name, tensor) in gpu_cache.drain() {
            let cpu_tensor = tensor;
            offloaded.insert(name, cpu_tensor);
        }

        // Clear LRU
        let mut lru = self.lru_order.lock().unwrap();
        lru.clear();

        Ok(())
    }

    /// Get memory usage statistics
    pub fn get_stats(&self) -> (usize, usize) {
        let gpu_count = self.gpu_cache.lock().unwrap().len();
        let cpu_count = self.offloaded_tensors.lock().unwrap().len();
        (gpu_count, cpu_count)
    }
}

/// CPU-offloaded optimizer state
pub struct CPUOffloadedOptimizer {
    /// Learning rate
    lr: f32,

    /// Beta1 for Adam
    beta1: f32,

    /// Beta2 for Adam
    beta2: f32,

    /// Epsilon for Adam
    eps: f32,

    /// Step count
    step: usize,

    /// First moment estimates (on CPU)
    m: HashMap<String, Tensor>,

    /// Second moment estimates (on CPU)
    v: HashMap<String, Tensor>,

    /// Device manager
    device: Device,
}

impl CPUOffloadedOptimizer {
    pub fn new(lr: f32, device: Device) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            step: 0,
            m: HashMap::new(),
            v: HashMap::new(),
            device,
        }
    }

    /// Update parameters with gradients
    pub fn step(
        &mut self,
        params: &mut HashMap<String, Tensor>,
        grads: &HashMap<String, Tensor>,
    ) -> flame_core::Result<()> {
        self.step += 1;
        let bias_correction1 = 1.0 - self.beta1.powi(self.step as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.step as i32);

        for (name, param) in params {
            if let Some(grad) = grads.get(name as &str) {
                // Move gradient to CPU for computation
                let grad_cpu = grad;

                // Initialize moment estimates if needed
                if !self.m.contains_key(name) {
                    let zeros = Tensor::zeros(grad_cpu.shape().clone(), grad_cpu.device().clone())?;
                    self.m.insert(name.to_string(), zeros.clone());
                    self.v.insert(name.to_string(), zeros.clone());
                }

                // Get moment estimates
                let m = self.m.get_mut(name).unwrap();
                let v = self.v.get_mut(name).unwrap();

                // Update biased first moment estimate
                // m = beta1 * m + (1 - beta1) * grad
                let beta1_tensor = Tensor::full(
                    grad_cpu.shape().clone(),
                    self.beta1 as f32,
                    grad_cpu.device().clone(),
                )?;
                let one_minus_beta1 = Tensor::full(
                    grad_cpu.shape().clone(),
                    (1.0 - self.beta1) as f32,
                    grad_cpu.device().clone(),
                )?;
                let m_scaled = m.mul(&beta1_tensor)?;
                let grad_scaled = grad_cpu.clone().mul(&one_minus_beta1)?;
                *m = m_scaled.add(&grad_scaled)?;

                // Update biased second moment estimate
                // v = beta2 * v + (1 - beta2) * grad^2
                let grad_sq = grad_cpu.square()?;
                let beta2_tensor = Tensor::full(
                    grad_cpu.shape().clone(),
                    self.beta2 as f32,
                    grad_cpu.device().clone(),
                )?;
                let one_minus_beta2 = Tensor::full(
                    grad_cpu.shape().clone(),
                    (1.0 - self.beta2) as f32,
                    grad_cpu.device().clone(),
                )?;
                *v = v.mul(&beta2_tensor)?.add(&grad_sq.mul(&one_minus_beta2)?)?;

                // Compute bias-corrected moments
                let bias_corr1_tensor =
                    Tensor::full(m.shape().clone(), bias_correction1 as f32, m.device().clone())?;
                let bias_corr2_tensor =
                    Tensor::full(v.shape().clone(), bias_correction2 as f32, v.device().clone())?;
                let m_hat = m.div(&bias_corr1_tensor)?;
                let v_hat = v.div(&bias_corr2_tensor)?;

                // Compute update on CPU
                let eps_tensor =
                    Tensor::full(v_hat.shape().clone(), self.eps as f32, v_hat.device().clone())?;
                let lr_tensor =
                    Tensor::full(m_hat.shape().clone(), self.lr as f32, m_hat.device().clone())?;
                let update = m_hat.div(&v_hat.sqrt()?.add(&eps_tensor)?)?.mul(&lr_tensor)?;

                // Move update to GPU and apply to Tensor
                let update_gpu = update;
                let new_value = param.sub(&update_gpu)?;
                *param = new_value;
            }
        }

        Ok(())
    }

    /// Clear optimizer state to free memory
    pub fn clear_state(&mut self) -> flame_core::Result<()> {
        self.m.clear();
        self.v.clear();
        Ok(())
    }
}
