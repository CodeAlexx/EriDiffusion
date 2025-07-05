use candle_core::{Device, Result, Tensor};

pub struct TrainingConfig {
    pub batch_size: usize,
    pub gradient_accumulation_steps: usize,
    pub mixed_precision: bool,
    pub gradient_checkpointing: bool,
    pub learning_rate: f64,
    pub weight_decay: f64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            batch_size: 1,
            gradient_accumulation_steps: 4,
            mixed_precision: true,
            gradient_checkpointing: true,
            learning_rate: 1e-4,
            weight_decay: 0.01,
        }
    }
}

pub fn calculate_optimal_batch_size(
    total_gpu_memory_mb: usize,
    model_params: usize,
    sequence_length: usize,
    hidden_size: usize,
) -> usize {
    // Rough estimates in MB
    let param_memory = (model_params * 4) / (1024 * 1024); // f32 params
    let optimizer_memory = param_memory * 2; // Adam optimizer states
    let base_memory = param_memory + optimizer_memory + 1024; // +1GB overhead
    
    // Activation memory per sample (rough estimate)
    let layers = 24; // Typical for SD 3.5
    let attention_memory_per_sample = (sequence_length * sequence_length * 4 * layers) / (1024 * 1024);
    let ffn_memory_per_sample = (sequence_length * hidden_size * 8 * layers) / (1024 * 1024);
    let activation_memory_per_sample = attention_memory_per_sample + ffn_memory_per_sample;
    
    let available_memory = total_gpu_memory_mb.saturating_sub(base_memory);
    let batch_size = available_memory / activation_memory_per_sample;
    
    batch_size.max(1)
}

// Helper to get GPU memory
pub fn get_gpu_memory_mb() -> Result<usize> {
    #[cfg(feature = "cuda")]
    {
        use cudarc::driver::CudaDevice;
        let device = CudaDevice::new(0).map_err(|e| candle_core::Error::Cuda(Box::new(e)))?;
        let (free, total) = device.mem_info().map_err(|e| candle_core::Error::Cuda(Box::new(e)))?;
        Ok((total / (1024 * 1024)) as usize)
    }
    
    #[cfg(not(feature = "cuda"))]
    {
        Ok(8192) // Default 8GB
    }
}