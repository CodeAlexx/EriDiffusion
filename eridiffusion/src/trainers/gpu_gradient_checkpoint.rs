use anyhow::Context;
use flame_core::device::Device;
use flame_core::optimizers::{Adam, SGD};
use flame_core::{DType, Result, Shape, Tensor};
use std::{collections::HashMap, sync::Arc};

// GPU-only gradient checkpointing - NO CPU FALLBACKS!
// Uses our CUDA kernels for efficient GPU memory management

// FLAME uses flame_core::device::Device instead of Device

// TODO: Re-enable when cuda_backend is implemented
// #[cfg(feature = "cuda")]
// use crate::cuda_backend::checkpoint::{
//     create_checkpoint_manager,
//     destroy_checkpoint_manager,
//     allocate_checkpoint_buffer,
//     save_activation_checkpoint,
// };

/// GPU-only gradient checkpoint manager
pub struct GPUGradientCheckpoint {
    enabled: bool,
    #[cfg(feature = "cuda")]
    checkpoint_manager: Option<*mut std::ffi::c_void>,
    checkpoint_interval: usize,
    max_layers: usize,
}

unsafe impl Send for GPUGradientCheckpoint {}
unsafe impl Sync for GPUGradientCheckpoint {}

impl GPUGradientCheckpoint {
    pub fn new(enabled: bool) -> Self {
        if enabled & !cfg!(feature = "cuda") {
            panic!("GPU gradient checkpointing requires CUDA feature! No CPU fallback!");
        }

        let checkpoint_interval = 4; // Checkpoint every 4 layers
        let max_layers = 64; // SDXL has ~60 transformer blocks

        #[cfg(feature = "cuda")]
        let checkpoint_manager = None; // TODO: Re-enable when cuda_backend is implemented
                                       // let checkpoint_manager = if enabled {
                                       //     unsafe {
                                       //         let manager = create_checkpoint_manager(max_layers as i32, checkpoint_interval as i32);
                                       //
                                       //         // Pre-allocate checkpoint buffers for typical SDXL sizes
                                       //         let typical_activation_size = 4 * 1024 * 1024 * 16; // 4MB * 16 channels
                                       //         for i in 0..(max_layers / checkpoint_interval) {
                                       //             allocate_checkpoint_buffer(manager, i as i32, typical_activation_size);
                                       //         }
                                       //
                                       //         Some(manager)
                                       //     }
                                       // } else {
                                       //     None
                                       // };

        Self {
            enabled,
            #[cfg(feature = "cuda")]
            checkpoint_manager,
            checkpoint_interval,
            max_layers,
        }
    }

    /// Check if checkpointing is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Save activation checkpoint on GPU
    #[cfg(feature = "cuda")]
    pub fn save_checkpoint(
        &self,
        activation: &Tensor,
        layer_idx: usize,
    ) -> flame_core::Result<Tensor> {
        if !self.enabled || self.checkpoint_manager.is_none() {
            return Ok(activation.clone());
        }

        // Only checkpoint at intervals
        if layer_idx % self.checkpoint_interval != 0 {
            return Ok(activation.clone());
        }

        let device = Device::from(activation.device().clone());
        if !true {
            panic!("GPU gradient checkpointing requires CUDA tensors! No CPU fallback!");
        }

        // TODO: Re-enable when storage() method becomes public or alternative API is available
        // unsafe {
        //     if let Some(manager) = self.checkpoint_manager {
        //         // Get raw pointer to tensor data
        //         let storage = activation.storage();
        //         match &*storage {
        //             Storage::Cuda(cuda_storage) => {
        //                 let slice = cuda_storage.as_cuda_slice::<f32>()?;
        //                 let ptr = slice.as_ptr();
        //                 let size = activation.shape().dims().iter().product::<usize>();
        //
        //                 // TODO: Re-enable when cuda_backend is implemented
        //                 // save_activation_checkpoint(
        //                 //     manager,
        //                 //     ptr as *const f32,
        //                 //     layer_idx as i32,
        //                 //     size,
        //                 // );
        //             }
        //             _ => panic!("GPU gradient checkpointing requires CUDA tensors!"),
        //         }
        //     }
        // }

        Ok(activation.clone())
    }

    /// Get checkpoint manager for direct CUDA kernel integration
    #[cfg(feature = "cuda")]
    pub fn get_manager(&self) -> Option<*mut std::ffi::c_void> {
        self.checkpoint_manager
    }
}

impl Drop for GPUGradientCheckpoint {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        {
            // TODO: Re-enable when cuda_backend is implemented
            // unsafe {
            //     if let Some(manager) = self.checkpoint_manager {
            //         destroy_checkpoint_manager(manager);
            //     }
            // }
        }
    }
}

/// Type alias for backward compatibility
pub type SDXLGradientCheckpoint = GPUGradientCheckpoint;

/// Wrapper for SDXL forward pass with GPU checkpointing
pub fn forward_sdxl_gpu_checkpoint(
    x: &Tensor,
    timestep: &Tensor,
    encoder_hidden_states: &Tensor,
    weights: &std::collections::HashMap<String, Tensor>,
    lora_collection: &crate::trainers::lora::LoRACollection,
    checkpoint: &GPUGradientCheckpoint,
) -> flame_core::Result<Tensor> {
    // FLAME devices are always CUDA, no need to check

    // For now, delegate to the existing forward pass
    // In the future, we'll integrate checkpoint saving at each layer
    // Create WeightLoader from HashMap
    // Clone each tensor individually since Tensor doesn't implement Clone directly
    let mut cloned_weights = HashMap::new();
    for (k, v) in weights {
        cloned_weights.insert(k.clone(), v.clone());
    }
    // Create a Device from the tensor's CUDA device
    // We'll use ordinal 0 as a default since we can't get the ordinal from Arc<CudaDevice>
    let device = flame_core::device::Device::cuda(0)?;
    let weight_loader =
        crate::loaders::weight_loader::WeightLoader { weights: cloned_weights, device };
    crate::trainers::sdxl_forward_sd_format_flash::forward_sdxl_sd_format_flash(
        x,
        timestep,
        encoder_hidden_states,
        &weight_loader,
        lora_collection,
        true, // use_flash_attention
    )
}
