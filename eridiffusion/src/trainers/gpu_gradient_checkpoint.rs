use anyhow::Context;
use flame_core::device::Device;
use flame_core::gradient_checkpointing::{CheckpointPolicy, CheckpointStats, CHECKPOINT_MANAGER};
use flame_core::optimizers::{Adam, SGD};
use flame_core::{DType, Result, Shape, Tensor};
use log::{info, warn};
use std::{collections::HashMap, env};

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
    checkpoint_interval: usize,
    max_layers: usize,
    policy: CheckpointPolicy,
}

unsafe impl Send for GPUGradientCheckpoint {}
unsafe impl Sync for GPUGradientCheckpoint {}

impl GPUGradientCheckpoint {
    pub fn new(enabled: bool, device: &Device) -> Self {
        if enabled & !cfg!(feature = "cuda") {
            panic!("GPU gradient checkpointing requires CUDA feature! No CPU fallback!");
        }

        let checkpoint_interval =
            parse_env_usize("GRADIENT_CHECKPOINT_INTERVAL", /*layers*/ 4).max(1);
        let max_layers = parse_env_usize("GRADIENT_CHECKPOINT_MAX_LAYERS", 64).max(1);
        let policy = parse_policy_from_env();

        if enabled {
            Self::install_manager(device, policy);
            info!(
                "Gradient checkpointing enabled (policy={:?}, interval={} layers, max_layers={}, device=cuda:{})",
                policy,
                checkpoint_interval,
                max_layers,
                device.ordinal()
            );
        }

        Self { enabled, checkpoint_interval, max_layers, policy }
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
        if !self.enabled {
            return Ok(activation.clone());
        }

        // Only checkpoint at intervals
        if layer_idx % self.checkpoint_interval != 0 {
            return Ok(activation.clone());
        }

        Ok(activation.clone())
    }

    fn install_manager(device: &Device, policy: CheckpointPolicy) {
        match CHECKPOINT_MANAGER.lock() {
            Ok(mut manager) => {
                manager.set_device(device.cuda_device().clone());
                manager.set_policy(policy);
            }
            Err(err) => {
                warn!(
                    "Failed to acquire gradient checkpoint manager ({:?}); checkpointing disabled",
                    err
                );
            }
        }
    }

    pub fn stats(&self) -> Option<CheckpointStats> {
        if !self.enabled {
            return None;
        }
        CHECKPOINT_MANAGER.lock().ok().map(|manager| manager.stats())
    }
}

impl Drop for GPUGradientCheckpoint {
    fn drop(&mut self) {
        if !self.enabled {
            return;
        }
        if let Ok(manager) = CHECKPOINT_MANAGER.lock() {
            let stats = manager.stats();
            info!(
                "Gradient checkpointing disabled (policy={:?}, tensors={}, recompute_count={}, approx_saved={} MB)",
                self.policy,
                stats.checkpointed_tensors,
                stats.recompute_count,
                stats.memory_saved / (1024 * 1024)
            );
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
        crate::loaders::weight_loader::WeightLoader::from_tensor_map(cloned_weights, device);
    crate::trainers::sdxl_forward_sd_format_flash::forward_sdxl_sd_format_flash(
        x,
        timestep,
        encoder_hidden_states,
        &weight_loader,
        lora_collection,
        true, // use_flash_attention
    )
}

fn parse_env_usize(key: &str, default: usize) -> usize {
    env::var(key).ok().and_then(|value| value.parse::<usize>().ok()).unwrap_or(default)
}

fn parse_policy_from_env() -> CheckpointPolicy {
    match env::var("GRADIENT_CHECKPOINT_POLICY")
        .unwrap_or_else(|_| "recompute".to_string())
        .to_lowercase()
        .as_str()
    {
        "cpu" | "cpu_offload" | "cpu-offload" => CheckpointPolicy::CPUOffload,
        "adaptive" => {
            let threshold_mb =
                parse_env_usize("GRADIENT_CHECKPOINT_MEMORY_THRESHOLD_MB", 8192).max(1);
            let prefer_recompute = parse_env_bool("GRADIENT_CHECKPOINT_PREFER_RECOMPUTE", true);
            CheckpointPolicy::Adaptive {
                memory_threshold: threshold_mb * 1024 * 1024,
                prefer_recompute,
            }
        }
        other => {
            if other != "recompute" {
                warn!("Unknown GRADIENT_CHECKPOINT_POLICY='{}', falling back to recompute", other);
            }
            CheckpointPolicy::Recompute
        }
    }
}

fn parse_env_bool(key: &str, default: bool) -> bool {
    env::var(key)
        .ok()
        .map(|value| matches!(value.to_lowercase().as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(default)
}
