use anyhow::Context;
use flame_core::device::Device;
use flame_core::Result;
use log::{debug, info};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::trainers::lora::LoRACollection;

#[derive(Debug, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    pub global_step: usize,
    pub epoch: usize,
    pub loss: f32,
    pub learning_rate: f32,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub config_hash: String,
}

pub struct CheckpointManager {
    output_dir: PathBuf,
    max_checkpoints: usize,
    checkpoints: Vec<(usize, PathBuf)>, // (step, path)
}

impl CheckpointManager {
    pub fn new(output_dir: PathBuf, max_checkpoints: usize) -> Self {
        Self { output_dir, max_checkpoints, checkpoints: Vec::new() }
    }

    /// Save a checkpoint
    pub fn save(
        &mut self,
        step: usize,
        lora_collection: &LoRACollection,
        optimizer_state: Option<HashMap<String, flame_core::Tensor>>,
        metadata: CheckpointMetadata,
    ) -> flame_core::Result<PathBuf> {
        // Create checkpoint directory
        let checkpoint_dir = self.output_dir.join(format!("checkpoint-{}", step));
        std::fs::create_dir_all(&checkpoint_dir)
            .map_err(|e| {
                flame_core::Error::InvalidOperation(format!(
                    "Failed to create directory: {}",
                    e
                ))
            })
            .map_err(|e| flame_core::Error::InvalidOperation(e.to_string()))?;

        // Save LoRA weights
        let lora_path = checkpoint_dir.join("sdxl_lora.safetensors");
        lora_collection.save_weights(&lora_path)?;

        // Save optimizer state if provided
        if let Some(state) = optimizer_state {
            let optimizer_path = checkpoint_dir.join("optimizer.safetensors");
            save_optimizer_state(&optimizer_path, state)?;
        }

        // Save metadata
        let metadata_path = checkpoint_dir.join("metadata.json");
        let metadata_json = serde_json::to_string_pretty(&metadata)
            .map_err(|e| {
                flame_core::Error::InvalidOperation(format!("Failed to serialize JSON: {}", e))
            })
            .map_err(|e| flame_core::Error::InvalidOperation(e.to_string()))?;
        std::fs::write(&metadata_path, metadata_json)
            .map_err(|e| flame_core::Error::InvalidOperation(e.to_string()))?;

        // Track checkpoint
        self.checkpoints.push((step, checkpoint_dir.clone()));

        // Clean up old checkpoints
        self.cleanup_old_checkpoints()?;

        info!("Saved checkpoint at step {} to {:?}", step, checkpoint_dir);
        Ok(checkpoint_dir)
    }

    /// Load a checkpoint
    pub fn load(
        &self,
        checkpoint_path: &Path,
    ) -> flame_core::Result<(
        LoRACollection,
        Option<HashMap<String, flame_core::Tensor>>,
        CheckpointMetadata,
    )> {
        info!("Loading checkpoint from {:?}", checkpoint_path);

        // Load metadata
        let metadata_path = checkpoint_path.join("metadata.json");
        let metadata_str = std::fs::read_to_string(&metadata_path).map_err(|e| {
            flame_core::Error::InvalidOperation(format!(
                "Failed to read checkpoint metadata: {}",
                e
            ))
        })?;
        let metadata: CheckpointMetadata = serde_json::from_str(&metadata_str).map_err(|e| {
            flame_core::Error::InvalidOperation(format!(
                "Failed to parse checkpoint metadata: {}",
                e
            ))
        })?;

        // Load LoRA weights
        let lora_path = checkpoint_path.join("sdxl_lora.safetensors");
        let lora_collection = load_lora_weights(&lora_path, &metadata)?;

        // Load optimizer state if exists
        let optimizer_path = checkpoint_path.join("optimizer.safetensors");
        let optimizer_state = if optimizer_path.exists() {
            let device = flame_core::device::Device::cuda(0)?;
            Some(load_optimizer_state(&optimizer_path, &device)?)
        } else {
            None
        };

        Ok((lora_collection, optimizer_state, metadata))
    }

    /// Get the latest checkpoint
    pub fn get_latest(&self) -> Option<&PathBuf> {
        self.checkpoints.iter().max_by_key(|(step, _)| step).map(|(_, path)| path)
    }

    /// Clean up old checkpoints to maintain max_checkpoints limit
    fn cleanup_old_checkpoints(&mut self) -> flame_core::Result<()> {
        if self.checkpoints.len() <= self.max_checkpoints {
            return Ok(());
        }

        // Sort by step number
        self.checkpoints.sort_by_key(|(step, _)| *step);

        // Remove oldest checkpoints
        while self.checkpoints.len() > self.max_checkpoints {
            let (step, path) = self.checkpoints.remove(0);
            debug!("Removing old checkpoint at step {}: {:?}", step, path);
            std::fs::remove_dir_all(&path)
                .map_err(|e| flame_core::Error::InvalidOperation(e.to_string()))?;
        }

        Ok(())
    }
}

/// Save checkpoint
pub fn save_checkpoint(
    checkpoint_path: &Path,
    lora_collection: &LoRACollection,
    optimizer_state: Option<HashMap<String, flame_core::Tensor>>,
    training_state: &crate::trainers::training::TrainingState,
) -> flame_core::Result<()> {
    std::fs::create_dir_all(checkpoint_path)
        .map_err(|e| {
            flame_core::Error::InvalidOperation(format!("Failed to create directory: {}", e))
        })
        .map_err(|e| flame_core::Error::InvalidOperation(e.to_string()))?;

    // Save LoRA weights
    let lora_path = checkpoint_path.join("sdxl_lora.safetensors");
    lora_collection.save_weights(&lora_path)?;

    // Save training state
    let state_path = checkpoint_path.join("training_state.json");
    let state_json = serde_json::json!({
        "global_step": training_state.global_step,
        "epoch": training_state.epoch,
        "average_loss": training_state.average_loss(),
        "best_loss": training_state.best_loss,
        "learning_rate": training_state.learning_rate,
    });
    std::fs::write(
        state_path,
        serde_json::to_string_pretty(&state_json).map_err(|e| {
            flame_core::Error::InvalidOperation(format!("Failed to serialize JSON: {}", e))
        })?,
    )
    .map_err(|e| flame_core::Error::InvalidOperation(e.to_string()))?;

    // Save optimizer state if provided
    if let Some(state) = optimizer_state {
        let optimizer_path = checkpoint_path.join("optimizer.safetensors");
        save_optimizer_state(&optimizer_path, state)?;
    }

    Ok(())
}

/// Load checkpoint
pub fn load_checkpoint(
    checkpoint_path: &Path,
    device: &flame_core::device::Device,
) -> flame_core::Result<(HashMap<String, flame_core::Tensor>, CheckpointMetadata)> {
    // Load LoRA weights
    let lora_path = checkpoint_path.join("sdxl_lora.safetensors");
    let weight_loader = crate::loaders::WeightLoader::from_safetensors(&lora_path, device.clone())?;

    // Convert WeightLoader to HashMap
    let mut lora_weights = HashMap::new();
    for key in weight_loader.keys() {
        if let Ok(tensor) = weight_loader.get(key) {
            lora_weights.insert(key.clone(), tensor.clone());
        }
    }

    // Load metadata
    let metadata_path = checkpoint_path.join("metadata.json");
    let metadata_str = std::fs::read_to_string(&metadata_path).map_err(|e| {
        flame_core::Error::InvalidOperation(format!("Failed to read file: {}", e))
    })?;
    let metadata: CheckpointMetadata = serde_json::from_str(&metadata_str).map_err(|e| {
        flame_core::Error::InvalidOperation(format!("Failed to parse JSON: {}", e))
    })?;

    Ok((lora_weights, metadata))
}

fn save_optimizer_state(
    path: &Path,
    state: HashMap<String, flame_core::Tensor>,
) -> flame_core::Result<()> {
    use safetensors::{serialize, tensor::TensorView, Dtype as SafeDtype};

    // Collect all data first to ensure proper lifetime
    let mut tensor_data: HashMap<String, (Vec<usize>, Vec<u8>)> = HashMap::new();

    for (name, tensor) in state {
        let tensor_cpu = tensor;
        let shape = tensor_cpu.shape().dims().to_vec();
        let data_f32 = tensor_cpu.to_vec1::<f32>()?;

        // Convert f32 vec to bytes
        let mut bytes = Vec::with_capacity(data_f32.len() * 4);
        for val in data_f32 {
            bytes.extend_from_slice(&val.to_le_bytes());
        }

        tensor_data.insert(name.clone(), (shape, bytes));
    }

    // Now create TensorViews with proper references
    let mut tensors = HashMap::new();
    for (name, (shape, data)) in &tensor_data {
        tensors.insert(
            name.clone(),
            TensorView::new(SafeDtype::F32, shape.clone(), data).map_err(|e| {
                flame_core::Error::InvalidOperation(format!(
                    "Failed to create TensorView: {}",
                    e
                ))
            })?,
        );
    }

    let data = serialize(tensors, &None).map_err(|e| {
        flame_core::Error::InvalidOperation(format!("Failed to serialize tensors: {}", e))
    })?;
    std::fs::write(path, data)
        .map_err(|e| flame_core::Error::InvalidOperation(e.to_string()))?;

    Ok(())
}

fn load_optimizer_state(
    path: &Path,
    device: &flame_core::device::Device,
) -> flame_core::Result<HashMap<String, flame_core::Tensor>> {
    let device = flame_core::device::Device::cuda(0)?;
    let weight_loader = crate::loaders::WeightLoader::from_safetensors(path, device.clone())?;

    // Convert WeightLoader to HashMap
    let mut state = HashMap::new();
    for key in weight_loader.keys() {
        if let Ok(tensor) = weight_loader.get(key) {
            state.insert(key.clone(), tensor.clone());
        }
    }

    Ok(state)
}

fn load_lora_weights(
    path: &Path,
    metadata: &CheckpointMetadata,
) -> flame_core::Result<LoRACollection> {
    // This is a placeholder - actual implementation would recreate LoRACollection from weights
    return Err(flame_core::Error::InvalidOperation(
        "LoRA weight loading not yet implemented".to_string(),
    ));
}
