//! Checkpoint saving and loading utilities for training
//!
//! This module provides functionality to save and load training checkpoints,
//! including model weights, optimizer states, and training metadata.

use flame_core::{DType, Device, Error, Result, Tensor};
use log::{debug, info, warn};
use safetensors::{serialize, tensor::TensorView, Dtype as SafeDtype, SafeTensors};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Training checkpoint metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    /// Global training step
    pub global_step: usize,
    /// Current epoch
    pub epoch: usize,
    /// Best loss achieved
    pub best_loss: f32,
    /// Total training time in seconds
    pub total_time_seconds: u64,
    /// Timestamp when checkpoint was created
    pub timestamp: String,
    /// Training configuration hash for validation
    pub config_hash: String,
    /// Model architecture version
    pub model_version: String,
    /// Additional custom metadata
    pub custom_metadata: HashMap<String, String>,
}

/// Checkpoint saver for training
pub struct CheckpointSaver {
    /// Base directory for checkpoints
    checkpoint_dir: PathBuf,
    /// Maximum number of checkpoints to keep
    max_checkpoints: usize,
    /// Whether to save optimizer state
    save_optimizer_state: bool,
    /// Device for tensor operations
    device: Device,
}

impl CheckpointSaver {
    /// Create a new checkpoint saver
    pub fn new(checkpoint_dir: PathBuf, max_checkpoints: usize, device: Device) -> Result<Self> {
        // Create checkpoint directory
        fs::create_dir_all(&checkpoint_dir)
            .map_err(|e| Error::Io(format!("Failed to create checkpoint dir: {}", e)))?;

        Ok(Self { checkpoint_dir, max_checkpoints, save_optimizer_state: true, device })
    }

    /// Expose the output directory (for auxiliary logs like telemetry)
    pub fn output_dir(&self) -> &PathBuf {
        &self.checkpoint_dir
    }

    /// Save a checkpoint
    pub fn save_checkpoint(
        &self,
        step: usize,
        epoch: usize,
        loss: f32,
        model_weights: &HashMap<String, Tensor>,
        optimizer_state: Option<&HashMap<String, Tensor>>,
        metadata: Option<HashMap<String, String>>,
    ) -> Result<PathBuf> {
        let checkpoint_name = format!("checkpoint-{:06}", step);
        let checkpoint_path = self.checkpoint_dir.join(&checkpoint_name);

        // Create checkpoint directory
        fs::create_dir_all(&checkpoint_path)
            .map_err(|e| Error::Io(format!("Failed to create checkpoint dir: {}", e)))?;

        info!("Saving checkpoint at step {} to {:?}", step, checkpoint_path);

        // Save model weights
        self.save_safetensors(&checkpoint_path.join("model.safetensors"), model_weights, "model")?;

        // Save optimizer state if requested
        if self.save_optimizer_state {
            if let Some(opt_state) = optimizer_state {
                self.save_safetensors(
                    &checkpoint_path.join("optimizer.safetensors"),
                    opt_state,
                    "optimizer",
                )?;
            }
        }

        // Create metadata
        let metadata = CheckpointMetadata {
            global_step: step,
            epoch,
            best_loss: loss,
            total_time_seconds: 0, // TODO: Track actual training time
            timestamp: chrono::Utc::now().to_rfc3339(),
            config_hash: "".to_string(), // TODO: Compute config hash
            model_version: "1.0".to_string(),
            custom_metadata: metadata.unwrap_or_default(),
        };

        // Save metadata
        let metadata_json = serde_json::to_string_pretty(&metadata)
            .map_err(|e| Error::InvalidOperation(format!("Failed to serialize metadata: {}", e)))?;
        fs::write(checkpoint_path.join("metadata.json"), metadata_json)
            .map_err(|e| Error::Io(format!("Failed to write metadata: {}", e)))?;

        // Cleanup old checkpoints
        self.cleanup_old_checkpoints()?;

        info!("Checkpoint saved successfully");
        Ok(checkpoint_path)
    }

    /// Save tensors to safetensors format
    fn save_safetensors(
        &self,
        path: &Path,
        tensors: &HashMap<String, Tensor>,
        prefix: &str,
    ) -> Result<()> {
        let mut tensor_data = HashMap::new();
        let mut tensor_info = HashMap::new();

        for (name, tensor) in tensors {
            let key = if prefix.is_empty() { name.clone() } else { format!("{}.{}", prefix, name) };

            // Get tensor data based on dtype
            let (bytes, dtype) = match tensor.dtype() {
                DType::F32 => {
                    let data = tensor.to_vec1::<f32>()?;
                    let bytes: Vec<u8> = data.iter().flat_map(|&f| f.to_le_bytes()).collect();
                    (bytes, SafeDtype::F32)
                }
                DType::F16 => {
                    // For F16, we need to handle differently
                    // Convert to F32 first, then save as F32
                    let data = tensor.to_vec1::<f32>()?;
                    let bytes: Vec<u8> = data.iter().flat_map(|&f| f.to_le_bytes()).collect();
                    (bytes, SafeDtype::F32)
                }
                DType::BF16 => {
                    // For BF16, convert to F32
                    let data = tensor.to_vec1::<f32>()?;
                    let bytes: Vec<u8> = data.iter().flat_map(|&f| f.to_le_bytes()).collect();
                    (bytes, SafeDtype::F32)
                }
                _ => {
                    warn!("Unsupported dtype {:?} for tensor {}, skipping", tensor.dtype(), name);
                    continue;
                }
            };

            // Store tensor data
            tensor_data.insert(key.clone(), bytes);

            // Create tensor view
            let shape = tensor.shape().dims().to_vec();
            tensor_info.insert(
                key,
                TensorInfo {
                    dtype,
                    shape,
                    data_offsets: (0, 0), // Will be set by safetensors
                },
            );
        }

        // Create safetensors data
        let mut safetensors_data = HashMap::new();
        for (key, bytes) in &tensor_data {
            let info = &tensor_info[key];
            let tensor_view = TensorView::new(info.dtype, info.shape.clone(), bytes)
                .map_err(|e| Error::InvalidOperation(e.to_string()))?;
            safetensors_data.insert(key.clone(), tensor_view);
        }

        // Add metadata
        let mut metadata = HashMap::new();
        metadata.insert("format".to_string(), "pt".to_string());
        metadata.insert("creator".to_string(), "flame-diffusers".to_string());

        // Serialize
        let serialized = serialize(safetensors_data, &Some(metadata))
            .map_err(|e| Error::InvalidOperation(e.to_string()))?;

        // Write to file
        fs::write(path, serialized).map_err(|e| Error::Io(e.to_string()))?;

        debug!("Saved {} tensors to {:?}", tensors.len(), path);
        Ok(())
    }

    /// Load a checkpoint
    pub fn load_checkpoint(
        &self,
        checkpoint_path: &Path,
    ) -> Result<(HashMap<String, Tensor>, Option<HashMap<String, Tensor>>, CheckpointMetadata)>
    {
        info!("Loading checkpoint from {:?}", checkpoint_path);

        // Load metadata
        let metadata_path = checkpoint_path.join("metadata.json");
        let metadata_json = fs::read_to_string(&metadata_path)
            .map_err(|e| Error::Io(format!("Failed to read metadata: {}", e)))?;
        let metadata: CheckpointMetadata = serde_json::from_str(&metadata_json)
            .map_err(|e| Error::InvalidOperation(format!("Failed to parse metadata: {}", e)))?;

        // Load model weights
        let model_path = checkpoint_path.join("model.safetensors");
        let model_weights = self.load_safetensors(&model_path)?;

        // Load optimizer state if exists
        let optimizer_path = checkpoint_path.join("optimizer.safetensors");
        let optimizer_state = if optimizer_path.exists() {
            Some(self.load_safetensors(&optimizer_path)?)
        } else {
            None
        };

        info!(
            "Checkpoint loaded successfully (step: {}, epoch: {})",
            metadata.global_step, metadata.epoch
        );

        Ok((model_weights, optimizer_state, metadata))
    }

    /// Load tensors from safetensors format
    fn load_safetensors(&self, path: &Path) -> Result<HashMap<String, Tensor>> {
        let data =
            fs::read(path).map_err(|e| Error::Io(format!("Failed to read safetensors: {}", e)))?;

        let tensors = SafeTensors::deserialize(&data)
            .map_err(|e| Error::InvalidOperation(format!("Failed to deserialize: {}", e)))?;

        let mut result = HashMap::new();

        for (name, tensor_view) in tensors.tensors() {
            let shape = tensor_view.shape();
            let dtype = tensor_view.dtype();

            // Convert to FLAME tensor
            let tensor = match dtype {
                SafeDtype::F32 => {
                    let data: Vec<f32> = tensor_view
                        .data()
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                        .collect();
                    Tensor::from_vec(
                        data,
                        flame_core::Shape::from_dims(shape),
                        self.device.cuda_device_arc(),
                    )?
                }
                SafeDtype::F16 | SafeDtype::BF16 => {
                    // For now, convert to F32
                    // TODO: Proper F16/BF16 support
                    warn!("Converting {:?} to F32 for tensor {}", dtype, name);
                    let data: Vec<f32> = tensor_view
                        .data()
                        .chunks_exact(2)
                        .map(|b| {
                            // This is a simplified conversion
                            // TODO: Proper half precision conversion
                            1.0f32
                        })
                        .collect();
                    Tensor::from_vec(
                        data,
                        flame_core::Shape::from_dims(shape),
                        self.device.cuda_device_arc(),
                    )?
                }
                _ => {
                    warn!("Unsupported dtype {:?} for tensor {}, skipping", dtype, name);
                    continue;
                }
            };

            result.insert(name.to_string(), tensor);
        }

        Ok(result)
    }

    /// Find the latest checkpoint
    pub fn find_latest_checkpoint(&self) -> Result<Option<PathBuf>> {
        let mut checkpoints = Vec::new();

        // List all checkpoint directories
        let entries = fs::read_dir(&self.checkpoint_dir)
            .map_err(|e| Error::Io(format!("Failed to read checkpoint dir: {}", e)))?;

        for entry in entries {
            let entry = entry.map_err(|e| Error::Io(e.to_string()))?;
            let path = entry.path();

            if path.is_dir() {
                if let Some(name) = path.file_name() {
                    if let Some(name_str) = name.to_str() {
                        if name_str.starts_with("checkpoint-") {
                            // Extract step number
                            if let Some(step_str) = name_str.strip_prefix("checkpoint-") {
                                if let Ok(step) = step_str.parse::<usize>() {
                                    checkpoints.push((step, path));
                                }
                            }
                        }
                    }
                }
            }
        }

        // Sort by step number
        checkpoints.sort_by_key(|(step, _)| *step);

        // Return the latest
        Ok(checkpoints.last().map(|(_, path)| path.clone()))
    }

    /// Cleanup old checkpoints
    fn cleanup_old_checkpoints(&self) -> Result<()> {
        if self.max_checkpoints == 0 {
            return Ok(()); // Keep all checkpoints
        }

        let mut checkpoints = Vec::new();

        // List all checkpoint directories
        let entries = fs::read_dir(&self.checkpoint_dir)
            .map_err(|e| Error::Io(format!("Failed to read checkpoint dir: {}", e)))?;

        for entry in entries {
            let entry = entry.map_err(|e| Error::Io(e.to_string()))?;
            let path = entry.path();

            if path.is_dir() {
                if let Some(name) = path.file_name() {
                    if let Some(name_str) = name.to_str() {
                        if name_str.starts_with("checkpoint-") {
                            // Extract step number
                            if let Some(step_str) = name_str.strip_prefix("checkpoint-") {
                                if let Ok(step) = step_str.parse::<usize>() {
                                    checkpoints.push((step, path));
                                }
                            }
                        }
                    }
                }
            }
        }

        // Sort by step number (newest first)
        checkpoints.sort_by_key(|(step, _)| std::cmp::Reverse(*step));

        // Remove old checkpoints
        for (_, path) in checkpoints.iter().skip(self.max_checkpoints) {
            info!("Removing old checkpoint: {:?}", path);
            fs::remove_dir_all(path)
                .map_err(|e| Error::Io(format!("Failed to remove checkpoint: {}", e)))?;
        }

        Ok(())
    }

    /// Set whether to save optimizer state
    pub fn set_save_optimizer_state(&mut self, save: bool) {
        self.save_optimizer_state = save;
    }
}

/// Helper struct for tensor info
#[derive(Debug)]
struct TensorInfo {
    dtype: SafeDtype,
    shape: Vec<usize>,
    data_offsets: (usize, usize),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_metadata_serialization() {
        let metadata = CheckpointMetadata {
            global_step: 1000,
            epoch: 5,
            best_loss: 0.123,
            total_time_seconds: 3600,
            timestamp: "2024-01-01T00:00:00Z".to_string(),
            config_hash: "abc123".to_string(),
            model_version: "1.0".to_string(),
            custom_metadata: HashMap::new(),
        };

        let json = serde_json::to_string(&metadata).unwrap();
        let deserialized: CheckpointMetadata = serde_json::from_str(&json).unwrap();

        assert_eq!(metadata.global_step, deserialized.global_step);
        assert_eq!(metadata.epoch, deserialized.epoch);
        assert_eq!(metadata.best_loss, deserialized.best_loss);
    }
}
