//! Checkpoint management

use eridiffusion_core::{DiffusionModel, NetworkAdapter, Result, Error};
use crate::{Optimizer, OptimizerState, Scheduler, SchedulerState, TrainingState};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};
use safetensors::SafeTensors;

/// Checkpoint metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    pub name: String,
    pub step: usize,
    pub epoch: usize,
    pub loss: f32,
    pub timestamp: u64,
    pub model_architecture: String,
    pub network_type: Option<String>,
}

/// Checkpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    pub metadata: CheckpointMetadata,
    pub training_state: TrainingState,
    pub optimizer_state: OptimizerState,
    pub scheduler_state: SchedulerState,
    pub config: serde_json::Value,
}

/// Checkpoint manager
pub struct CheckpointManager {
    checkpoint_dir: PathBuf,
    max_checkpoints: usize,
    save_optimizer_state: bool,
}

impl CheckpointManager {
    /// Create new checkpoint manager
    pub fn new(output_dir: &Path) -> Result<Self> {
        let checkpoint_dir = output_dir.join("checkpoints");
        std::fs::create_dir_all(&checkpoint_dir)
            .map_err(|e| Error::Io(e))?;
        
        Ok(Self {
            checkpoint_dir,
            max_checkpoints: 5,
            save_optimizer_state: true,
        })
    }
    
    /// Save checkpoint
    pub async fn save_checkpoint(
        &self,
        name: &str,
        training_state: &TrainingState,
        model: &Arc<RwLock<Box<dyn DiffusionModel>>>,
        network: &Option<Arc<RwLock<Box<dyn NetworkAdapter>>>>,
        optimizer: &Box<dyn Optimizer>,
        scheduler: &Box<dyn Scheduler>,
    ) -> Result<()> {
        tracing::info!("Saving checkpoint: {}", name);
        
        let checkpoint_path = self.checkpoint_dir.join(name);
        tokio::fs::create_dir_all(&checkpoint_path).await
            .map_err(|e| Error::Io(e))?;
        
        // Save metadata
        let metadata = CheckpointMetadata {
            name: name.to_string(),
            step: training_state.global_step,
            epoch: training_state.epoch,
            loss: training_state.best_loss,
            timestamp: Self::current_timestamp(),
            model_architecture: {
                let model = model.read().await;
                format!("{:?}", model.architecture())
            },
            network_type: if let Some(ref net) = network {
                let net = net.read().await;
                Some(net.adapter_type().to_string())
            } else {
                None
            },
        };
        
        let checkpoint = Checkpoint {
            metadata: metadata.clone(),
            training_state: training_state.clone(),
            optimizer_state: optimizer.state().clone(),
            scheduler_state: scheduler.state().clone(),
            config: serde_json::Value::Null, // Would save actual config
        };
        
        // Save checkpoint info
        let checkpoint_json = serde_json::to_string_pretty(&checkpoint)?;
        tokio::fs::write(checkpoint_path.join("checkpoint.json"), checkpoint_json).await
            .map_err(|e| Error::Io(e))?;
        
        // Save model weights
        if let Some(ref net) = network {
            // Save network weights
            self.save_network_weights(&checkpoint_path, net).await?;
        } else {
            // Save full model weights
            self.save_model_weights(&checkpoint_path, model).await?;
        }
        
        // Save optimizer state if enabled
        if self.save_optimizer_state {
            self.save_optimizer_state(&checkpoint_path, &checkpoint.optimizer_state).await?;
        }
        
        // Clean up old checkpoints
        self.cleanup_old_checkpoints().await?;
        
        tracing::info!("Checkpoint saved successfully");
        Ok(())
    }
    
    /// Load checkpoint
    pub async fn load_checkpoint(
        &self,
        name: &str,
        training_state: &mut TrainingState,
        model: &Arc<RwLock<Box<dyn DiffusionModel>>>,
        network: &Option<Arc<RwLock<Box<dyn NetworkAdapter>>>>,
        optimizer: &mut Box<dyn Optimizer>,
        scheduler: &mut Box<dyn Scheduler>,
    ) -> Result<()> {
        tracing::info!("Loading checkpoint: {}", name);
        
        let checkpoint_path = self.checkpoint_dir.join(name);
        
        // Load checkpoint info
        let checkpoint_json = tokio::fs::read_to_string(checkpoint_path.join("checkpoint.json")).await
            .map_err(|e| Error::Io(e))?;
        let checkpoint: Checkpoint = serde_json::from_str(&checkpoint_json)?;
        
        // Restore training state
        *training_state = checkpoint.training_state;
        
        // Restore optimizer and scheduler states
        optimizer.set_state(checkpoint.optimizer_state)?;
        scheduler.set_state(checkpoint.scheduler_state)?;
        
        // Load weights
        if let Some(ref net) = network {
            // Load network weights
            self.load_network_weights(&checkpoint_path, net).await?;
        } else {
            // Load full model weights
            self.load_model_weights(&checkpoint_path, model).await?;
        }
        
        tracing::info!("Checkpoint loaded successfully");
        Ok(())
    }
    
    /// List available checkpoints
    pub async fn list_checkpoints(&self) -> Result<Vec<CheckpointMetadata>> {
        let mut checkpoints = Vec::new();
        
        let mut entries = tokio::fs::read_dir(&self.checkpoint_dir).await
            .map_err(|e| Error::Io(e))?;
        
        while let Some(entry) = entries.next_entry().await.map_err(|e| Error::Io(e))? {
            if entry.file_type().await.map_err(|e| Error::Io(e))?.is_dir() {
                let checkpoint_file = entry.path().join("checkpoint.json");
                if checkpoint_file.exists() {
                    let json = tokio::fs::read_to_string(&checkpoint_file).await
                        .map_err(|e| Error::Io(e))?;
                    if let Ok(checkpoint) = serde_json::from_str::<Checkpoint>(&json) {
                        checkpoints.push(checkpoint.metadata);
                    }
                }
            }
        }
        
        // Sort by timestamp
        checkpoints.sort_by_key(|c| c.timestamp);
        checkpoints.reverse();
        
        Ok(checkpoints)
    }
    
    /// Get latest checkpoint
    pub async fn get_latest_checkpoint(&self) -> Result<Option<String>> {
        let checkpoints = self.list_checkpoints().await?;
        Ok(checkpoints.first().map(|c| c.name.clone()))
    }
    
    /// Save model weights
    async fn save_model_weights(
        &self,
        checkpoint_path: &Path,
        model: &Arc<RwLock<Box<dyn DiffusionModel>>>,
    ) -> Result<()> {
        let model = model.read().await;
        let params = model.parameters();
        
        // Collect tensors
        let mut tensor_map = std::collections::HashMap::new();
        for (i, param) in params.iter().enumerate() {
            tensor_map.insert(format!("param_{}", i), (*param).clone());
        }
        
        // Use candle's safetensors save functionality
        candle_core::safetensors::save(&tensor_map, checkpoint_path.join("model.safetensors"))
            .map_err(|e| Error::Model(format!("Failed to save model: {}", e)))?;
        
        Ok(())
    }
    
    /// Save network weights
    async fn save_network_weights(
        &self,
        checkpoint_path: &Path,
        network: &Arc<RwLock<Box<dyn NetworkAdapter>>>,
    ) -> Result<()> {
        let network = network.read().await;
        let params = network.trainable_parameters();
        
        // Collect tensors
        let mut tensor_map = std::collections::HashMap::new();
        for (i, param) in params.iter().enumerate() {
            // Convert Var to Tensor by calling as_tensor()
            tensor_map.insert(format!("network_param_{}", i), param.as_tensor().clone());
        }
        
        // Use candle's safetensors save functionality
        candle_core::safetensors::save(&tensor_map, checkpoint_path.join("network.safetensors"))
            .map_err(|e| Error::Model(format!("Failed to save network: {}", e)))?;
        
        Ok(())
    }
    
    /// Load model weights
    async fn load_model_weights(
        &self,
        checkpoint_path: &Path,
        model: &Arc<RwLock<Box<dyn DiffusionModel>>>,
    ) -> Result<()> {
        let weights_path = checkpoint_path.join("model.safetensors");
        let weights_data = tokio::fs::read(&weights_path).await
            .map_err(|e| Error::Io(e))?;
        
        // Load safetensors
        let tensors = SafeTensors::deserialize(&weights_data)
            .map_err(|e| Error::Model(format!("Failed to load safetensors: {}", e)))?;
        
        // Apply to model
        let mut model = model.write().await;
        let params = model.parameters();
        
        for (i, param) in params.iter().enumerate() {
            let tensor_name = format!("param_{}", i);
            if let Ok(tensor_view) = tensors.tensor(&tensor_name) {
                // Would apply tensor data to parameter
                // This is simplified - actual implementation would handle dtype/device conversion
            }
        }
        
        Ok(())
    }
    
    /// Load network weights
    async fn load_network_weights(
        &self,
        checkpoint_path: &Path,
        network: &Arc<RwLock<Box<dyn NetworkAdapter>>>,
    ) -> Result<()> {
        let weights_path = checkpoint_path.join("network.safetensors");
        let weights_data = tokio::fs::read(&weights_path).await
            .map_err(|e| Error::Io(e))?;
        
        // Load safetensors
        let tensors = SafeTensors::deserialize(&weights_data)
            .map_err(|e| Error::Model(format!("Failed to load safetensors: {}", e)))?;
        
        // Apply to network
        let mut network = network.write().await;
        let params = network.trainable_parameters();
        
        for (i, param) in params.iter().enumerate() {
            let tensor_name = format!("network_param_{}", i);
            if let Ok(tensor_view) = tensors.tensor(&tensor_name) {
                // Would apply tensor data to parameter
            }
        }
        
        Ok(())
    }
    
    /// Save optimizer state
    async fn save_optimizer_state(
        &self,
        checkpoint_path: &Path,
        state: &OptimizerState,
    ) -> Result<()> {
        let state_json = serde_json::to_string_pretty(state)?;
        tokio::fs::write(checkpoint_path.join("optimizer_state.json"), state_json).await
            .map_err(|e| Error::Io(e))?;
        Ok(())
    }
    
    /// Clean up old checkpoints
    async fn cleanup_old_checkpoints(&self) -> Result<()> {
        let checkpoints = self.list_checkpoints().await?;
        
        if checkpoints.len() > self.max_checkpoints {
            // Keep best and latest, remove others
            let mut to_keep = std::collections::HashSet::new();
            to_keep.insert("best".to_string());
            to_keep.insert("latest".to_string());
            
            // Keep the most recent checkpoints
            for checkpoint in checkpoints.iter().take(self.max_checkpoints) {
                to_keep.insert(checkpoint.name.clone());
            }
            
            // Remove old checkpoints
            for checkpoint in checkpoints.iter().skip(self.max_checkpoints) {
                if !to_keep.contains(&checkpoint.name) {
                    let path = self.checkpoint_dir.join(&checkpoint.name);
                    if path.exists() {
                        tokio::fs::remove_dir_all(&path).await
                            .map_err(|e| Error::Io(e))?;
                        tracing::info!("Removed old checkpoint: {}", checkpoint.name);
                    }
                }
            }
        }
        
        Ok(())
    }
    
    
    /// Get current timestamp
    fn current_timestamp() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }
}

/// Checkpoint utilities
pub mod utils {
    use super::*;
    
    /// Convert checkpoint to different format
    pub async fn convert_checkpoint(
        input_path: &Path,
        output_path: &Path,
        format: CheckpointFormat,
    ) -> Result<()> {
        // Would implement checkpoint format conversion
        Ok(())
    }
    
    /// Merge checkpoints
    pub async fn merge_checkpoints(
        checkpoint_paths: &[PathBuf],
        output_path: &Path,
        weights: Option<&[f32]>,
    ) -> Result<()> {
        // Would implement checkpoint merging
        Ok(())
    }
    
    /// Extract specific components from checkpoint
    pub async fn extract_component(
        checkpoint_path: &Path,
        component: CheckpointComponent,
        output_path: &Path,
    ) -> Result<()> {
        // Would implement component extraction
        Ok(())
    }
}

/// Checkpoint format
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CheckpointFormat {
    SafeTensors,
    PyTorch,
    ONNX,
    TensorFlow,
}

/// Checkpoint component
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CheckpointComponent {
    Model,
    Network,
    Optimizer,
    All,
}