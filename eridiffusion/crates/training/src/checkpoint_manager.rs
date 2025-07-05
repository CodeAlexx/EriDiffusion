//! Checkpoint management for training

use eridiffusion_core::{Result, Error, ModelArchitecture};
use eridiffusion_models::DiffusionModel;
use eridiffusion_core::NetworkAdapter;
use candle_core::Tensor;
use safetensors::{serialize, SafeTensors};
use serde::{Serialize, Deserialize};
use std::path::{Path, PathBuf};
use std::collections::HashMap;
use std::fs;

/// Checkpoint metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    pub step: usize,
    pub epoch: usize,
    pub loss: f32,
    pub learning_rate: f32,
    pub model_architecture: String,
    pub network_type: String,
    pub network_rank: Option<usize>,
    pub network_alpha: Option<f32>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub training_config: serde_json::Value,
}

/// Checkpoint manager for saving and loading training state
pub struct CheckpointManager {
    output_dir: PathBuf,
    save_latest: usize,
    save_total_limit: Option<usize>,
}

impl CheckpointManager {
    /// Create new checkpoint manager
    pub fn new(output_dir: PathBuf, save_latest: usize, save_total_limit: Option<usize>) -> Result<Self> {
        // Create output directory
        fs::create_dir_all(&output_dir)?;
        
        Ok(Self {
            output_dir,
            save_latest,
            save_total_limit,
        })
    }
    
    /// Save checkpoint
    pub fn save_checkpoint(
        &self,
        step: usize,
        epoch: usize,
        loss: f32,
        learning_rate: f32,
        network_adapter: &dyn NetworkAdapter,
        optimizer_state: Option<HashMap<String, Tensor>>,
        training_config: &serde_json::Value,
    ) -> Result<PathBuf> {
        // Create checkpoint directory
        let checkpoint_dir = self.output_dir.join(format!("checkpoint-{}", step));
        fs::create_dir_all(&checkpoint_dir)?;
        
        // Save network adapter weights
        let adapter_path = checkpoint_dir.join("adapter_model.safetensors");
        self.save_network_adapter(network_adapter, &adapter_path)?;
        
        // Save optimizer state if provided
        if let Some(opt_state) = optimizer_state {
            let optimizer_path = checkpoint_dir.join("optimizer.safetensors");
            self.save_optimizer_state(opt_state, &optimizer_path)?;
        }
        
        // Create metadata
        let metadata = CheckpointMetadata {
            step,
            epoch,
            loss,
            learning_rate,
            model_architecture: network_adapter.metadata().base_model.clone(),
            network_type: format!("{:?}", network_adapter.adapter_type()),
            network_rank: network_adapter.metadata().rank,
            network_alpha: network_adapter.metadata().alpha,
            timestamp: chrono::Utc::now(),
            training_config: training_config.clone(),
        };
        
        // Save metadata
        let metadata_path = checkpoint_dir.join("metadata.json");
        let metadata_json = serde_json::to_string_pretty(&metadata)?;
        fs::write(&metadata_path, metadata_json)?;
        
        // Update latest symlink
        let latest_link = self.output_dir.join("checkpoint-latest");
        if latest_link.exists() {
            fs::remove_file(&latest_link)?;
        }
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(&checkpoint_dir, &latest_link)?;
        }
        
        // Clean up old checkpoints if needed
        self.cleanup_old_checkpoints()?;
        
        Ok(checkpoint_dir)
    }
    
    /// Save network adapter weights
    fn save_network_adapter(&self, adapter: &dyn NetworkAdapter, path: &Path) -> Result<()> {
        // Get all parameters
        let params = adapter.parameters();
        
        // Convert to safetensors format
        let mut tensors = HashMap::new();
        for (name, tensor) in params {
            let data = tensor.flatten_all()?.to_vec1::<f32>()?;
            let shape = tensor.dims().to_vec();
            tensors.insert(name.clone(), (data, shape));
        }
        
        // Add metadata
        let mut metadata = HashMap::new();
        metadata.insert("format".to_string(), "eridiffusion".to_string());
        metadata.insert("network_type".to_string(), format!("{:?}", adapter.adapter_type()));
        if let Some(rank) = adapter.metadata().rank {
            metadata.insert("rank".to_string(), rank.to_string());
        }
        if let Some(alpha) = adapter.metadata().alpha {
            metadata.insert("alpha".to_string(), alpha.to_string());
        }
        
        // Serialize and save
        let serialized = serialize_tensors(&tensors, &metadata)?;
        fs::write(path, serialized)?;
        
        Ok(())
    }
    
    /// Save optimizer state
    fn save_optimizer_state(&self, state: HashMap<String, Tensor>, path: &Path) -> Result<()> {
        let mut tensors = HashMap::new();
        
        for (name, tensor) in state {
            let data = tensor.flatten_all()?.to_vec1::<f32>()?;
            let shape = tensor.dims().to_vec();
            tensors.insert(name, (data, shape));
        }
        
        let metadata = HashMap::new();
        let serialized = serialize_tensors(&tensors, &metadata)?;
        fs::write(path, serialized)?;
        
        Ok(())
    }
    
    /// Load checkpoint
    pub fn load_checkpoint(
        &self,
        checkpoint_path: &Path,
    ) -> Result<(CheckpointMetadata, HashMap<String, Vec<f32>>)> {
        // Load metadata
        let metadata_path = checkpoint_path.join("metadata.json");
        let metadata_json = fs::read_to_string(metadata_path)?;
        let metadata: CheckpointMetadata = serde_json::from_str(&metadata_json)?;
        
        // Load adapter weights
        let adapter_path = checkpoint_path.join("adapter_model.safetensors");
        let adapter_weights = self.load_safetensors(&adapter_path)?;
        
        Ok((metadata, adapter_weights))
    }
    
    /// Load safetensors file
    fn load_safetensors(&self, path: &Path) -> Result<HashMap<String, Vec<f32>>> {
        let data = fs::read(path)?;
        let tensors = SafeTensors::deserialize(&data)
            .map_err(|e| Error::Model(format!("Failed to deserialize safetensors: {}", e)))?;
        let mut result = HashMap::new();
        
        for (name, view) in tensors.tensors() {
            let data: Vec<f32> = view.data()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            result.insert(name.to_string(), data);
        }
        
        Ok(result)
    }
    
    /// Clean up old checkpoints
    fn cleanup_old_checkpoints(&self) -> Result<()> {
        if let Some(limit) = self.save_total_limit {
            // Get all checkpoint directories
            let mut checkpoints = Vec::new();
            
            for entry in fs::read_dir(&self.output_dir)? {
                let entry = entry?;
                let path = entry.path();
                
                if path.is_dir() {
                    if let Some(name) = path.file_name() {
                        if let Some(name_str) = name.to_str() {
                            if name_str.starts_with("checkpoint-") && name_str != "checkpoint-latest" {
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
            
            // Sort by step (newest first)
            checkpoints.sort_by(|a, b| b.0.cmp(&a.0));
            
            // Remove old checkpoints
            for (_, path) in checkpoints.iter().skip(limit) {
                fs::remove_dir_all(path)?;
            }
        }
        
        Ok(())
    }
    
    /// Get latest checkpoint
    pub fn get_latest_checkpoint(&self) -> Option<PathBuf> {
        let latest_link = self.output_dir.join("checkpoint-latest");
        if latest_link.exists() {
            fs::read_link(&latest_link).ok()
        } else {
            // Find highest numbered checkpoint
            let mut latest_step = 0;
            let mut latest_path = None;
            
            if let Ok(entries) = fs::read_dir(&self.output_dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_dir() {
                        if let Some(name) = path.file_name() {
                            if let Some(name_str) = name.to_str() {
                                if let Some(step_str) = name_str.strip_prefix("checkpoint-") {
                                    if let Ok(step) = step_str.parse::<usize>() {
                                        if step > latest_step {
                                            latest_step = step;
                                            latest_path = Some(path);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            
            latest_path
        }
    }
}

/// Helper function to serialize tensors
fn serialize_tensors(
    tensors: &HashMap<String, (Vec<f32>, Vec<usize>)>,
    metadata: &HashMap<String, String>,
) -> Result<Vec<u8>> {
    use safetensors::{Dtype, tensor::TensorView};
    
    // First convert all tensors to bytes
    let mut bytes_storage: HashMap<String, Vec<u8>> = HashMap::new();
    for (name, (values, _)) in tensors {
        let bytes: Vec<u8> = values.iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        bytes_storage.insert(name.clone(), bytes);
    }
    
    // Then create TensorViews
    let mut data: HashMap<String, TensorView> = HashMap::new();
    for (name, (_, shape)) in tensors {
        let bytes = bytes_storage.get(name).unwrap();
        let view = TensorView::new(
            Dtype::F32,
            shape.clone(),
            bytes,
        ).map_err(|e| Error::Model(format!("Failed to create tensor view: {}", e)))?;
        data.insert(name.clone(), view);
    }
    
    let metadata_opt = if metadata.is_empty() {
        None
    } else {
        Some(metadata.clone())
    };
    
    serialize(&data, &metadata_opt)
        .map_err(|e| Error::Model(format!("Failed to serialize tensors: {}", e)))
}