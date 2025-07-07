// dataset.rs - Base dataset trait and implementations

use eridiffusion_core::{Result, Error};
use candle_core::{Tensor, DType, Device};
use serde::{Serialize, Deserialize};
use std::path::{Path, PathBuf};
use std::collections::HashMap;
use tokio::sync::RwLock;
use std::sync::Arc;

/// Base dataset trait
pub trait Dataset: Send + Sync {
    /// Get dataset length
    fn len(&self) -> usize;
    
    /// Check if empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    
    /// Get item by index
    fn get_item(&self, index: usize) -> Result<DatasetItem>;
    
    /// Get metadata
    fn metadata(&self) -> &DatasetMetadata;
    
    /// Get all indices
    fn indices(&self) -> Vec<usize> {
        (0..self.len()).collect()
    }
}

/// Dataset item
#[derive(Debug, Clone)]
pub struct DatasetItem {
    pub image: Tensor,
    pub caption: String,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Dataset metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetMetadata {
    pub name: String,
    pub description: String,
    pub version: String,
    pub size: usize,
    pub features: HashMap<String, String>,
}

impl Default for DatasetMetadata {
    fn default() -> Self {
        Self {
            name: "Unknown".to_string(),
            description: "No description".to_string(),
            version: "1.0.0".to_string(),
            size: 0,
            features: HashMap::new(),
        }
    }
}

// Implement Dataset for Box<dyn Dataset>
impl Dataset for Box<dyn Dataset> {
    fn len(&self) -> usize {
        (**self).len()
    }
    
    fn get_item(&self, index: usize) -> Result<DatasetItem> {
        (**self).get_item(index)
    }
    
    fn metadata(&self) -> &DatasetMetadata {
        (**self).metadata()
    }
}