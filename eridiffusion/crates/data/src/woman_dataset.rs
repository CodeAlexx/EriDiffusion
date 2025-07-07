//! Woman dataset loader for LoRA training

use crate::{ImageDataset, DatasetConfig, Dataset, DatasetItem};
use eridiffusion_core::{Result, Error};
use std::path::PathBuf;
use tracing::info;

/// Configuration for the 40_woman dataset
pub struct WomanDatasetConfig {
    pub root_dir: PathBuf,
    pub resolution: usize,
    pub repeats: usize,
    pub cache_latents: bool,
}

impl Default for WomanDatasetConfig {
    fn default() -> Self {
        Self {
            root_dir: PathBuf::from("datasets/40_woman"),
            resolution: 1024,
            repeats: 20,
            cache_latents: true,
        }
    }
}

/// Dataset loader specifically for the 40_woman dataset
pub struct WomanDataset {
    inner: ImageDataset,
    repeats: usize,
}

impl WomanDataset {
    /// Create new dataset from the 40_woman directory
    pub fn new(config: WomanDatasetConfig) -> Result<Self> {
        info!("Loading 40_woman dataset from: {}", config.root_dir.display());
        
        // Create inner dataset config
        let dataset_config = DatasetConfig {
            root_dir: config.root_dir,
            caption_ext: "txt".to_string(),
            resolution: config.resolution,
            center_crop: true,
            random_flip: true,
            cache_latents: config.cache_latents,
            cache_dir: Some(PathBuf::from(".cache/latents")),
        };
        
        let inner = ImageDataset::new(dataset_config)?;
        
        info!(
            "Loaded {} images with {} repeats = {} total samples", 
            inner.len(), 
            config.repeats,
            inner.len() * config.repeats
        );
        
        Ok(Self {
            inner,
            repeats: config.repeats,
        })
    }
    
    /// Get base dataset length (without repeats)
    pub fn base_len(&self) -> usize {
        self.inner.len()
    }
}

impl Dataset for WomanDataset {
    fn len(&self) -> usize {
        self.inner.len() * self.repeats
    }
    
    fn get_item(&self, index: usize) -> Result<DatasetItem> {
        // Map repeated index to base index
        let base_index = index % self.inner.len();
        let repeat = index / self.inner.len();
        
        // Get base item
        let mut item = self.inner.get_item(base_index)?;
        
        // Add repeat information to metadata
        item.metadata.insert(
            "repeat".to_string(),
            serde_json::Value::Number(repeat.into()),
        );
        
        // Add special token if this is ohwx training
        if !item.caption.contains("ohwx") {
            item.caption = format!("ohwx woman, {}", item.caption);
        }
        
        Ok(item)
    }
    
    fn metadata(&self) -> &crate::dataset::DatasetMetadata {
        self.inner.metadata()
    }
}

/// Helper to validate dataset directory structure
pub fn validate_woman_dataset(root_dir: &PathBuf) -> Result<()> {
    use std::fs;
    
    if !root_dir.exists() {
        return Err(Error::DataError(format!(
            "Dataset directory does not exist: {}",
            root_dir.display()
        )));
    }
    
    // Check for expected structure
    let mut has_images = false;
    let mut has_captions = false;
    
    for entry in fs::read_dir(root_dir)? {
        let entry = entry?;
        let path = entry.path();
        
        if let Some(ext) = path.extension() {
            match ext.to_str().unwrap_or("").to_lowercase().as_str() {
                "jpg" | "jpeg" | "png" => has_images = true,
                "txt" | "caption" => has_captions = true,
                _ => {}
            }
        }
    }
    
    if !has_images {
        return Err(Error::DataError(
            "No images found in dataset directory".into()
        ));
    }
    
    if !has_captions {
        info!("Warning: No caption files found, will use filenames as captions");
    }
    
    Ok(())
}