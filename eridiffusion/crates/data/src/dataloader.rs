// dataloader.rs - DataLoader for batching and parallel loading

use crate::{Dataset, DatasetItem};
use eridiffusion_core::{Result, Error, Device};
use candle_core::{Tensor, DType};
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
use rand::seq::SliceRandom;
use tracing::{debug, warn};

/// DataLoader batch
#[derive(Debug, Clone)]
pub struct DataLoaderBatch {
    pub images: Tensor,
    pub captions: Vec<String>,
    pub masks: Option<Tensor>,
    pub loss_weights: Vec<f32>,
    pub metadata: std::collections::HashMap<String, Vec<serde_json::Value>>,
}

impl DataLoaderBatch {
    pub fn new(
        images: Tensor,
        captions: Vec<String>,
        masks: Option<Tensor>,
        loss_weights: Option<Vec<f32>>,
        metadata: std::collections::HashMap<String, Vec<serde_json::Value>>,
    ) -> Self {
        let batch_size = images.dims()[0];
        let loss_weights = loss_weights.unwrap_or_else(|| vec![1.0; batch_size]);
        
        Self {
            images,
            captions,
            masks,
            loss_weights,
            metadata,
        }
    }
    
    pub fn batch_size(&self) -> usize {
        self.images.dims()[0]
    }
    
    pub fn to_device(&self, device: &Device) -> Result<Self> {
        let candle_device = match device {
            Device::Cpu => candle_core::Device::Cpu,
            Device::Cuda(id) => candle_core::Device::new_cuda(*id)?,
        };
        
        let images = self.images.to_device(&candle_device)?;
        let masks = self.masks.as_ref().map(|m| m.to_device(&candle_device)).transpose()?;
        
        Ok(Self {
            images,
            captions: self.captions.clone(),
            masks,
            loss_weights: self.loss_weights.clone(),
            metadata: self.metadata.clone(),
        })
    }
}

/// DataLoader configuration
#[derive(Debug, Clone)]
pub struct DataLoaderConfig {
    pub batch_size: usize,
    pub shuffle: bool,
    pub drop_last: bool,
    pub num_workers: usize,
    pub prefetch_factor: usize,
}

impl Default for DataLoaderConfig {
    fn default() -> Self {
        Self {
            batch_size: 4,
            shuffle: true,
            drop_last: false,
            num_workers: 4,
            prefetch_factor: 2,
        }
    }
}

/// DataLoader for training
pub struct DataLoader<D: Dataset> {
    dataset: Arc<D>,
    config: DataLoaderConfig,
    device: Device,
    indices: Vec<usize>,
    current_position: Arc<Mutex<usize>>,
    epoch: Arc<Mutex<usize>>,
}

impl<D: Dataset + 'static> DataLoader<D> {
    /// Create new DataLoader
    pub fn new(dataset: D, config: DataLoaderConfig, device: Device) -> Self {
        let indices: Vec<usize> = (0..dataset.len()).collect();
        
        Self {
            dataset: Arc::new(dataset),
            config,
            device,
            indices,
            current_position: Arc::new(Mutex::new(0)),
            epoch: Arc::new(Mutex::new(0)),
        }
    }
    
    /// Get number of batches
    pub fn len(&self) -> usize {
        if self.config.drop_last {
            self.dataset.len() / self.config.batch_size
        } else {
            (self.dataset.len() + self.config.batch_size - 1) / self.config.batch_size
        }
    }
    
    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.dataset.is_empty()
    }
    
    /// Create iterator
    pub async fn iter(&self) -> DataLoaderIterator<D> {
        // Reset position
        *self.current_position.lock().await = 0;
        
        // Shuffle if needed
        let mut indices = self.indices.clone();
        if self.config.shuffle {
            let mut rng = rand::thread_rng();
            indices.shuffle(&mut rng);
        }
        
        DataLoaderIterator {
            loader: self.clone(),
            indices,
            prefetch_rx: None,
        }
    }
    
    /// Reset for new epoch
    pub async fn reset(&self) {
        *self.current_position.lock().await = 0;
        *self.epoch.lock().await += 1;
    }
}

impl<D: Dataset> Clone for DataLoader<D> {
    fn clone(&self) -> Self {
        Self {
            dataset: self.dataset.clone(),
            config: self.config.clone(),
            device: self.device.clone(),
            indices: self.indices.clone(),
            current_position: self.current_position.clone(),
            epoch: self.epoch.clone(),
        }
    }
}

/// DataLoader iterator
pub struct DataLoaderIterator<D: Dataset> {
    loader: DataLoader<D>,
    indices: Vec<usize>,
    prefetch_rx: Option<mpsc::Receiver<Result<DataLoaderBatch>>>,
}

impl<D: Dataset + 'static> DataLoaderIterator<D> {
    /// Initialize prefetching if enabled
    pub async fn start_prefetch(&mut self) {
        if self.loader.config.num_workers > 0 && self.prefetch_rx.is_none() {
            let (tx, rx) = mpsc::channel(self.loader.config.prefetch_factor);
            self.prefetch_rx = Some(rx);
            
            // Start prefetch task
            let loader = self.loader.clone();
            let indices = self.indices.clone();
            
            tokio::spawn(async move {
                let mut pos = 0;
                while pos < indices.len() {
                    let batch_size = loader.config.batch_size;
                    let end = (pos + batch_size).min(indices.len());
                    
                    if loader.config.drop_last && (end - pos) < batch_size {
                        break;
                    }
                    
                    let batch_indices = &indices[pos..end];
                    
                    // Load items in parallel
                    let mut items = Vec::new();
                    for &idx in batch_indices {
                        match loader.dataset.get_item(idx) {
                            Ok(item) => items.push(item),
                            Err(e) => {
                                let _ = tx.send(Err(e)).await;
                                return;
                            }
                        }
                    }
                    
                    // Collate batch
                    match collate_batch(items, &loader.device) {
                        Ok(batch) => {
                            if tx.send(Ok(batch)).await.is_err() {
                                break; // Receiver dropped
                            }
                        }
                        Err(e) => {
                            let _ = tx.send(Err(e)).await;
                            return;
                        }
                    }
                    
                    pos = end;
                }
            });
        }
    }
    
    /// Get next batch
    pub async fn next(&mut self) -> Option<Result<DataLoaderBatch>> {
        // Use prefetch if available
        if let Some(ref mut rx) = self.prefetch_rx {
            return rx.recv().await;
        }
        
        // Otherwise load synchronously
        let pos = *self.loader.current_position.lock().await;
        
        if pos >= self.indices.len() {
            return None;
        }
        
        let batch_size = self.loader.config.batch_size;
        let end = (pos + batch_size).min(self.indices.len());
        
        if self.loader.config.drop_last && (end - pos) < batch_size {
            return None;
        }
        
        let batch_indices = &self.indices[pos..end];
        
        // Update position
        *self.loader.current_position.lock().await = end;
        
        // Load items
        let mut items = Vec::new();
        for &idx in batch_indices {
            match self.loader.dataset.get_item(idx) {
                Ok(item) => items.push(item),
                Err(e) => return Some(Err(e)),
            }
        }
        
        // Collate batch
        Some(collate_batch(items, &self.loader.device))
    }
}

/// Collate items into batch
fn collate_batch(items: Vec<DatasetItem>, device: &Device) -> Result<DataLoaderBatch> {
    if items.is_empty() {
        return Err(Error::DataError("Cannot create batch from empty items".into()));
    }
    
    // Convert device
    let candle_device = match device {
        Device::Cpu => candle_core::Device::Cpu,
        Device::Cuda(id) => candle_core::Device::new_cuda(*id)?,
    };
    
    // Check all images have same shape
    let first_shape = items[0].image.dims();
    for (i, item) in items.iter().enumerate().skip(1) {
        if item.image.dims() != first_shape {
            return Err(Error::DataError(format!(
                "Image {} has shape {:?}, expected {:?}",
                i, item.image.dims(), first_shape
            )));
        }
    }
    
    // Stack images
    let mut images = Vec::new();
    for item in items.iter() {
        images.push(item.image.unsqueeze(0)?);
    }
    
    let images = Tensor::cat(&images, 0)?.to_device(&candle_device)?;
    
    // Collect captions
    let captions: Vec<String> = items.iter()
        .map(|item| item.caption.clone())
        .collect();
    
    // Collect metadata
    let mut metadata = std::collections::HashMap::new();
    if !items.is_empty() {
        for (key, _) in items[0].metadata.iter() {
            let values: Vec<serde_json::Value> = items.iter()
                .map(|item| item.metadata.get(key).cloned().unwrap_or(serde_json::Value::Null))
                .collect();
            metadata.insert(key.clone(), values);
        }
    }
    
    Ok(DataLoaderBatch::new(
        images,
        captions,
        None,
        None,
        metadata,
    ))
}