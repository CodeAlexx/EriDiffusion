//! Complete DataLoader implementation

use crate::{Dataset, DatasetItem, DataLoaderBatch, BucketSampler};
use eridiffusion_core::{Result, Error, Device};
use candle_core::Tensor;
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
use rand::seq::SliceRandom;

/// DataLoader for training
pub struct DataLoader<D: Dataset> {
    dataset: Arc<D>,
    sampler: Arc<Mutex<BucketSampler>>,
    batch_size: usize,
    drop_last: bool,
    num_workers: usize,
    device: Device,
}

impl<D: Dataset + 'static> DataLoader<D> {
    /// Create new DataLoader
    pub fn new(
        dataset: D,
        sampler: BucketSampler,
        batch_size: usize,
        drop_last: bool,
        num_workers: usize,
    ) -> Result<Self> {
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        
        Ok(Self {
            dataset: Arc::new(dataset),
            sampler: Arc::new(Mutex::new(sampler)),
            batch_size,
            drop_last,
            num_workers,
            device,
        })
    }
    
    /// Get number of batches
    pub fn len(&self) -> usize {
        let dataset_len = self.dataset.len();
        if self.drop_last {
            dataset_len / self.batch_size
        } else {
            (dataset_len + self.batch_size - 1) / self.batch_size
        }
    }
    
    /// Create iterator over batches
    pub async fn iter(&self) -> DataLoaderIterator<D> {
        // Create indices
        let mut indices: Vec<usize> = (0..self.dataset.len()).collect();
        
        // Shuffle if needed
        let sampler = self.sampler.lock().await;
        if sampler.shuffle {
            let mut rng = rand::thread_rng();
            indices.shuffle(&mut rng);
        }
        drop(sampler);
        
        DataLoaderIterator {
            dataset: self.dataset.clone(),
            indices,
            current: 0,
            batch_size: self.batch_size,
            drop_last: self.drop_last,
            device: self.device.clone(),
        }
    }
}

/// Iterator over batches
pub struct DataLoaderIterator<D: Dataset> {
    dataset: Arc<D>,
    indices: Vec<usize>,
    current: usize,
    batch_size: usize,
    drop_last: bool,
    device: Device,
}

impl<D: Dataset> DataLoaderIterator<D> {
    /// Get next batch
    pub async fn next(&mut self) -> Option<Result<DataLoaderBatch>> {
        if self.current >= self.indices.len() {
            return None;
        }
        
        // Get batch indices
        let end = (self.current + self.batch_size).min(self.indices.len());
        let batch_indices = &self.indices[self.current..end];
        
        // Check if we should drop last incomplete batch
        if self.drop_last && batch_indices.len() < self.batch_size {
            return None;
        }
        
        self.current = end;
        
        // Load items
        let mut items = Vec::new();
        for &idx in batch_indices {
            match self.dataset.get_item(idx) {
                Ok(item) => items.push(item),
                Err(e) => return Some(Err(e)),
            }
        }
        
        // Create batch
        let candle_device = match &self.device {
            Device::Cpu => candle_core::Device::Cpu,
            Device::Cuda(id) => match candle_core::Device::new_cuda(*id) {
                Ok(d) => d,
                Err(e) => return Some(Err(Error::from(e))),
            },
        };
        
        Some(collate_fn(items, &self.device))
    }
}

/// Collate function for creating batches
pub fn collate_fn(items: Vec<DatasetItem>, device: &Device) -> Result<DataLoaderBatch> {
    if items.is_empty() {
        return Err(Error::DataError("Cannot create batch from empty items".into()));
    }
    
    // Convert device
    let candle_device = match device {
        Device::Cpu => candle_core::Device::Cpu,
        Device::Cuda(id) => candle_core::Device::new_cuda(*id)?,
    };
    
    // Find the target size (assuming all items are the same size after transform)
    let first_dims = items[0].image.dims();
    let (channels, height, width) = (first_dims[0], first_dims[1], first_dims[2]);
    
    // Create batch tensor
    let batch_size = items.len();
    let mut batch_data = vec![0.0f32; batch_size * channels * height * width];
    
    // Copy each image into the batch
    for (i, item) in items.iter().enumerate() {
        let image_data = item.image.flatten_all()?.to_vec1::<f32>()?;
        let offset = i * channels * height * width;
        batch_data[offset..offset + image_data.len()].copy_from_slice(&image_data);
    }
    
    let images = Tensor::from_vec(
        batch_data,
        &[batch_size, channels, height, width],
        &candle_device,
    )?;
    
    // Collect other data
    let captions: Vec<String> = items.iter().map(|item| item.caption.clone()).collect();
    
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

/// Multi-threaded data loading with prefetching
pub struct PrefetchDataLoader<D: Dataset> {
    loader: DataLoader<D>,
    prefetch_factor: usize,
}

impl<D: Dataset + 'static> PrefetchDataLoader<D> {
    /// Create new prefetch data loader
    pub fn new(loader: DataLoader<D>, prefetch_factor: usize) -> Self {
        Self {
            loader,
            prefetch_factor,
        }
    }
    
    /// Start prefetching batches
    pub async fn start(&self) -> mpsc::Receiver<Result<DataLoaderBatch>> {
        let (tx, rx) = mpsc::channel(self.prefetch_factor);
        let mut iter = self.loader.iter().await;
        
        // Spawn prefetch task
        tokio::spawn(async move {
            while let Some(batch_result) = iter.next().await {
                if tx.send(batch_result).await.is_err() {
                    break; // Receiver dropped
                }
            }
        });
        
        rx
    }
}