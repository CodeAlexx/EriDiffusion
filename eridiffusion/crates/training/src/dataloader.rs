//! Data loading infrastructure

use eridiffusion_core::{ModelInputs, Result, Error, Device};
use candle_core::{Tensor, DType};
use std::path::Path;
use std::sync::Arc;
use tokio::sync::RwLock;
use async_trait::async_trait;
use serde::{Serialize, Deserialize};

/// Dataset trait
#[async_trait]
pub trait Dataset: Send + Sync {
    /// Get dataset length
    fn len(&self) -> usize;
    
    /// Check if empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    
    /// Get item at index
    async fn get_item(&self, index: usize) -> Result<DatasetItem>;
    
    /// Get metadata
    fn metadata(&self) -> &DatasetMetadata;
}

/// Dataset item
#[derive(Debug, Clone)]
pub struct DatasetItem {
    pub image: Tensor,
    pub caption: String,
    pub additional: std::collections::HashMap<String, Tensor>,
}

/// Dataset metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetMetadata {
    pub name: String,
    pub size: usize,
    pub image_size: (usize, usize),
    pub has_captions: bool,
    pub has_masks: bool,
}

/// Data loader
pub struct DataLoader {
    dataset: Arc<Box<dyn Dataset>>,
    batch_size: usize,
    shuffle: bool,
    drop_last: bool,
    num_workers: usize,
    device: Device,
    indices: Arc<RwLock<Vec<usize>>>,
}

impl DataLoader {
    /// Create new data loader
    pub fn new(
        dataset: Box<dyn Dataset>,
        batch_size: usize,
        shuffle: bool,
        drop_last: bool,
        num_workers: usize,
        device: Device,
    ) -> Self {
        let dataset_len = dataset.len();
        let indices: Vec<usize> = (0..dataset_len).collect();
        
        Self {
            dataset: Arc::new(dataset),
            batch_size,
            shuffle,
            drop_last,
            num_workers,
            device,
            indices: Arc::new(RwLock::new(indices)),
        }
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
    
    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    
    /// Create iterator
    pub fn iter(&self) -> DataLoaderIterator {
        DataLoaderIterator::new(self)
    }
    
    /// Shuffle indices
    async fn shuffle_indices(&self) {
        if self.shuffle {
            use rand::seq::SliceRandom;
            let mut rng = rand::thread_rng();
            let mut indices = self.indices.write().await;
            indices.shuffle(&mut rng);
        }
    }
}

/// Data loader iterator
pub struct DataLoaderIterator {
    loader: DataLoader,
    current_batch: usize,
    total_batches: usize,
}

impl DataLoaderIterator {
    fn new(loader: &DataLoader) -> Self {
        let loader = loader.clone();
        let total_batches = loader.len();
        
        Self {
            loader,
            current_batch: 0,
            total_batches,
        }
    }
    
    /// Get next batch
    pub async fn next(&mut self) -> Option<Result<ModelInputs>> {
        if self.current_batch >= self.total_batches {
            // Reset for next epoch
            self.current_batch = 0;
            self.loader.shuffle_indices().await;
            return None;
        }
        
        let batch_result = self.get_batch(self.current_batch).await;
        self.current_batch += 1;
        
        Some(batch_result)
    }
    
    /// Get specific batch
    async fn get_batch(&self, batch_idx: usize) -> Result<ModelInputs> {
        let start_idx = batch_idx * self.loader.batch_size;
        let end_idx = ((batch_idx + 1) * self.loader.batch_size).min(self.loader.dataset.len());
        
        let indices = self.loader.indices.read().await;
        let batch_indices: Vec<usize> = indices[start_idx..end_idx].to_vec();
        drop(indices);
        
        // Collect items in parallel
        let mut handles = Vec::new();
        for idx in batch_indices {
            let dataset = self.loader.dataset.clone();
            let handle = tokio::spawn(async move {
                dataset.get_item(idx).await
            });
            handles.push(handle);
        }
        
        // Wait for all items
        let mut items = Vec::new();
        for handle in handles {
            let item = handle.await
                .map_err(|e| Error::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;
            items.push(item?);
        }
        
        // Collate batch
        self.collate_batch(items).await
    }
    
    /// Collate items into batch
    async fn collate_batch(&self, items: Vec<DatasetItem>) -> Result<ModelInputs> {
        if items.is_empty() {
            return Err(Error::DataError("Empty batch".to_string()));
        }
        
        // Stack images
        let images: Vec<Tensor> = items.iter()
            .map(|item| item.image.clone())
            .collect();
        let latents = Tensor::stack(&images, 0)?;
        
        // Placeholder timesteps
        let batch_size = items.len();
        let candle_device = self.loader.device.to_candle()?;
        let timestep = Tensor::zeros(&[batch_size], DType::F32, &candle_device)?;
        
        // Placeholder encoder hidden states
        let encoder_hidden_states = Tensor::zeros(
            &[batch_size, 77, 768],
            DType::F32,
            &candle_device,
        )?;
        
        // Collect captions
        let captions: Vec<String> = items.iter()
            .map(|item| item.caption.clone())
            .collect();
        
        let mut additional = std::collections::HashMap::new();
        additional.insert("captions".to_string(), Tensor::new(captions.len() as i64, &candle_device)?);
        
        // Add any additional tensors from first item
        if let Some(first_item) = items.first() {
            for (key, _) in &first_item.additional {
                let tensors: Vec<Tensor> = items.iter()
                    .map(|item| item.additional.get(key).unwrap().clone())
                    .collect();
                let stacked = Tensor::stack(&tensors, 0)?;
                additional.insert(key.clone(), stacked);
            }
        }
        
        Ok(ModelInputs {
            latents,
            timestep,
            encoder_hidden_states: Some(encoder_hidden_states),
            pooled_projections: None,
            attention_mask: None,
            guidance_scale: None,
            additional,
        })
    }
}

impl Clone for DataLoader {
    fn clone(&self) -> Self {
        Self {
            dataset: self.dataset.clone(),
            batch_size: self.batch_size,
            shuffle: self.shuffle,
            drop_last: self.drop_last,
            num_workers: self.num_workers,
            device: self.device.clone(),
            indices: self.indices.clone(),
        }
    }
}

/// Image folder dataset
pub struct ImageFolderDataset {
    root_path: std::path::PathBuf,
    image_paths: Vec<std::path::PathBuf>,
    captions: Vec<String>,
    transform: Option<Box<dyn Transform>>,
    metadata: DatasetMetadata,
}

impl ImageFolderDataset {
    /// Create new image folder dataset
    pub async fn new(
        root_path: &Path,
        image_size: (usize, usize),
        caption_extension: Option<&str>,
    ) -> Result<Self> {
        let mut image_paths = Vec::new();
        let mut captions = Vec::new();
        
        // Scan directory for images
        let mut entries = tokio::fs::read_dir(root_path).await
            .map_err(|e| Error::Io(e))?;
        
        while let Some(entry) = entries.next_entry().await.map_err(|e| Error::Io(e))? {
            let path = entry.path();
            if let Some(ext) = path.extension() {
                let ext_str = ext.to_string_lossy().to_lowercase();
                if ["jpg", "jpeg", "png", "webp"].contains(&ext_str.as_str()) {
                    image_paths.push(path.clone());
                    
                    // Load caption if available
                    if let Some(caption_ext) = caption_extension {
                        let caption_path = path.with_extension(caption_ext);
                        if caption_path.exists() {
                            let caption = tokio::fs::read_to_string(&caption_path).await
                                .unwrap_or_else(|_| "".to_string());
                            captions.push(caption.trim().to_string());
                        } else {
                            captions.push("".to_string());
                        }
                    } else {
                        captions.push("".to_string());
                    }
                }
            }
        }
        
        let metadata = DatasetMetadata {
            name: root_path.file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string(),
            size: image_paths.len(),
            image_size,
            has_captions: caption_extension.is_some(),
            has_masks: false,
        };
        
        Ok(Self {
            root_path: root_path.to_path_buf(),
            image_paths,
            captions,
            transform: None,
            metadata,
        })
    }
    
    /// Set transform
    pub fn with_transform(mut self, transform: Box<dyn Transform>) -> Self {
        self.transform = Some(transform);
        self
    }
}

#[async_trait]
impl Dataset for ImageFolderDataset {
    fn len(&self) -> usize {
        self.image_paths.len()
    }
    
    async fn get_item(&self, index: usize) -> Result<DatasetItem> {
        if index >= self.len() {
            return Err(Error::DataError(format!("Index {} out of bounds", index)));
        }
        
        // Load image from file
        let image_path = &self.image_paths[index];
        let image_bytes = std::fs::read(image_path)
            .map_err(|e| Error::DataError(format!("Failed to read image {}: {}", image_path.display(), e)))?;
        
        // Decode image using image crate
        use image::io::Reader as ImageReader;
        use std::io::Cursor;
        
        let img = ImageReader::new(Cursor::new(image_bytes))
            .with_guessed_format()
            .map_err(|e| Error::DataError(format!("Failed to guess image format: {}", e)))?
            .decode()
            .map_err(|e| Error::DataError(format!("Failed to decode image: {}", e)))?
            .to_rgb8();
        
        // Convert to tensor
        let (width, height) = (img.width() as usize, img.height() as usize);
        let raw_data: Vec<f32> = img.pixels()
            .flat_map(|p| p.0.iter().map(|&v| v as f32 / 255.0))
            .collect();
        
        let image = Tensor::from_vec(
            raw_data,
            (height, width, 3),
            &candle_core::Device::Cpu,
        )?
        .permute((2, 0, 1))?; // HWC -> CHW
        
        // Apply transform if available
        let image = if let Some(ref transform) = self.transform {
            transform.apply(&image)?
        } else {
            image
        };
        
        let caption = self.captions[index].clone();
        
        Ok(DatasetItem {
            image,
            caption,
            additional: std::collections::HashMap::new(),
        })
    }
    
    fn metadata(&self) -> &DatasetMetadata {
        &self.metadata
    }
}

/// Transform trait
pub trait Transform: Send + Sync {
    /// Apply transform to tensor
    fn apply(&self, tensor: &Tensor) -> Result<Tensor>;
}

/// Random crop transform
pub struct RandomCrop {
    size: (usize, usize),
}

impl RandomCrop {
    pub fn new(size: (usize, usize)) -> Self {
        Self { size }
    }
}

impl Transform for RandomCrop {
    fn apply(&self, tensor: &Tensor) -> Result<Tensor> {
        // Simplified - would implement actual random cropping
        Ok(tensor.clone())
    }
}

/// Normalize transform
pub struct Normalize {
    mean: Vec<f32>,
    std: Vec<f32>,
}

impl Normalize {
    pub fn new(mean: Vec<f32>, std: Vec<f32>) -> Self {
        Self { mean, std }
    }
}

impl Transform for Normalize {
    fn apply(&self, tensor: &Tensor) -> Result<Tensor> {
        // Simplified - would implement actual normalization
        Ok(tensor.clone())
    }
}

/// Compose multiple transforms
pub struct Compose {
    transforms: Vec<Box<dyn Transform>>,
}

impl Compose {
    pub fn new(transforms: Vec<Box<dyn Transform>>) -> Self {
        Self { transforms }
    }
}

impl Transform for Compose {
    fn apply(&self, tensor: &Tensor) -> Result<Tensor> {
        let mut result = tensor.clone();
        for transform in &self.transforms {
            result = transform.apply(&result)?;
        }
        Ok(result)
    }
}