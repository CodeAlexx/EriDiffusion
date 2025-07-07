//! DataLoader batch implementation for diffusion training

use eridiffusion_core::{Result, Error};
use candle_core::{Tensor, Device};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Batch for diffusion model training
#[derive(Debug, Clone)]
pub struct DataLoaderBatch {
    /// Image tensors [batch_size, channels, height, width]
    pub images: Tensor,
    
    /// Text captions
    pub captions: Vec<String>,
    
    /// Optional masks for inpainting [batch_size, 1, height, width]
    pub masks: Option<Tensor>,
    
    /// Loss weights per sample
    pub loss_weights: Vec<f32>,
    
    /// Additional metadata
    pub metadata: HashMap<String, Vec<serde_json::Value>>,
}

impl DataLoaderBatch {
    /// Create a new batch
    pub fn new(
        images: Tensor,
        captions: Vec<String>,
        masks: Option<Tensor>,
        loss_weights: Option<Vec<f32>>,
        metadata: HashMap<String, Vec<serde_json::Value>>,
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
    
    /// Get batch size
    pub fn batch_size(&self) -> usize {
        self.images.dims()[0]
    }
    
    /// Move batch to device
    pub fn to_device(&self, device: &Device) -> Result<Self> {
        let images = self.images.to_device(device)?;
        let masks = self.masks.as_ref().map(|m| m.to_device(device)).transpose()?;
        
        Ok(Self {
            images,
            captions: self.captions.clone(),
            masks,
            loss_weights: self.loss_weights.clone(),
            metadata: self.metadata.clone(),
        })
    }
    
    /// Create batch from dataset items
    pub fn from_items(
        items: Vec<crate::DatasetItem>,
        device: &candle_core::Device,
    ) -> Result<Self> {
        if items.is_empty() {
            return Err(Error::DataError("Cannot create batch from empty items".into()));
        }
        
        // Stack images
        let images: Vec<Tensor> = items.iter()
            .map(|item| item.image.unsqueeze(0).map_err(Error::from))
            .collect::<Result<Vec<_>>>()?;
        let images = Tensor::cat(&images, 0)?;
        
        // Collect captions
        let captions: Vec<String> = items.iter()
            .map(|item| item.caption.clone())
            .collect();
        
        // Collect metadata
        let mut metadata = HashMap::new();
        for (key, _) in items[0].metadata.iter() {
            let values: Vec<serde_json::Value> = items.iter()
                .map(|item| item.metadata.get(key).cloned().unwrap_or(serde_json::Value::Null))
                .collect();
            metadata.insert(key.clone(), values);
        }
        
        Ok(Self::new(
            images.to_device(device)?,
            captions,
            None,
            None,
            metadata,
        ))
    }
}

/// Extended batch with latents
#[derive(Debug, Clone)]
pub struct LatentBatch {
    /// Pre-encoded latents
    pub latents: Tensor,
    
    /// Text captions
    pub captions: Vec<String>,
    
    /// Text embeddings (if pre-computed)
    pub text_embeds: Option<Tensor>,
    
    /// Pooled text embeddings
    pub pooled_embeds: Option<Tensor>,
    
    /// Loss weights
    pub loss_weights: Vec<f32>,
    
    /// Original image sizes
    pub original_sizes: Vec<(u32, u32)>,
    
    /// Crop coordinates
    pub crop_coords: Vec<(u32, u32)>,
    
    /// Additional metadata
    pub metadata: HashMap<String, Vec<serde_json::Value>>,
}

impl LatentBatch {
    /// Convert from DataLoaderBatch using VAE encoding
    pub async fn from_batch(
        batch: DataLoaderBatch,
        vae: &dyn eridiffusion_models::VAE,
        text_encoder: Option<&dyn eridiffusion_models::TextEncoder>,
    ) -> Result<Self> {
        // Encode images to latents
        let latents = vae.encode(&batch.images)?;
        
        // Encode text if encoder provided
        let (text_embeds, pooled_embeds) = if let Some(encoder) = text_encoder {
            let (embeds, pooled) = encoder.encode(&batch.captions)?;
            (Some(embeds), pooled)
        } else {
            (None, None)
        };
        
        // Extract original sizes and crop coords from metadata
        let original_sizes = batch.metadata.get("original_size")
            .map(|sizes| {
                sizes.iter().map(|v| {
                    if let serde_json::Value::Array(arr) = v {
                        let w = arr.get(0).and_then(|v| v.as_u64()).unwrap_or(1024) as u32;
                        let h = arr.get(1).and_then(|v| v.as_u64()).unwrap_or(1024) as u32;
                        (w, h)
                    } else {
                        (1024, 1024)
                    }
                }).collect()
            })
            .unwrap_or_else(|| vec![(1024, 1024); batch.batch_size()]);
        
        let crop_coords = batch.metadata.get("crop_coords")
            .map(|coords| {
                coords.iter().map(|v| {
                    if let serde_json::Value::Array(arr) = v {
                        let x = arr.get(0).and_then(|v| v.as_u64()).unwrap_or(0) as u32;
                        let y = arr.get(1).and_then(|v| v.as_u64()).unwrap_or(0) as u32;
                        (x, y)
                    } else {
                        (0, 0)
                    }
                }).collect()
            })
            .unwrap_or_else(|| vec![(0, 0); batch.batch_size()]);
        
        Ok(Self {
            latents,
            captions: batch.captions,
            text_embeds,
            pooled_embeds,
            loss_weights: batch.loss_weights,
            original_sizes,
            crop_coords,
            metadata: batch.metadata,
        })
    }
}