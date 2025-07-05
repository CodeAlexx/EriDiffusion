//! Flux dataset preprocessor - pre-computes latents and text embeddings
//! This is ESSENTIAL for fitting Flux training in 24GB VRAM

use eridiffusion_core::{Device, Result, Error, ModelArchitecture};
use eridiffusion_models::{VAE, TextEncoder};
use eridiffusion_data::{Dataset, DatasetItem, VAENormalizer};
use candle_core::{Tensor, DType};
use std::path::{Path, PathBuf};
use std::fs;
use safetensors::{SafeTensors, serialize};
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::HashMap;

/// Preprocessed item stored on disk
#[derive(Debug)]
pub struct PreprocessedFluxItem {
    /// Pre-encoded latents [16, H/8, W/8]
    pub latents_path: PathBuf,
    /// Pre-encoded T5 embeddings [512, 4096]
    pub t5_embeds_path: PathBuf,
    /// Pre-encoded CLIP pooled embeddings [768]
    pub clip_pooled_path: PathBuf,
    /// Original caption (for reference)
    pub caption: String,
    /// Metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Flux preprocessor configuration
#[derive(Debug, Clone)]
pub struct FluxPreprocessorConfig {
    /// Output directory for cached tensors
    pub cache_dir: PathBuf,
    /// Device for encoding (can be different from training device)
    pub device: Device,
    /// Batch size for encoding
    pub batch_size: usize,
    /// Whether to overwrite existing cache
    pub overwrite: bool,
}

/// Preprocessor for Flux training data
pub struct FluxPreprocessor {
    config: FluxPreprocessorConfig,
    vae: Option<Box<dyn VAE>>,
    t5_encoder: Option<Box<dyn TextEncoder>>,
    clip_encoder: Option<Box<dyn TextEncoder>>,
    vae_normalizer: VAENormalizer,
}

impl FluxPreprocessor {
    /// Create new preprocessor
    pub fn new(config: FluxPreprocessorConfig) -> Result<Self> {
        // Create cache directory
        fs::create_dir_all(&config.cache_dir)?;
        
        // Create VAE normalizer for Flux
        let vae_normalizer = VAENormalizer::new(ModelArchitecture::Flux);
        
        Ok(Self {
            config,
            vae: None,
            t5_encoder: None,
            clip_encoder: None,
            vae_normalizer,
        })
    }
    
    /// Set VAE for latent encoding
    pub fn with_vae(mut self, vae: Box<dyn VAE>) -> Self {
        self.vae = Some(vae);
        self
    }
    
    /// Set T5 encoder
    pub fn with_t5_encoder(mut self, encoder: Box<dyn TextEncoder>) -> Self {
        self.t5_encoder = Some(encoder);
        self
    }
    
    /// Set CLIP encoder
    pub fn with_clip_encoder(mut self, encoder: Box<dyn TextEncoder>) -> Self {
        self.clip_encoder = Some(encoder);
        self
    }
    
    /// Preprocess entire dataset
    pub async fn preprocess_dataset(
        &mut self,
        dataset: &dyn Dataset,
    ) -> Result<Vec<PreprocessedFluxItem>> {
        let vae = self.vae.as_ref()
            .ok_or_else(|| Error::Config("VAE not set".into()))?;
        let t5 = self.t5_encoder.as_ref()
            .ok_or_else(|| Error::Config("T5 encoder not set".into()))?;
        let clip = self.clip_encoder.as_ref()
            .ok_or_else(|| Error::Config("CLIP encoder not set".into()))?;
        
        let total_items = dataset.len();
        let pb = ProgressBar::new(total_items as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}")
                .unwrap()
        );
        
        let mut preprocessed_items = Vec::new();
        
        // Process in batches
        for batch_start in (0..total_items).step_by(self.config.batch_size) {
            let batch_end = (batch_start + self.config.batch_size).min(total_items);
            let batch_size = batch_end - batch_start;
            
            // Collect batch items
            let mut batch_items = Vec::new();
            for idx in batch_start..batch_end {
                let item = dataset.get_item(idx)?;
                batch_items.push(item);
            }
            
            // Process images through VAE
            pb.set_message("Encoding images with VAE...");
            let latents = self.encode_images_batch(vae.as_ref(), &batch_items).await?;
            
            // Process text through T5
            pb.set_message("Encoding text with T5...");
            let captions: Vec<String> = batch_items.iter()
                .map(|item| item.caption.clone())
                .collect();
            let (t5_embeds, _) = t5.encode(&captions)?;
            
            // Process text through CLIP
            pb.set_message("Encoding text with CLIP...");
            let (_, clip_pooled) = clip.encode(&captions)?;
            
            // Save each item
            for i in 0..batch_size {
                let item_idx = batch_start + i;
                let item = &batch_items[i];
                
                // Create unique filename for this item
                let item_hash = format!("{:016x}", item_idx);
                let latents_path = self.config.cache_dir.join(format!("{}_latents.safetensors", item_hash));
                let t5_path = self.config.cache_dir.join(format!("{}_t5.safetensors", item_hash));
                let clip_path = self.config.cache_dir.join(format!("{}_clip_pooled.safetensors", item_hash));
                
                // Check if already exists
                if !self.config.overwrite && 
                   latents_path.exists() && 
                   t5_path.exists() && 
                   clip_path.exists() {
                    pb.inc(1);
                    preprocessed_items.push(PreprocessedFluxItem {
                        latents_path,
                        t5_embeds_path: t5_path,
                        clip_pooled_path: clip_path,
                        caption: item.caption.clone(),
                        metadata: item.metadata.clone(),
                    });
                    continue;
                }
                
                // Extract individual tensors
                let latent_i = latents.narrow(0, i, 1)?;
                let t5_i = t5_embeds.narrow(0, i, 1)?;
                let clip_i = clip_pooled.as_ref()
                    .ok_or_else(|| Error::Training("CLIP pooled output missing".into()))?
                    .narrow(0, i, 1)?;
                
                // Save tensors
                self.save_tensor(&latent_i, &latents_path)?;
                self.save_tensor(&t5_i, &t5_path)?;
                self.save_tensor(&clip_i, &clip_path)?;
                
                preprocessed_items.push(PreprocessedFluxItem {
                    latents_path,
                    t5_embeds_path: t5_path,
                    clip_pooled_path: clip_path,
                    caption: item.caption.clone(),
                    metadata: item.metadata.clone(),
                });
                
                pb.inc(1);
            }
        }
        
        pb.finish_with_message("Preprocessing complete!");
        
        // Clear encoders from memory
        println!("Freeing encoder memory...");
        self.vae = None;
        self.t5_encoder = None;
        self.clip_encoder = None;
        
        Ok(preprocessed_items)
    }
    
    /// Encode a batch of images
    async fn encode_images_batch(
        &self,
        vae: &dyn VAE,
        items: &[DatasetItem],
    ) -> Result<Tensor> {
        // Stack images
        let images: Vec<Tensor> = items.iter()
            .map(|item| item.image.unsqueeze(0).map_err(Error::from))
            .collect::<Result<Vec<_>>>()?;
        let images = Tensor::cat(&images, 0)?;
        
        // Validate input images
        let image_data = images.flatten_all()?.to_vec1::<f32>()?;
        if image_data.iter().any(|x| x.is_nan() || x.is_infinite()) {
            return Err(Error::InvalidData("NaN or Inf detected in input images".into()));
        }
        
        // Normalize images for VAE using architecture-specific normalization
        let normalized = self.vae_normalizer.normalize_for_vae(&images)?;
        
        // Encode through VAE
        let latents = vae.encode(&normalized)?;
        
        // Apply VAE scaling factor using normalizer
        let scaled_latents = self.vae_normalizer.scale_latents(&latents)?;
        
        // Validate encoded latents
        let latent_data = scaled_latents.flatten_all()?.to_vec1::<f32>()?;
        if latent_data.iter().any(|x| x.is_nan() || x.is_infinite()) {
            return Err(Error::InvalidData("NaN or Inf detected in encoded latents".into()));
        }
        
        Ok(scaled_latents)
    }
    
    /// Save a tensor to disk
    fn save_tensor(&self, tensor: &Tensor, path: &Path) -> Result<()> {
        let mut tensors = HashMap::new();
        tensors.insert("data".to_string(), tensor.clone());
        
        let data = serialize(&tensors, &None)?;
        fs::write(path, data)?;
        
        Ok(())
    }
}

/// Preprocessed dataset that loads from disk
pub struct PreprocessedFluxDataset {
    items: Vec<PreprocessedFluxItem>,
    device: Device,
}

impl PreprocessedFluxDataset {
    pub fn new(items: Vec<PreprocessedFluxItem>, device: Device) -> Self {
        Self { items, device }
    }
    
    pub fn len(&self) -> usize {
        self.items.len()
    }
    
    pub fn get_item(&self, idx: usize) -> Result<PreprocessedFluxBatch> {
        let item = &self.items[idx];
        
        // Load tensors from disk
        let latents = self.load_tensor(&item.latents_path)?;
        let t5_embeds = self.load_tensor(&item.t5_embeds_path)?;
        let clip_pooled = self.load_tensor(&item.clip_pooled_path)?;
        
        Ok(PreprocessedFluxBatch {
            latents,
            t5_embeds,
            clip_pooled,
            caption: item.caption.clone(),
            metadata: item.metadata.clone(),
        })
    }
    
    fn load_tensor(&self, path: &Path) -> Result<Tensor> {
        let data = fs::read(path)?;
        let tensors = SafeTensors::deserialize(&data)?;
        let tensor_view = tensors.tensor("data")?;
        
        // Convert to Candle tensor
        let shape = tensor_view.shape().to_vec();
        let device = crate::flux_model_loader::to_candle_device(&self.device)?;
        
        match tensor_view.dtype() {
            safetensors::Dtype::F32 => {
                let data: Vec<f32> = tensor_view.data()
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();
                Tensor::from_vec(data, shape.as_slice(), &device)
                    .map_err(Error::from)
            }
            safetensors::Dtype::F16 => {
                let data: Vec<half::f16> = tensor_view.data()
                    .chunks_exact(2)
                    .map(|b| half::f16::from_le_bytes([b[0], b[1]]))
                    .collect();
                Tensor::from_vec(data, shape.as_slice(), &device)?
                    .to_dtype(DType::F32)
                    .map_err(Error::from)
            }
            _ => Err(Error::Training("Unsupported tensor dtype".into())),
        }
    }
}

/// Batch of preprocessed data
pub struct PreprocessedFluxBatch {
    pub latents: Tensor,
    pub t5_embeds: Tensor,
    pub clip_pooled: Tensor,
    pub caption: String,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Preprocessing statistics
pub fn print_memory_savings(dataset_size: usize) {
    println!("\n💾 Memory Savings with Preprocessing:");
    println!("─────────────────────────────────────");
    
    // Without preprocessing
    let vae_mem = 1.7; // GB
    let t5_mem = 11.0; // GB  
    let clip_mem = 0.5; // GB
    let flux_mem = 12.0; // GB
    let total_without = vae_mem + t5_mem + clip_mem + flux_mem;
    
    // With preprocessing
    let flux_only = 12.0; // GB
    let gradients = 3.0; // GB with checkpointing
    let optimizer = 6.0; // GB
    let activations = 2.0; // GB
    let total_with = flux_only + gradients + optimizer + activations;
    
    println!("Without preprocessing:");
    println!("  VAE:        {:>5.1} GB", vae_mem);
    println!("  T5-XXL:     {:>5.1} GB", t5_mem);
    println!("  CLIP-L:     {:>5.1} GB", clip_mem);
    println!("  Flux:       {:>5.1} GB", flux_mem);
    println!("  Total:      {:>5.1} GB ❌ Won't fit!", total_without);
    
    println!("\nWith preprocessing:");
    println!("  Flux only:  {:>5.1} GB", flux_only);
    println!("  Gradients:  {:>5.1} GB", gradients);
    println!("  Optimizer:  {:>5.1} GB", optimizer);
    println!("  Activations:{:>5.1} GB", activations);
    println!("  Total:      {:>5.1} GB ✅ Fits in 24GB!", total_with);
    
    println!("\n📈 Dataset preprocessing:");
    let images_size = dataset_size as f64 * 1024.0 * 1024.0 * 3.0 / 1e9; // 1024x1024 RGB
    let latents_size = dataset_size as f64 * 128.0 * 128.0 * 16.0 * 4.0 / 1e9; // 128x128x16 f32
    let text_size = dataset_size as f64 * (512.0 * 4096.0 + 768.0) * 4.0 / 1e9; // T5 + CLIP f32
    
    println!("  Raw images:    {:>6.1} GB", images_size);
    println!("  → Latents:     {:>6.1} GB", latents_size);
    println!("  → Text embeds: {:>6.1} GB", text_size);
    println!("  Disk total:    {:>6.1} GB", latents_size + text_size);
}