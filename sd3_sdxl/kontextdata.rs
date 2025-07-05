//! Flux Kontext extensions for the existing data pipeline

use eridiffusion_core::{Result, Error, Device, ModelArchitecture};
use candle_core::{Tensor, DType};
use std::path::{Path, PathBuf};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use tracing::{info, debug};

use crate::{
    DataPreprocessor, ResolutionConfig, 
    DatasetItem, DatasetConfig,
    VAEConfig,
};

// Extend ModelArchitecture enum (this would go in your core crate)
/*
pub enum ModelArchitecture {
    // ... existing variants ...
    FluxKontext,
}
*/

/// Flux Kontext specific dataset configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FluxKontextDatasetConfig {
    pub base: DatasetConfig,
    pub control_image_dir: Option<PathBuf>,
    pub control_image_ext: String,
    pub control_strength: f32,
    pub control_guidance_start: f32,
    pub control_guidance_end: f32,
    pub enable_control_preprocessing: bool,
}

impl Default for FluxKontextDatasetConfig {
    fn default() -> Self {
        Self {
            base: DatasetConfig {
                root_dir: PathBuf::from("./data"),
                caption_ext: "txt".to_string(),
                resolution: 1024,
                center_crop: false,
                random_flip: true,
                cache_latents: true,
                cache_dir: Some(PathBuf::from(".cache/flux_kontext_latents")),
            },
            control_image_dir: None,
            control_image_ext: "jpg".to_string(),
            control_strength: 1.0,
            control_guidance_start: 0.0,
            control_guidance_end: 1.0,
            enable_control_preprocessing: true,
        }
    }
}

/// Extended dataset item for Flux Kontext with control images
#[derive(Debug, Clone)]
pub struct FluxKontextDatasetItem {
    pub base: DatasetItem,
    pub control_image: Option<Tensor>,
    pub control_strength: f32,
    pub control_metadata: HashMap<String, serde_json::Value>,
}

impl FluxKontextDatasetItem {
    pub fn from_base(base: DatasetItem) -> Self {
        Self {
            base,
            control_image: None,
            control_strength: 1.0,
            control_metadata: HashMap::new(),
        }
    }
    
    pub fn with_control_image(mut self, control_image: Tensor, strength: f32) -> Self {
        self.control_image = Some(control_image);
        self.control_strength = strength;
        self
    }
}

/// Flux Kontext preprocessor
pub struct FluxKontextPreprocessor {
    base_preprocessor: Box<dyn DataPreprocessor>,
    enable_control: bool,
}

impl FluxKontextPreprocessor {
    pub fn new(enable_control: bool) -> Self {
        Self {
            base_preprocessor: Box::new(FluxBasePreprocessor::new()),
            enable_control,
        }
    }
    
    /// Preprocess control image
    pub fn preprocess_control_image(&self, control_image: &Tensor) -> Result<Tensor> {
        // Control images for Flux Kontext need special preprocessing
        // Similar to base image but may need different normalization
        
        // Ensure correct dtype
        let processed = if control_image.dtype() != DType::F32 {
            control_image.to_dtype(DType::F32)?
        } else {
            control_image.clone()
        };
        
        // Normalize to [-1, 1] (same as main image for Flux)
        let normalized = if processed.max(0)?.max(1)?.max(2)?.to_scalar::<f32>()? > 1.1 {
            // Assume [0, 255] range
            processed.affine(2.0 / 255.0, -1.0)?
        } else {
            // Assume [0, 1] range
            processed.affine(2.0, -1.0)?
        };
        
        Ok(normalized)
    }
    
    /// Combine main and control images for training
    pub fn prepare_training_inputs(
        &self,
        main_image: &Tensor,
        control_image: Option<&Tensor>,
    ) -> Result<Tensor> {
        let main_processed = self.base_preprocessor.preprocess_image(main_image)?;
        
        if let Some(control) = control_image {
            let control_processed = self.preprocess_control_image(control)?;
            
            // Ensure both images have the same spatial dimensions
            let main_dims = main_processed.dims();
            let control_dims = control_processed.dims();
            
            if main_dims[1..] != control_dims[1..] {
                // Resize control to match main
                let control_resized = resize_to_match(&control_processed, &main_processed)?;
                return Ok(control_resized);
            }
            
            Ok(control_processed)
        } else {
            Ok(main_processed)
        }
    }
}

impl DataPreprocessor for FluxKontextPreprocessor {
    fn preprocess_image(&self, image: &Tensor) -> Result<Tensor> {
        self.base_preprocessor.preprocess_image(image)
    }
    
    fn preprocess_caption(&self, caption: &str) -> Result<String> {
        self.base_preprocessor.preprocess_caption(caption)
    }
}

/// Base Flux preprocessor
struct FluxBasePreprocessor;

impl FluxBasePreprocessor {
    fn new() -> Self {
        Self
    }
}

impl DataPreprocessor for FluxBasePreprocessor {
    fn preprocess_image(&self, image: &Tensor) -> Result<Tensor> {
        // Flux specific image preprocessing
        let processed = if image.dtype() != DType::F32 {
            image.to_dtype(DType::F32)?
        } else {
            image.clone()
        };
        
        // Flux uses [-1, 1] normalization
        let normalized = if processed.max(0)?.max(1)?.max(2)?.to_scalar::<f32>()? > 1.1 {
            // Assume [0, 255] range
            processed.affine(2.0 / 255.0, -1.0)?
        } else {
            // Assume [0, 1] range  
            processed.affine(2.0, -1.0)?
        };
        
        Ok(normalized)
    }
    
    fn preprocess_caption(&self, caption: &str) -> Result<String> {
        // Flux Kontext supports longer captions (up to 512 tokens)
        let caption = caption.trim();
        let caption = caption.split_whitespace().collect::<Vec<_>>().join(" ");
        let words: Vec<&str> = caption.split_whitespace().collect();
        let truncated = words[..words.len().min(512)].join(" ");
        Ok(truncated)
    }
}

/// Extend ResolutionConfig for Flux Kontext
impl ResolutionConfig {
    pub fn flux_kontext() -> Self {
        Self {
            base_resolution: 1024,
            base_resolutions: vec![512, 768, 1024, 1536, 2048],
            min_resolution: 256,
            max_resolution: 2048,
            divisor: 16, // Flux uses 16x downsampling
            aspect_ratios: vec![
                1.0,    // Square
                0.75,   // 3:4 portrait
                1.33,   // 4:3 landscape  
                0.67,   // 2:3 portrait
                1.5,    // 3:2 landscape
                0.5,    // 1:2 portrait
                2.0,    // 2:1 landscape
                0.25,   // 1:4 portrait
                4.0,    // 4:1 landscape
                9.0/16.0, // 9:16 vertical video
                16.0/9.0, // 16:9 horizontal video
            ],
        }
    }
}

/// Extend VAEConfig for Flux Kontext
impl VAEConfig {
    pub fn flux_kontext() -> Self {
        Self {
            latent_channels: 16, // Flux uses 16-channel latents
            downsampling_factor: 8, // Flux VAE downsamples by 8x
            scale_factor: 0.3611, // Flux specific scaling factor
            use_tiling: true,
            tile_size: 1024,
            tile_overlap: 128,
        }
    }
}

/// Flux Kontext dataset that handles control images
pub struct FluxKontextDataset {
    config: FluxKontextDatasetConfig,
    base_dataset: Box<dyn crate::Dataset>,
    control_image_paths: HashMap<PathBuf, PathBuf>,
    preprocessor: FluxKontextPreprocessor,
    device: Device,
}

impl FluxKontextDataset {
    pub fn new(config: FluxKontextDatasetConfig) -> Result<Self> {
        // Create base dataset
        let base_dataset = Box::new(crate::ImageDataset::new(config.base.clone())?);
        
        // Scan for control images if directory specified
        let control_image_paths = if let Some(control_dir) = &config.control_image_dir {
            Self::scan_control_images(&base_dataset, control_dir, &config.control_image_ext)?
        } else {
            HashMap::new()
        };
        
        let preprocessor = FluxKontextPreprocessor::new(config.control_image_dir.is_some());
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        
        info!("Created Flux Kontext dataset with {} control images", 
              control_image_paths.len());
        
        Ok(Self {
            config,
            base_dataset,
            control_image_paths,
            preprocessor,
            device,
        })
    }
    
    /// Scan for control images matching base images
    fn scan_control_images(
        base_dataset: &Box<dyn crate::Dataset>,
        control_dir: &Path,
        control_ext: &str,
    ) -> Result<HashMap<PathBuf, PathBuf>> {
        let mut control_paths = HashMap::new();
        
        for i in 0..base_dataset.len() {
            let item = base_dataset.get_item(i)?;
            
            // Extract image path from metadata
            let image_path = item.metadata.get("image_path")
                .and_then(|v| v.as_str())
                .ok_or_else(|| Error::DataError("Missing image_path in metadata".into()))?;
            
            let base_path = PathBuf::from(image_path);
            
            // Look for corresponding control image
            let control_name = base_path.file_stem()
                .ok_or_else(|| Error::DataError("Invalid image path".into()))?;
            
            let control_path = control_dir.join(format!("{}.{}", 
                control_name.to_string_lossy(), control_ext));
            
            if control_path.exists() {
                control_paths.insert(base_path, control_path);
            }
        }
        
        Ok(control_paths)
    }
    
    /// Load control image
    fn load_control_image(&self, control_path: &Path) -> Result<Tensor> {
        use image::{ImageReader, DynamicImage};
        
        let img = ImageReader::open(control_path)
            .map_err(|e| Error::DataError(format!("Failed to open control image: {}", e)))?
            .decode()
            .map_err(|e| Error::DataError(format!("Failed to decode control image: {}", e)))?;
        
        // Convert to RGB
        let img = img.to_rgb8();
        let (width, height) = img.dimensions();
        
        // Convert device
        let candle_device = match &self.device {
            Device::Cpu => candle_core::Device::Cpu,
            Device::Cuda(id) => candle_core::Device::new_cuda(*id)?,
        };
        
        // Convert to tensor [3, H, W]
        let raw_pixels: Vec<u8> = img.into_raw();
        
        // Reorganize from HWC to CHW
        let mut data = vec![0.0f32; 3 * (width * height) as usize];
        for c in 0..3 {
            for y in 0..height {
                for x in 0..width {
                    let src_idx = ((y * width + x) * 3 + c) as usize;
                    let dst_idx = (c * height * width + y * width + x) as usize;
                    data[dst_idx] = raw_pixels[src_idx] as f32;
                }
            }
        }
        
        let tensor = Tensor::from_vec(
            data,
            &[3, height as usize, width as usize],
            &candle_device,
        )?;
        
        // Preprocess control image
        self.preprocessor.preprocess_control_image(&tensor)
    }
}

impl crate::Dataset for FluxKontextDataset {
    fn len(&self) -> usize {
        self.base_dataset.len()
    }
    
    fn get_item(&self, index: usize) -> Result<crate::DatasetItem> {
        let base_item = self.base_dataset.get_item(index)?;
        
        // Check if we have a control image for this item
        let control_image = if let Some(image_path_value) = base_item.metadata.get("image_path") {
            if let Some(image_path_str) = image_path_value.as_str() {
                let image_path = PathBuf::from(image_path_str);
                
                if let Some(control_path) = self.control_image_paths.get(&image_path) {
                    match self.load_control_image(control_path) {
                        Ok(control) => Some(control),
                        Err(e) => {
                            debug!("Failed to load control image {}: {}", control_path.display(), e);
                            None
                        }
                    }
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };
        
        // Add control metadata
        let mut metadata = base_item.metadata.clone();
        if control_image.is_some() {
            metadata.insert("has_control".to_string(), serde_json::Value::Bool(true));
            metadata.insert("control_strength".to_string(), 
                serde_json::Value::Number(serde_json::Number::from_f64(self.config.control_strength as f64).unwrap()));
        } else {
            metadata.insert("has_control".to_string(), serde_json::Value::Bool(false));
        }
        
        // For now, we'll store the control image in metadata as a special field
        // In a real implementation, you might want to extend DatasetItem
        let mut item = crate::DatasetItem {
            image: base_item.image,
            caption: base_item.caption,
            metadata,
        };
        
        // Store control image tensor in a special way (this is a hack for the existing structure)
        if let Some(control) = control_image {
            // You would want to extend DatasetItem to properly support this
            // For now, we'll indicate it's available in metadata
            debug!("Control image available for item {}", index);
        }
        
        Ok(item)
    }
    
    fn metadata(&self) -> &crate::DatasetMetadata {
        self.base_dataset.metadata()
    }
}

/// Extended DataLoader batch for Flux Kontext
#[derive(Debug, Clone)]
pub struct FluxKontextBatch {
    pub base: crate::DataLoaderBatch,
    pub control_images: Option<Tensor>,
    pub control_strengths: Vec<f32>,
    pub has_control: Vec<bool>,
}

impl FluxKontextBatch {
    pub fn from_base(base: crate::DataLoaderBatch) -> Self {
        let batch_size = base.batch_size();
        Self {
            base,
            control_images: None,
            control_strengths: vec![1.0; batch_size],
            has_control: vec![false; batch_size],
        }
    }
    
    pub fn with_control(mut self, control_images: Tensor, strengths: Vec<f32>) -> Self {
        let batch_size = control_images.dims()[0];
        self.control_images = Some(control_images);
        self.control_strengths = strengths;
        self.has_control = vec![true; batch_size];
        self
    }
    
    pub fn batch_size(&self) -> usize {
        self.base.batch_size()
    }
    
    pub fn to_device(&self, device: &Device) -> Result<Self> {
        let candle_device = match device {
            Device::Cpu => candle_core::Device::Cpu,
            Device::Cuda(id) => candle_core::Device::new_cuda(*id)?,
        };
        
        let base = self.base.to_device(device)?;
        let control_images = self.control_images.as_ref()
            .map(|c| c.to_device(&candle_device))
            .transpose()?;
        
        Ok(Self {
            base,
            control_images,
            control_strengths: self.control_strengths.clone(),
            has_control: self.has_control.clone(),
        })
    }
}

/// Collate function for Flux Kontext batches
pub fn collate_flux_kontext_batch(
    items: Vec<crate::DatasetItem>, 
    device: &Device
) -> Result<FluxKontextBatch> {
    // Create base batch
    let base_batch = crate::collate_fn(items.clone(), device)?;
    
    // Check for control images
    let has_control_data: Vec<bool> = items.iter()
        .map(|item| item.metadata.get("has_control")
            .and_then(|v| v.as_bool())
            .unwrap_or(false))
        .collect();
    
    let any_control = has_control_data.iter().any(|&x| x);
    
    if any_control {
        // For now, since we don't have control images in the actual tensors,
        // we'll create placeholder control images
        // In a real implementation, you'd extract them from the items
        
        let candle_device = match device {
            Device::Cpu => candle_core::Device::Cpu,
            Device::Cuda(id) => candle_core::Device::new_cuda(*id)?,
        };
        
        let batch_size = items.len();
        let (_, h, w) = match base_batch.images.dims() {
            [b, c, h, w] => (*b, *h, *w),
            _ => return Err(Error::InvalidShape("Expected BCHW format".into())),
        };
        
        // Create placeholder control images (zeros for items without control)
        let control_images = Tensor::zeros(
            &[batch_size, 3, h, w],
            DType::F32,
            &candle_device,
        )?;
        
        let control_strengths: Vec<f32> = items.iter()
            .map(|item| item.metadata.get("control_strength")
                .and_then(|v| v.as_f64())
                .map(|v| v as f32)
                .unwrap_or(1.0))
            .collect();
        
        Ok(FluxKontextBatch::from_base(base_batch)
            .with_control(control_images, control_strengths))
    } else {
        Ok(FluxKontextBatch::from_base(base_batch))
    }
}

/// Helper function to resize tensor to match another tensor's spatial dimensions
fn resize_to_match(source: &Tensor, target: &Tensor) -> Result<Tensor> {
    let target_dims = target.dims();
    let source_dims = source.dims();
    
    if target_dims.len() < 2 || source_dims.len() < 2 {
        return Err(Error::InvalidShape("Tensors must have at least 2 dimensions".into()));
    }
    
    let target_h = target_dims[target_dims.len() - 2];
    let target_w = target_dims[target_dims.len() - 1];
    
    // Simple nearest neighbor resize (in production, you'd want proper interpolation)
    // This is a placeholder implementation
    Ok(source.clone()) // TODO: Implement actual resizing
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_flux_kontext_resolution_config() {
        let config = ResolutionConfig::flux_kontext();
        assert_eq!(config.base_resolution, 1024);
        assert_eq!(config.latent_channels, 16);
        assert_eq!(config.divisor, 16);
        assert!(config.aspect_ratios.contains(&1.0)); // Square
        assert!(config.aspect_ratios.contains(&(16.0/9.0))); // Widescreen
    }
    
    #[test]
    fn test_flux_kontext_vae_config() {
        let config = VAEConfig::flux_kontext();
        assert_eq!(config.latent_channels, 16);
        assert_eq!(config.downsampling_factor, 8);
        assert_eq!(config.scale_factor, 0.3611);
        assert!(config.use_tiling);
    }
    
    #[test]
    fn test_flux_kontext_preprocessor() {
        let preprocessor = FluxKontextPreprocessor::new(true);
        
        // Test caption preprocessing
        let long_caption = "This is a very long caption ".repeat(100);
        let processed = preprocessor.preprocess_caption(&long_caption).unwrap();
        let word_count = processed.split_whitespace().count();
        assert!(word_count <= 512, "Caption should be truncated to 512 words");
    }
}
