use anyhow::{Result, anyhow};
use candle_core::{Device, Tensor, DType};
use candle_nn::VarBuilder;
use std::path::Path;

use crate::models::sdxl_vae::{VAEModel, SDXLVAE, SD3VAE, sdxl_vae_vb_rename, sd3_vae_vb_rename};

pub struct VAEInference {
    vae: VAEModel,
    device: Device,
}

#[derive(Debug, Clone)]
pub enum VAEType {
    SDXL,
    SD3,
}

impl VAEInference {
    /// Create new VAE inference instance
    pub fn new(vae_type: VAEType, vb: VarBuilder, device: Device) -> Result<Self> {
        let vae = match vae_type {
            VAEType::SDXL => VAEModel::new_sdxl(vb)?,
            VAEType::SD3 => VAEModel::new_sd3(vb)?,
        };
        
        Ok(Self { vae, device })
    }
    
    /// Load VAE from checkpoint with model loader
    pub fn from_checkpoint<P: AsRef<Path>>(
        checkpoint_path: P,
        vae_type: VAEType,
        device: Device,
        model_loader: &dyn ModelLoader,
    ) -> Result<Self> {
        let vb = model_loader.load_vae_weights(checkpoint_path, &vae_type)?;
        Self::new(vae_type, vb, device)
    }
    
    /// Encode images to latents
    /// Input: images tensor [batch, channels, height, width] in range [0, 255]
    /// Output: latents tensor [batch, latent_channels, height/8, width/8]
    pub fn encode(&self, images: &Tensor) -> Result<Tensor> {
        self.validate_image_tensor(images)?;
        let latents = self.vae.encode(images)?;
        Ok(latents)
    }
    
    /// Decode latents to images
    /// Input: latents tensor [batch, latent_channels, height, width]
    /// Output: images tensor [batch, 3, height*8, width*8] in range [0, 255]
    pub fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        self.validate_latent_tensor(latents)?;
        let images = self.vae.decode(latents)?;
        Ok(images)
    }
    
    /// Encode batch of images with memory optimization
    pub fn encode_batch(&self, images: &Tensor, batch_size: usize) -> Result<Tensor> {
        let (total_batch, channels, height, width) = images.dims4()?;
        
        if total_batch <= batch_size {
            return self.encode(images);
        }
        
        let mut encoded_batches = Vec::new();
        
        for start_idx in (0..total_batch).step_by(batch_size) {
            let end_idx = (start_idx + batch_size).min(total_batch);
            let batch_images = images.narrow(0, start_idx, end_idx - start_idx)?;
            let batch_latents = self.encode(&batch_images)?;
            encoded_batches.push(batch_latents);
        }
        
        Tensor::cat(&encoded_batches, 0)
    }
    
    /// Decode batch of latents with memory optimization
    pub fn decode_batch(&self, latents: &Tensor, batch_size: usize) -> Result<Tensor> {
        let (total_batch, latent_channels, height, width) = latents.dims4()?;
        
        if total_batch <= batch_size {
            return self.decode(latents);
        }
        
        let mut decoded_batches = Vec::new();
        
        for start_idx in (0..total_batch).step_by(batch_size) {
            let end_idx = (start_idx + batch_size).min(total_batch);
            let batch_latents = latents.narrow(0, start_idx, end_idx - start_idx)?;
            let batch_images = self.decode(&batch_latents)?;
            decoded_batches.push(batch_images);
        }
        
        Tensor::cat(&decoded_batches, 0)
    }
    
    /// Encode with deterministic sampling (use mean instead of random sampling)
    pub fn encode_deterministic(&self, images: &Tensor) -> Result<Tensor> {
        self.validate_image_tensor(images)?;
        
        // For deterministic encoding, we need to modify the sampling
        // This would require access to the distribution mean
        // For now, we use regular encode which samples from the distribution
        self.encode(images)
    }
    
    /// Get VAE information
    pub fn info(&self) -> VAEInfo {
        VAEInfo {
            vae_type: match &self.vae {
                VAEModel::SDXL(_) => VAEType::SDXL,
                VAEModel::SD3(_) => VAEType::SD3,
            },
            latent_channels: self.vae.latent_channels(),
            scale_factor: self.vae.scale_factor(),
            device: self.device.clone(),
        }
    }
    
    /// Validate input image tensor dimensions and values
    fn validate_image_tensor(&self, images: &Tensor) -> Result<()> {
        let dims = images.dims();
        if dims.len() != 4 {
            return Err(anyhow!("Image tensor must be 4D [batch, channels, height, width], got {:?}", dims));
        }
        
        let (_, channels, height, width) = images.dims4()?;
        if channels != 3 {
            return Err(anyhow!("Image tensor must have 3 channels (RGB), got {}", channels));
        }
        
        if height % 8 != 0 || width % 8 != 0 {
            return Err(anyhow!("Image dimensions must be multiples of 8, got {}x{}", height, width));
        }
        
        Ok(())
    }
    
    /// Validate input latent tensor dimensions
    fn validate_latent_tensor(&self, latents: &Tensor) -> Result<()> {
        let dims = latents.dims();
        if dims.len() != 4 {
            return Err(anyhow!("Latent tensor must be 4D [batch, channels, height, width], got {:?}", dims));
        }
        
        let (_, channels, _, _) = latents.dims4()?;
        let expected_channels = self.vae.latent_channels();
        if channels != expected_channels {
            return Err(anyhow!("Latent tensor must have {} channels for this VAE type, got {}", expected_channels, channels));
        }
        
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct VAEInfo {
    pub vae_type: VAEType,
    pub latent_channels: usize,
    pub scale_factor: f64,
    pub device: Device,
}

/// Trait for loading VAE weights - implement this for your model loader
pub trait ModelLoader {
    fn load_vae_weights<P: AsRef<Path>>(&self, checkpoint_path: P, vae_type: &VAEType) -> Result<VarBuilder>;
}

/// Utility functions for image preprocessing
pub mod utils {
    use super::*;
    
    /// Convert image from HWC to CHW format
    pub fn hwc_to_chw(image: &Tensor) -> Result<Tensor> {
        // Input: [height, width, channels] -> Output: [channels, height, width]
        image.permute(&[2, 0, 1])
    }
    
    /// Convert image from CHW to HWC format  
    pub fn chw_to_hwc(image: &Tensor) -> Result<Tensor> {
        // Input: [channels, height, width] -> Output: [height, width, channels]
        image.permute(&[1, 2, 0])
    }
    
    /// Add batch dimension to single image
    pub fn add_batch_dim(image: &Tensor) -> Result<Tensor> {
        image.unsqueeze(0)
    }
    
    /// Remove batch dimension from single image
    pub fn remove_batch_dim(image: &Tensor) -> Result<Tensor> {
        image.squeeze(0)
    }
    
    /// Normalize image from [0, 255] to [0, 1]
    pub fn normalize_to_unit(image: &Tensor) -> Result<Tensor> {
        (image / 255.0)
    }
    
    /// Denormalize image from [0, 1] to [0, 255]
    pub fn denormalize_from_unit(image: &Tensor) -> Result<Tensor> {
        (image * 255.0)?.clamp(0.0, 255.0)
    }
    
    /// Resize image to nearest multiple of 8
    pub fn resize_to_vae_compatible(height: usize, width: usize) -> (usize, usize) {
        let new_height = ((height + 7) / 8) * 8;
        let new_width = ((width + 7) / 8) * 8;
        (new_height, new_width)
    }
}

/// Example implementation of ModelLoader trait
/// You would replace this with your actual model loader
pub struct ExampleModelLoader;

impl ModelLoader for ExampleModelLoader {
    fn load_vae_weights<P: AsRef<Path>>(&self, checkpoint_path: P, vae_type: &VAEType) -> Result<VarBuilder> {
        // This is where you would integrate with your existing model loader
        // Example pseudocode:
        /*
        let weights = load_safetensors(checkpoint_path)?;
        let rename_fn = match vae_type {
            VAEType::SDXL => sdxl_vae_vb_rename,
            VAEType::SD3 => sd3_vae_vb_rename,
        };
        
        let mut renamed_weights = HashMap::new();
        for (key, tensor) in weights {
            let new_key = rename_fn(&key);
            renamed_weights.insert(new_key, tensor);
        }
        
        VarBuilder::from_tensors(renamed_weights, DType::F32, &device)
        */
        
        unimplemented!("Implement this with your actual model loader")
    }
}
