//! Architecture-specific VAE normalization for proper encoding/decoding

use eridiffusion_core::{Result, Error, ModelArchitecture};
use candle_core::{Tensor, DType};
use tracing::{debug, warn};

/// VAE normalization configuration for different architectures
#[derive(Debug, Clone)]
pub struct VAENormalization {
    /// Input normalization parameters
    pub input_mean: Vec<f32>,
    pub input_std: Vec<f32>,
    
    /// VAE scaling factor
    pub vae_scale_factor: f32,
    
    /// Additional scaling for specific architectures
    pub additional_scale: Option<f32>,
    
    /// Whether to use channel-wise normalization
    pub channel_wise: bool,
    
    /// Custom preprocessing function name
    pub preprocess_fn: Option<String>,
}

impl VAENormalization {
    /// Get normalization config for architecture
    pub fn for_architecture(arch: &ModelArchitecture) -> Self {
        match arch {
            ModelArchitecture::SD15 | ModelArchitecture::SD2 => Self {
                // SD 1.5/2.x uses standard ImageNet normalization
                input_mean: vec![0.5, 0.5, 0.5],
                input_std: vec![0.5, 0.5, 0.5],
                vae_scale_factor: 0.18215,
                additional_scale: None,
                channel_wise: false,
                preprocess_fn: None,
            },
            
            ModelArchitecture::SDXL => Self {
                // SDXL uses different VAE scaling
                input_mean: vec![0.5, 0.5, 0.5],
                input_std: vec![0.5, 0.5, 0.5],
                vae_scale_factor: 0.13025,
                additional_scale: None,
                channel_wise: false,
                preprocess_fn: None,
            },
            
            ModelArchitecture::SD3 | ModelArchitecture::SD35 => Self {
                // SD3/3.5 has specific normalization
                input_mean: vec![0.5, 0.5, 0.5],
                input_std: vec![0.5, 0.5, 0.5],
                vae_scale_factor: 0.13025,
                additional_scale: Some(1.5305),  // SD3 specific scaling
                channel_wise: false,
                preprocess_fn: Some("sd3_preprocess".into()),
            },
            
            ModelArchitecture::Flux | ModelArchitecture::FluxSchnell | ModelArchitecture::FluxDev => Self {
                // Flux uses different normalization completely
                input_mean: vec![0.0, 0.0, 0.0],
                input_std: vec![1.0, 1.0, 1.0],
                vae_scale_factor: 0.3611,
                additional_scale: Some(0.1),  // Additional Flux scaling
                channel_wise: false,
                preprocess_fn: Some("flux_preprocess".into()),
            },
            
            _ => Self::default(),
        }
    }
    
    /// Default normalization (SD 1.5 style)
    fn default() -> Self {
        Self {
            input_mean: vec![0.5, 0.5, 0.5],
            input_std: vec![0.5, 0.5, 0.5],
            vae_scale_factor: 0.18215,
            additional_scale: None,
            channel_wise: false,
            preprocess_fn: None,
        }
    }
}

/// VAE normalizer that handles architecture-specific normalization
pub struct VAENormalizer {
    config: VAENormalization,
    architecture: ModelArchitecture,
}

impl VAENormalizer {
    pub fn new(architecture: ModelArchitecture) -> Self {
        let config = VAENormalization::for_architecture(&architecture);
        Self {
            config,
            architecture,
        }
    }
    
    /// Normalize image for VAE encoding
    pub fn normalize_for_vae(&self, image: &Tensor) -> Result<Tensor> {
        debug!("Normalizing for VAE with architecture {:?}", self.architecture);
        
        // Validate input is in [0, 1] range
        let min = image.min_all()?.to_scalar::<f32>()?;
        let max = image.max_all()?.to_scalar::<f32>()?;
        
        if min < -0.1 || max > 1.1 {
            warn!("Input image outside expected [0, 1] range: min={}, max={}", min, max);
        }
        
        // Apply architecture-specific preprocessing
        let normalized = match self.config.preprocess_fn.as_deref() {
            Some("sd3_preprocess") => self.sd3_preprocess(image)?,
            Some("flux_preprocess") => self.flux_preprocess(image)?,
            _ => self.standard_normalize(image)?,
        };
        
        // Validate output
        self.validate_normalized(&normalized)?;
        
        Ok(normalized)
    }
    
    /// Standard normalization: [0, 1] -> [-1, 1]
    fn standard_normalize(&self, image: &Tensor) -> Result<Tensor> {
        if self.config.channel_wise && image.dims().len() >= 3 {
            // Channel-wise normalization
            let (c, h, w) = image.dims3()?;
            let mut channels = Vec::new();
            
            for i in 0..c {
                let ch = image.narrow(0, i, 1)?;
                let normalized = ch
                    .affine(
                        1.0 / self.config.input_std[i] as f64,
                        -self.config.input_mean[i] as f64 / self.config.input_std[i] as f64
                    )?;
                channels.push(normalized);
            }
            
            Tensor::cat(&channels, 0)
                .map_err(|e| Error::TensorError(e.to_string()))
        } else {
            // Global normalization (most common)
            // For [0.5, 0.5, 0.5] mean/std: (x - 0.5) / 0.5 = 2x - 1
            image.affine(2.0, -1.0)
                .map_err(|e| Error::TensorError(e.to_string()))
        }
    }
    
    /// SD3-specific preprocessing
    fn sd3_preprocess(&self, image: &Tensor) -> Result<Tensor> {
        // SD3 uses standard [-1, 1] normalization but may need special handling
        let normalized = self.standard_normalize(image)?;
        
        // Some SD3 implementations apply additional clipping
        normalized.clamp(-1.0, 1.0)
            .map_err(|e| Error::TensorError(e.to_string()))
    }
    
    /// Flux-specific preprocessing  
    fn flux_preprocess(&self, image: &Tensor) -> Result<Tensor> {
        // Flux expects [0, 1] input without normalization to [-1, 1]
        // Just ensure it's in the correct range
        image.clamp(0.0, 1.0)
            .map_err(|e| Error::TensorError(e.to_string()))
    }
    
    /// Apply VAE scaling factor to latents
    pub fn scale_latents(&self, latents: &Tensor) -> Result<Tensor> {
        let mut scaled = latents.affine(self.config.vae_scale_factor as f64, 0.0)?;
        
        // Apply additional architecture-specific scaling
        if let Some(additional) = self.config.additional_scale {
            scaled = scaled.affine(additional as f64, 0.0)?;
        }
        
        Ok(scaled)
    }
    
    /// Remove VAE scaling from latents for decoding
    pub fn unscale_latents(&self, latents: &Tensor) -> Result<Tensor> {
        let mut scale = self.config.vae_scale_factor;
        
        // Include additional scaling if present
        if let Some(additional) = self.config.additional_scale {
            scale *= additional;
        }
        
        latents.affine(1.0 / scale as f64, 0.0)
            .map_err(|e| Error::TensorError(e.to_string()))
    }
    
    /// Denormalize VAE output back to [0, 1]
    pub fn denormalize_from_vae(&self, image: &Tensor) -> Result<Tensor> {
        let denormalized = match self.config.preprocess_fn.as_deref() {
            Some("flux_preprocess") => {
                // Flux VAE output is already in [0, 1]
                image.clamp(0.0, 1.0)?
            },
            _ => {
                // Standard VAE output is in [-1, 1], convert to [0, 1]
                image.affine(0.5, 0.5)?.clamp(0.0, 1.0)?
            }
        };
        
        Ok(denormalized)
    }
    
    /// Validate normalized tensor
    fn validate_normalized(&self, tensor: &Tensor) -> Result<()> {
        let data = tensor.flatten_all()?.to_vec1::<f32>()?;
        
        // Check for NaN/Inf
        if data.iter().any(|&x| x.is_nan() || x.is_infinite()) {
            return Err(Error::DataError("NaN or Inf in normalized image".into()));
        }
        
        // Check range based on architecture
        let (expected_min, expected_max) = match self.config.preprocess_fn.as_deref() {
            Some("flux_preprocess") => (0.0, 1.0),
            _ => (-1.0, 1.0),
        };
        
        let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        
        if min < expected_min - 0.1 || max > expected_max + 0.1 {
            warn!(
                "Normalized values outside expected [{}, {}] range: min={}, max={}",
                expected_min, expected_max, min, max
            );
        }
        
        Ok(())
    }
    
    /// Get expected latent channels for architecture
    pub fn latent_channels(&self) -> usize {
        match self.architecture {
            ModelArchitecture::SD3 | ModelArchitecture::SD35 => 16,
            ModelArchitecture::Flux | ModelArchitecture::FluxSchnell | ModelArchitecture::FluxDev => 16,
            _ => 4,
        }
    }
}

/// Batch normalization utilities
pub mod batch_ops {
    use super::*;
    
    /// Normalize a batch of images
    pub fn normalize_batch(
        normalizer: &VAENormalizer,
        batch: &Tensor,
    ) -> Result<Tensor> {
        let (b, c, h, w) = batch.dims4()?;
        let mut normalized = Vec::with_capacity(b);
        
        for i in 0..b {
            let img = batch.narrow(0, i, 1)?;
            let norm = normalizer.normalize_for_vae(&img)?;
            normalized.push(norm);
        }
        
        Tensor::cat(&normalized, 0)
            .map_err(|e| Error::TensorError(e.to_string()))
    }
    
    /// Compute batch statistics for debugging
    pub fn batch_statistics(batch: &Tensor) -> Result<String> {
        let mean = batch.mean_all()?.to_scalar::<f32>()?;
        let mean_val = batch.mean_all()?.to_scalar::<f32>()?;
        let sq_diff = (batch.broadcast_sub(&Tensor::new(mean_val, batch.device())?))?;
        let variance = sq_diff.sqr()?.mean_all()?;
        let std = variance.sqrt()?.to_scalar::<f32>()?;
        let min = batch.min_all()?.to_scalar::<f32>()?;
        let max = batch.max_all()?.to_scalar::<f32>()?;
        
        Ok(format!(
            "Batch stats: mean={:.4}, std={:.4}, min={:.4}, max={:.4}",
            mean, std, min, max
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_architecture_configs() {
        // Test each architecture has valid config
        let architectures = vec![
            ModelArchitecture::SD15,
            ModelArchitecture::SDXL,
            ModelArchitecture::SD3,
            ModelArchitecture::SD35,
            ModelArchitecture::Flux,
        ];
        
        for arch in architectures {
            let config = VAENormalization::for_architecture(&arch);
            assert_eq!(config.input_mean.len(), 3);
            assert_eq!(config.input_std.len(), 3);
            assert!(config.vae_scale_factor > 0.0);
        }
    }
    
    #[test]
    fn test_standard_normalization() {
        // Test [0, 1] -> [-1, 1] conversion
        let normalizer = VAENormalizer::new(ModelArchitecture::SD15);
        
        // Create test tensor [0, 1]
        let input = Tensor::ones(&[1, 3, 64, 64], DType::F32, &candle_core::Device::Cpu).unwrap();
        let normalized = normalizer.normalize_for_vae(&input).unwrap();
        
        // Should be close to 1.0 after normalization (1.0 -> 2*1 - 1 = 1)
        let max = normalized.max_all().unwrap().to_scalar::<f32>().unwrap();
        assert!((max - 1.0).abs() < 0.01);
    }
}