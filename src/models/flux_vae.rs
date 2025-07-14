// flux_vae.rs
// Flux VAE wrapper using candle-transformers implementation

use candle_core::{Device, DType, Tensor};
use candle_nn::{VarBuilder, Module};
use candle_transformers::models::flux;
use std::path::Path;

/// Wrapper around candle-transformers Flux AutoEncoder
pub struct AutoencoderKL {
    inner: flux::autoencoder::AutoEncoder,
    scale_factor: f64,
    shift_factor: f64,
}

impl AutoencoderKL {
    /// Create a new AutoencoderKL with Flux configuration
    pub fn new(_config: &AutoencoderKLConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        // Use Flux dev config (same as Schnell for VAE)
        let config = flux::autoencoder::Config::dev();
        let inner = flux::autoencoder::AutoEncoder::new(&config, vb)?;
        
        Ok(Self {
            inner,
            scale_factor: 0.13025,  // Flux uses different scaling than SD
            shift_factor: 0.0,       // No shift for standard Flux
        })
    }
    
    /// Encode image to latents
    pub fn encode(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        // x should be in range [0, 1], convert to [-1, 1]
        let x = ((x * 2.0)? - 1.0)?;
        
        // Encode using the inner autoencoder
        let latents = self.inner.encode(&x)?;
        
        // Apply scaling
        latents.affine(self.scale_factor, self.shift_factor)
    }
    
    /// Decode latents to image
    pub fn decode(&self, z: &Tensor) -> candle_core::Result<Tensor> {
        // Remove scaling
        let z = ((z - self.shift_factor)? / self.scale_factor)?;
        
        // Decode
        let decoded = self.inner.decode(&z)?;
        
        // Convert from [-1, 1] to [0, 1]
        (decoded + 1.0)?.affine(0.5, 0.0)
    }
}

impl Module for AutoencoderKL {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        // For Module trait, forward does encode->decode round trip
        let latents = self.encode(x)?;
        self.decode(&latents)
    }
}

/// Configuration for backward compatibility
#[derive(Debug, Clone)]
pub struct AutoencoderKLConfig {
    pub block_out_channels: Vec<usize>,
    pub layers_per_block: usize,
    pub latent_channels: usize,
    pub norm_num_groups: usize,
    pub scaling_factor: f64,
}

impl Default for AutoencoderKLConfig {
    fn default() -> Self {
        Self {
            block_out_channels: vec![128, 256, 512, 512],
            layers_per_block: 2,
            latent_channels: 16,
            norm_num_groups: 32,
            scaling_factor: 0.13025,  // Flux scaling
        }
    }
}

/// Load Flux VAE from ae.safetensors
pub fn load_flux_vae(
    vae_path: &Path,
    device: &Device,
) -> candle_core::Result<AutoencoderKL> {
    println!("Loading Flux VAE from: {:?}", vae_path);
    
    let vb = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(&[vae_path], DType::F32, device)?
    };
    
    let config = AutoencoderKLConfig::default();
    AutoencoderKL::new(&config, vb)
}

// Example usage
pub fn decode_latents(
    vae: &AutoencoderKL,
    latents: &Tensor,
) -> candle_core::Result<Tensor> {
    // Latents should be [batch, 16, height/8, width/8]
    let images = vae.decode(latents)?;
    
    // Output is [batch, 3, height, width] in range [0, 1]
    Ok(images)
}