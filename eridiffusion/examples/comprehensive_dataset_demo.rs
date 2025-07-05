//! Comprehensive demonstration of dataset functionality for all models

use eridiffusion_core::{Result, Device, ModelArchitecture};
use eridiffusion_data::*;
use eridiffusion_models::{VAE, TextEncoder};
use std::path::PathBuf;
use std::sync::Arc;
use tracing::{info, debug};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    info!("🚀 AI-Toolkit Comprehensive Dataset Demo");
    info!("=======================================\n");
    
    // Demonstrate for each architecture
    demo_sd15_dataset().await?;
    demo_sdxl_dataset().await?;
    demo_sd3_dataset().await?;
    demo_flux_dataset().await?;
    
    info!("\n✅ All dataset demonstrations completed!");
    Ok(())
}

/// Demonstrate SD1.5 dataset functionality
async fn demo_sd15_dataset() -> Result<()> {
    info!("\n📊 SD1.5 Dataset Demo");
    info!("--------------------");
    
    let architecture = ModelArchitecture::SD15;
    let dataset_path = PathBuf::from("/home/alex/eridiffusion/datasets/40_woman");
    
    // Create dataset manager (without VAE for demo)
    let mut manager = DatasetManager::new(
        architecture.clone(),
        dataset_path,
        None, // No VAE for this demo
    )?;
    
    // Prepare dataset
    manager.prepare().await?;
    
    // Create resolution manager
    let resolution_manager = ResolutionManager::new(architecture.clone())?;
    info!("SD1.5 Resolution buckets: {:?}", resolution_manager.get_active_buckets());
    
    // Create caption preprocessor
    let caption_preprocessor = CaptionPreprocessor::new(architecture.clone())?;
    let sample_caption = "a photo of ohwx woman, portrait, masterpiece, best quality";
    let processed = caption_preprocessor.preprocess(sample_caption)?;
    info!("Processed caption: '{}' -> {} tokens", processed.text, processed.token_count);
    
    // Demonstrate bucket sampler
    let sampler = manager.create_bucket_sampler(4, true)?;
    info!("Created bucket sampler with batch size 4");
    
    Ok(())
}

/// Demonstrate SDXL dataset functionality
async fn demo_sdxl_dataset() -> Result<()> {
    info!("\n📊 SDXL Dataset Demo");
    info!("-------------------");
    
    let architecture = ModelArchitecture::SDXL;
    
    // Show SDXL-specific features
    let resolution_config = ResolutionConfig::for_architecture(&architecture);
    info!("SDXL Resolution config:");
    info!("  Base: {:?}", resolution_config.base_resolutions);
    info!("  Range: {} - {}", resolution_config.min_resolution, resolution_config.max_resolution);
    info!("  Aspect ratios: {:?}", resolution_config.aspect_ratios);
    
    // Caption handling for SDXL
    let caption_preprocessor = CaptionPreprocessor::new(architecture)?;
    let long_caption = "a professional photograph of ohwx woman wearing an elegant red dress, \
                        standing in a modern art gallery with soft natural lighting, \
                        high quality, detailed, award winning photography";
    let processed = caption_preprocessor.preprocess(long_caption)?;
    info!("SDXL supports dual text encoders");
    info!("Caption chunks: {}", processed.chunks.len());
    
    // VAE configuration
    let vae_config = VAEConfig::for_architecture(&architecture);
    info!("SDXL VAE config:");
    info!("  Latent channels: {}", vae_config.latent_channels);
    info!("  Downsampling: {}x", vae_config.downsampling_factor);
    info!("  Scale factor: {}", vae_config.scale_factor);
    info!("  Tiling enabled: {}", vae_config.use_tiling);
    
    Ok(())
}

/// Demonstrate SD3/SD3.5 dataset functionality
async fn demo_sd3_dataset() -> Result<()> {
    info!("\n📊 SD3/SD3.5 Dataset Demo");
    info!("------------------------");
    
    let architecture = ModelArchitecture::SD35;
    
    // SD3 specific features
    let resolution_config = ResolutionConfig::for_architecture(&architecture);
    info!("SD3.5 supports multiple resolutions:");
    for res in &resolution_config.base_resolutions {
        info!("  - {}x{}", res, res);
    }
    
    // SD3 uses 16-channel VAE
    let vae_config = VAEConfig::for_architecture(&architecture);
    info!("SD3.5 VAE has {} latent channels (vs 4 for SD1.5/SDXL)", vae_config.latent_channels);
    
    // Caption preprocessing for T5
    let caption_config = CaptionConfig::for_architecture(&architecture);
    info!("SD3.5 caption config:");
    info!("  Max tokens: {} (T5 support)", caption_config.max_tokens);
    info!("  Long caption support: {}", caption_config.supports_long_captions);
    
    // Demonstrate caption augmentation
    let augmenter = CaptionAugmenter::new(0.1, 0.2, 0.1);
    let original = "ohwx woman, portrait, smile, outdoor";
    let mut rng = rand::thread_rng();
    let augmented = augmenter.augment(original, &mut rng);
    info!("Caption augmentation: '{}' -> '{}'", original, augmented);
    
    Ok(())
}

/// Demonstrate Flux dataset functionality
async fn demo_flux_dataset() -> Result<()> {
    info!("\n📊 Flux Dataset Demo");
    info!("-------------------");
    
    let architecture = ModelArchitecture::Flux("dev".to_string());
    
    // Flux supports extreme resolutions
    let resolution_config = ResolutionConfig::for_architecture(&architecture);
    info!("Flux resolution range: {} - {}", 
        resolution_config.min_resolution, 
        resolution_config.max_resolution
    );
    info!("Flux supports {} different aspect ratios", resolution_config.aspect_ratios.len());
    
    // Show some extreme aspect ratios
    let buckets = resolution_config.get_buckets();
    let extreme_buckets: Vec<_> = buckets.iter()
        .filter(|(w, h)| {
            let ar = *w as f32 / *h as f32;
            ar > 3.0 || ar < 0.33
        })
        .collect();
    
    info!("Extreme aspect ratio buckets:");
    for (w, h) in extreme_buckets {
        info!("  - {}x{} (AR: {:.2})", w, h, *w as f32 / *h as f32);
    }
    
    // Flux caption handling
    let caption_config = CaptionConfig::for_architecture(&architecture);
    info!("Flux supports up to {} tokens (CLIP + T5)", caption_config.max_tokens);
    
    Ok(())
}

/// Create a complete data pipeline example
pub async fn demo_complete_pipeline() -> Result<()> {
    info!("\n🔄 Complete Data Pipeline Demo");
    info!("=============================");
    
    let architecture = ModelArchitecture::SD35;
    let dataset_path = PathBuf::from("/home/alex/eridiffusion/datasets/40_woman");
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    
    // Step 1: Create dataset manager with mock VAE
    info!("Step 1: Creating dataset manager...");
    let vae = create_mock_vae(architecture.clone(), &device)?;
    let mut manager = DatasetManager::new(
        architecture.clone(),
        dataset_path,
        Some(vae.clone()),
    )?;
    
    // Step 2: Prepare dataset (analyzes and caches)
    info!("Step 2: Preparing dataset...");
    manager.prepare().await?;
    
    // Step 3: Create batch processor with mock text encoders
    info!("Step 3: Creating batch processor...");
    let text_encoders = create_mock_text_encoders(&device)?;
    let batch_processor = BatchProcessor::new(
        architecture.clone(),
        Some(vae),
        Some(text_encoders),
    )?;
    
    // Step 4: Create bucket sampler
    info!("Step 4: Creating bucket sampler...");
    let sampler = manager.create_bucket_sampler(4, true)?;
    
    // Step 5: Process a batch
    info!("Step 5: Processing a batch...");
    if let Some(batch_indices) = sampler.next_batch() {
        let mut items = Vec::new();
        for idx in batch_indices {
            let item = manager.get_preprocessed_item(idx).await?;
            items.push(item);
        }
        
        let bucket_id = 0; // Use first bucket
        let processed_batch = batch_processor.process_batch(items, bucket_id).await?;
        
        info!("Processed batch:");
        info!("  Images shape: {:?}", processed_batch.images.shape());
        if let Some(latents) = &processed_batch.latents {
            info!("  Latents shape: {:?}", latents.shape());
        }
        if let Some(text_embeds) = &processed_batch.text_embeddings {
            info!("  Text embeddings shape: {:?}", text_embeds.primary_embeds.shape());
        }
        
        match &processed_batch.model_inputs {
            ModelInputs::SD3(inputs) => {
                info!("  SD3 model inputs: {} tensors", inputs.len());
                for (key, tensor) in inputs {
                    info!("    - {}: {:?}", key, tensor.shape());
                }
            }
            _ => {}
        }
    }
    
    info!("\n✅ Complete pipeline demo finished!");
    Ok(())
}

/// Create mock VAE for testing
fn create_mock_vae(architecture: ModelArchitecture, device: &Device) -> Result<Arc<dyn VAE>> {
    struct MockVAE {
        architecture: ModelArchitecture,
        device: Device,
    }
    
    impl VAE for MockVAE {
        fn encode(&self, images: &candle_core::Tensor) -> Result<candle_core::Tensor> {
            let (b, c, h, w) = match images.dims() {
                [b, c, h, w] => (*b, *c, *h, *w),
                _ => return Err(eridiffusion_core::Error::InvalidShape("Expected BCHW".into())),
            };
            
            let latent_channels = match self.architecture {
                ModelArchitecture::SD3 | ModelArchitecture::SD35 => 16,
                _ => 4,
            };
            
            // Simulate encoding
            candle_core::Tensor::randn(
                0.0f32,
                1.0f32,
                &[b, latent_channels, h / 8, w / 8],
                &self.device,
            ).map_err(|e| eridiffusion_core::Error::TensorError(e.to_string()))
        }
        
        fn decode(&self, latents: &candle_core::Tensor) -> Result<candle_core::Tensor> {
            let (b, c, h, w) = match latents.dims() {
                [b, c, h, w] => (*b, *c, *h, *w),
                _ => return Err(eridiffusion_core::Error::InvalidShape("Expected BCHW".into())),
            };
            
            // Simulate decoding
            candle_core::Tensor::randn(
                0.0f32,
                1.0f32,
                &[b, 3, h * 8, w * 8],
                &self.device,
            ).map_err(|e| eridiffusion_core::Error::TensorError(e.to_string()))
        }
        
        fn device(&self) -> &Device {
            &self.device
        }
        
        fn latent_channels(&self) -> usize {
            match self.architecture {
                ModelArchitecture::SD3 | ModelArchitecture::SD35 => 16,
                _ => 4,
            }
        }
    }
    
    Ok(Arc::new(MockVAE {
        architecture,
        device: device.clone(),
    }))
}

/// Create mock text encoders for testing
fn create_mock_text_encoders(device: &Device) -> Result<(Arc<dyn TextEncoder>, Option<Arc<dyn TextEncoder>>, Option<Arc<dyn TextEncoder>>)> {
    struct MockTextEncoder {
        embed_dim: usize,
        device: Device,
    }
    
    impl TextEncoder for MockTextEncoder {
        fn encode(&self, prompts: &[String]) -> Result<(candle_core::Tensor, Option<candle_core::Tensor>)> {
            let batch_size = prompts.len();
            let seq_len = 77;
            
            let embeds = candle_core::Tensor::randn(
                0.0f32,
                1.0f32,
                &[batch_size, seq_len, self.embed_dim],
                &self.device,
            ).map_err(|e| eridiffusion_core::Error::TensorError(e.to_string()))?;
            
            let pooled = candle_core::Tensor::randn(
                0.0f32,
                1.0f32,
                &[batch_size, self.embed_dim],
                &self.device,
            ).map_err(|e| eridiffusion_core::Error::TensorError(e.to_string()))?;
            
            Ok((embeds, Some(pooled)))
        }
        
        fn device(&self) -> &Device {
            &self.device
        }
    }
    
    let clip_l = Arc::new(MockTextEncoder {
        embed_dim: 768,
        device: device.clone(),
    });
    
    let clip_g = Arc::new(MockTextEncoder {
        embed_dim: 1280,
        device: device.clone(),
    });
    
    let t5 = Arc::new(MockTextEncoder {
        embed_dim: 4096,
        device: device.clone(),
    });
    
    Ok((clip_l, Some(clip_g), Some(t5)))
}