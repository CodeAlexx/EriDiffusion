//! Efficient batch preparation for different model architectures

use eridiffusion_core::{Result, Error, Device, ModelArchitecture};
use eridiffusion_models::{VAE, TextEncoder};
use candle_core::{Tensor, DType};
use std::sync::Arc;
use std::collections::HashMap;
use tracing::{info, debug};

use crate::{
    DataLoaderBatch, LatentBatch, PreprocessedItem,
    VAEPreprocessor, CaptionPreprocessor, ResolutionManager,
};

/// Batch processor for efficient data preparation
pub struct BatchProcessor {
    architecture: ModelArchitecture,
    vae_preprocessor: Option<Arc<VAEPreprocessor>>,
    caption_preprocessor: Arc<CaptionPreprocessor>,
    resolution_manager: Arc<ResolutionManager>,
    text_encoders: Option<TextEncoders>,
    config: BatchConfig,
    device: Device,
}

/// Text encoders container
struct TextEncoders {
    primary: Arc<dyn TextEncoder>,
    secondary: Option<Arc<dyn TextEncoder>>,
    tertiary: Option<Arc<dyn TextEncoder>>,
}

impl BatchProcessor {
    /// Create new batch processor
    pub fn new(
        architecture: ModelArchitecture,
        vae: Option<Arc<dyn VAE>>,
        text_encoders: Option<(Arc<dyn TextEncoder>, Option<Arc<dyn TextEncoder>>, Option<Arc<dyn TextEncoder>>)>,
    ) -> Result<Self> {
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        
        let vae_preprocessor = vae.map(|v| {
            Arc::new(VAEPreprocessor::new(v, architecture.clone()).unwrap())
        });
        
        let caption_preprocessor = Arc::new(CaptionPreprocessor::new(architecture.clone())?);
        let resolution_manager = Arc::new(ResolutionManager::new(architecture.clone())?);
        
        let text_encoders = text_encoders.map(|(primary, secondary, tertiary)| {
            TextEncoders { primary, secondary, tertiary }
        });
        
        let config = BatchConfig::for_architecture(&architecture);
        
        Ok(Self {
            architecture,
            vae_preprocessor,
            caption_preprocessor,
            resolution_manager,
            text_encoders,
            config,
            device,
        })
    }
    
    /// Process batch of items into model-ready format
    pub async fn process_batch(
        &self,
        items: Vec<PreprocessedItem>,
        bucket_id: usize,
    ) -> Result<ProcessedBatch> {
        let batch_size = items.len();
        debug!("Processing batch of {} items for bucket {}", batch_size, bucket_id);
        
        // Get bucket info
        let bucket = self.resolution_manager.get_bucket(bucket_id)?;
        
        // Process images
        let images = self.process_images(&items, bucket).await?;
        
        // Process captions
        let captions = self.process_captions(&items)?;
        
        // Encode to latents if VAE available
        let latents = if let Some(vae_prep) = &self.vae_preprocessor {
            Some(self.encode_images_to_latents(&images, vae_prep).await?)
        } else {
            None
        };
        
        // Encode text if encoders available
        let text_embeddings = if let Some(encoders) = &self.text_encoders {
            Some(self.encode_text(&captions.texts, encoders).await?)
        } else {
            None
        };
        
        // Create model-specific inputs
        let model_inputs = self.create_model_inputs(
            &images,
            &latents,
            &captions,
            &text_embeddings,
            &items,
        )?;
        
        Ok(ProcessedBatch {
            images,
            latents,
            captions,
            text_embeddings,
            model_inputs,
            bucket_id,
            metadata: self.collect_metadata(&items),
        })
    }
    
    /// Process images for batch
    async fn process_images(
        &self,
        items: &[PreprocessedItem],
        bucket: &ResolutionBucket,
    ) -> Result<Tensor> {
        let mut processed = Vec::new();
        
        for item in items {
            // Resize to bucket dimensions
            let resized = self.resolution_manager.resize_to_bucket(&item.image, bucket.id)?;
            processed.push(resized.tensor);
        }
        
        // Stack into batch
        Tensor::stack(&processed, 0).map_err(Error::from)
    }
    
    /// Process captions
    fn process_captions(&self, items: &[PreprocessedItem]) -> Result<ProcessedCaptions> {
        let mut texts = Vec::new();
        let mut token_ids = Vec::new();
        let mut attention_masks = Vec::new();
        
        for item in items {
            let processed = self.caption_preprocessor.preprocess(&item.caption)?;
            texts.push(processed.text.clone());
            
            // Simple tokenization (would use actual tokenizer)
            let tokens = self.simple_tokenize(&processed.text)?;
            token_ids.push(tokens.ids);
            attention_masks.push(tokens.attention_mask);
        }
        
        Ok(ProcessedCaptions {
            texts,
            token_ids,
            attention_masks,
        })
    }
    
    /// Encode images to latents
    async fn encode_images_to_latents(
        &self,
        images: &Tensor,
        vae_preprocessor: &VAEPreprocessor,
    ) -> Result<Tensor> {
        debug!("Encoding batch to latents");
        vae_preprocessor.encode_batch(images)
    }
    
    /// Encode text with all encoders
    async fn encode_text(
        &self,
        texts: &[String],
        encoders: &TextEncoders,
    ) -> Result<TextEmbeddings> {
        debug!("Encoding {} captions", texts.len());
        
        // Primary encoder (CLIP-L for most models)
        let (primary_embeds, primary_pooled) = encoders.primary.encode(texts)?;
        
        // Secondary encoder (CLIP-G for SDXL/SD3)
        let (secondary_embeds, secondary_pooled) = if let Some(encoder) = &encoders.secondary {
            let (embeds, pooled) = encoder.encode(texts)?;
            (Some(embeds), pooled)
        } else {
            (None, None)
        };
        
        // Tertiary encoder (T5 for SD3/Flux)
        let (tertiary_embeds, _) = if let Some(encoder) = &encoders.tertiary {
            let (embeds, pooled) = encoder.encode(texts)?;
            (Some(embeds), pooled)
        } else {
            (None, None)
        };
        
        Ok(TextEmbeddings {
            primary_embeds,
            primary_pooled,
            secondary_embeds,
            secondary_pooled,
            tertiary_embeds,
        })
    }
    
    /// Create model-specific inputs
    fn create_model_inputs(
        &self,
        images: &Tensor,
        latents: &Option<Tensor>,
        captions: &ProcessedCaptions,
        text_embeddings: &Option<TextEmbeddings>,
        items: &[PreprocessedItem],
    ) -> Result<ModelInputs> {
        match self.architecture {
            ModelArchitecture::SD15 | ModelArchitecture::SD15 => {
                self.create_sd_inputs(latents, text_embeddings)
            }
            ModelArchitecture::SDXL => {
                self.create_sdxl_inputs(latents, text_embeddings, items)
            }
            ModelArchitecture::SD3 | ModelArchitecture::SD35 => {
                self.create_sd3_inputs(latents, text_embeddings, items)
            }
            ModelArchitecture::Flux => {
                self.create_flux_inputs(latents, text_embeddings, items)
            }
            _ => {
                self.create_default_inputs(images, latents, text_embeddings)
            }
        }
    }
    
    /// Create SD1.5/SD2 inputs
    fn create_sd_inputs(
        &self,
        latents: &Option<Tensor>,
        text_embeddings: &Option<TextEmbeddings>,
    ) -> Result<ModelInputs> {
        let mut inputs = HashMap::new();
        
        if let Some(latents) = latents {
            inputs.insert("latents".to_string(), latents.clone());
        }
        
        if let Some(embeds) = text_embeddings {
            inputs.insert("encoder_hidden_states".to_string(), embeds.primary_embeds.clone());
        }
        
        Ok(ModelInputs::SD(inputs))
    }
    
    /// Create SDXL inputs
    fn create_sdxl_inputs(
        &self,
        latents: &Option<Tensor>,
        text_embeddings: &Option<TextEmbeddings>,
        items: &[PreprocessedItem],
    ) -> Result<ModelInputs> {
        let mut inputs = HashMap::new();
        
        if let Some(latents) = latents {
            inputs.insert("latents".to_string(), latents.clone());
        }
        
        if let Some(embeds) = text_embeddings {
            // SDXL uses concatenated CLIP embeddings
            if let Some(secondary) = &embeds.secondary_embeds {
                let concat_embeds = Tensor::cat(&[&embeds.primary_embeds, secondary], 2)?;
                inputs.insert("encoder_hidden_states".to_string(), concat_embeds);
            } else {
                inputs.insert("encoder_hidden_states".to_string(), embeds.primary_embeds.clone());
            }
            
            // Add pooled embeddings
            if let (Some(p1), Some(p2)) = (&embeds.primary_pooled, &embeds.secondary_pooled) {
                let pooled = Tensor::cat(&[p1, p2], 1)?;
                inputs.insert("text_embeds".to_string(), pooled);
            }
        }
        
        // Create time IDs
        let time_ids = self.create_time_ids(items)?;
        inputs.insert("time_ids".to_string(), time_ids);
        
        Ok(ModelInputs::SDXL(inputs))
    }
    
    /// Create SD3/SD3.5 inputs
    fn create_sd3_inputs(
        &self,
        latents: &Option<Tensor>,
        text_embeddings: &Option<TextEmbeddings>,
        items: &[PreprocessedItem],
    ) -> Result<ModelInputs> {
        let mut inputs = HashMap::new();
        
        if let Some(latents) = latents {
            inputs.insert("latents".to_string(), latents.clone());
        }
        
        if let Some(embeds) = text_embeddings {
            // SD3 uses all three text encoders
            let mut embeds_to_concat = vec![embeds.primary_embeds.clone()];
            
            if let Some(secondary) = &embeds.secondary_embeds {
                embeds_to_concat.push(secondary.clone());
            }
            
            if let Some(tertiary) = &embeds.tertiary_embeds {
                embeds_to_concat.push(tertiary.clone());
            }
            
            let concat_embeds = Tensor::cat(&embeds_to_concat, 2)?;
            inputs.insert("encoder_hidden_states".to_string(), concat_embeds);
            
            // Pooled projections
            if let (Some(p1), Some(p2)) = (&embeds.primary_pooled, &embeds.secondary_pooled) {
                let pooled = Tensor::cat(&[p1, p2], 1)?;
                inputs.insert("pooled_projections".to_string(), pooled);
            }
        }
        
        Ok(ModelInputs::SD3(inputs))
    }
    
    /// Create Flux inputs
    fn create_flux_inputs(
        &self,
        latents: &Option<Tensor>,
        text_embeddings: &Option<TextEmbeddings>,
        items: &[PreprocessedItem],
    ) -> Result<ModelInputs> {
        let mut inputs = HashMap::new();
        
        if let Some(latents) = latents {
            // Flux may use packed latents
            let packed = self.pack_latents_for_flux(latents)?;
            inputs.insert("latents".to_string(), packed);
        }
        
        if let Some(embeds) = text_embeddings {
            // Flux uses CLIP and T5
            inputs.insert("clip_embeds".to_string(), embeds.primary_embeds.clone());
            
            if let Some(t5) = &embeds.tertiary_embeds {
                inputs.insert("t5_embeds".to_string(), t5.clone());
            }
        }
        
        // Flux-specific inputs
        let img_ids = self.create_flux_image_ids(items)?;
        inputs.insert("img_ids".to_string(), img_ids);
        
        Ok(ModelInputs::Flux(inputs))
    }
    
    /// Create default inputs
    fn create_default_inputs(
        &self,
        images: &Tensor,
        latents: &Option<Tensor>,
        text_embeddings: &Option<TextEmbeddings>,
    ) -> Result<ModelInputs> {
        let mut inputs = HashMap::new();
        
        if let Some(latents) = latents {
            inputs.insert("latents".to_string(), latents.clone());
        } else {
            inputs.insert("images".to_string(), images.clone());
        }
        
        if let Some(embeds) = text_embeddings {
            inputs.insert("text_embeds".to_string(), embeds.primary_embeds.clone());
        }
        
        Ok(ModelInputs::Generic(inputs))
    }
    
    /// Create time IDs for SDXL
    fn create_time_ids(&self, items: &[PreprocessedItem]) -> Result<Tensor> {
        let mut time_ids = Vec::new();
        
        for item in items {
            // [original_height, original_width, crop_top, crop_left, target_height, target_width]
            let ids = vec![
                item.original_size.1 as f32,
                item.original_size.0 as f32,
                item.crop_coords.1 as f32,
                item.crop_coords.0 as f32,
                1024.0, // target height
                1024.0, // target width
            ];
            time_ids.push(ids);
        }
        
        // Stack into tensor [batch, 6]
        let flat: Vec<f32> = time_ids.into_iter().flatten().collect();
        let candle_device = match &self.device {
            Device::Cpu => candle_core::Device::Cpu,
            Device::Cuda(id) => candle_core::Device::new_cuda(*id)?,
        };
        Ok(Tensor::from_vec(flat, &[items.len(), 6], &candle_device)?)
    }
    
    /// Pack latents for Flux
    fn pack_latents_for_flux(&self, latents: &Tensor) -> Result<Tensor> {
        // Flux uses a specific packing format
        // This is a simplified version
        Ok(latents.clone())
    }
    
    /// Create image IDs for Flux
    fn create_flux_image_ids(&self, items: &[PreprocessedItem]) -> Result<Tensor> {
        // Flux uses positional encodings
        let batch_size = items.len();
        let latent_size = 128; // example
        
        let mut ids = Vec::new();
        for b in 0..batch_size {
            for i in 0..latent_size {
                ids.push(vec![b as f32, i as f32]);
            }
        }
        
        let flat: Vec<f32> = ids.into_iter().flatten().collect();
        let candle_device = match &self.device {
            Device::Cpu => candle_core::Device::Cpu,
            Device::Cuda(id) => candle_core::Device::new_cuda(*id)?,
        };
        Ok(Tensor::from_vec(flat, &[batch_size * latent_size, 2], &candle_device)?)
    }
    
    /// Simple tokenizer (placeholder)
    fn simple_tokenize(&self, text: &str) -> Result<TokenizedText> {
        let words: Vec<&str> = text.split_whitespace().collect();
        let max_len = self.config.max_sequence_length;
        
        // Simple word to ID mapping
        let mut ids = vec![101]; // [CLS] token
        for (i, _word) in words.iter().enumerate().take(max_len - 2) {
            ids.push((i + 1000) as i64); // Fake token IDs
        }
        ids.push(102); // [SEP] token
        
        // Pad to max length
        while ids.len() < max_len {
            ids.push(0);
        }
        
        let attention_mask = ids.iter()
            .map(|&id| if id == 0 { 0 } else { 1 })
            .collect();
        
        Ok(TokenizedText {
            ids,
            attention_mask,
        })
    }
    
    /// Collect metadata from items
    fn collect_metadata(&self, items: &[PreprocessedItem]) -> BatchMetadata {
        let mut metadata = BatchMetadata::default();
        
        for item in items {
            metadata.original_sizes.push(item.original_size);
            metadata.crop_coords.push(item.crop_coords);
            
            // Merge item metadata
            for (key, value) in &item.metadata {
                metadata.custom.entry(key.clone())
                    .or_insert_with(Vec::new)
                    .push(value.clone());
            }
        }
        
        metadata
    }
}

/// Batch configuration
#[derive(Debug, Clone)]
pub struct BatchConfig {
    pub max_sequence_length: usize,
    pub tokenizer_type: TokenizerType,
    pub use_attention_mask: bool,
    pub pad_token_id: i64,
}

impl BatchConfig {
    pub fn for_architecture(arch: &ModelArchitecture) -> Self {
        match arch {
            ModelArchitecture::SD15 | ModelArchitecture::SD15 => Self {
                max_sequence_length: 77,
                tokenizer_type: TokenizerType::CLIPTokenizer,
                use_attention_mask: true,
                pad_token_id: 49407,
            },
            ModelArchitecture::SDXL => Self {
                max_sequence_length: 77,
                tokenizer_type: TokenizerType::CLIPTokenizer,
                use_attention_mask: true,
                pad_token_id: 49407,
            },
            ModelArchitecture::SD3 | ModelArchitecture::SD35 => Self {
                max_sequence_length: 256,
                tokenizer_type: TokenizerType::T5Tokenizer,
                use_attention_mask: true,
                pad_token_id: 0,
            },
            _ => Self {
                max_sequence_length: 77,
                tokenizer_type: TokenizerType::CLIPTokenizer,
                use_attention_mask: true,
                pad_token_id: 0,
            },
        }
    }
}

/// Tokenizer type
#[derive(Debug, Clone, Copy)]
pub enum TokenizerType {
    CLIPTokenizer,
    T5Tokenizer,
    BertTokenizer,
}

/// Resolution bucket
use crate::resolution_manager::ResolutionBucket;

/// Processed batch ready for model
#[derive(Debug, Clone)]
pub struct ProcessedBatch {
    pub images: Tensor,
    pub latents: Option<Tensor>,
    pub captions: ProcessedCaptions,
    pub text_embeddings: Option<TextEmbeddings>,
    pub model_inputs: ModelInputs,
    pub bucket_id: usize,
    pub metadata: BatchMetadata,
}

/// Processed captions
#[derive(Debug, Clone)]
pub struct ProcessedCaptions {
    pub texts: Vec<String>,
    pub token_ids: Vec<Vec<i64>>,
    pub attention_masks: Vec<Vec<i64>>,
}

/// Text embeddings from all encoders
#[derive(Debug, Clone)]
pub struct TextEmbeddings {
    pub primary_embeds: Tensor,
    pub primary_pooled: Option<Tensor>,
    pub secondary_embeds: Option<Tensor>,
    pub secondary_pooled: Option<Tensor>,
    pub tertiary_embeds: Option<Tensor>,
}

/// Model-specific inputs
#[derive(Debug, Clone)]
pub enum ModelInputs {
    SD(HashMap<String, Tensor>),
    SDXL(HashMap<String, Tensor>),
    SD3(HashMap<String, Tensor>),
    Flux(HashMap<String, Tensor>),
    Generic(HashMap<String, Tensor>),
}

/// Batch metadata
#[derive(Debug, Clone, Default)]
pub struct BatchMetadata {
    pub original_sizes: Vec<(u32, u32)>,
    pub crop_coords: Vec<(u32, u32)>,
    pub custom: HashMap<String, Vec<serde_json::Value>>,
}

/// Tokenized text
#[derive(Debug, Clone)]
struct TokenizedText {
    pub ids: Vec<i64>,
    pub attention_mask: Vec<i64>,
}