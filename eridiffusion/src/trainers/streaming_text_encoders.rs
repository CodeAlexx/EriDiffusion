//! Streaming text encoders that process T5 layer-by-layer to save memory

use crate::loaders::WeightLoader;
use crate::models::streaming_t5::StreamingT5Encoder;
use crate::models::{clip, T5EncoderModel};
use flame_core::device::Device;
use flame_core::Result;
use flame_core::{CudaDevice, DType, Error, Shape, Tensor};
use log::info;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

pub struct StreamingTextEncoders {
    pub clip_l: Option<clip::ClipTextTransformer>,
    pub clip_g: Option<clip::ClipTextTransformer>,
    pub t5_streaming: Option<StreamingT5Encoder>,
    tokenizer_clip: Option<Tokenizer>,
    tokenizer_t5: Option<Tokenizer>,
    pub device: Device,
}

impl StreamingTextEncoders {
    pub fn new(device: Device) -> Self {
        Self {
            clip_l: None,
            clip_g: None,
            t5_streaming: None,
            tokenizer_clip: None,
            tokenizer_t5: None,
            device,
        }
    }

    pub fn load_clip_l(&mut self, model_path: &str) -> flame_core::Result<()> {
        println!("Loading CLIP-L from: {}", model_path);
        // CLIP-L is small (0.23GB), regular loading is fine
        let wl = crate::loaders::WeightLoader::from_safetensors_with_dtype(
            model_path,
            self.device.clone(),
            DType::F16,
        )?;
        let config = crate::models::text_encoder::CLIPConfig::clip_l();
        let weights = wl.weights;
        self.clip_l = Some(clip::ClipTextTransformer::new(config, &self.device, weights)?);
        println!("✅ CLIP-L loaded successfully with FP16 precision");
        Ok(())
    }

    pub fn load_clip_g(&mut self, model_path: &str) -> flame_core::Result<()> {
        println!("Loading CLIP-G from: {}", model_path);
        // CLIP-G is ~1.3GB, regular loading should be fine
        let wl = crate::loaders::WeightLoader::from_safetensors_with_dtype(
            model_path,
            self.device.clone(),
            DType::F16,
        )?;
        let config = crate::models::text_encoder::CLIPConfig::clip_g();
        let weights = wl.weights;
        self.clip_g = Some(clip::ClipTextTransformer::new(config, &self.device, weights)?);
        println!("✅ CLIP-G loaded successfully with FP16 precision");
        Ok(())
    }

    pub fn load_t5_streaming(&mut self, model_path: &str) -> flame_core::Result<()> {
        println!("Loading T5-XXL with GPU layer streaming (cuDNN optimized)...");
        println!("  Using GPU-based streaming T5 (loads layers on-demand)");
        println!("  ✨ cuDNN kernels enabled for maximum performance");
        self.t5_streaming = Some(StreamingT5Encoder::new(model_path, self.device.clone())?);
        println!("✅ T5-XXL configured for GPU-only memory-efficient processing");
        Ok(())
    }

    pub fn load_tokenizers(
        &mut self,
        clip_tokenizer_path: &str,
        t5_tokenizer_path: &str,
    ) -> flame_core::Result<()> {
        // Load CLIP tokenizer
        self.tokenizer_clip = Some(Tokenizer::from_file(clip_tokenizer_path).map_err(|e| {
            flame_core::Error::InvalidOperation(format!(
                "Failed to load CLIP tokenizer: {}",
                e
            ))
        })?);

        // Load T5 tokenizer
        self.tokenizer_t5 = Some(Tokenizer::from_file(t5_tokenizer_path).map_err(|e| {
            flame_core::Error::InvalidOperation(format!("Failed to load T5 tokenizer: {}", e))
        })?);

        println!("Tokenizers loaded successfully");
        Ok(())
    }

    pub fn encode_flux_prompt(
        &self,
        prompt: &str,
        neg_prompt: Option<&str>,
    ) -> flame_core::Result<(Tensor, Tensor, Option<Tensor>)> {
        let max_length = 77;
        let t5_max_length = 256;

        // Tokenize with CLIP
        let tokenizer_clip = self
            .tokenizer_clip
            .as_ref()
            .ok_or_else(|| Error::InvalidOperation("CLIP tokenizer not loaded".into()))?;

        let mut encoding = tokenizer_clip
            .encode(prompt, false)
            .map_err(|e| Error::InvalidOperation(format!("Tokenization failed: {}", e)))?;

        // Pad or truncate to max_length
        let ids = if encoding.get_ids().len() > max_length {
            encoding.get_ids()[..max_length].to_vec()
        } else {
            let mut ids = encoding.get_ids().to_vec();
            ids.resize(max_length, 0);
            ids
        };

        // Convert to tensor
        let input_ids = Tensor::from_vec(
            ids.iter().map(|&id| id as f32).collect::<Vec<_>>(),
            Shape::from_dims(&[1, max_length]),
            self.device.cuda_device_arc(),
        )?;

        // Encode with CLIP-L
        let clip_l = self
            .clip_l
            .as_ref()
            .ok_or_else(|| Error::InvalidOperation("CLIP-L not loaded".into()))?;

        let text_embeds_clip = clip_l.forward(&input_ids, None)?;
        let pooled_embeds = text_embeds_clip.pooled_output.clone(); // Extract pooled output

        // Encode with T5 if available
        let text_embeds_t5 = if self.t5_streaming.is_some() {
            let tokenizer_t5 = self.tokenizer_t5.as_ref().ok_or_else(|| {
                Error::InvalidOperation("T5 tokenizer not loaded".into())
            })?;

            let mut t5_encoding = tokenizer_t5.encode(prompt, false).map_err(|e| {
                Error::InvalidOperation(format!("T5 tokenization failed: {}", e))
            })?;

            // Pad or truncate
            let t5_ids = if t5_encoding.get_ids().len() > t5_max_length {
                t5_encoding.get_ids()[..t5_max_length].to_vec()
            } else {
                let mut ids = t5_encoding.get_ids().to_vec();
                ids.resize(t5_max_length, 0);
                ids
            };

            let t5_input = Tensor::from_vec(
                t5_ids.iter().map(|&id| id as f32).collect::<Vec<_>>(),
                Shape::from_dims(&[1, t5_max_length]),
                self.device.cuda_device_arc(),
            )?;

            // Use GPU streaming T5 encoder
            let t5_embeds = if let Some(t5_streaming) = &self.t5_streaming {
                // Process on GPU with layer streaming and cuDNN optimization
                t5_streaming.encode_batch(&t5_input)?
            } else {
                return Err(Error::InvalidOperation(
                    "T5 streaming encoder not loaded".to_string(),
                ));
            };

            Some(t5_embeds)
        } else {
            None
        };

        Ok((text_embeds_clip.last_hidden_state, pooled_embeds, text_embeds_t5))
    }

    /// Encode prompts one at a time to minimize memory usage
    pub fn encode_prompt_batch_streaming(
        &self,
        prompts: &[String],
        cache: &mut crate::trainers::text_embedding_cache::PersistentEmbeddingCache,
    ) -> flame_core::Result<()> {
        println!("Encoding {} prompts with streaming T5...", prompts.len());

        for (idx, prompt) in prompts.iter().enumerate() {
            // Check if already cached
            if let Some(_) = cache.get(prompt)? {
                continue;
            }

            // Encode single prompt
            let (clip_embeds, pooled_embeds, t5_embeds) = self.encode_flux_prompt(prompt, None)?;

            // Save to cache immediately
            if let Some(ref t5) = t5_embeds {
                cache.save(prompt, &clip_embeds, Some(t5))?;
            } else {
                cache.save(prompt, &clip_embeds, None)?;
            }

            // Progress update
            if idx % 10 == 0 {
                println!("  Encoded {}/{} prompts", idx + 1, prompts.len());
            }

            // Force memory cleanup periodically
            if idx % 50 == 49 {
                self.device.synchronize()?;
            }
        }

        println!("✅ All prompts encoded and cached");
        Ok(())
    }

    /// Get memory usage estimate
    pub fn memory_usage_mb(&self) -> f32 {
        let mut total_mb = 0.0;

        // CLIP-L: ~230MB
        if self.clip_l.is_some() {
            total_mb += 230.0;
        }

        // CLIP-G: ~1300MB
        if self.clip_g.is_some() {
            total_mb += 1300.0;
        }

        // T5 streaming: only embeddings + current layer (~400MB max)
        if self.t5_streaming.is_some() {
            total_mb += 400.0;
        }

        total_mb
    }
}
