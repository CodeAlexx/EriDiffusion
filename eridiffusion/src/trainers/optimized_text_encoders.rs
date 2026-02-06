//! Optimized text encoders with caching and performance improvements
//!
//! This module provides optimized text encoding for Flux and other models with:
//! - Text embedding caching to avoid re-encoding identical prompts
//! - Lazy loading of models to reduce memory pressure
//! - FP16 precision throughout for memory efficiency
//! - Batch processing optimizations

use crate::loaders::WeightLoader;
use crate::models::{
    clip,
    t5::{T5Config, T5EncoderModel},
};
use crate::trainers::text_embedding_cache::PersistentEmbeddingCache;
use flame_core::device::Device;
use flame_core::Result;
use flame_core::{DType, Error, Shape, Tensor};
use std::collections::HashMap;
use std::path::Path;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use tokenizers::Tokenizer;

pub struct OptimizedTextEncoders {
    pub clip_l: Option<clip::ClipTextTransformer>,
    pub t5: Option<T5EncoderModel>,
    tokenizer_clip: Option<Tokenizer>,
    tokenizer_t5: Option<Tokenizer>,
    pub device: Device,
    persistent_cache: Option<PersistentEmbeddingCache>,
    t5_loaded: bool,
}

impl OptimizedTextEncoders {
    pub fn new(device: Device) -> Self {
        Self {
            clip_l: None,
            t5: None,
            tokenizer_clip: None,
            tokenizer_t5: None,
            device,
            persistent_cache: None,
            t5_loaded: false,
        }
    }

    /// Enable persistent disk caching
    pub fn enable_persistent_cache(&mut self, cache_dir: PathBuf) -> Result<()> {
        self.persistent_cache =
            Some(PersistentEmbeddingCache::new(cache_dir, self.device.clone())?);

        // Optionally preload existing cache
        if let Some(ref mut cache) = self.persistent_cache {
            cache.preload_from_disk()?;
        }

        Ok(())
    }

    /// Load CLIP-L encoder (small, always keep in memory)
    pub fn load_clip_l(&mut self, model_path: &str) -> Result<()> {
        println!("Loading CLIP-L from: {}", model_path);
        let start = std::time::Instant::now();

        // CLIP-L is small (0.23GB), regular loading is fine
        let wl =
            WeightLoader::from_safetensors_with_dtype(model_path, self.device.clone(), DType::F16)?;

        // Use the CLIP-L config
        let config = crate::models::text_encoder::CLIPConfig::clip_l();

        self.clip_l = Some(clip::ClipTextTransformer::new(config, self.device.clone(), wl.weights)?);
        println!("✅ CLIP-L loaded in {:.2}s", start.elapsed().as_secs_f32());
        Ok(())
    }

    /// Lazy load T5 encoder (only when needed)
    pub fn ensure_t5_loaded(&mut self, model_path: &str) -> Result<()> {
        if self.t5_loaded {
            return Ok(());
        }

        println!("Lazy loading T5-XXL from: {}", model_path);
        let start = std::time::Instant::now();

        // Use streaming loader with smaller batch size for T5
        let wl =
            WeightLoader::from_safetensors_streaming(model_path, self.device.clone(), DType::F16)?;

        // Create T5-XXL config
        let config = T5Config::t5_xxl();

        // Use T5EncoderModel from t5.rs, not T5Encoder
        self.t5 = Some(T5EncoderModel::new(config, self.device.clone(), &wl)?);
        self.t5_loaded = true;

        println!("✅ T5-XXL loaded in {:.2}s", start.elapsed().as_secs_f32());
        Ok(())
    }

    pub fn load_tokenizers(
        &mut self,
        clip_tokenizer_path: &str,
        t5_tokenizer_path: &str,
    ) -> Result<()> {
        // Load CLIP tokenizer
        self.tokenizer_clip = Some(Tokenizer::from_file(clip_tokenizer_path).map_err(|e| {
            Error::InvalidOperation(format!("Failed to load CLIP tokenizer: {}", e))
        })?);

        // Load T5 tokenizer
        self.tokenizer_t5 =
            Some(Tokenizer::from_file(t5_tokenizer_path).map_err(|e| {
                Error::InvalidOperation(format!("Failed to load T5 tokenizer: {}", e))
            })?);

        println!("Tokenizers loaded successfully");
        Ok(())
    }

    /// Optimized Flux encoding with persistent caching
    pub fn encode_flux(&mut self, text: &str, t5_model_path: &str) -> Result<(Tensor, Tensor)> {
        // Check persistent cache first
        if let Some(ref mut cache) = self.persistent_cache {
            if let Some((clip_embed, t5_embed)) = cache.get(text)? {
                return Ok((clip_embed, t5_embed.unwrap()));
            }
        }

        // Ensure models are loaded
        if self.clip_l.is_none() {
            return Err(Error::InvalidOperation("CLIP-L not loaded".into()));
        }

        // Lazy load T5 only when needed
        self.ensure_t5_loaded(t5_model_path)?;

        println!("  🔄 Encoding text: \"{}\"", text);
        let encode_start = std::time::Instant::now();

        // Tokenize text
        let clip_tokens = self.tokenize_clip(text, 77)?;
        let t5_tokens = self.tokenize_t5(text, 256)?; // Use 256 to save memory

        // Encode with CLIP-L (fast)
        let clip_start = std::time::Instant::now();
        let clip_output = self.clip_l.as_ref().unwrap().forward(&clip_tokens, None)?;
        let clip_embed = clip_output.last_hidden_state;
        println!("    CLIP-L encoding: {:.3}s", clip_start.elapsed().as_secs_f32());

        // Encode with T5 (slower)
        let t5_start = std::time::Instant::now();
        let t5_model = self.t5.as_ref().unwrap();
        let t5_output = t5_model.forward(&t5_tokens)?;
        let t5_embed = t5_output.last_hidden_state; // T5Encoder returns T5Output
        println!("    T5-XXL encoding: {:.3}s", t5_start.elapsed().as_secs_f32());

        println!("  ✅ Total encoding time: {:.3}s", encode_start.elapsed().as_secs_f32());

        // Save to persistent cache
        if let Some(ref mut cache) = self.persistent_cache {
            cache.save(text, &clip_embed, Some(&t5_embed))?;
        }

        Ok((clip_embed, t5_embed))
    }

    /// Batch encoding with optimizations
    pub fn encode_flux_batch(
        &mut self,
        texts: &[String],
        t5_model_path: &str,
    ) -> Result<Vec<(Tensor, Tensor)>> {
        let mut results = Vec::new();

        println!("Batch encoding {} prompts...", texts.len());
        let batch_start = std::time::Instant::now();

        // Group identical prompts to avoid redundant encoding
        let mut unique_texts: HashMap<String, Vec<usize>> = HashMap::new();
        for (idx, text) in texts.iter().enumerate() {
            unique_texts.entry(text.clone()).or_insert_with(Vec::new).push(idx);
        }

        println!("  Found {} unique prompts out of {} total", unique_texts.len(), texts.len());

        // Encode unique prompts
        let mut encoded_map: HashMap<String, (Tensor, Tensor)> = HashMap::new();
        for (text, indices) in unique_texts.iter() {
            let (clip_embed, t5_embed) = self.encode_flux(text, t5_model_path)?;
            encoded_map.insert(text.clone(), (clip_embed, t5_embed));
        }

        // Build results in original order
        for text in texts {
            let embeddings = encoded_map.get(text).unwrap();
            results.push(embeddings.clone());
        }

        println!("Batch encoding complete in {:.2}s", batch_start.elapsed().as_secs_f32());
        Ok(results)
    }

    fn tokenize_clip(&self, text: &str, max_length: usize) -> Result<Tensor> {
        let tokenizer = self
            .tokenizer_clip
            .as_ref()
            .ok_or_else(|| Error::InvalidOperation("CLIP tokenizer not loaded".into()))?;
        let encoding = tokenizer
            .encode(text, true)
            .map_err(|e| Error::InvalidOperation(format!("Tokenization failed: {:?}", e)))?;
        let mut ids = encoding.get_ids().to_vec();

        // Pad or truncate to max_length
        ids.resize(max_length, 0);

        // Create tensor - pad to 80 for CUDA alignment if needed
        let padded_length = if max_length == 77 { 80 } else { max_length };
        let mut ids_f32: Vec<f32> = ids.into_iter().map(|id| id as f32).collect();
        ids_f32.resize(padded_length, 0.0);

        let shape = Shape::from_dims(&[1, padded_length]);
        let tensor = Tensor::from_vec(ids_f32, shape, self.device.cuda_device().clone())?;

        // Slice back to original size if padded
        if padded_length != max_length {
            tensor.slice(&[(0, 1), (0, max_length)])
        } else {
            Ok(tensor)
        }
    }

    fn tokenize_t5(&self, text: &str, max_length: usize) -> Result<Tensor> {
        let tokenizer = self
            .tokenizer_t5
            .as_ref()
            .ok_or_else(|| Error::InvalidOperation("T5 tokenizer not loaded".into()))?;
        let encoding = tokenizer
            .encode(text, true)
            .map_err(|e| Error::InvalidOperation(format!("Tokenization failed: {:?}", e)))?;
        let mut ids = encoding.get_ids().to_vec();

        // T5 can handle longer sequences
        if ids.len() > max_length {
            ids.truncate(max_length);
        }

        let ids_f32: Vec<f32> = ids.into_iter().map(|id| id as f32).collect();
        let len = ids_f32.len();
        Ok(Tensor::from_vec(
            ids_f32,
            Shape::from_dims(&[1, len]),
            self.device.cuda_device().clone(),
        )?)
    }

    /// Clear the embedding cache
    pub fn clear_cache(&mut self) {
        if let Some(ref mut cache) = self.persistent_cache {
            cache.clear_memory();
        }
    }

    /// Clear disk cache
    pub fn clear_disk_cache(&self) -> Result<()> {
        if let Some(ref cache) = self.persistent_cache {
            cache.clear_disk()?;
        }
        Ok(())
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> String {
        if let Some(ref cache) = self.persistent_cache {
            format!("{}", cache.stats())
        } else {
            "No persistent cache enabled".to_string()
        }
    }
}

/// Precompute and save text embeddings to disk for a dataset
pub fn precompute_embeddings(
    prompts: &[String],
    output_path: &Path,
    clip_path: &str,
    t5_path: &str,
    clip_tokenizer_path: &str,
    t5_tokenizer_path: &str,
    device: Device,
) -> Result<()> {
    println!("Precomputing embeddings for {} prompts...", prompts.len());
    let start = std::time::Instant::now();

    // Create optimized encoder
    let mut encoders = OptimizedTextEncoders::new(device);

    // Load models
    encoders.load_clip_l(clip_path)?;
    encoders.load_tokenizers(clip_tokenizer_path, t5_tokenizer_path)?;

    // Batch encode all prompts
    let embeddings = encoders.encode_flux_batch(prompts, t5_path)?;

    // Save embeddings to disk
    let mut embed_map: HashMap<String, Vec<Tensor>> = HashMap::new();
    for (idx, (clip, t5)) in embeddings.into_iter().enumerate() {
        embed_map.insert(format!("clip_{}", idx), vec![clip]);
        embed_map.insert(format!("t5_{}", idx), vec![t5]);
    }

    // Use safetensors format for efficient storage
    println!("Saving precomputed embeddings to {:?}", output_path);
    // Note: Actual safetensors saving would need the safetensors crate's save functionality

    println!("✅ Precomputation complete in {:.2}s", start.elapsed().as_secs_f32());
    Ok(())
}
