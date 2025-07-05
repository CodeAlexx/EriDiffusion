use anyhow::Result;
use candle_core::{DType, Device, Tensor, Module};
use candle_nn::VarBuilder;
use candle_transformers::models::stable_diffusion::clip;
use candle_transformers::models::t5;
use std::collections::HashMap;
use safetensors::{SafeTensors, serialize};
use std::fs;
use std::path::Path;
use std::thread;
use std::time::Duration;

use super::embedded_tokenizers::{tokenize_simple, tokenize_t5_simple};

/// Text encoder that uses simple tokenization
pub struct ProperTextEncoder {
    device: Device,
    cache_dir: String,
}

impl ProperTextEncoder {
    pub fn new(device: Device, cache_dir: String) -> Result<Self> {
        fs::create_dir_all(&cache_dir)?;
        Ok(Self { device, cache_dir })
    }

    pub fn encode_and_cache(
        &self,
        texts: &[(usize, String)],
        clip_l_path: &str,
        clip_g_path: &str,
        t5_path: &str,
        max_sequence_length: usize,
    ) -> Result<(HashMap<usize, Tensor>, HashMap<usize, Tensor>)> {
        let cache_path = format!("{}/text_embeddings_cache.safetensors", self.cache_dir);
        
        // Check if cache exists
        if Path::new(&cache_path).exists() {
            println!("Loading text embeddings from cache: {}", cache_path);
            return self.load_from_cache(&cache_path, texts);
        }

        println!("Generating text embeddings with proper tokenizers...");
        
        let mut all_context_embeds = HashMap::new();
        let mut all_pooled_embeds = HashMap::new();

        // Process in smaller batches to avoid OOM
        let batch_size = 5;
        let chunks: Vec<_> = texts.chunks(batch_size).collect();
        
        for (chunk_idx, chunk) in chunks.iter().enumerate() {
            println!("\nProcessing batch {}/{}", chunk_idx + 1, chunks.len());
            
            // Encode CLIP embeddings first
            let (clip_contexts, pooled_embeds) = self.encode_clip_batch(
                chunk,
                clip_l_path,
                clip_g_path,
            )?;
            
            // Clear GPU memory before T5
            if self.device.is_cuda() {
                println!("Clearing GPU memory before T5...");
                if let Err(e) = self.device.synchronize() {
                    println!("Warning: Failed to synchronize device: {}", e);
                }
                thread::sleep(Duration::from_millis(500));
            }
            
            // Encode T5 embeddings
            let t5_embeds = self.encode_t5_batch(
                chunk,
                t5_path,
                max_sequence_length,
            )?;
            
            // Combine embeddings
            for (i, (idx, _)) in chunk.iter().enumerate() {
                let clip_context = &clip_contexts[i];
                let t5_embed = &t5_embeds[i];
                let pooled = &pooled_embeds[i];
                
                // Move to GPU for concatenation if needed
                let clip_gpu = if clip_context.device().is_cpu() {
                    clip_context.to_device(&self.device)?
                } else {
                    clip_context.clone()
                };
                
                let t5_gpu = if t5_embed.device().is_cpu() {
                    t5_embed.to_device(&self.device)?
                } else {
                    t5_embed.clone()
                };
                
                // Concatenate CLIP and T5 embeddings
                let context = Tensor::cat(&[clip_gpu, t5_gpu], 1)?;
                
                // Move to CPU to save GPU memory
                let context_cpu = context.to_device(&Device::Cpu)?;
                
                all_context_embeds.insert(*idx, context_cpu);
                all_pooled_embeds.insert(*idx, pooled.clone());
            }
        }
        
        // Save to cache
        println!("\nSaving embeddings to cache...");
        self.save_to_cache(&cache_path, &all_context_embeds, &all_pooled_embeds)?;
        
        Ok((all_context_embeds, all_pooled_embeds))
    }

    fn encode_clip_batch(
        &self,
        texts: &[(usize, String)],
        clip_l_path: &str,
        clip_g_path: &str,
    ) -> Result<(Vec<Tensor>, Vec<Tensor>)> {
        println!("Loading CLIP models...");
        
        // Process CLIP-L
        println!("Processing CLIP-L...");
        let clip_l_embeds = {
            let tensors = candle_core::safetensors::load(clip_l_path, &self.device)?;
            let vb = VarBuilder::from_tensors(tensors, DType::F16, &self.device);
            let config = clip::Config::v1_5();
            let model = clip::ClipTextTransformer::new(vb, &config)?;
            
            let mut embeds = Vec::new();
            for (_, text) in texts {
                let tokens = self.tokenize_clip(text, 77)?;
                let token_tensor = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;
                let hidden = model.forward(&token_tensor)?;
                // Move to CPU immediately to free GPU memory
                let hidden_cpu = hidden.to_device(&Device::Cpu)?;
                embeds.push(hidden_cpu);
            }
            embeds
        };
        
        // Clear memory
        println!("Clearing memory before CLIP-G...");
        if self.device.is_cuda() {
            if let Err(e) = self.device.synchronize() {
                println!("Warning: Failed to synchronize device: {}", e);
            }
            thread::sleep(Duration::from_millis(500));
        }
        
        // Process CLIP-G
        println!("Processing CLIP-G...");
        let (clip_g_embeds, pooled_embeds) = {
            let tensors = candle_core::safetensors::load(clip_g_path, &self.device)?;
            let vb = VarBuilder::from_tensors(tensors, DType::F16, &self.device);
            let config = clip::Config::sdxl2();
            let model = clip::ClipTextTransformer::new(vb, &config)?;
            
            let mut embeds = Vec::new();
            let mut pooled = Vec::new();
            for (_, text) in texts {
                let tokens = self.tokenize_clip(text, 77)?;
                let token_tensor = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;
                let hidden = model.forward(&token_tensor)?;
                
                // Get pooled output (last token)
                let seq_len = hidden.dim(1)?;
                let pooled_output = hidden.narrow(1, seq_len - 1, 1)?.squeeze(1)?;
                
                // Move to CPU immediately
                let hidden_cpu = hidden.to_device(&Device::Cpu)?;
                let pooled_cpu = pooled_output.to_device(&Device::Cpu)?;
                
                embeds.push(hidden_cpu);
                pooled.push(pooled_cpu);
            }
            (embeds, pooled)
        };
        
        // Combine CLIP embeddings
        let mut combined_clip = Vec::new();
        for i in 0..texts.len() {
            let clip_l = &clip_l_embeds[i];
            let clip_g = &clip_g_embeds[i];
            
            // Move back to GPU for padding and concatenation
            let clip_l_gpu = clip_l.to_device(&self.device)?;
            let clip_g_gpu = clip_g.to_device(&self.device)?;
            
            // Pad to 2048 dims each
            let clip_l_padded = self.pad_to_dim(&clip_l_gpu, 2048)?;
            let clip_g_padded = self.pad_to_dim(&clip_g_gpu, 2048)?;
            
            // Concatenate
            let combined = Tensor::cat(&[clip_l_padded, clip_g_padded], 2)?;
            
            // Move back to CPU
            let combined_cpu = combined.to_device(&Device::Cpu)?;
            combined_clip.push(combined_cpu);
        }
        
        Ok((combined_clip, pooled_embeds))
    }

    fn encode_t5_batch(
        &self,
        texts: &[(usize, String)],
        t5_path: &str,
        max_sequence_length: usize,
    ) -> Result<Vec<Tensor>> {
        println!("Loading T5-XXL...");
        
        let tensors = candle_core::safetensors::load(t5_path, &self.device)?;
        let vb = VarBuilder::from_tensors(tensors, DType::F16, &self.device);
        
        // T5-XXL config
        let config = t5::Config {
            vocab_size: 32128,
            d_model: 4096,
            d_kv: 64,
            d_ff: 10240,
            num_layers: 24,
            num_decoder_layers: Some(24),
            num_heads: 64,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            dropout_rate: 0.1,
            layer_norm_epsilon: 1e-6,
            initializer_factor: 1.0,
            feed_forward_proj: t5::ActivationWithOptionalGating {
                gated: true,
                activation: candle_nn::Activation::NewGelu,
            },
            use_cache: true,
            tie_word_embeddings: true,
            decoder_start_token_id: Some(0),
            eos_token_id: 1,
            is_decoder: false,
            is_encoder_decoder: false,
            pad_token_id: 0,
        };
        
        let mut model = t5::T5EncoderModel::load(vb, &config)?;
        
        let mut embeds = Vec::new();
        for (_, text) in texts {
            let tokens = self.tokenize_t5(text, max_sequence_length)?;
            let token_tensor = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;
            let hidden = model.forward(&token_tensor)?;
            // Move to CPU immediately
            let hidden_cpu = hidden.to_device(&Device::Cpu)?;
            embeds.push(hidden_cpu);
        }
        
        Ok(embeds)
    }

    fn tokenize_clip(&self, text: &str, max_length: usize) -> Result<Vec<u32>> {
        tokenize_simple(text, max_length, 0)
    }

    fn tokenize_t5(&self, text: &str, max_length: usize) -> Result<Vec<u32>> {
        tokenize_t5_simple(text, max_length)
    }

    fn pad_to_dim(&self, tensor: &Tensor, target_dim: usize) -> Result<Tensor> {
        let shape = tensor.shape();
        let current_dim = shape.dims()[2];
        
        if current_dim == target_dim {
            return Ok(tensor.clone());
        }
        
        let pad_size = target_dim - current_dim;
        let padding = Tensor::zeros(&[shape.dims()[0], shape.dims()[1], pad_size], tensor.dtype(), &self.device)?;
        
        Ok(Tensor::cat(&[tensor, &padding], 2)?)
    }

    fn save_to_cache(
        &self,
        path: &str,
        context_embeds: &HashMap<usize, Tensor>,
        pooled_embeds: &HashMap<usize, Tensor>,
    ) -> Result<()> {
        let mut tensors = HashMap::new();
        
        for (idx, embed) in context_embeds {
            tensors.insert(format!("context_{}", idx), embed.clone());
        }
        
        for (idx, embed) in pooled_embeds {
            tensors.insert(format!("pooled_{}", idx), embed.clone());
        }
        
        let data = serialize(&tensors, &None)?;
        fs::write(path, data)?;
        
        println!("Saved {} embeddings to cache", context_embeds.len());
        Ok(())
    }

    fn load_from_cache(
        &self,
        path: &str,
        texts: &[(usize, String)],
    ) -> Result<(HashMap<usize, Tensor>, HashMap<usize, Tensor>)> {
        let data = fs::read(path)?;
        let tensors = SafeTensors::deserialize(&data)?;
        
        let mut context_embeds = HashMap::new();
        let mut pooled_embeds = HashMap::new();
        
        for (idx, _) in texts {
            let context_key = format!("context_{}", idx);
            let pooled_key = format!("pooled_{}", idx);
            
            if let Ok(context_data) = tensors.tensor(&context_key) {
                let shape = context_data.shape();
                let tensor = Tensor::from_raw_buffer(
                    context_data.data(),
                    context_data.dtype().try_into()?,
                    shape,
                    &Device::Cpu,
                )?;
                context_embeds.insert(*idx, tensor);
            }
            
            if let Ok(pooled_data) = tensors.tensor(&pooled_key) {
                let shape = pooled_data.shape();
                let tensor = Tensor::from_raw_buffer(
                    pooled_data.data(),
                    pooled_data.dtype().try_into()?,
                    shape,
                    &Device::Cpu,
                )?;
                pooled_embeds.insert(*idx, tensor);
            }
        }
        
        println!("Loaded {} cached embeddings", context_embeds.len());
        Ok((context_embeds, pooled_embeds))
    }
}