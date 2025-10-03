use crate::loaders::WeightLoader;
use crate::models::clip;
use crate::models::T5EncoderModel;
use flame_core::device::Device;
use flame_core::optimizers::{Adam, SGD};
use flame_core::Result;
use flame_core::{CudaDevice, DType, Error, Shape, Tensor};
use safetensors::{serialize, SafeTensors};
use std::{collections::HashMap, fs, path::Path, thread, time::Duration};
use tokenizers::Tokenizer;

pub struct PrefixedWeightLoader {
    loader: WeightLoader,
    prefix: String,
}
pub struct CachedTextEncoder {
    device: Device,
    cache_dir: String,
}

// FLAME uses flame_core::device::Device instead of Device

// FLAME uses flame_core::device::Device instead of Device

// FLAME uses flame_core::device::Device instead of Device

/// A memory-efficient text encoder that can load models one at a time
/// and cache embeddings to disk to avoid OOM issues

impl CachedTextEncoder {
    pub fn new(device: Device, cache_dir: String) -> flame_core::Result<Self> {
        // Create cache directory if it doesn't exist
        fs::create_dir_all(&cache_dir)
            .map_err(|e| Error::Io(format!("Failed to create cache directory: {}", e)))?;
        Ok(Self { device, cache_dir })
    }

    /// Encode texts and save to cache, or load from cache if available
    pub fn encode_and_cache(
        &self,
        texts: &[(usize, String)],
        clip_l_path: &str,
        clip_g_path: &str,
        t5_path: &str,
        tokenizer_clip_path: &str,
        tokenizer_t5_path: &str,
        max_sequence_length: usize,
    ) -> flame_core::Result<(HashMap<usize, Tensor>, HashMap<usize, Tensor>)> {
        let cache_path = format!("{}/text_embeddings_cache.safetensors", self.cache_dir);

        // Check if cache exists
        if Path::new(&cache_path).exists() {
            println!("Loading text embeddings from cache: {}", cache_path);
            return self.load_from_cache(&cache_path, texts);
        }

        println!("Generating text embeddings (this will be done in batches to save memory)...");

        let mut all_context_embeds = HashMap::new();
        let mut all_pooled_embeds = HashMap::new();

        // Process in smaller batches to avoid OOM
        let batch_size = 5; // Reduced from 10 to 5 for better memory management
        let chunks: Vec<_> = texts.chunks(batch_size).collect();

        for (chunk_idx, chunk) in chunks.iter().enumerate() {
            println!("\nProcessing batch {}/{}", chunk_idx + 1, chunks.len());

            // Encode CLIP embeddings first
            let (clip_contexts, pooled_embeds) =
                self.encode_clip_batch(chunk, clip_l_path, clip_g_path, tokenizer_clip_path)?;

            // Clear GPU memory before T5
            // Device synchronization not needed in FLAME
            println!("Clearing GPU memory before T5...");
            // Now actually attempts GPU memory cleanup via CUDA API
            // #[cfg(feature = "cuda")]
            // unsafe {
            // // CUDA memory advise is not available in FLAME
            // if let Ok(cuda_ctx) = cudarc::driver::cuCtxGetCurrent() {
            // let _ = cudarc::driver::cuMemAdvise(
            // 0,
            // 0,
            // cudarc::driver::CUmem_advise::CU_MEM_ADVISE_UNSET_READ_MOSTLY,
            // 0,
            // );
            // }
            // }

            // Encode T5 embeddings
            let t5_embeds =
                self.encode_t5_batch(chunk, t5_path, tokenizer_t5_path, max_sequence_length)?;

            // Combine embeddings
            for (i, (idx, _)) in chunk.iter().enumerate() {
                let clip_context = &clip_contexts[i];
                let t5_embed = &t5_embeds[i];
                let pooled = &pooled_embeds[i];

                // FLAME only supports CUDA devices, so no CPU check needed
                let clip_gpu = clip_context;
                let t5_gpu = t5_embed;

                // Concatenate CLIP and T5 embeddings
                let context = Tensor::cat(&[clip_gpu, t5_gpu], 1)?;

                // Move to CPU to save GPU memory
                let context_cpu = context;

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
        tokenizer_path: &str,
    ) -> flame_core::Result<(Vec<Tensor>, Vec<Tensor>)> {
        println!("Loading CLIP models...");

        // Load tokenizer
        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path).map_err(|e| {
            flame_core::Error::InvalidOperation(format!(
                "Failed to load CLIP tokenizer: {:?}",
                e
            ))
        })?;

        // Process CLIP-L
        println!("Processing CLIP-L...");
        let clip_l_embeds = {
            let wl = WeightLoader::from_safetensors(clip_l_path, self.device.clone())?;
            let config = crate::models::text_encoder::CLIPConfig::clip_l();
            let weights = wl.weights;
            let model = clip::ClipTextTransformer::new(config, &self.device, weights)?;

            let mut embeds = Vec::new();
            for (_, text) in texts {
                let tokens = self.tokenize_clip(&tokenizer, text, 77)?;
                let output = model.forward(&tokens, None)?;
                let hidden = output.last_hidden_state;
                // Move to CPU immediately to free GPU memory
                let hidden_cpu = hidden;
                embeds.push(hidden_cpu);
            }
            embeds
        };

        // Clear memory
        println!("Clearing memory before CLIP-G...");
        // Device synchronization not needed in FLAME

        // Process CLIP-G
        println!("Processing CLIP-G...");
        let (clip_g_embeds, pooled_embeds) = {
            let wl = WeightLoader::from_safetensors(clip_g_path, self.device.clone())?;
            let config = crate::models::text_encoder::CLIPConfig::clip_g();
            let weights = wl.weights;
            let model = clip::ClipTextTransformer::new(config, &self.device, weights)?;

            let mut embeds = Vec::new();
            let mut pooled = Vec::new();
            for (_, text) in texts {
                let tokens = self.tokenize_clip(&tokenizer, text, 77)?;
                let output = model.forward(&tokens, None)?;
                let hidden = output.last_hidden_state;

                // Get pooled output (last token)
                let seq_len = hidden.shape().dims()[1];
                let pooled_output = hidden.narrow(1, seq_len - 1, 1)?.squeeze(Some(1))?;

                // Move to CPU immediately
                let hidden_cpu = hidden;
                let pooled_cpu = pooled_output;

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
            let clip_l_gpu = clip_l;
            let clip_g_gpu = clip_g;

            // Pad to 2048 dims each
            let clip_l_padded = self.pad_to_dim(&clip_l_gpu, 2048)?;
            let clip_g_padded = self.pad_to_dim(&clip_g_gpu, 2048)?;

            // Concatenate
            let combined = Tensor::cat(&[&clip_l_padded, &clip_g_padded], 2)?;

            // Move back to CPU
            let combined_cpu = combined;
            combined_clip.push(combined_cpu);
        }

        Ok((combined_clip, pooled_embeds))
    }

    fn encode_t5_batch(
        &self,
        texts: &[(usize, String)],
        t5_path: &str,
        tokenizer_path: &str,
        max_sequence_length: usize,
    ) -> flame_core::Result<Vec<Tensor>> {
        println!("Loading T5-XXL...");

        // Load tokenizer
        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path).map_err(|e| {
            flame_core::Error::InvalidOperation(format!(
                "Failed to load T5 tokenizer: {:?}",
                e
            ))
        })?;

        let wl = WeightLoader::from_safetensors(t5_path, self.device.clone())?;

        // T5-XXL config
        let config = crate::models::text_encoder::T5Config {
            vocab_size: 32128,
            d_model: 4096,
            d_ff: 10240,
            num_layers: 24,
            num_heads: 64,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            dropout_rate: 0.1,
            layer_norm_epsilon: 1e-6,
        };

        // TODO: T5EncoderModel not yet implemented
        // let mut model = T5EncoderModel::load(wl, &config)?;
        return Err(flame_core::Error::InvalidOperation(
            "T5EncoderModel not yet implemented".to_string(),
        ));

        // Unreachable code below - commented out
        // let mut embeds = Vec::new();
        // for (_, text) in texts {
        //     let tokens = self.tokenize_t5(&tokenizer, text, max_sequence_length)?;
        //     let hidden = model.forward(&tokens)?;
        //     // Move to CPU immediately
        //     let hidden_cpu = hidden;
        //     embeds.push(hidden_cpu);
        // }
        //
        // Ok(embeds)
    }

    fn tokenize_clip(
        &self,
        tokenizer: &tokenizers::Tokenizer,
        text: &str,
        max_length: usize,
    ) -> flame_core::Result<Tensor> {
        let encoding = tokenizer.encode(text, true).map_err(|e| {
            flame_core::Error::InvalidOperation(format!("Tokenization failed: {:?}", e))
        })?;
        let mut ids = encoding.get_ids().to_vec();

        // Pad or truncate
        ids.resize(max_length, 0);

        let ids_u32: Vec<u32> = ids.into_iter().map(|id| id as u32).collect();
        let shape = Shape::from_dims(&[1, ids_u32.len()]);
        Tensor::from_vec(
            ids_u32.iter().map(|&x| x as f32).collect(),
            shape,
            self.device.cuda_device().clone(),
        )?
        .to_dtype(DType::U32)
    }

    fn tokenize_t5(
        &self,
        tokenizer: &tokenizers::Tokenizer,
        text: &str,
        max_length: usize,
    ) -> flame_core::Result<Tensor> {
        let encoding = tokenizer.encode(text, true).map_err(|e| {
            flame_core::Error::InvalidOperation(format!("Tokenization failed: {:?}", e))
        })?;
        let mut ids = encoding.get_ids().to_vec();

        if ids.len() > max_length {
            ids.truncate(max_length);
        }

        let ids_u32: Vec<u32> = ids.into_iter().map(|id| id as u32).collect();
        let shape = Shape::from_dims(&[1, ids_u32.len()]);
        Tensor::from_vec(
            ids_u32.iter().map(|&x| x as f32).collect(),
            shape,
            self.device.cuda_device().clone(),
        )?
        .to_dtype(DType::U32)
    }

    fn pad_to_dim(&self, tensor: &Tensor, target_dim: usize) -> flame_core::Result<Tensor> {
        let shape = tensor.shape();
        let current_dim = shape.dims()[2];

        if current_dim == target_dim {
            return Ok(tensor.clone());
        }

        let pad_size = target_dim - current_dim;
        let pad_shape = vec![shape.dims()[0], shape.dims()[0], pad_size];
        let padding = Tensor::zeros_dtype(
            Shape::from_dims(&pad_shape),
            tensor.dtype(),
            tensor.device().clone(),
        )?;

        Ok(Tensor::cat(&[tensor, &padding], 2)?)
    }

    fn save_to_cache(
        &self,
        path: &str,
        context_embeds: &HashMap<usize, Tensor>,
        pooled_embeds: &HashMap<usize, Tensor>,
    ) -> flame_core::Result<()> {
        let mut tensors = HashMap::new();

        for (idx, embed) in context_embeds {
            tensors.insert(format!("context_{}", idx), embed.clone());
        }

        for (idx, embed) in pooled_embeds {
            tensors.insert(format!("pooled_{}", idx), embed.clone());
        }

        // Use FLAME's save_file function to save tensors in safetensors format
        flame_core::serialization::save_file(&tensors, path)?;

        println!("Saved {} embeddings to cache", context_embeds.len());
        Ok(())
    }

    fn load_from_cache(
        &self,
        path: &str,
        texts: &[(usize, String)],
    ) -> flame_core::Result<(HashMap<usize, Tensor>, HashMap<usize, Tensor>)> {
        let data = fs::read(path)
            .map_err(|e| Error::Io(format!("Failed to read cache file: {}", e)))?;
        let tensors = SafeTensors::deserialize(&data).map_err(|e| {
            Error::InvalidOperation(format!("Failed to deserialize safetensors: {}", e))
        })?;

        let mut context_embeds = HashMap::new();
        let mut pooled_embeds = HashMap::new();

        for (idx, _) in texts {
            let context_key = format!("context_{}", idx);
            let pooled_key = format!("pooled_{}", idx);

            if let Ok(context_data) = tensors.tensor(&context_key) {
                let shape_dims: Vec<usize> = context_data.shape().to_vec();
                let shape = Shape::from_dims(&shape_dims);
                let data_f32: Vec<f32> = context_data
                    .data()
                    .chunks(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                let tensor = Tensor::from_vec(data_f32, shape, self.device.cuda_device().clone())?;
                context_embeds.insert(*idx, tensor);
            }

            if let Ok(pooled_data) = tensors.tensor(&pooled_key) {
                let shape_dims: Vec<usize> = pooled_data.shape().to_vec();
                let shape = Shape::from_dims(&shape_dims);
                let data_f32: Vec<f32> = pooled_data
                    .data()
                    .chunks(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                let tensor = Tensor::from_vec(data_f32, shape, self.device.cuda_device().clone())?;
                pooled_embeds.insert(*idx, tensor);
            }
        }

        println!("Loaded {} cached embeddings", context_embeds.len());
        Ok((context_embeds, pooled_embeds))
    }
}
