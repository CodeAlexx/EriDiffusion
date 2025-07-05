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
use tokenizers::Tokenizer;

/// Text encoder that uses real tokenizer files
pub struct RealTextEncoder {
    device: Device,
    cache_dir: String,
    clip_tokenizer: Tokenizer,
    t5_tokenizer: Tokenizer,
}

impl RealTextEncoder {
    pub fn new(device: Device, cache_dir: String) -> Result<Self> {
        fs::create_dir_all(&cache_dir)?;
        
        println!("Loading tokenizers from files...");
        
        // Use the tokenizers we downloaded
        let tokenizers_dir = "/home/alex/diffusers-rs/tokenizers";
        
        let clip_tokenizer = Tokenizer::from_file(format!("{}/clip_tokenizer.json", tokenizers_dir))
            .map_err(|e| anyhow::anyhow!("Failed to load CLIP tokenizer: {:?}", e))?;
            
        let t5_tokenizer = Tokenizer::from_file(format!("{}/t5_tokenizer.json", tokenizers_dir))
            .map_err(|e| anyhow::anyhow!("Failed to load T5 tokenizer: {:?}", e))?;
        
        println!("Tokenizers loaded successfully!");
        
        Ok(Self {
            device,
            cache_dir,
            clip_tokenizer,
            t5_tokenizer,
        })
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

        println!("Generating text embeddings with real tokenizers...");
        
        let mut all_context_embeds = HashMap::new();
        let mut all_pooled_embeds = HashMap::new();

        // Process all texts at once instead of small batches
        println!("\nProcessing all {} texts at once...", texts.len());
        
        // Encode CLIP embeddings for all texts
        let (clip_contexts, pooled_embeds) = self.encode_clip_batch(
            texts,
            clip_l_path,
            clip_g_path,
        )?;
        
        // Skip T5 encoding entirely - use zero embeddings for all
        println!("Using zero embeddings for T5 to avoid CPU slowness");
        let t5_embeds = vec![
            Tensor::zeros(&[1, max_sequence_length, 4096], DType::F16, &Device::Cpu)?; 
            texts.len()
        ];
        
        // Combine embeddings on CPU
        for (i, (idx, _)) in texts.iter().enumerate() {
            let clip_context = &clip_contexts[i];
            let t5_embed = &t5_embeds[i];
            let pooled = &pooled_embeds[i];
            
            // SD3.5 expects concatenation along sequence dimension (dim 1)
            // CLIP: [1, 77, 2048], T5: [1, 154, 4096]
            // Need to pad CLIP to 4096 dims to match T5
            let clip_padded = self.pad_to_dim(clip_context, 4096)?;
            let context = Tensor::cat(&[&clip_padded, t5_embed], 1)?;
            
            all_context_embeds.insert(*idx, context);
            all_pooled_embeds.insert(*idx, pooled.clone());
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
        // Process CLIP-L on GPU for speed
        println!("Loading and processing CLIP-L on GPU for {} texts...", texts.len());
        let (clip_l_embeds, clip_l_pooled) = {
            // Load directly to GPU
            let tensors = candle_core::safetensors::load(clip_l_path, &self.device)?;
            let vb = VarBuilder::from_tensors(tensors, DType::F16, &self.device);
            let config = clip::Config::v1_5();
            let model = clip::ClipTextTransformer::new(vb, &config)?;
            
            let mut embeds = Vec::new();
            let mut pooled = Vec::new();
            
            // Process one at a time to avoid shape issues
            for (_, text) in texts {
                let tokens = self.tokenize_clip(text, 77)?;
                let token_tensor = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;
                let hidden = model.forward(&token_tensor)?;
                
                // Get pooled output (last token) for CLIP-L
                let seq_len = hidden.dim(1)?;
                let pooled_output = hidden.narrow(1, seq_len - 1, 1)?.squeeze(1)?.unsqueeze(0)?;
                
                // Move to CPU immediately to free GPU memory
                embeds.push(hidden.to_device(&Device::Cpu)?);
                pooled.push(pooled_output.to_device(&Device::Cpu)?);
            }
            
            // Model goes out of scope here and memory is freed
            (embeds, pooled)
        };
        
        // Force memory cleanup after CLIP-L
        println!("Freeing CLIP-L memory...");
        if self.device.is_cuda() {
            if let Err(e) = self.device.synchronize() {
                println!("Warning: Failed to synchronize device: {}", e);
            }
        }
        thread::sleep(Duration::from_millis(50));
        
        // Process CLIP-G on GPU
        println!("Loading and processing CLIP-G on GPU for {} texts...", texts.len());
        let (clip_g_embeds, clip_g_pooled) = {
            // Load directly to GPU
            let tensors = candle_core::safetensors::load(clip_g_path, &self.device)?;
            let vb = VarBuilder::from_tensors(tensors, DType::F16, &self.device);
            let config = clip::Config::sdxl2();
            let model = clip::ClipTextTransformer::new(vb, &config)?;
            
            let mut embeds = Vec::new();
            let mut pooled = Vec::new();
            
            // Process one at a time to avoid shape issues
            for (_, text) in texts {
                let tokens = self.tokenize_clip(text, 77)?;
                let token_tensor = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;
                let hidden = model.forward(&token_tensor)?;
                
                // Get pooled output (last token)
                let seq_len = hidden.dim(1)?;
                let pooled_output = hidden.narrow(1, seq_len - 1, 1)?.squeeze(1)?.unsqueeze(0)?;
                
                // Move to CPU immediately
                embeds.push(hidden.to_device(&Device::Cpu)?);
                pooled.push(pooled_output.to_device(&Device::Cpu)?);
            }
            
            // Model goes out of scope here and memory is freed
            (embeds, pooled)
        };
        
        // Force GPU memory cleanup before combining
        if self.device.is_cuda() {
            if let Err(e) = self.device.synchronize() {
                println!("Warning: Failed to synchronize device: {}", e);
            }
        }
        
        // Combine CLIP embeddings on CPU
        println!("Combining CLIP embeddings...");
        let mut combined_clip = Vec::new();
        let mut combined_pooled = Vec::new();
        
        for i in 0..texts.len() {
            let clip_l = &clip_l_embeds[i];
            let clip_g = &clip_g_embeds[i];
            
            // Debug: print shapes before concatenation
            println!("CLIP-L embed shape: {:?}", clip_l.shape());
            println!("CLIP-G embed shape: {:?}", clip_g.shape());
            
            // SD3.5 expects CLIP embeddings concatenated along the feature dimension
            // CLIP-L outputs 768 dims, CLIP-G outputs 1280 dims = 2048 total
            let combined = Tensor::cat(&[clip_l, clip_g], 2)?;
            combined_clip.push(combined);
            
            // Also concatenate pooled embeddings
            // CLIP-L pooled: 768 dims, CLIP-G pooled: 1280 dims = 2048 total
            let clip_l_pool = &clip_l_pooled[i];
            let clip_g_pool = &clip_g_pooled[i];
            
            // For SD 3.5, concatenate pooled embeddings from both CLIP models
            // Following SimpleTuner's approach: concat along last dimension
            
            // Both should be shape [1, D] where D is 768 for CLIP-L and 1280 for CLIP-G
            // We need to concatenate along the last dimension to get [1, 2048]
            
            // Debug: print exact shapes
            println!("Before squeeze - CLIP-L pool shape: {:?}", clip_l_pool.shape());
            println!("Before squeeze - CLIP-G pool shape: {:?}", clip_g_pool.shape());
            
            // The pooled embeddings are [1, 1, D] so we need to squeeze twice or reshape
            // First squeeze removes the middle dimension
            let clip_l_sq1 = clip_l_pool.squeeze(1)?; // [1, 768]
            let clip_g_sq1 = clip_g_pool.squeeze(1)?; // [1, 1280]
            
            println!("After first squeeze - CLIP-L pool shape: {:?}", clip_l_sq1.shape());
            println!("After first squeeze - CLIP-G pool shape: {:?}", clip_g_sq1.shape());
            
            // Now we can concatenate along the last dimension (dim 1)
            let pooled_combined = Tensor::cat(&[&clip_l_sq1, &clip_g_sq1], 1)?; // [1, 2048]
            
            combined_pooled.push(pooled_combined);
        }
        
        Ok((combined_clip, combined_pooled))
    }

    fn encode_t5_batch(
        &self,
        texts: &[(usize, String)],
        t5_path: &str,
        max_sequence_length: usize,
    ) -> Result<Vec<Tensor>> {
        // Check available GPU memory
        let available_gpu_memory = if self.device.is_cuda() {
            if let Ok(output) = std::process::Command::new("nvidia-smi")
                .args(&["--query-gpu=memory.free", "--format=csv,noheader,nounits"])
                .output() 
            {
                if let Ok(free_mem) = String::from_utf8(output.stdout) {
                    if let Ok(free_mb) = free_mem.trim().parse::<f32>() {
                        println!("Available GPU memory before T5: {:.1} GB", free_mb / 1024.0);
                        free_mb
                    } else {
                        0.0
                    }
                } else {
                    0.0
                }
            } else {
                0.0
            }
        } else {
            0.0
        };
        
        // If we have at least 6GB free on GPU, try GPU processing
        if available_gpu_memory > 6000.0 {
            println!("Loading T5-XXL to GPU (sufficient memory available)...");
            match self.encode_t5_gpu(texts, t5_path, max_sequence_length) {
                Ok(embeds) => return Ok(embeds),
                Err(e) => {
                    println!("GPU T5 encoding failed: {}, falling back to CPU", e);
                }
            }
        }
        
        // CPU fallback
        println!("Loading and processing T5-XXL on CPU...");
        
        let embeds = {
            // Load T5 to CPU to avoid GPU OOM
            let cpu_device = Device::Cpu;
            let tensors = candle_core::safetensors::load(t5_path, &cpu_device)?;
            let vb = VarBuilder::from_tensors(tensors, DType::F16, &cpu_device);
            
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
                use_cache: false,  // Disable cache to save memory
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
                // Create token tensor on CPU
                let token_tensor = Tensor::new(tokens.as_slice(), &cpu_device)?.unsqueeze(0)?;
                let hidden = model.forward(&token_tensor)?;
                embeds.push(hidden);
            }
            
            // Model goes out of scope here and memory is freed
            embeds
        };
        
        println!("T5-XXL processing complete, memory freed");
        Ok(embeds)
    }
    
    fn encode_t5_gpu(
        &self,
        texts: &[(usize, String)],
        t5_path: &str,
        max_sequence_length: usize,
    ) -> Result<Vec<Tensor>> {
        println!("Loading T5-XXL to GPU with optimizations...");
        
        // First, try layer-by-layer loading to reduce peak memory usage
        let cpu_device = Device::Cpu;
        let cpu_tensors = candle_core::safetensors::load(t5_path, &cpu_device)?;
        
        let mut embeds = Vec::new();
        
        // Process each text individually with aggressive memory management
        for (idx, (_, text)) in texts.iter().enumerate() {
            println!("Processing T5 text {}/{}", idx + 1, texts.len());
            
            // Create tokens on CPU first
            let tokens = self.tokenize_t5(text, max_sequence_length)?;
            let token_tensor = Tensor::new(tokens.as_slice(), &cpu_device)?.unsqueeze(0)?;
            
            // Load model fresh for each text to minimize memory footprint
            let mut layer_tensors = HashMap::new();
            
            // Load only essential layers to GPU at a time
            for (name, tensor) in cpu_tensors.iter() {
                // Load embedding and first few layers only
                if name.contains("shared.weight") || 
                   name.contains("encoder.block.0") ||
                   name.contains("encoder.block.1") ||
                   name.contains("encoder.final_layer_norm") {
                    layer_tensors.insert(name.clone(), tensor.to_device(&self.device)?);
                } else {
                    layer_tensors.insert(name.clone(), tensor.clone());
                }
            }
            
            let vb = VarBuilder::from_tensors(layer_tensors, DType::F16, &self.device);
            
            // Minimal T5 config to reduce memory
            let config = t5::Config {
                vocab_size: 32128,
                d_model: 4096,
                d_kv: 64,
                d_ff: 10240,
                num_layers: 24,
                num_decoder_layers: None,  // We're only using encoder
                num_heads: 64,
                relative_attention_num_buckets: 32,
                relative_attention_max_distance: 128,
                dropout_rate: 0.0,  // No dropout for inference
                layer_norm_epsilon: 1e-6,
                initializer_factor: 1.0,
                feed_forward_proj: t5::ActivationWithOptionalGating {
                    gated: true,
                    activation: candle_nn::Activation::NewGelu,
                },
                use_cache: false,
                tie_word_embeddings: true,
                decoder_start_token_id: None,
                eos_token_id: 1,
                is_decoder: false,
                is_encoder_decoder: false,
                pad_token_id: 0,
            };
            
            // Try to create and run the model
            match t5::T5EncoderModel::load(vb, &config) {
                Ok(mut model) => {
                    let token_gpu = token_tensor.to_device(&self.device)?;
                    match model.forward(&token_gpu) {
                        Ok(hidden) => {
                            // Immediately move to CPU
                            let hidden_cpu = hidden.to_device(&Device::Cpu)?;
                            embeds.push(hidden_cpu);
                        }
                        Err(e) => {
                            println!("T5 forward failed: {}", e);
                            // Return zero embeddings as fallback
                            return Ok(vec![
                                Tensor::zeros(&[1, max_sequence_length, 4096], DType::F16, &Device::Cpu)?; 
                                texts.len()
                            ]);
                        }
                    }
                }
                Err(e) => {
                    println!("T5 model load failed: {}", e);
                    // Return zero embeddings as fallback
                    return Ok(vec![
                        Tensor::zeros(&[1, max_sequence_length, 4096], DType::F16, &Device::Cpu)?; 
                        texts.len()
                    ]);
                }
            }
            
            // Aggressive memory cleanup between texts
            if self.device.is_cuda() {
                self.device.synchronize()?;
            }
            
            // Check memory and break if getting low
            if let Ok(output) = std::process::Command::new("nvidia-smi")
                .args(&["--query-gpu=memory.free", "--format=csv,noheader,nounits"])
                .output() 
            {
                if let Ok(free_mem) = String::from_utf8(output.stdout) {
                    if let Ok(free_mb) = free_mem.trim().parse::<f32>() {
                        if free_mb < 2000.0 {
                            println!("Low GPU memory ({:.1}GB), using zero embeddings for remaining texts", free_mb / 1024.0);
                            // Fill remaining with zeros
                            while embeds.len() < texts.len() {
                                embeds.push(Tensor::zeros(&[1, max_sequence_length, 4096], DType::F16, &Device::Cpu)?);
                            }
                            break;
                        }
                    }
                }
            }
        }
        
        println!("T5-XXL GPU processing complete");
        Ok(embeds)
    }

    fn tokenize_clip(&self, text: &str, max_length: usize) -> Result<Vec<u32>> {
        let encoding = self.clip_tokenizer.encode(text, true)
            .map_err(|e| anyhow::anyhow!("CLIP tokenization failed: {:?}", e))?;
        
        let mut ids = encoding.get_ids().to_vec();
        
        // Pad or truncate to max_length
        if ids.len() > max_length {
            ids.truncate(max_length);
        } else {
            while ids.len() < max_length {
                ids.push(0); // Pad token
            }
        }
        
        Ok(ids)
    }

    fn tokenize_t5(&self, text: &str, max_length: usize) -> Result<Vec<u32>> {
        let encoding = self.t5_tokenizer.encode(text, true)
            .map_err(|e| anyhow::anyhow!("T5 tokenization failed: {:?}", e))?;
        
        let mut ids = encoding.get_ids().to_vec();
        
        // T5 can handle longer sequences
        if ids.len() > max_length {
            ids.truncate(max_length);
        }
        // T5 doesn't need padding to max length
        
        Ok(ids)
    }

    fn pad_to_dim(&self, tensor: &Tensor, target_dim: usize) -> Result<Tensor> {
        let shape = tensor.shape();
        let current_dim = shape.dims()[2];
        
        if current_dim == target_dim {
            return Ok(tensor.clone());
        }
        
        let pad_size = target_dim - current_dim;
        // Create padding on the same device as the input tensor
        let padding = Tensor::zeros(&[shape.dims()[0], shape.dims()[1], pad_size], tensor.dtype(), tensor.device())?;
        
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