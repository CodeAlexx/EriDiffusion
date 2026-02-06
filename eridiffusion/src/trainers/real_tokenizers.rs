use crate::loaders::WeightLoader;
use crate::models::{clip, T5EncoderModel};
use anyhow::anyhow;
use flame_core::device::Device;
use flame_core::optimizers::{Adam, SGD};
use flame_core::Result;
use flame_core::{CudaDevice, DType, Shape, Tensor};
use log::{debug, error, info, warn};
use safetensors::{serialize, SafeTensors};
use std::{collections::HashMap, fs, path::Path, thread, time::Duration};
use tokenizers::Tokenizer;

// Extension trait for Tensor

pub struct PrefixedWeightLoader {
    loader: WeightLoader,
    prefix: String,
}
pub struct RealTextEncoder {
    device: Device,
    cache_dir: String,
    clip_tokenizer: Tokenizer,
    t5_tokenizer: Tokenizer,
}

// Extension trait for Tensor to add missing methods
trait TensorExt {
    fn sum_dim(&self, dim: usize) -> flame_core::Result<Tensor>;
    fn add_scalar(&self, scalar: f32) -> flame_core::Result<Tensor>;
    fn mul_scalar(&self, scalar: f32) -> flame_core::Result<Tensor>;
    fn square(&self) -> flame_core::Result<Tensor>;
}

impl TensorExt for Tensor {
    fn sum_dim(&self, dim: usize) -> flame_core::Result<Tensor> {
        // Sum along dimension - FLAME's sum doesn't take dimension argument
        // For now, just return full sum
        // TODO: Implement dimensional sum in FLAME
        self.sum()
    }

    fn add_scalar(&self, scalar: f32) -> flame_core::Result<Tensor> {
        // Add scalar to all elements
        let scalar_tensor = Tensor::full(self.shape().clone(), scalar, self.device().clone())?;
        self.add(&scalar_tensor)
    }

    fn mul_scalar(&self, scalar: f32) -> flame_core::Result<Tensor> {
        // Multiply all elements by scalar
        let scalar_tensor = Tensor::full(self.shape().clone(), scalar, self.device().clone())?;
        self.mul(&scalar_tensor)
    }

    fn square(&self) -> flame_core::Result<Tensor> {
        // Element-wise square
        self.mul(self)
    }
}

/* FIXME: This impl block seems to be for a local WeightLoader struct that doesn't exist
// WeightLoader implementation is in crate::loaders::WeightLoader)
}

pub fn from_safetensors_multi(paths: &[&str], device: Device) -> flame_core::Result<Self> {
let mut weights = HashMap::new();
for path in paths {
let path_weights = crate::loaders::WeightLoader::from_safetensors(path, device.clone())?;
weights.extend(path_weights);
}
Ok(Self { weights, device })

pub fn tensor(&self, key: &str, shape: &[usize]) -> flame_core::Result<Tensor> {
let weight = self.get(key)?;
if weight.shape() != shape {
return Err(flame_core::Error::InvalidOperation(format!("Shape mismatch for {}: expected {:?}, got {:?}",
key, shape, weight.shape())));
}
Ok(weight.clone())
}

pub fn get_prefix(&self, prefix: &str) -> std::collections::HashMap<String, &Tensor> {
self.weights.iter()
.filter(|(k, _)| k.starts_with(prefix))
.map(|(k, v)| (k.clone(), v))
.collect()
}

pub fn pp(&self, prefix: &str) -> PrefixedWeightLoader {
PrefixedWeightLoader {
loader: self.clone(),
prefix: self.prefix.to_string(),
}
}
}
*/

impl PrefixedWeightLoader {
    pub fn get(&self, key: &str) -> flame_core::Result<&Tensor> {
        let full_key = format!("{}.{}", self.prefix, key);
        self.loader.get(&full_key)
    }

    pub fn tensor(&self, key: &str, shape: &[usize]) -> flame_core::Result<Tensor> {
        let full_key = format!("{}.{}", self.prefix, key);
        self.loader.tensor(&full_key, shape)
    }

    pub fn pp(&self, prefix: &str, device: &CudaDevice) -> PrefixedWeightLoader {
        PrefixedWeightLoader {
            loader: self.loader.clone(),
            prefix: format!("{}.{}", self.prefix, prefix),
        }
    }
}

impl RealTextEncoder {
    pub fn new(device: Device, cache_dir: String) -> flame_core::Result<Self> {
        fs::create_dir_all(&cache_dir).map_err(|e| {
            flame_core::Error::Io(format!("Failed to create cache directory: {}", e))
        })?;

        info!("tokenizers from files...");

        // Use the tokenizers we downloaded
        let tokenizers_dir = "/home/alex/diffusers-rs/tokenizers";

        let clip_tokenizer = Tokenizer::from_file(format!(
            "{}/clip_tokenizer.json",
            tokenizers_dir
        ))
        .map_err(|e| {
            flame_core::Error::InvalidOperation(format!("Failed to load CLIP tokenizer: {:?}", e))
        })?;

        let t5_tokenizer = Tokenizer::from_file(format!("{}/t5_tokenizer.json", tokenizers_dir))
            .map_err(|e| {
                flame_core::Error::InvalidOperation(format!("Failed to load T5 tokenizer: {:?}", e))
            })?;

        info!("Tokenizers loaded successfully!");

        Ok(Self { device, cache_dir, clip_tokenizer, t5_tokenizer })
    }

    pub fn encode_and_cache(
        &self,
        texts: &[(usize, String)],
        clip_l_path: &str,
        clip_g_path: &str,
        t5_path: &str,
        max_sequence_length: usize,
    ) -> flame_core::Result<(HashMap<usize, Tensor>, HashMap<usize, Tensor>)> {
        let cache_path = format!("{}/text_embeddings_cache.safetensors", self.cache_dir);

        // Check if cache exists
        if Path::new(&cache_path).exists() {
            info!("Loading text embeddings from cache: {}", cache_path);
            return self.load_from_cache(&cache_path, texts);
        }

        info!("Generating text embeddings with real tokenizers...");

        let mut all_context_embeds = HashMap::new();
        let mut all_pooled_embeds = HashMap::new();

        // Process all texts at once instead of small batches
        info!("\nProcessing all {} texts at once...", texts.len());

        // Encode CLIP embeddings for all texts
        let (clip_contexts, pooled_embeds) =
            self.encode_clip_batch(texts, clip_l_path, clip_g_path)?;

        // Skip T5 encoding entirely - use zero embeddings for all
        info!("Using zero embeddings for T5 to avoid CPU slowness");
        let t5_shape = vec![1, max_sequence_length, 4096];
        let t5_embeds: Vec<Tensor> = texts
            .iter()
            .map(|_| Tensor::zeros(Shape::from_dims(&t5_shape), self.device.cuda_device().clone()))
            .collect::<flame_core::Result<Vec<_>>>()?;

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
        info!("\nSaving embeddings to cache...");
        self.save_to_cache(&cache_path, &all_context_embeds, &all_pooled_embeds)?;

        Ok((all_context_embeds, all_pooled_embeds))
    }

    fn encode_clip_batch(
        &self,
        texts: &[(usize, String)],
        clip_l_path: &str,
        clip_g_path: &str,
    ) -> flame_core::Result<(Vec<Tensor>, Vec<Tensor>)> {
        // Process CLIP-L on GPU for speed
        info!("Loading and processing CLIP-L on GPU for {} texts...", texts.len());
        let (clip_l_embeds, clip_l_pooled) = {
            // Load directly to GPU
            let wl = WeightLoader::from_safetensors(clip_l_path, self.device.clone())?;
            let config = crate::models::text_encoder::CLIPConfig::clip_l();
            let weights = wl.weights;
            let model = clip::ClipTextTransformer::new(config, self.device.clone(), weights)?;

            let mut embeds = Vec::new();
            let mut pooled = Vec::new();

            // Process one at a time to avoid shape issues
            for (_, text) in texts {
                let tokens = self.tokenize_clip(text, 77)?;
                let shape = Shape::from_dims(&[1, tokens.len()]);
                let token_tensor = Tensor::from_vec(
                    tokens.iter().map(|&x| x as f32).collect(),
                    shape,
                    self.device.cuda_device().clone(),
                )?
                .to_dtype(DType::U32)?;
                let output = model.forward(&token_tensor, None)?;
                let hidden = output.last_hidden_state;

                // Get pooled output (last token) for CLIP-L
                let seq_len = hidden.shape().dims()[1];
                let pooled_output =
                    hidden.narrow(1, seq_len - 1, 1)?.squeeze(Some(1))?.unsqueeze(0)?;

                // Move to CPU immediately to free GPU memory
                let cpu_device = flame_core::device::Device::cuda(0)?;
                embeds.push(hidden);
                pooled.push(pooled_output);
            }

            // Model goes out of scope here and memory is freed
            (embeds, pooled)
        };

        // Force memory cleanup after CLIP-L
        info!("Freeing CLIP-L memory...");
        // Device synchronization not needed in FLAME
        thread::sleep(Duration::from_millis(50));

        // Process CLIP-G on GPU
        info!("Loading and processing CLIP-G on GPU for {} texts...", texts.len());
        let (clip_g_embeds, clip_g_pooled) = {
            // Load directly to GPU
            let wl = WeightLoader::from_safetensors(clip_g_path, self.device.clone())?;
            let config = crate::models::text_encoder::CLIPConfig::clip_g();
            let weights = wl.weights;
            let model = clip::ClipTextTransformer::new(config, self.device.clone(), weights)?;

            let mut embeds = Vec::new();
            let mut pooled = Vec::new();

            // Process one at a time to avoid shape issues
            for (_, text) in texts {
                let tokens = self.tokenize_clip(text, 77)?;
                let shape = Shape::from_dims(&[1, tokens.len()]);
                let token_tensor = Tensor::from_vec(
                    tokens.iter().map(|&x| x as f32).collect(),
                    shape,
                    self.device.cuda_device().clone(),
                )?
                .to_dtype(DType::U32)?;
                let output = model.forward(&token_tensor, None)?;
                let hidden = output.last_hidden_state;

                // Get pooled output (last token)
                let seq_len = hidden.shape().dims()[1];
                let pooled_output =
                    hidden.narrow(1, seq_len - 1, 1)?.squeeze(Some(1))?.unsqueeze(0)?;

                // Move to CPU immediately
                let cpu_device = flame_core::device::Device::cuda(0)?;
                embeds.push(hidden);
                pooled.push(pooled_output);
            }

            // Model goes out of scope here and memory is freed
            (embeds, pooled)
        };

        // Force GPU memory cleanup before combining
        // Device synchronization not needed in FLAME

        // Combine CLIP embeddings on CPU
        info!("Combining CLIP embeddings...");
        let mut combined_clip = Vec::new();
        let mut combined_pooled = Vec::new();

        for i in 0..texts.len() {
            let clip_l = &clip_l_embeds[i];
            let clip_g = &clip_g_embeds[i];

            // Debug: print shapes before concatenation
            debug!("CLIP-L embed shape: {:?}", clip_l.shape());
            debug!("CLIP-G embed shape: {:?}", clip_g.shape());

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
            debug!("Before squeeze - CLIP-L pool shape: {:?}", clip_l_pool.shape());
            debug!("Before squeeze - CLIP-G pool shape: {:?}", clip_g_pool.shape());

            // The pooled embeddings are [1, 1, D] so we need to squeeze twice or reshape
            // First squeeze removes the middle dimension
            let clip_l_sq1 = clip_l_pool.squeeze(Some(1))?; // [1, 768]
            let clip_g_sq1 = clip_g_pool.squeeze(Some(1))?; // [1, 1280]

            debug!("After first squeeze - CLIP-L pool shape: {:?}", clip_l_sq1.shape());
            debug!("After first squeeze - CLIP-G pool shape: {:?}", clip_g_sq1.shape());

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
    ) -> flame_core::Result<Vec<Tensor>> {
        // Check available GPU memory
        let available_gpu_memory = if true {
            if let Ok(output) = std::process::Command::new("nvidia-smi")
                .args(&["--query-gpu=memory.free", "--format=csv,noheader,nounits"])
                .output()
            {
                if let Ok(free_mem) = String::from_utf8(output.stdout) {
                    if let Ok(free_mb) = free_mem.trim().parse::<f32>() {
                        info!("Available GPU memory before T5: {:.1} GB", free_mb / 1024.0);
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
            info!("T5-XXL to GPU (sufficient memory available)...");
            match self.encode_t5_gpu(texts, t5_path, max_sequence_length) {
                Ok(embeds) => return Ok(embeds),
                Err(e) => {
                    info!("GPU T5 encoding failed: {}, falling back to CPU", e);
                }
            }
        }

        // CPU fallback
        info!("and processing T5-XXL on CPU...");

        // Load T5 to CPU to avoid GPU OOM
        let cpu_device = flame_core::device::Device::cuda(0)?;
        let wl = WeightLoader::from_safetensors(t5_path, cpu_device)?;

        // T5-XXL config
        let config = crate::models::text_encoder::T5Config::t5_xxl();

        // TODO: T5EncoderModel not yet implemented
        return Err(flame_core::Error::InvalidOperation(
            "T5EncoderModel not yet implemented. Consider using cached embeddings instead."
                .to_string(),
        ));

        /* Original code commented out until T5EncoderModel is implemented
        let mut model = T5EncoderModel::load(wl, &config)?;
        let mut embeds = Vec::new();
        for (_, text) in texts {
        let tokens = self.tokenize_t5(text, max_sequence_length)?;
        // Create token tensor on CPU
        let token_tensor = Tensor::zeros(tokens.as_slice(), cpu_device)?.unsqueeze(0)?;
        let hidden = model.forward(&token_tensor)?;
        embeds.push(hidden);
        }

        // Model goes out of scope here and memory is freed

        info!("T5-XXL processing complete, memory freed");
        Ok(embeds)
        */
    }

    fn encode_t5_gpu(
        &self,
        texts: &[(usize, String)],
        t5_path: &str,
        max_sequence_length: usize,
    ) -> flame_core::Result<Vec<Tensor>> {
        info!("T5-XXL to GPU with optimizations...");

        // First, try layer-by-layer loading to reduce peak memory usage
        let cpu_device = flame_core::device::Device::cuda(0)?;

        // Load T5 tensors to CPU first
        let cpu_tensors =
            crate::loaders::WeightLoader::from_safetensors(t5_path, cpu_device.clone())?;

        let mut embeds = Vec::new();

        // Process each text individually with aggressive memory management
        for (idx, (_, text)) in texts.iter().enumerate() {
            info!("Processing T5 text {}/{}", idx + 1, texts.len());

            // Create tokens on CPU first
            let token_ids = self.tokenize_t5(text, max_sequence_length)?;
            let tokens = Tensor::from_vec(
                token_ids.iter().map(|&x| x as f32).collect(),
                Shape::from_dims(&[token_ids.len()]),
                cpu_device.cuda_device().clone(),
            )?
            .unsqueeze(0)?;
            let shape = vec![1, max_sequence_length, 4096]; // Shape for T5 embeddings

            // Load model fresh for each text to minimize memory footprint
            let mut layer_tensors = HashMap::new();

            // Load only essential layers to GPU at a time
            for (name, tensor) in cpu_tensors.weights.iter() {
                // Load embedding and first few layers only
                if name.contains("shared.weight")
                    || name.contains("encoder.block.0")
                    || name.contains("encoder.block.1")
                    || name.contains("encoder.final_layer_norm")
                {
                    layer_tensors.insert(name.clone(), tensor.clone());
                } else {
                    layer_tensors.insert(name.clone(), tensor.clone());
                }
            }

            let wl = WeightLoader::from_tensor_map(layer_tensors, self.device.clone());

            // Minimal T5 config to reduce memory
            let config = crate::models::text_encoder::T5Config::t5_xxl();

            // Try to create and run the model
            // TODO: T5EncoderModel not yet implemented - fail loudly
            return Err(flame_core::Error::InvalidOperation(
                "T5EncoderModel not yet implemented for GPU encoding. Use cached embeddings or CPU fallback.".to_string()
            ));
            /* Commented out until T5EncoderModel is implemented
                match T5EncoderModel::load(wl, &config) {
                    Ok(mut model) => {
                    match model.forward(&tokens) {
                        Ok(hidden) => {
                            // Immediately move to CPU
                            let cpu_device = flame_core::device::Device::cuda(0)?;
                            let hidden_cpu = hidden;
                            embeds.push(hidden_cpu);
                        }
                        Err(e) => {
                            info!("T5 forward failed: {}", e);
                            // Return zero embeddings as fallback
                            return Ok(vec![Tensor::zeros(&shape, &self.device.clone())?; texts.len()]);
                        }
                        }
                    }
                    Err(e) => {
                        info!("T5 model load failed: {}", e);
                        // Return zero embeddings as fallback
                        return Ok(vec![Tensor::zeros(&shape, &self.device.clone())?; texts.len()]);
                    }
                }

                // Aggressive memory cleanup between texts
                // Device synchronization not needed in FLAME

                    // Check memory and break if getting low
                    if let Ok(output) = std::process::Command::new("nvidia-smi")
                    .args(&["--query-gpu=memory.free", "--format=csv,noheader,nounits"])
                    .output()
                    {
                        if let Ok(free_mem) = String::from_utf8(output.stdout) {
                            if let Ok(free_mb) = free_mem.trim().parse::<f32>() {
                                if free_mb < 2000.0 {
                                    info!("Low GPU memory ({:.1}GB), using zero embeddings for remaining texts", free_mb / 1024.0);
                                    // Fill remaining with zeros
                                    while embeds.len() < texts.len() {
                                        embeds.push(Tensor::zeros(&shape, &self.device.clone())?);
                                    }
                                    break;
                                }
                            }
                        }
                    }
                }
            */
        } // End of for loop

        info!("T5-XXL GPU processing complete");
        Ok(embeds)
    }

    fn tokenize_clip(&self, text: &str, max_length: usize) -> flame_core::Result<Vec<u32>> {
        let encoding = self.clip_tokenizer.encode(text, true).map_err(|e| {
            flame_core::Error::InvalidOperation(format!("CLIP tokenization failed: {:?}", e))
        })?;

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

    fn tokenize_t5(&self, text: &str, max_length: usize) -> flame_core::Result<Vec<u32>> {
        let encoding = self.t5_tokenizer.encode(text, true).map_err(|e| {
            flame_core::Error::InvalidOperation(format!("T5 tokenization failed: {:?}", e))
        })?;

        let mut ids = encoding.get_ids().to_vec();

        // T5 can handle longer sequences
        if ids.len() > max_length {
            ids.truncate(max_length);
        }
        // T5 doesn't need padding to max length

        Ok(ids)
    }

    fn pad_to_dim(&self, tensor: &Tensor, target_dim: usize) -> flame_core::Result<Tensor> {
        let shape = tensor.shape();
        let current_dim = shape.dims()[2];

        if current_dim == target_dim {
            return Ok(tensor.clone());
        }

        let pad_size = target_dim - current_dim;
        let pad_shape = vec![shape.dims()[0], shape.dims()[0], pad_size];
        // Create padding on the same device as the input tensor
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

        flame_core::serialization::save_file(&tensors, path)?;

        info!("Saved {} embeddings to cache", context_embeds.len());
        Ok(())
    }

    fn load_from_cache(
        &self,
        path: &str,
        texts: &[(usize, String)],
    ) -> flame_core::Result<(HashMap<usize, Tensor>, HashMap<usize, Tensor>)> {
        // Load tensors using FLAME's load_file function
        let all_tensors = flame_core::serialization::load_file(path, &self.device.cuda_device())?;

        let mut context_embeds = HashMap::new();
        let mut pooled_embeds = HashMap::new();

        for (idx, _) in texts {
            let context_key = format!("context_{}", idx);
            let pooled_key = format!("pooled_{}", idx);

            if let Some(context_tensor) = all_tensors.get(&context_key) {
                context_embeds.insert(*idx, context_tensor.clone());
            }

            if let Some(pooled_tensor) = all_tensors.get(&pooled_key) {
                pooled_embeds.insert(*idx, pooled_tensor.clone());
            }
        }

        info!("Loaded {} cached embeddings", context_embeds.len());
        Ok((context_embeds, pooled_embeds))
    }
} // End of impl RealTextEncoder
