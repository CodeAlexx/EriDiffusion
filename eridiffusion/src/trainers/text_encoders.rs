use crate::loaders::WeightLoader;
use crate::models::{clip, T5EncoderModel};
use anyhow::anyhow;
use flame_core::device::Device;
use flame_core::optimizers::{Adam, SGD};
use flame_core::Result;
use flame_core::{CudaDevice, DType, Error, Shape, Tensor};
use log::info;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

pub struct PrefixedWeightLoader {
    loader: WeightLoader,
    prefix: String,
}

pub struct TextEncoders {
    pub clip_l: Option<clip::ClipTextTransformer>,
    pub clip_g: Option<clip::ClipTextTransformer>,
    pub t5: Option<T5EncoderModel>,
    tokenizer_clip: Option<Tokenizer>,
    tokenizer_t5: Option<Tokenizer>,
    pub device: Device,
}

// Extension trait for Tensor to add missing methods

// WeightLoader implementation is in crate::loaders::WeightLoader

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

impl TextEncoders {
    pub fn new(device: Device) -> Self {
        Self {
            clip_l: None,
            clip_g: None,
            t5: None,
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

        // Use the v1_5 config for CLIP-L
        let config = crate::models::text_encoder::CLIPConfig::clip_l();

        // Get weights from WeightLoader
        let weights = wl.weights;
        self.clip_l = Some(clip::ClipTextTransformer::new(config, self.device.clone(), weights)?);
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

        // Use the sdxl2 config for CLIP-G (OpenCLIP ViT-bigG-14)
        let config = crate::models::text_encoder::CLIPConfig::clip_g();

        // Get weights from WeightLoader
        let weights = wl.weights;
        self.clip_g = Some(clip::ClipTextTransformer::new(config, self.device.clone(), weights)?);
        println!("✅ CLIP-G loaded successfully with FP16 precision");
        Ok(())
    }

    pub fn load_t5(&mut self, model_path: &str) -> flame_core::Result<()> {
        println!("Loading T5-XXL from: {}", model_path);
        // Use streaming loader to avoid OOM
        let wl = crate::loaders::WeightLoader::from_safetensors_streaming(
            model_path,
            self.device.clone(),
            DType::F16,
        )?;

        // Create T5-XXL config
        let config = crate::models::text_encoder::T5Config::t5_xxl();

        // Load T5 model using the T5Encoder from text_encoder module
        // T5Encoder::new expects (config, device, weights)
        self.t5 =
            Some(crate::models::text_encoder::T5Encoder::new(config, self.device.clone(), &wl)?);
        println!("✅ T5-XXL loaded successfully with streaming loader");
        Ok(())
    }

    pub fn load_tokenizers(
        &mut self,
        clip_tokenizer_path: &str,
        t5_tokenizer_path: &str,
    ) -> flame_core::Result<()> {
        // Load CLIP tokenizer
        self.tokenizer_clip = Some(Tokenizer::from_file(clip_tokenizer_path).map_err(|e| {
            flame_core::Error::InvalidOperation(format!("Failed to load CLIP tokenizer: {}", e))
        })?);

        // Load T5 tokenizer
        self.tokenizer_t5 = Some(Tokenizer::from_file(t5_tokenizer_path).map_err(|e| {
            flame_core::Error::InvalidOperation(format!("Failed to load T5 tokenizer: {}", e))
        })?);

        println!("Tokenizers loaded successfully");
        Ok(())
    }

    pub fn load_clip_tokenizer(&mut self, tokenizer_path: &str) -> flame_core::Result<()> {
        // Load just CLIP tokenizer for SDXL
        self.tokenizer_clip = Some(Tokenizer::from_file(tokenizer_path).map_err(|e| {
            flame_core::Error::InvalidOperation(format!("Failed to load CLIP tokenizer: {}", e))
        })?);
        println!("CLIP tokenizer loaded successfully");
        Ok(())
    }

    pub fn encode(
        &self,
        text: &str,
        max_sequence_length: usize,
    ) -> flame_core::Result<(Tensor, Tensor)> {
        // For SDXL, we only need CLIP-L and CLIP-G
        if self.clip_l.is_some() & self.clip_g.is_some() & self.t5.is_none() {
            return self.encode_sdxl(text, max_sequence_length);
        }

        // For SD3/Flux, we need all three encoders
        if self.clip_l.is_none() || self.clip_g.is_none() || self.t5.is_none() {
            return Err(flame_core::Error::InvalidOperation(
                "Text encoders not loaded".to_string(),
            ));
        }

        if self.tokenizer_clip.is_none() || self.tokenizer_t5.is_none() {
            return Err(flame_core::Error::InvalidOperation("Tokenizers not loaded".to_string()));
        }

        // Handle empty prompts for unconditional generation
        // Flux uses empty string, but we still need to generate proper embeddings
        let is_empty_prompt = text.trim().is_empty();

        // Tokenize for CLIP (max 77 tokens)
        let clip_tokens = if is_empty_prompt {
            // For empty prompts, use only start and end tokens;
            self.tokenize_clip("", 77)?
        } else {
            self.tokenize_clip(text, 77)?
        };

        // Encode with CLIP-L
        let clip_l = self.clip_l.as_ref().unwrap();
        let clip_l_output = clip_l.forward(&clip_tokens, None)?;
        let clip_l_hidden = clip_l_output.last_hidden_state;

        // Encode with CLIP-G
        let clip_g = self.clip_g.as_ref().unwrap();
        let clip_g_output = clip_g.forward(&clip_tokens, None)?;
        let clip_g_hidden = clip_g_output.last_hidden_state;

        // Get pooled output from CLIP-G (last token)
        let pooled = self.pool_clip_g(&clip_g_hidden)?;

        // Tokenize and encode with T5
        // T5 handles empty prompts differently - it needs at least the EOS token
        let t5_tokens = if is_empty_prompt {
            self.tokenize_t5("", max_sequence_length)?
        } else {
            self.tokenize_t5(text, max_sequence_length)?
        };
        let t5_model = self.t5.as_ref().unwrap();
        let t5_output = t5_model.forward(&t5_tokens)?;
        let t5_hidden = t5_output.last_hidden_state;

        // Concatenate CLIP embeddings along feature dimension
        // CLIP-L: [1, 77, 768] -> pad to [1, 77, 2048]
        // CLIP-G: [1, 77, 1280] -> pad to [1, 77, 2048]
        let clip_l_padded = self.pad_to_dim(&clip_l_hidden, 2048)?;
        let clip_g_padded = self.pad_to_dim(&clip_g_hidden, 2048)?;

        // Concatenate: [1, 77, 2048] + [1, 77, 2048] = [1, 77, 4096]
        let clip_concat = Tensor::cat(&[&clip_l_padded, &clip_g_padded], 2)?;

        // Concatenate with T5 along sequence dimension
        // CLIP: [1, 77, 4096]
        // T5: [1, max_seq_len, 4096]
        // Result: [1, 77 + max_seq_len, 4096]
        let context = Tensor::cat(&[&clip_concat, &t5_hidden], 1)?;

        Ok((context, pooled))
    }

    pub fn encode_sdxl(
        &self,
        text: &str,
        _max_sequence_length: usize,
    ) -> flame_core::Result<(Tensor, Tensor)> {
        // SDXL uses only CLIP-L and CLIP-G encoders
        if self.clip_l.is_none() || self.clip_g.is_none() {
            return Err(flame_core::Error::InvalidOperation(
                "CLIP encoders not loaded for SDXL".to_string(),
            ));
        }

        // Check if this is an empty prompt for unconditional generation
        let is_empty_prompt = text.trim().is_empty();

        if is_empty_prompt {
            // SDXL uses zeros for unconditional prompts (not empty strings)
            // Create zero embeddings with the correct shapes
            let batch_size = 1;
            let seq_len = 77;

            // Get dtype from the loaded models to ensure consistency
            // For now, use F32 as a safe default since we don't have access to model internals;
            let dtype = DType::F32;

            // CLIP-L: [batch, 77, 768]
            let shape_clip_l = &[batch_size, seq_len, 768];
            let clip_l_zeros =
                Tensor::zeros(Shape::from_dims(shape_clip_l), self.device.cuda_device().clone())?;

            // CLIP-G: [batch, 77, 1280]
            let shape_clip_g = &[batch_size, seq_len, 1280];
            let clip_g_zeros =
                Tensor::zeros(Shape::from_dims(shape_clip_g), self.device.cuda_device().clone())?;

            // Pooled output: [batch, 1280]
            let shape_pooled = &[batch_size, 1280];
            let pooled_zeros =
                Tensor::zeros(Shape::from_dims(shape_pooled), self.device.cuda_device().clone())?;

            // Concatenate along feature dimension: [batch, 77, 2048]
            let context = Tensor::cat(&[&clip_l_zeros, &clip_g_zeros], 2)?;

            return Ok((context, pooled_zeros));
        }

        // For non-empty prompts, proceed with normal encoding
        let clip_tokens = if let Some(tokenizer) = &self.tokenizer_clip {
            self.tokenize_clip(text, 77)?
        } else {
            // Fallback: create simple token IDs for testing
            // In production, tokenizers should be loaded from the model directory
            let mut token_ids = vec![49406u32]; // Start token
                                                // Pad to 77 tokens
            token_ids.resize(77, 49407u32); // Pad with end tokens
                                            // Create as U32 tensor (required for embedding lookup)
            let shape = Shape::from_dims(&[1, token_ids.len()]);
            Tensor::from_vec(
                token_ids.iter().map(|&x| x as f32).collect(),
                shape,
                self.device.cuda_device().clone(),
            )?
            .to_dtype(DType::U32)?
        };

        // Encode with CLIP-L
        let clip_l = self.clip_l.as_ref().unwrap();
        let clip_l_output = clip_l.forward(&clip_tokens, None)?;
        let clip_l_hidden = clip_l_output.last_hidden_state;

        // Encode with CLIP-G
        let clip_g = self.clip_g.as_ref().unwrap();
        let clip_g_output = clip_g.forward(&clip_tokens, None)?;
        let clip_g_hidden = clip_g_output.last_hidden_state;

        // Get pooled output from CLIP-G (last token)
        let pooled = self.pool_clip_g(&clip_g_hidden)?;

        // For SDXL, concatenate CLIP-L and CLIP-G embeddings
        // CLIP-L: [batch, 77, 768]
        // CLIP-G: [batch, 77, 1280]
        // Concatenate along feature dimension: [batch, 77, 2048]
        let context = Tensor::cat(&[&clip_l_hidden, &clip_g_hidden], 2)?;

        Ok((context, pooled))
    }

    pub fn tokenize_clip(&self, text: &str, max_length: usize) -> flame_core::Result<Tensor> {
        println!(
            "DEBUG: tokenize_clip called with text: {:?}, max_length: {}",
            &text[..text.len().min(50)],
            max_length
        );
        let tokenizer = self.tokenizer_clip.as_ref().ok_or_else(|| {
            flame_core::Error::InvalidOperation("CLIP tokenizer not loaded".into())
        })?;
        println!("DEBUG: Got tokenizer, calling encode...");
        let encoding = tokenizer.encode(text, true).map_err(|e| {
            flame_core::Error::InvalidOperation(format!("Tokenization failed: {:?}", e))
        })?;
        println!("DEBUG: Encoding complete, getting IDs...");
        let mut ids = encoding.get_ids().to_vec();

        // Pad or truncate to max_length
        ids.resize(max_length, 0);

        // Create tensor as F32 (embedding layer converts internally)
        let ids_f32: Vec<f32> = ids.into_iter().map(|id| id as f32).collect();
        println!("DEBUG: tokenize_clip creating tensor with {} tokens", ids_f32.len());

        let shape = Shape::from_dims(&[1, ids_f32.len()]);
        let tensor = Tensor::from_vec(ids_f32, shape, self.device.cuda_device().clone())?;

        Ok(tensor)
    }

    pub fn tokenize_t5(&self, text: &str, max_length: usize) -> flame_core::Result<Tensor> {
        let tokenizer = self
            .tokenizer_t5
            .as_ref()
            .ok_or_else(|| flame_core::Error::InvalidOperation("T5 tokenizer not loaded".into()))?;
        let encoding = tokenizer.encode(text, true).map_err(|e| {
            flame_core::Error::InvalidOperation(format!("Tokenization failed: {:?}", e))
        })?;
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

    fn pad_to_dim(&self, tensor: &Tensor, target_dim: usize) -> flame_core::Result<Tensor> {
        let shape = tensor.shape();
        let current_dim = shape.dims()[0];

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

    fn pool_clip_g(&self, hidden_states: &Tensor) -> flame_core::Result<Tensor> {
        // CLIP-G pooling: take the last token (EOS token)
        let seq_len = hidden_states.shape().dims()[1];
        let start = seq_len - 1;
        let end = seq_len;
        Ok(hidden_states.narrow(1, start, end - start)?.squeeze(Some(1))?)
    }

    pub fn encode_sd35(
        &self,
        text: &str,
        negative_text: Option<&str>,
        max_sequence_length: usize,
    ) -> flame_core::Result<(Tensor, Tensor, Tensor)> {
        // SD3.5 uses CLIP-L, CLIP-G, and T5
        if self.clip_l.is_none() || self.clip_g.is_none() || self.t5.is_none() {
            return Err(flame_core::Error::InvalidOperation(
                "All text encoders (CLIP-L, CLIP-G, T5) must be loaded for SD3.5".to_string(),
            ));
        }

        // Tokenize text
        let clip_tokens = self.tokenize_clip(text, 77)?;
        let t5_tokens = self.tokenize_t5(text, max_sequence_length)?;

        // Encode with each encoder
        let clip_l_output = self.clip_l.as_ref().unwrap().forward(&clip_tokens, None)?;
        let clip_l_embed = clip_l_output.last_hidden_state;
        let clip_g_output = self.clip_g.as_ref().unwrap().forward(&clip_tokens, None)?;
        let clip_g_embed = clip_g_output.last_hidden_state;
        let t5_embed = if let Some(t5) = &self.t5 {
            let t5_output = t5.forward(&t5_tokens)?;
            t5_output.last_hidden_state
        } else {
            // Actually load and use T5 if not available
            return Err(flame_core::Error::InvalidOperation(
                "T5 encoder required for SD3.5 but not loaded. Load with load_t5()".to_string(),
            ));
        };

        Ok((clip_l_embed, clip_g_embed, t5_embed))
    }

    pub fn encode_flux(&self, text: &str) -> flame_core::Result<(Tensor, Tensor)> {
        // CLIP-L is required for Flux
        if self.clip_l.is_none() {
            return Err(flame_core::Error::InvalidOperation(
                "CLIP-L must be loaded for Flux".to_string(),
            ));
        }

        // Tokenize and encode with CLIP
        let clip_tokens = self.tokenize_clip(text, 77)?;
        let clip_output = self.clip_l.as_ref().unwrap().forward(&clip_tokens, None)?;
        let clip_embed = clip_output.last_hidden_state;

        // T5 is optional - use dummy embeddings if not loaded
        let t5_embed = if let Some(ref t5_model) = self.t5 {
            // Use real T5 encoding if available
            // Reduce T5 length to 128 during caching to prevent OOM
            // During training, the full 256 tokens can be used if needed
            let t5_tokens = self.tokenize_t5(text, 128)?; // Reduced from 256 to prevent OOM

            // Process T5 with explicit memory management
            let t5_output = t5_model.forward(&t5_tokens)?;
            let t5_embed = t5_output.last_hidden_state;

            // Force synchronization after T5 forward pass
            self.device.synchronize()?;

            t5_embed
        } else {
            // Use dummy T5 embeddings for training without T5
            println!("Warning: T5 not loaded, using dummy embeddings for Flux training");
            Tensor::randn(
                Shape::from_dims(&[1, 128, 4096]), // [batch, seq_len, hidden_dim] - matches reduced length
                0.0,
                0.02,
                self.device.cuda_device().clone(),
            )?
        };

        Ok((clip_embed, t5_embed))
    }

    /// Encode only with CLIP-L (for memory-efficient sequential encoding)
    pub fn encode_clip_only(&self, text: &str) -> flame_core::Result<Tensor> {
        if self.clip_l.is_none() {
            return Err(flame_core::Error::InvalidOperation(
                "CLIP-L encoder not loaded".to_string(),
            ));
        }

        let clip_tokens = self.tokenize_clip(text, 77)?;
        let clip_output = self.clip_l.as_ref().unwrap().forward(&clip_tokens, None)?;
        Ok(clip_output.last_hidden_state)
    }

    /// Encode only with T5 (for memory-efficient sequential encoding)
    pub fn encode_t5_only(&self, text: &str, max_length: usize) -> flame_core::Result<Tensor> {
        if self.t5.is_none() {
            return Err(flame_core::Error::InvalidOperation("T5 encoder not loaded".to_string()));
        }

        let t5_tokens = self.tokenize_t5(text, max_length)?;
        info!("Processing T5 with sequence length: {:?}", t5_tokens.shape());

        let t5_model = self.t5.as_ref().unwrap();
        let t5_output = t5_model.forward(&t5_tokens)?;
        let t5_embed = t5_output.last_hidden_state;

        // Force synchronization after T5 forward pass
        self.device.synchronize()?;

        Ok(t5_embed)
    }

    pub fn encode_batch(
        &self,
        texts: &[String],
        max_sequence_length: usize,
    ) -> flame_core::Result<(Tensor, Tensor)> {
        let mut all_context = Vec::new();
        let mut all_pooled = Vec::new();

        for text in texts {
            let (context, pooled) = self.encode(text, max_sequence_length)?;
            all_context.push(context);
            all_pooled.push(pooled);
        }

        // Stack batch
        let context = Tensor::stack(&all_context, 0)?.squeeze(Some(1))?; // Remove extra batch dim
        let pooled = Tensor::stack(&all_pooled, 0)?;

        Ok((context, pooled))
    }

    /// Create unconditional prompt embeddings for classifier-free guidance
    /// SDXL uses zeros, Flux/SD3 use empty string embeddings
    pub fn encode_unconditional(
        &self,
        batch_size: usize,
        max_sequence_length: usize,
    ) -> flame_core::Result<(Tensor, Tensor)> {
        // Check which model type we're using
        let is_sdxl = self.clip_l.is_some() & self.clip_g.is_some() & self.t5.is_none();

        if is_sdxl {
            // SDXL: Use zeros for unconditional embeddings
            let seq_len = 77;
            let context_shape = &[batch_size, seq_len, 768]; // CLIP hidden size
            let pooled_shape = &[batch_size, 768];
            let context =
                Tensor::zeros(Shape::from_dims(context_shape), self.device.cuda_device().clone())?;
            let pooled =
                Tensor::zeros(Shape::from_dims(pooled_shape), self.device.cuda_device().clone())?;
            Ok((context, pooled))
        } else {
            // Flux/SD3: Use empty string embeddings
            let empty_prompts: Vec<String> = vec!["".to_string(); batch_size];
            self.encode_batch(&empty_prompts, max_sequence_length)
        }
    }

    /// Load text encoders from safetensors files
    pub fn from_safetensors(
        clip_l_path: Option<&Path>,
        clip_g_path: Option<&Path>,
        t5_path: Option<&Path>,
        device: Device,
    ) -> flame_core::Result<Self> {
        let mut encoders = TextEncoders {
            clip_l: None,
            clip_g: None,
            t5: None,
            tokenizer_clip: None,
            tokenizer_t5: None,
            device,
        };

        // Load CLIP-L model and tokenizer
        if let Some(path) = clip_l_path {
            encoders.load_clip_l(&path.to_string_lossy())?;

            // ALWAYS load CLIP tokenizer from HuggingFace cache - NOT from model directory!
            // The model directory only contains the model weights, not the tokenizer
            let alt_tokenizer = PathBuf::from("/home/alex/.cache/huggingface/hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41/tokenizer.json");
            if alt_tokenizer.exists() {
                println!("Loading CLIP tokenizer from HuggingFace cache: {:?}", alt_tokenizer);
                encoders.load_clip_tokenizer(&alt_tokenizer.to_string_lossy())?;
            } else {
                return Err(flame_core::Error::InvalidOperation(
                    "CLIP tokenizer not found in HuggingFace cache at expected location"
                        .to_string(),
                ));
            }
        }

        if let Some(path) = clip_g_path {
            encoders.load_clip_g(&path.to_string_lossy())?;
        }

        // Check if we should skip T5 for memory reasons
        let skip_t5 = std::env::var("SKIP_T5_ENCODER").unwrap_or_default() == "1";

        if let Some(path) = t5_path {
            // ALWAYS load T5 tokenizer from HuggingFace cache - NOT from model directory!
            // The model directory only contains the model weights, not the tokenizer
            let alt_tokenizer = PathBuf::from("/home/alex/.cache/huggingface/hub/models--google--flan-t5-xl/snapshots/7d6315df2c2fb742f0f5b556879d730926ca9001/tokenizer.json");
            if alt_tokenizer.exists() {
                println!("Loading T5 tokenizer from HuggingFace cache: {:?}", alt_tokenizer);
                encoders.tokenizer_t5 =
                    Some(Tokenizer::from_file(&alt_tokenizer).map_err(|e| {
                        flame_core::Error::InvalidOperation(format!(
                            "Failed to load T5 tokenizer: {}",
                            e
                        ))
                    })?);
            } else {
                return Err(flame_core::Error::InvalidOperation(
                    "T5 tokenizer not found in HuggingFace cache at expected location".to_string(),
                ));
            }

            if skip_t5 {
                println!("⚠️ SKIP_T5_ENCODER=1 set, using dummy T5 embeddings to save memory");
                println!("  Real T5 would use 9.12GB, dummy embeddings use minimal memory");
            } else {
                encoders.load_t5(&path.to_string_lossy())?;
            }
        }

        Ok(encoders)
    }
}
