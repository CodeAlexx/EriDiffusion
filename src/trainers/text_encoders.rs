use anyhow::Result;
use candle_core::{DType, Device, Tensor, Module};
use candle_nn::VarBuilder;
use candle_transformers::models::stable_diffusion::clip;
use candle_transformers::models::t5;
use tokenizers::Tokenizer;

pub struct TextEncoders {
    pub clip_l: Option<clip::ClipTextTransformer>,
    pub clip_g: Option<clip::ClipTextTransformer>,
    pub t5: Option<t5::T5EncoderModel>,
    tokenizer_clip: Option<Tokenizer>,
    tokenizer_t5: Option<Tokenizer>,
    pub device: Device,
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

    pub fn load_clip_l(&mut self, model_path: &str) -> Result<()> {
        println!("Loading CLIP-L from: {}", model_path);
        let tensors = candle_core::safetensors::load(model_path, &self.device)?;
        let vb = VarBuilder::from_tensors(tensors, DType::BF16, &self.device);
        
        // Use the v1_5 config for CLIP-L
        let config = clip::Config::v1_5();
        
        self.clip_l = Some(clip::ClipTextTransformer::new(vb, &config)?);
        println!("CLIP-L loaded successfully");
        Ok(())
    }

    pub fn load_clip_g(&mut self, model_path: &str) -> Result<()> {
        println!("Loading CLIP-G from: {}", model_path);
        let tensors = candle_core::safetensors::load(model_path, &self.device)?;
        let vb = VarBuilder::from_tensors(tensors, DType::BF16, &self.device);
        
        // Use the sdxl2 config for CLIP-G (OpenCLIP ViT-bigG-14)
        let config = clip::Config::sdxl2();
        
        self.clip_g = Some(clip::ClipTextTransformer::new(vb, &config)?);
        println!("CLIP-G loaded successfully");
        Ok(())
    }

    pub fn load_t5(&mut self, model_path: &str) -> Result<()> {
        println!("Loading T5-XXL from: {}", model_path);
        let tensors = candle_core::safetensors::load(model_path, &self.device)?;
        let vb = VarBuilder::from_tensors(tensors, DType::BF16, &self.device);
        
        // Create T5-XXL config
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
        
        self.t5 = Some(t5::T5EncoderModel::load(vb, &config)?);
        println!("T5-XXL loaded successfully");
        Ok(())
    }

    pub fn load_tokenizers(&mut self, clip_tokenizer_path: &str, t5_tokenizer_path: &str) -> Result<()> {
        // Load CLIP tokenizer
        self.tokenizer_clip = Some(Tokenizer::from_file(clip_tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load CLIP tokenizer: {}", e))?);
        
        // Load T5 tokenizer
        self.tokenizer_t5 = Some(Tokenizer::from_file(t5_tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load T5 tokenizer: {}", e))?);
        
        println!("Tokenizers loaded successfully");
        Ok(())
    }
    
    pub fn load_clip_tokenizer(&mut self, tokenizer_path: &str) -> Result<()> {
        // Load just CLIP tokenizer for SDXL
        self.tokenizer_clip = Some(Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load CLIP tokenizer: {}", e))?);
        println!("CLIP tokenizer loaded successfully");
        Ok(())
    }

    pub fn encode(&mut self, text: &str, max_sequence_length: usize) -> Result<(Tensor, Tensor)> {
        // For SDXL, we only need CLIP-L and CLIP-G
        if self.clip_l.is_some() && self.clip_g.is_some() && self.t5.is_none() {
            return self.encode_sdxl(text, max_sequence_length);
        }
        
        // For SD3/Flux, we need all three encoders
        if self.clip_l.is_none() || self.clip_g.is_none() || self.t5.is_none() {
            return Err(anyhow::anyhow!("Text encoders not loaded"));
        }
        
        if self.tokenizer_clip.is_none() || self.tokenizer_t5.is_none() {
            return Err(anyhow::anyhow!("Tokenizers not loaded"));
        }
        
        // Handle empty prompts for unconditional generation
        // Flux uses empty string, but we still need to generate proper embeddings
        let is_empty_prompt = text.trim().is_empty();
        
        // Tokenize for CLIP (max 77 tokens)
        let clip_tokens = if is_empty_prompt {
            // For empty prompts, use only start and end tokens
            self.tokenize_clip("", 77)?
        } else {
            self.tokenize_clip(text, 77)?
        };
        
        // Encode with CLIP-L
        let clip_l = self.clip_l.as_ref().unwrap();
        let clip_l_hidden = clip_l.forward(&clip_tokens)?;
        
        // Encode with CLIP-G
        let clip_g = self.clip_g.as_ref().unwrap();
        let clip_g_hidden = clip_g.forward(&clip_tokens)?;
        
        // Get pooled output from CLIP-G (last token)
        let pooled = self.pool_clip_g(&clip_g_hidden)?;
        
        // Tokenize and encode with T5
        // T5 handles empty prompts differently - it needs at least the EOS token
        let t5_tokens = if is_empty_prompt {
            self.tokenize_t5("", max_sequence_length)?
        } else {
            self.tokenize_t5(text, max_sequence_length)?
        };
        let t5_model = self.t5.as_mut().unwrap();
        let t5_hidden = t5_model.forward(&t5_tokens)?;
        
        // Concatenate CLIP embeddings along feature dimension
        // CLIP-L: [1, 77, 768] -> pad to [1, 77, 2048]
        // CLIP-G: [1, 77, 1280] -> pad to [1, 77, 2048]
        let clip_l_padded = self.pad_to_dim(&clip_l_hidden, 2048)?;
        let clip_g_padded = self.pad_to_dim(&clip_g_hidden, 2048)?;
        
        // Concatenate: [1, 77, 2048] + [1, 77, 2048] = [1, 77, 4096]
        let clip_concat = Tensor::cat(&[clip_l_padded, clip_g_padded], 2)?;
        
        // Concatenate with T5 along sequence dimension
        // CLIP: [1, 77, 4096]
        // T5: [1, max_seq_len, 4096]
        // Result: [1, 77 + max_seq_len, 4096]
        let context = Tensor::cat(&[clip_concat, t5_hidden], 1)?;
        
        Ok((context, pooled))
    }
    
    pub fn encode_sdxl(&mut self, text: &str, _max_sequence_length: usize) -> Result<(Tensor, Tensor)> {
        // SDXL uses only CLIP-L and CLIP-G encoders
        if self.clip_l.is_none() || self.clip_g.is_none() {
            return Err(anyhow::anyhow!("CLIP encoders not loaded for SDXL"));
        }
        
        // Check if this is an empty prompt for unconditional generation
        let is_empty_prompt = text.trim().is_empty();
        
        if is_empty_prompt {
            // SDXL uses zeros for unconditional prompts (not empty strings)
            // Create zero embeddings with the correct shapes
            let batch_size = 1;
            let seq_len = 77;
            
            // CLIP-L: [batch, 77, 768]
            let clip_l_zeros = Tensor::zeros(&[batch_size, seq_len, 768], DType::F32, &self.device)?;
            
            // CLIP-G: [batch, 77, 1280]
            let clip_g_zeros = Tensor::zeros(&[batch_size, seq_len, 1280], DType::F32, &self.device)?;
            
            // Pooled output: [batch, 1280]
            let pooled_zeros = Tensor::zeros(&[batch_size, 1280], DType::F32, &self.device)?;
            
            // Concatenate along feature dimension: [batch, 77, 2048]
            let context = Tensor::cat(&[clip_l_zeros, clip_g_zeros], 2)?;
            
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
            Tensor::new(token_ids.as_slice(), &self.device)?.unsqueeze(0)?
        };
        
        // Encode with CLIP-L
        let clip_l = self.clip_l.as_ref().unwrap();
        println!("DEBUG: CLIP tokens shape before CLIP-L: {:?}", clip_tokens.dims());
        let clip_l_hidden = clip_l.forward(&clip_tokens)?;
        println!("DEBUG: CLIP-L output shape: {:?}", clip_l_hidden.dims());
        
        // Encode with CLIP-G  
        let clip_g = self.clip_g.as_ref().unwrap();
        println!("DEBUG: CLIP tokens shape before CLIP-G: {:?}", clip_tokens.dims());
        let clip_g_hidden = clip_g.forward(&clip_tokens)?;
        println!("DEBUG: CLIP-G output shape: {:?}", clip_g_hidden.dims());
        
        // Get pooled output from CLIP-G (last token)
        let pooled = self.pool_clip_g(&clip_g_hidden)?;
        println!("DEBUG: Pooled output shape: {:?}", pooled.dims());
        
        // For SDXL, concatenate CLIP-L and CLIP-G embeddings
        // CLIP-L: [batch, 77, 768]
        // CLIP-G: [batch, 77, 1280]
        // Concatenate along feature dimension: [batch, 77, 2048]
        let context = Tensor::cat(&[clip_l_hidden, clip_g_hidden], 2)?;
        println!("DEBUG: Concatenated context shape: {:?}", context.dims());
        
        Ok((context, pooled))
    }

    fn tokenize_clip(&self, text: &str, max_length: usize) -> Result<Tensor> {
        let tokenizer = self.tokenizer_clip.as_ref().unwrap();
        let encoding = tokenizer.encode(text, true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {:?}", e))?;
        let mut ids = encoding.get_ids().to_vec();
        
        // Pad or truncate to max_length
        ids.resize(max_length, 0);
        
        let ids_u32: Vec<u32> = ids.into_iter().map(|id| id as u32).collect();
        Ok(Tensor::new(ids_u32.as_slice(), &self.device)?.unsqueeze(0)?)
    }

    fn tokenize_t5(&self, text: &str, max_length: usize) -> Result<Tensor> {
        let tokenizer = self.tokenizer_t5.as_ref().unwrap();
        let encoding = tokenizer.encode(text, true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {:?}", e))?;
        let mut ids = encoding.get_ids().to_vec();
        
        // T5 can handle longer sequences
        if ids.len() > max_length {
            ids.truncate(max_length);
        }
        
        let ids_u32: Vec<u32> = ids.into_iter().map(|id| id as u32).collect();
        Ok(Tensor::new(ids_u32.as_slice(), &self.device)?.unsqueeze(0)?)
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

    fn pool_clip_g(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // CLIP-G pooling: take the last token (EOS token)
        let seq_len = hidden_states.dim(1)?;
        Ok(hidden_states.narrow(1, seq_len - 1, 1)?.squeeze(1)?)
    }

    pub fn encode_batch(&mut self, texts: &[String], max_sequence_length: usize) -> Result<(Tensor, Tensor)> {
        let mut all_context = Vec::new();
        let mut all_pooled = Vec::new();
        
        for text in texts {
            let (context, pooled) = self.encode(text, max_sequence_length)?;
            all_context.push(context);
            all_pooled.push(pooled);
        }
        
        // Stack batch
        let context = Tensor::stack(&all_context, 0)?.squeeze(1)?; // Remove extra batch dim
        let pooled = Tensor::stack(&all_pooled, 0)?;
        
        Ok((context, pooled))
    }
    
    /// Create unconditional prompt embeddings for classifier-free guidance
    /// SDXL uses zeros, Flux/SD3 use empty string embeddings
    pub fn encode_unconditional(&mut self, batch_size: usize, max_sequence_length: usize) -> Result<(Tensor, Tensor)> {
        // Check which model type we're using
        let is_sdxl = self.clip_l.is_some() && self.clip_g.is_some() && self.t5.is_none();
        
        if is_sdxl {
            // SDXL: Use zeros for unconditional embeddings
            let seq_len = 77;
            let context = Tensor::zeros(&[batch_size, seq_len, 2048], DType::F32, &self.device)?;
            let pooled = Tensor::zeros(&[batch_size, 1280], DType::F32, &self.device)?;
            Ok((context, pooled))
        } else {
            // Flux/SD3: Use empty string embeddings
            let empty_prompts: Vec<String> = vec!["".to_string(); batch_size];
            self.encode_batch(&empty_prompts, max_sequence_length)
        }
    }
}