//! Flux model loading utilities using Candle

use eridiffusion_core::{Device, Result, Error, ModelArchitecture, TensorExt, FluxVariant, ModelOutput};
use eridiffusion_models::{DiffusionModel, TextEncoder, VAE};
use async_trait::async_trait;
use candle_core::{Tensor, DType, Module};
use candle_nn::VarBuilder;
use candle_transformers::models::{flux, t5, clip};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use tokenizers::Tokenizer;

/// Wrapper for T5 encoder to implement TextEncoder trait
pub struct T5TextEncoder {
    model: Mutex<t5::T5EncoderModel>,
    tokenizer: Tokenizer,
    device: Device,
    max_length: usize,
}

impl T5TextEncoder {
    pub fn new(
        model: t5::T5EncoderModel,
        tokenizer: Tokenizer,
        device: Device,
    ) -> Self {
        Self {
            model: Mutex::new(model),
            tokenizer,
            device,
            max_length: 256, // Flux uses 256 for T5
        }
    }
}

impl TextEncoder for T5TextEncoder {
    fn encode(&self, texts: &[String]) -> Result<(Tensor, Option<Tensor>)> {
        let device = to_candle_device(&self.device)?;
        let mut all_embeddings = Vec::new();
        
        for text in texts {
            // Tokenize
            let mut tokens = self.tokenizer
                .encode(text.as_str(), true)
                .map_err(|e| Error::Training(format!("Tokenization error: {}", e)))?
                .get_ids()
                .to_vec();
            
            // Pad or truncate to max_length
            tokens.resize(self.max_length, 0);
            
            // Convert to tensor
            let input_ids = Tensor::new(&tokens[..], &device)?.unsqueeze(0)?;
            
            // Forward pass
            let embeddings = self.model.lock().unwrap().forward(&input_ids)?;
            all_embeddings.push(embeddings);
        }
        
        // Stack all embeddings
        let stacked = Tensor::cat(&all_embeddings, 0)?;
        Ok((stacked, None))
    }
    
    fn get_hidden_size(&self) -> usize {
        4096 // T5-XXL hidden size
    }
    
    fn get_max_length(&self) -> usize {
        self.max_length
    }
}

/// Wrapper for CLIP encoder to implement TextEncoder trait
pub struct CLIPTextEncoder {
    model: clip::text_model::ClipTextTransformer,
    tokenizer: Tokenizer,
    device: Device,
    config: clip::text_model::ClipTextConfig,
}

impl CLIPTextEncoder {
    pub fn new(
        model: clip::text_model::ClipTextTransformer,
        tokenizer: Tokenizer,
        device: Device,
        config: clip::text_model::ClipTextConfig,
    ) -> Self {
        Self {
            model,
            tokenizer,
            device,
            config,
        }
    }
}

impl TextEncoder for CLIPTextEncoder {
    fn encode(&self, texts: &[String]) -> Result<(Tensor, Option<Tensor>)> {
        let device = to_candle_device(&self.device)?;
        let mut all_embeddings = Vec::new();
        let mut all_pooled = Vec::new();
        
        for text in texts {
            // Tokenize
            let tokens = self.tokenizer
                .encode(text.as_str(), true)
                .map_err(|e| Error::Training(format!("Tokenization error: {}", e)))?
                .get_ids()
                .to_vec();
            
            // Convert to tensor
            let input_ids = Tensor::new(&tokens[..], &device)?.unsqueeze(0)?;
            
            // Forward pass
            let embeddings = self.model.forward(&input_ids)?;
            
            // Get pooled output (last hidden state of [CLS] token)
            let pooled = embeddings.i((0, 0))?;
            
            all_embeddings.push(embeddings);
            all_pooled.push(pooled);
        }
        
        // Stack all embeddings
        let stacked_embeddings = Tensor::cat(&all_embeddings, 0)?;
        let stacked_pooled = Tensor::stack(&all_pooled, 0)?;
        
        Ok((stacked_embeddings, Some(stacked_pooled)))
    }
    
    fn get_hidden_size(&self) -> usize {
        self.config.embed_dim
    }
    
    fn get_max_length(&self) -> usize {
        self.config.max_position_embeddings
    }
}

/// Wrapper for Flux model to implement DiffusionModel trait
pub struct FluxDiffusionModel {
    model: flux::model::Flux,
    device: Device,
    metadata: eridiffusion_core::ModelMetadata,
}

impl FluxDiffusionModel {
    pub fn new(model: flux::model::Flux, device: Device) -> Self {
        let metadata = eridiffusion_core::ModelMetadata {
            architecture: ModelArchitecture::Flux,
            name: "flux-dev".to_string(),
            version: "1.0.0".to_string(),
            author: Some("Black Forest Labs".to_string()),
            description: Some("Flux Dev diffusion model".to_string()),
            license: Some("flux-1-dev-non-commercial-license".to_string()),
            tags: vec!["flux".to_string(), "diffusion".to_string()],
            created_at: chrono::Utc::now(),
            config: HashMap::new(),
        };
        Self { model, device, metadata }
    }
}

#[async_trait(?Send)]
impl DiffusionModel for FluxDiffusionModel {
    fn forward(&self, inputs: &eridiffusion_core::ModelInputs) -> Result<eridiffusion_core::ModelOutput> {
        // Extract inputs
        let noisy_latents = &inputs.latents;
        let timesteps = &inputs.timestep;
        let text_embeddings = inputs.encoder_hidden_states.as_ref()
            .ok_or_else(|| Error::Training("Text embeddings required for Flux".to_string()))?;
        let pooled_embeddings = inputs.pooled_projections.as_ref()
            .ok_or_else(|| Error::Training("Pooled embeddings required for Flux".to_string()))?;
        
        // Create Flux state
        let state = flux::sampling::State::new(text_embeddings, pooled_embeddings, noisy_latents)?;
        
        // Run model
        let guidance = inputs.guidance_scale.unwrap_or(3.5) as f64;
        let output = flux::sampling::denoise(
            &self.model,
            &state.img,
            &state.img_ids,
            &state.txt,
            &state.txt_ids,
            &state.vec,
            &[timesteps.to_scalar::<f32>()? as f64],
            guidance,
        )?;
        
        Ok(ModelOutput {
            sample: output,
            additional: HashMap::new(),
        })
    }
    
    fn architecture(&self) -> ModelArchitecture {
        ModelArchitecture::Flux
    }
    
    fn metadata(&self) -> &eridiffusion_core::ModelMetadata {
        &self.metadata
    }
    
    async fn load_pretrained(&mut self, _path: &Path) -> Result<()> {
        // Already loaded during construction
        Ok(())
    }
    
    async fn save_pretrained(&self, _path: &Path) -> Result<()> {
        // TODO: Implement model saving
        Ok(())
    }
    
    fn trainable_parameters(&self) -> Vec<&Tensor> {
        vec![]
    }
    
    fn parameters(&self) -> Vec<&Tensor> {
        vec![]
    }
    
    fn set_training(&mut self, _training: bool) {
        // TODO: Set training mode
    }
    
    fn device(&self) -> &Device {
        &self.device
    }
    
    fn to_device(&mut self, device: &Device) -> Result<()> {
        self.device = device.clone();
        Ok(())
    }
    
    fn memory_usage(&self) -> usize {
        12 * 1024 * 1024 * 1024 // 12GB estimate
    }
}

/// Wrapper for Flux VAE
pub struct FluxVAE {
    autoencoder: flux::autoencoder::AutoEncoder,
    device: Device,
}

impl FluxVAE {
    pub fn new(autoencoder: flux::autoencoder::AutoEncoder, device: Device) -> Self {
        Self { autoencoder, device }
    }
}

impl VAE for FluxVAE {
    fn encode(&self, images: &Tensor) -> Result<Tensor> {
        self.autoencoder.encode(images)
            .map_err(|e| Error::Model(format!("VAE encode error: {}", e)))
    }
    
    fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        self.autoencoder.decode(latents)
            .map_err(|e| Error::Model(format!("VAE decode error: {}", e)))
    }
    
    fn encode_deterministic(&self, images: &Tensor) -> Result<Tensor> {
        self.encode(images)
    }
    
    fn latent_channels(&self) -> usize {
        16 // Flux uses 16 latent channels
    }
    
    fn scaling_factor(&self) -> f64 {
        0.13025 // Flux VAE scaling factor
    }
}

/// Load T5 encoder from safetensors
pub async fn load_t5_encoder(
    model_path: &Path,
    tokenizer_path: &Path,
    device: &Device,
) -> Result<Box<dyn TextEncoder>> {
    let candle_device = to_candle_device(device)?;
    let dtype = candle_device.bf16_default_to_f32();
    
    // Load model weights
    let vb = unsafe { 
        VarBuilder::from_mmaped_safetensors(&[model_path], dtype, &candle_device)? 
    };
    
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
        tie_word_embeddings: true,
        is_decoder: false,
        is_encoder_decoder: true,
        use_cache: true,
        pad_token_id: 0,
        eos_token_id: 1,
        decoder_start_token_id: Some(0),
    };
    
    // Load model
    let model = t5::T5EncoderModel::load(vb, &config)?;
    
    // Load tokenizer
    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|e| Error::Training(format!("Failed to load T5 tokenizer: {}", e)))?;
    
    Ok(Box::new(T5TextEncoder::new(model, tokenizer, device.clone())))
}

/// Load CLIP encoder from safetensors
pub async fn load_clip_encoder(
    model_path: &Path,
    tokenizer_path: &Path,
    device: &Device,
) -> Result<Box<dyn TextEncoder>> {
    let candle_device = to_candle_device(device)?;
    let dtype = candle_device.bf16_default_to_f32();
    
    // Load model weights
    let vb = unsafe { 
        VarBuilder::from_mmaped_safetensors(&[model_path], dtype, &candle_device)? 
    };
    
    // CLIP-L config
    let config = clip::text_model::ClipTextConfig {
        vocab_size: 49408,
        projection_dim: 768,
        activation: clip::text_model::Activation::QuickGelu,
        intermediate_size: 3072,
        embed_dim: 768,
        max_position_embeddings: 77,
        pad_with: None,
        num_hidden_layers: 12,
        num_attention_heads: 12,
    };
    
    // Load model
    let model = clip::text_model::ClipTextTransformer::new(vb.pp("text_model"), &config)?;
    
    // Load tokenizer
    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|e| Error::Training(format!("Failed to load CLIP tokenizer: {}", e)))?;
    
    Ok(Box::new(CLIPTextEncoder::new(model, tokenizer, device.clone(), config)))
}

/// Load Flux model from safetensors
pub async fn load_flux_model(
    model_path: &Path,
    variant: FluxVariant,
    device: &Device,
) -> Result<Box<dyn DiffusionModel>> {
    let candle_device = to_candle_device(device)?;
    let dtype = candle_device.bf16_default_to_f32();
    
    // Load model weights
    let vb = unsafe { 
        VarBuilder::from_mmaped_safetensors(&[model_path], dtype, &candle_device)? 
    };
    
    // Get config based on variant
    let config = match variant {
        FluxVariant::Base => flux::model::Config::dev(), // Use dev config for base
        FluxVariant::Dev => flux::model::Config::dev(),
        FluxVariant::Schnell => flux::model::Config::schnell(),
    };
    
    // Load model
    let model = flux::model::Flux::new(&config, vb)?;
    
    Ok(Box::new(FluxDiffusionModel::new(model, device.clone())))
}

/// Load Flux VAE from safetensors
pub async fn load_flux_vae(
    vae_path: &Path,
    device: &Device,
) -> Result<Box<dyn VAE>> {
    let candle_device = to_candle_device(device)?;
    let dtype = DType::F32; // VAE typically uses F32
    
    // Load model weights
    let vb = unsafe { 
        VarBuilder::from_mmaped_safetensors(&[vae_path], dtype, &candle_device)? 
    };
    
    // Get config (same for dev and schnell)
    let config = flux::autoencoder::Config::dev();
    
    // Load model
    let autoencoder = flux::autoencoder::AutoEncoder::new(&config, vb)?;
    
    Ok(Box::new(FluxVAE::new(autoencoder, device.clone())))
}

/// Convert eridiffusion_core::Device to candle_core::Device
pub fn to_candle_device(device: &Device) -> Result<candle_core::Device> {
    device.to_candle()
}