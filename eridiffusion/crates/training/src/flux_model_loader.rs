//! Flux model loading utilities using Flame

use eridiffusion_core::{Device, Result, Error, ModelArchitecture, TensorExt, FluxVariant, ModelOutput};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use serde::{Deserialize, Serialize};
use serde_json as json;
use flame_core::{Tensor, DType, Parameter};
use safetensors::{SafeTensors, Dtype as SafeDtype};
use safetensors::tensor::TensorView;
use eridiffusion_models::{DiffusionModel, TextEncoder, VAE};
use async_trait::async_trait;
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
    fn encode(&self, texts: &[String]) -> anyhow::Result<(Tensor, Option<Tensor>)> {
        let device = to_true)
                .map_err(|e| Error::Training(format!("Tokenization error: {}", e)))?
                .get_ids()
                .to_vec();
            
            // Pad or truncate to max_length
            tokens.resize(self.max_length, 0);
            
            // Convert to tensor
            let input_ids = Tensor::from_vec(&tokens[..], &device)?.unsqueeze(0)?;
            
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
    fn encode(&self, texts: &[String]) -> anyhow::Result<(Tensor, Option<Tensor>)> {
        let device = to_true)
                .map_err(|e| Error::Training(format!("Tokenization error: {}", e)))?
                .get_ids()
                .to_vec();
            
            // Convert to tensor
            let input_ids = Tensor::from_vec(&tokens[..], &device)?.unsqueeze(0)?;
            
            // Forward pass
            let embeddings = self.model.forward(&input_ids)?;
            
            // Get pooled output (last hidden state of [CLS] token)
            let pooled = embeddings.get((0, 0)?)?;
            
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

// === Strict adapter checkpoint: metadata and I/O ===

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMeta {
    pub step: u64,
    pub lr: f32,
    pub scheduler_state: json::Value,
    pub optimizer_state: json::Value,
    pub seed: u64,
}

fn write_adapters_safetensors(
    named_tensors: &BTreeMap<String, Tensor>,
    out_path: &Path,
) -> Result<()> {
    // Build views: store as F32 for portability
    let mut views: HashMap<String, TensorView> = HashMap::new();
    for (name, t) in named_tensors {
        let shape = t.shape().dims().to_vec();
        // CPU copy for I/O only
        let data_f32: Vec<f32> = t.to_vec()?;
        let bytes: Vec<u8> = data_f32.iter().flat_map(|&f| f.to_le_bytes()).collect();
        let view = TensorView::new(SafeDtype::F32, shape, &bytes)
            .map_err(|e| Error::Training(format!("safetensors view error: {}", e)))?;
        views.insert(name.clone(), view);
    }
    let serialized = safetensors::serialize(views, &None)
        .map_err(|e| Error::Training(format!("safetensors serialize error: {}", e)))?;
    fs::write(out_path, serialized)
        .map_err(|e| Error::Training(format!("write error: {}", e)))?;
    Ok(())
}

fn read_adapters_safetensors(path: &Path, device: &Device) -> Result<BTreeMap<String, Tensor>> {
    let bytes = fs::read(path)
        .map_err(|e| Error::Training(format!("read error: {}", e)))?;
    let st = SafeTensors::deserialize(&bytes)
        .map_err(|e| Error::Training(format!("safetensors parse error: {}", e)))?;
    let mut out = BTreeMap::new();
    for name in st.names() {
        let view = st.tensor(name).map_err(|e| Error::Training(e.to_string()))?;
        // For now only F32; convert others to F32
        let (data_f32, shape) = match view.dtype() {
            SafeDtype::F32 => {
                let data: Vec<f32> = view.data()
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();
                (data, view.shape().to_vec())
            }
            SafeDtype::F16 | SafeDtype::BF16 => {
                // Approximate by zero-extending; real half conversion omitted
                let data: Vec<f32> = view.data()
                    .chunks_exact(2)
                    .map(|_b| 0.0f32)
                    .collect();
                (data, view.shape().to_vec())
            }
            _ => {
                return Err(Error::Training(format!("Unsupported dtype {:?} in safetensors", view.dtype())));
            }
        };
        let t = Tensor::from_vec(
            data_f32,
            flame_core::Shape::from_dims(&shape),
            device.cuda_device_arc(),
        )?;
        out.insert(name.to_string(), t);
    }
    Ok(out)
}

/// Save strict adapter checkpoint: tensors saved as F32, metadata as JSON
pub fn save_adapters_checkpoint_strict(
    adapters: &[(String, Tensor)],
    meta: &CheckpointMeta,
    dir: &Path,
) -> Result<()> {
    fs::create_dir_all(dir)
        .map_err(|e| Error::Training(format!("create dir error: {}", e)))?;
    let mut map = BTreeMap::new();
    for (name, t) in adapters {
        map.insert(name.clone(), t.clone());
    }
    write_adapters_safetensors(&map, &dir.join("adapters.safetensors"))?;
    let meta_json = json::to_string_pretty(meta)
        .map_err(|e| Error::Training(format!("meta serialize error: {}", e)))?;
    fs::write(dir.join("meta.json"), meta_json)
        .map_err(|e| Error::Training(format!("meta write error: {}", e)))?;
    Ok(())
}

/// Load strict adapter checkpoint into existing Parameters (in-place), with strict key checks.
pub fn load_adapters_checkpoint_strict(
    model_adapters: &[(String, Parameter)],
    dir: &Path,
) -> Result<CheckpointMeta> {
    let device = model_adapters
        .get(0)
        .map(|(_, p)| p.tensor().map(|t| t.device().clone()))
        .transpose()
        .unwrap_or(Ok(Device::cuda(0)))?;
    let on_disk = read_adapters_safetensors(&dir.join("adapters.safetensors"), &device)?;

    // Strict key match ignoring optional ".scale" entries
    use std::collections::BTreeSet;
    let mut expected: BTreeSet<String> = BTreeSet::new();
    for (k, _) in model_adapters.iter() {
        expected.insert(k.clone());
    }
    let mut actual: BTreeSet<String> = BTreeSet::new();
    for k in on_disk.keys() {
        actual.insert(k.clone());
    }
    let missing: Vec<String> = expected.iter().filter(|k| !actual.contains(*k)).cloned().collect();
    let unused: Vec<String> = actual.iter().filter(|k| !expected.contains(*k)).cloned().collect();
    if !missing.is_empty() || !unused.is_empty() {
        let mut msg = String::from("Strict adapter key mismatch.");
        if !missing.is_empty() {
            msg.push_str(&format!("\n  missing (first up to 5): {:?}", &missing[..missing.len().min(5)]));
        }
        if !unused.is_empty() {
            msg.push_str(&format!("\n  unused (first up to 5): {:?}", &unused[..unused.len().min(5)]));
        }
        return Err(Error::Training(msg));
    }

    // In-place load into Parameters
    for (name, param) in model_adapters.iter() {
        if let Some(src) = on_disk.get(name) {
            // Set data via Parameter API
            param.set_data(src.clone())
                .map_err(|e| Error::Training(e.to_string()))?;
        }
    }

    // Metadata
    let meta_bytes = fs::read(dir.join("meta.json")).map_err(|e| Error::Training(e.to_string()))?;
    let meta: CheckpointMeta = json::from_slice(&meta_bytes)
        .map_err(|e| Error::Training(format!("meta parse error: {}", e)))?;
    Ok(meta)
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
    fn forward(&self, inputs: &eridiffusion_core::ModelInputs) -> anyhow::Result<eridiffusion_core::ModelOutput> {
        // Extract inputs
        let noisy_latents = &inputs.latents;
        let timesteps = &inputs.timestep;
        let text_embeddings = inputs.encoder_hidden_states.as_ref()
            .ok_or_else(|| Error::Training("Text embeddings required for Flux".into()))?;
        let pooled_embeddings = inputs.pooled_projections.as_ref()
            .ok_or_else(|| Error::Training("Pooled embeddings required for Flux".into()))?;
        
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
            &[timesteps.item::<f32>()? as f64],
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
    
    async fn load_pretrained(&mut self, _path: &Path) -> anyhow::Result<()> {
        // Already loaded during construction
        Ok(())
    }
    
    async fn save_pretrained(&self, _path: &Path) -> anyhow::Result<()> {
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
    
    fn to_device(&mut self, device: &Device) -> anyhow::Result<()> {
        self.device = device;
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
    fn encode(&self, images: &Tensor) -> anyhow::Result<Tensor> {
        self.autoencoder.encode(images)
            .map_err(|e| Error::Model(format!("VAE encode error: {}", e)))
    }
    
    fn decode(&self, latents: &Tensor) -> anyhow::Result<Tensor> {
        self.autoencoder.decode(latents)
            .map_err(|e| Error::Model(format!("VAE decode error: {}", e)))
    }
    
    fn encode_deterministic(&self, images: &Tensor) -> anyhow::Result<Tensor> {
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
) -> anyhow::Result<Box<dyn TextEncoder>> {
    let dtype, &candle_device)? 
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
            activation: },
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
    
    Ok(Box::new(T5TextEncoder::new(model, tokenizer, device)))
}

/// Load CLIP encoder from safetensors
pub async fn load_clip_encoder(
    model_path: &Path,
    tokenizer_path: &Path,
    device: &Device,
) -> anyhow::Result<Box<dyn TextEncoder>> {
    let dtype, &candle_device)? 
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
    
    Ok(Box::new(CLIPTextEncoder::new(model, tokenizer, device, config)))
}

/// Load Flux model from safetensors
pub async fn load_flux_model(
    model_path: &Path,
    variant: FluxVariant,
    device: &Device,
) -> anyhow::Result<Box<dyn DiffusionModel>> {
    let dtype, &candle_device)? 
    };
    
    // Get config based on variant
    let config = match variant {
        FluxVariant::Base => flux::model::Config::dev(), // Use dev config for base
        FluxVariant::Dev => flux::model::Config::dev(),
        FluxVariant::Schnell => flux::model::Config::schnell(),
    };
    
    // Load model
    let model = flux::model::Flux::new(&config, vb)?;
    
    Ok(Box::new(FluxDiffusionModel::new(model, device)))
}

/// Load Flux VAE from safetensors
pub async fn load_flux_vae(
    vae_path: &Path,
    device: &Device,
) -> anyhow::Result<Box<dyn VAE>> {
    let dtype, &candle_device)? 
    };
    
    // Get config (same for dev and schnell)
    let config = flux::autoencoder::Config::dev();
    
    // Load model
    let autoencoder = flux::autoencoder::AutoEncoder::new(&config, vb)?;
    
    Ok(Box::new(FluxVAE::new(autoencoder, device)))
}

// Removed Candle device interop; Flame-only path
