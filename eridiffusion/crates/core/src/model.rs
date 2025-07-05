//! Core model traits and types

use crate::{Result, Device, DType};
use async_trait::async_trait;
use candle_core::{Tensor, Module};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Supported model architectures
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelArchitecture {
    /// Stable Diffusion 1.5
    SD15,
    /// Stable Diffusion 2 (deprecated)
    SD2,
    /// Stable Diffusion XL
    SDXL,
    /// Stable Diffusion 3
    SD3,
    /// Stable Diffusion 3.5
    SD35,
    /// Flux
    Flux,
    /// Flux Schnell
    FluxSchnell,
    /// Flux Dev
    FluxDev,
    /// PixArt-α
    PixArt,
    /// PixArt-Σ
    PixArtSigma,
    /// AuraFlow
    AuraFlow,
    /// KonText (Flux with ControlNet)
    KonText,
    /// HiDream
    HiDream,
    /// OmniGen 2
    OmniGen2,
    /// Flex 1
    Flex1,
    /// Flex 2
    Flex2,
    /// LTX Video
    LTX,
    /// Wan 2.1 Video
    Wan21,
    /// Hunyuan Video
    HunyuanVideo,
    /// Lumina
    Lumina,
    /// Chroma
    Chroma
}

/// Flux model variants
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FluxVariant {
    Base,
    Schnell,
    Dev,
}

impl std::fmt::Display for ModelArchitecture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SD15 => write!(f, "SD1.5"),
            Self::SD2 => write!(f, "SD2"),
            Self::SDXL => write!(f, "SDXL"),
            Self::SD3 => write!(f, "SD3"),
            Self::SD35 => write!(f, "SD3.5"),
            Self::Flux => write!(f, "Flux"),
            Self::FluxSchnell => write!(f, "Flux-Schnell"),
            Self::FluxDev => write!(f, "Flux-Dev"),
            Self::PixArt => write!(f, "PixArt-α"),
            Self::PixArtSigma => write!(f, "PixArt-Σ"),
            Self::AuraFlow => write!(f, "AuraFlow"),
            Self::KonText => write!(f, "KonText"),
            Self::HiDream => write!(f, "HiDream"),
            Self::OmniGen2 => write!(f, "OmniGen2"),
            Self::Flex1 => write!(f, "Flex1"),
            Self::Flex2 => write!(f, "Flex2"),
            Self::LTX => write!(f, "LTX"),
            Self::Wan21 => write!(f, "Wan2.1"),
            Self::HunyuanVideo => write!(f, "HunyuanVideo"),
            Self::Lumina => write!(f, "Lumina"),
            Self::Chroma => write!(f, "Chroma"),
        }
    }
}

impl ModelArchitecture {
    /// Get all supported architectures
    pub fn all() -> Vec<Self> {
        vec![
            Self::SD15,
            Self::SD2,
            Self::SDXL,
            Self::SD3,
            Self::SD35,
            Self::Flux,
            Self::FluxSchnell,
            Self::FluxDev,
            Self::PixArt,
            Self::PixArtSigma,
            Self::AuraFlow,
            Self::KonText,
            Self::HiDream,
            Self::OmniGen2,
            Self::Flex1,
            Self::Flex2,
            Self::LTX,
            Self::Wan21,
            Self::HunyuanVideo,
            Self::Lumina,
            Self::Chroma,
        ]
    }
    
    /// Get the default number of inference steps
    pub fn default_steps(&self) -> usize {
        match self {
            Self::FluxSchnell => 4,
            Self::FluxDev => 20,
            Self::SD35 => 28,
            _ => 50,
        }
    }
    
    /// Check if this architecture uses flow matching
    pub fn uses_flow_matching(&self) -> bool {
        matches!(self, 
            Self::SD3 | Self::SD35 | 
            Self::Flux | Self::FluxSchnell | Self::FluxDev |
            Self::AuraFlow | Self::LTX | Self::Lumina
        )
    }
    
    /// Get the latent channels for this architecture
    pub fn latent_channels(&self) -> usize {
        match self {
            Self::SD3 | Self::SD35 => 16,
            _ => 4,
        }
    }
}

/// Model inputs for diffusion models
#[derive(Debug, Clone)]
pub struct ModelInputs {
    /// Noisy latents
    pub latents: Tensor,
    /// Timestep
    pub timestep: Tensor,
    /// Text encoder hidden states
    pub encoder_hidden_states: Option<Tensor>,
    /// Pooled text projections (for SD3/SD3.5)
    pub pooled_projections: Option<Tensor>,
    /// Attention mask
    pub attention_mask: Option<Tensor>,
    /// Guidance scale (for models that need it)
    pub guidance_scale: Option<f32>,
    /// Additional inputs (architecture-specific)
    pub additional: HashMap<String, Tensor>,
}

impl Default for ModelInputs {
    fn default() -> Self {
        // This should not be used directly, but required for some patterns
        panic!("ModelInputs::default() should not be called - use proper constructor")
    }
}

/// Model output
#[derive(Debug)]
pub struct ModelOutput {
    /// Predicted noise or velocity
    pub sample: Tensor,
    /// Additional outputs (architecture-specific)
    pub additional: HashMap<String, Tensor>,
}

/// Base trait for all diffusion models
#[async_trait(?Send)]
pub trait DiffusionModel: Send + Sync {
    /// Get the model architecture
    fn architecture(&self) -> ModelArchitecture;
    
    /// Get model metadata
    fn metadata(&self) -> &ModelMetadata;
    
    /// Load pretrained weights
    async fn load_pretrained(&mut self, path: &Path) -> Result<()>;
    
    /// Save model weights
    async fn save_pretrained(&self, path: &Path) -> Result<()>;
    
    /// Forward pass
    fn forward(&self, inputs: &ModelInputs) -> Result<ModelOutput>;
    
    /// Forward pass with gradient checkpointing for memory efficiency
    fn forward_with_gradient_checkpointing(&self, inputs: &ModelInputs) -> Result<ModelOutput> {
        // Default implementation just calls forward
        // Models can override this for actual gradient checkpointing
        self.forward(inputs)
    }
    
    /// Get trainable parameters
    fn trainable_parameters(&self) -> Vec<&Tensor>;
    
    /// Get all parameters
    fn parameters(&self) -> Vec<&Tensor>;
    
    /// Set training mode
    fn set_training(&mut self, training: bool);
    
    /// Get device
    fn device(&self) -> &Device;
    
    /// Move to device
    fn to_device(&mut self, device: &Device) -> Result<()>;
    
    /// Get memory usage
    fn memory_usage(&self) -> usize;
}

/// Model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub name: String,
    pub architecture: ModelArchitecture,
    pub version: String,
    pub author: Option<String>,
    pub description: Option<String>,
    pub license: Option<String>,
    pub tags: Vec<String>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub config: HashMap<String, serde_json::Value>,
}

impl Default for ModelMetadata {
    fn default() -> Self {
        Self {
            name: "unnamed".to_string(),
            architecture: ModelArchitecture::SD15, // Default to SD1.5
            version: "1.0.0".to_string(),
            author: None,
            description: None,
            license: None,
            tags: vec![],
            created_at: chrono::Utc::now(),
            config: HashMap::new(),
        }
    }
}

/// Model loading configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelLoadConfig {
    pub device: Device,
    pub dtype: DType,
    pub use_safetensors: bool,
    pub load_tokenizers: bool,
    pub load_vae: bool,
    pub offload_device: Option<Device>,
}

impl Default for ModelLoadConfig {
    fn default() -> Self {
        Self {
            device: Device::cuda_if_available(0).unwrap_or(Device::Cpu),
            dtype: DType::F32,
            use_safetensors: true,
            load_tokenizers: true,
            load_vae: true,
            offload_device: None,
        }
    }
}