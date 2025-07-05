//! Static dispatch alternatives to reduce dynamic dispatch overhead

use crate::{Result, Error, ModelArchitecture};
use crate::model::{DiffusionModel, ModelInputs, ModelOutput};
use crate::network::{NetworkAdapter, NetworkOutput};
use candle_core::Tensor;
use std::collections::HashMap;

/// Enum-based model dispatch for known architectures
#[derive(Clone)]
pub enum ModelDispatch {
    SD15(Box<eridiffusion_models::SD15Model>),
    SDXL(Box<eridiffusion_models::SDXLModel>),
    SD3(Box<eridiffusion_models::SD3Model>),
    Flux(Box<eridiffusion_models::FluxModel>),
    // Dynamic fallback for plugins
    Dynamic(Box<dyn DiffusionModel + Send + Sync>),
}

impl ModelDispatch {
    /// Create from architecture
    pub fn from_architecture(arch: ModelArchitecture, device: crate::Device) -> Result<Self> {
        match arch {
            ModelArchitecture::SD15 => Ok(Self::SD15(Box::new(
                eridiffusion_models::SD15Model::new(device)?
            ))),
            ModelArchitecture::SDXL => Ok(Self::SDXL(Box::new(
                eridiffusion_models::SDXLModel::new(device)?
            ))),
            ModelArchitecture::SD3 => Ok(Self::SD3(Box::new(
                eridiffusion_models::SD3Model::new(device)?
            ))),
            ModelArchitecture::SD35 => Ok(Self::SD3(Box::new(
                eridiffusion_models::SD3Model::new(device)?
            ))),
            ModelArchitecture::Flux(variant) => Ok(Self::Flux(Box::new(
                eridiffusion_models::FluxModel::new(variant, device)?
            ))),
            ModelArchitecture::PixArt => Ok(Self::PixArt(Box::new(
                eridiffusion_models::PixArtModel::new(device)?
            ))),
            ModelArchitecture::AuraFlow => Ok(Self::AuraFlow(Box::new(
                eridiffusion_models::AuraFlowModel::new(device)?
            ))),
            _ => Err(Error::Unsupported(format!("Architecture {:?} not supported in static dispatch", arch))),
        }
    }
}

impl DiffusionModel for ModelDispatch {
    fn forward(&self, inputs: &ModelInputs) -> Result<ModelOutput> {
        match self {
            Self::SD15(model) => model.forward(inputs),
            Self::SDXL(model) => model.forward(inputs),
            Self::SD3(model) => model.forward(inputs),
            Self::Flux(model) => model.forward(inputs),
            Self::PixArt(model) => model.forward(inputs),
            Self::AuraFlow(model) => model.forward(inputs),
            Self::Dynamic(model) => model.forward(inputs),
        }
    }
    
    fn architecture(&self) -> ModelArchitecture {
        match self {
            Self::SD15(model) => model.architecture(),
            Self::SDXL(model) => model.architecture(),
            Self::SD3(model) => model.architecture(),
            Self::Flux(model) => model.architecture(),
            Self::PixArt(model) => model.architecture(),
            Self::AuraFlow(model) => model.architecture(),
            Self::Dynamic(model) => model.architecture(),
        }
    }
    
    fn device(&self) -> &crate::Device {
        match self {
            Self::SD15(model) => model.device(),
            Self::SDXL(model) => model.device(),
            Self::SD3(model) => model.device(),
            Self::Flux(model) => model.device(),
            Self::PixArt(model) => model.device(),
            Self::AuraFlow(model) => model.device(),
            Self::Dynamic(model) => model.device(),
        }
    }
    
    fn to_device(&mut self, device: &crate::Device) -> Result<()> {
        match self {
            Self::SD15(model) => model.to_device(device),
            Self::SDXL(model) => model.to_device(device),
            Self::SD3(model) => model.to_device(device),
            Self::Flux(model) => model.to_device(device),
            Self::PixArt(model) => model.to_device(device),
            Self::AuraFlow(model) => model.to_device(device),
            Self::Dynamic(model) => model.to_device(device),
        }
    }
    
    async fn load_pretrained(&mut self, path: &std::path::Path) -> Result<()> {
        match self {
            Self::SD15(model) => model.load_pretrained(path).await,
            Self::SDXL(model) => model.load_pretrained(path).await,
            Self::SD3(model) => model.load_pretrained(path).await,
            Self::Flux(model) => model.load_pretrained(path).await,
            Self::PixArt(model) => model.load_pretrained(path).await,
            Self::AuraFlow(model) => model.load_pretrained(path).await,
            Self::Dynamic(model) => model.load_pretrained(path).await,
        }
    }
    
    async fn save_pretrained(&self, path: &std::path::Path) -> Result<()> {
        match self {
            Self::SD15(model) => model.save_pretrained(path).await,
            Self::SDXL(model) => model.save_pretrained(path).await,
            Self::SD3(model) => model.save_pretrained(path).await,
            Self::Flux(model) => model.save_pretrained(path).await,
            Self::PixArt(model) => model.save_pretrained(path).await,
            Self::AuraFlow(model) => model.save_pretrained(path).await,
            Self::Dynamic(model) => model.save_pretrained(path).await,
        }
    }
    
    fn state_dict(&self) -> Result<HashMap<String, Tensor>> {
        match self {
            Self::SD15(model) => model.state_dict(),
            Self::SDXL(model) => model.state_dict(),
            Self::SD3(model) => model.state_dict(),
            Self::Flux(model) => model.state_dict(),
            Self::PixArt(model) => model.state_dict(),
            Self::AuraFlow(model) => model.state_dict(),
            Self::Dynamic(model) => model.state_dict(),
        }
    }
    
    fn load_state_dict(&mut self, state_dict: HashMap<String, Tensor>) -> Result<()> {
        match self {
            Self::SD15(model) => model.load_state_dict(state_dict),
            Self::SDXL(model) => model.load_state_dict(state_dict),
            Self::SD3(model) => model.load_state_dict(state_dict),
            Self::Flux(model) => model.load_state_dict(state_dict),
            Self::PixArt(model) => model.load_state_dict(state_dict),
            Self::AuraFlow(model) => model.load_state_dict(state_dict),
            Self::Dynamic(model) => model.load_state_dict(state_dict),
        }
    }
    
    fn trainable_parameters(&self) -> Vec<&Tensor> {
        match self {
            Self::SD15(model) => model.trainable_parameters(),
            Self::SDXL(model) => model.trainable_parameters(),
            Self::SD3(model) => model.trainable_parameters(),
            Self::Flux(model) => model.trainable_parameters(),
            Self::PixArt(model) => model.trainable_parameters(),
            Self::AuraFlow(model) => model.trainable_parameters(),
            Self::Dynamic(model) => model.trainable_parameters(),
        }
    }
    
    fn num_parameters(&self) -> usize {
        match self {
            Self::SD15(model) => model.num_parameters(),
            Self::SDXL(model) => model.num_parameters(),
            Self::SD3(model) => model.num_parameters(),
            Self::Flux(model) => model.num_parameters(),
            Self::PixArt(model) => model.num_parameters(),
            Self::AuraFlow(model) => model.num_parameters(),
            Self::Dynamic(model) => model.num_parameters(),
        }
    }
    
    fn memory_usage(&self) -> usize {
        match self {
            Self::SD15(model) => model.memory_usage(),
            Self::SDXL(model) => model.memory_usage(),
            Self::SD3(model) => model.memory_usage(),
            Self::Flux(model) => model.memory_usage(),
            Self::PixArt(model) => model.memory_usage(),
            Self::AuraFlow(model) => model.memory_usage(),
            Self::Dynamic(model) => model.memory_usage(),
        }
    }
}

/// Enum-based network adapter dispatch
pub enum NetworkDispatch {
    LoRA(eridiffusion_networks::LoRAAdapter),
    DoRA(Box<dyn NetworkAdapter + Send + Sync>),
    ControlNet(Box<dyn NetworkAdapter + Send + Sync>),
    IPAdapter(Box<dyn NetworkAdapter + Send + Sync>),
    // Dynamic fallback
    Dynamic(Box<dyn NetworkAdapter + Send + Sync>),
}

impl NetworkAdapter for NetworkDispatch {
    fn forward(&self, x: &Tensor, inputs: &ModelInputs) -> Result<NetworkOutput> {
        match self {
            Self::LoRA(adapter) => adapter.forward(x, inputs),
            Self::DoRA(adapter) => adapter.forward(x, inputs),
            Self::ControlNet(adapter) => adapter.forward(x, inputs),
            Self::IPAdapter(adapter) => adapter.forward(x, inputs),
            Self::Dynamic(adapter) => adapter.forward(x, inputs),
        }
    }
    
    fn adapter_type(&self) -> &str {
        match self {
            Self::LoRA(adapter) => adapter.adapter_type(),
            Self::DoRA(adapter) => adapter.adapter_type(),
            Self::ControlNet(adapter) => adapter.adapter_type(),
            Self::IPAdapter(adapter) => adapter.adapter_type(),
            Self::Dynamic(adapter) => adapter.adapter_type(),
        }
    }
    
    fn set_training(&mut self, training: bool) {
        match self {
            Self::LoRA(adapter) => adapter.set_training(training),
            Self::DoRA(adapter) => adapter.set_training(training),
            Self::ControlNet(adapter) => adapter.set_training(training),
            Self::IPAdapter(adapter) => adapter.set_training(training),
            Self::Dynamic(adapter) => adapter.set_training(training),
        }
    }
    
    fn trainable_parameters(&self) -> Vec<&Tensor> {
        match self {
            Self::LoRA(adapter) => adapter.trainable_parameters(),
            Self::DoRA(adapter) => adapter.trainable_parameters(),
            Self::ControlNet(adapter) => adapter.trainable_parameters(),
            Self::IPAdapter(adapter) => adapter.trainable_parameters(),
            Self::Dynamic(adapter) => adapter.trainable_parameters(),
        }
    }
    
    fn state_dict(&self) -> Result<HashMap<String, Tensor>> {
        match self {
            Self::LoRA(adapter) => adapter.state_dict(),
            Self::DoRA(adapter) => adapter.state_dict(),
            Self::ControlNet(adapter) => adapter.state_dict(),
            Self::IPAdapter(adapter) => adapter.state_dict(),
            Self::Dynamic(adapter) => adapter.state_dict(),
        }
    }
    
    fn load_state_dict(&mut self, state_dict: HashMap<String, Tensor>) -> Result<()> {
        match self {
            Self::LoRA(adapter) => adapter.load_state_dict(state_dict),
            Self::DoRA(adapter) => adapter.load_state_dict(state_dict),
            Self::ControlNet(adapter) => adapter.load_state_dict(state_dict),
            Self::IPAdapter(adapter) => adapter.load_state_dict(state_dict),
            Self::Dynamic(adapter) => adapter.load_state_dict(state_dict),
        }
    }
    
    fn merge_weights(&self, base_weights: &mut HashMap<String, Tensor>) -> Result<()> {
        match self {
            Self::LoRA(adapter) => adapter.merge_weights(base_weights),
            Self::DoRA(adapter) => adapter.merge_weights(base_weights),
            Self::ControlNet(adapter) => adapter.merge_weights(base_weights),
            Self::IPAdapter(adapter) => adapter.merge_weights(base_weights),
            Self::Dynamic(adapter) => adapter.merge_weights(base_weights),
        }
    }
    
    fn num_parameters(&self) -> usize {
        match self {
            Self::LoRA(adapter) => adapter.num_parameters(),
            Self::DoRA(adapter) => adapter.num_parameters(),
            Self::ControlNet(adapter) => adapter.num_parameters(),
            Self::IPAdapter(adapter) => adapter.num_parameters(),
            Self::Dynamic(adapter) => adapter.num_parameters(),
        }
    }
}