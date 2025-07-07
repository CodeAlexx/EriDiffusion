//! Universal tensor remapper that works with all model types

use anyhow::Result;
use candle_core::{Device, DType, Tensor};
use std::collections::HashMap;
use std::path::Path;
use super::tensor_remapper::TensorRemapper;

/// Model type detection based on tensor names
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelType {
    Flux,
    FluxDev,
    FluxSchnell,
    SD15,
    SD2,
    SDXL,
    SD3,
    SD35,
    Unknown,
}

impl ModelType {
    /// Detect model type from tensor names
    pub fn detect(tensors: &HashMap<String, Tensor>) -> Self {
        // Flux detection
        if tensors.contains_key("double_blocks.0.img_attn.qkv.weight") {
            if tensors.contains_key("guidance_in.weight") {
                ModelType::FluxDev
            } else {
                ModelType::FluxSchnell
            }
        }
        // SD3/3.5 detection (MMDiT architecture)
        else if tensors.contains_key("x_embedder.weight") || 
                tensors.contains_key("model.diffusion_model.x_embedder.weight") {
            if tensors.contains_key("pos_embed.proj.weight") {
                ModelType::SD35
            } else {
                ModelType::SD3
            }
        }
        // SDXL detection
        else if tensors.contains_key("conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight") ||
                tensors.contains_key("add_embedding.linear_1.weight") {
            ModelType::SDXL
        }
        // SD2 detection
        else if tensors.contains_key("cond_stage_model.model.token_embedding.weight") &&
                tensors.contains_key("cond_stage_model.model.ln_final.weight") {
            ModelType::SD2
        }
        // SD1.5 detection
        else if tensors.contains_key("cond_stage_model.transformer.text_model.embeddings.token_embedding.weight") ||
                tensors.contains_key("model.diffusion_model.input_blocks.0.0.weight") {
            ModelType::SD15
        }
        else {
            ModelType::Unknown
        }
    }
}

/// Universal remapper that handles all model types
pub struct UniversalRemapper {
    remapper: TensorRemapper,
    model_type: ModelType,
}

impl UniversalRemapper {
    /// Create a new universal remapper
    pub fn new(checkpoint_path: &Path, device: Device, dtype: DType) -> Result<Self> {
        let remapper = TensorRemapper::from_checkpoint(checkpoint_path, device, dtype)?;
        let model_type = ModelType::detect(remapper.tensors());
        
        println!("Detected model type: {:?}", model_type);
        
        let mut universal = Self {
            remapper,
            model_type,
        };
        
        // Add model-specific mappings
        universal.add_model_specific_mappings();
        
        Ok(universal)
    }
    
    /// Add mappings based on detected model type
    fn add_model_specific_mappings(&mut self) {
        match self.model_type {
            ModelType::Flux | ModelType::FluxDev | ModelType::FluxSchnell => {
                self.add_flux_mappings();
            }
            ModelType::SD35 => {
                self.add_sd35_mappings();
            }
            ModelType::SD3 => {
                self.add_sd3_mappings();
            }
            ModelType::SDXL => {
                self.add_sdxl_mappings();
            }
            ModelType::SD15 | ModelType::SD2 => {
                self.add_sd15_mappings();
            }
            ModelType::Unknown => {
                println!("Warning: Unknown model type, using generic mappings only");
            }
        }
    }
    
    /// Add Flux-specific mappings
    fn add_flux_mappings(&mut self) {
        // Time embeddings
        self.remapper.add_mapping(
            "time_in.mlp.0.fc1.weight".to_string(),
            "time_in.in_layer.weight".to_string(),
        );
        self.remapper.add_mapping(
            "time_in.mlp.0.fc1.bias".to_string(),
            "time_in.in_layer.bias".to_string(),
        );
        
        // MLP naming
        for i in 0..38 {
            for prefix in ["double_blocks", "single_blocks"] {
                for suffix in ["img_mlp", "txt_mlp", "mlp"] {
                    self.remapper.add_mapping(
                        format!("{}.{}.{}.linear1.weight", prefix, i, suffix),
                        format!("{}.{}.{}.0.weight", prefix, i, suffix),
                    );
                    self.remapper.add_mapping(
                        format!("{}.{}.{}.linear2.weight", prefix, i, suffix),
                        format!("{}.{}.{}.2.weight", prefix, i, suffix),
                    );
                }
            }
        }
    }
    
    /// Add SD3.5 mappings
    fn add_sd35_mappings(&mut self) {
        // MMDiT blocks
        for i in 0..38 {
            // Attention projections
            self.remapper.add_mapping(
                format!("mmdit_blocks.{}.attn.to_q.weight", i),
                format!("model.diffusion_model.joint_blocks.{}.x_block.attn.qkv.weight", i),
            );
            
            // Norm layers
            self.remapper.add_mapping(
                format!("mmdit_blocks.{}.norm1.weight", i),
                format!("model.diffusion_model.joint_blocks.{}.x_block.adaLN_modulation.1.weight", i),
            );
        }
        
        // Embedders
        self.remapper.add_mapping(
            "img_embedder.weight".to_string(),
            "model.diffusion_model.x_embedder.weight".to_string(),
        );
        self.remapper.add_mapping(
            "text_embedder.weight".to_string(),
            "model.diffusion_model.context_embedder.weight".to_string(),
        );
    }
    
    /// Add SD3 mappings (similar to SD3.5)
    fn add_sd3_mappings(&mut self) {
        self.add_sd35_mappings(); // SD3 and SD3.5 are very similar
    }
    
    /// Add SDXL mappings
    fn add_sdxl_mappings(&mut self) {
        // UNet mappings
        self.remapper.add_mapping(
            "time_embedding.linear_1.weight".to_string(),
            "model.diffusion_model.time_embed.0.weight".to_string(),
        );
        self.remapper.add_mapping(
            "time_embedding.linear_2.weight".to_string(),
            "model.diffusion_model.time_embed.2.weight".to_string(),
        );
        
        // Text encoders
        self.remapper.add_mapping(
            "text_encoder.embeddings.weight".to_string(),
            "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight".to_string(),
        );
        self.remapper.add_mapping(
            "text_encoder_2.embeddings.weight".to_string(),
            "conditioner.embedders.1.transformer.text_model.embeddings.token_embedding.weight".to_string(),
        );
    }
    
    /// Add SD1.5/SD2 mappings
    fn add_sd15_mappings(&mut self) {
        // Block naming
        for i in 0..12 {
            self.remapper.add_mapping(
                format!("down_blocks.{}.weight", i),
                format!("model.diffusion_model.input_blocks.{}.0.weight", i),
            );
        }
        
        for i in 0..12 {
            self.remapper.add_mapping(
                format!("up_blocks.{}.weight", i),
                format!("model.diffusion_model.output_blocks.{}.0.weight", i),
            );
        }
        
        // Mid block
        self.remapper.add_mapping(
            "mid_block.weight".to_string(),
            "model.diffusion_model.middle_block.0.weight".to_string(),
        );
    }
    
    /// Get the detected model type
    pub fn model_type(&self) -> ModelType {
        self.model_type
    }
    
    /// Load a tensor with model-specific handling
    pub fn load_tensor(&self, path: &str) -> Result<Tensor> {
        // First try the standard remapper
        if let Ok(tensor) = self.remapper.load_with_fallbacks(path) {
            return Ok(tensor);
        }
        
        // Try model-specific synthesis
        match self.model_type {
            ModelType::SD35 => self.synthesize_sd35_tensor(path),
            ModelType::SDXL => self.synthesize_sdxl_tensor(path),
            _ => Err(anyhow::anyhow!("No tensor found for: {}", path)),
        }
    }
    
    /// Synthesize SD3.5 specific tensors
    fn synthesize_sd35_tensor(&self, path: &str) -> Result<Tensor> {
        // Handle RMSNorm weights (SD3.5 uses RMSNorm instead of LayerNorm)
        if path.contains("norm") && path.ends_with(".weight") {
            let hidden_size = 1536; // SD3.5 hidden size
            return Ok(Tensor::ones(hidden_size, self.remapper.dtype, &Device::Cpu)?);
        }
        
        Err(anyhow::anyhow!("Cannot synthesize SD3.5 tensor: {}", path))
    }
    
    /// Synthesize SDXL specific tensors
    fn synthesize_sdxl_tensor(&self, path: &str) -> Result<Tensor> {
        // Handle add_embedding for SDXL
        if path.contains("add_embedding") {
            let dims = if path.contains("linear1") {
                (2816, 1280) // SDXL dimensions
            } else {
                (1280, 1280)
            };
            
            return Ok(Tensor::zeros(dims, self.remapper.dtype, &Device::Cpu)?);
        }
        
        Err(anyhow::anyhow!("Cannot synthesize SDXL tensor: {}", path))
    }
}

/// Helper function to create model-specific remapper
pub fn create_model_remapper(
    checkpoint_path: &Path,
    device: Device,
    dtype: DType,
) -> Result<UniversalRemapper> {
    UniversalRemapper::new(checkpoint_path, device, dtype)
}