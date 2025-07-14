//! Tensor name mapping for Flux model compatibility
//! 
//! Maps between Flux's tensor naming convention and our internal structure

use std::collections::HashMap;

/// Maps Flux tensor names to our expected structure
pub struct FluxTensorMapper {
    mappings: HashMap<String, String>,
}

impl FluxTensorMapper {
    pub fn new() -> Self {
        let mut mappings = HashMap::new();
        
        // We'll build mappings as we discover them
        Self { mappings }
    }
    
    /// Map a Flux tensor key to our structure
    /// Returns None if the tensor should be skipped (e.g., norm layers)
    pub fn map_key(&self, flux_key: &str) -> Option<String> {
        // Skip normalization layers - they're created, not loaded
        if flux_key.contains("img_norm") || flux_key.contains("txt_norm") {
            return None;
        }
        
        // Handle attention tensor mappings
        if flux_key.contains(".qkv.") {
            // Flux actually concatenates q,k,v into one tensor
            return Some(flux_key.to_string());
        }
        
        // Handle specific patterns
        let mapped = flux_key
            // MLP layers: Flux uses numbered layers (0, 2) instead of named
            .replace(".0.weight", ".w1.weight")
            .replace(".0.bias", ".w1.bias")
            .replace(".2.weight", ".w2.weight")
            .replace(".2.bias", ".w2.bias")
            // Projection layers
            .replace(".proj.", ".to_out.0.")
            // Final layer
            .replace("final_layer.linear", "final_layer.proj_out")
            .replace("final_layer.adaLN_modulation.1", "final_layer.ada_ln_modulation");
            
        Some(mapped)
    }
    
    /// Create a VarBuilder that automatically maps tensor names
    pub fn create_mapped_var_builder<'a>(
        &self,
        tensors: HashMap<String, candle_core::Tensor>,
        dtype: candle_core::DType,
        device: &candle_core::Device,
    ) -> candle_core::Result<candle_nn::VarBuilder<'a>> {
        // Create a new tensor map with our naming
        let mut mapped_tensors = HashMap::new();
        
        for (flux_key, tensor) in tensors {
            if let Some(our_key) = self.map_key(&flux_key) {
                mapped_tensors.insert(our_key, tensor);
            }
            // Skip tensors that map to None
        }
        
        Ok(candle_nn::VarBuilder::from_tensors(mapped_tensors, dtype, device))
    }
}

/// Helper to load Flux model with proper tensor mapping
pub fn load_flux_with_mapping<'a>(
    model_path: &std::path::Path,
    device: &candle_core::Device,
    dtype: candle_core::DType,
) -> candle_core::Result<candle_nn::VarBuilder<'a>> {
    // Load the raw tensors
    let tensors = candle_core::safetensors::load(model_path, device)?;
    
    // Print unmapped tensor names for debugging
    println!("Flux model tensors (first 20):");
    for (i, (name, _)) in tensors.iter().enumerate() {
        if i >= 20 { break; }
        println!("  {}: {}", i, name);
    }
    
    // Create mapper and return mapped VarBuilder
    let mapper = FluxTensorMapper::new();
    mapper.create_mapped_var_builder(tensors, dtype, device)
}

/// Create norm layers that Flux expects but doesn't store
pub fn create_flux_norm_layers(
    device: &candle_core::Device,
    dtype: candle_core::DType,
) -> candle_core::Result<HashMap<String, candle_core::Tensor>> {
    use candle_core::Tensor;
    let mut norm_tensors = HashMap::new();
    
    // Hidden size for Flux-dev
    let hidden_size = 3072;
    
    // Create norm weights as ones (Flux doesn't store these)
    let prefixes = vec![
        "double_blocks.0.img_norm1",
        "double_blocks.0.img_norm2", 
        "double_blocks.0.txt_norm1",
        "double_blocks.0.txt_norm2",
        // Add more as needed
    ];
    
    for prefix in prefixes {
        let weight = Tensor::ones(hidden_size, dtype, device)?;
        norm_tensors.insert(format!("{}.weight", prefix), weight);
    }
    
    Ok(norm_tensors)
}