// Complete production-ready Flux adaptive loader
// This handles all tensor transformations from Flux checkpoint to your model structure

use candle_core::{Device, DType, Result, Tensor, D};
use candle_nn::{VarBuilder, VarMap};
use std::collections::HashMap;
use std::path::Path;
use std::fs::File;
use std::io::BufReader;
use safetensors::{SafeTensors, tensor::TensorView};
use memmap2::Mmap;

// ===== Error Handling =====

#[derive(Debug, thiserror::Error)]
pub enum LoaderError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("SafeTensors error: {0}")]
    SafeTensors(String),
    
    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),
    
    #[error("Tensor not found: {0}")]
    TensorNotFound(String),
    
    #[error("Invalid tensor shape: expected {expected}, got {got}")]
    InvalidShape { expected: String, got: String },
}

// ===== Main Adaptive Loader =====

pub struct FluxAdaptiveLoader {
    tensors: HashMap<String, Tensor>,
    adapted_tensors: HashMap<String, Tensor>,
    device: Device,
    dtype: DType,
    debug: bool,
}

impl FluxAdaptiveLoader {
    /// Load a Flux checkpoint from disk
    pub fn from_file(path: &Path, device: Device, dtype: DType) -> Result<Self> {
        println!("Loading Flux checkpoint from: {:?}", path);
        
        // Load safetensors file
        let file = File::open(path)?;
        let buffer = unsafe { Mmap::map(&file)? };
        
        // Parse safetensors
        let tensors = SafeTensors::deserialize(&buffer)
            .map_err(|e| LoaderError::SafeTensors(e.to_string()))?;
        
        // Convert to Candle tensors
        let mut loaded_tensors = HashMap::new();
        
        for (name, view) in tensors.tensors() {
            let tensor = view_to_tensor(&view, &device)?;
            loaded_tensors.insert(name.to_string(), tensor);
        }
        
        println!("Loaded {} tensors from checkpoint", loaded_tensors.len());
        
        Ok(Self {
            tensors: loaded_tensors,
            adapted_tensors: HashMap::new(),
            device,
            dtype,
            debug: false,
        })
    }
    
    /// Enable debug output
    pub fn with_debug(mut self, debug: bool) -> Self {
        self.debug = debug;
        self
    }
    
    /// Main adaptation function - transforms all weights
    pub fn adapt_to_model(&mut self, num_double_blocks: usize, num_single_blocks: usize) -> Result<()> {
        println!("Adapting Flux weights for {} double blocks, {} single blocks", 
                 num_double_blocks, num_single_blocks);
        
        // 1. Adapt embeddings
        self.adapt_time_embeddings()?;
        self.adapt_vector_embeddings()?;
        
        // 2. Adapt input projections
        self.adapt_input_projections()?;
        
        // 3. Adapt all transformer blocks
        for i in 0..num_double_blocks {
            self.adapt_double_block(i)?;
        }
        
        for i in 0..num_single_blocks {
            self.adapt_single_block(i)?;
        }
        
        // 4. Adapt final layer
        self.adapt_final_layer()?;
        
        // 5. Copy position embeddings if present
        self.copy_position_embeddings()?;
        
        println!("Successfully adapted {} tensors", self.adapted_tensors.len());
        
        Ok(())
    }
    
    /// Adapt time embeddings: in_layer/out_layer -> mlp.0/mlp.2
    fn adapt_time_embeddings(&mut self) -> Result<()> {
        let mappings = [
            ("time_in.in_layer.weight", "time_in.mlp.0.weight"),
            ("time_in.in_layer.bias", "time_in.mlp.0.bias"),
            ("time_in.out_layer.weight", "time_in.mlp.2.weight"),
            ("time_in.out_layer.bias", "time_in.mlp.2.bias"),
        ];
        
        for (from, to) in &mappings {
            self.copy_and_convert(from, to)?;
        }
        
        if self.debug {
            println!("✓ Adapted time embeddings");
        }
        
        Ok(())
    }
    
    /// Adapt vector embeddings: in_layer/out_layer -> mlp.0/mlp.2
    fn adapt_vector_embeddings(&mut self) -> Result<()> {
        let mappings = [
            ("vector_in.in_layer.weight", "vector_in.mlp.0.weight"),
            ("vector_in.in_layer.bias", "vector_in.mlp.0.bias"),
            ("vector_in.out_layer.weight", "vector_in.mlp.2.weight"),
            ("vector_in.out_layer.bias", "vector_in.mlp.2.bias"),
        ];
        
        for (from, to) in &mappings {
            self.copy_and_convert(from, to)?;
        }
        
        if self.debug {
            println!("✓ Adapted vector embeddings");
        }
        
        Ok(())
    }
    
    /// Adapt input projections (img_in, txt_in)
    fn adapt_input_projections(&mut self) -> Result<()> {
        self.copy_and_convert("img_in.weight", "img_in.weight")?;
        self.copy_and_convert("img_in.bias", "img_in.bias")?;
        self.copy_and_convert("txt_in.weight", "txt_in.weight")?;
        self.copy_and_convert("txt_in.bias", "txt_in.bias")?;
        
        if self.debug {
            println!("✓ Adapted input projections");
        }
        
        Ok(())
    }
    
    /// Adapt a double block
    fn adapt_double_block(&mut self, idx: usize) -> Result<()> {
        let prefix = format!("double_blocks.{}", idx);
        
        // Image attention
        self.adapt_attention_block(&prefix, "img_attn")?;
        
        // Text attention
        self.adapt_attention_block(&prefix, "txt_attn")?;
        
        // Image MLP (0/2 -> fc1/fc2)
        self.adapt_mlp_numbered(&prefix, "img_mlp")?;
        
        // Text MLP
        self.adapt_mlp_numbered(&prefix, "txt_mlp")?;
        
        // Layer norms
        for norm in ["img_norm1", "img_norm2", "txt_norm1", "txt_norm2"] {
            self.copy_and_convert(
                &format!("{}.{}.weight", prefix, norm),
                &format!("{}.{}.weight", prefix, norm),
            )?;
        }
        
        // Modulation (if present)
        self.try_copy_modulation(&prefix, "img_mod");
        self.try_copy_modulation(&prefix, "txt_mod");
        
        if self.debug && idx % 5 == 0 {
            println!("✓ Adapted double blocks {}-{}", idx, (idx + 4).min(idx));
        }
        
        Ok(())
    }
    
    /// Adapt a single block
    fn adapt_single_block(&mut self, idx: usize) -> Result<()> {
        let prefix = format!("single_blocks.{}", idx);
        
        // Attention
        self.adapt_attention_block(&prefix, "attn")?;
        
        // MLP (linear1/linear2 -> mlp.fc1/mlp.fc2)
        self.adapt_mlp_named(&prefix)?;
        
        // Layer norms
        self.copy_and_convert(
            &format!("{}.norm1.weight", prefix),
            &format!("{}.norm1.weight", prefix),
        )?;
        self.copy_and_convert(
            &format!("{}.norm2.weight", prefix),
            &format!("{}.norm2.weight", prefix),
        )?;
        
        // Modulation (if present)
        self.try_copy_modulation(&prefix, "modulation");
        
        if self.debug && idx % 10 == 0 {
            println!("✓ Adapted single blocks {}-{}", idx, (idx + 9).min(idx));
        }
        
        Ok(())
    }
    
    /// Adapt attention block - handles QKV splitting
    fn adapt_attention_block(&mut self, block_prefix: &str, attn_name: &str) -> Result<()> {
        let prefix = format!("{}.{}", block_prefix, attn_name);
        
        // Split combined QKV weight
        if let Some(qkv_weight) = self.tensors.get(&format!("{}.qkv.weight", prefix)) {
            let (total_dim, in_dim) = qkv_weight.dims2()?;
            let head_dim = total_dim / 3;
            
            // Split into Q, K, V
            let q_weight = qkv_weight.narrow(0, 0, head_dim)?;
            let k_weight = qkv_weight.narrow(0, head_dim, head_dim)?;
            let v_weight = qkv_weight.narrow(0, head_dim * 2, head_dim)?;
            
            // Convert and store
            self.adapted_tensors.insert(
                format!("{}.to_q.weight", prefix),
                q_weight.to_device(&self.device)?.to_dtype(self.dtype)?,
            );
            self.adapted_tensors.insert(
                format!("{}.to_k.weight", prefix),
                k_weight.to_device(&self.device)?.to_dtype(self.dtype)?,
            );
            self.adapted_tensors.insert(
                format!("{}.to_v.weight", prefix),
                v_weight.to_device(&self.device)?.to_dtype(self.dtype)?,
            );
        }
        
        // Split combined QKV bias (if present)
        if let Some(qkv_bias) = self.tensors.get(&format!("{}.qkv.bias", prefix)) {
            let total_dim = qkv_bias.dims1()?;
            let head_dim = total_dim / 3;
            
            let q_bias = qkv_bias.narrow(0, 0, head_dim)?;
            let k_bias = qkv_bias.narrow(0, head_dim, head_dim)?;
            let v_bias = qkv_bias.narrow(0, head_dim * 2, head_dim)?;
            
            self.adapted_tensors.insert(
                format!("{}.to_q.bias", prefix),
                q_bias.to_device(&self.device)?.to_dtype(self.dtype)?,
            );
            self.adapted_tensors.insert(
                format!("{}.to_k.bias", prefix),
                k_bias.to_device(&self.device)?.to_dtype(self.dtype)?,
            );
            self.adapted_tensors.insert(
                format!("{}.to_v.bias", prefix),
                v_bias.to_device(&self.device)?.to_dtype(self.dtype)?,
            );
        }
        
        // Output projection (proj -> to_out.0)
        self.copy_and_convert(
            &format!("{}.proj.weight", prefix),
            &format!("{}.to_out.0.weight", prefix),
        )?;
        self.copy_and_convert(
            &format!("{}.proj.bias", prefix),
            &format!("{}.to_out.0.bias", prefix),
        )?;
        
        // Skip QK norm layers - not used in custom model
        
        Ok(())
    }
    
    /// Adapt MLP with numbered layers (0/2) to fc1/fc2
    fn adapt_mlp_numbered(&mut self, block_prefix: &str, mlp_name: &str) -> Result<()> {
        let prefix = format!("{}.{}", block_prefix, mlp_name);
        
        self.copy_and_convert(
            &format!("{}.0.weight", prefix),
            &format!("{}.fc1.weight", prefix),
        )?;
        self.copy_and_convert(
            &format!("{}.0.bias", prefix),
            &format!("{}.fc1.bias", prefix),
        )?;
        self.copy_and_convert(
            &format!("{}.2.weight", prefix),
            &format!("{}.fc2.weight", prefix),
        )?;
        self.copy_and_convert(
            &format!("{}.2.bias", prefix),
            &format!("{}.fc2.bias", prefix),
        )?;
        
        Ok(())
    }
    
    /// Adapt MLP with named layers (linear1/linear2) to mlp.fc1/fc2
    fn adapt_mlp_named(&mut self, block_prefix: &str) -> Result<()> {
        self.copy_and_convert(
            &format!("{}.linear1.weight", block_prefix),
            &format!("{}.mlp.fc1.weight", block_prefix),
        )?;
        self.copy_and_convert(
            &format!("{}.linear1.bias", block_prefix),
            &format!("{}.mlp.fc1.bias", block_prefix),
        )?;
        self.copy_and_convert(
            &format!("{}.linear2.weight", block_prefix),
            &format!("{}.mlp.fc2.weight", block_prefix),
        )?;
        self.copy_and_convert(
            &format!("{}.linear2.bias", block_prefix),
            &format!("{}.mlp.fc2.bias", block_prefix),
        )?;
        
        Ok(())
    }
    
    /// Adapt final layer
    fn adapt_final_layer(&mut self) -> Result<()> {
        // Check both possible names
        if self.tensors.contains_key("final_layer.linear.weight") {
            self.copy_and_convert("final_layer.linear.weight", "final_layer.weight")?;
            self.copy_and_convert("final_layer.linear.bias", "final_layer.bias")?;
        } else if self.tensors.contains_key("final_layer.weight") {
            self.copy_and_convert("final_layer.weight", "final_layer.weight")?;
            self.copy_and_convert("final_layer.bias", "final_layer.bias")?;
        }
        
        if self.debug {
            println!("✓ Adapted final layer");
        }
        
        Ok(())
    }
    
    /// Copy position embeddings if they exist
    fn copy_position_embeddings(&mut self) -> Result<()> {
        let pos_embed_names = [
            "pos_embed",
            "rope.freqs",
            "positional_embedding",
            "pe_embedder.positional_embedding",
        ];
        
        for name in &pos_embed_names {
            if let Some(tensor) = self.tensors.get(*name) {
                self.adapted_tensors.insert(
                    name.to_string(),
                    tensor.to_device(&self.device)?.to_dtype(self.dtype)?,
                );
                if self.debug {
                    println!("✓ Copied position embedding: {}", name);
                }
            }
        }
        
        Ok(())
    }
    
    /// Try to copy modulation layers (may not exist)
    fn try_copy_modulation(&mut self, block_prefix: &str, mod_name: &str) {
        let weight_name = format!("{}.{}.lin.weight", block_prefix, mod_name);
        let bias_name = format!("{}.{}.lin.bias", block_prefix, mod_name);
        
        if let Some(weight) = self.tensors.get(&weight_name) {
            if let Ok(converted) = weight.to_device(&self.device).and_then(|t| t.to_dtype(self.dtype)) {
                self.adapted_tensors.insert(weight_name.clone(), converted);
            }
        }
        
        if let Some(bias) = self.tensors.get(&bias_name) {
            if let Ok(converted) = bias.to_device(&self.device).and_then(|t| t.to_dtype(self.dtype)) {
                self.adapted_tensors.insert(bias_name.clone(), converted);
            }
        }
    }
    
    /// Helper to copy and convert a tensor
    fn copy_and_convert(&mut self, from: &str, to: &str) -> Result<()> {
        if let Some(tensor) = self.tensors.get(from) {
            let converted = tensor.to_device(&self.device)?.to_dtype(self.dtype)?;
            self.adapted_tensors.insert(to.to_string(), converted);
            
            if self.debug {
                let shape = tensor.shape().dims();
                println!("  {} -> {} {:?}", from, to, shape);
            }
        } else if self.debug {
            println!("  ⚠️  {} not found", from);
        }
        
        Ok(())
    }
    
    /// Create a VarBuilder from adapted tensors
    pub fn create_var_builder(self) -> Result<VarBuilder> {
        let mut var_map = VarMap::new();
        
        for (name, tensor) in self.adapted_tensors {
            var_map.set_one(&name, tensor)?;
        }
        
        Ok(VarBuilder::from_varmap(&var_map, self.dtype, &self.device))
    }
    
    /// Get the adapted tensors directly
    pub fn into_tensors(self) -> HashMap<String, Tensor> {
        self.adapted_tensors
    }
    
    /// Print statistics about the loaded tensors
    pub fn print_stats(&self) {
        println!("\n=== Loader Statistics ===");
        println!("Original tensors: {}", self.tensors.len());
        println!("Adapted tensors: {}", self.adapted_tensors.len());
        
        // Count by prefix
        let mut prefixes: HashMap<String, usize> = HashMap::new();
        for name in self.adapted_tensors.keys() {
            let prefix = name.split('.').next().unwrap_or("unknown");
            *prefixes.entry(prefix.to_string()).or_insert(0) += 1;
        }
        
        println!("\nTensors by prefix:");
        for (prefix, count) in prefixes {
            println!("  {}: {}", prefix, count);
        }
    }
}

// ===== Helper Functions =====

/// Convert SafeTensors view to Candle tensor
fn view_to_tensor(view: &TensorView, device: &Device) -> Result<Tensor> {
    let shape = view.shape();
    let dtype = match view.dtype() {
        safetensors::Dtype::F32 => DType::F32,
        safetensors::Dtype::F16 => DType::F16,
        safetensors::Dtype::BF16 => DType::BF16,
        _ => return Err(candle_core::Error::Msg("Unsupported dtype".into())),
    };
    
    let data = view.data();
    
    match dtype {
        DType::F32 => {
            let slice = unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const f32, data.len() / 4)
            };
            Tensor::from_slice(slice, shape, device)
        }
        DType::F16 => {
            let slice = unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const half::f16, data.len() / 2)
            };
            Tensor::from_slice(slice, shape, device)
        }
        DType::BF16 => {
            let slice = unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const half::bf16, data.len() / 2)
            };
            Tensor::from_slice(slice, shape, device)
        }
        _ => Err(candle_core::Error::Msg("Unsupported dtype".into())),
    }
}

// ===== Main Loading Functions =====

/// Load and adapt a Flux checkpoint for your model
pub fn load_flux_checkpoint(
    checkpoint_path: &Path,
    num_double_blocks: usize,
    num_single_blocks: usize,
    device: Device,
    dtype: DType,
    debug: bool,
) -> Result<VarBuilder> {
    let mut loader = FluxAdaptiveLoader::from_file(checkpoint_path, device, dtype)?
        .with_debug(debug);
    
    loader.adapt_to_model(num_double_blocks, num_single_blocks)?;
    
    if debug {
        loader.print_stats();
    }
    
    loader.create_var_builder()
}

/// Load and get raw tensors (for custom handling)
pub fn load_flux_tensors(
    checkpoint_path: &Path,
    num_double_blocks: usize,
    num_single_blocks: usize,
    device: Device,
    dtype: DType,
) -> Result<HashMap<String, Tensor>> {
    let mut loader = FluxAdaptiveLoader::from_file(checkpoint_path, device, dtype)?;
    loader.adapt_to_model(num_double_blocks, num_single_blocks)?;
    Ok(loader.into_tensors())
}

// ===== Integration with your model =====

/// Example: Create your Flux model with adapted weights
pub fn create_flux_model_from_checkpoint(
    checkpoint_path: &Path,
    config: &FluxConfig,
    device: Device,
    dtype: DType,
) -> Result<FluxModel> {
    // Load and adapt checkpoint
    let vb = load_flux_checkpoint(
        checkpoint_path,
        config.num_double_blocks,
        config.num_single_blocks,
        device,
        dtype,
        true, // Enable debug for first load
    )?;
    
    // Create your model with the adapted weights
    FluxModel::from_vb(config.clone(), vb, device, dtype)
}

// ===== Usage Examples =====

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_loader() -> Result<()> {
        let device = Device::cuda_if_available(0)?;
        let dtype = DType::BF16;
        
        // Load checkpoint
        let vb = load_flux_checkpoint(
            Path::new("flux_dev.safetensors"),
            19,  // double blocks
            38,  // single blocks
            device,
            dtype,
            true,
        )?;
        
        // Use with your model
        // let model = YourFluxModel::new(vb)?;
        
        Ok(())
    }
    
    #[test]
    fn test_tensor_loading() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F32;
        
        // Get raw tensors
        let tensors = load_flux_tensors(
            Path::new("flux_dev.safetensors"),
            19,
            38,
            device,
            dtype,
        )?;
        
        // Check some key tensors
        assert!(tensors.contains_key("time_in.mlp.0.weight"));
        assert!(tensors.contains_key("double_blocks.0.img_attn.to_q.weight"));
        
        Ok(())
    }
}

// Placeholder for your model - replace with actual implementation
struct FluxModel;
struct FluxConfig {
    num_double_blocks: usize,
    num_single_blocks: usize,
}

impl FluxModel {
    fn from_vb(_config: FluxConfig, _vb: VarBuilder, _device: Device, _dtype: DType) -> Result<Self> {
        Ok(FluxModel)
    }
}
