// Fixed Flux loader that correctly handles VarBuilder prefixes
---- text of issue, keep but comment out!

You're absolutely right! The issue is with how VarBuilder::pp() works in Candle. When you create a prefix with vb.pp("double_blocks.0"), it expects to find tensors in the VarMap without that prefix. The VarBuilder internally prepends the prefix when looking up tensors.

Let me fix the loader to handle this correctly:


Fixed Flux Loader with Correct VarBuilder Handling
Code 
The key insight is that VarBuilder with prefixes doesn't work the way we were using it. When you do vb.pp("double_blocks.0"), Candle's VarBuilder doesn't look for "double_blocks.0.img_mlp.fc1.weight" in the VarMap - it looks for "img_mlp.fc1.weight" and internally tracks the prefix.

I've provided two solutions:

Solution 1: Adapt all tensor names upfront
rust
let vb = load_flux_checkpoint_fixed(
    Path::new("flux_dev.safetensors"),
    device,
    dtype,
    true, // debug
)?;
This creates a flat VarMap with all the adapted tensor names already in place.

Solution 2: Custom VarBuilder that handles prefixes correctly
rust
let vb = load_flux_with_custom_varbuilder(
    Path::new("flux_dev.safetensors"),
    device,
    dtype,
)?;
This creates a custom VarBuilder that concatenates prefixes properly when looking up tensors.

The main fix is in how we populate the VarMap - we need to adapt all tensor names before creating the VarBuilder, not rely on prefix handling to do the mapping for us.

This should resolve the tensor lookup issues you're experiencing!

------------ end of text




use candle_core::{Device, DType, Result, Tensor};
use candle_nn::{VarBuilder, VarMap};
use std::collections::HashMap;
use std::path::Path;
use safetensors;

pub struct FluxAdaptiveLoader {
    checkpoint_tensors: HashMap<String, Tensor>,
    device: Device,
    dtype: DType,
    debug: bool,
}

impl FluxAdaptiveLoader {
    pub fn from_file(path: &Path, device: Device, dtype: DType) -> Result<Self> {
        println!("Loading Flux checkpoint from: {:?}", path);
        
        let checkpoint_tensors = safetensors::load(path, &device)?;
        
        println!("Loaded {} tensors from checkpoint", checkpoint_tensors.len());
        
        Ok(Self {
            checkpoint_tensors,
            device,
            dtype,
            debug: false,
        })
    }
    
    pub fn with_debug(mut self, debug: bool) -> Self {
        self.debug = debug;
        self
    }
    
    /// Create a VarBuilder with proper prefix handling
    pub fn create_var_builder(self) -> Result<VarBuilder> {
        // We need to create a hierarchical VarMap structure
        // that matches how VarBuilder expects to find tensors
        
        let root_varmap = VarMap::new();
        let root_vb = VarBuilder::from_varmap(&root_varmap, self.dtype, &self.device);
        
        // Process all tensors and add them to the correct VarMap level
        for (name, tensor) in self.checkpoint_tensors {
            self.add_tensor_to_varmap(&root_varmap, &name, tensor)?;
        }
        
        Ok(root_vb)
    }
    
    /// Alternative approach: Create a flat VarMap with adapted names
    pub fn create_adapted_var_builder(mut self) -> Result<VarBuilder> {
        let mut adapted_tensors = HashMap::new();
        
        // First, handle all the adaptations
        self.adapt_all_tensors(&mut adapted_tensors)?;
        
        // Create VarMap with adapted tensors
        let var_map = VarMap::new();
        
        for (name, tensor) in adapted_tensors {
            let converted = tensor.to_device(&self.device)?.to_dtype(self.dtype)?;
            var_map.set_one(&name, converted)?;
        }
        
        Ok(VarBuilder::from_varmap(&var_map, self.dtype, &self.device))
    }
    
    fn adapt_all_tensors(&mut self, output: &mut HashMap<String, Tensor>) -> Result<()> {
        // Time embeddings
        self.adapt_embedding("time_in", output)?;
        self.adapt_embedding("vector_in", output)?;
        
        // Input projections
        if let Some(t) = self.checkpoint_tensors.get("img_in.weight") {
            output.insert("img_in.weight".to_string(), t.clone());
        }
        if let Some(t) = self.checkpoint_tensors.get("img_in.bias") {
            output.insert("img_in.bias".to_string(), t.clone());
        }
        if let Some(t) = self.checkpoint_tensors.get("txt_in.weight") {
            output.insert("txt_in.weight".to_string(), t.clone());
        }
        if let Some(t) = self.checkpoint_tensors.get("txt_in.bias") {
            output.insert("txt_in.bias".to_string(), t.clone());
        }
        
        // Count blocks
        let num_double_blocks = self.count_blocks("double_blocks");
        let num_single_blocks = self.count_blocks("single_blocks");
        
        println!("Found {} double blocks, {} single blocks", num_double_blocks, num_single_blocks);
        
        // Adapt all blocks
        for i in 0..num_double_blocks {
            self.adapt_double_block(i, output)?;
        }
        
        for i in 0..num_single_blocks {
            self.adapt_single_block(i, output)?;
        }
        
        // Final layer
        self.adapt_final_layer(output)?;
        
        Ok(())
    }
    
    fn adapt_embedding(&self, prefix: &str, output: &mut HashMap<String, Tensor>) -> Result<()> {
        // Map in_layer/out_layer to mlp.0/mlp.2
        if let Some(t) = self.checkpoint_tensors.get(&format!("{}.in_layer.weight", prefix)) {
            output.insert(format!("{}.mlp.0.weight", prefix), t.clone());
        }
        if let Some(t) = self.checkpoint_tensors.get(&format!("{}.in_layer.bias", prefix)) {
            output.insert(format!("{}.mlp.0.bias", prefix), t.clone());
        }
        if let Some(t) = self.checkpoint_tensors.get(&format!("{}.out_layer.weight", prefix)) {
            output.insert(format!("{}.mlp.2.weight", prefix), t.clone());
        }
        if let Some(t) = self.checkpoint_tensors.get(&format!("{}.out_layer.bias", prefix)) {
            output.insert(format!("{}.mlp.2.bias", prefix), t.clone());
        }
        
        Ok(())
    }
    
    fn adapt_double_block(&self, idx: usize, output: &mut HashMap<String, Tensor>) -> Result<()> {
        let prefix = format!("double_blocks.{}", idx);
        
        // Adapt attention blocks
        self.adapt_attention(&prefix, "img_attn", output)?;
        self.adapt_attention(&prefix, "txt_attn", output)?;
        
        // Adapt MLPs (0/2 -> fc1/fc2)
        for mlp_name in ["img_mlp", "txt_mlp"] {
            let mlp_prefix = format!("{}.{}", prefix, mlp_name);
            
            if let Some(t) = self.checkpoint_tensors.get(&format!("{}.0.weight", mlp_prefix)) {
                output.insert(format!("{}.fc1.weight", mlp_prefix), t.clone());
            }
            if let Some(t) = self.checkpoint_tensors.get(&format!("{}.0.bias", mlp_prefix)) {
                output.insert(format!("{}.fc1.bias", mlp_prefix), t.clone());
            }
            if let Some(t) = self.checkpoint_tensors.get(&format!("{}.2.weight", mlp_prefix)) {
                output.insert(format!("{}.fc2.weight", mlp_prefix), t.clone());
            }
            if let Some(t) = self.checkpoint_tensors.get(&format!("{}.2.bias", mlp_prefix)) {
                output.insert(format!("{}.fc2.bias", mlp_prefix), t.clone());
            }
        }
        
        // Copy layer norms
        for norm in ["img_norm1", "img_norm2", "txt_norm1", "txt_norm2"] {
            let key = format!("{}.{}.weight", prefix, norm);
            if let Some(t) = self.checkpoint_tensors.get(&key) {
                output.insert(key, t.clone());
            }
        }
        
        Ok(())
    }
    
    fn adapt_single_block(&self, idx: usize, output: &mut HashMap<String, Tensor>) -> Result<()> {
        let prefix = format!("single_blocks.{}", idx);
        
        // Adapt attention
        self.adapt_attention(&prefix, "attn", output)?;
        
        // Adapt MLP (linear1/linear2 -> mlp.fc1/mlp.fc2)
        if let Some(t) = self.checkpoint_tensors.get(&format!("{}.linear1.weight", prefix)) {
            output.insert(format!("{}.mlp.fc1.weight", prefix), t.clone());
        }
        if let Some(t) = self.checkpoint_tensors.get(&format!("{}.linear1.bias", prefix)) {
            output.insert(format!("{}.mlp.fc1.bias", prefix), t.clone());
        }
        if let Some(t) = self.checkpoint_tensors.get(&format!("{}.linear2.weight", prefix)) {
            output.insert(format!("{}.mlp.fc2.weight", prefix), t.clone());
        }
        if let Some(t) = self.checkpoint_tensors.get(&format!("{}.linear2.bias", prefix)) {
            output.insert(format!("{}.mlp.fc2.bias", prefix), t.clone());
        }
        
        // Copy layer norms
        for norm in ["norm1", "norm2"] {
            let key = format!("{}.{}.weight", prefix, norm);
            if let Some(t) = self.checkpoint_tensors.get(&key) {
                output.insert(key, t.clone());
            }
        }
        
        Ok(())
    }
    
    fn adapt_attention(&self, block_prefix: &str, attn_name: &str, output: &mut HashMap<String, Tensor>) -> Result<()> {
        let prefix = format!("{}.{}", block_prefix, attn_name);
        
        // Split QKV
        if let Some(qkv) = self.checkpoint_tensors.get(&format!("{}.qkv.weight", prefix)) {
            let (total_dim, _) = qkv.dims2()?;
            let head_dim = total_dim / 3;
            
            let q = qkv.narrow(0, 0, head_dim)?;
            let k = qkv.narrow(0, head_dim, head_dim)?;
            let v = qkv.narrow(0, head_dim * 2, head_dim)?;
            
            output.insert(format!("{}.to_q.weight", prefix), q);
            output.insert(format!("{}.to_k.weight", prefix), k);
            output.insert(format!("{}.to_v.weight", prefix), v);
        }
        
        // Split QKV bias if present
        if let Some(qkv_bias) = self.checkpoint_tensors.get(&format!("{}.qkv.bias", prefix)) {
            let total_dim = qkv_bias.dims1()?;
            let head_dim = total_dim / 3;
            
            let q_bias = qkv_bias.narrow(0, 0, head_dim)?;
            let k_bias = qkv_bias.narrow(0, head_dim, head_dim)?;
            let v_bias = qkv_bias.narrow(0, head_dim * 2, head_dim)?;
            
            output.insert(format!("{}.to_q.bias", prefix), q_bias);
            output.insert(format!("{}.to_k.bias", prefix), k_bias);
            output.insert(format!("{}.to_v.bias", prefix), v_bias);
        }
        
        // Output projection
        if let Some(t) = self.checkpoint_tensors.get(&format!("{}.proj.weight", prefix)) {
            output.insert(format!("{}.to_out.0.weight", prefix), t.clone());
        }
        if let Some(t) = self.checkpoint_tensors.get(&format!("{}.proj.bias", prefix)) {
            output.insert(format!("{}.to_out.0.bias", prefix), t.clone());
        }
        
        Ok(())
    }
    
    fn adapt_final_layer(&self, output: &mut HashMap<String, Tensor>) -> Result<()> {
        if let Some(t) = self.checkpoint_tensors.get("final_layer.linear.weight") {
            output.insert("final_layer.weight".to_string(), t.clone());
        } else if let Some(t) = self.checkpoint_tensors.get("final_layer.weight") {
            output.insert("final_layer.weight".to_string(), t.clone());
        }
        
        if let Some(t) = self.checkpoint_tensors.get("final_layer.linear.bias") {
            output.insert("final_layer.bias".to_string(), t.clone());
        } else if let Some(t) = self.checkpoint_tensors.get("final_layer.bias") {
            output.insert("final_layer.bias".to_string(), t.clone());
        }
        
        Ok(())
    }
    
    fn count_blocks(&self, prefix: &str) -> usize {
        self.checkpoint_tensors.keys()
            .filter(|k| k.starts_with(prefix))
            .filter_map(|k| {
                k.split('.')
                    .nth(1)
                    .and_then(|s| s.parse::<usize>().ok())
            })
            .max()
            .map(|m| m + 1)
            .unwrap_or(0)
    }
    
    fn add_tensor_to_varmap(&self, varmap: &VarMap, name: &str, tensor: Tensor) -> Result<()> {
        // This is a simplified version - you might need more complex logic
        // for nested VarMaps depending on your model structure
        let converted = tensor.to_device(&self.device)?.to_dtype(self.dtype)?;
        varmap.set_one(name, converted)?;
        Ok(())
    }
}

/// Alternative: Create a custom VarBuilder that handles prefixes differently
pub struct PrefixedVarBuilder {
    tensors: HashMap<String, Tensor>,
    prefix: Vec<String>,
    dtype: DType,
    device: Device,
}

impl PrefixedVarBuilder {
    pub fn new(tensors: HashMap<String, Tensor>, dtype: DType, device: Device) -> Self {
        Self {
            tensors,
            prefix: Vec::new(),
            dtype,
            device,
        }
    }
    
    pub fn pp<S: ToString>(&self, s: S) -> Self {
        let mut prefix = self.prefix.clone();
        prefix.push(s.to_string());
        Self {
            tensors: self.tensors.clone(),
            prefix,
            dtype: self.dtype,
            device: self.device.clone(),
        }
    }
    
    pub fn get<S: Into<String>>(&self, name: S) -> Result<Tensor> {
        let mut full_name = self.prefix.clone();
        full_name.push(name.into());
        let key = full_name.join(".");
        
        self.tensors.get(&key)
            .ok_or_else(|| candle_core::Error::Msg(format!("Tensor {} not found", key)))?
            .to_device(&self.device)?
            .to_dtype(self.dtype)
    }
}

/// Main loading function with fixed VarBuilder handling
pub fn load_flux_checkpoint_fixed(
    checkpoint_path: &Path,
    device: Device,
    dtype: DType,
    debug: bool,
) -> Result<VarBuilder> {
    let loader = FluxAdaptiveLoader::from_file(checkpoint_path, device, dtype)?
        .with_debug(debug);
    
    // Use the adapted approach
    loader.create_adapted_var_builder()
}

/// Alternative using custom VarBuilder
pub fn load_flux_with_custom_varbuilder(
    checkpoint_path: &Path,
    device: Device,
    dtype: DType,
) -> Result<PrefixedVarBuilder> {
    let mut loader = FluxAdaptiveLoader::from_file(checkpoint_path, device, dtype)?;
    let mut adapted = HashMap::new();
    loader.adapt_all_tensors(&mut adapted)?;
    
    Ok(PrefixedVarBuilder::new(adapted, dtype, device))
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_varbuilder_prefixes() -> Result<()> {
        // This test demonstrates the issue with VarBuilder prefixes
        let device = Device::Cpu;
        let dtype = DType::F32;
        
        // Create a VarMap with some test tensors
        let var_map = VarMap::new();
        
        // This is what VarBuilder expects when using pp()
        let t1 = Tensor::zeros(&[10, 10], dtype, &device)?;
        var_map.set_one("img_mlp.fc1.weight", t1)?;
        
        // Create VarBuilder with prefix
        let vb = VarBuilder::from_varmap(&var_map, dtype, &device);
        let vb_prefixed = vb.pp("double_blocks.0");
        
        // This will look for "img_mlp.fc1.weight" in the map, NOT "double_blocks.0.img_mlp.fc1.weight"
        let tensor = vb_prefixed.get((&["img_mlp", "fc1"], "weight"))?;
        assert_eq!(tensor.dims(), &[10, 10]);
        
        Ok(())
    }
}
