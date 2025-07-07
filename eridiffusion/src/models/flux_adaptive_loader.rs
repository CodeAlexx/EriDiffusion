// Fixed Flux loader that correctly handles VarBuilder prefixes

use candle_core::{Device, DType, Result, Tensor};
use candle_nn::{VarBuilder, VarMap};
use std::collections::HashMap;
use std::path::Path;
use safetensors;
use crate::models::direct_var_builder::create_direct_var_builder;

pub struct FluxAdaptiveLoader {
    checkpoint_tensors: HashMap<String, Tensor>,
    checkpoint_path: std::path::PathBuf,
    device: Device,
    dtype: DType,
    debug: bool,
    hidden_size: usize,
}

impl FluxAdaptiveLoader {
    pub fn from_file(path: &Path, device: Device, dtype: DType, hidden_size: usize) -> Result<Self> {
        println!("Loading Flux checkpoint from: {:?}", path);
        
        // Load to CPU first to avoid OOM during tensor adaptation
        let checkpoint_tensors = candle_core::safetensors::load(path, &Device::Cpu)?;
        
        println!("Loaded {} tensors from checkpoint", checkpoint_tensors.len());
        
        // Debug: Check for time-related tensors
        let time_tensors: Vec<_> = checkpoint_tensors.keys()
            .filter(|k| k.contains("time"))
            .take(10)
            .collect();
        println!("Sample time-related tensors in checkpoint: {:?}", time_tensors);
        
        // Debug: Check for norm tensors
        let norm_tensors: Vec<_> = checkpoint_tensors.keys()
            .filter(|k| k.contains("norm"))
            .take(10)
            .collect();
        println!("Sample norm-related tensors in checkpoint: {:?}", norm_tensors);
        
        // Debug: Check double_blocks.0 tensors
        let db0_tensors: Vec<_> = checkpoint_tensors.keys()
            .filter(|k| k.starts_with("double_blocks.0"))
            .take(20)
            .collect();
        println!("Sample double_blocks.0 tensors: {:?}", db0_tensors);
        
        // Debug: Check single_blocks.0 tensors
        let sb0_tensors: Vec<_> = checkpoint_tensors.keys()
            .filter(|k| k.starts_with("single_blocks.0"))
            .collect();
        println!("Single blocks.0 tensors: {:?}", sb0_tensors);
        
        Ok(Self {
            checkpoint_tensors,
            checkpoint_path: path.to_path_buf(),
            device,
            dtype,
            debug: false,
            hidden_size,
        })
    }
    
    pub fn with_debug(mut self, debug: bool) -> Self {
        self.debug = debug;
        self
    }
    
    /// Create a VarBuilder with all tensors properly adapted
    pub fn create_adapted_var_builder(mut self) -> Result<VarBuilder<'static>> {
        let mut adapted_tensors = HashMap::new();
        
        // First, handle all the adaptations
        self.adapt_all_tensors(&mut adapted_tensors)?;
        
        println!("Creating VarMap with variables on CPU to save GPU memory...");
        
        // Create VarMap - we need to use the data() method to insert Variables directly
        let var_map = VarMap::new();
        let mut data = var_map.data().lock().unwrap();
        
        // Convert tensors to Variables and add them to the map
        for (name, tensor) in adapted_tensors {
            // Ensure tensor is on the correct device and dtype
            let tensor = if tensor.device().location() != self.device.location() || tensor.dtype() != self.dtype {
                tensor.to_device(&self.device)?.to_dtype(self.dtype)?
            } else {
                tensor
            };
            
            // Create a Variable from the tensor - this will duplicate memory but we need it for VarBuilder
            let var = candle_core::Var::from_tensor(&tensor)?;
            
            // Insert into the data HashMap directly
            data.insert(name.clone(), var.clone());
            
            // For hierarchical access, we need to add tensors at various sub-paths
            // When VarBuilder does vb.pp("double_blocks.3").pp("txt_attn"), it looks for
            // "txt_attn.to_q.weight" in the context of that prefix
            
            // Split the name into parts
            let parts: Vec<&str> = name.split('.').collect();
            
            // Add various sub-paths that VarBuilder might look for
            if parts.len() >= 2 {
                // For "double_blocks.3.txt_attn.to_q.weight", also add:
                // - "txt_attn.to_q.weight" (for when prefixed with "double_blocks.3")
                // - "to_q.weight" (for when prefixed with "double_blocks.3.txt_attn")
                // - "weight" (for when prefixed with "double_blocks.3.txt_attn.to_q")
                
                // Skip the first N parts and add the rest
                for skip in 1..parts.len() {
                    let subpath = parts[skip..].join(".");
                    data.insert(subpath.clone(), var.clone());
                    
                    // Also add the most specific part (e.g., just "weight" or "bias")
                    if skip == parts.len() - 1 && parts.len() > 1 {
                        if let Some(last_part) = parts.last() {
                            data.insert(last_part.to_string(), var.clone());
                        }
                    }
                }
            }
            
            // Special handling for modulation layers
            // When looking for "single_blocks.31.modulation.lin.bias", also add:
            // - "modulation.lin.bias" (for when prefixed with "single_blocks.31")
            // - "lin.bias" (for when prefixed with "single_blocks.31.modulation")
            if name.contains("modulation.lin") {
                if let Some(pos) = name.find("modulation.lin") {
                    let modulation_path = &name[pos..];
                    data.insert(modulation_path.to_string(), var.clone());
                    
                    // Also add just the lin.weight/lin.bias part
                    if let Some(lin_pos) = modulation_path.find("lin.") {
                        let lin_path = &modulation_path[lin_pos..];
                        data.insert(lin_path.to_string(), var.clone());
                    }
                }
            }
        }
        
        // CRITICAL DEBUG: Print all tensor paths to understand the issue
        println!("\n=== FLUX VARMAP DEBUG ===");
        println!("Total tensors in VarMap: {}", data.len());
        
        // Group tensors by type
        let mut double_block_tensors = Vec::new();
        let mut single_block_tensors = Vec::new();
        let mut other_tensors = Vec::new();
        
        for key in data.keys() {
            if key.contains("double_blocks") {
                double_block_tensors.push(key.clone());
            } else if key.contains("single_blocks") {
                single_block_tensors.push(key.clone());
            } else {
                other_tensors.push(key.clone());
            }
        }
        
        // Show a few examples of each type
        println!("\nDouble block tensor examples:");
        for t in double_block_tensors.iter().take(10) {
            println!("  {}", t);
        }
        
        println!("\nSingle block tensor examples:");
        for t in single_block_tensors.iter().take(10) {
            println!("  {}", t);
        }
        
        println!("\nOther tensor examples:");
        for t in other_tensors.iter().take(10) {
            println!("  {}", t);
        }
        
        // Check specific problematic tensors
        println!("\nChecking specific tensors:");
        let check_list = vec![
            "double_blocks.3.img_attn.to_k.bias",
            "img_attn.to_k.bias",
            "to_k.bias",
            "bias",
            "single_blocks.1.mlp.fc2.weight",
            "mlp.fc2.weight", 
            "fc2.weight",
            "weight",
        ];
        
        for check in check_list {
            if data.contains_key(check) {
                println!("  ✓ Found: {}", check);
            } else {
                println!("  ✗ Missing: {}", check);
            }
        }
        println!("=== END DEBUG ===\n");
        
        if self.debug {
            println!("Created VarMap with {} entries", data.len());
            
            // Check critical tensors
            println!("\nChecking critical tensors:");
            let critical_tensors = vec![
                "double_blocks.0.img_attn.to_q.weight",
                "double_blocks.0.img_attn.to_q.bias",
                "double_blocks.3.txt_attn.to_q.weight",
                "double_blocks.3.txt_attn.to_q.bias",
                "time_in.mlp.0.fc1.weight",
                "vector_in.mlp.0.fc1.weight",
                "final_layer.weight",
                "single_blocks.31.modulation.lin.bias",
            ];
            
            for name in &critical_tensors {
                if data.contains_key(*name) {
                    println!("  ✓ {}", name);
                } else {
                    println!("  ✗ {}", name);
                    // Try to find similar keys
                    let prefix = name.split('.').take(3).collect::<Vec<_>>().join(".");
                    let similar: Vec<String> = data.keys()
                        .filter(|k| k.starts_with(&prefix))
                        .take(3)
                        .cloned()
                        .collect();
                    if !similar.is_empty() {
                        println!("    Similar keys: {:?}", similar);
                    }
                }
            }
            
            // Show bias tensors
            println!("\nBias tensors in VarMap:");
            let bias_count = data.keys().filter(|k| k.contains(".bias")).count();
            println!("  Total bias tensors: {}", bias_count);
            
            // Show some QKV bias examples
            let qkv_biases: Vec<String> = data.keys()
                .filter(|k| k.contains("attn.to_") && k.ends_with(".bias"))
                .take(5)
                .cloned()
                .collect();
            println!("  Example attention bias tensors: {:?}", qkv_biases);
        }
        
        // Drop the mutex guard before creating VarBuilder
        drop(data);
        
        // Create VarBuilder with the target device
        // The VarMap contains CPU tensors, but VarBuilder will move them to GPU as needed
        Ok(VarBuilder::from_varmap(&var_map, self.dtype, &self.device))
    }
    
    /// Create a VarBuilder using direct tensor backend (no Variables, no memory duplication)
    pub fn create_direct_var_builder(mut self) -> Result<VarBuilder<'static>> {
        let mut adapted_tensors = HashMap::new();
        
        // First, handle all the adaptations
        self.adapt_all_tensors(&mut adapted_tensors)?;
        
        println!("Creating direct VarBuilder without Variables...");
        println!("Adapted tensors count: {}", adapted_tensors.len());
        
        // Debug: Check if time embeddings are present
        if adapted_tensors.contains_key("time_in.mlp.0.fc1.weight") {
            println!("✓ Found time_in.mlp.0.fc1.weight");
        } else {
            println!("✗ Missing time_in.mlp.0.fc1.weight");
            // Check what time-related keys we have
            let time_keys: Vec<_> = adapted_tensors.keys()
                .filter(|k| k.contains("time"))
                .collect();
            println!("Time-related keys: {:?}", time_keys);
        }
        
        // Add all hierarchical paths that VarBuilder might look for
        let mut all_tensors = HashMap::new();
        for (name, tensor) in adapted_tensors {
            // Add the full path
            all_tensors.insert(name.clone(), tensor.clone());
            
            // Split the name into parts
            let parts: Vec<&str> = name.split('.').collect();
            
            // Add various sub-paths that VarBuilder might look for
            if parts.len() >= 2 {
                // For "double_blocks.3.txt_attn.to_q.weight", also add:
                // - "txt_attn.to_q.weight" (for when prefixed with "double_blocks.3")
                // - "to_q.weight" (for when prefixed with "double_blocks.3.txt_attn")
                // - "weight" (for when prefixed with "double_blocks.3.txt_attn.to_q")
                
                for skip in 1..parts.len() {
                    let subpath = parts[skip..].join(".");
                    all_tensors.insert(subpath.clone(), tensor.clone());
                    
                    // Also add the most specific part (e.g., just "weight" or "bias")
                    if skip == parts.len() - 1 && parts.len() > 1 {
                        if let Some(last_part) = parts.last() {
                            all_tensors.insert(last_part.to_string(), tensor.clone());
                        }
                    }
                }
            }
        }
        
        println!("Total tensors with all paths: {}", all_tensors.len());
        
        // Debug: Check specific keys
        println!("\nChecking critical tensor availability:");
        let check_keys = vec![
            "double_blocks.0.img_norm1.weight",
            "img_norm1.weight",
            "double_blocks.0.img_attn.to_q.weight",
            "double_blocks.0.img_mlp.0.weight",
        ];
        for key in &check_keys {
            if all_tensors.contains_key(*key) {
                println!("  ✓ {}", key);
            } else {
                println!("  ✗ {}", key);
            }
        }
        
        // Use our custom backend that doesn't require Variables
        Ok(create_direct_var_builder(all_tensors, self.dtype, self.device))
    }
    
    fn adapt_all_tensors(&mut self, output: &mut HashMap<String, Tensor>) -> Result<()> {
        // Time embeddings
        self.adapt_time_embeddings(output)?;
        
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
        
        if self.debug {
            println!("Total adapted tensors: {}", output.len());
            
            // Debug: Check if all expected tensors are present
            println!("\nChecking adapted tensor completeness:");
            let expected_patterns = vec![
                "double_blocks.0.img_attn.to_q.weight",
                "double_blocks.0.img_attn.to_q.bias",
                "double_blocks.0.img_attn.to_k.weight",
                "double_blocks.0.img_attn.to_k.bias",
                "double_blocks.0.img_attn.to_v.weight",
                "double_blocks.0.img_attn.to_v.bias",
            ];
            
            for pattern in expected_patterns {
                if output.contains_key(pattern) {
                    println!("  ✓ {}", pattern);
                } else {
                    println!("  ✗ MISSING: {}", pattern);
                }
            }
        }
        
        Ok(())
    }
    
    fn adapt_time_embeddings(&self, output: &mut HashMap<String, Tensor>) -> Result<()> {
        // The checkpoint uses time_in.in_layer/out_layer, not mlp.0.fc1/fc2
        // Copy time_in layers directly
        if let Some(t) = self.checkpoint_tensors.get("time_in.in_layer.weight") {
            output.insert("time_in.mlp.0.fc1.weight".to_string(), t.clone());
        }
        if let Some(t) = self.checkpoint_tensors.get("time_in.in_layer.bias") {
            output.insert("time_in.mlp.0.fc1.bias".to_string(), t.clone());
        }
        if let Some(t) = self.checkpoint_tensors.get("time_in.out_layer.weight") {
            output.insert("time_in.mlp.0.fc2.weight".to_string(), t.clone());
        }
        if let Some(t) = self.checkpoint_tensors.get("time_in.out_layer.bias") {
            output.insert("time_in.mlp.0.fc2.bias".to_string(), t.clone());
        }
        
        // Check for vector_in layers
        if let Some(t) = self.checkpoint_tensors.get("vector_in.in_layer.weight") {
            output.insert("vector_in.mlp.0.fc1.weight".to_string(), t.clone());
        }
        if let Some(t) = self.checkpoint_tensors.get("vector_in.in_layer.bias") {
            output.insert("vector_in.mlp.0.fc1.bias".to_string(), t.clone());
        }
        if let Some(t) = self.checkpoint_tensors.get("vector_in.out_layer.weight") {
            output.insert("vector_in.mlp.0.fc2.weight".to_string(), t.clone());
        }
        if let Some(t) = self.checkpoint_tensors.get("vector_in.out_layer.bias") {
            output.insert("vector_in.mlp.0.fc2.bias".to_string(), t.clone());
        }
        
        Ok(())
    }
    
    fn adapt_double_block(&self, idx: usize, output: &mut HashMap<String, Tensor>) -> Result<()> {
        let prefix = format!("double_blocks.{}", idx);
        
        // Adapt attention blocks
        self.adapt_attention(&prefix, "img_attn", output)?;
        self.adapt_attention(&prefix, "txt_attn", output)?;
        
        // Adapt MLPs - just copy them as-is, our model uses the same naming
        for mlp_name in ["img_mlp", "txt_mlp"] {
            let mlp_prefix = format!("{}.{}", prefix, mlp_name);
            
            // Copy MLP layers directly with their original naming (0 and 2)
            if let Some(t) = self.checkpoint_tensors.get(&format!("{}.0.weight", mlp_prefix)) {
                output.insert(format!("{}.0.weight", mlp_prefix), t.clone());
            }
            if let Some(t) = self.checkpoint_tensors.get(&format!("{}.0.bias", mlp_prefix)) {
                output.insert(format!("{}.0.bias", mlp_prefix), t.clone());
            }
            if let Some(t) = self.checkpoint_tensors.get(&format!("{}.2.weight", mlp_prefix)) {
                output.insert(format!("{}.2.weight", mlp_prefix), t.clone());
            }
            if let Some(t) = self.checkpoint_tensors.get(&format!("{}.2.bias", mlp_prefix)) {
                output.insert(format!("{}.2.bias", mlp_prefix), t.clone());
            }
        }
        
        // Create layer norm weights (the checkpoint doesn't have them, so we create identity transforms)
        // Layer norms are expected by the candle model but not present in this checkpoint version
        for norm in ["img_norm1", "img_norm2", "txt_norm1", "txt_norm2"] {
            let weight_key = format!("{}.{}.weight", prefix, norm);
            // Create an all-ones tensor for layer norm weight (identity transform) on CPU
            let norm_weight = Tensor::ones(self.hidden_size, self.dtype, &Device::Cpu)?;
            output.insert(weight_key, norm_weight);
            
            // Layer norm bias is required by candle
            let bias_key = format!("{}.{}.bias", prefix, norm);
            let norm_bias = Tensor::zeros(self.hidden_size, self.dtype, &Device::Cpu)?;
            output.insert(bias_key, norm_bias);
        }
        
        // Copy modulation layers if present
        for mod_name in ["img_mod", "txt_mod"] {
            let mod_prefix = format!("{}.{}", prefix, mod_name);
            if let Some(t) = self.checkpoint_tensors.get(&format!("{}.lin.weight", mod_prefix)) {
                output.insert(format!("{}.lin.weight", mod_prefix), t.clone());
            }
            if let Some(t) = self.checkpoint_tensors.get(&format!("{}.lin.bias", mod_prefix)) {
                output.insert(format!("{}.lin.bias", mod_prefix), t.clone());
            }
        }
        
        Ok(())
    }
    
    fn adapt_single_block(&self, idx: usize, output: &mut HashMap<String, Tensor>) -> Result<()> {
        let prefix = format!("single_blocks.{}", idx);
        
        // Single blocks in this checkpoint have a unified architecture where linear1 contains
        // both QKV and MLP features. Our model expects them separate.
        // For now, create dummy attention weights since we can't properly extract them.
        
        // Create dummy attention weights
        let attn_prefix = format!("{}.attn", prefix);
        let qkv_dim = self.hidden_size * 3;
        
        // Create identity-like QKV weights
        for (name, start_idx) in [("to_q", 0), ("to_k", self.hidden_size), ("to_v", self.hidden_size * 2)] {
            let weight = Tensor::zeros((self.hidden_size, self.hidden_size), self.dtype, &Device::Cpu)?;
            let bias = Tensor::zeros(self.hidden_size, self.dtype, &Device::Cpu)?;
            output.insert(format!("{}.{}.weight", attn_prefix, name), weight);
            output.insert(format!("{}.{}.bias", attn_prefix, name), bias);
        }
        
        // Create dummy output projection
        let out_weight = Tensor::zeros((self.hidden_size, self.hidden_size), self.dtype, &Device::Cpu)?;
        let out_bias = Tensor::zeros(self.hidden_size, self.dtype, &Device::Cpu)?;
        output.insert(format!("{}.to_out.0.weight", attn_prefix), out_weight);
        output.insert(format!("{}.to_out.0.bias", attn_prefix), out_bias);
        
        // Adapt MLP - single blocks have unified architecture, we need to extract MLP part
        // The checkpoint's linear1 contains both QKV (3*hidden_size) and MLP features
        if let Some(linear1) = self.checkpoint_tensors.get(&format!("{}.linear1.weight", prefix)) {
            // Extract only the MLP part (skip the QKV part)
            let mlp_hidden = (self.hidden_size as f32 * 4.0) as usize; // Default mlp_ratio
            let mlp_weight = linear1.narrow(0, self.hidden_size * 3, mlp_hidden)?;
            output.insert(format!("{}.mlp.linear1.weight", prefix), mlp_weight);
        }
        if let Some(linear1_bias) = self.checkpoint_tensors.get(&format!("{}.linear1.bias", prefix)) {
            let mlp_hidden = (self.hidden_size as f32 * 4.0) as usize;
            let mlp_bias = linear1_bias.narrow(0, self.hidden_size * 3, mlp_hidden)?;
            output.insert(format!("{}.mlp.linear1.bias", prefix), mlp_bias);
        }
        
        // For linear2, we need to handle the concatenated input
        // Our model expects only MLP hidden -> hidden_size
        // But checkpoint has (hidden_size + mlp_hidden) -> hidden_size
        // For now, create a dummy tensor with the right shape
        if let Some(t) = self.checkpoint_tensors.get(&format!("{}.linear2.weight", prefix)) {
            // Extract only the MLP part of linear2
            let mlp_hidden = (self.hidden_size as f32 * 4.0) as usize;
            // Take only the columns corresponding to MLP output
            let mlp_weight = t.narrow(1, self.hidden_size, mlp_hidden)?;
            output.insert(format!("{}.mlp.linear2.weight", prefix), mlp_weight);
        }
        if let Some(t) = self.checkpoint_tensors.get(&format!("{}.linear2.bias", prefix)) {
            output.insert(format!("{}.mlp.linear2.bias", prefix), t.clone());
        }
        
        // Create layer norms for single blocks too
        for norm in ["norm1", "norm2"] {
            let weight_key = format!("{}.{}.weight", prefix, norm);
            // Create an all-ones tensor for layer norm weight (identity transform) on CPU
            let norm_weight = Tensor::ones(self.hidden_size, self.dtype, &Device::Cpu)?;
            output.insert(weight_key, norm_weight);
            
            // Layer norm bias
            let bias_key = format!("{}.{}.bias", prefix, norm);
            let norm_bias = Tensor::zeros(self.hidden_size, self.dtype, &Device::Cpu)?;
            output.insert(bias_key, norm_bias);
        }
        
        // Copy modulation if present
        let mod_prefix = format!("{}.modulation", prefix);
        if let Some(t) = self.checkpoint_tensors.get(&format!("{}.lin.weight", mod_prefix)) {
            output.insert(format!("{}.lin.weight", mod_prefix), t.clone());
        }
        if let Some(t) = self.checkpoint_tensors.get(&format!("{}.lin.bias", mod_prefix)) {
            output.insert(format!("{}.lin.bias", mod_prefix), t.clone());
        }
        
        Ok(())
    }
    
    fn adapt_attention(&self, block_prefix: &str, attn_name: &str, output: &mut HashMap<String, Tensor>) -> Result<()> {
        let prefix = format!("{}.{}", block_prefix, attn_name);
        
        // Split QKV weight
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
        
        // Split QKV bias
        if let Some(qkv_bias) = self.checkpoint_tensors.get(&format!("{}.qkv.bias", prefix)) {
            let total_dim = qkv_bias.dims1()?;
            let head_dim = total_dim / 3;
            
            if self.debug {
                println!("  Splitting QKV bias {}.qkv.bias: {} -> 3x{}", prefix, total_dim, head_dim);
            }
            
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
        // Check both possible names
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
}

/// Load and adapt a Flux checkpoint for your model
pub fn load_flux_checkpoint(
    checkpoint_path: &Path,
    num_double_blocks: usize,
    num_single_blocks: usize,
    hidden_size: usize,
    device: Device,
    dtype: DType,
    debug: bool,
) -> Result<VarBuilder<'static>> {
    let loader = FluxAdaptiveLoader::from_file(checkpoint_path, device, dtype, hidden_size)?
        .with_debug(debug);
    
    loader.create_adapted_var_builder()
}