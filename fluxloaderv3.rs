// Fixed adapt_attention function that properly handles QKV bias splitting
-----------
You're exactly right! The issue is that when we split the QKV weights, we also need to split the QKV bias. The Flux model stores them combined, but our model expects separate biases for Q, K, and V.
Let me provide the complete fix for the bias splitting:Fixed Flux Loader with QKV Bias SplittingCode // Fixed adapt_attention function that properly handles QKV bias splitting

use candle_core::{Device, DType, Result, Tensor};
use std::collections::HashMap;

/// Fixed adapt_attention that splits both weights AND biases
fn adapt_attention(
    &self,The key fix is in the adapt_attention function - we need to split BOTH the weights AND biases from the combined QKV tensors. The error occurs because candle_nn::linear expects both weight and bias tensors, but we were only providing the split weights.
The fix:

Split qkv.weight into to_q.weight, to_k.weight, to_v.weight ✓
Split qkv.bias into to_q.bias, to_k.bias, to_v.bias ✓ (this was missing)

Now when your model creates the attention layers with linear(in_features, out_features, vb), it will find both the weight and bias tensors it expects.
If some attention layers don't have bias in the checkpoint, you could use candle_nn::linear_no_bias instead, but most Flux checkpoints should have both weight and bias for the QKV projections.RetryClaude can make mistakes. Please double-check responses.
------------- end text

use candle_core::{Device, DType, Result, Tensor};
use std::collections::HashMap;

/// Fixed adapt_attention that splits both weights AND biases
fn adapt_attention(
    &self,
    block_prefix: &str,
    attn_name: &str,
    output: &mut HashMap<String, Tensor>,
) -> Result<()> {
    let prefix = format!("{}.{}", block_prefix, attn_name);
    
    // Split QKV weight
    if let Some(qkv_weight) = self.checkpoint_tensors.get(&format!("{}.qkv.weight", prefix)) {
        let (total_dim, in_features) = qkv_weight.dims2()?;
        let head_dim = total_dim / 3;
        
        if self.debug {
            println!("  Splitting QKV weight {} ({} -> 3x{})", 
                     format!("{}.qkv.weight", prefix), total_dim, head_dim);
        }
        
        let q_weight = qkv_weight.narrow(0, 0, head_dim)?;
        let k_weight = qkv_weight.narrow(0, head_dim, head_dim)?;
        let v_weight = qkv_weight.narrow(0, head_dim * 2, head_dim)?;
        
        output.insert(format!("{}.to_q.weight", prefix), q_weight);
        output.insert(format!("{}.to_k.weight", prefix), k_weight);
        output.insert(format!("{}.to_v.weight", prefix), v_weight);
    }
    
    // Split QKV bias - THIS IS THE KEY FIX
    if let Some(qkv_bias) = self.checkpoint_tensors.get(&format!("{}.qkv.bias", prefix)) {
        let total_dim = qkv_bias.dims1()?;
        let head_dim = total_dim / 3;
        
        if self.debug {
            println!("  Splitting QKV bias {} ({} -> 3x{})", 
                     format!("{}.qkv.bias", prefix), total_dim, head_dim);
        }
        
        let q_bias = qkv_bias.narrow(0, 0, head_dim)?;
        let k_bias = qkv_bias.narrow(0, head_dim, head_dim)?;
        let v_bias = qkv_bias.narrow(0, head_dim * 2, head_dim)?;
        
        output.insert(format!("{}.to_q.bias", prefix), q_bias);
        output.insert(format!("{}.to_k.bias", prefix), k_bias);
        output.insert(format!("{}.to_v.bias", prefix), v_bias);
    }
    
    // Output projection (proj -> to_out.0)
    if let Some(proj_weight) = self.checkpoint_tensors.get(&format!("{}.proj.weight", prefix)) {
        output.insert(format!("{}.to_out.0.weight", prefix), proj_weight.clone());
    }
    
    if let Some(proj_bias) = self.checkpoint_tensors.get(&format!("{}.proj.bias", prefix)) {
        output.insert(format!("{}.to_out.0.bias", prefix), proj_bias.clone());
    }
    
    // Skip norm layers (query_norm, key_norm) as they're not used in our model
    
    Ok(())
}

/// Alternative: Create linear layers without bias if not available
pub fn create_linear_no_bias(
    in_features: usize,
    out_features: usize,
    weight: Tensor,
    vb_device: &Device,
    vb_dtype: DType,
) -> Result<candle_nn::Linear> {
    use candle_nn::{VarBuilder, VarMap};
    
    // Create a temporary VarMap with just the weight
    let var_map = VarMap::new();
    var_map.set_one("weight", weight)?;
    
    // Create VarBuilder without bias
    let vb = VarBuilder::from_varmap(&var_map, vb_dtype, vb_device);
    
    // Use linear_no_bias instead of linear
    candle_nn::linear_no_bias(in_features, out_features, vb)
}

/// Complete working example of FluxAdaptiveLoader with bias handling
pub struct FluxAdaptiveLoader {
    checkpoint_tensors: HashMap<String, Tensor>,
    device: Device,
    dtype: DType,
    debug: bool,
}

impl FluxAdaptiveLoader {
    // ... other methods ...
    
    /// Create VarBuilder with all properly adapted tensors
    pub fn create_var_builder(&self) -> Result<candle_nn::VarBuilder> {
        let mut adapted_tensors = HashMap::new();
        
        // Adapt all tensors including bias splitting
        self.adapt_all_tensors(&mut adapted_tensors)?;
        
        // Create VarMap with all adapted tensors
        let var_map = candle_nn::VarMap::new();
        
        for (name, tensor) in adapted_tensors {
            let converted = tensor.to_device(&self.device)?.to_dtype(self.dtype)?;
            var_map.set_one(&name, converted)?;
        }
        
        if self.debug {
            println!("Created VarBuilder with {} tensors", var_map.all_vars().len());
        }
        
        Ok(candle_nn::VarBuilder::from_varmap(&var_map, self.dtype, &self.device))
    }
    
    fn adapt_all_tensors(&self, output: &mut HashMap<String, Tensor>) -> Result<()> {
        // Time/vector embeddings
        self.adapt_embedding("time_in", output)?;
        self.adapt_embedding("vector_in", output)?;
        
        // Input projections
        self.copy_tensor("img_in.weight", output)?;
        self.copy_tensor("img_in.bias", output)?;
        self.copy_tensor("txt_in.weight", output)?;
        self.copy_tensor("txt_in.bias", output)?;
        
        // Count blocks
        let num_double_blocks = self.count_blocks("double_blocks");
        let num_single_blocks = self.count_blocks("single_blocks");
        
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
    
    // Include the fixed adapt_attention method here
    fn adapt_attention(&self, block_prefix: &str, attn_name: &str, output: &mut HashMap<String, Tensor>) -> Result<()> {
        // Use the implementation above
        let prefix = format!("{}.{}", block_prefix, attn_name);
        
        // Split QKV weight
        if let Some(qkv_weight) = self.checkpoint_tensors.get(&format!("{}.qkv.weight", prefix)) {
            let (total_dim, _) = qkv_weight.dims2()?;
            let head_dim = total_dim / 3;
            
            let q_weight = qkv_weight.narrow(0, 0, head_dim)?;
            let k_weight = qkv_weight.narrow(0, head_dim, head_dim)?;
            let v_weight = qkv_weight.narrow(0, head_dim * 2, head_dim)?;
            
            output.insert(format!("{}.to_q.weight", prefix), q_weight);
            output.insert(format!("{}.to_k.weight", prefix), k_weight);
            output.insert(format!("{}.to_v.weight", prefix), v_weight);
        }
        
        // Split QKV bias
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
        if let Some(proj_weight) = self.checkpoint_tensors.get(&format!("{}.proj.weight", prefix)) {
            output.insert(format!("{}.to_out.0.weight", prefix), proj_weight.clone());
        }
        
        if let Some(proj_bias) = self.checkpoint_tensors.get(&format!("{}.proj.bias", prefix)) {
            output.insert(format!("{}.to_out.0.bias", prefix), proj_bias.clone());
        }
        
        Ok(())
    }
}

/// Debug helper to verify bias tensors are present
pub fn verify_attention_tensors(var_map: &candle_nn::VarMap, prefix: &str) -> Result<()> {
    let expected = vec![
        format!("{}.to_q.weight", prefix),
        format!("{}.to_q.bias", prefix),
        format!("{}.to_k.weight", prefix),
        format!("{}.to_k.bias", prefix),
        format!("{}.to_v.weight", prefix),
        format!("{}.to_v.bias", prefix),
        format!("{}.to_out.0.weight", prefix),
        format!("{}.to_out.0.bias", prefix),
    ];
    
    for tensor_name in expected {
        if var_map.all_vars().contains_key(&tensor_name) {
            println!("✓ Found {}", tensor_name);
        } else {
            println!("✗ Missing {}", tensor_name);
        }
    }
    
    Ok(())
}
