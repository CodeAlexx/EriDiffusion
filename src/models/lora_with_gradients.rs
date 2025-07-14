//! LoRA implementation with proper gradient tracking using Var
//! This implementation ensures LoRA weights are trainable and gradients flow correctly

use candle_core::{DType, Device, Module, Result, Tensor, Var, D};
use candle_nn::{VarBuilder, VarMap, Dropout, Init};

/// LoRA Linear layer with gradient tracking
#[derive(Clone)]
pub struct LoRALinearWithGradients {
    // Base layer weights (frozen - regular Tensor)
    base_weight: Tensor,
    base_bias: Option<Tensor>,
    
    // LoRA parameters (trainable - using Var)
    lora_a: Var,
    lora_b: Var,
    
    scale: f32,
    dropout: Option<Dropout>,
}

impl LoRALinearWithGradients {
    /// Create a new LoRA layer with gradient tracking
    pub fn new(
        vb: &VarBuilder,
        in_features: usize,
        out_features: usize,
        rank: usize,
        alpha: f32,
        dropout_rate: f32,
    ) -> Result<Self> {
        // Load base weights as regular tensors (frozen)
        let base_weight = vb.get((out_features, in_features), "weight")?;
        let base_bias = vb.get(out_features, "bias").ok();
        
        // Create LoRA A matrix with Var for gradient tracking
        let bound = (1.0 / (in_features as f64)).sqrt();
        let init_a = Init::Uniform { lo: -bound, up: bound };
        let lora_a_tensor = vb.pp("lora_a").get_with_hints((rank, in_features), "weight", init_a)?;
        let lora_a = Var::from_tensor(&lora_a_tensor)?;
        
        // Create LoRA B matrix with Var (zero initialized)
        let lora_b_tensor = Tensor::zeros((out_features, rank), vb.dtype(), vb.device())?;
        let lora_b = Var::from_tensor(&lora_b_tensor)?;
        
        let scale = alpha / rank as f32;
        
        let dropout = if dropout_rate > 0.0 {
            Some(Dropout::new(dropout_rate))
        } else {
            None
        };
        
        Ok(Self {
            base_weight,
            base_bias,
            lora_a,
            lora_b,
            scale,
            dropout,
        })
    }
    
    /// Create without base layer (for adding LoRA to existing layers)
    pub fn new_without_base(
        device: &Device,
        dtype: DType,
        in_features: usize,
        out_features: usize,
        rank: usize,
        alpha: f32,
        dropout_rate: f32,
    ) -> Result<Self> {
        // Create zero base weights
        let base_weight = Tensor::zeros((out_features, in_features), dtype, device)?;
        
        // Initialize LoRA A with kaiming uniform
        let bound = (1.0 / (in_features as f32)).sqrt();
        let lora_a_init = Tensor::rand(-bound, bound, (rank, in_features), device)?
            .to_dtype(dtype)?;
        let lora_a = Var::from_tensor(&lora_a_init)?;
        
        // Initialize LoRA B with zeros
        let lora_b_init = Tensor::zeros((out_features, rank), dtype, device)?;
        let lora_b = Var::from_tensor(&lora_b_init)?;
        
        let scale = alpha / rank as f32;
        
        let dropout = if dropout_rate > 0.0 {
            Some(Dropout::new(dropout_rate))
        } else {
            None
        };
        
        Ok(Self {
            base_weight,
            base_bias: None,
            lora_a,
            lora_b,
            scale,
            dropout,
        })
    }
    
    /// Get trainable parameters (returns Var references)
    pub fn trainable_vars(&self) -> Vec<&Var> {
        vec![&self.lora_a, &self.lora_b]
    }
    
    /// Forward pass with LoRA computation in the graph
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Base linear transformation: x @ W^T + b
        let mut output = x.matmul(&self.base_weight.t()?)?;
        if let Some(bias) = &self.base_bias {
            output = output.broadcast_add(bias)?;
        }
        
        // Apply dropout to input if training
        let lora_input = if let Some(dropout) = &self.dropout {
            dropout.forward(x, true)?
        } else {
            x.clone()
        };
        
        // LoRA computation: (x @ A^T) @ B^T * scale
        // Using as_tensor() to get Tensor view of Var for computation
        let h = lora_input.matmul(&self.lora_a.as_tensor().t()?)?;
        let lora_out = h.matmul(&self.lora_b.as_tensor().t()?)?;
        let scaled_lora = lora_out.affine(self.scale as f64, 0.)?;
        
        // Add LoRA to base output
        output.add(&scaled_lora)
    }
    
    /// Forward pass for LoRA only (to be added to external base output)
    pub fn forward_lora_only(&self, x: &Tensor) -> Result<Tensor> {
        // Apply dropout to input if training
        let lora_input = if let Some(dropout) = &self.dropout {
            dropout.forward(x, true)?
        } else {
            x.clone()
        };
        
        // LoRA computation: (x @ A^T) @ B^T * scale
        let h = lora_input.matmul(&self.lora_a.as_tensor().t()?)?;
        let lora_out = h.matmul(&self.lora_b.as_tensor().t()?)?;
        lora_out.affine(self.scale as f64, 0.)
    }
}

/// Attention layer with built-in LoRA using gradient tracking
pub struct AttentionWithLoRA {
    num_heads: usize,
    head_dim: usize,
    
    // Base projections (frozen)
    to_q_base: candle_nn::Linear,
    to_k_base: candle_nn::Linear,
    to_v_base: candle_nn::Linear,
    to_out_base: candle_nn::Linear,
    
    // LoRA modules (trainable with Var)
    to_q_lora: Option<LoRALinearWithGradients>,
    to_k_lora: Option<LoRALinearWithGradients>,
    to_v_lora: Option<LoRALinearWithGradients>,
    to_out_lora: Option<LoRALinearWithGradients>,
    
    scale: f64,
}

impl AttentionWithLoRA {
    pub fn new(
        vb: &VarBuilder,
        embed_dim: usize,
        num_heads: usize,
        use_lora: bool,
        lora_rank: usize,
        lora_alpha: f32,
        lora_dropout: f32,
        target_modules: &[String],
    ) -> Result<Self> {
        let head_dim = embed_dim / num_heads;
        let scale = 1.0 / (head_dim as f64).sqrt();
        
        // Load base layers
        let to_q_base = candle_nn::linear_no_bias(embed_dim, embed_dim, vb.pp("to_q"))?;
        let to_k_base = candle_nn::linear_no_bias(embed_dim, embed_dim, vb.pp("to_k"))?;
        let to_v_base = candle_nn::linear_no_bias(embed_dim, embed_dim, vb.pp("to_v"))?;
        let to_out_base = candle_nn::linear(embed_dim, embed_dim, vb.pp("to_out.0"))?;
        
        // Create LoRA layers if enabled
        let (to_q_lora, to_k_lora, to_v_lora, to_out_lora) = if use_lora {
            let device = vb.device();
            let dtype = vb.dtype();
            
            let q_lora = if target_modules.contains(&"to_q".to_string()) {
                Some(LoRALinearWithGradients::new_without_base(
                    device, dtype, embed_dim, embed_dim, lora_rank, lora_alpha, lora_dropout
                )?)
            } else {
                None
            };
            
            let k_lora = if target_modules.contains(&"to_k".to_string()) {
                Some(LoRALinearWithGradients::new_without_base(
                    device, dtype, embed_dim, embed_dim, lora_rank, lora_alpha, lora_dropout
                )?)
            } else {
                None
            };
            
            let v_lora = if target_modules.contains(&"to_v".to_string()) {
                Some(LoRALinearWithGradients::new_without_base(
                    device, dtype, embed_dim, embed_dim, lora_rank, lora_alpha, lora_dropout
                )?)
            } else {
                None
            };
            
            let out_lora = if target_modules.iter().any(|m| m == "to_out" || m == "to_out.0") {
                Some(LoRALinearWithGradients::new_without_base(
                    device, dtype, embed_dim, embed_dim, lora_rank, lora_alpha, lora_dropout
                )?)
            } else {
                None
            };
            
            (q_lora, k_lora, v_lora, out_lora)
        } else {
            (None, None, None, None)
        };
        
        Ok(Self {
            num_heads,
            head_dim,
            to_q_base,
            to_k_base,
            to_v_base,
            to_out_base,
            to_q_lora,
            to_k_lora,
            to_v_lora,
            to_out_lora,
            scale,
        })
    }
    
    fn apply_projection(
        &self,
        base: &candle_nn::Linear,
        lora: &Option<LoRALinearWithGradients>,
        x: &Tensor,
    ) -> Result<Tensor> {
        let base_out = base.forward(x)?;
        
        if let Some(lora_module) = lora {
            // Add LoRA to base output
            let lora_out = lora_module.forward_lora_only(x)?;
            base_out.add(&lora_out)
        } else {
            Ok(base_out)
        }
    }
    
    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let (b, n, c) = hidden_states.dims3()?;
        
        // Apply projections with LoRA
        let q = self.apply_projection(&self.to_q_base, &self.to_q_lora, hidden_states)?;
        let k = self.apply_projection(&self.to_k_base, &self.to_k_lora, hidden_states)?;
        let v = self.apply_projection(&self.to_v_base, &self.to_v_lora, hidden_states)?;
        
        // Reshape for multi-head attention
        let q = q.reshape((b, n, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let k = k.reshape((b, n, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let v = v.reshape((b, n, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        
        // Scaled dot-product attention
        let scores = q.matmul(&k.transpose(2, 3)?)?;
        let scores = (scores * self.scale)?;
        let attn = candle_nn::ops::softmax_last_dim(&scores)?;
        
        // Apply attention to values
        let out = attn.matmul(&v)?;
        let out = out.transpose(1, 2)?.reshape((b, n, c))?;
        
        // Output projection with LoRA
        self.apply_projection(&self.to_out_base, &self.to_out_lora, &out)
    }
    
    /// Get all trainable LoRA parameters
    pub fn trainable_vars(&self) -> Vec<&Var> {
        let mut vars = vec![];
        
        if let Some(lora) = &self.to_q_lora {
            vars.extend(lora.trainable_vars());
        }
        if let Some(lora) = &self.to_k_lora {
            vars.extend(lora.trainable_vars());
        }
        if let Some(lora) = &self.to_v_lora {
            vars.extend(lora.trainable_vars());
        }
        if let Some(lora) = &self.to_out_lora {
            vars.extend(lora.trainable_vars());
        }
        
        vars
    }
}

/// Cross-attention with LoRA and gradient tracking
pub struct CrossAttentionWithLoRA {
    num_heads: usize,
    head_dim: usize,
    
    to_q_base: candle_nn::Linear,
    to_k_base: candle_nn::Linear,
    to_v_base: candle_nn::Linear,
    to_out_base: candle_nn::Linear,
    
    to_q_lora: Option<LoRALinearWithGradients>,
    to_k_lora: Option<LoRALinearWithGradients>,
    to_v_lora: Option<LoRALinearWithGradients>,
    to_out_lora: Option<LoRALinearWithGradients>,
    
    scale: f64,
}

impl CrossAttentionWithLoRA {
    pub fn new(
        vb: &VarBuilder,
        query_dim: usize,
        context_dim: usize,
        num_heads: usize,
        use_lora: bool,
        lora_rank: usize,
        lora_alpha: f32,
        lora_dropout: f32,
        target_modules: &[String],
    ) -> Result<Self> {
        let head_dim = query_dim / num_heads;
        let scale = 1.0 / (head_dim as f64).sqrt();
        
        // Base layers
        let to_q_base = candle_nn::linear_no_bias(query_dim, query_dim, vb.pp("to_q"))?;
        let to_k_base = candle_nn::linear_no_bias(context_dim, query_dim, vb.pp("to_k"))?;
        let to_v_base = candle_nn::linear_no_bias(context_dim, query_dim, vb.pp("to_v"))?;
        let to_out_base = candle_nn::linear(query_dim, query_dim, vb.pp("to_out.0"))?;
        
        // LoRA layers
        let (to_q_lora, to_k_lora, to_v_lora, to_out_lora) = if use_lora {
            let device = vb.device();
            let dtype = vb.dtype();
            
            (
                if target_modules.contains(&"to_q".to_string()) {
                    Some(LoRALinearWithGradients::new_without_base(
                        device, dtype, query_dim, query_dim, lora_rank, lora_alpha, lora_dropout
                    )?)
                } else { None },
                
                if target_modules.contains(&"to_k".to_string()) {
                    Some(LoRALinearWithGradients::new_without_base(
                        device, dtype, context_dim, query_dim, lora_rank, lora_alpha, lora_dropout
                    )?)
                } else { None },
                
                if target_modules.contains(&"to_v".to_string()) {
                    Some(LoRALinearWithGradients::new_without_base(
                        device, dtype, context_dim, query_dim, lora_rank, lora_alpha, lora_dropout
                    )?)
                } else { None },
                
                if target_modules.iter().any(|m| m == "to_out" || m == "to_out.0") {
                    Some(LoRALinearWithGradients::new_without_base(
                        device, dtype, query_dim, query_dim, lora_rank, lora_alpha, lora_dropout
                    )?)
                } else { None },
            )
        } else {
            (None, None, None, None)
        };
        
        Ok(Self {
            num_heads,
            head_dim,
            to_q_base,
            to_k_base,
            to_v_base,
            to_out_base,
            to_q_lora,
            to_k_lora,
            to_v_lora,
            to_out_lora,
            scale,
        })
    }
    
    fn apply_projection(
        &self,
        base: &candle_nn::Linear,
        lora: &Option<LoRALinearWithGradients>,
        x: &Tensor,
    ) -> Result<Tensor> {
        let base_out = base.forward(x)?;
        
        if let Some(lora_module) = lora {
            // Add LoRA to base output
            let lora_out = lora_module.forward_lora_only(x)?;
            base_out.add(&lora_out)
        } else {
            Ok(base_out)
        }
    }
    
    pub fn forward(&self, hidden_states: &Tensor, encoder_hidden_states: &Tensor) -> Result<Tensor> {
        let (b, n, c) = hidden_states.dims3()?;
        let (_, n_enc, _) = encoder_hidden_states.dims3()?;
        
        // Projections with LoRA
        let q = self.apply_projection(&self.to_q_base, &self.to_q_lora, hidden_states)?;
        let k = self.apply_projection(&self.to_k_base, &self.to_k_lora, encoder_hidden_states)?;
        let v = self.apply_projection(&self.to_v_base, &self.to_v_lora, encoder_hidden_states)?;
        
        // Reshape for attention
        let q = q.reshape((b, n, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let k = k.reshape((b, n_enc, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let v = v.reshape((b, n_enc, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        
        // Attention
        let scores = q.matmul(&k.transpose(2, 3)?)?;
        let scores = (scores * self.scale)?;
        let attn = candle_nn::ops::softmax_last_dim(&scores)?;
        
        let out = attn.matmul(&v)?;
        let out = out.transpose(1, 2)?.reshape((b, n, c))?;
        
        // Output projection
        self.apply_projection(&self.to_out_base, &self.to_out_lora, &out)
    }
    
    pub fn trainable_vars(&self) -> Vec<&Var> {
        let mut vars = vec![];
        
        if let Some(lora) = &self.to_q_lora {
            vars.extend(lora.trainable_vars());
        }
        if let Some(lora) = &self.to_k_lora {
            vars.extend(lora.trainable_vars());
        }
        if let Some(lora) = &self.to_v_lora {
            vars.extend(lora.trainable_vars());
        }
        if let Some(lora) = &self.to_out_lora {
            vars.extend(lora.trainable_vars());
        }
        
        vars
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;
    
    #[test]
    fn test_lora_gradient_flow() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device);
        
        // Create LoRA layer
        let lora = LoRALinearWithGradients::new_without_base(
            &device, dtype, 768, 768, 16, 16.0, 0.0
        )?;
        
        // Test input
        let x = Tensor::randn(0f32, 1f32, (2, 10, 768), &device)?;
        
        // Forward pass
        let out = lora.forward(&x)?;
        
        // Compute loss
        let loss = out.mean_all()?;
        
        // Backward pass
        loss.backward()?;
        
        // Check gradients exist
        let vars = lora.trainable_vars();
        for var in vars {
            let grad = var.grad().expect("Gradient should exist");
            assert!(grad.dims() == var.as_tensor().dims());
        }
        
        println!("✓ Gradients flow through LoRA parameters correctly");
        
        Ok(())
    }
    
    #[test]
    fn test_attention_with_lora() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device);
        
        let target_modules = vec!["to_q".to_string(), "to_k".to_string(), "to_v".to_string(), "to_out".to_string()];
        
        let attn = AttentionWithLoRA::new(
            &vb, 512, 8, true, 16, 16.0, 0.0, &target_modules
        )?;
        
        // Test forward
        let x = Tensor::randn(0f32, 1f32, (2, 10, 512), &device)?;
        let out = attn.forward(&x)?;
        
        // Check we have trainable parameters
        let vars = attn.trainable_vars();
        assert_eq!(vars.len(), 8); // 4 modules * 2 vars each
        
        println!("✓ Attention with LoRA creates correct number of trainable parameters");
        
        Ok(())
    }
}