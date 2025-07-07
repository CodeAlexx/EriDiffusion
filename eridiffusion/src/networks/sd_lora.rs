//! Stable Diffusion LoRA/LoKr implementation
//! Supports SDXL, SD 3.5, and SD 1.5 with production-ready code

use anyhow::Result;
use candle_core::{Device, DType, Module, Tensor, D};
use candle_nn::{Init, VarBuilder, VarMap};
use std::collections::HashMap;

/// Adapter types for Stable Diffusion
#[derive(Debug, Clone, PartialEq)]
pub enum SDAdapterType {
   LoRA,
   LoKr,
}

/// LoRA module for SD models
#[derive(Debug, Clone)]
pub struct SDLoRAModule {
   pub lora_a: Tensor,
   pub lora_b: Tensor,
   pub scale: f32,
   rank: usize,
   alpha: f32,
   dropout: Option<f32>,
}

impl SDLoRAModule {
   pub fn new(
       in_features: usize,
       out_features: usize,
       rank: usize,
       alpha: f32,
       device: Device,
       dtype: DType,
   ) -> Result<Self> {
       // Initialize lora_A with kaiming uniform
       let bound = (1.0 / (in_features as f32)).sqrt();
       let lora_a = Tensor::rand(-bound, bound, (rank, in_features), &device)?.to_dtype(dtype)?;
       
       // Initialize lora_B with zeros
       let lora_b = Tensor::zeros((out_features, rank), dtype, &device)?;
       
       let scale = alpha / (rank as f32);
       
       Ok(Self {
           lora_a,
           lora_b,
           scale,
           rank,
           alpha,
           dropout: None,
       })
   }
   
   pub fn with_dropout(mut self, dropout: f32) -> Self {
       self.dropout = Some(dropout);
       self
   }
   
   pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
       // xs: [batch, seq_len, in_features] or [batch, in_features]
       let h = xs.matmul(&self.lora_a.t()?)?;
       let out = h.matmul(&self.lora_b.t()?)?;
       Ok(out.affine(self.scale as f64, 0.0)?)
   }
   
   pub fn forward_training(&self, xs: &Tensor, training: bool) -> Result<Tensor> {
       let h = xs.matmul(&self.lora_a.t()?)?;
       
       let h = if training && self.dropout.is_some() {
           candle_nn::ops::dropout(&h, self.dropout.unwrap())?
       } else {
           h
       };
       
       let out = h.matmul(&self.lora_b.t()?)?;
       Ok(out.affine(self.scale as f64, 0.0)?)
   }
   
   pub fn merge_weights(&self, base_weight: &Tensor) -> Result<Tensor> {
       // W' = W + (B @ A) * scale
       let lora_weight = self.lora_b.matmul(&self.lora_a)?;
       let scaled_weight = lora_weight.affine(self.scale as f64, 0.0)?;
       Ok(base_weight.add(&scaled_weight)?)
   }
   
   pub fn save_weights(&self, prefix: &str) -> HashMap<String, Tensor> {
       let mut tensors = HashMap::new();
       
       // Standard LoRA naming convention
       tensors.insert(format!("{}.lora_down.weight", prefix), self.lora_a.clone());
       tensors.insert(format!("{}.lora_up.weight", prefix), self.lora_b.clone());
       tensors.insert(
           format!("{}.alpha", prefix),
           Tensor::new(&[self.alpha], self.lora_a.device()).unwrap()
       );
       
       tensors
   }
   
   pub fn load_weights(
       prefix: &str,
       tensors: &HashMap<String, Tensor>,
       device: &Device,
       dtype: DType,
   ) -> Result<Self> {
       let lora_a = tensors
           .get(&format!("{}.lora_down.weight", prefix))
           .ok_or_else(|| anyhow::anyhow!("Missing {}.lora_down.weight", prefix))?
           .to_device(device)?
           .to_dtype(dtype)?;
           
       let lora_b = tensors
           .get(&format!("{}.lora_up.weight", prefix))
           .ok_or_else(|| anyhow::anyhow!("Missing {}.lora_up.weight", prefix))?
           .to_device(device)?
           .to_dtype(dtype)?;
       
       let rank = lora_a.dim(0)?;
       let alpha = tensors
           .get(&format!("{}.alpha", prefix))
           .and_then(|t| t.to_vec0::<f32>().ok())
           .unwrap_or(rank as f32);
       
       Ok(Self {
           lora_a,
           lora_b,
           scale: alpha / (rank as f32),
           rank,
           alpha,
           dropout: None,
       })
   }
   
   pub fn num_parameters(&self) -> usize {
       self.lora_a.elem_count() + self.lora_b.elem_count()
   }
}

/// LoKr (Kronecker product) module for SD models
#[derive(Debug, Clone)]
pub struct SDLoKrModule {
   pub lokr_w1: Tensor,
   pub lokr_w2: Tensor,
   pub lokr_t2: Option<Tensor>,
   pub scale: f32,
   rank: usize,
   alpha: f32,
   decompose_factor: i32,
}

impl SDLoKrModule {
   pub fn new(
       in_features: usize,
       out_features: usize,
       rank: usize,
       alpha: f32,
       decompose_factor: Option<i32>,
       device: Device,
       dtype: DType,
   ) -> Result<Self> {
       let factor = decompose_factor.unwrap_or(-1);
       
       // Calculate dimensions for w1 and w2
       let (w1_in, w1_out, w2_in, w2_out) = if factor > 0 {
           let f = factor as usize;
           // Ensure dimensions are divisible
           if in_features % f != 0 || out_features % f != 0 {
               anyhow::bail!("Features must be divisible by decompose_factor");
           }
           (in_features / f, rank, rank, out_features / f)
       } else {
           // Auto-calculate factor for square decomposition
           let total_params = in_features * out_features;
           let target_params = rank * rank * 2; // Approximate
           let f = ((total_params as f32 / target_params as f32).sqrt()) as usize;
           let f = f.max(1);
           
           // Find closest divisor
           let f = (1..=f).rev().find(|&x| in_features % x == 0 && out_features % x == 0).unwrap_or(1);
           
           (in_features / f, rank, rank, out_features / f)
       };
       
       // Initialize w1 with kaiming uniform
       let bound_w1 = (1.0 / (w1_in as f32)).sqrt();
       let lokr_w1 = Tensor::rand(-bound_w1, bound_w1, (w1_out, w1_in), &device)?.to_dtype(dtype)?;
       
       // Initialize w2 with zeros
       let lokr_w2 = Tensor::zeros((w2_out, w2_in), dtype, &device)?;
       
       // Optional full-rank component (small initialization)
       let lokr_t2 = if factor > 0 && factor as usize > 1 {
           let bound_t2 = (1.0 / (in_features as f32)).sqrt() * 0.1;
           Some(Tensor::rand(-bound_t2, bound_t2, (out_features, in_features), &device)?.to_dtype(dtype)?)
       } else {
           None
       };
       
       let scale = alpha / (rank as f32);
       
       Ok(Self {
           lokr_w1,
           lokr_w2,
           lokr_t2,
           scale,
           rank,
           alpha,
           decompose_factor: factor,
       })
   }
   
   pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
       // Get dimensions
       let original_shape = xs.dims();
       let last_dim = original_shape[original_shape.len() - 1];
       
       // Reshape input for kronecker product
       let (w1_in, w1_out) = self.lokr_w1.dims2()?;
       let (w2_out, w2_in) = self.lokr_w2.dims2()?;
       
       let f1 = last_dim / w1_in;
       let f2 = w2_out * f1;
       
       // First reshape and multiply with w1
       let xs_reshaped = xs.reshape(((), f1, w1_in))?;
       let h1 = xs_reshaped.matmul(&self.lokr_w1.t()?)?.reshape(((), w1_out * f1))?;
       
       // Second reshape and multiply with w2
       let h1_reshaped = h1.reshape(((), w2_in, f1))?;
       let h1_perm = h1_reshaped.permute((0, 2, 1))?;
       let h2 = h1_perm.reshape(((), f1, w2_in))?;
       let out = h2.matmul(&self.lokr_w2.t()?)?.reshape(((), f2))?;
       
       // Add optional full-rank component
       let out = if let Some(ref t2) = self.lokr_t2 {
           let t2_out = xs.matmul(&t2.t()?)?;
           (out + t2_out)?
       } else {
           out
       };
       
       // Reshape back to original shape (except last dim)
       let mut final_shape = original_shape.to_vec();
       final_shape[original_shape.len() - 1] = f2;
       let out = out.reshape(final_shape)?;
       
       Ok(out.affine(self.scale as f64, 0.0)?)
   }
   
   pub fn merge_weights(&self, base_weight: &Tensor) -> Result<Tensor> {
       // Compute the kronecker product of w2 and w1
       let (w1_in, w1_out) = self.lokr_w1.dims2()?;
       let (w2_out, w2_in) = self.lokr_w2.dims2()?;
       
       // w2 ⊗ w1 = [w2_out * w1_out, w2_in * w1_in]
       let w1_expanded = self.lokr_w1.unsqueeze(0)?.unsqueeze(2)?;
       let w2_expanded = self.lokr_w2.unsqueeze(1)?.unsqueeze(3)?;
       
       let kron = w2_expanded.broadcast_mul(&w1_expanded)?;
       let kron = kron.reshape((w2_out * w1_out, w2_in * w1_in))?;
       
       let mut merged = kron.affine(self.scale as f64, 0.0)?;
       
       if let Some(ref t2) = self.lokr_t2 {
           merged = (merged + t2.affine(self.scale as f64, 0.0)?)?;
       }
       
       Ok(base_weight.add(&merged)?)
   }
   
   pub fn save_weights(&self, prefix: &str) -> HashMap<String, Tensor> {
       let mut tensors = HashMap::new();
       
       tensors.insert(format!("{}.lokr_w1", prefix), self.lokr_w1.clone());
       tensors.insert(format!("{}.lokr_w2", prefix), self.lokr_w2.clone());
       
       if let Some(ref t2) = self.lokr_t2 {
           tensors.insert(format!("{}.lokr_t2", prefix), t2.clone());
       }
       
       tensors.insert(
           format!("{}.alpha", prefix),
           Tensor::new(&[self.alpha], self.lokr_w1.device()).unwrap()
       );
       
       tensors.insert(
           format!("{}.lokr_decompose_factor", prefix),
           Tensor::new(self.decompose_factor as f32, self.lokr_w1.device()).unwrap()
       );
       
       tensors
   }
   
   pub fn load_weights(
       prefix: &str,
       tensors: &HashMap<String, Tensor>,
       device: &Device,
       dtype: DType,
   ) -> Result<Self> {
       let lokr_w1 = tensors
           .get(&format!("{}.lokr_w1", prefix))
           .ok_or_else(|| anyhow::anyhow!("Missing {}.lokr_w1", prefix))?
           .to_device(device)?
           .to_dtype(dtype)?;
           
       let lokr_w2 = tensors
           .get(&format!("{}.lokr_w2", prefix))
           .ok_or_else(|| anyhow::anyhow!("Missing {}.lokr_w2", prefix))?
           .to_device(device)?
           .to_dtype(dtype)?;
           
       let lokr_t2 = tensors
           .get(&format!("{}.lokr_t2", prefix))
           .map(|t| t.to_device(device).and_then(|t| t.to_dtype(dtype)))
           .transpose()?;
           
       let rank = lokr_w1.dim(0)?;
       let alpha = tensors
           .get(&format!("{}.alpha", prefix))
           .and_then(|t| t.to_vec0::<f32>().ok())
           .unwrap_or(rank as f32);
           
       let decompose_factor = tensors
           .get(&format!("{}.lokr_decompose_factor", prefix))
           .and_then(|t| t.to_vec0::<f32>().ok())
           .unwrap_or(-1.0) as i32;
       
       Ok(Self {
           lokr_w1,
           lokr_w2,
           lokr_t2,
           scale: alpha / (rank as f32),
           rank,
           alpha,
           decompose_factor,
       })
   }
   
   pub fn num_parameters(&self) -> usize {
       self.lokr_w1.elem_count() + self.lokr_w2.elem_count() + 
       self.lokr_t2.as_ref().map(|t| t.elem_count()).unwrap_or(0)
   }
}

/// SD adapter configuration
#[derive(Debug, Clone)]
pub struct SDAdapterConfig {
   pub adapter_type: SDAdapterType,
   pub rank: usize,
   pub alpha: f32,
   pub dropout: Option<f32>,
   pub decompose_factor: Option<i32>,
   pub target_modules: Vec<String>,
}

impl Default for SDAdapterConfig {
   fn default() -> Self {
       Self {
           adapter_type: SDAdapterType::LoRA,
           rank: 16,
           alpha: 16.0,
           dropout: None,
           decompose_factor: None,
           target_modules: vec![],
       }
   }
}

/// Collection of SD adapters
pub struct SDAdapterCollection {
   lora_modules: HashMap<String, SDLoRAModule>,
   lokr_modules: HashMap<String, SDLoKrModule>,
   config: SDAdapterConfig,
   model_type: String,
}

impl SDAdapterCollection {
   pub fn new(config: SDAdapterConfig, model_type: &str) -> Self {
       let mut config = config;
       if config.target_modules.is_empty() {
           config.target_modules = get_target_modules(model_type);
       }
       
       Self {
           lora_modules: HashMap::new(),
           lokr_modules: HashMap::new(),
           config,
           model_type: model_type.to_string(),
       }
   }
   
   pub fn create_adapter(
       &mut self,
       name: &str,
       in_features: usize,
       out_features: usize,
       device: Device,
       dtype: DType,
   ) -> Result<()> {
       match self.config.adapter_type {
           SDAdapterType::LoRA => {
               let mut lora = SDLoRAModule::new(
                   in_features,
                   out_features,
                   self.config.rank,
                   self.config.alpha,
                   device,
                   dtype,
               )?;
               
               if let Some(dropout) = self.config.dropout {
                   lora = lora.with_dropout(dropout);
               }
               
               self.lora_modules.insert(name.to_string(), lora);
           }
           SDAdapterType::LoKr => {
               let lokr = SDLoKrModule::new(
                   in_features,
                   out_features,
                   self.config.rank,
                   self.config.alpha,
                   self.config.decompose_factor,
                   device,
                   dtype,
               )?;
               
               self.lokr_modules.insert(name.to_string(), lokr);
           }
       }
       
       Ok(())
   }
   
   pub fn forward(&self, name: &str, xs: &Tensor) -> Result<Option<Tensor>> {
       match self.config.adapter_type {
           SDAdapterType::LoRA => {
               self.lora_modules.get(name).map(|m| m.forward(xs)).transpose()
           }
           SDAdapterType::LoKr => {
               self.lokr_modules.get(name).map(|m| m.forward(xs)).transpose()
           }
       }
   }
   
   pub fn save_safetensors(&self) -> HashMap<String, Tensor> {
       let mut all_tensors = HashMap::new();
       
       // Save adapter weights
       match self.config.adapter_type {
           SDAdapterType::LoRA => {
               for (name, module) in &self.lora_modules {
                   let module_tensors = module.save_weights(name);
                   all_tensors.extend(module_tensors);
               }
           }
           SDAdapterType::LoKr => {
               for (name, module) in &self.lokr_modules {
                   let module_tensors = module.save_weights(name);
                   all_tensors.extend(module_tensors);
               }
           }
       }
       
       // Save metadata
       // Note: String metadata would need special handling
       // For now we'll skip adapter_type metadata
       
       all_tensors.insert(
           "metadata.rank".to_string(),
           Tensor::new(self.config.rank as f32, &Device::Cpu).unwrap()
       );
       
       all_tensors.insert(
           "metadata.alpha".to_string(),
           Tensor::new(self.config.alpha, &Device::Cpu).unwrap()
       );
       
       // Note: String metadata would need special handling
       // For now we'll skip model_type metadata
       
       all_tensors
   }
   
   pub fn load_safetensors(
       tensors: &HashMap<String, Tensor>,
       device: &Device,
       dtype: DType,
   ) -> Result<Self> {
       // Load metadata
       let model_type = tensors
           .get("metadata.model_type")
           .map(|t| t.to_string())
           .unwrap_or_else(|| "sdxl".to_string());
           
       let adapter_type = tensors
           .get("metadata.adapter_type")
           .map(|t| t.to_string())
           .map(|s| match s.as_str() {
               "lokr" => SDAdapterType::LoKr,
               _ => SDAdapterType::LoRA,
           })
           .unwrap_or(SDAdapterType::LoRA);
           
       let rank = tensors
           .get("metadata.rank")
           .and_then(|t| t.to_vec0::<f32>().ok())
           .unwrap_or(16.0) as usize;
           
       let alpha = tensors
           .get("metadata.alpha")
           .and_then(|t| t.to_vec0::<f32>().ok())
           .unwrap_or(16.0);
       
       let config = SDAdapterConfig {
           adapter_type: adapter_type.clone(),
           rank,
           alpha,
           dropout: None,
           decompose_factor: None,
           target_modules: get_target_modules(&model_type),
       };
       
       let mut collection = Self::new(config, &model_type);
       
       // Load modules based on type
       match adapter_type {
           SDAdapterType::LoRA => {
               for key in tensors.keys() {
                   if key.ends_with(".lora_down.weight") {
                       let module_name = key.trim_end_matches(".lora_down.weight");
                       let module = SDLoRAModule::load_weights(module_name, tensors, device, dtype)?;
                       collection.lora_modules.insert(module_name.to_string(), module);
                   }
               }
           }
           SDAdapterType::LoKr => {
               for key in tensors.keys() {
                   if key.ends_with(".lokr_w1") {
                       let module_name = key.trim_end_matches(".lokr_w1");
                       let module = SDLoKrModule::load_weights(module_name, tensors, device, dtype)?;
                       collection.lokr_modules.insert(module_name.to_string(), module);
                   }
               }
           }
       }
       
       Ok(collection)
   }
   
   pub fn num_parameters(&self) -> usize {
       match self.config.adapter_type {
           SDAdapterType::LoRA => {
               self.lora_modules.values().map(|m| m.num_parameters()).sum()
           }
           SDAdapterType::LoKr => {
               self.lokr_modules.values().map(|m| m.num_parameters()).sum()
           }
       }
   }
}

/// SDXL target modules (UNet)
pub fn sdxl_target_modules() -> Vec<String> {
   let mut modules = vec![];
   
   // Down blocks (3 blocks, 2 attention layers each)
   for i in 0..3 {
       for j in 0..2 {
           modules.extend(vec![
               format!("down_blocks.{}.attentions.{}.transformer_blocks.0.attn1.to_q", i, j),
               format!("down_blocks.{}.attentions.{}.transformer_blocks.0.attn1.to_k", i, j),
               format!("down_blocks.{}.attentions.{}.transformer_blocks.0.attn1.to_v", i, j),
               format!("down_blocks.{}.attentions.{}.transformer_blocks.0.attn1.to_out.0", i, j),
               format!("down_blocks.{}.attentions.{}.transformer_blocks.0.attn2.to_q", i, j),
               format!("down_blocks.{}.attentions.{}.transformer_blocks.0.attn2.to_k", i, j),
               format!("down_blocks.{}.attentions.{}.transformer_blocks.0.attn2.to_v", i, j),
               format!("down_blocks.{}.attentions.{}.transformer_blocks.0.attn2.to_out.0", i, j),
           ]);
       }
   }
   
   // Mid block
   modules.extend(vec![
       "mid_block.attentions.0.transformer_blocks.0.attn1.to_q",
       "mid_block.attentions.0.transformer_blocks.0.attn1.to_k",
       "mid_block.attentions.0.transformer_blocks.0.attn1.to_v",
       "mid_block.attentions.0.transformer_blocks.0.attn1.to_out.0",
       "mid_block.attentions.0.transformer_blocks.0.attn2.to_q",
       "mid_block.attentions.0.transformer_blocks.0.attn2.to_k",
       "mid_block.attentions.0.transformer_blocks.0.attn2.to_v",
       "mid_block.attentions.0.transformer_blocks.0.attn2.to_out.0",
   ].into_iter().map(String::from));
   
   // Up blocks (4 blocks, 3 attention layers each)
   for i in 0..4 {
       for j in 0..3 {
           modules.extend(vec![
               format!("up_blocks.{}.attentions.{}.transformer_blocks.0.attn1.to_q", i, j),
               format!("up_blocks.{}.attentions.{}.transformer_blocks.0.attn1.to_k", i, j),
               format!("up_blocks.{}.attentions.{}.transformer_blocks.0.attn1.to_v", i, j),
               format!("up_blocks.{}.attentions.{}.transformer_blocks.0.attn1.to_out.0", i, j),
               format!("up_blocks.{}.attentions.{}.transformer_blocks.0.attn2.to_q", i, j),
               format!("up_blocks.{}.attentions.{}.transformer_blocks.0.attn2.to_k", i, j),
               format!("up_blocks.{}.attentions.{}.transformer_blocks.0.attn2.to_v", i, j),
               format!("up_blocks.{}.attentions.{}.transformer_blocks.0.attn2.to_out.0", i, j),
           ]);
       }
   }
   
   modules
}

/// SD 3.5 target modules (MMDiT)
pub fn sd35_target_modules() -> Vec<String> {
   let mut modules = vec![];
   
   // Joint transformer blocks (typically 38 blocks)
   for i in 0..38 {
       modules.extend(vec![
           // Context block (cross-attention)
           format!("joint_blocks.{}.context_block.attn.to_q", i),
           format!("joint_blocks.{}.context_block.attn.to_k", i),
           format!("joint_blocks.{}.context_block.attn.to_v", i),
           format!("joint_blocks.{}.context_block.attn.to_out.0", i),
           // X-block (image self-attention)
           format!("joint_blocks.{}.x_block.attn.to_q", i),
           format!("joint_blocks.{}.x_block.attn.to_k", i),
           format!("joint_blocks.{}.x_block.attn.to_v", i),
           format!("joint_blocks.{}.x_block.attn.to_out.0", i),
           // Y-block (text self-attention)
           format!("joint_blocks.{}.y_block.attn.to_q", i),
           format!("joint_blocks.{}.y_block.attn.to_k", i),
           format!("joint_blocks.{}.y_block.attn.to_v", i),
           format!("joint_blocks.{}.y_block.attn.to_out.0", i),
       ]);
   }
   
   modules
}

/// SD 1.5 target modules
pub fn sd15_target_modules() -> Vec<String> {
   let mut modules = vec![];
   
   // Input blocks (12 blocks)
   for i in [1, 2, 4, 5, 7, 8, 10, 11] {
       modules.extend(vec![
           format!("model.diffusion_model.input_blocks.{}.1.transformer_blocks.0.attn1.to_q", i),
           format!("model.diffusion_model.input_blocks.{}.1.transformer_blocks.0.attn1.to_k", i),
           format!("model.diffusion_model.input_blocks.{}.1.transformer_blocks.0.attn1.to_v", i),
           format!("model.diffusion_model.input_blocks.{}.1.transformer_blocks.0.attn1.to_out.0", i),
           format!("model.diffusion_model.input_blocks.{}.1.transformer_blocks.0.attn2.to_q", i),
           format!("model.diffusion_model.input_blocks.{}.1.transformer_blocks.0.attn2.to_k", i),
           format!("model.diffusion_model.input_blocks.{}.1.transformer_blocks.0.attn2.to_v", i),
           format!("model.diffusion_model.input_blocks.{}.1.transformer_blocks.0.attn2.to_out.0", i),
       ]);
   }
   
   // Middle block
   modules.extend(vec![
       "model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_q",
       "model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_k",
       "model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_v",
       "model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_out.0",
       "model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_q",
       "model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_k",
       "model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_v",
       "model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_out.0",
   ].into_iter().map(String::from));
   
   // Output blocks (12 blocks)
   for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] {
       let block_idx = if i < 3 { 1 } else { 2 };
       modules.extend(vec![
           format!("model.diffusion_model.output_blocks.{}.{}.transformer_blocks.0.attn1.to_q", i, block_idx),
           format!("model.diffusion_model.output_blocks.{}.{}.transformer_blocks.0.attn1.to_k", i, block_idx),
           format!("model.diffusion_model.output_blocks.{}.{}.transformer_blocks.0.attn1.to_v", i, block_idx),
           format!("model.diffusion_model.output_blocks.{}.{}.transformer_blocks.0.attn1.to_out.0", i, block_idx),
           format!("model.diffusion_model.output_blocks.{}.{}.transformer_blocks.0.attn2.to_q", i, block_idx),
           format!("model.diffusion_model.output_blocks.{}.{}.transformer_blocks.0.attn2.to_k", i, block_idx),
           format!("model.diffusion_model.output_blocks.{}.{}.transformer_blocks.0.attn2.to_v", i, block_idx),
           format!("model.diffusion_model.output_blocks.{}.{}.transformer_blocks.0.attn2.to_out.0", i, block_idx),
       ]);
   }
   
   modules
}

/// Get target modules based on model type
pub fn get_target_modules(model_type: &str) -> Vec<String> {
   match model_type {
       "sd3.5" | "sd3" => sd35_target_modules(),
       "sdxl" => sdxl_target_modules(),
       "sd1.5" | "sd1.4" | "sd2" | "sd2.1" => sd15_target_modules(),
       _ => sdxl_target_modules(), // Default to SDXL
   }
}

/// Detect SD model type from state dict keys
pub fn detect_sd_model_type(state_dict_keys: &[String]) -> String {
   if state_dict_keys.iter().any(|k| k.contains("joint_blocks")) {
       "sd3.5".to_string()
   } else if state_dict_keys.iter().any(|k| k.contains("down_blocks") && !k.contains("diffusion_model")) {
       "sdxl".to_string()
   } else if state_dict_keys.iter().any(|k| k.contains("model.diffusion_model")) {
       "sd1.5".to_string()
   } else {
       "unknown".to_string()
   }
}

#[cfg(test)]
mod tests {
   use super::*;
   
   #[test]
   fn test_lora_forward() -> Result<()> {
       let device = Device::Cpu;
       let dtype = DType::F32;
       let batch = 2;
       let seq_len = 77;
       let in_features = 1024;
       let out_features = 1024;
       
       let lora = SDLoRAModule::new(in_features, out_features, 16, 16.0, device.clone(), dtype)?;
       let input = Tensor::randn(0.0, 1.0, (batch, seq_len, in_features), &device)?;
       let output = lora.forward(&input)?;
       
       assert_eq!(output.dims(), &[batch, seq_len, out_features]);
       Ok(())
   }
   
   #[test]
   fn test_lokr_forward() -> Result<()> {
       let device = Device::Cpu;
       let dtype = DType::F32;
       let batch = 2;
       let in_features = 256;
       let out_features = 256;
       
       let lokr = SDLoKrModule::new(in_features, out_features, 8, 8.0, Some(16), device.clone(), dtype)?;
       let input = Tensor::randn(0.0, 1.0, (batch, in_features), &device)?;
       let output = lokr.forward(&input)?;
       
       assert_eq!(output.dims(), &[batch, out_features]);
       Ok(())
   }
   
   #[test]
   fn test_model_detection() {
       let sdxl_keys = vec!["down_blocks.0.attentions.0.to_q.weight".to_string()];
       assert_eq!(detect_sd_model_type(&sdxl_keys), "sdxl");
       
       let sd35_keys = vec!["joint_blocks.0.x_block.attn.to_q.weight".to_string()];
       assert_eq!(detect_sd_model_type(&sd35_keys), "sd3.5");
       
       let sd15_keys = vec!["model.diffusion_model.input_blocks.0.0.weight".to_string()];
       assert_eq!(detect_sd_model_type(&sd15_keys), "sd1.5");
   }
}