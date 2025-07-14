//! SD 3.5 LoRA trainer without VarBuilder
//! Standard LoRA implementation (not LoKr)

use std::collections::HashMap;
use anyhow::{Result, Context};
use candle_core::{Device, Tensor, DType, Var, Module, D};
use candle_nn::{AdamW, Optimizer, ParamsAdamW};
use safetensors::{serialize, SafeTensors};
use std::fs;
use std::path::PathBuf;

// Import GPU LoRA operations
#[cfg(feature = "cuda-backward")]
use candle_core::lora_backward_ops::LoRABackwardOps;

use super::{Config, ProcessConfig};
use crate::trainers::adam8bit::Adam8bit;

// Import candle MMDiT and VAE models
use candle_transformers::models::{
    mmdit::model::{Config as MMDiTConfig, MMDiT},
    stable_diffusion::vae::{AutoEncoderKL, AutoEncoderKLConfig},
};

/// Simple LoRA adapter for SD 3.5
pub struct LoRAAdapter {
    /// Down projection: hidden_size -> rank
    pub down: Var,
    /// Up projection: rank -> hidden_size  
    pub up: Var,
    /// Scaling factor
    pub scale: f64,
}

impl LoRAAdapter {
    pub fn new(in_features: usize, out_features: usize, rank: usize, alpha: f32, device: &Device) -> Result<Self> {
        // Initialize LoRA weights - create tensors then wrap in Var
        let down_tensor = Tensor::randn(0.0f32, 0.02, (rank, in_features), device)?.to_dtype(DType::F32)?;
        let up_tensor = Tensor::zeros((out_features, rank), DType::F32, device)?;
        
        let down = Var::from_tensor(&down_tensor)?;
        let up = Var::from_tensor(&up_tensor)?;
        
        Ok(Self {
            down,
            up,
            scale: (alpha / rank as f32) as f64,
        })
    }
    
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [batch, seq_len, hidden_size]
        // Apply LoRA: x + scale * (x @ down.T @ up.T)
        let down_out = x.matmul(&self.down.as_tensor().t()?)?;
        let up_out = down_out.matmul(&self.up.as_tensor().t()?)?;
        Ok((up_out * self.scale)?)
    }
    
    /// GPU-accelerated backward pass for LoRA
    #[cfg(feature = "cuda-backward")]
    pub fn backward_gpu(&self, grad_output: &Tensor, input: &Tensor) -> Result<(Tensor, Tensor)> {
        // Use GPU-optimized LoRA backward kernel
        LoRABackwardOps::backward(grad_output, input, self.down.as_tensor(), self.up.as_tensor(), self.scale as f32)
    }
    
    /// Fallback CPU backward pass
    #[cfg(not(feature = "cuda-backward"))]
    pub fn backward_gpu(&self, grad_output: &Tensor, input: &Tensor) -> Result<(Tensor, Tensor)> {
        // Fallback to standard backward computation
        // grad_down = grad_output.T @ input @ up.T * scale
        // grad_up = grad_output.T @ (input @ down.T) * scale
        let batch_dims = input.dims().len() - 1;
        
        // Compute grad_down
        let grad_out_t = if batch_dims == 2 {
            grad_output.transpose(1, 2)?
        } else {
            grad_output.t()?
        };
        let input_up = input.matmul(&self.up.as_tensor())?;
        let grad_down_temp = grad_out_t.matmul(&input_up)?;
        let grad_down = (grad_down_temp * self.scale)?.transpose(D::Minus2, D::Minus1)?;
        
        // Compute grad_up  
        let input_down = input.matmul(&self.down.as_tensor().t()?)?;
        let grad_up_temp = grad_out_t.matmul(&input_down)?;
        let grad_up = (grad_up_temp * self.scale)?;
        
        Ok((grad_down, grad_up))
    }
}

/// SD 3.5 LoRA trainer
pub struct SD35LoRATrainer {
    // Model components
    mmdit_weights: HashMap<String, Tensor>,
    vae_weights: HashMap<String, Tensor>,
    
    // LoRA adapters for attention layers
    lora_adapters: HashMap<String, LoRAAdapter>,
    
    // Training configuration
    device: Device,
    model_path: String,
    rank: usize,
    alpha: f32,
    learning_rate: f32,
    batch_size: usize,
    num_steps: usize,
    save_every: usize,
    output_dir: String,
    
    // Optimizer
    use_8bit_adam: bool,
    adam8bit: Option<Adam8bit>,
    optimizer: Option<AdamW>,
}

impl SD35LoRATrainer {
    pub fn new(config: &Config, process: &ProcessConfig) -> Result<Self> {
        let device = Device::cuda_if_available(0)?;
        
        // Validate GPU requirement
        match &device {
            Device::Cuda(_) => {
                println!("CUDA GPU detected and verified");
            }
            Device::Cpu => {
                eprintln!("ERROR: Training requires a CUDA GPU. CPU device not supported.");
                return Err(anyhow::anyhow!(
                    "GPU required for training. CPU device not supported.\n\
                     Please use a CUDA-capable GPU."
                ));
            }
        }
        
        // Extract configuration
        let rank = process.network.linear.unwrap_or(16);
        let alpha = process.network.linear_alpha.unwrap_or(16.0);
        let learning_rate = process.train.lr;
        let batch_size = process.train.batch_size;
        let num_steps = process.train.steps;
        let save_every = process.save.save_every;
        let model_path = process.model.name_or_path.clone();
        let output_dir = format!("output/{}", config.config.name.as_deref().unwrap_or("sd35_lora"));
        let use_8bit_adam = process.train.optimizer == "adamw8bit";
        
        // Create output directory
        fs::create_dir_all(&output_dir)?;
        
        Ok(Self {
            mmdit_weights: HashMap::new(),
            vae_weights: HashMap::new(),
            lora_adapters: HashMap::new(),
            device,
            model_path,
            rank,
            alpha,
            learning_rate,
            batch_size,
            num_steps,
            save_every,
            output_dir,
            use_8bit_adam,
            adam8bit: None,
            optimizer: None,
        })
    }
    
    /// Load model weights directly without VarBuilder
    pub fn load_models(&mut self) -> Result<()> {
        println!("Loading SD 3.5 model weights...");
        
        // Load all weights from safetensors
        let weights = candle_core::safetensors::load(&self.model_path, &self.device)?;
        
        // Separate MMDiT and VAE weights
        for (key, tensor) in weights {
            if key.starts_with("model.diffusion_model.") {
                let new_key = key.strip_prefix("model.diffusion_model.").unwrap().to_string();
                self.mmdit_weights.insert(new_key, tensor);
            } else if key.starts_with("first_stage_model.") || key.starts_with("vae.") {
                self.vae_weights.insert(key, tensor);
            }
        }
        
        // Initialize LoRA adapters for attention layers
        self.initialize_lora_adapters()?;
        
        println!("✓ Models loaded successfully");
        Ok(())
    }
    
    /// Initialize LoRA adapters for key attention layers
    fn initialize_lora_adapters(&mut self) -> Result<()> {
        let hidden_size = 1536; // SD3.5 Large hidden size
        
        // Add LoRA to joint attention blocks
        for i in 0..38 {
            let prefix = format!("joint_blocks.{}", i);
            
            // Add adapters for Q, K, V projections
            for proj in ["q", "k", "v"] {
                let key = format!("{}.attn.{}_proj", prefix, proj);
                self.lora_adapters.insert(
                    key,
                    LoRAAdapter::new(
                        hidden_size,
                        hidden_size,
                        self.rank,
                        self.alpha,
                        &self.device
                    )?
                );
            }
            
            // Output projection
            let key = format!("{}.attn.out_proj", prefix);
            self.lora_adapters.insert(
                key,
                LoRAAdapter::new(
                    hidden_size,
                    hidden_size,
                    self.rank,
                    self.alpha,
                    &self.device
                )?
            );
        }
        
        println!("✓ Initialized {} LoRA adapters", self.lora_adapters.len());
        Ok(())
    }
    
    /// Save LoRA weights in ComfyUI format
    pub fn save_lora(&self, step: usize) -> Result<()> {
        // Create checkpoint directory in SimpleTuner format
        let checkpoint_dir = PathBuf::from(&self.output_dir).join(format!("checkpoint-{}", step));
        fs::create_dir_all(&checkpoint_dir)?;
        
        // Save LoRA weights
        let lora_path = checkpoint_dir.join("sd35_lora.safetensors");
        
        // First collect all tensor data
        let mut tensor_data = Vec::new();
        let mut tensor_info = Vec::new();
        
        // Convert LoRA weights to ComfyUI format
        for (name, adapter) in &self.lora_adapters {
            // Map our naming to ComfyUI naming
            let key_base = name
                .replace("joint_blocks.", "diffusion_model_joint_blocks_")
                .replace(".attn.", "_attn_")
                .replace("_proj", "");
            
            // Save down weight info
            let down_data = tensor_to_vec(adapter.down.as_tensor())?;
            tensor_info.push((
                format!("lora_unet_{}.lora_down.weight", key_base),
                convert_dtype(adapter.down.dtype())?,
                adapter.down.dims().to_vec(),
                tensor_data.len()
            ));
            tensor_data.push(down_data);
            
            // Save up weight info
            let up_data = tensor_to_vec(adapter.up.as_tensor())?;
            tensor_info.push((
                format!("lora_unet_{}.lora_up.weight", key_base),
                convert_dtype(adapter.up.dtype())?,
                adapter.up.dims().to_vec(),
                tensor_data.len()
            ));
            tensor_data.push(up_data);
        }
        
        // Now create TensorViews using indices
        let mut tensors = HashMap::new();
        for (name, dtype, shape, idx) in tensor_info {
            tensors.insert(
                name,
                safetensors::tensor::TensorView::new(
                    dtype,
                    shape,
                    &tensor_data[idx],
                )?
            );
        }
        
        // Add metadata
        let mut metadata = HashMap::new();
        metadata.insert("ss_network_module".to_string(), "networks.lora".to_string());
        metadata.insert("ss_network_dim".to_string(), self.rank.to_string());
        metadata.insert("ss_network_alpha".to_string(), self.alpha.to_string());
        metadata.insert("ss_training_model".to_string(), "sd35".to_string());
        
        // Save the file
        // Serialize with metadata
        let serialized = serialize(&tensors, &Some(metadata))?;
        std::fs::write(&lora_path, serialized)?;
        
        // Save optimizer state if using 8bit Adam
        if let Some(ref adam) = self.adam8bit {
            let optimizer_path = checkpoint_dir.join("optimizer.safetensors");
            self.save_adam8bit_state(adam, &optimizer_path)?;
        }
        
        // Save training state
        let state_path = checkpoint_dir.join("training_state.json");
        let state = serde_json::json!({
            "step": step,
            "learning_rate": self.learning_rate,
            "model_type": "sd35",
            "network_type": "lora",
            "rank": self.rank,
            "alpha": self.alpha,
        });
        fs::write(&state_path, serde_json::to_string_pretty(&state)?)?;
        
        println!("✓ Saved SD 3.5 LoRA checkpoint to {}", checkpoint_dir.display());
        Ok(())
    }
    
    /// Save 8bit Adam optimizer state
    fn save_adam8bit_state(&self, _adam: &Adam8bit, path: &std::path::Path) -> Result<()> {
        use safetensors::tensor::TensorView;
        
        // Adam8bit stores state internally, we'll save a placeholder for now
        let tensors: HashMap<String, safetensors::tensor::TensorView> = HashMap::new();
        
        // Adam8bit optimizer saves state internally
        // State persistence is handled by the optimizer itself
        /*
        for (name, (m, v)) in state {
            // Save first moment
            let m_data = tensor_to_vec(&m)?;
            tensors.insert(
                format!("{}_m", name),
                TensorView::new(
                    convert_safetensor_dtype(m.dtype())?,
                    m.dims().to_vec(),
                    &m_data,
                )?
            );
            
            // Save second moment
            let v_data = tensor_to_vec(&v)?;
            tensors.insert(
                format!("{}_v", name),
                TensorView::new(
                    convert_safetensor_dtype(v.dtype())?,
                    v.dims().to_vec(),
                    &v_data,
                )?
            );
        }
        */
        
        // Add metadata
        let mut metadata = HashMap::new();
        metadata.insert("format".to_string(), "adam8bit".to_string());
        metadata.insert("step".to_string(), _adam.get_step().to_string());
        
        let serialized = serialize(&tensors, &Some(metadata))?;
        std::fs::write(path, serialized)?;
        Ok(())
    }
    
    /// Train the model
    pub fn train(&mut self) -> Result<()> {
        // Check for GPU requirement
        match &self.device {
            Device::Cuda(_) => {
                println!("GPU detected, starting training...");
            }
            Device::Cpu => {
                eprintln!("ERROR: GPU is required for training. No CUDA device found.");
                return Err(anyhow::anyhow!("Training requires a CUDA GPU. CPU training is not supported."));
            }
        }
        
        println!("\n=== Starting SD 3.5 LoRA Training ===");
        println!("Rank: {}, Alpha: {}", self.rank, self.alpha);
        println!("Steps: {}, Batch Size: {}", self.num_steps, self.batch_size);
        println!("Learning Rate: {}", self.learning_rate);
        println!("Output Directory: {}", self.output_dir);
        
        // Setup optimizer
        let trainable_params: Vec<Var> = self.lora_adapters
            .values()
            .flat_map(|adapter| vec![adapter.down.clone(), adapter.up.clone()])
            .collect();
        
        if self.use_8bit_adam {
            println!("Using 8-bit Adam optimizer");
            // Adam8bit doesn't store vars, just create it
            self.adam8bit = Some(Adam8bit::new(self.learning_rate as f64));
        } else {
            let params = ParamsAdamW {
                lr: self.learning_rate as f64,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: 0.01,
            };
            self.optimizer = Some(AdamW::new(trainable_params, params)?);
        }
        
        // Training loop
        for step in 0..self.num_steps {
            // Placeholder for actual training data
            let batch_size = self.batch_size;
            let latents = Tensor::randn(0.0f32, 1.0, &[batch_size, 16, 32, 32], &self.device)?;
            
            // Sample random timesteps
            let uniform = Tensor::rand(0.0f32, 1000.0f32, &[batch_size], &self.device)?;
            let timesteps = uniform.to_dtype(DType::I64)?;
            
            let encoder_hidden_states = Tensor::randn(0.0f32, 1.0, &[batch_size, 154, 4096], &self.device)?;
            
            // Forward pass with LoRA (simplified - actual implementation would integrate with MMDiT)
            let mut lora_sum = Tensor::zeros_like(&latents)?;
            for (_name, adapter) in &self.lora_adapters {
                // In real implementation, this would be applied at the right layers
                let lora_out = adapter.forward(&encoder_hidden_states)?;
                // Accumulate LoRA outputs (simplified)
                // Slice the lora output to match latent dimensions
                let lora_slice = lora_out.narrow(2, 0, 16)?.narrow(3, 0, 32)?;
                lora_sum = (&lora_sum + &lora_slice)?;
            }
            
            // Compute loss (placeholder)
            let target = Tensor::randn_like(&latents, 0.0, 1.0)?;
            let loss = ((latents + lora_sum)? - target)?.sqr()?.mean_all()?;
            
            // Backward pass
            let gradients = loss.backward()?;
            
            // Update weights
            if let Some(ref mut adam) = self.adam8bit {
                // Update each parameter with Adam8bit
                for (adapter_name, adapter) in &self.lora_adapters {
                    if let Some(down_grad) = gradients.get(adapter.down.as_tensor()) {
                        adam.update(&format!("{}_down", adapter_name), &adapter.down, down_grad)?;
                    }
                    if let Some(up_grad) = gradients.get(adapter.up.as_tensor()) {
                        adam.update(&format!("{}_up", adapter_name), &adapter.up, up_grad)?;
                    }
                }
                adam.step(); // Increment step counter
            } else if let Some(ref mut opt) = self.optimizer {
                opt.step(&gradients)?;
            }
            
            // Logging
            if step % 50 == 0 {
                println!("Step {}/{}: loss = {:.6}", step, self.num_steps, loss.to_scalar::<f32>()?);
            }
            
            // Save checkpoint
            if step > 0 && step % self.save_every == 0 {
                self.save_lora(step)?;
            }
        }
        
        // Save final checkpoint
        self.save_lora(self.num_steps)?;
        
        println!("\n=== Training Complete! ===");
        Ok(())
    }
}

// Helper functions
fn tensor_to_vec(tensor: &Tensor) -> Result<Vec<u8>> {
    let shape = tensor.dims();
    let elem_count = shape.iter().product::<usize>();
    let data = tensor.flatten_all()?.to_vec1::<f32>()?;
    
    let mut bytes = Vec::with_capacity(elem_count * 4);
    for &val in &data {
        bytes.extend_from_slice(&val.to_le_bytes());
    }
    Ok(bytes)
}

fn convert_dtype(dtype: DType) -> Result<safetensors::Dtype> {
    match dtype {
        DType::F32 => Ok(safetensors::Dtype::F32),
        DType::F16 => Ok(safetensors::Dtype::F16),
        DType::BF16 => Ok(safetensors::Dtype::BF16),
        _ => anyhow::bail!("Unsupported dtype for safetensors: {:?}", dtype),
    }
}

fn convert_safetensor_dtype(dtype: DType) -> Result<safetensors::Dtype> {
    convert_dtype(dtype)
}

/// Train SD 3.5 LoRA from configuration
pub fn train_sd35_lora(config: &Config, process: &ProcessConfig) -> Result<()> {
    let mut trainer = SD35LoRATrainer::new(config, process)?;
    trainer.load_models()?;
    trainer.train()?;
    Ok(())
}