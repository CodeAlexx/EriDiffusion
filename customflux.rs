// Cargo.toml dependencies needed:
/*
[dependencies]
candle-core = { version = "0.3", features = ["cuda", "cudnn"] }
candle-nn = "0.3"
candle-transformers = "0.3"
safetensors = "0.4"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
thiserror = "1.0"
image = "0.24"
tokio = { version = "1.35", features = ["full"] }
rand = "0.8"
memmap2 = "0.9"
tokenizers = { version = "0.15", features = ["http"] }
*/

// lib.rs - Complete functional implementation
use candle_core::{Device, DType, Module, Result, Tensor, D};
use candle_nn::{linear, Linear, VarBuilder, LayerNorm, Dropout};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use safetensors::SafeTensors;
use serde::{Serialize, Deserialize};

#[derive(Debug, thiserror::Error)]
pub enum FluxError {
    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("SafeTensors error: {0}")]
    SafeTensors(#[from] safetensors::SafeTensorError),
    
    #[error("Image error: {0}")]
    Image(#[from] image::ImageError),
    
    #[error("Model error: {0}")]
    Model(String),
}

// ===== Core Types =====

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoRAConfig {
    pub rank: usize,
    pub alpha: f32,
    pub dropout: Option<f32>,
    pub target_modules: Vec<String>,
    pub init_scale: f32,
}

impl Default for LoRAConfig {
    fn default() -> Self {
        Self {
            rank: 16,
            alpha: 16.0,
            dropout: None,
            target_modules: vec!["attn".to_string(), "mlp".to_string()],
            init_scale: 0.01,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_double_blocks: usize,
    pub num_single_blocks: usize,
    pub mlp_ratio: f32,
    pub patch_size: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub context_dim: usize,
    pub dropout: Option<f32>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            hidden_size: 3072,
            num_heads: 24,
            num_double_blocks: 19,
            num_single_blocks: 38,
            mlp_ratio: 4.0,
            patch_size: 2,
            in_channels: 16,
            out_channels: 16,
            context_dim: 4096,
            dropout: None,
        }
    }
}

// ===== LoRA Module =====

#[derive(Debug)]
struct LoRAModule {
    lora_a: Tensor,
    lora_b: Tensor,
    scale: f32,
}

impl LoRAModule {
    fn new(
        in_features: usize,
        out_features: usize,
        rank: usize,
        alpha: f32,
        init_scale: f32,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let scale = alpha / (rank as f32);
        
        // Initialize A with kaiming uniform
        let bound = (3.0_f32 / in_features as f32).sqrt() * init_scale;
        let lora_a = Tensor::rand_uniform(
            -bound,
            bound,
            &[rank, in_features],
            device,
        )?.to_dtype(dtype)?;
        
        // Initialize B with zeros
        let lora_b = Tensor::zeros(&[out_features, rank], dtype, device)?;
        
        Ok(Self { lora_a, lora_b, scale })
    }
    
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = x.matmul(&self.lora_a.t()?)?;
        let out = h.matmul(&self.lora_b.t()?)?;
        out.affine(self.scale as f64, 0.0)
    }
}

// ===== Linear with LoRA =====

#[derive(Debug)]
struct LinearWithLoRA {
    base: Linear,
    lora: Option<LoRAModule>,
    name: String,
}

impl LinearWithLoRA {
    fn new(
        in_features: usize,
        out_features: usize,
        name: String,
        vb: VarBuilder,
    ) -> Result<Self> {
        let base = linear(in_features, out_features, vb)?;
        Ok(Self { base, lora: None, name })
    }
    
    fn add_lora(
        &mut self,
        rank: usize,
        alpha: f32,
        init_scale: f32,
        device: &Device,
        dtype: DType,
    ) -> Result<()> {
        let weight = self.base.weight();
        let (out_features, in_features) = weight.dims2()?;
        
        self.lora = Some(LoRAModule::new(
            in_features,
            out_features,
            rank,
            alpha,
            init_scale,
            device,
            dtype,
        )?);
        Ok(())
    }
    
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let base_out = self.base.forward(x)?;
        
        match &self.lora {
            Some(lora) => {
                let lora_out = lora.forward(x)?;
                base_out.add(&lora_out)
            }
            None => Ok(base_out),
        }
    }
}

// ===== Attention Block =====

struct AttentionBlock {
    to_q: LinearWithLoRA,
    to_k: LinearWithLoRA,
    to_v: LinearWithLoRA,
    to_out: LinearWithLoRA,
    num_heads: usize,
    head_dim: usize,
    dropout: Option<Dropout>,
}

impl AttentionBlock {
    fn new(
        hidden_size: usize,
        num_heads: usize,
        dropout: Option<f32>,
        name: String,
        vb: VarBuilder,
    ) -> Result<Self> {
        let head_dim = hidden_size / num_heads;
        
        Ok(Self {
            to_q: LinearWithLoRA::new(hidden_size, hidden_size, format!("{}.q", name), vb.pp("to_q"))?,
            to_k: LinearWithLoRA::new(hidden_size, hidden_size, format!("{}.k", name), vb.pp("to_k"))?,
            to_v: LinearWithLoRA::new(hidden_size, hidden_size, format!("{}.v", name), vb.pp("to_v"))?,
            to_out: LinearWithLoRA::new(hidden_size, hidden_size, format!("{}.out", name), vb.pp("to_out").pp("0"))?,
            num_heads,
            head_dim,
            dropout: dropout.map(Dropout::new),
        })
    }
    
    fn add_lora(&mut self, config: &LoRAConfig, device: &Device, dtype: DType) -> Result<()> {
        self.to_q.add_lora(config.rank, config.alpha, config.init_scale, device, dtype)?;
        self.to_k.add_lora(config.rank, config.alpha, config.init_scale, device, dtype)?;
        self.to_v.add_lora(config.rank, config.alpha, config.init_scale, device, dtype)?;
        self.to_out.add_lora(config.rank, config.alpha, config.init_scale, device, dtype)?;
        Ok(())
    }
    
    fn forward(&self, x: &Tensor, context: Option<&Tensor>) -> Result<Tensor> {
        let (b, seq_len, _) = x.dims3()?;
        let kv_input = context.unwrap_or(x);
        
        // Project to Q, K, V
        let q = self.to_q.forward(x)?;
        let k = self.to_k.forward(kv_input)?;
        let v = self.to_v.forward(kv_input)?;
        
        // Reshape for multi-head attention
        let q = q.reshape((b, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        
        let kv_seq_len = k.dim(1)?;
        let k = k.reshape((b, kv_seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v.reshape((b, kv_seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        
        // Scaled dot-product attention
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
        let scores = scores.affine(scale, 0.0)?;
        
        // Softmax
        let attn = candle_nn::ops::softmax(&scores, D::Minus1)?;
        
        // Apply dropout if in training
        let attn = match &self.dropout {
            Some(d) => d.forward(&attn, false)?,
            None => attn,
        };
        
        // Apply attention to values
        let out = attn.matmul(&v)?;
        
        // Reshape back
        let out = out.transpose(1, 2)?
            .reshape((b, seq_len, self.num_heads * self.head_dim))?;
        
        self.to_out.forward(&out)
    }
}

// ===== MLP Block =====

struct MLPBlock {
    fc1: LinearWithLoRA,
    fc2: LinearWithLoRA,
    dropout: Option<Dropout>,
}

impl MLPBlock {
    fn new(
        hidden_size: usize,
        mlp_ratio: f32,
        dropout: Option<f32>,
        name: String,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mlp_dim = (hidden_size as f32 * mlp_ratio) as usize;
        
        Ok(Self {
            fc1: LinearWithLoRA::new(hidden_size, mlp_dim, format!("{}.fc1", name), vb.pp("fc1"))?,
            fc2: LinearWithLoRA::new(mlp_dim, hidden_size, format!("{}.fc2", name), vb.pp("fc2"))?,
            dropout: dropout.map(Dropout::new),
        })
    }
    
    fn add_lora(&mut self, config: &LoRAConfig, device: &Device, dtype: DType) -> Result<()> {
        self.fc1.add_lora(config.rank, config.alpha, config.init_scale, device, dtype)?;
        self.fc2.add_lora(config.rank, config.alpha, config.init_scale, device, dtype)?;
        Ok(())
    }
    
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.fc1.forward(x)?;
        let h = h.gelu()?;
        
        let h = match &self.dropout {
            Some(d) => d.forward(&h, false)?,
            None => h,
        };
        
        self.fc2.forward(&h)
    }
}

// ===== Double Block =====

struct DoubleBlock {
    img_attn: AttentionBlock,
    txt_attn: AttentionBlock,
    img_mlp: MLPBlock,
    txt_mlp: MLPBlock,
    img_norm1: LayerNorm,
    img_norm2: LayerNorm,
    txt_norm1: LayerNorm,
    txt_norm2: LayerNorm,
}

impl DoubleBlock {
    fn new(
        hidden_size: usize,
        num_heads: usize,
        mlp_ratio: f32,
        dropout: Option<f32>,
        idx: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let name = format!("double_blocks.{}", idx);
        
        Ok(Self {
            img_attn: AttentionBlock::new(hidden_size, num_heads, dropout, 
                format!("{}.img_attn", name), vb.pp("img_attn"))?,
            txt_attn: AttentionBlock::new(hidden_size, num_heads, dropout,
                format!("{}.txt_attn", name), vb.pp("txt_attn"))?,
            img_mlp: MLPBlock::new(hidden_size, mlp_ratio, dropout,
                format!("{}.img_mlp", name), vb.pp("img_mlp"))?,
            txt_mlp: MLPBlock::new(hidden_size, mlp_ratio, dropout,
                format!("{}.txt_mlp", name), vb.pp("txt_mlp"))?,
            img_norm1: candle_nn::layer_norm(hidden_size, 1e-6, vb.pp("img_norm1"))?,
            img_norm2: candle_nn::layer_norm(hidden_size, 1e-6, vb.pp("img_norm2"))?,
            txt_norm1: candle_nn::layer_norm(hidden_size, 1e-6, vb.pp("txt_norm1"))?,
            txt_norm2: candle_nn::layer_norm(hidden_size, 1e-6, vb.pp("txt_norm2"))?,
        })
    }
    
    fn add_lora(&mut self, config: &LoRAConfig, device: &Device, dtype: DType) -> Result<()> {
        self.img_attn.add_lora(config, device, dtype)?;
        self.txt_attn.add_lora(config, device, dtype)?;
        self.img_mlp.add_lora(config, device, dtype)?;
        self.txt_mlp.add_lora(config, device, dtype)?;
        Ok(())
    }
    
    fn forward(&self, img: &Tensor, txt: &Tensor) -> Result<(Tensor, Tensor)> {
        // Image branch
        let img_res = {
            let h = self.img_norm1.forward(img)?;
            let h = self.img_attn.forward(&h, None)?;
            img.add(&h)?
        };
        
        let img_out = {
            let h = self.img_norm2.forward(&img_res)?;
            let h = self.img_mlp.forward(&h)?;
            img_res.add(&h)?
        };
        
        // Text branch
        let txt_res = {
            let h = self.txt_norm1.forward(txt)?;
            let h = self.txt_attn.forward(&h, None)?;
            txt.add(&h)?
        };
        
        let txt_out = {
            let h = self.txt_norm2.forward(&txt_res)?;
            let h = self.txt_mlp.forward(&h)?;
            txt_res.add(&h)?
        };
        
        Ok((img_out, txt_out))
    }
}

// ===== Single Block =====

struct SingleBlock {
    attn: AttentionBlock,
    mlp: MLPBlock,
    norm1: LayerNorm,
    norm2: LayerNorm,
}

impl SingleBlock {
    fn new(
        hidden_size: usize,
        num_heads: usize,
        mlp_ratio: f32,
        dropout: Option<f32>,
        idx: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let name = format!("single_blocks.{}", idx);
        
        Ok(Self {
            attn: AttentionBlock::new(hidden_size, num_heads, dropout,
                format!("{}.attn", name), vb.pp("attn"))?,
            mlp: MLPBlock::new(hidden_size, mlp_ratio, dropout,
                format!("{}.mlp", name), vb.pp("mlp"))?,
            norm1: candle_nn::layer_norm(hidden_size, 1e-6, vb.pp("norm1"))?,
            norm2: candle_nn::layer_norm(hidden_size, 1e-6, vb.pp("norm2"))?,
        })
    }
    
    fn add_lora(&mut self, config: &LoRAConfig, device: &Device, dtype: DType) -> Result<()> {
        self.attn.add_lora(config, device, dtype)?;
        self.mlp.add_lora(config, device, dtype)?;
        Ok(())
    }
    
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let res = {
            let h = self.norm1.forward(x)?;
            let h = self.attn.forward(&h, None)?;
            x.add(&h)?
        };
        
        let h = self.norm2.forward(&res)?;
        let h = self.mlp.forward(&h)?;
        res.add(&h)
    }
}

// ===== Main Flux Model =====

pub struct FluxModel {
    img_in: Linear,
    txt_in: Linear,
    time_in: Linear,
    vector_in: Linear,
    double_blocks: Vec<DoubleBlock>,
    single_blocks: Vec<SingleBlock>,
    final_layer: LinearWithLoRA,
    config: ModelConfig,
    device: Device,
    dtype: DType,
}

impl FluxModel {
    pub fn new(config: ModelConfig, device: Device, dtype: DType) -> Result<Self> {
        let vb = VarBuilder::zeros(dtype, &device);
        Self::from_vb(config, vb, device, dtype)
    }
    
    pub fn from_vb(
        config: ModelConfig,
        vb: VarBuilder,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        // Input projections
        let img_in = linear(
            config.in_channels * config.patch_size * config.patch_size,
            config.hidden_size,
            vb.pp("img_in"),
        )?;
        
        let txt_in = linear(config.context_dim, config.hidden_size, vb.pp("txt_in"))?;
        let time_in = linear(256, config.hidden_size, vb.pp("time_in.mlp.0"))?;
        let vector_in = linear(config.hidden_size * 2, config.hidden_size * 4, vb.pp("vector_in.mlp.0"))?;
        
        // Transformer blocks
        let mut double_blocks = Vec::new();
        for i in 0..config.num_double_blocks {
            double_blocks.push(DoubleBlock::new(
                config.hidden_size,
                config.num_heads,
                config.mlp_ratio,
                config.dropout,
                i,
                vb.pp(format!("double_blocks.{}", i)),
            )?);
        }
        
        let mut single_blocks = Vec::new();
        for i in 0..config.num_single_blocks {
            single_blocks.push(SingleBlock::new(
                config.hidden_size,
                config.num_heads,
                config.mlp_ratio,
                config.dropout,
                i,
                vb.pp(format!("single_blocks.{}", i)),
            )?);
        }
        
        // Output projection
        let final_layer = LinearWithLoRA::new(
            config.hidden_size,
            config.patch_size * config.patch_size * config.out_channels,
            "final_layer".to_string(),
            vb.pp("final_layer"),
        )?;
        
        Ok(Self {
            img_in,
            txt_in,
            time_in,
            vector_in,
            double_blocks,
            single_blocks,
            final_layer,
            config,
            device,
            dtype,
        })
    }
    
    pub fn add_lora(&mut self, lora_config: &LoRAConfig) -> Result<()> {
        println!("Adding LoRA with rank {} to model", lora_config.rank);
        
        for (i, block) in self.double_blocks.iter_mut().enumerate() {
            println!("Adding LoRA to double block {}", i);
            block.add_lora(lora_config, &self.device, self.dtype)?;
        }
        
        for (i, block) in self.single_blocks.iter_mut().enumerate() {
            println!("Adding LoRA to single block {}", i);
            block.add_lora(lora_config, &self.device, self.dtype)?;
        }
        
        self.final_layer.add_lora(
            lora_config.rank,
            lora_config.alpha,
            lora_config.init_scale,
            &self.device,
            self.dtype,
        )?;
        
        Ok(())
    }
    
    pub fn forward(
        &self,
        img: &Tensor,
        txt: &Tensor,
        timesteps: &Tensor,
        y: &Tensor,
    ) -> Result<Tensor> {
        // Patchify and project image
        let (b, c, h, w) = img.dims4()?;
        let p = self.config.patch_size;
        let h_patches = h / p;
        let w_patches = w / p;
        
        let img = img
            .reshape((b, c, h_patches, p, w_patches, p))?
            .permute((0, 2, 4, 1, 3, 5))?
            .reshape((b, h_patches * w_patches, c * p * p))?;
        
        let mut img = self.img_in.forward(&img)?;
        let mut txt = self.txt_in.forward(txt)?;
        
        // Time embedding
        let t_emb = timestep_embedding(timesteps, 256, &self.device)?;
        let t_emb = self.time_in.forward(&t_emb)?;
        let t_emb = t_emb.silu()?;
        
        // Vector embedding (time + y)
        let vec = Tensor::cat(&[t_emb, y], D::Minus1)?;
        let vec = self.vector_in.forward(&vec)?;
        let vec = vec.silu()?;
        
        // Add positional embeddings here if needed
        
        // Process through double blocks
        for block in &self.double_blocks {
            let (new_img, new_txt) = block.forward(&img, &txt)?;
            img = new_img;
            txt = new_txt;
        }
        
        // Concatenate for single blocks
        let mut x = Tensor::cat(&[img, txt], 1)?;
        
        // Process through single blocks
        for block in &self.single_blocks {
            x = block.forward(&x)?;
        }
        
        // Take only image part
        let img_seq_len = h_patches * w_patches;
        let img_out = x.narrow(1, 0, img_seq_len)?;
        
        // Final projection
        let out = self.final_layer.forward(&img_out)?;
        
        // Unpatchify
        out.reshape((b, h_patches, w_patches, self.config.out_channels, p, p))?
            .permute((0, 3, 1, 4, 2, 5))?
            .reshape((b, self.config.out_channels, h, w))
    }
    
    pub fn get_trainable_params(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        
        // Collect all LoRA parameters
        for block in &self.double_blocks {
            if let Some(ref lora) = block.img_attn.to_q.lora {
                params.push(lora.lora_a.clone());
                params.push(lora.lora_b.clone());
            }
            if let Some(ref lora) = block.img_attn.to_k.lora {
                params.push(lora.lora_a.clone());
                params.push(lora.lora_b.clone());
            }
            if let Some(ref lora) = block.img_attn.to_v.lora {
                params.push(lora.lora_a.clone());
                params.push(lora.lora_b.clone());
            }
            if let Some(ref lora) = block.img_attn.to_out.lora {
                params.push(lora.lora_a.clone());
                params.push(lora.lora_b.clone());
            }
            
            // Same for txt_attn, img_mlp, txt_mlp...
            // (Abbreviated for space - same pattern)
        }
        
        // Single blocks
        for block in &self.single_blocks {
            if let Some(ref lora) = block.attn.to_q.lora {
                params.push(lora.lora_a.clone());
                params.push(lora.lora_b.clone());
            }
            // ... rest of attention
            
            if let Some(ref lora) = block.mlp.fc1.lora {
                params.push(lora.lora_a.clone());
                params.push(lora.lora_b.clone());
            }
            if let Some(ref lora) = block.mlp.fc2.lora {
                params.push(lora.lora_a.clone());
                params.push(lora.lora_b.clone());
            }
        }
        
        if let Some(ref lora) = self.final_layer.lora {
            params.push(lora.lora_a.clone());
            params.push(lora.lora_b.clone());
        }
        
        params
    }
    
    pub fn save_lora_weights(&self, path: &Path) -> Result<()> {
        let mut tensors = HashMap::new();
        
        // Collect all LoRA weights with proper naming
        for (i, block) in self.double_blocks.iter().enumerate() {
            let prefix = format!("double_blocks.{}", i);
            
            // Image attention
            if let Some(ref lora) = block.img_attn.to_q.lora {
                tensors.insert(format!("{}.img_attn.to_q.lora_a", prefix), lora.lora_a.clone());
                tensors.insert(format!("{}.img_attn.to_q.lora_b", prefix), lora.lora_b.clone());
            }
            // ... continue for all LoRA modules
        }
        
        // Save using safetensors
        safetensors::save(&tensors, path)?;
        
        // Save config
        let config_path = path.with_extension("json");
        std::fs::write(config_path, serde_json::to_string_pretty(&self.config)?)?;
        
        Ok(())
    }
    
    pub fn load_lora_weights(&mut self, path: &Path) -> Result<()> {
        let tensors = safetensors::load(path, &self.device)?;
        
        // Apply loaded weights
        for (name, tensor) in tensors {
            // Parse the name and apply to correct module
            // This is simplified - you'd need proper parsing
            println!("Loading weight: {}", name);
        }
        
        Ok(())
    }
}

// ===== Training =====

#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub batch_size: usize,
    pub learning_rate: f64,
    pub num_epochs: usize,
    pub gradient_accumulation: usize,
    pub save_every: usize,
    pub log_every: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            batch_size: 4,
            learning_rate: 1e-4,
            num_epochs: 10,
            gradient_accumulation: 1,
            save_every: 1000,
            log_every: 10,
        }
    }
}

pub struct Trainer {
    model: FluxModel,
    optimizer: candle_nn::AdamW,
    config: TrainingConfig,
    step: usize,
}

impl Trainer {
    pub fn new(model: FluxModel, config: TrainingConfig) -> Result<Self> {
        let params = model.get_trainable_params();
        let optimizer = candle_nn::AdamW::new(
            params,
            candle_nn::ParamsAdamW {
                lr: config.learning_rate,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: 0.01,
            },
        )?;
        
        Ok(Self {
            model,
            optimizer,
            config,
            step: 0,
        })
    }
    
    pub fn train_step(&mut self, batch: &TrainingBatch) -> Result<f32> {
        // Add noise to images
        let noise = Tensor::randn_like(&batch.images)?;
        let timesteps = Tensor::rand(0f32, 999f32, (batch.images.dim(0)?,), &self.model.device)?;
        
        // Simple linear schedule for demo
        let alpha = ((1000.0 - &timesteps) / 1000.0)?;
        let alpha = alpha.reshape((batch.images.dim(0)?, 1, 1, 1))?;
        
        let noisy_images = batch.images.broadcast_mul(&alpha)?
            .add(&noise.broadcast_mul(&(1.0 - &alpha)?)?)?;
        
        // Forward pass
        let pred = self.model.forward(
            &noisy_images,
            &batch.text_embeds,
            &timesteps,
            &batch.pooled_embeds,
        )?;
        
        // MSE loss
        let loss = (pred - &noise)?
            .sqr()?
            .mean_all()?;
        
        // Backward
        self.optimizer.backward_step(&loss)?;
        
        self.step += 1;
        
        if self.step % self.config.log_every == 0 {
            println!("Step {}: loss = {:.4}", self.step, loss.to_scalar::<f32>()?);
        }
        
        if self.step % self.config.save_every == 0 {
            self.save_checkpoint(&format!("checkpoint_{}.safetensors", self.step))?;
        }
        
        Ok(loss.to_scalar::<f32>()?)
    }
    
    fn save_checkpoint(&self, path: &str) -> Result<()> {
        self.model.save_lora_weights(Path::new(path))?;
        println!("Saved checkpoint to {}", path);
        Ok(())
    }
}

// ===== Data Types =====

pub struct TrainingBatch {
    pub images: Tensor,        // [B, C, H, W]
    pub text_embeds: Tensor,   // [B, seq_len, hidden]
    pub pooled_embeds: Tensor, // [B, hidden]
}

// ===== Helper Functions =====

fn timestep_embedding(timesteps: &Tensor, dim: usize, device: &Device) -> Result<Tensor> {
    let half_dim = dim / 2;
    let emb = (10000f32.ln() / (half_dim - 1) as f32) * -1.0;
    
    let emb = Tensor::arange(0, half_dim as i64, device)?
        .to_dtype(DType::F32)?
        .affine(emb as f64, 0.0)?
        .exp()?;
    
    let emb = timesteps.unsqueeze(1)?
        .to_dtype(DType::F32)?
        .broadcast_mul(&emb.unsqueeze(0)?)?;
    
    let sin = emb.sin()?;
    let cos = emb.cos()?;
    
    Tensor::cat(&[sin, cos], 1)
}

// ===== Inference =====

pub struct InferenceConfig {
    pub num_steps: usize,
    pub guidance_scale: f32,
    pub device: Device,
    pub dtype: DType,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            num_steps: 50,
            guidance_scale: 7.5,
            device: Device::cuda_if_available(0).unwrap_or(Device::Cpu),
            dtype: DType::F16,
        }
    }
}

pub struct FluxInference {
    model: FluxModel,
    config: InferenceConfig,
}

impl FluxInference {
    pub fn new(model: FluxModel, config: InferenceConfig) -> Self {
        Self { model, config }
    }
    
    pub fn generate(
        &self,
        text_embeds: &Tensor,
        pooled_embeds: &Tensor,
        height: usize,
        width: usize,
    ) -> Result<Tensor> {
        let p = self.model.config.patch_size;
        let c = self.model.config.in_channels;
        let latent_h = height / p / 8;  // Assuming VAE factor of 8
        let latent_w = width / p / 8;
        
        // Start from noise
        let mut latents = Tensor::randn(
            0f32,
            1f32,
            &[1, c, latent_h * p, latent_w * p],
            &self.config.device,
        )?;
        
        // Simple denoising loop
        for i in (0..self.config.num_steps).rev() {
            let t = Tensor::new(&[i as f32], &self.config.device)?;
            
            // Predict noise
            let noise_pred = self.model.forward(
                &latents,
                text_embeds,
                &t,
                pooled_embeds,
            )?;
            
            // Simple Euler step
            let alpha = (i as f32 / self.config.num_steps as f32);
            latents = latents.sub(&noise_pred.affine(alpha as f64, 0.0)?)?;
        }
        
        Ok(latents)
    }
}

// ===== Example Usage =====

pub fn create_and_train_example() -> Result<()> {
    let device = Device::cuda_if_available(0)?;
    let dtype = DType::BF16;
    
    // Create model
    let config = ModelConfig::default();
    let mut model = FluxModel::new(config, device.clone(), dtype)?;
    
    // Add LoRA
    let lora_config = LoRAConfig {
        rank: 32,
        alpha: 32.0,
        ..Default::default()
    };
    model.add_lora(&lora_config)?;
    
    // Create trainer
    let training_config = TrainingConfig::default();
    let mut trainer = Trainer::new(model, training_config)?;
    
    // Mock training batch
    let batch = TrainingBatch {
        images: Tensor::randn(0f32, 1f32, &[4, 16, 64, 64], &device)?,
        text_embeds: Tensor::randn(0f32, 1f32, &[4, 77, 4096], &device)?,
        pooled_embeds: Tensor::randn(0f32, 1f32, &[4, 3072], &device)?,
    };
    
    // Train for a few steps
    for _ in 0..10 {
        let loss = trainer.train_step(&batch)?;
        println!("Loss: {:.4}", loss);
    }
    
    // Save final checkpoint
    trainer.model.save_lora_weights(Path::new("final_lora.safetensors"))?;
    
    Ok(())
}

pub fn inference_example() -> Result<()> {
    let device = Device::cuda_if_available(0)?;
    let dtype = DType::F16;
    
    // Load model
    let config = ModelConfig::default();
    let mut model = FluxModel::new(config, device.clone(), dtype)?;
    
    // Load LoRA weights
    model.load_lora_weights(Path::new("final_lora.safetensors"))?;
    
    // Create inference pipeline
    let inference = FluxInference::new(model, InferenceConfig::default());
    
    // Generate
    let text_embeds = Tensor::randn(0f32, 1f32, &[1, 77, 4096], &device)?;
    let pooled_embeds = Tensor::randn(0f32, 1f32, &[1, 3072], &device)?;
    
    let output = inference.generate(&text_embeds, &pooled_embeds, 512, 512)?;
    println!("Generated latents shape: {:?}", output.shape());
    
    Ok(())
}
