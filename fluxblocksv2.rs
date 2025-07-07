// Fixed transformer block mappings for Flux model

use candle_core::{Device, DType, Result, Tensor};
use candle_nn::{VarBuilder, VarMap};
use std::collections::HashMap;
use safetensors::SafeTensors;

/// Enhanced FluxTensorMapper with correct transformer block mappings
pub struct FluxTensorMapper {
    name_map: HashMap<String, String>,
    device: Device,
    dtype: DType,
}

impl FluxTensorMapper {
    pub fn new(device: Device, dtype: DType) -> Self {
        let mut name_map = HashMap::new();
        
        // Time embedding mappings (MlpEmbedder structure)
        name_map.insert("time_in.in_layer.weight".to_string(), "time_in.in_layer.weight".to_string());
        name_map.insert("time_in.in_layer.bias".to_string(), "time_in.in_layer.bias".to_string());
        name_map.insert("time_in.out_layer.weight".to_string(), "time_in.out_layer.weight".to_string());
        name_map.insert("time_in.out_layer.bias".to_string(), "time_in.out_layer.bias".to_string());
        
        // Vector embedding mappings (MlpEmbedder structure)
        name_map.insert("vector_in.in_layer.weight".to_string(), "vector_in.in_layer.weight".to_string());
        name_map.insert("vector_in.in_layer.bias".to_string(), "vector_in.in_layer.bias".to_string());
        name_map.insert("vector_in.out_layer.weight".to_string(), "vector_in.out_layer.weight".to_string());
        name_map.insert("vector_in.out_layer.bias".to_string(), "vector_in.out_layer.bias".to_string());
        
        // Guidance embedding mappings (if present)
        name_map.insert("guidance_in.in_layer.weight".to_string(), "guidance_in.in_layer.weight".to_string());
        name_map.insert("guidance_in.in_layer.bias".to_string(), "guidance_in.in_layer.bias".to_string());
        name_map.insert("guidance_in.out_layer.weight".to_string(), "guidance_in.out_layer.weight".to_string());
        name_map.insert("guidance_in.out_layer.bias".to_string(), "guidance_in.out_layer.bias".to_string());
        
        // Input projections
        name_map.insert("img_in.weight".to_string(), "img_in.weight".to_string());
        name_map.insert("img_in.bias".to_string(), "img_in.bias".to_string());
        name_map.insert("txt_in.weight".to_string(), "txt_in.weight".to_string());
        name_map.insert("txt_in.bias".to_string(), "txt_in.bias".to_string());
        
        // Final layer
        name_map.insert("final_layer.weight".to_string(), "final_layer.weight".to_string());
        name_map.insert("final_layer.bias".to_string(), "final_layer.bias".to_string());
        
        Self { name_map, device, dtype }
    }
    
    /// Add double block mappings with correct Flux naming
    pub fn add_double_block_mappings(&mut self, num_blocks: usize) {
        for i in 0..num_blocks {
            let prefix = format!("double_blocks.{}", i);
            
            // Image attention block
            self.add_flux_attention_mappings(&prefix, "img_attn");
            
            // Text attention block  
            self.add_flux_attention_mappings(&prefix, "txt_attn");
            
            // Image MLP - Flux uses numbered linear layers
            self.name_map.insert(
                format!("{}.img_mlp.0.weight", prefix),
                format!("{}.img_mlp.0.weight", prefix),
            );
            self.name_map.insert(
                format!("{}.img_mlp.0.bias", prefix),
                format!("{}.img_mlp.0.bias", prefix),
            );
            self.name_map.insert(
                format!("{}.img_mlp.2.weight", prefix),
                format!("{}.img_mlp.2.weight", prefix),
            );
            self.name_map.insert(
                format!("{}.img_mlp.2.bias", prefix),
                format!("{}.img_mlp.2.bias", prefix),
            );
            
            // Text MLP - same structure
            self.name_map.insert(
                format!("{}.txt_mlp.0.weight", prefix),
                format!("{}.txt_mlp.0.weight", prefix),
            );
            self.name_map.insert(
                format!("{}.txt_mlp.0.bias", prefix),
                format!("{}.txt_mlp.0.bias", prefix),
            );
            self.name_map.insert(
                format!("{}.txt_mlp.2.weight", prefix),
                format!("{}.txt_mlp.2.weight", prefix),
            );
            self.name_map.insert(
                format!("{}.txt_mlp.2.bias", prefix),
                format!("{}.txt_mlp.2.bias", prefix),
            );
            
            // Layer norms
            self.name_map.insert(
                format!("{}.img_norm1.weight", prefix),
                format!("{}.img_norm1.weight", prefix),
            );
            self.name_map.insert(
                format!("{}.img_norm2.weight", prefix),
                format!("{}.img_norm2.weight", prefix),
            );
            self.name_map.insert(
                format!("{}.txt_norm1.weight", prefix),
                format!("{}.txt_norm1.weight", prefix),
            );
            self.name_map.insert(
                format!("{}.txt_norm2.weight", prefix),
                format!("{}.txt_norm2.weight", prefix),
            );
            
            // Modulation layers
            self.name_map.insert(
                format!("{}.img_mod.lin.weight", prefix),
                format!("{}.img_mod.lin.weight", prefix),
            );
            self.name_map.insert(
                format!("{}.img_mod.lin.bias", prefix),
                format!("{}.img_mod.lin.bias", prefix),
            );
            self.name_map.insert(
                format!("{}.txt_mod.lin.weight", prefix),
                format!("{}.txt_mod.lin.weight", prefix),
            );
            self.name_map.insert(
                format!("{}.txt_mod.lin.bias", prefix),
                format!("{}.txt_mod.lin.bias", prefix),
            );
        }
    }
    
    /// Add single block mappings with correct Flux naming
    pub fn add_single_block_mappings(&mut self, num_blocks: usize) {
        for i in 0..num_blocks {
            let prefix = format!("single_blocks.{}", i);
            
            // Attention
            self.add_flux_attention_mappings(&prefix, "attn");
            
            // MLP layers - THIS IS THE KEY FIX
            // Single blocks use linear1/linear2 naming, not 0/2
            self.name_map.insert(
                format!("{}.linear1.weight", prefix),
                format!("{}.linear1.weight", prefix),
            );
            self.name_map.insert(
                format!("{}.linear1.bias", prefix),
                format!("{}.linear1.bias", prefix),
            );
            self.name_map.insert(
                format!("{}.linear2.weight", prefix),
                format!("{}.linear2.weight", prefix),
            );
            self.name_map.insert(
                format!("{}.linear2.bias", prefix),
                format!("{}.linear2.bias", prefix),
            );
            
            // Layer norm
            self.name_map.insert(
                format!("{}.norm.weight", prefix),
                format!("{}.norm.weight", prefix),
            );
            
            // Modulation
            self.name_map.insert(
                format!("{}.modulation.lin.weight", prefix),
                format!("{}.modulation.lin.weight", prefix),
            );
            self.name_map.insert(
                format!("{}.modulation.lin.bias", prefix),
                format!("{}.modulation.lin.bias", prefix),
            );
        }
    }
    
    fn add_flux_attention_mappings(&mut self, block_prefix: &str, attn_name: &str) {
        // QKV projection - Flux uses combined qkv
        self.name_map.insert(
            format!("{}.{}.qkv.weight", block_prefix, attn_name),
            format!("{}.{}.qkv.weight", block_prefix, attn_name),
        );
        self.name_map.insert(
            format!("{}.{}.qkv.bias", block_prefix, attn_name),
            format!("{}.{}.qkv.bias", block_prefix, attn_name),
        );
        
        // Norm layers
        self.name_map.insert(
            format!("{}.{}.norm.query_norm.weight", block_prefix, attn_name),
            format!("{}.{}.norm.query_norm.weight", block_prefix, attn_name),
        );
        self.name_map.insert(
            format!("{}.{}.norm.key_norm.weight", block_prefix, attn_name),
            format!("{}.{}.norm.key_norm.weight", block_prefix, attn_name),
        );
        
        // Output projection
        self.name_map.insert(
            format!("{}.{}.proj.weight", block_prefix, attn_name),
            format!("{}.{}.proj.weight", block_prefix, attn_name),
        );
        self.name_map.insert(
            format!("{}.{}.proj.bias", block_prefix, attn_name),
            format!("{}.{}.proj.bias", block_prefix, attn_name),
        );
    }
    
    /// Create VarBuilder that handles missing tensors gracefully
    pub fn create_var_builder(&self, tensors: &HashMap<String, Tensor>) -> Result<VarBuilder> {
        let mut var_map = VarMap::new();
        
        // First pass: add all tensors that exist
        for (name, tensor) in tensors {
            let tensor = tensor.to_device(&self.device)?.to_dtype(self.dtype)?;
            var_map.set_one(name, tensor)?;
        }
        
        Ok(VarBuilder::from_varmap(&var_map, self.dtype, &self.device))
    }
}

/// Enhanced model structure to match Flux checkpoint
pub mod flux_blocks {
    use super::*;
    use candle_nn::{Module, LayerNorm};
    
    /// Single block with correct linear1/linear2 naming
    pub struct FluxSingleBlock {
        pub attn: FluxAttention,
        pub norm: LayerNorm,
        pub linear1: candle_nn::Linear,
        pub linear2: candle_nn::Linear,
        pub modulation: Modulation,
    }
    
    impl FluxSingleBlock {
        pub fn new(vb: VarBuilder, hidden_size: usize, num_heads: usize) -> Result<Self> {
            Ok(Self {
                attn: FluxAttention::new(vb.pp("attn"), hidden_size, num_heads)?,
                norm: candle_nn::layer_norm(hidden_size, 1e-6, vb.pp("norm"))?,
                linear1: candle_nn::linear(hidden_size, hidden_size * 4, vb.pp("linear1"))?,
                linear2: candle_nn::linear(hidden_size * 4, hidden_size, vb.pp("linear2"))?,
                modulation: Modulation::new(vb.pp("modulation"), hidden_size)?,
            })
        }
        
        pub fn forward(&self, x: &Tensor, vec: &Tensor) -> Result<Tensor> {
            // Apply modulation
            let (shift, scale, gate) = self.modulation.forward(vec)?;
            
            // Norm + modulation
            let x_norm = self.norm.forward(x)?;
            let x_mod = x_norm.broadcast_mul(&(1.0 + scale)?)?.broadcast_add(&shift)?;
            
            // Self attention
            let attn_out = self.attn.forward(&x_mod)?;
            let x = x.add(&(attn_out.broadcast_mul(&gate)?)?)?;
            
            // MLP
            let mlp = self.linear1.forward(&x)?;
            let mlp = mlp.gelu()?;
            let mlp = self.linear2.forward(&mlp)?;
            
            x.add(&mlp)
        }
    }
    
    /// Double block with correct MLP structure
    pub struct FluxDoubleBlock {
        pub img_attn: FluxAttention,
        pub txt_attn: FluxAttention,
        pub img_norm1: LayerNorm,
        pub img_norm2: LayerNorm,
        pub txt_norm1: LayerNorm,
        pub txt_norm2: LayerNorm,
        pub img_mlp: FluxMLP,
        pub txt_mlp: FluxMLP,
        pub img_mod: Modulation,
        pub txt_mod: Modulation,
    }
    
    impl FluxDoubleBlock {
        pub fn new(vb: VarBuilder, hidden_size: usize, num_heads: usize) -> Result<Self> {
            Ok(Self {
                img_attn: FluxAttention::new(vb.pp("img_attn"), hidden_size, num_heads)?,
                txt_attn: FluxAttention::new(vb.pp("txt_attn"), hidden_size, num_heads)?,
                img_norm1: candle_nn::layer_norm(hidden_size, 1e-6, vb.pp("img_norm1"))?,
                img_norm2: candle_nn::layer_norm(hidden_size, 1e-6, vb.pp("img_norm2"))?,
                txt_norm1: candle_nn::layer_norm(hidden_size, 1e-6, vb.pp("txt_norm1"))?,
                txt_norm2: candle_nn::layer_norm(hidden_size, 1e-6, vb.pp("txt_norm2"))?,
                img_mlp: FluxMLP::new(vb.pp("img_mlp"), hidden_size)?,
                txt_mlp: FluxMLP::new(vb.pp("txt_mlp"), hidden_size)?,
                img_mod: Modulation::new(vb.pp("img_mod"), hidden_size)?,
                txt_mod: Modulation::new(vb.pp("txt_mod"), hidden_size)?,
            })
        }
    }
    
    /// MLP with numbered layers (0, 2) for double blocks
    pub struct FluxMLP {
        pub _0: candle_nn::Linear,  // First layer
        pub _2: candle_nn::Linear,  // Second layer (with skip for activation)
    }
    
    impl FluxMLP {
        pub fn new(vb: VarBuilder, hidden_size: usize) -> Result<Self> {
            let mlp_hidden = hidden_size * 4;
            Ok(Self {
                _0: candle_nn::linear(hidden_size, mlp_hidden, vb.pp("0"))?,
                _2: candle_nn::linear(mlp_hidden, hidden_size, vb.pp("2"))?,
            })
        }
        
        pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
            let x = self._0.forward(x)?;
            let x = x.gelu()?;
            self._2.forward(&x)
        }
    }
    
    /// Attention with QKV and normalization
    pub struct FluxAttention {
        pub qkv: candle_nn::Linear,
        pub norm: QKNorm,
        pub proj: candle_nn::Linear,
        pub num_heads: usize,
    }
    
    impl FluxAttention {
        pub fn new(vb: VarBuilder, hidden_size: usize, num_heads: usize) -> Result<Self> {
            Ok(Self {
                qkv: candle_nn::linear(hidden_size, hidden_size * 3, vb.pp("qkv"))?,
                norm: QKNorm::new(vb.pp("norm"), hidden_size / num_heads)?,
                proj: candle_nn::linear(hidden_size, hidden_size, vb.pp("proj"))?,
                num_heads,
            })
        }
        
        pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
            let (b, seq_len, _) = x.dims3()?;
            let qkv = self.qkv.forward(x)?;
            
            // Split QKV
            let qkv = qkv.reshape((b, seq_len, 3, self.num_heads, -1))?;
            let qkv = qkv.permute((2, 0, 3, 1, 4))?; // [3, B, heads, seq, dim]
            
            let q = qkv.get(0)?;
            let k = qkv.get(1)?;
            let v = qkv.get(2)?;
            
            // Apply normalization
            let (q, k) = self.norm.forward(&q, &k)?;
            
            // Attention
            let scale = 1.0 / (q.dim(D::Minus1)? as f64).sqrt();
            let scores = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
            let scores = scores.affine(scale, 0.0)?;
            let attn = candle_nn::ops::softmax(&scores, D::Minus1)?;
            let out = attn.matmul(&v)?;
            
            // Reshape and project
            let out = out.transpose(1, 2)?.reshape((b, seq_len, -1))?;
            self.proj.forward(&out)
        }
    }
    
    pub struct QKNorm {
        pub query_norm: LayerNorm,
        pub key_norm: LayerNorm,
    }
    
    impl QKNorm {
        pub fn new(vb: VarBuilder, dim: usize) -> Result<Self> {
            Ok(Self {
                query_norm: candle_nn::layer_norm(dim, 1e-6, vb.pp("query_norm"))?,
                key_norm: candle_nn::layer_norm(dim, 1e-6, vb.pp("key_norm"))?,
            })
        }
        
        pub fn forward(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
            Ok((
                self.query_norm.forward(q)?,
                self.key_norm.forward(k)?,
            ))
        }
    }
    
    pub struct Modulation {
        pub lin: candle_nn::Linear,
    }
    
    impl Modulation {
        pub fn new(vb: VarBuilder, hidden_size: usize) -> Result<Self> {
            Ok(Self {
                lin: candle_nn::linear(hidden_size, hidden_size * 3, vb.pp("lin"))?,
            })
        }
        
        pub fn forward(&self, vec: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
            let out = self.lin.forward(vec)?;
            let chunks = out.chunk(3, D::Minus1)?;
            Ok((
                chunks[0].clone(),
                chunks[1].clone(),
                chunks[2].clone(),
            ))
        }
    }
}

/// Load Flux model with correct structure
pub fn load_flux_model_fixed(
    checkpoint_path: &Path,
    device: &Device,
    dtype: DType,
) -> Result<()> {
    println!("Loading Flux checkpoint from {:?}", checkpoint_path);
    
    // Load all tensors
    let tensors = safetensors::load(checkpoint_path, device)?;
    
    // Create mapper
    let mut mapper = FluxTensorMapper::new(device.clone(), dtype);
    
    // Detect model structure from tensors
    let num_double_blocks = tensors.keys()
        .filter(|k| k.starts_with("double_blocks."))
        .map(|k| k.split('.').nth(1).and_then(|s| s.parse::<usize>().ok()).unwrap_or(0))
        .max()
        .unwrap_or(0) + 1;
    
    let num_single_blocks = tensors.keys()
        .filter(|k| k.starts_with("single_blocks."))
        .map(|k| k.split('.').nth(1).and_then(|s| s.parse::<usize>().ok()).unwrap_or(0))
        .max()
        .unwrap_or(0) + 1;
    
    println!("Detected {} double blocks, {} single blocks", num_double_blocks, num_single_blocks);
    
    // Add mappings
    mapper.add_double_block_mappings(num_double_blocks);
    mapper.add_single_block_mappings(num_single_blocks);
    
    // List some tensor names for debugging
    println!("\nSample tensor names from checkpoint:");
    for (i, name) in tensors.keys().enumerate() {
        if i < 10 || name.contains("single_blocks.30") {
            println!("  {}", name);
        }
    }
    
    // Create VarBuilder
    let vb = mapper.create_var_builder(&tensors)?;
    
    println!("\nModel loaded successfully!");
    Ok(())
}

/// Wrapper to handle both old and new MLP structures
pub fn load_mlp_block(vb: &VarBuilder, prefix: &str, hidden_size: usize) -> Result<Box<dyn Module>> {
    // Try to load with numbered structure first (0, 2)
    if vb.contains_tensor(&format!("{}.0.weight", prefix)) {
        Ok(Box::new(flux_blocks::FluxMLP::new(vb.set_prefix(prefix), hidden_size)?))
    } else if vb.contains_tensor(&format!("{}.linear1.weight", prefix)) {
        // Fall back to linear1/linear2 structure
        Ok(Box::new(LinearMLP::new(vb.set_prefix(prefix), hidden_size)?))
    } else {
        Err(candle_core::Error::Msg(format!("Could not find MLP weights at {}", prefix)))
    }
}

struct LinearMLP {
    linear1: candle_nn::Linear,
    linear2: candle_nn::Linear,
}

impl LinearMLP {
    fn new(vb: VarBuilder, hidden_size: usize) -> Result<Self> {
        Ok(Self {
            linear1: candle_nn::linear(hidden_size, hidden_size * 4, vb.pp("linear1"))?,
            linear2: candle_nn::linear(hidden_size * 4, hidden_size, vb.pp("linear2"))?,
        })
    }
}

impl Module for LinearMLP {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.linear1.forward(x)?;
        let x = x.gelu()?;
        self.linear2.forward(&x)
    }
}
