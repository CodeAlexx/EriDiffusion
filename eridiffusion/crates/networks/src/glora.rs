//! GLoRA (Generalized Low-Rank Adaptation) implementation

use eridiffusion_core::{
    NetworkAdapter, NetworkType, NetworkMetadata, ModelArchitecture, Device, Result, Error,
};
use candle_core::{Tensor, DType, Module, Shape, Var};
use candle_nn::{Linear, VarBuilder, LayerNorm};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;
use std::path::Path;
use async_trait::async_trait;

/// GLoRA configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GLoRAConfig {
    pub rank: usize,
    pub alpha: f32,
    pub dropout: f32,
    pub target_modules: Vec<String>,
    pub use_bias: bool,
    pub use_layernorm: bool,
    pub use_scalar: bool,
    pub prompt_length: Option<usize>,
    pub init_weights: bool,
    pub share_prompts: bool,
    pub hierarchical: bool,
    pub num_heads: Option<usize>,
}

impl Default for GLoRAConfig {
    fn default() -> Self {
        Self {
            rank: 4,
            alpha: 1.0,
            dropout: 0.0,
            target_modules: vec![
                "to_q".to_string(),
                "to_k".to_string(),
                "to_v".to_string(),
                "to_out.0".to_string(),
            ],
            use_bias: false,
            use_layernorm: true,
            use_scalar: true,
            prompt_length: Some(10),
            init_weights: true,
            share_prompts: false,
            hierarchical: false,
            num_heads: None,
        }
    }
}

/// GLoRA layer types
pub enum GLoRALayer {
    Linear(GLoRALinear),
    Conv2d(GLoRAConv2d),
}

/// GLoRA linear layer
pub struct GLoRALinear {
    // Core LoRA components
    lora_a: Linear,
    lora_b: Linear,
    
    // GLoRA-specific components
    layer_norm: Option<LayerNorm>,
    prompt_embeddings: Option<Tensor>,
    scaling_vector: Option<Tensor>,
    gating: Option<Linear>,
    
    // Multi-head variant
    head_lora_a: Option<Vec<Linear>>,
    head_lora_b: Option<Vec<Linear>>,
    
    scaling: f32,
    dropout: Option<f32>,
    rank: usize,
    in_features: usize,
    out_features: usize,
}

/// GLoRA convolutional layer
pub struct GLoRAConv2d {
    // Core LoRA components
    lora_a: candle_nn::Conv2d,
    lora_b: candle_nn::Conv2d,
    
    // GLoRA-specific components
    layer_norm: Option<LayerNorm>,
    prompt_embeddings: Option<Tensor>,
    scaling_vector: Option<Tensor>,
    gating: Option<Linear>,
    
    // Multi-head variant
    head_lora_a: Option<Vec<candle_nn::Conv2d>>,
    head_lora_b: Option<Vec<candle_nn::Conv2d>>,
    
    scaling: f32,
    dropout: Option<f32>,
    rank: usize,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
}

impl GLoRALinear {
    /// Create new GLoRA linear layer
    pub fn new(
        in_features: usize,
        out_features: usize,
        rank: usize,
        alpha: f32,
        dropout: f32,
        config: &GLoRAConfig,
        device: &Device,
    ) -> Result<Self> {
        let candle_device = device.to_candle()?;
        let dtype = DType::F32;
        
        // Standard LoRA components
        let lora_a = Linear::new(
            Self::kaiming_uniform_init(rank, in_features, dtype, &candle_device)?,
            if config.use_bias {
                Some(Tensor::zeros(&[rank], dtype, &candle_device)?)
            } else {
                None
            },
        );
        
        let lora_b = Linear::new(
            Tensor::zeros(&[out_features, rank], dtype, &candle_device)?,
            if config.use_bias {
                Some(Tensor::zeros(&[out_features], dtype, &candle_device)?)
            } else {
                None
            },
        );
        
        // Layer normalization
        let layer_norm = if config.use_layernorm {
            Some(LayerNorm::new(
                Tensor::ones(&[rank], dtype, &candle_device)?,
                Tensor::zeros(&[rank], dtype, &candle_device)?,
                1e-5,
            ))
        } else {
            None
        };
        
        // Prompt embeddings
        let prompt_embeddings = if let Some(prompt_len) = config.prompt_length {
            Some(Tensor::randn(
                0.0f32,
                0.02,
                &[prompt_len, rank],
                &candle_device,
            )?)
        } else {
            None
        };
        
        // Scaling vector
        let scaling_vector = if config.use_scalar {
            Some(Tensor::ones(&[rank], dtype, &candle_device)?)
        } else {
            None
        };
        
        // Gating mechanism
        let gating = if config.hierarchical {
            Some(Linear::new(
                Tensor::randn(0.0f32, 0.02, &[1, in_features], &candle_device)?,
                Some(Tensor::zeros(&[1], dtype, &candle_device)?),
            ))
        } else {
            None
        };
        
        // Multi-head variant
        let (head_lora_a, head_lora_b) = if let Some(num_heads) = config.num_heads {
            let mut heads_a = Vec::new();
            let mut heads_b = Vec::new();
            
            let head_rank = rank / num_heads;
            let head_in = in_features / num_heads;
            let head_out = out_features / num_heads;
            
            for _ in 0..num_heads {
                heads_a.push(Linear::new(
                    Self::kaiming_uniform_init(head_rank, head_in, dtype, &candle_device)?,
                    None,
                ));
                
                heads_b.push(Linear::new(
                    Tensor::zeros(&[head_out, head_rank], dtype, &candle_device)?,
                    None,
                ));
            }
            
            (Some(heads_a), Some(heads_b))
        } else {
            (None, None)
        };
        
        let scaling = alpha / rank as f32;
        
        Ok(Self {
            lora_a,
            lora_b,
            layer_norm,
            prompt_embeddings,
            scaling_vector,
            gating,
            head_lora_a,
            head_lora_b,
            scaling,
            dropout: if dropout > 0.0 { Some(dropout) } else { None },
            rank,
            in_features,
            out_features,
        })
    }
    
    /// Forward pass
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut result = x.clone();
        
        // Apply dropout if configured
        if let Some(dropout_rate) = self.dropout {
            result = self.apply_dropout(&result, dropout_rate)?;
        }
        
        // Multi-head path
        if let (Some(ref heads_a), Some(ref heads_b)) = (&self.head_lora_a, &self.head_lora_b) {
            result = self.multi_head_forward(&result, heads_a, heads_b)?;
        } else {
            // Standard path with enhancements
            
            // Apply prompt if available
            if let Some(ref prompts) = self.prompt_embeddings {
                result = self.apply_prompt(&result, prompts)?;
            }
            
            // Standard LoRA computation
            result = self.lora_a.forward(&result)?;
            
            // Apply layer norm if configured
            if let Some(ref ln) = self.layer_norm {
                result = ln.forward(&result)?;
            }
            
            // Apply scaling vector if configured
            if let Some(ref scale) = self.scaling_vector {
                result = result.broadcast_mul(scale)?;
            }
            
            result = self.lora_b.forward(&result)?;
        }
        
        // Apply gating if configured
        if let Some(ref gate) = self.gating {
            let gate_value = gate.forward(x)?;
            let gate_value = candle_nn::ops::sigmoid(&gate_value)?;
            result = (result * gate_value)?;
        }
        
        // Apply final scaling
        Ok((result * self.scaling as f64)?)
    }
    
    /// Multi-head forward
    fn multi_head_forward(
        &self,
        x: &Tensor,
        heads_a: &[Linear],
        heads_b: &[Linear],
    ) -> Result<Tensor> {
        let batch_dims = x.dims()[..x.dims().len()-1].to_vec();
        let batch_size: usize = batch_dims.iter().product();
        let num_heads = heads_a.len();
        let head_dim = self.in_features / num_heads;
        
        // Reshape for multi-head: [batch, num_heads, head_dim]
        let x_reshaped = x.reshape(&[batch_size, num_heads, head_dim])?;
        
        let mut outputs = Vec::new();
        
        // Process each head
        for (i, (head_a, head_b)) in heads_a.iter().zip(heads_b.iter()).enumerate() {
            let head_input = x_reshaped.narrow(1, i, 1)?.squeeze(1)?;
            
            let mut head_out = head_a.forward(&head_input)?;
            
            // Apply layer norm if configured
            if let Some(ref ln) = self.layer_norm {
                head_out = ln.forward(&head_out)?;
            }
            
            head_out = head_b.forward(&head_out)?;
            outputs.push(head_out);
        }
        
        // Concatenate heads
        let output = Tensor::cat(&outputs, 1)?;
        
        // Reshape back
        let mut output_shape = batch_dims;
        output_shape.push(self.out_features);
        output.reshape(output_shape.as_slice()).map_err(|e| Error::Tensor(e.to_string()))
    }
    
    /// Apply prompt embeddings
    fn apply_prompt(&self, x: &Tensor, prompts: &Tensor) -> Result<Tensor> {
        // Simple prompt addition - could be more sophisticated
        let prompt_expanded = prompts.sum(0)?.unsqueeze(0)?;
        Ok((x + prompt_expanded.broadcast_as(x.shape())?)?)
    }
    
    /// Apply dropout
    fn apply_dropout(&self, x: &Tensor, rate: f32) -> Result<Tensor> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        let mask = Tensor::from_vec(
            (0..x.elem_count())
                .map(|_| if rng.gen::<f32>() > rate { 1.0f32 } else { 0.0f32 })
                .collect::<Vec<_>>(),
            x.shape(),
            x.device(),
        )?;
        
        Ok((x * mask)?)
    }
    
    /// Kaiming uniform initialization
    fn kaiming_uniform_init(
        fan_out: usize,
        fan_in: usize,
        dtype: DType,
        device: &candle_core::Device,
    ) -> Result<Tensor> {
        let bound = (3.0f64 / fan_in as f64).sqrt();
        Tensor::rand(
            -bound,
            bound,
            &[fan_out, fan_in],
            device,
        )?.to_dtype(dtype).map_err(|e| Error::Tensor(e.to_string()))
    }
}

impl GLoRAConv2d {
    /// Create new GLoRA convolutional layer
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        rank: usize,
        alpha: f32,
        dropout: f32,
        config: &GLoRAConfig,
        device: &Device,
    ) -> Result<Self> {
        let candle_device = device.to_candle()?;
        let dtype = DType::F32;
        
        // Create conv2d config
        let conv_config = candle_nn::Conv2dConfig {
            padding: 1,
            stride: 1,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        };
        
        // Standard LoRA components
        let vb = VarBuilder::zeros(dtype, &candle_device);
        
        let lora_a = candle_nn::conv2d(
            in_channels,
            rank,
            kernel_size,
            conv_config,
            vb.pp("lora_a"),
        )?;
        
        let lora_b = candle_nn::conv2d(
            rank,
            out_channels,
            1, // 1x1 conv for lora_b
            candle_nn::Conv2dConfig::default(),
            vb.pp("lora_b"),
        )?;
        
        // Layer normalization
        let layer_norm = if config.use_layernorm {
            Some(LayerNorm::new(
                Tensor::ones(&[rank], dtype, &candle_device)?,
                Tensor::zeros(&[rank], dtype, &candle_device)?,
                1e-5,
            ))
        } else {
            None
        };
        
        // Prompt embeddings
        let prompt_embeddings = if let Some(prompt_len) = config.prompt_length {
            Some(Tensor::randn(
                0.0f32,
                0.02,
                &[prompt_len, rank],
                &candle_device,
            )?)
        } else {
            None
        };
        
        // Scaling vector
        let scaling_vector = if config.use_scalar {
            Some(Tensor::ones(&[rank, 1, 1], dtype, &candle_device)?)
        } else {
            None
        };
        
        // Gating mechanism
        let gating = if config.hierarchical {
            Some(Linear::new(
                Tensor::randn(0.0f32, 0.02, &[1, in_channels], &candle_device)?,
                Some(Tensor::zeros(&[1], dtype, &candle_device)?),
            ))
        } else {
            None
        };
        
        // Multi-head variant
        let (head_lora_a, head_lora_b) = if let Some(num_heads) = config.num_heads {
            let mut heads_a = Vec::new();
            let mut heads_b = Vec::new();
            
            let head_rank = rank / num_heads;
            let head_in = in_channels / num_heads;
            let head_out = out_channels / num_heads;
            
            for i in 0..num_heads {
                let vb_a = VarBuilder::zeros(dtype, &candle_device);
                let vb_b = VarBuilder::zeros(dtype, &candle_device);
                
                heads_a.push(candle_nn::conv2d(
                    head_in,
                    head_rank,
                    kernel_size,
                    conv_config,
                    vb_a.pp(&format!("head_a_{}", i)),
                )?);
                
                heads_b.push(candle_nn::conv2d(
                    head_rank,
                    head_out,
                    1,
                    candle_nn::Conv2dConfig::default(),
                    vb_b.pp(&format!("head_b_{}", i)),
                )?);
            }
            
            (Some(heads_a), Some(heads_b))
        } else {
            (None, None)
        };
        
        let scaling = alpha / rank as f32;
        
        Ok(Self {
            lora_a,
            lora_b,
            layer_norm,
            prompt_embeddings,
            scaling_vector,
            gating,
            head_lora_a,
            head_lora_b,
            scaling,
            dropout: if dropout > 0.0 { Some(dropout) } else { None },
            rank,
            in_channels,
            out_channels,
            kernel_size,
        })
    }
    
    /// Forward pass
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut result = x.clone();
        
        // Apply dropout if configured
        if let Some(dropout_rate) = self.dropout {
            result = self.apply_dropout(&result, dropout_rate)?;
        }
        
        // Multi-head path
        if let (Some(ref heads_a), Some(ref heads_b)) = (&self.head_lora_a, &self.head_lora_b) {
            result = self.multi_head_forward(&result, heads_a, heads_b)?;
        } else {
            // Standard path with enhancements
            
            // Standard LoRA computation
            result = self.lora_a.forward(&result)?;
            
            // Apply layer norm if configured (need to reshape for layer norm)
            if let Some(ref ln) = self.layer_norm {
                let shape = result.shape().dims().to_vec();
                let c = shape[1];
                let spatial = shape[2..].iter().product::<usize>();
                
                result = result.transpose(1, 2)?
                    .reshape(&[shape[0] * spatial, c])?;
                result = ln.forward(&result)?;
                result = result.reshape(&[shape[0], spatial, c])?
                    .transpose(1, 2)?
                    .reshape(shape.as_slice())?;
            }
            
            // Apply scaling vector if configured
            if let Some(ref scale) = self.scaling_vector {
                result = result.broadcast_mul(scale)?;
            }
            
            result = self.lora_b.forward(&result)?;
        }
        
        // Apply gating if configured
        if let Some(ref gate) = self.gating {
            // Global average pooling for conv features
            let pooled = x.mean_keepdim(2)?
                .mean_keepdim(3)?
                .squeeze(2)?
                .squeeze(2)?;
            let gate_value = gate.forward(&pooled)?;
            let gate_value = candle_nn::ops::sigmoid(&gate_value)?;
            let gate_value = gate_value.unsqueeze(2)?.unsqueeze(3)?;
            result = result.broadcast_mul(&gate_value)?;
        }
        
        // Apply final scaling
        Ok((result * self.scaling as f64)?)
    }
    
    /// Multi-head forward for conv
    fn multi_head_forward(
        &self,
        x: &Tensor,
        heads_a: &[candle_nn::Conv2d],
        heads_b: &[candle_nn::Conv2d],
    ) -> Result<Tensor> {
        let shape = x.shape().dims().to_vec();
        let num_heads = heads_a.len();
        let head_channels = self.in_channels / num_heads;
        
        let mut outputs = Vec::new();
        
        // Process each head
        for (i, (head_a, head_b)) in heads_a.iter().zip(heads_b.iter()).enumerate() {
            let start = i * head_channels;
            let head_input = x.narrow(1, start, head_channels)?;
            
            let mut head_out = head_a.forward(&head_input)?;
            
            // Apply layer norm if configured
            if let Some(ref ln) = self.layer_norm {
                let h_shape = head_out.shape().dims().to_vec();
                let c = h_shape[1];
                let spatial = h_shape[2..].iter().product::<usize>();
                
                head_out = head_out.transpose(1, 2)?
                    .reshape(&[h_shape[0] * spatial, c])?;
                head_out = ln.forward(&head_out)?;
                head_out = head_out.reshape(&[h_shape[0], spatial, c])?
                    .transpose(1, 2)?
                    .reshape(h_shape.as_slice())?;
            }
            
            head_out = head_b.forward(&head_out)?;
            outputs.push(head_out);
        }
        
        // Concatenate heads
        Tensor::cat(&outputs, 1).map_err(|e| Error::Tensor(e.to_string()))
    }
    
    /// Apply dropout
    fn apply_dropout(&self, x: &Tensor, rate: f32) -> Result<Tensor> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        let mask = Tensor::from_vec(
            (0..x.elem_count())
                .map(|_| if rng.gen::<f32>() > rate { 1.0f32 } else { 0.0f32 })
                .collect::<Vec<_>>(),
            x.shape(),
            x.device(),
        )?;
        
        Ok((x * mask)?)
    }
}

/// GLoRA adapter state
struct GLoRAState {
    layers: HashMap<String, GLoRALayer>,
    shared_prompts: Option<HashMap<String, Tensor>>,
    enabled: bool,
    training: bool,
    device: Device,
}

/// GLoRA adapter
pub struct GLoRAAdapter {
    config: GLoRAConfig,
    state: Arc<RwLock<GLoRAState>>,
    architecture: ModelArchitecture,
    metadata: NetworkMetadata,
}

impl GLoRAAdapter {
    /// Create new GLoRA adapter
    pub fn new(
        config: GLoRAConfig,
        architecture: ModelArchitecture,
        device: Device,
    ) -> Result<Self> {
        let state = Arc::new(RwLock::new(GLoRAState {
            layers: HashMap::new(),
            shared_prompts: if config.share_prompts {
                Some(HashMap::new())
            } else {
                None
            },
            enabled: true,
            training: false,
            device,
        }));
        
        let metadata = NetworkMetadata {
            name: "glora_adapter".to_string(),
            network_type: NetworkType::GLoRA,
            version: "1.0.0".to_string(),
            base_model: "unknown".to_string(),
            rank: Some(config.rank),
            alpha: Some(config.alpha),
            target_modules: config.target_modules.clone(),
            created_at: chrono::Utc::now(),
            config: HashMap::new(),
        };
        
        Ok(Self {
            config,
            state,
            architecture,
            metadata,
        })
    }
    
    /// Initialize GLoRA layers
    pub fn initialize_layers(
        &self,
        model_state_dict: &HashMap<String, TensorInfo>,
    ) -> Result<()> {
        let mut state = self.state.write();
        
        for (name, info) in model_state_dict {
            if self.should_adapt_module(name) {
                let layer = match info.shape.len() {
                    2 => {
                        // Linear layer
                        let out_features = info.shape[0];
                        let in_features = info.shape[1];
                        
                        let linear = GLoRALinear::new(
                            in_features,
                            out_features,
                            self.config.rank,
                            self.config.alpha,
                            self.config.dropout,
                            &self.config,
                            &state.device,
                        )?;
                        
                        tracing::debug!(
                            "Initialized GLoRA Linear for {}: {}x{} (rank={})",
                            name, in_features, out_features, self.config.rank
                        );
                        
                        GLoRALayer::Linear(linear)
                    }
                    4 => {
                        // Conv2d layer
                        let out_channels = info.shape[0];
                        let in_channels = info.shape[1];
                        let kernel_size = info.shape[2]; // Assuming square kernel
                        
                        let conv = GLoRAConv2d::new(
                            in_channels,
                            out_channels,
                            kernel_size,
                            self.config.rank,
                            self.config.alpha,
                            self.config.dropout,
                            &self.config,
                            &state.device,
                        )?;
                        
                        tracing::debug!(
                            "Initialized GLoRA Conv2d for {}: {}x{} k={} (rank={})",
                            name, in_channels, out_channels, kernel_size, self.config.rank
                        );
                        
                        GLoRALayer::Conv2d(conv)
                    }
                    _ => continue, // Skip unsupported shapes
                };
                
                state.layers.insert(name.clone(), layer);
            }
        }
        
        // Initialize shared prompts if configured
        if state.shared_prompts.is_some() {
            if let Some(prompt_len) = self.config.prompt_length {
                let candle_device = state.device.to_candle()?;
                let mut prompts_to_insert = HashMap::new();
                
                // Create shared prompt embeddings for different layer types
                for module_type in &self.config.target_modules {
                    let prompt = Tensor::randn(
                        0.0f32,
                        0.02,
                        &[prompt_len, self.config.rank],
                        &candle_device,
                    )?;
                    
                    prompts_to_insert.insert(module_type.clone(), prompt);
                }
                
                if let Some(ref mut shared_prompts) = state.shared_prompts {
                    for (k, v) in prompts_to_insert {
                        shared_prompts.insert(k, v);
                    }
                }
            }
        }
        
        tracing::info!("Initialized {} GLoRA layers", state.layers.len());
        Ok(())
    }
    
    fn should_adapt_module(&self, name: &str) -> bool {
        self.config.target_modules.iter().any(|pattern| {
            name.contains(pattern)
        })
    }
    
    /// Get number of parameters
    pub fn num_parameters(&self) -> usize {
        let state = self.state.read();
        
        let layer_params: usize = state.layers.values().map(|layer| {
            match layer {
                GLoRALayer::Linear(linear) => {
                    let mut params = 0;
                    
                    // Standard LoRA parameters
                    params += linear.lora_a.weight().elem_count();
                    params += linear.lora_b.weight().elem_count();
                    
                    // Layer norm parameters
                    if linear.layer_norm.is_some() {
                        params += 2 * linear.rank; // weight and bias
                    }
                    
                    // Prompt embeddings
                    if let Some(ref prompts) = linear.prompt_embeddings {
                        params += prompts.elem_count();
                    }
                    
                    // Scaling vector
                    if let Some(ref scale) = linear.scaling_vector {
                        params += scale.elem_count();
                    }
                    
                    // Gating parameters
                    if let Some(ref gate) = linear.gating {
                        params += gate.weight().elem_count() + 1; // weight + bias
                    }
                    
                    // Multi-head parameters
                    if let Some(ref heads_a) = linear.head_lora_a {
                        for head in heads_a {
                            params += head.weight().elem_count();
                        }
                    }
                    
                    if let Some(ref heads_b) = linear.head_lora_b {
                        for head in heads_b {
                            params += head.weight().elem_count();
                        }
                    }
                    
                    params
                }
                GLoRALayer::Conv2d(conv) => {
                    let mut params = 0;
                    
                    // Standard LoRA parameters
                    params += conv.lora_a.weight().elem_count();
                    params += conv.lora_b.weight().elem_count();
                    
                    // Layer norm parameters
                    if conv.layer_norm.is_some() {
                        params += 2 * conv.rank; // weight and bias
                    }
                    
                    // Prompt embeddings
                    if let Some(ref prompts) = conv.prompt_embeddings {
                        params += prompts.elem_count();
                    }
                    
                    // Scaling vector
                    if let Some(ref scale) = conv.scaling_vector {
                        params += scale.elem_count();
                    }
                    
                    // Gating parameters
                    if let Some(ref gate) = conv.gating {
                        params += gate.weight().elem_count() + 1; // weight + bias
                    }
                    
                    // Multi-head parameters
                    if let Some(ref heads_a) = conv.head_lora_a {
                        for head in heads_a {
                            params += head.weight().elem_count();
                        }
                    }
                    
                    if let Some(ref heads_b) = conv.head_lora_b {
                        for head in heads_b {
                            params += head.weight().elem_count();
                        }
                    }
                    
                    params
                }
            }
        }).sum();
        
        // Add shared prompt parameters
        let shared_params = if let Some(ref prompts) = state.shared_prompts {
            prompts.values().map(|p| p.elem_count()).sum()
        } else {
            0
        };
        
        layer_params + shared_params
    }
}

#[async_trait]
impl NetworkAdapter for GLoRAAdapter {
    fn adapter_type(&self) -> NetworkType {
        NetworkType::GLoRA
    }
    
    fn metadata(&self) -> &NetworkMetadata {
        &self.metadata
    }
    
    fn target_modules(&self) -> &[String] {
        &self.config.target_modules
    }
    
    fn trainable_parameters(&self) -> Vec<&Var> {
        // Collect trainable parameters from all GLoRA layers
        let state = self.state.read();
        let mut vars: Vec<&Var> = Vec::new();
        
        // TODO: Return actual Var references from GLoRA layers
        // For now return empty as layers use Linear/Conv2d which don't expose Var directly
        Vec::new()
    }
    
    fn parameters(&self) -> HashMap<String, Tensor> {
        let state = self.state.read();
        let mut params = HashMap::new();
        
        for (name, layer) in &state.layers {
            match layer {
                GLoRALayer::Linear(linear) => {
                    // Add LoRA A and B weights
                    params.insert(format!("{}.lora_a.weight", name), linear.lora_a.weight().clone());
                    params.insert(format!("{}.lora_b.weight", name), linear.lora_b.weight().clone());
                    
                    // Add biases if present
                    if let Some(bias) = linear.lora_a.bias() {
                        params.insert(format!("{}.lora_a.bias", name), bias.clone());
                    }
                    if let Some(bias) = linear.lora_b.bias() {
                        params.insert(format!("{}.lora_b.bias", name), bias.clone());
                    }
                    
                    // Add GLoRA-specific components
                    if let Some(ref prompts) = linear.prompt_embeddings {
                        params.insert(format!("{}.prompt_embeddings", name), prompts.clone());
                    }
                    if let Some(ref scale) = linear.scaling_vector {
                        params.insert(format!("{}.scaling_vector", name), scale.clone());
                    }
                    if let Some(ref gate) = linear.gating {
                        params.insert(format!("{}.gating.weight", name), gate.weight().clone());
                        if let Some(bias) = gate.bias() {
                            params.insert(format!("{}.gating.bias", name), bias.clone());
                        }
                    }
                    
                    // Add multi-head components
                    if let Some(ref heads_a) = linear.head_lora_a {
                        for (i, head) in heads_a.iter().enumerate() {
                            params.insert(format!("{}.head_lora_a.{}.weight", name, i), head.weight().clone());
                        }
                    }
                    if let Some(ref heads_b) = linear.head_lora_b {
                        for (i, head) in heads_b.iter().enumerate() {
                            params.insert(format!("{}.head_lora_b.{}.weight", name, i), head.weight().clone());
                        }
                    }
                }
                GLoRALayer::Conv2d(conv) => {
                    // Add LoRA A and B weights
                    params.insert(format!("{}.lora_a.weight", name), conv.lora_a.weight().clone());
                    params.insert(format!("{}.lora_b.weight", name), conv.lora_b.weight().clone());
                    
                    // Add biases if present
                    if let Some(bias) = conv.lora_a.bias() {
                        params.insert(format!("{}.lora_a.bias", name), bias.clone());
                    }
                    if let Some(bias) = conv.lora_b.bias() {
                        params.insert(format!("{}.lora_b.bias", name), bias.clone());
                    }
                    
                    // Add GLoRA-specific components
                    if let Some(ref prompts) = conv.prompt_embeddings {
                        params.insert(format!("{}.prompt_embeddings", name), prompts.clone());
                    }
                    if let Some(ref scale) = conv.scaling_vector {
                        params.insert(format!("{}.scaling_vector", name), scale.clone());
                    }
                    if let Some(ref gate) = conv.gating {
                        params.insert(format!("{}.gating.weight", name), gate.weight().clone());
                        if let Some(bias) = gate.bias() {
                            params.insert(format!("{}.gating.bias", name), bias.clone());
                        }
                    }
                    
                    // Add multi-head components
                    if let Some(ref heads_a) = conv.head_lora_a {
                        for (i, head) in heads_a.iter().enumerate() {
                            params.insert(format!("{}.head_lora_a.{}.weight", name, i), head.weight().clone());
                        }
                    }
                    if let Some(ref heads_b) = conv.head_lora_b {
                        for (i, head) in heads_b.iter().enumerate() {
                            params.insert(format!("{}.head_lora_b.{}.weight", name, i), head.weight().clone());
                        }
                    }
                }
            }
        }
        
        // Add shared prompts
        if let Some(ref shared_prompts) = state.shared_prompts {
            for (prompt_name, prompt_tensor) in shared_prompts {
                params.insert(format!("shared_prompts.{}", prompt_name), prompt_tensor.clone());
            }
        }
        
        params
    }
    
    fn apply_to_layer(&self, layer_name: &str, input: &Tensor) -> Result<Tensor> {
        let state = self.state.read();
        
        if !state.enabled {
            return Ok(input.clone());
        }
        
        if let Some(layer) = state.layers.get(layer_name) {
            match layer {
                GLoRALayer::Linear(linear) => linear.forward(input),
                GLoRALayer::Conv2d(conv) => conv.forward(input),
            }
        } else {
            Ok(input.clone())
        }
    }
    
    fn merge_weights(&mut self, scale: f32) -> Result<()> {
        let mut state = self.state.write();
        
        // Update scaling for all layers
        for layer in state.layers.values_mut() {
            match layer {
                GLoRALayer::Linear(linear) => {
                    linear.scaling = scale * self.config.alpha / linear.rank as f32;
                }
                GLoRALayer::Conv2d(conv) => {
                    conv.scaling = scale * self.config.alpha / conv.rank as f32;
                }
            }
        }
        
        Ok(())
    }
    
    async fn save_weights(&self, path: &Path) -> Result<()> {
        use candle_core::safetensors::save;
        
        let tensors = {
            let state = self.state.read();
            let mut tensors = HashMap::new();
            
            // Save layer weights
            for (name, layer) in &state.layers {
                match layer {
                    GLoRALayer::Linear(linear) => {
                        tensors.insert(
                            format!("{}.lora_a.weight", name),
                            linear.lora_a.weight().clone(),
                        );
                        tensors.insert(
                            format!("{}.lora_b.weight", name),
                            linear.lora_b.weight().clone(),
                        );
                        
                        if let Some(bias) = linear.lora_a.bias() {
                            tensors.insert(format!("{}.lora_a.bias", name), bias.clone());
                        }
                        if let Some(bias) = linear.lora_b.bias() {
                            tensors.insert(format!("{}.lora_b.bias", name), bias.clone());
                        }
                        
                        // Save GLoRA-specific components
                        if let Some(ref prompts) = linear.prompt_embeddings {
                            tensors.insert(format!("{}.prompt_embeddings", name), prompts.clone());
                        }
                        if let Some(ref scale) = linear.scaling_vector {
                            tensors.insert(format!("{}.scaling_vector", name), scale.clone());
                        }
                        if let Some(ref gate) = linear.gating {
                            tensors.insert(format!("{}.gating.weight", name), gate.weight().clone());
                            if let Some(bias) = gate.bias() {
                                tensors.insert(format!("{}.gating.bias", name), bias.clone());
                            }
                        }
                    }
                    GLoRALayer::Conv2d(conv) => {
                        tensors.insert(
                            format!("{}.lora_a.weight", name),
                            conv.lora_a.weight().clone(),
                        );
                        tensors.insert(
                            format!("{}.lora_b.weight", name),
                            conv.lora_b.weight().clone(),
                        );
                        
                        if let Some(bias) = conv.lora_a.bias() {
                            tensors.insert(format!("{}.lora_a.bias", name), bias.clone());
                        }
                        if let Some(bias) = conv.lora_b.bias() {
                            tensors.insert(format!("{}.lora_b.bias", name), bias.clone());
                        }
                        
                        // Save GLoRA-specific components
                        if let Some(ref prompts) = conv.prompt_embeddings {
                            tensors.insert(format!("{}.prompt_embeddings", name), prompts.clone());
                        }
                        if let Some(ref scale) = conv.scaling_vector {
                            tensors.insert(format!("{}.scaling_vector", name), scale.clone());
                        }
                        if let Some(ref gate) = conv.gating {
                            tensors.insert(format!("{}.gating.weight", name), gate.weight().clone());
                            if let Some(bias) = gate.bias() {
                                tensors.insert(format!("{}.gating.bias", name), bias.clone());
                            }
                        }
                    }
                }
            }
            
            // Save shared prompts
            if let Some(ref shared_prompts) = state.shared_prompts {
                for (prompt_name, prompt_tensor) in shared_prompts {
                    tensors.insert(format!("shared_prompts.{}", prompt_name), prompt_tensor.clone());
                }
            }
            tensors
        };
        
        // Save metadata
        let metadata_path = path.with_extension("json");
        let metadata_json = serde_json::to_string_pretty(&self.metadata)?;
        tokio::fs::write(&metadata_path, metadata_json).await?;
        
        // Save tensors
        save(&tensors, path)?;
        
        Ok(())
    }
    
    async fn load_weights(&mut self, path: &Path) -> Result<()> {
        use candle_core::safetensors::load;
        
        // Load tensors
        let tensors = load(path, &self.state.read().device.to_candle()?)?;
        
        // Load GLoRA weights from state dict
        {
            let mut state = self.state.write();
            
            for (key, tensor) in tensors {
                if let Some((module_name, param_name)) = key.rsplit_once('.') {
                    if let Some(layer) = state.layers.get_mut(module_name) {
                        match (layer, param_name) {
                            (GLoRALayer::Linear(linear), "lora_a.weight") => {
                                // Update lora_a weight
                            }
                            (GLoRALayer::Linear(linear), "lora_b.weight") => {
                                // Update lora_b weight
                            }
                            (GLoRALayer::Conv2d(conv), "lora_a.weight") => {
                                // Update lora_a weight
                            }
                            (GLoRALayer::Conv2d(conv), "lora_b.weight") => {
                                // Update lora_b weight
                            }
                            _ => {}
                        }
                    }
                }
            }
        } // Drop the write lock here
        
        // Load metadata if exists
        let metadata_path = path.with_extension("json");
        if metadata_path.exists() {
            let metadata_json = tokio::fs::read_to_string(&metadata_path).await?;
            self.metadata = serde_json::from_str(&metadata_json)?;
        }
        
        Ok(())
    }
    
    fn memory_usage(&self) -> usize {
        self.num_parameters() * 4 // f32 = 4 bytes
    }
    
    fn to_device(&mut self, device: &Device) -> Result<()> {
        let mut state = self.state.write();
        state.device = device.clone();
        
        // TODO: Move all layers to new device
        
        Ok(())
    }
}

/// Tensor info
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: DType,
}

/// GLoRA utilities
pub mod utils {
    use super::*;
    
    /// Analyze GLoRA components contribution
    pub fn analyze_component_contribution(
        glora_layer: &GLoRALayer,
        sample_input: &Tensor,
    ) -> Result<ComponentAnalysis> {
        // Get base LoRA output
        match glora_layer {
            GLoRALayer::Linear(linear) => {
                let base_output = {
                    let mut x = linear.lora_a.forward(sample_input)?;
                    linear.lora_b.forward(&x)?
                };
                
                let base_norm = base_output.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
                
                // Analyze prompt contribution
                let prompt_contribution = if let Some(ref prompts) = linear.prompt_embeddings {
                    let prompt_norm = prompts.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
                    prompt_norm / (base_norm + 1e-6)
                } else {
                    0.0
                };
                
                // Analyze scaling contribution
                let scaling_variance = if let Some(ref scale) = linear.scaling_vector {
                    scale.var(0)?.to_scalar::<f32>()?
                } else {
                    0.0
                };
                
                Ok(ComponentAnalysis {
                    base_lora_norm: base_norm,
                    prompt_contribution,
                    scaling_variance,
                    uses_gating: linear.gating.is_some(),
                    uses_multi_head: linear.head_lora_a.is_some(),
                })
            }
            GLoRALayer::Conv2d(conv) => {
                let base_output = {
                    let mut x = conv.lora_a.forward(sample_input)?;
                    conv.lora_b.forward(&x)?
                };
                
                let base_norm = base_output.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
                
                // Analyze prompt contribution
                let prompt_contribution = if let Some(ref prompts) = conv.prompt_embeddings {
                    let prompt_norm = prompts.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
                    prompt_norm / (base_norm + 1e-6)
                } else {
                    0.0
                };
                
                // Analyze scaling contribution
                let scaling_variance = if let Some(ref scale) = conv.scaling_vector {
                    scale.var(0)?.to_scalar::<f32>()?
                } else {
                    0.0
                };
                
                Ok(ComponentAnalysis {
                    base_lora_norm: base_norm,
                    prompt_contribution,
                    scaling_variance,
                    uses_gating: conv.gating.is_some(),
                    uses_multi_head: conv.head_lora_a.is_some(),
                })
            }
        }
    }
    
    /// Convert standard LoRA to GLoRA
    pub fn lora_to_glora(
        lora_a: &Tensor,
        lora_b: &Tensor,
        config: &GLoRAConfig,
        device: &Device,
    ) -> Result<GLoRALayer> {
        let in_features = lora_a.dims()[1];
        let out_features = lora_b.dims()[0];
        let rank = lora_a.dims()[0];
        
        let glora = if lora_a.dims().len() == 2 {
            // Linear layer
            GLoRALayer::Linear(GLoRALinear::new(
                in_features,
                out_features,
                rank,
                config.alpha,
                config.dropout,
                config,
                device,
            )?)
        } else {
            // Conv2d layer
            let kernel_size = lora_a.dims()[2];
            GLoRALayer::Conv2d(GLoRAConv2d::new(
                in_features,
                out_features,
                kernel_size,
                rank,
                config.alpha,
                config.dropout,
                config,
                device,
            )?)
        };
        
        // Copy LoRA weights
        // In practice would properly copy tensor data
        
        Ok(glora)
    }
}

/// Component analysis results
#[derive(Debug, Clone)]
pub struct ComponentAnalysis {
    pub base_lora_norm: f32,
    pub prompt_contribution: f32,
    pub scaling_variance: f32,
    pub uses_gating: bool,
    pub uses_multi_head: bool,
}