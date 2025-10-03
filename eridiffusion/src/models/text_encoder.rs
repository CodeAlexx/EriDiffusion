use crate::models::attention::{Attention, FeedForward, GELU};
use crate::ops::LayerNorm;
use crate::ops::Linear;
use anyhow::anyhow;
use flame_core::device::Device;
use flame_core::embedding::Embedding;
use flame_core::optimizers::{Adam, SGD};
use flame_core::{DType, Error, Result, Shape, Tensor};
use std::{collections::HashMap, sync::Arc};

pub struct CLIPTextEncoder {
    embeddings: CLIPTextEmbeddings,
    encoder: CLIPEncoder,
    final_layer_norm: LayerNorm,
    text_projection: Option<Linear>,
    config: CLIPConfig,
    device: flame_core::device::Device,
    tokenizer: Option<tokenizers::Tokenizer>,
}
pub struct CLIPTextEncoderOutput {
    pub last_hidden_state: Tensor,
    pub pooled_output: Tensor,
}

struct CLIPTextEmbeddings {
    token_embedding: Embedding,
    position_embedding: Embedding,
}
struct CLIPEncoder {
    layers: Vec<CLIPEncoderLayer>,
}

struct CLIPEncoderOutput {
    last_hidden_state: Tensor,
}
struct CLIPEncoderLayer {
    self_attn: CLIPAttention,
    layer_norm1: LayerNorm,
    mlp: CLIPMLP,
    layer_norm2: LayerNorm,
}

struct CLIPAttention {
    k_proj: Linear,
    v_proj: Linear,
    q_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f32,
}

struct CLIPMLP {
    fc1: Linear,
    fc2: Linear,
    activation: String,
}
pub struct T5Encoder {
    embeddings: T5Embeddings,
    encoder: T5Stack,
    config: T5Config,
    device: flame_core::device::Device,
}

pub struct T5Output {
    pub last_hidden_state: Tensor,
}

struct T5Embeddings {
    token_embeddings: Embedding,
}

/// T5 attention module
pub struct T5Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    num_heads: usize,
    head_dim: usize,
    relative_attention_bias: Option<Tensor>,
}

impl T5Attention {
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        has_relative_attention_bias: bool,
        device: flame_core::device::Device,
    ) -> Result<Self> {
        let head_dim = hidden_size / num_heads;
        Ok(Self {
            q: Linear::new(hidden_size, hidden_size, true, &device.cuda_device())?,
            k: Linear::new(hidden_size, hidden_size, true, &device.cuda_device())?,
            v: Linear::new(hidden_size, hidden_size, true, &device.cuda_device())?,
            o: Linear::new(hidden_size, hidden_size, true, &device.cuda_device())?,
            num_heads,
            head_dim,
            relative_attention_bias: if has_relative_attention_bias {
                Some(Tensor::zeros(
                    Shape::from_dims(&[32, num_heads]),
                    device.cuda_device().clone(),
                )?)
            } else {
                None
            },
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.shape().dims();
        let (b, n, _) = (dims[0], dims[1], dims[2]);

        let q = self
            .q
            .forward(x)?
            .reshape(&[b, n, self.num_heads, self.head_dim])?
            .permute(&[0, 2, 1, 3])?;
        let k = self
            .k
            .forward(x)?
            .reshape(&[b, n, self.num_heads, self.head_dim])?
            .permute(&[0, 2, 3, 1])?;
        let v = self
            .v
            .forward(x)?
            .reshape(&[b, n, self.num_heads, self.head_dim])?
            .permute(&[0, 2, 1, 3])?;

        let scale = (self.head_dim as f32).powf(-0.5);
        let attn = q.matmul(&k)?.mul_scalar(scale)?.softmax(-1)?;
        let out = attn.matmul(&v)?.permute(&[0, 2, 1, 3])?.reshape(&[
            b,
            n,
            self.num_heads * self.head_dim,
        ])?;

        self.o.forward(&out)
    }
}

type FlameShape = flame_core::Shape;
type FlameDevice = flame_core::device::Device;

// FLAME-based text encoder implementations
// Supports CLIP and T5 text encoders for diffusion models

// Extension trait for Tensor to add missing methods

/// CLIP text encoder configuration
#[derive(Clone, Debug)]
pub struct CLIPConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub max_position_embeddings: usize,
    pub layer_norm_eps: f32,
    pub hidden_act: String,
    pub projection_dim: Option<usize>,
}

impl CLIPConfig {
    /// SDXL CLIP-L config
    pub fn clip_l() -> Self {
        Self {
            vocab_size: 49408,
            hidden_size: 768,
            intermediate_size: 3072,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            max_position_embeddings: 77,
            layer_norm_eps: 1e-5,
            hidden_act: "quick_gelu".to_string(),
            projection_dim: None,
        }
    }

    /// SDXL CLIP-G config
    pub fn clip_g() -> Self {
        Self {
            vocab_size: 49408,
            hidden_size: 1280,
            intermediate_size: 5120,
            num_hidden_layers: 32,
            num_attention_heads: 20,
            max_position_embeddings: 77,
            layer_norm_eps: 1e-5,
            hidden_act: "gelu".to_string(),
            projection_dim: Some(1280),
        }
    }
}

/// T5 encoder configuration
#[derive(Clone, Debug)]
pub struct T5Config {
    pub vocab_size: usize,
    pub d_model: usize,
    pub d_ff: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub relative_attention_num_buckets: usize,
    pub relative_attention_max_distance: usize,
    pub layer_norm_epsilon: f32,
    pub dropout_rate: f32,
}

impl T5Config {
    /// T5-XXL config for SD3/Flux
    pub fn t5_xxl() -> Self {
        Self {
            vocab_size: 32128,
            d_model: 4096,
            d_ff: 10240,
            num_layers: 24,
            num_heads: 64,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            layer_norm_epsilon: 1e-6,
            dropout_rate: 0.1,
        }
    }
}

/// FLAME CLIP text encoder

impl CLIPTextEncoder {
    pub fn from_weights(
        weights: std::collections::HashMap<String, Tensor>,
        hidden_size: usize,
        device: &flame_core::device::Device,
    ) -> Result<Self> {
        // Infer config from weights - actually examine the loaded tensors
        let num_layers = weights
            .keys()
            .filter(|k| k.contains("encoder.layers.") & k.contains(".self_attn.k_proj.weight"))
            .count();
        let vocab_size = weights
            .get("embeddings.token_embedding.weight")
            .map(|t| t.shape().dims()[0])
            .unwrap_or(49408);
        let max_position_embeddings = weights
            .get("embeddings.position_embedding.weight")
            .map(|t| t.shape().dims()[0])
            .unwrap_or(77);

        let config = CLIPConfig {
            vocab_size,
            hidden_size,
            intermediate_size: hidden_size * 4,
            num_hidden_layers: num_layers,
            num_attention_heads: hidden_size / 64,
            max_position_embeddings,
            layer_norm_eps: 1e-5,
            hidden_act: "quick_gelu".to_string(),
            projection_dim: Some(hidden_size),
        };

        Self::new(config, device, weights)
    }

    pub fn new(
        config: CLIPConfig,
        device: &flame_core::device::Device,
        weights: std::collections::HashMap<String, Tensor>,
    ) -> Result<Self> {
        let embeddings = CLIPTextEmbeddings::new(&config, &device, &weights)?;
        let encoder = CLIPEncoder::new(&config, &device, &weights)?;

        // Load final layer norm weights
        let final_layer_norm = {
            let mut ln = LayerNorm::new(
                vec![config.hidden_size],
                config.layer_norm_eps,
                device.cuda_device().clone(),
            )?;

            // Try different possible key prefixes
            if let Some(weight) = weights
                .get("text_model.final_layer_norm.weight")
                .or_else(|| weights.get("final_layer_norm.weight"))
            {
                ln.weight = Some(weight.clone());
            }
            if let Some(bias) = weights
                .get("text_model.final_layer_norm.bias")
                .or_else(|| weights.get("final_layer_norm.bias"))
            {
                ln.bias = Some(bias.clone());
            }
            ln
        };

        let text_projection = if let Some(proj_dim) = config.projection_dim {
            // Load text projection weights if available
            if let Some(weight) = weights
                .get("text_projection.weight")
                .or_else(|| weights.get("text_model.text_projection.weight"))
            {
                let mut linear =
                    Linear::new(config.hidden_size, proj_dim, true, &device.cuda_device())?;
                linear.weight = weight.clone();
                if let Some(bias) = weights
                    .get("text_projection.bias")
                    .or_else(|| weights.get("text_model.text_projection.bias"))
                {
                    linear.bias = Some(bias.clone());
                }
                Some(linear)
            } else {
                Some(Linear::new(config.hidden_size, proj_dim, true, &device.cuda_device())?)
            }
        } else {
            None
        };

        // Try to load tokenizer
        let tokenizer =
            match tokenizers::Tokenizer::from_file("/home/alex/SwarmUI/Models/clip/tokenizer.json")
            {
                Ok(t) => Some(t),
                Err(e) => {
                    println!(
                        "Warning: Failed to load CLIP tokenizer: {}. Text encoding will fail.",
                        e
                    );
                    None
                }
            };

        Ok(Self {
            embeddings,
            encoder,
            final_layer_norm,
            text_projection,
            config,
            device: device.clone(),
            tokenizer,
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<CLIPTextEncoderOutput> {
        // Get embeddings
        let hidden_states = self.embeddings.forward(input_ids)?;

        // Create causal attention mask if needed
        let attention_mask = if let Some(mask) = attention_mask {
            Some(self.create_attention_mask(mask)?)
        } else {
            None
        };

        // Run through encoder
        let encoder_outputs = self.encoder.forward(&hidden_states, attention_mask.as_ref())?;

        // Final layer norm
        let last_hidden_state =
            self.final_layer_norm.forward(&encoder_outputs.last_hidden_state)?;

        // Get pooled output (EOS token)
        let pooled_output = self.get_pooled_output(&last_hidden_state, input_ids)?;

        // Apply text projection if exists
        let pooled_output = if let Some(proj) = &self.text_projection {
            proj.forward(&pooled_output)?
        } else {
            pooled_output
        };

        Ok(CLIPTextEncoderOutput { last_hidden_state, pooled_output })
    }

    pub fn encode(&self, text: &str, max_length: usize) -> Result<Tensor> {
        // Use the stored tokenizer
        let tokenizer = self.tokenizer.as_ref().ok_or_else(|| {
            flame_core::Error::InvalidOperation("Tokenizer not loaded".into())
        })?;

        // Tokenize the text
        let encoding = tokenizer.encode(text, false).map_err(|e| {
            flame_core::Error::InvalidOperation(format!("Tokenization failed: {}", e))
        })?;

        // Get the token IDs
        let mut tokens = encoding.get_ids().to_vec();
        let mut attention_mask = encoding.get_attention_mask().to_vec();

        // Truncate if needed
        if tokens.len() > max_length {
            tokens.truncate(max_length);
            attention_mask.truncate(max_length);
        }

        // Pad to max_length
        while tokens.len() < max_length {
            tokens.push(0);
            attention_mask.push(0);
        }

        // Convert to f32 for tensor
        let padded_ids: Vec<f32> = tokens.iter().map(|&id| id as f32).collect();

        // Convert to tensor
        let input_ids = Tensor::from_vec(
            padded_ids,
            Shape::from_dims(&[1, max_length]),
            self.device.cuda_device().clone(),
        )?;
        let attention_mask_f32: Vec<f32> = attention_mask.iter().map(|&m| m as f32).collect();
        let attention_mask = Tensor::from_vec(
            attention_mask_f32,
            Shape::from_dims(&[1, max_length]),
            self.device.cuda_device().clone(),
        )?;

        // Forward pass
        let output = self.forward(&input_ids, Some(&attention_mask))?;
        Ok(output.last_hidden_state)
    }

    fn create_attention_mask(&self, mask: &Tensor) -> Result<Tensor> {
        // Convert [batch, seq_len] to [batch, 1, 1, seq_len]
        let mask = mask.unsqueeze(1)?.unsqueeze(1)?;
        // Convert to attention mask values (0 -> -10000, 1 -> 0)
        let mask = mask.mul_scalar(-1.0 as f32)?.add_scalar(1.0)?;
        mask.mul_scalar(-10000.0 as f32)
    }

    fn get_pooled_output(&self, hidden_states: &Tensor, input_ids: &Tensor) -> Result<Tensor> {
        // Find EOS token positions (token_id = 49407 for CLIP)
        let eos_token_id = 49407;
        let input_shape = input_ids.shape();
        let input_dims = input_shape.dims();
        let batch_size = input_dims[0];
        let seq_len = input_dims[1];

        // For simplicity, assume EOS is at the last valid position
        // In production, would actually search for EOS token
        let indices = flame_core::Tensor::arange(
            0.0,
            batch_size as f32,
            1.0,
            self.device.cuda_device().clone(),
        )?;

        // Index into hidden states to get pooled output
        // Shape: [batch, hidden_size]
        Ok(hidden_states.index_select(0, &indices)?)
    }
}

/// CLIP text encoder output
impl CLIPTextEmbeddings {
    fn new(
        config: &CLIPConfig,
        device: &flame_core::device::Device,
        weights: &std::collections::HashMap<String, Tensor>,
    ) -> Result<Self> {
        // Try different possible key prefixes for the embeddings
        let token_weight = weights
            .get("text_model.embeddings.token_embedding.weight")
            .or_else(|| weights.get("embeddings.token_embedding.weight"))
            .or_else(|| weights.get("token_embedding.weight"))
            .ok_or_else(|| {
                flame_core::Error::InvalidOperation(
                    "Token embedding weight not found".to_string(),
                )
            })?;

        let position_weight = weights
            .get("text_model.embeddings.position_embedding.weight")
            .or_else(|| weights.get("embeddings.position_embedding.weight"))
            .or_else(|| weights.get("position_embedding.weight"))
            .ok_or_else(|| {
                flame_core::Error::InvalidOperation(
                    "Position embedding weight not found".to_string(),
                )
            })?;

        // Create embeddings with the loaded weights
        let token_embedding = {
            let dims = token_weight.shape().dims();
            println!("DEBUG: Token embedding weight shape: {:?}", dims);
            let mut emb = Embedding::new(dims[0], dims[1], device.cuda_device().clone())?;
            emb.weight = token_weight.clone();
            emb
        };

        let position_embedding = {
            let dims = position_weight.shape().dims();
            println!("DEBUG: Position embedding weight shape: {:?}", dims);
            let mut emb = Embedding::new(dims[0], dims[1], device.cuda_device().clone())?;
            emb.weight = position_weight.clone();
            emb
        };

        Ok(Self { token_embedding, position_embedding })
    }

    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        println!("DEBUG CLIPTextEmbeddings forward: input_ids shape = {:?}", input_ids.shape());
        let seq_len = input_ids.shape().dims()[1];

        // Get token embeddings
        let inputs_embeds = self.token_embedding.forward(input_ids)?;
        println!(
            "DEBUG CLIPTextEmbeddings: token_embedding output shape = {:?}",
            inputs_embeds.shape()
        );

        // Create position ids
        let position_ids =
            flame_core::Tensor::arange(0.0, seq_len as f32, 1.0, input_ids.device().clone())?
                .unsqueeze(0)?
                .broadcast_to(input_ids.shape())?;

        // Get position embeddings
        let position_embeds = self.position_embedding.forward(&position_ids)?;
        println!(
            "DEBUG CLIPTextEmbeddings: position_embedding output shape = {:?}",
            position_embeds.shape()
        );

        // Add embeddings
        let result = inputs_embeds.add(&position_embeds)?;
        println!("DEBUG CLIPTextEmbeddings: final embedding shape = {:?}", result.shape());
        Ok(result)
    }
}

/// CLIP encoder

impl CLIPEncoder {
    fn new(
        config: &CLIPConfig,
        device: &flame_core::device::Device,
        weights: &std::collections::HashMap<String, Tensor>,
    ) -> Result<Self> {
        let mut layers = Vec::new();

        for i in 0..config.num_hidden_layers {
            layers.push(CLIPEncoderLayer::new(config, device, weights, i)?);
        }

        Ok(Self { layers })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<CLIPEncoderOutput> {
        let mut hidden_states = hidden_states.clone();

        for (idx, layer) in self.layers.iter().enumerate() {
            let new_hidden = layer.forward(&hidden_states, attention_mask)?;
            // Drop the old hidden states to free memory immediately
            drop(hidden_states);
            hidden_states = new_hidden;

            // Every 3 layers, force a more aggressive cleanup
            // This prevents accumulation of intermediate tensors
            if (idx + 1) % 3 == 0 {
                // Force GPU synchronization to ensure cleanup happens
                let _ = hidden_states.device().synchronize();
            }
        }

        Ok(CLIPEncoderOutput { last_hidden_state: hidden_states })
    }
}

/// CLIP encoder output

impl CLIPEncoderLayer {
    fn new(
        config: &CLIPConfig,
        device: &flame_core::device::Device,
        weights: &std::collections::HashMap<String, Tensor>,
        layer_idx: usize,
    ) -> Result<Self> {
        let self_attn = CLIPAttention::new(config, device)?;
        let cuda_device = device.cuda_device();
        let layer_norm1 =
            LayerNorm::new(vec![config.hidden_size], config.layer_norm_eps, cuda_device.clone())?;
        let mlp = CLIPMLP::new(config, device)?;
        let layer_norm2 =
            LayerNorm::new(vec![config.hidden_size], config.layer_norm_eps, cuda_device.clone())?;

        Ok(Self { self_attn, layer_norm1, mlp, layer_norm2 })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        // Self attention
        let residual = hidden_states.clone();
        let hidden_states = self.layer_norm1.forward(hidden_states)?;
        let hidden_states = self.self_attn.forward(&hidden_states, attention_mask)?;
        let hidden_states = residual.add(&hidden_states)?;

        // Force cleanup of intermediate attention tensors
        // This is critical for preventing OOM during text encoding
        drop(residual);

        // MLP
        let residual = hidden_states.clone();
        let hidden_states = self.layer_norm2.forward(&hidden_states)?;
        let hidden_states = self.mlp.forward(&hidden_states)?;
        let output = residual.add(&hidden_states)?;

        // Another cleanup point
        drop(hidden_states);
        drop(residual);

        Ok(output)
    }
}

/// CLIP attention

impl CLIPAttention {
    fn new(config: &CLIPConfig, device: &flame_core::device::Device) -> Result<Self> {
        let embed_dim = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let head_dim = embed_dim / num_heads;

        Ok(Self {
            k_proj: Linear::new(embed_dim, embed_dim, true, device.cuda_device())?,
            v_proj: Linear::new(embed_dim, embed_dim, true, device.cuda_device())?,
            q_proj: Linear::new(embed_dim, embed_dim, true, device.cuda_device())?,
            out_proj: Linear::new(embed_dim, embed_dim, true, device.cuda_device())?,
            num_heads,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let hidden_shape = hidden_states.shape();
        let hidden_dims = hidden_shape.dims();
        let batch_size = hidden_dims[0];
        let seq_len = hidden_dims[1];

        // Project to Q, K, V
        let q = self
            .q_proj
            .forward(hidden_states)?
            .reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])?
            .transpose_dims(1, 2)?;
        let k = self
            .k_proj
            .forward(hidden_states)?
            .reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])?
            .transpose_dims(1, 2)?;
        let v = self
            .v_proj
            .forward(hidden_states)?
            .reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])?
            .transpose_dims(1, 2)?;

        // Attention scores
        let k_t = k.transpose_dims(2, 3)?;
        let scores = q.matmul(&k_t)?.mul_scalar(self.scale as f32)?;

        // Clean up Q and K tensors early
        drop(q);
        drop(k);
        drop(k_t);

        // Apply attention mask if provided
        let scores = if let Some(mask) = attention_mask { scores.add(mask)? } else { scores };

        // Softmax
        let attn_weights = scores.softmax(-1)?;
        drop(scores); // Free scores memory

        // Apply attention to values
        let attn_output = attn_weights.matmul(&v)?.transpose_dims(1, 2)?.reshape(&[
            batch_size,
            seq_len,
            self.num_heads * self.head_dim,
        ])?;

        // Clean up intermediate tensors
        drop(attn_weights);
        drop(v);

        // Output projection
        self.out_proj.forward(&attn_output)
    }
}

/// CLIP MLP

impl CLIPMLP {
    fn new(config: &CLIPConfig, device: &flame_core::device::Device) -> Result<Self> {
        Ok(Self {
            fc1: Linear::new(
                config.hidden_size,
                config.intermediate_size,
                true,
                device.cuda_device(),
            )?,
            fc2: Linear::new(
                config.intermediate_size,
                config.hidden_size,
                true,
                device.cuda_device(),
            )?,
            activation: config.hidden_act.clone(),
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let hidden_states = self.fc1.forward(hidden_states)?;

        let hidden_states = match self.activation.as_str() {
            "gelu" => hidden_states.gelu()?,
            "quick_gelu" => {
                // Quick GELU approximation: x * sigmoid(1.702 * x)
                let x_scaled = hidden_states.mul_scalar(1.702 as f32)?;
                let sigmoid = x_scaled.sigmoid()?;
                hidden_states.mul(&sigmoid)?
            }
            _ => hidden_states.gelu()?, // Default to GELU
        };

        self.fc2.forward(&hidden_states)
    }
}

/// FLAME T5 encoder

impl T5Encoder {
    pub fn new(
        config: T5Config,
        device: &flame_core::device::Device,
        weights: std::collections::HashMap<String, Tensor>,
    ) -> Result<Self> {
        let embeddings = T5Embeddings::new(&config, device, &weights)?;
        let encoder = T5Stack::new(&config, device, &weights)?;

        Ok(Self { embeddings, encoder, config, device: device.clone() })
    }

    pub fn forward(&self, input_ids: &Tensor) -> Result<T5Output> {
        // Get embeddings
        let hidden_states = self.embeddings.forward(input_ids)?;

        // Run through encoder
        let encoder_outputs = self.encoder.forward(&hidden_states)?;

        Ok(T5Output { last_hidden_state: encoder_outputs })
    }
}

/// T5 embeddings

impl T5Embeddings {
    fn new(
        config: &T5Config,
        device: &flame_core::device::Device,
        weights: &std::collections::HashMap<String, Tensor>,
    ) -> Result<Self> {
        // Use the pre-loaded embedding weights instead of creating new ones
        let embed_weight_key = "shared.weight";
        let embed_weight = weights.get(embed_weight_key).ok_or_else(|| {
            flame_core::Error::InvalidOperation(format!(
                "T5 embedding weight '{}' not found in loaded weights",
                embed_weight_key
            ))
        })?;

        // Create embedding with pre-loaded weights
        let mut embedding =
            Embedding::new(config.vocab_size, config.d_model, device.cuda_device().clone())?;
        // Replace the random weights with loaded weights
        embedding.weight = embed_weight.clone();

        Ok(Self { token_embeddings: embedding })
    }

    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.token_embeddings.forward(input_ids)
    }
}

struct T5Stack {
    layers: Vec<T5Layer>,
    final_layer_norm: LayerNorm,
}

impl T5Stack {
    fn new(
        config: &T5Config,
        device: &flame_core::device::Device,
        weights: &std::collections::HashMap<String, Tensor>,
    ) -> Result<Self> {
        let mut layers = Vec::new();

        for i in 0..config.num_layers {
            layers.push(T5Layer::new(config, device)?);
        }

        let final_layer_norm = LayerNorm::new(
            vec![config.d_model],
            config.layer_norm_epsilon,
            device.cuda_device().clone(),
        )?;

        Ok(Self { layers, final_layer_norm })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let mut hidden_states = hidden_states.clone();

        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states)?;
        }

        self.final_layer_norm.forward(&hidden_states)
    }
}

struct T5Layer {
    self_attention: T5Attention,
    layer_norm: LayerNorm,
    mlp: T5DenseGatedActDense,
    dropout: f32,
}

impl T5Layer {
    fn new(config: &T5Config, device: &flame_core::device::Device) -> Result<Self> {
        Ok(Self {
            self_attention: T5Attention::new(
                config.d_model,
                config.num_heads,
                true,
                device.clone(),
            )?,
            layer_norm: {
                LayerNorm::new(
                    vec![config.d_model],
                    config.layer_norm_epsilon,
                    device.cuda_device().clone(),
                )?
            },
            mlp: T5DenseGatedActDense::new(config, device)?,
            dropout: config.dropout_rate,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // T5 layer with pre-normalization

        // Self-attention with residual
        let normed = self.layer_norm.forward(hidden_states)?;
        let attention_output = self.self_attention.forward(&normed)?;
        let hidden_states = hidden_states.add(&attention_output)?;

        // MLP with residual
        let normed = self.layer_norm.forward(&hidden_states)?;
        let mlp_output = self.mlp.forward(&normed)?;
        let output = hidden_states.add(&mlp_output)?;

        Ok(output)
    }
}

// Helper layer types removed - using flame_core imports

// T5 component implementations
struct T5DenseGatedActDense {
    wi_0: Linear, // First gating linear layer
    wi_1: Linear, // Second gating linear layer
    wo: Linear,   // Output linear layer
    dropout: f32,
}

impl T5DenseGatedActDense {
    fn new(config: &T5Config, device: &flame_core::device::Device) -> Result<Self> {
        Ok(Self {
            wi_0: Linear::new(config.d_model, config.d_ff, true, device.cuda_device())?,
            wi_1: Linear::new(config.d_model, config.d_ff, true, device.cuda_device())?,
            wo: Linear::new(config.d_ff, config.d_model, true, device.cuda_device())?,
            dropout: config.dropout_rate,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // T5 uses Gated Linear Units (GLU)
        // hidden_gelu = gelu(wi_0(x))
        let hidden_gelu = self.wi_0.forward(hidden_states)?.gelu()?;

        // hidden_linear = wi_1(x)
        let hidden_linear = self.wi_1.forward(hidden_states)?;

        // hidden = hidden_gelu * hidden_linear
        let hidden = hidden_gelu.mul(&hidden_linear)?;

        // Apply dropout if in training mode (simplified - just return without dropout for now)
        // In real implementation would check training mode

        // output = wo(hidden)
        self.wo.forward(&hidden)
    }
}
