use crate::ops::{LayerNorm, Linear};
use anyhow::anyhow;
use flame_core::device::Device;
use flame_core::Result;
use flame_core::{embedding::Embedding, DType, Error, Shape, Tensor};
use std::{collections::HashMap, sync::Arc};

fn to_bf16_tensor(tensor: &Tensor) -> Result<Tensor> {
    if tensor.dtype() == DType::BF16 {
        Ok(tensor.clone())
    } else {
        tensor.to_dtype(DType::BF16)
    }
}

fn to_bf16_optional(tensor: Option<&Tensor>) -> Result<Option<Tensor>> {
    if let Some(t) = tensor {
        Ok(Some(to_bf16_tensor(t)?))
    } else {
        Ok(None)
    }
}

#[derive(Clone)]
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
    pub pad_token_id: usize,
}
pub struct CLIPTextEncoder {
    embeddings: CLIPTextEmbeddings,
    encoder: CLIPEncoder,
    final_layer_norm: LayerNorm,
    text_projection: Option<Linear>,
    config: CLIPConfig,
    device: flame_core::device::Device,
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
    final_layer_norm: LayerNorm,
    config: T5Config,
    device: flame_core::device::Device,
}
pub struct T5Output {
    pub last_hidden_state: Tensor,
}
struct T5Embeddings {
    token_embeddings: Embedding,
}
struct T5Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    num_heads: usize,
    d_kv: usize,
    scale: f32,
}
struct T5FeedForward {
    wi: Linear,
    wi_gate: Option<Linear>,
    wo: Linear,
    activation: String,
}

// FLAME uses flame_core::device::Device instead of Device

/// CLIP text encoder configuration
// Extension trait for Tensor to add missing methods
trait TensorExt {
    fn sum_dim(&self, dim: usize) -> Result<Tensor>;
    fn add_scalar(&self, scalar: f32) -> Result<Tensor>;
    fn mul_scalar(&self, scalar: f32) -> Result<Tensor>;
    fn square(&self) -> Result<Tensor>;
}

impl TensorExt for Tensor {
    fn sum_dim(&self, dim: usize) -> Result<Tensor> {
        // Sum along dimension
        Ok(self.sum_dim(dim)?)
    }

    fn add_scalar(&self, scalar: f32) -> Result<Tensor> {
        // Add scalar to all elements
        let scalar_tensor = Tensor::full(self.shape().clone(), scalar, self.device().clone())?;
        self.add(&scalar_tensor)
    }

    fn mul_scalar(&self, scalar: f32) -> Result<Tensor> {
        // Multiply all elements by scalar
        let scalar_tensor = Tensor::full(self.shape().clone(), scalar, self.device().clone())?;
        self.mul(&scalar_tensor)
    }

    fn square(&self) -> Result<Tensor> {
        // Element-wise square
        self.mul(self)
    }
}

#[derive(Clone)]
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
    pub pad_token_id: usize,
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
            projection_dim: Some(768),
            pad_token_id: 49407,
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
            pad_token_id: 49407,
        }
    }
}

/// T5 encoder configuration

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
            pad_token_id: 0,
        }
    }
}

/// FLAME CLIP text encoder

impl CLIPTextEncoder {
    pub fn new(
        config: CLIPConfig,
        device: flame_core::device::Device,
        weights: std::collections::HashMap<String, Tensor>,
    ) -> Result<Self> {
        let mut weights = weights;
        let mut alias_keys = Vec::new();
        for key in weights.keys() {
            if let Some(stripped) = key.strip_prefix("text_model.") {
                if !weights.contains_key(stripped) {
                    alias_keys.push((stripped.to_string(), key.clone()));
                }
            }
        }
        for (alias, original) in alias_keys {
            if let Some(tensor) = weights.get(&original) {
                weights.insert(alias, tensor.alias());
            }
        }

        let embeddings = CLIPTextEmbeddings::new(&config, &device, &weights)?;
        let encoder = CLIPEncoder::new(&config, &device, &weights)?;
        let final_layer_norm_weight =
            weights.get("text_model.final_layer_norm.weight").ok_or_else(|| {
                flame_core::Error::InvalidOperation(
                    "text_model.final_layer_norm.weight not found".to_string(),
                )
            })?;
        let final_layer_norm_bias =
            weights.get("text_model.final_layer_norm.bias").ok_or_else(|| {
                flame_core::Error::InvalidOperation(
                    "text_model.final_layer_norm.bias not found".to_string(),
                )
            })?;
        let final_layer_norm = {
            let mut ln = LayerNorm::new(
                vec![config.hidden_size],
                config.layer_norm_eps,
                device.cuda_device().clone(),
            )?;
            ln.weight = Some(to_bf16_tensor(final_layer_norm_weight)?);
            ln.bias = Some(to_bf16_tensor(final_layer_norm_bias)?);
            ln
        };

        let text_projection = if let Some(proj_dim) = config.projection_dim {
            if let Some(weight) = weights.get("text_projection.weight") {
                let bias = weights.get("text_projection.bias");
                Some({
                    let in_features = weight.shape().dims()[1];
                    let out_features = weight.shape().dims()[0];
                    let mut linear =
                        Linear::new(in_features, out_features, true, device.cuda_device())?;
                    linear.weight = to_bf16_tensor(weight)?;
                    linear.bias = to_bf16_optional(bias)?;
                    linear
                })
            } else {
                None
            }
        } else {
            None
        };

        Ok(Self { embeddings, encoder, final_layer_norm, text_projection, config, device })
    }

    /// Load from safetensors weights
    pub fn from_pretrained(
        model_path: &str,
        config: CLIPConfig,
        device: flame_core::device::Device,
    ) -> Result<Self> {
        // Load weights from safetensors file
        let loader = crate::loaders::WeightLoader::from_safetensors(model_path, device.clone())?;
        let mut weights = std::collections::HashMap::new();

        // Extract all weights into HashMap
        for (key, tensor) in loader.weights.iter() {
            weights.insert(key.clone(), tensor.clone());
        }

        Self::new(config, device, weights)
    }

    /// Forward pass
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
        log::debug!("CLIP encoder input dtype: {:?}", hidden_states.dtype());
        let encoder_outputs = self.encoder.forward(&hidden_states, attention_mask.as_ref())?;

        // Final layer norm
        let last_hidden_state = self.final_layer_norm.forward(&encoder_outputs)?;

        // Get pooled output (last token)
        let pooled_output = self.get_pooled_output(&last_hidden_state, input_ids)?;

        // Apply text projection if exists
        let pooled_output = if let Some(proj) = &self.text_projection {
            proj.forward(&pooled_output)?
        } else {
            pooled_output
        };

        Ok(CLIPTextEncoderOutput { last_hidden_state, pooled_output })
    }

    /// Encode a batch of text prompts
    pub fn encode_batch(&self, prompts: &[String]) -> Result<Tensor> {
        // Load CLIP tokenizer
        let tokenizer_path = "/home/alex/diffusers-rs/tokenizers/clip_tokenizer.json";
        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path).map_err(|e| {
            flame_core::Error::InvalidOperation(format!("Failed to load tokenizer: {:?}", e))
        })?;

        let mut all_ids = Vec::new();

        for prompt in prompts {
            // Tokenize the prompt
            let encoding = tokenizer.encode(prompt.as_str(), true).map_err(|e| {
                flame_core::Error::InvalidOperation(format!("Tokenization failed: {:?}", e))
            })?;

            let mut ids = encoding.get_ids().to_vec();

            // Pad or truncate to max length (77 for CLIP)
            let max_length = self.config.max_position_embeddings;
            if ids.len() > max_length {
                ids.truncate(max_length);
            } else {
                while ids.len() < max_length {
                    ids.push(self.config.pad_token_id as u32);
                }
            }

            all_ids.push(ids);
        }

        // Convert to tensor [batch_size, seq_len]
        let batch_size = all_ids.len();
        let seq_len = all_ids[0].len();
        let flat_ids: Vec<u32> = all_ids.into_iter().flatten().collect();

        let flat_ids_f32: Vec<f32> = flat_ids.iter().map(|&x| x as f32).collect();
        let input_ids = Tensor::from_vec(
            flat_ids_f32,
            Shape::from_dims(&[batch_size, seq_len]),
            self.device.cuda_device().clone(),
        )?;

        // Forward through encoder
        let output = self.forward(&input_ids, None)?;
        Ok(output.last_hidden_state)
    }

    fn create_attention_mask(&self, mask: &Tensor) -> Result<Tensor> {
        // Convert [batch, seq_len] to [batch, 1, 1, seq_len]
        let mask = mask.unsqueeze(1)?.unsqueeze(1)?;
        // Convert to attention mask values (0 -> -10000, 1 -> 0)
        let mask = mask.mul_scalar(-1.0f32)?.add_scalar(1.0f32)?;
        mask.mul_scalar(-10000.0f32)
    }

    fn get_pooled_output(&self, hidden_states: &Tensor, input_ids: &Tensor) -> Result<Tensor> {
        // Get the last valid token position for each sequence
        let batch_size = input_ids.shape().dims()[0];
        let seq_len = input_ids.shape().dims()[1];

        // Find EOS token positions (CLIP uses token ID 49407 for EOS)
        const EOS_TOKEN_ID: f32 = 49407.0;

        // Convert input_ids to vector to find EOS positions
        let input_ids_vec = input_ids.to_vec()?;
        let mut eos_positions = Vec::with_capacity(batch_size);

        for batch_idx in 0..batch_size {
            let start_idx = batch_idx * seq_len;
            let end_idx = start_idx + seq_len;
            let sequence = &input_ids_vec[start_idx..end_idx];

            // Find EOS token position, default to last position if not found
            let eos_pos =
                sequence.iter().position(|&token| token == EOS_TOKEN_ID).unwrap_or(seq_len - 1);

            eos_positions.push(eos_pos as f32);
        }

        // Create tensor with EOS positions
        let positions_tensor = Tensor::from_vec(
            eos_positions,
            Shape::from_dims(&[batch_size]),
            input_ids.device().clone(),
        )?;

        // Gather pooled output from the EOS positions
        let mut pooled_outputs = Vec::new();
        for (batch_idx, &pos) in positions_tensor.to_vec()?.iter().enumerate() {
            // Extract the hidden state at the EOS position for this batch
            let hidden_state = hidden_states
                .narrow(0, batch_idx, 1)? // Select batch
                .narrow(1, pos as usize, 1)? // Select position
                .squeeze(Some(1))?; // Remove sequence dimension
            pooled_outputs.push(hidden_state);
        }

        // Stack all pooled outputs
        let refs: Vec<&Tensor> = pooled_outputs.iter().collect();
        Tensor::cat(&refs, 0)
    }
}

/// CLIP text encoder output

/// CLIP text embeddings

impl CLIPTextEmbeddings {
    fn new(
        config: &CLIPConfig,
        device: &flame_core::device::Device,
        weights: &std::collections::HashMap<String, Tensor>,
    ) -> Result<Self> {
        let token_weight = weights.get("embeddings.token_embedding.weight").ok_or_else(|| {
            flame_core::Error::InvalidOperation("Missing token embedding weight".into())
        })?;
        let position_weight =
            weights.get("embeddings.position_embedding.weight").ok_or_else(|| {
                flame_core::Error::InvalidOperation("Missing position embedding weight".to_string())
            })?;

        // Create token embedding from weight tensor
        let token_embedding = {
            let dims = token_weight.shape().dims();
            let mut emb = Embedding::new(dims[0], dims[1], device.cuda_device().clone())?;
            emb.weight = to_bf16_tensor(token_weight)?;
            emb
        };
        // Create position embedding from weight tensor
        let position_embedding = {
            let dims = position_weight.shape().dims();
            let mut emb = Embedding::new(dims[0], dims[1], device.cuda_device().clone())?;
            emb.weight = to_bf16_tensor(position_weight)?;
            emb
        };
        Ok(Self { token_embedding, position_embedding })
    }

    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        log::debug!("CLIPTextEmbeddings forward: input_ids shape = {:?}", input_ids.shape());
        let seq_len = input_ids.shape().dims()[1];

        // Get token embeddings
        let inputs_embeds = self.token_embedding.forward(input_ids)?;
        log::debug!(
            "CLIPTextEmbeddings: token_embedding output shape = {:?}",
            inputs_embeds.shape()
        );

        // Create position ids
        let position_ids =
            flame_core::Tensor::arange(0.0, seq_len as f32, 1.0, input_ids.device().clone())?
                .unsqueeze(0)?
                .broadcast_to(input_ids.shape())?;
        log::debug!("CLIPTextEmbeddings: position_ids shape = {:?}", position_ids.shape());

        // Get position embeddings
        let position_embeds = self.position_embedding.forward(&position_ids)?;
        log::debug!("CLIPTextEmbeddings: position_embeds shape = {:?}", position_embeds.shape());

        // Add embeddings
        let result = inputs_embeds.add(&position_embeds)?;
        log::debug!("CLIPTextEmbeddings: final result shape = {:?}", result.shape());
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
            layers.push(CLIPEncoderLayer::new(config, device, weights, i)?)
        }

        Ok(Self { layers })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let mut hidden_states = hidden_states.clone();

        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, attention_mask)?;
        }

        Ok(hidden_states)
    }
}

/// CLIP encoder output

/// CLIP encoder layer

impl CLIPEncoderLayer {
    fn new(
        config: &CLIPConfig,
        device: &flame_core::device::Device,
        weights: &std::collections::HashMap<String, Tensor>,
        layer_idx: usize,
    ) -> Result<Self> {
        let prefix = format!("encoder.layers.{}", layer_idx);

        let self_attn = CLIPAttention::new(config, &device, weights, &prefix)?;
        let layer_norm1_weight =
            weights.get(&format!("{}.layer_norm1.weight", prefix)).ok_or_else(|| {
                flame_core::Error::InvalidOperation(format!(
                    "{}.layer_norm1.weight not found",
                    prefix
                ))
            })?;
        let layer_norm1_bias =
            weights.get(&format!("{}.layer_norm1.bias", prefix)).ok_or_else(|| {
                flame_core::Error::InvalidOperation(format!(
                    "{}.layer_norm1.bias not found",
                    prefix
                ))
            })?;
        let layer_norm1 = {
            let mut ln = LayerNorm::new(
                vec![config.hidden_size],
                config.layer_norm_eps,
                device.cuda_device().clone(),
            )?;
            ln.weight = Some(to_bf16_tensor(layer_norm1_weight)?);
            ln.bias = Some(to_bf16_tensor(layer_norm1_bias)?);
            ln
        };

        let mlp = CLIPMLP::new(config, &device, weights, &prefix)?;

        let layer_norm2_weight =
            weights.get(&format!("{}.layer_norm2.weight", prefix)).ok_or_else(|| {
                flame_core::Error::InvalidOperation(format!(
                    "{}.layer_norm2.weight not found",
                    prefix
                ))
            })?;
        let layer_norm2_bias =
            weights.get(&format!("{}.layer_norm2.bias", prefix)).ok_or_else(|| {
                flame_core::Error::InvalidOperation(format!(
                    "{}.layer_norm2.bias not found",
                    prefix
                ))
            })?;
        let layer_norm2 = {
            let mut ln = LayerNorm::new(
                vec![config.hidden_size],
                config.layer_norm_eps,
                device.cuda_device().clone(),
            )?;
            ln.weight = Some(to_bf16_tensor(layer_norm2_weight)?);
            ln.bias = Some(to_bf16_tensor(layer_norm2_bias)?);
            ln
        };

        Ok(Self { self_attn, layer_norm1, mlp, layer_norm2 })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        // Self attention with pre-norm
        let residual = hidden_states.clone();
        log::debug!("LayerNorm1 input dtype: {:?}", hidden_states.dtype());
        let hidden_states = self.layer_norm1.forward(hidden_states)?;
        let hidden_states = self.self_attn.forward(&hidden_states, attention_mask)?;
        let hidden_states = residual.add(&hidden_states)?;

        // MLP with pre-norm
        let residual = hidden_states.clone();
        let hidden_states = self.layer_norm2.forward(&hidden_states)?;
        let hidden_states = self.mlp.forward(&hidden_states)?;
        residual.add(&hidden_states)
    }
}

/// CLIP attention

impl CLIPAttention {
    fn new(
        config: &CLIPConfig,
        device: &flame_core::device::Device,
        weights: &std::collections::HashMap<String, Tensor>,
        prefix: &str,
    ) -> Result<Self> {
        let embed_dim = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let head_dim = embed_dim / num_heads;

        let q_weight =
            weights.get(&format!("{}.self_attn.q_proj.weight", prefix)).ok_or_else(|| {
                flame_core::Error::InvalidOperation(format!(
                    "{}.self_attn.q_proj.weight not found",
                    prefix
                ))
            })?;
        let k_weight =
            weights.get(&format!("{}.self_attn.k_proj.weight", prefix)).ok_or_else(|| {
                flame_core::Error::InvalidOperation(format!(
                    "{}.self_attn.k_proj.weight not found",
                    prefix
                ))
            })?;
        let v_weight =
            weights.get(&format!("{}.self_attn.v_proj.weight", prefix)).ok_or_else(|| {
                flame_core::Error::InvalidOperation(format!(
                    "{}.self_attn.v_proj.weight not found",
                    prefix
                ))
            })?;
        let out_weight =
            weights.get(&format!("{}.self_attn.out_proj.weight", prefix)).ok_or_else(|| {
                flame_core::Error::InvalidOperation(format!(
                    "{}.self_attn.out_proj.weight not found",
                    prefix
                ))
            })?;

        let q_bias = weights.get(&format!("{}.self_attn.q_proj.bias", prefix));
        let k_bias = weights.get(&format!("{}.self_attn.k_proj.bias", prefix));
        let v_bias = weights.get(&format!("{}.self_attn.v_proj.bias", prefix));
        let out_bias = weights.get(&format!("{}.self_attn.out_proj.bias", prefix));

        let scale = 1.0 / (head_dim as f32).sqrt();

        Ok(Self {
            k_proj: {
                let in_features = k_weight.shape().dims()[1];
                let out_features = k_weight.shape().dims()[0];
                let mut linear =
                    Linear::new(in_features, out_features, true, device.cuda_device())?;
                linear.weight = to_bf16_tensor(k_weight)?;
                linear.bias = to_bf16_optional(k_bias)?;
                linear
            },
            v_proj: {
                let in_features = v_weight.shape().dims()[1];
                let out_features = v_weight.shape().dims()[0];
                let mut linear =
                    Linear::new(in_features, out_features, true, device.cuda_device())?;
                linear.weight = to_bf16_tensor(v_weight)?;
                linear.bias = to_bf16_optional(v_bias)?;
                linear
            },
            q_proj: {
                let in_features = q_weight.shape().dims()[1];
                let out_features = q_weight.shape().dims()[0];
                let mut linear =
                    Linear::new(in_features, out_features, true, device.cuda_device())?;
                linear.weight = to_bf16_tensor(q_weight)?;
                linear.bias = to_bf16_optional(q_bias)?;
                linear
            },
            out_proj: {
                let in_features = out_weight.shape().dims()[1];
                let out_features = out_weight.shape().dims()[0];
                let mut linear =
                    Linear::new(in_features, out_features, true, device.cuda_device())?;
                linear.weight = to_bf16_tensor(out_weight)?;
                linear.bias = to_bf16_optional(out_bias)?;
                linear
            },
            num_heads,
            head_dim,
            scale,
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let batch_size = hidden_states.shape().dims()[0];
        let seq_len = hidden_states.shape().dims()[1];

        log::debug!(
            "CLIPAttention input dtype {:?}, q weight dtype {:?}",
            hidden_states.dtype(),
            self.q_proj.weight.dtype()
        );

        // Project to Q, K, V
        let q = self
            .q_proj
            .forward(hidden_states)?
            .reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])?
            .transpose_dims(1, 2)?;
        let q = if q.dtype() != DType::BF16 { q.to_dtype(DType::BF16)? } else { q };
        let k = self
            .k_proj
            .forward(hidden_states)?
            .reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])?
            .transpose_dims(1, 2)?;
        let k = if k.dtype() != DType::BF16 { k.to_dtype(DType::BF16)? } else { k };
        let v = self
            .v_proj
            .forward(hidden_states)?
            .reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])?
            .transpose_dims(1, 2)?;
        let v = if v.dtype() != DType::BF16 { v.to_dtype(DType::BF16)? } else { v };

        // Attention scores
        let k_t = k.transpose_dims(2, 3)?;
        let k_t = if k_t.dtype() != DType::BF16 { k_t.to_dtype(DType::BF16)? } else { k_t };
        log::debug!("CLIPAttention matmul lhs {:?} rhs {:?}", q.dtype(), k_t.dtype());
        
        let b = q.shape().dims()[0];
        let h = q.shape().dims()[1];
        let s = q.shape().dims()[2];
        let d = q.shape().dims()[3];
        let q_3d = q.reshape(&[b * h, s, d])?;
        let k_3d = k_t.reshape(&[b * h, d, s])?;
        let scores_3d = q_3d.bmm(&k_3d)?;
        let scores = scores_3d.reshape(&[b, h, s, s])?.mul_scalar(self.scale as f32)?;

        // Apply attention mask if provided
        let scores = if let Some(mask) = attention_mask { scores.add(mask)? } else { scores };

        // Softmax
        let attn_weights = scores.softmax(-1)?;

        // Apply attention to values
        let attn_weights_3d = attn_weights.reshape(&[b * h, s, s])?;
        let v_3d = v.reshape(&[b * h, s, d])?;
        let attn_output_3d = attn_weights_3d.bmm(&v_3d)?;
        let attn_output = attn_output_3d.reshape(&[b, h, s, d])?.transpose_dims(1, 2)?.reshape(&[
            batch_size,
            seq_len,
            self.num_heads * self.head_dim,
        ])?;

        // Output projection
        self.out_proj.forward(&attn_output)
    }
}

/// CLIP MLP

impl CLIPMLP {
    fn new(
        config: &CLIPConfig,
        device: &flame_core::device::Device,
        weights: &std::collections::HashMap<String, Tensor>,
        prefix: &str,
    ) -> Result<Self> {
        let fc1_weight = weights.get(&format!("{}.mlp.fc1.weight", prefix)).ok_or_else(|| {
            flame_core::Error::InvalidOperation(format!("{}.mlp.fc1.weight not found", prefix))
        })?;
        let fc2_weight = weights.get(&format!("{}.mlp.fc2.weight", prefix)).ok_or_else(|| {
            flame_core::Error::InvalidOperation(format!("{}.mlp.fc2.weight not found", prefix))
        })?;

        let fc1_bias = weights.get(&format!("{}.mlp.fc1.bias", prefix));
        let fc2_bias = weights.get(&format!("{}.mlp.fc2.bias", prefix));

        Ok(Self {
            fc1: {
                let in_features = fc1_weight.shape().dims()[1];
                let out_features = fc1_weight.shape().dims()[0];
                let mut linear =
                    Linear::new(in_features, out_features, true, device.cuda_device())?;
                linear.weight = to_bf16_tensor(fc1_weight)?;
                linear.bias = to_bf16_optional(fc1_bias)?;
                linear
            },
            fc2: {
                let in_features = fc2_weight.shape().dims()[1];
                let out_features = fc2_weight.shape().dims()[0];
                let mut linear =
                    Linear::new(in_features, out_features, true, device.cuda_device())?;
                linear.weight = to_bf16_tensor(fc2_weight)?;
                linear.bias = to_bf16_optional(fc2_bias)?;
                linear
            },
            activation: config.hidden_act.clone(),
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let hidden_states = self.fc1.forward(hidden_states)?;

        let hidden_states = match self.activation.as_str() {
            "gelu" => flame_core::cuda_ops_bf16::gelu_bf16(&hidden_states)?,
            "quick_gelu" => {
                // Quick GELU approximation: x * sigmoid(1.702 * x)
                let x_scaled = hidden_states.mul_scalar(1.702 as f32)?;
                let sigmoid = x_scaled.sigmoid()?;
                hidden_states.mul(&sigmoid)?
            }
            "relu" => hidden_states.relu()?,
            _ => flame_core::cuda_ops_bf16::gelu_bf16(&hidden_states)?, // Default to GELU
        };

        self.fc2.forward(&hidden_states)
    }
}

/// FLAME T5 encoder

impl T5Encoder {
    pub fn new(
        config: T5Config,
        device: flame_core::device::Device,
        weights: &crate::loaders::WeightLoader,
    ) -> Result<Self> {
        // Legacy path: must clone/alias because we only have reference
        let mut weight_map = std::collections::HashMap::new();
        for (key, tensor) in weights.weights.iter() {
            let insert_alias = |map: &mut std::collections::HashMap<String, Tensor>,
                                name: &str,
                                tensor: &Tensor| {
                map.entry(name.to_string()).or_insert_with(|| tensor.alias());
            };

            insert_alias(&mut weight_map, key, tensor);
            if let Some(stripped) = key.strip_prefix("text_model.") {
                insert_alias(&mut weight_map, stripped, tensor);
            }
            if let Some(stripped) = key.strip_prefix("encoder.") {
                insert_alias(&mut weight_map, stripped, tensor);
            }
        }
        Self::from_map(config, device, weight_map)
    }

    pub fn from_map(
        config: T5Config,
        device: flame_core::device::Device,
        mut weight_map: std::collections::HashMap<String, Tensor>,
    ) -> Result<Self> {
        // Add aliases for legacy keys if missing, but try to avoid cloning if possible.
        // Since we own the map, we can just insert aliases pointing to existing tensors.
        // Note: alias() might be deep copy, so we should avoid it if we can.
        // But we need to support multiple keys pointing to same tensor.
        // We can collect keys to process to avoid borrowing issues.
        let keys: Vec<String> = weight_map.keys().cloned().collect();
        for key in keys {
            // We need to clone the handle to alias it.
            // If alias() is deep copy, this is bad.
            // But we only alias if we need to add a missing key.
            // If the key already exists, we do nothing.
            
            let maybe_alias = |map: &mut std::collections::HashMap<String, Tensor>, src_key: &str, dst_key: &str| {
                if !map.contains_key(dst_key) {
                    if let Some(tensor) = map.get(src_key) {
                        let alias = tensor.clone(); // Use clone() which is shallow (Arc)
                        map.insert(dst_key.to_string(), alias);
                    }
                }
            };

            if let Some(stripped) = key.strip_prefix("text_model.") {
                maybe_alias(&mut weight_map, &key, stripped);
            }
            if let Some(stripped) = key.strip_prefix("encoder.") {
                maybe_alias(&mut weight_map, &key, stripped);
            }
        }

        #[cfg(debug_assertions)]
        {
            let mut sample: Vec<_> = weight_map.keys().take(5).cloned().collect();
            eprintln!("[t5_loader] sample keys: {:?}", sample);
        }

        let embeddings = T5Embeddings::new(&config, &device, &weight_map)?;
        let encoder = T5Stack::new(&config, &device, &weight_map)?;
        // T5 uses 'final_layer_norm' as the key for the final layer norm
        let final_ln_weight =
            weight_map.get("encoder.final_layer_norm.weight").ok_or_else(|| {
                flame_core::Error::InvalidOperation(
                    "encoder.final_layer_norm.weight not found".into(),
                )
            })?;
        let final_ln_bias = weight_map.get("encoder.final_layer_norm.bias");
        let final_layer_norm = {
            let mut ln = LayerNorm::new(
                vec![config.d_model],
                config.layer_norm_epsilon,
                device.cuda_device().clone(),
            )?;
            ln.weight = Some(to_bf16_tensor(final_ln_weight)?);
            ln.bias = Some(match final_ln_bias {
                Some(bias) => to_bf16_tensor(bias)?,
                None => Tensor::zeros_dtype(
                    Shape::from_dims(&[config.d_model]),
                    DType::BF16,
                    device.cuda_device().clone(),
                )?,
            });
            ln
        };

        Ok(Self { embeddings, encoder, final_layer_norm, config, device })
    }

    /// Load from pretrained weights
    pub fn from_pretrained(
        model_path: &str,
        config: T5Config,
        device: flame_core::device::Device,
    ) -> Result<Self> {
        let weights = crate::loaders::WeightLoader::from_safetensors(model_path, device.clone())?;
        Self::new(config, device, &weights)
    }

    pub fn forward(&self, input_ids: &Tensor) -> Result<T5Output> {
        // Get embeddings
        let hidden_states = self.embeddings.forward(input_ids)?;

        // Create attention mask
        let attention_mask = self.create_attention_mask(input_ids)?;

        // Run through encoder
        let encoder_outputs = self.encoder.forward(&hidden_states, Some(&attention_mask))?;

        // Apply final layer norm
        let last_hidden_state = self.final_layer_norm.forward(&encoder_outputs)?;

        Ok(T5Output { last_hidden_state })
    }

    fn create_attention_mask(&self, input_ids: &Tensor) -> Result<Tensor> {
        // Create attention mask from input_ids (pad tokens = 0)
        // Create a tensor filled with pad_token_id for comparison
        let pad_value = Tensor::full(
            input_ids.shape().clone(),
            self.config.pad_token_id as f32,
            input_ids.device().clone(),
        )?;
        // Use element-wise comparison - in FLAME, we'll use a different approach
        // Create mask where input_ids != pad_token_id
        let ones = Tensor::ones(input_ids.shape().clone(), input_ids.device().clone())?;
        let mask = ones; // TODO: Implement proper not-equal comparison

        // Convert to float and expand dimensions
        Ok(mask.unsqueeze(1)?.unsqueeze(1)?)
    }
}

/// T5 output

/// T5 embeddings

impl T5Embeddings {
    fn new(
        config: &T5Config,
        device: &flame_core::device::Device,
        weights: &std::collections::HashMap<String, Tensor>,
    ) -> Result<Self> {
        let token_weight = weights.get("shared.weight").ok_or_else(|| {
            flame_core::Error::InvalidOperation("Missing T5 token embeddings".into())
        })?;

        // Create embedding from weight tensor
        let token_embeddings = {
            let dims = token_weight.shape().dims();
            let mut emb = Embedding::new(dims[0], dims[1], device.cuda_device().clone())?;
            emb.weight = token_weight.clone();
            emb
        };

        Ok(Self { token_embeddings })
    }

    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.token_embeddings.forward(input_ids)
    }
}

/// T5 encoder stack
struct T5Stack {
    layers: Vec<T5Layer>,
}

impl T5Stack {
    fn new(
        config: &T5Config,
        device: &flame_core::device::Device,
        weights: &std::collections::HashMap<String, Tensor>,
    ) -> Result<Self> {
        let mut layers = Vec::new();

        for i in 0..config.num_layers {
            layers.push(T5Layer::new(config, &device, weights, i)?);
        }

        Ok(Self { layers })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let mut hidden_states = hidden_states.clone();

        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, attention_mask)?;
        }

        Ok(hidden_states)
    }
}

/// T5 layer
struct T5Layer {
    self_attention: T5Attention,
    layer_norm1: LayerNorm,
    feed_forward: T5FeedForward,
    layer_norm2: LayerNorm,
}

impl T5Layer {
    fn new(
        config: &T5Config,
        device: &flame_core::device::Device,
        weights: &std::collections::HashMap<String, Tensor>,
        layer_idx: usize,
    ) -> Result<Self> {
        let prefix = format!("encoder.block.{}", layer_idx);

        let ln1_weight =
            weights.get(&format!("{}.layer.0.layer_norm.weight", prefix)).ok_or_else(|| {
                flame_core::Error::InvalidOperation(format!(
                    "{}.layer.0.layer_norm.weight not found",
                    prefix
                ))
            })?;
        let ln1_bias = weights.get(&format!("{}.layer.0.layer_norm.bias", prefix));
        let layer_norm1 = {
            let mut ln = LayerNorm::new(
                vec![config.d_model],
                config.layer_norm_epsilon,
                device.cuda_device().clone(),
            )?;
            ln.weight = Some(to_bf16_tensor(ln1_weight)?);
            ln.bias = Some(match ln1_bias {
                Some(bias) => to_bf16_tensor(bias)?,
                None => Tensor::zeros_dtype(
                    Shape::from_dims(&[config.d_model]),
                    DType::BF16,
                    device.cuda_device().clone(),
                )?,
            });
            ln
        };

        let ln2_weight =
            weights.get(&format!("{}.layer.1.layer_norm.weight", prefix)).ok_or_else(|| {
                flame_core::Error::InvalidOperation(format!(
                    "{}.layer.1.layer_norm.weight not found",
                    prefix
                ))
            })?;
        let ln2_bias = weights.get(&format!("{}.layer.1.layer_norm.bias", prefix));
        let layer_norm2 = {
            let mut ln = LayerNorm::new(
                vec![config.d_model],
                config.layer_norm_epsilon,
                device.cuda_device().clone(),
            )?;
            ln.weight = Some(to_bf16_tensor(ln2_weight)?);
            ln.bias = Some(match ln2_bias {
                Some(bias) => to_bf16_tensor(bias)?,
                None => Tensor::zeros_dtype(
                    Shape::from_dims(&[config.d_model]),
                    DType::BF16,
                    device.cuda_device().clone(),
                )?,
            });
            ln
        };

        Ok(Self {
            self_attention: T5Attention::new(config, &device, weights, &prefix)?,
            layer_norm1,
            feed_forward: T5FeedForward::new(config, &device, weights, &prefix)?,
            layer_norm2,
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        // Self attention with pre-norm
        let normed = self.layer_norm1.forward(hidden_states)?;
        let attention_output = self.self_attention.forward(&normed, attention_mask)?;
        let hidden_states = hidden_states.add(&attention_output)?;

        // Feed forward with pre-norm
        let normed = self.layer_norm2.forward(&hidden_states)?;
        let ff_output = self.feed_forward.forward(&normed)?;
        hidden_states.add(&ff_output)
    }
}

/// T5 attention

impl T5Attention {
    fn new(
        config: &T5Config,
        device: &flame_core::device::Device,
        weights: &std::collections::HashMap<String, Tensor>,
        prefix: &str,
    ) -> Result<Self> {
        let d_kv = config.d_model / config.num_heads;

        let q_weight = weights
            .get(&format!("{}.layer.0.SelfAttention.q.weight", prefix))
            .ok_or_else(|| {
                flame_core::Error::InvalidOperation(format!(
                    "{}.layer.0.SelfAttention.q.weight not found",
                    prefix
                ))
            })?;
        let k_weight = weights
            .get(&format!("{}.layer.0.SelfAttention.k.weight", prefix))
            .ok_or_else(|| {
                flame_core::Error::InvalidOperation(format!(
                    "{}.layer.0.SelfAttention.k.weight not found",
                    prefix
                ))
            })?;
        let v_weight = weights
            .get(&format!("{}.layer.0.SelfAttention.v.weight", prefix))
            .ok_or_else(|| {
                flame_core::Error::InvalidOperation(format!(
                    "{}.layer.0.SelfAttention.v.weight not found",
                    prefix
                ))
            })?;
        let o_weight = weights
            .get(&format!("{}.layer.0.SelfAttention.o.weight", prefix))
            .ok_or_else(|| {
                flame_core::Error::InvalidOperation(format!(
                    "{}.layer.0.SelfAttention.o.weight not found",
                    prefix
                ))
            })?;

        // T5 attention layers typically don't have bias
        Ok(Self {
            q: {
                let in_features = q_weight.shape().dims()[1];
                let out_features = q_weight.shape().dims()[0];
                let mut linear =
                    Linear::new(in_features, out_features, false, device.cuda_device())?;
                linear.weight = to_bf16_tensor(q_weight)?;
                linear.bias = None;
                linear
            },
            k: {
                let in_features = k_weight.shape().dims()[1];
                let out_features = k_weight.shape().dims()[0];
                let mut linear =
                    Linear::new(in_features, out_features, false, device.cuda_device())?;
                linear.weight = to_bf16_tensor(k_weight)?;
                linear.bias = None;
                linear
            },
            v: {
                let in_features = v_weight.shape().dims()[1];
                let out_features = v_weight.shape().dims()[0];
                let mut linear =
                    Linear::new(in_features, out_features, false, device.cuda_device())?;
                linear.weight = to_bf16_tensor(v_weight)?;
                linear.bias = None;
                linear
            },
            o: {
                let in_features = o_weight.shape().dims()[1];
                let out_features = o_weight.shape().dims()[0];
                let mut linear =
                    Linear::new(in_features, out_features, false, device.cuda_device())?;
                linear.weight = to_bf16_tensor(o_weight)?;
                linear.bias = None;
                linear
            },
            num_heads: config.num_heads,
            d_kv,
            scale: 1.0 / (d_kv as f32).sqrt(),
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let batch_size = hidden_states.shape().dims()[0];
        let seq_len = hidden_states.shape().dims()[1];

        // Compute Q, K, V
        let q = self
            .q
            .forward(hidden_states)?
            .reshape(&[batch_size, seq_len, self.num_heads, self.d_kv])?
            .transpose_dims(1, 2)?;
        let q = if q.dtype() != DType::BF16 { q.to_dtype(DType::BF16)? } else { q };
        let k = self
            .k
            .forward(hidden_states)?
            .reshape(&[batch_size, seq_len, self.num_heads, self.d_kv])?
            .transpose_dims(1, 2)?;
        let k = if k.dtype() != DType::BF16 { k.to_dtype(DType::BF16)? } else { k };
        let v = self
            .v
            .forward(hidden_states)?
            .reshape(&[batch_size, seq_len, self.num_heads, self.d_kv])?
            .transpose_dims(1, 2)?;
        let v = if v.dtype() != DType::BF16 { v.to_dtype(DType::BF16)? } else { v };

        // Compute attention scores
        let k_t = k.transpose_dims(2, 3)?;
        let k_t = if k_t.dtype() != DType::BF16 { k_t.to_dtype(DType::BF16)? } else { k_t };
        println!("DEBUG T5Attention matmul lhs {:?} rhs {:?}", q.dtype(), k_t.dtype());
        
        let b = q.shape().dims()[0];
        let h = q.shape().dims()[1];
        let s = q.shape().dims()[2];
        let d = q.shape().dims()[3];
        let q_3d = q.reshape(&[b * h, s, d])?;
        let k_3d = k_t.reshape(&[b * h, d, s])?;
        let scores_3d = q_3d.bmm(&k_3d)?;
        let scores = scores_3d.reshape(&[b, h, s, s])?.mul_scalar(self.scale as f32)?;

        // Apply attention mask if provided
        let scores = if let Some(mask) = attention_mask {
            scores.add(&mask.mul_scalar(-1e9 as f32)?)?
        } else {
            scores
        };

        // Softmax
        let attn_weights = scores.softmax(-1)?;

        // Apply attention to values
        // Apply attention to values
        let attn_weights_3d = attn_weights.reshape(&[b * h, s, s])?;
        let v_3d = v.reshape(&[b * h, s, d])?;
        let attn_output_3d = attn_weights_3d.bmm(&v_3d)?;
        let attn_output = attn_output_3d.reshape(&[b, h, s, d])?.transpose_dims(1, 2)?.reshape(&[
            batch_size,
            seq_len,
            self.num_heads * self.d_kv,
        ])?;

        // Output projection
        self.o.forward(&attn_output)
    }
}

/// T5 feed forward

impl T5FeedForward {
    fn new(
        config: &T5Config,
        device: &flame_core::device::Device,
        weights: &std::collections::HashMap<String, Tensor>,
        prefix: &str,
    ) -> Result<Self> {
        let wi_key = format!("{}.layer.1.DenseReluDense.wi.weight", prefix);
        let wi_weight = weights.get(&wi_key);
        let wi0_key = format!("{}.layer.1.DenseReluDense.wi_0.weight", prefix);
        let wi1_key = format!("{}.layer.1.DenseReluDense.wi_1.weight", prefix);

        let wo_weight = weights
            .get(&format!("{}.layer.1.DenseReluDense.wo.weight", prefix))
            .ok_or_else(|| {
                flame_core::Error::InvalidOperation(format!(
                    "{}.layer.1.DenseReluDense.wo.weight not found",
                    prefix
                ))
            })?;

        // T5 feed forward layers typically don't have bias
        let (wi_linear, wi_gate_linear) = if let Some(wi_weight) = wi_weight {
            let in_features = wi_weight.shape().dims()[1];
            let out_features = wi_weight.shape().dims()[0];
            let mut linear = Linear::new(in_features, out_features, false, device.cuda_device())?;
            linear.weight = to_bf16_tensor(wi_weight)?;
            linear.bias = None;
            (linear, None)
        } else {
            let wi1 = weights.get(&wi1_key).ok_or_else(|| {
                flame_core::Error::InvalidOperation(format!(
                    "DenseReluDense gating requires {}, but it was not found",
                    wi1_key
                ))
            })?;
            let wi0 = weights.get(&wi0_key).ok_or_else(|| {
                flame_core::Error::InvalidOperation(format!(
                    "DenseReluDense gating requires {}, but it was not found",
                    wi0_key
                ))
            })?;
            let mut linear_main = Linear::new(
                wi1.shape().dims()[1],
                wi1.shape().dims()[0],
                false,
                device.cuda_device(),
            )?;
            linear_main.weight = to_bf16_tensor(wi1)?;
            linear_main.bias = None;

            let mut linear_gate = Linear::new(
                wi0.shape().dims()[1],
                wi0.shape().dims()[0],
                false,
                device.cuda_device(),
            )?;
            linear_gate.weight = to_bf16_tensor(wi0)?;
            linear_gate.bias = None;
            (linear_main, Some(linear_gate))
        };

        let mut wo_linear = Linear::new(
            wo_weight.shape().dims()[1],
            wo_weight.shape().dims()[0],
            false,
            device.cuda_device(),
        )?;
        wo_linear.weight = to_bf16_tensor(wo_weight)?;
        wo_linear.bias = None;

        Ok(Self {
            wi: wi_linear,
            wi_gate: wi_gate_linear,
            wo: wo_linear,
            activation: "relu".into(),
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let activated = if let Some(ref gate) = self.wi_gate {
            let gate_out = gate.forward(hidden_states)?.relu()?;
            let proj = self.wi.forward(hidden_states)?;
            gate_out.mul(&proj)?
        } else {
            let proj = self.wi.forward(hidden_states)?;
            match self.activation.as_str() {
                "relu" => proj.relu()?,
                _ => proj.relu()?,
            }
        };
        self.wo.forward(&activated)
    }
}

// End of file
