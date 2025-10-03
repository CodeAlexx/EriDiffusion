// Parameter is just Tensor in FLAME
use crate::ops::LayerNorm;
use crate::ops::Linear;
/// CLIP text encoder implementation using direct FLAME
use flame_core::device::Device;
use flame_core::{CudaDevice, Parameter, Result, Tensor};
// Module trait is in tensor module
use crate::flame_training::FLAMEModel;
use crate::loaders::WeightLoader;
use crate::models::attention::TensorAttentionExt;
use flame_core::embedding::Embedding;
use std::collections::HashMap;

/// CLIP Text Model
pub struct CLIPTextModel {
    embeddings: CLIPTextEmbeddings,
    encoder: CLIPEncoder,
    final_layer_norm: LayerNorm,
}

impl CLIPTextModel {
    pub fn load(weights: &WeightLoader) -> Result<Self> {
        let embeddings = CLIPTextEmbeddings::load(weights, "text_model.embeddings")?;
        let encoder = CLIPEncoder::load(weights, "text_model.encoder")?;

        let final_layer_norm =
            LayerNorm::new(vec![768], 1e-5, weights.device().cuda_device().clone())?;

        Ok(Self { embeddings, encoder, final_layer_norm })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<CLIPTextModelOutput> {
        let hidden_states = self.embeddings.forward(input_ids)?;
        let encoder_outputs = self.encoder.forward(&hidden_states, attention_mask)?;
        let last_hidden_state = self.final_layer_norm.forward(&encoder_outputs)?;

        // Get pooled output (EOS token)
        let batch_size = input_ids.shape().dims()[0];
        let sequence_length = input_ids.shape().dims()[1];

        // Find EOS token positions
        let eos_token_id = 49407;
        let eos_token_tensor = input_ids.mul_scalar(0.0)?.add_scalar(eos_token_id as f32)?; // Create filled tensor
        let eos_mask = input_ids.eq(&eos_token_tensor)?;

        let mut pooled_output = Vec::new();
        for b in 0..batch_size {
            // Find last EOS token position
            let batch_mask = eos_mask.slice(&[(b, b + 1), (0, sequence_length)])?;
            // Get the hidden state at EOS position
            // Simplified - would need proper indexing
            let pooled = last_hidden_state.slice(&[
                (b, b + 1),
                (sequence_length - 1, sequence_length),
                (0, 768),
            ])?;
            pooled_output.push(pooled);
        }

        let pooled_refs: Vec<&Tensor> = pooled_output.iter().collect();
        let pooled_output = Tensor::cat(&pooled_refs, 0)?;

        Ok(CLIPTextModelOutput { last_hidden_state, pooled_output })
    }
}

impl crate::flame_training::FLAMEModel for CLIPTextModel {
    fn parameters(&self) -> Vec<&Parameter> {
        let mut params = Vec::new();
        // Collect from embeddings, encoder, and final layer norm
        params
    }

    fn named_parameters(&self) -> std::collections::HashMap<String, &Parameter> {
        let mut params = HashMap::new();
        // Collect named parameters
        params
    }
}

/// CLIP Text Embeddings
struct CLIPTextEmbeddings {
    token_embedding: Embedding,
    position_embedding: Embedding,
}

impl CLIPTextEmbeddings {
    fn load(weights: &WeightLoader, prefix: &str) -> Result<Self> {
        let prefixed_weights = weights.pp(prefix);

        let token_embedding =
            Embedding::new(49408, 768, prefixed_weights.device().cuda_device().clone())?;
        let position_embedding =
            Embedding::new(77, 768, prefixed_weights.device().cuda_device().clone())?;

        Ok(Self { token_embedding, position_embedding })
    }

    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let seq_length = input_ids.shape().dims()[1];

        // Token embeddings
        let inputs_embeds = self.token_embedding.forward(input_ids)?;

        // Position embeddings
        let position_ids =
            flame_core::Tensor::arange(0.0, seq_length as f32, 1.0, input_ids.device().clone())?;
        let position_embeds = self.position_embedding.forward(&position_ids)?;

        // Combine
        inputs_embeds.add(&position_embeds)
    }
}

/// CLIP Encoder
struct CLIPEncoder {
    layers: Vec<CLIPEncoderLayer>,
}

impl CLIPEncoder {
    fn load(weights: &WeightLoader, prefix: &str) -> Result<Self> {
        let prefixed_weights = weights.pp(prefix);
        let mut layers = Vec::new();

        // CLIP-L has 12 layers
        for i in 0..12 {
            let layer = CLIPEncoderLayer::load(weights, &format!("layers.{}", i))?;
            layers.push(layer);
        }

        Ok(Self { layers })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let mut h = hidden_states.clone();

        for layer in &self.layers {
            h = layer.forward(&h, attention_mask)?;
        }

        Ok(h)
    }
}

/// CLIP Encoder Layer
struct CLIPEncoderLayer {
    self_attn: CLIPAttention,
    layer_norm1: LayerNorm,
    mlp: CLIPMLP,
    layer_norm2: LayerNorm,
}

impl CLIPEncoderLayer {
    fn load(weights: &WeightLoader, prefix: &str) -> Result<Self> {
        let prefixed_weights = weights.pp(prefix);

        let self_attn = CLIPAttention::load(weights, "self_attn")?;
        let layer_norm1 =
            LayerNorm::new(vec![768], 1e-5, prefixed_weights.device().cuda_device().clone())?;

        let mlp = CLIPMLP::load(weights, "mlp")?;
        let layer_norm2 =
            LayerNorm::new(vec![768], 1e-5, prefixed_weights.device().cuda_device().clone())?;

        Ok(Self { self_attn, layer_norm1, mlp, layer_norm2 })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        // Self attention with residual
        let residual = hidden_states.clone();
        let normed = self.layer_norm1.forward(hidden_states)?;
        let attended = self.self_attn.forward(&normed, attention_mask)?;
        let hidden_states = residual.add(&attended)?;

        // MLP with residual
        let residual = hidden_states.clone();
        let normed = self.layer_norm2.forward(&hidden_states)?;
        let mlp_out = self.mlp.forward(&normed)?;
        let hidden_states = residual.add(&mlp_out)?;

        Ok(hidden_states)
    }
}

// Additional components...
struct CLIPAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
}

impl CLIPAttention {
    fn load(weights: &WeightLoader, prefix: &str) -> Result<Self> {
        let prefixed_weights = weights.pp(prefix);

        let device = prefixed_weights.device().cuda_device();
        let q_proj = Linear::new(768, 768, true, &device)?;
        let k_proj = Linear::new(768, 768, true, &device)?;
        let v_proj = Linear::new(768, 768, true, &device)?;
        let out_proj = Linear::new(768, 768, true, &device)?;

        Ok(Self { q_proj, k_proj, v_proj, out_proj, num_heads: 12 })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let dims = hidden_states.shape().dims();
        let batch_size = dims[0];
        let seq_len = dims[1];
        let embed_dim = dims[2];
        let head_dim = embed_dim / self.num_heads;

        // Project to Q, K, V
        let q = self
            .q_proj
            .forward(hidden_states)?
            .reshape(&[batch_size, seq_len, self.num_heads, head_dim])?
            .transpose_dims(1, 2)?;

        let k = self
            .k_proj
            .forward(hidden_states)?
            .reshape(&[batch_size, seq_len, self.num_heads, head_dim])?
            .transpose_dims(1, 2)?;

        let v = self
            .v_proj
            .forward(hidden_states)?
            .reshape(&[batch_size, seq_len, self.num_heads, head_dim])?
            .transpose_dims(1, 2)?;

        // Attention
        let k_dims = k.shape().dims();
        let scores = q
            .matmul(&k.transpose_dims(k_dims.len() - 2, k_dims.len() - 1)?)?
            .mul_scalar(1.0 / (head_dim as f32).sqrt())?;

        // Apply mask if provided
        let scores = if let Some(mask) = attention_mask {
            scores.add(&mask.mul_scalar(-10000.0 as f32)?)?
        } else {
            scores
        };

        let attn_weights = scores.softmax(-1)?;
        let attn_output = attn_weights.matmul(&v)?;

        // Reshape back
        let attn_output =
            attn_output.transpose_dims(1, 2)?.reshape(&[batch_size, seq_len, embed_dim])?;

        // Output projection
        self.out_proj.forward(&attn_output)
    }
}

struct CLIPMLP {
    fc1: Linear,
    fc2: Linear,
}

impl CLIPMLP {
    fn load(weights: &WeightLoader, prefix: &str) -> Result<Self> {
        let prefixed_weights = weights.pp(prefix);

        let device = prefixed_weights.device().cuda_device();
        let fc1 = Linear::new(768, 3072, true, &device)?;
        let fc2 = Linear::new(3072, 768, true, &device)?;

        Ok(Self { fc1, fc2 })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let hidden_states = self.fc1.forward(hidden_states)?;
        let hidden_states = hidden_states.gelu()?; // CLIP uses GELU activation
        self.fc2.forward(&hidden_states)
    }
}

pub struct CLIPTextModelOutput {
    pub last_hidden_state: Tensor,
    pub pooled_output: Tensor,
}
