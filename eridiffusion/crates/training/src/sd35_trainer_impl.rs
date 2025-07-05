//! SD 3.5 trainer implementation

use eridiffusion_core::{Device, Result, Error, ModelInputs};
use eridiffusion_models::{SD3Model, TextEncoder, VAE, ModelOutput, TextEncoderOutput};
use candle_core::{Tensor, DType, D};
use std::collections::HashMap;

/// SD 3.5 model variants
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SD35ModelVariant {
    Medium,     // 2.5B params
    Large,      // 8B params  
    LargeTurbo, // 8B params, distilled
}

/// SD 3.5 training configuration
#[derive(Debug, Clone)]
pub struct SD35TrainingConfig {
    pub model_variant: SD35ModelVariant,
    pub resolution: usize,
    pub num_train_timesteps: usize,
    pub flow_matching: bool,
    pub flow_shift: f32,
    pub t5_max_length: usize,
    pub mixed_precision: bool,
    pub gradient_checkpointing: bool,
}

impl Default for SD35TrainingConfig {
    fn default() -> Self {
        Self {
            model_variant: SD35ModelVariant::Large,
            resolution: 1024,
            num_train_timesteps: 1000,
            flow_matching: true,
            flow_shift: 3.0,
            t5_max_length: 256,
            mixed_precision: true,
            gradient_checkpointing: true,
        }
    }
}

/// SD 3.5 trainer utilities
pub struct SD35Trainer;

impl SD35Trainer {
    /// Encode text with triple encoders (CLIP-L + CLIP-G + T5-XXL)
    pub fn encode_text_triple(
        prompts: &[String],
        clip_l: &dyn TextEncoder,
        clip_g: &dyn TextEncoder,
        t5: &dyn TextEncoder,
        t5_max_length: usize,
    ) -> Result<TextEncoderOutput> {
        let batch_size = prompts.len();
        
        // Encode with CLIP-L
        let (clip_l_hidden, clip_l_pooled) = clip_l.encode(prompts)?;
        
        // Encode with CLIP-G
        let (clip_g_hidden, clip_g_pooled) = clip_g.encode(prompts)?;
        
        // Encode with T5
        let (t5_hidden, _) = t5.encode(prompts)?;
        
        // Pad CLIP embeddings to 2048 each
        let clip_l_padded = Self::pad_embeddings(&clip_l_hidden, 2048)?;
        let clip_g_padded = Self::pad_embeddings(&clip_g_hidden, 2048)?;
        
        // Truncate T5 if needed
        let t5_truncated = if t5_hidden.dim(1)? > t5_max_length {
            t5_hidden.narrow(1, 0, t5_max_length)?
        } else {
            t5_hidden.clone()
        };
        
        // Concatenate all embeddings [CLIP-L, CLIP-G, T5]
        let encoder_hidden_states = Tensor::cat(&[
            &clip_l_padded,
            &clip_g_padded,
            &t5_truncated,
        ], 2)?; // Concatenate along hidden dimension
        
        // Concatenate pooled outputs from CLIP models
        let pooled_output = if let (Some(l_pooled), Some(g_pooled)) = (clip_l_pooled, clip_g_pooled) {
            Some(Tensor::cat(&[&l_pooled, &g_pooled], 1)?)
        } else {
            None
        };
        
        Ok(TextEncoderOutput {
            encoder_hidden_states,
            pooled_output,
            attention_mask: None,
        })
    }
    
    /// Pad embeddings to target dimension
    fn pad_embeddings(embeddings: &Tensor, target_dim: usize) -> Result<Tensor> {
        let current_dim = embeddings.dim(D::Minus1)?;
        
        if current_dim >= target_dim {
            // Truncate if larger
            embeddings.narrow(D::Minus1, 0, target_dim)
                .map_err(|e| Error::Training(e.to_string()))
        } else {
            // Pad with zeros
            let batch_size = embeddings.dim(0)?;
            let seq_len = embeddings.dim(1)?;
            let padding = target_dim - current_dim;
            
            let zeros = Tensor::zeros(
                (batch_size, seq_len, padding),
                embeddings.dtype(),
                embeddings.device(),
            )?;
            
            Tensor::cat(&[embeddings, &zeros], D::Minus1)
                .map_err(|e| Error::Training(e.to_string()))
        }
    }
    
    /// Prepare inputs for SD 3.5 training
    pub fn prepare_training_inputs(
        latents: &Tensor,
        timesteps: &Tensor,
        encoder_hidden_states: &Tensor,
        pooled_projections: &Tensor,
    ) -> Result<ModelInputs> {
        let mut additional = HashMap::new();
        
        // SD 3.5 uses pooled projections
        additional.insert("pooled_projections".to_string(), pooled_projections.clone());
        
        Ok(ModelInputs {
            latents: latents.clone(),
            timestep: timesteps.clone(),
            encoder_hidden_states: Some(encoder_hidden_states.clone()),
            pooled_projections: Some(pooled_projections.clone()),
            attention_mask: None,
            guidance_scale: None,
            additional,
        })
    }
    
    /// Compute flow matching loss with SNR weighting
    pub fn compute_flow_matching_loss(
        model_output: &Tensor,
        target: &Tensor,
        timesteps: &Tensor,
        snr_gamma: f32,
    ) -> Result<Tensor> {
        // Basic MSE loss
        let mse_loss = (model_output - target)?.sqr()?.mean_all()?;
        
        if snr_gamma > 0.0 {
            // Apply SNR weighting
            let one = Tensor::new(1.0f32, timesteps.device())?;
            let snr = timesteps.broadcast_div(&one.broadcast_sub(timesteps)?)?;
            let weight = snr.broadcast_div(&one.broadcast_add(&snr)?)?.sqrt()?;
            let weighted_loss = mse_loss.broadcast_mul(&weight.mean_all()?)?;
            Ok(weighted_loss)
        } else {
            Ok(mse_loss)
        }
    }
    
    /// Get target modules for SD 3.5 LoRA
    pub fn get_sd35_target_modules(include_text_encoder: bool) -> Vec<String> {
        let mut modules = vec![];
        
        // MMDiT blocks
        for i in 0..38 {
            // Joint attention blocks
            modules.extend(vec![
                format!("joint_blocks.{}.x_block.attn.qkv", i),
                format!("joint_blocks.{}.x_block.attn.proj", i),
                format!("joint_blocks.{}.x_block.mlp.fc1", i),
                format!("joint_blocks.{}.x_block.mlp.fc2", i),
                format!("joint_blocks.{}.context_block.attn.qkv", i),
                format!("joint_blocks.{}.context_block.attn.proj", i),
                format!("joint_blocks.{}.context_block.mlp.fc1", i),
                format!("joint_blocks.{}.context_block.mlp.fc2", i),
            ]);
        }
        
        // Final blocks
        modules.extend(vec![
            "x_embedder.proj".to_string(),
            "context_embedder.proj".to_string(),
            "final_layer.linear".to_string(),
        ]);
        
        // Text encoder modules if requested
        if include_text_encoder {
            // CLIP-L
            for i in 0..24 {
                modules.extend(vec![
                    format!("text_model.encoder.layers.{}.self_attn.q_proj", i),
                    format!("text_model.encoder.layers.{}.self_attn.k_proj", i),
                    format!("text_model.encoder.layers.{}.self_attn.v_proj", i),
                    format!("text_model.encoder.layers.{}.self_attn.out_proj", i),
                ]);
            }
            
            // CLIP-G
            for i in 0..32 {
                modules.extend(vec![
                    format!("text_model_g.encoder.layers.{}.self_attn.q_proj", i),
                    format!("text_model_g.encoder.layers.{}.self_attn.k_proj", i),
                    format!("text_model_g.encoder.layers.{}.self_attn.v_proj", i),
                    format!("text_model_g.encoder.layers.{}.self_attn.out_proj", i),
                ]);
            }
            
            // T5 encoder layers
            for i in 0..24 {
                modules.extend(vec![
                    format!("t5.encoder.block.{}.layer.0.SelfAttention.q", i),
                    format!("t5.encoder.block.{}.layer.0.SelfAttention.k", i),
                    format!("t5.encoder.block.{}.layer.0.SelfAttention.v", i),
                    format!("t5.encoder.block.{}.layer.0.SelfAttention.o", i),
                ]);
            }
        }
        
        modules
    }
    
    /// Sample training timesteps for flow matching
    pub fn sample_timesteps(
        batch_size: usize,
        num_train_timesteps: usize,
        device: &Device,
        flow_shift: f32,
    ) -> Result<Tensor> {
        if flow_shift > 1.0 {
            // Logit-normal sampling with shift
            let device_candle = device.to_candle()?;
            let u = Tensor::rand(0.0, 1.0, &[batch_size], &device_candle)?;
            let shift = flow_shift;
            
            // Apply logit-normal transformation
            let two = Tensor::new(2.0f32, u.device())?;
            let one = Tensor::new(1.0f32, u.device())?;
            let half = Tensor::new(0.5f32, u.device())?;
            let shift_tensor = Tensor::new(shift, u.device())?;
            
            let t = u.broadcast_mul(&two)?
                .broadcast_sub(&one)?
                .broadcast_mul(&shift_tensor)?
                .tanh()?
                .broadcast_mul(&half)?
                .broadcast_add(&half)?;
            Ok(t)
        } else {
            // Uniform sampling
            let device_candle = device.to_candle()?;
            Tensor::rand(0.0, 1.0, &[batch_size], &device_candle)
                .map_err(|e| Error::Training(e.to_string()))
        }
    }
    
    /// Create flow matching targets
    pub fn create_flow_targets(
        original_latents: &Tensor,
        noise: &Tensor,
        timesteps: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let batch_size = timesteps.dims()[0];
        let t = timesteps.reshape((batch_size, 1, 1, 1))?;
        
        // Interpolate between data and noise
        let one = Tensor::new(1.0f32, t.device())?;
        let one_minus_t = one.broadcast_sub(&t)?;
        let noisy_latents = original_latents.broadcast_mul(&one_minus_t)?.broadcast_add(&noise.broadcast_mul(&t)?)?;
        
        // Velocity target for flow matching
        let eps = Tensor::new(&[1e-5f32], timesteps.device())?;
        let t_safe = t.maximum(&eps)?;
        let velocity = noise.broadcast_sub(original_latents)?.broadcast_div(&t_safe)?;
        
        Ok((noisy_latents, velocity))
    }
}