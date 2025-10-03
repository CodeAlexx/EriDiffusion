//! Fixed Flux transformer blocks with proper AdaLN implementation
//!
//! This provides the corrected implementation of Flux blocks with proper
//! Adaptive Layer Normalization (AdaLN) as used in the Flux architecture.

use crate::ops::qk_norm::{apply_qk_norm, scaled_dot_product_attention, split_qkv};
use crate::ops::streaming_rms_norm::{apply_double_stream_rms_norm, apply_single_stream_rms_norm};
use flame_core::{DType, Device, Result, Tensor};
use std::collections::HashMap;

/// Layer normalization for AdaLN
fn layer_norm(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>, eps: f64) -> Result<Tensor> {
    // Compute mean and variance over the last dimension
    let shape = x.shape().dims();
    let last_dim = shape[shape.len() - 1] as f32;

    // Mean
    let mean = x.mean_dims(&[shape.len() - 1], true)?;

    // Variance
    let diff = x.sub(&mean)?;
    let var = diff.square()?.mean_dims(&[shape.len() - 1], true)?;

    // Normalize
    let std = var.add_scalar(eps as f32)?.sqrt()?;
    let norm = diff.div(&std)?;

    // Apply weight and bias
    let mut output = norm.mul(weight)?;
    if let Some(b) = bias {
        output = output.add(b)?;
    }

    Ok(output)
}

/// Fixed Double stream block with proper AdaLN
pub struct DoubleStreamBlockFixed {
    hidden_size: usize,
    num_heads: usize,
    mlp_ratio: f32,
    device: Device,
    weights: HashMap<String, Tensor>,
}

/// Fixed Single stream block with proper AdaLN
pub struct SingleStreamBlockFixed {
    hidden_size: usize,
    num_heads: usize,
    mlp_ratio: f32,
    device: Device,
    weights: HashMap<String, Tensor>,
}

impl DoubleStreamBlockFixed {
    /// Create from loaded weights
    pub fn from_weights(
        weights: &HashMap<String, Tensor>,
        hidden_size: usize,
        num_heads: usize,
        mlp_ratio: f32,
        device: Device,
    ) -> Result<Self> {
        // Clone all weights (Tensor::clone returns Result)
        let mut cloned_weights = HashMap::new();
        for (k, v) in weights {
            cloned_weights.insert(k.clone(), v.clone());
        }

        Ok(Self { hidden_size, num_heads, mlp_ratio, device, weights: cloned_weights })
    }

    /// Apply attention with QK-Norm
    fn apply_attention(
        &self,
        input: &Tensor,
        qkv_weight: &Tensor,
        qkv_bias: Option<&Tensor>,
        proj_weight: &Tensor,
        proj_bias: Option<&Tensor>,
        query_norm_scale: Option<&Tensor>,
        key_norm_scale: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Project to QKV
        let input_shape = input.shape().dims();
        let batch_size = input_shape[0];
        let seq_len = input_shape[1];
        let hidden_size = input_shape[2];

        // Flatten for matmul
        let input_flat = input.reshape(&[batch_size * seq_len, hidden_size])?;
        let qkv_weight_t = qkv_weight.transpose()?;
        let mut qkv = input_flat.matmul(&qkv_weight_t)?;
        if let Some(bias) = qkv_bias {
            qkv = qkv.add(bias)?;
        }

        // Reshape back
        let qkv_out_features = qkv_weight.shape().dims()[0];
        let qkv = qkv.reshape(&[batch_size, seq_len, qkv_out_features])?;

        // Split QKV
        let head_dim = self.hidden_size / self.num_heads;
        let (q, k, v) = split_qkv(&qkv, self.num_heads, head_dim)?;

        // Apply QK-Norm
        let (q_normed, k_normed) = apply_qk_norm(&q, &k, query_norm_scale, key_norm_scale, 1e-6)?;

        // Compute attention
        let scale = 1.0 / (head_dim as f32).sqrt();
        let attn_output = scaled_dot_product_attention(&q_normed, &k_normed, &v, scale)?;

        // Project output
        let attn_shape = attn_output.shape().dims();
        let attn_flat = attn_output.reshape(&[attn_shape[0] * attn_shape[1], attn_shape[2]])?;
        let proj_weight_t = proj_weight.transpose()?;
        let mut output = attn_flat.matmul(&proj_weight_t)?;
        if let Some(bias) = proj_bias {
            output = output.add(bias)?;
        }

        // Reshape back
        let output = output.reshape(&[batch_size, seq_len, self.hidden_size])?;

        Ok(output)
    }

    /// Forward pass with proper AdaLN
    pub fn forward(
        &self,
        img: &Tensor,
        txt: &Tensor,
        modulation: &Tensor, // [batch_size, modulation_dim]
        norm_weights: &HashMap<String, &Tensor>,
        guidance: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        // Debug input shapes
        println!(
            "🔍 DoubleStreamBlockFixed forward - img: {:?}, txt: {:?}, modulation: {:?}",
            img.shape().dims(),
            txt.shape().dims(),
            modulation.shape().dims()
        );
        // Extract layer norm weights
        let img_norm1_weight = norm_weights.get("img_norm1.weight").ok_or(
            flame_core::Error::InvalidOperation("Missing img_norm1.weight".into()),
        )?;
        let img_norm1_bias = norm_weights.get("img_norm1.bias").copied();

        let txt_norm1_weight = norm_weights.get("txt_norm1.weight").ok_or(
            flame_core::Error::InvalidOperation("Missing txt_norm1.weight".into()),
        )?;
        let txt_norm1_bias = norm_weights.get("txt_norm1.bias").copied();

        let img_norm2_weight = norm_weights.get("img_norm2.weight").ok_or(
            flame_core::Error::InvalidOperation("Missing img_norm2.weight".into()),
        )?;
        let img_norm2_bias = norm_weights.get("img_norm2.bias").copied();

        let txt_norm2_weight = norm_weights.get("txt_norm2.weight").ok_or(
            flame_core::Error::InvalidOperation("Missing txt_norm2.weight".into()),
        )?;
        let txt_norm2_bias = norm_weights.get("txt_norm2.bias").copied();

        // Step 1: Project modulation to get scale/shift parameters
        let img_mod_weight = self.weights.get("img_mod.lin.weight").ok_or(
            flame_core::Error::InvalidOperation("Missing img_mod.lin.weight".into()),
        )?;
        let img_mod_bias = self.weights.get("img_mod.lin.bias").ok_or(
            flame_core::Error::InvalidOperation("Missing img_mod.lin.bias".into()),
        )?;

        let txt_mod_weight = self.weights.get("txt_mod.lin.weight").ok_or(
            flame_core::Error::InvalidOperation("Missing txt_mod.lin.weight".into()),
        )?;
        let txt_mod_bias = self.weights.get("txt_mod.lin.bias").ok_or(
            flame_core::Error::InvalidOperation("Missing txt_mod.lin.bias".into()),
        )?;

        // Project modulation: [batch, mod_dim] -> [batch, 6 * hidden_size]
        let img_mod_params = modulation.matmul(&img_mod_weight.transpose()?)?.add(&img_mod_bias)?;
        let txt_mod_params = modulation.matmul(&txt_mod_weight.transpose()?)?.add(&txt_mod_bias)?;

        // Split modulation parameters into (shift, scale) for each sub-block
        // Expected shape: [batch, 6 * hidden_size] for shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
        let chunk_size = self.hidden_size;
        let img_params = img_mod_params.chunk(6, 1)?;
        let txt_params = txt_mod_params.chunk(6, 1)?;

        // Extract parameters
        let img_shift_msa = &img_params[0];
        let img_scale_msa = &img_params[1];
        let img_gate_msa = &img_params[2];
        let img_shift_mlp = &img_params[3];
        let img_scale_mlp = &img_params[4];
        let img_gate_mlp = &img_params[5];

        let txt_shift_msa = &txt_params[0];
        let txt_scale_msa = &txt_params[1];
        let txt_gate_msa = &txt_params[2];
        let txt_shift_mlp = &txt_params[3];
        let txt_scale_mlp = &txt_params[4];
        let txt_gate_mlp = &txt_params[5];

        // Step 2: Apply attention with AdaLN
        // Normalize inputs
        let img_norm1 = layer_norm(img, img_norm1_weight, img_norm1_bias, 1e-6)?;
        let txt_norm1 = layer_norm(txt, txt_norm1_weight, txt_norm1_bias, 1e-6)?;

        // Apply AdaLN modulation: output = norm * (1 + scale) + shift
        // SAFETY: Clamp scale values to prevent explosion
        let img_scale_clamped = img_scale_msa.clamp(-2.0, 2.0)?;
        let txt_scale_clamped = txt_scale_msa.clamp(-2.0, 2.0)?;

        // Need to broadcast modulation from [batch, hidden] to [batch, seq_len, hidden]
        // For batch_size=1, we unsqueeze and expand
        let img_seq_len = img_norm1.shape().dims()[1];
        let txt_seq_len = txt_norm1.shape().dims()[1];

        // Unsqueeze to add sequence dimension: [batch, hidden] -> [batch, 1, hidden]
        let img_scale_unsqueezed = img_scale_clamped.unsqueeze(1)?;
        let img_shift_unsqueezed = img_shift_msa.unsqueeze(1)?;
        let txt_scale_unsqueezed = txt_scale_clamped.unsqueeze(1)?;
        let txt_shift_unsqueezed = txt_shift_msa.unsqueeze(1)?;

        // Expand along sequence dimension
        let img_scale_expanded =
            img_scale_unsqueezed.expand(&[1, img_seq_len, self.hidden_size])?;
        let img_shift_expanded =
            img_shift_unsqueezed.expand(&[1, img_seq_len, self.hidden_size])?;
        let txt_scale_expanded =
            txt_scale_unsqueezed.expand(&[1, txt_seq_len, self.hidden_size])?;
        let txt_shift_expanded =
            txt_shift_unsqueezed.expand(&[1, txt_seq_len, self.hidden_size])?;

        let one = Tensor::ones_like(&img_scale_expanded)?;
        let img_mod1 = img_norm1.mul(&one.add(&img_scale_expanded)?)?.add(&img_shift_expanded)?;
        let one_txt = Tensor::ones_like(&txt_scale_expanded)?;
        let txt_mod1 =
            txt_norm1.mul(&one_txt.add(&txt_scale_expanded)?)?.add(&txt_shift_expanded)?;

        // Apply attention
        let img_attn = if let (Some(qkv_w), Some(proj_w)) =
            (self.weights.get("img_attn.qkv.weight"), self.weights.get("img_attn.proj.weight"))
        {
            let attn_out = self.apply_attention(
                &img_mod1,
                qkv_w,
                self.weights.get("img_attn.qkv.bias"),
                proj_w,
                self.weights.get("img_attn.proj.bias"),
                norm_weights.get("img_attn.norm.query_norm.scale").copied(),
                norm_weights.get("img_attn.norm.key_norm.scale").copied(),
            )?;
            // Apply gate with broadcasting
            let img_gate_msa_unsqueezed = img_gate_msa.unsqueeze(1)?;
            let img_gate_msa_expanded =
                img_gate_msa_unsqueezed.expand(&[1, img_seq_len, self.hidden_size])?;
            attn_out.mul(&img_gate_msa_expanded)?
        } else {
            return Err(flame_core::Error::InvalidOperation(
                "Missing attention weights".to_string(),
            ));
        };

        let txt_attn = if let (Some(qkv_w), Some(proj_w)) =
            (self.weights.get("txt_attn.qkv.weight"), self.weights.get("txt_attn.proj.weight"))
        {
            let attn_out = self.apply_attention(
                &txt_mod1,
                qkv_w,
                self.weights.get("txt_attn.qkv.bias"),
                proj_w,
                self.weights.get("txt_attn.proj.bias"),
                norm_weights.get("txt_attn.norm.query_norm.scale").copied(),
                norm_weights.get("txt_attn.norm.key_norm.scale").copied(),
            )?;
            // Apply gate with broadcasting
            let txt_gate_msa_unsqueezed = txt_gate_msa.unsqueeze(1)?;
            let txt_gate_msa_expanded =
                txt_gate_msa_unsqueezed.expand(&[1, txt_seq_len, self.hidden_size])?;
            attn_out.mul(&txt_gate_msa_expanded)?
        } else {
            return Err(flame_core::Error::InvalidOperation(
                "Missing attention weights".to_string(),
            ));
        };

        // Add residual
        let img_post_attn = img.add(&img_attn)?;
        let txt_post_attn = txt.add(&txt_attn)?;

        // Step 3: Apply MLP with AdaLN
        // Normalize
        let img_norm2 = layer_norm(&img_post_attn, img_norm2_weight, img_norm2_bias, 1e-6)?;
        let txt_norm2 = layer_norm(&txt_post_attn, txt_norm2_weight, txt_norm2_bias, 1e-6)?;

        // Apply AdaLN modulation with safety clamping
        let img_scale_mlp_clamped = img_scale_mlp.clamp(-2.0, 2.0)?;
        let txt_scale_mlp_clamped = txt_scale_mlp.clamp(-2.0, 2.0)?;

        // Need to broadcast MLP modulation too
        let img_scale_mlp_unsqueezed = img_scale_mlp_clamped.unsqueeze(1)?;
        let img_shift_mlp_unsqueezed = img_shift_mlp.unsqueeze(1)?;
        let txt_scale_mlp_unsqueezed = txt_scale_mlp_clamped.unsqueeze(1)?;
        let txt_shift_mlp_unsqueezed = txt_shift_mlp.unsqueeze(1)?;

        let img_scale_mlp_expanded =
            img_scale_mlp_unsqueezed.expand(&[1, img_seq_len, self.hidden_size])?;
        let img_shift_mlp_expanded =
            img_shift_mlp_unsqueezed.expand(&[1, img_seq_len, self.hidden_size])?;
        let txt_scale_mlp_expanded =
            txt_scale_mlp_unsqueezed.expand(&[1, txt_seq_len, self.hidden_size])?;
        let txt_shift_mlp_expanded =
            txt_shift_mlp_unsqueezed.expand(&[1, txt_seq_len, self.hidden_size])?;

        let one_img_mlp = Tensor::ones_like(&img_scale_mlp_expanded)?;
        let img_mod2 = img_norm2
            .mul(&one_img_mlp.add(&img_scale_mlp_expanded)?)?
            .add(&img_shift_mlp_expanded)?;
        let one_txt_mlp = Tensor::ones_like(&txt_scale_mlp_expanded)?;
        let txt_mod2 = txt_norm2
            .mul(&one_txt_mlp.add(&txt_scale_mlp_expanded)?)?
            .add(&txt_shift_mlp_expanded)?;

        // Apply MLP
        let img_mlp = if let (Some(fc1_w), Some(fc2_w)) =
            (self.weights.get("img_mlp.0.weight"), self.weights.get("img_mlp.2.weight"))
        {
            // Flatten for MLP
            let shape = img_mod2.shape().dims();
            let batch_seq = shape[0] * shape[1];
            let hidden = shape[2];
            let img_flat = img_mod2.reshape(&[batch_seq, hidden])?;

            // First layer
            let fc1_w_t = fc1_w.transpose()?;
            let mut h = img_flat.matmul(&fc1_w_t)?;
            if let Some(fc1_b) = self.weights.get("img_mlp.0.bias") {
                h = h.add(fc1_b)?;
            }
            h = h.gelu()?;

            // Second layer
            let fc2_w_t = fc2_w.transpose()?;
            let mut mlp_out = h.matmul(&fc2_w_t)?;
            if let Some(fc2_b) = self.weights.get("img_mlp.2.bias") {
                mlp_out = mlp_out.add(fc2_b)?;
            }

            // Reshape back and apply gate
            let img_gate_mlp_unsqueezed = img_gate_mlp.unsqueeze(1)?;
            let img_gate_mlp_expanded =
                img_gate_mlp_unsqueezed.expand(&[1, img_seq_len, self.hidden_size])?;
            mlp_out.reshape(&[shape[0], shape[1], self.hidden_size])?.mul(&img_gate_mlp_expanded)?
        } else {
            return Err(flame_core::Error::InvalidOperation(
                "Missing MLP weights".to_string(),
            ));
        };

        let txt_mlp = if let (Some(fc1_w), Some(fc2_w)) =
            (self.weights.get("txt_mlp.0.weight"), self.weights.get("txt_mlp.2.weight"))
        {
            // Flatten for MLP
            let shape = txt_mod2.shape().dims();
            let batch_seq = shape[0] * shape[1];
            let hidden = shape[2];
            let txt_flat = txt_mod2.reshape(&[batch_seq, hidden])?;

            // First layer
            let fc1_w_t = fc1_w.transpose()?;
            let mut h = txt_flat.matmul(&fc1_w_t)?;
            if let Some(fc1_b) = self.weights.get("txt_mlp.0.bias") {
                h = h.add(fc1_b)?;
            }
            h = h.gelu()?;

            // Second layer
            let fc2_w_t = fc2_w.transpose()?;
            let mut mlp_out = h.matmul(&fc2_w_t)?;
            if let Some(fc2_b) = self.weights.get("txt_mlp.2.bias") {
                mlp_out = mlp_out.add(fc2_b)?;
            }

            // Reshape back and apply gate
            let txt_gate_mlp_unsqueezed = txt_gate_mlp.unsqueeze(1)?;
            let txt_gate_mlp_expanded =
                txt_gate_mlp_unsqueezed.expand(&[1, txt_seq_len, self.hidden_size])?;
            mlp_out.reshape(&[shape[0], shape[1], self.hidden_size])?.mul(&txt_gate_mlp_expanded)?
        } else {
            return Err(flame_core::Error::InvalidOperation(
                "Missing MLP weights".to_string(),
            ));
        };

        // Final residual
        let img_final = img_post_attn.add(&img_mlp)?;
        let txt_final = txt_post_attn.add(&txt_mlp)?;

        // Add assertions to catch explosions
        let img_max = img_final.max_all()?;
        let txt_max = txt_final.max_all()?;
        if img_max > 1e6 || txt_max > 1e6 || img_max.is_nan() || txt_max.is_nan() {
            return Err(flame_core::Error::InvalidOperation(format!(
                "Explosion detected in block output: img_max={}, txt_max={}",
                img_max, txt_max
            )));
        }

        Ok((img_final, txt_final))
    }
}

impl SingleStreamBlockFixed {
    /// Create from loaded weights
    pub fn from_weights(
        weights: &HashMap<String, Tensor>,
        hidden_size: usize,
        num_heads: usize,
        mlp_ratio: f32,
        device: Device,
    ) -> Result<Self> {
        // Clone all weights (Tensor::clone returns Result)
        let mut cloned_weights = HashMap::new();
        for (k, v) in weights {
            cloned_weights.insert(k.clone(), v.clone());
        }

        Ok(Self { hidden_size, num_heads, mlp_ratio, device, weights: cloned_weights })
    }

    /// Apply attention with QK-Norm
    fn apply_attention(
        &self,
        input: &Tensor,
        qkv_weight: &Tensor,
        qkv_bias: Option<&Tensor>,
        proj_weight: &Tensor,
        proj_bias: Option<&Tensor>,
        query_norm_scale: Option<&Tensor>,
        key_norm_scale: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Project to QKV
        let input_shape = input.shape().dims();
        let batch_size = input_shape[0];
        let seq_len = input_shape[1];
        let hidden_size = input_shape[2];

        // Flatten for matmul
        let input_flat = input.reshape(&[batch_size * seq_len, hidden_size])?;
        let qkv_weight_t = qkv_weight.transpose()?;
        let mut qkv = input_flat.matmul(&qkv_weight_t)?;
        if let Some(bias) = qkv_bias {
            qkv = qkv.add(bias)?;
        }

        // Reshape back
        let qkv_out_features = qkv_weight.shape().dims()[0];
        let qkv = qkv.reshape(&[batch_size, seq_len, qkv_out_features])?;

        // Split QKV
        let head_dim = self.hidden_size / self.num_heads;
        let (q, k, v) = split_qkv(&qkv, self.num_heads, head_dim)?;

        // Apply QK-Norm
        let (q_normed, k_normed) = apply_qk_norm(&q, &k, query_norm_scale, key_norm_scale, 1e-6)?;

        // Compute attention
        let scale = 1.0 / (head_dim as f32).sqrt();
        let attn_output = scaled_dot_product_attention(&q_normed, &k_normed, &v, scale)?;

        // Project output
        let attn_shape = attn_output.shape().dims();
        let attn_flat = attn_output.reshape(&[attn_shape[0] * attn_shape[1], attn_shape[2]])?;
        let proj_weight_t = proj_weight.transpose()?;
        let mut output = attn_flat.matmul(&proj_weight_t)?;
        if let Some(bias) = proj_bias {
            output = output.add(bias)?;
        }

        // Reshape back
        let output = output.reshape(&[batch_size, seq_len, self.hidden_size])?;

        Ok(output)
    }

    /// Forward pass with proper AdaLN for single stream
    pub fn forward(
        &self,
        x: &Tensor,
        modulation: &Tensor, // [batch_size, modulation_dim]
        norm_weights: &HashMap<String, &Tensor>,
        guidance: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Debug shapes
        println!(
            "🔍 SingleStreamBlockFixed forward - x shape: {:?}, modulation shape: {:?}",
            x.shape().dims(),
            modulation.shape().dims()
        );
        // Extract layer norm weights
        let norm1_weight = norm_weights
            .get("norm1.weight")
            .ok_or(flame_core::Error::InvalidOperation("Missing norm1.weight".into()))?;
        let norm1_bias = norm_weights.get("norm1.bias").copied();

        let norm2_weight = norm_weights
            .get("norm2.weight")
            .ok_or(flame_core::Error::InvalidOperation("Missing norm2.weight".into()))?;
        let norm2_bias = norm_weights.get("norm2.bias").copied();

        // Step 1: Project modulation to get scale/shift parameters
        let mod_weight = self.weights.get("modulation.lin.weight").ok_or(
            flame_core::Error::InvalidOperation("Missing modulation.lin.weight".into()),
        )?;
        let mod_bias = self.weights.get("modulation.lin.bias").ok_or(
            flame_core::Error::InvalidOperation("Missing modulation.lin.bias".into()),
        )?;

        // Project modulation: [batch, mod_dim] -> [batch, 6 * hidden_size]
        let mod_params = modulation.matmul(&mod_weight.transpose()?)?.add(&mod_bias)?;

        // Split modulation parameters into (shift, scale) for each sub-block
        let chunk_size = self.hidden_size;
        let params = mod_params.chunk(6, 1)?;

        // Extract parameters
        let shift_msa = &params[0];
        let scale_msa = &params[1];
        let gate_msa = &params[2];
        let shift_mlp = &params[3];
        let scale_mlp = &params[4];
        let gate_mlp = &params[5];

        // Step 2: Apply attention with AdaLN
        // Normalize inputs
        let norm1 = layer_norm(x, norm1_weight, norm1_bias, 1e-6)?;

        // Apply AdaLN modulation: output = norm * (1 + scale) + shift
        // SAFETY: Clamp scale values to prevent explosion
        let scale_clamped = scale_msa.clamp(-2.0, 2.0)?;

        // Need to broadcast modulation from [batch, hidden] to [batch, seq_len, hidden]
        let seq_len = norm1.shape().dims()[1];

        // Unsqueeze to add sequence dimension: [batch, hidden] -> [batch, 1, hidden]
        let scale_unsqueezed = scale_clamped.unsqueeze(1)?;
        let shift_unsqueezed = shift_msa.unsqueeze(1)?;

        // Expand along sequence dimension
        let scale_expanded = scale_unsqueezed.expand(&[1, seq_len, self.hidden_size])?;
        let shift_expanded = shift_unsqueezed.expand(&[1, seq_len, self.hidden_size])?;

        let one = Tensor::ones_like(&scale_expanded)?;
        let mod1 = norm1.mul(&one.add(&scale_expanded)?)?.add(&shift_expanded)?;

        // Apply attention - SingleStreamBlock uses linear1/linear2 instead of separate attention/mlp
        // But we can still apply attention if weights are present
        let attn_out = if let (Some(qkv_w), Some(proj_w)) =
            (self.weights.get("attn.qkv.weight"), self.weights.get("attn.proj.weight"))
        {
            let attn = self.apply_attention(
                &mod1,
                qkv_w,
                self.weights.get("attn.qkv.bias"),
                proj_w,
                self.weights.get("attn.proj.bias"),
                norm_weights.get("attn.norm.query_norm.scale").copied(),
                norm_weights.get("attn.norm.key_norm.scale").copied(),
            )?;
            // Apply gate with broadcasting
            let gate_msa_unsqueezed = gate_msa.unsqueeze(1)?;
            let gate_msa_expanded = gate_msa_unsqueezed.expand(&[1, seq_len, self.hidden_size])?;
            attn.mul(&gate_msa_expanded)?
        } else if let Some(linear1_w) = self.weights.get("linear1.weight") {
            // Fallback to linear1 if no attention weights
            let shape = mod1.shape().dims();
            let batch_seq = shape[0] * shape[1];
            let hidden = shape[2];
            let x_flat = mod1.reshape(&[batch_seq, hidden])?;

            let linear1_w_t = linear1_w.transpose()?;
            let mut h = x_flat.matmul(&linear1_w_t)?;
            if let Some(linear1_b) = self.weights.get("linear1.bias") {
                h = h.add(linear1_b)?;
            }

            // Apply gate with broadcasting
            let gate_msa_unsqueezed = gate_msa.unsqueeze(1)?;
            let gate_msa_expanded = gate_msa_unsqueezed.expand(&[1, seq_len, self.hidden_size])?;
            h.reshape(&[shape[0], shape[1], self.hidden_size])?.mul(&gate_msa_expanded)?
        } else {
            return Err(flame_core::Error::InvalidOperation(
                "Missing attention or linear1 weights".to_string(),
            ));
        };

        // Add residual
        let post_attn = x.add(&attn_out)?;

        // Step 3: Apply MLP with AdaLN
        // Normalize
        let norm2 = layer_norm(&post_attn, norm2_weight, norm2_bias, 1e-6)?;

        // Apply AdaLN modulation with safety clamping
        let scale_mlp_clamped = scale_mlp.clamp(-2.0, 2.0)?;

        // Need to broadcast MLP modulation too
        let scale_mlp_unsqueezed = scale_mlp_clamped.unsqueeze(1)?;
        let shift_mlp_unsqueezed = shift_mlp.unsqueeze(1)?;

        let scale_mlp_expanded = scale_mlp_unsqueezed.expand(&[1, seq_len, self.hidden_size])?;
        let shift_mlp_expanded = shift_mlp_unsqueezed.expand(&[1, seq_len, self.hidden_size])?;

        let one_mlp = Tensor::ones_like(&scale_mlp_expanded)?;
        let mod2 = norm2.mul(&one_mlp.add(&scale_mlp_expanded)?)?.add(&shift_mlp_expanded)?;

        // Apply MLP or linear2
        let mlp_out = if let (Some(fc1_w), Some(fc2_w)) =
            (self.weights.get("mlp.0.weight"), self.weights.get("mlp.2.weight"))
        {
            // Standard MLP
            let shape = mod2.shape().dims();
            let batch_seq = shape[0] * shape[1];
            let hidden = shape[2];
            let x_flat = mod2.reshape(&[batch_seq, hidden])?;

            // First layer
            let fc1_w_t = fc1_w.transpose()?;
            let mut h = x_flat.matmul(&fc1_w_t)?;
            if let Some(fc1_b) = self.weights.get("mlp.0.bias") {
                h = h.add(fc1_b)?;
            }
            h = h.gelu()?;

            // Second layer
            let fc2_w_t = fc2_w.transpose()?;
            let mut mlp_out = h.matmul(&fc2_w_t)?;
            if let Some(fc2_b) = self.weights.get("mlp.2.bias") {
                mlp_out = mlp_out.add(fc2_b)?;
            }

            // Reshape back and apply gate
            let gate_mlp_unsqueezed = gate_mlp.unsqueeze(1)?;
            let gate_mlp_expanded = gate_mlp_unsqueezed.expand(&[1, seq_len, self.hidden_size])?;
            mlp_out.reshape(&[shape[0], shape[1], self.hidden_size])?.mul(&gate_mlp_expanded)?
        } else if let Some(linear2_w) = self.weights.get("linear2.weight") {
            // Fallback to linear2
            let shape = mod2.shape().dims();
            let batch_seq = shape[0] * shape[1];
            let hidden = shape[2];
            let x_flat = mod2.reshape(&[batch_seq, hidden])?;

            let linear2_w_t = linear2_w.transpose()?;
            let mut h = x_flat.matmul(&linear2_w_t)?;
            if let Some(linear2_b) = self.weights.get("linear2.bias") {
                h = h.add(linear2_b)?;
            }

            // Apply gate with broadcasting
            let gate_mlp_unsqueezed = gate_mlp.unsqueeze(1)?;
            let gate_mlp_expanded = gate_mlp_unsqueezed.expand(&[1, seq_len, self.hidden_size])?;
            h.reshape(&[shape[0], shape[1], self.hidden_size])?.mul(&gate_mlp_expanded)?
        } else {
            return Err(flame_core::Error::InvalidOperation(
                "Missing MLP or linear2 weights".to_string(),
            ));
        };

        // Final residual
        let final_out = post_attn.add(&mlp_out)?;

        // Add assertions to catch explosions
        let max_val = final_out.max_all()?;
        if max_val > 1e6 || max_val.is_nan() {
            return Err(flame_core::Error::InvalidOperation(format!(
                "Explosion detected in single block output: max={}",
                max_val
            )));
        }

        Ok(final_out)
    }
}
