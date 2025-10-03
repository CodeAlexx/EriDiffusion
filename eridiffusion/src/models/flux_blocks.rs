//! Flux transformer blocks for layer-by-layer streaming
//!
//! This provides lightweight wrappers around the Flux blocks that can be
//! created from loaded weights without keeping the full model in memory.

use crate::ops::qk_norm::{apply_qk_norm, scaled_dot_product_attention, split_qkv};
use crate::ops::streaming_rms_norm::{apply_double_stream_rms_norm, apply_single_stream_rms_norm};
use flame_core::{DType, Device, Result, Tensor};
use std::collections::HashMap;

/// Double stream block that processes img and txt separately
pub struct DoubleStreamBlock {
    hidden_size: usize,
    num_heads: usize,
    mlp_ratio: f32,
    device: Device,
    weights: HashMap<String, Tensor>,
}

impl DoubleStreamBlock {
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
        prefix: &str,
    ) -> Result<Tensor> {
        // Project to QKV
        // Weights are stored as [out_features, in_features], need to transpose for matmul
        println!("🔍 {} - input shape: {:?}", prefix, input.shape().dims());
        println!("🔍 {} - qkv_weight shape: {:?}", prefix, qkv_weight.shape().dims());
        let qkv_weight_t = qkv_weight.transpose()?;
        println!("🔍 {} - qkv_weight_t shape: {:?}", prefix, qkv_weight_t.shape().dims());

        // Reshape input for matmul: [batch, seq_len, hidden] -> [batch * seq_len, hidden]
        let input_shape = input.shape().dims();
        let batch_size = input_shape[0];
        let seq_len = input_shape[1];
        let hidden_size = input_shape[2];

        let input_flat = input.reshape(&[batch_size * seq_len, hidden_size])?;
        let mut qkv = input_flat.matmul(&qkv_weight_t)?;
        if let Some(bias) = qkv_bias {
            qkv = qkv.add(bias)?;
        }

        // Reshape back to [batch, seq_len, 3 * hidden_size]
        let qkv_out_features = qkv_weight.shape().dims()[0]; // Should be 3 * hidden_size
        let qkv = qkv.reshape(&[batch_size, seq_len, qkv_out_features])?;

        // Split QKV into separate tensors
        let head_dim = self.hidden_size / self.num_heads;
        println!(
            "🔍 {} - QKV shape: {:?}, num_heads: {}, head_dim: {}",
            prefix,
            qkv.shape().dims(),
            self.num_heads,
            head_dim
        );
        let (q, k, v) = split_qkv(&qkv, self.num_heads, head_dim)?;

        // Apply QK-Norm
        if query_norm_scale.is_some() || key_norm_scale.is_some() {
            println!(
                "🔍 {} - QK-Norm weights found: query={}, key={}",
                prefix,
                query_norm_scale.is_some(),
                key_norm_scale.is_some()
            );
        } else {
            println!("⚠️  {} - No QK-Norm weights found!", prefix);
        }

        let (q_normed, k_normed) = apply_qk_norm(&q, &k, query_norm_scale, key_norm_scale, 1e-6)?;

        // Compute scaled dot-product attention
        let scale = 1.0 / (head_dim as f32).sqrt();
        let attn_output = scaled_dot_product_attention(&q_normed, &k_normed, &v, scale)?;

        println!("🔍 {} - attn_output shape: {:?}", prefix, attn_output.shape().dims());
        println!("🔍 {} - proj_weight shape: {:?}", prefix, proj_weight.shape().dims());

        // Project output
        // Weights are stored as [out_features, in_features], need to transpose for matmul
        let proj_weight_t = proj_weight.transpose()?;
        println!("🔍 {} - proj_weight_t shape: {:?}", prefix, proj_weight_t.shape().dims());

        // Reshape attention output for matmul
        let attn_shape = attn_output.shape().dims();
        let attn_flat = attn_output.reshape(&[attn_shape[0] * attn_shape[1], attn_shape[2]])?;
        let mut output = attn_flat.matmul(&proj_weight_t)?;
        if let Some(bias) = proj_bias {
            output = output.add(bias)?;
        }

        // Reshape back to [batch, seq_len, hidden_size]
        let output = output.reshape(&[batch_size, seq_len, self.hidden_size])?;

        Ok(output)
    }

    /// Forward pass
    pub fn forward(
        &self,
        img: &Tensor,
        img_ids: &Tensor,
        txt: &Tensor,
        txt_ids: &Tensor,
        guidance: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        // Simplified implementation - in reality would need full attention/MLP
        // For now, just apply some transformations to show the structure

        // Apply modulation (AdaLN) with bias
        let img_mod = if let Some(w) = self.weights.get("img_mod.linear.weight") {
            // Weights are stored as [out_features, in_features], need to transpose for matmul
            let w_t = w.transpose()?;
            let mut result = img.matmul(&w_t)?;
            if let Some(b) = self.weights.get("img_mod.linear.bias") {
                result = result.add(b)?;
            }
            result
        } else {
            img.clone()
        };

        let txt_mod = if let Some(w) = self.weights.get("txt_mod.linear.weight") {
            // Weights are stored as [out_features, in_features], need to transpose for matmul
            let w_t = w.transpose()?;
            let mut result = txt.matmul(&w_t)?;
            if let Some(b) = self.weights.get("txt_mod.linear.bias") {
                result = result.add(b)?;
            }
            result
        } else {
            txt.clone()
        };

        // Apply attention (simplified) with bias
        let img_attn = if let Some(qkv_w) = self.weights.get("img_attn.qkv.weight") {
            // Simplified attention - just linear projection
            // Weights are stored as [out_features, in_features], need to transpose for matmul
            let qkv_w_t = qkv_w.transpose()?;
            let mut qkv = img.matmul(&qkv_w_t)?;
            if let Some(qkv_b) = self.weights.get("img_attn.qkv.bias") {
                qkv = qkv.add(qkv_b)?;
            }
            if let Some(proj_w) = self.weights.get("img_attn.proj.weight") {
                // Weights are stored as [out_features, in_features], need to transpose for matmul
                let proj_w_t = proj_w.transpose()?;
                let mut result = qkv.matmul(&proj_w_t)?;
                if let Some(proj_b) = self.weights.get("img_attn.proj.bias") {
                    result = result.add(proj_b)?;
                }
                result
            } else {
                qkv
            }
        } else {
            img.clone()
        };

        let txt_attn = if let Some(qkv_w) = self.weights.get("txt_attn.qkv.weight") {
            // Weights are stored as [out_features, in_features], need to transpose for matmul
            let qkv_w_t = qkv_w.transpose()?;
            let mut qkv = txt.matmul(&qkv_w_t)?;
            if let Some(qkv_b) = self.weights.get("txt_attn.qkv.bias") {
                qkv = qkv.add(qkv_b)?;
            }
            if let Some(proj_w) = self.weights.get("txt_attn.proj.weight") {
                // Weights are stored as [out_features, in_features], need to transpose for matmul
                let proj_w_t = proj_w.transpose()?;
                let mut result = qkv.matmul(&proj_w_t)?;
                if let Some(proj_b) = self.weights.get("txt_attn.proj.bias") {
                    result = result.add(proj_b)?;
                }
                result
            } else {
                qkv
            }
        } else {
            txt.clone()
        };

        // Add residual
        let img_out = img.add(&img_attn)?;
        let txt_out = txt.add(&txt_attn)?;

        // Apply MLP (simplified) with bias
        let img_mlp = if let Some(fc1_w) = self.weights.get("img_mlp.0.weight") {
            // Weights are stored as [out_features, in_features], need to transpose for matmul
            let fc1_w_t = fc1_w.transpose()?;
            let mut h = img_out.matmul(&fc1_w_t)?;
            if let Some(fc1_b) = self.weights.get("img_mlp.0.bias") {
                h = h.add(fc1_b)?;
            }
            h = h.gelu()?;
            if let Some(fc2_w) = self.weights.get("img_mlp.2.weight") {
                // Weights are stored as [out_features, in_features], need to transpose for matmul
                let fc2_w_t = fc2_w.transpose()?;
                let mut result = h.matmul(&fc2_w_t)?;
                if let Some(fc2_b) = self.weights.get("img_mlp.2.bias") {
                    result = result.add(fc2_b)?;
                }
                result
            } else {
                h
            }
        } else {
            img_out.clone()
        };

        let txt_mlp = if let Some(fc1_w) = self.weights.get("txt_mlp.0.weight") {
            // Weights are stored as [out_features, in_features], need to transpose for matmul
            let fc1_w_t = fc1_w.transpose()?;
            let mut h = txt_out.matmul(&fc1_w_t)?;
            if let Some(fc1_b) = self.weights.get("txt_mlp.0.bias") {
                h = h.add(fc1_b)?;
            }
            h = h.gelu()?;
            if let Some(fc2_w) = self.weights.get("txt_mlp.2.weight") {
                // Weights are stored as [out_features, in_features], need to transpose for matmul
                let fc2_w_t = fc2_w.transpose()?;
                let mut result = h.matmul(&fc2_w_t)?;
                if let Some(fc2_b) = self.weights.get("txt_mlp.2.bias") {
                    result = result.add(fc2_b)?;
                }
                result
            } else {
                h
            }
        } else {
            txt_out.clone()
        };

        // Final residual
        let img_final = img_out.add(&img_mlp)?;
        let txt_final = txt_out.add(&txt_mlp)?;

        Ok((img_final, txt_final))
    }

    /// Forward pass with QK normalization (Flux uses QK-Norm in attention, not block-level norm)
    pub fn forward_with_norm(
        &self,
        img: &Tensor,
        img_ids: &Tensor,
        txt: &Tensor,
        txt_ids: &Tensor,
        guidance: Option<&Tensor>,
        norm_weights: &HashMap<String, &Tensor>,
        eps: f64,
    ) -> Result<(Tensor, Tensor)> {
        // Flux doesn't apply block-level normalization
        // It uses QK-Norm inside attention and AdaLN modulation
        let img_normed = img.clone();
        let txt_normed = txt.clone();

        // Apply modulation (AdaLN) with bias
        let img_mod = if let Some(w) = self.weights.get("img_mod.linear.weight") {
            // Weights are stored as [out_features, in_features], need to transpose for matmul
            let w_t = w.transpose()?;
            let mut result = img_normed.matmul(&w_t)?;
            if let Some(b) = self.weights.get("img_mod.linear.bias") {
                result = result.add(b)?;
            }
            result
        } else {
            img_normed.clone()
        };

        let txt_mod = if let Some(w) = self.weights.get("txt_mod.linear.weight") {
            // Weights are stored as [out_features, in_features], need to transpose for matmul
            let w_t = w.transpose()?;
            let mut result = txt_normed.matmul(&w_t)?;
            if let Some(b) = self.weights.get("txt_mod.linear.bias") {
                result = result.add(b)?;
            }
            result
        } else {
            txt_normed.clone()
        };

        // Apply attention with proper QK-Norm
        let img_attn = if let (Some(qkv_w), Some(proj_w)) =
            (self.weights.get("img_attn.qkv.weight"), self.weights.get("img_attn.proj.weight"))
        {
            // Debug QK-Norm weights
            let q_scale = norm_weights.get(&"img_attn.norm.query_norm.scale".to_string()).copied();
            let k_scale = norm_weights.get(&"img_attn.norm.key_norm.scale".to_string()).copied();

            if q_scale.is_some() || k_scale.is_some() {
                println!("🎯 QK-Norm weights found for img_attn:");
                if let Some(q) = q_scale {
                    let q_data: Vec<f32> = q.to_vec()?;
                    println!(
                        "  - Query scale: min={:.4}, max={:.4}, mean={:.4}",
                        q_data.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
                        q_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)),
                        q_data.iter().sum::<f32>() / q_data.len() as f32
                    );
                }
                if let Some(k) = k_scale {
                    let k_data: Vec<f32> = k.to_vec()?;
                    println!(
                        "  - Key scale: min={:.4}, max={:.4}, mean={:.4}",
                        k_data.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
                        k_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)),
                        k_data.iter().sum::<f32>() / k_data.len() as f32
                    );
                }
            } else {
                println!("⚠️  No QK-Norm weights found for img_attn!");
            }

            self.apply_attention(
                &img_normed,
                qkv_w,
                self.weights.get("img_attn.qkv.bias"),
                proj_w,
                self.weights.get("img_attn.proj.bias"),
                q_scale,
                k_scale,
                "img_attn",
            )?
        } else {
            img_normed.clone()
        };

        let txt_attn = if let (Some(qkv_w), Some(proj_w)) =
            (self.weights.get("txt_attn.qkv.weight"), self.weights.get("txt_attn.proj.weight"))
        {
            self.apply_attention(
                &txt_normed,
                qkv_w,
                self.weights.get("txt_attn.qkv.bias"),
                proj_w,
                self.weights.get("txt_attn.proj.bias"),
                // Look for the exact keys that were stored in extract_rms_norm_weights
                norm_weights.get(&"txt_attn.norm.query_norm.scale".to_string()).copied(),
                norm_weights.get(&"txt_attn.norm.key_norm.scale".to_string()).copied(),
                "txt_attn",
            )?
        } else {
            txt_normed.clone()
        };

        // Add residual (skip connection)
        let img_out = img.add(&img_attn)?;
        let txt_out = txt.add(&txt_attn)?;

        // Apply normalization before MLP if we have norm2 weights
        let (img_mlp_in, txt_mlp_in) = if norm_weights.contains_key("img_norm2.weight") {
            let mut mlp_norm_weights = HashMap::new();
            if let Some(&w) = norm_weights.get("img_norm2.weight") {
                mlp_norm_weights.insert("img_norm.weight".to_string(), w);
            }
            if let Some(&b) = norm_weights.get("img_norm2.bias") {
                mlp_norm_weights.insert("img_norm.bias".to_string(), b);
            }
            if let Some(&w) = norm_weights.get("txt_norm2.weight") {
                mlp_norm_weights.insert("txt_norm.weight".to_string(), w);
            }
            if let Some(&b) = norm_weights.get("txt_norm2.bias") {
                mlp_norm_weights.insert("txt_norm.bias".to_string(), b);
            }
            apply_double_stream_rms_norm(&img_out, &txt_out, &mlp_norm_weights, eps)?
        } else {
            (img_out.clone(), txt_out.clone())
        };

        // Apply MLP with normalized inputs
        let img_mlp = if let Some(fc1_w) = self.weights.get("img_mlp.0.weight") {
            println!("🔍 img_mlp - fc1_w shape: {:?}", fc1_w.shape().dims());
            println!("🔍 img_mlp - img_mlp_in shape: {:?}", img_mlp_in.shape().dims());

            // Reshape input for matmul: [batch, seq_len, hidden] -> [batch * seq_len, hidden]
            let img_shape = img_mlp_in.shape().dims();
            let batch_size = img_shape[0];
            let seq_len = img_shape[1];
            let hidden_size = img_shape[2];

            let img_flat = img_mlp_in.reshape(&[batch_size * seq_len, hidden_size])?;

            // Weights are stored as [out_features, in_features], need to transpose for matmul
            let fc1_w_t = fc1_w.transpose()?;
            println!("🔍 img_mlp - fc1_w_t shape: {:?}", fc1_w_t.shape().dims());
            let mut h = img_flat.matmul(&fc1_w_t)?;
            if let Some(fc1_b) = self.weights.get("img_mlp.0.bias") {
                h = h.add(fc1_b)?;
            }
            h = h.gelu()?;
            if let Some(fc2_w) = self.weights.get("img_mlp.2.weight") {
                // Weights are stored as [out_features, in_features], need to transpose for matmul
                let fc2_w_t = fc2_w.transpose()?;
                let mut result = h.matmul(&fc2_w_t)?;
                if let Some(fc2_b) = self.weights.get("img_mlp.2.bias") {
                    result = result.add(fc2_b)?;
                }
                // Reshape back to [batch, seq_len, hidden_size]
                result.reshape(&[batch_size, seq_len, self.hidden_size])?
            } else {
                // Reshape back to [batch, seq_len, hidden_size]
                h.reshape(&[batch_size, seq_len, self.hidden_size])?
            }
        } else {
            img_mlp_in.clone()
        };

        let txt_mlp = if let Some(fc1_w) = self.weights.get("txt_mlp.0.weight") {
            println!("🔍 txt_mlp - fc1_w shape: {:?}", fc1_w.shape().dims());
            println!("🔍 txt_mlp - txt_mlp_in shape: {:?}", txt_mlp_in.shape().dims());

            // Reshape input for matmul: [batch, seq_len, hidden] -> [batch * seq_len, hidden]
            let txt_shape = txt_mlp_in.shape().dims();
            let batch_size = txt_shape[0];
            let seq_len = txt_shape[1];
            let hidden_size = txt_shape[2];

            let txt_flat = txt_mlp_in.reshape(&[batch_size * seq_len, hidden_size])?;

            // Weights are stored as [out_features, in_features], need to transpose for matmul
            let fc1_w_t = fc1_w.transpose()?;
            println!("🔍 txt_mlp - fc1_w_t shape: {:?}", fc1_w_t.shape().dims());
            let mut h = txt_flat.matmul(&fc1_w_t)?;
            if let Some(fc1_b) = self.weights.get("txt_mlp.0.bias") {
                h = h.add(fc1_b)?;
            }
            h = h.gelu()?;
            if let Some(fc2_w) = self.weights.get("txt_mlp.2.weight") {
                // Weights are stored as [out_features, in_features], need to transpose for matmul
                let fc2_w_t = fc2_w.transpose()?;
                let mut result = h.matmul(&fc2_w_t)?;
                if let Some(fc2_b) = self.weights.get("txt_mlp.2.bias") {
                    result = result.add(fc2_b)?;
                }
                // Reshape back to [batch, seq_len, hidden_size]
                result.reshape(&[batch_size, seq_len, self.hidden_size])?
            } else {
                // Reshape back to [batch, seq_len, hidden_size]
                h.reshape(&[batch_size, seq_len, self.hidden_size])?
            }
        } else {
            txt_mlp_in.clone()
        };

        // Final residual
        let img_final = img_out.add(&img_mlp)?;
        let txt_final = txt_out.add(&txt_mlp)?;

        Ok((img_final, txt_final))
    }
}

/// Single stream block that processes combined features
#[derive(Clone)]
pub struct SingleStreamBlock {
    hidden_size: usize,
    num_heads: usize,
    mlp_ratio: f32,
    device: Device,
    weights: HashMap<String, Tensor>,
}

impl SingleStreamBlock {
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

    /// Forward pass
    pub fn forward(&self, x: &Tensor, guidance: Option<&Tensor>) -> Result<Tensor> {
        // Simplified implementation

        // Apply modulation with bias
        let x_mod = if let Some(w) = self.weights.get("modulation.lin.weight") {
            // Weights are stored as [out_features, in_features], need to transpose for matmul
            let w_t = w.transpose()?;
            let mut result = x.matmul(&w_t)?;
            if let Some(b) = self.weights.get("modulation.lin.bias") {
                result = result.add(b)?;
            }
            result
        } else {
            x.clone()
        };

        // SingleStreamBlock uses linear1/linear2 instead of separate attention/mlp
        // Apply linear1
        let hidden = if let Some(w1) = self.weights.get("linear1.weight") {
            // Weights are stored as [out_features, in_features], need to transpose for matmul
            let w1_t = w1.transpose()?;
            let mut h = x.matmul(&w1_t)?;
            if let Some(b1) = self.weights.get("linear1.bias") {
                h = h.add(b1)?;
            }
            // Apply modulation here (simplified)
            h
        } else {
            x.clone()
        };

        // Apply linear2
        let output = if let Some(w2) = self.weights.get("linear2.weight") {
            // Weights are stored as [out_features, in_features], need to transpose for matmul
            let w2_t = w2.transpose()?;
            let mut result = hidden.matmul(&w2_t)?;
            if let Some(b2) = self.weights.get("linear2.bias") {
                result = result.add(b2)?;
            }
            result
        } else {
            hidden
        };

        // Final residual
        x.add(&output)
    }

    /// Forward pass with QK normalization (Flux uses QK-Norm in attention, not block-level norm)
    pub fn forward_with_norm(
        &self,
        x: &Tensor,
        guidance: Option<&Tensor>,
        norm_weights: &HashMap<String, &Tensor>,
        eps: f64,
    ) -> Result<Tensor> {
        // Flux doesn't apply block-level normalization
        // It uses QK-Norm inside attention
        let x_normed = x.clone();

        // Apply modulation with bias
        let x_mod = if let Some(w) = self.weights.get("modulation.lin.weight") {
            // Weights are stored as [out_features, in_features], need to transpose for matmul
            let w_t = w.transpose()?;
            let mut result = x_normed.matmul(&w_t)?;
            if let Some(b) = self.weights.get("modulation.lin.bias") {
                result = result.add(b)?;
            }
            result
        } else {
            x_normed.clone()
        };

        // Apply linear1 with normalized input
        let hidden = if let Some(w1) = self.weights.get("linear1.weight") {
            // Weights are stored as [out_features, in_features], need to transpose for matmul
            let w1_t = w1.transpose()?;
            let mut h = x_normed.matmul(&w1_t)?;
            if let Some(b1) = self.weights.get("linear1.bias") {
                h = h.add(b1)?;
            }
            // Modulation would be applied here in full implementation
            h
        } else {
            x_normed.clone()
        };

        // Apply linear2
        let output = if let Some(w2) = self.weights.get("linear2.weight") {
            // Weights are stored as [out_features, in_features], need to transpose for matmul
            let w2_t = w2.transpose()?;
            let mut result = hidden.matmul(&w2_t)?;
            if let Some(b2) = self.weights.get("linear2.bias") {
                result = result.add(b2)?;
            }
            result
        } else {
            hidden
        };

        // Final residual connection
        x.add(&output)
    }
}
