//! ACE-Step DiT decoder forward pass for LoRA training.
//!
//! Ported from flame-diffusion/acestep-trainer/src/model.rs (2026-05-05).
//! Reference: acestep/models/base/modeling_acestep_v15_base.py
//! Architecture: 24-layer DiT with self-attn + cross-attn, AdaLN modulation,
//!   GQA (16 heads, 8 KV heads), RoPE, SiLU-gated MLP (SwiGLU).
//!   Patch embed via Conv1d(in=192, out=2048, k=2, s=2).
//!
//! Weight key patterns (from safetensors):
//!   decoder.layers.{i}.self_attn.{q,k,v,o}_proj.weight
//!   decoder.layers.{i}.cross_attn.{q,k,v,o}_proj.weight
//!   decoder.layers.{i}.self_attn.{q,k}_norm.weight
//!   decoder.layers.{i}.cross_attn.{q,k}_norm.weight
//!   decoder.layers.{i}.{self_attn_norm,cross_attn_norm,mlp_norm}.weight
//!   decoder.layers.{i}.mlp.{gate_proj,up_proj,down_proj}.weight
//!   decoder.layers.{i}.scale_shift_table  [1, 6, 2048]
//!   decoder.time_embed{,_r}.{linear_1,linear_2,time_proj}.{weight,bias}
//!   decoder.{proj_in.1,proj_out.1}.{weight,bias}  (Conv1d / ConvTranspose1d)
//!   decoder.{condition_embedder,norm_out}.weight
//!   decoder.scale_shift_table  [1, 2, 2048]
//!   null_condition_emb  [1, 1, 2048]

use cudarc::driver::CudaDevice;
use flame_core::{
    conv1d,
    parameter::Parameter,
    serialization, DType, Error, Shape, Tensor,
};

use crate::lora::LoRALinear;
use crate::Result;
use std::{collections::HashMap, path::Path, sync::Arc};

/// ACE-Step DiT decoder configuration, auto-detected from weights.
#[derive(Debug, Clone)]
pub struct AceStepConfig {
    pub hidden_size: usize,     // 2048 (turbo) or 2560 (xl-base)
    pub num_heads: usize,       // 16 or 32
    pub num_kv_heads: usize,    // 8 or 32
    pub head_dim: usize,        // 128
    pub intermediate_size: usize, // 6144 or larger
    pub num_layers: usize,      // 24 or 32
    pub in_channels: usize,     // 192
    pub acoustic_dim: usize,    // 64
    pub patch_size: usize,      // 2
    pub rope_theta: f32,        // 1_000_000
    pub rms_norm_eps: f32,      // 1e-6
}

/// LoRA adapters for one DiT layer (self-attn + cross-attn).
#[derive(Clone)]
struct DiTLayerAdapters {
    self_q: LoRALinear,
    self_k: LoRALinear,
    self_v: LoRALinear,
    self_o: LoRALinear,
    cross_q: LoRALinear,
    cross_k: LoRALinear,
    cross_v: LoRALinear,
    cross_o: LoRALinear,
}

pub struct AceStepLoRAModel {
    weights: Arc<HashMap<String, Tensor>>,
    config: AceStepConfig,
    adapters: Vec<DiTLayerAdapters>,
    /// Null condition embedding [1, 1, hidden_size] for CFG dropout.
    null_condition_emb: Tensor,
}

impl AceStepLoRAModel {
    pub fn from_safetensors(
        path: &Path,
        rank: usize,
        alpha: f32,
        device: Arc<CudaDevice>,
    ) -> Result<Self> {
        let weights = serialization::load_file(path, &device)?;
        Self::from_weights(weights, rank, alpha, device)
    }

    pub fn from_weights(
        mut weights: HashMap<String, Tensor>,
        rank: usize,
        alpha: f32,
        device: Arc<CudaDevice>,
    ) -> Result<Self> {
        let hidden_size = {
            let w = weights.get("decoder.condition_embedder.weight")
                .ok_or_else(|| Error::InvalidInput("Missing decoder.condition_embedder.weight".into()))?;
            w.shape().dims()[0]
        };

        let num_heads = {
            let q = weights.get("decoder.layers.0.self_attn.q_proj.weight")
                .ok_or_else(|| Error::InvalidInput("Missing decoder.layers.0.self_attn.q_proj.weight".into()))?;
            q.shape().dims()[0] / 128
        };

        let num_kv_heads = {
            let k = weights.get("decoder.layers.0.self_attn.k_proj.weight")
                .ok_or_else(|| Error::InvalidInput("Missing decoder.layers.0.self_attn.k_proj.weight".into()))?;
            k.shape().dims()[0] / 128
        };

        let intermediate_size = {
            let g = weights.get("decoder.layers.0.mlp.gate_proj.weight")
                .ok_or_else(|| Error::InvalidInput("Missing decoder.layers.0.mlp.gate_proj.weight".into()))?;
            g.shape().dims()[0]
        };

        let in_channels = {
            let p = weights.get("decoder.proj_in.1.weight")
                .ok_or_else(|| Error::InvalidInput("Missing decoder.proj_in.1.weight".into()))?;
            p.shape().dims()[1]
        };

        let acoustic_dim = {
            let p = weights.get("decoder.proj_out.1.weight")
                .ok_or_else(|| Error::InvalidInput("Missing decoder.proj_out.1.weight".into()))?;
            p.shape().dims()[1]
        };

        let mut num_layers = 0;
        while weights.contains_key(&format!("decoder.layers.{num_layers}.self_attn.q_proj.weight")) {
            num_layers += 1;
        }

        let config = AceStepConfig {
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim: 128,
            intermediate_size,
            num_layers,
            in_channels,
            acoustic_dim,
            patch_size: 2,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
        };

        log::info!(
            "ACE-Step config: hidden={} heads={} kv_heads={} layers={} mlp={} in_ch={} acoustic={}",
            config.hidden_size, config.num_heads, config.num_kv_heads,
            config.num_layers, config.intermediate_size, config.in_channels,
            config.acoustic_dim,
        );

        let null_condition_emb = weights.remove("null_condition_emb")
            .ok_or_else(|| Error::InvalidInput("Missing null_condition_emb".into()))?
            .to_dtype(DType::BF16)?;

        // Convert all decoder weights to BF16, frozen, pre-transpose 2D weights
        let decoder_keys: Vec<String> = weights.keys()
            .filter(|k| k.starts_with("decoder."))
            .cloned()
            .collect();
        for key in decoder_keys {
            let tensor = weights.remove(&key).unwrap();
            let tensor = tensor.to_dtype(DType::BF16)?.requires_grad_(false);
            let dims = tensor.shape().dims().to_vec();
            let frozen = if key.ends_with(".weight") && dims.len() == 2 {
                tensor.transpose()?.requires_grad_(false)
            } else {
                tensor
            };
            weights.insert(key, frozen);
        }

        // Create LoRA adapters for each layer.
        // ACE-Step targets q, k, v, o projections in self-attn and cross-attn.
        let mut adapters = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let seed_base = 1000 + (i as u64) * 32;
            let q_dim = config.num_heads * config.head_dim;
            let kv_dim = config.num_kv_heads * config.head_dim;
            adapters.push(DiTLayerAdapters {
                self_q: LoRALinear::new(config.hidden_size, q_dim, rank, alpha, device.clone(), seed_base)?,
                self_k: LoRALinear::new(config.hidden_size, kv_dim, rank, alpha, device.clone(), seed_base + 1)?,
                self_v: LoRALinear::new(config.hidden_size, kv_dim, rank, alpha, device.clone(), seed_base + 2)?,
                self_o: LoRALinear::new(config.hidden_size, q_dim, rank, alpha, device.clone(), seed_base + 3)?,
                cross_q: LoRALinear::new(config.hidden_size, q_dim, rank, alpha, device.clone(), seed_base + 4)?,
                cross_k: LoRALinear::new(config.hidden_size, kv_dim, rank, alpha, device.clone(), seed_base + 5)?,
                cross_v: LoRALinear::new(config.hidden_size, kv_dim, rank, alpha, device.clone(), seed_base + 6)?,
                cross_o: LoRALinear::new(config.hidden_size, q_dim, rank, alpha, device.clone(), seed_base + 7)?,
            });
        }

        Ok(Self {
            weights: Arc::new(weights),
            config,
            adapters,
            null_condition_emb,
        })
    }

    /// Collect all trainable LoRA parameters for the optimizer.
    pub fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        for adapter in &self.adapters {
            params.extend(adapter.self_q.parameters());
            params.extend(adapter.self_k.parameters());
            params.extend(adapter.self_v.parameters());
            params.extend(adapter.self_o.parameters());
            params.extend(adapter.cross_q.parameters());
            params.extend(adapter.cross_k.parameters());
            params.extend(adapter.cross_v.parameters());
            params.extend(adapter.cross_o.parameters());
        }
        params
    }

    /// Same as [`parameters`] but each Parameter is paired with the canonical
    /// safetensors key used by [`save_lora`]. Required by
    /// `eridiffusion-core::training::checkpoint::save_full`.
    /// Order MUST mirror `parameters()` (LoRALinear::parameters returns `[lora_a, lora_b]`).
    pub fn named_parameters(&self) -> Vec<(String, Parameter)> {
        let mut out = Vec::new();
        let suffixes = [
            "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
            "cross_attn.q_proj", "cross_attn.k_proj", "cross_attn.v_proj", "cross_attn.o_proj",
        ];
        for (i, adapter) in self.adapters.iter().enumerate() {
            let layers = [
                &adapter.self_q, &adapter.self_k, &adapter.self_v, &adapter.self_o,
                &adapter.cross_q, &adapter.cross_k, &adapter.cross_v, &adapter.cross_o,
            ];
            for (sfx, lora) in suffixes.iter().zip(layers.iter()) {
                let prefix = format!("decoder.layers.{i}.{sfx}");
                out.push((format!("{prefix}.lora_A.weight"), lora.lora_a().clone()));
                out.push((format!("{prefix}.lora_B.weight"), lora.lora_b().clone()));
            }
        }
        out
    }

    /// Save LoRA weights as safetensors.
    pub fn save_lora(&self, path: &Path) -> Result<()> {
        let mut tensors = HashMap::new();
        for (i, adapter) in self.adapters.iter().enumerate() {
            let prefix = format!("decoder.layers.{i}");
            adapter.self_q.save_tensors(&format!("{prefix}.self_attn.q_proj"), &mut tensors)?;
            adapter.self_k.save_tensors(&format!("{prefix}.self_attn.k_proj"), &mut tensors)?;
            adapter.self_v.save_tensors(&format!("{prefix}.self_attn.v_proj"), &mut tensors)?;
            adapter.self_o.save_tensors(&format!("{prefix}.self_attn.o_proj"), &mut tensors)?;
            adapter.cross_q.save_tensors(&format!("{prefix}.cross_attn.q_proj"), &mut tensors)?;
            adapter.cross_k.save_tensors(&format!("{prefix}.cross_attn.k_proj"), &mut tensors)?;
            adapter.cross_v.save_tensors(&format!("{prefix}.cross_attn.v_proj"), &mut tensors)?;
            adapter.cross_o.save_tensors(&format!("{prefix}.cross_attn.o_proj"), &mut tensors)?;
        }
        serialization::save_file(&tensors, path)?;
        Ok(())
    }

    /// Get a reference to the null condition embedding for CFG dropout.
    pub fn null_condition_emb(&self) -> &Tensor {
        &self.null_condition_emb
    }

    pub fn config(&self) -> &AceStepConfig {
        &self.config
    }

    #[inline]
    fn w(&self, key: &str) -> flame_core::Result<&Tensor> {
        self.weights.get(key)
            .ok_or_else(|| Error::InvalidInput(format!("Missing weight: {key}")))
    }

    fn timestep_embedding(&self, t: &Tensor, dim: usize, scale: f32, prefix: &str)
        -> flame_core::Result<Tensor>
    {
        let device = t.device();
        let t_scaled = t.mul_scalar(scale)?;
        let half = dim / 2;
        let freqs_data: Vec<f32> = (0..half)
            .map(|i| (-(10000.0f32.ln()) * (i as f32) / (half as f32)).exp())
            .collect();
        let freqs = Tensor::from_vec(freqs_data, Shape::from_dims(&[1, half]), device.clone())?;

        let t_2d = t_scaled.to_dtype(DType::F32)?.reshape(&[t.shape().dims()[0], 1])?;
        let args = t_2d.broadcast_to(&Shape::from_dims(&[t.shape().dims()[0], half]))?
            .mul(&freqs.broadcast_to(&Shape::from_dims(&[t.shape().dims()[0], half]))?)?;
        let cos_args = args.cos()?;
        let sin_args = args.sin()?;
        let embedding = Tensor::cat(&[&cos_args, &sin_args], 1)?;
        let embedding = embedding.to_dtype(DType::BF16)?;

        let temb = linear3d_bias(
            &embedding.reshape(&[embedding.shape().dims()[0], 1, dim])?,
            self.w(&format!("{prefix}.linear_1.weight"))?,
            self.w(&format!("{prefix}.linear_1.bias"))?,
        )?;
        let temb = temb.silu()?;
        let temb = linear3d_bias(
            &temb,
            self.w(&format!("{prefix}.linear_2.weight"))?,
            self.w(&format!("{prefix}.linear_2.bias"))?,
        )?;

        let temb_act = temb.silu()?;
        let timestep_proj = linear3d_bias(
            &temb_act,
            self.w(&format!("{prefix}.time_proj.weight"))?,
            self.w(&format!("{prefix}.time_proj.bias"))?,
        )?;
        let proj_dims = timestep_proj.shape().dims().to_vec();
        let b = proj_dims[0];
        let _timestep_proj = timestep_proj.reshape(&[b, 6, self.config.hidden_size])?;

        let temb = temb.reshape(&[b, self.config.hidden_size])?;
        Ok(temb)
    }

    /// Run the decoder forward pass for training.
    ///
    /// # Arguments
    /// * `hidden_states` — Noised latents x_t [B, T, acoustic_dim=64]
    /// * `timestep` — Timestep values t [B]
    /// * `timestep_r` — Timestep r values [B] (= t for training)
    /// * `encoder_hidden_states` — Condition encoder output [B, L, hidden_size]
    /// * `context_latents` — Source context [B, T, in_channels=192]
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        timestep: &Tensor,
        timestep_r: &Tensor,
        encoder_hidden_states: &Tensor,
        context_latents: &Tensor,
    ) -> Result<Tensor> {
        let device = hidden_states.device();
        let b = hidden_states.shape().dims()[0];
        let original_seq_len = hidden_states.shape().dims()[1];
        let h = self.config.hidden_size;

        // Timestep embeddings
        let temb_t = self.timestep_embedding(timestep, 256, 1000.0, "decoder.time_embed")?;
        let t_minus_r = timestep.sub(timestep_r)?;
        let temb_r = self.timestep_embedding(&t_minus_r, 256, 1000.0, "decoder.time_embed_r")?;
        let temb = temb_t.add(&temb_r)?;

        let proj_t = self.compute_timestep_proj(timestep, "decoder.time_embed")?;
        let proj_r = self.compute_timestep_proj(&t_minus_r, "decoder.time_embed_r")?;
        let timestep_proj = proj_t.add(&proj_r)?;

        let last_dim = context_latents.shape().dims().len() - 1;
        let x = Tensor::cat(&[context_latents, hidden_states], last_dim)?;

        let seq_len = x.shape().dims()[1];
        let ps = self.config.patch_size;
        let pad_length = if seq_len % ps != 0 { ps - (seq_len % ps) } else { 0 };
        let x = if pad_length > 0 {
            let pad_shape = Shape::from_dims(&[b, pad_length, x.shape().dims()[2]]);
            let pad = Tensor::zeros_dtype(pad_shape, DType::BF16, device.clone())?;
            Tensor::cat(&[&x, &pad], 1)?
        } else {
            x
        };

        let x = conv1d_forward(
            &x,
            self.w("decoder.proj_in.1.weight")?,
            self.w("decoder.proj_in.1.bias")?,
            self.config.patch_size,
        )?;

        let encoder_hs = linear3d(
            encoder_hidden_states,
            self.w("decoder.condition_embedder.weight")?,
        )?;
        let cond_bias = self.w("decoder.condition_embedder.bias")?
            .reshape(&[1, 1, h])?
            .broadcast_to(encoder_hs.shape())?;
        let encoder_hs = encoder_hs.add(&cond_bias)?;

        let patched_seq_len = x.shape().dims()[1];
        let (cos, sin) = self.compute_rope(patched_seq_len, device.clone())?;

        let mut x = x;
        for layer_idx in 0..self.config.num_layers {
            x = self.dit_layer_forward(&x, &timestep_proj, &encoder_hs, &cos, &sin, layer_idx)?;
        }

        // Output: AdaLN + proj_out
        let out_sst = self.w("decoder.scale_shift_table")?;
        let temb_unsqueeze = temb.reshape(&[b, 1, h])?;
        let sst_plus_temb = out_sst.broadcast_to(&Shape::from_dims(&[b, 2, h]))?.add(&temb_unsqueeze)?;
        let shift = sst_plus_temb.narrow(1, 0, 1)?;
        let scale = sst_plus_temb.narrow(1, 1, 1)?;

        let x_normed = rms_norm(&x, self.w("decoder.norm_out.weight")?, self.config.rms_norm_eps)?;
        let ones = Tensor::ones_dtype(scale.shape().clone(), DType::BF16, device.clone())?;
        let x = x_normed.mul(&ones.add(&scale)?)?.add(&shift)?;

        let x = conv_transpose1d_forward(
            &x,
            self.w("decoder.proj_out.1.weight")?,
            self.w("decoder.proj_out.1.bias")?,
            self.config.patch_size,
        )?;

        // Force requires_grad=true then identity op so backward can trace.
        // The LoRA ops deep inside attention DO record on the tape, but the final
        // AdaLN + proj_out path uses base weights (requires_grad=false), so the
        // output tensor doesn't carry requires_grad. Identity mul records a
        // MulScalar op, giving backward a tape entry to start from.
        let x = x.requires_grad_(true);
        let x = x.mul_scalar(1.0)?;

        let x = if x.shape().dims()[1] > original_seq_len {
            x.narrow(1, 0, original_seq_len)?
        } else {
            x
        };

        Ok(x)
    }

    fn compute_timestep_proj(&self, t: &Tensor, prefix: &str) -> flame_core::Result<Tensor> {
        let device = t.device();
        let b = t.shape().dims()[0];
        let h = self.config.hidden_size;

        let t_scaled = t.mul_scalar(1000.0)?;
        let half = 128;
        let freqs_data: Vec<f32> = (0..half)
            .map(|i| (-(10000.0f32.ln()) * (i as f32) / (half as f32)).exp())
            .collect();
        let freqs = Tensor::from_vec(freqs_data, Shape::from_dims(&[1, half]), device.clone())?;
        let t_2d = t_scaled.to_dtype(DType::F32)?.reshape(&[b, 1])?;
        let args = t_2d.broadcast_to(&Shape::from_dims(&[b, half]))?
            .mul(&freqs.broadcast_to(&Shape::from_dims(&[b, half]))?)?;
        let embedding = Tensor::cat(&[&args.cos()?, &args.sin()?], 1)?.to_dtype(DType::BF16)?;

        let emb_3d = embedding.reshape(&[b, 1, 256])?;
        let temb = linear3d_bias(&emb_3d, self.w(&format!("{prefix}.linear_1.weight"))?,
            self.w(&format!("{prefix}.linear_1.bias"))?)?;
        let temb = temb.silu()?;
        let temb = linear3d_bias(&temb, self.w(&format!("{prefix}.linear_2.weight"))?,
            self.w(&format!("{prefix}.linear_2.bias"))?)?;

        let proj = linear3d_bias(&temb.silu()?,
            self.w(&format!("{prefix}.time_proj.weight"))?,
            self.w(&format!("{prefix}.time_proj.bias"))?)?;

        proj.reshape(&[b, 6, h])
    }

    fn compute_rope(&self, seq_len: usize, device: Arc<CudaDevice>)
        -> flame_core::Result<(Tensor, Tensor)>
    {
        let hd = self.config.head_dim;
        let half = hd / 2;
        let theta = self.config.rope_theta;

        let inv_freq: Vec<f32> = (0..half)
            .map(|i| 1.0 / theta.powf((2 * i) as f32 / hd as f32))
            .collect();
        let inv_freq_t = Tensor::from_vec(inv_freq, Shape::from_dims(&[1, half]), device.clone())?;

        let positions: Vec<f32> = (0..seq_len).map(|i| i as f32).collect();
        let pos_t = Tensor::from_vec(positions, Shape::from_dims(&[seq_len, 1]), device.clone())?;

        let angles = pos_t.matmul(&inv_freq_t)?;
        let cos = angles.cos()?.to_dtype(DType::BF16)?;
        let sin = angles.sin()?.to_dtype(DType::BF16)?;

        Ok((cos, sin))
    }

    fn dit_layer_forward(
        &self,
        hidden_states: &Tensor,
        timestep_proj: &Tensor,
        encoder_hidden_states: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let prefix = format!("decoder.layers.{layer_idx}");
        let h = self.config.hidden_size;
        let b = hidden_states.shape().dims()[0];

        let sst = self.w(&format!("{prefix}.scale_shift_table"))?;
        let modulation = sst.broadcast_to(&Shape::from_dims(&[b, 6, h]))?.add(timestep_proj)?;

        let shift_msa = modulation.narrow(1, 0, 1)?;
        let scale_msa = modulation.narrow(1, 1, 1)?;
        let gate_msa = modulation.narrow(1, 2, 1)?;
        let c_shift = modulation.narrow(1, 3, 1)?;
        let c_scale = modulation.narrow(1, 4, 1)?;
        let c_gate = modulation.narrow(1, 5, 1)?;

        // Self-attention with AdaLN
        let x_normed = rms_norm(
            hidden_states,
            self.w(&format!("{prefix}.self_attn_norm.weight"))?,
            self.config.rms_norm_eps,
        )?;
        let ones_s = Tensor::ones_dtype(scale_msa.shape().clone(), DType::BF16, hidden_states.device().clone())?;
        let norm_hs = x_normed.mul(&ones_s.add(&scale_msa)?)?.add(&shift_msa)?;

        let attn_out = self.self_attention_forward(&norm_hs, cos, sin, layer_idx)?;

        let x = hidden_states.add(&attn_out.mul(&gate_msa)?)?;

        // Cross-attention
        let cross_normed = rms_norm(
            &x,
            self.w(&format!("{prefix}.cross_attn_norm.weight"))?,
            self.config.rms_norm_eps,
        )?;
        let cross_out = self.cross_attention_forward(&cross_normed, encoder_hidden_states, layer_idx)?;
        let x = x.add(&cross_out)?;

        // MLP with AdaLN
        let mlp_normed = rms_norm(
            &x,
            self.w(&format!("{prefix}.mlp_norm.weight"))?,
            self.config.rms_norm_eps,
        )?;
        let ones_c = Tensor::ones_dtype(c_scale.shape().clone(), DType::BF16, x.device().clone())?;
        let mlp_in = mlp_normed.mul(&ones_c.add(&c_scale)?)?.add(&c_shift)?;

        let gate = linear3d(&mlp_in, self.w(&format!("{prefix}.mlp.gate_proj.weight"))?)?.silu()?;
        let up = linear3d(&mlp_in, self.w(&format!("{prefix}.mlp.up_proj.weight"))?)?;
        let mlp_out = linear3d(
            &gate.mul(&up)?,
            self.w(&format!("{prefix}.mlp.down_proj.weight"))?,
        )?;

        let x = x.add(&mlp_out.mul(&c_gate)?)?;

        Ok(x)
    }

    fn self_attention_forward(
        &self,
        hidden_states: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let prefix = format!("decoder.layers.{layer_idx}.self_attn");
        let adapter = &self.adapters[layer_idx];
        let b = hidden_states.shape().dims()[0];
        let s = hidden_states.shape().dims()[1];
        let nh = self.config.num_heads;
        let nkv = self.config.num_kv_heads;
        let hd = self.config.head_dim;

        let q = linear3d(hidden_states, self.w(&format!("{prefix}.q_proj.weight"))?)?
            .add(&adapter.self_q.forward_delta(hidden_states)?)?;
        let k = linear3d(hidden_states, self.w(&format!("{prefix}.k_proj.weight"))?)?
            .add(&adapter.self_k.forward_delta(hidden_states)?)?;
        let v = linear3d(hidden_states, self.w(&format!("{prefix}.v_proj.weight"))?)?
            .add(&adapter.self_v.forward_delta(hidden_states)?)?;

        let q = q.reshape(&[b, s, nh, hd])?;
        let k = k.reshape(&[b, s, nkv, hd])?;
        let v = v.reshape(&[b, s, nkv, hd])?;

        let q = rms_norm_per_head(&q, self.w(&format!("{prefix}.q_norm.weight"))?, self.config.rms_norm_eps)?;
        let k = rms_norm_per_head(&k, self.w(&format!("{prefix}.k_norm.weight"))?, self.config.rms_norm_eps)?;

        let q = q.transpose_dims(1, 2)?;
        let k = k.transpose_dims(1, 2)?;
        let v = v.transpose_dims(1, 2)?;

        let q = apply_rope(&q, cos, sin)?;
        let k = apply_rope(&k, cos, sin)?;

        let k = repeat_kv(&k, nh / nkv)?;
        let v = repeat_kv(&v, nh / nkv)?;

        let scale = (hd as f32).sqrt().recip();
        let attn = q.bmm(&k.transpose_dims(2, 3)?)?.mul_scalar(scale)?;
        let attn = attn.softmax(-1)?;
        let out = attn.bmm(&v)?;

        let out = out.transpose_dims(1, 2)?.reshape(&[b, s, nh * hd])?;

        let o_input = out;
        let out = linear3d(&o_input, self.w(&format!("{prefix}.o_proj.weight"))?)?
            .add(&adapter.self_o.forward_delta(&o_input)?)?;

        Ok(out)
    }

    fn cross_attention_forward(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: &Tensor,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let prefix = format!("decoder.layers.{layer_idx}.cross_attn");
        let adapter = &self.adapters[layer_idx];
        let b = hidden_states.shape().dims()[0];
        let s_q = hidden_states.shape().dims()[1];
        let s_kv = encoder_hidden_states.shape().dims()[1];
        let nh = self.config.num_heads;
        let nkv = self.config.num_kv_heads;
        let hd = self.config.head_dim;

        let q = linear3d(hidden_states, self.w(&format!("{prefix}.q_proj.weight"))?)?
            .add(&adapter.cross_q.forward_delta(hidden_states)?)?;
        let k = linear3d(encoder_hidden_states, self.w(&format!("{prefix}.k_proj.weight"))?)?
            .add(&adapter.cross_k.forward_delta(encoder_hidden_states)?)?;
        let v = linear3d(encoder_hidden_states, self.w(&format!("{prefix}.v_proj.weight"))?)?
            .add(&adapter.cross_v.forward_delta(encoder_hidden_states)?)?;

        let q = q.reshape(&[b, s_q, nh, hd])?;
        let k = k.reshape(&[b, s_kv, nkv, hd])?;
        let v = v.reshape(&[b, s_kv, nkv, hd])?;

        let q = rms_norm_per_head(&q, self.w(&format!("{prefix}.q_norm.weight"))?, self.config.rms_norm_eps)?;
        let k = rms_norm_per_head(&k, self.w(&format!("{prefix}.k_norm.weight"))?, self.config.rms_norm_eps)?;

        let q = q.transpose_dims(1, 2)?;
        let k = k.transpose_dims(1, 2)?;
        let v = v.transpose_dims(1, 2)?;

        // No RoPE for cross-attention.
        let k = repeat_kv(&k, nh / nkv)?;
        let v = repeat_kv(&v, nh / nkv)?;

        let scale = (hd as f32).sqrt().recip();
        let attn = q.bmm(&k.transpose_dims(2, 3)?)?.mul_scalar(scale)?;
        let attn = attn.softmax(-1)?;
        let out = attn.bmm(&v)?;

        let out = out.transpose_dims(1, 2)?.reshape(&[b, s_q, nh * hd])?;

        let o_input = out;
        let out = linear3d(&o_input, self.w(&format!("{prefix}.o_proj.weight"))?)?
            .add(&adapter.cross_o.forward_delta(&o_input)?)?;

        Ok(out)
    }
}

// ===========================================================================
// Helper functions (flame_core::Result — auto-converts to crate::Result via ?)
// ===========================================================================

fn linear3d(input: &Tensor, weight_t: &Tensor) -> flame_core::Result<Tensor> {
    input.matmul(weight_t)
}

fn linear3d_bias(input: &Tensor, weight_t: &Tensor, bias: &Tensor) -> flame_core::Result<Tensor> {
    let out = input.matmul(weight_t)?;
    let bias_expanded = bias.reshape(&[1, 1, bias.shape().elem_count()])?
        .broadcast_to(out.shape())?;
    out.add(&bias_expanded)
}

fn rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> flame_core::Result<Tensor> {
    let x_f32 = x.to_dtype(DType::F32)?;
    let sq = x_f32.mul(&x_f32)?;
    let last_dim = sq.shape().dims().len() - 1;
    let mean_sq = sq.mean_dim(&[last_dim], true)?;
    let rsqrt = mean_sq.add_scalar(eps)?.rsqrt()?;
    let normed = x_f32.mul(&rsqrt)?;
    let w = weight.to_dtype(DType::F32)?
        .reshape(&[1, 1, weight.shape().elem_count()])?;
    normed.mul(&w)?.to_dtype(DType::BF16)
}

fn rms_norm_per_head(x: &Tensor, weight: &Tensor, eps: f32) -> flame_core::Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    let (b, s, heads, hd) = (dims[0], dims[1], dims[2], dims[3]);
    let flat = x.reshape(&[b * s * heads, hd])?;
    let flat_f32 = flat.to_dtype(DType::F32)?;
    let sq = flat_f32.mul(&flat_f32)?;
    let last_dim = sq.shape().dims().len() - 1;
    let mean_sq = sq.mean_dim(&[last_dim], true)?;
    let rsqrt = mean_sq.add_scalar(eps)?.rsqrt()?;
    let normed = flat_f32.mul(&rsqrt)?;
    let w = weight.to_dtype(DType::F32)?
        .reshape(&[1, weight.shape().elem_count()])?;
    let result = normed.mul(&w)?.to_dtype(DType::BF16)?;
    result.reshape(&[b, s, heads, hd])
}

fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> flame_core::Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    let (_b, _heads, s, hd) = (dims[0], dims[1], dims[2], dims[3]);
    let half = hd / 2;

    let x1 = x.narrow(3, 0, half)?;
    let x2 = x.narrow(3, half, half)?;

    let cos_b = cos.narrow(0, 0, s)?.reshape(&[1, 1, s, half])?;
    let sin_b = sin.narrow(0, 0, s)?.reshape(&[1, 1, s, half])?;

    let r1 = x1.mul(&cos_b)?.sub(&x2.mul(&sin_b)?)?;
    let r2 = x1.mul(&sin_b)?.add(&x2.mul(&cos_b)?)?;

    Tensor::cat(&[&r1, &r2], 3)
}

fn repeat_kv(kv: &Tensor, n_rep: usize) -> flame_core::Result<Tensor> {
    if n_rep == 1 {
        return Ok(kv.clone());
    }
    let dims = kv.shape().dims().to_vec();
    let (b, kv_heads, s, hd) = (dims[0], dims[1], dims[2], dims[3]);
    let expanded = kv.reshape(&[b, kv_heads, 1, s, hd])?
        .broadcast_to(&Shape::from_dims(&[b, kv_heads, n_rep, s, hd]))?;
    expanded.reshape(&[b, kv_heads * n_rep, s, hd])
}

/// Conv1d forward as reshape+matmul (autograd-compatible).
/// For kernel_size == stride (non-overlapping patches), Conv1d is equivalent to:
///   reshape [B, T, C] -> [B, T/k, k*C] -> matmul + bias -> [B, T/k, C_out]
/// Weight: [C_out, C_in, kernel_size], Bias: [C_out]
fn conv1d_forward(
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    stride: usize,
) -> flame_core::Result<Tensor> {
    let dims = input.shape().dims().to_vec();
    let (b, t, c_in) = (dims[0], dims[1], dims[2]);
    let w_dims = weight.shape().dims().to_vec();
    let (c_out, _c_in_k, k) = (w_dims[0], w_dims[1], w_dims[2]);
    debug_assert_eq!(stride, k, "conv1d_forward requires stride == kernel_size");

    let t_out = t / k;
    let x = input.reshape(&[b, t_out, k * c_in])?;
    let w_2d = weight.reshape(&[c_out, k * c_in])?.transpose()?;
    let out = x.matmul(&w_2d)?;
    let bias_expanded = bias.reshape(&[1, 1, c_out])?
        .broadcast_to(out.shape())?;
    out.add(&bias_expanded)
}

/// ConvTranspose1d forward as matmul+reshape (autograd-compatible).
/// For kernel_size == stride (non-overlapping), ConvTranspose1d is:
///   matmul [B, T, C_in] @ W[C_in, k*C_out] -> [B, T, k*C_out] -> reshape -> [B, T*k, C_out]
/// Weight: [C_in, C_out, kernel_size], Bias: [C_out]
fn conv_transpose1d_forward(
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    stride: usize,
) -> flame_core::Result<Tensor> {
    let dims = input.shape().dims().to_vec();
    let (b, t, _c_in) = (dims[0], dims[1], dims[2]);
    let w_dims = weight.shape().dims().to_vec();
    let (c_in, c_out, k) = (w_dims[0], w_dims[1], w_dims[2]);
    debug_assert_eq!(stride, k, "conv_transpose1d_forward requires stride == kernel_size");

    let w_2d = weight.reshape(&[c_in, k * c_out])?;
    let out = input.matmul(&w_2d)?;
    let out = out.reshape(&[b, t * k, c_out])?;
    let bias_expanded = bias.reshape(&[1, 1, c_out])?
        .broadcast_to(out.shape())?;
    out.add(&bias_expanded)
}

#[allow(dead_code)]
fn _conv_transpose1d_old(
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    stride: usize,
) -> flame_core::Result<Tensor> {
    let x = input.transpose_dims(1, 2)?;
    let out = conv1d::conv_transpose1d(&x, weight, Some(bias), stride, 0, 0, 1)?;
    out.transpose_dims(1, 2)
}
