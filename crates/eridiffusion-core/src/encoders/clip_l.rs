//! CLIP-L text encoder for FLUX 1 — pure flame-core implementation.
//!
//! Small model (12 layers, 768-dim, ~250MB) that produces:
//! - `last_hidden_state`: [B, 77, 768] — hidden states (NOT used by FLUX 1 directly)
//! - `pooled_output`: [B, 768] — CLS pooled (used as `vector` input to DiT)
//!
//! Architecture: standard CLIP text transformer.
//!   - Vocab: 49408, max_pos: 77, hidden: 768, heads: 12, layers: 12
//!   - Activation: quick_gelu (x * sigmoid(1.702 * x))
//!   - Causal attention mask
//!   - Layer norm (not RMSNorm)
//!   - Weight key prefix: none (keys like `text_model.encoder.layers.N...`)
//!
//! ⚠️ STANDALONE — does NOT connect to any inference pipeline.

use flame_core::{CudaDevice, DType, Result, Shape, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ClipConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub layer_norm_eps: f32,
    pub projection_dim: usize,
    /// CLIP-L default EOS token id (CLIPTokenizer also pads with this value).
    pub eos_token_id: i32,
    /// Tokenizer pad-token id. For CLIP-L this is the EOS (49407 — `<|endoftext|>`),
    /// but for CLIP-G the HF `tokenizer_2/tokenizer_config.json` sets
    /// `"pad_token": "!"` → id `0`. Fix per SDXL audit H1: prior code used
    /// `eos_token_id` as the pad fallback for both encoders, corrupting CLIP-G
    /// pad-position hidden states (silent — pooled is unaffected because EOS
    /// is the first match in argmax, but UNet cross-attn sees garbage at the
    /// 70+ pad positions of every short caption).
    /// Reference: HF `stabilityai/stable-diffusion-xl-base-1.0/tokenizer_2/tokenizer_config.json`.
    pub pad_token_id: i32,
    /// CLIP-L uses quick_gelu (x * sigmoid(1.702x)), CLIP-G uses standard gelu.
    pub use_quick_gelu: bool,
}

impl Default for ClipConfig {
    /// CLIP-L config (768-dim, 12 layers, quick_gelu).
    fn default() -> Self {
        Self {
            vocab_size: 49408,
            hidden_size: 768,
            num_layers: 12,
            intermediate_size: 3072,
            num_heads: 12,
            head_dim: 64, // 768 / 12
            max_position_embeddings: 77,
            layer_norm_eps: 1e-5,
            projection_dim: 768,
            eos_token_id: 49407, // HF CLIPTextConfig default
            pad_token_id: 49407, // CLIP-L pads with EOS per HF tokenizer config
            use_quick_gelu: true,
        }
    }
}

impl ClipConfig {
    /// CLIP-G config (1280-dim, 32 layers, standard gelu).
    /// OpenCLIP ViT-bigG-14, used as text_encoder_2 in SD3/SD3.5/SDXL.
    pub fn clip_g() -> Self {
        Self {
            vocab_size: 49408,
            hidden_size: 1280,
            num_layers: 32,
            intermediate_size: 5120,
            num_heads: 20,
            head_dim: 64, // 1280 / 20
            max_position_embeddings: 77,
            layer_norm_eps: 1e-5,
            projection_dim: 1280,
            eos_token_id: 49407,
            // SDXL audit H1: HF tokenizer_2 declares `"pad_token": "!"` → id 0.
            // CLIP-G's pad is NOT EOS — it has its own dedicated pad id.
            pad_token_id: 0,
            use_quick_gelu: false, // CLIP-G uses standard GELU
        }
    }
}

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

/// CLIP-L text encoder — loads all weights to GPU (small model, ~250MB).
pub struct ClipEncoder {
    weights: HashMap<String, Tensor>,
    config: ClipConfig,
    device: Arc<CudaDevice>,
}

impl ClipEncoder {
    pub fn new(
        weights: HashMap<String, Tensor>,
        config: ClipConfig,
        device: Arc<CudaDevice>,
    ) -> Self {
        Self { weights, config, device }
    }

    fn w(&self, key: &str) -> Result<&Tensor> {
        self.weights.get(key).ok_or_else(|| {
            flame_core::Error::InvalidInput(format!("Missing CLIP weight: {key}"))
        })
    }

    /// Linear: [B, N, C] x [out, in]^T -> [B, N, out], with optional bias.
    fn linear(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
        let wt = flame_core::bf16_elementwise::transpose2d_bf16(weight)?;
        let shape = x.shape().dims().to_vec();
        let (b, n, c) = (shape[0], shape[1], shape[2]);
        let x_2d = x.reshape(&[b * n, c])?;
        let mut out = x_2d.matmul(&wt)?;
        if let Some(bias) = bias {
            out = out.add(&bias.unsqueeze(0)?.expand(&out.shape().dims().to_vec())?)?;
        }
        let out_dim = out.shape().dims()[1];
        out.reshape(&[b, n, out_dim])
    }

    /// LayerNorm (NOT RMSNorm): normalize, scale, shift.
    /// ## PyTorch: F.layer_norm(x, (hidden_size,), weight, bias, eps)
    fn layer_norm(x: &Tensor, weight: &Tensor, bias: &Tensor, eps: f32) -> Result<Tensor> {
        let dims = x.shape().dims().to_vec();
        let hidden = *dims.last().unwrap();
        let batch: usize = dims[..dims.len() - 1].iter().product();
        let x_2d = x.reshape(&[batch, hidden])?;
        let out = flame_core::cuda_ops_bf16::layer_norm_bf16(
            &x_2d, Some(weight), Some(bias), eps,
        )?;
        out.reshape(&dims)
    }

    /// quick_gelu: x * sigmoid(1.702 * x)
    fn quick_gelu(x: &Tensor) -> Result<Tensor> {
        let scaled = x.mul_scalar(1.702)?;
        let sig = scaled.sigmoid()?;
        x.mul(&sig)
    }

    /// Build causal mask [1, 1, 77, 77] — standard CLIP causal mask.
    fn build_causal_mask(seq_len: usize, device: &Arc<CudaDevice>) -> Result<Tensor> {
        let mut data = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                if j <= i {
                    data[i * seq_len + j] = 1.0;
                }
            }
        }
        Tensor::from_vec(
            data,
            Shape::from_dims(&[1, 1, seq_len, seq_len]),
            device.clone(),
        )?.to_dtype(DType::BF16)
    }

    /// Combined causal + key-padding mask.
    ///
    /// `valid_key_end` is the index of the LAST valid key (inclusive). Tokens
    /// at positions `> valid_key_end` are pad and are masked out in the keys.
    /// Queries at any position can attend to keys `j` iff `j <= i` (causal)
    /// AND `j <= valid_key_end` (no pad keys). For SDXL CLIP-L/G with HF
    /// `attention_mask` semantics, `valid_key_end` is the index of the first
    /// EOS token (the "real" EOS, before the pad-EOS run).
    fn build_pad_mask(
        seq_len: usize,
        valid_key_end: usize,
        device: &Arc<CudaDevice>,
    ) -> Result<Tensor> {
        let mut data = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                if j <= i && j <= valid_key_end {
                    data[i * seq_len + j] = 1.0;
                }
            }
        }
        Tensor::from_vec(
            data,
            Shape::from_dims(&[1, 1, seq_len, seq_len]),
            device.clone(),
        )?
        .to_dtype(DType::BF16)
    }

    /// Single CLIP transformer layer.
    ///
    /// ## PyTorch reference:
    /// ```python
    /// # Self-attention
    /// residual = hidden_states
    /// hidden_states = self.layer_norm1(hidden_states)
    /// hidden_states = self.self_attn(hidden_states, causal_mask)
    /// hidden_states = residual + hidden_states
    /// # MLP
    /// residual = hidden_states
    /// hidden_states = self.layer_norm2(hidden_states)
    /// hidden_states = self.mlp(hidden_states)  # fc1 -> quick_gelu -> fc2
    /// hidden_states = residual + hidden_states
    /// ```
    fn layer_forward(
        &self,
        hidden: &Tensor,
        layer_idx: usize,
        mask: &Tensor,
    ) -> Result<Tensor> {
        let cfg = &self.config;
        let h = cfg.num_heads;
        let d = cfg.head_dim;
        let prefix = format!("text_model.encoder.layers.{layer_idx}");

        let dims = hidden.shape().dims().to_vec();
        let (b, n) = (dims[0], dims[1]);

        // --- Self-attention ---
        let ln1_w = self.w(&format!("{prefix}.layer_norm1.weight"))?;
        let ln1_b = self.w(&format!("{prefix}.layer_norm1.bias"))?;
        let normed = Self::layer_norm(hidden, ln1_w, ln1_b, cfg.layer_norm_eps)?;

        // Separate Q, K, V projections (CLIP uses separate, not fused QKV)
        let q_w = self.w(&format!("{prefix}.self_attn.q_proj.weight"))?;
        let q_b = self.w(&format!("{prefix}.self_attn.q_proj.bias"))?;
        let k_w = self.w(&format!("{prefix}.self_attn.k_proj.weight"))?;
        let k_b = self.w(&format!("{prefix}.self_attn.k_proj.bias"))?;
        let v_w = self.w(&format!("{prefix}.self_attn.v_proj.weight"))?;
        let v_b = self.w(&format!("{prefix}.self_attn.v_proj.bias"))?;

        let q = Self::linear(&normed, q_w, Some(q_b))?;
        let k = Self::linear(&normed, k_w, Some(k_b))?;
        let v = Self::linear(&normed, v_w, Some(v_b))?;

        let q = q.reshape(&[b, n, h, d])?.permute(&[0, 2, 1, 3])?;
        let k = k.reshape(&[b, n, h, d])?.permute(&[0, 2, 1, 3])?;
        let v = v.reshape(&[b, n, h, d])?.permute(&[0, 2, 1, 3])?;

        let attn_out = flame_core::attention::sdpa(&q, &k, &v, Some(mask))?;
        let attn_out = attn_out.permute(&[0, 2, 1, 3])?.reshape(&[b, n, h * d])?;

        let out_w = self.w(&format!("{prefix}.self_attn.out_proj.weight"))?;
        let out_b = self.w(&format!("{prefix}.self_attn.out_proj.bias"))?;
        let attn_out = Self::linear(&attn_out, out_w, Some(out_b))?;

        let hidden = hidden.add(&attn_out)?;

        // --- MLP ---
        let ln2_w = self.w(&format!("{prefix}.layer_norm2.weight"))?;
        let ln2_b = self.w(&format!("{prefix}.layer_norm2.bias"))?;
        let normed2 = Self::layer_norm(&hidden, ln2_w, ln2_b, cfg.layer_norm_eps)?;

        let fc1_w = self.w(&format!("{prefix}.mlp.fc1.weight"))?;
        let fc1_b = self.w(&format!("{prefix}.mlp.fc1.bias"))?;
        let fc2_w = self.w(&format!("{prefix}.mlp.fc2.weight"))?;
        let fc2_b = self.w(&format!("{prefix}.mlp.fc2.bias"))?;

        let mlp_hidden = Self::linear(&normed2, fc1_w, Some(fc1_b))?;
        let mlp_hidden = if cfg.use_quick_gelu {
            Self::quick_gelu(&mlp_hidden)?
        } else {
            mlp_hidden.gelu()?
        };
        let mlp_out = Self::linear(&mlp_hidden, fc2_w, Some(fc2_b))?;

        hidden.add(&mlp_out)
    }

    /// Encode text: returns (last_hidden_state, pooled_output).
    ///
    /// Matches BFL `HFEmbedder` behavior — reads `pooler_output` which is the final
    /// layer norm applied to the hidden state at the FIRST EOS token position.
    /// Pads/truncates `token_ids` to `max_position_embeddings` (77) to match
    /// `padding="max_length", max_length=77, truncation=True`.
    /// SDXL-style encode: combines causal mask with HF's `attention_mask`
    /// (key-padding mask) so pad-position outputs match HF's CLIPTextModel
    /// when the cross-attention context includes ALL 77 positions.
    ///
    /// Without the pad mask, pad-position queries attend to other pad keys
    /// and accumulate divergent context across layers — passes the pooled
    /// cos_sim ≥ 0.9999 (real EOS) but the full context is cos~0.5 vs HF.
    /// SDXL UNet attends to the full context, so this matters.
    ///
    /// Returns `(last_hidden_state [1, seq, dim], pooler_output [1, dim])`.
    pub fn encode_sdxl(&self, token_ids: &[i32]) -> Result<(Tensor, Tensor)> {
        let cfg = &self.config;
        let max_len = cfg.max_position_embeddings;

        let mut padded: Vec<i32> = token_ids.to_vec();
        if padded.len() > max_len {
            padded.truncate(max_len);
        } else {
            // SDXL audit H1: pad with the tokenizer's actual pad id (0 for CLIP-G,
            // 49407 for CLIP-L), NOT eos_token_id, to match HF tokenizer behaviour.
            padded.resize(max_len, cfg.pad_token_id);
        }
        let seq_len = max_len;

        // Find the real EOS — argmax over (id == eos_token_id) returns the
        // FIRST 1, matching HF's pooler-output gather. EOS is not the pad
        // value, so the first hit is always the real EOS.
        let real_eos = padded
            .iter()
            .position(|&id| id == cfg.eos_token_id)
            .unwrap_or(seq_len - 1);

        let token_w = self.w("text_model.embeddings.token_embedding.weight")?;
        let pos_w = self.w("text_model.embeddings.position_embedding.weight")?;

        let ids = Tensor::from_vec(
            padded.iter().map(|&id| id as f32).collect(),
            Shape::from_dims(&[seq_len]),
            self.device.clone(),
        )?
        .to_dtype(DType::I32)?;
        let token_embeds = token_w.index_select0(&ids)?;

        let pos_ids = Tensor::from_vec(
            (0..seq_len as i32).map(|i| i as f32).collect(),
            Shape::from_dims(&[seq_len]),
            self.device.clone(),
        )?
        .to_dtype(DType::I32)?;
        let pos_embeds = pos_w.index_select0(&pos_ids)?;

        let mut hidden = token_embeds.add(&pos_embeds)?.unsqueeze(0)?;

        // Combined causal + key-padding mask (pad keys are tokens beyond the
        // real EOS).
        let mask = Self::build_pad_mask(seq_len, real_eos, &self.device)?;

        for i in 0..cfg.num_layers {
            hidden = self.layer_forward(&hidden, i, &mask)?;
        }

        let final_ln_w = self.w("text_model.final_layer_norm.weight")?;
        let final_ln_b = self.w("text_model.final_layer_norm.bias")?;
        let hidden = Self::layer_norm(&hidden, final_ln_w, final_ln_b, cfg.layer_norm_eps)?;
        let pooled = hidden.narrow(1, real_eos, 1)?.squeeze(Some(1))?;

        Ok((hidden, pooled))
    }

    pub fn encode(&self, token_ids: &[i32]) -> Result<(Tensor, Tensor)> {
        let cfg = &self.config;
        let max_len = cfg.max_position_embeddings; // 77

        // CLIP.H1 fix: always run at seq_len=77. CLIPTokenizer pads with eos_token_id.
        let mut padded: Vec<i32> = token_ids.to_vec();
        if padded.len() > max_len {
            padded.truncate(max_len);
        } else {
            padded.resize(max_len, cfg.eos_token_id);
        }
        let seq_len = max_len;

        // 1. Token + position embeddings
        let token_w = self.w("text_model.embeddings.token_embedding.weight")?;
        let pos_w = self.w("text_model.embeddings.position_embedding.weight")?;

        let ids = Tensor::from_vec(
            padded.iter().map(|&id| id as f32).collect(),
            Shape::from_dims(&[seq_len]),
            self.device.clone(),
        )?.to_dtype(DType::I32)?;

        let token_embeds = token_w.index_select0(&ids)?;

        // Position IDs: 0..seq_len
        let pos_ids = Tensor::from_vec(
            (0..seq_len as i32).map(|i| i as f32).collect(),
            Shape::from_dims(&[seq_len]),
            self.device.clone(),
        )?.to_dtype(DType::I32)?;
        let pos_embeds = pos_w.index_select0(&pos_ids)?;

        let mut hidden = token_embeds.add(&pos_embeds)?.unsqueeze(0)?; // [1, seq, 768]

        // 2. Causal mask
        let mask = Self::build_causal_mask(seq_len, &self.device)?;

        // 3. Transformer layers
        for i in 0..cfg.num_layers {
            hidden = self.layer_forward(&hidden, i, &mask)?;
        }

        // 4. Final layer norm
        let final_ln_w = self.w("text_model.final_layer_norm.weight")?;
        let final_ln_b = self.w("text_model.final_layer_norm.bias")?;
        let hidden = Self::layer_norm(&hidden, final_ln_w, final_ln_b, cfg.layer_norm_eps)?;

        // 5. Pooled output: hidden state at the FIRST EOS token position.
        //
        // ## HF reference (modeling_clip.py CLIPTextTransformer.forward):
        // ```python
        // pooled_output = last_hidden_state[
        //     torch.arange(last_hidden_state.shape[0], device=...),
        //     (input_ids.to(dtype=torch.int) == self.eos_token_id).int().argmax(dim=-1),
        // ]
        // ```
        // `argmax` on {0,1} returns the FIRST index of the max. CLIPTokenizer pads
        // with eos_token_id=49407, so this finds the real EOS, not a pad slot.
        //
        // CLIP.C1 fix: previously used `max_by_key` which returns the LAST tied
        // element, landing on the final pad position instead of the real EOS.
        let eos_pos = padded
            .iter()
            .position(|&id| id == cfg.eos_token_id)
            .unwrap_or(seq_len - 1);

        let pooled = hidden.narrow(1, eos_pos, 1)?.squeeze(Some(1))?; // [1, 768]

        Ok((hidden, pooled))
    }

    /// Encode for SD3: returns (penultimate_hidden_state, pooled_output).
    ///
    /// SD3 uses `hidden_states[-2]` (output after the second-to-last transformer
    /// layer, NOT final-layer-normed) for the sequence embedding, and
    /// `pooler_output` (final layer output → final_layer_norm → EOS position)
    /// for the pooled vector.
    ///
    /// ## HF reference (modeling_clip.py):
    /// ```python
    /// # hidden_states[-2] = encoder output after layer (num_layers - 2)
    /// clip_out = clip(tokens).hidden_states[-2]       # [B, 77, hidden]
    /// clip_pooled = clip(tokens).pooler_output         # [B, hidden]
    /// ```
    pub fn encode_sd3(&self, token_ids: &[i32]) -> Result<(Tensor, Tensor)> {
        let cfg = &self.config;
        let max_len = cfg.max_position_embeddings;

        let mut padded: Vec<i32> = token_ids.to_vec();
        if padded.len() > max_len {
            padded.truncate(max_len);
        } else {
            // SDXL audit H1: pad with `pad_token_id` (CLIP-L: 49407, CLIP-G: 0)
            // — matches HF tokenizer behaviour. Using eos for both encoders
            // silently corrupted CLIP-G pad-position hidden states (which
            // SDXL UNet's cross-attn reads).
            padded.resize(max_len, cfg.pad_token_id);
        }
        let seq_len = max_len;

        // 1. Token + position embeddings
        let token_w = self.w("text_model.embeddings.token_embedding.weight")?;
        let pos_w = self.w("text_model.embeddings.position_embedding.weight")?;

        let ids = Tensor::from_vec(
            padded.iter().map(|&id| id as f32).collect(),
            Shape::from_dims(&[seq_len]),
            self.device.clone(),
        )?.to_dtype(DType::I32)?;
        let token_embeds = token_w.index_select0(&ids)?;

        let pos_ids = Tensor::from_vec(
            (0..seq_len as i32).map(|i| i as f32).collect(),
            Shape::from_dims(&[seq_len]),
            self.device.clone(),
        )?.to_dtype(DType::I32)?;
        let pos_embeds = pos_w.index_select0(&pos_ids)?;

        let mut hidden = token_embeds.add(&pos_embeds)?.unsqueeze(0)?;

        // 2. Causal mask
        let mask = Self::build_causal_mask(seq_len, &self.device)?;

        // 3. Transformer layers — capture penultimate output
        let penultimate_idx = cfg.num_layers - 2; // hidden_states[-2]
        let mut penultimate_hidden = None;

        for i in 0..cfg.num_layers {
            hidden = self.layer_forward(&hidden, i, &mask)?;
            if i == penultimate_idx {
                penultimate_hidden = Some(hidden.clone());
            }
        }

        // 4. Pooled output: final_layer_norm on LAST layer → EOS position
        let final_ln_w = self.w("text_model.final_layer_norm.weight")?;
        let final_ln_b = self.w("text_model.final_layer_norm.bias")?;
        let normed = Self::layer_norm(&hidden, final_ln_w, final_ln_b, cfg.layer_norm_eps)?;

        let eos_pos = padded
            .iter()
            .position(|&id| id == cfg.eos_token_id)
            .unwrap_or(seq_len - 1);
        let pooled_pre_proj = normed.narrow(1, eos_pos, 1)?.squeeze(Some(1))?;

        // SD3 / SDXL: CLIP-G applies `text_projection` to pooler_output to
        // produce `text_embeds`. CLIP-L doesn't ship a text_projection in
        // the SD3 CLIP-L safetensors, so we fall back to raw pooler_output.
        // Without this, CLIP-G pooled has wrong scale → y_emb has outliers
        // (e.g. max_abs ~32) → adaLN modulation explodes → SD3.5 Large hidden
        // state blows up in block 0 → uniform-color output.
        let pooled = if let Ok(proj_w) = self.w("text_projection.weight") {
            let x3 = pooled_pre_proj.unsqueeze(1)?; // [1, 1, C]
            let y = Self::linear(&x3, proj_w, None)?; // [1, 1, C_out]
            y.squeeze(Some(1))? // [1, C_out]
        } else {
            pooled_pre_proj
        };

        Ok((penultimate_hidden.unwrap(), pooled))
    }

    /// Encode for Stable Cascade: returns (last_hidden_state, pooled).
    ///
    /// Stable Cascade uses CLIP-ViT-bigG-14 with the `text_projection` matrix.
    /// Output:
    /// - `hidden_states`: [1, seq_len, 1280] — post-final-LN last hidden state
    /// - `pooled`: [1, 1280] — `text_projection.T @ hidden[EOS]`
    ///
    /// ## HF reference (modeling_clip.py CLIPTextModelWithProjection):
    /// ```python
    /// text_outputs = self.text_model(input_ids)  # last_hidden_state + pooler_output
    /// text_embeds = self.text_projection(text_outputs.pooler_output)
    /// ```
    ///
    /// `pooler_output` is: final_layer_norm → gather EOS position
    pub fn encode_cascade(&self, token_ids: &[i32]) -> Result<(Tensor, Tensor)> {
        let cfg = &self.config;
        let max_len = cfg.max_position_embeddings;

        let mut padded: Vec<i32> = token_ids.to_vec();
        if padded.len() > max_len {
            padded.truncate(max_len);
        } else {
            padded.resize(max_len, cfg.eos_token_id);
        }
        let seq_len = max_len;

        // 1. Token + position embeddings
        let token_w = self.w("text_model.embeddings.token_embedding.weight")?;
        let pos_w = self.w("text_model.embeddings.position_embedding.weight")?;

        let ids = Tensor::from_vec(
            padded.iter().map(|&id| id as f32).collect(),
            Shape::from_dims(&[seq_len]),
            self.device.clone(),
        )?.to_dtype(DType::I32)?;
        let token_embeds = token_w.index_select0(&ids)?;

        let pos_ids = Tensor::from_vec(
            (0..seq_len as i32).map(|i| i as f32).collect(),
            Shape::from_dims(&[seq_len]),
            self.device.clone(),
        )?.to_dtype(DType::I32)?;
        let pos_embeds = pos_w.index_select0(&pos_ids)?;

        let mut hidden = token_embeds.add(&pos_embeds)?.unsqueeze(0)?; // [1, seq, C]

        // 2. Causal mask
        let mask = Self::build_causal_mask(seq_len, &self.device)?;

        // 3. Transformer layers
        for i in 0..cfg.num_layers {
            hidden = self.layer_forward(&hidden, i, &mask)?;
        }

        // Stable Cascade prior pipeline uses `hidden_states[-1]` BEFORE
        // `final_layer_norm` for `clip_text`. (HF CLIP decorates the text
        // transformer with `@capture_outputs(tie_last_hidden_states=False)`
        // so `hidden_states[-1]` = last encoder layer output, NOT the LN'd
        // `last_hidden_state`.) That's the "pre-LN" tensor we return below.
        let pre_ln_hidden = hidden.clone();

        // 4. Final layer norm on the full last-hidden-state — needed only
        //    to compute `pooler_output`. The prior UNet receives the pre-LN
        //    `clip_text` and the post-LN+projected pool separately.
        let final_ln_w = self.w("text_model.final_layer_norm.weight")?;
        let final_ln_b = self.w("text_model.final_layer_norm.bias")?;
        let last_hidden_post_ln = Self::layer_norm(&hidden, final_ln_w, final_ln_b, cfg.layer_norm_eps)?;

        // 5. Pooled: gather EOS position, then project.
        let eos_pos = padded
            .iter()
            .position(|&id| id == cfg.eos_token_id)
            .unwrap_or(seq_len - 1);
        // [1, 1, C] -> [1, C]
        let pooled_pre_proj = last_hidden_post_ln.narrow(1, eos_pos, 1)?.squeeze(Some(1))?;

        // text_projection: [1280, 1280]; apply as pooled_pre @ text_projection.T
        // PyTorch nn.Linear: y = x @ W.T, weight shape [out, in] -> we apply via linear().
        // But note this is stored as a plain weight (no bias in CLIPTextModelWithProjection).
        // Key may or may not exist; fall back to the pooled_pre if absent.
        let pooled = if let Ok(proj_w) = self.w("text_projection.weight") {
            // x [1, C_in]; W [C_out, C_in] -> need to reshape pooled_pre to [1, 1, C] for Self::linear.
            let x3 = pooled_pre_proj.unsqueeze(1)?; // [1, 1, C]
            let y = Self::linear(&x3, proj_w, None)?; // [1, 1, C_out]
            y.squeeze(Some(1))? // [1, C_out]
        } else {
            pooled_pre_proj
        };

        Ok((pre_ln_hidden, pooled))
    }

    pub fn config(&self) -> &ClipConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clip_config() {
        let cfg = ClipConfig::default();
        assert_eq!(cfg.hidden_size, 768);
        assert_eq!(cfg.num_layers, 12);
        assert_eq!(cfg.num_heads * cfg.head_dim, 768);
        assert!(cfg.use_quick_gelu);

        let cfg_g = ClipConfig::clip_g();
        assert_eq!(cfg_g.hidden_size, 1280);
        assert_eq!(cfg_g.num_layers, 32);
        assert_eq!(cfg_g.num_heads, 20);
        assert_eq!(cfg_g.num_heads * cfg_g.head_dim, 1280);
        assert!(!cfg_g.use_quick_gelu);
    }
}
