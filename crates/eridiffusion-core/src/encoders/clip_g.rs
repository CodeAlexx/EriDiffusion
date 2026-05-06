//! CLIP-G/14 (OpenCLIP ViT-bigG-14) text encoder for SDXL.
//!
//! Architecture: same CLIPTextTransformer shape as CLIP-L but bigger:
//!   - hidden=1280, layers=32, heads=20, head_dim=64 (vs CLIP-L 768/12/12/64)
//!   - intermediate=5120, max_pos=77, vocab=49408
//!   - **standard GELU** (NOT quick_gelu, unlike CLIP-L)
//!   - has `text_projection.weight` (CLIP-L typically does not)
//!
//! For SDXL we need:
//!   - `hidden_states[-2]`: penultimate layer output, NOT final-LN, used as the
//!     1280-dim slice of the dual-encoder context (concat with CLIP-L 768 → 2048).
//!   - `pooled_output`: text_projection @ final_LN(hidden[-1])[EOS] → [B, 1280],
//!     used as part of the SDXL `add_text_embeds` ADM input.
//!
//! Both behaviors live in the shared `ClipEncoder::encode_sd3` (CLIP-G branch
//! for SDXL is structurally identical: penultimate hidden + projected pool).
//! This module just bundles the right config + a clearer constructor name.
//!
//! ⚠️ STANDALONE — does NOT mutate inference pipelines; consumed by SDXL prepare/sample bins.

use std::collections::HashMap;
use std::sync::Arc;

use flame_core::{CudaDevice, Result, Tensor};

use crate::encoders::clip_l::{ClipConfig, ClipEncoder};

/// CLIP-G text encoder for SDXL — wraps `ClipEncoder` with the bigG-14 config.
pub struct ClipGEncoder {
    inner: ClipEncoder,
}

impl ClipGEncoder {
    /// Construct from a pre-loaded weight HashMap (HF `text_encoder_2/`-style keys).
    pub fn new(weights: HashMap<String, Tensor>, device: Arc<CudaDevice>) -> Self {
        Self { inner: ClipEncoder::new(weights, ClipConfig::clip_g(), device) }
    }

    /// SDXL convention: returns `(penultimate_hidden [1, 77, 1280], pooled [1, 1280])`.
    ///
    /// `penultimate_hidden` = hidden_states[-2] (no final LN), used as the
    /// 1280-d slice of the dual-encoder context.
    /// `pooled` = text_projection(final_LN(hidden[-1])[EOS]), used as the
    /// 1280-d slice of `add_text_embeds`.
    ///
    /// Implementation note: this matches `encode_sd3`, which already returns
    /// `(penultimate_pre_ln, projected_pooled)` — the SDXL CLIP-G output is
    /// structurally identical to SD3's CLIP-G output.
    pub fn encode_sdxl(&self, token_ids: &[i32]) -> Result<(Tensor, Tensor)> {
        self.inner.encode_sd3(token_ids)
    }

    /// Hidden size (1280 for CLIP-G).
    pub fn hidden_size(&self) -> usize {
        self.inner.config().hidden_size
    }

    /// Tokenizer pad token id used by this encoder. SDXL audit H1: callers
    /// MUST pad CLIP-G token streams with id `0` (the HF `tokenizer_2`
    /// `"pad_token": "!"`), not `49407` (CLIP-L's EOS).
    pub fn pad_token_id(&self) -> i32 {
        self.inner.config().pad_token_id
    }
}
