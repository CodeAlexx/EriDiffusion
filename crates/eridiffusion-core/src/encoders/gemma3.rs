//! Gemma-3 12B text encoder (LTX-2 caption path).
//!
//! Status: **STUB / placeholder**. Per LTX2_PORT_AUDIT_AITOOLKIT.md §5 row
//! "Gemma3-12B text encoder", a full Rust port is ~1500 LoC and
//! the audit explicitly recommends NOT porting it for the trainer:
//!
//!   "the text encoder doesn't need a Rust port for LoRA training,
//!    only for inference."  — LTX2_PORT_AUDIT_MUSUBI.md §5.3
//!
//! Strategy:
//!   - Trainer consumes pre-computed Gemma3 embeddings from disk
//!     (cached by an external Python pass; or by `prepare_ltx2`
//!     using its image-as-frame bootstrap with placeholder embeds).
//!   - Sampler ships with the same loader contract so users can swap
//!     in real Gemma3 embeddings produced by upstream Python.
//!
//! This module declares the *interface* (`Gemma3Encoder::load`,
//! `Gemma3Encoder::encode`) so callers can compile against a stable
//! API. The forward path returns zeros of the correct shape with a
//! `log::warn!` so a downstream NaN can be traced to "real Gemma3 not
//! plumbed in yet" rather than to model math.
//!
//! ## Architecture (for the future port)
//! - `Gemma3ForConditionalGeneration` (multimodal, vision tower stripped).
//! - hidden_size=3840, num_hidden_layers=48, num_attention_heads=16,
//!   num_key_value_heads=8 (GQA), head_dim=256, intermediate_size=15360.
//! - hidden_activation=gelu_pytorch_tanh, rms_norm_eps=1e-6.
//! - rope_theta=1000000 (global) + rope_local_base_freq=10000 (sliding).
//! - sliding_window=1024, sliding_window_pattern=6 (every 6th = global).
//! - rope_scaling = {"factor": 8.0, "rope_type": "linear"}.
//! - Output: stack(all_49_hidden_states, dim=-1).flatten(2,3) →
//!   shape `[B, 1024, 3840 * 49]`.
//! - Tokenizer: GemmaTokenizerFast, **left padding**, max length 1024.

use cudarc::driver::CudaDevice;
use flame_core::{DType, Shape, Tensor};
use std::sync::Arc;

use crate::Result;

/// Hidden dim of a single Gemma-3-12B layer's hidden_states.
pub const GEMMA3_HIDDEN: usize = 3840;
/// Number of layers whose hidden states are stacked into the LTX-2 caption.
pub const GEMMA3_NUM_LAYERS_FOR_LTX: usize = 49;
/// Combined caption channel count after layer-stacking.
/// LTX-2 connector consumes this; the DiT's `caption_channels` config = 3840
/// is the *per-layer* channels (they get reshaped to 3840 via the connector).
pub const GEMMA3_CAPTION_CHANNELS: usize = GEMMA3_HIDDEN * GEMMA3_NUM_LAYERS_FOR_LTX;
/// LTX-2 connector requires fixed prompt length of 1024 tokens.
pub const GEMMA3_PROMPT_LEN: usize = 1024;

/// Stub Gemma-3 encoder. Loads checkpoint footprint without inhabiting
/// it; `encode()` returns zeros at the correct shape. Replace with a
/// real port (see module docs) before training real T2V LoRA.
pub struct Gemma3Encoder {
    pub device: Arc<CudaDevice>,
    /// Whether weights were actually loaded. False ⇒ `encode` warns.
    pub weights_loaded: bool,
}

impl Gemma3Encoder {
    /// Load a Gemma-3 checkpoint from a directory of safetensors shards.
    /// **Currently a no-op.** Real implementation will populate per-layer
    /// weights (W_q, W_k, W_v, W_o, W_gate, W_up, W_down, norm tensors,
    /// embedding table, output norm).
    pub fn load(_ckpt_dir: &std::path::Path, device: Arc<CudaDevice>) -> Result<Self> {
        log::warn!(
            "Gemma3Encoder::load — STUB. Real Gemma-3 forward not yet ported. \
             encode() will return zeros. Train against pre-cached embeddings instead."
        );
        Ok(Self { device, weights_loaded: false })
    }

    /// Encode a single (already-tokenized, already-left-padded) input id sequence.
    ///
    /// Returns a `[1, GEMMA3_PROMPT_LEN, GEMMA3_HIDDEN]` BF16 tensor.
    /// (NOT the layer-stacked `GEMMA3_CAPTION_CHANNELS` form — the LTX-2
    /// connector projects from per-layer 3840-d to whatever the DiT consumes;
    /// per-layer is what `caption_channels=3840` in the DiT config wants,
    /// matching the audit's "already-projected" LTX-2.3 path.)
    pub fn encode(&self, _input_ids: &[i32]) -> Result<Tensor> {
        if !self.weights_loaded {
            log::warn!(
                "Gemma3Encoder::encode — returning ZEROS placeholder. \
                 Train output will be meaningless until a real Gemma-3 \
                 forward is plumbed or pre-cached embeddings are loaded."
            );
        }
        let shape = Shape::from_dims(&[1, GEMMA3_PROMPT_LEN, GEMMA3_HIDDEN]);
        let zeros = Tensor::zeros_dtype(shape, DType::BF16, self.device.clone())?;
        Ok(zeros)
    }
}
