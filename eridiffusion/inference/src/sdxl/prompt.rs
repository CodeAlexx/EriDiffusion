use anyhow::{Context, Result};
use eridiffusion_common_text::{clip_l::ClipL, openclip_g::OpenClipG, HfTokenizer};
use flame_core::{Device as FlameDevice, Tensor};

use crate::sdxl::config::SdxlPaths;

/// Encoded prompt outputs used for classifier-free guidance.
pub struct PromptEmbeddings {
    pub context: Tensor,
    pub pooled: Tensor,
}

/// SDXL text encoder bundle (CLIP-L + OpenCLIP-G + tokenizer).
pub struct PromptEncoder {
    tokenizer: HfTokenizer,
    clip_l: ClipL,
    clip_g: OpenClipG,
    device: FlameDevice,
    seq_len: usize,
}

impl PromptEncoder {
    /// Load tokenizer and CLIP encoders on the requested CUDA device.
    pub fn load(paths: &SdxlPaths, seq_len: usize, device_index: usize) -> Result<Self> {
        let tokenizer_path =
            paths.tokenizer.to_str().context("tokenizer path contains invalid unicode")?;
        let clip_l_path = paths.clip_l.to_str().context("clip-l path contains invalid unicode")?;
        let clip_g_path = paths.clip_g.to_str().context("clip-g path contains invalid unicode")?;

        let device = FlameDevice::cuda(device_index)?;
        let tokenizer = HfTokenizer::from_path(tokenizer_path, seq_len)?;
        let clip_l = ClipL::from_weights_auto(clip_l_path, &device, seq_len)?;
        let clip_g = OpenClipG::from_weights_auto(clip_g_path, &device, seq_len)?;

        Ok(Self { tokenizer, clip_l, clip_g, device, seq_len })
    }

    /// Encode a batch of prompts, returning concatenated context and pooled embeddings.
    pub fn encode(&self, prompts: &[String]) -> Result<PromptEmbeddings> {
        if prompts.is_empty() {
            anyhow::bail!("encode requires at least one prompt");
        }
        let (ids, _lengths, _pad) = self.tokenizer.encode_batch_on(prompts, &self.device)?;
        let clip_l_ctx = self.clip_l.forward(&ids)?; // [B,seq,dim_l]
        let clip_g_ctx = self.clip_g.forward(&ids)?; // [B,seq,dim_g]
        let pooled = self.clip_g.pooled(&clip_g_ctx)?; // [B,dim_g]
        let context = Tensor::cat(&[&clip_l_ctx, &clip_g_ctx], 2)?; // [B,seq,dim_l+dim_g]
        Ok(PromptEmbeddings { context, pooled })
    }

    /// Convenience helper for single prompt encoding.
    pub fn encode_single(&self, prompt: &str) -> Result<PromptEmbeddings> {
        self.encode(&[prompt.to_owned()])
    }

    pub fn encode_pair(
        &self,
        prompt: &str,
        negative_prompt: &str,
    ) -> Result<(PromptEmbeddings, PromptEmbeddings)> {
        let cond = self.encode(&[prompt.to_owned()])?;
        let uncond = self.encode(&[negative_prompt.to_owned()])?;
        Ok((cond, uncond))
    }

    /// Provide access to the underlying CUDA device for downstream modules.
    pub fn device(&self) -> &FlameDevice {
        &self.device
    }

    pub fn device_clone(&self) -> FlameDevice {
        self.device.clone()
    }

    pub fn seq_len(&self) -> usize {
        self.seq_len
    }
}
