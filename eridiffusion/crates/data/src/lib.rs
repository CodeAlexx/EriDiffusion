//! Data types used by training pipelines (Flame-only)

use std::collections::HashMap;
use eridiffusion_core::Result;
use flame_core::Tensor;

pub mod image_dataset;
pub mod vae_preprocessor;
pub mod wan22_image_dataset;
pub mod wan22_video_dataset;

pub mod types {
    use super::*;
    use serde_json::Value;

    /// Minimal batch used by training pipelines
    #[derive(Clone, Debug)]
    pub struct DataLoaderBatch {
        pub images: Tensor,                  // images or latents (BCHW or NHWC)
        pub captions: Vec<String>,           // per-sample captions
        pub attention_mask: Option<Tensor>,  // optional mask
        pub loss_weights: Option<Tensor>,    // optional per-sample weights
        pub metadata: HashMap<String, Value>,
    }

    impl DataLoaderBatch {
        pub fn new(
            images: Tensor,
            captions: Vec<String>,
            attention_mask: Option<Tensor>,
            loss_weights: Option<Tensor>,
            metadata: HashMap<String, Value>,
        ) -> Self {
            Self { images, captions, attention_mask, loss_weights, metadata }
        }
    }

    /// Latent batch used by training code
    #[derive(Clone, Debug)]
    pub struct LatentBatch {
        pub latents: Tensor,
        pub text_embeds: Option<Tensor>,
        pub pooled_embeds: Option<Tensor>,
        pub captions: Vec<String>,
        pub loss_weights: Tensor,
        pub metadata: HashMap<String, Value>,
    }
}

pub use types::{DataLoaderBatch, LatentBatch};

pub mod latent {
    use super::*;
    use eridiffusion_models::{VAE, TextEncoder};

    /// Build a LatentBatch from a DataLoaderBatch using provided VAE and optional text encoder.
    pub async fn batch_to_latents(
        batch: types::DataLoaderBatch,
        vae: &dyn VAE,
        text_encoder: Option<&dyn TextEncoder>,
    ) -> Result<types::LatentBatch> {
        let images = &batch.images;

        // Heuristic: if channels == 3 treat as RGB and encode, else assume already latents
        let latents: Tensor = match images.shape().dims() {
            [_, c, _, _] if *c == 3 => vae.encode(images)?,
            [_, _, _, c] if *c == 3 => vae.encode(&images.permute(&[0, 3, 1, 2])?)?,
            _ => images.clone(),
        };

        // Text embeddings (optional)
        let (text_embeds, pooled_embeds) = if let Some(te) = text_encoder {
            let (emb, pooled) = te.encode(&batch.captions)?;
            (Some(emb), pooled)
        } else {
            (None, None)
        };

        // Loss weights: default ones if not provided
        let loss_weights: Tensor = if let Some(w) = &batch.loss_weights { w.clone() } else {
            // ones per-sample (B)
            let b = latents.shape().dims()[0];
            Tensor::ones(flame_core::Shape::from_dims(&[b]), latents.device().clone())?
        };

        Ok(types::LatentBatch {
            latents,
            text_embeds,
            pooled_embeds,
            captions: batch.captions,
            loss_weights,
            metadata: batch.metadata,
        })
    }
}

// Re-export helper for training code
pub use latent::batch_to_latents;
pub use vae_preprocessor::VAENormalizer;
