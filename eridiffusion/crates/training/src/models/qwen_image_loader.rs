//! Qwen-Image v1 loader — specs derived from SimpleTuner (notes)
// Source: /home/alex/diffusers-rs/SimpleTuner/helpers/models/qwen_image/
// Expect vision transformer backbone + tokenizer/encoder; accept pre-encoded embeddings path.
use eridiffusion_core::Result;
use crate::model_registry::{TrainCfg, ModelBundle};

pub fn load_qwen_image(_cfg: &TrainCfg) -> Result<ModelBundle> {
    Ok(ModelBundle { name: "qwen_image_v1".into() })
}
