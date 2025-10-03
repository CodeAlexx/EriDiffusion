//! Hunyuan loader — specs derived from SimpleTuner (notes)
// Source: /home/alex/diffusers-rs/SimpleTuner/helpers/models/hunyuan/
// Record latent sizes, vision/text paths, LoRA mapping.
use eridiffusion_core::Result;
use crate::model_registry::{TrainCfg, ModelBundle};

pub fn load_hunyuan(_cfg: &TrainCfg) -> Result<ModelBundle> {
    Ok(ModelBundle { name: "hunyuan".into() })
}
