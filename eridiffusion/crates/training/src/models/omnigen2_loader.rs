//! Omnigen2 loader — specs derived from SimpleTuner (notes)
// Source: /home/alex/diffusers-rs/SimpleTuner/helpers/models/omnigen2/
// Capture latent shapes, text requirements, LoRA mapping.
use eridiffusion_core::Result;
use crate::model_registry::{TrainCfg, ModelBundle};

pub fn load_omnigen2(_cfg: &TrainCfg) -> Result<ModelBundle> {
    Ok(ModelBundle { name: "omnigen2".into() })
}
