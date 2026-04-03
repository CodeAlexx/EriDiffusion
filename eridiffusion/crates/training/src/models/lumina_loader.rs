//! Lumina loader — specs derived from SimpleTuner (notes)
// Source: /home/alex/diffusers-rs/SimpleTuner/helpers/models/lumina/
// Capture dims and LoRA sites.
use eridiffusion_core::Result;
use crate::model_registry::{TrainCfg, ModelBundle};

pub fn load_lumina(_cfg: &TrainCfg) -> Result<ModelBundle> {
    Ok(ModelBundle { name: "lumina".into() })
}
