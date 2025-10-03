//! Wan 2.2 loader — specs derived from SimpleTuner (notes)
// Source: /home/alex/diffusers-rs/SimpleTuner/helpers/models/wan/
// Confirm expected latent sizes, attention block structure, LoRA sites.
use eridiffusion_core::Result;
use crate::model_registry::{TrainCfg, ModelBundle};

pub fn load_wan22(_cfg: &TrainCfg) -> Result<ModelBundle> {
    Ok(ModelBundle { name: "wan22".into() })
}
