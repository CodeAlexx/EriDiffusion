//! LTX loader — specs derived from SimpleTuner (notes)
// Source: /home/alex/diffusers-rs/SimpleTuner/helpers/models/ltx/
// Confirm text encoder dims and LoRA sites.
use eridiffusion_core::Result;
use crate::model_registry::{TrainCfg, ModelBundle};

pub fn load_ltx(_cfg: &TrainCfg) -> Result<ModelBundle> {
    Ok(ModelBundle { name: "ltx".into() })
}
