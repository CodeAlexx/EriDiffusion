//! SD3 loader — specs derived from SimpleTuner (notes)
// Source references:
// - /home/alex/diffusers-rs/SimpleTuner/helpers/models/sd3/
// Extract hybrid details (UNet/MMDiT as used), LoRA sites, text encoder dims, latent shapes.
use eridiffusion_core::Result;
use crate::model_registry::{TrainCfg, ModelBundle};

pub fn load_sd3(_cfg: &TrainCfg) -> Result<ModelBundle> {
    Ok(ModelBundle { name: "sd3".into() })
}
