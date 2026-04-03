//! HiDream-I1 loader — specs derived from SimpleTuner (notes)
// Source: /home/alex/diffusers-rs/SimpleTuner/helpers/models/hidream/
// Identify text/image paths, latent dims, LoRA attach points.
use eridiffusion_core::Result;
use crate::model_registry::{TrainCfg, ModelBundle};

pub fn load_hidream(_cfg: &TrainCfg) -> Result<ModelBundle> {
    Ok(ModelBundle { name: "hidream_i1".into() })
}
