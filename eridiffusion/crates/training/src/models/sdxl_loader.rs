//! SDXL loader — specs derived from SimpleTuner (notes)
// Source references:
// - /home/alex/diffusers-rs/SimpleTuner/helpers/models/sdxl/
// Extract:
// - UNet weights: attention blocks q/k/v/o, feed-forward; LoRA attach points same
// - VAE latent interface (NHWC), text encoder dims (CLIP/T5 as configured)
// - Scheduler spacing (training/inference) knobs; keep BF16 params
use eridiffusion_core::Result;
use crate::model_registry::{TrainCfg, ModelBundle};

pub fn load_sdxl(_cfg: &TrainCfg) -> Result<ModelBundle> {
    Ok(ModelBundle { name: "sdxl".into() })
}
