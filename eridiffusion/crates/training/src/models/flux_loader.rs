//! Flux loader — specs derived from SimpleTuner (notes)
// Source references (scan locally):
// - /home/alex/diffusers-rs/SimpleTuner/helpers/models/flux/
// Extract:
// - Required weight keys: transformer blocks (q/k/v/o), mlp (fc1/fc2), layer norms
// - Text encoders: T5 (hidden=4096, max_len=256) and/or CLIP; accept pre-encoded embeddings
// - Expected inputs: NHWC latents (packed 2x2?), timesteps, text embeds; outputs predict eps
// - Scheduler: flowmatch/EDM variants; LoRA attach points: q/k/v/o + mlp
// - Keep BF16 params where possible; grads FP32 via accumulator
// - Respect layouts: Linear [IN, OUT], Conv [KH,KW,IC,OC]
use eridiffusion_core::Result;
use crate::model_registry::{TrainCfg, ModelBundle};
use std::fs;
use tracing::info;

fn try_inspect_weights(path: &str) {
    if let Ok(bytes) = fs::read(path) {
        if let Ok(st) = safetensors::SafeTensors::deserialize(&bytes) {
            let mut prefix_hist = std::collections::BTreeMap::<String, usize>::new();
            for name in st.names().iter().take(2000) {
                let prefix = name.split('.').next().unwrap_or("").to_string();
                *prefix_hist.entry(prefix).or_default() += 1;
            }
            info!("flux_loader: weights keys prefixes (sample) = {:?}", prefix_hist);
        }
    }
}

pub fn load_flux(_cfg: &TrainCfg) -> Result<ModelBundle> {
    if let Some(w) = &_cfg.weights { try_inspect_weights(w); }
    Ok(ModelBundle { name: "flux".into() })
}
