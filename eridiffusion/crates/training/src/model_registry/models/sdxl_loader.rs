use eridiffusion_core::Result;

use crate::model_registry::{ModelBundle, TrainCfg};

pub fn load_sdxl(_cfg: &TrainCfg) -> Result<ModelBundle> {
    Ok(ModelBundle { name: "sdxl".into() })
}
