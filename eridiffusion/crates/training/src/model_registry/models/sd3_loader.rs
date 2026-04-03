use eridiffusion_core::Result;

use crate::model_registry::{ModelBundle, TrainCfg};

pub fn load_sd3(_cfg: &TrainCfg) -> Result<ModelBundle> {
    Ok(ModelBundle { name: "sd3".into() })
}
