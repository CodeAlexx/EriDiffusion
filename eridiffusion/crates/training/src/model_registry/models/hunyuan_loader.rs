use eridiffusion_core::Result;

use crate::model_registry::{ModelBundle, TrainCfg};

pub fn load_hunyuan(_cfg: &TrainCfg) -> Result<ModelBundle> {
    Ok(ModelBundle { name: "hunyuan".into() })
}
