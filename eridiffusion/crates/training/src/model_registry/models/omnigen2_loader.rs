use eridiffusion_core::Result;

use crate::model_registry::{ModelBundle, TrainCfg};

pub fn load_omnigen2(_cfg: &TrainCfg) -> Result<ModelBundle> {
    Ok(ModelBundle { name: "omnigen2".into() })
}
