use eridiffusion_core::Result;

use crate::model_registry::{ModelBundle, TrainCfg};

pub fn load_wan22(_cfg: &TrainCfg) -> Result<ModelBundle> {
    Ok(ModelBundle { name: "wan22".into() })
}
