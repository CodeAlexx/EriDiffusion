use eridiffusion_core::Result;

use crate::model_registry::{ModelBundle, TrainCfg};

pub fn load_qwen_image(_cfg: &TrainCfg) -> Result<ModelBundle> {
    Ok(ModelBundle { name: "qwen_image".into() })
}
