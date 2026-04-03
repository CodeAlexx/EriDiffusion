use anyhow::Result;
use serde::{Serialize, Deserialize};
use eridiffusion_common_weights::ParamRegistry;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RunMeta { pub step: usize, pub notes: Option<String> }

pub fn save_lora(_reg: &ParamRegistry, _path: &str, _step: usize, _meta: &RunMeta) -> Result<()> { Ok(()) }
pub fn load_lora(_reg: &mut ParamRegistry, _path: &str) -> Result<()> { Ok(()) }

