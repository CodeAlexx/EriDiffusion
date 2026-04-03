use anyhow::Result;
use eridiffusion_common_weights::ParamRegistry;

pub fn accumulate_start(_k: usize) {}
pub fn accumulate_step(_i: usize) {}
pub fn accumulate_finish() {}

pub fn clip_grad_global_norm(_reg: &ParamRegistry, _max_norm: f32) -> Result<f32> { Ok(0.0) }

