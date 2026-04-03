use anyhow::Result;
use eridiffusion_common_weights::{ParamId, ParamRegistry};

/// Enable grads only on a set of LoRA ParamIds
pub fn set_lora_grad_flags(reg: &mut ParamRegistry, ids: &[ParamId], on: bool) {
    reg.set_requires_grad(ids, on);
}

